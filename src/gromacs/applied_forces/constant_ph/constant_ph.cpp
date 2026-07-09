/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2021,2022, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
/*! \file
 * \brief
 * Implements constant pH lambda dynamics code.
 *
 * \author Noora Aho <noora.s.aho@jyu.fi>
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 *
 */

#include <algorithm>
#include <array>
#include <numeric>
#include <optional>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include "gromacs/fileio/enxio.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/lambda_dynamics_params.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/units.h"
#include "gromacs/random/seed.h"
#include "gromacs/random/normaldistribution.h"
#include "gromacs/random/tabulatednormaldistribution.h"
#include "gromacs/random/threefry.h"
#include "gromacs/random/uniformrealdistribution.h"
#include "gromacs/trajectory/energyframe.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/linearalgebra/matrix.h"

#include "gromacs/domdec/ga2la.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"

#include "constant_ph.h"

/* Initialize double well barrier potential
   Previous work taken from constant pH version 5.1
 */
namespace
{
real dwpPotential(const real xx,
                  const real s,
                  const real f,
                  const real st,
                  const real m,
                  const real a,
                  const real b,
                  const real k,
                  const real d)
{
    const real sInvSq = 1 / gmx::square(s);
    const real aInvSq = 1 / gmx::square(a);

    const real devRight    = xx - 1 - b;
    const real expRight    = std::exp(-0.5_real * gmx::square(devRight) * aInvSq);
    const real devLeft     = xx + b;
    const real expLeft     = std::exp(-0.5_real * gmx::square(devLeft) * aInvSq);
    const real devMid      = xx - 0.5_real;
    const real expMid      = std::exp(-0.5_real * gmx::square(devMid) * sInvSq);
    const real erfArgRight = st * (xx - 1 - m);
    const real erfArgLeft  = st * (xx + m);

    return -k * (expRight + expLeft) + d * expMid
           + f * 0.5_real * (2 + std::erf(erfArgRight) - std::erf(erfArgLeft));
}

real dwpDvdl(const real xx,
             const real s,
             const real f,
             const real st,
             const real m,
             const real a,
             const real b,
             const real k,
             const real d)
{
    const real sInvSq = 1 / gmx::square(s);
    const real aInvSq = 1 / gmx::square(a);

    const real devRight    = xx - 1 - b;
    const real expRight    = std::exp(-0.5_real * gmx::square(devRight) * aInvSq);
    const real devLeft     = xx + b;
    const real expLeft     = std::exp(-0.5_real * gmx::square(devLeft) * aInvSq);
    const real devMid      = xx - 0.5_real;
    const real expMid      = std::exp(-0.5_real * gmx::square(devMid) * sInvSq);
    const real erfArgRight = st * (xx - 1 - m);
    const real erfArgLeft  = st * (xx + m);

    return k * (devRight * expRight + devLeft * expLeft) * aInvSq - d * devMid * expMid * sInvSq
           + f * st * (std::exp(-gmx::square(erfArgRight)) - std::exp(-gmx::square(erfArgLeft))) * M_1_SQRTPI;
}

real pHPotential(const real xx, const real st, const real a, const real pH, const real pKa, const real referenceTemperature)
{
    const real expArg = (pKa > pH) ? -5 * st * (xx - 2 * a) : -5 * st * (xx - 1 + 2 * a);
    return (expArg > 5) ? gmx::c_boltz * referenceTemperature * std::log(10.0_real) * (pKa - pH)
                                  * std::exp(-expArg) / (1 + std::exp(-expArg))
                        : gmx::c_boltz * referenceTemperature * std::log(10.0_real) * (pKa - pH)
                                  / (1 + std::exp(expArg));
}

real pHDvdl(const real xx, const real st, const real a, const real pH, const real pKa, const real referenceTemperature)
{
    const real expArg = (pKa > pH) ? -5 * st * (xx - 2 * a) : -5 * st * (xx - 1 + 2 * a);
    return (expArg > 5) ? gmx::c_boltz * referenceTemperature * std::log(10.0_real) * (pKa - pH) * (5 * st)
                                  * std::exp(-expArg) / gmx::square(1 + std::exp(-expArg))
                        : gmx::c_boltz * referenceTemperature * std::log(10.0_real) * (pKa - pH) * (5 * st)
                                  * std::exp(expArg) / gmx::square(1 + std::exp(expArg));
    ;
}

std::array<real, 15> init_lambda_dwp(const real barrier)
{
    std::array<real, 15> lambda_dwp = { 0 };
    /* fields of array lambda_dwp are filled 08.08.2014 */
    FILE* fw1;
    real  s = 0.3;
    real  e1, e10, st, m;
    real  sig0, sig, x, x0;
    real  eps = 0.005;
    real  dg;
    real  v, vmin, vmin0, dv, iii;
    real  totx, totp, totxp;
    real  flagfor;
    int   ready;
    real  a, b, k, d;
    int   iter;
    int   max_iter;
    b        = -0.1;
    a        = 0.05;
    iter     = 0;
    max_iter = 10000;

    fw1   = fopen("lambda_dwp.dat", "w");
    iii   = barrier - 1.0;
    sig0  = 0.02;
    sig   = sig0;
    x0    = 0;
    vmin0 = 10.0;

    /* Note that the height of outer walls will be f+barrier/2 */

    // for f=50 follows
    //  real f=50.0;
    //  e1=1.45222;         /*e1=InverseErf[1-2.0/f]*/
    // e10=0.595116;        /*e10=InverseErf[1-20.0/f]*/

    // real f=100.0;
    // e1=1.64498;           /*e1=InverseErf[1-2.0/f]*/
    // e10=0.906194;         /*e10=InverseErf[1-20.0/f]*/

    // For f=1000 follows
    real f = 1000.0;
    e1     = 2.185;
    e10    = 1.645;

    lambda_dwp[0] = s;
    lambda_dwp[1] = f;
    lambda_dwp[2] = e1;
    lambda_dwp[3] = e10;

    st = (e1 - e10) / (2.0 * sig0);
    m  = 2.0 * sig0 * (2.0 * e1 - e10) / (e1 - e10);

    lambda_dwp[4] = st;
    lambda_dwp[5] = m;

    flagfor = iii;

    for (iii = flagfor; iii <= flagfor; iii = iii + 1)
    {
        dg   = 1.0 + iii;
        sig0 = 0.02;
        sig  = sig0;
        b    = -0.1;
        a    = 0.05;
        k    = dg / 2.0;
        d    = dg / 2.0;

        /* correct for the fact that the two minima are shallower than k=dg/2 due to the tails of
         * the central gaussian and the erf functions (basically bring the minima to the same value
         * of the two side gaussians before the central gaussian was added) */

        vmin = vmin0;
        for (x = -0.1; x <= 0.2; x = x + 0.001)
        {
            v = dwpPotential(x, s, f, st, m, a, b, k, d);
            if (v < vmin)
            {
                vmin = v;
            }
        }
        k = k + dg / 2.0 + vmin;

        /* adjust location minima and width boltzmann distribution therein to the target values (0,1 and sig0) */

        ready = 0;
        while (ready == 0) /*while which correspond to repeat begin*/

        {
            vmin = vmin0;
            for (x = -0.1; x < 0.2; x = x + 0.001)
            {
                v = dwpPotential(x, s, f, st, m, a, b, k, d);
                if (v < vmin)
                {
                    vmin = v;
                }
            }
            k = k + dg / 2.0 + vmin;

            x    = 0.5;
            totp = 0;
            totx = 0;
            while (x > -0.2)
            {
                v = dwpPotential(x, s, f, st, m, a, b, k, d);
                if (v <= 0)
                {
                    totp = totp + std::exp(-v);
                    totx = totx + x * std::exp(-v);
                }
                x = x - 0.001;
            }
            x0 = totx / totp;

            x     = 0.5;
            totxp = 0;
            while (x > -0.2)
            {
                v = dwpPotential(x, s, f, st, m, a, b, k, d);
                if (v <= 0)
                {
                    totxp = totxp + ((x - x0) * (x - x0) * std::exp(-v));
                }
                x = x - 0.001;
            }
            sig = std::sqrt(totxp / totp);
            b   = b + 0.01 * x0;
            a   = a / (1.0 + 0.01 * (sig - sig0) / sig0);

            for (x = -0.2; x <= 1.2; x = x + 0.001)
            {
                v = dwpPotential(x, s, f, st, m, a, b, k, d);
            }

            if ((std::abs(x0) <= eps) && (std::abs(sig - sig0) / sig0 <= eps))
            {
                ready = 1;
                for (x = -1.0; x <= 2.0; x = x + 0.001)
                {
                    v  = dwpPotential(x, s, f, st, m, a, b, k, d);
                    dv = dwpPotential(x, s, f, st, m, a, b, k, d);
                    fprintf(fw1, "%f %f\n", x, v);
                }
                lambda_dwp[6] = a;
                lambda_dwp[7] = b;
                lambda_dwp[8] = k;
                lambda_dwp[9] = d;
            }

            if (iii < -0.55)
            {
                k     = 0;
                d     = 0;
                ready = 1;
                for (x = -1.0; x <= 2.0; x = x + 0.001)
                {
                    v  = dwpPotential(x, s, f, st, m, a, b, k, d);
                    dv = dwpDvdl(x, s, f, st, m, a, b, k, d);
                    fprintf(fw1, "%f %f\n", x, v);
                }
                lambda_dwp[6] = a;
                lambda_dwp[7] = b;
                lambda_dwp[8] = k;
                lambda_dwp[9] = d;
                if (barrier != 0.)
                {
                    fprintf(stderr, "Warning: Barrier changed from %f to zero\n", barrier);
                }
                fprintf(stderr, "Parameters double well potential:\n");
                fprintf(stderr, "a = %f b = %f k = %f d = %f dg = %f \n", a, b, k, d, dg);
                fprintf(stderr, "m = %f st = %f \n", m, st);
                fprintf(stderr, "\n");
            }

            iter++;

            if (iter > max_iter)
            {
                ready = 1;
                for (x = -1.0; x <= 2.0; x = x + 0.001)
                {
                    v  = dwpPotential(x, s, f, st, m, a, b, k, d);
                    dv = dwpDvdl(x, s, f, st, m, a, b, k, d);
                    fprintf(fw1, "%f %f %f\n", x, v, dv);
                }
                lambda_dwp[6] = a;
                lambda_dwp[7] = b;
                lambda_dwp[8] = k;
                lambda_dwp[9] = d;
                fprintf(stderr,
                        "Warning: double well potential did not converge to tolerance limit eps=%f "
                        "in max iter=%d\n",
                        eps, max_iter);
                fprintf(stderr,
                        "Warning: check the shape of the double well potential in "
                        "lambda_dwp.dat\n");
                fprintf(stderr, "Parameters double well potential:\n");
                fprintf(stderr, "a = %f b = %f k = %f d = %f dg = %f \n", a, b, k, d, dg);
                fprintf(stderr, "m = %f st = %f \n", m, st);
                fprintf(stderr, "\n");
            }

        } /* endwhile of repeat begin */

    } /* end for */

    fclose(fw1);
    return lambda_dwp;
}

real gaussdist(gmx::DefaultRandomEngine* rng, real sigma)
{
    constexpr real                     two = 2.0;
    real                               r   = two;
    gmx::UniformRealDistribution<real> uniformDist;
    real                               x = 0;
    do
    {
        x            = two * uniformDist(*rng) - 1.0;
        const real y = 2.0 * uniformDist(*rng) - 1.0;
        r            = x * x + y * y;
    } while (r > 1.0 || r == 0.0);
    r = x * std::sqrt(-two * std::log(r) / r);
    r = r * sigma;
    return (r);
}

int maxNumChargeConstraintGroups(const gmx::LambdaDynamicsSimulationParameters& lambdaParams)
{
    int numIndices = 0;
    for (const auto& atomSet : lambdaParams.lambdaAtomsCollections())
    {
        if (atomSet.isInConstraintGroup())
        {
            numIndices = std::max(numIndices, atomSet.chargeConstraintGroup());
        }
    }
    return numIndices;
}

template<bool isInitialLambda>
void calculateChargeConstraint(gmx::ArrayRef<gmx::LambdaCoordinate> lambdaCoordinates)
{
    for (auto& lambdaCoordinate : lambdaCoordinates)
    {
        const real coordinate = isInitialLambda ? lambdaCoordinate.x0 : lambdaCoordinate.x;
        lambdaCoordinate.lambdaChargeConstraint =
                coordinate * (lambdaCoordinate.totalChargeB - lambdaCoordinate.totalChargeA);
        /*if (lambdaCoordinateIt->constraintGroupMembers.empty())
        {
            lambdaCoordinateIt->lambdaChargeConstraint = coordinate;
        }
        else
        {
            real totalCharge = lambdaCoordinateIt->totalChargeA * (1 - coordinate)
                               + lambdaCoordinateIt->totalChargeB * coordinate;
            for (auto otherIt = lambdaCoordinateIt->constraintGroupMembers.begin();
                 otherIt != lambdaCoordinateIt->constraintGroupMembers.end(); ++otherIt)
            {
                const real otherCoord = isInitialLambda ? (*otherIt)->x0 : (*otherIt)->x;
                totalCharge += (*otherIt)->totalChargeA * (1 - otherCoord)
                               + (*otherIt)->totalChargeB * otherCoord;
            }
            lambdaCoordinateIt->lambdaChargeConstraint = totalCharge;
            lambdaCoordinateIt += lambdaCoordinateIt->constraintGroupMembers.size();
        }*/
    }
}

std::vector<real> lambdaConstraintStates(gmx::ArrayRef<const gmx::LambdaCoordinate> allCoordinates,
                                         int numMultiStateConstraintGroups,
                                         int numChargeConstraintGroups)
{
    if (numChargeConstraintGroups + numMultiStateConstraintGroups <= 0)
    {
        return std::vector<real>();
    }
    std::vector<real> sumInitialLambdas(numMultiStateConstraintGroups + numChargeConstraintGroups, 0);

    //! for multistate groups sum of lambdas should always be one
    for (int i = 0; i < numMultiStateConstraintGroups; i++)
    {
        sumInitialLambdas[i] = 1;
    }
    //! for charge constraint groups we are using initial charges
    for (const auto& coord : allCoordinates)
    {
        const int index = numMultiStateConstraintGroups + coord.chargeConstraintGroup - 1;
        if (index >= numMultiStateConstraintGroups)
        {
            if (index - numMultiStateConstraintGroups == coord.chargeConstraintGroup - 1)
            {
                sumInitialLambdas[index] += coord.lambdaChargeConstraint * coord.bufferResidueMultiplier;
            }
        }
    }
    return sumInitialLambdas;
}

int numMultiStateConstraintGroups(gmx::ArrayRef<const gmx::LambdaCoordinate> coordinates)
{
    int groups = 0;
    for (auto coordIt = coordinates.begin(); coordIt != coordinates.end(); coordIt++)
    {
        if (coordIt->isInConstraintGroup)
        {
            ++groups;
            coordIt += coordIt->constraintGroupMembers.size();
        }
    }
    return groups;
}

int numMultiGroupConstraintGroupStates(gmx::ArrayRef<const gmx::LambdaCoordinate> coordinates)
{
    int numStates = 0;
    for (auto coordIt = coordinates.begin(); coordIt != coordinates.end(); coordIt++)
    {
        if (coordIt->isInConstraintGroup)
        {
            const int size = coordIt->constraintGroupMembers.size();
            numStates += size + 1; // add one for this state!
            coordIt += size;       // advance over the other members!
        }
    }
    return numStates;
}

std::vector<gmx::LambdaCoordinate> createCoordinates(const t_inputrec&    ir,
                                                     int                  randomSeed,
                                                     const gmx::MDLogger& mdlog)
{
    GMX_LOG(mdlog.info).asParagraph().appendText("Constant pH MD initialization:");
    std::vector<gmx::LambdaCoordinate> lambdaCoordinates;
    const auto&                        lambdaParams = *ir.lambdaDynamicsSimulationParameters;

    int totalNumOfCoordinates = 0;
    for (const auto& atomCollection : lambdaParams.lambdaAtomsCollections())
    {
        const auto& lambdaResidue = lambdaParams.lambdaResidues()[atomCollection.lambdaResidueIndex()];

        // We need to get the correct residue for the multi state constraint case
        // here. So we make a temp vector of the residues and loop over this one.
        // In case of no multi state constraints we just have a single entry at the current index.
        const auto multiStateConstraintGroupIndices =
                lambdaResidue.isInMultiStateConstraintGroup()
                        ? constructMultiStateConstraintGroupIndices(
                                lambdaParams.lambdaResidues(), lambdaResidue.multiStateConstraintGroup())
                        : std::vector<int>(1, atomCollection.lambdaResidueIndex());

        int positionInGroup = 0;
        for (const auto& residueIndex : multiStateConstraintGroupIndices)
        {
            const auto& currentLambdaResidue = lambdaParams.lambdaResidues()[residueIndex];
            // We need to get the correct residue for the multi state constraint case
            // here. So we make a temp vector of the
            gmx::LambdaCoordinate newCoordinate;

            real assignInitialLambda = atomCollection.initialLambda()[positionInGroup];

            newCoordinate.groupNumber              = totalNumOfCoordinates;
            newCoordinate.globalAtomIndices        = atomCollection.atomIndicies();
            newCoordinate.chargeA                  = currentLambdaResidue.chargeA();
            newCoordinate.chargeB                  = currentLambdaResidue.chargeB();
            newCoordinate.dvdlCoefficients         = currentLambdaResidue.dvdlCoefficients();
            newCoordinate.maximumNumberPowerSeries = currentLambdaResidue.maximumNumberPowerSeries();
            newCoordinate.isBufferResidue          = atomCollection.isBufferResidue();
            newCoordinate.x0                       = assignInitialLambda;
            newCoordinate.referencePka             = newCoordinate.isBufferResidue
                                                             ? lambdaParams.simulationpH()
                                                             : currentLambdaResidue.referencePka();
            newCoordinate.bufferResidueMultiplier  = atomCollection.bufferResidueMultiplier();
            newCoordinate.totalChargeA             = 0;
            for (const auto& charge : currentLambdaResidue.chargeA())
            {
                newCoordinate.totalChargeA += charge;
            }
            newCoordinate.totalChargeB = 0;
            for (const auto& charge : currentLambdaResidue.chargeB())
            {
                newCoordinate.totalChargeB += charge;
            }

            // initialize lambda structure
            newCoordinate.x   = newCoordinate.x0;
            newCoordinate.tau = lambdaParams.lambdatau();
            newCoordinate.bar = atomCollection.barrier();
            if (atomCollection.isInConstraintGroup())
            {
                newCoordinate.chargeConstraintGroup = atomCollection.chargeConstraintGroup();
            }

            if (lambdaParams.isCalibrationRun() || newCoordinate.isBufferResidue)
            {
                newCoordinate.v     = 0.0_real;
                newCoordinate.v_old = 0.0_real;
            }
            else
            {
                // initial velocity of lambda particle
                gmx::UniformRealDistribution<real> uniformDist;
                gmx::DefaultRandomEngine           Rng(randomSeed);
                double sigma    = sqrt(newCoordinate.T * gmx::c_boltz / lambdaParams.lambdaParticleMass());
                newCoordinate.v = gaussdist(&Rng, sigma); /* random start velocity to lambda */
                newCoordinate.v_old = 0.0_real;
            }

            // initialize double well potential
            newCoordinate.lambda_dwp = init_lambda_dwp(newCoordinate.bar);

            newCoordinate.isInConstraintGroup = (multiStateConstraintGroupIndices.size() > 1);
            if (multiStateConstraintGroupIndices.size() > 1)
            {
                newCoordinate.constraintGroupMembers.resize(multiStateConstraintGroupIndices.size() - 1);
            }
            lambdaCoordinates.emplace_back(newCoordinate);
            totalNumOfCoordinates++;
            positionInGroup++;
        }
    }

    if (lambdaParams.isCalibrationRun())
    {
        GMX_LOG(mdlog.info)
                .asParagraph()
                .appendText(
                        "Current run will calibrate a set of dvdl parameters and not update the "
                        "lambda "
                        "coordinate");
    }

    fprintf(stderr, "Total number of lambda coordinates is %d\n", totalNumOfCoordinates);

    for (auto& lambdaCoordinate : lambdaCoordinates)
    {
        if (!lambdaCoordinate.isBufferResidue)
        {
            GMX_LOG(mdlog.info)
                    .appendTextFormatted("Parameters double well potential of group %d:",
                                         lambdaCoordinate.groupNumber + 1);
            GMX_LOG(mdlog.info)
                    .appendTextFormatted("s = %f f = %f st = %f m = %f", lambdaCoordinate.lambda_dwp[0],
                                         lambdaCoordinate.lambda_dwp[1], lambdaCoordinate.lambda_dwp[4],
                                         lambdaCoordinate.lambda_dwp[5]);
            GMX_LOG(mdlog.info)
                    .appendTextFormatted("a = %f b = %f k = %f d = %f", lambdaCoordinate.lambda_dwp[6],
                                         lambdaCoordinate.lambda_dwp[7], lambdaCoordinate.lambda_dwp[8],
                                         lambdaCoordinate.lambda_dwp[9]);
        }
        else
        {
            GMX_LOG(mdlog.info)
                    .appendTextFormatted("Group %d is a buffer. Double well potential is set to 0.",
                                         lambdaCoordinate.groupNumber + 1);
        }
    }

    return lambdaCoordinates;
}

} // namespace

gmx::ArrayRef<real> ConstantPHOutputStorage::accessOutput(int index)
{
    return output_[index];
}

gmx::ArrayRef<const real> ConstantPHOutputStorage::accessOutput(int index) const
{
    return output_[index];
}

void ConstantPHOutputStorage::init(int numLambdas)
{
    output_.resize(numLambdas);
}

bool ConstantPH::isOutputStep(int64_t step) const
{
    return (step % lambdaNst_ == 0);
}

void ConstantPH::writeToEnergyFrame(int64_t step, t_enxframe* frame)
{
    GMX_ASSERT(frame != nullptr, "Need a valid energy frame");
    if (!isOutputStep(step))
    {
        return;
    }
    const int numSubblocks = lambdaCoordinates_.size();
    // one block for each lambda coordinate to write dvdl to one subblock
    // one additional one for lambda coordinate value and velocity
    GMX_ASSERT(numSubblocks > 0, "We should always have data to write");

    // We always write dvdl
    add_blocks_enxframe(frame, frame->nblock + 1);

    t_enxblock* cpHMDDvdlBlock = &(frame->block[frame->nblock - 1]);
    add_subblocks_enxblock(cpHMDDvdlBlock, numSubblocks);
    cpHMDDvdlBlock->id = enxCPHMD;

    int blockIndex = 0;
    for (auto& lambdaCoordinate : lambdaCoordinates_)
    {
        auto storage = outputStorage_.accessOutput(blockIndex);
        storage[static_cast<int>(gmx::CpHMDOutputSelection::Coordinate)] = lambdaCoordinate.x;
        storage[static_cast<int>(gmx::CpHMDOutputSelection::Velocity)]   = lambdaCoordinate.v;
        storage[static_cast<int>(gmx::CpHMDOutputSelection::Dvdl)] = lambdaCoordinate.dvdl_pot;


        cpHMDDvdlBlock->sub[blockIndex].type = XdrDataType::Float;
        cpHMDDvdlBlock->sub[blockIndex].nr   = 3;
        cpHMDDvdlBlock->sub[blockIndex].fval = storage.data();
        blockIndex++;
    }
}

namespace
{

void linkMultiStateCoordinates(gmx::ArrayRef<gmx::LambdaCoordinate> lambdaCoordinates)
{
    int positionInGroup = 0;
    for (auto lambdaCoordinateIt = lambdaCoordinates.begin();
         lambdaCoordinateIt != lambdaCoordinates.end(); ++lambdaCoordinateIt)
    {
        if (!lambdaCoordinateIt->constraintGroupMembers.empty())
        {
            int size  = lambdaCoordinateIt->constraintGroupMembers.size();
            int index = 0;
            for (int pos = size - positionInGroup; pos >= -positionInGroup; --pos)
            {
                auto otherIt = lambdaCoordinateIt + pos;
                if (otherIt != lambdaCoordinateIt)
                {
                    lambdaCoordinateIt->constraintGroupMembers[index] = &(*otherIt);
                    ++index;
                }
            }
            ++positionInGroup;
            if (positionInGroup > size)
            {
                positionInGroup = 0;
            }
        }
    }
}

// calculate all polynomial components of particular power
std::vector<real> calcPowerComponents(gmx::ArrayRef<const real> lambdaGroupPowerComponent, int pow)
{
    std::vector<real> result;
    if (lambdaGroupPowerComponent.size() == 1)
    {
        result.push_back(std::pow(lambdaGroupPowerComponent[0], pow));
        return result;
    }

    if (pow == 0)
    {
        result.push_back(1);
        return result;
    }

    if (pow == 1)
    {
        for (const auto& l : lambdaGroupPowerComponent)
        {
            result.push_back(l);
        }
        return result;
    }

    gmx::ArrayRef<const real> subLambdaGroupPowerComponent = { lambdaGroupPowerComponent.begin() + 1,
                                                               lambdaGroupPowerComponent.end() };
    for (int i = 0; i <= pow; ++i)
    {
        std::vector<real> intermediate = calcPowerComponents(subLambdaGroupPowerComponent, i);
        for (const auto& j : intermediate)
        {
            result.push_back(std::pow(lambdaGroupPowerComponent[0], pow - i) * j);
        }
    }

    return result;
}

// calculate integral of all polynomial components of particular power
std::vector<real> calcIntPowerComponents(gmx::ArrayRef<const real> lambdaGroupPowerComponent, int pow, int intLambda)
{
    if ((lambdaGroupPowerComponent.ssize() < intLambda) || (intLambda <= -1))
    {
        return calcPowerComponents(lambdaGroupPowerComponent, pow);
    }

    std::vector<real> result;
    if (lambdaGroupPowerComponent.ssize() == intLambda)
    {
        for (int i = 0; i < lambdaGroupPowerComponent.ssize(); ++i)
        {
            std::vector<real> accumulator = calcIntPowerComponents(lambdaGroupPowerComponent, pow, i);
            if (result.empty())
            {
                for (const auto& j : accumulator)
                {
                    result.push_back(-j);
                }
            }
            else
            {
                for (int j = 0; j < gmx::ssize(accumulator); ++j)
                {
                    result[j] -= accumulator[j];
                }
            }
        }
        return result;
    }

    if (lambdaGroupPowerComponent.ssize() == 1)
    {
        result.push_back(std::pow(lambdaGroupPowerComponent[0], pow + 1) / (pow + 1));
        return result;
    }

    if (pow == 0)
    {
        result.push_back(lambdaGroupPowerComponent[intLambda]);
        return result;
    }

    if (pow == 1)
    {
        for (const auto& l : lambdaGroupPowerComponent)
        {
            result.push_back((0 == intLambda ? l * l / 2. : l * lambdaGroupPowerComponent[intLambda]));
        }
        return result;
    }

    gmx::ArrayRef<const real> subLambdaGroupPowerComponent = { lambdaGroupPowerComponent.begin() + 1,
                                                               lambdaGroupPowerComponent.end() };
    if (intLambda == 0)
    {
        for (int i = 0; i <= pow; ++i)
        {
            std::vector<real> intermediate = calcPowerComponents(subLambdaGroupPowerComponent, i);
            for (const auto& j : intermediate)
            {
                result.push_back(std::pow(lambdaGroupPowerComponent[0], pow - i + 1) * j / (pow - i + 1));
            }
        }

        return result;
    }

    for (int i = 0; i <= pow; ++i)
    {
        std::vector<real> intermediate =
                calcIntPowerComponents(subLambdaGroupPowerComponent, i, intLambda - 1);
        for (const auto& j : intermediate)
        {
            result.push_back(std::pow(lambdaGroupPowerComponent[0], pow - i) * j);
        }
    }

    return result;
}

// calculate convolution of lambda polynomials with dvdl coefficients
real calcPolynomial(gmx::ArrayRef<const real> lambdaGroupPowerComponent,
                    gmx::ArrayRef<const real> dvdlCoefficients,
                    const int                 maximumNumberPowerSeries)
{
    real              result = 0.;
    std::vector<real> polynomial;
    for (int i = maximumNumberPowerSeries; i >= 0; --i)
    {
        for (const auto& m : calcPowerComponents(lambdaGroupPowerComponent, i))
        {
            polynomial.push_back(m);
        }
    }

    for (gmx::Index i = 0; i < dvdlCoefficients.ssize(); i++)
    {
        result -= dvdlCoefficients[i] * polynomial[i];
    }

    return result;
}

// calculate convolution of integrated lambda polynomials with dvdl coefficients
real calcIntPolynomial(gmx::ArrayRef<const real> lambdaGroupPowerComponent,
                       gmx::ArrayRef<const real> dvdlCoefficients,
                       const int                 intLambda,
                       const int                 maximumNumberPowerSeries)
{
    real              result = 0.;
    std::vector<real> polynomial;
    for (int i = maximumNumberPowerSeries; i >= 0; --i)
    {
        for (const auto& m : calcIntPowerComponents(lambdaGroupPowerComponent, i, intLambda))
        {
            polynomial.push_back(m);
        }
    }

    for (gmx::Index i = 0; i < dvdlCoefficients.ssize(); i++)
    {
        result -= dvdlCoefficients[i] * polynomial[i];
    }

    return result;
}

void computeForces(gmx::LambdaCoordinate* currentLambdaCoordinate,
                   gmx::ArrayRef<real>    groupPotential,
                   bool                   useMultiStateConstraits,
                   bool                   isCalibrationRun,
                   const real             referenceTemperature,
                   const real             simulationpH)
{
    real xx = currentLambdaCoordinate->x;

    // real  dvdl = 0.0;
    real corr_ph       = 0.0;
    real corr_dvdl     = 0.0;
    real corr_pot_ph   = 0.0;
    real corr_pot_dvdl = 0.0;

    const real                ref_pka   = currentLambdaCoordinate->referencePka;
    gmx::ArrayRef<const real> dvdlCoefs = currentLambdaCoordinate->dvdlCoefficients;

    // effect of bond breaking (free energy dvdl) -> multi state or simple two state
    if (useMultiStateConstraits && currentLambdaCoordinate->isInConstraintGroup)
    {
        // three lambdas contribute in total here.
        // Currently only works for three lambdas constrained
        // This should be generalized
        // ?? How to represent multivariate polynomials?
        std::vector<const gmx::LambdaCoordinate*> multiStateLambdas;
        multiStateLambdas.emplace_back(currentLambdaCoordinate);
        for (const auto& otherCoord : currentLambdaCoordinate->constraintGroupMembers)
        {
            multiStateLambdas.emplace_back(&(*otherCoord));
        }

        std::sort(multiStateLambdas.begin(), multiStateLambdas.end(),
                  [](const auto& lambda1, const auto& lambda2) {
                      return lambda1->groupNumber < lambda2->groupNumber;
                  });
        // only firs n-1 lambdas of constraint group are needed
        multiStateLambdas.pop_back();
        std::vector<real> multiStateLambdaValues;
        multiStateLambdaValues.reserve(multiStateLambdas.size());
        for (const auto multiStateLambda : multiStateLambdas)
        {
            multiStateLambdaValues.emplace_back(multiStateLambda->x);
        }

        corr_dvdl = calcPolynomial(multiStateLambdaValues, dvdlCoefs,
                                   currentLambdaCoordinate->maximumNumberPowerSeries);

        auto      posLambda = std::find_if(multiStateLambdas.begin(), multiStateLambdas.end(),
                                      [&currentLambdaCoordinate](const auto& lambda) {
                                          return lambda->groupNumber == currentLambdaCoordinate->groupNumber;
                                      });
        const int intLambda = (posLambda == multiStateLambdas.end())
                                      ? multiStateLambdas.size()
                                      : posLambda - multiStateLambdas.begin();

        corr_pot_dvdl = calcIntPolynomial(multiStateLambdaValues, dvdlCoefs, intLambda,
                                          currentLambdaCoordinate->maximumNumberPowerSeries);
    }
    else
    {
        int order = dvdlCoefs.size() - 1;
        for (const auto& coeff : dvdlCoefs)
        {
            // Should we save potential for all constrained lambda?
            corr_dvdl += -1.0 * coeff * pow(xx, order);
            corr_pot_dvdl += -1.0 * coeff * pow(xx, order + 1) / (order + 1);
            --order;
        }
    }

    // dvdl array now has the force so no need to loop over atoms (lambdaAtoms array can be ordered)

    const real s  = currentLambdaCoordinate->lambda_dwp[0];
    const real f  = currentLambdaCoordinate->lambda_dwp[1];
    const real st = currentLambdaCoordinate->lambda_dwp[4];
    const real m  = currentLambdaCoordinate->lambda_dwp[5];
    const real a  = currentLambdaCoordinate->lambda_dwp[6];
    const real b  = currentLambdaCoordinate->lambda_dwp[7];
    const real k  = currentLambdaCoordinate->lambda_dwp[8];
    const real d  = currentLambdaCoordinate->lambda_dwp[9];

    // effect of external pH

    corr_ph += (currentLambdaCoordinate->isBufferResidue)
                       ? 0
                       : pHDvdl(currentLambdaCoordinate->x, st, a, simulationpH, ref_pka,
                                referenceTemperature);
    corr_pot_ph += (currentLambdaCoordinate->isBufferResidue)
                           ? 0
                           : pHPotential(currentLambdaCoordinate->x, st, a, simulationpH, ref_pka,
                                         referenceTemperature);

    // effect of double barrier potential - taken from previous constant ph version 5.1
    const real corr_dwp     = (currentLambdaCoordinate->isBufferResidue)
                                      ? 0
                                      : dwpDvdl(currentLambdaCoordinate->x, s, f, st, m, a, b, k, d);
    const real corr_pot_dwp = (currentLambdaCoordinate->isBufferResidue)
                                      ? 0
                                      : dwpPotential(currentLambdaCoordinate->x, s, f, st, m, a, b, k, d);

    // for dvdl output
    currentLambdaCoordinate->dvdl_dwp = corr_dwp;
    currentLambdaCoordinate->dvdl_ref = corr_dvdl;
    currentLambdaCoordinate->dvdl_ph  = corr_ph;
    currentLambdaCoordinate->pot_dwp  = corr_pot_dwp;
    currentLambdaCoordinate->pot_ref  = corr_pot_dvdl;
    currentLambdaCoordinate->pot_ph   = corr_pot_ph;
    currentLambdaCoordinate->dvdl_pot = groupPotential[currentLambdaCoordinate->groupNumber];

    // If mass zero then don't update -> reference free energy calculation
    if (!isCalibrationRun)
    {
        // if not calibraton run, we correct for multiple residues in the same
        // group before applying the rest of corrections.
        groupPotential[currentLambdaCoordinate->groupNumber] /=
                currentLambdaCoordinate->bufferResidueMultiplier;
        groupPotential[currentLambdaCoordinate->groupNumber] += corr_ph + corr_dvdl + corr_dwp;
    }
}

std::vector<real> getCurrrentConstraints(gmx::ArrayRef<const gmx::LambdaCoordinate> lambdaCoordinates,
                                         int numMultiStateConstraintGroups,
                                         int numChargeConstraintGroups)
{
    std::vector<real> currentConstraints(numMultiStateConstraintGroups + numChargeConstraintGroups, 0);

    int         multiStateIt = 0;
    std::size_t pos          = 0;
    for (const auto& lambdaCoordinate : lambdaCoordinates)
    {
        // first, parse all multistate constraints
        if (!lambdaCoordinate.constraintGroupMembers.empty())
        {
            currentConstraints[multiStateIt] += lambdaCoordinate.x;
            ++pos;
            if (pos == lambdaCoordinate.constraintGroupMembers.size() + 1)
            {
                ++multiStateIt;
                pos = 0;
            }
        }

        // second, parse all charge constraints
        for (int i = 0; i < numChargeConstraintGroups; ++i)
        {
            if (i == lambdaCoordinate.chargeConstraintGroup - 1)
            {
                currentConstraints[numMultiStateConstraintGroups + i] +=
                        lambdaCoordinate.lambdaChargeConstraint * lambdaCoordinate.bufferResidueMultiplier;
            }
        }
    }
    return currentConstraints;
}

std::vector<real> getCurrentMultiplier(double const* const*      constraintMultipliers,
                                       gmx::ArrayRef<const real> currentConstraints,
                                       gmx::ArrayRef<const real> initialConstraints)
{
    GMX_RELEASE_ASSERT(currentConstraints.size() == initialConstraints.size(),
                       "Need to have same number of current and initial constraints");
    std::vector<real> currentMultipliers;
    for (size_t i = 0; i < initialConstraints.size(); ++i)
    {
        real multiplier = 0.0;
        for (size_t j = 0; j < initialConstraints.size(); ++j)
        {
            multiplier += constraintMultipliers[i][j] * (currentConstraints[j] - initialConstraints[j]);
        }
        currentMultipliers.emplace_back(multiplier);
    }
    return currentMultipliers;
}

void do_constraints(gmx::ArrayRef<gmx::LambdaCoordinate> lambdaCoordinates,
                    double const* const*                 constraintMultipliers,
                    gmx::ArrayRef<const real>            sumOfInitialLambdaConstraints,
                    const int                            numMultiStateConstraintGroups,
                    const real                           dt,
                    const real                           lambdaMass)
{
    int numChargeConstraintGroups = sumOfInitialLambdaConstraints.size() - numMultiStateConstraintGroups;
    //! Calculate charge constraints components
    calculateChargeConstraint<false>(lambdaCoordinates);
    //! Get current constraints
    std::vector<real> currentConstraints = getCurrrentConstraints(
            lambdaCoordinates, numMultiStateConstraintGroups, numChargeConstraintGroups);
    //! Get current multipliers
    std::vector<real> currentMultipliers = getCurrentMultiplier(
            constraintMultipliers, currentConstraints, sumOfInitialLambdaConstraints);
    //! Update positions
    for (auto& lambdaCoordinate : lambdaCoordinates)
    {
        GMX_RELEASE_ASSERT(currentMultipliers.size() == lambdaCoordinate.constraintCoefficients.size(),
                           "Need to have same number of multipliers and constraint coefficients");
        for (gmx::Index i = 0; i < gmx::ssize(currentMultipliers); i++)
        {
            lambdaCoordinate.x -= currentMultipliers[i] * lambdaCoordinate.constraintCoefficients[i];
        }
    }

    // advance lambda velocities, kinetic energies, temperatures
    // (also for non-constrained groups these now computed again but should not matter)

    for (auto& lambdaCoordinate : lambdaCoordinates)
    {
        lambdaCoordinate.v    = (lambdaCoordinate.x - lambdaCoordinate.x_old) / dt;
        lambdaCoordinate.ekin = 0.5 * lambdaMass * gmx::square(lambdaCoordinate.v);
        lambdaCoordinate.T    = 2.0 * (lambdaCoordinate.ekin) / gmx::c_boltz;
    }
}


/*  Perform T coupling with v-rescale thermostat
 *  - couple all lambdas together
 *  - Nf = number of degrees of freedom = N - 1
 *  - according to Bussi et al JCP (2007) appendix
 *
 *  Returns the new kinetic energy
 */
real tcouple_vrescale_collective(gmx::ArrayRef<gmx::LambdaCoordinate> lambdaCoordinates,
                                 const real                           Tref,
                                 const real                           tau,
                                 const real                           dt,
                                 const real                           Ekin_total,
                                 const int                            numMultiStateConstraintGroups,
                                 const int64_t                        step,
                                 const int64_t                        seed,
                                 const bool                           useChargeConstraints,
                                 const bool                           useMultiStateConstraits)
{
    gmx::ThreeFry2x64<64> rng(seed, gmx::RandomDomain::ConstantPH);

    gmx::NormalDistribution<real> normalDist;

    const real factor = exp(-1.0 * dt / tau);

    // number of degrees of freedom (one less if constraint on)
    int Nf = lambdaCoordinates.size();
    if (useChargeConstraints)
    {
        Nf = Nf - 1;
    }
    if (useMultiStateConstraits)
    {
        Nf = Nf - numMultiStateConstraintGroups;
    }

    // reference kinetic energy from desired temperature
    const real Ekin_ref = Nf * 0.5 * Tref * gmx::c_boltz;

    // first random number
    rng.restart(step, 0);
    const real r1 = normalDist(rng);

    // sum of gaussian squared random numbers
    real sum_r2 = 0;
    for (int j = 1; j < Nf; j++)
    {
        const real r = normalDist(rng);
        sum_r2       = sum_r2 + r * r;
    }
    const real Ekin_new = Ekin_total + (1.0 - factor) * (Ekin_ref * (r1 * r1 + sum_r2) / Nf - Ekin_total)
                          + 2.0 * r1 * std::sqrt(Ekin_ref * Ekin_total / Nf * (1.0 - factor) * factor);

    // Analytically Ek_new>=0, but we check for rounding errors (from gromacs coupling.cpp)
    real alpha = 0.0;
    if (Ekin_new > 0)
    {
        alpha = std::sqrt(Ekin_new / Ekin_total);
    }

    // scale kinetic energies and velocities with alpha
    const int gmx_unused numThreads = gmx_omp_nthreads_get(ModuleMultiThread::Update);
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (gmx::Index i = 0; i < gmx::ssize(lambdaCoordinates); i++)
    {
        auto& lambdaCoordinate = lambdaCoordinates[i];

        lambdaCoordinate.v    = alpha * lambdaCoordinate.v;
        lambdaCoordinate.x    = lambdaCoordinate.x_old + lambdaCoordinate.v * dt;
        lambdaCoordinate.ekin = alpha * alpha * lambdaCoordinate.ekin;
        lambdaCoordinate.T    = 2.0 * (lambdaCoordinate.ekin) / gmx::c_boltz;
    }

    return Ekin_new;
}

/*
 *  Update lambda and velocity using normal leap-frog
 *  - needs thermostat (collective v-rescale)
 */
void updateLambda(gmx::LambdaCoordinate* currentLambdaCoordinate,
                  gmx::ArrayRef<real>    groupPotential,
                  real                   lambdaMass,
                  bool                   isCalibrationRun,
                  const real             deltaT)
{
    if (!isCalibrationRun)
    {
        GMX_RELEASE_ASSERT(lambdaMass != 0.0,
                           "Can't have zero mass for non calibration simulations");
        const real inverseMass = currentLambdaCoordinate->bufferResidueMultiplier / lambdaMass;

        currentLambdaCoordinate->v =
                currentLambdaCoordinate->v
                + inverseMass * (-1.0) * groupPotential[currentLambdaCoordinate->groupNumber] * deltaT;
        currentLambdaCoordinate->ekin  = 0.5 * lambdaMass * gmx::square(currentLambdaCoordinate->v);
        currentLambdaCoordinate->T     = 2.0 * (currentLambdaCoordinate->ekin) / gmx::c_boltz;
        currentLambdaCoordinate->x_old = currentLambdaCoordinate->x;
        currentLambdaCoordinate->x = currentLambdaCoordinate->x + currentLambdaCoordinate->v * deltaT;
    }
}

void getConstraintCoefficients(gmx::ArrayRef<gmx::LambdaCoordinate> lambdaCoordinates,
                               int                                  numMultiStateConstraintGroups,
                               int                                  numChargeConstraintGroups)
{
    int         multiStateIt = 0;
    std::size_t pos          = 0;
    for (auto& lambdaCoordinate : lambdaCoordinates)
    {
        lambdaCoordinate.constraintCoefficients.resize(numMultiStateConstraintGroups
                                                       + numChargeConstraintGroups);

        for (int i = 0; i < numMultiStateConstraintGroups + numChargeConstraintGroups; ++i)
        {
            lambdaCoordinate.constraintCoefficients[i] = 0;
        }

        // first, parse all multistate constraints
        if (!lambdaCoordinate.constraintGroupMembers.empty())
        {
            lambdaCoordinate.constraintCoefficients[multiStateIt] = 1;
            ++pos;
            if (pos == lambdaCoordinate.constraintGroupMembers.size() + 1)
            {
                ++multiStateIt;
                pos = 0;
            }
        }

        // second, parse all charge constraints
        for (int i = 0; i < numChargeConstraintGroups; ++i)
        {
            if (i == lambdaCoordinate.chargeConstraintGroup - 1)
            {
                lambdaCoordinate.constraintCoefficients[numMultiStateConstraintGroups + i] =
                        (lambdaCoordinate.totalChargeB - lambdaCoordinate.totalChargeA)
                        * lambdaCoordinate.bufferResidueMultiplier;
            }
        }
    }
}

void getConstraintMultipliers(gmx::ArrayRef<const gmx::LambdaCoordinate> lambdaCoordinates,
                              int      numMultiStateConstraintGroups,
                              int      numChargeConstraintGroups,
                              double** constraintMultipliers)
{
    int n = numMultiStateConstraintGroups + numChargeConstraintGroups;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            constraintMultipliers[i][j] = 0.;
            for (const auto& lambdaCoordinate : lambdaCoordinates)
            {
                constraintMultipliers[i][j] += lambdaCoordinate.constraintCoefficients[i]
                                               * lambdaCoordinate.constraintCoefficients[j];
            }
        }
    }
    GMX_RELEASE_ASSERT(matrix_invert(nullptr, n, constraintMultipliers) == 0,
                       "The constraint groups should be linearly independent");
    {
    }
}

} // namespace

ConstantPH::ConstantPH(const t_inputrec& ir, int natoms, t_commrec* commrec, const gmx::MDLogger& mdlog) :
    lambdaCoordinates_(createCoordinates(ir, ir.lambdaDynamicsSimulationParameters->randomSeed(), mdlog)),
    eLambdaThermostat_(ir.lambdaDynamicsSimulationParameters->eLambdaThermostat()),
    numMultiStateConstraintGroupStates_(numMultiGroupConstraintGroupStates(lambdaCoordinates_)),
    lambdaMass_(ir.lambdaDynamicsSimulationParameters->lambdaParticleMass()),
    lambdaTau_(ir.lambdaDynamicsSimulationParameters->lambdatau()),
    lambdaNst_(ir.lambdaDynamicsSimulationParameters->lambdaNst()),
    simulationpH_(ir.lambdaDynamicsSimulationParameters->simulationpH()),
    referenceTemperature_(EI_DYNAMICS(ir.eI) ? ir.opts.ref_t[0] : 0.0_real),
    simulationDeltaT_(ir.delta_t),
    simulationLDSeed_(ir.lambdaDynamicsSimulationParameters->randomVVSeed()),
    useChargeConstraints_(ir.lambdaDynamicsSimulationParameters->useChargeConstraints()),
    useMultiStateConstraits_(ir.lambdaDynamicsSimulationParameters->useMultiStateConstraints()),
    isCalibrationRun_(ir.lambdaDynamicsSimulationParameters->isCalibrationRun()),
    commrec_(commrec)
{
    GMX_RELEASE_ASSERT(ir.lambda_dynamics, "We should only set up ConstantPH with lambda dynamics");
    linkMultiStateCoordinates(lambdaCoordinates_);
    calculateChargeConstraint<true>(lambdaCoordinates_);
    numMultiStateConstraintGroups_ = numMultiStateConstraintGroups(lambdaCoordinates_);

    getConstraintCoefficients(lambdaCoordinates_, numMultiStateConstraintGroups_,
                              maxNumChargeConstraintGroups(*ir.lambdaDynamicsSimulationParameters));
    constraintMultipliers_ =
            numMultiStateConstraintGroups_ + maxNumChargeConstraintGroups(*ir.lambdaDynamicsSimulationParameters)
                            > 0
                    ? alloc_matrix(
                            numMultiStateConstraintGroups_
                                    + maxNumChargeConstraintGroups(*ir.lambdaDynamicsSimulationParameters),
                            numMultiStateConstraintGroups_
                                    + maxNumChargeConstraintGroups(*ir.lambdaDynamicsSimulationParameters))
                    : nullptr;
    if (constraintMultipliers_)
    {
        getConstraintMultipliers(lambdaCoordinates_, numMultiStateConstraintGroups_,
                                 maxNumChargeConstraintGroups(*ir.lambdaDynamicsSimulationParameters),
                                 constraintMultipliers_);
    }

    sumOfInitialLambdaConstraints_ = lambdaConstraintStates(
            lambdaCoordinates_, numMultiStateConstraintGroups_,
            maxNumChargeConstraintGroups(*ir.lambdaDynamicsSimulationParameters));
    int i = 0;
    i     = 0;
    for (const auto& initialLambda : sumOfInitialLambdaConstraints_)
    {
        if (i < numMultiStateConstraintGroups_)
        {
            GMX_LOG(mdlog.info)
                    .appendTextFormatted("Initial value for multistate constraint group %d = %f",
                                         i + 1, initialLambda);
        }
        else
        {
            GMX_LOG(mdlog.info)
                    .appendTextFormatted("Initial value for charge constraint group %d = %f",
                                         i - numMultiStateConstraintGroups_ + 1, initialLambda);
        }
        i++;
    }
    potential_.resize(natoms);
    groupPotential_.resize(lambdaCoordinates_.size());
    outputStorage_.init(lambdaCoordinates_.size());

    // With domain decomposition all MPI ranks update all lambda coordinates

    // Atom indices are set up before step 0 with DD and at DD partitioning
}

ConstantPH::~ConstantPH()
{
    // TODO: We should actually clean up or use only C++ constructs
    if (constraintMultipliers_)
    {
        free_matrix(constraintMultipliers_);
    }
}

void ConstantPH::setLambdaCharges(gmx::ArrayRef<real> localCharges) const
{
    // printf("setting charges\n");
    // interpolate the charges for atoms that are part of lambda group
    for (auto lambdaCoordinateIt = lambdaCoordinates_.begin();
         lambdaCoordinateIt != lambdaCoordinates_.end(); ++lambdaCoordinateIt)
    {
        std::vector<real> finalCharges(lambdaCoordinateIt->chargeA.size(), 0);
        // when part of a constraint group, interpolate charges from all coordinates
        // in the group to get the final charges on the atom collection
        if (lambdaCoordinateIt->isInConstraintGroup)
        {
            // charges assume that we use State A -> all charges are the same
            // and State B -> charges for the actual state
            for (auto otherCoordIt = lambdaCoordinateIt->constraintGroupMembers.begin();
                 otherCoordIt != lambdaCoordinateIt->constraintGroupMembers.end(); ++otherCoordIt)
            {
                for (int atomIndex = 0; atomIndex < gmx::ssize(lambdaCoordinateIt->chargeA); atomIndex++)
                {
                    finalCharges[atomIndex] +=
                            (1 - (*otherCoordIt)->x) * (*otherCoordIt)->chargeA[atomIndex]
                            + (*otherCoordIt)->x * (*otherCoordIt)->chargeB[atomIndex];
                }
            }
        }
        for (int atomIndex = 0; atomIndex < gmx::ssize(lambdaCoordinateIt->chargeA); atomIndex++)
        {
            finalCharges[atomIndex] +=
                    (1 - lambdaCoordinateIt->x) * lambdaCoordinateIt->chargeA[atomIndex]
                    + lambdaCoordinateIt->x * lambdaCoordinateIt->chargeB[atomIndex];
        }
        GMX_ASSERT(lambdaCoordinateIt->localAtomIndices.size()
                           == lambdaCoordinateIt->groupChargeIndices.size(),
                   "The sizes of the atom indices and charge indices should match");
        auto groupChargeIndexIt = lambdaCoordinateIt->groupChargeIndices.begin();
        for (const int localAtomIndex : lambdaCoordinateIt->localAtomIndices)
        {
            localCharges[localAtomIndex] = finalCharges[*groupChargeIndexIt];
            groupChargeIndexIt++;
        }

        // Jump over other members of constraint group.
        lambdaCoordinateIt += lambdaCoordinateIt->constraintGroupMembers.size();
    }
    // printf("setting charges done\n");
}

void ConstantPH::updateAfterPartition(const gmx_ga2la_t* ga2la, int numLocalAtoms, int numForceAtoms)
{
    potential_.resize(numForceAtoms);
    isLambdaAtom_.resize(numForceAtoms);

    // Rebuild the atom indices
    std::fill(isLambdaAtom_.begin(), isLambdaAtom_.end(), false);
    for (auto& lambdaCoordinate : lambdaCoordinates_)
    {
        lambdaCoordinate.localAtomIndices.clear();
        lambdaCoordinate.groupChargeIndices.clear();
        lambdaCoordinate.localChargeDifferences.clear();
        auto atomIt = lambdaCoordinate.globalAtomIndices.begin();
        for (int i = 0; i < lambdaCoordinate.bufferResidueMultiplier; i++)
        {
            for (int atomIndex = 0; atomIndex < gmx::ssize(lambdaCoordinate.chargeA); atomIndex++)
            {
                // Only include HOME atoms. ga2la->find() also returns halo (imported) copies
                // (Entry.cell > 0); an atom that is home on one rank and a halo copy on another
                // would then be added to this group on BOTH ranks and double-counted in the
                // group-potential sumReduce. findHome() returns non-null only for home atoms
                // (Entry.cell == 0), so each titratable atom is counted exactly once.
                const int* localIndexPtr = (ga2la ? ga2la->findHome(*atomIt) : nullptr);
                if (ga2la == nullptr || localIndexPtr != nullptr)
                {
                    const int localIndex = (ga2la ? *localIndexPtr : *atomIt);
                    lambdaCoordinate.localAtomIndices.push_back(localIndex);
                    lambdaCoordinate.groupChargeIndices.push_back(atomIndex);
                    lambdaCoordinate.localChargeDifferences.push_back(
                            (lambdaCoordinate.chargeB[atomIndex] - lambdaCoordinate.chargeA[atomIndex]));

                    isLambdaAtom_[localIndex] = true;
                }
                atomIt++;
            }
        }
    }

    // Rebuild the local lambda atom list
    allLambdaAtoms_.clear();
    for (int i = 0; i < numLocalAtoms; ++i)
    {
        if (isLambdaAtom_[i])
        {
            allLambdaAtoms_.push_back(i);
        }
    }

    // With more than one rank, we broadcast the lambda state from the master
    // rank to the other ranks to avoid divergence due to rounding errors
    // This is likely not needed, as the MPI reduction of dV/dl gives
    // the same answer on all ranks and the integration should be indentical
    // But as this could lead to subtle errors in the results and this
    // broadcast at partitioning is cheap, we do it anyhow.
    if (commrec_ && commrec_->commMyGroup.size() > 1)
    {
        std::vector<real> state(2 * lambdaCoordinates_.size());

        if (commrec_->commMyGroup.isMainRank())
        {
            for (gmx::Index i = 0; i < gmx::ssize(lambdaCoordinates_); i++)
            {
                state[i * 2]     = lambdaCoordinates_[i].x;
                state[i * 2 + 1] = lambdaCoordinates_[i].v;
            }
        }

        gmx_bcast(gmx::ssize(state) * sizeof(real), state.data(), commrec_->commMyGroup.comm());

        if (!commrec_->commMyGroup.isMainRank())
        {
            for (gmx::Index i = 0; i < gmx::ssize(lambdaCoordinates_); i++)
            {
                lambdaCoordinates_[i].x = state[i * 2];
                lambdaCoordinates_[i].v = state[i * 2 + 1];
            }
        }
    }
}

// Returns the kinetic energy
static real kineticEnergy(const real lambdaMass, gmx::ArrayRef<const gmx::LambdaCoordinate> lambdaCoordinates)
{
    const int gmx_unused numThreads = gmx_omp_nthreads_get(ModuleMultiThread::Update);

    real ekin = 0;
#pragma omp parallel for num_threads(numThreads) schedule(static) reduction(+ : ekin)
    for (gmx::Index i = 0; i < gmx::ssize(lambdaCoordinates); i++)
    {
        const auto& lambdaCoordinate = lambdaCoordinates[i];
        ekin += 0.5_real * lambdaMass * lambdaCoordinate.v * lambdaCoordinate.v;
    }

    return ekin;
}

/* TODO
 * loop over lambda groups
 *     lambda = do_lambdadyn(...)
 * output lambda value, temperature etc.
 *
 * Returns the change in kinetic energy due to T-coupling
 */
ConstantPHLambdaEnergies ConstantPH::updateLambdas(const int64_t step)
{
    // here we should change the behaviour
    // use gmx::ListOfLists
    // printf("update lambdas\n");
    // printf("%lu\n",groupPotential_.size());

    const int gmx_unused numThreads = gmx_omp_nthreads_get(ModuleMultiThread::Update);

#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (gmx::Index i = 0; i < gmx::ssize(lambdaCoordinates_); i++)
    {
        const auto& lambdaCoordinate = lambdaCoordinates_[i];

        // printf("%d\n",lambdaCoordinate.groupNumber);
        groupPotential_[lambdaCoordinate.groupNumber] = 0;
        auto atomIt                                   = lambdaCoordinate.localAtomIndices.begin();
        for (const auto chargeDifference : lambdaCoordinate.localChargeDifferences)
        {
            if (*atomIt != -1)
            {
                groupPotential_[lambdaCoordinate.groupNumber] += potential_[*atomIt] * chargeDifference;
                atomIt++;
            }
        }
    }
    if (commrec_ && commrec_->commMyGroup.size() > 1)
    {
        commrec_->commMyGroup.sumReduce(groupPotential_.size(), groupPotential_.data());
    }
    // compute forces acting on each lambda
    real lambdaPotentialSum = 0;
#pragma omp parallel for num_threads(numThreads) schedule(static) reduction(+ : lambdaPotentialSum)
    for (gmx::Index i = 0; i < gmx::ssize(lambdaCoordinates_); i++)
    {
        auto& lambdaCoordinate = lambdaCoordinates_[i];

        computeForces(&lambdaCoordinate, groupPotential_, useMultiStateConstraits_,
                      isCalibrationRun_, referenceTemperature_, simulationpH_);
        lambdaPotentialSum += lambdaCoordinate.pot_dwp + lambdaCoordinate.pot_ref + lambdaCoordinate.pot_ph;
    }
    // total kinetic energy
    real Ekin_total = 0;

    // TODO: Compute Ekin for all thermostats

    /* Thermostat is controlled by mdp option.
     */

    // if we have thermostat = langevin
    real deltaEkin = 0;
    switch (eLambdaThermostat_)
    {
        case (eLambdaTcVRESCALE):
        {
#pragma omp parallel for num_threads(numThreads) schedule(static) reduction(+ : deltaEkin)
            for (gmx::Index i = 0; i < gmx::ssize(lambdaCoordinates_); i++)
            {
                auto& lambdaCoordinate = lambdaCoordinates_[i];

                updateLambda(&lambdaCoordinate, groupPotential_, lambdaMass_, isCalibrationRun_,
                             simulationDeltaT_);
            }

            Ekin_total = kineticEnergy(lambdaMass_, lambdaCoordinates_);

            if (!isCalibrationRun_)
            {
                real Ekin_new = tcouple_vrescale_collective(
                        lambdaCoordinates_, referenceTemperature_, lambdaTau_, simulationDeltaT_,
                        Ekin_total, numMultiStateConstraintGroups_, step, simulationLDSeed_,
                        useChargeConstraints_, useMultiStateConstraits_);
                deltaEkin  = Ekin_new - Ekin_total;
                Ekin_total = Ekin_new;
            }

            // multiple states
            if (useMultiStateConstraits_ || useChargeConstraints_)
            {
                do_constraints(lambdaCoordinates_, constraintMultipliers_, sumOfInitialLambdaConstraints_,
                               numMultiStateConstraintGroups_, simulationDeltaT_, lambdaMass_);
            }
            if (!isCalibrationRun_)
            {
                for (auto& lambdaCoordinate : lambdaCoordinates_)
                {
                    GMX_RELEASE_ASSERT((lambdaCoordinate.x < 1.15 && lambdaCoordinate.x > -0.15),
                                       "Lambda coordinate left the range for which it has been "
                                       "parametrised. Check your input parameters");
                    {
                    }
                }
            }
        }
        break;
        default: GMX_RELEASE_ASSERT(false, "How did we end up here?");
    }

    return { lambdaPotentialSum, Ekin_total, deltaEkin };
}

std::vector<int> constructMultiStateConstraintGroupIndices(gmx::ArrayRef<const gmx::LambdaDynamicsResidue> residues,
                                                           int multiStateConstraintGroup)
{
    std::vector<int> indices;
    for (int i = 0; i < gmx::ssize(residues); i++)
    {
        if (residues[i].isInMultiStateConstraintGroup()
            && residues[i].multiStateConstraintGroup() == multiStateConstraintGroup)
        {
            indices.emplace_back(i);
        }
    }
    return indices;
}
