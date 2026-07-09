/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019,2020,2021,2022, by the GROMACS development team, led by
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

#ifndef GMX_MDLIB_CONSTANT_PH_H
#define GMX_MDLIB_CONSTANT_PH_H

#include <array>
#include <memory>
#include <vector>

#include "gromacs/mdrunutility/handlerestart.h"
#include "gromacs/mdtypes/lambda_dynamics_params.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/real.h"
#include "gromacs/random/threefry.h"

class gmx_ga2la_t;
struct t_enxframe;
struct t_inputrec;
struct t_mdatoms;
struct t_commrec;

namespace gmx
{
class LambdaDynamicsSimulationParameters;
class MDLogger;
} // namespace gmx

struct ConstantPHLambdaEnergies
{
    real lambdaPotentialSum = 0;
    real EkinTotal          = 0;
    real deltaEkin          = 0;
};

//! Make multi state indices available
std::vector<int> constructMultiStateConstraintGroupIndices(gmx::ArrayRef<const gmx::LambdaDynamicsResidue> residues,
                                                           int multiStateConstraintGroup);


/*! \libinternal
 * \brief Helper to write data to output file
 */
class ConstantPHOutputStorage
{
public:
    void                init(int numLambdas);
    gmx::ArrayRef<real> accessOutput(int index);

    gmx::ArrayRef<const real> accessOutput(int index) const;

private:
    std::vector<gmx::EnumerationArray<gmx::CpHMDOutputSelection, real>> output_;
};

/*! \libinternal
 * \brief This holds the constant pH data and provides methods for mdrun to interact with
 */
class ConstantPH
{
public:
    ConstantPH(const t_inputrec& ir, int natoms, t_commrec* commrec, const gmx::MDLogger& mdlog);

    ~ConstantPH();

    //! Write data to energy file.
    void writeToEnergyFrame(int64_t step, t_enxframe* frame);

    //! Return the buffer to add electrostatic potential contributions to, to beindexed by local atom index
    gmx::ArrayRef<real> potential() { return potential_; }
    //! Return the dvdl acting on whole lambda group
    gmx::ArrayRef<real> groupPotential() { return groupPotential_; }
    //! Return local atoms that are being used for lambda dynamics calculations. Used for PME
    gmx::ArrayRef<const int> allLambdaAtoms() const { return allLambdaAtoms_; }
    //! Return a vector of booleans which tells whether atoms are subject to lambda dynamics.
    const std::vector<bool>& isLambdaAtom() const { return isLambdaAtom_; }

    //! Sets charges for particles coupled to lambda's, indexed by local index
    void setLambdaCharges(gmx::ArrayRef<real> localCharges) const;

    /*! \brief Set the communicator used for multi-rank (DD) reductions.
     *
     * ConstantPH is constructed early (before the commrec exists) so a checkpoint can
     * populate its lambda coordinates. The commrec is only needed at run time, for the
     * cross-domain group-potential reduction in updateLambdas() and the lambda-state
     * broadcast in updateAfterPartition(). Inject it once it is available. Passing nullptr
     * (or a single-rank commrec) keeps the single-rank behaviour. */
    void setCommrec(t_commrec* commrec) { commrec_ = commrec; }

    //! Updates localAtomIndices, isLambda, allLambdaAtoms after partition
    //
    // Without DD, pass nullptr for ga2la
    void updateAfterPartition(const gmx_ga2la_t* ga2la, int numLocalAtoms, int numForceAtoms);

    //! Update the lambda variables using the computed potential
    //
    // Returns the lambda potential energy, kinetic energy and change in kinetic energy due to T-coupling
    ConstantPHLambdaEnergies updateLambdas(int64_t step);

    //! Access to populate data from checkpoint.
    gmx::ArrayRef<gmx::LambdaCoordinate> lambdaCoordinates() { return lambdaCoordinates_; }
    //! Read only handle to coordinates.
    gmx::ArrayRef<const gmx::LambdaCoordinate> lambdaCoordinates() const
    {
        return lambdaCoordinates_;
    }
    //! Decide if we print output or not.
    bool isOutputStep(int64_t step) const;

private:
    //! All the lambda coordinates used for simulation.
    std::vector<gmx::LambdaCoordinate> lambdaCoordinates_;
    //! Handle to potential.
    std::vector<real> potential_;
    //! Global dvdl per lambda group
    std::vector<real> groupPotential_;
    //! Vector of initial lambda states summed up for charge constraint groups.
    std::vector<real> sumOfInitialLambdaConstraints_;
    //! Matrix of constraint multipliers
    double** constraintMultipliers_;
    //! Vector of all lambda atoms.
    std::vector<int> allLambdaAtoms_;
    //! Booleans indicates whether an atom is a lambda atom
    std::vector<bool> isLambdaAtom_;
    //! equal to eLambdaTcVRESCALE, ugly but effective.
    int eLambdaThermostat_ = 0;
    //! Total number of multi state constraint groups.
    int numMultiStateConstraintGroups_ = 0;
    //! Number of states for multi state constraint groups.
    int numMultiStateConstraintGroupStates_ = 0;
    //! Lambda mass.
    real lambdaMass_ = 0;
    //! Lambda tau.
    real lambdaTau_ = 0;
    //! Lambda NST.
    int lambdaNst_ = 0;
    //! Simulation pH value.
    real simulationpH_ = 0;
    //! Reference temperature.
    real referenceTemperature_ = 0;
    //! Simulation time step.
    real simulationDeltaT_ = 0;
    //! Simulation LD seed.
    real simulationLDSeed_ = 0;
    //! Using charge constraints today?
    bool useChargeConstraints_ = false;
    //! Using multi state constraints today?
    bool useMultiStateConstraits_ = false;
    //! Are we just calibrating?
    bool isCalibrationRun_ = false;
    //! Commrec
    t_commrec* commrec_;
    //! Storage for output values.
    ConstantPHOutputStorage outputStorage_;
};

void update_lambda(const t_inputrec&                    ir,
                   int64_t                              step,
                   gmx::ArrayRef<gmx::LambdaCoordinate> coordinates,
                   gmx::ArrayRef<const real>            pot);

std::array<real, 15> init_lambda_dwp(real barrier);


#endif
