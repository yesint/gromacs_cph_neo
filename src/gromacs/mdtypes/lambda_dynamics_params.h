/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2020,2021,2022, by the GROMACS development team, led by
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

/*! \libinternal \file
 *
 * \brief
 * Declares Lambda Dynamics parameter data types.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \inlibraryapi
 * \ingroup module_mdtypes
 */

#ifndef GMX_MDTYPES_LAMBA_DYNAMICS_PARAMS_H
#define GMX_MDTYPES_LAMBA_DYNAMICS_PARAMS_H

#include <array>
#include <optional>
#include <string>
#include <vector>
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

struct t_blocka;
struct t_inpfile;
struct t_inputrec;
struct IndexGroup;

//! Constant-pH port: warninp_t is the modern WarningHandler pointer in 2026.
class WarningHandler;
using warninp_t = WarningHandler*;

namespace gmx
{

class ISerializer;

struct LambdaCoordinate;

//! Which field it is for output
enum class CpHMDOutputSelection : int
{
    //! Lambda coordinate value.
    Coordinate,
    //! Lambda dvdl.
    Dvdl,
    //! Lambda velocity.
    Velocity,
    //! Count
    Count
};

//! Lambda coordinate information.
struct LambdaCoordinate
{
    //! initial lambda
    real x0 = 0;
    //! Actual lambda coordinate value.
    real x = 0;
    //!  Lambda coordinate for previous step.
    real x_old = 0;
    //! Lambda coordinate two steps ago.
    real x_old_old = 0;
    //! velocity of lambda.
    real v = 0;
    //! velocity of previous step
    real v_old = 0;
    //! velocity two steps ago
    real v_old_old = 0;
    //! kinetic energy of lambda
    real ekin = 0;
    //! Lambda temperature
    real T = 0;
    //! tcouple time constant
    int tau = 0;
    //! reference pKa.
    real referencePka = 0;
    //! double well barrier
    real bar = 0;
    //! dvdl of previous step
    real dvdl_old = 0;
    //! double well potential array
    std::array<real, 15> lambda_dwp;
    //! dvdl from environment
    real dvdl_pot = 0;
    //! dvdl from ph
    real dvdl_ph = 0;
    //! dvdl from double well potential
    real dvdl_dwp = 0;
    //! dvdl from reference free energy
    real dvdl_ref = 0;
    //! potential from ph
    real pot_ph = 0;
    //! potential from double well potential
    real pot_dwp = 0;
    //! potential from reference free energy
    real pot_ref = 0;
    //! Group number
    int groupNumber = -1;
    //! If this coordinate is part of a constraint group.
    bool isInConstraintGroup = false;
    //! Charge constraint group index.
    int chargeConstraintGroup = 0;
    //! Vector of pointers for all lambdas in a multi state constraint group.
    std::vector<LambdaCoordinate*> constraintGroupMembers;
    //! Proxy for lambda state for charge constraint. Equal to x for non-multi-state-constraints.
    real lambdaChargeConstraint = 0;
    //! Global atom indices used.
    ArrayRef<const int> globalAtomIndices;
    //! Local atom indices used.
    std::vector<int> localAtomIndices;
    //! Group charge indices, indexes into chargeA/B with same index as goes into localAtomIndices.
    std::vector<int> groupChargeIndices;
    //! Local charge differences.
    std::vector<real> localChargeDifferences;
    //! Charge state A.
    ArrayRef<const real> chargeA;
    //! Charge state B.
    ArrayRef<const real> chargeB;
    //! Accumulated charge for state A, for calculating effective charge constraint.
    real totalChargeA;
    //! Accumulated charge for state B, for calculating effective charge constraint.
    real totalChargeB;
    //! dvdl coefficients.
    ArrayRef<const real> dvdlCoefficients;
    //! Maximum number in power series.
    int maximumNumberPowerSeries;
    //! array of constraint coefficients
    std::vector<real> constraintCoefficients;
    //! Is this a buffer residue?
    bool isBufferResidue = false;
    //! Buffer residue multiplier.
    int bufferResidueMultiplier = 0;
};

//! Parameters for a lambda dynamics residue.
class LambdaDynamicsResidue
{
public:
    //! Constructor from input rec
    LambdaDynamicsResidue(std::vector<t_inpfile>* inp, const std::string& prefix, int state, warninp_t wi, bool bComment);
    //! Constructor from serializer
    explicit LambdaDynamicsResidue(ISerializer* serializer);

    //! Move constructor.
    LambdaDynamicsResidue(LambdaDynamicsResidue&&) = default;
    //! Move assignment operator.
    LambdaDynamicsResidue& operator=(LambdaDynamicsResidue&&) = default;
    //! Delete copy constructor.
    LambdaDynamicsResidue(const LambdaDynamicsResidue&) = delete;
    //! Delete copy assignment.
    LambdaDynamicsResidue& operator=(const LambdaDynamicsResidue&) = delete;

    //! Write data to serializer.
    void serialize(ISerializer* serializer);

    //! Access to residue name.
    const std::string& residueName() const { return residueName_; }
    //! Access to number of states.
    int residueNStates() { return residueNStates_; }
    //! Access to dvdl coefficients.
    ArrayRef<const real> dvdlCoefficients() const { return dvdlCoefficients_; }
    //! Access to maximum number in power series.
    int maximumNumberPowerSeries() const { return maximumNumberPowerSeries_; }

    //! Calculate maximum number in power series.
    void setMaximumNumberPowerSeries(int maximumNumberPowerSeries, warninp_t wi);
    //! Calculate maximum number in power series.
    void setMultistateConstraintGroup(int multiStateConstraintGroup);

    //! Access to reference pka value.
    real referencePka() const { return referencePka_; }
    //! Access to A state charges.
    ArrayRef<const real> chargeA() const { return chargeA_; }
    //! Access to B state charges.
    ArrayRef<const real> chargeB() const { return chargeB_; }
    //! Are we part of a multi state constraint group?
    bool isInMultiStateConstraintGroup() const { return multiStateConstraintGroup_.has_value(); }
    //! Number of out multi state constraint group.
    int multiStateConstraintGroup() const
    {
        return isInMultiStateConstraintGroup() ? multiStateConstraintGroup_.value() : -1;
    }

private:
    //! Name of residue, for output.
    std::string residueName_;
    //! Number of states in the group.
    int residueNStates_;
    //! Parametrization coefficients;
    std::vector<real> dvdlCoefficients_;
    //! Maximum number in power series.
    int maximumNumberPowerSeries_;
    //! Reference pKa value.
    real referencePka_;
    //! Charges for residue state A.
    std::vector<real> chargeA_;
    //! Charges for residue state B.
    std::vector<real> chargeB_;
    //! Multi state constraint group for this residue.
    std::optional<int> multiStateConstraintGroup_;
};

/*!\brief Parameters for an atom collection in lambda dynamics.
 *
 * Contains information about atoms that are controlled by a single lambda.
 * Also states if residue is charge constraint buffer particle or not.
 */
class LambdaDynamicsAtomCollection
{
public:
    //! Constructor from input rec
    LambdaDynamicsAtomCollection(std::vector<t_inpfile>*  inp,
                                 const std::string&       prefix,
                                 bool                     useChargeConstraints,
                                 std::vector<std::string> lambdaGroupTypeNames,
                                 warninp_t                wi,
                                 bool                     bComment);
    //! Constructor from serializer
    explicit LambdaDynamicsAtomCollection(ISerializer* serializer);

    //! Move constructor.
    LambdaDynamicsAtomCollection(LambdaDynamicsAtomCollection&&) = default;
    //! Move assignment operator.
    LambdaDynamicsAtomCollection& operator=(LambdaDynamicsAtomCollection&&) = default;
    //! Delete copy constructor.
    LambdaDynamicsAtomCollection(const LambdaDynamicsAtomCollection&) = delete;
    //! Delete copy assignment.
    LambdaDynamicsAtomCollection& operator=(const LambdaDynamicsAtomCollection&) = delete;

    //! Set up the correct atom indices from the index group name.
    void setAtomIndices(ArrayRef<const IndexGroup> indexGroups, warninp_t wi);

    //! Write datastructure to serializer
    void serialize(ISerializer* serializer);
    //! Access collection name.
    const std::string& collectionName() const { return collectionName_; }
    //! Access to index for corresponding LambdaDynamicsResidue.
    int lambdaResidueIndex() const { return lambdaResidueIndex_; }
    //! Access to atom indices.
    ArrayRef<const int> atomIndicies() const { return atomIndicies_; }
    //! Access to barrier.
    real barrier() const { return barrier_; }
    //! Need to know if we are in a constraint group?
    bool isInConstraintGroup() const { return chargeConstraintGroup_.has_value(); }
    //! Need to know which constraint group?
    int chargeConstraintGroup() const
    {
        return isInConstraintGroup() ? chargeConstraintGroup_.value() : -1;
    }
    //! Need to know if this is a buffer residue?
    bool isBufferResidue() const { return isBufferResidue_; }
    //! Need to know how many multipliers the buffer residue has?
    int bufferResidueMultiplier() const { return isBufferResidue() ? bufferResidueMultiplier_ : 1; }
    //! Read only handle to lambda coordinate info.
    ArrayRef<const real> initialLambda() const { return initialLambdas_; }

private:
    //! Name of collection, for output. Usually residue name.
    std::string collectionName_;
    //! Which entry in LambdaDynamicsResidue represents this collection.
    int lambdaResidueIndex_;
    //! Holds the name of the index group used to populate atomIndices_.
    std::string indexGroupName_;
    //! Global atom numbers for this collection.
    std::vector<int> atomIndicies_;
    //! Constraint group for this collection.
    std::optional<int> chargeConstraintGroup_;
    //! Barrier for crossing between states.
    real barrier_;
    //! Is this collection a buffer residue?
    bool isBufferResidue_;
    //! Initial lambda coordinate value.
    std::vector<real> initialLambdas_;
    //! Multiplier for having multiple residues grouped together for buffers.
    int bufferResidueMultiplier_;
};

//! Parameters for LambdaDynamics.
class LambdaDynamicsSimulationParameters
{
public:
    //! Constructor from input rec
    LambdaDynamicsSimulationParameters(std::vector<t_inpfile>* inp, warninp_t wi);
    //! Constructor from serializer
    explicit LambdaDynamicsSimulationParameters(ISerializer* serializer);

    //! Move constructor.
    LambdaDynamicsSimulationParameters(LambdaDynamicsSimulationParameters&&) = default;
    //! Move assignment operator.
    LambdaDynamicsSimulationParameters& operator=(LambdaDynamicsSimulationParameters&&) = default;
    //! Delete copy constructor.
    LambdaDynamicsSimulationParameters(const LambdaDynamicsSimulationParameters&) = delete;
    //! Delete copy assignment.
    LambdaDynamicsSimulationParameters& operator=(const LambdaDynamicsSimulationParameters&) = delete;

    //! Write datastructure to serializer
    void serialize(ISerializer* serializer);
    //! Access to simulation pH.
    real simulationpH() const { return simulationpH_; }
    //! Access to integrator type.
    int eLambdaThermostat() const { return eLambdaThermostat_; }
    //! Read only access to lambda dynamics residues.
    ArrayRef<const LambdaDynamicsResidue> lambdaResidues() const { return lambdaResidues_; }
    //! Read only access to atom collection.
    ArrayRef<const LambdaDynamicsAtomCollection> lambdaAtomsCollections() const
    {
        return lambdaAtomsCollections_;
    }
    //! Read/Write access to atom collection.
    ArrayRef<LambdaDynamicsAtomCollection> lambdaAtomsCollections()
    {
        return lambdaAtomsCollections_;
    }
    //! Access to lambda mass.
    real lambdaParticleMass() const { return lambdaParticleMass_; }
    //! Access to update interval.
    int lambdaNst() const { return lambdaNst_; }
    //! Access to lambda tau.
    real lambdatau() const { return lambdaTau_; }
    //! Need to know about multi state constraints?
    bool useMultiStateConstraints() const { return useMultiStateConstraints_; }
    //! Need to know about charge constraints?
    bool useChargeConstraints() const { return useChargeConstraints_; }
    //! Need to know if we are calibrating?
    bool isCalibrationRun() const { return isCalibrationRun_; }
    //! Access to seed.
    int randomSeed() const { return randomSeed_; }
    //! Access to temperature coupling seed.
    int randomVVSeed() const { return randomVVSeed_; }
    //! Access lambda constraint states data.
    ArrayRef<const int> multiStateConstraintGroupLambdaStates() const
    {
        return multiStateConstraintGroupLambdaStates_;
    }
    //! Access to constrained lambda indicies.
    ArrayRef<const int> constrainedLambdaIndicies() const { return constrainedLambdaIndicies_; }

private:
    //! Simulation pH.
    real simulationpH_;
    //! Thermostat being used.
    int eLambdaThermostat_;
    //! Vector of lambda residues.
    std::vector<LambdaDynamicsResidue> lambdaResidues_;
    //! Vector of lambda atom collections.
    std::vector<LambdaDynamicsAtomCollection> lambdaAtomsCollections_;
    //! Mass of lambda particles, for now the same for all.
    real lambdaParticleMass_;
    //! Update interval for lambda.
    int lambdaNst_;
    //! Tau for lambda.
    real lambdaTau_;
    //! Seed for velocity generation random engine.
    int randomSeed_;
    //! Seed for temperature coupling random engine.
    int randomVVSeed_;
    //! Whether we are using multi state constraints.
    bool useMultiStateConstraints_;
    //! Whether we are using charge constraints.
    bool useChargeConstraints_;
    //! Whether this simulation will be used to calibrate the dvdl parameters.
    bool isCalibrationRun_;
    //! Vector of multi state constraint group lambda states, empty when none.
    std::vector<int> multiStateConstraintGroupLambdaStates_;
    //! Vector of charge constraint lambda indicies.
    std::vector<int> constrainedLambdaIndicies_;
    //! List of lambda group type names
    std::vector<std::string> groupTypeNames_;
};

/*! \endcond */

} // namespace gmx

#endif
