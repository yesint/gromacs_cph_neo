/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2016,2017 by the GROMACS development team.
 * Copyright (c) 2018,2019,2020,2021,2022, by the GROMACS development team, led by
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
#include <numeric>
#include "gmxpre.h"

#include "read_params.h"

#include "gromacs/applied_forces/constant_ph/constant_ph.h"
#include "gromacs/gmxpreprocess/readir.h"
#include "gromacs/fileio/readinp.h"
#include "gromacs/fileio/warninp.h"
#include "gromacs/math/units.h"
#include "gromacs/math/utilities.h"
#include "gromacs/utility/vec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/lambda_dynamics_params.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/random/seed.h"
#include "gromacs/topology/index.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/serialization/iserializer.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

/* Constant-pH port shims: 2026 replaced the free warning_error()/warning_note()
 * functions with WarningHandler methods, and yesno_names is no longer a global. */
namespace
{
inline void warning_error(WarningHandler* wi, std::string_view msg)
{
    wi->addError(msg);
}
inline void warning_note(WarningHandler* wi, std::string_view msg)
{
    wi->addNote(msg);
}
const char* const yesno_names[] = { "no", "yes", nullptr };

/* Ported from the fork's readinp addition: read a whitespace-separated list of
 * reals from an mdp parameter. Kept module-local to avoid patching stock readinp. */
std::vector<double> get_ereal_vector(std::vector<t_inpfile>* inp, const std::string& name, WarningHandler* wi)
{
    std::vector<t_inpfile>& inpRef = *inp;
    const int               ii     = get_einp(inp, name.c_str());
    if (ii == -1)
    {
        wi->addWarning(formatString("Parameter %s has not been found", name.c_str()));
        return {};
    }
    std::vector<double> realValues;
    for (const auto& realValueString : splitString(inpRef[ii].value_))
    {
        char*  ptr;
        double v = std::strtod(realValueString.c_str(), &ptr);
        if (*ptr != '\0')
        {
            wi->addError(formatString(
                    "Value '%s' in vector for parameter '%s' in parameter file is not a real value",
                    realValueString.c_str(), inpRef[ii].name_.c_str()));
        }
        realValues.push_back(v);
    }
    return realValues;
}
} // namespace

LambdaDynamicsResidue::LambdaDynamicsResidue(std::vector<t_inpfile>* inp,
                                             const std::string&      prefix,
                                             int                     state,
                                             warninp_t               wi,
                                             bool                    bComment)
{
    std::string opt;
    if (bComment)
    {
        printStringNoNewline(inp, "The name for this group type.");
    }

    opt          = prefix + "-name";
    residueName_ = get_estr(inp, opt, "31337");

    if (bComment)
    {
        printStringNoNewline(inp, "The number of states for this group type.");
    }

    opt             = prefix + "-n-states";
    residueNStates_ = get_eint(inp, opt, 1, wi);

    if (bComment)
    {
        printStringNoNewline(inp, "The dvdl references values for this group type");
    }
    opt = prefix + "-state-" + std::to_string(state) + "-dvdl-coefficients";
    {
        std::vector<double> temp = get_ereal_vector(inp, opt, wi);
        for (const auto& value : temp)
        {
            dvdlCoefficients_.emplace_back(value);
        }
        maximumNumberPowerSeries_ = dvdlCoefficients_.size();
    }

    if (bComment)
    {
        printStringNoNewline(inp, "Group type reference pKa");
    }
    opt           = prefix + "-state-" + std::to_string(state) + "-reference-pka";
    referencePka_ = get_ereal(inp, opt, 0, wi);

    if (bComment)
    {
        printStringNoNewline(inp, "Atomic charges for state A");
    }
    opt = prefix + "state-0-charges";
    {
        std::vector<double> temp = get_ereal_vector(inp, opt, wi);
        for (const auto& value : temp)
        {
            chargeA_.emplace_back(value);
        }
    }

    if (bComment)
    {
        printStringNoNewline(inp, "Atomic charges for state B");
    }
    opt = prefix + "-state-" + std::to_string(state) + "-charges";
    {
        std::vector<double> temp = get_ereal_vector(inp, opt, wi);
        for (const auto& value : temp)
        {
            chargeB_.emplace_back(value);
        }
    }

    if (chargeA_.size() != chargeB_.size())
    {
        warning_error(wi, gmx::formatString("Different size for charge arrays for %s", prefix.c_str()));
    }
}

void LambdaDynamicsResidue::setMaximumNumberPowerSeries(int maximumNumberPowerSeries, warninp_t wi)
{
    if (maximumNumberPowerSeries < 0)
    {
        std::string message = formatString(
                "Invalid number of dvdl coefficients are provided"
                "for group type %s",
                residueName_.c_str());
        warning_error(wi, message);
    }
    else
    {
        maximumNumberPowerSeries_ = maximumNumberPowerSeries;
    }
}

void LambdaDynamicsResidue::setMultistateConstraintGroup(int multiStateConstraintGroup)
{
    multiStateConstraintGroup_ = multiStateConstraintGroup;
}

LambdaDynamicsResidue::LambdaDynamicsResidue(ISerializer* serializer)
{
    GMX_RELEASE_ASSERT(serializer->reading(),
                       "Can not use writing serializer for creating datastructure");
    serializer->doString(&residueName_);
    serializer->doInt(&residueNStates_);
    int numDvdlCoeff = 0;
    serializer->doInt(&numDvdlCoeff);
    dvdlCoefficients_.resize(numDvdlCoeff);
    serializer->doRealArray(dvdlCoefficients_.data(), numDvdlCoeff);
    serializer->doInt(&maximumNumberPowerSeries_);
    bool hasMultiStateConstraint;
    serializer->doBool(&hasMultiStateConstraint);
    if (hasMultiStateConstraint)
    {
        int group;
        serializer->doInt(&group);
        multiStateConstraintGroup_ = group;
    }
    serializer->doReal(&referencePka_);
    // Assumes same size of charges, as enforced during grompp time.
    int numCharges = 0;
    serializer->doInt(&numCharges);
    chargeA_.resize(numCharges);
    chargeB_.resize(numCharges);
    serializer->doRealArray(chargeA_.data(), numCharges);
    serializer->doRealArray(chargeB_.data(), numCharges);
}

void LambdaDynamicsResidue::serialize(ISerializer* serializer)
{
    GMX_RELEASE_ASSERT(!serializer->reading(),
                       "Can not use reading serializer for writing datastructure");
    serializer->doString(&residueName_);
    serializer->doInt(&residueNStates_);
    int numDvdlCoeff = dvdlCoefficients_.size();
    serializer->doInt(&numDvdlCoeff);
    serializer->doRealArray(dvdlCoefficients_.data(), numDvdlCoeff);
    serializer->doInt(&maximumNumberPowerSeries_);
    bool hasMultiStateConstraint = isInMultiStateConstraintGroup();
    serializer->doBool(&hasMultiStateConstraint);
    if (hasMultiStateConstraint)
    {
        int group = multiStateConstraintGroup();
        serializer->doInt(&group);
    }
    serializer->doReal(&referencePka_);
    // Assumes same size of charges, as enforced during grompp time.
    int numCharges = chargeA_.size();
    serializer->doInt(&numCharges);
    serializer->doRealArray(chargeA_.data(), numCharges);
    serializer->doRealArray(chargeB_.data(), numCharges);
}

namespace
{
int getAtomCollectionResidueIndex(ArrayRef<std::string> lambdaGroupTypeNames, const std::string& collectionName)
{
    int index = 0;
    for (const auto& name : lambdaGroupTypeNames)
    {
        if (name == collectionName)
        {
            return index;
        }
        ++index;
    }
    return -1;
}

} // namespace

LambdaDynamicsAtomCollection::LambdaDynamicsAtomCollection(std::vector<t_inpfile>* inp,
                                                           const std::string&      prefix,
                                                           bool useChargeConstraints,
                                                           std::vector<std::string> lambdaGroupTypeNames,
                                                           warninp_t                wi,
                                                           bool                     bComment)
{
    if (bComment)
    {
        printStringNoNewline(inp, "Name for this collection");
    }

    std::string opt     = prefix + "-name";
    collectionName_     = get_estr(inp, opt, "31337");
    lambdaResidueIndex_ = getAtomCollectionResidueIndex(lambdaGroupTypeNames, collectionName_);
    if (lambdaResidueIndex_ < 0)
    {
        std::string message =
                formatString("Atom collection (%s) doesn't match any lambda group type name",
                             collectionName_.c_str());
        warning_error(wi, message);
    }

    if (useChargeConstraints)
    {
        if (bComment)
        {
            printStringNoNewline(inp, "Charge restraint group index for this collection");
        }
        opt                    = prefix + "-charge-restraint-group-index";
        chargeConstraintGroup_ = get_eint(inp, opt, -1, wi);
    }

    if (bComment)
    {
        printStringNoNewline(inp, "Barrier between states");
    }
    opt      = prefix + "-barrier";
    barrier_ = get_ereal(inp, opt, 7.5, wi);

    if (bComment)
    {
        printStringNoNewline(inp, "Name of index group for this collection");
    }
    opt             = prefix + "-index-group-name";
    indexGroupName_ = get_estr(inp, opt, "index");

    if (bComment)
    {
        printStringNoNewline(inp, "If this residue is a buffer residue or not");
    }
    opt              = prefix + "-buffer-residue";
    isBufferResidue_ = get_eeenum(inp, opt, yesno_names, wi) != 0;
    if (isBufferResidue_)
    {
        barrier_ = 0.;
    }

    if (bComment)
    {
        printStringNoNewline(inp, "Initial value for the lambda of this collection");
    }
    opt = prefix + "-initial-lambda";
    {
        std::vector<double> temp = get_ereal_vector(inp, opt, wi);
        for (const auto& value : temp)
        {
            initialLambdas_.emplace_back(value);
        }
    }

    if (isBufferResidue_)
    {
        if (bComment)
        {
            printStringNoNewline(inp, "How many multiply residues are in the group");
        }
        opt                      = prefix + "-buffer-residue-multiplier";
        bufferResidueMultiplier_ = get_eint(inp, opt, 1, wi);
        if (bufferResidueMultiplier_ < 1)
        {
            warning_error(wi, "Can't have multiplier less than 0");
        }
    }
}

void LambdaDynamicsAtomCollection::setAtomIndices(ArrayRef<const IndexGroup> indexGroups, warninp_t wi)
{
    // 2026 represents index groups as a list of IndexGroup{name, particleIndices},
    // replacing the old t_blocka/gnames pair.
    int gid = -1;
    for (int i = 0; i < ssize(indexGroups); i++)
    {
        if (gmx_strcasecmp(indexGroupName_.c_str(), indexGroups[i].name.c_str()) == 0)
        {
            gid = i;
            break;
        }
    }
    if (gid < 0)
    {
        warning_error(wi, formatString("No index group named %s found", indexGroupName_.c_str()));
        return;
    }
    const auto& particleIndices = indexGroups[gid].particleIndices;
    if (particleIndices.empty())
    {
        warning_error(wi, formatString("No atoms found for index group %s", indexGroupName_.c_str()));
    }
    atomIndicies_.assign(particleIndices.begin(), particleIndices.end());
}

LambdaDynamicsAtomCollection::LambdaDynamicsAtomCollection(ISerializer* serializer)
{
    GMX_RELEASE_ASSERT(serializer->reading(),
                       "Can not use writing serializer to create datastructure");

    serializer->doString(&collectionName_);
    serializer->doInt(&lambdaResidueIndex_);
    bool useChargeConstraints;
    serializer->doBool(&useChargeConstraints);
    if (useChargeConstraints)
    {
        int group;
        serializer->doInt(&group);
        chargeConstraintGroup_ = group;
    }
    serializer->doReal(&barrier_);
    int numAtoms = 0;
    serializer->doInt(&numAtoms);
    atomIndicies_.resize(numAtoms);
    serializer->doIntArray(atomIndicies_.data(), numAtoms);
    serializer->doBool(&isBufferResidue_);
    if (isBufferResidue_)
    {
        serializer->doInt(&bufferResidueMultiplier_);
    }
    int numInitialLambdas = 0;
    serializer->doInt(&numInitialLambdas);
    initialLambdas_.resize(numInitialLambdas);
    serializer->doRealArray(initialLambdas_.data(), numInitialLambdas);
}

void LambdaDynamicsAtomCollection::serialize(ISerializer* serializer)
{
    GMX_RELEASE_ASSERT(!serializer->reading(),
                       "Can not use reading serializer to write datastructure");
    serializer->doString(&collectionName_);
    serializer->doInt(&lambdaResidueIndex_);
    bool useChargeConstraints = isInConstraintGroup();
    serializer->doBool(&useChargeConstraints);
    if (useChargeConstraints)
    {
        int group = chargeConstraintGroup();
        serializer->doInt(&group);
    }
    serializer->doReal(&barrier_);
    int numAtoms = atomIndicies_.size();
    serializer->doInt(&numAtoms);
    serializer->doIntArray(atomIndicies_.data(), numAtoms);
    serializer->doBool(&isBufferResidue_);
    if (isBufferResidue_)
    {
        serializer->doInt(&bufferResidueMultiplier_);
    }
    int numInitialLambdas = initialLambdas_.size();
    serializer->doInt(&numInitialLambdas);
    serializer->doRealArray(initialLambdas_.data(), numInitialLambdas);
}

namespace
{
int multiStateConstraintGroupSize(ArrayRef<const LambdaDynamicsResidue> lambdaResidues, int indexGroup)
{
    int counter = 0;
    for (const auto& residue : lambdaResidues)
    {
        if (residue.isInMultiStateConstraintGroup() && residue.multiStateConstraintGroup() == indexGroup)
        {
            counter++;
        }
    }
    return counter;
}

int numPolCoefs(int order, int nVariables)
{
    if (order == 0)
    {
        return 1;
    }
    if (nVariables == 1)
    {
        return order + 1;
    }
    int nCoefficients = 0;
    for (int i = order; i >= 0; --i)
    {
        nCoefficients += numPolCoefs(order - i, nVariables - 1);
    }
    return nCoefficients;
}

int getMaxNumberPowerCoefficients(ArrayRef<const LambdaDynamicsResidue> lambdaResidues, int residueIndex)
{
    if (lambdaResidues[residueIndex].isInMultiStateConstraintGroup())
    {
        int numMultiStateConstraintGroups = multiStateConstraintGroupSize(
                lambdaResidues, lambdaResidues[residueIndex].multiStateConstraintGroup());
        int order         = 0;
        int nCoefficients = numPolCoefs(order, numMultiStateConstraintGroups - 1);

        while (nCoefficients <= gmx::ssize(lambdaResidues[residueIndex].dvdlCoefficients()))
        {
            if (nCoefficients == gmx::ssize(lambdaResidues[residueIndex].dvdlCoefficients()))
            {
                return order;
            }
            order++;
            nCoefficients = numPolCoefs(order, numMultiStateConstraintGroups - 1);
        }
        return -1;
    }

    return gmx::ssize(lambdaResidues[residueIndex].dvdlCoefficients());
}

} // namespace

LambdaDynamicsSimulationParameters::LambdaDynamicsSimulationParameters(std::vector<t_inpfile>* inp,
                                                                       warninp_t               wi)
{
    const std::string base = "lambda-dynamics";
    std::string       opt;

    printStringNoNewline(inp, "Simulation pH value to use");
    opt           = base + "-simulation-ph";
    simulationpH_ = get_ereal(inp, opt, 0, wi);

    printStringNoNewline(inp, "Thermostat being used for lambda particles");
    opt                = base + "-themorstat";
    eLambdaThermostat_ = get_eeenum(inp, opt, eLambdaTcoupl_names, wi);
    GMX_RELEASE_ASSERT(eLambdaThermostat_ != eLambdaTcLangevin,
                       "Langevin dynamics currently doesn't work in constant pH");

    printStringNoNewline(inp, "Mass for the lambda particles");
    opt                 = base + "-lambda-particle-mass";
    lambdaParticleMass_ = get_ereal(inp, opt, 0, wi);

    printStringNoNewline(inp, "Update interval for lambda");
    opt        = base + "-update-nst";
    lambdaNst_ = get_eint(inp, opt, 100, wi);

    printStringNoNewline(inp, "Tau for lambda");
    opt        = base + "-tau";
    lambdaTau_ = get_ereal(inp, opt, 0.1, wi);

    printStringNoNewline(inp, "Are multi state constraints being used?");
    opt                       = base + "-multistate-constraints";
    useMultiStateConstraints_ = get_eeenum(inp, opt, yesno_names, wi) != 0;

    printStringNoNewline(inp, "Are charge constraints being used?");
    opt                   = base + "-charge-constraints";
    useChargeConstraints_ = get_eeenum(inp, opt, yesno_names, wi) != 0;

    printStringNoNewline(inp, "Is this a calibration run?");
    opt               = base + "-calibration";
    isCalibrationRun_ = get_eeenum(inp, opt, yesno_names, wi) != 0;

    printStringNoNewline(inp, "Number of different lambda dynamics residues");
    opt                   = base + "-number-lambda-group-types";
    const int numResidues = get_eint(inp, opt, 0, wi);
    int       numGroups   = numResidues;

    printStringNoNewline(inp, "Seed for velocity generation. -1 means generate seed");
    opt                 = base + "-random-seed";
    const int inputSeed = get_eint(inp, opt, -1, wi);
    randomSeed_         = (inputSeed == -1) ? static_cast<int>(gmx::makeRandomSeed()) : inputSeed;
    fprintf(stderr, "Setting the Constant PH lambda velocity seed to %d\n", randomSeed_);

    printStringNoNewline(inp, "Seed for temperature coupling. -1 means generate seed");
    opt                   = base + "-random-vv-seed";
    const int inputVVSeed = get_eint(inp, opt, -1, wi);
    randomVVSeed_ = (inputVVSeed == -1) ? static_cast<int>(gmx::makeRandomSeed()) : inputVVSeed;
    fprintf(stderr, "Setting the Constant PH lambda temperature coupling seed to %d\n", randomVVSeed_);

    if (numResidues == 0)
    {
        warning_error(wi, "No lambda dynamics residues are set up");
    }

    int multiStateConstraintGroup = 1;
    for (int i = 0; i < numResidues; i++)
    {
        std::string prefix = base + formatString("-group-type%d", i + 1);
        lambdaResidues_.emplace_back(LambdaDynamicsResidue(inp, prefix, 1, wi, (i == 0)));
        groupTypeNames_.emplace_back(lambdaResidues_.back().residueName());

        if (lambdaResidues_.back().residueNStates() > 1)
        {
            // for each multistate residue create a note, that one of pKa should be equal to pH
            std::string warningMessage = gmx::formatString(
                    "You are multistate representation of titratable site %d. For correct behavior "
                    "the reference pKa value for one of the states has to be equal to pH.",
                    i + 1);
            warning_note(wi, warningMessage);
            lambdaResidues_.back().setMultistateConstraintGroup(multiStateConstraintGroup);
            for (int j = 2; j <= lambdaResidues_.back().residueNStates(); j++)
            {
                numGroups++;
                lambdaResidues_.emplace_back(LambdaDynamicsResidue(inp, prefix, j, wi, (i == 0)));
                groupTypeNames_.emplace_back(lambdaResidues_.back().residueName());
                lambdaResidues_.back().setMultistateConstraintGroup(multiStateConstraintGroup);
            }
            multiStateConstraintGroup++;
        }
    }

    for (int i = 0; i < numGroups; i++)
    {
        lambdaResidues_[i].setMaximumNumberPowerSeries(
                getMaxNumberPowerCoefficients(lambdaResidues_, i), wi);
    }

    printStringNoNewline(inp, "Number of atom sets to apply lambda dynamics to");
    opt                          = base + "-number-atom-collections";
    const int numAtomCollections = get_eint(inp, opt, 0, wi);

    if (numAtomCollections == 0)
    {
        warning_error(wi, "No atom sets to apply lambda dynamics to");
    }

    for (int i = 0; i < numAtomCollections; i++)
    {
        std::string prefix = base + formatString("-atom-set%d", i + 1);
        lambdaAtomsCollections_.emplace_back(LambdaDynamicsAtomCollection(
                inp, prefix, useChargeConstraints_, groupTypeNames_, wi, (i == 0)));
    }
}

LambdaDynamicsSimulationParameters::LambdaDynamicsSimulationParameters(ISerializer* serializer)
{
    GMX_RELEASE_ASSERT(serializer->reading(),
                       "Can not use writing serializer to read AWH parameters");
    serializer->doReal(&simulationpH_);
    serializer->doInt(&eLambdaThermostat_);
    serializer->doReal(&lambdaParticleMass_);
    serializer->doInt(&lambdaNst_);
    serializer->doReal(&lambdaTau_);
    serializer->doBool(&useMultiStateConstraints_);
    serializer->doBool(&useChargeConstraints_);
    serializer->doBool(&isCalibrationRun_);
    serializer->doInt(&randomSeed_);
    serializer->doInt(&randomVVSeed_);
    if (useMultiStateConstraints_)
    {
        int numMultiStateLambdaStates = 0;
        serializer->doInt(&numMultiStateLambdaStates);
        multiStateConstraintGroupLambdaStates_.resize(numMultiStateLambdaStates);
        serializer->doIntArray(multiStateConstraintGroupLambdaStates_.data(), numMultiStateLambdaStates);
    }
    int numLambdaResidues = 0;
    serializer->doInt(&numLambdaResidues);
    for (int i = 0; i < numLambdaResidues; i++)
    {
        lambdaResidues_.emplace_back(LambdaDynamicsResidue(serializer));
    }

    if (useChargeConstraints_)
    {
        int numConstrainedLambdas = 0;
        serializer->doInt(&numConstrainedLambdas);
        constrainedLambdaIndicies_.resize(numConstrainedLambdas);
        serializer->doIntArray(constrainedLambdaIndicies_.data(), numConstrainedLambdas);
    }
    int numAtomCollections = 0;
    serializer->doInt(&numAtomCollections);
    for (int i = 0; i < numAtomCollections; i++)
    {
        lambdaAtomsCollections_.emplace_back(LambdaDynamicsAtomCollection(serializer));
    }
    int numGroupTypeNames = 0;
    serializer->doInt(&numGroupTypeNames);
    groupTypeNames_.resize(numGroupTypeNames);
    for (auto& groupTypeName : groupTypeNames_)
    {
        serializer->doString(&groupTypeName);
    }
    checkLambdaParams(*this, nullptr);
}

void LambdaDynamicsSimulationParameters::serialize(ISerializer* serializer)
{
    GMX_RELEASE_ASSERT(!serializer->reading(),
                       "Can not use reading serializer to write AWH parameters");
    checkLambdaParams(*this, nullptr);
    serializer->doReal(&simulationpH_);
    serializer->doInt(&eLambdaThermostat_);
    serializer->doReal(&lambdaParticleMass_);
    serializer->doInt(&lambdaNst_);
    serializer->doReal(&lambdaTau_);
    serializer->doBool(&useMultiStateConstraints_);
    serializer->doBool(&useChargeConstraints_);
    serializer->doBool(&isCalibrationRun_);
    serializer->doInt(&randomSeed_);
    serializer->doInt(&randomVVSeed_);
    if (useMultiStateConstraints_)
    {
        int numMultiStateLambdaStates = multiStateConstraintGroupLambdaStates_.size();
        serializer->doInt(&numMultiStateLambdaStates);
        serializer->doIntArray(multiStateConstraintGroupLambdaStates_.data(), numMultiStateLambdaStates);
    }
    int numLambdaResidues = lambdaResidues_.size();
    serializer->doInt(&numLambdaResidues);
    for (auto& lambdaResidue : lambdaResidues_)
    {
        lambdaResidue.serialize(serializer);
    }

    if (useChargeConstraints_)
    {
        int numConstrainedLambdas = constrainedLambdaIndicies_.size();
        serializer->doInt(&numConstrainedLambdas);
        serializer->doIntArray(constrainedLambdaIndicies_.data(), numConstrainedLambdas);
    }

    int numAtomCollections = lambdaAtomsCollections_.size();
    serializer->doInt(&numAtomCollections);
    for (auto& lambdaAtomsCollection : lambdaAtomsCollections_)
    {
        lambdaAtomsCollection.serialize(serializer);
    }
    int numGroupTypeNames = groupTypeNames_.size();
    serializer->doInt(&numGroupTypeNames);
    for (auto& groupTypeName : groupTypeNames_)
    {
        serializer->doString(&groupTypeName);
    }
}

void checkLambdaParams(const LambdaDynamicsSimulationParameters& lambdaParams, warninp_t wi)
{
    std::string opt = "lambda-dynamics-update-nstout";
    if (lambdaParams.lambdaNst() <= 0)
    {
        auto message =
                formatString("Not writing cPHMD output with AWH (%s = %d) does not make sense",
                             opt.c_str(), lambdaParams.lambdaNst());
        warning_error(wi, message);
    }

    const auto lambdaResidues = lambdaParams.lambdaResidues();
    for (auto& lambdaAtomsCollection : lambdaParams.lambdaAtomsCollections())
    {
        {
            int numAtoms = lambdaAtomsCollection.atomIndicies().size();
            int numCharges =
                    lambdaParams.lambdaResidues()[lambdaAtomsCollection.lambdaResidueIndex()]
                            .chargeA()
                            .size()
                    * lambdaAtomsCollection.bufferResidueMultiplier();
            if (numAtoms != numCharges)
            {
                std::string message = formatString(
                        "Number of atoms in collection %s does "
                        "not match number of charges in the requested residue %d (internal name "
                        "%s. Numbers are %d and %d, respectively",
                        lambdaAtomsCollection.collectionName().c_str(),
                        lambdaAtomsCollection.lambdaResidueIndex() + 1,
                        lambdaParams.lambdaResidues()[lambdaAtomsCollection.lambdaResidueIndex()]
                                .residueName()
                                .c_str(),
                        numAtoms, numCharges);
                warning_error(wi, message);
            }
        }
        if (lambdaResidues[lambdaAtomsCollection.lambdaResidueIndex()].isInMultiStateConstraintGroup())
        {
            int initialLambdaSize         = lambdaAtomsCollection.initialLambda().ssize();
            int numConstraintGroupMembers = multiStateConstraintGroupSize(
                    lambdaResidues,
                    lambdaResidues[lambdaAtomsCollection.lambdaResidueIndex()].multiStateConstraintGroup());
            if (initialLambdaSize != numConstraintGroupMembers)
            {
                auto message = formatString(
                        "Number of initial lambda states (%d) doesn't match number of members of "
                        "multistate constraint group (%d)",
                        initialLambdaSize, numConstraintGroupMembers);
                warning_error(wi, message);
            }
        }
        else
        {
            if (lambdaAtomsCollection.initialLambda().ssize() != 1)
            {
                warning_error(wi,
                              "Number of initial lambda states need to be one without multi state "
                              "constraints");
            }
        }
    }
}

} // namespace gmx
