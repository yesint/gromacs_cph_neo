/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2021, by the GROMACS development team, led by
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
/*! \internal \file
 * \brief
 * Tests for functionality of the constant ph module.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "gromacs/applied_forces/constant_ph/constant_ph.h"

#include <gtest/gtest.h>
#include <array>
#include <memory>
#include <vector>

#include "gromacs/domdec/ga2la.h"
#include "gromacs/domdec/mdsetup.h"
#include "gromacs/fileio/readinp.h"
#include "gromacs/fileio/warninp.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/stringstream.h"

#include "testutils/testasserts.h"
#include "testutils/topologyhelpers.h"

namespace gmx
{
namespace test
{
namespace
{

/********************************************************************
 * ConstantPHTest
 */

TEST(ConstantPHTest, InitDoubleWellPotentialBarrierZero)
{
    {
        const real barrier = 0;
        auto       dwp     = init_lambda_dwp(barrier);
        EXPECT_REAL_EQ(dwp[0], 0.3);
        EXPECT_REAL_EQ(dwp[1], 1000);
        EXPECT_REAL_EQ(dwp[2], 2.185);
        EXPECT_REAL_EQ(dwp[3], 1.645);
        EXPECT_REAL_EQ(dwp[4], 13.5);
        EXPECT_REAL_EQ(dwp[5], 0.201852);
        EXPECT_REAL_EQ(dwp[6], 0.0476433);
        EXPECT_REAL_EQ(dwp[7], -0.097055);
        EXPECT_REAL_EQ(dwp[8], 0);
        EXPECT_REAL_EQ(dwp[9], 0);
        // EXPECT_REAL_EQ(dwp[10], -1.35e32);
        EXPECT_REAL_EQ(dwp[11], 0);
        // EXPECT_REAL_EQ(dwp[12], -1.36e32);
        EXPECT_REAL_EQ(dwp[13], 0);
        // EXPECT_REAL_EQ(dwp[14], -1.35e32);
    }
}

TEST(ConstantPHTest, InitDoubleWellPotentialBarrierFive)
{
    {
        const real barrier = 5;
        auto       dwp     = init_lambda_dwp(barrier);
        EXPECT_REAL_EQ(dwp[0], 0.3);
        EXPECT_REAL_EQ(dwp[1], 1000);
        EXPECT_REAL_EQ(dwp[2], 2.185);
        EXPECT_REAL_EQ(dwp[3], 1.645);
        EXPECT_REAL_EQ(dwp[4], 13.5);
        EXPECT_REAL_EQ(dwp[5], 0.201852);
        EXPECT_REAL_EQ(dwp[6], 0.0363033);
        EXPECT_REAL_EQ(dwp[7], 0.00443207);
        EXPECT_REAL_EQ(dwp[8], 3.18892);
        EXPECT_REAL_EQ(dwp[9], 2.5);
        EXPECT_REAL_EQ(dwp[10], 0);
        EXPECT_REAL_EQ(dwp[11], 0);
        EXPECT_REAL_EQ(dwp[12], 0);
        EXPECT_REAL_EQ(dwp[13], 0);
        EXPECT_REAL_EQ(dwp[14], 0);
    }
}


class ConstantPHPartitionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        WarningHandler warnings(false, 0);
        StringInputStream input(
                "lambda-dynamics-themorstat = v-rescale\n"
                "lambda-dynamics-lambda-particle-mass = 5\n"
                "lambda-dynamics-calibration = yes\n"
                "lambda-dynamics-random-seed = 12345\n"
                "lambda-dynamics-random-vv-seed = 67890\n"
                "lambda-dynamics-number-lambda-group-types = 1\n"
                "lambda-dynamics-group-type1-name = TEST\n"
                "lambda-dynamics-group-type1-state-1-dvdl-coefficients = 0\n"
                "lambda-dynamics-group-type1state-0-charges = 0 0\n"
                "lambda-dynamics-group-type1-state-1-charges = 1 2\n"
                "lambda-dynamics-number-atom-collections = 1\n"
                "lambda-dynamics-atom-set1-name = TEST\n"
                "lambda-dynamics-atom-set1-barrier = 0\n"
                "lambda-dynamics-atom-set1-initial-lambda = 0.75\n");
        auto parameters = read_inpfile(&input, "constant-ph-test.mdp", &warnings);
        inputrec_.lambda_dynamics = true;
        inputrec_.eI              = IntegrationAlgorithm::Steep;
        inputrec_.delta_t         = 0.002;
        inputrec_.lambdaDynamicsSimulationParameters =
                std::make_unique<LambdaDynamicsSimulationParameters>(&parameters, &warnings);
        constantPH_ = std::make_unique<ConstantPH>(inputrec_, 2, nullptr, MDLogger{});
        constantPH_->lambdaCoordinates()[0].globalAtomIndices = globalAtomIndices_;
    }

    std::array<int, 2> globalAtomIndices_{0, 1};
    t_inputrec inputrec_;
    std::unique_ptr<ConstantPH> constantPH_;
};

TEST_F(ConstantPHPartitionTest, UpdatesHomeAndHaloChargesAndPotentialMask)
{
    gmx_ga2la_t mapping(2, 2);
    mapping.insert(0, {0, 0});
    mapping.insert(1, {1, 1});
    constantPH_->updateAfterPartition(&mapping, 1, 2);

    std::array<real, 2> charges{0, 0};
    constantPH_->setLambdaCharges(charges);
    EXPECT_REAL_EQ(charges[0], 0.75);
    EXPECT_REAL_EQ(charges[1], 1.5);
    EXPECT_TRUE(constantPH_->isLambdaAtom()[0]);
    EXPECT_TRUE(constantPH_->isLambdaAtom()[1]);
    ASSERT_EQ(constantPH_->allLambdaAtoms().size(), 1);
    EXPECT_EQ(constantPH_->allLambdaAtoms()[0], 0);

    constantPH_->lambdaCoordinates()[0].x = 0.25;
    constantPH_->setLambdaCharges(charges);
    EXPECT_REAL_EQ(charges[0], 0.25);
    EXPECT_REAL_EQ(charges[1], 0.5);
}

TEST_F(ConstantPHPartitionTest, ReducesOnlyHomePotentialsWithReorderedAtoms)
{
    gmx_ga2la_t mapping(2, 2);
    mapping.insert(0, {1, 1});
    mapping.insert(1, {0, 0});
    constantPH_->updateAfterPartition(&mapping, 1, 2);
    constantPH_->potential()[0] = 3;
    constantPH_->potential()[1] = 1000;

    constantPH_->updateLambdas(0);
    EXPECT_REAL_EQ(constantPH_->lambdaCoordinates()[0].dvdl_pot, 6);
}

TEST_F(ConstantPHPartitionTest, RepartitionDropsAbsentAtomsAndOldMappings)
{
    constantPH_->updateAfterPartition(nullptr, 2, 2);
    gmx_ga2la_t mapping(2, 1);
    mapping.insert(1, {0, 0});
    constantPH_->updateAfterPartition(&mapping, 1, 1);

    std::array<real, 1> charges{0};
    constantPH_->setLambdaCharges(charges);
    EXPECT_REAL_EQ(charges[0], 1.5);
    EXPECT_EQ(constantPH_->isLambdaAtom().size(), 1);
    EXPECT_EQ(constantPH_->lambdaCoordinates()[0].localAtomIndices.size(), 1);
    constantPH_->potential()[0] = 3;
    constantPH_->updateLambdas(0);
    EXPECT_REAL_EQ(constantPH_->lambdaCoordinates()[0].dvdl_pot, 6);
}

TEST_F(ConstantPHPartitionTest, HaloOnlyRankContributesNoGroupPotential)
{
    gmx_ga2la_t mapping(2, 2);
    mapping.insert(0, {0, 1});
    mapping.insert(1, {1, 1});
    constantPH_->updateAfterPartition(&mapping, 0, 2);
    constantPH_->potential()[0] = 1000;
    constantPH_->potential()[1] = 1000;

    constantPH_->updateLambdas(0);
    EXPECT_REAL_EQ(constantPH_->lambdaCoordinates()[0].dvdl_pot, 0);
    EXPECT_TRUE(constantPH_->allLambdaAtoms().empty());
    EXPECT_TRUE(constantPH_->isLambdaAtom()[0]);
    EXPECT_TRUE(constantPH_->isLambdaAtom()[1]);
}

TEST_F(ConstantPHPartitionTest, SingleRankIncludesAllAtomPotentials)
{
    constantPH_->updateAfterPartition(nullptr, 2, 2);
    constantPH_->potential()[0] = 3;
    constantPH_->potential()[1] = 5;

    constantPH_->updateLambdas(0);
    EXPECT_REAL_EQ(constantPH_->lambdaCoordinates()[0].dvdl_pot, 13);
    EXPECT_EQ(constantPH_->allLambdaAtoms().size(), 2);
}

TEST_F(ConstantPHPartitionTest, AtomSetupRefreshesPotentialViewAfterResize)
{
    t_forcerec forceRecord;
    forceRecord.ic = std::make_unique<interaction_const_t>();
    forceRecord.constantPH = constantPH_.get();
    forceRecord.electrostaticPotential = constantPH_->potential();

    for (const int numWaters : {1, 8, 1})
    {
        gmx_mtop_t topology;
        addNWaterMolecules(&topology, numWaters);
        topology.finalize();
        gmx_localtop_t localTopology(topology.ffparams);
        auto mdAtoms = makeMDAtoms(nullptr, topology, inputrec_, false);
        forceRecord.natoms_force = topology.natoms;

        mdAlgorithmsSetupAtomData(nullptr, inputrec_, topology, &localTopology, &forceRecord,
                                  nullptr, mdAtoms.get(), nullptr, nullptr, nullptr);

        ASSERT_EQ(forceRecord.electrostaticPotential.data(), constantPH_->potential().data());
        ASSERT_EQ(forceRecord.electrostaticPotential.size(), constantPH_->potential().size());
        EXPECT_EQ(forceRecord.electrostaticPotential.size(), topology.natoms);
        forceRecord.electrostaticPotential[topology.natoms - 1] = 7;
        EXPECT_REAL_EQ(constantPH_->potential()[topology.natoms - 1], 7);
    }
}

} // namespace
} // namespace test
} // namespace gmx
