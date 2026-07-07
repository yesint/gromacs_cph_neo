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

#include "testutils/testasserts.h"

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


} // namespace
} // namespace test
} // namespace gmx
