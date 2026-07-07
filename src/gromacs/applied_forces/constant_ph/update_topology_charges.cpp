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
/*! \file
 * \brief
 * Implements updating of charges in gmx_mtop_t for constant-pH
 *
 * \author Berk Hess <hess@kth.se>
 */

#include "update_topology_charges.h"

#include <vector>

#include "constant_ph.h"

#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/logger.h"

namespace gmx
{

void updateTopologyChargerForConstantPH(gmx_mtop_t* mtop, const t_inputrec& inputrec)
{
    const ConstantPH constantPH(inputrec, mtop->natoms, nullptr, MDLogger());

    std::vector<bool> chargeIsSet(mtop->natoms, false);
    std::vector<real> charge(mtop->natoms, 0.0_real);

    for (const LambdaCoordinate& coord : constantPH.lambdaCoordinates())
    {
        // If this coordinate is a buffer with multiple copies
        // with identical charges, the charge(s) are stored for
        // one copy only. So we should index module numCharges.
        const int numCharges = coord.chargeA.ssize();
        for (gmx::Index i = 0; i < coord.globalAtomIndices.ssize(); i++)
        {
            const int atomIndex    = coord.globalAtomIndices[i];
            chargeIsSet[atomIndex] = true;
            const int chargeIndex  = i % numCharges;
            charge[atomIndex] += (1 - coord.x0) * coord.chargeA[chargeIndex]
                                 + coord.x0 * coord.chargeB[chargeIndex];
        }
    }

    std::vector<std::vector<std::pair<int, int>>> moltypeRepeatsList(mtop->moltype.size());


    int atomOffset = 0;
    for (const gmx_molblock_t& mb : mtop->molblock)
    {
        if (mb.nmol > 0)
        {
            moltypeRepeatsList[mb.type].push_back({ atomOffset, mb.nmol });
            atomOffset += mb.nmol * mtop->moltype[mb.type].atoms.nr;
        }
    }

    for (gmx::Index moltypeIndex = 0; moltypeIndex < gmx::ssize(mtop->moltype); moltypeIndex++)
    {
        const auto& moltypeRepeats = moltypeRepeatsList[moltypeIndex];
        t_atoms&    atoms          = mtop->moltype[moltypeIndex].atoms;
        for (int atomIndex = 0; atomIndex < atoms.nr; atomIndex++)
        {
            bool changed = false;
            for (const auto& moltypeRepeat : moltypeRepeats)
            {
                for (int i = 0; i < moltypeRepeat.second; i++)
                {
                    const int globalAtomIndex = moltypeRepeat.first + i * atoms.nr + atomIndex;
                    if (chargeIsSet[globalAtomIndex])
                    {
                        if (!changed)
                        {
                            atoms.atom[atomIndex].q  = charge[globalAtomIndex];
                            atoms.atom[atomIndex].qB = charge[globalAtomIndex];
                            changed                  = true;
                        }
                        else if (std::abs(atoms.atom[atomIndex].q - charge[globalAtomIndex]) > 1e-6_real)
                        {
                            gmx_fatal(
                                    FARGS,
                                    "We do not allow initializing protonation states of the same "
                                    "group in different copies of a molecule to different values");
                        }
                    }
                }
            }
        }
    }
}

} // namespace gmx
