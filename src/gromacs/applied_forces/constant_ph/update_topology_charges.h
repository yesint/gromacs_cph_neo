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

/*! \libinternal \file
 *
 * \brief
 * Declares a function to modify charge of titratable groups in gmx_mtop_t.
 *
 * \author Berk Hess <hess@kth.se>
 * \inlibraryapi
 */

#ifndef GMX_CONSTANT_PH_UPDATE_TOPOLOGY_CHARGES_H
#define GMX_CONSTANT_PH_UPDATE_TOPOLOGY_CHARGES_H

struct gmx_mtop_t;
struct t_inputrec;

namespace gmx
{

/*! \brief Set the charges of titratable groups in \p mtop to the initial values given in \p inputrec
 *
 * Note that when multiple copies of a moleculetype with titriable groups
 * are present, the initial state for each group in all copies has to be
 * the same. If this is not the case, a fatal error is issued.
 *
 * \param[in,out] mtop      Molecular topology to modify
 * \param[in,out] inputrec  The input record including the constant-pH settings
 */
void updateTopologyChargerForConstantPH(gmx_mtop_t* mtop, const t_inputrec& inputrec);

} // namespace gmx

#endif /* GMX_CONSTANT_PH_UPDATE_TOPOLOGY_CHARGES_H */
