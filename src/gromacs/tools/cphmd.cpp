/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2022, by the GROMACS development team, led by
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
 *  \brief
 *  Tool for extracting cpHMD  data from energy files.
 *
 *  \author Paul Bauer <paul.bauer.q@gmail.com>
 */

#include "gmxpre.h"

#include "cphmd.h"

#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <string>

#include "gromacs/analysisdata/modules/plot.h"
#include "gromacs/applied_forces/constant_ph/constant_ph.h"
#include "gromacs/commandline/pargs.h"
#include "gromacs/fileio/enxio.h"
#include "gromacs/fileio/filetypes.h"
#include "gromacs/fileio/gmxfio.h"
#include "gromacs/fileio/oenv.h"
#include "gromacs/fileio/tpxio.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/options/timeunitmanager.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/gmxana/gmx_ana.h"
#include "gromacs/math/units.h"
#include "gromacs/mdtypes/lambda_dynamics_params.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/trajectory/energyframe.h"
#include "gromacs/utility/arraysize.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/fixedcapacityvector.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/path.h"
#include "gromacs/utility/programcontext.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

namespace
{

//! Strings for legends of different graphs
constexpr gmx::EnumerationArray<CpHMDOutputSelection, const char*> cpHMDGraphLegendStrings = {
    "Lambda Coordinate", "Lambda dvdl", "Lambda velocity"
};

//! Strings for file name of different graphs
constexpr gmx::EnumerationArray<CpHMDOutputSelection, const char*> cpHMDGraphFileNameStrings = {
    "coord", "dvdl", "vel"
};

/*! \brief General output file type for lambda dynamics data.
 *
 * Depending on what kind of data is written, fields are populated differently.
 */
class OutputFile
{
public:
    /*! \brief Constructor, Set the output base file name and title.
     *
     * Result is a valid object, but will produce empty output files.
     *
     * \param[in] filename     The name for output files, frame time, and possibly bias number, will
     * be added per file/frame. \param[in] baseTitle    The base title of the plot, the lambda
     * information might be added. \param[in] graphType    What kind of graph this is going to be.
     * \param[in] timeUnit     Time unit used for output file.
     * \param[in] collectionNames  Names of collections of lambdaCoordinates used for this output.
     * \param[in] firstLambda  First lambda that will be printed here.
     * \param[in] lastLambda   Last lambda printed to this file.
     */
    OutputFile(const std::string&          filename,
               const std::string&          baseTitle,
               CpHMDOutputSelection        graphType,
               TimeUnit                    timeUnit,
               ArrayRef<const std::string> collectionNames,
               int                         firstLambda,
               int                         lastLambda);

    /*! Clean up file pointer when closing.
     */
    ~OutputFile();
    /*! \brief Gets access to output file for a collection lambdas.
     *
     * If file is not yet open, opens it and prints title and legends.
     *
     * \param[in] oenv  The output environment.
     */
    void openLambdaOutputFile(const gmx_output_env_t* oenv);

    /*! \brief Writes data selected from \p block to file.
     *
     * \param[in] block          The energy block with the data to write.
     */
    void writeData(const t_enxblock& block, double time) const;

private:
    std::string          baseFilename_; /**< Base of the output file name. */
    std::string          title_;        /**< Title for the graph. */
    CpHMDOutputSelection graphType_;    /**< Which kind of output we are writing. */
    TimeUnit             timeUnit_;     /**< Time unit used for plot. */
    real                 xscale_;
    int                  numLambdas_;
    int                  firstLambda_;
    FILE*                outputFilePointer_ = nullptr;
    bool                 fileIsOpen_        = false;

    std::vector<std::string> legend_; /**< Legends for the output */
    std::string              xLabel_; /**< Label for the x-axis. */
    std::string              yLabel_; /**< Label for the y-axis. */
};

/*! \brief All options and meta-data needed for the cpHMD output */
class CpHMDReader
{
public:
    //! Constructor
    CpHMDReader(ArrayRef<const LambdaDynamicsResidue>        lambdaResidues,
                ArrayRef<const LambdaDynamicsAtomCollection> lambdaAtoms,
                ArrayRef<CpHMDOutputSelection>               graphTypes,
                TimeUnit                                     timeUnit,
                const std::string&                           baseFilename,
                int                                          maxLambdaPerPlot);

    //! Open and prepare all output files.
    void openAllOutputFiles(const gmx_output_env_t* oenv);
    //! Extract and print cpHMD data for a cpHMD block of one energy frame
    void processCpHMDFrame(const t_enxblock& block, double time) const;

private:
    //! Storage for output graphs
    gmx::EnumerationArray<CpHMDOutputSelection, std::vector<OutputFile>> files_;
};

/*! \brief Constructs a legend for a standard awh output file */
std::vector<std::string> makeLegend(ArrayRef<const std::string> collectionNames, int firstLambda)
{
    std::vector<std::string> legend;
    int                      lambdaNum = firstLambda;
    /* Give legends to dimensions higher than the first */
    for (const auto& collectionName : collectionNames)
    {
        legend.emplace_back(gmx::formatString("Atom Collection %s, index (%d)",
                                              collectionName.c_str(), lambdaNum + 1));
        lambdaNum++;
    }
    return legend;
}

OutputFile::OutputFile(const std::string&          filename,
                       const std::string&          baseTitle,
                       CpHMDOutputSelection        graphType,
                       TimeUnit                    timeUnit,
                       ArrayRef<const std::string> collectionNames,
                       int                         firstLambda,
                       int                         lastLambda) :
    graphType_(graphType),
    timeUnit_(timeUnit),
    numLambdas_(lastLambda - firstLambda + 1),
    firstLambda_(firstLambda)
{
    baseFilename_ = filename;
    title_        = baseTitle;
    if (firstLambda != lastLambda)
    {
        baseFilename_ = gmx::concatenateBeforeExtension(
                                baseFilename_, formatString("-%d-%d", firstLambda + 1, lastLambda + 1))
                                .string();
        title_ += formatString(" %s for Lambda %d - %d", cpHMDGraphLegendStrings[graphType],
                               firstLambda + 1, lastLambda + 1);
    }
    else
    {
        baseFilename_ =
                gmx::concatenateBeforeExtension(baseFilename_, formatString("-%d", firstLambda + 1))
                        .string();
        title_ += formatString(" %s for Lambda %d", cpHMDGraphLegendStrings[graphType], firstLambda + 1);
    }
    gmx::TimeUnitManager manager(timeUnit_);
    xLabel_ = formatString("Time (%s)", manager.timeUnitAsString());
    xscale_ = manager.inverseTimeScaleFactor();
    yLabel_ = cpHMDGraphLegendStrings[graphType];
    legend_ = makeLegend(collectionNames, firstLambda);
}

OutputFile::~OutputFile()
{
    if (fileIsOpen_)
    {
        gmx_ffclose(outputFilePointer_);
    }
}

CpHMDReader::CpHMDReader(ArrayRef<const LambdaDynamicsResidue>        lambdaResidues,
                         ArrayRef<const LambdaDynamicsAtomCollection> lambdaAtoms,
                         ArrayRef<CpHMDOutputSelection>               graphTypes,
                         TimeUnit                                     timeUnit,
                         const std::string&                           baseFilename,
                         int                                          maxLambdaPerPlot)
{
    // the whole loop assumes the flat construction of the lambda coordinates
    // from the collections and residues the same way as the actual code does
    std::vector<std::string> collectionNames;
    for (const auto& atomCollection : lambdaAtoms)
    {
        const std::string& partOne       = atomCollection.collectionName();
        const auto&        lambdaResidue = lambdaResidues[atomCollection.lambdaResidueIndex()];

        // We need to get the correct residue for the multi state constraint case
        // here. So we make a temp vector of the residues and loop over this one.
        // In case of no multi state constraints we just have a single entry at the current index.
        const auto multiStateConstraintGroupIndices =
                lambdaResidue.isInMultiStateConstraintGroup()
                        ? constructMultiStateConstraintGroupIndices(
                                  lambdaResidues, lambdaResidue.multiStateConstraintGroup())
                        : std::vector<int>(1, atomCollection.lambdaResidueIndex());

        for (const auto& residueIndex : multiStateConstraintGroupIndices)
        {
            const std::string partTwo = formatString(
                    "%s %s", partOne.c_str(), lambdaResidues[residueIndex].residueName().c_str());
            collectionNames.emplace_back(partTwo);
        }
    }
    const int maxLambdas           = collectionNames.size();
    int       currentLambdasInPlot = 0;

    for (int index = 0; index < maxLambdas; ++index)
    {
        // open new file if needed
        if (index == 0 || currentLambdasInPlot == maxLambdaPerPlot)
        {
            currentLambdasInPlot                          = 0;
            ArrayRef<const std::string> names             = collectionNames;
            const int                   firstLambdaInPlot = index;
            const int                   lastLambda        = index + maxLambdaPerPlot;
            const int lastLambdaInPlot = (lastLambda > maxLambdas) ? maxLambdas - 1 : lastLambda - 1;
            const int arraySize = (lastLambda > maxLambdas) ? (maxLambdas - index) : maxLambdaPerPlot;
            for (auto graphType : graphTypes)
            {
                std::string fullName =
                        gmx::concatenateBeforeExtension(
                                baseFilename, formatString("-%s", cpHMDGraphFileNameStrings[graphType]))
                                .string();

                files_[graphType].emplace_back(fullName, "Constant pH", graphType, timeUnit,
                                               names.subArray(firstLambdaInPlot, arraySize),
                                               firstLambdaInPlot, lastLambdaInPlot);
            }
        }
        currentLambdasInPlot++;
    }
}

void OutputFile::openLambdaOutputFile(const gmx_output_env_t* oenv)
{
    if (!fileIsOpen_)
    {
        outputFilePointer_ = xvgropen(baseFilename_.c_str(), title_.c_str(), xLabel_, yLabel_, oenv);
        xvgrLegend(outputFilePointer_, legend_, oenv);
        fileIsOpen_ = true;
    }
}

/*! \brief Prints data selected by \p outputFile from \p block to \p fp */
void OutputFile::writeData(const t_enxblock& block, double time) const
{
    GMX_ASSERT(fileIsOpen_, "Need to have open output files for writing");
    fprintf(outputFilePointer_, "%8.4f", time);
    for (int i = 0; i < numLambdas_; ++i)
    {
        fprintf(outputFilePointer_, "    %8.4f",
                block.sub[firstLambda_ + i].fval[static_cast<int>(graphType_)]);
    }
    fprintf(outputFilePointer_, "\n");
}

void CpHMDReader::openAllOutputFiles(const gmx_output_env_t* oenv)
{
    for (auto& graphType : files_)
    {
        for (auto& file : graphType)
        {
            file.openLambdaOutputFile(oenv);
        }
    }
}

void CpHMDReader::processCpHMDFrame(const t_enxblock& block, double time) const
{
    /* We look for AWH data every energy frame and count the no of AWH frames found. We only extract every 'skip' AWH frame. */

    for (auto& graphType : files_)
    {
        for (auto& file : graphType)
        {
            file.writeData(block, time);
        }
    }
}

class CpHMD : public ICommandLineOptionsModule
{
public:
    CpHMD() {}

    // From ICommandLineOptionsModule
    void init(CommandLineModuleSettings* /*settings*/) override {}

    void initOptions(IOptionsContainer* options, ICommandLineOptionsModuleSettings* settings) override;

    void optionsFinished() override;

    int run() override;

private:
    //! Commandline options
    //! \{
    bool     bStartTimeSet_ = false;
    bool     bEndTimeSet_   = false;
    double   startTime_;
    double   endTime_;
    TimeUnit timeUnit_         = TimeUnit::Default;
    int      maxLambdaPerPlot_ = 4;
    bool     plotCoordinate_   = true;
    bool     plotDvdl_         = false;
    bool     plotVelocity_     = false;
    //! \}
    //! Commandline file options
    //! \{
    std::string inputTprFilename_;
    std::string inputEnergyFilename_;
    std::string outputFileBaseName_;
    //! \}
    //! Stuff for time unit
    std::shared_ptr<TimeUnitBehavior> timeUnitBehavior_;
};

void CpHMD::initOptions(IOptionsContainer* options, ICommandLineOptionsModuleSettings* settings)
{
    const char* desc[] = {
        "[THISMODULE] analyses and plots results from simulations run",
        "using the constant pH implementation. Users can plot the lambda coordinate values,",
        "lambda velocities and lambda dvdl from energy ([REF].edr[ref]) files. A valid",
        "run input ([REF].tpr[ref]) with constant pH enabled is required. More analysis",
        "features might be added in the future."
    };
    settings->setHelpText(desc);

    const char* bugs[] = { "Please use this with caution, no guarantee about it being bug free." };
    settings->setBugText(bugs);

    timeUnitBehavior_ = std::make_shared<TimeUnitBehavior>();

    settings->addOptionsBehavior(timeUnitBehavior_);
    options->addOption(FileNameOption("s")
                               .filetype(OptionFileType::RunInput)
                               .inputFile()
                               .store(&inputTprFilename_)
                               .required()
                               .defaultBasename("topol")
                               .description("Run input file with cpHMD enabled"));
    options->addOption(FileNameOption("e")
                               .filetype(OptionFileType::Energy)
                               .inputFile()
                               .store(&inputEnergyFilename_)
                               .required()
                               .defaultBasename("energy")
                               .description("Energy file from cpHMD simulation"));
    options->addOption(FileNameOption("o")
                               .required()
                               .filetype(OptionFileType::Plot)
                               .outputFile()
                               .store(&outputFileBaseName_)
                               .defaultBasename("cphmd")
                               .description("Base name for output files generated"));
    // Add options for trajectory time control.
    options->addOption(DoubleOption("begin")
                               .store(&startTime_)
                               .storeIsSet(&bStartTimeSet_)
                               .timeValue()
                               .description("First frame (%t) to read from trajectory"));
    options->addOption(
            DoubleOption("end").store(&endTime_).storeIsSet(&bEndTimeSet_).timeValue().description("Last frame (%t) to read from trajectory"));

    options->addOption(
            IntegerOption("numplot")
                    .store(&maxLambdaPerPlot_)
                    .description("How many lambda coordinates should be combined into one plot"));
    options->addOption(
            BooleanOption("coordinate").store(&plotCoordinate_).description("Whether coordinates should be plotted or not"));
    options->addOption(
            BooleanOption("dvdl").store(&plotDvdl_).description("Whether dvdl values should be plotted or not"));
    options->addOption(
            BooleanOption("velocity").store(&plotVelocity_).description("Whether velocites should be plotted or not"));


    timeUnitBehavior_->setTimeUnitFromEnvironment();
    timeUnitBehavior_->addTimeUnitOption(options, "tu");
    timeUnitBehavior_->setTimeUnitStore(&timeUnit_);
}

void CpHMD::optionsFinished()
{
    if (!plotCoordinate_ && !plotVelocity_ && !plotDvdl_)
    {
        GMX_THROW(InconsistentInputError("Need to plot at least something"));
    }
}

int CpHMD::run()
{
    t_inputrec        ir;
    ener_file_t       fp;
    gmx_enxnm_t*      enm = nullptr;
    t_enxframe*       frame;
    int               nre;
    gmx_output_env_t* oenv;
    gmx_mtop_t        mtop;
    int               natoms;
    matrix            box;

    read_tpx(inputTprFilename_.c_str(), &ir, box, &natoms, nullptr, nullptr, &mtop);
    if (!ir.lambda_dynamics)
    {
        gmx_fatal(FARGS, "This is not the TPR for a cpHMD simulation\n");
    }

    output_env_init(&oenv, getProgramContext(), timeUnit_, false, XvgFormat::Xmgrace, 0);

    snew(frame, 1);
    fp = open_enx(inputEnergyFilename_.c_str(), "r");
    do_enxnms(fp, &nre, &enm);

    gmx::FixedCapacityVector<CpHMDOutputSelection, 3> graphs;
    if (plotCoordinate_)
    {
        graphs.emplace_back(CpHMDOutputSelection::Coordinate);
    }
    if (plotVelocity_)
    {
        graphs.emplace_back(CpHMDOutputSelection::Velocity);
    }
    if (plotDvdl_)
    {
        graphs.emplace_back(CpHMDOutputSelection::Dvdl);
    }

    std::unique_ptr<CpHMDReader> cpHMDReader = std::make_unique<CpHMDReader>(
            ir.lambdaDynamicsSimulationParameters->lambdaResidues(),
            ir.lambdaDynamicsSimulationParameters->lambdaAtomsCollections(), graphs, timeUnit_,
            outputFileBaseName_, maxLambdaPerPlot_);

    /* Initiate counters */
    gmx_bool haveFrame;
    int      frameCounter = 0;
    int      timeCheck    = 0;
    do
    {
        haveFrame = do_enx(fp, frame);

        bool useFrame = false;

        t_enxblock* block = nullptr;

        if (haveFrame)
        {
            timeCheck = check_times(frame->t);

            if (timeCheck == 0)
            {
                block = find_block_id_enxframe(frame, enxCPHMD, nullptr);

                if (block != nullptr)
                {
                    useFrame = true;
                    frameCounter++;
                }
            }
        }
        if (useFrame)
        {
            cpHMDReader->openAllOutputFiles(oenv);

            cpHMDReader->processCpHMDFrame(*block, frame->t);
        }
    } while (haveFrame && (timeCheck <= 0));

    close_enx(fp);
    output_env_done(oenv);
    return 0;
}


} // namespace

const char                       CpHMDInfo::name[]             = "cphmd";
const char                       CpHMDInfo::shortDescription[] = "Plot data from cpHMD simulations";
ICommandLineOptionsModulePointer CpHMDInfo::create()
{
    return std::make_unique<CpHMD>();
}

} // namespace gmx
