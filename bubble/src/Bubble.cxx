/*!
 * \file   main.cxx
 * \brief    
 * \author Thomas Helfer
 * \date   29/09/2024
 */

#include <cstdlib>
#include "MFEMMGIS/Material.hxx"
#include "MFEMMGIS/Profiler.hxx"
#include "MFEMMGIS/PeriodicNonLinearEvolutionProblem.hxx"
#include "MFEMMGIS/UniformImposedPressureBoundaryCondition.hxx"
#include "OperaHPC/BubbleDescription.hxx"
#include "OperaHPC/Utilities.hxx"
#include "Common/Print.hxx"
#include "Common/Memory.hxx"

namespace opera_hpc {

  struct Bubble : BubbleDescription {
    bool broken = false;
  };

  bool areAllBroken(const std::vector<Bubble>& bubbles) {
    for (const auto& b : bubbles) {
      if (!b.broken) {
        return false;
      }
    }
    return true;
  }
}  // end of namespace opera_hpc

// add postprocessing
template <typename Problem>
void post_process(Problem& p, double start, double end) {
  CatchTimeSection("common::post_processing_step");
  p.executePostProcessings(start, end);
}

struct TestParameters {
  const char* mesh_file = "mesh/single_sphere.msh";
  const char* behaviour = "Elasticity";
  const char* library = "src/libBehaviour.so";
  const char* bubble_file = "mesh/single_bubble.txt";
  int order = 1;
  int refinement = 0;
  int post_processing = 1; // default value : activated
  int verbosity_level = 0; // default value : lower level
};


void fill_parameters(mfem::OptionsParser& args, TestParameters& p)
{
  args.AddOption(&p.mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&p.library, "-l", "--library", "Material library.");
  args.AddOption(&p.bubble_file, "-f", "--bubble-file", "File containing the bubbles.");
  args.AddOption(&p.order, "-o", "--order", "Finite element order (polynomial degree).");
  args.AddOption(&p.refinement, "-r", "--refinement", "refinement level of the mesh, default = 0");
  args.AddOption(&p.post_processing, "-p", "--post-processing", "run post processing step");
  args.AddOption(&p.verbosity_level, "-v", "--verbosity-level", "choose the verbosity level");

  args.Parse();

  if (!args.Good()) {
    if (mfem_mgis::getMPIrank() == 0)
      args.PrintUsage(std::cout);
    mfem_mgis::finalize();
    exit(0);
  }
  if (p.mesh_file == nullptr) { 
    if (mfem_mgis::getMPIrank() == 0)
      std::cout << "ERROR: Mesh file missing" << std::endl;
    args.PrintUsage(std::cout);
    mfem_mgis::abort(EXIT_FAILURE);
  }
  if (mfem_mgis::getMPIrank() == 0)
    args.PrintOptions(std::cout);
}



int main(int argc, char** argv) {
  using namespace mfem_mgis::Profiler::Utils; // Use Message
  // options treatment
  mfem_mgis::initialize(argc, argv);
  mfem_mgis::Profiler::timers::init_timers();

  // get parameters
  TestParameters p;
  mfem::OptionsParser args(argc, argv);
  fill_parameters(args, p);

  // reference pressure
  constexpr auto pref = mfem_mgis::real{1e6};
  constexpr auto maximumNumberSteps = mfem_mgis::size_type{100};
  // distance to determine if a bubble breaks
  constexpr auto dmin = mfem_mgis::real{650e-9};
  //
  auto bubbles = [p] {
    auto r = std::vector<opera_hpc::Bubble>{};
    for (const auto& d : opera_hpc::BubbleDescription::read(p.bubble_file)) {
      auto b = opera_hpc::Bubble{};
      static_cast<opera_hpc::BubbleDescription&>(b) = d;
      r.push_back(b);
    }
    return r;
  }();
  // finite element space
  auto fed = std::make_shared<mfem_mgis::FiniteElementDiscretization>(
      mfem_mgis::Parameters{
          {"MeshFileName", p.mesh_file},
          {"FiniteElementFamily", "H1"},
          {"FiniteElementOrder", p.order},
          {"UnknownsSize", 3},
          {"NumberOfUniformRefinements", p.refinement}, // faster for testing
          //{"NumberOfUniformRefinements", parameters.parallel ? 1 : 0},
          //          {"Hypothesis", "Tridimensional"},
          {"Parallel", true}});
  // definition of the nonlinear problem
  auto problem = mfem_mgis::PeriodicNonLinearEvolutionProblem{fed};

  // get problem information
  print_mesh_information(problem.getImplementation<true>());
  print_memory_footprint("[Building problem]");

  // macroscopic strain
  std::vector<mfem_mgis::real> e(6, mfem_mgis::real{0});
  problem.setMacroscopicGradientsEvolution(
      [e](const mfem_mgis::real) { return e; });
  // imposing pressure on the bubble boundaries
  for (const auto &b : bubbles) {
    problem.addBoundaryCondition(
        std::make_unique<mfem_mgis::UniformImposedPressureBoundaryCondition>(
            problem.getFiniteElementDiscretizationPointer(),
            b.boundary_identifier, [&b](const mfem_mgis::real) {
              // pressure evolution in the bubble
              // if sound, return the reference pressure
              // if broken, return 0
              return b.broken ? 0 : pref;
            }));
    }
    // choix du solver linéaire +
    int verbosity = p.verbosity_level;
    int post_processing = p.post_processing;
    auto solverParameters = mfem_mgis::Parameters{};
    solverParameters.insert(
        mfem_mgis::Parameters{{"VerbosityLevel", verbosity}});
    //  solverParameters.insert(mfem_mgis::Parameters{{"MaximumNumberOfIterations",
    //  defaultMaxNumOfIt}});
    //  solverParameters.insert(mfem_mgis::Parameters{{"Tolerance", Tol}});
    //
    auto options = mfem_mgis::Parameters{{"VerbosityLevel", verbosity},
                                         {"Strategy", "Elasticity"}};
    auto preconditionner =
        mfem_mgis::Parameters{{"Name", "HypreBoomerAMG"}, {"Options", options}};
    solverParameters.insert(
        mfem_mgis::Parameters{{"Preconditioner", preconditionner}});
    problem.setLinearSolver("HyprePCG", solverParameters);
    // matrix is considered elastic
    problem.addBehaviourIntegrator("Mechanics", 1, p.library,
                                   p.behaviour);
    auto &m = problem.getMaterial(1);
    for (auto &s : {&m.s0, &m.s1}) {
      mgis::behaviour::setMaterialProperty(*s, "YoungModulus", 150e9);
      mgis::behaviour::setMaterialProperty(*s, "PoissonRatio", 0.3);
      mgis::behaviour::setExternalStateVariable(*s, "Temperature", 293.15);
      mgis::behaviour::setExternalStateVariable(*s, "Temperature", 293.15);
    }
    if (post_processing) {
      auto results = std::vector<mfem_mgis::Parameter>{"Stress"};
      problem.addPostProcessing(
          "ParaviewExportIntegrationPointResultsAtNodes",
          {{"OutputFileName", "TestCaseOneBubble"}, {"Results", results}});
    }
    //
    auto nstep = mfem_mgis::size_type{};
    while ((nstep != maximumNumberSteps) &&
           (!opera_hpc::areAllBroken(bubbles))) {
      problem.solve(0, 1);
      const auto r = opera_hpc::findFirstPrincipalStressValueAndLocation(
          problem.getMaterial(1));
      auto nbroken = mfem_mgis::size_type{};
      auto all_broken_bubbles_identifiers = std::vector<mfem_mgis::size_type>{};
      for (auto &b : bubbles) {
        const auto d = opera_hpc::distance(b, r.location);
        if (d < dmin) {
          b.broken = true;
          ++nbroken;
        }
        if (b.broken) {
          all_broken_bubbles_identifiers.push_back(b.boundary_identifier);
        }
      }
      if (nbroken == 0) {
        Message("no bubble broke at step ", nstep,"\n");
        break;
      }


      Message("step:",nstep);
      Message("-", nbroken, "bubbles broke at this step.");
      Message("-", all_broken_bubbles_identifiers.size(), "bubbles are broken");
      Message("- value of the first principal stress:", r.value, "at coordinate (", r.location[0], ",", r.location[1], "," , r.location[2], ")");
      if (post_processing)
        post_process(problem, 0+double(nstep), 1+double(nstep));
      
      ++nstep;
    }
    print_memory_footprint("[End]");
    mfem_mgis::Profiler::timers::print_and_write_timers();
    return EXIT_SUCCESS;
}
