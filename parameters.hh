#ifndef PARAMETERS_HH
#define PARAMETERS_HH

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/conditional_ostream.h>

#include "my_components.hh" 

using namespace dealii;

namespace Parameters
{
// ----------------------------------
// --- STRUCT SOLVER
// ----------------------------------

// ----------------------------------
// --- enums // Neumann=0, Dirichlet=1, None=2
// ----------------------------------
  enum BCType { Neumann, Dirichlet, None };
  
  enum ViscosityType { constant_viscosity, first_order_viscosity, entropy_viscosity} ;

  struct Solver
    {
      enum  OutputType { quiet, verbose };

      // nonlinear
      enum NonLinSolverType { newton };
      NonLinSolverType NLsolver;

      OutputType NLoutput;
      
      double nonlinear_atol;
      double nonlinear_rtol;
      double damping;
      unsigned int max_nonlin_iterations;

      //linear
      enum LinSolverType { krylov, direct };
      LinSolverType solver;

      OutputType output;
        
      double linear_atol;
      double linear_rtol;
      double ilut_fill;
      double ilut_atol;
      double ilut_rtol;
      double ilut_drop;
      int max_linear_iterations;
        
      // methods
      static void declare_parameters (ParameterHandler &prm); // jcr why is one static?
      void parse_parameters (ParameterHandler &prm);
    };
    
// ----------------------------------
// --- DECLARE SOLVER PARAMETERS
// ----------------------------------

  void Solver::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("nonlinear solver");
        {
          prm.declare_entry("output", "quiet",
                            Patterns::Selection("quiet|verbose"),
                            "State whether output from nonlinear solver runs should be printed. "
                            "Choices are <quiet|verbose>.");
          prm.declare_entry("nonlinear method", "newton",
                            Patterns::Selection("newton"),
                            "The kind of nonlinear solver for the linear system. "
                            "Choices are <newton>.");
          prm.declare_entry("nonlinear absolute tolerance", "1e-10",
                            Patterns::Double(),
                            "Nonlinear absolute tolerance");
          prm.declare_entry("nonlinear relative tolerance", "1e-10",
                            Patterns::Double(),
                            "Nonlinear relative tolerance");
          prm.declare_entry("damping", "1.0",
                            Patterns::Double(),
                            "damping");
          prm.declare_entry("max nonlinear iters", "300",
                            Patterns::Integer(),
                            "Maximum nonlinear iterations");
        }
      prm.leave_subsection();

      prm.enter_subsection("linear solver");
        {
          prm.declare_entry("output", "quiet",
                            Patterns::Selection("quiet|verbose"),
                            "State whether output from linear solver runs should be printed. "
                            "Choices are <quiet|verbose>.");
          prm.declare_entry("linear method", "krylov",
                            Patterns::Selection("krylov|direct"),
                            "The kind of linear solver for the linear system. "
                            "Choices are <krylov|direct>.");
          prm.declare_entry("linear absolute tolerance", "1e-10",
                            Patterns::Double(),
                            "Linear absolute tolerance");
          prm.declare_entry("linear relative tolerance", "1e-10",
                            Patterns::Double(),
                            "Linear relative tolerance");
          prm.declare_entry("max linear iters", "300",
                            Patterns::Integer(),
                            "Maximum linear solver iterations");
          prm.declare_entry("ilut fill", "2.0",
                            Patterns::Double(),
                            "Ilut preconditioner fill");
          prm.declare_entry("ilut absolute tolerance", "1e-9",
                            Patterns::Double(),
                            "Ilut preconditioner tolerance");
          prm.declare_entry("ilut relative tolerance", "1.1",
                            Patterns::Double(),
                            "Ilut relative tolerance");
          prm.declare_entry("ilut drop tolerance", "1e-10",
                            Patterns::Double(),
                            "Ilut drop tolerance");
        }
      prm.leave_subsection();
    }    

// ----------------------------------
// --- PARSE SOLVER PARAMETERS
// ----------------------------------
  void Solver::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("nonlinear solver");
        {
          const std::string op = prm.get("output");
          if (op == "verbose") NLoutput = verbose;
          if (op == "quiet")   NLoutput = quiet;
            
          const std::string sv = prm.get("nonlinear method");
          if (sv == "newton")     NLsolver = newton;
            
          nonlinear_atol         = prm.get_double("nonlinear absolute tolerance");
          nonlinear_rtol         = prm.get_double("nonlinear relative tolerance");
          damping                = prm.get_double("damping");
          max_nonlin_iterations  = prm.get_integer("max nonlinear iters");
        }
      prm.leave_subsection();

      prm.enter_subsection("linear solver");
        {
          const std::string op = prm.get("output");
          if (op == "verbose") output = verbose;
          if (op == "quiet")   output = quiet;
            
          const std::string sv = prm.get("linear method");
          if (sv == "direct")     solver = direct;
          else if (sv == "krylov") solver = krylov;
            
          linear_atol     = prm.get_double("linear absolute tolerance");
          linear_rtol     = prm.get_double("linear relative tolerance");
          max_linear_iterations  = prm.get_integer("max linear iters");
          ilut_fill       = prm.get_double("ilut fill");
          ilut_atol       = prm.get_double("ilut absolute tolerance");
          ilut_rtol       = prm.get_double("ilut relative tolerance");
          ilut_drop       = prm.get_double("ilut drop tolerance");
        }
      prm.leave_subsection();
      }

// ----------------------------------
// --- STRUCT ALL PARAMETERS
// ----------------------------------

  template <int dim>
  struct AllParameters : public Solver
    {
      AllParameters ();
      
      static const unsigned n_boundaries = 6;
      
      // enum ViscosityType { constant_viscosity, first_order_viscosity, entropy_viscosity} ;
      
      struct BoundaryConditions
        {
          BCType type_of_bc[MyComponents::n_components];
          double values    [MyComponents::n_components];
        };
      
      BoundaryConditions boundary_conditions[n_boundaries];
      
      double time_step, initial_time, final_time;
      double theta;
      bool perform_steady_state_first;
      bool is_transient;
      std::string    time_discretization_scheme_name;

      unsigned int n_init_refinements;
      unsigned int n_init_refinements_x;
      unsigned int n_init_refinements_y;
      
      unsigned int console_print_out;
      unsigned int vtk_output_frequency;
      std::string  output_dir;
      std::string  output_name;

      unsigned int degree_p;
      unsigned int n_quad_points;
      unsigned int n_face_quad_points;
        
      double length;

      double amplitude;
      int spatial_shape_option;
      int temporal_function_option;
      double time_a0;
      bool has_exact_solution;
      int  pbID;


      double c_max;
      double c_ent;
      double c_jmp;
      double cfl;
      ViscosityType viscosity_option;
      double const_visc;
      bool is_cfl_time_adaptive;

      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
    };

// ----------------------------------
// --- CONSTRUCTOR ALL PARAMETERS
// ----------------------------------
  template <int dim>
  AllParameters<dim>::AllParameters ()
    {}

// ----------------------------------
// --- DECLARE ALL PARAMETERS
// ---------------------------------- 

  template <int dim>
  void AllParameters<dim>::declare_parameters (ParameterHandler &prm)
    {
      // ------------------------------------
      prm.enter_subsection("time stepping");
        {
          prm.declare_entry("time step", "0.2",
                            Patterns::Double(0),
                            "simulation time step");
          prm.declare_entry("perform steady state first", "true",
                            Patterns::Bool(),
                            "whether to do a SS first or not");
          prm.declare_entry("initial time", "10.0",
                            Patterns::Double(0),
                            "simulation start time");
          prm.declare_entry("final time", "10.0",
                            Patterns::Double(0),
                            "simulation end time");
          prm.declare_entry("theta scheme value", "0.5",
                            Patterns::Double(0,1),
                            "value for theta that interpolated between explicit "
                            "Euler (theta=0), Crank-Nicolson (theta=0.5), and "
                            "implicit Euler (theta=1).");
          prm.declare_entry("time discretization name", "ERK33",
                            Patterns::Anything(),
                            "time disc. name");
          prm.declare_entry("cfl", "0.5",
                            Patterns::Double(0),
                            "cfl value");
          prm.declare_entry("is cfl time adaptive", "true",
                            Patterns::Bool(),
                            "is cfl time adaptive");                            
        }
      prm.leave_subsection();

      // ------------------------------------
      prm.enter_subsection("refinement");
        {
          prm.declare_entry("number initial mesh refinements", "2",
                            Patterns::Integer(0),
                            "number initial mesh refinements");
          prm.declare_entry("number initial mesh refinements_x", "10",
                            Patterns::Integer(0),
                            "number initial mesh refinements_x");
          prm.declare_entry("number initial mesh refinements_y", "1",
                            Patterns::Integer(0),
                            "number initial mesh refinements_y");
        }
      prm.leave_subsection();

      // ------------------------------------
      prm.enter_subsection("verbose level");
        {
          prm.declare_entry("console print out level", "0",
                            Patterns::Integer(0),
                            "console print out level");
          prm.declare_entry("vtk output frequency", "1",
                            Patterns::Integer(0),
                            "vtk output frequency");
          prm.declare_entry("output directory", ".",
                            Patterns::Anything(),
                            "output directory");
          prm.declare_entry("output name", "solu-",
                            Patterns::Anything(),
                            "output name");
        }
      prm.leave_subsection();

     // ------------------------------------
     prm.enter_subsection("fe space");
        {
          prm.declare_entry("polynomials of degree p", "2",
                            Patterns::Integer(0),
                            "use polynomials of degree p for the shape functions");
          prm.declare_entry("number quadrature points", "3",
                            Patterns::Integer(0),
                            "number of quadrature points in each space direction");
          prm.declare_entry("number face quadrature points", "3",
                            Patterns::Integer(0),
                            "number of quadrature points to integrate over boundary");
        }
      prm.leave_subsection();
        
      // ------------------------------------
      prm.enter_subsection("domain");
        {
          prm.declare_entry("length", "1.0",
                            Patterns::Double(0),
                            "length of the domain");
        }
      prm.leave_subsection();
        
      // ------------------------------------
      prm.enter_subsection("entropy");
        {
          prm.declare_entry("c_max", "0.5",
                            Patterns::Double(0),
                            "c_max constant for the first-order viscosity term");
          prm.declare_entry("c_ent", "1.0",
                            Patterns::Double(0),
                            "c_ent constant for the entropy viscosity term");
          prm.declare_entry("c_jmp", "1.0",
                            Patterns::Double(0),
                            "c_jmp constant for the entropy jump term");
          prm.declare_entry("viscosity option", "first_order",
                            Patterns::Selection("constant|first_order|entropy"),
                            "viscosity option: constant, first_order, entropy");
          prm.declare_entry("constant viscosity", "1.0",
                            Patterns::Double(0),
                            "constant viscosity, for testing purposes");
        }
      prm.leave_subsection();
        
      // ------------------------------------
      prm.enter_subsection("exact solution");
        {
          prm.declare_entry("has exact solution", "true",
                            Patterns::Bool(),
                            "whether an exact solution exists or has been implemented");
          prm.declare_entry("amplitude", "1.0",
                            Patterns::Double(0),
                            "amplitude of the solution");
          prm.declare_entry("spatial shape option", "0",
                            Patterns::Integer(0),
                            "spatial shape option: 0=x(L-x) per dim, 1=sin per dim");
          prm.declare_entry("temporal function option", "0",
                            Patterns::Integer(0),
                            "temporal function option: 0=exp(a0.t), 1=not done yet");
          prm.declare_entry("time constant a0", "1.0",
                            Patterns::Double(0),
                            "time constant a0 (in exp(a0.t) )");
          prm.declare_entry("problem ID", "1",
                            Patterns::Integer(0),
                            "problem ID: 1=sinus, 2=Riemann pb#xxx");        
        }
      prm.leave_subsection();

      
      // ------------------------------------
      for (unsigned int boundary_id = 0; boundary_id < n_boundaries; ++boundary_id)
        {
          prm.enter_subsection("boundary_" + Utilities::int_to_string(boundary_id));
            {
              prm.declare_entry("Dirichlet", "false", Patterns::Bool()  , "to specify Dirichlet boundary");
              prm.declare_entry("Neumann"  , "false", Patterns::Bool()  , "to specify Neumann boundary");
              prm.declare_entry("value"    , "0.0"  , Patterns::Double(), "to specify the value");
            }
          prm.leave_subsection();
        }
      
      // ------------------------------------
      Parameters::Solver::declare_parameters (prm);
    }
    
// ----------------------------------
// --- PARSE ALL PARAMETERS
// ---------------------------------- 
  template <int dim>
  void AllParameters<dim>::parse_parameters (ParameterHandler &prm)
    {
      // ------------------------------------
      prm.enter_subsection("time stepping");
        {
          time_step = prm.get_double("time step");
          perform_steady_state_first = prm.get_bool("perform steady state first");
          if (time_step <= 0.0)
            {
              time_step    = 1.0;
              initial_time = 0.0;
              final_time   = 0.0;
              is_transient = false;
              cfl          = 0.1;
              is_cfl_time_adaptive = false;
            }
          else
            {
              is_transient = true;
              initial_time = prm.get_double("initial time");
              final_time   = prm.get_double("final time");
              theta        = prm.get_double("theta scheme value");
              time_discretization_scheme_name = prm.get("time discretization name");
              cfl   = prm.get_double("cfl");
              is_cfl_time_adaptive = prm.get_bool("is cfl time adaptive");
            }
        }
      prm.leave_subsection();
        
      // ------------------------------------
      prm.enter_subsection("refinement");
        {
          n_init_refinements   = prm.get_integer("number initial mesh refinements");
          n_init_refinements_x = prm.get_integer("number initial mesh refinements_x");
          n_init_refinements_y = prm.get_integer("number initial mesh refinements_y");
        }
      prm.leave_subsection();

      // ------------------------------------
      prm.enter_subsection("verbose level");
        {
          console_print_out = prm.get_integer("console print out level");
          vtk_output_frequency = prm.get_integer("vtk output frequency");
          output_dir  = prm.get("output directory");
          output_name = prm.get("output name");
        }
      prm.leave_subsection();

      // ------------------------------------
      prm.enter_subsection("fe space");
        {
          degree_p           = prm.get_integer("polynomials of degree p");
          n_quad_points      = prm.get_integer("number quadrature points");
          n_face_quad_points = prm.get_integer("number face quadrature points");
        }
      prm.leave_subsection();

      // ------------------------------------
      prm.enter_subsection("domain");
        {
          length = prm.get_double("length");
        }
      prm.leave_subsection();

      // ------------------------------------
      prm.enter_subsection("entropy");
        {
          c_max = prm.get_double("c_max");
          c_ent = prm.get_double("c_ent");
          c_jmp = prm.get_double("c_jmp");

          const std::string vo = prm.get("viscosity option");
          if (vo == "constant")  
            viscosity_option = constant_viscosity;
          else if (vo == "first_order") 
            viscosity_option = first_order_viscosity;
          else  if (vo == "entropy") 
            viscosity_option = entropy_viscosity;
          else
            AssertThrow (false, ExcNotImplemented());
      
          const_visc = prm.get_double("constant viscosity");
                  }
      prm.leave_subsection();

      // ------------------------------------
      prm.enter_subsection("exact solution");
        {
          has_exact_solution       = prm.get_bool("has exact solution");
          amplitude                = prm.get_double("amplitude");
          spatial_shape_option     = prm.get_integer("spatial shape option");
          temporal_function_option = prm.get_integer("temporal function option");
          time_a0                  = prm.get_double("time constant a0");
          pbID                     = prm.get_integer("problem ID");
        }
      prm.leave_subsection();

      // loop on boundaries
      for (unsigned int boundary_id = 0; boundary_id < n_boundaries; ++boundary_id)
        {
          prm.enter_subsection("boundary_" + Utilities::int_to_string(boundary_id));
            {
              if (prm.get_bool("Dirichlet")) 
                {
                  boundary_conditions[boundary_id].type_of_bc[MyComponents::burgers_component] = Dirichlet;
                  boundary_conditions[boundary_id].values    [MyComponents::burgers_component] = prm.get_double("value");
                }
              else if (prm.get_bool("Neumann")) 
                {
                  boundary_conditions[boundary_id].type_of_bc[MyComponents::burgers_component] = Neumann;
                  boundary_conditions[boundary_id].values    [MyComponents::burgers_component] = prm.get_double("value");
                }
              else
                {
                  boundary_conditions[boundary_id].type_of_bc[MyComponents::burgers_component] = None;
                }
            } // end subsection
          prm.leave_subsection();
        } // end loop on boundaries
      
      // ------------------------------------
      Parameters::Solver::parse_parameters (prm);
    }
  
}  // namespace


#endif // PARAMETERS_HH
