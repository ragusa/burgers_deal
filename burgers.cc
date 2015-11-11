/* Burgers equation stabilized with entropy viscosity and solved on distributed meshes (p4est)
 code written  using deal.ii 
   JCR, Texas A&M University, 2012-14 
   BT , Texas A&M University, 2013-14 
*/

#include <deal.II/base/quadrature_lib.h> 
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>
#define USE_PETSC_LA
namespace LA
{
#ifdef USE_PETSC_LA
  using namespace dealii::LinearAlgebraPETSc;
#else
  using namespace dealii::LinearAlgebraTrilinos;
#endif
}

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>

// #include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/base/index_set.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#ifdef USE_PETSC_LA
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#else
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#endif


/** std */
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <cmath>
#include <cstdlib>

/** trilinos */

using namespace dealii;

/** perso **/
#include "my_components.hh"
#include "parameters.hh"
#include "my_flux.hh"
#include "exact_solution.hh"
#include "initial_data.hh"
#include "butcher_tableaux.hh"

void mypause(std::string,bool);
void mypause(std::string mystring, bool B)
{
  if(B){
    std::cout<<" ... pausing in " << mystring << std::endl;
    std::cin.ignore();
    std::cin.get();
  }
}
static int time_step_no=0;

// **********************************************************************************
// ---                   ---
// --- BURGERS PROBLEM   ---
// ---                   ---
// ---                                       ---
// --- template class BurgersPRoblem<dim>    ---
// ---                                       ---
// **********************************************************************************

template <int dim>
class BurgersProblem
{
public:
  BurgersProblem (const Parameters::AllParameters<dim> *const params, const MyFlux<dim> flx);
  ~BurgersProblem();
  
  void run ();
  
private:
  void setup_system ();
  
  void assemble_mass_matrix ();
  
  void compute_ss_residual (const bool is_steady_state,
                            const double time);
  
  void assemble_ss_jacobian (const bool is_steady_state,
                             const double time);
  
/*  void compute_ss_residual_cell_term (const FEValues<dim>             &fe_values      ,
                                        const std::vector<unsigned int> &dofs           ,
                                        const double                     time           ,
                                        const bool                       is_steady_state,
                                        const typename DoFHandler<dim>::active_cell_iterator cell);
  
  void assemble_ss_jacobian_cell_term (const FEValues<dim>             &fe_values,
                                       const std::vector<unsigned int> &dofs     ,
                                       const double                     time     );
*/  
  void compute_tr_residual (const unsigned short stage,
                            const double         time ,
                            const double         dt   );
  
  void assemble_tr_jacobian (const unsigned short stage,
                             const double         time ,
                             const double         dt   ,
                             const bool           is_explicit);
  
  std::pair<unsigned int, double> linear_solve (LA::MPI::Vector   &solution); 
  std::pair<unsigned int, double> mass_solve   (LA::MPI::Vector   &solution); 
  
  void refine_grid ();
  
  //void output_viscosity (); // jcr const or not const!
  void output_solution (const unsigned int time_step_no) const;
  void output_exact_solution (const unsigned int time_step_no) const;
  //void process_solution () const;
  void extract_map(std::map<typename DoFHandler<dim>::active_cell_iterator, double> &my_map);
  void extract_map(std::map<typename DoFHandler<dim>::active_cell_iterator, Vector<double> > &my_map);

  double compute_dt ();
  void compute_entropy_residual (const double dt);
  void compute_first_order_viscosity ();
  void compute_entropy_viscosity ();
  void compute_viscosity (const double dt);
  void compute_jumps ();

  MPI_Comm                                   mpi_communicator;
  const Parameters::AllParameters<dim>       *const parameters;
  MyFlux<dim>                                flux;

  parallel::distributed::Triangulation<dim>  triangulation;
  DoFHandler<dim>                            dof_handler;
  const FESystem<dim>                        fe;
  const QGauss<dim>                          quadrature;
  const QGauss<dim-1>                        face_quadrature;
  const unsigned int                         n_q_pts;
  const unsigned int                         n_q_pts_face;
  
  ConditionalOStream                         pcout;
  TimerOutput                                computing_timer;

  const unsigned int                         console_print_out_;
  const unsigned int                         vtk_output_frequency_;
  
  ConstraintMatrix                           constraints;

  IndexSet                                   locally_owned_dofs;
  IndexSet                                   locally_relevant_dofs;

  LA::MPI::Vector                 current_solution;
  LA::MPI::Vector                 old_solution;
  LA::MPI::Vector                 older_solution;
  LA::MPI::Vector                 oldest_solution;
  LA::MPI::Vector                 old_entropy;

  //  step-40 has ::MPI for matrices ...
  LA::MPI::SparseMatrix system_matrix;
  LA::MPI::SparseMatrix mass_matrix;
  LA::MPI::Vector  system_rhs; 

  LA::MPI::Vector              tmp_vect_relev;
  LA::MPI::Vector              tmp_vect_owned;
  std::vector<LA::MPI::Vector> Y;
  std::vector<LA::MPI::Vector> previous_f;

  double      h_min;
  double      h_min_;
  double      volume_of_domain;
  double      entropy_average;
  double      entropy_normalization;
  double      c_max;
  double      c_ent;
  double      c_jmp;

  std::map<typename DoFHandler<dim>::active_cell_iterator, double>  map_h_local; // local h value
  
  std::map<typename DoFHandler<dim>::active_cell_iterator, double>  map_entropy_residual_K; 
  std::map<typename DoFHandler<dim>::active_cell_iterator, double>  map_entropy_viscosity_K; 
  std::map<typename DoFHandler<dim>::active_cell_iterator, double>  map_first_order_viscosity_K; 
  std::map<typename DoFHandler<dim>::active_cell_iterator, double>  map_viscosity_K; 
  std::map<typename DoFHandler<dim>::active_cell_iterator, double>  map_jumps_K; 
  
  std::map<typename DoFHandler<dim>::active_cell_iterator, Vector<double> > map_entropy_residual_K_q; 
  std::map<typename DoFHandler<dim>::active_cell_iterator, Vector<double> > map_entropy_viscosity_K_q; 
  std::map<typename DoFHandler<dim>::active_cell_iterator, Vector<double> > map_first_order_viscosity_K_q; 
  std::map<typename DoFHandler<dim>::active_cell_iterator, Vector<double> > map_viscosity_K_q; 

  Vector<double> visc_output_K; // viscosity for output via DataOut

  // time integration, Butcher tableau
  std::string                        method_name; 
  unsigned short                     n_stages;
  std::vector<std::vector<double> >  a;
  std::vector<double>                b, c; 
  bool  is_explicit;

};


// **********************************************************************************
// ---             ---
// --- constructor ---
// ---             ---
// **********************************************************************************

template <int dim>
BurgersProblem<dim>::BurgersProblem (const Parameters::AllParameters<dim> *const params, const MyFlux<dim> flx)
  : 
  mpi_communicator (MPI_COMM_WORLD),
  parameters(params),
  flux(flx),
  triangulation (mpi_communicator,
                 typename Triangulation<dim>::MeshSmoothing
                 (Triangulation<dim>::smoothing_on_refinement |
                  Triangulation<dim>::smoothing_on_coarsening)),
  dof_handler (triangulation),
  fe (FE_Q<dim>(parameters->degree_p), MyComponents::n_components), 
  quadrature(parameters->n_quad_points),
  face_quadrature(parameters->n_face_quad_points),
  n_q_pts(quadrature.size()),
  n_q_pts_face(face_quadrature.size()),
  pcout (std::cout,
         (Utilities::MPI::this_mpi_process(mpi_communicator)
          == 0)),
  computing_timer (pcout,
                   TimerOutput::summary,
                   TimerOutput::wall_times),
 console_print_out_(parameters->console_print_out),
 vtk_output_frequency_(parameters->vtk_output_frequency)
{
  //pcout.set_condition (parameters->output == Parameters::Solver::verbose);
  
  init_butcher_tableau(method_name=parameters->time_discretization_scheme_name,
                       n_stages, a, b, c, is_explicit);
  
  /** print chosen Butcher tableau to the screen */
  pcout << "time discretization method name is : " << method_name << std::endl;

  for (unsigned short s = 0; s < n_stages; ++s) 
  {
    pcout << "c[" << s << "]=" << c[s] << "  ";
    for (unsigned short t = 0; t < n_stages; ++t) 
      pcout << "a[" << s << "][" << t << "]=" << a[s][t] << "  ";
  }
  pcout<<"\n";
  for (unsigned short s = 0; s < n_stages; ++s)
    pcout << "b[" << s << "]=" << b[s] <<"    ";
  pcout << "\n";
  
  //AssertThrow(false, ExcMessage(" stopping after constructor"));
}

// **********************************************************************************
// ---            ---
// --- destructor ---
// ---            ---
// **********************************************************************************

template <int dim>
BurgersProblem<dim>::~BurgersProblem ()
{ 
  dof_handler.clear ();
}

// **********************************************************************************
// ---              ---
// --- setup_system ---
// ---              ---
// **********************************************************************************

template <int dim>
void BurgersProblem<dim>::setup_system ()
{
  TimerOutput::Scope t(computing_timer, "setup");

  static bool first_time_here = true;
  if (first_time_here)
  {
    first_time_here = false;
        
    //const unsigned int n_act_cells = triangulation.n_locally_owned_active_cells();
    
    // clear all maps
    map_entropy_residual_K.clear(); 
    map_entropy_viscosity_K.clear(); 
    map_first_order_viscosity_K.clear(); 
    map_viscosity_K.clear(); 
    map_jumps_K.clear(); 
    map_entropy_residual_K_q.clear(); 
    map_entropy_viscosity_K_q.clear(); 
    map_first_order_viscosity_K_q.clear(); 
    map_viscosity_K_q.clear(); 
    
    // dof handler
    dof_handler.clear();
    dof_handler.distribute_dofs (fe);
    DoFRenumbering::component_wise (dof_handler); 
    // DoFRenumbering::Cuthill_McKee(dof_handler);
    
    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler, locally_relevant_dofs);
    
    // 4 solution vectors with ghost entries
    current_solution.reinit (locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
    old_solution.reinit     (locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
    older_solution.reinit   (locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
    oldest_solution.reinit  (locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
    // previous entropy with ghost entries. jcr 
    old_entropy.reinit      (locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
    // tmp vector. what for? jcr
    tmp_vect_relev.reinit   (locally_owned_dofs,locally_relevant_dofs,mpi_communicator);

    // tmp vector of only locally-owned size
    tmp_vect_owned.reinit   (locally_owned_dofs,mpi_communicator);
    // rhs
    system_rhs.reinit       (locally_owned_dofs,mpi_communicator);
    
    // Y is a solution. It needs to have ghost entries
    Y = std::vector<LA::MPI::Vector> (n_stages, LA::MPI::Vector ());
    for (unsigned short stage = 0; stage < n_stages; ++stage) 
      Y[stage].reinit (locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
      
    // previous_f contributes to the system rhs. It only needs to be of size locally_owned
    previous_f = std::vector<LA::MPI::Vector> (n_stages, LA::MPI::Vector ());
    for (unsigned short stage = 0; stage < n_stages; ++stage) 
      previous_f[stage].reinit (locally_owned_dofs,mpi_communicator);
      
    // initialize cell iterators
    typename DoFHandler<dim>::active_cell_iterator  cell = dof_handler.begin_active(),
                                                    endc = dof_handler.end();
      
    // ------------------------------
    // compute h_min and total volume
    // ------------------------------
    double h_min_local = 1.E100;
    double volume_of_domain_local = 0.0;
    map_h_local.clear();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
      {
        h_min_local = std::min( h_min_local , cell->diameter() );
        volume_of_domain_local += cell->measure();
        map_h_local[cell] = cell->diameter();
        // alt.: map_h_local[cell] = std::pow( cell->measure(), 1./dim ) ;
      }
    // get values from all partitions
    MPI_Allreduce(&h_min_local, &h_min, 1, MPI_DOUBLE, MPI_MIN, mpi_communicator);
    // make sure volume_of_domain has been reset to 0.
    volume_of_domain = 0.0;
    MPI_Allreduce(&volume_of_domain_local, &volume_of_domain, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    
  }
  
  // -------------------
  // compute constraints
  // -------------------
  constraints.clear ();
  constraints.reinit (locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints (dof_handler, constraints);
  
  for (unsigned int bd_id = 0; bd_id < parameters->n_boundaries; ++bd_id)
    for (unsigned int cmp_i = 0; cmp_i < MyComponents::n_components; ++cmp_i)
      if (parameters->boundary_conditions[bd_id].type_of_bc[cmp_i] == Parameters::Dirichlet) 
      {
        if(console_print_out_>=5)
          pcout << "boundary_id = " << bd_id << "\t"             << parameters->boundary_conditions[bd_id].type_of_bc[cmp_i]  << std::endl;
        std::vector<bool> mask (MyComponents::n_components, false);
        mask[cmp_i] = true;
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  bd_id,
                                                  ZeroFunction<dim>(MyComponents::n_components),
                                                  constraints,
                                                  mask);
      }
  
  constraints.close(); 

  // ----------------
  // sparsity pattern
  // ----------------

  /*
  In case the constraints are already taken care of in this function, it is possible 
  to neglect off-diagonal entries in the sparsity pattern. 
  When using ConstraintMatrix::distribute_local_to_global during assembling, 
  no entries will ever be written into these matrix position, so that one can save 
  some computing time in matrix-vector products by not even creating these elements. 
  In that case, the variable keep_constrained_dofs needs to be set to false.
  */ // jcr ask about this

  const bool keep_constrained_dofs = false;
  DynamicSparsityPattern dyn_sparsity_pattern (locally_relevant_dofs);
  DoFTools::make_sparsity_pattern (dof_handler, 
                                   dyn_sparsity_pattern, 
                                   constraints, keep_constrained_dofs); 
  SparsityTools::distribute_sparsity_pattern (dyn_sparsity_pattern,
                                              dof_handler.n_locally_owned_dofs_per_processor(),
                                              mpi_communicator,
                                              locally_relevant_dofs);
  
  // system matrix 
  system_matrix.reinit (locally_owned_dofs,locally_owned_dofs,dyn_sparsity_pattern,mpi_communicator);
  // mass matrix 
  mass_matrix.reinit (locally_owned_dofs,locally_owned_dofs,dyn_sparsity_pattern,mpi_communicator);

  // -----------------------------------
  // entropy viscosity tuning parameters
  // -----------------------------------
  c_max = parameters->c_max;
  c_ent = parameters->c_ent;
  c_jmp = parameters->c_jmp;
  
}


// **********************************************************************************
// ---                          ---
// --- compute_entropy_residual ---
// ---                          ---
// **********************************************************************************

template<int dim>
void BurgersProblem<dim>::compute_entropy_residual(const double dt)
{
  /** computes the entropy residual as 
    R = (Eold-Eolder)/dt + E'.f'.grad(u)
    
    the residual eq is obtained by multiplying the governing equation, dU/dU+div(f) by E'=dE/du
    The spatial term becomes: 
      E' div(f) = E' (df1/dx + df2/dy)  
                = E' (df1/du du/dx + df1/dv dv/dx + df2/du du/dy + df2/dv dv/dy)
    For Burgers, f=[f1,f2], with f1=u^/2 and f2=v^2/2
  */
  
  const UpdateFlags update_flags = update_values | update_gradients | update_JxW_values ;
  FEValues<dim> fe_values (fe, quadrature, update_flags);

  // local vars
  std::vector<double> u_old_local   (n_q_pts);
  std::vector<double> u_older_local (n_q_pts);
  std::vector<Tensor<1,dim> > grad_u_old_local    (n_q_pts);
  std::vector<Tensor<1,dim> > flx_prime_old_local (n_q_pts);
  
  std::vector<double> entropy_old_local      (n_q_pts);
  std::vector<double> entropy_older_local    (n_q_pts);
  std::vector<double> entropy_prime_old_local(n_q_pts);
  std::vector<double> ent_residual_local     (n_q_pts);

  if(console_print_out_>=5)
    pcout <<"dt in entropy residual " << dt << "\tvolume " << volume_of_domain << std::endl;

  typename DoFHandler<dim>::active_cell_iterator  cell = dof_handler.begin_active(),
                                                  endc = dof_handler.end();
  
  // reset the average entropy value
  double entropy_average_local = 0.;
  
  // loop on the active cells
  for (; cell!=endc; ++cell) 
    if (cell->is_locally_owned()) 
    {
      fe_values.reinit (cell);                 
      // local values of u      
      fe_values.get_function_values(old_solution   ,u_old_local);
      fe_values.get_function_values(older_solution ,u_older_local);
      // local values of grad u      
      fe_values.get_function_gradients(old_solution ,grad_u_old_local);
      
      for (unsigned int q = 0; q < n_q_pts; ++q) 
      {
        // local values of entropy at quad pts
        entropy_old_local[q]     = flux.entropy(u_old_local[q]);
        entropy_older_local[q]   = flux.entropy(u_older_local[q]);       
        // local values of entropy prime
        entropy_prime_old_local[q] = flux.entropy_prime(u_old_local[q]);
        // local values of f_ prime
        flx_prime_old_local[q]     = flux.flx_f_prime(u_old_local[q]);
        
        // increment the average entropy
        entropy_average_local +=entropy_old_local[q] * fe_values.JxW(q);
        
        /*        
        std::cout << "entropies "   << entropy_old_local[q]       << " " << entropy_older_local[q] 
                  << ", ent prime " << entropy_prime_old_local[q] 
                  << ", f  prime  " << flx_prime_old_local[q]     
                  << ", grad u    " << grad_u_old_local[q]
                  << std::endl;  
        */

        // evaluate the entropy residual on each cell, at each qp
        ent_residual_local[q]  = ( entropy_old_local[q] - entropy_older_local[q] ) / dt;
        ent_residual_local[q] += entropy_prime_old_local[q] * flx_prime_old_local[q] * grad_u_old_local[q] ;
        ent_residual_local[q] = std::fabs(ent_residual_local[q]); 
      } // end quadrature pts loop

      // save the local residual at each qp
      // using maps (converting std::vector to Vector)
      map_entropy_residual_K_q[cell] = Vector<double>(ent_residual_local.begin(),ent_residual_local.end());      
      // obtain the largest residual on a given cell
      map_entropy_residual_K[cell] = *max_element( ent_residual_local.begin(), ent_residual_local.end() );;
      
      /*
      std::cout << "entropy_residual_K_q[ii](q) = ";
      for (unsigned int q = 0; q < n_q_pts; ++q)
        std::cout << map_entropy_residual_K_q[cell](q) << ", ";
      std::cout << std::endl;  
      std::cout << " entropy_residual_K(ii) = " << map_entropy_residual_K[cell] << std::endl;
      */

    } // end locally_owned if statement
  
  // finalize entropy average value (obtain values from partitions)
  entropy_average_local /= volume_of_domain;
  entropy_average=0.0;
  MPI_Allreduce(&entropy_average_local, &entropy_average, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
  if(console_print_out_>=5)
    pcout << "entropy_average = " << entropy_average << std::endl;
  
  // compute || S - Save || _\infty
  double entropy_normalization_local = -1.0;
  cell = dof_handler.begin_active(); // need to reset cell to the beginning of the list
  for (; cell!=endc; ++cell) 
    if (cell->is_locally_owned()) 
    {
      fe_values.reinit (cell);                 
      fe_values.get_function_values(old_solution,u_old_local);
      for (unsigned int q = 0; q < n_q_pts; ++q)
        entropy_normalization_local = std::max( entropy_normalization_local , 
                                                std::fabs(entropy_average - flux.entropy(u_old_local[q]))
                                                );
    } // end locally_owned if statement
  
  // compute and distribute entropy_normalization among all procs
  entropy_normalization=-1.0;
  MPI_Allreduce(&entropy_normalization_local, &entropy_normalization, 1, MPI_DOUBLE, MPI_MAX, mpi_communicator);
  if(console_print_out_>=5)
    pcout << "entropy_normalization = " << entropy_normalization << std::endl;

}
  
// **********************************************************************************
// ---                           ---
// --- compute_entropy_viscosity ---
// ---                           ---
// **********************************************************************************

template<int dim>
void BurgersProblem<dim>::compute_entropy_viscosity()
{
//  std::string str = "visc_ent_" + Utilities::int_to_string(time_step_no,4) + ".txt"; std::ofstream o (str.c_str()); 
//  entropy_viscosity_K.print(o, 10,true,false);

  Vector<double> ones(n_q_pts);
  
  //  using maps instead
  typename DoFHandler<dim>::active_cell_iterator  cell = dof_handler.begin_active(),
                                                  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned()) 
    {
      
      // compute entropy viscosity on the whole cell
      map_entropy_viscosity_K[cell] = std::pow(map_h_local[cell],2) 
                    * ( c_ent * map_entropy_residual_K[cell] + c_jmp * map_jumps_K[cell] ) 
                    / entropy_normalization ;
      
      // compute entropy viscosity at qp
      // store in aux Vector the entropy residual on cell K for all quadrature pts
      Vector<double> aux = map_entropy_residual_K_q[cell]; 
      aux *= c_ent;
      // add the constant jump value for cell K
      // old way: aux += c_jmp * map_jumps_K[cell];
      // new way:
      ones = 1.0; ones *= c_jmp * map_jumps_K[cell];
      aux += ones ;
      // multiply by c_ent h^2 / norm_value
      aux *= ( std::pow(map_h_local[cell],2) / entropy_normalization );
      // store in appropriate map
      map_entropy_viscosity_K_q[cell] =  aux ;
      
      // now that the raw jump value is no longer needed, put the final value for output purposes
      map_jumps_K[cell] *= std::pow(map_h_local[cell],2) / entropy_normalization;
    }
  
}

// **********************************************************************************
// ---                               ---
// --- compute_first_order_viscosity ---
// ---                               ---
// **********************************************************************************

template<int dim>
void BurgersProblem<dim>::compute_first_order_viscosity()
{
  // fe stuff
  const UpdateFlags update_cell_flags = update_values ;
  FEValues<dim> fe_values (fe, quadrature, update_cell_flags);

  typename DoFHandler<dim>::active_cell_iterator  cell = dof_handler.begin_active(),
                                                  endc = dof_handler.end();
  // local vars
  std::vector<double> u_current_local(n_q_pts);
  Tensor<1,dim> flx_prime_current_local;
  std::vector<double> propagation_speed_local(n_q_pts);
  double max_speed;

  // loop over active cells
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned()) 
    {
      fe_values.reinit (cell);                 
      fe_values.get_function_values(current_solution,u_current_local);
      
      for (unsigned int q = 0; q < n_q_pts; ++q)
      {
        flx_prime_current_local    = flux.flx_f_prime(u_current_local[q]);
        propagation_speed_local[q] = flux.propagation_speed(flx_prime_current_local);
      }
      max_speed = *max_element( propagation_speed_local.begin(), propagation_speed_local.end() );
      // use maps for 1st order visc on cell
      map_first_order_viscosity_K[cell] = c_max * map_h_local[cell] * max_speed;
      // use maps for 1st order visc at quad pts
      Vector<double> aux(propagation_speed_local.begin(),propagation_speed_local.end());
      aux *= c_max * map_h_local[cell];
      map_first_order_viscosity_K_q[cell] =  aux ;
      
      /*      std::cout << "first_order_K_q[ii](q) = ";
      for (unsigned int q = 0; q < n_q_pts; ++q)
        std::cout << map_first_order_viscosity_K_q[cell](q) << ", ";
      std::cout << std::endl;  
      std::cout << " first_order_K(ii) = " << map_first_order_viscosity_K[cell] 
                << "\n max speed \t"       << max_speed 
                << "\t h_local "           << map_h_local[cell] 
                << " cmax "                << c_max 
                << std::endl;
      */

    } // end locally_owned if statement
  
  //  std::string str = "visc_1st_" + Utilities::int_to_string(time_step_no,4) + ".txt"; std::ofstream o (str.c_str()); 
  //  first_order_viscosity_K.print(o, 10,true,false);
}

// **********************************************************************************
// ---               ---
// --- compute_jumps ---
// ---               ---
// **********************************************************************************

template<int dim>
void BurgersProblem<dim>::compute_jumps()
{
  // fe stuff
  const unsigned int faces_per_cell = GeometryInfo<dim>::faces_per_cell;
  const UpdateFlags update_flags = update_values | update_gradients | update_normal_vectors;

  FEFaceValues<dim> fe_values_face(      fe, face_quadrature, update_flags );
  FEFaceValues<dim> fe_values_face_neigh(fe, face_quadrature, update_flags);

  // local vars
  std::vector<double> local_jumps_face(n_q_pts_face);
  std::vector<double> u_old_local(n_q_pts_face);

  typename DoFHandler<dim>::active_cell_iterator  cell = dof_handler.begin_active(),
                                                  endc = dof_handler.end();
  // loop over active cells
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
    {
      double max_jump_in_cell = -1.0;
      double max_jump_on_face;
      // loop on faces
      for (unsigned int iface = 0; iface < faces_per_cell; ++iface)
      {
        typename DoFHandler<dim>::face_iterator face = cell->face(iface);
        if( face->at_boundary() == false )
        {
          Assert (cell->neighbor(iface).state() == IteratorState::valid,ExcInternalError());
          typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(iface);
          //Return the how-many'th neighbor this cell is of cell->neighbor(neighbor), i.e. return the face_no such that
          //      cell->neighbor(neighbor)->neighbor(face_no)==cell. 
          // This function is the right one if you want to know how to get back from a neighbor to the present cell.
          // find which number the current face has relative to the neighboring cell
          const unsigned int ineighbor = cell->neighbor_of_neighbor(iface);
          
          Assert (ineighbor < GeometryInfo<dim>::faces_per_cell, ExcInternalError());
          
          // reinit the FE face values
          fe_values_face.reinit(      cell    ,iface    );
          fe_values_face_neigh.reinit(neighbor,ineighbor);
          
          // local arrays (should move memory allocation outside!!!!)
          std::vector< Tensor<1,dim> > grad_face( n_q_pts_face);
          std::vector< Tensor<1,dim> > grad_neigh(n_q_pts_face);
          double aux_face;
          
          // get the face gradients for u
          fe_values_face.get_function_gradients(      old_solution,grad_face );
          fe_values_face_neigh.get_function_gradients(old_solution,grad_neigh);
          
          // get the face values for u and the entropy
          fe_values_face.get_function_values(old_solution,u_old_local);
          
          // get the face normal vector
          std::vector< Point<dim> > normal_vectors(n_q_pts_face);
          normal_vectors = fe_values_face.get_normal_vectors();
          
          // compute the absolute value of jump in the normal derivative
          max_jump_on_face = -1.0;
          for (unsigned int q=0; q<n_q_pts_face; ++q) 
          {
            grad_face[q] -= grad_neigh[q];                // compute difference in gradients
            aux_face = grad_face[q] * normal_vectors[q];  // dot with normal
            aux_face *= flux.entropy(u_old_local[q]);
            aux_face = std::abs(aux_face);                // take asbolute value
            max_jump_on_face = std::max( max_jump_on_face , aux_face ); // find maximum value on that face
          }
          // double max_jump_on face = *std::max_element( jumps_face.begin(), jumps_face.end() );
          
        } // end conditional at_boundary() 
        
        // update max jump in current cell
        max_jump_in_cell = std::max( max_jump_in_cell, max_jump_on_face ); 
      } // end loop on iface
      
      // store max jump in map (not yet devided by the entropy normalization constant)
      map_jumps_K[cell] =  max_jump_in_cell ;
      //map_jumps_K[cell] =  0.0 ; // jcr dbg
      
    } // end locally_owned if statement

  // debug print out
  //  std::string str = "jumps_" + Utilities::int_to_string(time_step_no,4) + ".txt"; std::ofstream o (str.c_str()); 
  //  jumps_K.print(o, 10,true,false);

}

// **********************************************************************************
// ---                   ---
// --- compute_viscosity ---
// ---                   ---
// **********************************************************************************

template<int dim>
void BurgersProblem<dim>::compute_viscosity(const double dt)
{

  enum visc_t {constant_viscosity=0, first_order_viscosity=1, entropy_viscosity=2};
  
  //  pcout << "visc OPTION : " << parameters->viscosity_option << ", " << first_order_viscosity << std::endl;

//  std::string str = "visc_" + Utilities::int_to_string(time_step_no,4) + ".txt"; std::ofstream o (str.c_str()); 
//  viscosity_K.print(o, 10,true,false);

  Vector<double> ones(n_q_pts);
       
  typename DoFHandler<dim>::active_cell_iterator  cell = dof_handler.begin_active(),
                                                  endc = dof_handler.end();
  switch(parameters->viscosity_option){
  case constant_viscosity:
    {
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
      {
        map_viscosity_K[cell] = parameters->const_visc;
	      // ones *= 0.0; ones.add(1.0);
        ones = 1.0;
        ones *= parameters->const_visc;
        map_viscosity_K_q[cell] = ones;
      } // end locally_owned if statement
    break;
    }
  case first_order_viscosity:
    {
    compute_first_order_viscosity();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
      {
        map_viscosity_K[cell]   = map_first_order_viscosity_K[cell];
        // map_viscosity_K_q[cell] = map_first_order_viscosity_K_q[cell];
	      // ones *= 0.0; ones.add(1.0);
	      ones = 1.0;
        ones *= map_first_order_viscosity_K[cell];
        map_viscosity_K_q[cell] = ones;
      } // end locally_owned if statement
    break;
    }
  case entropy_viscosity:
    {
    compute_entropy_residual(dt);
    compute_jumps();
    compute_entropy_viscosity();
    compute_first_order_viscosity();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
      {
        map_viscosity_K[cell]   = std::min( map_entropy_viscosity_K[cell]  , map_first_order_viscosity_K[cell]   ) ;
        // should use the line below
        Vector<double> aux = map_entropy_viscosity_K_q[cell];
        for( unsigned int k=1 ; k<n_q_pts; ++k)
           if ( map_first_order_viscosity_K_q[cell](k)< map_entropy_viscosity_K_q[cell](k) )
              aux(k) = map_first_order_viscosity_K_q[cell](k);           
        map_viscosity_K_q[cell] = aux ;
        // for comparison, I temporary use
	      // ones *= 0.0; ones.add(1.0);
	      ones = 1.0;
        ones *= std::min( map_entropy_viscosity_K[cell],map_first_order_viscosity_K[cell] ); 
        map_viscosity_K_q[cell] = ones;      
      } // end locally_owned if statement
    break;
    }
  }

 /*
 // viscosity for output
  unsigned int n_act_cells = triangulation.n_active_cells();
  visc_output_K.reinit(n_act_cells);
  unsigned int iel=0;
  for (cell=dof_handler.begin_active(); cell!=endc; ++cell)
    {
     visc_output_K(iel) = map_viscosity_K[cell];
     ++iel;
  } // end loop over active cells
*/

}

// **********************************************************************************
// ---             ---
// --- extract map ---
// ---             ---
// **********************************************************************************
template <int dim>
void BurgersProblem<dim>::extract_map(std::map<typename DoFHandler<dim>::active_cell_iterator, double> &my_map)
{
    unsigned int n_act_cells = triangulation.n_locally_owned_active_cells();
    visc_output_K.reinit(n_act_cells);
    
    unsigned int iel=0;
    typename DoFHandler<dim>::active_cell_iterator  cell = dof_handler.begin_active(),
                                                    endc = dof_handler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
      {
        visc_output_K(iel) = my_map[cell];
        ++iel;
      } // end locally_owned if statement
}

// **********************************************************************************
// ---                  ---
// --- extract map_quad ---
// ---                  ---
// **********************************************************************************
template <int dim>
void BurgersProblem<dim>::extract_map(std::map<typename DoFHandler<dim>::active_cell_iterator, Vector<double> > &my_map)
{
    unsigned int n_act_cells = triangulation.n_locally_owned_active_cells();
    unsigned int total_points = n_act_cells * n_q_pts;
    visc_output_K.reinit(total_points);
    
    unsigned int i=0;
    typename DoFHandler<dim>::active_cell_iterator  cell = dof_handler.begin_active(),
                                                    endc = dof_handler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
      {
        for (unsigned int q = 0; q < n_q_pts; ++q) 
        {
          visc_output_K(i) = my_map[cell](q);
          ++i;
        }
      } // end locally_owned if statement
}

// **********************************************************************************
// ---            ---
// --- compute_dt ---
// ---            ---
// **********************************************************************************

template<int dim>
double BurgersProblem<dim>::compute_dt()
{
   
  const UpdateFlags update_cell_flags = update_values ;
  FEValues<dim> fe_values (fe, quadrature, update_cell_flags);
  typename DoFHandler<dim>::active_cell_iterator  cell = dof_handler.begin_active(),
                                                  endc = dof_handler.end();
  // local vars
  std::vector<double> u_current_local(n_q_pts);
  Tensor<1,dim> flx_prime_current_local;
  std::vector<double> propagation_speed_local(n_q_pts);
  double max_speed_local=-1.0;
  double aux;

  // loop over active cells
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
    {
      fe_values.reinit (cell);                 
      fe_values.get_function_values(current_solution,u_current_local);
      
      for (unsigned int q = 0; q < n_q_pts; ++q)
      {
        flx_prime_current_local    = flux.flx_f_prime(u_current_local[q]);
        propagation_speed_local[q] = flux.propagation_speed(flx_prime_current_local);
 //       pcout<<u_current_local[q]<<"\t"<<flx_prime_current_local<<"\t"<<propagation_speed_local[q]<<std::endl;
      }
      aux = *max_element( propagation_speed_local.begin(), propagation_speed_local.end() );
      max_speed_local = std::max( max_speed_local, aux );
  //    pcout<<max_speed_local<<std::endl;
    } // end locally_owned if statement

  // get values from all partitions
  double max_speed=0.;
  MPI_Allreduce(&max_speed_local, &max_speed, 1, MPI_DOUBLE, MPI_MAX, mpi_communicator);

  // update dt
  double dt = parameters->cfl * h_min / max_speed;
  if(console_print_out_ >= 1)
  {
    pcout << "compute_dt : \n" 
          << "\t cfl="       << parameters->cfl 
          << "\t h_min="     << h_min 
          << "\t max_speed=" << max_speed 
          << "\t dt="        << dt 
          << std::endl; 
  }
  return dt;
}

// **********************************************************************************
// ---                      ---
// --- assemble_mass_matrix ---
// ---                      ---
// **********************************************************************************

template <int dim>
void BurgersProblem<dim>::assemble_mass_matrix ()
{        
  // reset the matrix
  mass_matrix = 0.;

  const UpdateFlags update_cell_flags = update_values | update_JxW_values ;
  FEValues<dim> fe_values (fe, quadrature, update_cell_flags);

  const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);

  const unsigned int burgers_component  = MyComponents::burgers_component;
  
  typename DoFHandler<dim>::active_cell_iterator  cell = dof_handler.begin_active(),
                                                  endc = dof_handler.end();
  
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
    {
      fe_values.reinit (cell);                 
      cell->get_dof_indices (local_dof_indices);
      
      FullMatrix<double> local_mass (dofs_per_cell, dofs_per_cell);
    
      for (unsigned int q_point = 0; q_point < n_q_pts; ++q_point)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j) 
            local_mass(i,j) +=   fe_values.shape_value_component(i, q_point, burgers_component)
                               * fe_values.shape_value_component(j, q_point, burgers_component)
                              * fe_values.JxW(q_point);
      // add to global matrix
      constraints.distribute_local_to_global (local_mass, local_dof_indices, mass_matrix );
      
    } // end locally_owned if statement
  
  mass_matrix.compress(VectorOperation::add);
}

// **********************************************************************************
// ---                     ---
// --- compute_ss_residual ---
// ---                     ---
// **********************************************************************************

template <int dim>
void BurgersProblem<dim>::compute_ss_residual (bool is_steady_state,
                                               const double time   )
{        
  // reset to zero the global rhs vector
  system_rhs = 0.;

  // we also update the values in order to get real point to integrate, for instance, source terms
  const UpdateFlags update_cell_flags = update_values | update_gradients | update_q_points | update_JxW_values,
                    update_face_flags = update_values | update_gradients | update_q_points | update_JxW_values |
                                        update_normal_vectors;

  FEValues<dim>     fe_values      (fe, quadrature,      update_cell_flags);
  FEFaceValues<dim> fe_face_values (fe, face_quadrature, update_face_flags);

  const unsigned int burgers_component  = MyComponents::burgers_component;

  // local vars
  const unsigned int dofs_per_cell  = dof_handler.get_fe().dofs_per_cell;
  const unsigned int faces_per_cell = GeometryInfo<dim>::faces_per_cell;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  
  Vector<double>     local_f      (dofs_per_cell);
  std::vector<double>          current_solution_values(n_q_pts);
  std::vector<Tensor<1, dim> > current_solution_gradients(n_q_pts);
  std::vector<Tensor<1, dim> > flx_prime_local(n_q_pts);


  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                 endc = dof_handler.end();

  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
    {
      fe_values.reinit (cell);                 
      cell->get_dof_indices (local_dof_indices);
      
      fe_values.get_function_values   (current_solution,current_solution_values   );
      fe_values.get_function_gradients(current_solution,current_solution_gradients);
      
      for(unsigned int q = 0; q < n_q_pts; ++q)
        flx_prime_local[q] = flux.flx_f_prime(current_solution_values[q]);
          
      // the LOCAL steady state residual is (we write the SS residual on the RHS of the governing law)
      //        Burgers equation is: dU/dt + div(iflx+vflx) = 0
      //  where iflx = inviscid flux = U^2/2 = [u^2; v^2]/2
      //        vflx = viscous  flux = -mu grad(U)
      //
      // Weak form on div(fluxes) on the RHS (hence the - sign):
      // \int_K [-div(fluxes)b]  = -\int_K b div(u^2)/2      - \int_K div(vflx) b   
      //                         = -\int_K b f_prime.grad(u) + \int_K (vflx).grad(b)     - \int_{dK} vflx.n b  
      //                         = -\int_K b f_prime.grad(u) - \int_K mu grad(u).grad(b) + \int_{dK} mu grad(u).n b  
      // Note that we could also write:
      //   -\int_K b div(u^2)/2  =  \int_K grad(b) iflx  - \int_{dK} iflx.n b  
        
      local_f=0.;
      for (unsigned int q_point = 0; q_point < n_q_pts; ++q_point)
         for (unsigned int i=0; i<dofs_per_cell; ++i)
            // steady state local residual (the f(t,u) function in du/dt = f(t,u) that is on the RHS)
            local_f(i)-= (  fe_values.shape_value_component(i, q_point, burgers_component)
                            * flx_prime_local[q_point]
                            * current_solution_gradients[q_point]
                          +fe_values.shape_grad_component(i, q_point, burgers_component)
        //                  * viscosity_K_q[ii](q_point) // bad idea with 1st order visc
        //                    * viscosity_K(ii)
        //                    * map_viscosity_K[cell]
                            * map_viscosity_K_q[cell](q_point)
                            * current_solution_gradients[q_point]
                         )
                         * fe_values.JxW(q_point);
  
      // add to global vector
      constraints.distribute_local_to_global (local_f, local_dof_indices, system_rhs);
  
      for (unsigned int face = 0; face < faces_per_cell; ++face)
        if (cell->at_boundary(face))
        {
          fe_face_values.reinit (cell, face);
            
          /*
            assemble_ss_face_term (face, fe_face_values,
            local_dof_indices,
            cell->at_boundary(face),
            cell->face(face)->boundary_indicator());
          */
        }
    } // end locally_owned if statement
  
  system_rhs.compress(VectorOperation::add);

  // if we are doing a SS calc, we put the current f on the system_rhs of Newton's solve
  // and enforce the BC
  if( is_steady_state )
  {
    // the rhs of the steady state is -f ( J delta = -f )
    system_rhs *= -1.0;
    // zero-out the residual at the Dirichlet nodes
    std::map<unsigned int, double> boundary_values; // jcr scope of bv?
    const unsigned short n_boundaries = parameters->n_boundaries;
    for (unsigned short bd_id = 0; bd_id < n_boundaries; ++bd_id) 
      for (unsigned int cmp_i = 0; cmp_i < MyComponents::n_components; ++cmp_i) 
        if (parameters->boundary_conditions[bd_id].type_of_bc[cmp_i] == Parameters::Dirichlet) 
        {
          std::vector<bool> mask (MyComponents::n_components, false);
          mask[cmp_i] = true;
          VectorTools::interpolate_boundary_values (dof_handler,
                                                    bd_id,
                                                    ConstantFunction<dim>(parameters->boundary_conditions[bd_id].values[cmp_i], MyComponents::n_components),
                                                    boundary_values,              
                                                    mask);
        } // end if Dirichlet

    for (std::map<unsigned int, double>::const_iterator it = boundary_values.begin(); it != boundary_values.end(); ++it)
      system_rhs(it->first) = (it->second) ;

    system_rhs.compress(VectorOperation::insert); // jcr check insert
  }

}

// **********************************************************************************
// ---                      ---
// --- assemble_ss_jacobian ---
// ---                      ---
// **********************************************************************************

template <int dim>
void BurgersProblem<dim>::assemble_ss_jacobian (const bool is_steady_state,
                                                const double time         )

{
  // reset the jacobian matrix in steady state (in transient, it already contains the Mass matrix)
  if(is_steady_state) 
     system_matrix = 0.;

  // we also update the values in order to get real point to integrate, for instance, source terms
  const UpdateFlags update_cell_flags = update_values | update_gradients | update_q_points | update_JxW_values,
                    update_face_flags = update_values | update_gradients | update_q_points | update_JxW_values |
                                        update_normal_vectors;

  FEValues<dim>     fe_values      (fe, quadrature     , update_cell_flags);
  FEFaceValues<dim> fe_face_values (fe, face_quadrature, update_face_flags);

  const unsigned int burgers_component = MyComponents::burgers_component;

  const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
  const unsigned int faces_per_cell = GeometryInfo<dim>::faces_per_cell;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);

  FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);
  std::vector<double>          current_solution_values(n_q_pts);
  std::vector<Tensor<1, dim> > current_solution_gradients(n_q_pts);
  std::vector<Tensor<1, dim> > flx_prime_local(n_q_pts);
  std::vector<double>          viscosity(n_q_pts,0.0);
  Tensor<1, dim>  flx_second;
  for(unsigned int d=0; d<dim; ++d)
     flx_second[d]=1.0;

  typename DoFHandler<dim>::active_cell_iterator  cell = dof_handler.begin_active(),
                                                  endc = dof_handler.end();
  
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
    {
      fe_values.reinit (cell);                 
      cell->get_dof_indices (local_dof_indices);

      fe_values.get_function_values   (current_solution,current_solution_values   );
      fe_values.get_function_gradients(current_solution,current_solution_gradients);
      for(unsigned int q = 0; q < n_q_pts; ++q)
        flx_prime_local[q] = flux.flx_f_prime(current_solution_values[q]);
      
      // the LOCAL steady state residual is (we write the SS residual on the RHS of the governing law)
      //        Burgers equation is: du/dt + div(iflx+vflx) = 0
      //  where iflx = inviscid flux = u^2/2
      //        vflx = viscous  flux = -mu grad(u)
      //
      // Weak form on div(fluxes) on the RHS:
      // \int_K [-div(fluxes)b]  = -\int_K b div(u^2)/2      - \int_K div(vflx) b   
      //                         = -\int_K b f_prime.grad(u) + \int_K (vflx).grad(b)     - \int_{dK} vflx.n b  
      //                         = -\int_K b f_prime.grad(u) - \int_K mu grad(u).grad(b) + \int_{dK} mu grad(u).n b  
      // Note that we could also write:
      //   -\int_K b div(u^2)/2  =  \int_K grad(b) iflx  - \int_{dK} iflx.n b  
      //
      // The ss jacobian is:
      //        -\int_K bi bjf_second.grad(u) -\int_K bi f_prime.grad(bj) 
      //        - \int_K mu grad(bi).grad(bj)  + \int_{dK} mu grad(bj).n bi  
      //
      // Alternate SSJac:
      //        +\int_K grad(bi).f_prime bj -\int_{dK} bi bj f_prime.n bi
      //        - \int_K mu grad(bi).grad(bj)  + \int_{dK} mu grad(bj).n bi  

      local_matrix=0.0;
      for (unsigned int q_point = 0; q_point < n_q_pts; ++q_point)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                local_matrix(i,j) -= (fe_values.shape_value_component(i, q_point, burgers_component)
                                       * (  fe_values.shape_value_component(j, q_point, burgers_component)
                                           *flx_second
                                           *current_solution_gradients[q_point]
                                          + flx_prime_local[q_point]
                                           *fe_values.shape_grad_component(j, q_point, burgers_component)
                                         )
                                       + fe_values.shape_grad_component(i, q_point, burgers_component)
                                        *viscosity[q_point]
                                        *fe_values.shape_grad_component(j, q_point, burgers_component)
                                     )  
                                     * fe_values.JxW(q_point);
      
      // add to global matrix
      constraints.distribute_local_to_global (local_matrix, local_dof_indices, system_matrix );
  
      for (unsigned int face = 0; face < faces_per_cell; ++face)
        if (cell->at_boundary(face))
        {
          fe_face_values.reinit (cell, face);
            
                /*assemble_ss_face_term (face, fe_face_values,
                local_dof_indices,
                cell->at_boundary(face),
                cell->face(face)->boundary_indicator());*/
        }
    } // end locally_owned if statement

  system_matrix.compress(VectorOperation::add);

}


// **********************************************************************************
// ---                     ---
// --- compute_tr_residual ---
// ---                     ---
// **********************************************************************************

template <int dim>
void BurgersProblem<dim>::compute_tr_residual (const unsigned short stage_i   ,
                                               const double         stage_time,
                                               const double         dt        )
{        
  
  // these vectors do not contain locally relevant dofs
  LA::MPI::Vector u     (locally_owned_dofs,mpi_communicator);
  LA::MPI::Vector u_old (locally_owned_dofs,mpi_communicator);
  
  // put M(Y_i - u_n) in residual G_i, current_solution plays the role of Y_i
  // jcr why only locally owned when doing M*u ?
  u     = current_solution;
  u_old = old_solution;
  u -= u_old;
  // mass_matrix.vmult(previous_f[stage_i], u); 
  mass_matrix.vmult(tmp_vect_owned, u); 
  
  // compute f(t_i,Y_i) ( system_rhs is reset to 0 in compute_ss_residual to keep f(Y_i) )
  bool is_steady_state = false ;
  compute_ss_residual(is_steady_state,stage_time);
  // save f(t_i,Y_i)
  previous_f[stage_i] = system_rhs ;
  
  // add contribution (steady state residual) of the current stage
  // after this line, rhs = M(Y_i-un) -dt.a_{ii} f(t_i,Y_i)
  system_rhs.sadd( -dt*a[stage_i][stage_i] , tmp_vect_owned) ; 

  // add contribution of the previous stages
  for (unsigned short stage_j = 0; stage_j < stage_i; ++stage_j) 
    system_rhs.add(-dt*a[stage_i][stage_j] , previous_f[stage_j]);
    // system_rhs -= (dt*a[stage_i][stage_j]) *  previous_f[stage_j];
 
  // make this the rhs of the linear system J^{tr} delta = -G
  system_rhs *= -1.0; // if matrix-free, we do not want this anymore

  // zero-out the residual G(Y) at the Dirichlet nodes ???????
  std::map<unsigned int, double> boundary_values; 
  const unsigned short n_boundaries = parameters->n_boundaries;
  for (unsigned short bd_id = 0; bd_id < n_boundaries; ++bd_id) 
    for (unsigned int cmp_i = 0; cmp_i < MyComponents::n_components; ++cmp_i) 
      if (parameters->boundary_conditions[bd_id].type_of_bc[cmp_i] == Parameters::Dirichlet) 
      {
        std::vector<bool> mask (MyComponents::n_components, false);
        mask[cmp_i] = true;
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  bd_id,
                                                  ZeroFunction<dim>(MyComponents::n_components), 
// why is it a constant function to zero out the residual? ConstantFunction<dim>(parameters->boundary_conditions[bd_id].values[cmp_i], MyComponents::n_components),
                                                  boundary_values,              
                                                  mask);
      } // end if Dirichlet

  for (std::map<unsigned int, double>::const_iterator it = boundary_values.begin(); it != boundary_values.end(); ++it)
    system_rhs(it->first) = (it->second);

  system_rhs.compress(VectorOperation::insert);

}


// **********************************************************************************
// ---                      ---
// --- assemble_tr_jacobian ---
// ---                      ---
// **********************************************************************************

template <int dim>
void BurgersProblem<dim>::assemble_tr_jacobian (const unsigned short stage_i    ,
                                                const double         stage_time ,
                                                const double         dt         ,
                                                const bool           is_explicit)
{        

  // if explicit, no need to spend time in computing the steady-state jacobian, 
  // nor in re-assigning M to J
  if(is_explicit)
     return;
  
  // reset system matrix (put 0's)
  system_matrix = 0.;

  // if ESDIRK method, a[i][i] can be zero for a given stage (usually the first one)
  if( std::abs( a[stage_i][stage_i] ) < 1e-8 ) 
  {
    system_matrix.add( mass_matrix , 1.0 );
    //system_matrix.add( 1.0 , mass_matrix);
    return;
  }

  // we do this so that the system matrix will have 1 on the diagonal for Dirichlet nodes, 
  // otherwise it is:  1-dt.a_{ii}  and we do not want this to be = to zero for some given dt
  // Also: J = M -dt.a.SSjac = -dt.a (M/(-dt.a) + SSjac)
  system_matrix.add( mass_matrix, -1.0/(dt*a[stage_i][stage_i]) );
  //system_matrix.add( -1.0/(dt*a[stage_i][stage_i]) , mass_matrix);
  // compute ss jacobian 
  // (system_matrix is NOT reset to 0 in compute_ss_jacobian if is_steady_state=false is given)
  bool is_steady_state = false;
  assemble_ss_jacobian(is_steady_state, stage_time);
  // multiply by: -dt.a_{ii} in order to get J = M - dt.a_{ii}*SSjac
  //   remember that Dirichlet BCs have G=0 so it is not important 
  //   that we multiplied the diagonal entries of these rows here
  system_matrix *= ( -dt*a[stage_i][stage_i] ) ;

  /* alternate that may not work for non-zero Dirichlet BC
  //    jcr: why, G=0 for any Dirichlet BCs...
  // compute ss jacobian 
  assemble_ss_jacobian(stage_time);

  system_matrix *= ( -dt*a[stage_i][stage_i] ) ;
    
  system_matrix.add( 1.0 , mass_matrix);
  */

}


/*

// **********************************************************************************
// ---                               ---
// --- compute_ss_residual_cell_term ---
// ---                               ---
// **********************************************************************************

template <int dim>
void BurgersProblem<dim>::compute_ss_residual_cell_term (const FEValues<dim>             &fe_values,
                                                         const std::vector<unsigned int> &local_dof_indices,
                                                         const double                     time             ,
                                                         const bool                       is_steady_state  ,
                                                         const typename DoFHandler<dim>::active_cell_iterator cell)
{

  // jcr this better inside this function or outside the loop ?
  const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
  //  const unsigned int n_components  = MyComponents::n_components;
  const unsigned int burgers_component  = MyComponents::burgers_component;
  // jcr this better inside this function or outside the loop ?
  Vector<double>     local_f      (dofs_per_cell);
  // same
  std::vector<double>          current_solution_values(n_q_pts);
  std::vector<Tensor<1, dim> > current_solution_gradients(n_q_pts);
  std::vector<Tensor<1, dim> > flx_prime_local(n_q_pts);
  std::vector<double>          viscosity(n_q_pts,0.0);

  fe_values.get_function_values   (current_solution,current_solution_values   );
  fe_values.get_function_gradients(current_solution,current_solution_gradients);

  for(unsigned int q = 0; q < n_q_pts; ++q)
    flx_prime_local[q] = flux.flx_f_prime(current_solution_values[q]);
          
  // the LOCAL steady state residual is (we write the SS residual on the RHS of the governing law)
  //        Burgers equation is: du/dt + div(iflx+vflx) = 0
  //  where iflx = inviscid flux = u^2/2
  //        vflx = viscous  flux = -mu grad(u)
  //
  // Weak form on div(fluxes) on the RHS:
  // \int_K [-div(fluxes)b]  = -\int_K b div(u^2)/2      - \int_K div(vflx) b   
  //                         = -\int_K b f_prime.grad(u) + \int_K (vflx).grad(b)     - \int_{dK} vflx.n b  
  //                         = -\int_K b f_prime.grad(u) - \int_K mu grad(u).grad(b) + \int_{dK} mu grad(u).n b  
  // Note that we could also write:
  //   -\int_K b div(u^2)/2  =  \int_K grad(b) iflx  - \int_{dK} iflx.n b  
  
  local_f=0.;
  for (unsigned int q_point = 0; q_point < n_q_pts; ++q_point)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        // steady state local residual (the f(t,u) function in du/dt = f(t,u), that is on the RHS of that eq. )
        local_f(i)-= (  fe_values.shape_value_component(i, q_point, burgers_component)
                        * flx_prime_local[q_point]
                        * current_solution_gradients[q_point]
                      + fe_values.shape_grad_component(i, q_point, burgers_component)
      //                  * viscosity_K_q[ii](q_point) // bad idea with 1st order visc
      //                    * viscosity_K(ii)
      //                    * map_viscosity_K[cell]
                          * map_viscosity_K_q[cell](q_point)
                        * current_solution_gradients[q_point]
                     )
                     * fe_values.JxW(q_point);
  
  // add to global vector
  constraints.distribute_local_to_global (local_f, local_dof_indices, system_rhs);

}


// **********************************************************************************
// ---                       ---
// --- assemble_ss_cell_term ---
// ---                       ---
// **********************************************************************************

template <int dim>
void BurgersProblem<dim>::assemble_ss_jacobian_cell_term (const FEValues<dim>             &fe_values,
                                                          const std::vector<unsigned int> &local_dof_indices,
                                                          const double                     time              )
{

  // jcr this better inside this function or outside the loop ?
  const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
  const unsigned int burgers_component  = MyComponents::burgers_component;
  // jcr this better inside this function or outside the loop ?
  FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);
  // same question
  std::vector<double>          current_solution_values(n_q_pts);
  std::vector<Tensor<1, dim> > current_solution_gradients(n_q_pts);
  std::vector<Tensor<1, dim> > flx_prime_local(n_q_pts);
  std::vector<double>          viscosity(n_q_pts,0.0);
  Tensor<1, dim>  flx_second;
  for(unsigned int d=0; d<dim; ++d)
     flx_second[d]=1.0;
       
  //  std::vector<std::vector<double> > conductivity_values(n_q_points,std::vector<double>(2,0.0));

  fe_values.get_function_values   (current_solution,current_solution_values   );
  fe_values.get_function_gradients(current_solution,current_solution_gradients);
  for(unsigned int q = 0; q < n_q_pts; ++q)
    flx_prime_local[q] = flux.flx_f_prime(current_solution_values[q]);
      
  // the LOCAL steady state residual is (we write the SS residual on the RHS of the governing law)
  //        Burgers equation is: du/dt + div(iflx+vflx) = 0
  //  where iflx = inviscid flux = u^2/2
  //        vflx = viscous  flux = -mu grad(u)
  //
  // Weak form on div(fluxes) on the RHS:
  // \int_K [-div(fluxes)b]  = -\int_K b div(u^2)/2      - \int_K div(vflx) b   
  //                         = -\int_K b f_prime.grad(u) + \int_K (vflx).grad(b)     - \int_{dK} vflx.n b  
  //                         = -\int_K b f_prime.grad(u) - \int_K mu grad(u).grad(b) + \int_{dK} mu grad(u).n b  
  // Note that we could also write:
  //   -\int_K b div(u^2)/2  =  \int_K grad(b) iflx  - \int_{dK} iflx.n b  
  //
  // The ss jacobian is:
  //        -\int_K bi bjf_second.grad(u) -\int_K bi f_prime.grad(bj) 
  //        - \int_K mu grad(bi).grad(bj)  + \int_{dK} mu grad(bj).n bi  
  //
  // Alternate SSJac:
  //        +\int_K grad(bi).f_prime bj -\int_{dK} bi bj f_prime.n bi
  //        - \int_K mu grad(bi).grad(bj)  + \int_{dK} mu grad(bj).n bi  

  for (unsigned int q_point = 0; q_point < n_q_pts; ++q_point)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            local_matrix(i,j) -= (fe_values.shape_value_component(i, q_point, burgers_component)
                                   * (  fe_values.shape_value_component(j, q_point, burgers_component)
                                       *flx_second
                                       *current_solution_gradients[q_point]
                                      + flx_prime_local[q_point]
                                       *fe_values.shape_grad_component(j, q_point, burgers_component)
                                     )
                                   + fe_values.shape_grad_component(i, q_point, burgers_component)
                                    *viscosity[q_point]
                                    *fe_values.shape_grad_component(j, q_point, burgers_component)
                                 )  
                                 * fe_values.JxW(q_point);
  
  // add to global matrix
  constraints.distribute_local_to_global (local_matrix, local_dof_indices, system_matrix );

}
*/

// **********************************************************************************
// ---        ---
// --- solve  ---
// ---        ---
// **********************************************************************************


template <int dim>
std::pair<unsigned int, double> BurgersProblem<dim>::linear_solve (LA::MPI::Vector &newton_update)
{ 
  switch (parameters->solver)
  {
    case Parameters::Solver::direct:
    {
      // compute linear tolerance based on atol and rtol values
      const double linear_tol = parameters->linear_rtol*system_rhs.l2_norm() + parameters->linear_atol ;

      SolverControl  solver_control (system_rhs.size(), linear_tol );
      LA::SolverCG  solver (solver_control);
      // jcr : is it ok for performance to create this solution vector here each time?
      LA::MPI::Vector completely_dist_solution(locally_owned_dofs,mpi_communicator);
      
//      LA::MPI::PreconditionSSOR preconditioner;
      dealii::PETScWrappers::PreconditionNone preconditioner ;
      preconditioner.initialize(system_matrix);

      // solver only accepts locally owned dofs, thus use completely_dist_solution as container for the solve result
      solver.solve (system_matrix, completely_dist_solution, system_rhs, preconditioner );
      constraints.distribute(completely_dist_solution); // jcr see void ConstraintMatrix::distribute_local_to_global
      // copy ghost elements needed for the "locally relevant" part of the solution vector on each subdomain
      newton_update = completely_dist_solution;

      return std::pair<unsigned int, double> (solver_control.last_step(),
                                              solver_control.last_value() ); 

    }
    default:
      Assert( false , ExcNotImplemented() );
  }  
   
  Assert (false, ExcNotImplemented());
  return std::pair<unsigned int, double> (0,0);

}

// **********************************************************************************
// ---             ---
// --- solve Mass  ---
// ---             ---
// **********************************************************************************


template <int dim>
std::pair<unsigned int, double> BurgersProblem<dim>::mass_solve (LA::MPI::Vector &solu)
{ 
  switch (parameters->solver)
  {
    case Parameters::Solver::direct:
    {
      // compute linear tolerance based on atol and rtol values
      const double linear_tol = parameters->linear_rtol*system_rhs.l2_norm() + parameters->linear_atol ;

      SolverControl  solver_control (system_rhs.size(), linear_tol );
      LA::SolverCG solver (solver_control);
      LA::MPI::Vector completely_dist_solution(locally_owned_dofs,mpi_communicator);
 
      //PreconditionSSOR<> preconditioner;
      // const double omega = parameters->ssor_omega;
//      LA::MPI::PreconditionSSOR preconditioner;
      dealii::PETScWrappers::PreconditionNone preconditioner ;
      preconditioner.initialize(mass_matrix);
  
      // solver only accepts locally owned dofs, thus use completely_dist_solution as container for the solve result
      solver.solve (mass_matrix, completely_dist_solution, system_rhs, preconditioner );    
                     
      constraints.distribute(completely_dist_solution); 
      // copy ghost elements needed for the "locally relevant" part of the solution vector on each subdomain
      solu = completely_dist_solution;

      return std::pair<unsigned int, double> (solver_control.last_step(),
                                              solver_control.last_value() ); 
    }
    default:
      Assert( false , ExcNotImplemented() );
  }  
   
  Assert (false, ExcNotImplemented());
  return std::pair<unsigned int, double> (0,0);

}

// **********************************************************************************
// ---              ---
// --- refine_grid  ---
// ---              ---
// **********************************************************************************

template <int dim>
void BurgersProblem<dim>::refine_grid ()
{
  TimerOutput::Scope t(computing_timer, "refine");
  
  Vector<float> estimated_error_per_cell (triangulation.n_locally_owned_active_cells());

  KellyErrorEstimator<dim>::estimate (dof_handler,
                                      quadrature,
                                      typename FunctionMap<dim>::type(),
                                      current_solution,
                                      estimated_error_per_cell);

  parallel::distributed::GridRefinement::
  refine_and_coarsen_fixed_number (triangulation,
                                   estimated_error_per_cell,
                                   0.3, 0.03);

  triangulation.execute_coarsening_and_refinement ();
}


// **********************************************************************************
// ---                 ---
// --- output_solution ---
// ---                 ---
// **********************************************************************************

template <int dim>
void BurgersProblem<dim>::output_solution (const unsigned int time_step_no) const
{

  // ----------------------
  // create data_out for solution. 
  DataOut<dim> data_out;
  // attach dof
  data_out.attach_dof_handler (dof_handler);
  // add data
  data_out.add_data_vector (current_solution,
                            MyComponents::component_names (),
                            DataOut<dim>::type_dof_data,
                            MyComponents::component_interpretation ());
  Vector<float> subdomain (triangulation.n_active_cells());
  for (unsigned int i=0; i<subdomain.size(); ++i)
     subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector (subdomain, "subdomain");
  // build patches
  data_out.build_patches();

  // ----------------------
  // output solution to vtu
  const  std::string filename = "solution-" +  
                                Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) + 
                                "." +
                                Utilities::int_to_string (time_step_no, 4) ;
  std::ofstream output ((filename + ".vtu").c_str());  // write vtu file
  data_out.write_vtu (output);
  // create master record
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  {
    std::vector<std::string> filenames;
    for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
        filenames.push_back ("solution-" +
                             Utilities::int_to_string (i, 4) +
                             "." +
                             Utilities::int_to_string (time_step_no, 4) +
                             ".vtu");
    std::ofstream master_output ((filename + ".pvtu").c_str());
    data_out.write_pvtu_record (master_output, filenames);
//    data_out.write_visit_record (master_output, filenames);
  }
  // ----------------------

}

// **********************************************************************************
// ---                 ---
// --- output_viscosity  ---
// ---                 ---
// **********************************************************************************
/*
template <int dim>
void BurgersProblem<dim>::output_viscosity () 
{

  // ----------------------
    
    // create data_out for viscosity used. only PIECEWISE output    
    DataOut<dim> data_out_visc_K;
    // attach dof
    data_out_visc_K.attach_dof_handler(dof_handler);
    // add data
    extract_map(map_viscosity_K);
    data_out_visc_K.add_data_vector(visc_output_K,"viscosity_PW");
    // build patches
    data_out_visc_K.build_patches();
    // output piece-wise viscosity values to vtu
    std::string filename = "visc_K-" + Utilities::int_to_string (time_step_no, 4) + ".vtu";
    std::ofstream ofstream_viscosity_K( filename.c_str() );
    // write vtu file
    data_out_visc_K.write_vtu(ofstream_viscosity_K);

    if(parameters->viscosity_option==2){
      data_out_visc_K.clear_data_vectors ();
      // add data FIRST-ORDER VISCOSITY
      extract_map(map_first_order_viscosity_K);
      data_out_visc_K.add_data_vector(visc_output_K,"first_viscosity_PW");
      // build patches
      data_out_visc_K.build_patches();
      // output piece-wise viscosity values to vtu
      filename = "first_visc_K-" + Utilities::int_to_string (time_step_no, 4) + ".vtu";
      // write vtu file
      std::ofstream ofstream_first_K( filename.c_str() );
      data_out_visc_K.write_vtu(ofstream_first_K);

      data_out_visc_K.clear_data_vectors ();
      // add data ENTROPY VISCOSITY
      extract_map(map_entropy_viscosity_K);
      data_out_visc_K.add_data_vector(visc_output_K,"entropy_viscosity_PW");
      // build patches
      data_out_visc_K.build_patches();
      // output piece-wise viscosity values to vtu
      filename = "entr_visc_K-" + Utilities::int_to_string (time_step_no, 4) + ".vtu";
      // write vtu file
      std::ofstream ofstream_entr_K( filename.c_str() );
      data_out_visc_K.write_vtu(ofstream_entr_K);

      data_out_visc_K.clear_data_vectors ();
      // add data JUMPS VISCOSITY
      extract_map(map_jumps_K);
      data_out_visc_K.add_data_vector(visc_output_K,"jumps_viscosity_PW");
      // build patches
      data_out_visc_K.build_patches();
      // output piece-wise viscosity values to vtu
      filename = "jumps_K-" + Utilities::int_to_string (time_step_no, 4) + ".vtu";
      // write vtu file
      std::ofstream ofstream_jumps_K( filename.c_str() );
      data_out_visc_K.write_vtu(ofstream_jumps_K);
    }
    
    // output piece-wise viscosity values to vtu
    // TO-DO output viscosity at quadrature points for plotting    
    
}

*/
// **********************************************************************************
// ---                        ---
// --- output_exact_solution  ---
// ---                        ---
// **********************************************************************************


template <int dim>
void BurgersProblem<dim>::output_exact_solution (const unsigned int time_step_no) const
{
  // if the exact solution is not implemented/requested/available, skip this
  if(!parameters->has_exact_solution)
    return;

  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);

//  ExactSolution<dim>::debugprint=true;
  
  LA::MPI::Vector exact_solution(locally_owned_dofs,mpi_communicator);
  VectorTools::interpolate (dof_handler,
                            ExactSolution<dim>( MyComponents::n_components,
                                                parameters->length,
                                                parameters->amplitude,
                                                parameters->spatial_shape_option, 
                                                parameters->time_a0 ),
                            exact_solution);

  data_out.add_data_vector (exact_solution, 
                            MyComponents::component_names (),                             
                            DataOut<dim>::type_dof_data,
                            MyComponents::component_interpretation ());
  Vector<float> subdomain (triangulation.n_active_cells());
  for (unsigned int i=0; i<subdomain.size(); ++i)
     subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector (subdomain, "subdomain");
  data_out.build_patches ();
  
  // ----------------------
  // output solution to vtu
  const  std::string filename = "exact_solution-" +  
                                Utilities::int_to_string (time_step_no, 4) + 
                                "." +
                                Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4);
  std::ofstream output ((filename + ".vtu").c_str());  // write vtu file
  data_out.write_vtu (output);
  // create master record
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  {
    std::vector<std::string> filenames;
    for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
        filenames.push_back ("exact_solution-" +
                             Utilities::int_to_string (time_step_no, 4) +
                             "." +
                             Utilities::int_to_string (i, 4) +
                             ".vtu");
    std::ofstream master_output ((filename + ".pvtu").c_str());
    data_out.write_pvtu_record (master_output, filenames);
//    data_out.write_visit_record (master_output, filenames);
  }
  // ----------------------

}


// **********************************************************************************
// ---                   ---
// --- process_solution  ---
// ---                   ---
// **********************************************************************************
/*
template <int dim>
void BurgersProblem<dim>::process_solution () const
{
  // if the exact solution is not implemented/requested/available, skip this
  if(!parameters->has_exact_solution)
   return;
   
  Vector<float> difference_per_cell (triangulation.n_locally_owned_active_cells());

  VectorTools::integrate_difference (dof_handler,
                                     current_solution,
                                     ExactSolution<dim>( MyComponents::n_components, 
                                                         parameters->length,
                                                         parameters->amplitude,
                                                         parameters->spatial_shape_option,
                                                         parameters->time_a0 ),
                                     difference_per_cell,
                                     quadrature,
                                     VectorTools::L2_norm);
  const double L2_error = difference_per_cell.l2_norm();
    
  std::cout << "   Error in L2 norm: "  << L2_error
            << " Rel Err in L2 norm: " << L2_error/current_solution.l2_norm()
            << std::endl;

  const Point<dim> pt = (dim == 2   ?   Point<dim> (parameters->length/2.,parameters->length/2.)
                                    :   Point<dim> (parameters->length/2.,parameters->length/2.,parameters->length/2.)
                                 );
  Vector<double> diff (dim-1);
  VectorTools::point_difference (dof_handler,
                                 current_solution,
                                 ExactSolution<dim>( MyComponents::n_components, 
                                                     parameters->length,
                                                     parameters->amplitude,
                                                     parameters->spatial_shape_option,
                                                     parameters->time_a0 ),
                                                     diff,
                                                     pt);
  pcout << "pt diff = : " << diff(0) << std::endl;
}
*/

// **********************************************************************************
// ---      ---
// --- run  ---
// ---      ---
// **********************************************************************************

template <int dim>
void BurgersProblem<dim>::run ()
{

  // -------------------------------------------
  // initial uniform grid
  // -------------------------------------------
  const Point<dim> bottom_left = Point<dim>();
  Point<dim> upper_right;
  switch(dim){
  case 1:
    {
      upper_right = Point<dim> (parameters->length);
      break;
    }
  case 2:
    {
      // upper_right = Point<dim> (parameters->length,parameters->length);
      upper_right = Point<dim> (parameters->length,1./double(parameters->n_init_refinements_x));
      break;
    }
  case 3:
    {
      upper_right = Point<dim> (parameters->length,parameters->length,parameters->length);
      break;
    }
  }
  bool colorize=true;
  // GridGenerator::hyper_rectangle(triangulation,repetitions,bottom_left,upper_right,colorize);
  // triangulation.refine_global (parameters->n_init_refinements);
  static const unsigned int arr[] = {parameters->n_init_refinements_x,parameters->n_init_refinements_y};
  const std::vector<unsigned int> repetitions(arr, arr+sizeof(arr)/sizeof(arr[0]) );
  GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions,bottom_left,upper_right,colorize);
  if (console_print_out_ >= 10)
    pcout << "bottom " << bottom_left[0] << " " << bottom_left[1]
          << "top    " << upper_right[0] << " " << upper_right[1] << std::endl;

  // -------------------------------------------
  // system setup (DOFs, sparsity, etc.)
  // -------------------------------------------
  setup_system ();
  if (console_print_out_ >= 10)
  {
    pcout << "   Number of active cells:       " << triangulation.n_global_active_cells() << std::endl;
    pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs()                  << std::endl;
  }

  // -------------------------------------------
  // time step init
  // -------------------------------------------
  // jcr &
  const double       &time_step    = parameters->time_step;
  const double       &initial_time = parameters->initial_time;
  const double       &final_time   = parameters->final_time;
  const bool         &steady_state_first = parameters->perform_steady_state_first;
  const unsigned int  n_components = MyComponents::n_components;
  
  // time stepping
  double begin_time = initial_time;
  
  // synchronize time at which exact solution is evaluated
  ExactSolution<dim>::synchronize (&begin_time);

  // -------------------------------------------
  // distributed solution vectors
  // -------------------------------------------
  // use initial data as a starting point
  LA::MPI::Vector completely_dist_solution(locally_owned_dofs,mpi_communicator);
  VectorTools::interpolate(dof_handler,
                           InitialData<dim>(n_components,parameters->pbID), 
                           //ZeroFunction<dim>(n_components), 
                           completely_dist_solution);
                           
  // apply Dirichlet conditions at first iteration
  std::map<unsigned int, double> boundary_values; // jcr scope of bv?
  const unsigned short n_boundaries = parameters->n_boundaries;
  for (unsigned short bd_id = 0; bd_id < n_boundaries; ++bd_id) 
    for (unsigned int cmp_i = 0; cmp_i < n_components; ++cmp_i) 
      if (parameters->boundary_conditions[bd_id].type_of_bc[cmp_i] == 1) 
      {
        std::vector<bool> mask (n_components, false);
        mask[cmp_i] = true;
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  bd_id,
                                                  ConstantFunction<dim>(parameters->boundary_conditions[bd_id].values[cmp_i], n_components),
                                                  boundary_values,              
                                                  mask);
      } // end if Dirichlet
  // insert BC values in solution vector
  for (std::map<unsigned int, double>::const_iterator it = boundary_values.begin(); it != boundary_values.end(); ++it)
    completely_dist_solution(it->first) = (it->second) ;
  // synch the parallel vector
  // jcr: should we do this at the end or each time we mess with the parallel vector?
  completely_dist_solution.compress(VectorOperation::insert);

  // auxiliary vector in Newton solve
  LA::MPI::Vector newton_update (locally_owned_dofs,locally_relevant_dofs,mpi_communicator);

  // ------                             ------- //
  // ------ prepare for transient solve ------- //
  // ------                             ------- //
  if (console_print_out_ >= 1)
    pcout << "about to start transient ..." << std::endl;
  
  // jcr is this a copy? Not sure, current solution also has relevant dofs ...
  current_solution = completely_dist_solution;
  old_solution     = completely_dist_solution;
  older_solution   = completely_dist_solution;
  
  //process_solution ();
  output_solution (0);
  //  AssertThrow(false, ExcMessage(" stopping after initial output_solution"));
  output_exact_solution (0);

  assemble_mass_matrix ();
  // for fully explicit techniques, the jacobian matrix will be constant = Mass
  if(is_explicit) {
    system_matrix = 0.;
    system_matrix.add(mass_matrix , 1.0);
    //system_matrix.add(1, mass_matrix);
  }
  // AssertThrow (false, ExcMessage ("jcr stopping after steady state"));


  bool is_time_adaptive = parameters->is_cfl_time_adaptive;
  double delta_t = parameters->time_step;  // above, reference, here not, why?
  unsigned int n_time_steps=0;
  if(!is_time_adaptive)
  {
    // we want to check time convergence and compare numerical solution
    // against exact solution at final_time not around it
    unsigned int n0, n1=0;
    const double dt_accuracy = 1e-14;
    n0 = (final_time - initial_time) / delta_t;
    if( std::abs(initial_time + n0 * delta_t -final_time) <dt_accuracy)
      n1 = n0;
    else if( std::abs(initial_time + (n0+1) * delta_t -final_time) <dt_accuracy)
      n1 = n0+1;
    else if( std::abs(initial_time + (n0-1) * delta_t -final_time) <dt_accuracy)
      n1 = n0-1;
    else 
    {
      pcout << "initial_time=" << initial_time << "\tn_time_steps=" << n0
            << "\ttime_step="  << delta_t      << "\tfinal_time="   << final_time << "\n"
            << "initial_time + n_time_steps * time_step=" << initial_time + n0 * delta_t << std::endl; 
      AssertThrow(initial_time + n0 * delta_t == final_time, ExcMessage("bad choice for time step"));
    }
    // set the number of time steps
    n_time_steps = n1;
  }

  if(dim==1 && console_print_out_ >= 5)
  {
    std::string str = "initial_solution.txt"; std::ofstream o (str.c_str()); 
    current_solution.print(o, 10,true,false);
  }

  // ------                  ------- //
  // ------  transient solve ------- //
  // ------                  ------- //
  while ( begin_time < final_time ) {

    // ---------------------
    // compute a new delta t
    // ---------------------
    if(is_time_adaptive)
      delta_t = compute_dt();
    else
      delta_t = parameters->time_step;

    // adjust dt to yield the user-requested end time
    if( begin_time + delta_t >= final_time ) {
      delta_t = final_time - begin_time;
      begin_time += 1E-14; // to make sure this is the last time step
      if(console_print_out_ >= 1)
        pcout << "Reducing time step size from " << (final_time - begin_time) << " to " << delta_t 
              << "\tbegin time=" << begin_time << "\tfinal_time=" << final_time << std::endl;
    }
    // increment time step counter
    ++time_step_no;

    if(console_print_out_ >= 1)
    { 
      pcout<<"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
      pcout<<"!!-- begin_time="<< begin_time << " dt= " << delta_t << " time step no=" << time_step_no << std::endl;
    }
    
    // do we need the all setup call again? jcr, only if refinement i believe
    //system_matrix.reinit (sparsity_pattern);  

    // -----------------
    // compute viscosity
    // -----------------
    compute_viscosity( delta_t );

    // ---------------------
    // loop on the RK stages
    // ---------------------
    for (unsigned short stage_i = 0; stage_i < n_stages; ++stage_i) 
    {

      // compute c_i dt
      double stage_time = begin_time+c[stage_i]*delta_t;
      if(console_print_out_ >= 3)
        pcout<<"!!-- solving stage "<< stage_i << " at stage_time " << stage_time << std::endl;

      // ------------              ------------ //
      // ------------ Newton solve ------------ //
      // ------------              ------------ //
      unsigned int nonlin_iter = 0;
      bool newton_convergence = false;

      // compute the TR residual G(Y_i) at stage i   
      compute_tr_residual(stage_i , stage_time, delta_t );
      // compute the initial residual norm
      double res_norm = system_rhs.l2_norm();
      // compute the tolerance to satisfy based on ATOL and RTOL  
      const double tol = parameters->nonlinear_atol + parameters->nonlinear_rtol * res_norm;
      if(console_print_out_ >= 3)
        pcout << "Initial residual norm = "   << res_norm << "\t" 
              << "Tolerance value adopted = " << tol      << std::endl;

      // Newton's WHILE loop
      while ( nonlin_iter < parameters->max_nonlin_iterations && !newton_convergence ) 
      {
        if(console_print_out_ >= 3)
          pcout <<  "Newton iteration # " << nonlin_iter << "\t:";
        
        // compute the residual and the jacobiam matrix J delta u = -F(u)    
        assemble_tr_jacobian (stage_i, stage_time, delta_t, is_explicit);
        // compute initial residual
        double res_norm = res_norm;
        
        // what is below looks like BS. One can have newton_update be a locally-owned vector 
        // only, pass it to the solver, and only do the vector trick below for the 
        // current_solution update ... jcr
        
        // zero out the update vector
        newton_update.reinit(locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
        // newton_update.reinit(locally_owned_dofs,mpi_communicator);
        // solve the linear system J delta = -f  and put the solution in newton_update
        std::pair<unsigned int, double> linear_convergence = linear_solve (newton_update);
        // update Newton solution
        //current_solution.add(parameters->damping, newton_update);
        
        // create a tmp vector with only locally_owned dofs
        LA::MPI::Vector tmp(locally_owned_dofs,mpi_communicator);
        LA::MPI::Vector tm2(locally_owned_dofs,mpi_communicator);
        // copy the locally owned dofs in it
        /*        std::cout << " size " << current_solution.size() 
                  << " size " << tmp.size() 
                  << " size " << newton_update.size() 
                  << " size " << tm2.size() 
                  << std::endl;*/
        tmp = current_solution;
        tm2 = newton_update;
        
        //o.close(); str = "update" + Utilities::int_to_string(nonlin_iter,4) + ".txt"; o.open(str.c_str()); 
        //newton_update.print(o, 10,true,false);

        // add the newton update from the solve (newton_update only contains locally_owned dofs)
	      // tmp.add(1.0, tm2);
	      tmp += tm2;
        current_solution = tmp;


        if( (is_explicit) && (std::fabs(parameters->damping-1)<1e-12) ) 
        { //jcr: need {} for one line if when there is an else????
          // no need for nonlinear iteration with explicit schemes
          newton_convergence=true;
          // print out residual attained according to printout level requested
          if(console_print_out_ >= 3)
          {
            compute_tr_residual(stage_i, stage_time, delta_t);
            res_norm = system_rhs.l2_norm();
            std::printf("   %-16.3e %04d        %-5.2e %-5.2e\n", res_norm, 
                linear_convergence.first, linear_convergence.second, newton_update.l2_norm() );
          }
        }
        else
        {
          // this computes the Transient residual G(Y_i) at stage i   
          compute_tr_residual(stage_i, stage_time, delta_t);
          // compute the initial residual norm
          res_norm = system_rhs.l2_norm();
          // 
          // output_viscosity();
          // console print for convergence status: 
          //  Newton res, # of lin solve, lin solve residual, norm of Newton update vector
          if(console_print_out_ >= 3)
            std::printf("   %-16.3e %04d        %-5.2e %-5.2e\n", res_norm, 
                linear_convergence.first, linear_convergence.second, newton_update.l2_norm() );
          // check Newton convergence
          newton_convergence = res_norm < tol ;
          if ( newton_convergence && console_print_out_ >= 1)
            pcout << "  --- Newton has converged --- " << std::endl;
        }
        // increment iteration counter
        ++nonlin_iter;
      } // END of Newton's WHILE loop
     
      //      AssertThrow (false, ExcMessage ("jcr stopping"));
      if( ! newton_convergence )
        AssertThrow (false, ExcMessage ("No convergence in Newton solver"));
      
      // ------------              ------------ //
      // ------------ Newton solve ------------ //
      // ------------ END END END  ------------ //
  
    // ---------------------
    }  // END of  RK stages
    // ---------------------

    // solve: M current_sol = M_old_sol + dt sum_{i} b_i previous_f_i
    system_rhs = 0.0;
    LA::MPI::Vector tmp(locally_owned_dofs,mpi_communicator);
    tmp = old_solution;       
    mass_matrix.vmult(system_rhs,tmp);    
    for (unsigned short stage_i = 0; stage_i < n_stages; ++stage_i) 
      system_rhs.add (delta_t * b[stage_i], previous_f[stage_i]);
      // system_rhs += (delta_t * b[stage_i]) * previous_f[stage_i];
    std::pair<unsigned int, double> mass_convergence = mass_solve (current_solution);

    // output solution
    unsigned int multiple = time_step_no/vtk_output_frequency_;
    if(multiple*vtk_output_frequency_-time_step_no ==0 )
    {
      if(Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
      {
        TimerOutput::Scope t(computing_timer, "output");
        output_solution (time_step_no);
      }
    }
    // prepare next time step
    begin_time    += delta_t;
    older_solution = old_solution;
    old_solution   = current_solution;

  // ------                  ------- //
  } // end of transient loop
  // ------  transient solve ------- //
  // ------  END  END  END   ------- //

  if(console_print_out_ >= 1)
    pcout<<"\nUSING TIME METHOD "<<method_name<<"\n";

  // print out timer stats
  computing_timer.print_summary ();
  computing_timer.reset ();

}


// **********************************************************************************
// ---      ---
// --- main ---
// ---      ---
// **********************************************************************************

int main (int argc, char* argv[])
{
  deallog.depth_console (0);
  if (argc != 2)
  {
    std::cout << "Usage:" << argv[0] << " input_file" << std::endl;
    std::exit(1);
  }

  try
  {
    // Utilities::System::MPI_InitFinalize mpi_initialization (argc, argv);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    {
      const unsigned int dim = 2;

      ParameterHandler prm;
      Parameters::AllParameters<dim>::declare_parameters (prm); // done this way b/c static (this is ugly)

      char *input_filename = argv[1];
      // prm.print_parameters (std::cout, ParameterHandler::Text);
      prm.read_input (input_filename);

      Parameters::AllParameters<dim> parameters;
      parameters.parse_parameters (prm);  // done this way b/c not static

    // AssertThrow(false, ExcMessage(" stopping after parameters"));

      const MyFlux<dim> flux;
      BurgersProblem<dim> burgers_problem(&parameters, flux);
      burgers_problem.run();
    }
    
  }

  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }

  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
