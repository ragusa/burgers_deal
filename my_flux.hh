#ifndef MY_FLUX_HH
#define MY_FLUX_HH

//#include "parameters.hh"

#include "my_components.hh"

using namespace dealii;

// --- template class Flux<dim> ---

template<int dim>
class MyFlux
{
public:

  MyFlux();   // constructor
  ~MyFlux();  // destructor

  // methods to compute the flux
  Tensor<1,dim> flx_f(const double) const;
  
  // methods to compute its derivative wrt to the unknown u
  Tensor<1,dim> flx_f_prime(const double) const;

  // methods to compute the entropy
  double entropy(const double) const;
  // methods to compute the entropy
  void entropy_values(const std::vector<double>, std::vector<double>) const;
	
  // methods to compute its derivative wrt to u
  double entropy_prime(const double) const;

  // methods to compute the propagation speed
  double propagation_speed(const Tensor<1,dim>&) const;
};

// ---       ---
// --- constructor ---
// ---       ---
template<int dim>
MyFlux<dim>::MyFlux() { }

// ---       ---
// --- destructor ---
// ---       ---
template<int dim>
MyFlux<dim>::~MyFlux() { }

// ---       ---
// --- flux_f ---
// ---       ---
template<int dim>
Tensor<1,dim> MyFlux<dim>::flx_f(const double u) const
{
	Tensor<1,dim> return_value;

	const double aux = 0.5*std::pow(u,2);
	for(unsigned int d = 0; d < dim; ++d)
		return_value[d] = aux;  

	return return_value;
}

// ---       ---
// --- flux_f_prime ---
// ---       ---
template<int dim>
Tensor<1,dim> MyFlux<dim>::flx_f_prime(const double u) const
{
	Tensor<1,dim> return_value;

	for(unsigned int d = 0; d < dim; ++d)
		return_value[d] = u;                  

	return return_value;
}

// ---       ---
// --- entropy ---
// ---       ---
template<int dim>
double MyFlux<dim>::entropy(const double u) const
{
	return 0.5*std::pow(u,2);               
}
// ---       ---
// --- entropy values list---
// ---       ---
template<int dim>
void MyFlux<dim>::entropy_values(const std::vector<double> u,
                                       std::vector<double> values) const
{
  const unsigned int n_points= u.size();
  Assert( values.size() == n_points, ExcDimensionMismatch(values.size(),n_points) );
  for (unsigned int i=0; i< n_points; ++i)
    values[i]=entropy(u[i]) ;
}

// ---       ---
// --- entropy_prime ---
// ---       ---
template<int dim>
double MyFlux<dim>::entropy_prime(const double u) const
{
	return u;                               
}

// ---       ---
// --- propagation_speed ---
// ---       ---
template<int dim>
double MyFlux<dim>::propagation_speed(const Tensor<1,dim> &flux_f_prime) const
{
	return flux_f_prime.norm();
}


#endif // MY_FLUX_HH
