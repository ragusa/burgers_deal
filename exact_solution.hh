// **********************************************************************************
// ---                  ---
// --- EXACT SOLUTION   ---
// ---                  ---
// ---                                   ---
// --- template class ExactSolution<dim> ---
// ---                                   ---
// **********************************************************************************

template <int dim>
class ExactSolution : public Function<dim>
{
public:
  
  ExactSolution (int n_components, double length, double amplitude, 
                 int pbID, double time_a0);      
    
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0 ) const;
    
  virtual void value_list (const std::vector<Point<dim> > &points,
                           std::vector<double>            &values,
                           const unsigned int              component = 0 ) const;
  
  static double *p_time;
  static void synchronize (double *p_t) { p_time = p_t; }
  
private:
  bool  is_initial_time  ;
  double length ;
  double amplitude;
  int    pbID;
  double time_a0;


  double math_formula(const Point<dim>&,
                      const unsigned int component = 0) const ;
};


// ---       ---
// --- constructor ---
// ---       ---

template<int dim>
ExactSolution<dim>::ExactSolution(int n_components, double length, double amplitude, 
                                  int pbID, double time_a0):
  Function<dim>(n_components),
  is_initial_time(true),  
  length(length),
  amplitude(amplitude),
  pbID(pbID),
  time_a0(time_a0)
{
  Assert( pbID <= 1 , ExcIndexRange( pbID, 0 , 2 ) );
}

// jcr ?? *
template <int dim>
double *ExactSolution<dim>::p_time = NULL;

// ---            ---
// --- value_list ---
// ---            ---
template <int dim>
void ExactSolution<dim>::value_list (const std::vector<Point<dim> > &points,
                                     std::vector<double>            &values,
                                     const unsigned int              component) const
{  
  const unsigned int n_points = points.size();

  Assert (values.size() == n_points, ExcDimensionMismatch (values.size(), n_points));
    
  for (unsigned int i=0; i<n_points; ++i)
    values[i] = value(points[i], component);
}

// ---       ---
// --- value ---
// ---       ---
template <int dim>
double ExactSolution<dim>::value (const Point<dim>   &point,
                                  const unsigned int  component ) const 
                                                                  // the last const keeps the method 
                                                                  // from altering variables in the class
{
  Assert( component == 0 , ExcIndexRange( component , 0 , 1 ) ); 

  // retrieve time
  double t = *p_time;

  double returned_value = -1.;
  if (component == MyComponents::burgers_component) {
      returned_value = math_formula(point,component);
  }
  else {
    Assert (false, ExcNotImplemented());
  } // end if component
  return returned_value;
}

// ---              ---
// --- math_formula ---
// ---              ---
template<int dim>
double ExactSolution<dim>::math_formula(const Point<dim>   &point,
                                        const unsigned int  comp ) const
{
  Assert (comp < this->n_components, ExcIndexRange( comp, 0, this->n_components) );

  switch(dim)
    {
    case 2:
      break;
    default:
      Assert( false , ExcNotImplemented() );
    }

  const double x = point[0];
  const double y = point[1];

  //const double t = Function<dim>::get_time();
  double t = *p_time;
    
  double return_value;

  //  std::cout << "here math 1 " << std::endl;
  if(is_initial_time)
    {
      //        is_initial_time = false ; // set it to false for the rest of the computation
        
      if     (x <= 0.5 && y <= 0.5)
          return_value =  0.5;
      else if(x <= 0.5 && y >  0.5)
          return_value = -0.2;
      else if(x >  0.5 && y <= 0.5)
          return_value =  0.8;
      else //if(x >  0.5 && y >  0.5)
          return_value = -1.0;
      
    }
  else
    {
      if(x <= 0.5-0.6*t)
        {
        if(y >= 0.5+0.15*t)
          return_value = -0.2;
        else
        return_value =  0.5;
        }
      else if(x > 0.5-0.6*t && x <= 0.5-0.25*t)
        {
        if(y >= -8.0*x/7.0 + 15.0/14.0 - 15.0*t/28.0)
          return_value = -1.0;
        else
          return_value =  0.5;
        }
      else if(x > 0.5-0.25*t && x <= 0.5+0.5*t)
        {
        if(y >= x/6.0 + 5.0/12.0 - 5.0*t/24.0)
          return_value = -1.0;
        else
          return_value =  0.5;
        }
      else if(x > 0.5+0.5*t && x <= 0.5+0.8*t)
        {
        if(y >= x - (5.0/(18.0*t))*std::pow((x+t-0.5),2))
          return_value = -1.0;
        else
          return_value = (2.0*x-1.0)/(2.0*t);
        }
      else //if(x > 0.5+0.8*t)
        {
        if(y >= 0.5-0.1*t)
          return_value = -1.0;
        else
          return_value =  0.8;
        }
    }

  return return_value;
}

