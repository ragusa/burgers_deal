// **********************************************************************************
// ---                ---
// --- INITAL DATA    ---
// ---                ---
// ---                                 ---
// --- template class InitialData<dim> ---
// ---                                 ---
// **********************************************************************************

template <int dim>
class InitialData : public Function<dim>
{
public:
  
  InitialData (int n_components, int pbID);      
    
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0 ) const;
    
  virtual void value_list (const std::vector<Point<dim> > &points,
                           std::vector<double>            &values,
                           const unsigned int              component = 0 ) const;
private:
  int    pbID;

  double math_formula(const Point<dim>&,
                      const unsigned int component = 0) const ;
};


// ---             ---
// --- constructor ---
// ---             ---
template<int dim>
InitialData<dim>::InitialData(int n_components_, int pbID_):
  Function<dim>(n_components_),
  pbID(pbID_)
{
  Assert( pbID <= 1 , ExcIndexRange( pbID, 0 , 2 ) );
}

// ---            ---
// --- value_list ---
// ---            ---
template <int dim>
void InitialData<dim>::value_list (const std::vector<Point<dim> > &points,
                                         std::vector<double>      &values,
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
double InitialData<dim>::value (const Point<dim>   &point,
                                const unsigned int  component ) const 
                                                                  // the last const keeps the method 
                                                                  // from altering variables in the class
{
  Assert( component == 0 , ExcIndexRange( component , 0 , 1 ) ); 

  double returned_value = -999.;
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
double InitialData<dim>::math_formula(const Point<dim>   &point,
                                      const unsigned int  comp ) const
{
  Assert (comp < this->n_components, ExcIndexRange( comp, 0, this->n_components) );

  double value = 1.0;

// std::cout << "pbID = " << pbID << std::endl;

  switch(pbID)
    {
    case 1: // sinus 1D-like problem in multi-D
      {
        double r = point[0];
        value *= std::sin(2.0*numbers::PI*r);
        break;
      }
    case 2: // Riemann problem # xxx
      {
        Assert( dim==2 , ExcNotImplemented() );
        double x = point[0];
        double y = point[1];
        if     (x <= 0.5 && y <= 0.5)
          value =  0.5;
        else if(x <= 0.5 && y >  0.5)
          value = -0.2;
        else if(x >  0.5 && y <= 0.5)
          value =  0.8;
        else if(x >  0.5 && y >  0.5)
          value = -1.0;
        break;
      }
    case 3: // sinus in multi-D
      {
        for(unsigned int d=0; d<dim; ++d){
          double r = point[d];
          value *= std::sin(2.0*numbers::PI*r);
        }
        break;
      }
    default:
      {
        Assert( false , ExcNotImplemented() );
      }
    }
  
  return value;
  
}

