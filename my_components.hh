#ifndef MY_COMPONENTS_HH
#define MY_COMPONENTS_HH

using namespace dealii;

struct MyComponents
{
  ///////////////////////////////////////////////////////////////////////////////////////////
  //// EQUATIONS SYSTEM
  ///////////////////////////////////////////////////////////////////////////////////////////  

  static std::vector<std::string> exact_component_names ()
  {
    std::vector<std::string> names;
    names.push_back ("Burgers (exact)");

    return names;
  }
  
  static const unsigned int n_components = 1;

  static const unsigned int burgers_component = 0;
  
  static std::vector<std::string> component_names ()
  {
    std::vector<std::string> names;
    names.push_back ("Burgers");

    return names;
  }
 
  static std::vector<DataComponentInterpretation::DataComponentInterpretation> component_interpretation ()
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation;
    data_component_interpretation.push_back (DataComponentInterpretation::component_is_scalar);

    return data_component_interpretation;
  }
 
};

#endif // MY_COMPONENTS_HH
