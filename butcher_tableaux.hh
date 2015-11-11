
// **********************************************************************************
// ---                  ---
// --- Butcher tableaux ---
// ---                  ---
// **********************************************************************************
// jcr global var ?

enum temporal_disc {BE, IM, SDIRK22, SDIRK32, SDIRK33, FE, CN, ERK22, ERK44, ERK33, not_implemented};

void init_butcher_tableau(const std::string                 &method_name,
                          unsigned short                    &n_stages,
                          std::vector<std::vector<double> > &a,
                          std::vector<double>               &b,
                          std::vector<double>               &c,
                          bool                              &is_explicit) {
                          
  std::cout << "time discretization method name is : " << method_name << std::endl;

  temporal_disc method;
  std::string erk22("ERK22");
  std::string erk33("ERK33");
  std::string erk44("ERK44");
  std::string cn   ("CN");
  std::string fe   ("Explicit-Euler");
  std::string be   ("Implicit-Euler");
  std::string im   ("Implicit-Midpoint");
  std::string sdirk22("SDIRK22");
  std::string sdirk32("SDIRK32");
  std::string sdirk33("SDIRK33");
  
  method = not_implemented;
  if (erk22.compare(method_name) == 0)
    method = ERK22;
  if (erk33.compare(method_name) == 0)
    method = ERK33;
  if (erk44.compare(method_name) == 0)
    method = ERK44;
  if (fe.compare(method_name) == 0)
    method = FE;
  if (be.compare(method_name) == 0)
    method = BE;
  if (im.compare(method_name) == 0)
    method = IM;
  if (cn.compare(method_name) == 0)
    method = CN;
  if (sdirk22.compare(method_name) == 0)
    method = SDIRK22;
  if (sdirk32.compare(method_name) == 0)
    method = SDIRK32;
  if (sdirk33.compare(method_name) == 0)
    method = SDIRK33;
  
  std::cout << "time discretization method ENUM is : " << method << std::endl;

  if (method == BE) { // verified
    n_stages = 1;
    a = std::vector<std::vector<double> > (n_stages, std::vector<double> (n_stages, 0.0));
    a[0][0] = 1.0;
    b = std::vector<double> (n_stages, 0.0);
    b[0] = 1.0;
    c = std::vector<double> (n_stages, 0.0);
    c[0] = 1.0;
  } else if (method == IM) {
    n_stages = 1;
    a = std::vector<std::vector<double> > (n_stages, std::vector<double> (n_stages, 0.0));
    a[0][0] = 0.5;
    b = std::vector<double> (n_stages, 0.0);
    b[0] = 1.0;
    c = std::vector<double> (n_stages, 0.0);
    c[0] = 0.5;
  } else if (method == SDIRK22) { // verified
    double gamma = 1.0 - 1.0 / sqrt(2.0);
    n_stages = 2;
    a = std::vector<std::vector<double> > (n_stages, std::vector<double> (n_stages, 0.0));
    a[0][0] = gamma;
    a[1][0] = 1.0 - gamma;
    a[1][1] = gamma;
    b = std::vector<double> (n_stages, 0.0);
    b[0] = 1.0 - gamma;
    b[1] = gamma;
    c = std::vector<double> (n_stages, 0.0);
    c[0] = gamma;
    c[1] = 1.0;
  } else if (method == SDIRK32) { // verified
    double gamma = (3.0 + sqrt(3.0)) / 6.0;
    n_stages = 2;
    a = std::vector<std::vector<double> > (n_stages, std::vector<double> (n_stages, 0.0));
    a[0][0] = gamma;
    a[1][0] = 1.0 - 2.0 * gamma;
    a[1][1] = gamma;
    b = std::vector<double> (n_stages, 0.0);
    b[0] = 0.5;
    b[1] = 0.5;
    c = std::vector<double> (n_stages, 0.0);
    c[0] = gamma;
    c[1] = 1.0 - gamma;
  } else if (method == SDIRK33) { // verified
    double gamma = 0.435866521508459;
    n_stages = 3;
    a = std::vector<std::vector<double> > (n_stages, std::vector<double> (n_stages, 0.0));
    a[0][0] = gamma;
    a[1][0] = (1.0 - gamma) / 2.0;
    a[1][1] = gamma;
    a[2][0] = (-6.0 * gamma * gamma + 16.0 * gamma - 1.0) / 4.0;
    a[2][1] = (6.0 * gamma * gamma - 20.0 * gamma + 5.0) / 4.0;
    a[2][2] = gamma;
    b = std::vector<double> (n_stages, 0.0);
    b[0] = (-6.0 * gamma * gamma + 16.0 * gamma - 1.0) / 4.0;
    b[1] = (6.0 * gamma * gamma - 20.0 * gamma + 5.0) / 4.0;
    b[2] = gamma;
    c = std::vector<double> (n_stages, 0.0);
    c[0] = gamma;
    c[1] = (1.0 + gamma) / 2.0;
    c[2] = 1.0;
  } else if (method == FE) { // verified (but remember about stability issues, dt/dx^2 < C for HC for instance)
    n_stages = 1;
    a = std::vector<std::vector<double> > (n_stages, std::vector<double> (n_stages, 0.0));
    a[0][0] = 0.0;
    b = std::vector<double> (n_stages, 0.0);
    b[0] = 1.0;
    c = std::vector<double> (n_stages, 0.0);
    c[0] = 0.0;
  } else if (method == CN) { // verified
    n_stages = 2;
    a = std::vector<std::vector<double> > (n_stages, std::vector<double> (n_stages, 0.0));
    a[0][0] = 0.0;
    a[1][0] = 0.5;
    a[1][1] = 0.5;
    b = std::vector<double> (n_stages, 0.0);
    b[0] = 0.5;
    b[1] = 0.5;
    c = std::vector<double> (n_stages, 0.0);
    c[0] = 0.0;
    c[1] = 1.0;
  } else if (method == ERK22) {
    n_stages = 2;
    a = std::vector<std::vector<double> > (n_stages, std::vector<double> (n_stages, 0.0));
    a[0][0] = 0.0;
    a[1][0] = 0.5;
    a[1][1] = 0.0;
    b = std::vector<double> (n_stages, 0.0);
    b[0] = 0.0;
    b[1] = 1.0;
    c = std::vector<double> (n_stages, 0.0);
    c[0] = 0.0;
    c[1] = 0.5;
  } else if (method == ERK44) {
    n_stages = 4;
    a = std::vector<std::vector<double> > (n_stages, std::vector<double> (n_stages, 0.0));
    a[0][0] = 0.0;
    a[1][0] = 0.5;
    a[1][1] = 0.0;
    a[2][0] = 0.0;
    a[2][1] = 0.5;
    a[2][2] = 0.0;
    a[3][0] = 0.0;
    a[3][1] = 0.0;
    a[3][2] = 1.0;
    a[3][3] = 0.0;
    b = std::vector<double> (n_stages, 0.0);
    b[0] = 1.0 / 6.0;
    b[1] = 1.0 / 3.0;
    b[2] = 1.0 / 3.0;
    b[3] = 1.0 / 6.0;
    c = std::vector<double> (n_stages, 0.0);
    c[0] = 0.0;
    c[1] = 0.5;
    c[2] = 0.5;
    c[3] = 1.0;
  } else if (method == ERK33) {
    n_stages = 3;
    a = std::vector<std::vector<double> > (n_stages, std::vector<double> (n_stages, 0.0));
    a[0][0] = 0.0;
    a[1][0] = 1.0;
    a[1][1] = 0.0;
    a[2][0] = 0.25;
    a[2][1] = 0.25;
    a[2][2] = 0.0;
    b = std::vector<double> (n_stages, 0.0);
    b[0] = 1.0 / 6.0;
    b[1] = 1.0 / 6.0;
    b[2] = 2.0 / 3.0;
    c = std::vector<double> (n_stages, 0.0);
    c[0] = 0.0;
    c[1] = 1.0;
    c[2] = 0.5;
  } else if (method == not_implemented) {
    AssertThrow(false, ExcNotImplemented());
  } else {
    AssertThrow(false, ExcNotImplemented());
  }

  // determine whether the method is fully explicit or not
  is_explicit=true;
  for(int i=0; i<n_stages; ++i)
    for(int j=i; j<n_stages; ++j)
      if( std::abs(a[i][j]) > 1e-6 )
        is_explicit = false;
       
}

