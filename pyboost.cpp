#include <boost/python.hpp>
#include <cstdlib> // setenv
#include <iostream>

int main()
{
  // Allow Python to load modules from the current directory.
  setenv("PYTHONPATH", ".", 1);
  // Initialize Python.
  Py_Initialize();

  namespace python = boost::python;
  try
    {
      // >>> import shout_filter
      // python::object my_python_class_module = python::import("shout_filter");
      python::object my_python_class_module = python::import("predictor");

        //python::object myFunc = my_python_class_module.attr("filterFunc");
      python::object predictFunc = my_python_class_module.attr("predict");
      // python::object result = myFunc("hola mundo");
      python::object result = predictFunc(-147.536,-75.7903, -30.7109, -136.171, 9.0898, 123.918, 1.937, 1.1107, 0.7442, 1.48799, 0.174372, 1, 1);

      std::string resultstr = python::extract<std::string>(result);

      python::object result2 = predictFunc(-167.113, 1000.0, -80.0162, 1000.0, -12.0881, 1000.0, 3.5589, -1.0, 0.7390, -1.0, -1.0, 1, 0);

      std::string resultstr2 = python::extract<std::string>(result2);
      std::cout << resultstr << std::endl;
      std::cout << resultstr2 << std::endl;
    }
  catch (const python::error_already_set&)
    {
      PyErr_Print();
      return 1;
    }

  return 0;
  // Do not call Py_Finalize() with Boost.Python.
}

//Compile: g++ -o program pyboost.cpp -L/usr/lib -lboost_python -I/usr/include/python2.7 -Wall -lpython2.7
