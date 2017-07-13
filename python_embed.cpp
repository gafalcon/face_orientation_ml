#include <Python.h>
#include <iostream>
#include <string>


static const char* PLUGIN_NAME = "shout_filter";

std::string CallPlugIn(const std::string &ln){
  PyObject* name = PyString_FromString(PLUGIN_NAME);
  PyObject* pluginModule = PyImport_Import(name);
  Py_DECREF(name);
  if (!pluginModule){
    PyErr_Print();
    return "Error importing module";
  }

  PyObject* filterFunc = PyObject_GetAttrString(pluginModule, "filterFunc");
  Py_DECREF(pluginModule);
  if (!filterFunc){
    PyErr_Print();
    return "Error retrieving 'filterFunc'";
  }

  PyObject* args = Py_BuildValue("(s)", ln.c_str());
  if (!args){
    PyErr_Print();
    Py_DECREF(filterFunc);
    return "Error building args tuple";
  }

  PyObject* resultObj = PyObject_CallObject(filterFunc, args);
  Py_DECREF(filterFunc);
  Py_DECREF(args);
  if (!resultObj){
    PyErr_Print();
    return "Error invoking 'filterFunc'";
  }
  const char* resultStr = PyString_AsString(resultObj);
  if (!resultStr){
    PyErr_Print();
    Py_DECREF(resultObj);
    return "Error converting result to C string";
  }
  std::string result = resultStr;
  Py_DECREF(resultObj);
  return result;
}

int main(int argc, char *argv[])
{
  Py_Initialize();
  PyObject* sysPath = PySys_GetObject((char*)"path");
  PyObject* curDir = PyString_FromString(".");
  PyList_Append(sysPath, curDir);
  Py_DECREF(curDir);

  std::clog << "Type lines of text:" << std::endl;
  std::string input;
  while(true){
    std::getline(std::cin, input);
    if(!std::cin.good())
      break;
    std::cout << CallPlugIn(input) << std::endl;
  }
  Py_Finalize();
  return 0;
}
