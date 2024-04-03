#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#if defined(ABQ_LINUX)
#include <aba_for_c.h>
#endif // ABQ_LINUX

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

namespace {
  std::ofstream _logFile;
  std::string _configFileName;
  bool _isInitialized = false;
  PyObject *_pyDriver;
  PyObject *_evaluateMethod;
  PyThreadState * _threadState;

  //
  // Return a string to the ini config file
  // Currently uses PYVUMAT_CONF_FILE environment variable
  //
  std::string
  getConfFileName(std::ofstream & logFile)
  {
    std::string returnValue;
      
    char const* tmpChar = getenv("PYVUMAT_CONF_FILE");
    if ( tmpChar == NULL ) {
      
      //
      // Write warning that configure file was not found
      //
      std::stringstream message;
      message << "Error: Could not find the INI conf file."
	      << "Make sure PYVUMAT_CONF_FILE environment variable "
	      << " is set." << std::endl;
      logFile << message.str();
    }

    //
    // Set the conf file name
    //
    returnValue = tmpChar;
    return returnValue;
  }   

}

//
// VUMAT function called by FEM codes.
//
extern "C" void vumat_(
// Read only
  const int *blockInfo, const int &ndir, const int &nshr, const int &nstatev, 
  const int &nfieldv, const int &nprops, const int &lanneal, 
  const double &stepTime, const double &totalTime, const double &dt, 
  const char *cmname, const double *coordMp, const double *charLength,
  const double *props, const double *density, const double *strainInc, 
  const double *relSpinInc, const double *tempOld, const double *stretchOld, 
  const double *defgradOld, const double *fieldOld, const double *stressOld, 
  const double *stateOld, const double *enerInternOld, 
  const double *enerInelasOld, const double *tempNew, const double *stretchNew,
  const double *defgradNew, const double *fieldNew,
// Write only
  double *stressNew, double *stateNew, double *enerInternNew, 
  double *enerInelasNew )
{

  //
  // Perform initialization on the first step
  //
  if (!_isInitialized) {

    //
    // Create logging file
    //
    _logFile.open("vumat_c++.log",std::ofstream::app);
    _logFile.precision(18);

    //
    // Get the INI configuration file
    // 
    _configFileName = getConfFileName(_logFile);
    
    //
    // Initializes the Python interpreter
    //
    Py_Initialize();      
    _import_array();
    
    //
    // Load the module object
    //
    PyObject * pModuleName = PyUnicode_FromString("pyvumat.driver");
    
    PyObject * pModule = PyImport_Import(pModuleName);
    if (pModule == NULL) {
      PyErr_Print();
      const std::string message("Failed to import pyvumat.driver module");
      throw std::runtime_error(message);
    }
    Py_XDECREF(pModuleName);
    
    //
    // Load the module's dict
    //
    PyObject * pDict = PyModule_GetDict(pModule);
    if (pDict == NULL) {
      PyErr_Print();
      const std::string message("Failed to get dict for the "
				"pyvumat.driver module");
      throw std::runtime_error(message);
    }
    Py_XDECREF(pModule);
    
    //
    // Build the name of a callable class
    //
    PyObject * pClass = PyDict_GetItemString(pDict, "Driver");
    if (pClass == NULL || !PyCallable_Check(pClass)) {
      PyErr_Print();
      const std::string message("Failed to get the pyvumat.driver class");
      throw std::runtime_error(message);
    }
    Py_XDECREF(pDict);
    
    //
    // Creates an instance of the class
    //
    PyObject * pClassArg;
    pClassArg = Py_BuildValue("(s)",
			      _configFileName.c_str());
    
    _pyDriver = PyObject_CallObject(pClass, pClassArg);
    
    Py_XDECREF(pClass);
    Py_XDECREF(pClassArg);
    
    if (_pyDriver == NULL) {
      PyErr_Print();
      const std::string message("Failed to instantiate pyvumat driver object");
      throw std::runtime_error(message);
    }

    //
    // Create the evaluate() method object
    //
    _evaluateMethod = PyObject_GetAttrString(_pyDriver,"evaluate");

    if (_evaluateMethod == NULL) {
      PyErr_Print();
      const std::string message("Failed to get evaluate() method from driver object");
      throw std::runtime_error(message);
    }

    _threadState = PyEval_SaveThread();    
    _isInitialized = true;
    
  } /* End initialization */
  
  //
  // Get lock to python interpreter
  //
  PyGILState_STATE gState;
  gState =  PyGILState_Ensure();

  // 
  int nblock = blockInfo[0];

  // Start timing
  time_t _stm =time(NULL );  
  struct tm * startTime = localtime ( &_stm );

  //
  // Create an empty tuple to pass as the argument
  //
  PyObject *emptyArg = Py_BuildValue("()");
  
  //
  // Create the keyword dict containing all inputs
  //
  PyObject *keywords = PyDict_New();

  //
  // Add scalar arguments
  //
  PyObject *lanneal_py = Py_BuildValue("i",lanneal);
  PyDict_SetItemString(keywords,"lanneal",lanneal_py);

  PyObject *stepTime_py = Py_BuildValue("d",stepTime);
  PyDict_SetItemString(keywords,"stepTime",stepTime_py);

  PyObject *totalTime_py = Py_BuildValue("d",totalTime);
  PyDict_SetItemString(keywords,"totalTime",totalTime_py);

  PyObject *dt_py = Py_BuildValue("d",dt);
  PyDict_SetItemString(keywords,"dt",dt_py);

  //
  // Convert the arrays to python arrays
  //
  // NOTE: All arrays are in Fortran (column-major) order. It is easier to
  // keep them that way for now. Must transpose in python if necessary!
  //
  npy_intp dim2D[2];
  npy_intp dim1D[1];

  //  CoordMp
  // Fixme: Is dimension 3, ndir, or something else?
  dim2D[0] = ndir;
  dim2D[1] = nblock;
  PyArrayObject * coordMpArray=
    (PyArrayObject *)PyArray_SimpleNewFromData(2,dim2D,
					       NPY_DOUBLE,
					       (void *)coordMp);  
  PyDict_SetItemString(keywords,"coordMp",(PyObject*)coordMpArray);

  //  CharLength
  dim1D[0] = nblock;
  PyArrayObject * charLengthArray=
    (PyArrayObject *)PyArray_SimpleNewFromData(1,dim1D,
					       NPY_DOUBLE,
					       (void *)charLength);  
  PyDict_SetItemString(keywords,"charLength",(PyObject*)charLengthArray);

  //  Props
  dim1D[0] = nprops;
  PyArrayObject * propsArray=
    (PyArrayObject *)PyArray_SimpleNewFromData(1,dim1D,
					       NPY_DOUBLE,
					       (void *)props);  
  PyDict_SetItemString(keywords,"props",(PyObject*)propsArray);

  //  Density
  dim1D[0] = nblock;
  PyArrayObject * densityArray=
    (PyArrayObject *)PyArray_SimpleNewFromData(1,dim1D,
					       NPY_DOUBLE,
					       (void *)density);
  PyDict_SetItemString(keywords,"density",(PyObject*)densityArray);  

  // StrainInc
  dim2D[0] = ndir+nshr;
  dim2D[1] = nblock;
  PyArrayObject * strainIncArray=
  (PyArrayObject *)PyArray_SimpleNewFromData(2,dim2D,
					     NPY_DOUBLE,
					     (void *)strainInc);  
  PyDict_SetItemString(keywords,"strainInc",(PyObject*)strainIncArray);

  // RelSpinInc
  dim2D[0] = nshr;
  dim2D[1] = nblock;
  PyArrayObject * relSpinIncArray=
  (PyArrayObject *)PyArray_SimpleNewFromData(2,dim2D,
					     NPY_DOUBLE,
					     (void *)relSpinInc);  
  PyDict_SetItemString(keywords,"relSpinInc",(PyObject*)relSpinIncArray);

  //  TempOld
  dim1D[0] = nblock;
  PyArrayObject * tempOldArray=
    (PyArrayObject *)PyArray_SimpleNewFromData(1,dim1D,
					       NPY_DOUBLE,
					       (void *)tempOld);  
  PyDict_SetItemString(keywords,"tempOld",(PyObject*)tempOldArray);

  //  StretchOld
  dim2D[0] = ndir+nshr;
  dim2D[1] = nblock;
  PyArrayObject * stretchOldArray=
  (PyArrayObject *)PyArray_SimpleNewFromData(2,dim2D,
					     NPY_DOUBLE,
					     (void *)stretchOld);  
  PyDict_SetItemString(keywords,"stretchOld",(PyObject*)stretchOldArray);

  //  DefGradOld
  dim2D[0] = ndir+nshr+nshr;
  dim2D[1] = nblock;
  PyArrayObject * defgradOldArray=
    (PyArrayObject *)PyArray_SimpleNewFromData(2,dim2D,
					       NPY_DOUBLE,
					       (void *)defgradOld);
  PyDict_SetItemString(keywords,"defgradOld",(PyObject*)defgradOldArray);

  //  FieldOld
  dim2D[0] = nfieldv;
  dim2D[1] = nblock;
  PyArrayObject * fieldOldArray=
    (PyArrayObject *)PyArray_SimpleNewFromData(2,dim2D,
					       NPY_DOUBLE,
					       (void *)fieldOld);
  PyDict_SetItemString(keywords,"fieldOld",(PyObject*)fieldOldArray);

  //  StressOld
  dim2D[0] = ndir+nshr;
  dim2D[1] = nblock;
  PyArrayObject * stressOldArray=
    (PyArrayObject *)PyArray_SimpleNewFromData(2,dim2D,
					       NPY_DOUBLE,
					       (void *)stressOld);
  PyDict_SetItemString(keywords,"stressOld",(PyObject*)stressOldArray);

  //  StateOld
  dim2D[0] = nstatev;
  dim2D[1] = nblock;
  PyArrayObject * stateOldArray=
  (PyArrayObject *)PyArray_SimpleNewFromData(2,dim2D,
					     NPY_DOUBLE,
					     (void *)stateOld);
  PyDict_SetItemString(keywords,"stateOld",(PyObject*)stateOldArray);

  //  enerInternOld
  dim1D[0] = nblock;
  PyArrayObject * enerInternOldArray=
    (PyArrayObject *)PyArray_SimpleNewFromData(1,dim1D,
					       NPY_DOUBLE,
					       (void *)enerInternOld);  
  PyDict_SetItemString(keywords,"enerInternOld",(PyObject*)enerInternOldArray);

  //  enerInelasOld
  dim1D[0] = nblock;
  PyArrayObject * enerInelasOldArray=
    (PyArrayObject *)PyArray_SimpleNewFromData(1,dim1D,
					       NPY_DOUBLE,
					       (void *)enerInelasOld);  
  PyDict_SetItemString(keywords,"enerInelasOld",(PyObject*)enerInelasOldArray);

  //  tempNew
  dim1D[0] = nblock;
  PyArrayObject * tempNewArray=
    (PyArrayObject *)PyArray_SimpleNewFromData(1,dim1D,
					       NPY_DOUBLE,
					       (void *)tempNew);  
  PyDict_SetItemString(keywords,"tempNew",(PyObject*)tempNewArray);

  //  StretchNew
  dim2D[0] = ndir+nshr;
  dim2D[1] = nblock;
  PyArrayObject * stretchNewArray=
  (PyArrayObject *)PyArray_SimpleNewFromData(2,dim2D,
					     NPY_DOUBLE,
					     (void *)stretchNew);  
  PyDict_SetItemString(keywords,"stretchNew",(PyObject*)stretchNewArray);
  
  //  DefGradNew
  dim2D[0] = ndir+nshr+nshr;
  dim2D[1] = nblock;
  PyArrayObject * defgradNewArray=
    (PyArrayObject *)PyArray_SimpleNewFromData(2,dim2D,
					       NPY_DOUBLE,
					       (void *)defgradNew);  
  PyDict_SetItemString(keywords,"defgradNew",(PyObject*)defgradNewArray);

  //  FieldNew
  dim2D[0] = nfieldv;
  dim2D[1] = nblock;
  PyArrayObject * fieldNewArray=
    (PyArrayObject *)PyArray_SimpleNewFromData(2,dim2D,
					       NPY_DOUBLE,
					       (void *)fieldNew);
  PyDict_SetItemString(keywords,"fieldNew",(PyObject*)fieldNewArray);

  //
  // Evaluate the pyML model for the inputs
  //
  PyObject * retVal = PyObject_Call(_evaluateMethod,emptyArg,keywords);

  if (retVal == NULL) {
    PyErr_Print();
    const std::string message("Failed in python call to evaluate)");
    throw std::runtime_error(message);
  }

  //
  // Extract the results
  //
  PyArrayObject *stressNewArray;
  PyArrayObject *stateNewArray;
  PyArrayObject *enerInternNewArray;  
  PyArrayObject *enerInelasNewArray;
  int parseSuccess = PyArg_ParseTuple(retVal,"OOOO",
				      &stressNewArray,
				      &stateNewArray,
				      &enerInternNewArray,
				      &enerInelasNewArray);
  if (!parseSuccess) {
    PyErr_Print();
    const std::string message("Failed to parse results from python "
			      "call to evaluate()");
    throw std::runtime_error(message);
  }

  //
  // Store the results to return from VUMAT
  //

  // StressNew
  // Make sure the returned array is 2D (ndir+nshr x nblock);
  unsigned int numOutDims = PyArray_NDIM(stressNewArray);
  npy_intp * outDims = PyArray_DIMS(stressNewArray);
  assert(numOutDims == 2);  
  assert(outDims[0] == ndir+nshr);  
  assert(outDims[1] == nblock);  
  double * tmpStressNew = (double *)PyArray_DATA(stressNewArray);
  std::copy(tmpStressNew,
	    tmpStressNew+(nblock*(ndir+nshr)),
	    stressNew);
  
  // StateNew
  // Make sure the returned array is 2D (nstatev x nblock);
  numOutDims = PyArray_NDIM(stateNewArray);
  outDims = PyArray_DIMS(stateNewArray);
  assert(numOutDims == 2);  
  assert(outDims[0] == nstatev);  
  assert(outDims[1] == nblock);  
  double * tmpStateNew = (double *)PyArray_DATA(stateNewArray);
  std::copy(tmpStateNew,
	    tmpStateNew+(nblock*nstatev),
	    stateNew);

  // EnerInternNew
  // Make sure the returned array is 1D (nblock);
  numOutDims = PyArray_NDIM(enerInternNewArray);
  outDims = PyArray_DIMS(enerInternNewArray);
  assert(numOutDims == 1);  
  assert(outDims[0] == nblock);  
  double * tmpEnerInternNew = (double *)PyArray_DATA(enerInternNewArray);
  std::copy(tmpEnerInternNew,
	    tmpEnerInternNew+nblock,
	    enerInternNew);

  // EnerInelasNew
  // Make sure the returned array is 1D (nblock);
  numOutDims = PyArray_NDIM(enerInelasNewArray);
  outDims = PyArray_DIMS(enerInelasNewArray);
  assert(numOutDims == 1);  
  assert(outDims[0] == nblock);  
  double * tmpEnerInelasNew = (double *)PyArray_DATA(enerInelasNewArray);
  std::copy(tmpEnerInelasNew,
	    tmpEnerInelasNew+nblock,
	    enerInelasNew);

  Py_XDECREF(lanneal_py);  
  Py_XDECREF(stepTime_py);    
  Py_XDECREF(totalTime_py);  
  Py_XDECREF(dt_py);  
  Py_XDECREF(coordMpArray);  
  Py_XDECREF(charLengthArray);  
  Py_XDECREF(propsArray);  
  Py_XDECREF(densityArray);  
  Py_XDECREF(strainIncArray);  
  Py_XDECREF(relSpinIncArray);  
  Py_XDECREF(tempOldArray);  
  Py_XDECREF(stretchOldArray);  
  Py_XDECREF(defgradOldArray);
  Py_XDECREF(fieldOldArray);  
  Py_XDECREF(stressOldArray);  
  Py_XDECREF(stateOldArray);
  Py_XDECREF(enerInternOldArray);  
  Py_XDECREF(enerInelasOldArray);  
  Py_XDECREF(tempNewArray);  
  Py_XDECREF(stretchNewArray);
  Py_XDECREF(defgradNewArray);
  Py_XDECREF(fieldNewArray);  

  Py_XDECREF(retVal);

  Py_XDECREF(emptyArg);  
  Py_XDECREF(keywords);  

  //
  // Release GIL and return
  //
  PyGILState_Release(gState);

  return;
}

extern "C" void vumat(
// Read only
  const int *blockInfo, const int &ndir, const int &nshr, const int &nstatev, 
  const int &nfieldv, const int &nprops, const int &lanneal, 
  const double &stepTime, const double &totalTime, const double &dt, 
  const char *cmname, const double *coordMp, const double *charLength,
  const double *props, const double *density, const double *strainInc, 
  const double *relSpinInc, const double *tempOld, const double *stretchOld, 
  const double *defgradOld, const double *fieldOld, const double *stressOld, 
  const double *stateOld, const double *enerInternOld, 
  const double *enerInelasOld, const double *tempNew, const double *stretchNew,
  const double *defgradNew, const double *fieldNew,
// Write only
  double *stressNew, double *stateNew, double *enerInternNew, 
  double *enerInelasNew )
{
  vumat_(blockInfo, ndir, nshr, nstatev, 
	 nfieldv, nprops, lanneal, 
	 stepTime, totalTime, dt, 
	 cmname, coordMp, charLength,
	 props, density, strainInc, 
	 relSpinInc, tempOld, stretchOld, 
	 defgradOld, fieldOld, stressOld, 
	 stateOld, enerInternOld, 
	 enerInelasOld, tempNew, stretchNew,
	 defgradNew, fieldNew,
// Write only
	 stressNew, stateNew, enerInternNew, 
	 enerInelasNew );
  return;
}
