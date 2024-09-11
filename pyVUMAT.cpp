#include <stdio.h>
#include <stdlib.h>

#include <fenv.h>

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

// Determine if compilation is in Abaqus
#if defined(ABQ_LINUX) || defined(ABQ_WIN86_64) || defined(ABQ_WIN86_32)
#define IN_ABAQUS
#include <omi_for_c.h>
#else
#define FOR_NAME(lower_case_name,upper_case_name) lower_case_name
#endif // In Abaqus

// Define flag for memory layout of NumPy arrays
#if defined(C_ORDERING)
#define MEM_FLAG NPY_ARRAY_CARRAY_RO
#else
#define MEM_FLAG NPY_ARRAY_FARRAY_RO
#endif

PyObject *
create1dNpArray(npy_intp dim, const double *data) {
  
  // Create numpy array
  npy_intp dims[1] = {dim};
  return PyArray_New(&PyArray_Type,
		     1,dims,
		     NPY_DOUBLE, NULL,
		     (void *)data,
		     0, MEM_FLAG,
		     NULL);  
}

PyObject *
create2dNpArray(npy_intp dim1, npy_intp dim2, const double *data) {

  npy_intp dims[2];
  
#if defined(C_ORDERING)
  dims[0] = dim2;
  dims[1] = dim1;
#else
  dims[0] = dim1;
  dims[1] = dim2;
#endif
    
    // Create numpy array
    return PyArray_New(&PyArray_Type,
		       2,dims,
		       NPY_DOUBLE, NULL,
		       (void *)data,
		       0, MEM_FLAG,
		       NULL);  
}

//
// VUMAT function called by FEA codes.
//
extern "C" void FOR_NAME(vumat,VUMAT)(
// Read only
  const int *blockInfo, const int *ndirPtr, const int *nshrPtr,
  const int *nstatevPtr, const int *nfieldvPtr, const int *npropsPtr,
  const int *lanneal, const double *stepTime, const double *totalTime,
  const double *dt, const char *cmname, const double *coordMp,
  const double *charLength, const double *props, const double *density,
  const double *strainInc, const double *relSpinInc, const double *tempOld,
  const double *stretchOld, const double *defgradOld, const double *fieldOld,
  const double *stressOld, const double *stateOld, const double *enerInternOld,
  const double *enerInelasOld, const double *tempNew, const double *stretchNew,
  const double *defgradNew, const double *fieldNew,
// Write only
  double *stressNew, double *stateNew, double *enerInternNew,
  double *enerInelasNew )
{

  static FILE *_logFile;
  static int _isInitialized = 0;
  static PyObject *_evaluateMethod;
  static PyThreadState *_threadState;

  //
  // Perform initialization on the first step
  //
  if (!_isInitialized) {

    //
    // Create logging file
    //
    _logFile = fopen("pyvumat.log","a");

    //
    // Get the INI configuration file
    //
    char *configFileName = getenv("PYVUMAT_CONF_FILE");

    if ( configFileName == NULL ) {
      //
      // Write warning that configure file was not found
      //
      fprintf(_logFile, "Warning: Could not find the INI conf file.\n");
      fprintf(_logFile, "Make sure PYVUMAT_CONF_FILE environment variable is set.");
    }
    
    //
    // Initializes the Python interpreter
    //
    Py_Initialize();
    
    // _import_array() causes floating point exception on some systems.
    // Temporarily disable floating point exceptions to get through
    // NumPy initialization
    fenv_t orig_feenv;
    feholdexcept(&orig_feenv);
    _import_array();
    fesetenv(&orig_feenv);

    //
    // Load the module object
    //
    PyObject *pModuleName = PyUnicode_FromString("pyvumat.driver");
    
    PyObject *pModule = PyImport_Import(pModuleName);
    if (pModule == NULL) PyErr_Print();
    Py_XDECREF(pModuleName);
    
    //
    // Load the module's dict
    //
    PyObject *pDict = PyModule_GetDict(pModule);
    if (pDict == NULL) PyErr_Print();
    Py_XDECREF(pModule);
    
    //
    // Build the name of a callable class
    //
    PyObject *pClass = PyDict_GetItemString(pDict, "Driver");
    if (pClass == NULL || !PyCallable_Check(pClass)) PyErr_Print();
    Py_XDECREF(pDict);
    
    //
    // Creates an instance of the class
    //
    PyObject *pClassArg = NULL;
    if (configFileName != NULL) {
      pClassArg = Py_BuildValue("(s)",
				configFileName);
    }

    PyObject *pyDriver = PyObject_CallObject(pClass, pClassArg);
    if (pyDriver == NULL) PyErr_Print();

    Py_XDECREF(pClass);
    Py_XDECREF(pClassArg);

    //
    // Create the evaluate() method object
    //
    _evaluateMethod = PyObject_GetAttrString(pyDriver,"evaluate");

    if (_evaluateMethod == NULL) PyErr_Print();

    Py_XDECREF(pyDriver);
    
    _threadState = PyEval_SaveThread();    
    _isInitialized = 1;
    
  } /* End initialization */
  
  //
  // Get lock to python interpreter
  //
  PyGILState_STATE gState;
  gState =  PyGILState_Ensure();

  //
  // Get array sizes
  //
  const int nblock = blockInfo[0];
  const int ndir = ndirPtr[0];
  const int nshr = nshrPtr[0];
  const int nprops = npropsPtr[0];
  const int nfieldv = nfieldvPtr[0];
  const int nstatev = nstatevPtr[0];

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
  PyObject *lanneal_py = Py_BuildValue("i",*lanneal);
  PyDict_SetItemString(keywords,"lanneal",lanneal_py);

  PyObject *stepTime_py = Py_BuildValue("d",*stepTime);
  PyDict_SetItemString(keywords,"stepTime",stepTime_py);

  PyObject *totalTime_py = Py_BuildValue("d",*totalTime);
  PyDict_SetItemString(keywords,"totalTime",totalTime_py);

  PyObject *dt_py = Py_BuildValue("d",*dt);
  PyDict_SetItemString(keywords,"dt",dt_py);

  PyObject *nblock_py = Py_BuildValue("i",nblock);
  PyDict_SetItemString(keywords,"nblock",nblock_py);

  PyObject *ndir_py = Py_BuildValue("i",ndir);
  PyDict_SetItemString(keywords,"ndir",ndir_py);

  PyObject *nshr_py = Py_BuildValue("i",nshr);
  PyDict_SetItemString(keywords,"nshr",nshr_py);

  PyObject *nprops_py = Py_BuildValue("i",nprops);
  PyDict_SetItemString(keywords,"nprops",nprops_py);

  PyObject *nfieldv_py = Py_BuildValue("i",nfieldv);
  PyDict_SetItemString(keywords,"nfieldv",nfieldv_py);

  PyObject *nstatev_py = Py_BuildValue("i",nstatev);
  PyDict_SetItemString(keywords,"nstatev",nstatev_py);
  
  //
  // Convert the arrays to python arrays
  //
  
  //  CoordMp
  // Fixme: Is dimension 3, ndir, or something else?
  PyObject *coordMpArray = create2dNpArray(nblock,ndir,coordMp);
  PyDict_SetItemString(keywords,"coordMp",coordMpArray);

  //  CharLength
  PyObject *charLengthArray = create1dNpArray(nblock,charLength);
  PyDict_SetItemString(keywords,"charLength",charLengthArray);

  //  Props
  PyObject *propsArray = create1dNpArray(nprops,props);
  PyDict_SetItemString(keywords,"props",propsArray);

  //  Density
  PyObject *densityArray = create1dNpArray(nblock,density);
  PyDict_SetItemString(keywords,"density",densityArray);  

  // StrainInc
  PyObject *strainIncArray = create2dNpArray(nblock,ndir+nshr,
					   strainInc);
  PyDict_SetItemString(keywords,"strainInc",strainIncArray);

  // RelSpinInc
  PyObject *relSpinIncArray = create2dNpArray(nblock,nshr,
					      relSpinInc);
  PyDict_SetItemString(keywords,"relSpinInc",relSpinIncArray);

  //  TempOld
  PyObject *tempOldArray = create1dNpArray(nblock,tempOld);
  PyDict_SetItemString(keywords,"tempOld",tempOldArray);

  //  StretchOld
  PyObject *stretchOldArray= create2dNpArray(nblock,ndir+nshr,
					     stretchOld);
  PyDict_SetItemString(keywords,"stretchOld",stretchOldArray);

  //  DefGradOld
  PyObject *defgradOldArray = create2dNpArray(nblock,
					      ndir+nshr+nshr,
					      defgradOld);
  PyDict_SetItemString(keywords,"defgradOld",defgradOldArray);

  //  FieldOld
  PyObject *fieldOldArray = create2dNpArray(nblock,nfieldv,
					    fieldOld);
  PyDict_SetItemString(keywords,"fieldOld",fieldOldArray);

  //  StressOld
  PyObject *stressOldArray = create2dNpArray(nblock,ndir+nshr,
					     stressOld);
  PyDict_SetItemString(keywords,"stressOld",stressOldArray);

  //  StateOld
  PyObject *stateOldArray = create2dNpArray(nblock,nstatev,
					    stateOld);
  PyDict_SetItemString(keywords,"stateOld",stateOldArray);

  //  enerInternOld
  PyObject *enerInternOldArray = create1dNpArray(nblock,enerInternOld);
  PyDict_SetItemString(keywords,"enerInternOld",enerInternOldArray);

  //  enerInelasOld
  PyObject *enerInelasOldArray = create1dNpArray(nblock,enerInelasOld);
  PyDict_SetItemString(keywords,"enerInelasOld",enerInelasOldArray);

  //  tempNew
  PyObject *tempNewArray = create1dNpArray(nblock,tempNew);
  PyDict_SetItemString(keywords,"tempNew",tempNewArray);

  //  StretchNew
  PyObject *stretchNewArray = create2dNpArray(nblock,ndir+nshr,
					      stretchNew);
  PyDict_SetItemString(keywords,"stretchNew",stretchNewArray);
  
  //  DefGradNew
  PyObject *defgradNewArray = create2dNpArray(nblock,
					      ndir+nshr+nshr,
					      defgradNew);
  PyDict_SetItemString(keywords,"defgradNew",defgradNewArray);

  //  FieldNew
  PyObject *fieldNewArray = create2dNpArray(nblock,nfieldv,
					    fieldNew);
  PyDict_SetItemString(keywords,"fieldNew",fieldNewArray);

  //
  // Evaluate the pyVUMAT model for the inputs
  //
  PyObject *retVal = PyObject_Call(_evaluateMethod,emptyArg,keywords);
  if (retVal == NULL) PyErr_Print();

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
  if (!parseSuccess) PyErr_Print();

  //
  // Store the results to return to FEA code
  //

  // StressNew
  // Make sure the returned array is 2D (nblock x ndir+nshr);
  unsigned int numOutDims = PyArray_NDIM(stressNewArray);
  npy_intp *outDims = PyArray_DIMS(stressNewArray);
  assert(numOutDims == 2);
#if defined(C_ORDERING)
  assert(outDims[0] == ndir+nshr);
  assert(outDims[1] == nblock);  
#else  
  assert(outDims[0] == nblock);  
  assert(outDims[1] == ndir+nshr);
#endif
  double *tmpStressNew = (double *)PyArray_DATA(stressNewArray);
  memcpy(stressNew,tmpStressNew,nblock*(ndir+nshr)*sizeof(double));

  // StateNew
  // Make sure the returned array is 2D (nblock x nstatev);
  numOutDims = PyArray_NDIM(stateNewArray);
  outDims = PyArray_DIMS(stateNewArray);
  assert(numOutDims == 2);
#if defined(C_ORDERING)
  assert(outDims[0] == nstatev);
  assert(outDims[1] == nblock);  
#else  
  assert(outDims[0] == nblock);  
  assert(outDims[1] == nstatev);
#endif
  double *tmpStateNew = (double *)PyArray_DATA(stateNewArray);
  memcpy(stateNew,tmpStateNew,nblock*nstatev*sizeof(double));

  // EnerInternNew
  // Make sure the returned array is 1D (nblock);
  numOutDims = PyArray_NDIM(enerInternNewArray);
  outDims = PyArray_DIMS(enerInternNewArray);
  assert(numOutDims == 1);  
  assert(outDims[0] == nblock);  
  double *tmpEnerInternNew = (double *)PyArray_DATA(enerInternNewArray);
  memcpy(enerInternNew,tmpEnerInternNew,nblock*sizeof(double));
  
  // EnerInelasNew
  // Make sure the returned array is 1D (nblock);
  numOutDims = PyArray_NDIM(enerInelasNewArray);
  outDims = PyArray_DIMS(enerInelasNewArray);
  assert(numOutDims == 1);  
  assert(outDims[0] == nblock);  
  double *tmpEnerInelasNew = (double *)PyArray_DATA(enerInelasNewArray);
  memcpy(enerInelasNew,tmpEnerInelasNew,nblock*sizeof(double));

  Py_XDECREF(lanneal_py);  
  Py_XDECREF(stepTime_py);    
  Py_XDECREF(totalTime_py);  
  Py_XDECREF(dt_py);
  Py_XDECREF(nblock_py);
  Py_XDECREF(ndir_py);
  Py_XDECREF(nshr_py);
  Py_XDECREF(nprops_py);
  Py_XDECREF(nfieldv_py);
  Py_XDECREF(nstatev_py);  
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
  
  // Uncomment this line if you are printing to the
  // log file and need to see the output in real time
  //  fflush(_logFile);

  return;
}

#if !defined(IN_ABAQUS)
extern "C" void vumat_(
// Read only
  const int *blockInfo, const int *ndir, const int *nshr, const int *nstatev,
  const int *nfieldv, const int *nprops, const int *lanneal,
  const double *stepTime, const double *totalTime, const double *dt,
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
  vumat(blockInfo, ndir, nshr, nstatev,
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
#endif // Not in Abaqus
