# Makefile for PyVUMAT

#################### START USER INPUT ####################

# C++ compiler
CC = icpc  

PYTHON_LIB_DIR = ${HOME}/.conda/envs/pytorch/lib 
PYTHON_LIB = python3.9

# Find with: python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
PYTHON_INC_DIR = ${HOME}/.conda/envs/pytorch/include/python3.9 

# Find with: python -c "from numpy import get_include; print(get_include())"
NUMPY_INC_DIR = ${HOME}/.conda/envs/pytorch/lib/python3.9/site-packages/numpy/core/include

##################### END USER INPUT #####################

CFLAGS = -fPIC -Wall -Wextra -O2 -g  # C flags
LDFLAGS = -shared   # linking flags
INCLUDES = -I$(PYTHON_INC_DIR) -I$(NUMPY_INC_DIR)
LIBRARIES = -Wl,-rpath,${PYTHON_LIB_DIR} -L${PYTHON_LIB_DIR} -l${PYTHON_LIB}

TARGET_LIB = libpyVUMAT.so  # target lib
SRCS = pyVUMAT.cpp  # source files
OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: ${TARGET_LIB}

$(TARGET_LIB): $(OBJS)
	$(CC) $(LDFLAGS) $(LIBRARIES) -o $@ $^

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

.PHONY: clean
clean:
	-rm ${TARGET_LIB} ${OBJS}
