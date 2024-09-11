# Makefile for PyVUMAT

#################### START USER INPUT ####################

# C++ compiler
CC = g++

PYTHON_LIB_DIR = 
PYTHON_LIB = pythonX.Y

# Find with: python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
PYTHON_INC_DIR =

# Find with: python -c "from numpy import get_include; print(get_include())"
NUMPY_INC_DIR =

##################### END USER INPUT #####################

CFLAGS = -fPIC -Wall -Wextra -O3 -g -Wno-unused-parameter #-DC_ORDERING 
LDFLAGS = -shared
INCLUDES = -I$(PYTHON_INC_DIR) -I$(NUMPY_INC_DIR) 
LIBRARIES = -Wl,-rpath,${PYTHON_LIB_DIR} -L${PYTHON_LIB_DIR} -l${PYTHON_LIB}

TARGET_LIB = libpyvumat.so  # target lib
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
