#include <string.h>
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

#define INT_VAL_1D(x,i) ( *(npy_int32*)(x->data + x->strides[0]*i) )
#define INT_VAL_2D(x,i,j) ( *(npy_int32*)(x->data + x->strides[0]*i + x->strides[1]*j) )

#define debug 0


static PyObject* compute_intersection_matrix(PyObject *self, PyObject *arg) {

  // TODO: handle position_cost
  PyObject *o_parcellation1, *o_parcellation2, *o_inter_graph;

  PyArrayObject *p1, *p2, *inter_graph;
  int i, npos;

  PyArg_ParseTuple(arg, "OOO", &o_parcellation1, &o_parcellation2, 
                   &o_inter_graph);

  p1 = (PyArrayObject*) PyArray_ContiguousFromObject(o_parcellation1,
                                                     PyArray_INT32,  
                                                     1, 1);

  p2 = (PyArrayObject*) PyArray_ContiguousFromObject(o_parcellation2,
                                                     PyArray_INT32,  
                                                     1, 1);
  
  inter_graph = (PyArrayObject*) PyArray_ContiguousFromObject(o_inter_graph,
                                                              PyArray_INT32,
                                                              2, 2);
  if (debug) {
    printf("Wrapping done\n");
    fflush(stdout);
  }
  npos = p1->dimensions[0];
  
  if (debug) {    
    printf("Npos: %d\n", npos);
    fflush(stdout);
  }
  
  for(i=0; i<npos ; i++){
    if (debug) {
        printf("%d ",i);
        fflush(stdout);
    }
    INT_VAL_2D(inter_graph, INT_VAL_1D(p1,i), INT_VAL_1D(p2,i)) += 1;
  }
  
  if (debug) {
      printf("Computation of intersection matrix done !\n");
      fflush(stdout);
  }
  Py_DECREF(p1);
  Py_DECREF(p2);

  
  Py_INCREF(Py_None);
  if (debug) {
      printf("Return\n");
      fflush(stdout);
  }
  return Py_None;
}

static PyMethodDef methods[] = {
  {"compute_intersection_matrix", compute_intersection_matrix, METH_VARARGS, 
   "Compute the intersection matrix M between two parcellations," 
   "so that M(i,j)=|S1_i \\cap S2_j| where Sn_p is the p^th patch of "
   "parcellation n"},
  {NULL, NULL, 0, NULL},
};

void initcparcellation(void)
{
  Py_InitModule("cparcellation", methods);
  import_array();	
}
