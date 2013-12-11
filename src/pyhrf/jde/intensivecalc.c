#include <string.h>
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

#define PYA_TVAL(x,i) ( *(npy_float64*)(x->data + x->strides[0]*i) ) // TO TEST 
#define PYAI_TVAL(x,i) ( *(npy_int32*)(x->data + x->strides[0]*i) ) // TO TEST 

#define PYA_MVAL(x,i,j,T) ( *(npy_float64*)(x->data + x->strides[0]*(i*(1-T)+j*T) + x->strides[1]*(j*(1-T)+i*T)) )

//TODO : support transposition ?
#define PYA_M2VAL(x,i,j,k) ( *(npy_float64*)(x->data + x->strides[0]*i + x->strides[1]*j + x->strides[2]*k) )

#define PYAI_MVAL(x,i,j,T) ( *(npy_int32*)(x->data + x->strides[0]*(i*(1-T)+j*T) + x->strides[1]*(j*(1-T)+i*T)) )

#define CA_MVAL(x,i,j,nc,T) (x[(i*(1-T)+j*T)*nc + (j*(1-T)+i*T)])

//#define MVAL(x,i,j,nc) ( x[i*nc+j] )

#define L_CI 0 // inactivated class label
#define L_CA 1 // activated class label
#define L_CD 2 // deactivated class label

#define debug 0

void pyADotPyAToDbl(PyArrayObject* loper, int nll, int ncl, int lTransp, 
                    PyArrayObject* roper, int nlr, int ncr, int rTransp, 
                    npy_float64* res) 
{ 
  int i,j,k; 
  int imax, jmax, kmax, ncRes; 

  // Handle transpostions for matrix dimensions : 
  imax = nll*(1-lTransp)+ncl*lTransp; 
  jmax = ncr*(1-rTransp)+nlr*rTransp; 
  kmax = nlr*(1-rTransp)+ncr*rTransp; 
  
  ncRes = ncr*(1-rTransp)+nlr*rTransp;

  for (i=0 ; i<imax ; i++) 
    for (j=0 ; j<jmax ; j++) 
      { 
        CA_MVAL(res,i,j,ncRes,0) = 0.0; 
        for (k=0 ; k<kmax ; k++) 
          { 
            //printf("[i=%d,j=%d] += %f * %f\n",i,j,PYA_MVAL(loper,i,k,lTransp), 
            //       PYA_MVAL(roper,k,j,rTransp));
            //fflush(stdout);
            CA_MVAL(res,i,j,ncRes,0) +=                        \
              PYA_MVAL(loper,i,k,lTransp) *                    \
              PYA_MVAL(roper,k,j,rTransp);
          } 
        //printf("[i=%d,j=%d] = %f\n",i,j, CA_MVAL(res,i,j,ncRes,0));
        //fflush(stdout);
      } 
} 

void dblDotPyAToPyA(npy_float64* loper, int nll, int ncl, int lTransp, 
                    PyArrayObject* roper, int nlr, int ncr, int rTransp, 
                    PyArrayObject* res)
{ 
  int i,j,k; 
  int imax, jmax, kmax; 

  // Handle transpostions for matrix dimensions : 
  imax = nll*(1-lTransp)+ncl*lTransp; 
  jmax = ncr*(1-rTransp)+nlr*rTransp; 
  kmax = nlr*(1-rTransp)+ncr*rTransp; 

  for (i=0 ; i<imax ; i++) 
    for (j=0 ; j<jmax ; j++) 
      { 
        PYA_MVAL(res,i,j,0) = 0.0; 
        for (k=0 ; k<kmax ; k++) 
          { 
            //printf("[i=%d,j=%d] += %f * %f\n",i,j,CA_MVAL(loper,i,k,ncl,lTransp),
            //       PYA_MVAL(roper,k,j,rTransp)); 
            //fflush(stdout);
            PYA_MVAL(res,i,j,0) += \
              CA_MVAL(loper,i,k,ncl,lTransp) * \
              PYA_MVAL(roper,k,j,rTransp); 
          } 
        //printf("[i=%d,j=%d] = %f\n",i,j, PYA_MVAL(res,i,j,0)); 
        //fflush(stdout);
      } 
} 
 

static PyObject* quadFormNorm(PyObject *self, PyObject *arg)  
{ 
  PyObject *oMatX, *oMatQ, *oRes;  
  PyArrayObject* matX, *matQ, *res;  
  int nlx, ncx, nlq, ncq;  
  npy_float64 *matXTQ; 
  PyArg_ParseTuple(arg, "OOO", &oMatX, &oMatQ, &oRes);  
  matX = (PyArrayObject*) PyArray_ContiguousFromObject(oMatX, PyArray_FLOAT64,  
                                                       2, 2);  
  matQ = (PyArrayObject*) PyArray_ContiguousFromObject(oMatQ, PyArray_FLOAT64,  
                                                       2, 2);  
  res = (PyArrayObject*) PyArray_ContiguousFromObject(oRes, PyArray_FLOAT64,  
                                                      2, 2);  
  nlx = matX->dimensions[0]; 
  ncx = matX->dimensions[1]; 

  nlq = matQ->dimensions[0]; 
  ncq = matQ->dimensions[1]; 
  
  matXTQ = malloc(sizeof(npy_float64)*ncx*ncq); 

  pyADotPyAToDbl(matX, nlx, ncx, 1, matQ, nlq, ncq, 0, matXTQ); 
  //printf("xT.Q done\n");
  dblDotPyAToPyA(matXTQ, ncx, ncq, 0, matX, nlx, ncx, 0, res);
  //printf("xT.Q.X done\n");

  free(matXTQ);

  Py_DECREF(matQ);  
  Py_DECREF(matX);  
  Py_DECREF(res);  
   
  Py_INCREF(Py_None); 
  return Py_None;  
} 


static PyObject* computeStLambda(PyObject *self, PyObject *arg)
//PyArray* nrls, PyArray* stackX, npy_float64* delta, npy_float64* varMBY, PyArray* destStDS, PyArray* destStDY)
{	
  npy_float64* stDelta;
  PyObject *oNrls, *oStackX, *oDelta, *oVarMBY, *oDestStDS, *oDestStDY;
  PyArrayObject* nrls, *stackX, *delta, *varMBY, *destStDS, *destStDY;
  npy_float64* s;
  npy_float64 * pdestStDSuvi, *pdestStDYui;
  npy_float64 nrlij;
  int nbVox, hrflgt, ny, nbCond;
  int idx, i, j, k, u, v;

  //  printf("trying arg parsing ... \n");
  //  fflush(stdout);
  PyArg_ParseTuple(arg, "OOOOOO", &oNrls, &oStackX, &oDelta, &oVarMBY, &oDestStDS, &oDestStDY);
  //  printf("arg parsing ok\n");
  //  fflush(stdout);
  nrls = (PyArrayObject*) PyArray_ContiguousFromObject(oNrls, PyArray_FLOAT64, 2, 2);
  stackX = (PyArrayObject*) PyArray_ContiguousFromObject(oStackX, PyArray_FLOAT64, 2, 2);
  delta = (PyArrayObject*) PyArray_ContiguousFromObject(oDelta, PyArray_FLOAT64, 2, 2);
  varMBY = (PyArrayObject*) PyArray_ContiguousFromObject(oVarMBY, PyArray_FLOAT64, 2, 2);
  destStDS = (PyArrayObject*) PyArray_ContiguousFromObject(oDestStDS, PyArray_FLOAT64, 3, 3);
  destStDY = (PyArrayObject*) PyArray_ContiguousFromObject(oDestStDY, PyArray_FLOAT64, 2, 2);
  
  //  printf("wrapping ok\n");
  //  fflush(stdout);

  nbCond = nrls->dimensions[0];
  nbVox = nrls->dimensions[1];
  ny = delta->dimensions[0];
  hrflgt = stackX->dimensions[1];
  
  stDelta = malloc(sizeof(npy_float64)*(hrflgt*ny));

  for (i=0 ; i<nbVox ; i++) 
    {
      s = calloc(sizeof(npy_float64),ny*hrflgt);
      for (j=0 ; j<nbCond ; j++)
	{
	  nrlij = *(npy_float64*)(nrls->data+
			     j*nrls->strides[0]+
			     i*nrls->strides[1]);
	  for (u=0 ; u<ny ; u++)
	    for (v=0 ; v<hrflgt ; v++)
	      s[u*hrflgt+v] += nrlij * (*(npy_float64*)(stackX->data
						   +(u+j*ny)*stackX->strides[0]
						   +v*stackX->strides[1]));
	}

      for (u=0 ; u<hrflgt ; u++)
	{
	  for (v=0 ; v<ny ; v++)
	    {
	      idx = u*ny+v;
	      stDelta[idx] = 0.0;
	      for (k=0 ; k<ny ; k++)
		{
		  stDelta[idx] += s[k*hrflgt+u] * (*(npy_float64*)(delta->data
							      + k*delta->strides[0]
							      + v*delta->strides[1]));
		}
	    }
	}
    
      // Calculation of St*Delta*S :
      for (u=0 ; u<hrflgt ; u++)
	for (v=0 ; v<hrflgt ; v++)
	  {
	    pdestStDSuvi = (npy_float64*) (destStDS->data
				      + u*destStDS->strides[0]
				      + v*destStDS->strides[1]
				      + i*destStDS->strides[2]);
	    *pdestStDSuvi = 0.0;
	    for (k=0 ; k<ny ; k++)
	      *pdestStDSuvi += stDelta[u*ny+k] * s[k*hrflgt+v];
	  }
    
      // Calculation of St*Delta*Y :
      for (u=0 ; u<hrflgt ; u++)
	{
	  pdestStDYui = (npy_float64*) (destStDY->data
				   + u*destStDY->strides[0]
				   + i*destStDY->strides[1]);
	  *pdestStDYui = 0.0;
	  for (k=0 ; k<ny ; k++)
	    *pdestStDYui += stDelta[u*ny+k] * (*(npy_float64*)(varMBY->data
							  + k*varMBY->strides[0]
							  + i*varMBY->strides[1]));
	}
    
      free(s);
    }
  
  free(stDelta);
  
  Py_DECREF(nrls);
  Py_DECREF(stackX);
  Py_DECREF(delta);
  Py_DECREF(varMBY);
  Py_DECREF(destStDS);
  Py_DECREF(destStDY);

  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject* computeStLambdaARModel(PyObject *self, PyObject *arg)
//PyArray* nrls, PyArray* stackX, npy_float64* delta, npy_float64* varMBY, PyArray* destStDS, PyArray* destStDY, PyArray* rb)
{	
  npy_float64* stDelta;
  PyObject *oNrls, *oStackX, *oDelta, *oVarMBY, *oDestStDS, *oDestStDY, *orb;
  PyArrayObject* nrls, *stackX, *delta, *varMBY, *destStDS, *destStDY, *rb;
  npy_float64* s;
  npy_float64 * pdestStDSuvi, *pdestStDYui;
  npy_float64 nrlij;
  int nbVox, hrflgt, ny, nbCond;
  int idx, i, j, k, u, v;

//  printf("trying arg parsing ... \n");
//  fflush(stdout);
  PyArg_ParseTuple(arg, "OOOOOOO", &oNrls, &oStackX, &oDelta, &oVarMBY, &oDestStDS, &oDestStDY,&orb);
//  printf("arg parsing ok\n");
  nrls = (PyArrayObject*) PyArray_ContiguousFromObject(oNrls, PyArray_FLOAT64, 2, 2);
  stackX = (PyArrayObject*) PyArray_ContiguousFromObject(oStackX, PyArray_FLOAT64, 2, 2);
  // In this case, delta is defined in 3d since the autocorrelation structure varies
  // from one voxel to another
//  printf("before wrapping\n");

  delta = (PyArrayObject*) PyArray_ContiguousFromObject(oDelta, PyArray_FLOAT64, 3, 3);
  varMBY = (PyArrayObject*) PyArray_ContiguousFromObject(oVarMBY, PyArray_FLOAT64, 2, 2);
  destStDS = (PyArrayObject*) PyArray_ContiguousFromObject(oDestStDS, PyArray_FLOAT64, 3, 3);
  destStDY = (PyArrayObject*) PyArray_ContiguousFromObject(oDestStDY, PyArray_FLOAT64, 2, 2);
  rb = (PyArrayObject*) PyArray_ContiguousFromObject(orb, PyArray_FLOAT64, 1, 1);
  
//  printf("wrapping ok\n");
//  fflush(stdout);

  nbCond = nrls->dimensions[0];
  nbVox = nrls->dimensions[1];
  ny = varMBY->dimensions[0];
  hrflgt = stackX->dimensions[1];
  
  stDelta = malloc(sizeof(npy_float64)*(hrflgt*ny));

  for (i=0 ; i<nbVox ; i++)  {
      s = calloc(sizeof(npy_float64),ny*hrflgt);
      for (j=0 ; j<nbCond ; j++) {
	    nrlij = *(npy_float64*)(nrls->data+
			     j*nrls->strides[0]+
			     i*nrls->strides[1]);
	    for (u=0 ; u<ny ; u++)
	        for (v=0 ; v<hrflgt ; v++)
	            s[u*hrflgt+v] += nrlij * (*(npy_float64*)(stackX->data
						      +(u+j*ny)*stackX->strides[0]
						      +v*stackX->strides[1]));
	    }
      for (u=0 ; u<hrflgt ; u++) {
	    for (v=0 ; v<ny ; v++) {
//	      idx = i*hrflgt + u*ny + v;
          idx = u*ny + v;
	      stDelta[idx] = 0.0;
	      for (k=0 ; k<ny ; k++) {
		    stDelta[idx] += s[k*hrflgt+u] * (*(npy_float64*)(delta->data
							      + k*delta->strides[0]
							      + v*delta->strides[1]
							      + i*delta->strides[2]));
		    }
	      }
	    }
      // Calculation of St*Delta*S :
      for (u=0 ; u<hrflgt ; u++)
        for (v=0 ; v<hrflgt ; v++) {
	      pdestStDSuvi = (npy_float64*) (destStDS->data
				         + u*destStDS->strides[0]
				         + v*destStDS->strides[1]
				         + i*destStDS->strides[2]);
	      *pdestStDSuvi = 0.0;
	      for (k=0 ; k<ny ; k++)
	        *pdestStDSuvi += stDelta[u*ny+k] * s[k*hrflgt+v];
	      *pdestStDSuvi /= * (npy_float64*)(rb->data + i*rb->strides[0]);
	    }
    
      // Calculation of St*Delta*Y :
      for (u=0 ; u<hrflgt ; u++) {
	    pdestStDYui = (npy_float64*) (destStDY->data
				    + u*destStDY->strides[0]
				    + i*destStDY->strides[1]);
	    *pdestStDYui = 0.0;
	    for (k=0 ; k<ny ; k++)
	      *pdestStDYui += stDelta[u*ny+k] * (*(npy_float64*)(varMBY->data
					    + k*varMBY->strides[0]
					    + i*varMBY->strides[1]));
	    *pdestStDYui /= *(npy_float64*)(rb->data + i*rb->strides[0]);
	  }
      free(s);
    }
  free(stDelta);
  
  Py_DECREF(nrls);
  Py_DECREF(stackX);
  Py_DECREF(delta);
  Py_DECREF(varMBY);
  Py_DECREF(destStDS);
  Py_DECREF(destStDY);

  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject* computePtLambdaARModel(PyObject *self, PyObject *arg)
//npy_float64 * varP, npy_float64* delta, npy_float64 * nrls, npy_float64 *Xh, npy_float64* varMBY, npy_float64*  reps, PyArray* destPtDP, PyArray* destPtDY)
//            intensivecalc.computePtLambdaARModel(self.P,
//                                                  invAutoCorrNoise,
//                                                  varNrls,
//                                                  varXh,
//                                                  self.dataInput.varMBY,
//                                                  reps,
//                                                  self.varPtLambdaP,
//                                                  self.varPtLambdaYmP);

{	
  npy_float64* ptDelta, *ymPred;
  PyObject *oVarP, *oDelta, *oVarMBY, *oVarNRLs, *oVarXh, *oDestPtDP, *oDestPtDY, *oReps;
  PyArrayObject *varP, *delta, *varMBY, *varNRLs, *varXh, *destPtDP, *destPtDY, *varReps;
  npy_float64 *pdestPtDPuvi, *pdestPtDYui;
  int nbVox, dimdrift, ny,nbCond;
  int idx, i, k, u, v, j, start, stop; //, it;
  npy_float64 sumNrl;

//  PyArg_ParseTuple(arg, "OOOOOO", &oVarP, &oDelta, &oVarDmP, &oDestPtDP, &oDestPtDY, &oReps);
  PyArg_ParseTuple(arg, "OOOOOOOO", &oVarP, &oDelta, &oVarNRLs, &oVarXh, &oVarMBY, &oReps, &oDestPtDP, &oDestPtDY);
  varP = (PyArrayObject*) PyArray_ContiguousFromObject(oVarP, PyArray_FLOAT64, 2, 2);
  delta = (PyArrayObject*) PyArray_ContiguousFromObject(oDelta, PyArray_FLOAT64, 3, 3);
  varNRLs = (PyArrayObject*) PyArray_ContiguousFromObject(oVarNRLs, PyArray_FLOAT64, 2, 2);
  varXh = (PyArrayObject*) PyArray_ContiguousFromObject(oVarXh, PyArray_FLOAT64, 2, 2);
  varMBY = (PyArrayObject*) PyArray_ContiguousFromObject(oVarMBY, PyArray_FLOAT64, 2, 2);
  destPtDP = (PyArrayObject*) PyArray_ContiguousFromObject(oDestPtDP, PyArray_FLOAT64, 3, 3);
  destPtDY = (PyArrayObject*) PyArray_ContiguousFromObject(oDestPtDY, PyArray_FLOAT64, 2, 2);
  varReps = (PyArrayObject*) PyArray_ContiguousFromObject(oReps, PyArray_FLOAT64, 1, 1);

  ny = varMBY->dimensions[0];
  nbVox = varMBY->dimensions[1];
  dimdrift = varP->dimensions[1];
  nbCond = varNRLs->dimensions[0];
//  printf("dimDrift=%d\t Ny=%d\t NbVox=%d\n",dimdrift,ny,nbVox);
//  printf("dimP[0]=%d\t dimP[1]=%d\n",varP->dimensions[0],dimdrift);
//  printf("dimMBY[0]=%d\t dimMBY[1]=%d\n",varMBY->dimensions[0],varMBY->dimensions[1]);

//  ptDelta = malloc(sizeof(npy_float64)*(dimdrift*nbVox));
  ptDelta = malloc(sizeof(npy_float64)*(dimdrift*ny));
  for (i=0 ; i<nbVox ; i++)  {
/*      sumNrl = calloc(sizeof(npy_float64),ny);*/
      ymPred = calloc(sizeof(npy_float64),ny);
//      printf("\t Apres alloc\n");
      for (u=0 ; u<ny ; u++) {
        sumNrl=0.;
        for (j=0 ; j<nbCond ; j++)
	  sumNrl += *(npy_float64*)(varNRLs->data 
			  + j*varNRLs->strides[0]
			  + i*varNRLs->strides[1]) * 
                         (*(npy_float64 *)(varXh->data 
                          + u*varXh->strides[0]
                          + j*varXh->strides[1]));
//        printf("\t u=%d, Apres calcul sumNrl\n",u);
        ymPred[u] +=  *(npy_float64*)(varMBY->data 
                        + u*varMBY->strides[0] 
                        + i*varMBY->strides[1]) - sumNrl;
//        printf("\t u=%d, Apres calcul ymPred\n",u);
      }
//        printf("\t Apres calcul ymPred\n");

      for (u=0 ; u<dimdrift ; u++) {
        start=0;
        stop=2;
        for (v=0 ; v<ny ; v++) {
          idx = u*ny + v;
	  ptDelta[idx] = 0.0;
	  for (k=start ; k<stop ; k++) {
//	  for (k=0 ; k<ny ; k++) {
	    ptDelta[idx] += (*(npy_float64*)(varP->data + k*varP->strides[0] 
                                                   + u*varP->strides[1])) * 
                            (*(npy_float64*)(delta->data + k*delta->strides[0] 
                                                    + v*delta->strides[1] 
                                                    + i*delta->strides[2]));
	  }
          if (v>0)
            start+=1;
          if (v<ny-1)
stop+=1;
	}
      }
//      printf("\t Apres calcul ptDelta\n");

      // Calculation of Pt*Delta*P :
      for (u=0 ; u<dimdrift ; u++)
        for (v=0 ; v<dimdrift ; v++) {
	      pdestPtDPuvi = (npy_float64*) (destPtDP->data
				         + u*destPtDP->strides[0]
				         + v*destPtDP->strides[1]
				         + i*destPtDP->strides[2]);
	      *pdestPtDPuvi = 0.0;
	      for (k=0 ; k<ny ; k++) {
//                printf("\t PtD[%d,%d] =%1.3f\n",k,u,ptDelta[u*ny+k]); 
	        *pdestPtDPuvi += ptDelta[u*ny+k] * (*(npy_float64*)(varP->data 
                                                    + k*varP->strides[0] 
                                                    + v*varP->strides[1]));
              }
	      *pdestPtDPuvi /= *(npy_float64*)(varReps->data + i*varReps->strides[0]);
	    }
//       printf("\t Apres calcul PtDeltaP\n");
      // Calculation of Pt*Delta*(Y-Predict):
      for (u=0 ; u<dimdrift ; u++) {
	    pdestPtDYui = (npy_float64*) (destPtDY->data
				    + u*destPtDY->strides[0]
				    + i*destPtDY->strides[1]);
	    *pdestPtDYui = 0.0;
	    for (k=0 ; k<ny ; k++)
	      *pdestPtDYui += ptDelta[u*ny+k] * ymPred[k];
	    *pdestPtDYui /= *(npy_float64*)(varReps->data + i*varReps->strides[0]);
	}
//      printf("\t Vox %d:\t Apres calcul PtDeltaY\n",i);
/*      free(sumNrl);*/
      free(ymPred);
    }

  free(ptDelta);  
  Py_DECREF(varNRLs);
  Py_DECREF(varXh);
  Py_DECREF(delta);
  Py_DECREF(varMBY);
  Py_DECREF(destPtDP);
  Py_DECREF(destPtDY);
  Py_DECREF(varReps);

  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject* computeStLambdaSparse(PyObject *self, PyObject *arg)
//PyArray* nrls, PyArray* stackX, PyArray* idxStackX, npy_float64* delta, npy_float64* varMBY, PyArray* destStDS, PyArray* destStDY, PyArray* rb)
{	
  npy_float64* stDelta;
  PyObject *oNrls, *oStackX, *oDelta, *oVarMBY, *oDestStDS, *oDestStDY, *orb, *oidxStackX;
  PyArrayObject* nrls, *stackX, *delta, *varMBY, *destStDS, *destStDY, *rb, *idxStackX;
  npy_float64* s;
  npy_float64 * pdestStDSuvi, *pdestStDYui;
  npy_float64 nrlij;
  int nbVox, hrflgt, ny, nbCond;
  int idx, i, j, k, u, v, ik, nbIdxStackX, x, y, iStackX;
  int* sparseIdxSInCol;
  int* nbSparseIdxSInCol;

  //  printf("trying arg parsing ... \n");
  //  fflush(stdout);
  PyArg_ParseTuple(arg, "OOOOOOOO", &oNrls, &oStackX, &oidxStackX, &oDelta, &oVarMBY, &oDestStDS, &oDestStDY, &orb);
  //  printf("arg parsing ok\n");
  //  fflush(stdout);
  nrls = (PyArrayObject*) PyArray_ContiguousFromObject(oNrls, PyArray_FLOAT64, 2, 2);
  stackX = (PyArrayObject*) PyArray_ContiguousFromObject(oStackX, PyArray_INT, 2, 2);
  delta = (PyArrayObject*) PyArray_ContiguousFromObject(oDelta, PyArray_FLOAT64, 2, 2);
  varMBY = (PyArrayObject*) PyArray_ContiguousFromObject(oVarMBY, PyArray_FLOAT64, 2, 2);
  destStDS = (PyArrayObject*) PyArray_ContiguousFromObject(oDestStDS, PyArray_FLOAT64, 3, 3);
  destStDY = (PyArrayObject*) PyArray_ContiguousFromObject(oDestStDY, PyArray_FLOAT64, 2, 2);
  rb = (PyArrayObject*) PyArray_ContiguousFromObject(orb, PyArray_FLOAT64, 1, 1);
  idxStackX = (PyArrayObject*) PyArray_ContiguousFromObject(oidxStackX, PyArray_INT, 1, 1);

  //printf("wrapping ok\n");
  //fflush(stdout);

  nbCond = nrls->dimensions[0];
  nbVox = nrls->dimensions[1];
  ny = delta->dimensions[0];
  hrflgt = stackX->dimensions[1];
    
  sparseIdxSInCol = malloc(sizeof(int)*(hrflgt*ny));
  nbSparseIdxSInCol = malloc(sizeof(int)*hrflgt);
  
  nbIdxStackX = idxStackX->dimensions[0];
  
  nbSparseIdxSInCol = calloc(sizeof(int), hrflgt);
  s = malloc(sizeof(npy_float64)*ny*hrflgt);
  for (i=0 ; i<nbVox ; i++) 
    {

      stDelta = calloc(sizeof(npy_float64),(hrflgt*ny));

      for( iStackX=0 ; iStackX<nbIdxStackX ; iStackX+=2)
	{
	  x = *(int*) (idxStackX->data + iStackX*idxStackX->strides[0]);
	  y = *(int*) (idxStackX->data + (iStackX+1)*idxStackX->strides[0]);
	  
	  //printf("treating [x=%d,y=%d]=%d from stackX\n", x, y, (*(int*)(stackX->data + x*stackX->strides[0] + y*stackX->strides[1])));

	  u = x%ny;
	  v = y ;
	  j = x/ny;
	  //printf(" |-> u=%d, v=%d, j=%d\n", u,v,j);
 	  nrlij = *(npy_float64*)(nrls->data + j*nrls->strides[0] + i*nrls->strides[1]); 
	  //printf(" s[u*hrflgt+v] before = %1.2f\n", s[u*hrflgt+v]);
	  s[u*hrflgt+v] = nrlij;
	  if (i==0)
	    {
	      sparseIdxSInCol[v*ny+nbSparseIdxSInCol[v]] = u;
	      nbSparseIdxSInCol[v]++;
	    }
	  //printf(" s[u*hrflgt+v] after = %1.2f\n", s[u*hrflgt+v]);
	  //s[u*hrflgt+v] +=  (*(npy_float64*)(stackX->data + (u+j*ny)*stackX->strides[0] + v*stackX->strides[1])) == 0 ? 0.0 : nrlij;
	}

      for (u=0 ; u<hrflgt ; u++)
	{
	  if (nbSparseIdxSInCol[u] > 0)
	    {
	      for (v=0 ; v<ny ; v++)
		{
		  idx = u*ny+v;
		  //stDelta[idx] = 0.0;
		  //printf("doing : stDelta[u=%d,v=%d] :\n",u,v); 
		  for (ik=0 ; ik<nbSparseIdxSInCol[u] ; ik++)
		    {
		      k = sparseIdxSInCol[u*ny+ik];
		      //printf("  s[%d,%d]=%1.0e * d[%d,%d]=%1.0e\n", k, u, s[k*hrflgt+u],k,v, vd);
		      stDelta[idx] += s[k*hrflgt+u] * (*(npy_float64*)(delta->data + 
								  k*delta->strides[0] + 
								  v*delta->strides[1]));
		    }
		  //printf("\n");
		}
	    }
	}

    
      // Calculation of St*Delta*S :
      for (u=0 ; u<hrflgt ; u++)
	for (v=0 ; v<hrflgt ; v++)
	  {
	    pdestStDSuvi = (npy_float64*) (destStDS->data
				      + u*destStDS->strides[0]
				      + v*destStDS->strides[1]
				      + i*destStDS->strides[2]);
	    *pdestStDSuvi = 0.0;
	    for (ik=0 ; ik<nbSparseIdxSInCol[v] ; ik++)
	      {
		k = sparseIdxSInCol[v*ny+ik];
		*pdestStDSuvi += stDelta[u*ny+k] * s[k*hrflgt+v];
	      }
	  }
      for (u=0 ; u<hrflgt ; u++)
	for (v=0 ; v<hrflgt ; v++)
	  {
	    pdestStDSuvi = (npy_float64*) (destStDS->data
				      + u*destStDS->strides[0]
				      + v*destStDS->strides[1]
				      + i*destStDS->strides[2]);

	    *pdestStDSuvi /= * (npy_float64*)(rb->data + i*rb->strides[0]);
	  }


      //printf("calc of stdeltas ok\n");
      //fflush(stdout);
      
      // Calculation of St*Delta*Y :
      for (u=0 ; u<hrflgt ; u++)
	{
	  pdestStDYui = (npy_float64*) (destStDY->data
				   + u*destStDY->strides[0]
				   + i*destStDY->strides[1]);
	  *pdestStDYui = 0.0;
	  for (k=0 ; k<ny ; k++)
	    *pdestStDYui += stDelta[u*ny+k] * (*(npy_float64*)(varMBY->data
							  + k*varMBY->strides[0]
							  + i*varMBY->strides[1]));
	}
      for (u=0 ; u<hrflgt ; u++)
	{
	  pdestStDYui = (npy_float64*) (destStDY->data
				   + u*destStDY->strides[0]
				   + i*destStDY->strides[1]);
	  *pdestStDYui /= *(npy_float64*)(rb->data + i*rb->strides[0]);
	}

      free(stDelta);
    }
  free(s);
  free(nbSparseIdxSInCol);
  free(sparseIdxSInCol);


  Py_DECREF(nrls);
  Py_DECREF(stackX);
  Py_DECREF(delta); 
  Py_DECREF(varMBY);
  Py_DECREF(destStDS);
  Py_DECREF(destStDY);
  Py_DECREF(rb);
  Py_DECREF(idxStackX); 

  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject* computeStLambdaARModelSparse(PyObject *self, PyObject *arg)
//PyArray* nrls, PyArray* stackX, PyArray* idxStackX, npy_float64* delta, npy_float64* varMBY, PyArray* destStDS, PyArray* destStDY, PyArray* rb)
{	
  npy_float64* stDelta;
  PyObject *oNrls, *oStackX, *oDelta, *oVarMBY, *oDestStDS, *oDestStDY, *orb, *oidxStackX;
  PyArrayObject* nrls, *stackX, *delta, *varMBY, *destStDS, *destStDY, *rb, *idxStackX;
  npy_float64* s;
  npy_float64 * pdestStDSuvi, *pdestStDYui;
  npy_float64 nrlij;
  int nbVox, hrflgt, ny, nbCond;
  int idx, i, j, k, u, v, ik, nbIdxStackX, x, y, iStackX;
  int* sparseIdxSInCol;
  int* nbSparseIdxSInCol;

  //  printf("trying arg parsing ... \n");
  //  fflush(stdout);
  PyArg_ParseTuple(arg, "OOOOOOOO", &oNrls, &oStackX, &oidxStackX, &oDelta, &oVarMBY, &oDestStDS, &oDestStDY, &orb);
  //  printf("arg parsing ok\n");
  //  fflush(stdout);
  nrls = (PyArrayObject*) PyArray_ContiguousFromObject(oNrls, PyArray_FLOAT64, 2, 2);
  stackX = (PyArrayObject*) PyArray_ContiguousFromObject(oStackX, PyArray_INT, 2, 2);
  delta = (PyArrayObject*) PyArray_ContiguousFromObject(oDelta, PyArray_FLOAT64, 3, 3);
  varMBY = (PyArrayObject*) PyArray_ContiguousFromObject(oVarMBY, PyArray_FLOAT64, 2, 2);
  destStDS = (PyArrayObject*) PyArray_ContiguousFromObject(oDestStDS, PyArray_FLOAT64, 3, 3);
  destStDY = (PyArrayObject*) PyArray_ContiguousFromObject(oDestStDY, PyArray_FLOAT64, 2, 2);
  rb = (PyArrayObject*) PyArray_ContiguousFromObject(orb, PyArray_FLOAT64, 1, 1);
  idxStackX = (PyArrayObject*) PyArray_ContiguousFromObject(oidxStackX, PyArray_INT, 1, 1);

  //printf("wrapping ok\n");
  //fflush(stdout);

  nbCond = nrls->dimensions[0];
  nbVox = nrls->dimensions[1];
  ny = varMBY->dimensions[0];
  hrflgt = stackX->dimensions[1];
    
  sparseIdxSInCol = malloc(sizeof(int)*(hrflgt*ny));
  nbSparseIdxSInCol = malloc(sizeof(int)*hrflgt);
  
  nbIdxStackX = idxStackX->dimensions[0];
  
  nbSparseIdxSInCol = calloc(sizeof(int), hrflgt);
  s = malloc(sizeof(npy_float64)*ny*hrflgt);
  for (i=0 ; i<nbVox ; i++) 
    {

      stDelta = calloc(sizeof(npy_float64),(hrflgt*ny));

      for( iStackX=0 ; iStackX<nbIdxStackX ; iStackX+=2)
	{
	  x = *(int*) (idxStackX->data + iStackX*idxStackX->strides[0]);
	  y = *(int*) (idxStackX->data + (iStackX+1)*idxStackX->strides[0]);
	  
	  u = x%ny;
	  v = y ;
	  j = x/ny;
	  //printf(" |-> u=%d, v=%d, j=%d\n", u,v,j);
 	  nrlij = *(npy_float64*)(nrls->data + j*nrls->strides[0] + i*nrls->strides[1]); 
	  //printf(" s[u*hrflgt+v] before = %1.2f\n", s[u*hrflgt+v]);
	  s[u*hrflgt+v] = nrlij;
	  if (i==0)
	    {
	      sparseIdxSInCol[v*ny+nbSparseIdxSInCol[v]] = u;
	      nbSparseIdxSInCol[v]++;
	    }
	  //printf(" s[u*hrflgt+v] after = %1.2f\n", s[u*hrflgt+v]);
	  //s[u*hrflgt+v] +=  (*(npy_float64*)(stackX->data + (u+j*ny)*stackX->strides[0] + v*stackX->strides[1])) == 0 ? 0.0 : nrlij;
	}

      for (u=0 ; u<hrflgt ; u++)
	{
	  if (nbSparseIdxSInCol[u] > 0)
	    {
	      for (v=0 ; v<ny ; v++)
		{
		  idx = u*ny+v;
		  //stDelta[idx] = 0.0;
		  //printf("doing : stDelta[u=%d,v=%d] :\n",u,v); 
		  for (ik=0 ; ik<nbSparseIdxSInCol[u] ; ik++)
		    {
		      k = sparseIdxSInCol[u*ny+ik];
		      //printf("  s[%d,%d]=%1.0e * d[%d,%d]=%1.0e\n", k, u, s[k*hrflgt+u],k,v, vd);
		      stDelta[idx] += s[k*hrflgt+u] * (*(npy_float64*)(delta->data
							      + k*delta->strides[0]
							      + v*delta->strides[1]
							      + i*delta->strides[2]));
		    }
		}
	    }
	}

    
      // Calculation of St*Delta*S :
      for (u=0 ; u<hrflgt ; u++)
	for (v=0 ; v<hrflgt ; v++)
	  {
	    pdestStDSuvi = (npy_float64*) (destStDS->data
				      + u*destStDS->strides[0]
				      + v*destStDS->strides[1]
				      + i*destStDS->strides[2]);
	    *pdestStDSuvi = 0.0;
	    for (ik=0 ; ik<nbSparseIdxSInCol[v] ; ik++)
	      {
		k = sparseIdxSInCol[v*ny+ik];
		*pdestStDSuvi += stDelta[u*ny+k] * s[k*hrflgt+v];
	      }
	  }
      for (u=0 ; u<hrflgt ; u++)
	for (v=0 ; v<hrflgt ; v++)
	  {
	    pdestStDSuvi = (npy_float64*) (destStDS->data
				      + u*destStDS->strides[0]
				      + v*destStDS->strides[1]
				      + i*destStDS->strides[2]);

	    *pdestStDSuvi /= * (npy_float64*)(rb->data + i*rb->strides[0]);
	  }

      // Calculation of St*Delta*Y :
      for (u=0 ; u<hrflgt ; u++)
	{
	  pdestStDYui = (npy_float64*) (destStDY->data
				   + u*destStDY->strides[0]
				   + i*destStDY->strides[1]);
	  *pdestStDYui = 0.0;
	  for (k=0 ; k<ny ; k++)
	    *pdestStDYui += stDelta[u*ny+k] * (*(npy_float64*)(varMBY->data
							  + k*varMBY->strides[0]
							  + i*varMBY->strides[1]));
	}
      for (u=0 ; u<hrflgt ; u++)
	{
	  pdestStDYui = (npy_float64*) (destStDY->data
				   + u*destStDY->strides[0]
				   + i*destStDY->strides[1]);
	  *pdestStDYui /= *(npy_float64*)(rb->data + i*rb->strides[0]);
	}

      free(stDelta);
    }
  free(s);
  free(nbSparseIdxSInCol);
  free(sparseIdxSInCol);


  Py_DECREF(nrls);
  Py_DECREF(stackX);
  Py_DECREF(delta); 
  Py_DECREF(varMBY);
  Py_DECREF(destStDS);
  Py_DECREF(destStDY);
  Py_DECREF(rb);
  Py_DECREF(idxStackX); 

  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject* calcCorrEnergies(PyObject *self, PyObject *arg)
{
  PyObject *oCurLabels, *oEnergies, *oNeighbours;
  PyArrayObject *curLabels, *energies, *neighbours;
  int i, j, c, nn, spCorr, cond;
  int nbVox, nbClasses;
  int in, lab, lab1, lab2;
  int* nCount;
  npy_float64 deltaColor;

  //printf("calcCorrEnergies ... Arg parsing ...\n"); 
  //fflush(stdout); 
  PyArg_ParseTuple(arg, "iOOOdiii",&cond, &oCurLabels, &oEnergies, &oNeighbours, 
                   &deltaColor, &nbClasses, &lab1, &lab2);
  //printf("done !\n"); 
  //fflush(stdout); 
  //printf("Wrapping arrays ...\n"); 
  //fflush(stdout); 
  curLabels = (PyArrayObject*) PyArray_ContiguousFromObject(oCurLabels, 
                                                            PyArray_INT,  
                                                            2, 2);
  energies = (PyArrayObject*) PyArray_ContiguousFromObject(oEnergies, 
                                                           PyArray_FLOAT64,  
                                                           2, 2);
  neighbours = (PyArrayObject*) PyArray_ContiguousFromObject(oNeighbours, 
                                                             PyArray_INT,  
                                                             2, 2);
  /*
  labelFlags = (PyArrayObject*) PyArray_ContiguousFromObject(oLabels, 
                                                             PyArray_INT,  
                                                             1, 1);
  */
  //printf("done !\n"); 
  //fflush(stdout); 

  nCount = malloc(sizeof(int)*nbClasses);
  
  nbVox = curLabels->dimensions[1];
  //printf("nbVox = %d\n", nbVox); 
  //fflush(stdout); 

  j = cond;
  for (i=0 ; i<nbVox ; i++)
    {
      //printf("reset nCount ... \n"); 
      //fflush(stdout);
      for(c=0; c<nbClasses ; c++)
        nCount[c] = 0;
      //printf("done \n"); 
      //fflush(stdout);

      nn = 0;
      //printf(" i=%d\n", i);
      //fflush(stdout);
      while(PYAI_MVAL(neighbours,i,nn,0) != -1)
        {
          in = PYAI_MVAL(neighbours,i,nn,0);
          //printf("neighbours[%d,%d] = %d\n", i, nn, in ); 
          //fflush(stdout); 
          lab = PYAI_MVAL(curLabels,j,in,0);
          //printf("labels[%d,%d] = %d\n", j, in, lab); 
          //fflush(stdout); 
          nCount[lab]++;
          nn ++;
        }
      spCorr = (nCount[lab1] - nCount[lab2]);
      PYA_MVAL(energies,j,i,0) = spCorr + deltaColor;
    }
  //printf("Test\n");
  free(nCount);
  Py_DECREF(curLabels);
  Py_DECREF(energies);
  Py_DECREF(neighbours);
  
  Py_INCREF(Py_None);
  return Py_None;
}


/*
 * Compute the diff of label counts in the neighbourhood of the given position iv
 * Return: sum_{j~iv} (q_j = lab1) - sum_{j~iv} (q_j = lab2)
 */
static int deltaECorr(PyArrayObject* neighbours, int maxNeighbours, int iv, 
                      PyArrayObject* labels, int lab1, int lab2, int nbClasses)
{
  int nn, in, lab, ic;
  int* nCount, dCount;
  if (debug) {
    printf("deltaECorr ... \n");
    fflush(stdout);
  }
  nCount = calloc(nbClasses, sizeof(int));
  nn = 0;
  if (debug) {
    printf("neighbours[%d] : \n", iv);
    fflush(stdout);
  }
  while(PYAI_MVAL(neighbours,iv,nn,0) != -1 && nn<maxNeighbours) {
    in = PYAI_MVAL(neighbours,iv,nn,0);

    lab = PYAI_TVAL(labels,in);
    if (debug){
      printf("%d ", lab ); 
      fflush(stdout); 
    }
    nCount[lab]++;
    nn++;
  }
  dCount = nCount[lab1] - nCount[lab2];
  if (debug){
    printf("\n"); 
    printf("counts:");
    for (ic=0 ; ic<nbClasses ; ic++)
      printf(" %d - %d |", ic, nCount[ic]);
    printf("\n");
    printf("lab1 = %d - lab2 = %d, count[%d]=%d, count[%d]=%d ", 
           lab1, lab2, lab1, nCount[lab1], lab2, nCount[lab2] );
    printf("-> Delta = %d\n", dCount);
  }
  free(nCount);
  // Py_Decref ??
  return dCount;
}


static npy_float64 deltaWCorr0 (int nbVox, int sumqj, float t1, float t2)
{
  npy_float64 num, denom;

  // If we consider proportion of activation
//   num = exp( t1 * (((float)(sumqj + 1)/(float)(nbVox)) - t2)) + 1.;
//   denom = exp( t1 * (((float)(sumqj)/(float)(nbVox)) - t2)) + 1.;
  
  // If we consider number of activated voxels
  num = exp( t1 * ((npy_float64)(sumqj) + 1. - t2)) + 1.;
  denom = exp( t1 * ((npy_float64)(sumqj) - t2)) + 1.;
  
  return (npy_float64)(num/denom);
  
}

static npy_float64 deltaWCorr1 (int nbVox, int sumqj, float t1, float t2)
{
  npy_float64 num, denom;
  
  // If we consider proportion of activation
  
//   num = exp( - t1 * (((float)(sumqj + 1)/(float)(nbVox)) - t2)) + 1.;
//   denom = exp( - t1 * (((float)(sumqj)/(float)(nbVox)) - t2)) + 1.;
  
  // If we consider number of activated voxels
  
  num = exp( - t1 * ((npy_float64)(sumqj) + 1. - t2)) + 1.;
  denom = exp( - t1 * ((npy_float64)(sumqj)- t2)) + 1.;
  
  return (npy_float64)(num/denom);
}


/*
 * Compute the diff of label counts in the neighbourhood of the given position iv
 * Return: sum_{j~iv} (q_j = lab1) - sum_{j~iv} (q_j = lab2)
 */
static int deltaECorr2(PyArrayObject* neighbours, int maxNeighbours, 
                       npy_int32 iv, 
                       PyArrayObject* labels, npy_int32 lab1, npy_int32 lab2, 
                       int nbClasses, int cond)
{
  int nn, ic, dCount;
  npy_int32 in, lab; 
  int* nCount;
  
  nCount = calloc(nbClasses, sizeof(int));
  nn = 0;
  if (debug) {
    printf("deltaECorr2 ...\n");
    printf("neighbours[%d] : ", iv);
    fflush(stdout);
  }
  fflush(stdout);
  
  while(PYAI_MVAL(neighbours,iv,nn,0) != -1 && nn<maxNeighbours) {
    in = PYAI_MVAL(neighbours,iv,nn,0);
    lab = PYAI_MVAL(labels,cond,in,0);
    fflush(stdout);
    if (debug){
      printf("%d ", lab ); 
      fflush(stdout); 
    }
    nCount[lab]++;
    nn++;
  }
  dCount = nCount[lab1] - nCount[lab2];
  if (debug){
    printf("\n"); 
    printf("counts:");
    for (ic=0 ; ic<nbClasses ; ic++)
      printf(" %d - %d |", ic, nCount[ic]);
    printf("\n");
    printf("lab1 = %d - lab2 = %d, count[%d]=%d, count[%d]=%d ", 
           lab1, lab2, lab1, nCount[lab1], lab2, nCount[lab2] );
    printf("-> Delta = %d\n", dCount);
    fflush(stdout); 
  }
  free(nCount);
  // Py_Decref ??
  return dCount;
}

static npy_float64 compute_ratio_lambda(int iv, npy_float64 beta, PyArrayObject* neighbours, 
                                   int maxNeighbours, PyArrayObject* labels, 
                                   int l1,
                                   int l2, int nbClasses, npy_float64* mApost, 
                                   npy_float64* vApost, npy_float64* sApost,
                                   PyArrayObject* mean, 
                                   PyArrayObject* var) {
  
  npy_float64 ratio, l, dECorr, e;
  
  ratio = sqrt(PYA_TVAL(var, l2) / PYA_TVAL(var, l1));
  
  l = ratio * (sApost[l1] / sApost[l2] * exp(0.5 * (mApost[l1] *         \
                                                    mApost[l1] /        \
                                                    vApost[l1] -        \
                                                    mApost[l2] *        \
                                                    mApost[l2] /        \
                                                    vApost[l2] -        \
                                                    PYA_TVAL(mean, l1) * \
                                                    PYA_TVAL(mean, l1) / \
                                                    PYA_TVAL(var, l1) + \
                                                    PYA_TVAL(mean, l2) * \
                                                    PYA_TVAL(mean, l2)/ \
                                                    PYA_TVAL(var, l2))
                                            )
               );
  e = exp(0.5 * (mApost[l1] *                                               \
                 mApost[l1] /                                           \
                 vApost[l1] -                                           \
                 mApost[l2] *                                           \
                 mApost[l2] /                                           \
                 vApost[l2] -                                           \
                 PYA_TVAL(mean, l1) *                                   \
                 PYA_TVAL(mean, l1) /                                   \
                 PYA_TVAL(var, l1) +                                    \
                 PYA_TVAL(mean, l2) *                                   \
                 PYA_TVAL(mean, l2)/                                    \
                 PYA_TVAL(var, l2)));
  if (debug) {
    printf(" ~~~~~~~~~~~~~~~ \n");
    printf("ratio * sApost[%d]/sApost[%d] = %1.2e * (%1.2e/%1.2e) = %1.2e\n",
           l1, l2, ratio, sApost[l1], sApost[l2], ratio*(sApost[l1]/sApost[l2]));
    printf("exp(0.5* (mp[%d]**2/vp[%d] - mp[%d]**2/vp[%d] - m[%d]**2/v[%d] + m[%d]**2/v[%d])=\n",
           l1, l1, l2, l2, l1, l1, l2, l2);
    printf("exp(0.5* (%1.2e**2/%1.2e - %1.2e**2/%1.2e - %1.2e**2/%1.2e + %1.2e**2/%1.2e)\n",
           mApost[l1], vApost[l1], mApost[l2], vApost[l2], PYA_TVAL(mean, l1),
           PYA_TVAL(var, l1), PYA_TVAL(mean, l2), PYA_TVAL(var, l2));
    printf("exp(0.5* (%1.2e - %1.2e - %1.2e\t + %1.2e) = %1.2e\n",
           mApost[l1]*mApost[l1]/vApost[l1], mApost[l2]*mApost[l2]/vApost[l2],
           PYA_TVAL(mean, l1)*PYA_TVAL(mean, l1)/PYA_TVAL(var, l1),
           PYA_TVAL(mean, l2)*PYA_TVAL(mean, l2)/PYA_TVAL(var, l2), e);
    printf(" ==> l = %1.2e\n", l);
  }

  // compute delta of correlation energy:
  dECorr = deltaECorr(neighbours, maxNeighbours, iv, labels, l1, l2, nbClasses);
  if (debug) {

    printf("exp(beta*dEcorr) = exp(%f*%f) = %f\n", beta, dECorr, exp(beta*dECorr));
    fflush(stdout);
    printf(" ~~~~~~~~~~~~~~~ \n");
  }

  l *= exp(beta*dECorr);

  return l;
}


static npy_float64 compute_ratio_lambda_WithRelCond(npy_float64 l , int nbVox, int sumqj, float t1, float t2) {
  
  npy_float64 dWCorr1, lRelCond;
  
  dWCorr1 = deltaWCorr1(nbVox, sumqj, t1, t2);
//   printf("dWCorr1 = %f\n", dWCorr1);
  if (debug) {

    printf("exp(dWCorr1) = exp(%f) = %f\n", dWCorr1, exp(dWCorr1));
    fflush(stdout);
    printf(" ~~~~~~~~~~~~~~~ \n");
  }
  
  lRelCond = l*dWCorr1;
  
  return lRelCond;
}

static npy_float64 compute_ratio_lambda_WithIRRelCond(int iv, npy_float64 beta, PyArrayObject* neighbours, 
                                   int maxNeighbours, PyArrayObject* labels, 
                                   int l1,
                                   int l2, int nbClasses, int nbVox, int sumqj, float t1, float t2) 
{
  
  npy_float64 lIRRelCond, dWCorr0, dECorr;
  
  dWCorr0 = deltaWCorr0(nbVox, sumqj, t1, t2);
//   printf("dWCorr0 = %f", dWCorr0);
  if (debug) {

    printf("exp(dWCorr0) = exp(%f) = %f\n", dWCorr0, exp(dWCorr0));
    fflush(stdout);
    printf(" ~~~~~~~~~~~~~~~ \n");
  }
  
  // compute delta of correlation energy:
  dECorr = deltaECorr(neighbours, maxNeighbours, iv, labels, l1, l2, nbClasses);
  if (debug) {

    printf("exp(beta*dEcorr) = exp(%f*%f) = %f\n", beta, dECorr, exp(beta*dECorr));
    fflush(stdout);
    printf(" ~~~~~~~~~~~~~~~ \n");
  }
  lIRRelCond = dWCorr0*exp(beta*dECorr);
  
  return lIRRelCond;
}
  

static npy_float64 compute_ratio_lambda2(npy_int32 iv, npy_float64 beta, 
					 PyArrayObject* neighbours, 
					 int maxNeighbours, 
					 PyArrayObject* labels, 
					 int l1, int l2, int nbClasses, 
					 npy_float64* mApost, 
					 npy_float64* vApost, 
					 npy_float64* sApost,
					 PyArrayObject* mean, 
					 PyArrayObject* var, int cond) 
{
  
  npy_float64 ratio, l, dECorr, e;
  
  ratio = sqrt(PYA_MVAL(var,l2,cond,0) / PYA_MVAL(var, l1,cond,0));
  
  l = ratio * (sApost[l1] / sApost[l2] * exp(0.5 * (mApost[l1] *         \
                                                    mApost[l1] /        \
                                                    vApost[l1] -        \
                                                    mApost[l2] *        \
                                                    mApost[l2] /        \
                                                    vApost[l2] -        \
                                                    PYA_MVAL(mean,l1,cond,0) * \
                                                    PYA_MVAL(mean,l1,cond,0) / \
                                                    PYA_MVAL(var,l1,cond,0) + \
                                                    PYA_MVAL(mean,l2,cond,0) * \
                                                    PYA_MVAL(mean,l2,cond,0)/ \
                                                    PYA_MVAL(var,l2,cond,0))
                                            )
               );
  e = exp(0.5 * (mApost[l1] *                                               \
                 mApost[l1] /                                           \
                 vApost[l1] -                                           \
                 mApost[l2] *                                           \
                 mApost[l2] /                                           \
                 vApost[l2] -                                           \
                 PYA_MVAL(mean,l1,cond,0) *                                   \
                 PYA_MVAL(mean,l1,cond,0) /                                   \
                 PYA_MVAL(var,l1,cond,0) +                                    \
                 PYA_MVAL(mean,l2,cond,0) *                                   \
                 PYA_MVAL(mean,l2,cond,0)/                                    \
                 PYA_MVAL(var,l2,cond,0)));
  if (debug) {
    printf(" ~~~~~~~~~~~~~~~ \n");
    printf("ratio * sApost[%d]/sApost[%d] = %1.2e * (%1.2e/%1.2e) = %1.2e\n",
           l1, l2, ratio, sApost[l1], sApost[l2], ratio*(sApost[l1]/sApost[l2]));
    printf("exp(0.5* (mp[%d]**2/vp[%d] - mp[%d]**2/vp[%d] - m[%d]**2/v[%d] + m[%d]**2/v[%d])=\n",
           l1, l1, l2, l2, l1, l1, l2, l2);
    printf("exp(0.5* (%1.2e**2/%1.2e - %1.2e**2/%1.2e - %1.2e**2/%1.2e + %1.2e**2/%1.2e)\n",
           mApost[l1], vApost[l1], mApost[l2], vApost[l2], PYA_MVAL(mean, l1,cond,0),
           PYA_MVAL(var, l1,cond,0), PYA_MVAL(mean, l2,cond,0), PYA_MVAL(var, l2,cond,0));
    printf("exp(0.5* (%1.2e - %1.2e - %1.2e\t + %1.2e) = %1.2e\n",
           mApost[l1]*mApost[l1]/vApost[l1], mApost[l2]*mApost[l2]/vApost[l2],
           PYA_MVAL(mean, l1,cond,0)*PYA_MVAL(mean, l1,cond,0)/PYA_MVAL(var, l1,cond,0),
           PYA_MVAL(mean, l2,cond,0)*PYA_MVAL(mean, l2,cond,0)/PYA_MVAL(var, l2,cond,0), e);
    printf(" ==> l = %1.2e\n", l);
  }

  // compute delta of correlation energy:
  dECorr = deltaECorr2(neighbours, maxNeighbours, iv, labels, l1, l2, nbClasses, cond);
  if (debug) {

    printf("exp(beta*dEcorr) = exp(%f*%f) = %f\n", beta, dECorr, exp(beta*dECorr));
    fflush(stdout);
    printf(" ~~~~~~~~~~~~~~~ \n");
  }

  l *= exp(beta*dECorr);

  return l;
}

static npy_float64 compute_ratio_lambda2_withRelCond_NEW(npy_int32 iv, npy_float64 beta, 
                     PyArrayObject* neighbours, 
                     int maxNeighbours, 
                     PyArrayObject* labels, 
                     int l1, int l2, int nbClasses, 
                     npy_float64* mApost, 
                     npy_float64* vApost, 
                     npy_float64* sApost,
                     PyArrayObject* mean, 
                     PyArrayObject* var, int cond,
                     int sumqj, float t1, float t2, int nbVox, float wj, int sampleW) 
{
  
  npy_float64 ratio, l, dECorr, dWCorr, dWCorr0, dWCorr1, proba, pm0, pm1;
  
  ratio = sqrt(PYA_MVAL(var,l1,cond,0) / PYA_MVAL(var, l2,cond,0));
  
  l = ratio * (sApost[l2] / sApost[l1] * exp(-0.5 * (mApost[l1] *         \
                                                    mApost[l1] /        \
                                                    vApost[l1] -        \
                                                    mApost[l2] *        \
                                                    mApost[l2] /        \
                                                    vApost[l2] -        \
                                                    PYA_MVAL(mean,l1,cond,0) * \
                                                    PYA_MVAL(mean,l1,cond,0) / \
                                                    PYA_MVAL(var,l1,cond,0) + \
                                                    PYA_MVAL(mean,l2,cond,0) * \
                                                    PYA_MVAL(mean,l2,cond,0)/ \
                                                    PYA_MVAL(var,l2,cond,0))
                                            )
               );
  // compute delta of correlation energy:
  dECorr = deltaECorr2(neighbours, maxNeighbours, iv, labels, l1, l2, nbClasses, cond);
  
  if(sampleW)
  {
      pm0 = 1. / (1. + exp( - t1 * ((npy_float64)(sumqj) - t2)) ); // pm when q_j^m = 0  
      pm1 = 1. / (1. + exp( - t1 * ((npy_float64)(sumqj) + 1. - t2)) ); // pm when q_j^m = 1
  
      dWCorr0 = wj * pm0 + (1. - wj)*(1. - pm0);
      dWCorr1 = wj * pm1 + (1. - wj)*(1. - pm1);
  
      dWCorr = dWCorr0/dWCorr1;
  }
  else
      dWCorr = 1.;
  
  proba = exp(beta*dECorr) * dWCorr / (1. - wj*(1. - l));
  
  return 1./(1. + proba);
}

static npy_float64 compute_ratio_lambda_WithIRRelCond2(int iv, npy_float64 beta, PyArrayObject* neighbours, 
                                   int maxNeighbours, PyArrayObject* labels, 
                                   int l1,
                                   int l2, int nbClasses, int nbVox, int sumqj, float t1, float t2, int cond) 
{
  
  npy_float64 lIRRelCond, dWCorr0, dECorr;
  
  dWCorr0 = deltaWCorr0(nbVox, sumqj, t1, t2);
//   printf("dWCorr0 = %f\n", dWCorr0);
  if (debug) {

    printf("exp(dWCorr0) = exp(%f) = %f\n", dWCorr0, exp(dWCorr0));
    fflush(stdout);
    printf(" ~~~~~~~~~~~~~~~ \n");
  }
  
  // compute delta of correlation energy:
  dECorr = deltaECorr2(neighbours, maxNeighbours, iv, labels, l1, l2, nbClasses, cond);
  if (debug) {
    printf("exp(beta*dEcorr) = exp(%f*%f) = %f\n", beta, dECorr, exp(beta*dECorr));
    fflush(stdout);
    printf(" ~~~~~~~~~~~~~~~ \n");
  }
  lIRRelCond = dWCorr0*exp(beta*dECorr);
  
  return lIRRelCond;  
  
}


/* static int deltaECorr2C(PyArrayObject* neighbours, int maxNeighbours, int iv,  */
/*                         PyArrayObject* labels) */
/* { */
/*   int c, nn, in, lab; */
/*   int nCount[2]; */
/*   for(c=0; c<2 ; c++) */
/*     nCount[c] = 0; */
/*   nn = 0; */
/*   while(PYAI_MVAL(neighbours,iv,nn,0) != -1 && nn<maxNeighbours) { */
/*     in = PYAI_MVAL(neighbours,iv,nn,0); */
/*     if (debug){ */
/*       printf("neighbours[%d,%d] = %d\n", iv, nn, in );  */
/*       fflush(stdout);  */
/*     } */
/*     lab = PYAI_TVAL(labels,in); */
/*     if (debug){ */
/*       printf("labels[%d] = %d\n", in, lab);  */
/*       fflush(stdout);  */
/*     } */
/*     nCount[lab]++; */
/*     nn++; */
/*   } */
/*   if (debug){ */
/*     printf("sp correlation done !\n"); */
/*     fflush(stdout);  */
/*   } */
/*   return nCount[0] - nCount[1]; */
/* } */

static PyObject* sampleSmmNrl(PyObject *self, PyObject *arg)
{  
  PyObject *oVoxOrder, *oNoiseVars, *oNeighbours, *oyTilde;
  PyObject *oLabels, *oVarXh, *oNrls, *oNrlsSamples, *oLabelsSamples;
  PyObject *oVarXhtQ, *oMean, *oVar;

  PyArrayObject *voxOrder, *neighbours, *yTilde;
  PyArrayObject *labels, *varXh, *nrls, *nrlsSamples, *labelsSamples, *rb;
  PyArrayObject *varXhtQ, *mean, *var;

  int i, iv, n, nMax, nbVox, lab, maxNeighbours;
  int ic, nbClasses, sampleLabels, it, cond, xhVoxelWise;
  npy_float64 gTQg, varXjhtQjej, gTQgrb;
  npy_float64 *vApost, *sApost, *mApost;
  npy_float64 deltaNrl, oldNrl, beta;
  npy_float64 rl_I_A, rl_D_A, rl_A_D, rl_I_D, rl_D_I, rl_A_I, lApostD;
  npy_float64 *ej;
  npy_float64 *lApost2;

  PyArg_ParseTuple(arg, "OOOOOOOOOOddOOiiii", 
                   &oVoxOrder, &oNoiseVars, &oNeighbours,
                   &oyTilde, &oLabels, &oVarXh, &oNrls, &oNrlsSamples, 
                   &oLabelsSamples, &oVarXhtQ, &gTQg, &beta, &oMean, 
                   &oVar, &nbClasses, &sampleLabels, &it, &cond);


  rb = (PyArrayObject*) PyArray_ContiguousFromObject(oNoiseVars, 
                                                     PyArray_FLOAT64, 1, 1);


  voxOrder = (PyArrayObject*) PyArray_ContiguousFromObject(oVoxOrder, 
                                                           PyArray_INT32,  
                                                           1, 1);

  neighbours = (PyArrayObject*) PyArray_ContiguousFromObject(oNeighbours, 
                                                             PyArray_INT32,  
                                                             2, 2);
  
  yTilde = (PyArrayObject*) PyArray_ContiguousFromObject(oyTilde, 
                                                         PyArray_FLOAT64,  
                                                         2, 2);

  labels = (PyArrayObject*) PyArray_ContiguousFromObject(oLabels, 
                                                         PyArray_INT32,  
                                                         1, 1);

  varXh = (PyArrayObject*) PyArray_ContiguousFromObject(oVarXh, 
                                                        PyArray_FLOAT64,  
                                                        2, 2);

  nrls = (PyArrayObject*) PyArray_ContiguousFromObject(oNrls, PyArray_FLOAT64, 
                                                       1, 1);

  nrlsSamples = (PyArrayObject*) PyArray_ContiguousFromObject(oNrlsSamples, 
                                                              PyArray_FLOAT64,  
                                                              1, 1);

  var = (PyArrayObject*) PyArray_ContiguousFromObject(oVar, PyArray_FLOAT64, 
                                                       1, 1);

  mean = (PyArrayObject*) PyArray_ContiguousFromObject(oMean, PyArray_FLOAT64, 
                                                       1, 1);

  labelsSamples = (PyArrayObject*) PyArray_ContiguousFromObject(oLabelsSamples, 
                                                           PyArray_FLOAT64, 1, 1);


  varXhtQ = (PyArrayObject*) PyArray_ContiguousFromObject(oVarXhtQ, 
                                                          PyArray_FLOAT64, 2, 2);
  if (1 && debug){
    printf("Wrapping done\n "); 
    fflush(stdout); 
  }

  vApost = malloc(sizeof(npy_float64)*nbClasses);
  sApost = malloc(sizeof(npy_float64)*nbClasses);
  mApost = malloc(sizeof(npy_float64)*nbClasses);

  lApost2 = malloc(sizeof(npy_float64)*(nbClasses-1));

  maxNeighbours = neighbours->dimensions[1];
  nbVox = voxOrder->dimensions[0];
  if (debug){
    printf("nbVox=%d , nbClasses=%d\n", nbVox, nbClasses);
    fflush(stdout); 
  }

  nMax = varXh->dimensions[1];
  //nMax = varXh->dimensions[0];
  if (debug){
    printf("nMax=%d \n", nMax);
    fflush(stdout); 
  }

  
  xhVoxelWise = 0;  
  if (varXh->dimensions[0] > 1) {
    xhVoxelWise = 1;
    //printf("varXh voxel wise!\n");
  }
  //else
  //  printf("varXh not voxel wise!\n");

  ej = malloc(sizeof(npy_float64)*nMax);
  if (debug){  
    printf("Current components :\n");
    printf("mean CI = %f, var CI = %f \n", PYA_TVAL(mean, L_CI), 
           PYA_TVAL(var, L_CI));
    printf("mean CA = %f, var CA = %f \n", PYA_TVAL(mean,L_CA), 
           PYA_TVAL(var,L_CA));

    if (nbClasses == 3)
      printf("mCD = %f, vCD = %f \n", PYA_TVAL(mean,L_CD), PYA_TVAL(var,L_CD));
    
    printf("beta = %f \n", beta);
    printf("gTQg = %f \n", gTQg);
    fflush(stdout); 
  }

  for(i=0 ; i<nbVox ; i++) {
    iv = PYAI_TVAL(voxOrder,i);
    
    if (debug) {
      printf("it%04d-cond%02d-Vox%03d ... \n", it,cond,iv);
      fflush(stdout); 
    }

    /*********************************/
    /* Compute posterior components  */
    /*********************************/
    for(n=0 ; n<nMax ; n++){
      //ej[n] = PYA_MVAL(yTilde,n,iv,0) + PYA_TVAL(nrls,iv) * PYA_TVAL(varXh,n);
      if (!xhVoxelWise)
        ej[n] = PYA_MVAL(yTilde,n,iv,0) + PYA_TVAL(nrls,iv) * PYA_MVAL(varXh,0,n,0);
      else
        ej[n] = PYA_MVAL(yTilde,n,iv,0) + PYA_TVAL(nrls,iv) * PYA_MVAL(varXh,iv,n,0);
    }

    if (0 && debug){
      printf("ej :\n");
      for( n=0 ; n<nMax ; n++)
        printf("%f ", ej[n]);
      printf("\n");
      fflush(stdout); 
    }
    
    varXjhtQjej = 0.;
    for(n=0 ; n<nMax ; n++){
      if (!xhVoxelWise)
        varXjhtQjej += PYA_MVAL(varXhtQ,0,n,0) * ej[n];
      else
        varXjhtQjej += PYA_MVAL(varXhtQ,iv,n,0) * ej[n];
    }
    if (0 && debug){
      printf("varXhtQjej = %f !\n", varXjhtQjej);
      fflush(stdout); 
    }

    varXjhtQjej /= PYA_TVAL(rb,iv);

    if (0 && debug){
      printf("PYA_TVAL(rb,i) = %f !\n", PYA_TVAL(rb,iv));
      printf("varXhtQjej = %f !\n", varXjhtQjej);
      fflush(stdout); 
    }
    
    

    if (xhVoxelWise){
      gTQg = 0.;
      for (n=0 ; n<nMax ; n++)
        gTQg += PYA_MVAL(varXhtQ,iv,n,0) * PYA_MVAL(varXh,iv,n,0);
    }
    gTQgrb = gTQg / PYA_TVAL(rb,iv);

    for (ic=0 ; ic<nbClasses ; ic++) {
      vApost[ic] = 1. / (1. / PYA_TVAL(var,ic) + gTQgrb);
      sApost[ic] = sqrt(vApost[ic]);
      mApost[ic] = vApost[ic] * (PYA_TVAL(mean,ic) / PYA_TVAL(var,ic) + \
                                 varXjhtQjej);
    }

    if (debug){
      printf("gTQgrb = %e\n", gTQgrb);
      printf("mApostCA = %e, sApostCA = %e\n", mApost[L_CA], sApost[L_CA]);
      printf("mApostCI = %e, sApostCI = %e\n", mApost[L_CI], sApost[L_CI]);
      if (nbClasses == 3)
        printf("mApostCD = %e, sApostCD = %e\n", mApost[L_CD], sApost[L_CD]);
      fflush(stdout); 
    }
    
    /********************/
    /* label sampling   */
    /********************/
    
    if (sampleLabels) {
      if (debug){
        printf("label sampling is ON\n");
        fflush(stdout);
      }
      // Compute lambdaTilde :
      rl_I_A = compute_ratio_lambda(iv, beta, neighbours, maxNeighbours, labels, 
                                    L_CI, L_CA, nbClasses, mApost, vApost, sApost, 
                                    mean, var);

      if (nbClasses == 2) {
        lApost2[L_CI] = 1. - 1. / (1. + rl_I_A);
        if (debug) {
          printf("rl_I_A = %e\n", rl_I_A);
          fflush(stdout);
        }
      }
      else if (nbClasses == 3) {
        rl_D_A = compute_ratio_lambda(iv, beta, neighbours, maxNeighbours, labels, 
                                      L_CD, L_CA, nbClasses, mApost, vApost, 
                                      sApost, mean, var);
        rl_A_D = compute_ratio_lambda(iv, beta, neighbours, maxNeighbours, labels, 
                                      L_CA, L_CD, nbClasses, mApost, vApost, 
                                      sApost, mean, var);
        rl_I_D = compute_ratio_lambda(iv, beta, neighbours, maxNeighbours, labels, 
                                      L_CI, L_CD, nbClasses, mApost, vApost, 
                                      sApost, mean, var);
        rl_A_I = compute_ratio_lambda(iv, beta, neighbours, maxNeighbours, labels, 
                                      L_CA, L_CI, nbClasses, mApost, vApost, 
                                      sApost, mean, var);
        rl_D_I = compute_ratio_lambda(iv, beta, neighbours, maxNeighbours, labels, 
                                      L_CD, L_CI, nbClasses, mApost, vApost, 
                                      sApost, mean, var);
          
        lApostD = 1. / (1. + rl_I_D + rl_A_D);
        lApost2[L_CI] = 1. - 1. / (1. + rl_I_A + rl_D_A) - lApostD ;
        lApost2[L_CA] = 1. - lApostD - lApost2[L_CI];

        if (debug) {
          printf("rl_I_A = %e\n", rl_I_A);
          printf("rl_D_A = %e\n", rl_D_A);
          printf("rl_A_D = %e\n", rl_A_D);
          printf("rl_I_D = %e\n", rl_I_D);
          printf("rl_A_I = %e\n", rl_A_I);
          printf("rl_D_I = %e\n", rl_D_I);
          //printf("LApostI = %f\n", 1. / (1. + rl_I_A + rl_D_A));
          printf("P apost I = %e\n", 1. / (1. + rl_A_I + rl_D_I));
          printf("P apost A = %e\n", 1. / (1. + rl_I_A + rl_D_A));
          printf("P apost D = %e\n", lApostD);
          fflush(stdout);
        }
      }
        
      
      lab = nbClasses-1;
      for (ic=0 ; ic<nbClasses-1 ; ic++)
        if (PYA_TVAL(labelsSamples,iv) <= lApost2[ic])
          lab = ic;
      
      //lab = (int) (PYA_TVAL(labelsSamples, iv) <= 1. / (1. + rl_I_A));
      PYAI_TVAL(labels,iv) = lab;
      if (debug){
        printf("nrl = %f\n", PYA_TVAL(nrls,iv));
        printf("cumul p CI = %e\n", lApost2[L_CI]);
        if (nbClasses == 3)
          printf("cumul p CA = %e\n", lApost2[L_CA]);
        printf("random = %f !\n", PYA_TVAL(labelsSamples, iv));
        printf("-> label = %d !\n", lab);
        fflush(stdout); 
      }
    }
    else { // label = true one
      if (debug) {
        printf("label sampling is OFF\n");
        fflush(stdout); 
      }
      lab = PYAI_TVAL(labels,iv);
      if (debug) {
        printf("label: %d\n", lab);
        fflush(stdout);
      }
    }

    //drawNrl();
    oldNrl = PYA_TVAL(nrls,iv);
    PYA_TVAL(nrls,iv) = sApost[lab] * PYA_TVAL(nrlsSamples,iv) + mApost[lab];
    
    if (debug) {
      printf("nrl = %f !\n", PYA_TVAL(nrls,iv));
      fflush(stdout); 
    }
    //if ( isnan(PYA_TVAL(nrls,iv)) ) {
      //printf("nan in nrls\n");
      //printf("ej:\n");
      //for( n=0 ; n<nMax ; n++)
      //  printf("%f ", ej[n]);
      //printf("\n");
      //exit(1);
    //}
    
    /*********************/
    /* Update y tilde    */
    /*********************/
    deltaNrl = oldNrl - PYA_TVAL(nrls,iv);
    for(n=0 ; n<nMax ; n++){
      //PYA_MVAL(yTilde,n,iv,0) += deltaNrl * PYA_TVAL(varXh,n);
      if (!xhVoxelWise)
        PYA_MVAL(yTilde,n,iv,0) += deltaNrl * PYA_MVAL(varXh,0,n,0);
      else
        PYA_MVAL(yTilde,n,iv,0) += deltaNrl * PYA_MVAL(varXh,iv,n,0);
    }

    if (1 && debug){
      printf("deltaNrl = %f ", deltaNrl);
      printf("updated ytilde:\n");
      for(n=0 ; n<nMax ; n++){
        printf("%f ", PYA_MVAL(yTilde,n,iv,0));
      }
      printf("\n");
      fflush(stdout); 
    }
  }  

  free(ej);
  free(sApost);
  free(mApost);
  free(vApost);
  free(lApost2);

  Py_DECREF(voxOrder);
  Py_DECREF(yTilde);
  Py_DECREF(labels);
  Py_DECREF(varXh);
  Py_DECREF(nrls);
  Py_DECREF(nrlsSamples);
  Py_DECREF(mean);
  Py_DECREF(var);
  Py_DECREF(labelsSamples);
  Py_DECREF(rb);
  Py_DECREF(varXhtQ);
  Py_DECREF(neighbours);
  Py_INCREF(Py_None);
  return Py_None; 
}


static PyObject* sampleSmmNrlWithRelVar(PyObject *self, PyObject *arg)
{  
  PyObject *oVoxOrder, *oNoiseVars, *oNeighbours, *oyTilde;
  PyObject *oLabels, *oVarXh, *oNrls, *oNrlsSamples, *oLabelsSamples;
  PyObject *oVarXhtQ, *oMean, *oVar;
  PyObject *oCardClass;

  PyArrayObject *voxOrder, *neighbours, *yTilde;
  PyArrayObject *labels, *varXh, *nrls, *nrlsSamples, *labelsSamples, *rb;
  PyArrayObject *varXhtQ, *mean, *var;
  PyArrayObject *CardClass;

  int i, iv, n, nMax, nbVox, lab, maxNeighbours;
  int ic, nbClasses, sampleLabels, it, cond, xhVoxelWise;
  npy_int32 wj;
  npy_float64 gTQg, varXjhtQjej, gTQgrb;
  npy_float64 *vApost, *sApost, *mApost;
  npy_float64 deltaNrl, oldNrl, beta;
  npy_float64 rl_I_A; //, rl_D_A, rl_A_D, rl_I_D, rl_D_I, rl_A_I;
  npy_float64 rl_I_A_RelCond; //, rl_D_A_RelCond, rl_A_D_RelCond, rl_I_D_RelCond, rl_D_I_RelCond, rl_A_I_RelCond, lApostD_RelCond;
  npy_float64 rl_I_A_IRRelCond; //, rl_D_A_IRRelCond, rl_A_D_IRRelCond, rl_I_D_IRRelCond, rl_D_I_IRRelCond, rl_A_I_IRRelCond, lApostD_IRRelCond;
  npy_float64 *ej;
  npy_float64 *lApost2_RelCond, *lApost2_IRRelCond;

  // Parameters of Sigmoid function
  float t1 = 1.0, t2 = 94.0;

  PyArg_ParseTuple(arg, "OOOOOOOOOOddOOiiiiiO", 
                   &oVoxOrder, &oNoiseVars, &oNeighbours,
                   &oyTilde, &oLabels, &oVarXh, &oNrls, &oNrlsSamples, 
                   &oLabelsSamples, &oVarXhtQ, &gTQg, &beta, &oMean, 
                   &oVar, &nbClasses, &sampleLabels, &it, &cond, &wj, &oCardClass);

  printf("PyArg_ParseTuple OK\n");
  fflush(stdout);
                   
  rb = (PyArrayObject*) PyArray_ContiguousFromObject(oNoiseVars, 
                                                     PyArray_FLOAT64, 1, 1);


  voxOrder = (PyArrayObject*) PyArray_ContiguousFromObject(oVoxOrder, 
                                                           PyArray_INT32,  
                                                           1, 1);

  neighbours = (PyArrayObject*) PyArray_ContiguousFromObject(oNeighbours, 
                                                             PyArray_INT32,  
                                                             2, 2);
  
  yTilde = (PyArrayObject*) PyArray_ContiguousFromObject(oyTilde, 
                                                         PyArray_FLOAT64,  
                                                         2, 2);

  labels = (PyArrayObject*) PyArray_ContiguousFromObject(oLabels, 
                                                         PyArray_INT32,  
                                                         1, 1);

  varXh = (PyArrayObject*) PyArray_ContiguousFromObject(oVarXh, 
                                                        PyArray_FLOAT64,  
                                                        2, 2);

  nrls = (PyArrayObject*) PyArray_ContiguousFromObject(oNrls, PyArray_FLOAT64, 
                                                       1, 1);

  nrlsSamples = (PyArrayObject*) PyArray_ContiguousFromObject(oNrlsSamples, 
                                                              PyArray_FLOAT64,  
                                                              1, 1);

  var = (PyArrayObject*) PyArray_ContiguousFromObject(oVar, PyArray_FLOAT64, 
                                                       1, 1);

  mean = (PyArrayObject*) PyArray_ContiguousFromObject(oMean, PyArray_FLOAT64, 
                                                       1, 1);

  CardClass = (PyArrayObject*) PyArray_ContiguousFromObject(oCardClass, PyArray_INT32, 
                                                       1, 1);

  labelsSamples = (PyArrayObject*) PyArray_ContiguousFromObject(oLabelsSamples, 
                                                           PyArray_FLOAT64, 1, 1);


  varXhtQ = (PyArrayObject*) PyArray_ContiguousFromObject(oVarXhtQ, 
                                                          PyArray_FLOAT64, 2, 2);
  if (1 && debug){
    printf("Wrapping done\n "); 
    fflush(stdout); 
  }

  vApost = malloc(sizeof(npy_float64)*nbClasses);
  sApost = malloc(sizeof(npy_float64)*nbClasses);
  mApost = malloc(sizeof(npy_float64)*nbClasses);

  lApost2_RelCond = malloc(sizeof(npy_float64)*(nbClasses-1));
  lApost2_IRRelCond = malloc(sizeof(npy_float64)*(nbClasses-1));
  
  maxNeighbours = neighbours->dimensions[1];
  nbVox = voxOrder->dimensions[0];
  if (0 && debug){
    printf("nbVox=%d \n", nbVox);
    fflush(stdout); 
  }

  nMax = varXh->dimensions[1];
  //nMax = varXh->dimensions[0];
  if (0 && debug){
    printf("nMax=%d \n", nMax);
    fflush(stdout); 
  }

  
  xhVoxelWise = 0;  
  if (varXh->dimensions[0] > 1) {
    xhVoxelWise = 1;
    //printf("varXh voxel wise!\n");
  }
  //else
  //  printf("varXh not voxel wise!\n");

  ej = malloc(sizeof(npy_float64)*nMax);
  if (debug){  
    printf("Current components :\n");
    printf("mean CI = %f, var CI = %f \n", PYA_TVAL(mean, L_CI), 
           PYA_TVAL(var, L_CI));
    printf("mean CA = %f, var CA = %f \n", PYA_TVAL(mean,L_CA), 
           PYA_TVAL(var,L_CA));

    if (nbClasses == 3)
      printf("mCD = %f, vCD = %f \n", PYA_TVAL(mean,L_CD), PYA_TVAL(var,L_CD));
    
    printf("beta = %f \n", beta);
    printf("gTQg = %f \n", gTQg);
    fflush(stdout); 
  }


  for(i=0 ; i<nbVox ; i++) {
    iv = PYAI_TVAL(voxOrder,i);
    
    if (debug) {
      printf("it%04d-cond%02d-Vox%03d ... \n", it,cond,iv);
      fflush(stdout); 
    }

    /*********************************/
    /* Compute posterior components  */
    /*********************************/

    // Compute mean of ROI labels
//     printf("Compute mean of ROI labels (without q_%d^%d) for condition : %d\n", iv, cond, cond);
    int sumqj = 0.;
    sumqj = (PYAI_TVAL(CardClass, L_CA) - PYAI_TVAL(labels,iv));
//    printf("Mean = %f\n",sumqj);

    if (wj)
    {
        
       /* If wj = 1 The codition is relevant and we compute the posterior components
          as we did before introducing the relevant variable w */
         
       for(n=0 ; n<nMax ; n++){
        //ej[n] = PYA_MVAL(yTilde,n,iv,0) + PYA_TVAL(nrls,iv) * PYA_TVAL(varXh,n);
        if (!xhVoxelWise)
            ej[n] = PYA_MVAL(yTilde,n,iv,0) + PYA_TVAL(nrls,iv) * PYA_MVAL(varXh,0,n,0);
        else
            ej[n] = PYA_MVAL(yTilde,n,iv,0) + PYA_TVAL(nrls,iv) * PYA_MVAL(varXh,iv,n,0);
        }

        if (0 && debug){
            printf("ej :\n");
            for( n=0 ; n<nMax ; n++)
                printf("%f ", ej[n]);
            printf("\n");
            fflush(stdout); 
        }
    
        varXjhtQjej = 0.;
        for(n=0 ; n<nMax ; n++){
            if (!xhVoxelWise)
                varXjhtQjej += PYA_MVAL(varXhtQ,0,n,0) * ej[n];
            else
                varXjhtQjej += PYA_MVAL(varXhtQ,iv,n,0) * ej[n];
        }
        if (0 && debug){
            printf("varXhtQjej = %f !\n", varXjhtQjej);
            fflush(stdout); 
        }

        varXjhtQjej /= PYA_TVAL(rb,iv);

        if (0 && debug){
            printf("PYA_TVAL(rb,i) = %f !\n", PYA_TVAL(rb,iv));
            printf("varXhtQjej = %f !\n", varXjhtQjej);
            fflush(stdout); 
        }
    
        if (xhVoxelWise){
            gTQg = 0.;
            for (n=0 ; n<nMax ; n++)
            gTQg += PYA_MVAL(varXhtQ,iv,n,0) * PYA_MVAL(varXh,iv,n,0);
        }
        gTQgrb = gTQg / PYA_TVAL(rb,iv);

        for (ic=0 ; ic<nbClasses ; ic++) {
            vApost[ic] = 1. / (1. / PYA_TVAL(var,ic) + gTQgrb);
            sApost[ic] = sqrt(vApost[ic]);
            mApost[ic] = vApost[ic] * (PYA_TVAL(mean,ic) / PYA_TVAL(var,ic) + \
                                 varXjhtQjej);
        } 
        
    }

    else
    {
        for (ic=0 ; ic<nbClasses ; ic++) {
            vApost[ic] = PYA_TVAL(var,0);
            sApost[ic] = sqrt(vApost[ic]);
            mApost[ic] = PYA_TVAL(mean,0);
        } 
    }

    if (debug){
      printf("gTQgrb = %e\n", gTQgrb);
      printf("mApostCA = %e, sApostCA = %e\n", mApost[L_CA], sApost[L_CA]);
      printf("mApostCI = %e, sApostCI = %e\n", mApost[L_CI], sApost[L_CI]);
      if (nbClasses == 3)
        printf("mApostCD = %e, sApostCD = %e\n", mApost[L_CD], sApost[L_CD]);
      fflush(stdout); 
    }
    
    /********************/
    /* label sampling   */
    /********************/
    if (sampleLabels) {
      if (wj) {
        // Compute lambdaTilde :
        rl_I_A = compute_ratio_lambda(iv, beta, neighbours, maxNeighbours, labels, 
                                    L_CI, L_CA, nbClasses, mApost, vApost, sApost, 
                                    mean, var);
        rl_I_A_RelCond = compute_ratio_lambda_WithRelCond(rl_I_A, nbVox, sumqj, t1, t2);

        if (nbClasses == 2) {
            lApost2_RelCond[L_CI] = 1. - 1. / (1. + rl_I_A_RelCond);
            if (debug) {
                printf("rl_I_A_RelCond = %e\n", rl_I_A_RelCond);
                fflush(stdout);
            }
        }
//         else if (nbClasses == 3) {
//             rl_D_A = compute_ratio_lambda(iv, beta, neighbours, maxNeighbours, labels, 
//                                       L_CD, L_CA, nbClasses, mApost, vApost, 
//                                       sApost, mean, var);
//             rl_D_A_RelCond = compute_ratio_lambda_WithRelCond(rl_D_A, nbVox, sumqj, t1, t2);
// 	  
//             rl_A_D = compute_ratio_lambda(iv, beta, neighbours, maxNeighbours, labels, 
//                                       L_CA, L_CD, nbClasses, mApost, vApost, 
//                                       sApost, mean, var);
//             rl_A_D_RelCond = compute_ratio_lambda_WithRelCond(rl_A_D, nbVox, sumqj, t1, t2);
// 	  
//             rl_I_D = compute_ratio_lambda(iv, beta, neighbours, maxNeighbours, labels, 
//                                       L_CI, L_CD, nbClasses, mApost, vApost, 
//                                       sApost, mean, var);
//             rl_I_D_RelCond = compute_ratio_lambda_WithRelCond(rl_I_D, nbVox, sumqj, t1, t2);
// 	  
//             rl_A_I = compute_ratio_lambda(iv, beta, neighbours, maxNeighbours, labels, 
//                                       L_CA, L_CI, nbClasses, mApost, vApost, 
//                                       sApost, mean, var);
//             rl_A_I_RelCond = compute_ratio_lambda_WithRelCond(rl_A_I, nbVox, sumqj, t1, t2);
// 	  
//             rl_D_I = compute_ratio_lambda(iv, beta, neighbours, maxNeighbours, labels, 
//                                       L_CD, L_CI, nbClasses, mApost, vApost, 
//                                       sApost, mean, var);
//             rl_D_I_RelCond = compute_ratio_lambda_WithRelCond(rl_D_I, nbVox, sumqj, t1, t2);
//           
//             lApostD_RelCond = 1. / (1. + rl_I_D_RelCond + rl_A_D_RelCond);
//             lApost2_RelCond[L_CI] = 1. - 1. / (1. + rl_I_A_RelCond + rl_D_A_RelCond) - lApostD_RelCond ;
//             lApost2_RelCond[L_CA] = 1. - lApostD_RelCond - lApost2_RelCond[L_CI];
// 
//             if (debug) {
//                 printf("rl_I_A_RelCond = %e\n", rl_I_A_RelCond);
//                 printf("rl_D_A_RelCond = %e\n", rl_D_A_RelCond);
//                 printf("rl_A_D_RelCond = %e\n", rl_A_D_RelCond);
//                 printf("rl_I_D_RelCond = %e\n", rl_I_D_RelCond);
//                 printf("rl_A_I_RelCond = %e\n", rl_A_I_RelCond);
//                 printf("rl_D_I_RelCond = %e\n", rl_D_I_RelCond);
//                 //printf("LApostI = %f\n", 1. / (1. + rl_I_A + rl_D_A));
//                 printf("P apost I with Rel Cond = %e\n", 1. / (1. + rl_A_I_RelCond + rl_D_I_RelCond));
//                 printf("P apost A with Rel Cond = %e\n", 1. / (1. + rl_I_A_RelCond + rl_D_A_RelCond));
//                 printf("P apost D with Rel Cond = %e\n", lApostD_RelCond);
//                 fflush(stdout);
//             }
//         }
      
        lab = nbClasses-1;
        for (ic=0 ; ic<nbClasses-1 ; ic++)
            if (PYA_TVAL(labelsSamples,iv) <= lApost2_RelCond[ic])
                lab = ic;
      
        //lab = (int) (PYA_TVAL(labelsSamples, iv) <= 1. / (1. + rl_I_A));
        PYAI_TVAL(labels,iv) = lab;
        if (debug){
            printf("nrl = %f\n", PYA_TVAL(nrls,iv));
            printf("cumul p CI with Rel Cond = %e\n", lApost2_RelCond[L_CI]);
            if (nbClasses == 3)
                printf("cumul p CA with Rel Cond = %e\n", lApost2_RelCond[L_CA]);
            printf("random = %f !\n", PYA_TVAL(labelsSamples, iv));
            printf("-> label = %d !\n", lab);
            fflush(stdout); 
        }
      }
            
      else {
        // Compute lambdaTilde :
	
        rl_I_A_IRRelCond = compute_ratio_lambda_WithIRRelCond(iv, beta, neighbours, maxNeighbours, labels, 
				      L_CI, L_CA, nbClasses, nbVox, sumqj, t1, t2);

        if (nbClasses == 2) {
            lApost2_IRRelCond[L_CI] = 1. - 1. / (1. + rl_I_A_IRRelCond);
            if (debug) {
                printf("rl_I_A_RelCond = %e\n", rl_I_A_IRRelCond);
                fflush(stdout);
            }
        }
//         else if (nbClasses == 3) {
//             rl_D_A_IRRelCond = compute_ratio_lambda_WithIRRelCond(iv, beta, neighbours, maxNeighbours, labels, 
//                                       L_CD, L_CA, nbClasses, nbVox, sumqj, t1, t2);
// 	  
//             rl_A_D_IRRelCond = compute_ratio_lambda_WithIRRelCond(iv, beta, neighbours, maxNeighbours, labels, 
//                                       L_CA, L_CD, nbClasses, nbVox, sumqj, t1, t2);
// 	  
//             rl_I_D_IRRelCond = compute_ratio_lambda_WithIRRelCond(iv, beta, neighbours, maxNeighbours, labels, 
//                                       L_CI, L_CD, nbClasses, nbVox, sumqj, t1, t2);
// 	  
//             rl_A_I_IRRelCond = compute_ratio_lambda_WithIRRelCond(iv, beta, neighbours, maxNeighbours, labels, 
//                                       L_CA, L_CI, nbClasses, nbVox, sumqj, t1, t2);
// 	  
//             rl_D_I_IRRelCond = compute_ratio_lambda_WithIRRelCond(iv, beta, neighbours, maxNeighbours, labels, 
//                                       L_CD, L_CI, nbClasses, nbVox, sumqj, t1, t2);
//           
//             lApostD_IRRelCond = 1. / (1. + rl_I_D_IRRelCond + rl_A_D_IRRelCond);
//             lApost2_IRRelCond[L_CI] = 1. - 1. / (1. + rl_I_A_IRRelCond + rl_D_A_IRRelCond) - lApostD_IRRelCond ;
//             lApost2_IRRelCond[L_CA] = 1. - lApostD_IRRelCond - lApost2_IRRelCond[L_CI];
// 
//             if (debug) {
//                 printf("rl_I_A_IRRelCond = %e\n", rl_I_A_IRRelCond);
//                 printf("rl_D_A_IRRelCond = %e\n", rl_D_A_IRRelCond);
//                 printf("rl_A_D_IRRelCond = %e\n", rl_A_D_IRRelCond);
//                 printf("rl_I_D_IRRelCond = %e\n", rl_I_D_IRRelCond);
//                 printf("rl_A_I_IRRelCond = %e\n", rl_A_I_IRRelCond);
//                 printf("rl_D_I_IRRelCond = %e\n", rl_D_I_IRRelCond);
//                 //printf("LApostI = %f\n", 1. / (1. + rl_I_A + rl_D_A));
//                 printf("P apost I with IRRel Cond = %e\n", 1. / (1. + rl_A_I_IRRelCond + rl_D_I_IRRelCond));
//                 printf("P apost A with IRRel Cond = %e\n", 1. / (1. + rl_I_A_IRRelCond + rl_D_A_IRRelCond));
//                 printf("P apost D with IRRel Cond = %e\n", lApostD_IRRelCond);
//                 fflush(stdout);
//             }
//         }
       
        lab = nbClasses-1;
        for (ic=0 ; ic<nbClasses-1 ; ic++)
            if (PYA_TVAL(labelsSamples,iv) <= lApost2_IRRelCond[ic])
                lab = ic;
      
        //lab = (int) (PYA_TVAL(labelsSamples, iv) <= 1. / (1. + rl_I_A));
        PYAI_TVAL(labels,iv) = lab;
        if (debug){
            printf("nrl = %f\n", PYA_TVAL(nrls,iv));
            printf("cumul p CI with IRRel Cond = %e\n", lApost2_IRRelCond[L_CI]);
            if (nbClasses == 3)
                printf("cumul p CA with IRRel Cond = %e\n", lApost2_IRRelCond[L_CA]);
            printf("random = %f !\n", PYA_TVAL(labelsSamples, iv));
            printf("-> label = %d !\n", lab);
            fflush(stdout); 
        }
      
      }
      
    }
    
    else // label = true one
      lab = PYAI_TVAL(labels,iv);

    //drawNrl();
    oldNrl = PYA_TVAL(nrls,iv);
    PYA_TVAL(nrls,iv) = sApost[lab] * PYA_TVAL(nrlsSamples,iv) + mApost[lab];
    
    if (debug) {
      printf("nrl = %f !\n", PYA_TVAL(nrls,iv));
      fflush(stdout); 
    }
    //if ( isnan(PYA_TVAL(nrls,iv)) ) {
      //printf("nan in nrls\n");
      //printf("ej:\n");
      //for( n=0 ; n<nMax ; n++)
      //  printf("%f ", ej[n]);
      //printf("\n");
      //exit(1);
    //}
    
    /*********************/
    /* Update y tilde    */
    /*********************/
    deltaNrl = oldNrl - PYA_TVAL(nrls,iv);
    for(n=0 ; n<nMax ; n++){
      //PYA_MVAL(yTilde,n,iv,0) += deltaNrl * PYA_TVAL(varXh,n);
      if (!xhVoxelWise)
        PYA_MVAL(yTilde,n,iv,0) += wj * deltaNrl * PYA_MVAL(varXh,0,n,0);
      else
        PYA_MVAL(yTilde,n,iv,0) += wj * deltaNrl * PYA_MVAL(varXh,iv,n,0);
    }

    if (1 && debug){
      printf("deltaNrl = %f ", deltaNrl);
      printf("updated ytilde:\n");
      for(n=0 ; n<nMax ; n++){
        printf("%f ", PYA_MVAL(yTilde,n,iv,0));
      }
      printf("\n");
      fflush(stdout); 
    }
  }  

  free(ej);
  free(sApost);
  free(mApost);
  free(vApost);
  free(lApost2_RelCond);
  free(lApost2_IRRelCond);

  Py_DECREF(voxOrder);
  Py_DECREF(yTilde);
  Py_DECREF(labels);
  Py_DECREF(varXh);
  Py_DECREF(nrls);
  Py_DECREF(nrlsSamples);
  Py_DECREF(mean);
  Py_DECREF(var);
  Py_DECREF(labelsSamples);
  Py_DECREF(rb);
  Py_DECREF(varXhtQ);
  Py_DECREF(neighbours);
  Py_INCREF(Py_None);
  
  return Py_None; 
}

static PyObject* sampleSmmNrl2(PyObject *self, PyObject *arg)
{  
  // Generic Python objects retrieved when parsing args
  PyObject *oVoxOrder, *oNoiseVars, *oNeighbours, *oyTilde;
  PyObject *oLabels, *oVarXh, *oNrls, *oNrlsSamples, *oLabelsSamples;
  PyObject *o_current_mean_apost, *o_current_var_apost;
  PyObject *oVarXhtQ, *oMean, *oVar, *oGTQg, *oBeta;

  // C numpy array objects which will be obtained from generic python objects
  // with function PyArray_ContiguousFromObject
  PyArrayObject *voxOrder, *neighbours, *yTilde;
  PyArrayObject *current_mean_apost, *current_var_apost;
  PyArrayObject *labels, *varXh, *nrls, *nrlsSamples, *labelsSamples, *rb;
  PyArrayObject *varXhtQ, *mean, *var, *gTQg, *beta;

  int i, iv, n, nMax, nbVox, maxNeighbours, j, d;
  int ic, nbClasses, sampleLabels, it, nbCond, xhVoxelWise;
  npy_float64 varXjhtQjej, gTQgrb, betaj;
  npy_float64 *vApost, *sApost, *mApost;
  npy_float64 deltaNrl, oldNrl;
  npy_float64 rl_I_A, rl_D_A, rl_A_D, rl_I_D, rl_D_I, rl_A_I, lApostD;
  npy_float64 *ej;
  npy_float64 *lApost2;

  npy_int32 lab;

  if (debug) {
      printf("SampleSmmNrl2 ...\n "); 
      fflush(stdout);                
  }
  
  PyArg_ParseTuple(arg, "OOOOOOOOOOOOOOOOiiii", 
                   &oVoxOrder, &oNoiseVars, &oNeighbours,
                   &oyTilde, &oLabels, &oVarXh, &oNrls, &oNrlsSamples, 
                   &oLabelsSamples, &oVarXhtQ, &oGTQg, &oBeta, &oMean, 
                   &oVar, &o_current_mean_apost, &o_current_var_apost,
                   &nbClasses, &sampleLabels, &it, &nbCond);
  if (debug) {
    printf("Arg parsing done\n "); 
    fflush(stdout);                
  }
  
//   printf("PyArg_ParseTuple OK\n");
//   fflush(stdout);
  
  voxOrder = (PyArrayObject*) PyArray_ContiguousFromObject(oVoxOrder, 
                                                           PyArray_INT32,  
                                                           1, 1);

  neighbours = (PyArrayObject*) PyArray_ContiguousFromObject(oNeighbours, 
                                                             PyArray_INT32,  
                                                             2, 2);
  
  yTilde = (PyArrayObject*) PyArray_ContiguousFromObject(oyTilde, 
                                                         PyArray_FLOAT64,  
                                                         2, 2);

  labels = (PyArrayObject*) PyArray_ContiguousFromObject(oLabels, 
                                                         PyArray_INT32,  
                                                         2, 2);

  varXh = (PyArrayObject*) PyArray_ContiguousFromObject(oVarXh, 
                                                        PyArray_FLOAT64,  
                                                        3, 3);

  nrls = (PyArrayObject*) PyArray_ContiguousFromObject(oNrls, PyArray_FLOAT64, 
                                                       2, 2);

  nrlsSamples = (PyArrayObject*) PyArray_ContiguousFromObject(oNrlsSamples, 
                                                              PyArray_FLOAT64,  
                                                              2, 2);

  var = (PyArrayObject*) PyArray_ContiguousFromObject(oVar, PyArray_FLOAT64, 
                                                       2, 2);

  mean = (PyArrayObject*) PyArray_ContiguousFromObject(oMean, PyArray_FLOAT64, 
                                                       2, 2);

  labelsSamples = (PyArrayObject*) PyArray_ContiguousFromObject(oLabelsSamples, 
                                                           PyArray_FLOAT64, 2, 2);
  varXhtQ = (PyArrayObject*) PyArray_ContiguousFromObject(oVarXhtQ,
                                                          PyArray_FLOAT64, 3, 3);
  rb = (PyArrayObject*) PyArray_ContiguousFromObject(oNoiseVars,
                                                     PyArray_FLOAT64, 1, 1);


  beta = (PyArrayObject*) PyArray_ContiguousFromObject(oBeta, PyArray_FLOAT64, 1, 1);
  gTQg = (PyArrayObject*) PyArray_ContiguousFromObject(oGTQg, PyArray_FLOAT64, 1, 1);
  
  current_mean_apost = (PyArrayObject*) PyArray_ContiguousFromObject(o_current_mean_apost, PyArray_FLOAT64, 3, 3);

  current_var_apost = (PyArrayObject*) PyArray_ContiguousFromObject(o_current_var_apost, PyArray_FLOAT64, 3, 3);

  if (debug){
    printf("Wrapping done\n "); 
    fflush(stdout); 
  }

//   printf("Array wrap OK\n");
//   fflush(stdout);

  vApost = malloc(sizeof(npy_float64)*nbClasses);
  sApost = malloc(sizeof(npy_float64)*nbClasses);
  mApost = malloc(sizeof(npy_float64)*nbClasses);

  lApost2 = malloc(sizeof(npy_float64)*(nbClasses-1));

  if (debug){
      printf("Allocation done\n "); 
      fflush(stdout); 
  }
  
  maxNeighbours = neighbours->dimensions[1];
  nbVox = voxOrder->dimensions[0];
  if (debug){
    printf("nbVox=%d, nbClasses=%d, sampleLabels=%d \n", nbVox, nbClasses, 
           sampleLabels);
    fflush(stdout); 
  }


  if (debug){
    int varXh_dimensions = varXh->dimensions[0];
    printf("varXh->dim[0]=%d \n", varXh_dimensions);
    fflush(stdout); 
  }
  nMax = varXh->dimensions[1];
  //nMax = varXh->dimensions[0];
  if (debug){
    printf("nMax=%d \n", nMax);
    fflush(stdout); 
  }
  
  if (debug){
    int rb_dimensions = rb->dimensions[0];
    printf("rb dimension: %d\n", rb_dimensions);
    printf("Xh dimensions:\n");
    for(d=0 ; d<3 ; d++)
    {
      int varXh_dimensions = varXh->dimensions[d];
      printf("%d ", varXh_dimensions);
    }
    printf("\n");
    
    printf("varXhtQ dimensions:\n");
    for(d=0 ; d<3 ; d++)
    {
      int varXhtQ_dimensions = varXhtQ->dimensions[d];
      printf("%d ", varXhtQ_dimensions);
    }
    printf("\n");
  }


  xhVoxelWise = 0;  
  if (varXh->dimensions[0] > 1) {
    xhVoxelWise = 1;
    //printf("varXh voxel wise!\n");
  }
  //else
  //  printf("varXh not voxel wise!\n");

  ej = malloc(sizeof(npy_float64)*nMax);

  if (debug) {
    int labels_dimensions_0 = labels->dimensions[0];
    int labels_dimensions_1 = labels->dimensions[1];
    printf("input labels: %d %d\n", labels_dimensions_0, 
            labels_dimensions_1);
    for(j=0;j<nbCond;j++)
      for(iv=0;iv<nbVox;iv++)
        printf("%d ", PYAI_MVAL(labels,j,iv,0));
    printf("\n");
  }

  for(j=0 ; j<nbCond ; j++) {
    if (debug){  
      printf("Treating condition : %d\n",j);
      printf("Current components :\n");
      printf("mean CI = %f, var CI = %f \n", PYA_MVAL(mean,L_CI,j,0), 
             PYA_MVAL(var,L_CI,j,0));
      printf("mean CA = %f, var CA = %f \n", PYA_MVAL(mean,L_CA,j,0), 
             PYA_MVAL(var,L_CA,j,0));

      if (nbClasses == 3)
        printf("mCD = %f, vCD = %f \n", PYA_MVAL(mean,L_CD,j,0), PYA_MVAL(var,L_CD,j,0));
    
      printf("beta = %f \n", PYA_TVAL(beta,j));
      printf("gTQg = %f \n", PYA_TVAL(gTQg,j));
      fflush(stdout); 
    }

    for(i=0 ; i<nbVox ; i++) {
      iv = PYAI_TVAL(voxOrder,i);
    
      if (debug) {
        printf("it%04d-cond%02d-Vox%03d ... \n", it,j,iv);
        fflush(stdout); 
      }

      /*********************************/
      /* Compute posterior components  */
      /*********************************/
      for(n=0 ; n<nMax ; n++){
        //ej[n] = PYA_MVAL(yTilde,n,iv,0) + PYA_TVAL(nrls,iv) * PYA_TVAL(varXh,n);
        //printf("n=%d, nrl: %f \n",n, PYA_MVAL(nrls,j,iv,0));
        //fflush(stdout); 
        if (!xhVoxelWise)
          ej[n] = PYA_MVAL(yTilde,n,iv,0) + PYA_MVAL(nrls,j,iv,0) * PYA_M2VAL(varXh,0,n,j);
        else
          ej[n] = PYA_MVAL(yTilde,n,iv,0) + PYA_MVAL(nrls,j,iv,0) * PYA_M2VAL(varXh,iv,n,j);
      }

      if (debug){
        printf("ej :\n");
        for( n=0 ; n<nMax ; n++)
          printf("%f ", ej[n]);
        printf("\n");
        
        printf("varXhtQj :\n");
        for( n=0 ; n<nMax ; n++)
          printf("%f ", PYA_M2VAL(varXhtQ,0,j,n));
        printf("\n");
        
        fflush(stdout); 
      }
    
      varXjhtQjej = 0.;
      for(n=0 ; n<nMax ; n++){
        if (!xhVoxelWise)
          varXjhtQjej += PYA_M2VAL(varXhtQ,0,j,n) * ej[n]; 
        else
          varXjhtQjej += PYA_M2VAL(varXhtQ,iv,j,n) * ej[n];
      }
      if (1 && debug){
        printf("varXhtQjej = %f !\n", varXjhtQjej);
        printf("PYA_TVAL(rb,iv=%d) = %f !\n",iv,PYA_TVAL(rb,iv));
        fflush(stdout); 
      }
    
      varXjhtQjej /= PYA_TVAL(rb,iv);

      if (1 && debug){
        printf("varXhtQjej/rb = %f !\n", varXjhtQjej);
        fflush(stdout); 
      }

      if (1 && debug){
        printf("PYA_TVAL(rb,iv=%d) = %f !\n",iv, PYA_TVAL(rb,iv));
        printf("varXhtQjej = %f !\n", varXjhtQjej);
        fflush(stdout); 
      }
    
    

      if (xhVoxelWise){
        PYA_TVAL(gTQg,j) = 0.;
        for (n=0 ; n<nMax ; n++) {
          printf("PYA_M2VAL(varXhtQ,iv=%d,j=%d,n=%d)=%f\n",iv,j,n,PYA_M2VAL(varXhtQ,iv,j,n));
          fflush(stdout); 
          PYA_TVAL(gTQg,j) += PYA_M2VAL(varXhtQ,iv,j,n) * PYA_M2VAL(varXh,iv,n,j);
        }
      }
      gTQgrb = PYA_TVAL(gTQg,j) / PYA_TVAL(rb,iv);

      for (ic=0 ; ic<nbClasses ; ic++) {
        vApost[ic] = 1. / (1. / PYA_MVAL(var,ic,j,0) + gTQgrb);
        sApost[ic] = sqrt(vApost[ic]);
        
        mApost[ic] = vApost[ic] * (PYA_MVAL(mean,ic,j,0) / PYA_MVAL(var,ic,j,0) + \
                                   varXjhtQjej);
        
        PYA_M2VAL(current_mean_apost, ic, j, iv) = mApost[ic];
        if (vApost[ic] < -0.000001) {
          printf("!!neg var!! vApost[%d] = %f !\n", ic, vApost[ic]);
          fflush(stdout);
        }
        if (vApost[ic] <= 0.)
          vApost[ic] = 0.00000001;
        PYA_M2VAL(current_var_apost, ic, j, iv) = vApost[ic];
      }

      if (debug){
        printf("gTQgrb = %e\n", gTQgrb);
        printf("mApostCA = %e, sApostCA = %e\n", mApost[L_CA], sApost[L_CA]);
        printf("mApostCI = %e, sApostCI = %e\n", mApost[L_CI], sApost[L_CI]);
        if (nbClasses == 3)
          printf("mApostCD = %e, sApostCD = %e\n", mApost[L_CD], sApost[L_CD]);
        fflush(stdout); 
      }


      if (debug) {
        int labels_dimensions_0 = labels->dimensions[0];
        int labels_dimensions_1 = labels->dimensions[1];
        printf("After comp a post .... input labels: %d %d\n", 
               labels_dimensions_0, labels_dimensions_1);
        for(d=0;d<nbVox;d++)
          printf("%d ", PYAI_MVAL(labels,j,d,0));
        printf("\n");
      }

    
      /********************/
      /* label sampling   */
      /********************/
      betaj = PYA_TVAL(beta,j);
      //printf("betaj = %e\n", betaj);
      //fflush(stdout); 
      if (sampleLabels) {
        if (debug) {
          printf("label sampling is ON\n");
          fflush(stdout); 
        }
        
        // Compute lambdaTilde :
        rl_I_A = compute_ratio_lambda2(iv, betaj, neighbours, maxNeighbours, 
                                       labels, L_CI, L_CA, nbClasses, mApost, 
                                       vApost, sApost, mean, var, j);
        
        if (nbClasses == 2) {
          lApost2[L_CI] = 1. - 1. / (1. + rl_I_A);
//           printf("lApost2[%d] = %e\n", L_CI, lApost2[L_CI]);
          fflush(stdout);
          if (debug) {
            printf("rl_I_A = %e\n", rl_I_A);
            fflush(stdout);
          }
        }
        else if (nbClasses == 3) {
          rl_D_A = compute_ratio_lambda2(iv, betaj, neighbours, maxNeighbours, labels, 
                                         L_CD, L_CA, nbClasses, mApost, vApost, 
                                         sApost, mean, var,j);
          rl_A_D = compute_ratio_lambda2(iv, betaj, neighbours, maxNeighbours, labels, 
                                         L_CA, L_CD, nbClasses, mApost, vApost, 
                                         sApost, mean, var,j);
          rl_I_D = compute_ratio_lambda2(iv, betaj, neighbours, maxNeighbours, labels, 
                                         L_CI, L_CD, nbClasses, mApost, vApost, 
                                         sApost, mean, var,j);
          rl_A_I = compute_ratio_lambda2(iv, betaj, neighbours, maxNeighbours, labels, 
                                         L_CA, L_CI, nbClasses, mApost, vApost, 
                                         sApost, mean, var,j);
          rl_D_I = compute_ratio_lambda2(iv, betaj, neighbours, maxNeighbours, labels, 
                                         L_CD, L_CI, nbClasses, mApost, vApost, 
                                         sApost, mean, var,j);
          
          lApostD = 1. / (1. + rl_I_D + rl_A_D);
          lApost2[L_CI] = 1. - 1. / (1. + rl_I_A + rl_D_A) - lApostD ;
          lApost2[L_CA] = 1. - lApostD - lApost2[L_CI];

          if (debug) {
            printf("rl_I_A = %e\n", rl_I_A);
            printf("rl_D_A = %e\n", rl_D_A);
            printf("rl_A_D = %e\n", rl_A_D);
            printf("rl_I_D = %e\n", rl_I_D);
            printf("rl_A_I = %e\n", rl_A_I);
            printf("rl_D_I = %e\n", rl_D_I);
            //printf("LApostI = %f\n", 1. / (1. + rl_I_A + rl_D_A));
            printf("P apost I = %e\n", 1. / (1. + rl_A_I + rl_D_I));
            printf("P apost A = %e\n", 1. / (1. + rl_I_A + rl_D_A));
            printf("P apost D = %e\n", lApostD);
            fflush(stdout);
          }
        }

        lab = nbClasses-1;
        for (ic=0 ; ic<nbClasses-1 ; ic++)
          if (PYA_MVAL(labelsSamples,j,iv,0) <= lApost2[ic])
            lab = ic;
      
        //lab = (int) (PYA_TVAL(labelsSamples,j, iv,0) <= 1. / (1. + rl_I_A));
        PYAI_MVAL(labels,j,iv,0) = lab;
        if (debug){
          printf("nrl = %f\n", PYA_MVAL(nrls,j,iv,0));
          printf("cumul p CI = %e\n", lApost2[L_CI]);
          if (nbClasses == 3)
            printf("cumul p CA = %e\n", lApost2[L_CA]);
          printf("random = %f !\n", PYA_MVAL(labelsSamples,j,iv,0));
          printf("-> label = %d !\n", lab);
          fflush(stdout); 
        }
      }
      else { // label = true one
        if (debug){
          printf("label sampling is OFF\n");
          fflush(stdout); 
          printf("j=%d, iv=%d\n", j,iv);
        }
        lab = PYAI_MVAL(labels,j,iv,0);
        if (debug){
          printf("label=%d\n", lab);
          fflush(stdout); 
        }
      }

      //drawNrl();
      oldNrl = PYA_MVAL(nrls,j,iv,0);
      PYA_MVAL(nrls,j,iv,0) = sApost[lab] * PYA_MVAL(nrlsSamples,j,iv,0) + \
                              mApost[lab];
    
      if (debug) {
        printf("nrl = %f !\n", PYA_MVAL(nrls,j,iv,0));
        fflush(stdout); 
      }
      //if ( isnan(PYA_TVAL(nrls,iv)) ) {
      //printf("nan in nrls\n");
      //printf("ej:\n");
      //for( n=0 ; n<nMax ; n++)
      //  printf("%f ", ej[n]);
      //printf("\n");
      //exit(1);
      //}
    
      /*********************/
      /* Update y tilde    */
      /*********************/
      deltaNrl = oldNrl - PYA_MVAL(nrls,j,iv,0);
      for(n=0 ; n<nMax ; n++){
        //PYA_MVAL(yTilde,n,iv,0) += deltaNrl * PYA_TVAL(varXh,n);
        if (!xhVoxelWise)
          PYA_MVAL(yTilde,n,iv,0) += deltaNrl * PYA_M2VAL(varXh,0,n,j);
        else
          PYA_MVAL(yTilde,n,iv,0) += deltaNrl * PYA_M2VAL(varXh,iv,n,j);
      }

      if (0 && debug){
        printf("deltaNrl = %f ", deltaNrl);
        printf("updated ytilde:\n");
        for(n=0 ; n<nMax ; n++){
          printf("%f ", PYA_MVAL(yTilde,n,iv,0));
        }
        printf("\n");
        fflush(stdout); 
      }
    }
  }

  free(ej);
  free(sApost);
  free(mApost);
  free(vApost);
  free(lApost2);

  Py_DECREF(voxOrder);
  Py_DECREF(yTilde);
  Py_DECREF(labels);
  Py_DECREF(varXh);
  Py_DECREF(nrls);
  Py_DECREF(nrlsSamples);
  Py_DECREF(mean);
  Py_DECREF(var);
  Py_DECREF(labelsSamples);
  Py_DECREF(rb);
  Py_DECREF(varXhtQ);
  Py_DECREF(neighbours);
  Py_DECREF(beta);
  Py_DECREF(gTQg);
  Py_DECREF(current_mean_apost);
  Py_DECREF(current_var_apost);
  
  Py_INCREF(Py_None);
  return Py_None; 
}


static PyObject* sampleSmmNrlBar(PyObject *self, PyObject *arg)
{  //printf("badam!");
    //fflush(stdout);
  // Generic Python objects retrieved when parsing args
  PyObject *oVoxOrder, *oNeighbours;
  PyObject *oLabels, *oNrls, *oNrlsSamples, *oLabelsSamples;
  PyObject *o_current_mean_apost, *o_current_var_apost;
  PyObject *oMean, *oVar,  *oBeta, *oVarSess, *oSumNrlSess;

  // C numpy array objects which will be obtained from generic python objects
  // with function PyArray_ContiguousFromObject
  PyArrayObject *voxOrder, *neighbours;
  PyArrayObject *current_mean_apost, *current_var_apost;
  PyArrayObject *labels, *nrls, *nrlsSamples, *labelsSamples;
  PyArrayObject  *mean, *var, *beta;
  PyArrayObject  *var_sess, *sum_nrl_sess;

  int i, iv, n, nMax, nbVox, maxNeighbours, j, d;
  int ic, nbClasses, sampleLabels, it, nbCond, nb_sess;
  npy_float64 betaj;
  npy_float64 *vApost, *sApost, *mApost;
  npy_float64 deltaNrl, oldNrl;
  npy_float64 rl_I_A, rl_D_A, rl_A_D, rl_I_D, rl_D_I, rl_A_I, lApostD;
  npy_float64 *lApost2;

  npy_int32 lab;

  if (debug) {
      printf("SampleSmmNrlBar ...\n "); 
      fflush(stdout);                
  }
  
  PyArg_ParseTuple(arg, "OOOOOOOOOOOOOiiiii", 
                   &oVoxOrder, &oNeighbours,
                   &oLabels, &oNrls, &oNrlsSamples, 
                   &oLabelsSamples, &oBeta, &oMean, 
                   &oVar, &o_current_mean_apost, &o_current_var_apost,
                   &oVarSess, &oSumNrlSess,
                   &nbClasses, &sampleLabels, &it, &nbCond, &nb_sess);
  if (debug) {
    printf("Arg parsing done\n "); 
    fflush(stdout);                
  }
  //printf("bidim "); 
  //fflush(stdout);  
//   printf("PyArg_ParseTuple OK\n");
//   fflush(stdout);
  
  voxOrder = (PyArrayObject*) PyArray_ContiguousFromObject(oVoxOrder, 
                                                           PyArray_INT32,  
                                                           1, 1);
//printf("bidam "); 
//fflush(stdout); 
  neighbours = (PyArrayObject*) PyArray_ContiguousFromObject(oNeighbours, 
                                                             PyArray_INT32,  
                                                             2, 2);
                                                             

  labels = (PyArrayObject*) PyArray_ContiguousFromObject(oLabels, 
                                                         PyArray_INT32,  
                                                         2, 2);
                                                         
  nrls = (PyArrayObject*) PyArray_ContiguousFromObject(oNrls, 
                                                       PyArray_FLOAT64,  
                                                       2, 2);                                                       

  nrlsSamples = (PyArrayObject*) PyArray_ContiguousFromObject(oNrlsSamples, 
                                                              PyArray_FLOAT64,  
                                                              2, 2);

  var = (PyArrayObject*) PyArray_ContiguousFromObject(oVar, PyArray_FLOAT64, 
                                                       2, 2);

  mean = (PyArrayObject*) PyArray_ContiguousFromObject(oMean, PyArray_FLOAT64, 
                                                       2, 2);
                                                       
  sum_nrl_sess = (PyArrayObject*) PyArray_ContiguousFromObject(oSumNrlSess, PyArray_FLOAT64, 
                                                               2, 2);

  labelsSamples = (PyArrayObject*) PyArray_ContiguousFromObject(oLabelsSamples, 
                                                           PyArray_FLOAT64, 2, 2);


  beta = (PyArrayObject*) PyArray_ContiguousFromObject(oBeta, PyArray_FLOAT64, 1, 1);
  
  var_sess = (PyArrayObject*) PyArray_ContiguousFromObject(oVarSess, PyArray_FLOAT64, 1, 1);
  
  current_mean_apost = (PyArrayObject*) PyArray_ContiguousFromObject(o_current_mean_apost, PyArray_FLOAT64, 3, 3);

  current_var_apost = (PyArrayObject*) PyArray_ContiguousFromObject(o_current_var_apost, PyArray_FLOAT64, 3, 3);
  
  if (debug){
    printf("Wrapping done\n "); 
    fflush(stdout); 
    printf("varsess = %f \n", PYA_TVAL(var_sess,0));
    fflush(stdout);
  }
  
//   printf("Array wrap OK\n");
//   fflush(stdout);

  vApost = malloc(sizeof(npy_float64)*nbClasses);
  sApost = malloc(sizeof(npy_float64)*nbClasses);
  mApost = malloc(sizeof(npy_float64)*nbClasses);

  lApost2 = malloc(sizeof(npy_float64)*(nbClasses-1));

  if (debug){
      printf("Allocation done\n "); 
      fflush(stdout); 
  }
  
  maxNeighbours = neighbours->dimensions[1];
  nbVox = voxOrder->dimensions[0];
  if (debug){
    printf("nbVox=%d, nbClasses=%d, sampleLabels=%d \n", nbVox, nbClasses, 
           sampleLabels);
    fflush(stdout); 
  }

  if (debug) {
    int labels_dimensions_0 = labels->dimensions[0];
    int labels_dimensions_1 = labels->dimensions[1];
    printf("input labels: %d %d\n", labels_dimensions_0, 
            labels_dimensions_1);
    for(j=0;j<nbCond;j++)
      for(iv=0;iv<nbVox;iv++)
        printf("%d ", PYAI_MVAL(labels,j,iv,0));
    printf("\n");
  }

  for(j=0 ; j<nbCond ; j++) {
    if (debug){  
      printf("Treating condition : %d\n",j);
      printf("Current components :\n");
      printf("mean CI = %f, var CI = %f \n", PYA_MVAL(mean,L_CI,j,0), 
             PYA_MVAL(var,L_CI,j,0));
      printf("mean CA = %f, var CA = %f \n", PYA_MVAL(mean,L_CA,j,0), 
             PYA_MVAL(var,L_CA,j,0));

      if (nbClasses == 3)
        printf("mCD = %f, vCD = %f \n", PYA_MVAL(mean,L_CD,j,0), PYA_MVAL(var,L_CD,j,0));
    
      printf("beta = %f \n", PYA_TVAL(beta,j));
      fflush(stdout); 
    }

    for(i=0 ; i<nbVox ; i++) {
      iv = PYAI_TVAL(voxOrder,i);
    
      if (debug) {
        printf("it%04d-cond%02d-Vox%03d ... \n", it,j,iv);
        fflush(stdout); 
      }

      /*********************************/
      /* Compute posterior components  */
      /*********************************/
   

      for (ic=0 ; ic<nbClasses ; ic++) {
        vApost[ic] = 1. / (1. / PYA_MVAL(var,ic,j,0) + nb_sess / PYA_TVAL(var_sess,0) );
        sApost[ic] = sqrt(vApost[ic]);
        
        mApost[ic] = vApost[ic] * (PYA_MVAL(mean,ic,j,0) / PYA_MVAL(var,ic,j,0) + \
                     PYA_MVAL(sum_nrl_sess,j,iv,0) / PYA_TVAL(var_sess,0) );
        
        PYA_M2VAL(current_mean_apost, ic, j, iv) = mApost[ic];
        if (vApost[ic] < -0.000001) {
          printf("!!neg var!! vApost[%d] = %f !\n", ic, vApost[ic]);
          fflush(stdout);
        }
        if (vApost[ic] <= 0.)
          vApost[ic] = 0.00000001;
        PYA_M2VAL(current_var_apost, ic, j, iv) = vApost[ic];
      }

      if (debug){
        printf("mApostCA = %e, sApostCA = %e\n", mApost[L_CA], sApost[L_CA]);
        printf("mApostCI = %e, sApostCI = %e\n", mApost[L_CI], sApost[L_CI]);
        if (nbClasses == 3)
          printf("mApostCD = %e, sApostCD = %e\n", mApost[L_CD], sApost[L_CD]);
        fflush(stdout); 
      }


      if (debug) {
        int labels_dimensions_0 = labels->dimensions[0];
        int labels_dimensions_1 = labels->dimensions[1];
        printf("After comp a post .... input labels: %d %d\n", 
               labels_dimensions_0, labels_dimensions_1);
        for(d=0;d<nbVox;d++)
          printf("%d ", PYAI_MVAL(labels,j,d,0));
        printf("\n");
      }

    
      /********************/
      /* label sampling   */
      /********************/
      betaj = PYA_TVAL(beta,j);
      //printf("betaj = %e\n", betaj);
      //fflush(stdout); 
      if (sampleLabels) {
        if (debug) {
          printf("label sampling is ON\n");
          fflush(stdout); 
        }
        
        // Compute lambdaTilde :
        rl_I_A = compute_ratio_lambda2(iv, betaj, neighbours, maxNeighbours, 
                                       labels, L_CI, L_CA, nbClasses, mApost, 
                                       vApost, sApost, mean, var, j);
        
        if (nbClasses == 2) {
          lApost2[L_CI] = 1. - 1. / (1. + rl_I_A);
//           printf("lApost2[%d] = %e\n", L_CI, lApost2[L_CI]);
          fflush(stdout);
          if (debug) {
            printf("rl_I_A = %e\n", rl_I_A);
            fflush(stdout);
          }
        }
        else if (nbClasses == 3) {
          rl_D_A = compute_ratio_lambda2(iv, betaj, neighbours, maxNeighbours, labels, 
                                         L_CD, L_CA, nbClasses, mApost, vApost, 
                                         sApost, mean, var,j);
          rl_A_D = compute_ratio_lambda2(iv, betaj, neighbours, maxNeighbours, labels, 
                                         L_CA, L_CD, nbClasses, mApost, vApost, 
                                         sApost, mean, var,j);
          rl_I_D = compute_ratio_lambda2(iv, betaj, neighbours, maxNeighbours, labels, 
                                         L_CI, L_CD, nbClasses, mApost, vApost, 
                                         sApost, mean, var,j);
          rl_A_I = compute_ratio_lambda2(iv, betaj, neighbours, maxNeighbours, labels, 
                                         L_CA, L_CI, nbClasses, mApost, vApost, 
                                         sApost, mean, var,j);
          rl_D_I = compute_ratio_lambda2(iv, betaj, neighbours, maxNeighbours, labels, 
                                         L_CD, L_CI, nbClasses, mApost, vApost, 
                                         sApost, mean, var,j);
          
          lApostD = 1. / (1. + rl_I_D + rl_A_D);
          lApost2[L_CI] = 1. - 1. / (1. + rl_I_A + rl_D_A) - lApostD ;
          lApost2[L_CA] = 1. - lApostD - lApost2[L_CI];

          if (debug) {
            printf("rl_I_A = %e\n", rl_I_A);
            printf("rl_D_A = %e\n", rl_D_A);
            printf("rl_A_D = %e\n", rl_A_D);
            printf("rl_I_D = %e\n", rl_I_D);
            printf("rl_A_I = %e\n", rl_A_I);
            printf("rl_D_I = %e\n", rl_D_I);
            //printf("LApostI = %f\n", 1. / (1. + rl_I_A + rl_D_A));
            printf("P apost I = %e\n", 1. / (1. + rl_A_I + rl_D_I));
            printf("P apost A = %e\n", 1. / (1. + rl_I_A + rl_D_A));
            printf("P apost D = %e\n", lApostD);
            fflush(stdout);
          }
        }

        lab = nbClasses-1;
        for (ic=0 ; ic<nbClasses-1 ; ic++)
          if (PYA_MVAL(labelsSamples,j,iv,0) <= lApost2[ic])
            lab = ic;
      
        //lab = (int) (PYA_TVAL(labelsSamples,j, iv,0) <= 1. / (1. + rl_I_A));
        PYAI_MVAL(labels,j,iv,0) = lab;
        if (debug){
          printf("cumul p CI = %e\n", lApost2[L_CI]);
          if (nbClasses == 3)
            printf("cumul p CA = %e\n", lApost2[L_CA]);
          printf("random = %f !\n", PYA_MVAL(labelsSamples,j,iv,0));
          printf("-> label = %d !\n", lab);
          fflush(stdout); 
        }
      }
      else { // label = true one
        if (debug){
          printf("label sampling is OFF\n");
          fflush(stdout); 
          printf("j=%d, iv=%d\n", j,iv);
        }
        lab = PYAI_MVAL(labels,j,iv,0);
        if (debug){
          printf("label=%d\n", lab);
          fflush(stdout); 
        }
      }
      

      //drawNrl();
      PYA_MVAL(nrls,j,iv,0) = sApost[lab] * PYA_MVAL(nrlsSamples,j,iv,0) + \
                                mApost[lab];
      
      
      if (debug) {
          printf("nrl = %f !\n", PYA_MVAL(nrls,j,iv,0));
          fflush(stdout); 
      }

      if (j==0 && iv==10 && debug) {
        printf("nrl[0,10] = %f !\n", PYA_MVAL(nrls,0,10,0));
        fflush(stdout); 
      }
      
    }
  }

  free(sApost);
  free(mApost);
  free(vApost);
  free(lApost2);

  Py_DECREF(voxOrder);
  Py_DECREF(labels);
  Py_DECREF(nrls);
  Py_DECREF(nrlsSamples);
  Py_DECREF(mean);
  Py_DECREF(var);
  Py_DECREF(labelsSamples);
  Py_DECREF(neighbours);
  Py_DECREF(sum_nrl_sess);
  Py_DECREF(var_sess);
  
  Py_DECREF(beta);
  Py_DECREF(current_mean_apost);
  Py_DECREF(current_var_apost);
  
  Py_INCREF(Py_None);
  return Py_None; 
}




static PyObject* sampleSmmNrl2WithRelVar(PyObject *self, PyObject *arg)
{  
  // Generic Python objects retrieved when parsing args
  PyObject *oVoxOrder, *oNoiseVars, *oNeighbours, *oyTilde;
  PyObject *oLabels, *oVarXh, *oNrls, *oNrlsSamples, *oLabelsSamples;
  PyObject *o_current_mean_apost, *o_current_var_apost;
  PyObject *oVarXhtQ, *oMean, *oVar, *oGTQg, *oBeta;
  PyObject *oW;
  PyObject *oNbVoxAct;

  // C numpy array objects which will be obtained from generic python objects
  // with function PyArray_ContiguousFromObject
  PyArrayObject *voxOrder, *neighbours, *yTilde;
  PyArrayObject *current_mean_apost, *current_var_apost;
  PyArrayObject *labels, *varXh, *nrls, *nrlsSamples, *labelsSamples, *rb;
  PyArrayObject *varXhtQ, *mean, *var, *gTQg, *beta;
  PyArrayObject *w;
  PyArrayObject *NbVoxAct;

  int i, iv, n, nMax, nbVox, lab, maxNeighbours, j, d;
  int ic, nbClasses, sampleLabels, it, nbCond, xhVoxelWise;
  float t1,t2;
  npy_float64 varXjhtQjej, gTQgrb, betaj;
  npy_float64 *vApost, *sApost, *mApost;
  npy_float64 deltaNrl, oldNrl;
  npy_float64 rl_I_A; //rl_D_A, rl_A_D, rl_I_D, rl_D_I, rl_A_I;
  npy_float64 rl_I_A_RelCond; //rl_D_A_RelCond, rl_A_D_RelCond, rl_I_D_RelCond, rl_D_I_RelCond, rl_A_I_RelCond, lApostD_RelCond;
  npy_float64 rl_I_A_IRRelCond; //rl_D_A_IRRelCond, rl_A_D_IRRelCond, rl_I_D_IRRelCond, rl_D_I_IRRelCond, rl_A_I_IRRelCond, lApostD_IRRelCond;
  npy_float64 *ej;
  npy_float64 *lApost2_RelCond, *lApost2_IRRelCond;
  
  if (debug) {
      printf("SampleSmmNrl2 ...\n "); 
      fflush(stdout);                
  }
  
  PyArg_ParseTuple(arg, "OOOOOOOOOOOOOOOOOffOiiii", 
                   &oVoxOrder, &oNoiseVars, &oNeighbours,
                   &oyTilde, &oLabels, &oVarXh, &oNrls, &oNrlsSamples, 
                   &oLabelsSamples, &oVarXhtQ, &oGTQg, &oBeta, &oMean, 
                   &oVar, &o_current_mean_apost, &o_current_var_apost,
                   &oW, &t1, &t2, &oNbVoxAct, &nbClasses, &sampleLabels, &it, &nbCond);
  
//   printf("tau1 = %f",t1);
//   printf("   , tau2 = %f\n",t2);
  
  if (debug) {
    printf("Arg parsing done\n "); 
    fflush(stdout);                
  }
  
  voxOrder = (PyArrayObject*) PyArray_ContiguousFromObject(oVoxOrder, 
                                                           PyArray_INT32,  
                                                           1, 1);
                                                           
  neighbours = (PyArrayObject*) PyArray_ContiguousFromObject(oNeighbours, 
                                                             PyArray_INT32,  
                                                             2, 2);
  
  yTilde = (PyArrayObject*) PyArray_ContiguousFromObject(oyTilde, 
                                                         PyArray_FLOAT64,  
                                                         2, 2);

  labels = (PyArrayObject*) PyArray_ContiguousFromObject(oLabels, 
                                                         PyArray_INT32,  
                                                         2, 2);

  varXh = (PyArrayObject*) PyArray_ContiguousFromObject(oVarXh, 
                                                        PyArray_FLOAT64,  
                                                        3, 3);

  nrls = (PyArrayObject*) PyArray_ContiguousFromObject(oNrls, PyArray_FLOAT64, 
                                                       2, 2);

  nrlsSamples = (PyArrayObject*) PyArray_ContiguousFromObject(oNrlsSamples, 
                                                              PyArray_FLOAT64,  
                                                              2, 2);

  var = (PyArrayObject*) PyArray_ContiguousFromObject(oVar, PyArray_FLOAT64, 
                                                       2, 2);

  mean = (PyArrayObject*) PyArray_ContiguousFromObject(oMean, PyArray_FLOAT64, 
                                                       2, 2);

  labelsSamples = (PyArrayObject*) PyArray_ContiguousFromObject(oLabelsSamples, 
                                                           PyArray_FLOAT64, 2, 2);

  varXhtQ = (PyArrayObject*) PyArray_ContiguousFromObject(oVarXhtQ,
                                                          PyArray_FLOAT64, 3, 3);

  rb = (PyArrayObject*) PyArray_ContiguousFromObject(oNoiseVars,
                                                     PyArray_FLOAT64, 1, 1);                           
                                                     
  beta = (PyArrayObject*) PyArray_ContiguousFromObject(oBeta, PyArray_FLOAT64, 1, 1);
  gTQg = (PyArrayObject*) PyArray_ContiguousFromObject(oGTQg, PyArray_FLOAT64, 1, 1);
  
  current_mean_apost = (PyArrayObject*) PyArray_ContiguousFromObject(o_current_mean_apost, PyArray_FLOAT64, 3, 3);

  current_var_apost = (PyArrayObject*) PyArray_ContiguousFromObject(o_current_var_apost, PyArray_FLOAT64, 3, 3);
                                                     
  w = (PyArrayObject*) PyArray_ContiguousFromObject(oW, PyArray_INT32, 
                                                       1, 1); 
  
  NbVoxAct = (PyArrayObject*) PyArray_ContiguousFromObject(oNbVoxAct, PyArray_INT32, 
                                                       1, 1);
  
  if (debug){
    printf("Wrapping done\n "); 
    fflush(stdout); 
  }

  vApost = malloc(sizeof(npy_float64)*nbClasses);
  sApost = malloc(sizeof(npy_float64)*nbClasses);
  mApost = malloc(sizeof(npy_float64)*nbClasses);

  lApost2_RelCond = malloc(sizeof(npy_float64)*(nbClasses-1));
  lApost2_IRRelCond = malloc(sizeof(npy_float64)*(nbClasses-1));
  
  if (debug){
      printf("Allocation done\n "); 
      fflush(stdout); 
  }
  
  maxNeighbours = neighbours->dimensions[1];
  nbVox = voxOrder->dimensions[0];
  if (debug){
    printf("nbVox=%d \n", nbVox);
    fflush(stdout); 
  }

  nMax = varXh->dimensions[1];
  //nMax = varXh->dimensions[0];
  if (debug){
    printf("nMax=%d \n", nMax);
    fflush(stdout); 
  }
  
  if (debug){
    int rb_dimensions = rb->dimensions[0];
    printf("rb dimension: %d\n", rb_dimensions);
    printf("Xh dimensions:\n");
    for(d=0 ; d<3 ; d++)
    {
      int varXh_dimensions = varXh->dimensions[d];
      printf("%d ", varXh_dimensions);
    }
    printf("\n");
    
    printf("varXhtQ dimensions:\n");
    for(d=0 ; d<3 ; d++)
    {
      int varXh_dimensions = varXh->dimensions[d];
      printf("%d ", varXh_dimensions);
    }
    printf("\n");
  }


  xhVoxelWise = 0;  
  if (varXh->dimensions[0] > 1) {
    xhVoxelWise = 1;
  }

  ej = malloc(sizeof(npy_float64)*nMax);
  
  for(j=0 ; j<nbCond ; j++) {
    if (debug){  
      printf("Treating condition : %d\n",j);
      printf("Current components :\n");
      printf("mean CI = %f, var CI = %f \n", PYA_MVAL(mean,L_CI,j,0), 
             PYA_MVAL(var,L_CI,j,0));
      printf("mean CA = %f, var CA = %f \n", PYA_MVAL(mean,L_CA,j,0), 
             PYA_MVAL(var,L_CA,j,0));

      if (nbClasses == 3)
        printf("mCD = %f, vCD = %f \n", PYA_MVAL(mean,L_CD,j,0), PYA_MVAL(var,L_CD,j,0));
    
      printf("beta = %f \n", PYA_TVAL(beta,j));
      printf("gTQg = %f \n", PYA_TVAL(gTQg,j));
      fflush(stdout); 
    }
    
    for(i=0 ; i<nbVox ; i++) {
      iv = PYAI_TVAL(voxOrder,i);
      
      if (debug) {
        printf("it%04d-cond%02d-Vox%03d ... \n", it,j,iv);
        fflush(stdout); 
      }

      //Compute sum of ROI labels without label of voxel iv (based on labels computed in the last iterations)
      int sumqj = 0;
//       sumqj = PYAI_TVAL(NbVoxAct, j) - PYAI_MVAL(labels,j,iv,0);
      int mm;
      for(mm=0 ; mm<nbVox ; mm++)
          if(mm!=iv)
              sumqj += PYAI_MVAL(labels,j,mm,0);

      /*********************************/
      /* Compute posterior components  */
      /*********************************/
      
      if(PYAI_TVAL(w,j))
      {
          /* If wj = 1 condition is relevant and we compute the posterior components
          as we did before introducing the relevant variable w */
          
          for(n=0 ; n<nMax ; n++){
              //ej[n] = PYA_MVAL(yTilde,n,iv,0) + PYA_TVAL(nrls,iv) * PYA_TVAL(varXh,n);
              if (!xhVoxelWise)
                  ej[n] = PYA_MVAL(yTilde,n,iv,0) + PYA_MVAL(nrls,j,iv,0) * PYA_M2VAL(varXh,0,n,j);
              else
                  ej[n] = PYA_MVAL(yTilde,n,iv,0) + PYA_MVAL(nrls,j,iv,0) * PYA_M2VAL(varXh,iv,n,j);
          }

          if (0 && debug){
              printf("ej :\n");
              for( n=0 ; n<nMax ; n++)
                  printf("%f ", ej[n]);
              printf("\n");
              fflush(stdout); 
          }
    
          varXjhtQjej = 0.;
          for(n=0 ; n<nMax ; n++){
            if (!xhVoxelWise)
                varXjhtQjej += PYA_M2VAL(varXhtQ,0,j,n) * ej[n]; 
            else
                varXjhtQjej += PYA_M2VAL(varXhtQ,iv,j,n) * ej[n];
          }
          if (1 && debug){
            printf("varXhtQjej = %f !\n", varXjhtQjej);
            printf("PYA_TVAL(rb,iv=%d) = %f !\n",iv,PYA_TVAL(rb,iv));
            fflush(stdout); 
          }
    
          varXjhtQjej /= PYA_TVAL(rb,iv);

          if (1 && debug){
            printf("varXhtQjej/rb = %f !\n", varXjhtQjej);
            fflush(stdout); 
          }

          if (1 && debug){
            printf("PYA_TVAL(rb,i) = %f !\n", PYA_TVAL(rb,iv));
            printf("varXhtQjej = %f !\n", varXjhtQjej);
            fflush(stdout); 
          }

          if (xhVoxelWise){
            PYA_TVAL(gTQg,j) = 0.;
            for (n=0 ; n<nMax ; n++) {
                printf("PYA_M2VAL(varXhtQ,iv=%d,j=%d,n=%d)=%f\n",iv,j,n,PYA_M2VAL(varXhtQ,iv,j,n));
                fflush(stdout); 
                PYA_TVAL(gTQg,j) += PYA_M2VAL(varXhtQ,iv,j,n) * PYA_M2VAL(varXh,iv,n,j);
            }
          }
          gTQgrb = PYA_TVAL(gTQg,j) / PYA_TVAL(rb,iv);

          for (ic=0 ; ic<nbClasses ; ic++) {
            vApost[ic] = 1. / (1. / PYA_MVAL(var,ic,j,0) + gTQgrb);
            sApost[ic] = sqrt(vApost[ic]);
            mApost[ic] = vApost[ic] * (PYA_MVAL(mean,ic,j,0) / PYA_MVAL(var,ic,j,0) + \
                                   varXjhtQjej);

            PYA_M2VAL(current_mean_apost, ic, j, iv) = mApost[ic];
            if (vApost[ic] < -0.000001) {
                printf("!!neg var ParsiMod Rel!! vApost[%d] = %f !\n", ic, vApost[ic]);
                fflush(stdout);
                }
            if (vApost[ic] <= 0.)
                vApost[ic] = 0.00000001;
            PYA_M2VAL(current_var_apost, ic, j, iv) = vApost[ic];
          }   
      }
      
      else
      {
          for (ic=0 ; ic<nbClasses ; ic++) {
            vApost[ic] = PYA_MVAL(var,0,j,0);
            sApost[ic] = sqrt(vApost[ic]);
            mApost[ic] = PYA_MVAL(mean,0,j,0);
            
            PYA_M2VAL(current_mean_apost, ic, j, iv) = mApost[ic];
            if (vApost[ic] < -0.000001) {
                printf("!!neg var ParsiMod Irrel!! vApost[%d] = %f !\n", ic, vApost[ic]);
                fflush(stdout);
            }
            if (vApost[ic] <= 0.)
                vApost[ic] = 0.00000001;
            PYA_M2VAL(current_var_apost, ic, j, iv) = vApost[ic];
            
          }    
      }

      if (debug){
        printf("gTQgrb = %e\n", gTQgrb);
        printf("mApostCA = %e, sApostCA = %e\n", mApost[L_CA], sApost[L_CA]);
        printf("mApostCI = %e, sApostCI = %e\n", mApost[L_CI], sApost[L_CI]);
        if (nbClasses == 3)
          printf("mApostCD = %e, sApostCD = %e\n", mApost[L_CD], sApost[L_CD]);
        fflush(stdout); 
      }
      
      /********************/
      /* label sampling   */
      /********************/
      betaj = PYA_TVAL(beta,j);
      if (sampleLabels) {
            if (PYAI_TVAL(w,j)) {
                
                // Compute lambdaTilde :
                rl_I_A = compute_ratio_lambda2(iv, betaj, neighbours, maxNeighbours, labels, 
                                    L_CI, L_CA, nbClasses, mApost, vApost, sApost, 
                                    mean, var, j);
                
                rl_I_A_RelCond = compute_ratio_lambda_WithRelCond(rl_I_A, nbVox, sumqj, t1, t2);
                
                if (nbClasses == 2) {
                    lApost2_RelCond[L_CI] = 1. - 1. / (1. + rl_I_A_RelCond);
                    if (debug) {
                        printf("rl_I_A_RelCond = %e\n", rl_I_A_RelCond);
                        fflush(stdout);
                    }
                }
      
                lab = nbClasses-1;
                for (ic=0 ; ic<nbClasses-1 ; ic++)
                {
                    if (PYA_MVAL(labelsSamples,j,iv,0) <= lApost2_RelCond[ic])
                        lab = ic;
                }
      
                //lab = (int) (PYA_TVAL(labelsSamples, iv) <= 1. / (1. + rl_I_A));
                PYAI_MVAL(labels,j,iv,0) = lab;
                if (debug){
                    printf("nrl = %f\n", PYA_MVAL(nrls,j,iv,0));
                    printf("cumul p CI with Rel Cond = %e\n", lApost2_RelCond[L_CI]);
                    printf("random = %f !\n", PYA_MVAL(labelsSamples,j,iv,0));
                    printf("-> label = %d !\n", lab);
                    fflush(stdout); 
                }
            }
      
            else {
                
                // Compute lambdaTilde :                
                rl_I_A_IRRelCond = compute_ratio_lambda_WithIRRelCond2(iv, betaj, neighbours, maxNeighbours, labels, 
                                                L_CI, L_CA, nbClasses, nbVox, sumqj, t1, t2, j);

                if (nbClasses == 2) {
                    lApost2_IRRelCond[L_CI] = 1. - 1. / (1. + rl_I_A_IRRelCond);
                    if (debug) {
                        printf("rl_I_A_IRRelCond = %e\n", rl_I_A_IRRelCond);
                        fflush(stdout);
                    }
                }
      
                lab = nbClasses-1;
                for (ic=0 ; ic<nbClasses-1 ; ic++)
                {
                    if (PYA_MVAL(labelsSamples,j,iv,0) <= lApost2_IRRelCond[ic])
                        lab = ic;
                }
      
                //lab = (int) (PYA_MVAL(labelsSamples,j,iv,0) <= 1. / (1. + rl_I_A));
                PYAI_MVAL(labels,j,iv,0) = lab;
                if (debug){
                    printf("nrl = %f\n", PYA_MVAL(nrls,j,iv,0));
                    printf("cumul p CI with IRRel Cond = %e\n", lApost2_IRRelCond[L_CI]);
                    printf("random = %f !\n", PYA_MVAL(labelsSamples,j,iv,0));
                    printf("-> label = %d !\n", lab);
                    fflush(stdout); 
                }  
            }
      }
    
      else // label = true one
        lab = PYAI_MVAL(labels,j,iv,0);

      //drawNrl();
      oldNrl = PYA_MVAL(nrls,j,iv,0);
      PYA_MVAL(nrls,j,iv,0) = sApost[lab] * PYA_MVAL(nrlsSamples,j,iv,0) + \
                        mApost[lab];
    
      if (debug) {
        printf("nrl = %f !\n", PYA_MVAL(nrls,j,iv,0));
        fflush(stdout); 
      }
    
      /*********************/
      /* Update y tilde    */
      /*********************/
      deltaNrl = oldNrl - PYA_MVAL(nrls,j,iv,0);
      for(n=0 ; n<nMax ; n++){
        if (!xhVoxelWise)
          PYA_MVAL(yTilde,n,iv,0) += PYAI_TVAL(w,j) * deltaNrl * PYA_M2VAL(varXh,0,n,j);
        else
          PYA_MVAL(yTilde,n,iv,0) += PYAI_TVAL(w,j) * deltaNrl * PYA_M2VAL(varXh,iv,n,j);
      }

      if (0 && debug){
        printf("deltaNrl = %f ", deltaNrl);
        printf("updated ytilde:\n");
        for(n=0 ; n<nMax ; n++){
          printf("%f ", PYA_MVAL(yTilde,n,iv,0));
        }
        printf("\n");
        fflush(stdout); 
      }
    }
  }
  
  free(ej);
  free(sApost);
  free(mApost);
  free(vApost);
  free(lApost2_RelCond);
  free(lApost2_IRRelCond);

  Py_DECREF(voxOrder);
  Py_DECREF(yTilde);
  Py_DECREF(labels);
  Py_DECREF(varXh);
  Py_DECREF(nrls);
  Py_DECREF(nrlsSamples);
  Py_DECREF(mean);
  Py_DECREF(var);
  Py_DECREF(labelsSamples);
  Py_DECREF(rb);
  Py_DECREF(varXhtQ);
  Py_DECREF(neighbours);
  Py_DECREF(beta);
  Py_DECREF(gTQg);
  Py_DECREF(w);
  Py_DECREF(NbVoxAct);
  Py_DECREF(current_mean_apost);
  Py_DECREF(current_var_apost);
  
  Py_INCREF(Py_None);
  
  return Py_None; 
}

static PyObject* sampleSmmNrl2WithRelVar_NEW(PyObject *self, PyObject *arg)
{  
    // Generic Python objects retrieved when parsing args
    PyObject *oVoxOrder, *oNoiseVars, *oNeighbours, *oyTilde, *oy, *omatPL;
    PyObject *oLabels, *oVarXh, *oNrls, *oNrlsSamples, *oLabelsSamples;
    PyObject *o_current_mean_apost, *o_current_var_apost;
    PyObject *oVarXhtQ, *oMean, *oVar, *oGTQg, *oBeta;
    PyObject *oW;
    PyObject *oNbVoxAct;
    
    // C numpy array objects which will be obtained from generic python objects
    // with function PyArray_ContiguousFromObject
    PyArrayObject *voxOrder, *neighbours, *yTilde, *y, *PL;
    PyArrayObject *current_mean_apost, *current_var_apost;
    PyArrayObject *labels, *varXh, *nrls, *nrlsSamples, *labelsSamples, *rb;
    PyArrayObject *varXhtQ, *mean, *var, *gTQg, *beta;
    PyArrayObject *w;
    PyArrayObject *NbVoxAct;
    
    int i, ii, iv, n, nMax, nbVox, lab, maxNeighbours, j;
    int ic, nbClasses, sampleLabels, it, nbCond, xhVoxelWise, sampleW;
    float t1,t2;
    npy_float64 varXjhtQjej, gTQgrb, betaj;
    npy_float64 *vApost, *sApost, *mApost;
    npy_float64 probaLab1, probaLab0;
    npy_float64 *ej;
    npy_int32 wj;
    npy_float64 deltaNrl, oldNrl;
    
    if (debug) {
        printf("SampleSmmNrl2 ...\n "); 
        fflush(stdout);                
    }
    
    PyArg_ParseTuple(arg, "OOOOOOOOOOOOOOOOOOOffOiiiii", 
                     &oVoxOrder, &oNoiseVars, &oNeighbours,&oyTilde,
                     &oy, &omatPL, &oLabels, &oVarXh, &oNrls, &oNrlsSamples, 
                     &oLabelsSamples, &oVarXhtQ, &oGTQg, &oBeta, &oMean, 
                     &oVar, &o_current_mean_apost, &o_current_var_apost,
                     &oW, &t1, &t2, &oNbVoxAct, &nbClasses, &sampleLabels, &it, &nbCond, &sampleW);
    
    if (debug) {
        printf("Arg parsing done\n "); 
        fflush(stdout);                
    }
    
    voxOrder = (PyArrayObject*) PyArray_ContiguousFromObject(oVoxOrder, 
                                                             PyArray_INT32,  
                                                             1, 1);
    
    neighbours = (PyArrayObject*) PyArray_ContiguousFromObject(oNeighbours, 
                                                               PyArray_INT32,  
                                                               2, 2);
    
    yTilde = (PyArrayObject*) PyArray_ContiguousFromObject(oyTilde, 
                                                           PyArray_FLOAT64,  
                                                           2, 2);
    
    y = (PyArrayObject*) PyArray_ContiguousFromObject(oy, 
                                                      PyArray_FLOAT64,  
                                                      2, 2);
    
    PL = (PyArrayObject*) PyArray_ContiguousFromObject(omatPL, 
                                                       PyArray_FLOAT64,  
                                                       2, 2);
    
    labels = (PyArrayObject*) PyArray_ContiguousFromObject(oLabels, 
                                                           PyArray_INT32,  
                                                           2, 2);
    
    varXh = (PyArrayObject*) PyArray_ContiguousFromObject(oVarXh, 
                                                          PyArray_FLOAT64,  
                                                          3, 3);
    
    nrls = (PyArrayObject*) PyArray_ContiguousFromObject(oNrls, PyArray_FLOAT64, 
                                                         2, 2);
    
    nrlsSamples = (PyArrayObject*) PyArray_ContiguousFromObject(oNrlsSamples, 
                                                                PyArray_FLOAT64,  
                                                                2, 2);
    
    var = (PyArrayObject*) PyArray_ContiguousFromObject(oVar, PyArray_FLOAT64, 
                                                        2, 2);
    
    mean = (PyArrayObject*) PyArray_ContiguousFromObject(oMean, PyArray_FLOAT64, 
                                                         2, 2);
    
    labelsSamples = (PyArrayObject*) PyArray_ContiguousFromObject(oLabelsSamples, 
                                                                  PyArray_FLOAT64, 2, 2);
    
    varXhtQ = (PyArrayObject*) PyArray_ContiguousFromObject(oVarXhtQ,
                                                            PyArray_FLOAT64, 3, 3);
    
    rb = (PyArrayObject*) PyArray_ContiguousFromObject(oNoiseVars,
                                                       PyArray_FLOAT64, 1, 1);                           
    
    beta = (PyArrayObject*) PyArray_ContiguousFromObject(oBeta, PyArray_FLOAT64, 1, 1);
    gTQg = (PyArrayObject*) PyArray_ContiguousFromObject(oGTQg, PyArray_FLOAT64, 1, 1);
    
    current_mean_apost = (PyArrayObject*) PyArray_ContiguousFromObject(o_current_mean_apost, PyArray_FLOAT64, 3, 3);
    
    current_var_apost = (PyArrayObject*) PyArray_ContiguousFromObject(o_current_var_apost, PyArray_FLOAT64, 3, 3);
    
    w = (PyArrayObject*) PyArray_ContiguousFromObject(oW, PyArray_INT32, 
                                                      1, 1); 
    
    NbVoxAct = (PyArrayObject*) PyArray_ContiguousFromObject(oNbVoxAct, PyArray_INT32, 
                                                             1, 1);
    
    vApost = malloc(sizeof(npy_float64)*nbClasses);
    sApost = malloc(sizeof(npy_float64)*nbClasses);
    mApost = malloc(sizeof(npy_float64)*nbClasses);

    maxNeighbours = neighbours->dimensions[1];
    nbVox = voxOrder->dimensions[0];
    
    nMax = varXh->dimensions[1]; // Number of scans

    xhVoxelWise = 0;  
    if (varXh->dimensions[0] > 1) {
        xhVoxelWise = 1;
    }
    
    ej = malloc(sizeof(npy_float64)*nMax);
    
    for(j=0 ; j<nbCond ; j++) {
        
        betaj = PYA_TVAL(beta,j);
        wj = PYAI_TVAL(w,j);
        
        for(i=0 ; i<nbVox ; i++) {
            iv = PYAI_TVAL(voxOrder,i);
            //Compute sum of ROI labels without label of voxel iv (based on labels computed in the last iterations)
            int sumqj = 0;
            int mm;
            for(mm=0 ; mm<nbVox ; mm++)
                if(mm!=iv)
                    sumqj += PYAI_MVAL(labels,j,mm,0);
                
            /*********************************/
            /* Compute posterior components  */
            /*********************************/
            
            for(n=0 ; n<nMax ; n++){
                float StimIndSign = 0.;
                if (!xhVoxelWise)
                {
                    for(ii=0; ii<nbCond ; ii++) {
                        if (ii!=j)
                            StimIndSign += PYAI_TVAL(w,ii) * PYA_MVAL(nrls,ii,iv,0) * PYA_M2VAL(varXh,0,n,ii);
                    }
                    ej[n] = PYA_MVAL(y,n,iv,0) - StimIndSign - PYA_MVAL(PL,n,iv,0);
                }
                
                else
                {
                    for(ii=0; ii<nbCond ; ii++) {
                        if (ii!=j)
                            StimIndSign += PYAI_TVAL(w,ii) * PYA_MVAL(nrls,ii,iv,0) * PYA_M2VAL(varXh,iv,n,ii);
                    }
                    ej[n] = PYA_MVAL(y,n,iv,0) - StimIndSign - PYA_MVAL(PL,n,iv,0);
                }                
            }
            
            varXjhtQjej = 0.;
            for(n=0 ; n<nMax ; n++){
                if (!xhVoxelWise)
                    varXjhtQjej += PYA_M2VAL(varXhtQ,0,j,n) * ej[n] * PYAI_TVAL(w,j); 
                else
                    varXjhtQjej += PYA_M2VAL(varXhtQ,iv,j,n) * ej[n] * PYAI_TVAL(w,j);
            }
            varXjhtQjej /= PYA_TVAL(rb,iv);
            if (xhVoxelWise){
                PYA_TVAL(gTQg,j) = 0.;
                for (n=0 ; n<nMax ; n++) {
                    printf("PYA_M2VAL(varXhtQ,iv=%d,j=%d,n=%d)=%f\n",iv,j,n,PYA_M2VAL(varXhtQ,iv,j,n));
                    fflush(stdout); 
                    PYA_TVAL(gTQg,j) += PYA_M2VAL(varXhtQ,iv,j,n) * PYA_M2VAL(varXh,iv,n,j);
                }
            }
            gTQgrb = PYA_TVAL(gTQg,j) * PYAI_TVAL(w,j) / PYA_TVAL(rb,iv);
            
            for (ic=0 ; ic<nbClasses ; ic++) {
                vApost[ic] = 1. / (1. / PYA_MVAL(var,ic,j,0) + gTQgrb);
                sApost[ic] = sqrt(vApost[ic]);
                mApost[ic] = vApost[ic] * ( PYA_MVAL(mean,ic,j,0)/PYA_MVAL(var,ic,j,0) + varXjhtQjej);
                
                PYA_M2VAL(current_mean_apost, ic, j, iv) = mApost[ic];
                if (vApost[ic] < -0.000001) {
                    printf("!!neg var ParsiMod Rel!! vApost[%d] = %f !\n", ic, vApost[ic]);
                    fflush(stdout);
                }
                if (vApost[ic] <= 0.)
                    vApost[ic] = 0.00000001;
                PYA_M2VAL(current_var_apost, ic, j, iv) = vApost[ic];
            }
                
            /********************/
            /* label sampling   */
            /********************/
            if (sampleLabels) {
                
                probaLab1 = compute_ratio_lambda2_withRelCond_NEW(iv, betaj, neighbours, maxNeighbours, labels,
                                                               L_CI, L_CA, nbClasses, mApost,vApost, sApost,
                                                               mean, var, j, sumqj, t1, t2, nbVox, wj, sampleW);
//                printf("cond = %d,   vox = %d,   proba1 =%f\n",j,iv,probaLab1);
               probaLab0 = 1. - probaLab1;
               lab = 0;
               if(PYA_MVAL(labelsSamples,j,iv,0) <= probaLab1)
                   lab = 1;
                
               PYAI_MVAL(labels,j,iv,0) = lab;
                
            }
                
            else // label = true one
                lab = PYAI_MVAL(labels,j,iv,0);
        
            /********************/
            /* NRLs sampling   */
            /********************/
            oldNrl = PYA_MVAL(nrls,j,iv,0);
            if(wj==1)
                PYA_MVAL(nrls,j,iv,0) = sApost[lab] * PYA_MVAL(nrlsSamples,j,iv,0) + mApost[lab];
            else
                PYA_MVAL(nrls,j,iv,0) = sApost[0] * PYA_MVAL(nrlsSamples,j,iv,0) + mApost[0];            
        }
    }
    
    free(ej);
    free(sApost);
    free(mApost);
    free(vApost);
    
    Py_DECREF(voxOrder);
    Py_DECREF(y);
    Py_DECREF(labels);
    Py_DECREF(varXh);
    Py_DECREF(nrls);
    Py_DECREF(nrlsSamples);
    Py_DECREF(mean);
    Py_DECREF(var);
    Py_DECREF(labelsSamples);
    Py_DECREF(rb);
    Py_DECREF(varXhtQ);
    Py_DECREF(neighbours);
    Py_DECREF(beta);
    Py_DECREF(gTQg);
    Py_DECREF(w);
    Py_DECREF(NbVoxAct);
    Py_DECREF(current_mean_apost);
    Py_DECREF(current_var_apost);
    
    Py_INCREF(Py_None);
    
    return Py_None; 
}

/*
static PyObject* alloc_test(PyObject *self, PyObject *arg) {  
  int size, i;
  

  printf("begin\n");
  fflush(stdout);

  PyArg_ParseTuple(arg, "i", &size);

  printf("after parse\n");
  fflush(stdout);

  //  float array[size];
  
  float *array2;

  array2 = malloc(sizeof(float)*size);

  printf("test\n");
  fflush(stdout);

  for (i=0; i<size; i++) {
    //printf("i=%d\n", i);
    //fflush(stdout);
    array2[i] = i;
  }

  
  free(array2);
  Py_INCREF(Py_None);
  return Py_None; 
}
*/

static PyObject* computeYtilde(PyObject *self, PyObject *arg) {  

  PyObject *oVarXh, *oNrls, *oMBY, *oYTilde, *oSumaXh;

  PyArrayObject *varXh, *nrls, *mby, *yTilde, *sumaXh;

  int i, n, nMax, nbVox, j, nbCond;
  
  PyArg_ParseTuple(arg, "OOOOO", &oVarXh, &oNrls, &oMBY, &oYTilde, &oSumaXh);

  varXh = (PyArrayObject*) PyArray_ContiguousFromObject(oVarXh, 
                                                        PyArray_FLOAT64,  
                                                        2, 2);
  nrls = (PyArrayObject*) PyArray_ContiguousFromObject(oNrls, 
                                                        PyArray_FLOAT64,  
                                                        2, 2);
  mby = (PyArrayObject*) PyArray_ContiguousFromObject(oMBY, 
                                                      PyArray_FLOAT64,  
                                                      2, 2);
  yTilde = (PyArrayObject*) PyArray_ContiguousFromObject(oYTilde, 
                                                         PyArray_FLOAT64,  
                                                         2, 2);
  sumaXh = (PyArrayObject*) PyArray_ContiguousFromObject(oSumaXh, 
                                                         PyArray_FLOAT64,  
                                                         2, 2);

  nbVox = nrls->dimensions[1];
  nbCond = nrls->dimensions[0];
  /* printf("nbvox=%d, nbCond=%d\n", nbVox, nbCond); */
  /* fflush(stdout); */
  /* printf("Varxh : (%d,%d)\n", varXh->dimensions[0], varXh->dimensions[1]); */
  /* printf("sumaXh : (%d,%d)\n", sumaXh->dimensions[0], sumaXh->dimensions[1]); */
  /* fflush(stdout); */
  /* printf("yTilde : (%d,%d)\n", yTilde->dimensions[0], yTilde->dimensions[1]); */
  /* fflush(stdout); */
  nMax = mby->dimensions[0]; // number of scans
  /* printf("nMax=%d\n", nMax); */
  /* fflush(stdout); */
  
  for(i=0 ; i<nbVox ; i++)
    for(n=0 ; n<nMax ; n++) {
      PYA_MVAL(sumaXh,n,i,0) = 0.0;
      for(j=0 ; j<nbCond ; j++)
        PYA_MVAL(sumaXh,n,i,0) += PYA_MVAL(nrls,j,i,0) * PYA_MVAL(varXh,n,j,0);
      PYA_MVAL(yTilde,n,i,0) = PYA_MVAL(mby,n,i,0) - PYA_MVAL(sumaXh,n,i,0);
    }

  Py_DECREF(varXh);
  Py_DECREF(nrls);
  Py_DECREF(mby);
  Py_DECREF(yTilde);
  Py_DECREF(sumaXh);
  Py_INCREF(Py_None);
  return Py_None; 

}


static PyObject* sample_potts(PyObject *self, PyObject *arg) {  

  // Generic Python objects retrieved when parsing args
  PyObject *oVoxOrder, *oNeighbours, *oLabelsSamples, *oExtField;
  PyObject *oBeta, *oLabels;

  // C numpy array objects which will be obtained from generic python objects
  // with function PyArray_ContiguousFromObject
  PyArrayObject *voxOrder, *neighbours, *ext_field;
  PyArrayObject *labelsSamples, *labels, *beta;

  int i, iv, nbVox, maxNeighbours, j;
  int ic, nbClasses, it, nbCond, lab, in, nn;
  npy_float64 betaj, sum_proba, *proba;
  int* sp_corr;

  if (debug) {
      printf("sample_potts ...\n "); 
      fflush(stdout);                
  }
  
  PyArg_ParseTuple(arg, "OOOOOOi", 
                   &oVoxOrder, &oNeighbours, &oExtField, &oBeta,
                   &oLabelsSamples, &oLabels, &it);
  if (debug) {
    printf("Arg parsing done\n "); 
    fflush(stdout);                
  }

  voxOrder = (PyArrayObject*) PyArray_ContiguousFromObject(oVoxOrder, 
                                                           PyArray_INT32,  
                                                           1, 1);

  neighbours = (PyArrayObject*) PyArray_ContiguousFromObject(oNeighbours, 
                                                             PyArray_INT32,  
                                                             2, 2);

  labels = (PyArrayObject*) PyArray_ContiguousFromObject(oLabels, 
                                                         PyArray_INT32,  
                                                         2, 2);

  labelsSamples = (PyArrayObject*) PyArray_ContiguousFromObject(oLabelsSamples, 
                                                                PyArray_FLOAT64, 
                                                                2, 2);

  beta = (PyArrayObject*) PyArray_ContiguousFromObject(oBeta, PyArray_FLOAT64, 
                                                       1, 1);

  ext_field = (PyArrayObject*) PyArray_ContiguousFromObject(oExtField, 
                                                            PyArray_FLOAT64, 
                                                            3, 3);

  if (debug){
    printf("Wrapping done\n "); 
    fflush(stdout); 
  }

  maxNeighbours = neighbours->dimensions[1];
  nbVox = voxOrder->dimensions[0];
  nbCond = ext_field->dimensions[1];
  nbClasses = ext_field->dimensions[0];

  proba = malloc(sizeof(npy_float64) * nbClasses);

  if (debug){
      printf("nbVox=%d, nbClasses=%d,  nbCond=%d\n", nbVox, nbClasses, nbCond);
    fflush(stdout); 
  }
  
  if (debug) {
    int labels_dimensions_0 = labels->dimensions[0];
    int labels_dimensions_1 = labels->dimensions[1];
    printf("input labels: %d %d\n", labels_dimensions_0, 
            labels_dimensions_1);
    for(j=0;j<nbCond;j++)
      for(iv=0;iv<nbVox;iv++)
        printf("%d ", PYAI_MVAL(labels,j,iv,0));
    printf("\n");
  }

  for(j=0 ; j<nbCond ; j++) {
    if (debug){  
      printf("Treating condition : %d\n",j);
      printf("beta = %f \n", PYA_TVAL(beta,j));
      fflush(stdout); 
    }
    for(i=0 ; i<nbVox ; i++) {
      iv = PYAI_TVAL(voxOrder,i);
      betaj = PYA_TVAL(beta,j);

      // compute spatial correlation (count of neighbours with same state)
      sp_corr = calloc(nbClasses, sizeof(int));
      nn = 0;
      while(PYAI_MVAL(neighbours,iv,nn,0) != -1 && nn<maxNeighbours) {
        in = PYAI_MVAL(neighbours,iv,nn,0);
        lab = PYAI_MVAL(labels,j,in,0);
        if (0 && debug){
          printf("in=%d\n",in);
          printf("lab=%d\n",lab);
          printf("%d ", lab ); 
          fflush(stdout); 
        }
        sp_corr[lab]++;
        nn++;
      }

      
      // Compute proba apost
      sum_proba = 0.0;
      for (ic=0 ; ic<nbClasses ; ic++) {
        proba[ic] = exp(betaj * sp_corr[ic] + PYA_M2VAL(ext_field,ic,j,iv));
        sum_proba += proba[ic];
      }

      // pick label
      lab = nbClasses-1;
      for (ic=0 ; ic<nbClasses-1 ; ic++)
        if (PYA_MVAL(labelsSamples,j,iv,0) <= proba[ic]/sum_proba)
          lab = ic;

      PYAI_MVAL(labels,j,iv,0) = lab;
    }
  }

  free(proba);
  free(sp_corr);

  Py_DECREF(voxOrder);
  Py_DECREF(labels);
  Py_DECREF(neighbours);
  Py_DECREF(ext_field);
  Py_DECREF(labelsSamples);
  Py_DECREF(beta);

  
  Py_INCREF(Py_None);
  return Py_None; 
}



static PyObject* asl_compute_y_tilde(PyObject *self, PyObject *arg) {  

  PyObject *oVarXh, *oVarXg, *oMBY, *oYTilde, *oSumaXh, *oSumcXg;
  PyObject *oPrls, *oBrls;

  PyArrayObject *varXh, *varXg, *brls, *prls, *mby, *y_tilde, *sumaXh, *sumcXg;


  int i, n, nMax, nbVox, j, nbCond;
  int compute_bold, compute_perf;

  if (debug) {
    printf("Parse args ...\n");
    fflush(stdout);
  }

  PyArg_ParseTuple(arg, "OOOOOOOOii", &oVarXh, &oVarXg, &oBrls, &oPrls, 
                   &oMBY, &oYTilde, &oSumaXh, &oSumcXg, &compute_bold,
                   &compute_perf);

  if (debug) {
    printf("Arg parsing done!\n");
    fflush(stdout);
  }

  varXh = (PyArrayObject*) PyArray_ContiguousFromObject(oVarXh, 
                                                        PyArray_FLOAT64,  
                                                        2, 2);

  varXg = (PyArrayObject*) PyArray_ContiguousFromObject(oVarXg, 
                                                        PyArray_FLOAT64,  
                                                        2, 2);

  brls = (PyArrayObject*) PyArray_ContiguousFromObject(oBrls, 
                                                       PyArray_FLOAT64,  
                                                       2, 2);

  prls = (PyArrayObject*) PyArray_ContiguousFromObject(oPrls, 
                                                       PyArray_FLOAT64,  
                                                       2, 2);

  mby = (PyArrayObject*) PyArray_ContiguousFromObject(oMBY, 
                                                      PyArray_FLOAT64,  
                                                      2, 2);

  y_tilde = (PyArrayObject*) PyArray_ContiguousFromObject(oYTilde, 
                                                          PyArray_FLOAT64,  
                                                          2, 2);

  sumaXh = (PyArrayObject*) PyArray_ContiguousFromObject(oSumaXh,
                                                         PyArray_FLOAT64,
                                                         2, 2);

  sumcXg = (PyArrayObject*) PyArray_ContiguousFromObject(oSumcXg,
                                                         PyArray_FLOAT64,
                                                         2, 2);


  nbVox = brls->dimensions[1];
  nbCond = brls->dimensions[0];
  nMax = mby->dimensions[0]; // number of scans

  if (debug) {
    printf("nbvox=%d, nbCond=%d\n", nbVox, nbCond);
    fflush(stdout);
    printf("Varxh : (%d,%d)\n", varXh->dimensions[0], varXh->dimensions[1]);
    /* printf("sumaXh : (%d,%d)\n", sumaXh->dimensions[0], sumaXh->dimensions[1]); */
    /* fflush(stdout); */
    printf("y_tilde : (%d,%d)\n", y_tilde->dimensions[0],y_tilde->dimensions[1]);
    fflush(stdout);
    printf("nMax=%d\n", nMax);
    fflush(stdout);
  }  
  for(i=0 ; i<nbVox ; i++)
    for(n=0 ; n<nMax ; n++) {

      if (compute_bold) {
        PYA_MVAL(sumaXh,n,i,0) = 0.0;
        for (j=0 ; j<nbCond ; j++)
          PYA_MVAL(sumaXh,n,i,0) += PYA_MVAL(brls,j,i,0) * \
            PYA_MVAL(varXh,n,j,0);  
      }

      if (compute_perf) {
        PYA_MVAL(sumcXg,n,i,0) = 0.0;
        for (j=0 ; j<nbCond ; j++)
          PYA_MVAL(sumcXg,n,i,0) += PYA_MVAL(prls,j,i,0) * \
            PYA_MVAL(varXg,n,j,0);  
      }

      PYA_MVAL(y_tilde,n,i,0) = PYA_MVAL(mby,n,i,0) - PYA_MVAL(sumaXh,n,i,0) - \
        PYA_MVAL(sumcXg,n,i,0);
    }

  Py_DECREF(varXh);
  Py_DECREF(varXg);
  Py_DECREF(prls);
  Py_DECREF(brls);
  Py_DECREF(mby);
  Py_DECREF(y_tilde);

  Py_DECREF(sumaXh);
  Py_DECREF(sumcXg);

  Py_INCREF(Py_None);
  return Py_None; 

}




static PyObject* computeYtildeWithRelVar(PyObject *self, PyObject *arg) {  

 
  PyObject *oVarXh, *oNrls, *oMBY, *oYTilde, *oSumWaXh, *oW, *oWa;

  PyArrayObject *varXh, *nrls, *mby, *yTilde, *sumWaXh, *w, *Wa;

  int i, n, nMax, nbVox, j, nbCond;

/*  printf("computeYtildeWithRelVar ...\n");  
  fflush(stdout);*/  
  
  PyArg_ParseTuple(arg, "OOOOOOO", &oVarXh, &oNrls, &oMBY, &oYTilde, &oSumWaXh, &oW, &oWa);

/*  printf("PyArg_ParseTuple OK\n");  
  fflush(stdout);*/  
  
  varXh = (PyArrayObject*) PyArray_ContiguousFromObject(oVarXh, 
                                                        PyArray_FLOAT64,  
                                                         2, 2);
  nrls = (PyArrayObject*) PyArray_ContiguousFromObject(oNrls, 
                                                        PyArray_FLOAT64,  
                                                         2, 2);
  mby = (PyArrayObject*) PyArray_ContiguousFromObject(oMBY, 
                                                      PyArray_FLOAT64,  
							 2, 2);
  yTilde = (PyArrayObject*) PyArray_ContiguousFromObject(oYTilde, 
                                                         PyArray_FLOAT64,  
                                                         2, 2);
  sumWaXh = (PyArrayObject*) PyArray_ContiguousFromObject(oSumWaXh, 
                                                         PyArray_FLOAT64,  
                                                         2, 2);							 
  w = (PyArrayObject*) PyArray_ContiguousFromObject(oW, 
						    PyArray_INT32,
							 1, 1);
  Wa = (PyArrayObject*) PyArray_ContiguousFromObject(oWa, 
                                                     PyArray_FLOAT64,  
                                                         2, 2);
 
/*  printf("Array wrap OK\n");  
  fflush(stdout);*/  
  
  nbVox = nrls->dimensions[1];
  nbCond = nrls->dimensions[0];
  nMax = mby->dimensions[0]; // number of scans

  for(i=0 ; i<nbVox ; i++)
    for(n=0 ; n<nMax ; n++) {
      PYA_MVAL(sumWaXh,n,i,0) = 0.0;
      for(j=0 ; j<nbCond ; j++)
      {
	    PYA_MVAL(Wa,j,i,0) = PYA_MVAL(nrls,j,i,0) * PYAI_TVAL(w,j);
	    PYA_MVAL(sumWaXh,n,i,0) += PYA_MVAL(Wa,j,i,0) * PYA_MVAL(varXh,n,j,0);
      }
      PYA_MVAL(yTilde,n,i,0) = PYA_MVAL(mby,n,i,0) - PYA_MVAL(sumWaXh,n,i,0);
    }   

  Py_DECREF(varXh);
  Py_DECREF(nrls);
  Py_DECREF(mby);
  Py_DECREF(yTilde);
  Py_DECREF(sumWaXh);
  Py_DECREF(Wa);
  Py_DECREF(w);
  Py_INCREF(Py_None);
  return Py_None; 

}


  static PyMethodDef methods[] = {
    {"computeStLambda", computeStLambda, METH_VARARGS, "Compute StLambdaS and StLambdaY for HRF sampling"},
    {"computeStLambdaARModel", computeStLambdaARModel, METH_VARARGS, "Compute StLambdaS and StLambdaY for HRF sampling with an AR noise model"},
    {"computePtLambdaARModel", computePtLambdaARModel, METH_VARARGS, "Compute PtLambdaP and PtLambdaY for HDrift sampling with an AR noise model"},
    {"computeStLambdaSparse", computeStLambdaSparse, METH_VARARGS, "Compute StLambdaS and StLambdaY for HRF sampling using sparse calculation"},
    {"computeStLambdaARModelSparse", computeStLambdaARModelSparse, METH_VARARGS, "Compute StLambdaS and StLambdaY for HRF sampling with an AR noise model using sparse calculation"},
    {"quadFormNorm", quadFormNorm, METH_VARARGS, "xT.Q.x matrix multiplication"},
    {"calcCorrEnergies", calcCorrEnergies, METH_VARARGS, "compute all delta of energy for the Ising Field, for all voxels."},
    {"sampleSmmNrl", sampleSmmNrl, METH_VARARGS , "NRL sampling for one condition with a SMM model (serial sampling)"},
    {"sampleSmmNrl2", sampleSmmNrl2, METH_VARARGS , "NRL sampling for all conditions with a SMM model (serial sampling)"},
    {"sampleSmmNrlBar", sampleSmmNrlBar, METH_VARARGS , "NRL sampling for all conditions with a SMM model (serial sampling) and multisession"},
    {"computeYtilde", computeYtilde, METH_VARARGS , "compute Ytilde"},    
    {"sampleSmmNrlWithRelVar", sampleSmmNrlWithRelVar, METH_VARARGS , "NRL sampling for one condition with a SMM model (serial sampling) and a relevant variable w"},
    {"sampleSmmNrl2WithRelVar", sampleSmmNrl2WithRelVar, METH_VARARGS , "NRL sampling for all conditions with a SMM model (serial sampling) and a relevant variable w"},
    {"sampleSmmNrl2WithRelVar_NEW", sampleSmmNrl2WithRelVar_NEW, METH_VARARGS , "NRL sampling for all conditions with a SMM model (serial sampling) and a relevant variable w"},
    {"computeYtildeWithRelVar", computeYtildeWithRelVar, METH_VARARGS , "compute Ytilde with relevant variable w"},
    {"sample_potts", sample_potts, METH_VARARGS , "generate realisations of 2 different Potts fields"},
    {"asl_compute_y_tilde", asl_compute_y_tilde, METH_VARARGS , "Compute Y tilde for ASL JDE model"},
    {NULL, NULL, 0, NULL},
  };

  void	initintensivecalc(void)
  {
    Py_InitModule("intensivecalc", methods);
    import_array();	
  }
