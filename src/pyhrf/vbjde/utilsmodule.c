#include "Python.h"
#include "math.h"
#include <time.h>
#include <numpy/arrayobject.h>

#define eps_FreeEnergy 0.0001
#define eps 0.0001
// #define eps 0.00000001
#define pi 3.1415926
#define GetValueInt(array,x,y) ( *(int*)(array->data + array->strides[0]*x + array->strides[1]*y) )
#define GetValue(array,x,y) ( *(npy_float64*)(array->data + array->strides[0]*x + array->strides[1]*y) )
#define GetValue1D(array,x) ( *(npy_float64*)(array->data + array->strides[0]*x) )
#define GetValue3D(array,x,y,z) ( *(npy_float64*)(array->data + array->strides[0]*x + array->strides[1]*y + array->strides[2]*z) )
#define GetValue3DInt(array,x,y,z) ( *(int*)(array->data + array->strides[0]*x + array->strides[1]*y + array->strides[2]*z) )
#define GetValue4D(array,x,y,z,t) ( *(npy_float64*)(array->data + array->strides[0]*x + array->strides[1]*y + array->strides[2]*z + array->strides[3]*t) )

#define profiling 0

long difftimeval(struct timeval* t1, struct timeval* t2)
{
    long seconds, useconds, mtime;  
    seconds  = t1->tv_sec  - t2->tv_sec;
    useconds = t1->tv_usec - t2->tv_usec;
    
    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    return mtime;
}

npy_float64 normpdf(npy_float64 x, npy_float64 mu, npy_float64 sigma){
  npy_float64 y, u;
  u = (x-mu)/sigma;
  y = exp(-u*u/2) / (sigma * sqrt(2*pi));
  return y;
}

void invMat(npy_float64 *A,int M)  {
  int i,j,k;
  npy_float64 sum,x;
    for (i=1; i < M; i++) A[i] /= A[0]; // normalize row 0
    for (i=1; i < M; i++)  { 
      for (j=i; j < M; j++)  { // do a column of L
        sum = 0.0;
        for (k = 0; k < i; k++)  
            sum += A[j*M+k] * A[k*M+i];
        A[j*M+i] -= sum;
        }
      if (i == M-1) continue;
      for (j=i+1; j < M; j++)  {  // do a row of U
        sum = 0.0;
        for (k = 0; k < i; k++)
            sum += A[i*M+k]*A[k*M+j];
        A[i*M+j] = (A[i*M+j]-sum) / A[i*M+i];
        }
      }
    for ( i = 0; i < M; i++ )  // invert L
      for ( j = i; j < M; j++ )  {
        x = 1.0;
        if ( i != j ) {
          x = 0.0;
          for ( k = i; k < j; k++ ) 
              x -= A[j*M+k]*A[k*M+i];
          }
        A[j*M+i] = x / A[j*M+j];
        }
    for ( i = 0; i < M; i++ )   // invert U
      for ( j = i; j < M; j++ )  {
        if ( i == j ) continue;
        sum = 0.0;
        for ( k = i; k < j; k++ )
            sum += A[k*M+j]*( (i==k) ? 1.0 : A[i*M+k] );
        A[i*M+j] = -sum;
        }
    for ( i = 0; i < M; i++ )   // final inversion
      for ( j = 0; j < M; j++ )  {
        sum = 0.0;
        for ( k = ((i>j)?i:j); k < M; k++ )  
            sum += ((j==k)?1.0:A[j*M+k])*A[k*M+i];
        A[j*M+i] = sum;
        }
    }

void addMat (npy_float64 *A, npy_float64 *B,int M,int N, npy_float64 alpha, npy_float64 beta){
  // B = alpha*A + beta*B          
  int i, j;   
            for (i = 0; i < M; i++) {
                    for (j = 0; j < N; j++) {
                            B[i*N+j] = alpha * A[i*N+j] + beta * B[i*N+j];
                    }
            }
  }

void prodMat(npy_float64 *A,npy_float64 *B,npy_float64 *C,int M, int N, int K,npy_float64 alpha, npy_float64 beta){
  //A: MxK, B:KxN, C:MxN
  // C = alpha A*B + beta C
  int i,j,k;
  npy_float64 tmp;
            for (i = 0; i < M; i++) {
                  for (j = 0; j < N; j++) {
			  C[i*N+j] *= beta;
                          tmp = 0;
                           for (k = 0; k < K; k++) {
                                  tmp += alpha * A[i*K+k] * B[k*N+j];
                           }
                           C[i*N+j] += tmp;
                  }
          }
}
void prodaddMat(npy_float64 *A,npy_float64 *B,npy_float64 *C,int M, int N, int K){
    //A: MxK, B:KxN, C:MxN
    // C = A*B + C
    int i,j,k;
    npy_float64 tmp;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            tmp = 0;
            for (k = 0; k < K; k++) {
                tmp += A[i*K+k] * B[k*N+j];
            }
            C[i*N+j] += tmp;
        }
    }
}
void prodaddMatAlpha(int *A,npy_float64 *B,npy_float64 *C,int M, int N, int K,npy_float64 alpha){
    //A: MxK, B:KxN, C:MxN
    // C = alpha*A*B + C
    int i,j,k;
    npy_float64 tmp;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            tmp = 0;
            for (k = 0; k < K; k++) {
                tmp += alpha * A[i*K+k] * B[k*N+j];
            }
            C[i*N+j] += tmp;
        }
    }
}


void prodMatfast(npy_float64 *A,npy_float64 *B,npy_float64 *C,int M, int N, int K,npy_float64 alpha){
    //A: MxK, B:KxN, C:MxN
    // C = alpha A*B + C
    int i,j,k;
    npy_float64 tmp;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            tmp = 0;
            for (k = 0; k < K; k++) {
                tmp += alpha * A[i*K+k] * B[k*N+j];
            }
            C[i*N+j] = tmp;
        }
    }
}
void prodMat2(npy_float64 *A,npy_float64 *B,npy_float64 *C,int M, int N, int K){
  int i,j,k;
  npy_float64 tmp;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            tmp = A[i*N] * B[j];
            for (k = 1; k < K; k++) {
                tmp += A[i*K+k] * B[k*N+j];
            }
            C[i*N+j] = tmp;
        }
    }
}

void prodMatVect(npy_float64 *A,npy_float64 *B,npy_float64 *C,int M, int N){
    int i,j;
//  npy_float64 tmp;
    for (i = 0; i < M; i++) {
        C[i] = 0;
        for (j = 0; j < N; j++) {
            C[i] += A[i*N+j] * B[j];
        }
    }
}

void prodMat3(int *A,npy_float64 *B,npy_float64 *C,int M, int N, int K){
    //C = A*B where A is a matrix of integers
    int i,j,k;
    npy_float64 tmp;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            tmp = 0;//A[i*N] * B[j];
            for (k = 0; k < K; k++) {
                    tmp += A[i*K+k] * B[k*N+j];  
            }
            C[i*N+j] = tmp;
        }
    }
}

void prodMat4(npy_float64 *A,int *B,npy_float64 *C,int M, int N, int K){
  //C = A*B where B is a matrix of integers
    int i,j,k;
    npy_float64 tmp;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            tmp = 0;//A[i*N] * B[j];
            for (k = 0; k < K; k++) {
                    tmp += A[i*K+k] * B[k*N+j]; 
            }
            C[i*N+j] = tmp;
        }
    }
}

void prodMat4b(PyArrayObject *A,int *B,npy_float64 *C,int M, int N, int K, int vox){
  //C = A*B where B is a matrix of integers
    int i,j,k;
    npy_float64 tmp;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            tmp = 0;//A[i*N] * B[j];
            for (k = 0; k < K; k++) {
                    tmp += GetValue3D(A,i,k,vox) * B[k*N+j]; 
            }
            C[i*N+j] = tmp;
        }
    }
}

npy_float64 prodMatScal(npy_float64 *A,npy_float64 *B,int M){
  int i;
  npy_float64 tmp = 0;
  for (i = 0; i < M; i++) {
      tmp += A[i] * B[i];
  }
  return tmp;
}

npy_float64 traceMat(npy_float64 *A,int M){
  int i;
  npy_float64 trace = 0;
  for (i = 0; i < M; i++) {
      trace += A[i*M+i];
  }
  return trace;
}

void prodMat5(int *A,npy_float64 *B,npy_float64 *C,int M, int N, int K,npy_float64 alpha){
  int i,j,k;
  npy_float64 tmp;
            for (i = 0; i < M; i++) {
                  for (j = 0; j < N; j++) {
                          tmp = 0;//A[i*N] * B[j];
                           for (k = 0; k < K; k++) {
                                  tmp += A[i*K+k] * B[k*N+j] * alpha;				  
                           }
                           C[i*N+j] = tmp;
                  }
          }
}

void transMat(npy_float64 *A,int M,int N){
  int i,j;
  npy_float64 *B;//[N*M];
  B = malloc(sizeof(npy_float64)*M*N);
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      B[i+j*M] = A[j+i*N];
    }
  }
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
     A[i*N+j] = B[i*N+j];
    }
  }
  free(B);
}

void copyMat(npy_float64 *A,npy_float64 *B, int M,int N,npy_float64 alpha)  {
  int i;
  for (i = 0; i < M*N; i++)
      B[i] = alpha * A[i];
    
}
void addMatfast (npy_float64 *A, npy_float64 *B,int M,int N){
    // B = A+B          
    int i, j;   
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            B[i*N+j] += A[i*N+j];
        }
    }
  }

void SubMatfast (npy_float64 *A, npy_float64 *B,int M,int N){
    // B = B - A          
    int i, j;   
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            B[i*N+j] -= A[i*N+j];
        }
    }
  }
  
void SubMatfastVect (npy_float64 *A, npy_float64 *B,int M){
    // B = B - A          
    int i;   
    for (i = 0; i < M; i++) {
        B[i] = B[i] - A[i];
    }
  }

npy_float64 max(npy_float64 A, npy_float64 B){
//   printf("returning the max\n");
  if (A > B)
    return A;
  else
    return B;
}
  
npy_float64 min(npy_float64 A, npy_float64 B){
//   printf("returning the min\n");
  if (A < B)
  {
//     printf("A=%f\n",A);
    return A;
  }
  else
  {
//     printf("B=%f\n",B);
    return B;
  }
}

npy_float64 Compute_Pmfj(npy_float64 Beta_m, npy_float64 SUM_Q_Z_neighbours_qjm, npy_float64 SUM_Q_Z_neighbours_0, npy_float64 SUM_Q_Z_neighbours_1)
{
    npy_float64 numer, denom, Pmfj;
    
    denom = exp(Beta_m * SUM_Q_Z_neighbours_0) + exp(Beta_m * SUM_Q_Z_neighbours_1);
    
    numer = exp(Beta_m * SUM_Q_Z_neighbours_qjm);
    
    Pmfj = numer / denom;
    
    if(Pmfj > 1.0)
        printf("NOT OK : Pmfj is higher than 1 ...\n");
    
    return Pmfj;
}

npy_float64 Compute_Pmfj1(npy_float64 Beta_m, npy_float64 SUM_Q_Z_neighbours_qjm, npy_float64 SUM_Q_Z_neighbours_0, npy_float64 SUM_Q_Z_neighbours_1, npy_float64 Qmj, npy_float64 alpha_0)
{
    npy_float64 numer, denom, Pmfj;
    
    denom = exp( alpha_0 + Beta_m * SUM_Q_Z_neighbours_0) + exp(Beta_m * SUM_Q_Z_neighbours_1);
    
    numer = exp( alpha_0 * (1 - Qmj) + Beta_m * SUM_Q_Z_neighbours_qjm );
    
    Pmfj = numer / denom;
    
    if(Pmfj > 1.0)
        printf("NOT OK : Pmfj1 is higher than 1 ...\n");
    
    return Pmfj;
}

npy_float64 Pzmvoxclass(npy_float64 beta,PyArrayObject *q_Zarray,PyArrayObject *grapharray,int maxNeighbours,int K,int j,int class)
{
    int k,nn;
    npy_float64 tmp2[K],Emax,Sum,Pzmjk;
    Emax = 0;
    for (k=0;k<K;k++){
        tmp2[k] = 0;
        nn = 0;
        while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours) {
            tmp2[k] += beta * GetValue(q_Zarray,k,GetValueInt(grapharray,j,nn));
            nn++;
        }
        if (tmp2[k] > Emax) Emax = tmp2[k];
    }
    Sum = 0;
    for (k=0;k<K;k++){
        Sum += exp(tmp2[k] - Emax);
    }
    Pzmjk = exp(tmp2[class] - Emax) / (Sum + eps);  
    return Pzmjk;
}

static PyObject *UtilsC_maximization_sigma_noiseP(PyObject *self, PyObject *args){
PyObject *Y,*m_A,*m_H,*X,*Sigma_A,*sigma_epsilone,*Sigma_H,*PL;
PyArrayObject *Yarray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Sigma_Aarray,*PLarray;
int j,J,m,m2,d,D,nCond,Nrep,nr;
PyArg_ParseTuple(args, "OOOOOOOOiiii",&PL,&sigma_epsilone,&Sigma_H,&Y,&m_A,&m_H,&Sigma_A,&X,&J,&D,&nCond,&Nrep);
Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
PLarray = (PyArrayObject *) PyArray_ContiguousFromObject(PL,PyArray_FLOAT64,2,2);
m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,2,2);
sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
Yarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Y,PyArray_FLOAT64,2,2);
Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,3,3);
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
npy_float64 *S,*H,*Htilde,*Yj,*PLj,*Sigma_A0,*m_Aj,*tmpnCondnCond,*tmpnCond,*tmpD,*tmp,*tmpDD,*SSigma_H;
int *XX,*XXT;

Sigma_A0 = malloc(sizeof(npy_float64)*nCond*nCond);
H = malloc(sizeof(npy_float64)*D);
m_Aj = malloc(sizeof(npy_float64)*nCond);
Yj = malloc(sizeof(npy_float64)*Nrep);
PLj = malloc(sizeof(npy_float64)*Nrep);
S = malloc(sizeof(npy_float64)*Nrep);
tmpnCondnCond = malloc(sizeof(npy_float64)*nCond*nCond);
tmpnCond = malloc(sizeof(npy_float64)*nCond);
tmpD = malloc(sizeof(npy_float64)*D);
tmpDD = malloc(sizeof(npy_float64)*D*D);
tmp = malloc(sizeof(npy_float64)*D*Nrep);
Htilde = malloc(sizeof(npy_float64)*nCond*nCond);
XX = malloc(sizeof(int)*D*Nrep);
XXT = malloc(sizeof(int)*D*Nrep);
SSigma_H = malloc(sizeof(npy_float64)*D*D);

for (j=0;j<J;j++){ 
  for (d=0;d<D;d++){
      H[d] = GetValue(m_Harray,d,j);
      for (nr=0;nr<D;nr++){
	  SSigma_H[d*D+nr] = GetValue3D(Sigma_Harray,d,nr,j);
      }
    }
    for (m=0;m<nCond;m++){
      for (m2=0;m2<nCond;m2++){
	for (d=0;d<D;d++){
	  for (nr=0;nr<Nrep;nr++){
	      XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
	      XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
	  }
	}
	prodMat4(H,XXT,S,1,Nrep,D);
	prodMat4(S,XX,tmpD,1, D, Nrep);
	Htilde[m*nCond + m2] = prodMatScal(tmpD,H,D);
	prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
	prodMat4(tmp,XX,tmpDD,D, D, Nrep);
	Htilde[m*nCond + m2] += traceMat(tmpDD,D);
      }
    }
  
  
  for (nr=0;nr<Nrep;nr++){
    S[nr] = 0;
    Yj[nr] = GetValue(Yarray,nr,j);
    PLj[nr] = GetValue(PLarray,nr,j);
  }
  for (m=0;m<nCond;m++){
      for (d=0;d<D;d++){
	for (nr=0;nr<Nrep;nr++){
	    XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
	}
      }
      prodaddMatAlpha(XX,H,S,Nrep,1,D,GetValue(m_Aarray,j,m));
      
       for (m2=0;m2<nCond;m2++){
	  Sigma_A0[m*nCond+m2] = GetValue3D(Sigma_Aarray,m,m2,j);
       }
       m_Aj[m] = GetValue(m_Aarray,j,m);
  }
  SubMatfastVect(PLj,Yj,Nrep);
  
  GetValue1D(sigma_epsilonearray,j) = -2*prodMatScal(S,Yj,Nrep);
  prodMat2(Sigma_A0,Htilde,tmpnCondnCond,nCond,nCond,nCond);
  GetValue1D(sigma_epsilonearray,j) += traceMat(tmpnCondnCond,nCond);
  prodMat2(m_Aj,Htilde,tmpnCond,1,nCond,nCond);
  GetValue1D(sigma_epsilonearray,j) += prodMatScal(tmpnCond,m_Aj,nCond);
  GetValue1D(sigma_epsilonearray,j) += prodMatScal(Yj,Yj,Nrep);
  GetValue1D(sigma_epsilonearray,j) /= Nrep;

}

free(Sigma_A0);
free(H);
free(m_Aj);
free(Yj);
free(PLj);
free(S);
free(tmpnCondnCond);
free(tmpnCond);
free(tmpD);
free(tmpDD);
free(tmp);
free(Htilde);
free(XX);
free(XXT);
free(SSigma_H);
// free(SQ);

Py_DECREF(Yarray);
Py_DECREF(m_Aarray);
Py_DECREF(m_Harray);
Py_DECREF(sigma_epsilonearray);
Py_DECREF(Sigma_Harray);
Py_DECREF(Xarray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(PLarray);

Py_INCREF(Py_None);
return Py_None;
}   

static PyObject *UtilsC_maximization_sigma_noise(PyObject *self, PyObject *args){
PyObject *Y,*m_A,*m_H,*X,*Sigma_A,*sigma_epsilone,*Sigma_H,*PL,*Gamma;
PyArrayObject *Yarray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Sigma_Aarray,*PLarray,*Gammaarray;
int j,J,m,m2,d,D,nCond,Nrep,nr;
PyArg_ParseTuple(args, "OOOOOOOOOiiii",&Gamma,&PL,&sigma_epsilone,&Sigma_H,&Y,&m_A,&m_H,&Sigma_A,&X,&J,&D,&nCond,&Nrep);
Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
PLarray = (PyArrayObject *) PyArray_ContiguousFromObject(PL,PyArray_FLOAT64,2,2);
m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
Yarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Y,PyArray_FLOAT64,2,2);
Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);

npy_float64 *S,*H,*Htilde,*Yj,*PLj,*Sigma_A0,*m_Aj,*tmpnCondnCond,*tmpnCond,*tmpD,*tmp,*tmpDD,*SSigma_H,*GGamma;
int *XX,*XXT;

Sigma_A0 = malloc(sizeof(npy_float64)*nCond*nCond);
H = malloc(sizeof(npy_float64)*D);
m_Aj = malloc(sizeof(npy_float64)*nCond);
Yj = malloc(sizeof(npy_float64)*Nrep);
PLj = malloc(sizeof(npy_float64)*Nrep);
S = malloc(sizeof(npy_float64)*Nrep);
tmpnCondnCond = malloc(sizeof(npy_float64)*nCond*nCond);
tmpnCond = malloc(sizeof(npy_float64)*nCond);
tmpD = malloc(sizeof(npy_float64)*D);
tmpDD = malloc(sizeof(npy_float64)*D*D);
tmp = malloc(sizeof(npy_float64)*D*Nrep);
Htilde = malloc(sizeof(npy_float64)*nCond*nCond);
XX = malloc(sizeof(int)*D*Nrep);
XXT = malloc(sizeof(int)*D*Nrep);
SSigma_H = malloc(sizeof(npy_float64)*D*D);
GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);

npy_float64 *SQ, *YGamma;
SQ = malloc(sizeof(npy_float64)*Nrep);
YGamma = malloc(sizeof(npy_float64)*Nrep);

for (d=0;d<Nrep;d++){
  for (nr=0;nr<Nrep;nr++){
    GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
  }
}

for (d=0;d<D;d++){
  H[d] = GetValue1D(m_Harray,d);
  for (nr=0;nr<D;nr++){
      SSigma_H[d*D+nr] = GetValue(Sigma_Harray,d,nr);
  }
}
for (m=0;m<nCond;m++){
  for (m2=0;m2<nCond;m2++){
    for (d=0;d<D;d++){
      for (nr=0;nr<Nrep;nr++){
        XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
        XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
      }
    }
    prodMat4(H,XXT,S,1,Nrep,D);
    prodMat4(S,XX,tmpD,1, D, Nrep);
    Htilde[m*nCond + m2] = prodMatScal(tmpD,H,D);
    prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
    prodMat4(tmp,XX,tmpDD,D, D, Nrep);
    Htilde[m*nCond + m2] += traceMat(tmpDD,D);
  }
}
for (j=0;j<J;j++){
  for (nr=0;nr<Nrep;nr++){
    S[nr] = 0;
    Yj[nr] = GetValue(Yarray,nr,j);
    PLj[nr] = GetValue(PLarray,nr,j);
//      printf("Yj[%d] = %f\n",nr, Yj[nr]);
//      printf("PLj[%d] = %f\n",nr, PLj[nr]);
//       getchar();
  }
  for (m=0;m<nCond;m++){
      for (d=0;d<D;d++){
          for (nr=0;nr<Nrep;nr++){
              XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
          }
      }
      prodaddMatAlpha(XX,H,S,Nrep,1,D,GetValue(m_Aarray,j,m));
      prodMat2(S,GGamma,SQ,1,Nrep,Nrep);
      
      for (m2=0;m2<nCond;m2++){
          Sigma_A0[m*nCond+m2] = GetValue3D(Sigma_Aarray,m,m2,j);
      }
      m_Aj[m] = GetValue(m_Aarray,j,m);
  }
  SubMatfastVect(PLj,Yj,Nrep);
//   for (nr=0;nr<Nrep;nr++){
//       printf("Yj[%d] = %f\n",nr, Yj[nr]);
//       getchar();
//   }
  
  GetValue1D(sigma_epsilonearray,j) = -2*prodMatScal(SQ,Yj,Nrep);
  prodMat2(Sigma_A0,Htilde,tmpnCondnCond,nCond,nCond,nCond);
  GetValue1D(sigma_epsilonearray,j) += traceMat(tmpnCondnCond,nCond);
  prodMat2(m_Aj,Htilde,tmpnCond,1,nCond,nCond);
  GetValue1D(sigma_epsilonearray,j) += prodMatScal(tmpnCond,m_Aj,nCond);
  prodMat2(Yj,GGamma,YGamma,1,Nrep,Nrep);
  GetValue1D(sigma_epsilonearray,j) += prodMatScal(YGamma,Yj,Nrep);
  GetValue1D(sigma_epsilonearray,j) /= Nrep;
//   printf("sigma_epsilonearray[%d] = %f\n",j, GetValue1D(sigma_epsilonearray,j));
//   getchar();
}

free(Sigma_A0);
free(H);
free(m_Aj);
free(Yj);
free(PLj);
free(S);
free(tmpnCondnCond);
free(tmpnCond);
free(tmpD);
free(tmpDD);
free(tmp);
free(Htilde);
free(XX);
free(XXT);
free(SSigma_H);
free(GGamma);
free(SQ);
free(YGamma);

Py_DECREF(Yarray);
Py_DECREF(m_Aarray);
Py_DECREF(m_Harray);
Py_DECREF(sigma_epsilonearray);
Py_DECREF(Sigma_Harray);
Py_DECREF(Xarray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(PLarray);
Py_DECREF(Gammaarray);

Py_INCREF(Py_None);
return Py_None;
}   

static PyObject *UtilsC_maximization_sigma_noise_ParsiMod(PyObject *self, PyObject *args){
PyObject *y_tilde,*m_A,*m_H,*X,*Sigma_A,*sigma_epsilone,*Sigma_H,*Gamma,*p_W;
PyArrayObject *y_tildearray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Sigma_Aarray,*Gammaarray,*p_Warray;
int j,J,m,m2,d,D,nCond,Nrep,nr;
PyArg_ParseTuple(args, "OOOOOOOOOiiii",&p_W,&Gamma,&sigma_epsilone,&Sigma_H,&y_tilde,&m_A,&m_H,&Sigma_A,&X,&J,&D,&nCond,&Nrep);
Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
y_tildearray  = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);

npy_float64 *H,*tmp,*tmpT,*GGamma,*tmpDD,*tmpD,*tmpNrep,*tmpNrep2,*Htilde,*Sigma_A0;
npy_float64 *yy_tilde,*SSigma_H,*m_Aj,*S,*tmpnCondnCond,*tmpnCond;

int *XX,*XXT;

SSigma_H = malloc(sizeof(npy_float64)*D*D);
H = malloc(sizeof(npy_float64)*D);
tmp = malloc(sizeof(npy_float64)*Nrep*D);
tmpD = malloc(sizeof(npy_float64)*D);
tmpDD = malloc(sizeof(npy_float64)*D*D);
tmpNrep = malloc(sizeof(npy_float64)*Nrep);
tmpNrep2 = malloc(sizeof(npy_float64)*Nrep);
tmpT = malloc(sizeof(npy_float64)*Nrep*D);
GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
yy_tilde = malloc(sizeof(npy_float64)*Nrep);
XX = malloc(sizeof(int)*D*Nrep);
XXT = malloc(sizeof(int)*D*Nrep);
Htilde = malloc(sizeof(npy_float64)*nCond*nCond);

Sigma_A0 = malloc(sizeof(npy_float64)*nCond*nCond);
m_Aj = malloc(sizeof(npy_float64)*nCond);
S = malloc(sizeof(npy_float64)*Nrep);
tmpnCondnCond = malloc(sizeof(npy_float64)*nCond*nCond);
tmpnCond = malloc(sizeof(npy_float64)*nCond);


npy_float64 *SQ, *yGamma;
SQ = malloc(sizeof(npy_float64)*Nrep);
yGamma = malloc(sizeof(npy_float64)*Nrep);

for (d=0;d<Nrep;d++){
  for (nr=0;nr<Nrep;nr++){
    GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
  }
}
for (d=0;d<D;d++){
  H[d] = GetValue1D(m_Harray,d);
  for (nr=0;nr<D;nr++){
      SSigma_H[d*D+nr] = GetValue(Sigma_Harray,d,nr);
  }
}

for (m=0;m<nCond;m++){
  for (m2=0;m2<nCond;m2++){
    for (d=0;d<D;d++){
      for (nr=0;nr<Nrep;nr++){
        XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
        XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
      }
    }
    prodMat4(H,XXT,tmpNrep,1,Nrep,D);
    prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
    prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
    Htilde[m*nCond + m2] = prodMatScal(tmpD,H,D);
    prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
    prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
    prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
    Htilde[m*nCond + m2] += traceMat(tmpDD,D);
    Htilde[m*nCond + m2] *= GetValue(p_Warray,m2,1);
    Htilde[m*nCond + m2] *= GetValue(p_Warray,m,1);
  }
}

npy_float64 alpha;

for (j=0;j<J;j++){
  for (nr=0;nr<Nrep;nr++){
      S[nr] = 0;
      yy_tilde[nr] = GetValue(y_tildearray,nr,j);
  }
  for (m=0;m<nCond;m++){
      for (d=0;d<D;d++){
	for (nr=0;nr<Nrep;nr++){
	    XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
	}
      }
      alpha = GetValue(m_Aarray,j,m) *GetValue(p_Warray,m,1);
      prodaddMatAlpha(XX,H,S,Nrep,1,D,alpha);
      prodMat2(S,GGamma,SQ,1,Nrep,Nrep);
      
      for (m2=0;m2<nCond;m2++){
          Sigma_A0[m*nCond+m2] = GetValue3D(Sigma_Aarray,m,m2,j);
      }
       m_Aj[m] = GetValue(m_Aarray,j,m);
  }
  
  GetValue1D(sigma_epsilonearray,j) = -2*prodMatScal(SQ,yy_tilde,Nrep);
  prodMat2(Sigma_A0,Htilde,tmpnCondnCond,nCond,nCond,nCond);
  GetValue1D(sigma_epsilonearray,j) += traceMat(tmpnCondnCond,nCond);
  prodMat2(m_Aj,Htilde,tmpnCond,1,nCond,nCond);
  GetValue1D(sigma_epsilonearray,j) += prodMatScal(tmpnCond,m_Aj,nCond);
  prodMat2(yy_tilde,GGamma,yGamma,1,Nrep,Nrep);
  GetValue1D(sigma_epsilonearray,j) += prodMatScal(yGamma,yy_tilde,Nrep);
  GetValue1D(sigma_epsilonearray,j) /= Nrep;

}

free(Sigma_A0);
free(H);
free(m_Aj);
free(yy_tilde);
free(S);
free(tmpnCondnCond);
free(tmpnCond);
free(tmpD);
free(tmpDD);
free(tmp);
free(Htilde);
free(XX);
free(XXT);
free(SSigma_H);
free(tmpNrep);
free(tmpNrep2);
free(tmpT);
free(GGamma);
free(SQ);
free(yGamma);

Py_DECREF(y_tildearray);
Py_DECREF(m_Aarray);
Py_DECREF(m_Harray);
Py_DECREF(sigma_epsilonearray);
Py_DECREF(Sigma_Harray);
Py_DECREF(Xarray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(p_Warray);
Py_DECREF(Gammaarray);

Py_INCREF(Py_None);
return Py_None;
}

static PyObject *UtilsC_maximization_sigma_noise_ParsiMod_RVM(PyObject *self, PyObject *args){
PyObject *y_tilde,*m_A,*m_H,*X,*Sigma_A,*sigma_epsilone,*Sigma_H,*Gamma,*m_W, *v_W;
PyArrayObject *y_tildearray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Sigma_Aarray,*Gammaarray,*m_Warray,*v_Warray;
int j,J,m,m2,d,D,nCond,Nrep,nr;
PyArg_ParseTuple(args, "OOOOOOOOOOiiii",&m_W,&v_W,&Gamma,&sigma_epsilone,&Sigma_H,&y_tilde,&m_A,&m_H,&Sigma_A,&X,&J,&D,&nCond,&Nrep);
Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
y_tildearray  = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
m_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(m_W,PyArray_FLOAT64,1,1);
v_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(v_W,PyArray_FLOAT64,2,2);

npy_float64 *H,*tmp,*tmpT,*GGamma,*tmpDD,*tmpD,*tmpNrep,*tmpNrep2,*Htilde,*Sigma_A0;
npy_float64 *yy_tilde,*SSigma_H,*m_Aj,*S,*tmpnCondnCond,*tmpnCond;

int *XX,*XXT;

SSigma_H = malloc(sizeof(npy_float64)*D*D);
H = malloc(sizeof(npy_float64)*D);
tmp = malloc(sizeof(npy_float64)*Nrep*D);
tmpD = malloc(sizeof(npy_float64)*D);
tmpDD = malloc(sizeof(npy_float64)*D*D);
tmpNrep = malloc(sizeof(npy_float64)*Nrep);
tmpNrep2 = malloc(sizeof(npy_float64)*Nrep);
tmpT = malloc(sizeof(npy_float64)*Nrep*D);
GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
yy_tilde = malloc(sizeof(npy_float64)*Nrep);
XX = malloc(sizeof(int)*D*Nrep);
XXT = malloc(sizeof(int)*D*Nrep);
Htilde = malloc(sizeof(npy_float64)*nCond*nCond);

Sigma_A0 = malloc(sizeof(npy_float64)*nCond*nCond);
m_Aj = malloc(sizeof(npy_float64)*nCond);
S = malloc(sizeof(npy_float64)*Nrep);
tmpnCondnCond = malloc(sizeof(npy_float64)*nCond*nCond);
tmpnCond = malloc(sizeof(npy_float64)*nCond);


npy_float64 *SQ, *yGamma;
SQ = malloc(sizeof(npy_float64)*Nrep);
yGamma = malloc(sizeof(npy_float64)*Nrep);

for (d=0;d<Nrep;d++){
  for (nr=0;nr<Nrep;nr++){
    GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
  }
}
for (d=0;d<D;d++){
  H[d] = GetValue1D(m_Harray,d);
  for (nr=0;nr<D;nr++){
      SSigma_H[d*D+nr] = GetValue(Sigma_Harray,d,nr);
  }
}

for (m=0;m<nCond;m++){
  for (m2=0;m2<nCond;m2++){
    for (d=0;d<D;d++){
      for (nr=0;nr<Nrep;nr++){
        XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
        XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
      }
    }
    prodMat4(H,XXT,tmpNrep,1,Nrep,D);
    prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
    prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
    Htilde[m*nCond + m2] = prodMatScal(tmpD,H,D);
    prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
    prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
    prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
    Htilde[m*nCond + m2] += traceMat(tmpDD,D);
    Htilde[m*nCond + m2] *= ( GetValue1D(m_Warray,m)*GetValue1D(m_Warray,m2) + GetValue(v_Warray,m,m2) ) ;
  }
}

npy_float64 alpha;

for (j=0;j<J;j++){
  for (nr=0;nr<Nrep;nr++){
      S[nr] = 0;
      yy_tilde[nr] = GetValue(y_tildearray,nr,j);
  }
  for (m=0;m<nCond;m++){
      for (d=0;d<D;d++){
          for (nr=0;nr<Nrep;nr++){
              XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
          }
      }
      alpha = GetValue(m_Aarray,j,m) *GetValue1D(m_Warray,m);
      prodaddMatAlpha(XX,H,S,Nrep,1,D,alpha);
      prodMat2(S,GGamma,SQ,1,Nrep,Nrep);
      
      for (m2=0;m2<nCond;m2++){
          Sigma_A0[m*nCond+m2] = GetValue3D(Sigma_Aarray,m,m2,j);
      }
      m_Aj[m] = GetValue(m_Aarray,j,m);
  }
  
  GetValue1D(sigma_epsilonearray,j) = -2*prodMatScal(SQ,yy_tilde,Nrep);
  prodMat2(Sigma_A0,Htilde,tmpnCondnCond,nCond,nCond,nCond);
  GetValue1D(sigma_epsilonearray,j) += traceMat(tmpnCondnCond,nCond);
  prodMat2(m_Aj,Htilde,tmpnCond,1,nCond,nCond);
  GetValue1D(sigma_epsilonearray,j) += prodMatScal(tmpnCond,m_Aj,nCond);
  prodMat2(yy_tilde,GGamma,yGamma,1,Nrep,Nrep);
  GetValue1D(sigma_epsilonearray,j) += prodMatScal(yGamma,yy_tilde,Nrep);
  GetValue1D(sigma_epsilonearray,j) /= Nrep;

}

free(Sigma_A0);
free(H);
free(m_Aj);
free(yy_tilde);
free(S);
free(tmpnCondnCond);
free(tmpnCond);
free(tmpD);
free(tmpDD);
free(tmp);
free(Htilde);
free(XX);
free(XXT);
free(SSigma_H);
free(tmpNrep);
free(tmpNrep2);
free(tmpT);
free(GGamma);
free(SQ);
free(yGamma);

Py_DECREF(m_Warray);
Py_DECREF(v_Warray);
Py_DECREF(y_tildearray);
Py_DECREF(m_Aarray);
Py_DECREF(m_Harray);
Py_DECREF(sigma_epsilonearray);
Py_DECREF(Sigma_Harray);
Py_DECREF(Xarray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(Gammaarray);

Py_INCREF(Py_None);
return Py_None;
}

static PyObject *UtilsC_expectation_AP(PyObject *self, PyObject *args){
PyObject *Y,*y_tilde,*m_A,*m_H,*X,*Gamma,*Sigma_A,*sigma_epsilone,*Sigma_H,*PL,*mu_MK,*sigma_MK,*q_Z;
PyArrayObject *q_Zarray,*mu_MKarray,*sigma_MKarray,*PLarray,*Yarray,*y_tildearray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Gammaarray,*Sigma_Aarray;
int j,J,m,m2,d,D,nCond,Nrep,nr,K,k;
PyArg_ParseTuple(args, "OOOOOOOOOOOOOiiiii",&q_Z,&mu_MK,&sigma_MK,&PL,&sigma_epsilone,&Gamma,&Sigma_H,&Y,&y_tilde,&m_A,&m_H,&Sigma_A,&X,&J,&D,&nCond,&Nrep,&K);
Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
Sigma_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,3,3);
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
mu_MKarray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_MK,PyArray_FLOAT64,2,2);
sigma_MKarray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_MK,PyArray_FLOAT64,2,2);
PLarray = (PyArrayObject *) PyArray_ContiguousFromObject(PL,PyArray_FLOAT64,2,2);
m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,2,2);
y_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
Yarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Y,PyArray_FLOAT64,2,2);
Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
npy_float64 *H,*tmp,*tmpT,*GGamma,*tmp2,*X_tilde,*Sigma_Aj,*Delta,*tmpDD,*tmpD,*tmpNrep,*tmpNrep2,*tmpnCond,*tmpnCond2;
npy_float64 *yy_tilde,*SSigma_H,*Sigma_A0;
int *XX,*XXT;
tmpnCond = malloc(sizeof(npy_float64)*nCond);
tmpnCond2 = malloc(sizeof(npy_float64)*nCond);
SSigma_H = malloc(sizeof(npy_float64)*D*D);
Delta = malloc(sizeof(npy_float64)*nCond*nCond);
Sigma_A0 = malloc(sizeof(npy_float64)*nCond*nCond);
Sigma_Aj = malloc(sizeof(npy_float64)*nCond*nCond);
X_tilde = malloc(sizeof(npy_float64)*nCond*D);
H = malloc(sizeof(npy_float64)*D);
tmp = malloc(sizeof(npy_float64)*Nrep*D);
tmpD = malloc(sizeof(npy_float64)*D);
tmpDD = malloc(sizeof(npy_float64)*D*D);
tmpNrep = malloc(sizeof(npy_float64)*Nrep);
tmpNrep2 = malloc(sizeof(npy_float64)*Nrep);
tmpT = malloc(sizeof(npy_float64)*Nrep*D);
GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
tmp2 = malloc(sizeof(npy_float64)*Nrep*D);
yy_tilde = malloc(sizeof(npy_float64)*Nrep);
XX = malloc(sizeof(int)*D*Nrep);
XXT = malloc(sizeof(int)*D*Nrep);

for (d=0;d<Nrep;d++){
  for (nr=0;nr<Nrep;nr++){
    GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
  }
}

for (j=0;j<J;j++){
  for (d=0;d<D;d++){
      H[d] = GetValue(m_Harray,d,j);
      for (nr=0;nr<D;nr++){
	  SSigma_H[d*D+nr] = GetValue3D(Sigma_Harray,d,nr,j);
      }
    }
    for (m=0;m<nCond;m++){
      for (m2=0;m2<nCond;m2++){
	for (d=0;d<D;d++){
	  for (nr=0;nr<Nrep;nr++){
	      XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
	      XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
	  }
	}
	prodMat4(H,XXT,tmpNrep,1,Nrep,D);
	prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
	prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
	Sigma_A0[m*nCond + m2] = prodMatScal(tmpD,H,D);
	prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
// 	prodMat4b(Sigma_Harray,XXT,tmp,D,Nrep,D,j);
	prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
	prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
	Sigma_A0[m*nCond + m2] += traceMat(tmpDD,D);
	Delta[m*nCond+m2] = 0;
      }
    }
  
  
  if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
  for (nr=0;nr<Nrep;nr++){
    yy_tilde[nr] = GetValue(y_tildearray,nr,j);
  }
  for (m=0;m<nCond;m++){
      for (d=0;d<D;d++){
        for (nr=0;nr<Nrep;nr++){
            XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
        }
      }
      prodMat2(GGamma,yy_tilde,tmpNrep,Nrep,1,Nrep);
      prodMat4(tmpNrep,XX,tmpD,1,D,Nrep);
      for (d=0;d<D;d++){ 
        X_tilde[m*D+d] = tmpD[d] / GetValue1D(sigma_epsilonearray,j);
      }
      for (m2=0;m2<nCond;m2++){
        Sigma_Aj[m*nCond + m2] = Sigma_A0[m*nCond + m2] / GetValue1D(sigma_epsilonearray,j);
      }
  }
  prodMat2(X_tilde,H,tmpnCond,nCond,1,D);
  for (k=0;k<K;k++){
      for (m=0;m<nCond;m++){
        Delta[m*nCond+m] = GetValue3D(q_Zarray,m,k,j) / (GetValue(sigma_MKarray,m,k) + eps);
        tmpnCond2[m] = GetValue(mu_MKarray,m,k);
      }
      prodaddMat(Delta,tmpnCond2,tmpnCond,nCond,1,nCond);
      addMatfast(Delta,Sigma_Aj,nCond,nCond);
  }
  invMat(Sigma_Aj,nCond);
  for (m=0;m<nCond;m++){
      for (m2=0;m2<nCond;m2++){
        GetValue3D(Sigma_Aarray,m,m2,j) = Sigma_Aj[m*nCond + m2];
      }
  }
  prodMat2(Sigma_Aj,tmpnCond,tmpnCond2,nCond,1,nCond);
  for (m=0;m<nCond;m++){
     GetValue(m_Aarray,j,m) = tmpnCond2[m];
  }
}

free(X_tilde);
free(Sigma_Aj);
free(Sigma_A0);
free(Delta);
free(tmpDD);
free(tmpD);
free(tmpNrep);
free(tmpNrep2);
free(tmpnCond);
free(tmpnCond2);
free(H);
free(tmp);
free(tmpT);
free(GGamma);
free(tmp2);
free(yy_tilde);
free(XX);
free(XXT);

Py_DECREF(q_Zarray);
Py_DECREF(Sigma_Harray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(m_Aarray);
Py_DECREF(m_Harray);
Py_DECREF(Xarray);
Py_DECREF(Yarray);
Py_DECREF(sigma_MKarray);
Py_DECREF(mu_MKarray);
Py_DECREF(PLarray);
Py_DECREF(y_tildearray);
Py_DECREF(Gammaarray);
Py_DECREF(sigma_epsilonearray);
Py_INCREF(Py_None);
return Py_None;
}   



static PyObject *UtilsC_expectation_A(PyObject *self, PyObject *args){
PyObject *Y,*y_tilde,*m_A,*m_H,*X,*Gamma,*Sigma_A,*sigma_epsilone,*Sigma_H,*PL,*mu_MK,*sigma_MK,*q_Z;
PyArrayObject *q_Zarray,*mu_MKarray,*sigma_MKarray,*PLarray,*Yarray,*y_tildearray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Gammaarray,*Sigma_Aarray;
int j,J,m,m2,d,D,nCond,Nrep,nr,K,k;
PyArg_ParseTuple(args, "OOOOOOOOOOOOOiiiii",&q_Z,&mu_MK,&sigma_MK,&PL,&sigma_epsilone,&Gamma,&Sigma_H,&Y,&y_tilde,&m_A,&m_H,&Sigma_A,&X,&J,&D,&nCond,&Nrep,&K);
Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
Sigma_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
mu_MKarray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_MK,PyArray_FLOAT64,2,2);
sigma_MKarray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_MK,PyArray_FLOAT64,2,2);
PLarray = (PyArrayObject *) PyArray_ContiguousFromObject(PL,PyArray_FLOAT64,2,2);
m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
y_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
Yarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Y,PyArray_FLOAT64,2,2);
Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
npy_float64 *H,*tmp,*tmpT,*GGamma,*tmp2,*X_tilde,*Sigma_Aj,*Delta,*tmpDD,*tmpD,*tmpNrep,*tmpNrep2,*tmpnCond,*tmpnCond2;
npy_float64 *yy_tilde,*SSigma_H,*Sigma_A0;
int *XX,*XXT;
tmpnCond = malloc(sizeof(npy_float64)*nCond);
tmpnCond2 = malloc(sizeof(npy_float64)*nCond);
SSigma_H = malloc(sizeof(npy_float64)*D*D);
Delta = malloc(sizeof(npy_float64)*nCond*nCond);
Sigma_A0 = malloc(sizeof(npy_float64)*nCond*nCond);
Sigma_Aj = malloc(sizeof(npy_float64)*nCond*nCond);
X_tilde = malloc(sizeof(npy_float64)*nCond*D);
H = malloc(sizeof(npy_float64)*D);
tmp = malloc(sizeof(npy_float64)*Nrep*D);
tmpD = malloc(sizeof(npy_float64)*D);
tmpDD = malloc(sizeof(npy_float64)*D*D);
tmpNrep = malloc(sizeof(npy_float64)*Nrep);
tmpNrep2 = malloc(sizeof(npy_float64)*Nrep);
tmpT = malloc(sizeof(npy_float64)*Nrep*D);
GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
tmp2 = malloc(sizeof(npy_float64)*Nrep*D);
yy_tilde = malloc(sizeof(npy_float64)*Nrep);
XX = malloc(sizeof(int)*D*Nrep);
XXT = malloc(sizeof(int)*D*Nrep);

for (d=0;d<Nrep;d++){
  for (nr=0;nr<Nrep;nr++){
    GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
  }
}
for (d=0;d<D;d++){
  H[d] = GetValue1D(m_Harray,d);
  for (nr=0;nr<D;nr++){
      SSigma_H[d*D+nr] = GetValue(Sigma_Harray,d,nr);
  }
}
for (m=0;m<nCond;m++){
  for (m2=0;m2<nCond;m2++){
    for (d=0;d<D;d++){
      for (nr=0;nr<Nrep;nr++){
        XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
        XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
      }
    }
    prodMat4(H,XXT,tmpNrep,1,Nrep,D);
    prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
    prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
    Sigma_A0[m*nCond + m2] = prodMatScal(tmpD,H,D);
    prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
    prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
    prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
    Sigma_A0[m*nCond + m2] += traceMat(tmpDD,D);
    Delta[m*nCond+m2] = 0;
  }
}
for (j=0;j<J;j++){
  if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
  for (nr=0;nr<Nrep;nr++){
    yy_tilde[nr] = GetValue(y_tildearray,nr,j);
  }
  for (m=0;m<nCond;m++){
      for (d=0;d<D;d++){
        for (nr=0;nr<Nrep;nr++){
            XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
        }
      }
      prodMat2(GGamma,yy_tilde,tmpNrep,Nrep,1,Nrep);
      prodMat4(tmpNrep,XX,tmpD,1,D,Nrep);
      for (d=0;d<D;d++){ 
        X_tilde[m*D+d] = tmpD[d] / GetValue1D(sigma_epsilonearray,j);
      }
      for (m2=0;m2<nCond;m2++){
        Sigma_Aj[m*nCond + m2] = Sigma_A0[m*nCond + m2] / GetValue1D(sigma_epsilonearray,j);
      }
  }
  prodMat2(X_tilde,H,tmpnCond,nCond,1,D);
  for (k=0;k<K;k++){
      for (m=0;m<nCond;m++){
//         Delta[m*nCond+m] = GetValue3D(q_Zarray,m,k,j) / (GetValue(sigma_MKarray,m,k) + eps);
        Delta[m*nCond+m] = GetValue3D(q_Zarray,m,k,j) / GetValue(sigma_MKarray,m,k);
        tmpnCond2[m] = GetValue(mu_MKarray,m,k);
      }
      prodaddMat(Delta,tmpnCond2,tmpnCond,nCond,1,nCond);
      addMatfast(Delta,Sigma_Aj,nCond,nCond);
  }
  invMat(Sigma_Aj,nCond);
  for (m=0;m<nCond;m++){
      for (m2=0;m2<nCond;m2++){
        GetValue3D(Sigma_Aarray,m,m2,j) = Sigma_Aj[m*nCond + m2];
      }
  }
  prodMat2(Sigma_Aj,tmpnCond,tmpnCond2,nCond,1,nCond);
  for (m=0;m<nCond;m++){
     GetValue(m_Aarray,j,m) = tmpnCond2[m];
  }
}

free(X_tilde);
free(Sigma_Aj);
free(Sigma_A0);
free(Delta);
free(tmpDD);
free(tmpD);
free(tmpNrep);
free(tmpNrep2);
free(tmpnCond);
free(tmpnCond2);
free(H);
free(tmp);
free(tmpT);
free(GGamma);
free(tmp2);
free(yy_tilde);
free(XX);
free(XXT);

Py_DECREF(q_Zarray);
Py_DECREF(Sigma_Harray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(m_Aarray);
Py_DECREF(m_Harray);
Py_DECREF(Xarray);
Py_DECREF(Yarray);
Py_DECREF(sigma_MKarray);
Py_DECREF(mu_MKarray);
Py_DECREF(PLarray);
Py_DECREF(y_tildearray);
Py_DECREF(Gammaarray);
Py_DECREF(sigma_epsilonearray);
Py_INCREF(Py_None);
return Py_None;
}   

static PyObject *UtilsC_expectation_A_ParsiMod(PyObject *self, PyObject *args){
PyObject *y_tilde,*m_A,*m_H,*X,*Gamma,*Sigma_A,*sigma_epsilone,*Sigma_H,*mu_MK,*sigma_MK,*q_Z,*p_W;
PyArrayObject *p_Warray,*q_Zarray,*mu_MKarray,*sigma_MKarray,*y_tildearray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Gammaarray,*Sigma_Aarray;
int j,J,m,m2,d,D,nCond,Nrep,nr,K;
PyArg_ParseTuple(args, "OOOOOOOOOOOOiiiii",&p_W,&q_Z,&mu_MK,&sigma_MK,&sigma_epsilone,&Gamma,&Sigma_H,&y_tilde,&m_A,&m_H,&Sigma_A,&X,&J,&D,&nCond,&Nrep,&K);
Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
Sigma_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
mu_MKarray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_MK,PyArray_FLOAT64,2,2);
sigma_MKarray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_MK,PyArray_FLOAT64,2,2);

m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
y_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);

npy_float64 *H,*tmp,*tmpT,*GGamma,*tmp2,*X_tilde,*Sigma_Aj,*Delta0,*Delta1,*tmpDD,*tmpD,*tmpNrep,*tmpNrep2,*tmpnCond,*tmpnCond2,*tmpnCond3;
npy_float64 *yy_tilde,*SSigma_H,*Sigma_A0;
int *XX,*XXT;
tmpnCond = malloc(sizeof(npy_float64)*nCond);
tmpnCond2 = malloc(sizeof(npy_float64)*nCond);
tmpnCond3 = malloc(sizeof(npy_float64)*nCond);
SSigma_H = malloc(sizeof(npy_float64)*D*D);
Delta0 = malloc(sizeof(npy_float64)*nCond*nCond);
Delta1 = malloc(sizeof(npy_float64)*nCond*nCond);
Sigma_A0 = malloc(sizeof(npy_float64)*nCond*nCond);
Sigma_Aj = malloc(sizeof(npy_float64)*nCond*nCond);
X_tilde = malloc(sizeof(npy_float64)*nCond*D);
H = malloc(sizeof(npy_float64)*D);
tmp = malloc(sizeof(npy_float64)*Nrep*D);
tmpD = malloc(sizeof(npy_float64)*D);
tmpDD = malloc(sizeof(npy_float64)*D*D);
tmpNrep = malloc(sizeof(npy_float64)*Nrep);
tmpNrep2 = malloc(sizeof(npy_float64)*Nrep);
tmpT = malloc(sizeof(npy_float64)*Nrep*D);
GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
tmp2 = malloc(sizeof(npy_float64)*Nrep*D);
yy_tilde = malloc(sizeof(npy_float64)*Nrep);
XX = malloc(sizeof(int)*D*Nrep);
XXT = malloc(sizeof(int)*D*Nrep);

for (d=0;d<Nrep;d++){
  for (nr=0;nr<Nrep;nr++){
    GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
  }
}
for (d=0;d<D;d++){
  H[d] = GetValue1D(m_Harray,d);
  for (nr=0;nr<D;nr++){
      SSigma_H[d*D+nr] = GetValue(Sigma_Harray,d,nr);
  }
}
for (m=0;m<nCond;m++){
  for (m2=0;m2<nCond;m2++){
    for (d=0;d<D;d++){
      for (nr=0;nr<Nrep;nr++){
        XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
        XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
      }
    }
    prodMat4(H,XXT,tmpNrep,1,Nrep,D);
    prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
    prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
    Sigma_A0[m*nCond + m2] = prodMatScal(tmpD,H,D);
    prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
    prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
    prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
    Sigma_A0[m*nCond + m2] += traceMat(tmpDD,D);
    Delta0[m*nCond+m2] = 0;
    Delta1[m*nCond+m2] = 0;
  }
}

for (j=0;j<J;j++){
  if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
  for (nr=0;nr<Nrep;nr++){
    yy_tilde[nr] = GetValue(y_tildearray,nr,j);
  }
  
  for (m=0;m<nCond;m++){
      for (d=0;d<D;d++){
        for (nr=0;nr<Nrep;nr++){
            XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
        }
      }
      prodMat2(GGamma,yy_tilde,tmpNrep,Nrep,1,Nrep);
      prodMat4(tmpNrep,XX,tmpD,1,D,Nrep);
      for (d=0;d<D;d++){ 
        X_tilde[m*D+d] = tmpD[d] * GetValue(p_Warray,m,1) / GetValue1D(sigma_epsilonearray,j);
      }
      for (m2=0;m2<nCond;m2++){
        Sigma_Aj[m*nCond + m2] = (Sigma_A0[m*nCond + m2] / GetValue1D(sigma_epsilonearray,j)) * GetValue(p_Warray,m,1) * GetValue(p_Warray,m2,1);	
      }
  }
  prodMat2(X_tilde,H,tmpnCond,nCond,1,D);
  
  for (m=0;m<nCond;m++){
    Delta0[m*nCond+m] = ( 1.0 - GetValue3D(q_Zarray,m,1,j)*GetValue(p_Warray,m,1) ) / GetValue(sigma_MKarray,m,0);
    Delta1[m*nCond+m] = GetValue3D(q_Zarray,m,1,j)*GetValue(p_Warray,m,1) / GetValue(sigma_MKarray,m,1);
    tmpnCond2[m] = GetValue(mu_MKarray,m,0);
    tmpnCond3[m] = GetValue(mu_MKarray,m,1);
  }
      
  prodaddMat(Delta0,tmpnCond2,tmpnCond,nCond,1,nCond);
  prodaddMat(Delta1,tmpnCond3,tmpnCond,nCond,1,nCond);
  
  addMatfast(Delta0,Sigma_Aj,nCond,nCond);
  addMatfast(Delta1,Sigma_Aj,nCond,nCond);
  
  invMat(Sigma_Aj,nCond);
  
  for (m=0;m<nCond;m++){
      for (m2=0;m2<nCond;m2++){
        GetValue3D(Sigma_Aarray,m,m2,j) = Sigma_Aj[m*nCond + m2];
      }
  }
  prodMat2(Sigma_Aj,tmpnCond,tmpnCond2,nCond,1,nCond);
  for (m=0;m<nCond;m++){
     GetValue(m_Aarray,j,m) = tmpnCond2[m];
  }
}

free(X_tilde);
free(Sigma_Aj);
free(Sigma_A0);
free(Delta0);
free(Delta1);
free(tmpDD);
free(tmpD);
free(tmpNrep);
free(tmpNrep2);
free(tmpnCond);
free(tmpnCond2);
free(tmpnCond3);
free(H);
free(tmp);
free(tmpT);
free(GGamma);
free(tmp2);
free(yy_tilde);
free(XX);
free(XXT);

Py_DECREF(p_Warray);
Py_DECREF(q_Zarray);
Py_DECREF(Sigma_Harray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(m_Aarray);
Py_DECREF(m_Harray);
Py_DECREF(Xarray);
Py_DECREF(sigma_MKarray);
Py_DECREF(mu_MKarray);

Py_DECREF(y_tildearray);
Py_DECREF(Gammaarray);
Py_DECREF(sigma_epsilonearray);
Py_INCREF(Py_None);
return Py_None;
}   

static PyObject *UtilsC_expectation_A_ParsiMod_RVM(PyObject *self, PyObject *args){
PyObject *y_tilde,*m_A,*m_H,*X,*Gamma,*Sigma_A,*sigma_epsilone,*Sigma_H,*mu_MK,*sigma_MK,*q_Z,*m_W, *v_W;
PyArrayObject *m_Warray,*v_Warray,*q_Zarray,*mu_MKarray,*sigma_MKarray,*y_tildearray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Gammaarray,*Sigma_Aarray;
int j,J,m,m2,d,D,nCond,Nrep,nr,k,K;
PyArg_ParseTuple(args, "OOOOOOOOOOOOOiiiii",&m_W,&v_W,&q_Z,&mu_MK,&sigma_MK,&sigma_epsilone,&Gamma,&Sigma_H,&y_tilde,&m_A,&m_H,&Sigma_A,&X,&J,&D,&nCond,&Nrep,&K);
Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
Sigma_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
mu_MKarray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_MK,PyArray_FLOAT64,2,2);
sigma_MKarray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_MK,PyArray_FLOAT64,2,2);
m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
y_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
m_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(m_W,PyArray_FLOAT64,1,1);
v_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(v_W,PyArray_FLOAT64,2,2);

npy_float64 *H,*tmp,*tmpT,*GGamma,*X_tilde,*Sigma_Aj,*Delta,*tmpDD,*tmpD,*tmpNrep,*tmpNrep2,*tmpnCond,*tmpnCond2;
npy_float64 *yy_tilde,*SSigma_H,*Sigma_A0;
int *XX,*XXT;
tmpnCond = malloc(sizeof(npy_float64)*nCond);
tmpnCond2 = malloc(sizeof(npy_float64)*nCond);
SSigma_H = malloc(sizeof(npy_float64)*D*D);
Sigma_A0 = malloc(sizeof(npy_float64)*nCond*nCond);
Sigma_Aj = malloc(sizeof(npy_float64)*nCond*nCond);
X_tilde = malloc(sizeof(npy_float64)*nCond*D);
H = malloc(sizeof(npy_float64)*D);
Delta = malloc(sizeof(npy_float64)*nCond*nCond);
tmp = malloc(sizeof(npy_float64)*Nrep*D);
tmpD = malloc(sizeof(npy_float64)*D);
tmpDD = malloc(sizeof(npy_float64)*D*D);
tmpNrep = malloc(sizeof(npy_float64)*Nrep);
tmpNrep2 = malloc(sizeof(npy_float64)*Nrep);
tmpT = malloc(sizeof(npy_float64)*Nrep*D);
GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
yy_tilde = malloc(sizeof(npy_float64)*Nrep);
XX = malloc(sizeof(int)*D*Nrep);
XXT = malloc(sizeof(int)*D*Nrep);

for (d=0;d<Nrep;d++){
  for (nr=0;nr<Nrep;nr++){
    GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
  }
}
for (d=0;d<D;d++){
  H[d] = GetValue1D(m_Harray,d);
  for (nr=0;nr<D;nr++){
      SSigma_H[d*D+nr] = GetValue(Sigma_Harray,d,nr);
  }
}
for (m=0;m<nCond;m++){
  for (m2=0;m2<nCond;m2++){
    for (d=0;d<D;d++){
      for (nr=0;nr<Nrep;nr++){
        XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
        XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
      }
    }
    prodMat4(H,XXT,tmpNrep,1,Nrep,D);
    prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
    prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
    Sigma_A0[m*nCond + m2] = prodMatScal(tmpD,H,D);
    prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
    prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
    prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
    Sigma_A0[m*nCond + m2] += traceMat(tmpDD,D);
    Delta[m*nCond+m2] = 0;
  }
}

for (j=0;j<J;j++){
  if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
  for (nr=0;nr<Nrep;nr++){
    yy_tilde[nr] = GetValue(y_tildearray,nr,j);
  }
  
  for (m=0;m<nCond;m++){
      for (d=0;d<D;d++){
        for (nr=0;nr<Nrep;nr++){
            XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
        }
      }
      prodMat2(GGamma,yy_tilde,tmpNrep,Nrep,1,Nrep);
      prodMat4(tmpNrep,XX,tmpD,1,D,Nrep);
      for (d=0;d<D;d++){ 
        X_tilde[m*D+d] = tmpD[d] * GetValue1D(m_Warray,m) / GetValue1D(sigma_epsilonearray,j);
      }
      for (m2=0;m2<nCond;m2++){
        Sigma_Aj[m*nCond + m2] = (Sigma_A0[m*nCond + m2] / GetValue1D(sigma_epsilonearray,j)) * ( GetValue1D(m_Warray,m) * GetValue1D(m_Warray,m2) + GetValue(v_Warray,m,m2) );   
      }
  }
  prodMat2(X_tilde,H,tmpnCond,nCond,1,D);
  
  for (k=0;k<K;k++){
      for (m=0;m<nCond;m++){
          Delta[m*nCond+m] = GetValue3D(q_Zarray,m,k,j) / GetValue(sigma_MKarray,m,k);
          tmpnCond2[m] = GetValue(mu_MKarray,m,k);
      }
      prodaddMat(Delta,tmpnCond2,tmpnCond,nCond,1,nCond);
      addMatfast(Delta,Sigma_Aj,nCond,nCond);
  }
  
  invMat(Sigma_Aj,nCond);
  
  for (m=0;m<nCond;m++){
      for (m2=0;m2<nCond;m2++){
        GetValue3D(Sigma_Aarray,m,m2,j) = Sigma_Aj[m*nCond + m2];
      }
  }
  prodMat2(Sigma_Aj,tmpnCond,tmpnCond2,nCond,1,nCond);
  for (m=0;m<nCond;m++){
     GetValue(m_Aarray,j,m) = tmpnCond2[m];
  }
}

free(X_tilde);
free(Sigma_Aj);
free(Sigma_A0);
free(Delta);
free(tmpDD);
free(tmpD);
free(tmpNrep);
free(tmpNrep2);
free(tmpnCond);
free(tmpnCond2);
free(H);
free(tmp);
free(tmpT);
free(GGamma);
free(yy_tilde);
free(XX);
free(XXT);

Py_DECREF(m_Warray);
Py_DECREF(v_Warray);
Py_DECREF(q_Zarray);
Py_DECREF(Sigma_Harray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(m_Aarray);
Py_DECREF(m_Harray);
Py_DECREF(Xarray);
Py_DECREF(sigma_MKarray);
Py_DECREF(mu_MKarray);

Py_DECREF(y_tildearray);
Py_DECREF(Gammaarray);
Py_DECREF(sigma_epsilonearray);
Py_INCREF(Py_None);
return Py_None;
}   

static PyObject *UtilsC_expectation_H(PyObject *self, PyObject *args){
PyObject *XGamma,*Y,*y_tilde,*m_A,*m_H,*X,*R,*Gamma,*Sigma_A,*sigma_epsilone,*Sigma_H,*QQ_barnCond;
PyArrayObject *XGammaarray,*QQ_barnCondarray,*Yarray,*y_tildearray,*Rarray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Gammaarray,*Sigma_Aarray;
int j,J,m,m2,d,D,nCond,Nrep,nr;
npy_float64 scale,sigmaH;
PyArg_ParseTuple(args, "OOOOOOOOOOOOiiiidd",&XGamma,&QQ_barnCond,&sigma_epsilone,&Gamma,&R,&Sigma_H,&Y,&y_tilde,&m_A,&m_H,&Sigma_A,&X,&J,&D,&nCond,&Nrep,&scale,&sigmaH);
Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
Rarray = (PyArrayObject *) PyArray_ContiguousFromObject(R,PyArray_FLOAT64,2,2);
m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
y_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
Yarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Y,PyArray_FLOAT64,2,2);
Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
QQ_barnCondarray = (PyArrayObject *) PyArray_ContiguousFromObject(QQ_barnCond,PyArray_FLOAT64,4,4);
XGammaarray = (PyArrayObject *) PyArray_ContiguousFromObject(XGamma,PyArray_FLOAT64,3,3);
npy_float64 *S,*H,*tmp,*tmpT,*GGamma,*tmp2;
npy_float64 *yy_tilde,*Y_bar_tilde,*Q_bar,*Q_barnCond;
struct timeval tstart, tend;

S = malloc(sizeof(npy_float64)*Nrep);
H = malloc(sizeof(npy_float64)*D);
tmp = malloc(sizeof(npy_float64)*Nrep*D);
tmpT = malloc(sizeof(npy_float64)*Nrep*D);
GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
tmp2 = malloc(sizeof(npy_float64)*Nrep*D);
yy_tilde = malloc(sizeof(npy_float64)*Nrep);
Y_bar_tilde = malloc(sizeof(npy_float64)*D);
Q_bar = malloc(sizeof(npy_float64)*D*D);
Q_barnCond = malloc(sizeof(npy_float64)*nCond*nCond*D*D);
for (d=0;d<Nrep;d++){
  for (nr=0;nr<Nrep;nr++){
    GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
  }
}
for (d=0;d<D;d++){
  for (nr=0;nr<D;nr++){
    Q_bar[d*D+nr] = GetValue(Rarray,d,nr) * scale/sigmaH;
  }
  Y_bar_tilde[d] = 0;
}
gettimeofday(&tstart, NULL);
for (j=0;j<J;j++){
  if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
  for (nr=0;nr<Nrep;nr++){
    yy_tilde[nr] = GetValue(y_tildearray,nr,j);
    for (d=0;d<D;d++){   
      tmp[nr*D + d] = 0;
      tmp2[nr*D + d] = 0;
      tmpT[nr*D + d] = 0;
    }
  }
  for (m=0;m<nCond;m++){
    for (nr=0;nr<Nrep;nr++){  
      for (d=0;d<D;d++){
        tmp2[d*Nrep + nr] += GetValue3D(XGammaarray,m,d,nr) * GetValue(m_Aarray,j,m) / GetValue1D(sigma_epsilonearray,j);
        tmpT[d*Nrep+nr] += GetValue(m_Aarray,j,m) * GetValue3DInt(Xarray,m,nr,d);
      }
    }
  }
  prodaddMat(tmp2,yy_tilde,Y_bar_tilde,D,1,Nrep);
  for (m=0;m<nCond;m++){
    for (m2=0;m2<nCond;m2++){
      for (d=0;d<D;d++){
        for (nr=0;nr<D;nr++){
        Q_bar[d*D+nr] += GetValue4D(QQ_barnCondarray,m,m2,d,nr) * ( GetValue3D(Sigma_Aarray,m,m2,j) + GetValue(m_Aarray,j,m2) * GetValue(m_Aarray,j,m)) / GetValue1D(sigma_epsilonearray,j) ;
        }
      }
    }
  }
}
gettimeofday(&tend, NULL);
if (profiling)
    printf ("Ytilde and Q_bar step took %ld msec\n", difftimeval(&tend,&tstart));

gettimeofday(&tstart, NULL);
invMat(Q_bar,D);
gettimeofday(&tend, NULL);
if (profiling)
    printf ("Inv Q_bar step took %ld msec\n", difftimeval(&tend,&tstart));

gettimeofday(&tstart, NULL);
prodMat2(Q_bar,Y_bar_tilde,H,D,1,D);
gettimeofday(&tend, NULL);
if (profiling)
    printf ("Prod Q_bar Y_bar step took %ld msec\n", difftimeval(&tend,&tstart));

for (d=0;d<D;d++){
  GetValue1D(m_Harray,d) = H[d];
  for (m=0;m<D;m++){
    GetValue(Sigma_Harray,d,m) = Q_bar[d*D+m];
  }
}
free(S);
free(H);
free(tmp);
free(tmpT);
free(GGamma);
free(tmp2);
free(yy_tilde);
free(Y_bar_tilde);
free(Q_bar);
free(Q_barnCond);
Py_DECREF(QQ_barnCondarray);
Py_DECREF(XGammaarray);
Py_DECREF(Sigma_Harray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(m_Aarray);
Py_DECREF(m_Harray);
Py_DECREF(Xarray);
Py_DECREF(Yarray);
Py_DECREF(Rarray);
Py_DECREF(y_tildearray);
Py_DECREF(Gammaarray);
Py_DECREF(sigma_epsilonearray);
Py_INCREF(Py_None);
return Py_None;
}   

static PyObject *UtilsC_expectation_H_ParsiMod(PyObject *self, PyObject *args){
PyObject *XGamma,*y_tilde,*m_A,*m_H,*X,*R,*Gamma,*Sigma_A,*sigma_epsilone,*Sigma_H,*QQ_barnCond,*p_W;
PyArrayObject *p_Warray,*XGammaarray,*QQ_barnCondarray,*y_tildearray,*Rarray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Gammaarray,*Sigma_Aarray;
int j,J,m,m2,d,D,nCond,Nrep,nr;
npy_float64 scale,sigmaH;
PyArg_ParseTuple(args, "OOOOOOOOOOOOiiiidd",&p_W,&XGamma,&QQ_barnCond,&sigma_epsilone,&Gamma,&R,&Sigma_H,&y_tilde,&m_A,&m_H,&Sigma_A,&X,&J,&D,&nCond,&Nrep,&scale,&sigmaH);
Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
Rarray = (PyArrayObject *) PyArray_ContiguousFromObject(R,PyArray_FLOAT64,2,2);
m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
y_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
QQ_barnCondarray = (PyArrayObject *) PyArray_ContiguousFromObject(QQ_barnCond,PyArray_FLOAT64,4,4);
XGammaarray = (PyArrayObject *) PyArray_ContiguousFromObject(XGamma,PyArray_FLOAT64,3,3);
p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);

npy_float64 *S,*H,*tmp,*tmpT,*GGamma,*tmp2;
npy_float64 *yy_tilde,*Y_bar_tilde,*Q_bar,*Q_barnCond;
struct timeval tstart, tend;

S = malloc(sizeof(npy_float64)*Nrep);
H = malloc(sizeof(npy_float64)*D);
tmp = malloc(sizeof(npy_float64)*Nrep*D);
tmpT = malloc(sizeof(npy_float64)*Nrep*D);
GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
tmp2 = malloc(sizeof(npy_float64)*Nrep*D);
yy_tilde = malloc(sizeof(npy_float64)*Nrep);
Y_bar_tilde = malloc(sizeof(npy_float64)*D);
Q_bar = malloc(sizeof(npy_float64)*D*D);
Q_barnCond = malloc(sizeof(npy_float64)*nCond*nCond*D*D);
for (d=0;d<Nrep;d++){
  for (nr=0;nr<Nrep;nr++){
    GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
  }
}
for (d=0;d<D;d++){
  for (nr=0;nr<D;nr++){
    Q_bar[d*D+nr] = GetValue(Rarray,d,nr) * scale/sigmaH;
  }
  Y_bar_tilde[d] = 0;
}
gettimeofday(&tstart, NULL);
for (j=0;j<J;j++){
  if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
  for (nr=0;nr<Nrep;nr++){
    yy_tilde[nr] = GetValue(y_tildearray,nr,j);
    for (d=0;d<D;d++){   
      tmp[nr*D + d] = 0;
      tmp2[nr*D + d] = 0;
      tmpT[nr*D + d] = 0;
    }
  }
  for (m=0;m<nCond;m++){
    for (nr=0;nr<Nrep;nr++){  
      for (d=0;d<D;d++){
        tmp2[d*Nrep + nr] += GetValue3D(XGammaarray,m,d,nr) * GetValue(m_Aarray,j,m) * GetValue(p_Warray,m,1) / GetValue1D(sigma_epsilonearray,j);
        tmpT[d*Nrep+nr] += GetValue(m_Aarray,j,m) * GetValue3DInt(Xarray,m,nr,d);
      }
    }
  }
  prodaddMat(tmp2,yy_tilde,Y_bar_tilde,D,1,Nrep);
  for (m=0;m<nCond;m++){
    for (m2=0;m2<nCond;m2++){
      for (d=0;d<D;d++){
        for (nr=0;nr<D;nr++){
            Q_bar[d*D+nr] += GetValue4D(QQ_barnCondarray,m,m2,d,nr) * ( GetValue3D(Sigma_Aarray,m,m2,j) + GetValue(m_Aarray,j,m2) * GetValue(m_Aarray,j,m)) * (GetValue(p_Warray,m,1) *GetValue(p_Warray,m2,1)) / GetValue1D(sigma_epsilonearray,j) ;
        }
      }
    }
  }
}
gettimeofday(&tend, NULL);
if (profiling)
    printf ("Ytilde and Q_bar step took %ld msec\n", difftimeval(&tend,&tstart));

gettimeofday(&tstart, NULL);
invMat(Q_bar,D);
gettimeofday(&tend, NULL);
if (profiling)
    printf ("Inv Q_bar step took %ld msec\n", difftimeval(&tend,&tstart));

gettimeofday(&tstart, NULL);
prodMat2(Q_bar,Y_bar_tilde,H,D,1,D);
gettimeofday(&tend, NULL);
if (profiling)
    printf ("Prod Q_bar Y_bar step took %ld msec\n", difftimeval(&tend,&tstart));

for (d=0;d<D;d++){
  GetValue1D(m_Harray,d) = H[d];
  for (m=0;m<D;m++){
    GetValue(Sigma_Harray,d,m) = Q_bar[d*D+m];
  }
}
free(S);
free(H);
free(tmp);
free(tmpT);
free(GGamma);
free(tmp2);
free(yy_tilde);
free(Y_bar_tilde);
free(Q_bar);
free(Q_barnCond);
Py_DECREF(p_Warray);
Py_DECREF(QQ_barnCondarray);
Py_DECREF(XGammaarray);
Py_DECREF(Sigma_Harray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(m_Aarray);
Py_DECREF(m_Harray);
Py_DECREF(Xarray);
Py_DECREF(Rarray);
Py_DECREF(y_tildearray);
Py_DECREF(Gammaarray);
Py_DECREF(sigma_epsilonearray);
Py_INCREF(Py_None);
return Py_None;
}   

static PyObject *UtilsC_expectation_H_ParsiMod_RVM(PyObject *self, PyObject *args){
PyObject *XGamma,*y_tilde,*m_A,*m_H,*X,*R,*Gamma,*Sigma_A,*sigma_epsilone,*Sigma_H,*QQ_barnCond,*m_W,*v_W;
PyArrayObject *m_Warray,*v_Warray,*XGammaarray,*QQ_barnCondarray,*y_tildearray,*Rarray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Gammaarray,*Sigma_Aarray;
int j,J,m,m2,d,D,nCond,Nrep,nr;
npy_float64 scale,sigmaH;
PyArg_ParseTuple(args, "OOOOOOOOOOOOOiiiidd",&m_W,&v_W,&XGamma,&QQ_barnCond,&sigma_epsilone,&Gamma,&R,&Sigma_H,&y_tilde,&m_A,&m_H,&Sigma_A,&X,&J,&D,&nCond,&Nrep,&scale,&sigmaH);
Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
Rarray = (PyArrayObject *) PyArray_ContiguousFromObject(R,PyArray_FLOAT64,2,2);
m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
y_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
QQ_barnCondarray = (PyArrayObject *) PyArray_ContiguousFromObject(QQ_barnCond,PyArray_FLOAT64,4,4);
XGammaarray = (PyArrayObject *) PyArray_ContiguousFromObject(XGamma,PyArray_FLOAT64,3,3);
m_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(m_W,PyArray_FLOAT64,1,1);
v_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(v_W,PyArray_FLOAT64,2,2);

npy_float64 *S,*H,*tmp,*tmpT,*GGamma,*tmp2;
npy_float64 *yy_tilde,*Y_bar_tilde,*Q_bar,*Q_barnCond;
struct timeval tstart, tend;

S = malloc(sizeof(npy_float64)*Nrep);
H = malloc(sizeof(npy_float64)*D);
tmp = malloc(sizeof(npy_float64)*Nrep*D);
tmpT = malloc(sizeof(npy_float64)*Nrep*D);
GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
tmp2 = malloc(sizeof(npy_float64)*Nrep*D);
yy_tilde = malloc(sizeof(npy_float64)*Nrep);
Y_bar_tilde = malloc(sizeof(npy_float64)*D);
Q_bar = malloc(sizeof(npy_float64)*D*D);
Q_barnCond = malloc(sizeof(npy_float64)*nCond*nCond*D*D);
for (d=0;d<Nrep;d++){
  for (nr=0;nr<Nrep;nr++){
    GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
  }
}
for (d=0;d<D;d++){
  for (nr=0;nr<D;nr++){
    Q_bar[d*D+nr] = GetValue(Rarray,d,nr) * scale/sigmaH;
  }
  Y_bar_tilde[d] = 0;
}
gettimeofday(&tstart, NULL);
for (j=0;j<J;j++){
  if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
  for (nr=0;nr<Nrep;nr++){
    yy_tilde[nr] = GetValue(y_tildearray,nr,j);
    for (d=0;d<D;d++){   
      tmp[nr*D + d] = 0;
      tmp2[nr*D + d] = 0;
      tmpT[nr*D + d] = 0;
    }
  }
  for (m=0;m<nCond;m++){
    for (nr=0;nr<Nrep;nr++){  
      for (d=0;d<D;d++){
        tmp2[d*Nrep + nr] += GetValue3D(XGammaarray,m,d,nr) * GetValue(m_Aarray,j,m) * GetValue1D(m_Warray,m) / GetValue1D(sigma_epsilonearray,j);
        tmpT[d*Nrep+nr] += GetValue(m_Aarray,j,m) * GetValue3DInt(Xarray,m,nr,d);
      }
    }
  }
  prodaddMat(tmp2,yy_tilde,Y_bar_tilde,D,1,Nrep);
  for (m=0;m<nCond;m++){
    for (m2=0;m2<nCond;m2++){
      for (d=0;d<D;d++){
        for (nr=0;nr<D;nr++){
            Q_bar[d*D+nr] += GetValue4D(QQ_barnCondarray,m,m2,d,nr) * ( GetValue3D(Sigma_Aarray,m,m2,j) + GetValue(m_Aarray,j,m2) * GetValue(m_Aarray,j,m)) * ( GetValue1D(m_Warray,m) * GetValue1D(m_Warray,m2) + GetValue(v_Warray,m,m2) ) / GetValue1D(sigma_epsilonearray,j) ;
        }
      }
    }
  }
}
gettimeofday(&tend, NULL);
if (profiling)
    printf ("Ytilde and Q_bar step took %ld msec\n", difftimeval(&tend,&tstart));

gettimeofday(&tstart, NULL);
invMat(Q_bar,D);
gettimeofday(&tend, NULL);
if (profiling)
    printf ("Inv Q_bar step took %ld msec\n", difftimeval(&tend,&tstart));

gettimeofday(&tstart, NULL);
prodMat2(Q_bar,Y_bar_tilde,H,D,1,D);
gettimeofday(&tend, NULL);
if (profiling)
    printf ("Prod Q_bar Y_bar step took %ld msec\n", difftimeval(&tend,&tstart));

for (d=0;d<D;d++){
  GetValue1D(m_Harray,d) = H[d];
  for (m=0;m<D;m++){
    GetValue(Sigma_Harray,d,m) = Q_bar[d*D+m];
  }
}
free(S);
free(H);
free(tmp);
free(tmpT);
free(GGamma);
free(tmp2);
free(yy_tilde);
free(Y_bar_tilde);
free(Q_bar);
free(Q_barnCond);
Py_DECREF(m_Warray);
Py_DECREF(v_Warray);
Py_DECREF(QQ_barnCondarray);
Py_DECREF(XGammaarray);
Py_DECREF(Sigma_Harray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(m_Aarray);
Py_DECREF(m_Harray);
Py_DECREF(Xarray);
Py_DECREF(Rarray);
Py_DECREF(y_tildearray);
Py_DECREF(Gammaarray);
Py_DECREF(sigma_epsilonearray);
Py_INCREF(Py_None);
return Py_None;
}
  
// npy_float64 MC_step_log(float tau1,float tau2,int m,int j,npy_float64 q_Zj,npy_float64 *q_Zarray, int M,int J,int S,int K)
// {
// 
// int s,i,k,lab,SUM;
// npy_float64 log_term, sum_log_term,E_log_term,alea;
// 
// sum_log_term = 0.;
// for(s=0;s<S;s++)
// {
//   SUM = 0;
//   for(i=0;i<J;j++)
//   {
//     if(i != j)
//     {
//       alea = rand() / RAND_MAX;
//       lab = K - 1;
//       for(k=0;k<K;k++)
//       {
// 	if(alea <= GetValue3D(q_Zarray,m,k,i))
// 	  lab = 0;
//       }
//       SUM += lab;
//     }
//   }
//   
//   log_term = log( 1 + exp( - tau1 * (SUM + q_Zj - tau2) ) );
//   sum_log_term += log_term;
// }  
// E_log_term = sum_log_term/S;
// 
// return E_log_term;
//   
// }

static PyObject *UtilsC_expectation_Z_ParsiMod_1(PyObject *self, PyObject *args){

PyObject *q_Z,*p_W,*graph,*Sigma_A,*m_A,*sigma_M,*Beta,*mu_M,*MC_mean;
PyArrayObject *q_Zarray,*p_Warray,*grapharray,*Sigma_Aarray,*m_Aarray,*sigma_Marray,*Betaarray,*mu_Marray,*MC_meanarray;
int j,J,K,k,m,maxNeighbours,nn,nCond,S;
npy_float64 tau1,tau2;

PyArg_ParseTuple(args, "OOOOOOOOiiiiiddO",&Sigma_A,&m_A,&sigma_M, &Beta,&p_W,&mu_M,&q_Z,&graph,&nCond,&J,&S,&K,&maxNeighbours,&tau1,&tau2,&MC_mean);
q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
grapharray = (PyArrayObject *) PyArray_ContiguousFromObject(graph,PyArray_INT,2,2); 
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3); 
sigma_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_M,PyArray_FLOAT64,2,2); 
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2); 
mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2); 
Betaarray = (PyArrayObject *) PyArray_ContiguousFromObject(Beta,PyArray_FLOAT64,1,1); 
MC_meanarray = (PyArrayObject *) PyArray_ContiguousFromObject(MC_mean,PyArray_FLOAT64,4,4); 
// printf("tau1 =%f\n",tau1);
// printf("tau2 =%f\n",tau2);

npy_float64 part_5, part_6, part_7;
npy_float64 *part_4, *tmp, *E_log_term, *proba;

part_4 = malloc(sizeof(npy_float64)*K);
tmp = malloc(sizeof(npy_float64)*K);
proba = malloc(sizeof(npy_float64)*K);
E_log_term = malloc(sizeof(npy_float64)*K);

// npy_float64 Var;

npy_float64 E_max, sum_proba, val0, val1;

int s,i,lab,SUM,k1;
npy_float64 log_term, sum_log_term,alea;

for(m=0;m<nCond;m++){

  for(j=0;j<J;j++){
    
    for(k=0;k<K;k++){
      tmp[k] = 0;
      nn = 0;
      while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours){
          tmp[k] += GetValue3D(q_Zarray,m,k,GetValueInt(grapharray,j,nn));
          nn++;
      }
      
      part_4[k] = log( normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,k), sqrt(GetValue(sigma_Marray,m,k)) ) + eps );
      
      // MC Step of S Iterations
      sum_log_term = 0.;
      for(s=0;s<S;s++)
      {
        SUM = 0;
        for(i=0;i<J;i++)
        {
            if(i != j)
            {
                alea = rand()/ (npy_float64)RAND_MAX;
                lab = K - 1;
                for(k1=0;k1<K-1;k1++)
                {
                    if(alea <= GetValue3D(q_Zarray,m,k1,i))
                        lab = 0;
                }
                SUM += lab;
            }
        }
        
        if(SUM + k < tau2)
        {
            log_term = (- tau1 * (SUM + k - tau2)) + log( 1. + exp( tau1 * (SUM + k - tau2) ) );
        }
        else
        {
            log_term = log( 1. + exp( - tau1 * (SUM + k - tau2) ) );
        }
        sum_log_term += log_term;
        GetValue4D(MC_meanarray,m,j,s,k) = sum_log_term/(npy_float64)(s+1);
      } 
      E_log_term[k] = sum_log_term/(npy_float64)S; 
    }
    
    part_5 = GetValue(p_Warray,m,1) * (part_4[1] - part_4[0]);
    
    part_6 = GetValue(Betaarray,m,0) * tmp[1];
    
    part_7 = 0.5*GetValue3D(Sigma_Aarray,m,m,j) * GetValue(p_Warray,m,1) * ( 1./GetValue(sigma_Marray,m,0) - 1./GetValue(sigma_Marray,m,1) ) - tau1*GetValue(p_Warray,m,0) + part_6;     
    
    val0 = GetValue(Betaarray,m,0) * tmp[0] - E_log_term[0];
    val1 = part_7 + part_5 - E_log_term[1];
    
    E_max = val0;
    if(val1 > E_max) E_max = val1;
    
    proba[0] = exp(val0 - E_max);
    proba[1] = exp(val1 - E_max);
    sum_proba = 0.0;
    for(k=0;k<K;k++)
        sum_proba += proba[k];
    for(k=0;k<K;k++)
    {
        proba[k] /= sum_proba;
        GetValue3D(q_Zarray,m,k,j) = proba[k];
    }   
    
/*    if((c - part_7 - part_5 + E_log_term[1]) < 700.)
      Var = exp(GetValue(Betaarray,m,0) * tmp[0] - E_log_term[0] - part_7 - part_5 + E_log_term[1]);
    else
      Var = exp(700.);
    
    GetValue3D(q_Zarray,m,1,j) = 1. / (1. + Var);
    GetValue3D(q_Zarray,m,0,j) = 1. - GetValue3D(q_Zarray,m,1,j);   */ 
  }
}

free(part_4);
free(tmp);
free(proba);
free(E_log_term);

Py_DECREF(grapharray);
Py_DECREF(q_Zarray);
Py_DECREF(p_Warray);
Py_DECREF(m_Aarray);
Py_DECREF(mu_Marray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(Betaarray);
Py_DECREF(sigma_Marray);
Py_INCREF(Py_None);
return Py_None;
}

static PyObject *UtilsC_expectation_Z_ParsiMod_1_MeanLabels(PyObject *self, PyObject *args){

PyObject *q_Z,*p_W,*graph,*Sigma_A,*m_A,*sigma_M,*Beta,*mu_M,*MC_mean;
PyArrayObject *q_Zarray,*p_Warray,*grapharray,*Sigma_Aarray,*m_Aarray,*sigma_Marray,*Betaarray,*mu_Marray,*MC_meanarray;
int j,J,K,k,m,maxNeighbours,nn,nCond,S;
npy_float64 tau1,tau2;

PyArg_ParseTuple(args, "OOOOOOOOiiiiiddO",&Sigma_A,&m_A,&sigma_M, &Beta,&p_W,&mu_M,&q_Z,&graph,&nCond,&J,&S,&K,&maxNeighbours,&tau1,&tau2,&MC_mean);
q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
grapharray = (PyArrayObject *) PyArray_ContiguousFromObject(graph,PyArray_INT,2,2); 
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3); 
sigma_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_M,PyArray_FLOAT64,2,2); 
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2); 
mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2); 
Betaarray = (PyArrayObject *) PyArray_ContiguousFromObject(Beta,PyArray_FLOAT64,1,1); 
MC_meanarray = (PyArrayObject *) PyArray_ContiguousFromObject(MC_mean,PyArray_FLOAT64,4,4); 
// printf("tau1 =%f\n",tau1);
// printf("tau2 =%f\n",tau2);

npy_float64 part_5, part_6, part_7;
npy_float64 *part_4, *tmp, *E_log_term, *proba;

part_4 = malloc(sizeof(npy_float64)*K);
tmp = malloc(sizeof(npy_float64)*K);
proba = malloc(sizeof(npy_float64)*K);
E_log_term = malloc(sizeof(npy_float64)*K);

// npy_float64 Var;

npy_float64 E_max, sum_proba, val0, val1;

int s,i,lab,SUM,k1;
npy_float64 log_term, sum_log_term,alea;

for(m=0;m<nCond;m++){

  for(j=0;j<J;j++){
    
    for(k=0;k<K;k++){
      tmp[k] = 0;
      nn = 0;
      while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours){
          tmp[k] += GetValue3D(q_Zarray,m,k,GetValueInt(grapharray,j,nn));
          nn++;
      }
      
      part_4[k] = log( normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,k), sqrt(GetValue(sigma_Marray,m,k)) ) + eps );
      
      // MC Step of S Iterations
      sum_log_term = 0.;
      for(s=0;s<S;s++)
      {
        SUM = 0;
        for(i=0;i<J;i++)
        {
            if(i != j)
            {
                alea = rand()/ (npy_float64)RAND_MAX;
                lab = K - 1;
                for(k1=0;k1<K-1;k1++)
                {
                    if(alea <= GetValue3D(q_Zarray,m,k1,i))
                        lab = 0;
                }
                SUM += lab;
            }
        }
        
        if( ((SUM + k)/(float)J) < tau2 )
        {
            log_term = (- tau1 * ( ((SUM + k)/(float)J) - tau2 ) ) + log( 1. + exp( tau1 * ( ((SUM + k)/(float)J) - tau2 ) ) );
        }
        else
        {
            log_term = log( 1. + exp( - tau1 * ( ((SUM + k)/(float)J) - tau2) ) );
        }
        sum_log_term += log_term;
        GetValue4D(MC_meanarray,m,j,s,k) = sum_log_term/(npy_float64)(s+1);
      } 
      E_log_term[k] = sum_log_term/(npy_float64)S; 
    }
    
    part_5 = GetValue(p_Warray,m,1) * (part_4[1] - part_4[0]);
    
    part_6 = GetValue(Betaarray,m,0) * tmp[1];
    
    part_7 = 0.5*GetValue3D(Sigma_Aarray,m,m,j) * GetValue(p_Warray,m,1) * ( 1./GetValue(sigma_Marray,m,0) - 1./GetValue(sigma_Marray,m,1) ) + (tau1/(float)J)*(GetValue(p_Warray,m,1)-1.) + part_6;     
    
    val0 = GetValue(Betaarray,m,0) * tmp[0] - E_log_term[0];
    val1 = part_7 + part_5 - E_log_term[1];
    
    E_max = val0;
    if(val1 > E_max) E_max = val1;
    
    proba[0] = exp(val0 - E_max);
    proba[1] = exp(val1 - E_max);
    sum_proba = 0.0;
    for(k=0;k<K;k++)
        sum_proba += proba[k];
    for(k=0;k<K;k++)
    {
        proba[k] /= sum_proba;
        GetValue3D(q_Zarray,m,k,j) = proba[k];
    }   
    
/*    if((c - part_7 - part_5 + E_log_term[1]) < 700.)
      Var = exp(GetValue(Betaarray,m,0) * tmp[0] - E_log_term[0] - part_7 - part_5 + E_log_term[1]);
    else
      Var = exp(700.);
    
    GetValue3D(q_Zarray,m,1,j) = 1. / (1. + Var);
    GetValue3D(q_Zarray,m,0,j) = 1. - GetValue3D(q_Zarray,m,1,j);   */ 
  }
}

free(part_4);
free(tmp);
free(proba);
free(E_log_term);

Py_DECREF(grapharray);
Py_DECREF(q_Zarray);
Py_DECREF(p_Warray);
Py_DECREF(m_Aarray);
Py_DECREF(mu_Marray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(Betaarray);
Py_DECREF(sigma_Marray);
Py_INCREF(Py_None);
return Py_None;
}

static PyObject *UtilsC_expectation_Z_ParsiMod_2(PyObject *self, PyObject *args){
    
    PyObject *q_Z,*p_W,*graph,*Sigma_A,*m_A,*sigma_M,*Beta,*mu_M;
    PyArrayObject *q_Zarray,*p_Warray,*grapharray,*Sigma_Aarray,*m_Aarray,*sigma_Marray,*Betaarray,*mu_Marray;
    int j,J,K,k,m,maxNeighbours,nn,nCond;
    npy_float64 alpha_0;
    
    PyArg_ParseTuple(args, "OOOOOOOOiiiid",&Sigma_A,&m_A,&sigma_M, &Beta,&p_W,&mu_M,&q_Z,&graph,&nCond,&J,&K,&maxNeighbours,&alpha_0);
    q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
    p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
    grapharray = (PyArrayObject *) PyArray_ContiguousFromObject(graph,PyArray_INT,2,2); 
    Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3); 
    sigma_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_M,PyArray_FLOAT64,2,2); 
    m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2); 
    mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2); 
    Betaarray = (PyArrayObject *) PyArray_ContiguousFromObject(Beta,PyArray_FLOAT64,1,1);
    
//     printf("alpha_0 =%f\n",alpha_0);
    
    npy_float64 part, part_5, part_6, part_7, Var, alea;
    npy_float64 *part_4, *tmp;
    
    part_4 = malloc(sizeof(npy_float64)*K);
    tmp = malloc(sizeof(npy_float64)*K);
    
    npy_int32 *Qm;
    Qm = malloc(sizeof(npy_int32)*J); 
    
    for(m=0;m<nCond;m++){

        for(j=0;j<J;j++){
            Qm[j] = 1;
            alea = rand()/ (float)RAND_MAX;
            if(alea <= GetValue3D(q_Zarray,m,0,j))
                Qm[j] = 0;
        }
        
        for(j=0;j<J;j++){
            for(k=0;k<K;k++){
                tmp[k] = 0;
                nn = 0;
                while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours){
                    if(Qm[j] == k && Qm[GetValueInt(grapharray,j,nn)] == k){
                        tmp[k] += GetValue3D(q_Zarray,m,k,GetValueInt(grapharray,j,nn));
                    }
                    nn++;
                }
                part_4[k] = log( normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,k), sqrt(GetValue(sigma_Marray,m,k)) ) + eps );
            }
            part_5 = GetValue(p_Warray,m,1) * (part_4[1] - part_4[0]);
            
            part_6 = GetValue(Betaarray,m,0) * tmp[1];
            
            part_7 = 0.5*GetValue3D(Sigma_Aarray,m,m,j) * GetValue(p_Warray,m,1) * ( 1./GetValue(sigma_Marray,m,0) - 1./GetValue(sigma_Marray,m,1) ) + part_6;     
            
            part = ( alpha_0 * (1. - GetValue(p_Warray,m,1)) ) + ( GetValue(Betaarray,m,0) * tmp[0] );
            
            if( (part - part_5 - part_7) < 700. )
                Var = exp(part - part_5 - part_7);
            else
                Var = exp(700.);
            
            GetValue3D(q_Zarray,m,1,j) = 1. / (1. + Var);
            GetValue3D(q_Zarray,m,0,j) = 1. - GetValue3D(q_Zarray,m,1,j);    
        }
    }
    
    free(part_4);
    free(tmp);
    free(Qm);
    
    Py_DECREF(grapharray);
    Py_DECREF(q_Zarray);
    Py_DECREF(p_Warray);
    Py_DECREF(m_Aarray);
    Py_DECREF(mu_Marray);
    Py_DECREF(Sigma_Aarray);
    Py_DECREF(Betaarray);
    Py_DECREF(sigma_Marray);
    
    Py_INCREF(Py_None);
    
    return Py_None;
}

static PyObject *UtilsC_expectation_Z_ParsiMod_3(PyObject *self, PyObject *args){
    
    PyObject *q_Z,*p_W,*graph,*Sigma_A,*m_A,*sigma_M,*Beta,*mu_M;
    PyArrayObject *q_Zarray,*p_Warray,*grapharray,*Sigma_Aarray,*m_Aarray,*sigma_Marray,*Betaarray,*mu_Marray;
    int j,J,K,k,m,maxNeighbours,nn,nCond;
    
    PyArg_ParseTuple(args, "OOOOOOOOiiii",&Sigma_A,&m_A,&sigma_M, &Beta,&p_W,&mu_M,&q_Z,&graph,&nCond,&J,&K,&maxNeighbours);
    q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
    p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
    grapharray = (PyArrayObject *) PyArray_ContiguousFromObject(graph,PyArray_INT,2,2); 
    Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3); 
    sigma_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_M,PyArray_FLOAT64,2,2); 
    m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2); 
    mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2); 
    Betaarray = (PyArrayObject *) PyArray_ContiguousFromObject(Beta,PyArray_FLOAT64,1,1); 
    
    npy_float64 part_5, part_6, part_7;
    npy_float64 *part_4, *tmp,*proba;
    
    part_4 = malloc(sizeof(npy_float64)*K);
    tmp = malloc(sizeof(npy_float64)*K);
    proba = malloc(sizeof(npy_float64)*K);
    
    npy_float64 E_max,sum_proba, val0, val1;
    
    for(m=0;m<nCond;m++){
        for(j=0;j<J;j++){
            for(k=0;k<K;k++){
                tmp[k] = 0;
                nn = 0;
                while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours){
                    tmp[k] += GetValue3D(q_Zarray,m,k,GetValueInt(grapharray,j,nn));
                    nn++;
                }
                part_4[k] = log( normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,k), sqrt(GetValue(sigma_Marray,m,k)) ) + eps);
            }

            part_5 = GetValue(p_Warray,m,1) * (part_4[1] - part_4[0]);
            
            part_6 = GetValue(Betaarray,m,0) * tmp[1];
            
            part_7 = 0.5*GetValue3D(Sigma_Aarray,m,m,j) * GetValue(p_Warray,m,1) * ( 1./GetValue(sigma_Marray,m,0) - 1./GetValue(sigma_Marray,m,1) ) + part_6;
            
            val0 = GetValue(Betaarray,m,0) * tmp[0];
            val1 = part_5 + part_7;

            E_max = val0;
            if(val1 > E_max) E_max=val1;
            
            proba[0] = exp(val0 - E_max);
            proba[1] = exp(val1 - E_max);
            
            sum_proba = 0.0;
            for(k=0;k<K;k++)
                sum_proba += proba[k];
            
            for(k=0;k<K;k++)
            {
                proba[k] /= sum_proba;
                GetValue3D(q_Zarray,m,k,j) = proba[k];
            }   
       } 
    }
    
    free(part_4);
    free(tmp);
    free(proba);
    
    Py_DECREF(grapharray);
    Py_DECREF(q_Zarray);
    Py_DECREF(p_Warray);
    Py_DECREF(m_Aarray);
    Py_DECREF(mu_Marray);
    Py_DECREF(Sigma_Aarray);
    Py_DECREF(Betaarray);
    Py_DECREF(sigma_Marray);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *UtilsC_expectation_Z_ParsiMod_RVM_and_CompMod(PyObject *self, PyObject *args){
    
    PyObject *q_Z,*graph,*Sigma_A,*m_A,*sigma_M,*Beta,*mu_M;
    PyArrayObject *q_Zarray,*grapharray,*Sigma_Aarray,*m_Aarray,*sigma_Marray,*Betaarray,*mu_Marray;
    int j,J,K,k,m,maxNeighbours,nn,nCond;
    
    PyArg_ParseTuple(args, "OOOOOOOiiii",&Sigma_A,&m_A,&sigma_M, &Beta,&mu_M,&q_Z,&graph,&nCond,&J,&K,&maxNeighbours);
    q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
    grapharray = (PyArrayObject *) PyArray_ContiguousFromObject(graph,PyArray_INT,2,2); 
    Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3); 
    sigma_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_M,PyArray_FLOAT64,2,2); 
    m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2); 
    mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2); 
    Betaarray = (PyArrayObject *) PyArray_ContiguousFromObject(Beta,PyArray_FLOAT64,1,1); 
    
    npy_float64 part_5, part_6, part_7;
    npy_float64 *part_4, *tmp,*proba;
    
    part_4 = malloc(sizeof(npy_float64)*K);
    tmp = malloc(sizeof(npy_float64)*K);
    proba = malloc(sizeof(npy_float64)*K);
    
    npy_float64 E_max,sum_proba, val0, val1;
    
    for(m=0;m<nCond;m++){
        for(j=0;j<J;j++){
            for(k=0;k<K;k++){
                tmp[k] = 0;
                nn = 0;
                while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours){
                    tmp[k] += GetValue3D(q_Zarray,m,k,GetValueInt(grapharray,j,nn));
                    nn++;
                }
                part_4[k] = log( normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,k), sqrt(GetValue(sigma_Marray,m,k)) ) + eps);
            }

            part_5 = (part_4[1] - part_4[0]);
            
            part_6 = GetValue(Betaarray,m,0) * tmp[1];
            
            part_7 = 0.5*GetValue3D(Sigma_Aarray,m,m,j) * ( 1./GetValue(sigma_Marray,m,0) - 1./GetValue(sigma_Marray,m,1) ) + part_6;
            
            val0 = GetValue(Betaarray,m,0) * tmp[0];
            val1 = part_5 + part_7;

            E_max = val0;
            if(val1 > E_max) E_max=val1;
            
            proba[0] = exp(val0 - E_max);
            proba[1] = exp(val1 - E_max);
            
            sum_proba = 0.0;
            for(k=0;k<K;k++)
                sum_proba += proba[k];
            
            for(k=0;k<K;k++)
            {
                proba[k] /= sum_proba;
                GetValue3D(q_Zarray,m,k,j) = proba[k];
            }   
       } 
    }
    
    free(part_4);
    free(tmp);
    free(proba);
    
    Py_DECREF(grapharray);
    Py_DECREF(q_Zarray);
    Py_DECREF(m_Aarray);
    Py_DECREF(mu_Marray);
    Py_DECREF(Sigma_Aarray);
    Py_DECREF(Betaarray);
    Py_DECREF(sigma_Marray);
    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *UtilsC_expectation_W_ParsiMod_1(PyObject *self, PyObject *args){
  
PyObject *HXGamma,*y_tilde,*m_A,*m_H,*X,*Gamma,*Sigma_A,*sigma_epsilone,*Sigma_H,*p_W,*q_Z,*mu_M,*sigma_M;
PyArrayObject *p_Warray,*q_Zarray,*HXGammaArray,*y_tildearray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Gammaarray,*Sigma_Aarray,*sigma_Marray,*mu_Marray;    
int j,J,m,m2,d,D,nCond,Nrep,nr,K,k;
npy_float64 tau1,tau2;

PyArg_ParseTuple(args, "OOOOOOOOOOOOOiiiiidd",&p_W,&q_Z,&HXGamma,&sigma_epsilone,&Gamma,&Sigma_H,&y_tilde,&m_A,&m_H,&Sigma_A,&X,&mu_M,&sigma_M,&J,&D,&nCond,&Nrep,&K,&tau1,&tau2);
Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
y_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
HXGammaArray = (PyArrayObject *) PyArray_ContiguousFromObject(HXGamma,PyArray_FLOAT64,2,2);
p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
sigma_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_M,PyArray_FLOAT64,2,2); 
mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2);

npy_float64 *H,*tmp,*tmpT,*GGamma,*tmpDD,*tmpD,*tmpNrep,*tmpNrep2,*part_6,*part_4,*part_1,*proba;
npy_float64 *yy_tilde,*SSigma_H;

int *XX,*XXT;

SSigma_H = malloc(sizeof(npy_float64)*D*D);
H = malloc(sizeof(npy_float64)*D);
tmp = malloc(sizeof(npy_float64)*Nrep*D);
tmpD = malloc(sizeof(npy_float64)*D);
tmpDD = malloc(sizeof(npy_float64)*D*D);
tmpNrep = malloc(sizeof(npy_float64)*Nrep);
tmpNrep2 = malloc(sizeof(npy_float64)*Nrep);
tmpT = malloc(sizeof(npy_float64)*Nrep*D);
GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
yy_tilde = malloc(sizeof(npy_float64)*Nrep);
XX = malloc(sizeof(int)*D*Nrep);
XXT = malloc(sizeof(int)*D*Nrep);

part_1 = malloc(sizeof(npy_float64)*K);
proba = malloc(sizeof(npy_float64)*K);
part_4 = malloc(sizeof(npy_float64)*nCond);
part_6 = malloc(sizeof(npy_float64)*nCond*nCond);

for (k=0;k<K;k++){
    part_1[k] = 0.0 ;
}

for (m=0;m<nCond;m++){
    part_4[m] = 0.0 ;
    for (m2=0;m2<nCond;m2++){
        part_6[m*nCond + m2] = 0.0 ;
    }
}

npy_float64 part, part_2, part_3, part_5, part_7, part_8, part_9, part_10, Sum_Z;

for (d=0;d<Nrep;d++){
  for (nr=0;nr<Nrep;nr++){
    GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
  }
}
for (d=0;d<D;d++){
  H[d] = GetValue1D(m_Harray,d);
  for (nr=0;nr<D;nr++){
      SSigma_H[d*D+nr] = GetValue(Sigma_Harray,d,nr);
  }
}

for (m=0;m<nCond;m++){
  for (m2=0;m2<nCond;m2++){
    for (d=0;d<D;d++){
      for (nr=0;nr<Nrep;nr++){
        XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
        XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
      }
    }
    if(m2 != m){
      prodMat4(H,XXT,tmpNrep,1,Nrep,D);
      prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
      prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
      part_6[m*nCond + m2] = prodMatScal(tmpD,H,D);
      prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
      prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
      prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
      part_6[m*nCond + m2] += traceMat(tmpDD,D);
      part_6[m*nCond + m2] *= GetValue(p_Warray,m2,1);
    }
  }
}


for (m=0;m<nCond;m++){
  for (d=0;d<D;d++){
    for (nr=0;nr<Nrep;nr++){
        XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
        XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
    }
  }
  prodMat4(H,XXT,tmpNrep,1,Nrep,D);
  prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
  prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
  part_4[m] = prodMatScal(tmpD,H,D);
  prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
  prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
  prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
  part_4[m] += traceMat(tmpDD,D); 
}

// npy_float64 Var2;
npy_float64 val0, val1, E_max;

for (m=0;m<nCond;m++){
  part_2 = 0.;
  part_3 = 0.;
  part_5 = 0.;
  part_8 = 0.;
  part_10 = 0.;
  Sum_Z = 0.;
  
  for (j=0;j<J;j++){
    if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
    for (nr=0;nr<Nrep;nr++){
      yy_tilde[nr] = GetValue(y_tildearray,nr,j);
    }
    
    Sum_Z += GetValue3D(q_Zarray,m,1,j);
    
    for (k=0;k<K;k++){
      part_1[k] = log( normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,k), sqrt(GetValue(sigma_Marray,m,k)) ) + eps );
    }

    part_2 += GetValue3D(q_Zarray,m,1,j) * (part_1[1] - part_1[0]);
    
    part_3 += 0.5 * GetValue3D(q_Zarray,m,1,j) * GetValue3D(Sigma_Aarray,m,m,j) * ( 1./GetValue(sigma_Marray,m,0) - 1./GetValue(sigma_Marray,m,1) );    

    part_5 += 0.5 * part_4[m] * ( GetValue3D(Sigma_Aarray,m,m,j) + pow(GetValue(m_Aarray,j,m),2) ) / GetValue1D(sigma_epsilonearray,j);

    part_9 = 0.;
    for (nr=0;nr<Nrep;nr++){
      part_9 += GetValue(HXGammaArray,m,nr) * yy_tilde[nr];

    }
    part_10 += part_9 * GetValue(m_Aarray,j,m) / GetValue1D(sigma_epsilonearray,j); 
    
    part_7 = 0.;
    for (m2=0;m2<nCond;m2++)
    {
      if (m2 != m)
        part_7 += (1./GetValue1D(sigma_epsilonearray,j)) * part_6[m*nCond + m2] * ( GetValue3D(Sigma_Aarray,m,m2,j) + GetValue(m_Aarray,j,m2) * GetValue(m_Aarray,j,m) );
    }
    part_8 += part_7;
  }
  
//   printf("m = %d, sum = %f\n",m,Sum_Z);
  part = - tau1 * ( Sum_Z - tau2);
  
  val0 = part;
  val1 = part_2 + part_3 - part_5 - part_8 + part_10;
  printf("m = %d,    val0 = %f, val1 = %f\n",m,val0,val1);
//   printf("           part_2 = %f,   part_3 = %f,    part_5 = %f,    part_8 = %f,    part_10 = %f\n",part_2,part_3,part_5,part_8,part_10);
  E_max = val0;
  if(val1 > E_max) E_max = val1;
  
  proba[1] = 1. / (1. + ( exp(val0 - E_max) / exp(val1 - E_max) ) );
  proba[0] = 1. - proba[1];
  
  for(k=0;k<K;k++)
  {
      GetValue(p_Warray,m,k) = proba[k];
  }
  
//   if((part - (part_2 + part_3 - part_5 - part_8 + part_10)) < 700.)
//     Var2 = exp( part - (part_2 + part_3 - part_5 - part_8 + part_10) );
//   else
//     Var2 = exp(700.);
//   
//   GetValue(p_Warray,m,1) = 1. / (1. + Var2);
//   GetValue(p_Warray,m,0) = 1. - GetValue(p_Warray,m,1);
  
}

free(SSigma_H);
free(H);
free(tmp);
free(tmpD);
free(tmpDD);
free(tmpNrep);
free(tmpNrep2);
free(tmpT);
free(GGamma);
free(yy_tilde);
free(XX);
free(XXT);
free(part_1);
free(part_4);
free(part_6);
free(proba);

Py_DECREF(p_Warray);
Py_DECREF(q_Zarray);
Py_DECREF(HXGammaArray);
Py_DECREF(Sigma_Harray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(m_Aarray);
Py_DECREF(m_Harray);
Py_DECREF(Xarray);
Py_DECREF(y_tildearray);
Py_DECREF(Gammaarray);
Py_DECREF(sigma_epsilonearray);
Py_INCREF(Py_None);

return Py_None;

}

static PyObject *UtilsC_expectation_W_ParsiMod_1_MeanLabels(PyObject *self, PyObject *args){
  
PyObject *HXGamma,*y_tilde,*m_A,*m_H,*X,*Gamma,*Sigma_A,*sigma_epsilone,*Sigma_H,*p_W,*q_Z,*mu_M,*sigma_M;
PyArrayObject *p_Warray,*q_Zarray,*HXGammaArray,*y_tildearray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Gammaarray,*Sigma_Aarray,*sigma_Marray,*mu_Marray;    
int j,J,m,m2,d,D,nCond,Nrep,nr,K,k;
npy_float64 tau1,tau2;

PyArg_ParseTuple(args, "OOOOOOOOOOOOOiiiiidd",&p_W,&q_Z,&HXGamma,&sigma_epsilone,&Gamma,&Sigma_H,&y_tilde,&m_A,&m_H,&Sigma_A,&X,&mu_M,&sigma_M,&J,&D,&nCond,&Nrep,&K,&tau1,&tau2);
Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
y_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
HXGammaArray = (PyArrayObject *) PyArray_ContiguousFromObject(HXGamma,PyArray_FLOAT64,2,2);
p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
sigma_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_M,PyArray_FLOAT64,2,2); 
mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2);

npy_float64 *H,*tmp,*tmpT,*GGamma,*tmpDD,*tmpD,*tmpNrep,*tmpNrep2,*part_6,*part_4,*part_1,*proba;
npy_float64 *yy_tilde,*SSigma_H;

int *XX,*XXT;

SSigma_H = malloc(sizeof(npy_float64)*D*D);
H = malloc(sizeof(npy_float64)*D);
tmp = malloc(sizeof(npy_float64)*Nrep*D);
tmpD = malloc(sizeof(npy_float64)*D);
tmpDD = malloc(sizeof(npy_float64)*D*D);
tmpNrep = malloc(sizeof(npy_float64)*Nrep);
tmpNrep2 = malloc(sizeof(npy_float64)*Nrep);
tmpT = malloc(sizeof(npy_float64)*Nrep*D);
GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
yy_tilde = malloc(sizeof(npy_float64)*Nrep);
XX = malloc(sizeof(int)*D*Nrep);
XXT = malloc(sizeof(int)*D*Nrep);

part_1 = malloc(sizeof(npy_float64)*K);
proba = malloc(sizeof(npy_float64)*K);
part_4 = malloc(sizeof(npy_float64)*nCond);
part_6 = malloc(sizeof(npy_float64)*nCond*nCond);

for (k=0;k<K;k++){
    part_1[k] = 0.0 ;
}

for (m=0;m<nCond;m++){
    part_4[m] = 0.0 ;
    for (m2=0;m2<nCond;m2++){
        part_6[m*nCond + m2] = 0.0 ;
    }
}

npy_float64 part, part_2, part_3, part_5, part_7, part_8, part_9, part_10, Sum_Z;

for (d=0;d<Nrep;d++){
  for (nr=0;nr<Nrep;nr++){
    GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
  }
}
for (d=0;d<D;d++){
  H[d] = GetValue1D(m_Harray,d);
  for (nr=0;nr<D;nr++){
      SSigma_H[d*D+nr] = GetValue(Sigma_Harray,d,nr);
  }
}

for (m=0;m<nCond;m++){
  for (m2=0;m2<nCond;m2++){
    for (d=0;d<D;d++){
      for (nr=0;nr<Nrep;nr++){
        XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
        XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
      }
    }
    if(m2 != m){
      prodMat4(H,XXT,tmpNrep,1,Nrep,D);
      prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
      prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
      part_6[m*nCond + m2] = prodMatScal(tmpD,H,D);
      prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
      prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
      prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
      part_6[m*nCond + m2] += traceMat(tmpDD,D);
      part_6[m*nCond + m2] *= GetValue(p_Warray,m2,1);
    }
  }
}


for (m=0;m<nCond;m++){
  for (d=0;d<D;d++){
    for (nr=0;nr<Nrep;nr++){
        XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
        XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
    }
  }
  prodMat4(H,XXT,tmpNrep,1,Nrep,D);
  prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
  prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
  part_4[m] = prodMatScal(tmpD,H,D);
  prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
  prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
  prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
  part_4[m] += traceMat(tmpDD,D); 
}

// npy_float64 Var2;
npy_float64 val0, val1, E_max;

for (m=0;m<nCond;m++){
  part_2 = 0.;
  part_3 = 0.;
  part_5 = 0.;
  part_8 = 0.;
  part_10 = 0.;
  Sum_Z = 0.;
  
  for (j=0;j<J;j++){
    if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
    for (nr=0;nr<Nrep;nr++){
      yy_tilde[nr] = GetValue(y_tildearray,nr,j);
    }
    
    Sum_Z += GetValue3D(q_Zarray,m,1,j);
    
    for (k=0;k<K;k++){
      part_1[k] = log( normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,k), sqrt(GetValue(sigma_Marray,m,k)) ) + eps );
    }

    part_2 += GetValue3D(q_Zarray,m,1,j) * (part_1[1] - part_1[0]);
    
    part_3 += 0.5 * GetValue3D(q_Zarray,m,1,j) * GetValue3D(Sigma_Aarray,m,m,j) * ( 1./GetValue(sigma_Marray,m,0) - 1./GetValue(sigma_Marray,m,1) );    

    part_5 += 0.5 * part_4[m] * ( GetValue3D(Sigma_Aarray,m,m,j) + pow(GetValue(m_Aarray,j,m),2) ) / GetValue1D(sigma_epsilonearray,j);

    part_9 = 0.;
    for (nr=0;nr<Nrep;nr++){
      part_9 += GetValue(HXGammaArray,m,nr) * yy_tilde[nr];

    }
    part_10 += part_9 * GetValue(m_Aarray,j,m) / GetValue1D(sigma_epsilonearray,j); 
    
    part_7 = 0.;
    for (m2=0;m2<nCond;m2++)
    {
      if (m2 != m)
        part_7 += (1./GetValue1D(sigma_epsilonearray,j)) * part_6[m*nCond + m2] * ( GetValue3D(Sigma_Aarray,m,m2,j) + GetValue(m_Aarray,j,m2) * GetValue(m_Aarray,j,m) );
    }
    part_8 += part_7;
  }
  
  printf("m = %d, sum = %f\n",m,Sum_Z);
  
  part = - tau1 * ( (Sum_Z/(float)J) - tau2);
  
  val0 = part;
  val1 = part_2 + part_3 - part_5 - part_8 + part_10;
  E_max = val0;
  if(val1 > E_max) E_max = val1;
  
  proba[1] = 1. / (1. + ( exp(val0 - E_max) / exp(val1 - E_max) ) );
  proba[0] = 1. - proba[1];
  
  for(k=0;k<K;k++)
  {
      GetValue(p_Warray,m,k) = proba[k];
  }
  
//   if((part - (part_2 + part_3 - part_5 - part_8 + part_10)) < 700.)
//     Var2 = exp( part - (part_2 + part_3 - part_5 - part_8 + part_10) );
//   else
//     Var2 = exp(700.);
//   
//   GetValue(p_Warray,m,1) = 1. / (1. + Var2);
//   GetValue(p_Warray,m,0) = 1. - GetValue(p_Warray,m,1);
  
}

free(SSigma_H);
free(H);
free(tmp);
free(tmpD);
free(tmpDD);
free(tmpNrep);
free(tmpNrep2);
free(tmpT);
free(GGamma);
free(yy_tilde);
free(XX);
free(XXT);
free(part_1);
free(part_4);
free(part_6);
free(proba);

Py_DECREF(p_Warray);
Py_DECREF(q_Zarray);
Py_DECREF(HXGammaArray);
Py_DECREF(Sigma_Harray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(m_Aarray);
Py_DECREF(m_Harray);
Py_DECREF(Xarray);
Py_DECREF(y_tildearray);
Py_DECREF(Gammaarray);
Py_DECREF(sigma_epsilonearray);
Py_INCREF(Py_None);

return Py_None;

}

static PyObject *UtilsC_expectation_W_ParsiMod_2(PyObject *self, PyObject *args){
    
    PyObject *HXGamma,*y_tilde,*m_A,*m_H,*X,*Gamma,*Sigma_A,*sigma_epsilone,*Sigma_H,*p_W,*q_Z,*mu_M,*sigma_M,*Beta,*graph;
    PyArrayObject *p_Warray,*q_Zarray,*HXGammaArray,*y_tildearray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Gammaarray,*Sigma_Aarray,*sigma_Marray,*mu_Marray,*Betaarray,*grapharray;    
    int j,J,m,m2,d,D,nCond,Nrep,nr,K,k,nn,maxNeighbours;
    npy_float64 alpha, alpha_0;
    
    PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOiiiiiidd",&p_W,&q_Z,&HXGamma,&sigma_epsilone,&Gamma,&Sigma_H,&y_tilde,&m_A,&m_H,&Sigma_A,&X,&mu_M,&sigma_M,&Beta,&graph,&J,&D,&nCond,&Nrep,&K,&maxNeighbours,&alpha,&alpha_0);
    Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
    m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
    m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
    y_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
    sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
    Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
    Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
    Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
    HXGammaArray = (PyArrayObject *) PyArray_ContiguousFromObject(HXGamma,PyArray_FLOAT64,2,2);
    p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
    q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
    sigma_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_M,PyArray_FLOAT64,2,2); 
    mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2);
    Betaarray = (PyArrayObject *) PyArray_ContiguousFromObject(Beta,PyArray_FLOAT64,1,1);
    grapharray = (PyArrayObject *) PyArray_ContiguousFromObject(graph,PyArray_INT,2,2);
    
    npy_float64 *H,*tmp,*tmpT,*GGamma,*tmpDD,*tmpD,*tmpNrep,*tmpNrep2,*part_6,*part_4,*part_1;
    npy_float64 *yy_tilde,*SSigma_H;
    
    int *XX,*XXT;
    
    SSigma_H = malloc(sizeof(npy_float64)*D*D);
    H = malloc(sizeof(npy_float64)*D);
    tmp = malloc(sizeof(npy_float64)*Nrep*D);
    tmpD = malloc(sizeof(npy_float64)*D);
    tmpDD = malloc(sizeof(npy_float64)*D*D);
    tmpNrep = malloc(sizeof(npy_float64)*Nrep);
    tmpNrep2 = malloc(sizeof(npy_float64)*Nrep);
    tmpT = malloc(sizeof(npy_float64)*Nrep*D);
    GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
    yy_tilde = malloc(sizeof(npy_float64)*Nrep);
    XX = malloc(sizeof(int)*D*Nrep);
    XXT = malloc(sizeof(int)*D*Nrep);
    
    part_1 = malloc(sizeof(npy_float64)*K);
    part_4 = malloc(sizeof(npy_float64)*nCond);
    part_6 = malloc(sizeof(npy_float64)*nCond*nCond);
    
    for (k=0;k<K;k++){
        part_1[k] = 0.0 ;
    }
    
    for (m=0;m<nCond;m++){
        part_4[m] = 0.0 ;
        for (m2=0;m2<nCond;m2++){
            part_6[m*nCond + m2] = 0.0 ;
        }
    }
    
    npy_float64 part, part_2, part_3, part_5, part_7, part_8, part_9, part_10, Sum_Z, part_11, part_12;
    
    for (d=0;d<Nrep;d++){
        for (nr=0;nr<Nrep;nr++){
            GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
        }
    }
    for (d=0;d<D;d++){
        H[d] = GetValue1D(m_Harray,d);
        for (nr=0;nr<D;nr++){
            SSigma_H[d*D+nr] = GetValue(Sigma_Harray,d,nr);
        }
    }
    
    for (m=0;m<nCond;m++){
        for (m2=0;m2<nCond;m2++){
            for (d=0;d<D;d++){
                for (nr=0;nr<Nrep;nr++){
                    XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
                    XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
                }
            }
            if(m2 != m){
                prodMat4(H,XXT,tmpNrep,1,Nrep,D);
                prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
                prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
                part_6[m*nCond + m2] = prodMatScal(tmpD,H,D);
                prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
                prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
                prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
                part_6[m*nCond + m2] += traceMat(tmpDD,D);
                part_6[m*nCond + m2] *= GetValue(p_Warray,m2,1);
            }
        }
    }

    for (m=0;m<nCond;m++){
        for (d=0;d<D;d++){
            for (nr=0;nr<Nrep;nr++){
                XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
                XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
            }
        }
        prodMat4(H,XXT,tmpNrep,1,Nrep,D);
        prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
        prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
        part_4[m] = prodMatScal(tmpD,H,D);
        prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
        prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
        prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
        part_4[m] += traceMat(tmpDD,D); 
    }
    
    npy_float64 Var2;
    
    npy_float64 Beta_m, val1, val2;
    
    npy_float64 *SUM_Q_Z_neighbours,*SUM_Pmfj1_neighbours,*SUM_Pmfj2_neighbours,*Pmfj1, *Pmfj2;
    SUM_Q_Z_neighbours = malloc(sizeof(npy_float64)*K*J);
    SUM_Pmfj1_neighbours = malloc(sizeof(npy_float64)*K*J);
    SUM_Pmfj2_neighbours = malloc(sizeof(npy_float64)*K*J);
    Pmfj1 = malloc(sizeof(npy_float64)*K*J);
    Pmfj2 = malloc(sizeof(npy_float64)*K*J);
    
    npy_int32 *Qm;
    Qm = malloc(sizeof(npy_int32)*J); 
    
    for (m=0;m<nCond;m++){
        part_2 = 0.;
        part_3 = 0.;
        part_5 = 0.;
        part_8 = 0.;
        part_10 = 0.;
        Sum_Z = 0.;
        val1 = 0.0;
        val2 = 0.0;
        
        Beta_m = GetValue(Betaarray,m,0);
        
        float alea;
        for(j=0;j<J;j++){
            Qm[j] = 1;
            for(k=0;k<K;k++){
                SUM_Q_Z_neighbours[k*J + j] = 0.0;
                SUM_Pmfj1_neighbours[k*J + j] = 0.0;
                SUM_Pmfj2_neighbours[k*J + j] = 0.0;
            }
            alea = rand()/ (float)RAND_MAX;
            if(alea <= GetValue3D(q_Zarray,m,0,j))
                Qm[j] = 0;
        }
        
        for(j=0;j<J;j++){
            for(k=0;k<K;k++){
                nn = 0;
                while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours){
                    SUM_Q_Z_neighbours[k*J + j] += GetValue3D(q_Zarray,m,k,GetValueInt(grapharray,j,nn));
                    nn++;
                } 
            }
        }
        
        for(j=0;j<J;j++){
            for(k=0;k<K;k++)
            {
                Pmfj2[k*J+j] = Compute_Pmfj(Beta_m, SUM_Q_Z_neighbours[k*J+j], SUM_Q_Z_neighbours[j], SUM_Q_Z_neighbours[J+j]);
                Pmfj1[k*J+j] = Compute_Pmfj1(Beta_m, SUM_Q_Z_neighbours[k*J+j], SUM_Q_Z_neighbours[j], SUM_Q_Z_neighbours[J+j], k, alpha_0);
            }  
        }
        
        for(j=0;j<J;j++){
            for(k=0;k<K;k++){
                nn = 0;
                while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours){
                    if(Qm[j] == k && Qm[GetValueInt(grapharray,j,nn)] == k){
                        SUM_Pmfj1_neighbours[k*J+j] += Pmfj1[k*J+GetValueInt(grapharray,j,nn)];
                        SUM_Pmfj2_neighbours[k*J+j] += Pmfj2[k*J+GetValueInt(grapharray,j,nn)];  
                    }
                    nn++;
                }
            }
        }
        
        for (j=0;j<J;j++){
            if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
            for (nr=0;nr<Nrep;nr++){
                yy_tilde[nr] = GetValue(y_tildearray,nr,j);
            }

            Sum_Z += GetValue3D(q_Zarray,m,0,j);
            
            for (k=0;k<K;k++){
                part_1[k] = log( normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,k), sqrt(GetValue(sigma_Marray,m,k)) ) + eps );
            }
            
            part_2 += GetValue3D(q_Zarray,m,1,j) * (part_1[1] - part_1[0]);
            
            part_3 += 0.5 * GetValue3D(q_Zarray,m,1,j) * GetValue3D(Sigma_Aarray,m,m,j) * ( 1./GetValue(sigma_Marray,m,0) - 1./GetValue(sigma_Marray,m,1) );    
            
            part_5 += 0.5 * part_4[m] * ( GetValue3D(Sigma_Aarray,m,m,j) + pow(GetValue(m_Aarray,j,m),2) ) / GetValue1D(sigma_epsilonearray,j);
            
            part_9 = 0.;
            for (nr=0;nr<Nrep;nr++){
                part_9 += GetValue(HXGammaArray,m,nr) * yy_tilde[nr];
                
            }
            part_10 += part_9 * GetValue(m_Aarray,j,m) / GetValue1D(sigma_epsilonearray,j); 
            
            part_7 = 0.;
            for (m2=0;m2<nCond;m2++)
            {
                if (m2 != m){
                    part_7 += (1./GetValue1D(sigma_epsilonearray,j)) * part_6[m*nCond + m2] * ( GetValue3D(Sigma_Aarray,m,m2,j) + GetValue(m_Aarray,j,m2) * GetValue(m_Aarray,j,m) );
                }     
            }
            part_8 += part_7;
            
            val1 += log ( Pmfj2[J+j] + Pmfj2[j]*exp(alpha_0) );
            for(k=0;k<K;k++){
                val2 += ( 0.5*Pmfj1[k*J+j]*SUM_Pmfj1_neighbours[k*J+j] - 0.5*Pmfj2[k*J+j]*SUM_Pmfj2_neighbours[k*J+j] - (Pmfj1[k*J+j]-Pmfj2[k*J+j])*SUM_Q_Z_neighbours[k*J+j] );
            }
            
        }

        part_11 = log(alpha) + val1 + Beta_m * val2; // Z1_Z2 = val1*exp(Beta_m * val2), part_11 = log(alpha * Z1/Z2)
        
        part_12 = alpha_0 * Sum_Z;
        
        part = log(1. - alpha);
        
        if((part - (part_2 + part_3 - part_5 - part_8 + part_10 + part_11 - part_12)) < 700.)
            Var2 = exp( part - (part_2 + part_3 - part_5 - part_8 + part_10 + part_11 - part_12) );
        else
            Var2 = exp(700.);
        
        GetValue(p_Warray,m,1) = 1. / (1. + Var2);
        GetValue(p_Warray,m,0) = 1. - GetValue(p_Warray,m,1);
        
    }
     
    free(SSigma_H);
    free(H);
    free(tmp);
    free(tmpD);
    free(tmpDD);
    free(tmpNrep);
    free(tmpNrep2);
    free(tmpT);
    free(GGamma);
    free(yy_tilde);
    free(XX);
    free(XXT);
    free(part_1);
    free(part_4);
    free(part_6);
    free(SUM_Q_Z_neighbours);
    free(SUM_Pmfj1_neighbours);
    free(SUM_Pmfj2_neighbours);
    free(Pmfj1);
    free(Pmfj2);
    
    Py_DECREF(p_Warray);
    Py_DECREF(q_Zarray);
    Py_DECREF(HXGammaArray);
    Py_DECREF(Sigma_Harray);
    Py_DECREF(Sigma_Aarray);
    Py_DECREF(m_Aarray);
    Py_DECREF(m_Harray);
    Py_DECREF(Xarray);
    Py_DECREF(y_tildearray);
    Py_DECREF(Gammaarray);
    Py_DECREF(sigma_epsilonearray);
    Py_INCREF(Py_None);
    
    return Py_None;
    
}

static PyObject *UtilsC_expectation_W_ParsiMod_3(PyObject *self, PyObject *args){
    
    PyObject *HXGamma,*y_tilde,*m_A,*m_H,*X,*Gamma,*Sigma_A,*sigma_epsilone,*Sigma_H,*p_W,*q_Z,*mu_M,*sigma_M;
    PyArrayObject *p_Warray,*q_Zarray,*HXGammaArray,*y_tildearray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Gammaarray,*Sigma_Aarray,*sigma_Marray,*mu_Marray;    
    int j,J,m,m2,d,D,nCond,Nrep,nr,K,k;
    npy_float64 tau1,tau2;
    
    PyArg_ParseTuple(args, "OOOOOOOOOOOOOiiiiidd",&p_W,&q_Z,&HXGamma,&sigma_epsilone,&Gamma,&Sigma_H,&y_tilde,&m_A,&m_H,&Sigma_A,&X,&mu_M,&sigma_M,&J,&D,&nCond,&Nrep,&K,&tau1,&tau2);
    Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
    m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
    m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
    y_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
    sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
    Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
    Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
    Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
    HXGammaArray = (PyArrayObject *) PyArray_ContiguousFromObject(HXGamma,PyArray_FLOAT64,2,2);
    p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
    q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
    sigma_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_M,PyArray_FLOAT64,2,2); 
    mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2);
    
    npy_float64 *H,*tmp,*tmpT,*GGamma,*tmpDD,*tmpD,*tmpNrep,*tmpNrep2,*part_6,*part_4,*part_1,*proba;
    npy_float64 *yy_tilde,*SSigma_H;
    
    int *XX,*XXT;
    
    SSigma_H = malloc(sizeof(npy_float64)*D*D);
    H = malloc(sizeof(npy_float64)*D);
    tmp = malloc(sizeof(npy_float64)*Nrep*D);
    tmpD = malloc(sizeof(npy_float64)*D);
    tmpDD = malloc(sizeof(npy_float64)*D*D);
    tmpNrep = malloc(sizeof(npy_float64)*Nrep);
    tmpNrep2 = malloc(sizeof(npy_float64)*Nrep);
    tmpT = malloc(sizeof(npy_float64)*Nrep*D);
    GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
    yy_tilde = malloc(sizeof(npy_float64)*Nrep);
    XX = malloc(sizeof(int)*D*Nrep);
    XXT = malloc(sizeof(int)*D*Nrep);
    
    part_1 = malloc(sizeof(npy_float64)*K);
    proba = malloc(sizeof(npy_float64)*K);
    part_4 = malloc(sizeof(npy_float64)*nCond);
    part_6 = malloc(sizeof(npy_float64)*nCond*nCond);
    
    for (k=0;k<K;k++){
        part_1[k] = 0.0 ;
    }
    
    for (m=0;m<nCond;m++){
        part_4[m] = 0.0 ;
        for (m2=0;m2<nCond;m2++){
            part_6[m*nCond + m2] = 0.0 ;
        }
    }
    
    npy_float64 part, part_2, part_3, part_5, part_7, part_8, part_9, part_10;
    
    for (d=0;d<Nrep;d++){
        for (nr=0;nr<Nrep;nr++){
            GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
        }
    }
    for (d=0;d<D;d++){
        H[d] = GetValue1D(m_Harray,d);
        for (nr=0;nr<D;nr++){
            SSigma_H[d*D+nr] = GetValue(Sigma_Harray,d,nr);
        }
    }
    
    for (m=0;m<nCond;m++){
        for (m2=0;m2<nCond;m2++){
            for (d=0;d<D;d++){
                for (nr=0;nr<Nrep;nr++){
                    XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
                    XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
                }
            }
            if(m2 != m){
                prodMat4(H,XXT,tmpNrep,1,Nrep,D);
                prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
                prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
                part_6[m*nCond + m2] = prodMatScal(tmpD,H,D);
                prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
                prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
                prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
                part_6[m*nCond + m2] += traceMat(tmpDD,D);
                part_6[m*nCond + m2] *= GetValue(p_Warray,m2,1);
            }
        }
    }
    
    
    for (m=0;m<nCond;m++){
        for (d=0;d<D;d++){
            for (nr=0;nr<Nrep;nr++){
                XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
                XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
            }
        }
        prodMat4(H,XXT,tmpNrep,1,Nrep,D);
        prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
        prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
        part_4[m] = prodMatScal(tmpD,H,D);
        prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
        prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
        prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
        part_4[m] += traceMat(tmpDD,D); 
    }
    
    npy_float64 val0, val1, E_max;
//     npy_float64 pm=0., sum_proba;
    
    for (m=0;m<nCond;m++){
        part_2 = 0.;
        part_3 = 0.;
        part_5 = 0.;
        part_8 = 0.;
        part_10 = 0.;
        
//         if(  (tau1 * (pow(GetValue(mu_Marray,m,1),2) - tau2)) >= 0.0  )
//             pm = 1. / ( 1. + exp( -tau1 * (pow(GetValue(mu_Marray,m,1),2) - tau2) ) );
//         if(  (tau1 * (pow(GetValue(mu_Marray,m,1),2) - tau2)) < 0.0  )
//             pm = exp( tau1 * (pow(GetValue(mu_Marray,m,1),2) - tau2) ) / ( 1. + exp( tau1 * (pow(GetValue(mu_Marray,m,1),2) - tau2) ) );
        
        part = - tau1 * (pow(GetValue(mu_Marray,m,1),2) - tau2);
        
        for (j=0;j<J;j++){
            if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
            for (nr=0;nr<Nrep;nr++){
                yy_tilde[nr] = GetValue(y_tildearray,nr,j);
            }
            
            for (k=0;k<K;k++){
                part_1[k] = log( normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,k), sqrt(GetValue(sigma_Marray,m,k)) )  + eps );
            }
            
            part_2 += GetValue3D(q_Zarray,m,1,j) * (part_1[1] - part_1[0]);
            
            part_3 += 0.5 * GetValue3D(q_Zarray,m,1,j) * GetValue3D(Sigma_Aarray,m,m,j) * ( 1./GetValue(sigma_Marray,m,0) - 1./GetValue(sigma_Marray,m,1) );    
            
            part_5 += 0.5 * part_4[m] * ( GetValue3D(Sigma_Aarray,m,m,j) + pow(GetValue(m_Aarray,j,m),2) ) / GetValue1D(sigma_epsilonearray,j);
            
            part_9 = 0.;
            for (nr=0;nr<Nrep;nr++){
                part_9 += GetValue(HXGammaArray,m,nr) * yy_tilde[nr];
                
            }
            part_10 += part_9 * GetValue(m_Aarray,j,m) / GetValue1D(sigma_epsilonearray,j); 
            
            part_7 = 0.;
            for (m2=0;m2<nCond;m2++)
            {
                if (m2 != m)
                    part_7 += (1./GetValue1D(sigma_epsilonearray,j)) * part_6[m*nCond + m2] * ( GetValue3D(Sigma_Aarray,m,m2,j) + GetValue(m_Aarray,j,m2) * GetValue(m_Aarray,j,m) );
            }
            part_8 += part_7;
       }
        
//        val0 = log(1.0 - pm + eps);
//        val1 = log(pm + eps) + part_2 + part_3 - part_5 - part_8 + part_10;
//         
//        E_max = val0;
//        if(val1 > E_max) E_max = val1;
//        
//        proba[0] = exp(val0 - E_max);
//        proba[1] = exp(val1 - E_max);
//        
//        sum_proba = 0.0;
//        for(k=0;k<K;k++)
//            sum_proba += proba[k];
//        
//        for(k=0;k<K;k++)
//        {
//            proba[k] /= sum_proba;
//            GetValue(p_Warray,m,k) = proba[k];
//        }

        val0 = part;
        val1 = part_2 + part_3 - part_5 - part_8 + part_10;

//         printf("m = %d, tau1 = %f, tau2 = %f, mu_1^2 = %f\n",m,tau1,tau2,pow(GetValue(mu_Marray,m,1),2));
//         printf("        part2 = %f, part3 = %f, part5 = %f, part8 = %f, part10 = %f\n",part_2,part_3,part_5,part_8,part_10);
//         printf("m = %d,     val0 = %f, val1 = %f\n",m,val0,val1);
//         printf("\n");

        E_max = val0;
        if(val1 > E_max) E_max = val1;

        proba[1] = 1. / (1. + ( exp(val0 - E_max) / exp(val1 - E_max) ) );
        proba[0] = 1. - proba[1];
        
        for(k=0;k<K;k++)
        {
            GetValue(p_Warray,m,k) = proba[k];
        }
      
    }
    
    free(SSigma_H);
    free(H);
    free(tmp);
    free(tmpD);
    free(tmpDD);
    free(tmpNrep);
    free(tmpNrep2);
    free(tmpT);
    free(GGamma);
    free(yy_tilde);
    free(XX);
    free(XXT);
    free(part_1);
    free(part_4);
    free(part_6);
    free(proba);
    
    Py_DECREF(p_Warray);
    Py_DECREF(q_Zarray);
    Py_DECREF(HXGammaArray);
    Py_DECREF(Sigma_Harray);
    Py_DECREF(Sigma_Aarray);
    Py_DECREF(m_Aarray);
    Py_DECREF(m_Harray);
    Py_DECREF(Xarray);
    Py_DECREF(y_tildearray);
    Py_DECREF(Gammaarray);
    Py_DECREF(sigma_epsilonearray);
    Py_INCREF(Py_None);
    
    return Py_None;
    
}

static PyObject *UtilsC_expectation_W_ParsiMod_RVM(PyObject *self, PyObject *args){
    
    PyObject *HXGamma,*y_tilde,*m_A,*m_H,*X,*Gamma,*Sigma_A,*sigma_epsilone,*Sigma_H,*m_W,*v_W,*alpha_RVM,*q_Z,*mu_M,*sigma_M;
    PyArrayObject *m_Warray,*v_Warray,*alpha_RVMarray,*q_Zarray,*HXGammaArray,*y_tildearray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Gammaarray,*Sigma_Aarray,*sigma_Marray,*mu_Marray;    
    int j,J,m,m2,d,D,nCond,Nrep,nr;
    
    PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOiiii",&m_W,&v_W,&alpha_RVM,&q_Z,&HXGamma,&sigma_epsilone,&Gamma,&Sigma_H,&y_tilde,&m_A,&m_H,&Sigma_A,&X,&mu_M,&sigma_M,&J,&D,&nCond,&Nrep);
    Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
    m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
    m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
    y_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
    sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
    Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
    Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
    Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
    HXGammaArray = (PyArrayObject *) PyArray_ContiguousFromObject(HXGamma,PyArray_FLOAT64,2,2);
    m_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(m_W,PyArray_FLOAT64,1,1);
    v_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(v_W,PyArray_FLOAT64,2,2);
    alpha_RVMarray = (PyArrayObject *) PyArray_ContiguousFromObject(alpha_RVM,PyArray_FLOAT64,1,1);
    q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
    sigma_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_M,PyArray_FLOAT64,2,2); 
    mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2);
    
    npy_float64 *H,*tmp,*tmpT,*GGamma,*tmpDD,*tmpD,*tmpNrep,*tmpNrep2,*part_6,*part_4,*part_1,*part_3,*part_5;
    npy_float64 *yy_tilde,*SSigma_H;
    
    int *XX,*XXT;
    
    SSigma_H = malloc(sizeof(npy_float64)*D*D);
    H = malloc(sizeof(npy_float64)*D);
    tmp = malloc(sizeof(npy_float64)*Nrep*D);
    tmpD = malloc(sizeof(npy_float64)*D);
    tmpDD = malloc(sizeof(npy_float64)*D*D);
    tmpNrep = malloc(sizeof(npy_float64)*Nrep);
    tmpNrep2 = malloc(sizeof(npy_float64)*Nrep);
    tmpT = malloc(sizeof(npy_float64)*Nrep*D);
    GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
    yy_tilde = malloc(sizeof(npy_float64)*Nrep);
    XX = malloc(sizeof(int)*D*Nrep);
    XXT = malloc(sizeof(int)*D*Nrep);

    part_1 = malloc(sizeof(npy_float64)*nCond);
    part_3 = malloc(sizeof(npy_float64)*nCond);
    part_4 = malloc(sizeof(npy_float64)*nCond);
    part_5 = malloc(sizeof(npy_float64)*nCond);
    part_6 = malloc(sizeof(npy_float64)*nCond*nCond);
    
    for (m=0;m<nCond;m++){
        part_4[m] = 0.0 ;
        part_1[m] = 0.0 ;
        part_3[m] = 0.0 ;
        part_5[m] = 0.0 ;
        for (m2=0;m2<nCond;m2++){
            part_6[m*nCond + m2] = 0.0 ;
        }
    }
    
    for (d=0;d<Nrep;d++){
        for (nr=0;nr<Nrep;nr++){
            GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
        }
    }
    for (d=0;d<D;d++){
        H[d] = GetValue1D(m_Harray,d);
        for (nr=0;nr<D;nr++){
            SSigma_H[d*D+nr] = GetValue(Sigma_Harray,d,nr);
        }
    }
    
    for (m=0;m<nCond;m++){
        for (m2=0;m2<nCond;m2++){
            for (d=0;d<D;d++){
                for (nr=0;nr<Nrep;nr++){
                    XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
                    XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
                }
            }
            if(m2 != m){
                prodMat4(H,XXT,tmpNrep,1,Nrep,D);
                prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
                prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
                part_6[m*nCond + m2] = prodMatScal(tmpD,H,D);
                prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
                prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
                prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
                part_6[m*nCond + m2] += traceMat(tmpDD,D);
            }
        }
    }
    
    for (m=0;m<nCond;m++){
        for (d=0;d<D;d++){
            for (nr=0;nr<Nrep;nr++){
                XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
                XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
            }
        }
        prodMat4(H,XXT,tmpNrep,1,Nrep,D);
        prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
        prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
        part_4[m] = prodMatScal(tmpD,H,D);
        prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
        prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
        prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
        part_4[m] += traceMat(tmpDD,D); 
    }
    
    npy_float64 part_2;
    
    for (m=0;m<nCond;m++){
        
        for (j=0;j<J;j++){
            if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
            for (nr=0;nr<Nrep;nr++){
                yy_tilde[nr] = GetValue(y_tildearray,nr,j);
            }
            
            part_1[m] += part_4[m] * ( GetValue3D(Sigma_Aarray,m,m,j) + pow(GetValue(m_Aarray,j,m),2) ) / GetValue1D(sigma_epsilonearray,j);
            
            part_2 = 0.;
            for (nr=0;nr<Nrep;nr++){
                part_2 += GetValue(HXGammaArray,m,nr) * yy_tilde[nr];  
            }
            part_3[m] += part_2 * GetValue(m_Aarray,j,m) / GetValue1D(sigma_epsilonearray,j); 
            
            for (m2=0;m2<nCond;m2++)
            {
                if (m2 != m)
                    part_5[m] += (1./GetValue1D(sigma_epsilonearray,j)) * part_6[m*nCond + m2] * GetValue1D(m_Warray,m2) * ( GetValue3D(Sigma_Aarray,m,m2,j) + GetValue(m_Aarray,j,m2) * GetValue(m_Aarray,j,m) );
            }
        }
        
        GetValue(v_Warray,m,m) = 1. / (GetValue1D(alpha_RVMarray,m) + part_1[m] + eps);
        GetValue1D(m_Warray,m) = GetValue(v_Warray,m,m) * (part_3[m] - part_5[m]);
    }
    
    
    
    free(SSigma_H);
    free(H);
    free(tmp);
    free(tmpD);
    free(tmpDD);
    free(tmpNrep);
    free(tmpNrep2);
    free(tmpT);
    free(GGamma);
    free(yy_tilde);
    free(XX);
    free(XXT);
    free(part_1);
    free(part_3);
    free(part_5);
    free(part_4);
    free(part_6);
    
    Py_DECREF(m_Warray);
    Py_DECREF(v_Warray);
    Py_DECREF(alpha_RVMarray);
    Py_DECREF(q_Zarray);
    Py_DECREF(HXGammaArray);
    Py_DECREF(Sigma_Harray);
    Py_DECREF(Sigma_Aarray);
    Py_DECREF(m_Aarray);
    Py_DECREF(m_Harray);
    Py_DECREF(Xarray);
    Py_DECREF(y_tildearray);
    Py_DECREF(Gammaarray);
    Py_DECREF(sigma_epsilonearray);
    Py_INCREF(Py_None);
    
    return Py_None;
    
}

static PyObject *UtilsC_expectation_W_ParsiMod_3_Cond(PyObject *self, PyObject *args){
    
    PyObject *HXGamma,*y_tilde,*m_A,*m_H,*X,*Gamma,*Sigma_A,*sigma_epsilone,*Sigma_H,*p_W,*q_Z,*mu_M,*sigma_M,*tau1,*tau2;
    PyArrayObject *p_Warray,*q_Zarray,*HXGammaArray,*y_tildearray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Gammaarray,*Sigma_Aarray,*sigma_Marray,*mu_Marray,*tau1_array,*tau2_array;    
    int j,J,m,m2,d,D,nCond,Nrep,nr,K,k;
    
    PyArg_ParseTuple(args, "OOOOOOOOOOOOOiiiiiOO",&p_W,&q_Z,&HXGamma,&sigma_epsilone,&Gamma,&Sigma_H,&y_tilde,&m_A,&m_H,&Sigma_A,&X,&mu_M,&sigma_M,&J,&D,&nCond,&Nrep,&K,&tau1,&tau2);
    Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
    m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
    m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
    y_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
    sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
    Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
    Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
    Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
    HXGammaArray = (PyArrayObject *) PyArray_ContiguousFromObject(HXGamma,PyArray_FLOAT64,2,2);
    p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
    q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
    sigma_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_M,PyArray_FLOAT64,2,2); 
    mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2);
    tau1_array = (PyArrayObject *) PyArray_ContiguousFromObject(tau1,PyArray_FLOAT64,1,1);
    tau2_array = (PyArrayObject *) PyArray_ContiguousFromObject(tau2,PyArray_FLOAT64,1,1);
    
    npy_float64 *H,*tmp,*tmpT,*GGamma,*tmpDD,*tmpD,*tmpNrep,*tmpNrep2,*part_6,*part_4,*part_1,*proba;
    npy_float64 *yy_tilde,*SSigma_H;
    
    int *XX,*XXT;
    
    SSigma_H = malloc(sizeof(npy_float64)*D*D);
    H = malloc(sizeof(npy_float64)*D);
    tmp = malloc(sizeof(npy_float64)*Nrep*D);
    tmpD = malloc(sizeof(npy_float64)*D);
    tmpDD = malloc(sizeof(npy_float64)*D*D);
    tmpNrep = malloc(sizeof(npy_float64)*Nrep);
    tmpNrep2 = malloc(sizeof(npy_float64)*Nrep);
    tmpT = malloc(sizeof(npy_float64)*Nrep*D);
    GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
    yy_tilde = malloc(sizeof(npy_float64)*Nrep);
    XX = malloc(sizeof(int)*D*Nrep);
    XXT = malloc(sizeof(int)*D*Nrep);
    
    part_1 = malloc(sizeof(npy_float64)*K);
    proba = malloc(sizeof(npy_float64)*K);
    part_4 = malloc(sizeof(npy_float64)*nCond);
    part_6 = malloc(sizeof(npy_float64)*nCond*nCond);
    
    for (k=0;k<K;k++){
        part_1[k] = 0.0 ;
    }
    
    for (m=0;m<nCond;m++){
        part_4[m] = 0.0 ;
        for (m2=0;m2<nCond;m2++){
            part_6[m*nCond + m2] = 0.0 ;
        }
    }
    
    npy_float64 part, part_2, part_3, part_5, part_7, part_8, part_9, part_10;
    
    for (d=0;d<Nrep;d++){
        for (nr=0;nr<Nrep;nr++){
            GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
        }
    }
    for (d=0;d<D;d++){
        H[d] = GetValue1D(m_Harray,d);
        for (nr=0;nr<D;nr++){
            SSigma_H[d*D+nr] = GetValue(Sigma_Harray,d,nr);
        }
    }
    
    for (m=0;m<nCond;m++){
        for (m2=0;m2<nCond;m2++){
            for (d=0;d<D;d++){
                for (nr=0;nr<Nrep;nr++){
                    XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
                    XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
                }
            }
            if(m2 != m){
                prodMat4(H,XXT,tmpNrep,1,Nrep,D);
                prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
                prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
                part_6[m*nCond + m2] = prodMatScal(tmpD,H,D);
                prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
                prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
                prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
                part_6[m*nCond + m2] += traceMat(tmpDD,D);
                part_6[m*nCond + m2] *= GetValue(p_Warray,m2,1);
            }
        }
    }
    
    
    for (m=0;m<nCond;m++){
        for (d=0;d<D;d++){
            for (nr=0;nr<Nrep;nr++){
                XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
                XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
            }
        }
        prodMat4(H,XXT,tmpNrep,1,Nrep,D);
        prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
        prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
        part_4[m] = prodMatScal(tmpD,H,D);
        prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
        prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
        prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
        part_4[m] += traceMat(tmpDD,D); 
    }
    
    npy_float64 val0, val1, E_max;
//     npy_float64 pm=0., sum_proba;
    
    for (m=0;m<nCond;m++){
        part_2 = 0.;
        part_3 = 0.;
        part_5 = 0.;
        part_8 = 0.;
        part_10 = 0.;
        
//         if(  (GetValue1D(tau1_array,m) * (pow(GetValue(mu_Marray,m,1),2) - GetValue1D(tau2_array,m))) >= 0.0  )
//             pm = 1. / ( 1. + exp( -GetValue1D(tau1_array,m) * (pow(GetValue(mu_Marray,m,1),2) - GetValue1D(tau2_array,m)) ) );
//         if(  (GetValue1D(tau1_array,m) * (pow(GetValue(mu_Marray,m,1),2) - GetValue1D(tau2_array,m))) < 0.0  )
//             pm = exp( GetValue1D(tau1_array,m) * (pow(GetValue(mu_Marray,m,1),2) - GetValue1D(tau2_array,m)) ) / ( 1. + exp( GetValue1D(tau1_array,m) * (pow(GetValue(mu_Marray,m,1),2) - GetValue1D(tau2_array,m)) ) );

        part = - GetValue1D(tau1_array,m) * (pow(GetValue(mu_Marray,m,1),2) - GetValue1D(tau2_array,m));
        
        for (j=0;j<J;j++){
            if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
            for (nr=0;nr<Nrep;nr++){
                yy_tilde[nr] = GetValue(y_tildearray,nr,j);
            }
            
            for (k=0;k<K;k++){
                part_1[k] = log( normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,k), sqrt(GetValue(sigma_Marray,m,k)) )  + eps );
            }
            
            //             printf("part_1[0] = %f, part_1[1] = %f\n",part_1[0],part_1[k]);
            part_2 += GetValue3D(q_Zarray,m,1,j) * (part_1[1] - part_1[0]);
            
            part_3 += 0.5 * GetValue3D(q_Zarray,m,1,j) * GetValue3D(Sigma_Aarray,m,m,j) * ( 1./GetValue(sigma_Marray,m,0) - 1./GetValue(sigma_Marray,m,1) );    
            
            part_5 += 0.5 * part_4[m] * ( GetValue3D(Sigma_Aarray,m,m,j) + pow(GetValue(m_Aarray,j,m),2) ) / GetValue1D(sigma_epsilonearray,j);
            
            part_9 = 0.;
            for (nr=0;nr<Nrep;nr++){
                part_9 += GetValue(HXGammaArray,m,nr) * yy_tilde[nr];
                
            }
            part_10 += part_9 * GetValue(m_Aarray,j,m) / GetValue1D(sigma_epsilonearray,j); 
            
            part_7 = 0.;
            for (m2=0;m2<nCond;m2++)
            {
                if (m2 != m)
                    part_7 += (1./GetValue1D(sigma_epsilonearray,j)) * part_6[m*nCond + m2] * ( GetValue3D(Sigma_Aarray,m,m2,j) + GetValue(m_Aarray,j,m2) * GetValue(m_Aarray,j,m) );
            }
            part_8 += part_7;    
        }
        
        
        // This computation way causes problems in the case of fixed HRF (not estimated) because the pm=1 (for relevant cond)
        // and theoritically val0 = -inf --> val0 < val1 --> p(w=0) < p(w=1)
        // du to eps in val0 we have val0 = log(eps)=log(0.0001)=-9.2 --> w is estimated to 0 because val1=-7694.269051 < val0
        // while val1 should be > val0 cause val0 should be -inf
//         val0 = log(1.0 - pm + eps);
//         val1 = log(pm + eps) + part_2 + part_3 - part_5 - part_8 + part_10;
//         
//         printf("val0 = %f, val1 = %f \n",val0,val1);
//         
//         E_max = val0;
//         if(val1 > E_max) E_max = val1;
//         
//         proba[0] = exp(val0 - E_max);
//         proba[1] = exp(val1 - E_max);
//         
//         sum_proba = 0.0;
//         for(k=0;k<K;k++)
//             sum_proba += proba[k];
//         
//         for(k=0;k<K;k++)
//         {
//             proba[k] /= sum_proba;
//             GetValue(p_Warray,m,k) = proba[k];
//         }

       
        val0 = part;
        val1 = part_2 + part_3 - part_5 - part_8 + part_10;
        
//         printf("m = %d, tau1 = %f, tau2 = %f, mu_1^2 = %f\n",m,GetValue1D(tau1_array,m),GetValue1D(tau2_array,m),pow(GetValue(mu_Marray,m,1),2));
//         printf("        val0 = %f, val1 = %f\n",val0,val1);
//         printf("part2 = %f, part3 = %f, part5 = %f, part8 = %f, part10 = %f\n",part_2,part_3,part_5,part_8,part_10);
//         printf("\n");
        
        E_max = val0;
        if(val1 > E_max) E_max = val1;

        proba[1] = 1. / (1. + ( exp(val0 - E_max) / exp(val1 - E_max) ) );
        proba[0] = 1. - proba[1];
        
        for(k=0;k<K;k++)
        {
            GetValue(p_Warray,m,k) = proba[k];
        }
   
    }
    
    free(SSigma_H);
    free(H);
    free(tmp);
    free(tmpD);
    free(tmpDD);
    free(tmpNrep);
    free(tmpNrep2);
    free(tmpT);
    free(GGamma);
    free(yy_tilde);
    free(XX);
    free(XXT);
    free(part_1);
    free(part_4);
    free(part_6);
    free(proba);
    
    Py_DECREF(p_Warray);
    Py_DECREF(q_Zarray);
    Py_DECREF(HXGammaArray);
    Py_DECREF(Sigma_Harray);
    Py_DECREF(Sigma_Aarray);
    Py_DECREF(m_Aarray);
    Py_DECREF(m_Harray);
    Py_DECREF(Xarray);
    Py_DECREF(y_tildearray);
    Py_DECREF(Gammaarray);
    Py_DECREF(sigma_epsilonearray);
    Py_INCREF(Py_None);
    
    return Py_None;
    
}

static PyObject *UtilsC_expectation_W_ParsiMod_4(PyObject *self, PyObject *args){
    
    PyObject *HXGamma,*y_tilde,*m_A,*m_H,*X,*Gamma,*Sigma_A,*sigma_epsilone,*Sigma_H,*p_W,*q_Z,*mu_M,*sigma_M;
    PyArrayObject *p_Warray,*q_Zarray,*HXGammaArray,*y_tildearray,*m_Aarray,*m_Harray,*sigma_epsilonearray,*Sigma_Harray,*Xarray,*Gammaarray,*Sigma_Aarray,*sigma_Marray,*mu_Marray;    
    int j,J,m,m2,d,D,nCond,Nrep,nr,K,k;
    npy_float64 tau1,tau2;
    
    PyArg_ParseTuple(args, "OOOOOOOOOOOOOiiiiidd",&p_W,&q_Z,&HXGamma,&sigma_epsilone,&Gamma,&Sigma_H,&y_tilde,&m_A,&m_H,&Sigma_A,&X,&mu_M,&sigma_M,&J,&D,&nCond,&Nrep,&K,&tau1,&tau2);
    Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
    m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
    m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
    y_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
    sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
    Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
    Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
    Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
    HXGammaArray = (PyArrayObject *) PyArray_ContiguousFromObject(HXGamma,PyArray_FLOAT64,2,2);
    p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
    q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
    sigma_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_M,PyArray_FLOAT64,2,2); 
    mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2);
    
    npy_float64 *H,*tmp,*tmpT,*GGamma,*tmpDD,*tmpD,*tmpNrep,*tmpNrep2,*part_6,*part_4,*part_1,*proba;
    npy_float64 *yy_tilde,*SSigma_H;
    
    int *XX,*XXT;
    
    SSigma_H = malloc(sizeof(npy_float64)*D*D);
    H = malloc(sizeof(npy_float64)*D);
    tmp = malloc(sizeof(npy_float64)*Nrep*D);
    tmpD = malloc(sizeof(npy_float64)*D);
    tmpDD = malloc(sizeof(npy_float64)*D*D);
    tmpNrep = malloc(sizeof(npy_float64)*Nrep);
    tmpNrep2 = malloc(sizeof(npy_float64)*Nrep);
    tmpT = malloc(sizeof(npy_float64)*Nrep*D);
    GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
    yy_tilde = malloc(sizeof(npy_float64)*Nrep);
    XX = malloc(sizeof(int)*D*Nrep);
    XXT = malloc(sizeof(int)*D*Nrep);
    
    part_1 = malloc(sizeof(npy_float64)*K);
    proba = malloc(sizeof(npy_float64)*K);
    part_4 = malloc(sizeof(npy_float64)*nCond);
    part_6 = malloc(sizeof(npy_float64)*nCond*nCond);
    
    for (k=0;k<K;k++){
        part_1[k] = 0.0 ;
    }
    
    for (m=0;m<nCond;m++){
        part_4[m] = 0.0 ;
        for (m2=0;m2<nCond;m2++){
            part_6[m*nCond + m2] = 0.0 ;
        }
    }
    
    npy_float64 part, part_2, part_3, part_5, part_7, part_8, part_9, part_10;
    
    for (d=0;d<Nrep;d++){
        for (nr=0;nr<Nrep;nr++){
            GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
        }
    }
    for (d=0;d<D;d++){
        H[d] = GetValue1D(m_Harray,d);
        for (nr=0;nr<D;nr++){
            SSigma_H[d*D+nr] = GetValue(Sigma_Harray,d,nr);
        }
    }
    
    for (m=0;m<nCond;m++){
        for (m2=0;m2<nCond;m2++){
            for (d=0;d<D;d++){
                for (nr=0;nr<Nrep;nr++){
                    XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
                    XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
                }
            }
            if(m2 != m){
                prodMat4(H,XXT,tmpNrep,1,Nrep,D);
                prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
                prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
                part_6[m*nCond + m2] = prodMatScal(tmpD,H,D);
                prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
                prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
                prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
                part_6[m*nCond + m2] += traceMat(tmpDD,D);
                part_6[m*nCond + m2] *= GetValue(p_Warray,m2,1);
            }
        }
    }
    
    
    for (m=0;m<nCond;m++){
        for (d=0;d<D;d++){
            for (nr=0;nr<Nrep;nr++){
                XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
                XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
            }
        }
        prodMat4(H,XXT,tmpNrep,1,Nrep,D);
        prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
        prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
        part_4[m] = prodMatScal(tmpD,H,D);
        prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
        prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
        prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
        part_4[m] += traceMat(tmpDD,D); 
    }
    
    npy_float64 val0, val1, E_max, dKL, m1, v1, v0;
//     npy_float64 pm=0., sum_proba;
    
    for (m=0;m<nCond;m++){
        part_2 = 0.;
        part_3 = 0.;
        part_5 = 0.;
        part_8 = 0.;
        part_10 = 0.;
        
        m1 = GetValue(mu_Marray,m,1);
        v1 = GetValue(sigma_Marray,m,1);
        v0 = GetValue(sigma_Marray,m,0);
        
        dKL = 0.5*pow(m1,2)*(1./v1 + 1./v0) + pow(v1-v0,2)/(2.*v1*v0);
        
//         if(  (tau1 * (dKL - tau2)) >= 0.0  )
//             pm = 1. / ( 1. + exp( -tau1 * (dKL - tau2) ) );
//         if(  (tau1 * (dKL - tau2)) < 0.0  )
//             pm = exp( tau1 * (dKL - tau2) ) / ( 1. + exp( tau1 * (dKL - tau2) ) );
        
        part = - tau1 * (dKL - tau2);
        
        for (j=0;j<J;j++){
            if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
            for (nr=0;nr<Nrep;nr++){
                yy_tilde[nr] = GetValue(y_tildearray,nr,j);
            }
            
            for (k=0;k<K;k++){
                part_1[k] = log( normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,k), sqrt(GetValue(sigma_Marray,m,k)) )  + eps );
            }
            
            part_2 += GetValue3D(q_Zarray,m,1,j) * (part_1[1] - part_1[0]);
            
            part_3 += 0.5 * GetValue3D(q_Zarray,m,1,j) * GetValue3D(Sigma_Aarray,m,m,j) * ( 1./GetValue(sigma_Marray,m,0) - 1./GetValue(sigma_Marray,m,1) );    
            
            part_5 += 0.5 * part_4[m] * ( GetValue3D(Sigma_Aarray,m,m,j) + pow(GetValue(m_Aarray,j,m),2) ) / GetValue1D(sigma_epsilonearray,j);
            
            part_9 = 0.;
            for (nr=0;nr<Nrep;nr++){
                part_9 += GetValue(HXGammaArray,m,nr) * yy_tilde[nr];
            }
            part_10 += part_9 * GetValue(m_Aarray,j,m) / GetValue1D(sigma_epsilonearray,j); 
            
            part_7 = 0.;
            for (m2=0;m2<nCond;m2++)
            {
                if (m2 != m)
                    part_7 += (1./GetValue1D(sigma_epsilonearray,j)) * part_6[m*nCond + m2] * ( GetValue3D(Sigma_Aarray,m,m2,j) + GetValue(m_Aarray,j,m2) * GetValue(m_Aarray,j,m) );
            }
            part_8 += part_7;
        }
        
//         val0 = log(1.0 - pm + eps);
//         val1 = log(pm + eps) + part_2 + part_3 - part_5 - part_8 + part_10;
//    
//         E_max = val0;
//         if(val1 > E_max) E_max = val1;
//         
//         proba[0] = exp(val0 - E_max);
//         proba[1] = exp(val1 - E_max);
//         
//         sum_proba = 0.0;
//         for(k=0;k<K;k++)
//             sum_proba += proba[k];
//         
//         for(k=0;k<K;k++)
//         {
//             proba[k] /= sum_proba;
//             GetValue(p_Warray,m,k) = proba[k];
//         }

        val0 = part;
        val1 = part_2 + part_3 - part_5 - part_8 + part_10;
        
        
//         printf("m = %d, tau1 = %f, tau2 = %f, mu_1^2 = %f\n",m,tau1,tau2,pow(GetValue(mu_Marray,m,1),2));
//         printf("        part2 = %f, part3 = %f, part5 = %f, part8 = %f, part10 = %f\n",part_2,part_3,part_5,part_8,part_10);
//         printf("        val0 = %f, val1 = %f\n",val0,val1);
//         printf("\n");

        E_max = val0;
        if(val1 > E_max) E_max = val1;

        proba[1] = 1. / (1. + ( exp(val0 - E_max) / exp(val1 - E_max) ) );
        proba[0] = 1. - proba[1];

        for(k=0;k<K;k++)
        {
            GetValue(p_Warray,m,k) = proba[k];
        }

    }
    
    free(SSigma_H);
    free(H);
    free(tmp);
    free(tmpD);
    free(tmpDD);
    free(tmpNrep);
    free(tmpNrep2);
    free(tmpT);
    free(GGamma);
    free(yy_tilde);
    free(XX);
    free(XXT);
    free(part_1);
    free(part_4);
    free(part_6);
    free(proba);
    
    Py_DECREF(p_Warray);
    Py_DECREF(q_Zarray);
    Py_DECREF(HXGammaArray);
    Py_DECREF(Sigma_Harray);
    Py_DECREF(Sigma_Aarray);
    Py_DECREF(m_Aarray);
    Py_DECREF(m_Harray);
    Py_DECREF(Xarray);
    Py_DECREF(y_tildearray);
    Py_DECREF(Gammaarray);
    Py_DECREF(sigma_epsilonearray);
    Py_INCREF(Py_None);
    
    return Py_None;
    
}

static PyObject *UtilsC_maximization_LP(PyObject *self, PyObject *args){
PyObject *Y,*m_A,*m_H,*L,*P,*X;
PyArrayObject *Yarray,*m_Aarray,*m_Harray,*Larray,*Parray,*Xarray;
int j,J,m,d,D,nCond,Ndrift,Nrep,nr;
PyArg_ParseTuple(args, "OOOOOOiiiii",&Y,&m_A,&m_H,&L,&P,&X,&J,&D,&nCond,&Ndrift,&Nrep);
Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3); 
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2); 
m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,2,2); 
Yarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Y,PyArray_FLOAT64,2,2); 
Larray   = (PyArrayObject *) PyArray_ContiguousFromObject(L,PyArray_FLOAT64,2,2); 
Parray   = (PyArrayObject *) PyArray_ContiguousFromObject(P,PyArray_FLOAT64,2,2); 
npy_float64 *PP,*S,*H,*Yj,*Lj,*tmp,*XH;
int *XX;
PP = malloc(sizeof(npy_float64)*Nrep*Ndrift);
S = malloc(sizeof(npy_float64)*Nrep);
tmp = malloc(sizeof(npy_float64)*Nrep);
H = malloc(sizeof(npy_float64)*D*J);
Yj = malloc(sizeof(npy_float64)*Nrep);
Lj = malloc(sizeof(npy_float64)*Ndrift);
XX = malloc(sizeof(int)*Nrep*D);
XH = malloc(sizeof(npy_float64)*Nrep*nCond);
for (nr=0;nr<Nrep;nr++){
  for (d=0;d<Ndrift;d++){
      PP[nr+d*Nrep] = GetValue(Parray,nr,d);
    }
}

for (j=0;j<J;j++){
  
  for (d=0;d<D;d++){
  H[d] = GetValue(m_Harray,d,j);
  }
  for (m=0;m<nCond;m++){
      for (d=0;d<D;d++){
        for (nr=0;nr<Nrep;nr++){
        XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
        }
      }
      prodMat3(XX,H,tmp,Nrep,1,D);
      for (nr=0;nr<Nrep;nr++){
        XH[m*Nrep + nr] = tmp[nr];
      }
  }
  
  for (nr=0;nr<Nrep;nr++){
    S[nr] = GetValue(m_Aarray,j,0) * XH[nr];
    Yj[nr] = GetValue(Yarray,nr,j);
  }
  for (m=1;m<nCond;m++){
    for (nr=0;nr<Nrep;nr++){
        S[nr] +=  GetValue(m_Aarray,j,m) * XH[m*Nrep+nr];
    }
  }
  SubMatfastVect(S,Yj,Nrep);
  prodMatVect(PP,Yj,Lj,Ndrift,Nrep);
  for (d=0;d<Ndrift;d++){
    GetValue(Larray,d,j) = Lj[d];
  }
}
free(PP);
free(S);
free(H);
free(Yj);
free(Lj);
free(XX);
free(tmp);
free(XH);
Py_DECREF(m_Aarray);
Py_DECREF(m_Harray);
Py_DECREF(Xarray);
Py_DECREF(Yarray);
Py_DECREF(Larray);
Py_DECREF(Parray);
Py_INCREF(Py_None);
return Py_None;
}


static PyObject *UtilsC_maximization_L(PyObject *self, PyObject *args){
PyObject *Y,*m_A,*m_H,*L,*P,*X;
PyArrayObject *Yarray,*m_Aarray,*m_Harray,*Larray,*Parray,*Xarray;
int j,J,m,d,D,nCond,Ndrift,Nrep,nr;
PyArg_ParseTuple(args, "OOOOOOiiiii",&Y,&m_A,&m_H,&L,&P,&X,&J,&D,&nCond,&Ndrift,&Nrep);
Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3); 
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2); 
m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1); 
Yarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Y,PyArray_FLOAT64,2,2); 
Larray   = (PyArrayObject *) PyArray_ContiguousFromObject(L,PyArray_FLOAT64,2,2); 
Parray   = (PyArrayObject *) PyArray_ContiguousFromObject(P,PyArray_FLOAT64,2,2); 
npy_float64 *PP,*S,*H,*Yj,*Lj,*tmp,*XH;
int *XX;
PP = malloc(sizeof(npy_float64)*Nrep*Ndrift);
S = malloc(sizeof(npy_float64)*Nrep);
tmp = malloc(sizeof(npy_float64)*Nrep);
H = malloc(sizeof(npy_float64)*D);
Yj = malloc(sizeof(npy_float64)*Nrep);
Lj = malloc(sizeof(npy_float64)*Ndrift);
XX = malloc(sizeof(int)*Nrep*D);
XH = malloc(sizeof(npy_float64)*Nrep*nCond);
for (nr=0;nr<Nrep;nr++){
  for (d=0;d<Ndrift;d++){
      PP[nr+d*Nrep] = GetValue(Parray,nr,d);
    }
}
for (d=0;d<D;d++){
  H[d] = GetValue1D(m_Harray,d);
}
for (m=0;m<nCond;m++){
     for (d=0;d<D;d++){
      for (nr=0;nr<Nrep;nr++){
        XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
      }
    }
    prodMat3(XX,H,tmp,Nrep,1,D);
    for (nr=0;nr<Nrep;nr++){
        XH[m*Nrep + nr] = tmp[nr];
    }
}
for (j=0;j<J;j++){
  for (nr=0;nr<Nrep;nr++){
    S[nr] = GetValue(m_Aarray,j,0) * XH[nr];
    Yj[nr] = GetValue(Yarray,nr,j);
  }
  for (m=1;m<nCond;m++){
     for (nr=0;nr<Nrep;nr++){
        S[nr] +=  GetValue(m_Aarray,j,m) * XH[m*Nrep+nr];
     }
  }
  SubMatfastVect(S,Yj,Nrep);
  prodMatVect(PP,Yj,Lj,Ndrift,Nrep);
  for (d=0;d<Ndrift;d++){
    GetValue(Larray,d,j) = Lj[d];
  }
}
free(PP);
free(S);
free(H);
free(Yj);
free(Lj);
free(XX);
free(tmp);
free(XH);
Py_DECREF(m_Aarray);
Py_DECREF(m_Harray);
Py_DECREF(Xarray);
Py_DECREF(Yarray);
Py_DECREF(Larray);
Py_DECREF(Parray);
Py_INCREF(Py_None);
return Py_None;
}

static PyObject *UtilsC_maximization_L_ParsiMod(PyObject *self, PyObject *args){
    PyObject *Y,*m_A,*m_H,*L,*P,*X,*p_W;
    PyArrayObject *Yarray,*m_Aarray,*m_Harray,*Larray,*Parray,*Xarray,*p_Warray;
    int j,J,m,d,D,nCond,Ndrift,Nrep,nr;
    PyArg_ParseTuple(args, "OOOOOOOiiiii",&Y,&m_A,&m_H,&L,&P,&X,&p_W,&J,&D,&nCond,&Ndrift,&Nrep);
    Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3); 
    m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2); 
    m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1); 
    Yarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Y,PyArray_FLOAT64,2,2); 
    Larray   = (PyArrayObject *) PyArray_ContiguousFromObject(L,PyArray_FLOAT64,2,2); 
    Parray   = (PyArrayObject *) PyArray_ContiguousFromObject(P,PyArray_FLOAT64,2,2); 
    p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
    npy_float64 *PP,*S,*H,*Yj,*Lj,*tmp,*XH;
    int *XX;
    PP = malloc(sizeof(npy_float64)*Nrep*Ndrift);
    S = malloc(sizeof(npy_float64)*Nrep);
    tmp = malloc(sizeof(npy_float64)*Nrep);
    H = malloc(sizeof(npy_float64)*D);
    Yj = malloc(sizeof(npy_float64)*Nrep);
    Lj = malloc(sizeof(npy_float64)*Ndrift);
    XX = malloc(sizeof(int)*Nrep*D);
    XH = malloc(sizeof(npy_float64)*Nrep*nCond);
    for (nr=0;nr<Nrep;nr++){
        for (d=0;d<Ndrift;d++){
            PP[nr+d*Nrep] = GetValue(Parray,nr,d);
        }
    }
    for (d=0;d<D;d++){
        H[d] = GetValue1D(m_Harray,d);
    }
    for (m=0;m<nCond;m++){
        for (d=0;d<D;d++){
            for (nr=0;nr<Nrep;nr++){
                XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
            }
        }
        prodMat3(XX,H,tmp,Nrep,1,D);
        for (nr=0;nr<Nrep;nr++){
            XH[m*Nrep + nr] = tmp[nr];
        }
    }
    for (j=0;j<J;j++){
        for (nr=0;nr<Nrep;nr++){
            S[nr] = GetValue(p_Warray,0,1) * GetValue(m_Aarray,j,0) * XH[nr];
            Yj[nr] = GetValue(Yarray,nr,j);
        }
        for (m=1;m<nCond;m++){
            for (nr=0;nr<Nrep;nr++){
                S[nr] +=  GetValue(p_Warray,m,1) * GetValue(m_Aarray,j,m) * XH[m*Nrep+nr];
            }
        }
        SubMatfastVect(S,Yj,Nrep);
        prodMatVect(PP,Yj,Lj,Ndrift,Nrep);
        for (d=0;d<Ndrift;d++){
            GetValue(Larray,d,j) = Lj[d];
        }
    }
    free(PP);
    free(S);
    free(H);
    free(Yj);
    free(Lj);
    free(XX);
    free(tmp);
    free(XH);
    Py_DECREF(m_Aarray);
    Py_DECREF(m_Harray);
    Py_DECREF(Xarray);
    Py_DECREF(Yarray);
    Py_DECREF(Larray);
    Py_DECREF(Parray);
    Py_DECREF(p_Warray);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *UtilsC_maximization_L_ParsiMod_RVM(PyObject *self, PyObject *args){
    PyObject *Y,*m_A,*m_H,*L,*P,*X,*m_W;
    PyArrayObject *Yarray,*m_Aarray,*m_Harray,*Larray,*Parray,*Xarray,*m_Warray;
    int j,J,m,d,D,nCond,Ndrift,Nrep,nr;
    PyArg_ParseTuple(args, "OOOOOOOiiiii",&Y,&m_A,&m_H,&L,&P,&X,&m_W,&J,&D,&nCond,&Ndrift,&Nrep);
    Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3); 
    m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2); 
    m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1); 
    Yarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Y,PyArray_FLOAT64,2,2); 
    Larray   = (PyArrayObject *) PyArray_ContiguousFromObject(L,PyArray_FLOAT64,2,2); 
    Parray   = (PyArrayObject *) PyArray_ContiguousFromObject(P,PyArray_FLOAT64,2,2); 
    m_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(m_W,PyArray_FLOAT64,1,1);
    
    npy_float64 *PP,*S,*H,*Yj,*Lj,*tmp,*XH;
    int *XX;
    PP = malloc(sizeof(npy_float64)*Nrep*Ndrift);
    S = malloc(sizeof(npy_float64)*Nrep);
    tmp = malloc(sizeof(npy_float64)*Nrep);
    H = malloc(sizeof(npy_float64)*D);
    Yj = malloc(sizeof(npy_float64)*Nrep);
    Lj = malloc(sizeof(npy_float64)*Ndrift);
    XX = malloc(sizeof(int)*Nrep*D);
    XH = malloc(sizeof(npy_float64)*Nrep*nCond);
    for (nr=0;nr<Nrep;nr++){
        for (d=0;d<Ndrift;d++){
            PP[nr+d*Nrep] = GetValue(Parray,nr,d);
        }
    }
    for (d=0;d<D;d++){
        H[d] = GetValue1D(m_Harray,d);
    }
    for (m=0;m<nCond;m++){
        for (d=0;d<D;d++){
            for (nr=0;nr<Nrep;nr++){
                XX[nr*D + d] = GetValue3DInt(Xarray,m,nr,d);
            }
        }
        prodMat3(XX,H,tmp,Nrep,1,D);
        for (nr=0;nr<Nrep;nr++){
            XH[m*Nrep + nr] = tmp[nr];
        }
    }
    for (j=0;j<J;j++){
        for (nr=0;nr<Nrep;nr++){
            S[nr] = GetValue1D(m_Warray,0) * GetValue(m_Aarray,j,0) * XH[nr];
            Yj[nr] = GetValue(Yarray,nr,j);
        }
        for (m=1;m<nCond;m++){
            for (nr=0;nr<Nrep;nr++){
                S[nr] +=  GetValue1D(m_Warray,m) * GetValue(m_Aarray,j,m) * XH[m*Nrep+nr];
            }
        }
        SubMatfastVect(S,Yj,Nrep);
        prodMatVect(PP,Yj,Lj,Ndrift,Nrep);
        for (d=0;d<Ndrift;d++){
            GetValue(Larray,d,j) = Lj[d];
        }
    }
    free(PP);
    free(S);
    free(H);
    free(Yj);
    free(Lj);
    free(XX);
    free(tmp);
    free(XH);
    Py_DECREF(m_Aarray);
    Py_DECREF(m_Harray);
    Py_DECREF(Xarray);
    Py_DECREF(Yarray);
    Py_DECREF(Larray);
    Py_DECREF(Parray);
    Py_DECREF(m_Warray);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *UtilsC_maximization_beta(PyObject *self, PyObject *args){
PyObject *q_Z,*Z_tilde,*graph;
PyArrayObject *q_Zarray,*Z_tildearray,*grapharray;
npy_float64 beta,gamma,Gr,gradientStep;
int j,J,K,k,ni,maxNeighbours,nn,MaxItGrad;

PyArg_ParseTuple(args, "dOOiiOdiid", &beta,&q_Z,&Z_tilde,&J,&K,&graph,&gamma,&maxNeighbours,&MaxItGrad,&gradientStep);
q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,2,2); 
Z_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(Z_tilde,PyArray_FLOAT64,2,2); 
grapharray = (PyArrayObject *) PyArray_ContiguousFromObject(graph,PyArray_INT,2,2); 
npy_float64 tmp2[K],Emax,Sum,tmp,Pzmi;
Gr = gamma; 
ni = 0;
while( (ni<MaxItGrad) && (fabs(Gr) > eps) ){
  Gr = gamma;
  for (j=0;j<J;j++){
    Emax = 0;
    for (k=0;k<K;k++){
      tmp2[k] = 0;
      nn = 0;
      while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours) {
        tmp2[k] += beta * GetValue(Z_tildearray,k,GetValueInt(grapharray,j,nn));
        nn++;
      }
      if (tmp2[k] > Emax) Emax = tmp2[k];
    }
    Sum = 0;
    for (k=0;k<K;k++){
      Sum += exp(tmp2[k] - Emax);
    }
    for (k=0;k<K;k++){
      tmp = 0;
      nn = 0;
      while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours) {
        tmp += GetValue(Z_tildearray,k,GetValueInt(grapharray,j,nn));
        nn++;
      }
      Pzmi = exp(beta * tmp - Emax) / (Sum + eps);
      Gr += tmp * ( Pzmi - GetValue(q_Zarray,k,j) );
    }
  }
  beta -= gradientStep*Gr;
  ni++;
}
if (eps > beta) beta = 0.01;
Py_DECREF(grapharray);
Py_DECREF(q_Zarray);
Py_DECREF(Z_tildearray);
Py_INCREF(Py_None);
return Py_BuildValue("d", beta);
}

static PyObject *UtilsC_maximization_beta_CB(PyObject *self, PyObject *args){
PyObject *q_Z,*graph;
PyArrayObject *q_Zarray,*grapharray;
npy_float64 beta,gamma,Gr,gradientStep,Pzmvoisk_sum,Pzmjk;
int j,J,K,k,ni,maxNeighbours,nn,MaxItGrad;

PyArg_ParseTuple(args, "dOiiOdiid", &beta,&q_Z,&J,&K,&graph,&gamma,&maxNeighbours,&MaxItGrad,&gradientStep);
q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,2,2); 
grapharray = (PyArrayObject *) PyArray_ContiguousFromObject(graph,PyArray_INT,2,2); 
npy_float64 tmp;
Gr = gamma; 
ni = 0;
while( (ni<MaxItGrad) && (fabs(Gr) > eps) ){
  Gr = gamma;
  for (j=0;j<J;j++){
    for (k=0;k<K;k++){
      tmp = 0;
      nn = 0;
      Pzmvoisk_sum = 0;
      while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours) {
        tmp += GetValue(q_Zarray,k,GetValueInt(grapharray,j,nn));
        Pzmvoisk_sum += Pzmvoxclass(beta,q_Zarray,grapharray,maxNeighbours,K,GetValueInt(grapharray,j,nn),k);
        nn++;
      }
      Pzmjk = Pzmvoxclass(beta,q_Zarray,grapharray,maxNeighbours,K,j,k);
      Gr += 0.5 * ( Pzmjk * Pzmvoisk_sum - GetValue(q_Zarray,k,j)*tmp );
    }
  }
  beta -= gradientStep*Gr;
  ni++;
}
if (eps > beta) beta = 0.01; // Pourquoi pas beta = eps ?
Py_DECREF(grapharray);
Py_DECREF(q_Zarray);
Py_INCREF(Py_None);
return Py_BuildValue("d", beta);
}

static PyObject *UtilsC_expectation_Z(PyObject *self, PyObject *args){
PyObject *q_Z,*Z_tilde,*graph,*Sigma_A,*m_A,*sigma_M,*Beta,*mu_M;
PyArrayObject *q_Zarray,*Z_tildearray,*grapharray,*Sigma_Aarray,*m_Aarray,*sigma_Marray,*Betaarray,*mu_Marray;
int j,J,K,k,m,maxNeighbours,nn,M;
PyArg_ParseTuple(args, "OOOOOOOOiiii",&Sigma_A,&m_A,&sigma_M, &Beta,&Z_tilde,&mu_M,&q_Z,&graph,&M,&J,&K,&maxNeighbours);
q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
Z_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(Z_tilde,PyArray_FLOAT64,3,3); 
grapharray = (PyArrayObject *) PyArray_ContiguousFromObject(graph,PyArray_INT,2,2); 
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3); 
sigma_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_M,PyArray_FLOAT64,2,2); 
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2); 
mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2); 
Betaarray = (PyArrayObject *) PyArray_ContiguousFromObject(Beta,PyArray_FLOAT64,1,1); 

npy_float64 tmp[K],Emax,Sum,alpha[K],Malpha,extern_field,Gauss[K],local_energy,energy[K],Probas[K];
for (j=0;j<J;j++){
  for (m=0;m<M;m++){
    Malpha = 0;
    for (k=0;k<K;k++){

      alpha[k] = -0.5*GetValue3D(Sigma_Aarray,m,m,j) / (eps + GetValue(sigma_Marray,m,k) );
      Malpha += alpha[k]/K;
    }
    for (k=0;k<K;k++){
      alpha[k] /= Malpha;
      tmp[k] = 0;
      nn = 0;
      while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours){
        tmp[k] += GetValue3D(Z_tildearray,m,k,GetValueInt(grapharray,j,nn));
        nn++;
      }
    }
    Emax = 0;
    for (k=0;k<K;k++){
      energy[k] = 0;
      extern_field = log( normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,k), sqrt(GetValue(sigma_Marray,m,k)) ) +eps);
      if (extern_field < -100) extern_field = -100;
      extern_field += alpha[k];
      local_energy = GetValue(Betaarray,m,0) * tmp[k];
      energy[k] += local_energy + extern_field;
      if (energy[k] > Emax) Emax = energy[k];
    }
    Sum = 0;
    for (k=0;k<K;k++){
      Probas[k] = exp( energy[k] - Emax );
      Sum += Probas[k];
    }
    for (k=0;k<K;k++){
      GetValue3D(Z_tildearray,m,k,j) = Probas[k] / (Sum + eps);
    }
  }
}
/*--------------------------------------------------------------*/
for (j=0;j<J;j++){
  for (m=0;m<M;m++){
    Malpha = 0;
    for (k=0;k<K;k++){
      alpha[k] = -0.5*GetValue3D(Sigma_Aarray,m,m,j) / (eps + GetValue(sigma_Marray,m,k) );
      Malpha += alpha[k]/K;
    }
    for (k=0;k<K;k++){
      tmp[k] = 0;
      nn = 0;
      alpha[k] /= Malpha;
      while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours){
        tmp[k] += GetValue3D(Z_tildearray,m,k,GetValueInt(grapharray,j,nn));
        nn++;
      }
    }
    Emax = 0;
    for (k=0;k<K;k++){
      energy[k] = 0;
      extern_field = alpha[k];
      local_energy = GetValue(Betaarray,m,0) * tmp[k];
      energy[k] += local_energy + extern_field;
      if (energy[k] > Emax) Emax = energy[k];
      Gauss[k] = normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,k), sqrt(GetValue(sigma_Marray,m,k)) );
    }
    Sum = 0;
    for (k=0;k<K;k++){
      Probas[k] = exp( energy[k] - Emax );
      Sum += Probas[k];
    }
    for (k=0;k<K;k++){
      GetValue3D(q_Zarray,m,k,j) = Gauss[k]*Probas[k] / (Sum + eps);
    }
    Sum = 0;
    for (k=0;k<K;k++){
      Sum += GetValue3D(q_Zarray,m,k,j);
    }
    for (k=0;k<K;k++){
      GetValue3D(q_Zarray,m,k,j) /= Sum;
    }
  } 
}
Py_DECREF(grapharray);
Py_DECREF(q_Zarray);
Py_DECREF(Z_tildearray);
Py_DECREF(m_Aarray);
Py_DECREF(mu_Marray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(Betaarray);
Py_DECREF(sigma_Marray);
Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *UtilsC_expectation_Z_MF_ParsiMod_3(PyObject *self, PyObject *args){
PyObject *q_Z,*Z_tilde,*graph,*Sigma_A,*m_A,*sigma_M,*Beta,*mu_M,*p_W;
PyArrayObject *q_Zarray,*Z_tildearray,*grapharray,*Sigma_Aarray,*m_Aarray,*sigma_Marray,*Betaarray,*mu_Marray,*p_Warray;
int j,J,K,k,m,maxNeighbours,nn,M;
PyArg_ParseTuple(args, "OOOOOOOOOiiii",&p_W,&Sigma_A,&m_A,&sigma_M, &Beta,&Z_tilde,&mu_M,&q_Z,&graph,&M,&J,&K,&maxNeighbours);
p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
Z_tildearray = (PyArrayObject *) PyArray_ContiguousFromObject(Z_tilde,PyArray_FLOAT64,3,3); 
grapharray = (PyArrayObject *) PyArray_ContiguousFromObject(graph,PyArray_INT,2,2); 
Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3); 
sigma_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_M,PyArray_FLOAT64,2,2); 
m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2); 
mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2); 
Betaarray = (PyArrayObject *) PyArray_ContiguousFromObject(Beta,PyArray_FLOAT64,1,1); 

npy_float64 tmp[K],Emax,Sum,alpha[K],Malpha,extern_field,Gauss[K],local_energy,energy[K],Probas[K],C;
for (j=0;j<J;j++){
  for (m=0;m<M;m++){
    Malpha = 0;
    C = -0.5*GetValue3D(Sigma_Aarray,m,m,j)*GetValue(p_Warray,m,0) / (eps + GetValue(sigma_Marray,m,0) );
    for (k=0;k<K;k++){

      alpha[k] = C -0.5*GetValue3D(Sigma_Aarray,m,m,j)*GetValue(p_Warray,m,1) / (eps + GetValue(sigma_Marray,m,k) );
      Malpha += alpha[k]/K;
    }
    for (k=0;k<K;k++){
      alpha[k] /= Malpha;
      tmp[k] = 0;
      nn = 0;
      while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours){
        tmp[k] += GetValue3D(Z_tildearray,m,k,GetValueInt(grapharray,j,nn));
        nn++;
      }
    }
    Emax = 0;
    for (k=0;k<K;k++){
      energy[k] = 0;
      extern_field = GetValue(p_Warray,m,1) * log( normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,k), sqrt(GetValue(sigma_Marray,m,k)) ) +eps);
      extern_field += GetValue(p_Warray,m,0) * log( normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,0), sqrt(GetValue(sigma_Marray,m,0)) ) +eps);
      if (extern_field < -100) extern_field = -100;
      extern_field += alpha[k];
      local_energy = GetValue(Betaarray,m,0) * tmp[k];
      energy[k] += local_energy + extern_field;
      if (energy[k] > Emax) Emax = energy[k];
    }
    Sum = 0;
    for (k=0;k<K;k++){
      Probas[k] = exp( energy[k] - Emax );
      Sum += Probas[k];
    }
    for (k=0;k<K;k++){
      GetValue3D(Z_tildearray,m,k,j) = Probas[k] / (Sum + eps);
    }
  }
}
/*--------------------------------------------------------------*/
for (j=0;j<J;j++){
  for (m=0;m<M;m++){
    Malpha = 0;
    C = -0.5*GetValue3D(Sigma_Aarray,m,m,j)*GetValue(p_Warray,m,0) / (eps + GetValue(sigma_Marray,m,0) );
    for (k=0;k<K;k++){
      alpha[k] = C -0.5*GetValue3D(Sigma_Aarray,m,m,j)*GetValue(p_Warray,m,1) / (eps + GetValue(sigma_Marray,m,k) );
      Malpha += alpha[k]/K;
    }
    for (k=0;k<K;k++){
      tmp[k] = 0;
      nn = 0;
      alpha[k] /= Malpha;
      while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours){
        tmp[k] += GetValue3D(Z_tildearray,m,k,GetValueInt(grapharray,j,nn));
        nn++;
      }
    }
    Emax = 0;
    for (k=0;k<K;k++){
      energy[k] = 0;
      extern_field = alpha[k];
      local_energy = GetValue(Betaarray,m,0) * tmp[k];
      energy[k] += local_energy + extern_field;
      if (energy[k] > Emax) Emax = energy[k];
      Gauss[k] = pow( normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,k), sqrt(GetValue(sigma_Marray,m,k)) ) , GetValue(p_Warray,m,1));
      Gauss[k] *= pow( normpdf(GetValue(m_Aarray,j,m),GetValue(mu_Marray,m,0), sqrt(GetValue(sigma_Marray,m,0)) ) , GetValue(p_Warray,m,0));
    }
    Sum = 0;
    for (k=0;k<K;k++){
      Probas[k] = exp( energy[k] - Emax );
      Sum += Probas[k];
    }
    for (k=0;k<K;k++){
      GetValue3D(q_Zarray,m,k,j) = Gauss[k]*Probas[k] / (Sum + eps);
    }
    Sum = 0;
    for (k=0;k<K;k++){
      Sum += GetValue3D(q_Zarray,m,k,j);
    }
    for (k=0;k<K;k++){
      GetValue3D(q_Zarray,m,k,j) /= Sum;
    }
  } 
}
Py_DECREF(grapharray);
Py_DECREF(q_Zarray);
Py_DECREF(Z_tildearray);
Py_DECREF(m_Aarray);
Py_DECREF(mu_Marray);
Py_DECREF(Sigma_Aarray);
Py_DECREF(Betaarray);
Py_DECREF(sigma_Marray);
Py_INCREF(Py_None);
  return Py_None;
}
 
static PyObject *UtilsC_expectation_Ptilde_Likelihood(PyObject *self, PyObject *args)
{
    PyObject *y_tilde,*m_A,*m_H,*X,*Sigma_A,*sigma_epsilone,*Sigma_H,*Gamma,*p_W,*XGamma;
    PyArrayObject *y_tildearray,*m_Aarray,*m_Harray,*Xarray,*Sigma_Aarray,*sigma_epsilonearray,*Sigma_Harray,*Gammaarray,*p_Warray,*XGammaarray;
    int j,J,m,m2,d,D,nCond,Nrep,nr;
    npy_float64 Det_Gamma;
    PyArg_ParseTuple(args, "OOOOOOOOOOiiiid", &y_tilde,&m_A,&m_H,&X,&Sigma_A,&sigma_epsilone,&Sigma_H,&Gamma,&p_W,&XGamma,&J,&D,&nCond,&Nrep,&Det_Gamma);
    y_tildearray   = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
    m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
    m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
    Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
    Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
    sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
    Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
    Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
    p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
    XGammaarray = (PyArrayObject *) PyArray_ContiguousFromObject(XGamma,PyArray_FLOAT64,3,3);
    
    npy_float64 Const;
    Const = - (0.5*Nrep*J*log(2*pi)) + 0.5*J*log(Det_Gamma + eps_FreeEnergy);

    npy_float64 *SSigma_H,*Sigma_A0,*H,*tmp,*tmpD,*tmpDD,*tmpNrep,*tmpNrep2,*tmpT,*GGamma;
    int *XX,*XXT;
    SSigma_H = malloc(sizeof(npy_float64)*D*D);
    Sigma_A0 = malloc(sizeof(npy_float64)*nCond*nCond);
    H = malloc(sizeof(npy_float64)*D);
    tmp = malloc(sizeof(npy_float64)*Nrep*D);
    tmpD = malloc(sizeof(npy_float64)*D);
    tmpDD = malloc(sizeof(npy_float64)*D*D);
    tmpNrep = malloc(sizeof(npy_float64)*Nrep);
    tmpNrep2 = malloc(sizeof(npy_float64)*Nrep);
    tmpT = malloc(sizeof(npy_float64)*Nrep*D);
    GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
    XX = malloc(sizeof(int)*D*Nrep);
    XXT = malloc(sizeof(int)*D*Nrep);
    
    for (d=0;d<Nrep;d++){
        for (nr=0;nr<Nrep;nr++){
            GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
        }
    }
    for (d=0;d<D;d++){
        H[d] = GetValue1D(m_Harray,d);
        for (nr=0;nr<D;nr++){
            SSigma_H[d*D+nr] = GetValue(Sigma_Harray,d,nr);
        }
    }
    for (m=0;m<nCond;m++){
        for (m2=0;m2<nCond;m2++){
            for (d=0;d<D;d++){
                for (nr=0;nr<Nrep;nr++){
                    XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
                    XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
                }
            }
            prodMat4(H,XXT,tmpNrep,1,Nrep,D);
            prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
            prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
            Sigma_A0[m*nCond + m2] = prodMatScal(tmpD,H,D);
            prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
            prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
            prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
            Sigma_A0[m*nCond + m2] += traceMat(tmpDD,D);
        }
    }
    
    npy_float64 val1, val2, part_1 = 0.0;
    for (j=0;j<J;j++){
        if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
        for (m=0;m<nCond;m++){
            for (m2=0;m2<nCond;m2++){
                val1 = GetValue3D(Sigma_Aarray,m,m2,j) + GetValue(m_Aarray,j,m2) * GetValue(m_Aarray,j,m);
                val2 = val1 * GetValue(p_Warray,m,1) * GetValue(p_Warray,m2,1) * Sigma_A0[m*nCond + m2];
                part_1 += val2/GetValue1D(sigma_epsilonearray,j);
            }
        }
    }
    
    npy_float64 *tmp2, *yy_tilde, *Y_bar_tilde;
    tmp2 = malloc(sizeof(npy_float64)*Nrep*D);
    yy_tilde = malloc(sizeof(npy_float64)*Nrep);
    Y_bar_tilde = malloc(sizeof(npy_float64)*D);

    for (d=0;d<D;d++){
        Y_bar_tilde[d] = 0;
    }
    for (j=0;j<J;j++){
        if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
        for (nr=0;nr<Nrep;nr++){
            yy_tilde[nr] = GetValue(y_tildearray,nr,j);
            for (d=0;d<D;d++){   
                tmp2[nr*D + d] = 0;
            }
        }
        for (m=0;m<nCond;m++){
            for (nr=0;nr<Nrep;nr++){  
                for (d=0;d<D;d++){
                    tmp2[d*Nrep + nr] += GetValue3D(XGammaarray,m,d,nr) * GetValue(m_Aarray,j,m) * GetValue(p_Warray,m,1) / GetValue1D(sigma_epsilonearray,j);
                }
            }
        }
        prodaddMat(tmp2,yy_tilde,Y_bar_tilde,D,1,Nrep);
    }
    npy_float64 part_2 = 0.0;
    for (d=0;d<D;d++){
        part_2 += H[d]*Y_bar_tilde[d];
    }
    
    npy_float64 *yGamma;
    yGamma = malloc(sizeof(npy_float64)*Nrep);
    npy_float64 part_3 = 0.0;
    for (j=0;j<J;j++){
        if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
        for (nr=0;nr<Nrep;nr++){
            yy_tilde[nr] = GetValue(y_tildearray,nr,j);
        }
        
        prodMat2(yy_tilde,GGamma,yGamma,1,Nrep,Nrep);
        part_3 += prodMatScal(yGamma,yy_tilde,Nrep)/GetValue1D(sigma_epsilonearray,j);
    }
    
    npy_float64 part_4 = 0.0;
    for (j=0;j<J;j++){
        if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
        part_4 += 0.5*log(GetValue1D(sigma_epsilonearray,j));
    } 

    npy_float64 EPtilde1;
    EPtilde1 = Const - Nrep*part_4 - 0.5*part_1 + part_2 - 0.5*part_3;
//     printf("Const = %f, part_1 = %f,    part_2 = %f,    part_3 = %f, part_4 =%f\n",Const,part_1,part_2,part_3,part_4);
    
    free(SSigma_H);
    free(Sigma_A0);
    free(H);
    free(tmp);
    free(tmpD);
    free(tmpDD);
    free(tmpNrep);
    free(tmpNrep2);
    free(tmpT);
    free(GGamma);
    free(XX);
    free(XXT);
    free(tmp2);
    free(yy_tilde);
    free(Y_bar_tilde);
    free(yGamma);
    
    Py_DECREF(y_tildearray);
    Py_DECREF(m_Aarray);
    Py_DECREF(m_Harray);
    Py_DECREF(Xarray);
    Py_DECREF(Sigma_Aarray);
    Py_DECREF(sigma_epsilonearray);
    Py_DECREF(Sigma_Harray);
    Py_DECREF(Gammaarray);
    Py_DECREF(p_Warray);
    Py_DECREF(XGammaarray);
    
    Py_INCREF(Py_None);
    
    return Py_BuildValue("d", EPtilde1);   
}

static PyObject *UtilsC_expectation_Ptilde_Likelihood_RVM(PyObject *self, PyObject *args)
{
    PyObject *y_tilde,*m_A,*m_H,*X,*Sigma_A,*sigma_epsilone,*Sigma_H,*Gamma,*m_W,*v_W,*XGamma;
    PyArrayObject *y_tildearray,*m_Aarray,*m_Harray,*Xarray,*Sigma_Aarray,*sigma_epsilonearray,*Sigma_Harray,*Gammaarray,*m_Warray,*v_Warray,*XGammaarray;
    int j,J,m,m2,d,D,nCond,Nrep,nr;
    npy_float64 Det_Gamma;
    PyArg_ParseTuple(args, "OOOOOOOOOOOiiiid", &y_tilde,&m_A,&m_H,&X,&Sigma_A,&sigma_epsilone,&Sigma_H,&Gamma,&m_W,&v_W,&XGamma,&J,&D,&nCond,&Nrep,&Det_Gamma);
    y_tildearray   = (PyArrayObject *) PyArray_ContiguousFromObject(y_tilde,PyArray_FLOAT64,2,2);
    m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
    m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
    Xarray   = (PyArrayObject *) PyArray_ContiguousFromObject(X,PyArray_INT,3,3);
    Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
    sigma_epsilonearray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_epsilone,PyArray_FLOAT64,1,1);
    Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
    Gammaarray   = (PyArrayObject *) PyArray_ContiguousFromObject(Gamma,PyArray_FLOAT64,2,2);
    m_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(m_W,PyArray_FLOAT64,1,1);
    v_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(v_W,PyArray_FLOAT64,2,2);
    XGammaarray = (PyArrayObject *) PyArray_ContiguousFromObject(XGamma,PyArray_FLOAT64,3,3);
    
    npy_float64 Const;
    Const = - (0.5*Nrep*J*log(2*pi)) + 0.5*J*log(Det_Gamma + eps_FreeEnergy);
    
    npy_float64 *SSigma_H,*Sigma_A0,*H,*tmp,*tmpD,*tmpDD,*tmpNrep,*tmpNrep2,*tmpT,*GGamma;
    int *XX,*XXT;
    SSigma_H = malloc(sizeof(npy_float64)*D*D);
    Sigma_A0 = malloc(sizeof(npy_float64)*nCond*nCond);
    H = malloc(sizeof(npy_float64)*D);
    tmp = malloc(sizeof(npy_float64)*Nrep*D);
    tmpD = malloc(sizeof(npy_float64)*D);
    tmpDD = malloc(sizeof(npy_float64)*D*D);
    tmpNrep = malloc(sizeof(npy_float64)*Nrep);
    tmpNrep2 = malloc(sizeof(npy_float64)*Nrep);
    tmpT = malloc(sizeof(npy_float64)*Nrep*D);
    GGamma = malloc(sizeof(npy_float64)*Nrep*Nrep);
    XX = malloc(sizeof(int)*D*Nrep);
    XXT = malloc(sizeof(int)*D*Nrep);
    
    for (d=0;d<Nrep;d++){
        for (nr=0;nr<Nrep;nr++){
            GGamma[d*Nrep + nr] = GetValue(Gammaarray,nr,d);
        }
    }
    for (d=0;d<D;d++){
        H[d] = GetValue1D(m_Harray,d);
        for (nr=0;nr<D;nr++){
            SSigma_H[d*D+nr] = GetValue(Sigma_Harray,d,nr);
        }
    }
    for (m=0;m<nCond;m++){
        for (m2=0;m2<nCond;m2++){
            for (d=0;d<D;d++){
                for (nr=0;nr<Nrep;nr++){
                    XX[nr*D + d] = GetValue3DInt(Xarray,m2,nr,d);
                    XXT[d*Nrep + nr] = GetValue3DInt(Xarray,m,nr,d);
                }
            }
            prodMat4(H,XXT,tmpNrep,1,Nrep,D);
            prodMat2(tmpNrep,GGamma,tmpNrep2,1,Nrep,Nrep);
            prodMat4(tmpNrep2,XX,tmpD,1, D, Nrep);
            Sigma_A0[m*nCond + m2] = prodMatScal(tmpD,H,D);
            prodMat4(SSigma_H,XXT,tmp,D,Nrep,D);
            prodMat2(tmp,GGamma,tmpT,D,Nrep,Nrep);
            prodMat4(tmpT,XX,tmpDD,D, D, Nrep);
            Sigma_A0[m*nCond + m2] += traceMat(tmpDD,D);
        }
    }
    
    npy_float64 val1, val2, part_1 = 0.0;
    for (j=0;j<J;j++){
        if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
        for (m=0;m<nCond;m++){
            for (m2=0;m2<nCond;m2++){
                val1 = GetValue3D(Sigma_Aarray,m,m2,j) + GetValue(m_Aarray,j,m2) * GetValue(m_Aarray,j,m);
                val2 = val1 * ( GetValue1D(m_Warray,m)*GetValue1D(m_Warray,m2) + GetValue(v_Warray,m,m2) ) * Sigma_A0[m*nCond + m2];
                part_1 += val2/GetValue1D(sigma_epsilonearray,j);
            }
        }
    }
    
    npy_float64 *tmp2, *yy_tilde, *Y_bar_tilde;
    tmp2 = malloc(sizeof(npy_float64)*Nrep*D);
    yy_tilde = malloc(sizeof(npy_float64)*Nrep);
    Y_bar_tilde = malloc(sizeof(npy_float64)*D);
    
    for (d=0;d<D;d++){
        Y_bar_tilde[d] = 0;
    }
    for (j=0;j<J;j++){
        if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
        for (nr=0;nr<Nrep;nr++){
            yy_tilde[nr] = GetValue(y_tildearray,nr,j);
            for (d=0;d<D;d++){   
                tmp2[nr*D + d] = 0;
            }
        }
        for (m=0;m<nCond;m++){
            for (nr=0;nr<Nrep;nr++){  
                for (d=0;d<D;d++){
                    tmp2[d*Nrep + nr] += GetValue3D(XGammaarray,m,d,nr) * GetValue(m_Aarray,j,m) * GetValue1D(m_Warray,m) / GetValue1D(sigma_epsilonearray,j);
                }
            }
        }
        prodaddMat(tmp2,yy_tilde,Y_bar_tilde,D,1,Nrep);
    }
    npy_float64 part_2 = 0.0;
    for (d=0;d<D;d++){
        part_2 += H[d]*Y_bar_tilde[d];
    }
    
    npy_float64 *yGamma;
    yGamma = malloc(sizeof(npy_float64)*Nrep);
    npy_float64 part_3 = 0.0;
    for (j=0;j<J;j++){
        if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
        for (nr=0;nr<Nrep;nr++){
            yy_tilde[nr] = GetValue(y_tildearray,nr,j);
        }
        
        prodMat2(yy_tilde,GGamma,yGamma,1,Nrep,Nrep);
        part_3 += prodMatScal(yGamma,yy_tilde,Nrep)/GetValue1D(sigma_epsilonearray,j);
    }
    
    npy_float64 part_4 = 0.0;
    for (j=0;j<J;j++){
        if (GetValue1D(sigma_epsilonearray,j) < eps) GetValue1D(sigma_epsilonearray,j) = eps;
        part_4 += 0.5*log(GetValue1D(sigma_epsilonearray,j));
    } 
    
    npy_float64 EPtilde1;
    EPtilde1 = Const - Nrep*part_4 - 0.5*part_1 + part_2 - 0.5*part_3;
    //     printf("Const = %f, part_1 = %f,    part_2 = %f,    part_3 = %f, part_4 =%f\n",Const,part_1,part_2,part_3,part_4);
    
    free(SSigma_H);
    free(Sigma_A0);
    free(H);
    free(tmp);
    free(tmpD);
    free(tmpDD);
    free(tmpNrep);
    free(tmpNrep2);
    free(tmpT);
    free(GGamma);
    free(XX);
    free(XXT);
    free(tmp2);
    free(yy_tilde);
    free(Y_bar_tilde);
    free(yGamma);
    
    Py_DECREF(y_tildearray);
    Py_DECREF(m_Aarray);
    Py_DECREF(m_Harray);
    Py_DECREF(Xarray);
    Py_DECREF(Sigma_Aarray);
    Py_DECREF(sigma_epsilonearray);
    Py_DECREF(Sigma_Harray);
    Py_DECREF(Gammaarray);
    Py_DECREF(m_Warray);
    Py_DECREF(v_Warray);
    Py_DECREF(XGammaarray);
    
    Py_INCREF(Py_None);
    
    return Py_BuildValue("d", EPtilde1);   
}

static PyObject *UtilsC_expectation_Ptilde_A(PyObject *self, PyObject *args)
{
    PyObject *m_A,*Sigma_A,*p_W,*q_Z,*mu_MK,*sigma_MK;
    PyArrayObject *m_Aarray,*Sigma_Aarray,*p_Warray,*q_Zarray,*mu_MKarray,*sigma_MKarray;
    int j,J,m,nCond,k,K;
    PyArg_ParseTuple(args, "OOOOOOiii", &m_A,&Sigma_A,&p_W,&q_Z,&mu_MK,&sigma_MK,&J,&nCond,&K);
    m_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(m_A,PyArray_FLOAT64,2,2);
    Sigma_Aarray = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_A,PyArray_FLOAT64,3,3);
    p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
    q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
    mu_MKarray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_MK,PyArray_FLOAT64,2,2);
    sigma_MKarray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_MK,PyArray_FLOAT64,2,2);

    npy_float64 *val1,*val2,*part;
    npy_float64 EPtilde2 = 0.0;
    val1 = malloc(sizeof(npy_float64)*K);
    val2 = malloc(sizeof(npy_float64)*K);
    part = malloc(sizeof(npy_float64)*K);
    
    for(j=0;j<J;j++){
        for(m=0;m<nCond;m++){
            for(k=0;k<K;k++)
            {
                val1[k] = - log( sqrt( 2.*pi*GetValue(sigma_MKarray,m,k) ) );
                val2[k] = ( pow(GetValue(m_Aarray,j,m) - GetValue(mu_MKarray,m,k), 2) + GetValue3D(Sigma_Aarray,m,m,j) ) / ( 2.*GetValue(sigma_MKarray,m,k) );
                part[k] = val1[k] - val2[k];
            }
            EPtilde2 += ( 1. - GetValue3D(q_Zarray,m,1,j)*GetValue(p_Warray,m,1) ) * part[0] + GetValue3D(q_Zarray,m,1,j)*GetValue(p_Warray,m,1)* part[1];
        }
    }

    free(val1);
    free(val2);
    free(part);

    Py_DECREF(m_Aarray);
    Py_DECREF(Sigma_Aarray);
    Py_DECREF(p_Warray);
    Py_DECREF(q_Zarray);
    Py_DECREF(mu_MKarray);
    Py_DECREF(sigma_MKarray);
    
    Py_INCREF(Py_None);
    
    return Py_BuildValue("d", EPtilde2);
}
    
static PyObject *UtilsC_expectation_Ptilde_H(PyObject *self, PyObject *args)
{
    PyObject *R, *m_H, *Sigma_H;
    PyArrayObject *Rarray, *m_Harray, *Sigma_Harray;
    int d,D,nr;
    npy_float64 v_h, HRH = 0.0, Det_invR;
    PyArg_ParseTuple(args, "OOOidd", &R, &m_H, &Sigma_H, &D, &v_h, &Det_invR);
    
    Rarray = (PyArrayObject *) PyArray_ContiguousFromObject(R,PyArray_FLOAT64,2,2);
    m_Harray = (PyArrayObject *) PyArray_ContiguousFromObject(m_H,PyArray_FLOAT64,1,1);
    Sigma_Harray   = (PyArrayObject *) PyArray_ContiguousFromObject(Sigma_H,PyArray_FLOAT64,2,2);
    
    npy_float64 *Rarray2,*RH,*VhR,*H,*SSigma_H;
    Rarray2 = malloc(sizeof(npy_float64)*D*D);
    RH = malloc(sizeof(npy_float64)*D);
    VhR = malloc(sizeof(npy_float64)*D*D);
    H = malloc(sizeof(npy_float64)*D);
    SSigma_H = malloc(sizeof(npy_float64)*D*D);

    for (d=0;d<D;d++){
        for (nr=0;nr<D;nr++){
            Rarray2[d*D+nr] = GetValue(Rarray,d,nr);
        }
    }
    for (d=0;d<D;d++){
        H[d] = GetValue1D(m_Harray,d);
        for (nr=0;nr<D;nr++){
            SSigma_H[d*D+nr] = GetValue(Sigma_Harray,d,nr);
        }
    }
    
    prodMatVect(Rarray2,H,RH,D,D);
    for(d=0;d<D;d++){
            HRH += H[d]*RH[d];
    }
    prodMat2(SSigma_H,Rarray2,VhR,D,D,D);
    HRH += traceMat(VhR,D);
    
    npy_float64 Const;
    Const = -0.5*D*log(2*pi*v_h) - 0.5*log(Det_invR);
//     printf("log(Det_inv)R = %f\n",log(Det_invR));
    
    npy_float64 EPtilde3;
    
    EPtilde3 = Const - 0.5*v_h*HRH;

    free(Rarray2);
    free(RH);
    free(VhR);
    free(H);
    free(SSigma_H);
    
    Py_DECREF(Rarray);
    Py_DECREF(m_Harray);
    Py_DECREF(Sigma_Harray);
    
    Py_INCREF(Py_None);
    
    return Py_BuildValue("d", EPtilde3);
}
  
static PyObject *UtilsC_expectation_Ptilde_W_ParsiMod1(PyObject *self, PyObject *args)
{
    PyObject *q_Z, *p_W;
    PyArrayObject *q_Zarray, *p_Warray;
    int m,nCond,s,S,j,J,k,K;
    npy_float64 tau1, tau2;
    PyArg_ParseTuple(args, "OOiiiidd", &q_Z, &p_W, &nCond, &S, &J, &K, &tau1, &tau2);
    q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3);
    p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
    
    npy_float64 sum_log_term, alea, log_term, E_log_term,SUM_Q_Z_1;
    int SUM, lab;
    
    npy_float64 EPtilde4 = 0.0;
    
    // MC Step of S Iterations
    for(m=0;m<nCond;m++){
        sum_log_term = 0.;
        for(s=0;s<S;s++){
            SUM = 0;
            for(j=0;j<J;j++){
                alea = rand()/ (float)RAND_MAX;
                lab = K - 1;
                for(k=0;k<K-1;k++)
                {
                    if(alea <= GetValue3D(q_Zarray,m,k,j))
                        lab = 0;
                }
                SUM += lab;
            }
            if(SUM < tau2)
            {
                log_term = ( tau1 * (SUM - tau2)) - log( 1. + exp( tau1 * (SUM - tau2) ) );
            }
            else
            {
                log_term = - log( 1. + exp( - tau1 * (SUM - tau2) ) );
            }
            sum_log_term += log_term;
        } 
        E_log_term = sum_log_term/(float)S;
        
        SUM_Q_Z_1 = 0.0;
        for(j=0;j<J;j++){
            SUM_Q_Z_1 +=  GetValue3D(q_Zarray,m,1,j);
        }
        
        EPtilde4 += E_log_term - ( tau1 * (1 - GetValue(p_Warray,m,1)) * (SUM_Q_Z_1 - tau2) );  
    }
    
    Py_DECREF(q_Zarray);
    Py_DECREF(p_Warray);
    
    Py_INCREF(Py_None);
    
    return Py_BuildValue("d", EPtilde4);
}

static PyObject *UtilsC_expectation_Ptilde_W_ParsiMod3(PyObject *self, PyObject *args)
{
    
    PyObject *p_W, *mu_M;
    PyArrayObject *p_Warray, *mu_Marray;
    int m,nCond;
    npy_float64 tau1, tau2;
    PyArg_ParseTuple(args, "OOidd", &p_W, &mu_M, &nCond, &tau1, &tau2);

    p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
    mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2);
    
    npy_float64 EPtilde4 = 0.0;
    npy_float64 part_1 = 0.0, part_2;
    
    for(m=0;m<nCond;m++)
    {
        if( (tau1*(pow(GetValue(mu_Marray,m,1),2) - tau2)) <= 0.0 )
            part_1 = tau1*(pow(GetValue(mu_Marray,m,1),2) - tau2) - log( 1.0 + exp( tau1 * ( pow(GetValue(mu_Marray,m,1),2) - tau2 )) );
        if( (tau1*(pow(GetValue(mu_Marray,m,1),2) - tau2)) > 0.0 )
            part_1 = - log( 1.0 + exp( -tau1 * ( pow(GetValue(mu_Marray,m,1),2) - tau2 )) );

        part_2 = tau1 * ( pow(GetValue(mu_Marray,m,1),2) - tau2 ) * GetValue(p_Warray,m,0);
        EPtilde4 += (part_1 - part_2);
    }
    
    Py_DECREF(mu_Marray);
    Py_DECREF(p_Warray);
    
    Py_INCREF(Py_None);
    
    return Py_BuildValue("d", EPtilde4);
}

static PyObject *UtilsC_expectation_Ptilde_W_ParsiMod3_Cond(PyObject *self, PyObject *args)
{
    
    PyObject *p_W, *mu_M, *tau1, *tau2;
    PyArrayObject *p_Warray, *mu_Marray,*tau1_array, *tau2_array;
    int m,nCond;

    PyArg_ParseTuple(args, "OOiOO", &p_W, &mu_M, &nCond, &tau1, &tau2);
    
    p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
    mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2);
    tau1_array = (PyArrayObject *) PyArray_ContiguousFromObject(tau1,PyArray_FLOAT64,1,1);
    tau2_array = (PyArrayObject *) PyArray_ContiguousFromObject(tau2,PyArray_FLOAT64,1,1);
    
    npy_float64 EPtilde4 = 0.0;
    npy_float64 part_1 = 0.0, part_2;
    
    for(m=0;m<nCond;m++)
    {
        if( (GetValue1D(tau1_array,m)*(pow(GetValue(mu_Marray,m,1),2) - GetValue1D(tau2_array,m))) <= 0.0 )
            part_1 = GetValue1D(tau1_array,m)*(pow(GetValue(mu_Marray,m,1),2) - GetValue1D(tau2_array,m)) - log( 1.0 + exp( GetValue1D(tau1_array,m) * ( pow(GetValue(mu_Marray,m,1),2) - GetValue1D(tau2_array,m) )) );
        if( (GetValue1D(tau1_array,m)*(pow(GetValue(mu_Marray,m,1),2) - GetValue1D(tau2_array,m))) > 0.0 )
            part_1 = - log( 1.0 + exp( -GetValue1D(tau1_array,m) * ( pow(GetValue(mu_Marray,m,1),2) - GetValue1D(tau2_array,m) )) );
        
        part_2 = GetValue1D(tau1_array,m) * ( pow(GetValue(mu_Marray,m,1),2) - GetValue1D(tau2_array,m) ) * GetValue(p_Warray,m,0);
        EPtilde4 += (part_1 - part_2);
    }
    
    Py_DECREF(mu_Marray);
    Py_DECREF(p_Warray);
    
    Py_INCREF(Py_None);
    
    return Py_BuildValue("d", EPtilde4);
}

static PyObject *UtilsC_expectation_Ptilde_W_ParsiMod4(PyObject *self, PyObject *args)
{
    
    PyObject *p_W, *mu_M, *sigma_M;
    PyArrayObject *p_Warray, *mu_Marray, *sigma_Marray;
    int m,nCond;
    npy_float64 tau1, tau2;
    PyArg_ParseTuple(args, "OOOidd", &p_W, &mu_M, &sigma_M, &nCond, &tau1, &tau2);

    p_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(p_W,PyArray_FLOAT64,2,2);
    mu_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(mu_M,PyArray_FLOAT64,2,2);
    sigma_Marray = (PyArrayObject *) PyArray_ContiguousFromObject(sigma_M,PyArray_FLOAT64,2,2);
    
    npy_float64 EPtilde4 = 0.0, dKL, m1, v1, v0;
    npy_float64 part_1 = 0.0, part_2;
    
    for(m=0;m<nCond;m++)
    {
        m1 = GetValue(mu_Marray,m,1);
        v1 = GetValue(sigma_Marray,m,1);
        v0 = GetValue(sigma_Marray,m,0);
        
        dKL = 0.5*pow(m1,2)*(1./v1 + 1./v0) + pow(v1-v0,2)/(2.*v1*v0);
        
        if( (-tau1*(dKL - tau2)) >= 0.0 )
            part_1 = tau1*(dKL - tau2) - log( 1.0 + exp( tau1 * ( dKL - tau2 )) );
        if( (-tau1*(dKL - tau2)) < 0.0 )
            part_1 = - log( 1.0 + exp( -tau1 * ( dKL - tau2 )) );
        
        part_2 = tau1 * ( dKL - tau2 ) * GetValue(p_Warray,m,0);
        
        EPtilde4 += (part_1 - part_2); 
    }
    
    Py_DECREF(mu_Marray);
    Py_DECREF(p_Warray);
    
    Py_INCREF(Py_None);
    
    return Py_BuildValue("d", EPtilde4);
} 

static PyObject *UtilsC_expectation_Ptilde_W_RVM(PyObject *self, PyObject *args)
{
    PyObject *q_Z, *m_W, *v_W, *alpha_RVM;
    PyArrayObject *q_Zarray, *m_Warray, *v_Warray, *alpha_RVMarray;
    int m,nCond;
    PyArg_ParseTuple(args, "OOOiO", &q_Z, &m_W, &v_W, &nCond, &alpha_RVM);
    q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3);
    m_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(m_W,PyArray_FLOAT64,1,1);
    v_Warray = (PyArrayObject *) PyArray_ContiguousFromObject(v_W,PyArray_FLOAT64,2,2);
    alpha_RVMarray = (PyArrayObject *) PyArray_ContiguousFromObject(alpha_RVM,PyArray_FLOAT64,1,1);
    
    npy_float64 EPtilde4 = 0.0;
    
    for(m=0;m<nCond;m++){
        EPtilde4 += 0.5*log(GetValue1D(alpha_RVMarray,m)+eps_FreeEnergy) - 0.5*log(2.*pi) - 0.5*GetValue1D(alpha_RVMarray,m)*( pow(GetValue1D(m_Warray,m),2) + GetValue(v_Warray,m,m) );
    }
    
    Py_DECREF(q_Zarray);
    Py_DECREF(m_Warray);
    Py_DECREF(v_Warray);
    Py_DECREF(alpha_RVMarray);
    
    Py_INCREF(Py_None);
    
    return Py_BuildValue("d", EPtilde4);
}


static PyObject *UtilsC_expectation_Ptilde_Z_MF_Begin(PyObject *self, PyObject *args)
{
    PyObject *q_Z,*graph, *Beta;
    PyArrayObject *q_Zarray, *grapharray, *Betaarray;
    int nn, j, J, k, K, m, nCond, maxNeighbours;
    
    PyArg_ParseTuple(args, "OOOiiii", &q_Z, &graph, &Beta, &J, &K, &nCond, &maxNeighbours);
    
    q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
    grapharray = (PyArrayObject *) PyArray_ContiguousFromObject(graph,PyArray_INT,2,2); 
    Betaarray = (PyArrayObject *) PyArray_ContiguousFromObject(Beta,PyArray_FLOAT64,1,1);
    
    npy_float64 *tmp;
    tmp = malloc(sizeof(npy_float64)*K);
    
    npy_float64 Beta_m;
    
    npy_float64 EPtilde5 = 0.0;
    npy_float64 tmp2, part_1, part_2;
    
    for(m=0;m<nCond;m++){
        Beta_m = GetValue(Betaarray,m,0);
        for(j=0;j<J;j++){
            part_1 = 0.0;
            tmp2 = 0.0;
            for(k=0;k<K;k++){
                tmp[k]=0.0;
                nn = 0;
                while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours){
                    tmp[k] += GetValue3D(q_Zarray,m,k,GetValueInt(grapharray,j,nn));
                    nn++;
                }
                part_1 += GetValue3D(q_Zarray,m,k,j) * tmp[k];
                tmp2 += exp(Beta_m * tmp[k]);
            }
            part_2 = Beta_m * part_1;
            EPtilde5 += part_2 - log(tmp2);
        }
    }
    
    free(tmp);
    
    Py_DECREF(q_Zarray);
    Py_DECREF(grapharray);
    Py_DECREF(Betaarray);
    
    Py_INCREF(Py_None);
    
    return Py_BuildValue("d", EPtilde5);
}
        
static PyObject *UtilsC_expectation_Ptilde_Z(PyObject *self, PyObject *args)
{
    PyObject *q_Z,*graph, *Beta;
    PyArrayObject *q_Zarray, *grapharray, *Betaarray;
    int nn, j, J, k, K, m, nCond, maxNeighbours;
    
    PyArg_ParseTuple(args, "OOOiiii", &q_Z, &graph, &Beta, &J, &K, &nCond, &maxNeighbours);
    
    q_Zarray = (PyArrayObject *) PyArray_ContiguousFromObject(q_Z,PyArray_FLOAT64,3,3); 
    grapharray = (PyArrayObject *) PyArray_ContiguousFromObject(graph,PyArray_INT,2,2); 
    Betaarray = (PyArrayObject *) PyArray_ContiguousFromObject(Beta,PyArray_FLOAT64,1,1);
    
    npy_float64 Beta_m, part_1, part_2, part_3, part_4;
    
    npy_float64 EPtilde5 = 0.0;
    
    npy_float64 *Pmf, *SUM_Q_Z_neighbours, *tmp, *tmp2;
    Pmf = malloc(sizeof(npy_float64)*K*J);
    SUM_Q_Z_neighbours = malloc(sizeof(npy_float64)*K*J);
    tmp = malloc(sizeof(npy_float64)*K);
    tmp2 = malloc(sizeof(npy_float64)*K);
    
    for(m=0;m<nCond;m++){
        Beta_m = GetValue(Betaarray,m,0);
        for(j=0;j<J;j++){
            for(k=0;k<K;k++){
                SUM_Q_Z_neighbours[k*J + j] = 0.0;
            }
        }
        
        for(j=0;j<J;j++){
            for(k=0;k<K;k++){
                nn = 0;
                while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours){
                    SUM_Q_Z_neighbours[k*J + j] += GetValue3D(q_Zarray,m,k,GetValueInt(grapharray,j,nn));
                    nn++;
                }
            }
        }

        for(j=0;j<J;j++){
            for(k=0;k<K;k++)
            {
                Pmf[k*J+j] = Compute_Pmfj(Beta_m, SUM_Q_Z_neighbours[k*J+j], SUM_Q_Z_neighbours[j], SUM_Q_Z_neighbours[J+j]);
            }
        }
        
        for(j=0;j<J;j++){
            part_1 = 0.0;
            part_2 = 0.0;
            for(k=0;k<K;k++){
                tmp[k] = 0.0;
                tmp2[k] = 0.0;
                nn = 0;
                while(GetValueInt(grapharray,j,nn) != -1 && nn<maxNeighbours){
                        tmp[k] += GetValue3D(q_Zarray,m,k,GetValueInt(grapharray,j,nn));
                        tmp2[k] += Pmf[k*J + GetValueInt(grapharray,j,nn)];
                    
                    nn++;
                }
                part_1 += ( 0.5*GetValue3D(q_Zarray,m,k,j)*tmp[k] - 0.5*Pmf[k*J+j]*tmp2[k] + Pmf[k*J+j]*SUM_Q_Z_neighbours[k*J+j] );
                part_2 += exp( Beta_m*SUM_Q_Z_neighbours[k*J+j] );
            }
            part_3 = exp(Beta_m*part_1);
            part_4 = log (part_3 / part_2);
            EPtilde5 += part_4;
        }
    }
    
    free(tmp);
    free(tmp2);
    free(Pmf);
    free(SUM_Q_Z_neighbours);
    
    Py_DECREF(q_Zarray);
    Py_DECREF(grapharray);
    Py_DECREF(Betaarray);
    
    Py_INCREF(Py_None);
    
    return Py_BuildValue("d", EPtilde5);
    
}         
 
static PyMethodDef UtilsMethods[] = {
    {"maximization_L", UtilsC_maximization_L, METH_VARARGS, "Maximization L"},
    {"maximization_LP", UtilsC_maximization_LP, METH_VARARGS, "Maximization L Parcellation"},
    {"maximization_L_ParsiMod", UtilsC_maximization_L_ParsiMod, METH_VARARGS, "Maximization L ParsiMod"},
    {"maximization_L_ParsiMod_RVM", UtilsC_maximization_L_ParsiMod_RVM, METH_VARARGS, "Maximization L ParsiMod RVM"},
    {"maximization_beta", UtilsC_maximization_beta, METH_VARARGS, "Maximization Beta"},
    {"maximization_beta_CB", UtilsC_maximization_beta_CB, METH_VARARGS, "Maximization Beta CB"},
    {"expectation_Z", UtilsC_expectation_Z, METH_VARARGS, "Expectation Z"},
    {"expectation_Z_MF_ParsiMod_3", UtilsC_expectation_Z_MF_ParsiMod_3, METH_VARARGS, "Expectation Z MF ParsiMod3"},
    {"expectation_Z_ParsiMod_1", UtilsC_expectation_Z_ParsiMod_1, METH_VARARGS, "Expectation Z ParsiMod 1"},
    {"expectation_Z_ParsiMod_1_MeanLabels", UtilsC_expectation_Z_ParsiMod_1_MeanLabels, METH_VARARGS, "Expectation Z ParsiMod 1 *Mean Labels*"},
    {"expectation_Z_ParsiMod_2", UtilsC_expectation_Z_ParsiMod_2, METH_VARARGS, "Expectation Z ParsiMod2 2"},
    {"expectation_Z_ParsiMod_3", UtilsC_expectation_Z_ParsiMod_3, METH_VARARGS, "Expectation Z ParsiMod2 3"},
    {"expectation_Z_ParsiMod_RVM_and_CompMod", UtilsC_expectation_Z_ParsiMod_RVM_and_CompMod, METH_VARARGS, "Expectation Z ParsiMod_RVM and CompMod"},
    {"expectation_H", UtilsC_expectation_H, METH_VARARGS, "Expectation H"},
    {"expectation_H_ParsiMod", UtilsC_expectation_H_ParsiMod, METH_VARARGS, "Expectation H ParsiMod"},
    {"expectation_H_ParsiMod_RVM", UtilsC_expectation_H_ParsiMod_RVM, METH_VARARGS, "Expectation H ParsiMod RVM"},
    {"expectation_A", UtilsC_expectation_A, METH_VARARGS, "Expectation A"},
    {"expectation_AP", UtilsC_expectation_AP, METH_VARARGS, "Expectation A Parcellation"},
    {"expectation_A_ParsiMod", UtilsC_expectation_A_ParsiMod, METH_VARARGS, "Expectation A ParsiMod"},
    {"expectation_A_ParsiMod_RVM", UtilsC_expectation_A_ParsiMod_RVM, METH_VARARGS, "Expectation A ParsiMod RVM"},
    {"maximization_sigma_noise", UtilsC_maximization_sigma_noise, METH_VARARGS, "Maximization Noise Variance"},
    {"maximization_sigma_noiseP", UtilsC_maximization_sigma_noiseP, METH_VARARGS, "Maximization Noise Variance Parcellation"},
    {"maximization_sigma_noise_ParsiMod", UtilsC_maximization_sigma_noise_ParsiMod, METH_VARARGS, "Maximization Noise Variance ParsiMod"},
    {"maximization_sigma_noise_ParsiMod_RVM", UtilsC_maximization_sigma_noise_ParsiMod_RVM, METH_VARARGS, "Maximization Noise Variance ParsiMod RVM"},
    {"expectation_W_ParsiMod_1", UtilsC_expectation_W_ParsiMod_1, METH_VARARGS, "Expectation W ParsiMod 1"},
    {"expectation_W_ParsiMod_1_MeanLabels", UtilsC_expectation_W_ParsiMod_1_MeanLabels, METH_VARARGS, "Expectation W ParsiMod 1 *Mean Labels*"},
    {"expectation_W_ParsiMod_2", UtilsC_expectation_W_ParsiMod_2, METH_VARARGS, "Expectation W ParsiMod 2"},
    {"expectation_W_ParsiMod_3", UtilsC_expectation_W_ParsiMod_3, METH_VARARGS, "Expectation W ParsiMod 3"},
    {"expectation_W_ParsiMod_3_Cond", UtilsC_expectation_W_ParsiMod_3_Cond, METH_VARARGS, "Expectation W ParsiMod 3 Cond"},
    {"expectation_W_ParsiMod_4", UtilsC_expectation_W_ParsiMod_4, METH_VARARGS, "Expectation W ParsiMod 4"},
    {"expectation_W_ParsiMod_RVM", UtilsC_expectation_W_ParsiMod_RVM, METH_VARARGS, "Expectation W ParsiMod RVM"},
    {"expectation_Ptilde_Likelihood", UtilsC_expectation_Ptilde_Likelihood, METH_VARARGS, "Expectation Ptilde Likelihood"},
    {"expectation_Ptilde_Likelihood_RVM", UtilsC_expectation_Ptilde_Likelihood_RVM, METH_VARARGS, "Expectation Ptilde Likelihood RVM"},
    {"expectation_Ptilde_A", UtilsC_expectation_Ptilde_A, METH_VARARGS, "Expectation Ptilde A"},
    {"expectation_Ptilde_H", UtilsC_expectation_Ptilde_H, METH_VARARGS, "Expectation Ptilde H"},
    {"expectation_Ptilde_W_ParsiMod1", UtilsC_expectation_Ptilde_W_ParsiMod1, METH_VARARGS, "Expectation Ptilde W ParsiMod1"},
    {"expectation_Ptilde_W_ParsiMod3", UtilsC_expectation_Ptilde_W_ParsiMod3, METH_VARARGS, "Expectation Ptilde W ParsiMod3"},
    {"expectation_Ptilde_W_ParsiMod3_Cond", UtilsC_expectation_Ptilde_W_ParsiMod3_Cond, METH_VARARGS, "Expectation Ptilde W ParsiMod3 Cond"},
    {"expectation_Ptilde_W_ParsiMod4", UtilsC_expectation_Ptilde_W_ParsiMod4, METH_VARARGS, "Expectation Ptilde W ParsiMod4"},
    {"expectation_Ptilde_W_RVM", UtilsC_expectation_Ptilde_W_RVM, METH_VARARGS, "Expectation Ptilde W RVM"},
    {"expectation_Ptilde_Z", UtilsC_expectation_Ptilde_Z, METH_VARARGS, "Expectation Ptilde Z"},
    {"expectation_Ptilde_Z_MF_Begin", UtilsC_expectation_Ptilde_Z_MF_Begin, METH_VARARGS, "Expectation Ptilde Z MF Begin"},
    {NULL, NULL, 0, NULL}
};

void initUtilsC(void){
    (void) Py_InitModule("UtilsC", UtilsMethods);
    import_array();
}


// void transMat(float A[],int M,int N)  {
//   int i,j;
//   float tmp;
//   for (i = 0; i < M; i++) {
//     for (j = i + 1; j < N; j++) {
//       tmp = A[i*N + j];
//       A[i*N+j] = A[j*M+i];
//       A[j*M+i] = tmp;
//     }
//   }
// }
