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

long difftimeval(struct timeval* t1, struct timeval* t2){
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

void prodaddMat(npy_float64 *A,npy_float64 *B,npy_float64 *C,int M, int N, int K){
    // A: MxK, B:KxN, C:MxN
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
    // A: MxK, B:KxN, C:MxN
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
    // npy_float64 tmp;
    for (i = 0; i < M; i++) {
        C[i] = 0;
        for (j = 0; j < N; j++) {
            C[i] += A[i*N+j] * B[j];
        }
    }
}

void prodMat3(int *A,npy_float64 *B,npy_float64 *C,int M, int N, int K){
    // C = A*B where A is a matrix of integers
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
    // C = A*B where B is a matrix of integers
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

void addMatfast(npy_float64 *A, npy_float64 *B,int M,int N){
    // B = A+B          
    int i, j;   
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            B[i*N+j] += A[i*N+j];
        }
    }
}
  
void SubMatfastVect(npy_float64 *A, npy_float64 *B,int M){
    // B = B - A          
    int i;   
    for (i = 0; i < M; i++) {
        B[i] = B[i] - A[i];
    }
}

  

npy_float64 Compute_Pmfj(npy_float64 Beta_m, npy_float64 SUM_Q_Z_neighbours_qjm, npy_float64 SUM_Q_Z_neighbours_0, npy_float64 SUM_Q_Z_neighbours_1){
    npy_float64 numer, denom, Pmfj;
    denom = exp(Beta_m * SUM_Q_Z_neighbours_0) + exp(Beta_m * SUM_Q_Z_neighbours_1);
    numer = exp(Beta_m * SUM_Q_Z_neighbours_qjm);
    Pmfj = numer / denom;
    if(Pmfj > 1.0)
        printf("NOT OK : Pmfj is higher than 1 ...\n");
    return Pmfj;
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
      
 
static PyMethodDef UtilsMethods[] = {
    {"maximization_L", UtilsC_maximization_L, METH_VARARGS, "Maximization L"},
    {"maximization_beta", UtilsC_maximization_beta, METH_VARARGS, "Maximization Beta"},
    {"expectation_Z", UtilsC_expectation_Z, METH_VARARGS, "Expectation Z"},
    {"expectation_H", UtilsC_expectation_H, METH_VARARGS, "Expectation H"},
    {"expectation_A", UtilsC_expectation_A, METH_VARARGS, "Expectation A"},
    {"maximization_sigma_noise", UtilsC_maximization_sigma_noise, METH_VARARGS, "Maximization Noise Variance"},
    {NULL, NULL, 0, NULL}
};

void initUtilsC(void){
    (void) Py_InitModule("moduleC_asl", UtilsMethods);
    import_array();
}

