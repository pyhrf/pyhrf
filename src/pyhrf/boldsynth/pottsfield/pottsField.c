#include <stdio.h>
#include <stdlib.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <assert.h>

#define debug 0

#define TRUE 1
#define FALSE 0

#define ROUND(x) ( x-(int)(x) >= 0.5 ? ((int)(x))+1 : (int)(x)) 
#ifndef MAX
#define MAX(x,y) ((x) < (y) ? (y) : (x))
#endif
#ifndef MIN
#define MIN(x,y) ((x) > (y) ? (y) : (x))
#endif
#define CARRE(x) ((x)*(x))

#define NB_VOISINS_MAX 4



/*
  Grid coordinates handling
*/
#define COR_COORD(x,n) (x<0 ? (-(x)/n+1)*n+x : (x)%n)

void getVoisinsTorus(int i, int j, int width, int height, int* voisins, int* nbVoisins)
{
  voisins[0] = COR_COORD(i-1,height)*width + COR_COORD(j,width);

  voisins[1] = COR_COORD(i,height)*width + COR_COORD(j+1,width);

  voisins[2] = COR_COORD(i+1,height)*width + COR_COORD(j,width);

  voisins[3] = COR_COORD(i,height)*width + COR_COORD(j-1,width);
  
  *nbVoisins = 4;
}

void getVoisinsNoTorus(int i, int j, int width, int height, int* voisins, int* nbVoisins)
{
  int iv = 0;
  if (i>0) {
    // Voisin du haut :
    voisins[iv] = (i-1)*width + j;
    iv++;
  }
  if (i<height-1) {
    // voisin du bas :
    voisins[iv] = (i+1)*width + j;
    iv++;
  }

  if (j>0) {
    // voisin de gauche :
    voisins[iv] = i*width + j-1;
    iv++;
  }

  if (j<width-1) {
    // voisin de droite :
    voisins[iv] = i*width + j+1;
    iv++;
  }
  
  *nbVoisins = iv;

}




/***
    Definition de l'environnement de simulation
    !defenvsim 
***/ 

typedef struct T_envSimul {

  int width;
  int height;
  int nbSites;
  
  int* grille;
  void (*getVoisins)(int i, int j, int width, int height, int* voisins, int* nbVoisins);
  
  int nbEtats;

  double beta;
  double* fieldForce;
  int constJ;

  int bitRes;
  int lvMax;

  double*** kernel;
  double*** energie;

} EnvSimul;


void initEnvSimul(EnvSimul* e, int w, int h, int torusModeON, int constJ, double* fieldForce, int* dataIni, int nbEtats)
{
  int bitRes, tmpBitRes;
  int lvMax;
  int i,v; //j, idx;

  e->constJ = constJ;
  e->fieldForce = fieldForce;
  
  if(torusModeON)
    {
      if (debug)
        printf("torus mode ON\n");
      e->getVoisins = getVoisinsTorus;
    }
  else
    {
      if (debug) 
        printf("torus mode OFF\n");
      e->getVoisins = getVoisinsNoTorus;
    }

  // Initialisation de la grille de simulation :
  e->width = w;
  e->height = h;
  e->nbSites = w*h;
  e->grille = dataIni;
  
  e->nbEtats = nbEtats;

  bitRes = 0;
  tmpBitRes = nbEtats-1;
  while(tmpBitRes){tmpBitRes>>=1; bitRes++;}
  //printf("nombre d'etats : %d -> bitRes = %d\n", nbEtats, bitRes);

  e->bitRes = bitRes;

  e->kernel = malloc(sizeof(double**)*(NB_VOISINS_MAX+1));
  e->energie = malloc(sizeof(double**)*(NB_VOISINS_MAX+1));

  lvMax = (1 << (bitRes*NB_VOISINS_MAX))-1;
  //printf("lvMax = %d \n", lvMax);
  e->lvMax = lvMax;

  for(v=0 ; v<=NB_VOISINS_MAX ; v++)
    {
      e->kernel[v] = malloc(sizeof(double*)*nbEtats);
      e->energie[v] = malloc(sizeof(double*)*nbEtats);
      for(i=0 ; i<nbEtats ; i++) 
	{
	  e->kernel[v][i] = malloc(sizeof(double)*(lvMax+1)); //TODO optimiser, lvMax change ...
	  e->energie[v][i] = malloc(sizeof(double)*(lvMax+1));
	}
    }

  //   printf("Allocation kernel OK !\n"); 
  //   fflush(stdout); 


}

/*  Initialisation d'un tour de simulation
  !initsimul
*/

void generateKernel(EnvSimul* env, int nbVoisins, double beta);

void initSimulation(EnvSimul* env, double beta)
{
  int nv;

  //Precalcul des proba de transition locales :
  //  printf("Precalcul du kernel ...\n"); 
  //  fflush(stdout);
  for (nv=0 ; nv <= NB_VOISINS_MAX ; nv++)
    generateKernel(env, nv, beta);

}

void generateKernel(EnvSimul* env, int nbVoisins, double beta)
{
  int lv = 0;
  int maxState = env->nbEtats-1;
  int bitRes = env->bitRes;
  int bitMask, stateNext, p=1;
  int qv[NB_VOISINS_MAX];
  double* energie = malloc(sizeof(double)*env->nbEtats);
  double* expEnergie = malloc(sizeof(double)*env->nbEtats);
  double sumExpEnergie;
  int v, energieV, q, qvTmp;

  bitMask = (1 << env->bitRes) -1;
/*   printf("bitMask = %d\n", bitMask); */
    
  while(p!=-1)
    {
      for (v=0 ; v<nbVoisins ; v++)
	{
	  qvTmp = ( lv >> (env->bitRes*v))&bitMask;
	  assert(lv<=env->lvMax);
/* 	  printf(" %d", qvTmp); */
/* 	  fflush(stdout); */
	  qv[v] = qvTmp;
	}
/*       printf("\n"); */
      sumExpEnergie = 0.0;
      for (q=0 ; q<env->nbEtats ; q++)
	{
	  energieV = 0.0;
	  for(v=0; v<nbVoisins ; v++)
	    energieV += env->nbEtats*(qv[v] == q)-1;

	  energie[q] = -env->constJ*energieV - env->fieldForce[q];
	  expEnergie[q] = exp(-beta*energie[q]);
	  /* 	      printf(" energie state %d -> energie=%1.2f, expEnergie=%1.2f\n", q, energie[q], expEnergie[q]); */
	  sumExpEnergie += expEnergie[q];
	}
      /* 	  printf("sumExpEnergie=%1.2f", sumExpEnergie); */
      /* 	  printf(" => \n"); */
      for (q=0 ; q<env->nbEtats ; q++)
	{
	  assert(nbVoisins<=NB_VOISINS_MAX);
	  assert(q<env->nbEtats);
	  assert(lv<=env->lvMax);
	  env->energie[nbVoisins][q][lv] = energie[q];
	  env->kernel[nbVoisins][q][lv] = expEnergie[q]/sumExpEnergie;
/* 	  printf(" ener[%d][%d][%d] = %1.4f\n", nbVoisins, q, lv, env->energie[nbVoisins][q][lv]);  */
/* 	  printf(" kern[%d][%d][%d] = %1.4f\n", nbVoisins, q, lv, env->kernel[nbVoisins][q][lv]);  */
	}


      if ( (lv&bitMask) < maxState ) lv ++;
      else {
	p = 1;
	while(p!=-1)
	  {
	    stateNext = (lv >> (bitRes*p)) & bitMask;
	    if (stateNext < maxState)
	      {
		lv += 1 << (bitRes*p);
		lv >>=  bitRes*p;
		lv <<=  bitRes*p;
		break;
	      }
	    else
	      p = p < nbVoisins-1 ? p+1 : -1;
	  }
      }
/*       printf("%03d ->", lv); */
/*       for (v=0 ; v<nbVoisins ; v++)  */
/* 	printf(" %d", (lv>>(bitRes*v))&bitMask); */
/*       printf("\n"); */
      
    }

  free(energie);
  free(expEnergie);
}


/***
    Fonction realisation un pas de simulation
    !dosimulstep
***/
void doSimulStep(EnvSimul* env)
{
  int* voisins = malloc(sizeof(int)*NB_VOISINS_MAX); //[NB_VOISINS_MAX];
  int nbVoisins;
  double rndm;
  int prevEtat, q, newEtat;
  int v, idx;
  double sumProba = 0.0;
  int lv;
  int i,j;

  // Tirage de la position à swapper :
  rndm = rand()/(double)RAND_MAX*(env->height-1);
  i = ROUND(rndm);
  rndm = rand()/(double)RAND_MAX*(env->width-1);
  j = ROUND(rndm);
  assert(i < env->height);
  assert(j < env->width);

  idx = i*env->width+j;
  prevEtat = env->grille[idx];
  
  //   printf("Considering : cell(%d, %d) -> q=%d\n",i,j, prevEtat);
/*   fflush(stdout); */

/*   printf("Extracting neighbours ...\n"); */
/*   fflush(stdout); */
  env->getVoisins(i,j,env->width, env->height, voisins, &nbVoisins);
  assert(nbVoisins <= NB_VOISINS_MAX);

  lv = 0;
/*   printf(" etat voisins -> "); */
  for (v=0 ; v<nbVoisins ; v++)
    {
/*       printf(" voisin : (%d, %d)\n", voisins[v], voisins[v+1]); */
/*       printf(" etat du voisin : %d ", env->grille[voisins[v]][voisins[v+1]]); */
      assert(voisins[v] < env->nbSites);

      lv |= env->grille[voisins[v]] << (env->bitRes*v);      

    }
  assert(lv <= env->lvMax);
/*   printf(" => lv=%d \n",lv); */
  sumProba = env->kernel[nbVoisins][0][lv];
  rndm = rand()/(double)RAND_MAX;
/*
  printf(" randm=%1.2f\n", rndm);  
  printf(" comparing with : ");   
*/
  for (q=0 ; q<env->nbEtats ; q++)
    {
      //       printf(" %1.2f ", sumProba);  
      if ( rndm <= sumProba)
	{
	  idx = i*env->width+j;
	  newEtat = env->grille[idx] = q;
	  //  	  printf(" ->OK for state %d\n", q);  
	  break;
	}
      else
	//	printf("not OK");
      sumProba += (q!=env->nbEtats-1? env->kernel[nbVoisins][q+1][lv] : 0.0);
/*
      printf("new sumProba : %g", sumProba);
      fflush(stdout);
*/
      assert(sumProba <= 1.00000001);
    }

  //printf("newEtat = %d\n", newEtat);
/*   printf("env->energie[%d][%d][%d] = %f\n",nbVoisins, prevEtat, lv, env->energie[nbVoisins][prevEtat][lv]); */
/*   printf("env->energie[%d][%d][%d] = %f\n",nbVoisins, newEtat, lv, env->energie[nbVoisins][newEtat][lv]); */
  
  free(voisins);
}

void genGridPottsInternal(int width, int height, int boundMode, double beta, double* fieldForce, int nbIt, int* dataIni, int nbEtats)
{
  int it;
  EnvSimul env;
  int constJ = 1;

  initEnvSimul(&env, width, height, boundMode, constJ, fieldForce, dataIni, nbEtats);
  initSimulation(&env, beta);
  for (it=0 ; it<nbIt ; it++)
    doSimulStep(&env);
}

static PyObject *genPottsField(PyObject *self, PyObject *arg) {
     
	double beta;
	int i, j;
	int nbIt;
	int nbEtats;
	int boundMode;
	PyObject* oData;
	PyArrayObject* arrayData;
	//char* data;
	int s0, height, width; //s1
	int* input;
	double* fieldForce;
	//PyArray_Descr	*typecode = PyArray_DescrNewFromType(PyArray_OBJECT);
    if (debug)
      printf("genPottsField ...\n");

	PyArg_ParseTuple(arg, "iiidiOi", &width, &height, &boundMode, &beta, &nbIt, &oData, &nbEtats);
	arrayData = (PyArrayObject*) PyArray_ContiguousFromObject(oData, 
                                                              PyArray_INT32, 
                                                              1, 1);
	//data = arrayData->data;
	s0 = arrayData->strides[0];
    if (debug) {
      printf("w=%d, h=%d\n", width, height);
      fflush(stdout);
    }
	input = malloc(sizeof(int*)*(height*width));
	for (i=0 ; i<height ; i++)
	  {
	    //input[i] = malloc(sizeof(int)*width);
	    for (j=0 ; j<width ; j++)
	      {
            input[i*width+j] = *(npy_int32*)(arrayData->data+(i*width+j)*s0);
            if (debug)
              printf(" %d", input[i*width+j]);
	      }
        if (debug)
          printf("\n");
	  }
    
    if (debug)
      printf(" beta=%f, ni=%d, boundM=%d, nbEtats=%d\n", beta, nbIt, 
             boundMode, nbEtats);
	fieldForce = calloc(nbEtats, sizeof(double));

	genGridPottsInternal(width, height, boundMode, beta, fieldForce, 
                         nbIt, input, nbEtats);

	//printf(" gen potts done !\n");


	for (i=0 ; i<height ; i++)
	  {
	    for (j=0 ; j<width ; j++)
	      {
		*(npy_int32*)(arrayData->data+(i*width+j)*s0) = input[i*width+j];
		//printf(" %d", input[i*width+j]);
	      }
	    //printf("\n");
	  }

	free(fieldForce);

	//Py_XDECRREF(arrayData);

    Py_DECREF(Py_None);
	return Py_None;
}


static PyMethodDef methods[] = {
	{"genPottsField", genPottsField, METH_VARARGS, "Generate a Potts Field"},
	{NULL, NULL, 0, NULL}
};

void  initpottsfield_c(void)
{
	Py_InitModule("pottsfield_c", methods);
	import_array();	
}

