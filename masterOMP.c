#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <emmintrin.h>
#include <malloc.h>

#define SM (CLS/sizeof(double))

void init_vect(double *M, int N)
{
  int j;
  // random numbers in the range [0.5, 1.5)

  for (j=0; j<N; j++)
    M[j] = 0.5 + (double)rand()/RAND_MAX;  
}

void init_mat(double *M, int N)
{
  int j, k;
  // random numbers in the range [0.5, 1.5)

  for (k=0; k<N; k++) 
    for (j=0; j<N; j++)
      M[k*N+j] = 0.5 + (double)rand()/RAND_MAX;
}

void zero_mat(double *M, int N)
{
  int j, k;

  for (k=0; k<N; k++) 
    for (j=0; j<N; j++)
      M[k*N+j] = 0.0;
}


double checksum_vect ( double *const c, int N )
{
  int i;
  double S= 0.0;

  for (i=0; i<N; i++)
    S += c[i];
  return S;
}

double checksum_mat ( double *const c, int N )
{
  int i, j;
  double S= 0.0;

  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      S += c[i*N+j];
  return S;
}

void f1_mat ( double *const x, double *const y, double *restrict a, int N )
{
  int i, j;

  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      a[i*N+j] = x[i] * y[j];
}

void f2_mat ( double *const x, double *const y, double *restrict a, int N )
{
  int i, j;
#pragma omp parallel for
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      a[i+N*j] = x[i] * y[j];
}


void f1_vect ( double *x, double r, int N )
{
  int i;

  for (i=0; i<N; i++)
    x[i] = x[i] / r;
}

void mult_mat ( double *const a, double *const b, double *restrict c, int N )
{
  int i, j, k;
  int i2,j2,k2;
  double *restrict a2; 
  double *restrict b2;
  double *restrict c2;
#pragma omp parallel for
  for (i=0;i<N; i+=SM)
    for(j=0;j<N;j+=SM)
      for(k=0;k<N;k+=SM)
        for(i2=0,c2=&c[i*N+j],a2=&a[i*N+k];i2<SM;++i2,c2+=N,a2+=N)
        {
            //printf("first inner\n");
          _mm_prefetch (&a2[8],_MM_HINT_NTA);
          printf("ok load a2\n");
          for(k2=0,b2=&b[k*N+j];k2<SM;++k2,b2+=N)
          {
            //printf("second inner\n");
            __m128d m1d = _mm_load_sd(&a2[k2]);
            m1d=_mm_unpacklo_pd (m1d,m1d);
            for(j2=0;j2<SM;j2+=2)
            {
                /*printf("third inner\n");
                printf("First pointer: %p\n", &b2[j2]);
                printf("Second pointer: %p\n", &b2[j2+1]);
                printf("Third pointer: %p\n", &b2[j2+2]);*/
                
              __m128d m2 = _mm_load_pd(&b2[j2]);
              //printf("first load\n");
              __m128d r2 = _mm_load_pd(&c2[j2]);
              //printf("second load\n");
              _mm_store_pd (&c2[j2],_mm_add_pd (_mm_mul_pd(m2,m1d),r2));
              //printf("store\n");
              //c2[j2] += a2[k2]*b2[j2];
            }
          }
        }
/*  double *T;
  T = (double *) malloc ( N*N*sizeof(double));
  zero_mat(T,N);
#pragma omp parallel
{
#pragma omp for
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
        T[i*N+j] = b [j*N+i];
  
#pragma omp for 
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      for (k=0; k<N; k++)
        c[i*N+j] += a[i*N+k] * T[j*N+k];//b[k*N+j];
}
//  free(T);*/

}

void mat_transpose (double *M, int N)
{
  int j, k;
  double T;

  for (k=0; k<N; k++) 
    for (j=k+1; j<N; j++) {
      T = M[k*N+j];
      M[k*N+j] = M[j*N+k];
      M[j*N+k] = T;
    }
}

//////// MAIN ////////////
int main (int argc, char **argv)
{
  int N=2000;
  printf("before attr\n");
  double *A; 
  double *B;
  double *C;
  double *X;
  double *Y;
  double R;
  int ok = 0;
  if (argc>1) {  N  = atoll(argv[1]); }
  if (N<1 || N>20000) {
     printf("input parameter: N (1-20000)\n");
     return 0;
  }

  // dynamic allocation of 2-D matrices
  /*A = (double *) _mm_malloc ( N*N*sizeof(double),64);
  B = (double *) _mm_malloc ( N*N*sizeof(double),64);
  C = (double *) _mm_malloc ( N*N*sizeof(double),64);*/
  ok += posix_memalign((void**)&A,64,N*N*sizeof(double));
  ok += posix_memalign((void**)&B,64,N*N*sizeof(double));
  ok += posix_memalign((void**)&C,64,N*N*sizeof(double));
  printf ("malloc ok\n");
  // Dynamic allocation of vectors
  /*X = (double *) _mm_malloc ( N*sizeof(double),64);
  Y = (double *) _mm_malloc ( N*sizeof(double),64);*/
  ok += posix_memalign((void**)&X,64,N*sizeof(double));
  ok += posix_memalign((void**)&Y,64,N*sizeof(double));
  
  if (ok > 0)
  {
    printf("posix memalign failed: %d\n",ok);
    exit(1);
  }
  // initial seed for random generation
  srand(1);

  // Initialize input data with random data
  init_vect (X, N);
  init_vect (Y, N);
  R=0.0;

  // Main computation
  f1_mat        (X, Y, A, N);
  f2_mat        (X, Y, B, N);
  R +=          checksum_vect (Y, N);
  f1_vect       (X, R, N);
  R +=          checksum_vect (X, N);
  zero_mat      (C, N);
  mult_mat      (A, B, C, N);
  mat_transpose (B, N);
  mult_mat      (B, A, C, N); 
  R +=          checksum_mat(C, N);

  // Output a single value
  printf("Final Result  (N= %d ) = %e\n", N, R);

  free (A);  free (B);  free (C);  free (X);  free (Y);
  return 0;
}
