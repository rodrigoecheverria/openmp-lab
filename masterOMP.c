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
  int N2 = N-SM;
#pragma omp parallel 
{
#pragma omp for schedule(static,1) private(i,j,k,i2,j2,k2,a2,b2,c2)
  for (i=0;i<N2; i+=SM)
    for(j=0;j<N;j+=SM)
      for(k=0;k<N;k+=SM)
        for(i2=0,c2=&c[i*N+j],a2=&a[i*N+k];i2<SM;++i2,c2+=N,a2+=N)
        {
          _mm_prefetch (&a2[8],_MM_HINT_NTA);
          for(k2=0,b2=&b[k*N+j];k2<SM;++k2,b2+=N)
          {
            __m128d m1d = _mm_load_sd(&a2[k2]);
            m1d=_mm_unpacklo_pd (m1d,m1d);
            
            for(j2=0;j2<SM;j2+=2)
            {
                __m128d m2 = _mm_load_pd(&b2[j2]);
                __m128d r2 = _mm_load_pd(&c2[j2]);
                _mm_store_pd (&c2[j2],_mm_add_pd (_mm_mul_pd(m2,m1d),r2));
            }
          }
        }
//LAST ITERATION OF i UNROLLED (i = N-SM )
i = N2; //The value of i is unknown at this point if multithreaded

#pragma omp for schedule(static,1) private(i,j,k,i2,j2,k2,a2,b2,c2)
for(j=0;j<N;j+=SM)
  for(k=0;k<N;k+=SM)
    for(i2=0,c2=&c[i*N+j],a2=&a[i*N+k];i2<SM;++i2,c2+=N,a2+=N)
    {
      _mm_prefetch (&a2[8],_MM_HINT_NTA);
      for(k2=0,b2=&b[k*N+j];k2<SM;++k2,b2+=N)
      {
        __m128d m1d = _mm_load_sd(&a2[k2]);
        m1d=_mm_unpacklo_pd (m1d,m1d);
        for(j2=0;j2<SM;j2+=2)
        {
          __m128d m2,r2;
          if ((i*N + j+ j2) < (N*N))
          {
            m2 = _mm_load_pd(&b2[j2]);
            r2 = _mm_load_pd(&c2[j2]);
            _mm_store_pd (&c2[j2],_mm_add_pd (_mm_mul_pd(m2,m1d),r2));
          }
        }
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
  int N=2000,ok=0;
  double *A, *B, *C, *X, *Y, R; 

  if (argc>1) {  N  = atoll(argv[1]); }
  if (N<1 || N>20000) {
     printf("input parameter: N (1-20000)\n");
     return 0;
  }

  // Dynamic allocation of 2-D matrices (aligned)
  ok += posix_memalign((void**)&A,64,N*N*sizeof(double));
  ok += posix_memalign((void**)&B,64,N*N*sizeof(double));
  ok += posix_memalign((void**)&C,64,N*N*sizeof(double));
  
  // Dynamic allocation of vectors (aligned)
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
