#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

/* Program Parameters */
#define MAXN 8000  /* Max value of N */
int N;  /* Matrix size */
int nt;  /* Number of Threads */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
/* ------------------ Cuda Code --------------------- */

/****** Parallel Cuda code *******/
/* Defined global variables are 
 * maximun matrix size = MAXN
 * Given matrix size = N
 * Input matrix repesentation in 1D A[N][N]
 * Output matrix in 1D is B[N][N]
 */

__global__ void meanCalculation(float* d_in, float* d_mean, int N, int nt)
{
  extern __shared__ float col_data[];
  __shared__ float col_total;

  //each thread loads one element from global to shared mem
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
 	 
  unsigned int thread_id = threadIdx.y;
  unsigned int j = idx_y * N + idx_x;
  __syncthreads();

  /*Calculation for each thread id*/	
  col_data[thread_id]=d_in[j];
	
  /*below for loop is for if number of thread < Matrix size.*/
  for(int i=0;i<N;i+=nt){
	if(N*(nt+thread_id+i)+blockIdx.x < N*N){
		col_data[thread_id]+=d_in[(N*(nt+thread_id+i))+blockIdx.x];
	}	
  }

 /* Sum reduction performed on each column data which is corresponding to 
  * one block by zeroth thread of each block
  */
 if(thread_id==0){

	for(int s=0;s<nt;s++){
		col_total+=col_data[thread_id+s];
	}
	d_mean[blockIdx.x]=col_total/N;
 } 

}

__global__ void calculate_SD(float* d_in, float* d_mean, float* d_sd, int N, int nt)
{
  extern __shared__ float col_sd_data[];
  __shared__ float col_sd_total;

  //each thread loads one element from global to shared mem
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  
  unsigned int thread_id = threadIdx.y;
  unsigned int j = idx_y * N + idx_x;
  __syncthreads();

  col_sd_data[thread_id] = powf(d_in[j] - d_mean[blockIdx.x], 2.0);
  
   for(int i=0;i<N;i+=nt){
	if(N*(nt+thread_id+i)+blockIdx.x < N*N){
		col_sd_data[thread_id]+=powf(d_in[(N*(nt+thread_id+i))+blockIdx.x] - d_mean[blockIdx.x], 2.0);
	}
   }

 if(thread_id==0){
	col_sd_total=0;
	for(int s=0;s<nt;s++){
	col_sd_total+=col_sd_data[thread_id+s];
	}
	d_sd[blockIdx.x] = col_sd_total/(float) N;
 } 
    
}

__global__ void matrixColumnNorm(float* d_in, float* d_out, float* d_mean, float* d_sd, int N, int nt,int c)
{  
  unsigned int thread_id = threadIdx.y;
  
  d_out[thread_id+blockIdx.x*N] = (d_in[thread_id+blockIdx.x*N] - d_mean[blockIdx.x]) / d_sd[blockIdx.x];	
	
	for(int i=0;i<c;i++){
		if((nt+thread_id)+blockIdx.x*N < N*N){
			d_out[(nt+thread_id)+blockIdx.x*N] = (d_in[(nt+thread_id)+blockIdx.x*N] - d_mean[blockIdx.x])/d_sd[blockIdx.x];
		}	
  	}
}


/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int seed = 0;  /* Random seed */
  
  /* Read command-line arguments */
  srand(time_seed());  /* Randomize */

  if (argc == 4) {	
    seed = atoi(argv[3]);
    srand(seed);
    printf("Random seed = %i\n", seed);
  } 
  if (argc >= 3) {
    N = atoi(argv[1]);
    nt = atoi(argv[2]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
    
    if (nt > 1024) {
      printf("nt = %i is out of range.Please provide number of thread less than 1024.\n", N);
      exit(0);
    }
  }
  else {
    printf("Usage: %s <matrix_dimension> <number_of_thread> [random seed]\n",
           argv[0]);    
    exit(0);
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}


int main(int argc, char **argv) {
  /* Timing variables */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2;
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */

  /* Process program parameters */
  parameters(argc, argv);

  float* A = new float [N * N];
  float* B = new float [N * N];

  int i,j;
  
  /*initializing input A*/
  printf("\nInitializing...\n");
  for(i=0;i<N;i++)
  {
    for(j=0;j<N;j++)
    {
      A[j* N + i] = (float)rand()/ 64000.00;
    }
  }
  
  /*print inputs.*/
  if (N < 10) {
    printf("\nA =\n\t");
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
      printf("%5.2f%s", A[i* N + j], (j < N-1) ? ", " : ";\n\t");
      }
    }
  }

  float* d_in;
  float* d_out;
  float* d_mean;
  float* d_sd;
  size_t sizeof2d = N * N * sizeof(float);
  size_t sizeof1d = N * sizeof(float);

  //allocated the device memory for source array
  cudaMalloc(&d_in, sizeof2d);
  cudaMemcpy(d_in, A, sizeof2d, cudaMemcpyHostToDevice);

  //allocate the device memory for destination array
  cudaMalloc(&d_out, sizeof2d);

  //allocate the device memory for mean arry
  cudaMalloc(&d_mean, sizeof1d);

  //allocate the device memory for sd array
  cudaMalloc(&d_sd, sizeof1d);

  dim3 dimBlock;
  dim3 dimGrid;

  if( N < nt)
  {
    dimBlock.x = 1;
    dimBlock.y = N;
    dimGrid.x = N;
    dimGrid.y = 1;
  }
  else
  {
    dimBlock.x = 1;
    dimBlock.y = nt;
    dimGrid.x = N;
    dimGrid.y = 1;
  }

  /* Start Clock */
  printf("\nStarting clock.\n");
  cudaEventRecord(start);
  gettimeofday(&etstart, &tzdummy);
  etstart2 = times(&cputstart);

  double c1=(double)N/(double)nt;
 int c=ceil(c1);

  meanCalculation<<<dimGrid, dimBlock, sizeof1d>>>(d_in, d_mean, N,nt);
  cudaDeviceSynchronize();
  calculate_SD<<<dimGrid, dimBlock, sizeof1d>>>(d_in, d_mean, d_sd, N,nt);
  cudaDeviceSynchronize();
  matrixColumnNorm<<<dimGrid, dimBlock>>>(d_in, d_out, d_mean, d_sd, N,nt,c);
  cudaDeviceSynchronize();

  /* Stop Clock */
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");

  cudaMemcpy(B, d_out, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

   if (N < 10) {
  printf("\nB =\n\t");
    for (i= 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%1.10f%s", B[i* N + j], (j < N-1) ? ", " : ";\n\t");
        }
    }
  }  

  /* Display timing results */
  printf("\nElapsed CPU Time = %g ms.\n", (float)(usecstop - usecstart)/(float)1000);
  printf("Elapsed Cuda Time = %g ms \n",milliseconds);
  printf("Effective Bandwidth (GB/s): %f \n", (2*sizeof2d/milliseconds)/1e6);
  float mean_work = N * log2((float)N) + N;
  float sd_work = N * log2((float)N) + (2*N) + (2*N*N);
  float norm_work = 2 * N * N;
  printf("Effective Throughput (GFLOPS/s): %f \n", ((mean_work+sd_work+norm_work)*1e-9)/(milliseconds*1e-3)); 
  printf("--------------------------------------------\n");

  //deallocate device memory
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_mean);
  cudaFree(d_sd);

  free(A);
  free(B);

  exit(0);
}
