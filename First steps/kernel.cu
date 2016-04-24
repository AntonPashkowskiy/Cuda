#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <time.h>
#include <malloc.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define FIBER 32
#define MATRIX_SIZE 2048
#define DATA_SIZE MATRIX_SIZE * MATRIX_SIZE * sizeof(int)
#define MAX_MATRIX_SIZE (MATRIX_SIZE * MATRIX_SIZE)

__global__ void kernel_shared(int *A, int *C, int *B, int *result) {
	__shared__ int shared_memory[FIBER][FIBER];

	int i = blockIdx.x * blockDim.x + threadIdx.y;
	int j = blockIdx.y * blockDim.y + threadIdx.x;

	shared_memory[threadIdx.y][threadIdx.x] = B[i * MATRIX_SIZE + j];

	__syncthreads();

	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	
	int first_index = i + j * MATRIX_SIZE;
	int second_index = j + i * MATRIX_SIZE;

	if (first_index < MAX_MATRIX_SIZE && second_index < MAX_MATRIX_SIZE)
	{
		result[first_index] = (A[first_index] + A[first_index]) * shared_memory[threadIdx.x][threadIdx.y] - C[first_index];
	}
}

__global__ void kernel(int *A, int *C, int *B, int *result) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int first_index = i + j * MATRIX_SIZE;
	int second_index = j + i * MATRIX_SIZE;

	if (first_index < MAX_MATRIX_SIZE && second_index < MAX_MATRIX_SIZE)
	{	
		result[first_index] = (A[first_index] + A[first_index]) * B[second_index] - C[first_index];
	}
}

using namespace std;

int* simple_matrix_multiplication(int* A, int* B, int* C) {
	int *result = (int*)_aligned_malloc(DATA_SIZE, 32);
	
	for (int i = 0; i < MATRIX_SIZE; i++) 
	{
		for (int j = 0; j < MATRIX_SIZE; j++) 
		{
			int first_index = i * MATRIX_SIZE + j;
			int second_index = j * MATRIX_SIZE + i;

			result[first_index] = (A[first_index] + A[first_index]) * B[second_index] - C[first_index];
		}
	}
	return result;
}

void cuda_memory_allocation(int **pointer) {
	cudaError_t result = cudaMalloc((void**)pointer, DATA_SIZE);
	
	if (result != cudaSuccess) 
	{
		printf("%s\n", cudaGetErrorString(result));
	}
}

void cuda_memcpy_host_to_device(int *source, int *destination) {
	cudaError_t result = cudaMemcpy(destination, source, DATA_SIZE, cudaMemcpyHostToDevice);

	if (result != cudaSuccess) 
	{
		printf("%s\n", cudaGetErrorString(result));
	}
}

void cuda_memcpy_device_to_host(int *source, int *destination) {
	cudaError_t result = cudaMemcpy(source, destination, DATA_SIZE, cudaMemcpyDeviceToHost);
	
	if (result != cudaSuccess) 
	{
		printf("%s\n", cudaGetErrorString(result));
	}
}

bool is_matrix_equals(int *first_matrix, int *second_matrix) {
	for (int i = 0; i < MATRIX_SIZE; i++) 
	{
		for (int j = 0; j < MATRIX_SIZE; j++)
		{
			if (first_matrix[i * MATRIX_SIZE + j] != second_matrix[i * MATRIX_SIZE + j])
			{
				printf("\n%d != %d [%d]\n", first_matrix[i * MATRIX_SIZE + j], second_matrix[i * MATRIX_SIZE + j], i * MATRIX_SIZE + j);
				return false;
			}
		}
	}
	return true;
}

int* fill_matrix(int *matrix)
{
	if (matrix == NULL) 
	{
		return NULL;
	}

	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		for (int j = 0; j < MATRIX_SIZE; j++)
		{
			matrix[i * MATRIX_SIZE + j] = rand() % 1000;
		}
	}

	return matrix;
}

void print_matrix(int** matrix)
{
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		for (int j = 0; j < MATRIX_SIZE; j++)
		{
			printf("%u\t", matrix[i][j]);
			if (j == MATRIX_SIZE - 1)
			{
				printf("\n");
			}
		}
	}
	printf("\n");
	printf("\n");
}

int* process_matrix_cpu(int *A, int *B, int *C) {
	int *result;

	fill_matrix(A);
	fill_matrix(C);
	fill_matrix(B);

	clock_t start, stop;
	start = clock();

	result = simple_matrix_multiplication(A, B, C);

	stop = clock();
	printf("Run time CPU =  %d \n", stop - start);

	return result;
}

int* process_matrix_gpu(int *A, int *B, int *X, bool shared) {
	int *device_memory;
	int *result = (int*)_aligned_malloc(DATA_SIZE, 32);
	memset(result, 0, DATA_SIZE);

	cuda_memory_allocation(&device_memory);
	cuda_memcpy_host_to_device(result, device_memory);

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	dim3 threads(FIBER, FIBER);
	dim3 blocks((MATRIX_SIZE + (FIBER - 1)) / FIBER, (MATRIX_SIZE + (FIBER - 1)) / FIBER);

	cudaEventSynchronize(start);

	if (shared) 
	{
		kernel_shared <<< blocks, threads >>> (A, X, B, device_memory);
	} 
	else 
	{
		kernel <<< blocks, threads >>> (A, X, B, device_memory);
	}

	cudaError_t error = cudaGetLastError();
	
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	float timer = 0;
	cudaEventElapsedTime(&timer, start, stop);
	
	if (!shared)
	{
		cout << "Run time GPU = " << timer << endl;
	}
	else 
	{
		cout << "Run time GPU (shared) = " << timer << endl;
	}
	cudaEventRecord(start);
	cuda_memcpy_device_to_host(result, device_memory);
	
	return result;
}

int main(int argc, char* argv[])
{
	int *gpu_A, *gpu_B, *gpu_C;
	int *gpu_shared_A, *gpu_shared_B, *gpu_shared_C;
	int *A, *B, *C;
	int *cpu_result, *gpu_result, *gpu_shared_result;

	A = (int*)_aligned_malloc(DATA_SIZE, 32);
	B = (int*)_aligned_malloc(DATA_SIZE, 32);
	C = (int*)_aligned_malloc(DATA_SIZE, 32);

	cpu_result = process_matrix_cpu(A, B, C);

	cuda_memory_allocation(&gpu_A);
	cuda_memory_allocation(&gpu_B);
	cuda_memory_allocation(&gpu_C);
	cuda_memory_allocation(&gpu_shared_A);
	cuda_memory_allocation(&gpu_shared_B);
	cuda_memory_allocation(&gpu_shared_C);

	cuda_memcpy_host_to_device(A, gpu_A);
	cuda_memcpy_host_to_device(B, gpu_B);
	cuda_memcpy_host_to_device(C, gpu_C);
	cuda_memcpy_host_to_device(A, gpu_shared_A);
	cuda_memcpy_host_to_device(B, gpu_shared_B);
	cuda_memcpy_host_to_device(C, gpu_shared_C);

	gpu_result = process_matrix_gpu(gpu_A, gpu_B, gpu_C, false);
	gpu_shared_result = process_matrix_gpu(gpu_shared_A, gpu_shared_B, gpu_shared_C, true);

	if (!is_matrix_equals(cpu_result, gpu_result))
	{
		printf("Errors occured!\n");
	}

	if (!is_matrix_equals(cpu_result, gpu_shared_result)) 
	{
		printf("Error occured! (shared)\n");
	}

	_aligned_free(A);
	_aligned_free(B);
	_aligned_free(C);
	_aligned_free(cpu_result);
	system("pause");
}
