# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:58:39 2020

@author: ojaro
"""


import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np

mod = SourceModule("""
#include <stdio.h>
#include<cmath>
                
#include <stdio.h>
#include<cmath>

__global__ void FloydWarshall( float* matrix, float* path, int N,int k) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N)return;
	int idx = N * blockIdx.y + i;
	__shared__ float best;
	if (threadIdx.x == 0)
		best = matrix[N * blockIdx.y + k];
	__syncthreads();
	if (best == INT_MAX / 2)
		return;
	float tmp_b = matrix[k * N + i];
	if (tmp_b == INT_MAX)
		return;
	float current = best + tmp_b;
	if (current < matrix[idx]) {
		matrix[idx] = current;
		path[idx] = k;
	}
}
                
""")

BLOCK_SIZE = 32

def cudaFloydWarshall(matrix):
    N = len(matrix)
    matrix = np.array(matrix,dtype = np.float32)
    matrix = matrix.flatten()
    matrixDevice = cuda.mem_alloc(matrix.nbytes)
    cuda.memcpy_htod(matrixDevice, matrix)
    
    path = np.full_like(matrix,-1)
    pathDevice = cuda.mem_alloc(matrix.nbytes)
    cuda.memcpy_htod(pathDevice, path)
    
    dimGrid = (int((N+BLOCK_SIZE-1)/BLOCK_SIZE),N)
    
    for k in range(N):
        func = mod.get_function("FloydWarshall")
        func(matrixDevice,pathDevice,np.int32(N),np.int32(k),block = (BLOCK_SIZE,1,1),grid = dimGrid)
        pycuda.driver.Context.synchronize()
        
    cuda.memcpy_dtoh(matrix, matrixDevice)
    cuda.memcpy_dtoh(path,pathDevice)
    return matrix, path
    
