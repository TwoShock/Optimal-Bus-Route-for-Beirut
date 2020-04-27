# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:58:39 2020

@author: ojaro
"""


import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np
import math
import matplotlib.pyplot as plt
import osmnx as ox
from sklearn.cluster import KMeans
import random
import time
import pandas as pd
import pickle

mod = SourceModule("""
#include <stdio.h>
#include<cmath>
                
__global__ void _Wake_GPU(int reps){
	int idx=blockIdx.x*blockDim.x + threadIdx.x;
	if(idx>=reps)return;
}

__global__ void _GPU_Floyd_kernel(int k, float *G,float *P, int N){//G will be the adjacency matrix, P will be path matrix
	int col=blockIdx.x*blockDim.x + threadIdx.x;
	if(col>=N)return;
	int idx=N*blockIdx.y+col;

	__shared__ float best;
	if(threadIdx.x==0)
		best=G[N*blockIdx.y+k];
	__syncthreads();
	if(best==INT_MAX/2)
        return;
	float tmp_b=G[k*N+col];
	if(tmp_b==INT_MAX)
        return;
	float cur=best+tmp_b;
	if(cur<G[idx]){
		G[idx]=cur;
		P[idx]=k;
	}
}

                
""")

BLOCK_SIZE = 32
NumBytes = 4

func = mod.get_function("_Wake_GPU")
func(np.int32(NumBytes),block = (BLOCK_SIZE,1,1))
OrigGraph = [[0,8,math.inf,1],
             [math.inf,0,1,math.inf],
             [4,math.inf,0,math.inf],
             [math.inf,2,9,0]]

OrigGraph = np.array(OrigGraph,dtype = np.float32)

OrigGraph = OrigGraph.flatten()
H_G = np.empty_like(OrigGraph)
H_G[:] = OrigGraph
D_G = cuda.mem_alloc(OrigGraph.nbytes)
cuda.memcpy_htod(D_G, H_G)


H_P	= np.full_like(OrigGraph,-1)
D_P = cuda.mem_alloc(OrigGraph.nbytes)
cuda.memcpy_htod(D_P, H_P)


Grid = (int((NumBytes+BLOCK_SIZE-1)/BLOCK_SIZE),NumBytes)


for k in range(NumBytes):
    func = mod.get_function("_GPU_Floyd_kernel")
    func(np.int32(k),D_G,D_P,np.int32(NumBytes),block = (BLOCK_SIZE,1,1),grid = Grid)
    pycuda.driver.Context.synchronize()
    

cuda.memcpy_dtoh(H_G, D_G)
cuda.memcpy_dtoh(H_P,D_P)

print(H_G,'\n')
print(H_P)

