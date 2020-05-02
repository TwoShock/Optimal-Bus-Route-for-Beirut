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
import pickle


TPB = 32
K = 26
MAX_ITER = 500

mod = SourceModule("""
#include <stdio.h>

#define TPB 32
#define K 26
#define MAX_ITER 10

__device__ float distance(float x1, float x2,float y1, float y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

__global__ void kMeansClusterAssignment(float *d_datapoints_X,float *d_datapoints_Y, int *d_clust_assn, float *d_centroids_X,float *d_centroids_Y,int N)
{
	//get idx for this datapoint
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//bounds check
	if (idx >= N) return;

	//find the closest centroid to this datapoint
	float min_dist = INFINITY;
	int closest_centroid = 0;

	for(int c = 0; c<K;++c)
	{
		float dist = distance(d_datapoints_X[idx],d_centroids_X[c],d_datapoints_Y[idx],d_centroids_Y[c]);

		if(dist < min_dist)
		{
			min_dist = dist;
			closest_centroid=c;
		}
	}

	//assign closest cluster id for this datapoint/thread
	d_clust_assn[idx]=closest_centroid;
}
        
__global__ void kMeansFinalStep(float *d_centroids_X,float *d_centroids_Y,int *d_clust_sizes){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx<K){
        d_centroids_X[idx] = d_centroids_X[idx]/d_clust_sizes[idx];
        d_centroids_Y[idx] = d_centroids_Y[idx]/d_clust_sizes[idx];

    }

}


__global__ void kMeansCentroidUpdate(float *d_datapoints_X, float *d_datapoints_Y, int *d_clust_assn, float *d_centroids_X,float *d_centroids_Y, int *d_clust_sizes,int N)
{

	//get idx of thread at grid level
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//bounds check
	if (idx >= N) return;

	//get idx of thread at the block level
	const int s_idx = threadIdx.x;

	//put the datapoints and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
	__shared__ float s_datapoints_X[TPB];
    __shared__ float s_datapoints_Y[TPB];
	__shared__ int s_clust_assn[TPB];

	s_datapoints_X[s_idx]= d_datapoints_X[idx];
    s_datapoints_Y[s_idx]= d_datapoints_Y[idx];

	s_clust_assn[s_idx] = d_clust_assn[idx];

	__syncthreads();

	//it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
	if(s_idx==0)
	{
		float b_clust_datapoint_sums_X[K]={0};
        float b_clust_datapoint_sums_Y[K]={0};

		int b_clust_sizes[K]={0};

		for(int j=0; j< blockDim.x; ++j)
		{
			int clust_id = s_clust_assn[j];
			b_clust_datapoint_sums_X[clust_id]+=s_datapoints_X[j];
            b_clust_datapoint_sums_Y[clust_id]+=s_datapoints_Y[j];

			b_clust_sizes[clust_id]+=1;
		}

		//Now we add the sums to the global centroids and add the counts to the global counts.
		for(int z=0; z < K; ++z)
		{
			atomicAdd(&d_centroids_X[z],b_clust_datapoint_sums_X[z]);
            atomicAdd(&d_centroids_Y[z],b_clust_datapoint_sums_Y[z]);

			atomicAdd(&d_clust_sizes[z],b_clust_sizes[z]);
		}
	}

	__syncthreads();

}
""")

def CudaKMeans (x_coord,y_coord,N):

    h_centroids_X = np.zeros((K,),dtype = np.float32)
    h_centroids_Y = np.zeros((K,),dtype = np.float32)
    
    h_datapoints_X = np.zeros((N,),dtype = np.float32)
    h_datapoints_Y = np.zeros((N,),dtype = np.float32)
    
    h_clust_sizes =  np.zeros((K,),dtype = int)
    h_clust_assn = np.zeros((N,),dtype = int)
    
    d_centroids_X = cuda.mem_alloc(h_centroids_X.nbytes)
    d_centroids_Y = cuda.mem_alloc(h_centroids_Y.nbytes)
    
    d_datapoints_X = cuda.mem_alloc(h_datapoints_X.nbytes)
    d_datapoints_Y = cuda.mem_alloc(h_datapoints_Y.nbytes)
    
    d_clust_sizes = cuda.mem_alloc(h_clust_sizes.nbytes)
    d_clust_assn = cuda.mem_alloc(h_clust_assn.nbytes)
    
    for i in range(N):
        h_datapoints_X[i] = x_coord[i]
        h_datapoints_Y[i] = y_coord[i]
    
    for i in range(K):
        seed = int(random.uniform(1,N))
        h_centroids_X[i] = x_coord[seed]
        h_centroids_Y[i] = y_coord[seed]
    
    cuda.memcpy_htod(d_centroids_X, h_centroids_X)
    cuda.memcpy_htod(d_centroids_Y, h_centroids_Y)
    
    cuda.memcpy_htod(d_datapoints_X, h_datapoints_X)
    cuda.memcpy_htod(d_datapoints_Y, h_datapoints_Y)
    
    cuda.memcpy_htod(d_clust_sizes, h_clust_sizes)
    
    cur_iter = 1;
    while cur_iter<MAX_ITER:
        func = mod.get_function("kMeansClusterAssignment")
        func(d_datapoints_X,d_datapoints_Y,d_clust_assn,d_centroids_X,d_centroids_Y, np.int32(N),block=(TPB,1,1),grid=(math.ceil(N/TPB),1))
        cuda.memcpy_dtoh(h_clust_assn, d_clust_assn)
        
        temp = np.zeros((K,),dtype = np.float32)
        cuda.memcpy_htod(d_centroids_X,temp)
        cuda.memcpy_htod(d_centroids_Y,temp)
        
        temp = np.zeros((K,),dtype = int)
        cuda.memcpy_htod(d_clust_sizes,temp)
        
        func = mod.get_function("kMeansCentroidUpdate")
        func(d_datapoints_X,d_datapoints_Y,d_clust_assn,d_centroids_X,d_centroids_Y,d_clust_sizes, np.int32(N),block=(TPB,1,1),grid=(math.ceil(N/TPB),1))
    
        func = mod.get_function("kMeansFinalStep")
        func(d_centroids_X,d_centroids_Y,d_clust_sizes, block=(TPB,1,1),grid=(math.ceil(K/TPB),1))
    
        cur_iter+=1
    
    cuda.memcpy_dtoh(h_clust_assn, d_clust_assn)
    cuda.memcpy_dtoh(h_centroids_X, d_centroids_X)
    cuda.memcpy_dtoh(h_centroids_Y, d_centroids_Y)
    cuda.memcpy_dtoh(h_clust_sizes, d_clust_sizes)
    
        
    nodes = []    
    for i in range(K):
        nodes.append([h_centroids_X[i],h_centroids_Y[i]])
    
    return np.array(nodes)


