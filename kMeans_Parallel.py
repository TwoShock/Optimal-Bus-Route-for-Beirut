import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np
import math
import random



TPB = 32
K = 26
MAX_ITER = 500

mod = SourceModule("""

#include <stdio.h>

#define TPB 32
#define CLUSTER_COUNT 26 // this is the k parameter in kmeans


__device__ float distance(float x1, float x2, float y1, float y2){
	return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

__global__ void AssignClusters(float* d_X, float* d_Y, int* d_assigned_cluster, float* d_centroids_X, float* d_centroids_Y, int N){
	/*
	Input:
		d_X: array containing the X coordinates stored on the device 
		d_Y: array containing the Y coordinates stored on the device
		d_assigned_cluster: array which will contain the clusters that each point belong to
		d_centroids_X: array containing the centroids for X
		d_centroids_Y: array containing the centroids for Y
		N: number of points
	Function: Populates the d_assigned_cluster array with the correct values. Namely each point will belong to a given cluster and that is represented in the d_assigne_cluster array
		ex: Assume the output of the function gives the following for an array of 6 points with 2 clusters
			d_assignned_cluster = {0,0,0,1,1,1}
			This means that points 0->2 are part of cluster 0 and points 2->5 are part of cluster 1
		Each thread computes and assigns a single point to a cluster array so each thread works on one index of d_assigned_cluster
	*/
	const int i = blockIdx.x * blockDim.x + threadIdx.x;//computes global thread index
	if (i >= N) return; //checks if given thread is out of bounds

	//find the closest centroid to this datapoint
	float minDist = INFINITY;
	int nearestCentroid = 0;//start out assuming the nearest is cluster 0
	for (int centroidIndex = 0; centroidIndex < CLUSTER_COUNT; ++centroidIndex){
		float dist = distance(d_X[i], d_centroids_X[centroidIndex], d_Y[i], d_centroids_Y[centroidIndex]);//compute distance from the current point and the current centroid choosen
		if (dist < minDist){ //updates the nearestCentroid field if the distance is smaller than the previous distance
			minDist = dist;
			nearestCentroid = centroidIndex;
		}
	}
	//assign closest cluster id for this datapoint/thread
	d_assigned_cluster[i] = nearestCentroid;
}

__global__ void FinalStep(float* d_centroids_X, float* d_centroids_Y, int* d_clust_sizes) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < CLUSTER_COUNT) {
		d_centroids_X[idx] = d_centroids_X[idx] / d_clust_sizes[idx];
		d_centroids_Y[idx] = d_centroids_Y[idx] / d_clust_sizes[idx];
	}
}
__global__ void UpdateCentroids(float* d_X, float* d_Y, int* d_assigned_cluster, float* d_centroids_X, float* d_centroids_Y, int* d_clust_sizes, int N){
	/*
	Function: Recomputes the new centroids given current d_X d_Y 
	*/
	const int i = blockIdx.x * blockDim.x + threadIdx.x;//computes global thread index
	if (i >= N) return;//checks if thread is out of bounds
	//get idx of thread at the block level
	const int s_idx = threadIdx.x;
	//put the datapoints and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
	__shared__ float s_X[TPB];
	__shared__ float s_Y[TPB];
	__shared__ int s_assigned_cluster[TPB];
	s_X[s_idx] = d_X[i];
	s_Y[s_idx] = d_Y[i];
	s_assigned_cluster[s_idx] = d_assigned_cluster[i];
	__syncthreads();
	//it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
	if (s_idx == 0)
	{
		float b_clust_datapoint_sums_X[CLUSTER_COUNT] = { 0 };
		float b_clust_datapoint_sums_Y[CLUSTER_COUNT] = { 0 };
		int b_clust_sizes[CLUSTER_COUNT] = { 0 };
		for (int j = 0; j < blockDim.x; ++j)
		{
			int clust_id = s_assigned_cluster[j];
			b_clust_datapoint_sums_X[clust_id] += s_X[j];
			b_clust_datapoint_sums_Y[clust_id] += s_Y[j];
			b_clust_sizes[clust_id] += 1;
		}
		//Now we add the sums to the global centroids and add the counts to the global counts.
		for (int z = 0; z < CLUSTER_COUNT; ++z)
		{
			atomicAdd(&d_centroids_X[z], b_clust_datapoint_sums_X[z]);
			atomicAdd(&d_centroids_Y[z], b_clust_datapoint_sums_Y[z]);
			atomicAdd(&d_clust_sizes[z], b_clust_sizes[z]);
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
        func = mod.get_function("AssignClusters")
        func(d_datapoints_X,d_datapoints_Y,d_clust_assn,d_centroids_X,d_centroids_Y, np.int32(N),block=(TPB,1,1),grid=(math.ceil(N/TPB),1))
        cuda.memcpy_dtoh(h_clust_assn, d_clust_assn)
        
        temp = np.zeros((K,),dtype = np.float32)
        cuda.memcpy_htod(d_centroids_X,temp)
        cuda.memcpy_htod(d_centroids_Y,temp)
        
        temp = np.zeros((K,),dtype = int)
        cuda.memcpy_htod(d_clust_sizes,temp)
        
        func = mod.get_function("UpdateCentroids")
        func(d_datapoints_X,d_datapoints_Y,d_clust_assn,d_centroids_X,d_centroids_Y,d_clust_sizes, np.int32(N),block=(TPB,1,1),grid=(math.ceil(N/TPB),1))
    
        func = mod.get_function("FinalStep")
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


