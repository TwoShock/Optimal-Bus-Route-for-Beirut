import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import pycuda.driver as drv
import random
import pycuda.gpuarray as gpuArr
import math
from matplotlib import pyplot as plt

mod = SourceModule("""
#include <stdio.h>

#define N 2048
#define TPB 32
#define K 3
#define MAX_ITER 10

__device__ float distance(float x1, float x2)
{
	return sqrt((x2-x1)*(x2-x1));
}

__global__ void kMeansClusterAssignment(float *d_datapoints, int *d_clust_assn, float *d_centroids)
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
		float dist = distance(d_datapoints[idx],d_centroids[c]);

		if(dist < min_dist)
		{
			min_dist = dist;
			closest_centroid=c;
		}
	}

	//assign closest cluster id for this datapoint/thread
	d_clust_assn[idx]=closest_centroid;
}


__global__ void kMeansCentroidUpdate(float *d_datapoints, int *d_clust_assn, float *d_centroids, int *d_clust_sizes)
{

	//get idx of thread at grid level
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//bounds check
	if (idx >= N) return;

	//get idx of thread at the block level
	const int s_idx = threadIdx.x;

	//put the datapoints and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
	__shared__ float s_datapoints[TPB];
	s_datapoints[s_idx]= d_datapoints[idx];

	__shared__ int s_clust_assn[TPB];
	s_clust_assn[s_idx] = d_clust_assn[idx];

	__syncthreads();

	//it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
	if(s_idx==0)
	{
		float b_clust_datapoint_sums[K]={0};
		int b_clust_sizes[K]={0};

		for(int j=0; j< blockDim.x; ++j)
		{
			int clust_id = s_clust_assn[j];
			b_clust_datapoint_sums[clust_id]+=s_datapoints[j];
			b_clust_sizes[clust_id]+=1;
		}

		//Now we add the sums to the global centroids and add the counts to the global counts.
		for(int z=0; z < K; ++z)
		{
			atomicAdd(&d_centroids[z],b_clust_datapoint_sums[z]);
			atomicAdd(&d_clust_sizes[z],b_clust_sizes[z]);
		}
	}

	__syncthreads();

	//currently centroids are just sums, so divide by size to get actual centroids
	if(idx < K){
		d_centroids[idx] = d_centroids[idx]/d_clust_sizes[idx]; 
	}

}
""")
N =  2048
TPB = 32
K = 3
MAX_ITER = 10

h_centroids = np.zeros((K,),dtype = np.float32)
h_datapoints = np.zeros((N,),dtype = np.float32)
h_clust_sizes =  np.zeros((K,),dtype = int)
h_clust_assn = np.zeros((N,),dtype = int)

d_centroids = cuda.mem_alloc(h_centroids.nbytes)
d_datapoints = cuda.mem_alloc(h_datapoints.nbytes)
d_clust_sizes = cuda.mem_alloc(h_clust_sizes.nbytes)
d_clust_assn = cuda.mem_alloc(h_clust_assn.nbytes)



cuda.memcpy_dtoh(h_centroids,d_centroids)


for d in range(0,N):
    np.put(h_datapoints,d,random.uniform(0,100))

np.put(h_centroids,0,h_datapoints[0])
np.put(h_centroids,1,h_datapoints[250])
np.put(h_centroids,2,h_datapoints[500])

plt.plot(h_datapoints,'*')

cuda.memcpy_htod(d_centroids, h_centroids)
cuda.memcpy_htod(d_datapoints, h_datapoints)
cuda.memcpy_htod(d_clust_sizes, h_clust_sizes)


cur_iter = 1;

while(cur_iter < MAX_ITER):
    func = mod.get_function("kMeansClusterAssignment")
    func(d_datapoints,d_clust_assn,d_centroids, block=(TPB,1,1),grid=(math.ceil(N/TPB),1))

    temp = np.zeros((K,),dtype = np.float32)
    cuda.memcpy_htod(d_centroids,temp)

    temp = np.zeros((K,),dtype = int)
    cuda.memcpy_htod(d_clust_sizes,temp)

    func = mod.get_function("kMeansCentroidUpdate")
    func(d_datapoints,d_clust_assn,d_centroids,d_clust_sizes, block=(TPB,1,1),grid=(math.ceil(N/TPB),1))
    cur_iter+=1
    
res = np.empty_like(h_clust_assn)
cuda.memcpy_dtoh(res, d_clust_assn)
cuda.memcpy_dtoh(h_centroids, d_centroids)

for i in range (0,K):
        print("centroid ",i,": ",h_centroids[i],"\n");

a = []
b = []
c = []
for idx,data in enumerate(h_datapoints):
    if res[idx] == 0:
        a.append(data)
    elif res[idx]== 1:
        b.append(data)
    else:
        c.append(data)
plt.show()
x = np.linspace(0,512,len(a))
y = np.linspace(0,512,len(b))
z = np.linspace(0,512,len(c))
plt.plot(x,a,'*',color = 'red')
plt.plot(y,b,'*',color = 'green')
plt.plot(z,c,'*',color = 'blue')
plt.show()

