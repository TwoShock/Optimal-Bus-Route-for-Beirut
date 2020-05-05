import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np
import random
import math

mod = SourceModule("""
#include <stdio.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define Idx1D(width, i, j) (i*width+j)

extern "C" {
__global__ void ConstructTour(int* tour, int* visited, double* choice, double* prob, int N) {

        int idx = blockDim.x * blockIdx.x + threadIdx.x;

        for (int step = 1; step < N; step++) {

            int current = tour[Idx1D(N, idx, step - 1)];
            double currentProb = 0.0;

            for (int i = 0; i < N; i++) {
                if (visited[Idx1D(N, idx, i)] == 1)
                    prob[Idx1D(N, idx, i)] = 0.0;
                else {
                    double temp = choice[Idx1D(N, current, i)];
                    prob[Idx1D(N, idx, i)] = temp;
                    currentProb = currentProb + temp;
                }
            }

            double random;
            curandState_t state;
            curand_init((unsigned long long) clock(), 0, 0, &state);
            random = curand_uniform(&state);
            random = random * currentProb;

            int next;
            double sum = prob[Idx1D(N, idx, 0)];

            for (next = 0; sum < random; next++) {
                sum =sum+ prob[Idx1D(N, idx, next + 1)];
            }

            tour[Idx1D(N, idx, step)] = next;
            visited[Idx1D(N, idx, next)] = 1;
        }
    }
    __global__ void Evaporate(double* pheromone, int N, double evapRate) {
        /*
        This function modifies the pheremone trail to account for the evaporation constant
        */
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        pheromone[idx] = pheromone[idx] * evapRate;
    }

    __global__ void Reinforce(double* pheromone, int* distances, int* path, int N, double amount) {

        int currentXThread = threadIdx.x + blockIdx.x * blockDim.x;

        int source = path[currentXThread];
        int dest = path[currentXThread + 1];
        int idx = Idx1D(N, source, dest);
        pheromone[idx] = pheromone[idx] + amount;
        pheromone[Idx1D(N, dest, source)] = pheromone[Idx1D(N, dest, source)] + amount;
    }

}
""",no_extern_c=True)

INF = math.pow(2,30) #2^31 causes overflow
INITIAL_PHEROMONE = 0.5
EVAPORATION_CONSTANT = 0.8
ALPHA = 1
BETA = 3
MAX_ITER = 100

def Idx1D(width, i, j):
    return i * width + j

def dimBlockCalc(nbAnts):
	blocks = int(math.log(nbAnts))
	return int(math.pow(2, blocks))


def calculateCost (distances,path, N):
    cost = 0
    for i in range(N-1):
        idx = Idx1D(N, path [i], path [i + 1])
        cost += distances[idx]
    return cost 

def evaporate (pheromone,N):
    pheromoneDevice = np.empty_like(pheromone)
    
    pheromoneDevice = cuda.mem_alloc(pheromone.nbytes)
    cuda.memcpy_htod(pheromoneDevice,pheromone)
    func = mod.get_function("Evaporate")
    func(pheromoneDevice, np.int32(N), np.int32(EVAPORATION_CONSTANT),block = (N,1,1),grid = (N,1))
    cuda.memcpy_dtoh(pheromone,pheromoneDevice)
    return pheromone

def reinforce(pheromone,distances, path,N):
    amount = float(1.0 / calculateCost (distances, path, N))
    
    path_device = np.empty_like(path)
    distancesDevice = np.empty_like(distances)
    pheromoneDevice = np.empty_like(pheromone)
    
    path_device = cuda.mem_alloc(path.nbytes)
    distanceArray_device = cuda.mem_alloc(distances.nbytes)
    pheromoneDevice = cuda.mem_alloc(pheromone.nbytes)

    cuda.memcpy_htod(path_device,path)
    cuda.memcpy_htod(distanceArray_device,distances)
    cuda.memcpy_htod(pheromoneDevice,pheromone)
    
    func = mod.get_function("Reinforce")
    func(pheromoneDevice, distanceArray_device, path_device, np.int32(N),np.float32(amount),block=(N-1,1,1),grid=(1,1))
    
    cuda.memcpy_dtoh(distances,distancesDevice)
    cuda.memcpy_dtoh(pheromone,pheromoneDevice)
    
    return (pheromone,distances)



def bestSolution (tours, distances, numberOfAnts, N):
    bestTour = tours
    for tour in range(numberOfAnts):
        bestCost = calculateCost(distances, bestTour, N)
        currentCost = calculateCost(distances,tours [Idx1D (N, tour, 0):], N)
        if (currentCost<bestCost): 
            bestTour = tours[Idx1D(N, tour, 0):]
    return bestTour
	

def cudaACO (distances, N, nbAnts):
    
    pheromone = np.empty((N*N,),dtype=np.float32)
    choice = np.empty((N*N,),dtype=np.float32)

    tours = np.empty((N*nbAnts,),dtype=int)
    visited = np.empty((N*nbAnts,),dtype=int)
    
    
    distancesDevice = np.empty_like(distances)
    tours_device = np.empty_like(tours)
    visited_device = np.empty_like(visited)
    choiceInfo_device = np.empty_like(choice)
    probDevice = np.empty_like(pheromone)
    
    distancesDevice= cuda.mem_alloc(distances.nbytes)
    toursDevice = cuda.mem_alloc(tours.nbytes)
    visitedDevice = cuda.mem_alloc(visited.nbytes)
    choiceDevice = cuda.mem_alloc(choice.nbytes)
    probDevice = cuda.mem_alloc(pheromone.nbytes)

    cuda.memcpy_htod(distancesDevice,distances)
    
    for i in range(N):
        for j in range(N):
            pheromone [Idx1D (N, i, j)] = INITIAL_PHEROMONE
            
    for iteration in range(MAX_ITER):

        for i in range(nbAnts):
            for j in range(N):
                tours [Idx1D(N, i, j)] = INF 
        
        for i in range(nbAnts):
            for j in range(N):
                visited [Idx1D (N, i, j)] = 0 
                
        for i in range(N):
            for j in range(N):
                edge_pherom = pheromone [Idx1D (N, i, j)]
                edge_weight = distances [Idx1D (N, i, j)]
                prob = 0.0
                if (edge_weight != 0):
                    prob = math.pow(edge_pherom, ALPHA) * math.pow((1 / edge_weight), BETA)
                else:
                    prob = math.pow(edge_pherom, ALPHA) * math.pow(INF, BETA)
                choice [Idx1D (N, i, j)] = prob
        
        cuda.memcpy_htod(choiceDevice,choice)
        
        for ant in range(nbAnts):
            init = int(random.uniform(0,N))
            tours[Idx1D(N, ant, 0)] = init
            visited [Idx1D (N, ant, init)] = 1
        
        cuda.memcpy_htod(visited_device,visited)
        cuda.memcpy_htod(tours_device,tours)
        
        b = dimBlockCalc(nbAnts)
        dimGrid = int(nbAnts / b)
        antsPerBlock = int(b)
        
        func = mod.get_function("ConstructTour")
        func(toursDevice, visitedDevice, choiceDevice, probDevice, np.int32(N),block = (antsPerBlock,1,1),grid = (dimGrid,1))
        
        cuda.memcpy_dtoh(tours,toursDevice)
        cuda.memcpy_dtoh(visited,visitedDevice)
        
        pheromone = evaporate (pheromone, N)
        best = bestSolution (tours, distances, nbAnts, N)
        pheromones,distanceArray = reinforce (pheromone, distances, best, N) 
    best = bestSolution (tours, distances, nbAnts, N)
    return best 

