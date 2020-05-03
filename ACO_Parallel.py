import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import time
import pickle
import networkx as nx
import pants
import random
import math
from cuda_kmeans import CudaKMeans
import geopy.distance

mod = SourceModule("""
#include <stdio.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define index(length, line, column) (column + line * length) 
extern "C" {
    __global__ void evaporate(double* pheromoneValues, int numberOfCities, double evaporationRate) {
        /*
        This function modifies the pheremone trail to account for the evaporation constant
        */
        int edgeID = threadIdx.x + blockIdx.x * blockDim.x;
        pheromoneValues[edgeID] = pheromoneValues[edgeID] * evaporationRate;
    }

    __global__ void reinforcePath(double* pheromoneValues, int* distances, int* pathArray, int numberOfCities, double reinforcementAmount) {

        int currentXThread = threadIdx.x + blockIdx.x * blockDim.x;

        int source = pathArray[currentXThread];
        int dest = pathArray[currentXThread + 1];
        int idx = index(numberOfCities, source, dest);
        pheromoneValues[idx] = pheromoneValues[idx] + reinforcementAmount;
        pheromoneValues[index(numberOfCities, dest, source)] = pheromoneValues[index(numberOfCities, dest, source)] + reinforcementAmount;
    }

    __global__ void constructTour(int* tourArray, int* visitedArray, double* choiceInfo, double* probabilityArray, int numberOfCities) {

        int currentXThread = blockDim.x * blockIdx.x + threadIdx.x;

        for (int step = 1; step < numberOfCities; step++) {

            int current = tourArray[index(numberOfCities, currentXThread, step - 1)];
            double currentProbabilitySum = 0.0;

            for (int i = 0; i < numberOfCities; i++) {
                if (visitedArray[index(numberOfCities, currentXThread, i)] == 1)
                    probabilityArray[index(numberOfCities, currentXThread, i)] = 0.0;
                else {
                    double currentProbability = choiceInfo[index(numberOfCities, current, i)];
                    probabilityArray[index(numberOfCities, currentXThread, i)] = currentProbability;
                    currentProbabilitySum = currentProbabilitySum + currentProbability;
                }
            }

            double random;
            curandState_t state;
            curand_init((unsigned long long) clock(), 0, 0, &state);
            random = curand_uniform(&state);
            random = random * currentProbabilitySum;

            int next;
            double sum = probabilityArray[index(numberOfCities, currentXThread, 0)];

            for (next = 0; sum < random; next++) {
                sum += probabilityArray[index(numberOfCities, currentXThread, next + 1)];
            }

            tourArray[index(numberOfCities, currentXThread, step)] = next;
            visitedArray[index(numberOfCities, currentXThread, next)] = 1;
        }
    }
}
""",no_extern_c=True)



INF = 65536 
ITERATIONS = 50
INTIALPHEREMONEAMOUNT = 1.0
EVAPORATION_CONSTANT = 0.5
ALPHA = 1
BETA = 2


def index(length, line, column):
    return (column + line * length) 

def threads(numberOfAnts):
    n_threds = 1
    while(n_threds * 2 <numberOfAnts):
        n_threds*= 2
    return n_threds

def threadsPerBlock(numberOfAnts):
	blocks = math.log(numberOfAnts)
	return math.pow(2, blocks)

def evaporate (pheromones,numberOfCities):
    pheremoneDeviceArray = np.empty_like(pheromones)
    
    pheremoneDeviceArray = cuda.mem_alloc(pheromones.nbytes)
    cuda.memcpy_htod(pheremoneDeviceArray,pheromones)
    func = mod.get_function("evaporate")
    func(pheremoneDeviceArray, np.int32(numberOfCities), np.int32(EVAPORATION_CONSTANT),block = (numberOfCities,1,1),grid = (numberOfCities,1))
    cuda.memcpy_dtoh(pheromones,pheremoneDeviceArray)
    return pheromones

def reinforce(pheromones,distanceArray, path,numberOfCities):
    amount = float(1.0 / calcPathCost (distanceArray, path, numberOfCities))
    
    path_device = np.empty_like(path)
    distanceArray_device = np.empty_like(distanceArray)
    pheremoneDeviceArray = np.empty_like(pheromones)
    
    path_device = cuda.mem_alloc(path.nbytes)
    distanceArray_device = cuda.mem_alloc(distanceArray.nbytes)
    pheremoneDeviceArray = cuda.mem_alloc(pheromones.nbytes)

    cuda.memcpy_htod(path_device,path)
    cuda.memcpy_htod(distanceArray_device,distanceArray)
    cuda.memcpy_htod(pheremoneDeviceArray,pheromones)
    
    func = mod.get_function("reinforcePath")
    func(pheremoneDeviceArray, distanceArray_device, path_device, np.int32(numberOfCities),np.float32(amount),block=(1,1,1),grid=(numberOfCities,1))
    
    cuda.memcpy_dtoh(distanceArray,distanceArray_device)
    cuda.memcpy_dtoh(pheromones,pheremoneDeviceArray)
    
    return (pheromones,distanceArray)


def calcPathCost (distanceArray,path, numberOfCities):
    cost = 0
    for count in range(numberOfCities-1):
        idx = index(numberOfCities, path [count], path [count + 1])
        if idx<INF:
            cost += distanceArray[idx]
        else:
            cost+=INF
    return cost 


def bestSolution (tours, distanceArray, numberOfAnts, numberOfCities):
    bestTour = tours
    for tour in range(numberOfAnts):
        bestCost = calcPathCost(distanceArray, bestTour, numberOfCities)
        currentCost = calcPathCost(distanceArray,tours [index (numberOfCities, tour, 0):], numberOfCities)
        if (currentCost<bestCost): 
            bestTour = tours[index(numberOfCities, tour, 0):]
    return bestTour
	

def run (distanceArray, numberOfCities, numberOfAnts):
    
    pheromones = np.empty((numberOfCities*numberOfCities,),dtype=np.float32)
    tours = np.empty((numberOfCities*numberOfAnts,),dtype=int)
    visited = np.empty((numberOfCities*numberOfAnts,),dtype=int)
    choiceInfo = np.empty((numberOfCities*numberOfCities,),dtype=np.float32)
    
    
    distanceArray_device = np.empty_like(distanceArray)
    tours_device = np.empty_like(tours)
    visited_device = np.empty_like(visited)
    choiceInfo_device = np.empty_like(choiceInfo)
    probabilityArray = np.empty_like(pheromones)
    
    distanceArray_device = cuda.mem_alloc(distanceArray.nbytes)
    tours_device = cuda.mem_alloc(tours.nbytes)
    visited_device = cuda.mem_alloc(visited.nbytes)
    choiceInfo_device = cuda.mem_alloc(choiceInfo.nbytes)
    probabilityArray = cuda.mem_alloc(pheromones.nbytes)

    cuda.memcpy_htod(distanceArray_device,distanceArray)
    
    for i in range(numberOfCities):
        for j in range(numberOfCities):
            pheromones [index (numberOfCities, i, j)] = INTIALPHEREMONEAMOUNT
            
    for iteration in range(ITERATIONS):

        for i in range(numberOfAnts):
            for j in range(numberOfCities):
                tours [index(numberOfCities, i, j)] = INF 
        
        for i in range(numberOfAnts):
            for j in range(numberOfCities):
                visited [index (numberOfCities, i, j)] = 0 
                
        for i in range(numberOfCities):
            for j in range(numberOfCities):
                edge_pherom = pheromones [index (numberOfCities, i, j)]
                edge_weight = distanceArray [index (numberOfCities, i, j)]
                prob = 0.0
                if (edge_weight != 0):
                    prob = math.pow(edge_pherom, ALPHA) * math.pow((1 / edge_weight), BETA)
                else:
                    prob = math.pow(edge_pherom, ALPHA) * math.pow(INF, BETA)
                choiceInfo [index (numberOfCities, i, j)] = prob
        
        cuda.memcpy_htod(choiceInfo_device,choiceInfo)
        
        for ant in range(numberOfAnts):
            init = int(random.uniform(0,numberOfCities))
            tours[index(numberOfCities, ant, 0)] = init
            visited [index (numberOfCities, ant, init)] = 1
        
        cuda.memcpy_htod(visited_device,visited)
        cuda.memcpy_htod(tours_device,tours)
        
        gridDim = int(numberOfAnts / threadsPerBlock(numberOfAnts))
        antsPerBlock = int(threadsPerBlock (numberOfAnts))
        
        func = mod.get_function("constructTour")
        func(tours_device, visited_device, choiceInfo_device, probabilityArray, np.int32(numberOfCities),block = (antsPerBlock,1,1),grid = (gridDim,1))
        
        cuda.memcpy_dtoh(tours,tours_device)
        cuda.memcpy_dtoh(visited,visited_device)
        
        pheromones = evaporate (pheromones, numberOfCities)
        best = bestSolution (tours, distanceArray, numberOfAnts, numberOfCities)
        pheromones,distanceArray = reinforce (pheromones, distanceArray, best, numberOfCities) 
    best = bestSolution (tours, distanceArray, numberOfAnts, numberOfCities)
    return best 
