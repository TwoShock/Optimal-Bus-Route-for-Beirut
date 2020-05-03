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
    __global__ void cuda_evaporate(double * pheromones, int n_cities, double evap_rate) { 
    
    	int edge_id = threadIdx.x + blockIdx.x * blockDim.x; 
    	pheromones[edge_id] =pheromones[edge_id]* evap_rate;
    } 
    
    __global__ void cuda_reinforce(double * pheromones, int * distances, int * path, int n_cities, double amount) {
    
    	int col_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    	int origin = path [col_id];
    	int dest = path [col_id + 1];
        int idx = index(n_cities, origin, dest);
    	pheromones[idx] = pheromones[idx] + amount;
    	pheromones [index(n_cities, dest, origin)] = pheromones [index(n_cities, dest, origin)]+amount; 
    }
    
    __global__ void cuda_construct_tour(int * tours, int * visited, double * choiceinfo, double * probs, int n_cities) {
    
    	int line_id = blockDim.x * blockIdx.x + threadIdx.x; 
    
    	for (int step = 1; step <n_cities; step++) { 
    
    		int current = tours[index(n_cities, line_id, step - 1)];
    		double sum_probs = 0.0;
    
    		for (int i = 0; i <n_cities; i ++) {
    			if (visited [index(n_cities, line_id, i)] == 1) 
    				probs [index(n_cities, line_id, i)] = 0.0;
    			else {
    				double current_prob = choiceinfo[index(n_cities, current, i)]; 
    				probs [index(n_cities, line_id, i)] = current_prob; 
    				sum_probs = sum_probs+current_prob; 
    			}
    		}
    
    		double random;
    		curandState_t state;
    		curand_init ((unsigned long long) clock(), 0, 0, & state); 
    		random = curand_uniform(& state); 
    		random = random * sum_probs; 
    
    		int next;
    		double sum = probs [index(n_cities, line_id, 0)];
    
    		for (next = 0; sum <random; next ++) {
    			sum  = sum + probs [index(n_cities, line_id, next + 1)];
    		} 
    
    		tours [index(n_cities, line_id, step)] = next; 
    		visited [index(n_cities, line_id, next)] = 1;
    	} 
    }
}
""",no_extern_c=True)



#define randdouble () ((double) rand () / (double) RAND_MAX) 

Infinity = 65536 
NUMBER_OF_ITERATIONS = 50
INIT_PHEROMONE_AMOUNT = 1.0
EVAPORATION_RATE = 0.5
ALFA = 1
BETA = 2

def index(length, line, column):
    return (column + line * length) 

def threads( n_ants):
    n_threds = 1
    while(n_threds * 2 <n_ants):
        n_threds*= 2
    return n_threds

def thread_per_block(n_ants):
	blocks = math.log(n_ants)
	return math.pow(2, blocks)

def calculate_pathcost (distances,path, n_cities):
    cost = 0
    for count in range(n_cities-1):
        idx = index(n_cities, path [count], path [count + 1])
        if idx<Infinity:
            cost += distances[idx]
        else:
            cost+=Infinity
    return cost 

def best_solution (tours, distances, n_ants, n_cities):
    best_tour = tours
    for tour in range(n_ants):
        bestCost = calculate_pathcost(distances, best_tour, n_cities)
        currentCost = calculate_pathcost(distances,tours [index (n_cities, tour, 0):], n_cities)
        if (currentCost<bestCost): 
            best_tour = tours[index(n_cities, tour, 0):]
    return best_tour


def evaporate (pheromones,n_cities):
    pheromones_device = np.empty_like(pheromones)
    
    pheromones_device = cuda.mem_alloc(pheromones.nbytes)
    cuda.memcpy_htod(pheromones_device,pheromones)
    func = mod.get_function("cuda_evaporate")
    func(pheromones_device, np.int32(n_cities), np.int32(EVAPORATION_RATE),block = (n_cities,1,1),grid = (n_cities,1))
    cuda.memcpy_dtoh(pheromones,pheromones_device)
    return pheromones

def reinforce(pheromones,distances, path,n_cities):
    amount = float(1.0 / calculate_pathcost (distances, path, n_cities))
    
    path_device = np.empty_like(path)
    distances_device = np.empty_like(distances)
    pheromones_device = np.empty_like(pheromones)
    
    path_device = cuda.mem_alloc(path.nbytes)
    distances_device = cuda.mem_alloc(distances.nbytes)
    pheromones_device = cuda.mem_alloc(pheromones.nbytes)

    cuda.memcpy_htod(path_device,path)
    cuda.memcpy_htod(distances_device,distances)
    cuda.memcpy_htod(pheromones_device,pheromones)
    
    func = mod.get_function("cuda_reinforce")
    func(pheromones_device, distances_device, path_device, np.int32(n_cities),np.float32(amount),block=(1,1,1),grid=(n_cities,1))
    
    cuda.memcpy_dtoh(distances,distances_device)
    cuda.memcpy_dtoh(pheromones,pheromones_device)
    
    return (pheromones,distances)
	

def run (distances, n_cities, n_ants):
    
    pheromones = np.empty((n_cities*n_cities,),dtype=np.float32)
    tours = np.empty((n_cities*n_ants,),dtype=int)
    visited = np.empty((n_cities*n_ants,),dtype=int)
    choiceinfo = np.empty((n_cities*n_cities,),dtype=np.float32)
    
    
    distances_device = np.empty_like(distances)
    tours_device = np.empty_like(tours)
    visited_device = np.empty_like(visited)
    choiceinfo_device = np.empty_like(choiceinfo)
    probs = np.empty_like(pheromones)
    
    distances_device = cuda.mem_alloc(distances.nbytes)
    tours_device = cuda.mem_alloc(tours.nbytes)
    visited_device = cuda.mem_alloc(visited.nbytes)
    choiceinfo_device = cuda.mem_alloc(choiceinfo.nbytes)
    probs = cuda.mem_alloc(pheromones.nbytes)

    cuda.memcpy_htod(distances_device,distances)
    
    for i in range(n_cities):
        for j in range(n_cities):
            pheromones [index (n_cities, i, j)] = INIT_PHEROMONE_AMOUNT
            
    for iteration in range(NUMBER_OF_ITERATIONS):

        for i in range(n_ants):
            for j in range(n_cities):
                tours [index(n_cities, i, j)] = Infinity 
        
        for i in range(n_ants):
            for j in range(n_cities):
                visited [index (n_cities, i, j)] = 0 
                
        for i in range(n_cities):
            for j in range(n_cities):
                edge_pherom = pheromones [index (n_cities, i, j)]
                edge_weight = distances [index (n_cities, i, j)]
                prob = 0.0
                if (edge_weight != 0):
                    prob = math.pow(edge_pherom, ALFA) * math.pow((1 / edge_weight), BETA)
                else:
                    prob = math.pow(edge_pherom, ALFA) * math.pow(Infinity, BETA)
                choiceinfo [index (n_cities, i, j)] = prob
        
        cuda.memcpy_htod(choiceinfo_device,choiceinfo)
        
        for ant in range(n_ants):
            init = int(random.uniform(0,n_cities))
            tours[index(n_cities, ant, 0)] = init
            visited [index (n_cities, ant, init)] = 1
        
        cuda.memcpy_htod(visited_device,visited)
        cuda.memcpy_htod(tours_device,tours)
        
        gridDim = int(n_ants / thread_per_block(n_ants))
        antsPerBlock = int(thread_per_block (n_ants))
        
        func = mod.get_function("cuda_construct_tour")
        func(tours_device, visited_device, choiceinfo_device, probs, np.int32(n_cities),block = (antsPerBlock,1,1),grid = (gridDim,1))
        
        cuda.memcpy_dtoh(tours,tours_device)
        cuda.memcpy_dtoh(visited,visited_device)
        
        pheromones = evaporate (pheromones, n_cities)
        best = best_solution (tours, distances, n_ants, n_cities)
        pheromones,distances = reinforce (pheromones, distances, best, n_cities) 
    best = best_solution (tours, distances, n_ants, n_cities)
    return best 
    
