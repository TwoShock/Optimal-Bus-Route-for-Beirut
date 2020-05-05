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
from cuda_kmeans import CudaKMeans
from sklearn.cluster import KMeans

def pickleObject(obj,filename):#used for object serilaization
    outFile = open(filename,'wb')
    pickle.dump(obj,outFile)
    outFile.close()

def loadPickle(filename):#used for object deserilaization
    inFile = open(filename,'rb')
    obj = pickle.load(inFile)
    inFile.close()
    return obj

def getBuildingCentroids(buildingMap):
    """
        Input: building Map
        Output: numpy array contianing centroid locations
    """
    buildingCentroids = buildingMap.centroid
    X = []
    Y = []
    for i in range(len(buildingCentroids)):
        X.append(buildingCentroids.iloc[i].x)
        Y.append(buildingCentroids.iloc[i].y)
    return buildingCentroids,CudaKMeans(X,Y,len(X))

def getListOfNearestNodesToCentroids(centroids,driveMap):
    return ox.get_nearest_nodes(driveMap,[x[0] for x in centroids],[y[1] for y in centroids])


driveMap = loadPickle('driveMap.pkl')

buildingMap = loadPickle('buildingMap.pkl')

def distance(node_a,node_b):
    return nx.shortest_path_length(G=driveMap,source=node_a,target=node_b,weight='length')



def main():
    centroids,Stops = getBuildingCentroids(buildingMap)   
            
    nearest_nodes = getListOfNearestNodesToCentroids(Stops,driveMap)

    x = [driveMap.nodes[nearest_nodes[i]]['x'] for i in range(len(nearest_nodes))]
    y = [driveMap.nodes[nearest_nodes[i]]['y'] for i in range(len(nearest_nodes))]
    
    G = ox.graph_from_place('Beirut,Lebanon',network_type = 'drive')
    fig,ax = ox.plot_graph(G, show=False, close=False)

    for i in range(len(x)):
        ax.scatter(x[i], y[i], c='red')
    plt.show()
    world = pants.World(nearest_nodes,distance)
    solver = pants.Solver()
    solution = solver.solve(world)
    print(solution.distance)
    print(solution.path)
    

    routes = []
    for i in range(len(solution.tour)-1):
        routes.append(nx.shortest_path(driveMap,solution.tour[i],solution.tour[i+1]))
    
    fig,ax = ox.plot_graph_routes(driveMap,routes)

   
if __name__ == "__main__":
    main()
