import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
import osmnx as ox
import pickle
import numpy as np
import pants

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
    X = np.empty((0,2),dtype = float)
    for i in range(len(buildingCentroids)):
        currentPoint = list(zip(*buildingCentroids.iloc[i].xy))
        X = np.append(X,currentPoint,axis = 0)
    return KMeans(n_clusters=27,random_state=0).fit(X).cluster_centers_

def getListOfNearestNodesToCentroids(centroids,driveMap):
    return ox.get_nearest_nodes(driveMap,[x[0] for x in centroids],[y[1] for y in centroids])
driveMap,buildingMap = loadPickle('driveMap.pkl'),loadPickle('buildingMap.pkl')
def distance(node_a,node_b):
    return nx.shortest_path_length(G=driveMap,source=node_a,target=node_b,weight='length')
def main():
    centroids = getBuildingCentroids(buildingMap)    
    nearest_nodes = getListOfNearestNodesToCentroids(centroids,driveMap)
    
    x = [driveMap.nodes[nearest_nodes[i]]['x'] for i in range(len(nearest_nodes))]
    y = [[driveMap.nodes[nearest_nodes[i]]['y'] for i in range(len(nearest_nodes))]]
    
    world = pants.World(nearest_nodes,distance)
    solver = pants.Solver()
    solution = solver.solve(world)
    print(solution.distance)
    print(type(solution.tour))    # Nodes visited in order
    print(solution.path)    # Edges taken in order
    

    #nodes = [driveMap.nodes[solution.tour[i]] for i in range(len(solution.tour))]
    routes = []
    for i in range(len(solution.tour)-1):
        routes.append(nx.shortest_path(driveMap,solution.tour[i],solution.tour[i+1]))
    
    fig,ax = ox.plot_graph_routes(driveMap,routes)
    
    #plt.scatter(x,y)
    #buildingMap.plot(ax=ax,facecolor='red',alpha = 0.7)
    #plt.show()
    #route = nx.shortest_path(driveMap,nearest_nodes[20],nearest_nodes[13])
    #route_length_km = sum([driveMap.edges[u][v][0]['length'] for u, v in zip(route, route[1:])]) / 1000
    #print(nx.shortest_path_length(G=driveMap,source=nearest_nodes[0],target=nearest_nodes[20],weight='length'))
    #print('here',route)
    #ox.plot_graph_route(driveMap,route,fig_height=10,fig_width=10)
    
if __name__ == "__main__":
    main()