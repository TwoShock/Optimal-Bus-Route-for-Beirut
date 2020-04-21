import networkx as nx
import matplotlib.pyplot as plt
import random
import osmnx as ox

random.seed(42)
def createRandomGraph(size = 1000,minPopulation = 500,maxPopulation = 20000,minEdgeWeight = 10,maxEdgeWeight = 100):
    G = nx.Graph()
    for i in range(size):
        G.add_node(i,population = random.randint(minPopulation,maxPopulation))
    for i in range(size):
        for j in range(i+1,size):
            G.add_edge(i,j,weight = random.randint(minEdgeWeight,maxEdgeWeight))
    return G

def plotGraph(G):
    plt.subplot(121)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()

def main():
    G = ox.graph_from_place('Beirut,Lebanon', network_type='drive')
    ox.plot_graph(G)
    print(list(G.nodes(data=True))) 
    
if __name__ == "__main__":
    main()