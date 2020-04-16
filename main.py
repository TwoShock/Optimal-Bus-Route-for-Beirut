import networkx as nx
import matplotlib.pyplot as plt
import random

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
    G = createRandomGraph(size = 25)
    plotGraph(G)
    
if __name__ == "__main__":
    main()