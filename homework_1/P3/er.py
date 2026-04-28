import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def read_graph(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    n = int(lines[0].strip())

    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))

    for line in lines[1:]:
        if line.strip() == "":
            continue
        a, b = map(int, line.split())
        G.add_edge(a, b)

    return G


def dfs_size(node, G, visited):
    stack = [node]
    size = 0

    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            size += 1
            stack.extend(G.neighbors(current))

    return size


def our_giant_component_size(G):
    visited = set()
    max_size = 0

    for node in G.nodes():
        if node not in visited:
            component_size = dfs_size(node, G, visited)
            max_size = max(max_size, component_size)

    return max_size


def nx_giant_component_size(G):
    largest_cc = max(nx.connected_components(G), key=len)
    return len(largest_cc)



def nx_generate_er_graph(n, p, seed=None):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return G


def our_generate_er_graph(n, p, seed=None):
    if seed is not None:
        random.seed(seed)

    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))  # nodes labeled 1..n

    # iterate over all unordered pairs (i, j)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if random.random() < p:
                G.add_edge(i, j)

    return G

def save_graph(G, filename):
    with open(filename, "w") as f:
        f.write(f"{G.number_of_nodes()}\n")
        for u, v in G.edges():
            f.write(f"{u} {v}\n")

def p_experiment(start, stop, step, N=2000, seed=42):
    results = []
    for p in  :
        G = our_generate_er_graph(N, p, seed=seed)
        size = our_giant_component_size(G)
        results.append({
            'p':p,
            'size':size
        })

    return pd.DataFrame(results)



if __name__ == "__main__":

    G1 = our_generate_er_graph(2000, 0.0001, seed=42)
    G2 = our_generate_er_graph(2000, 0.005, seed=42)
    g1_size = our_giant_component_size(G1)
    g2_size = our_giant_component_size(G2)

    save_graph(G1, 'random1.txt')
    save_graph(G2, 'random2.txt')

    print(f'ER({2000}, {0.0001}), GC size: {g1_size}')
    print(f'ER({2000}, {0.005}), GC size: {g2_size}')
    
    df = p_experiment(0.0001, 0.005, 0.0001, N=2000)
    plt.plot(df['p'], df['size'])
    plt.xlabel('p (probabilidade de ligação entre vértices)')
    plt.ylabel('Tamanho Maior Componente')
    plt.axvline(0.0005, color='r')
    plt.show()
