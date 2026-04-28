import numpy as np
import pandas as pd
import networkx as nx

G = nx.Graph()
edges =  [('A', 'B', 1), ('A', 'E', 1), ('A', 'F', 1), ('B', 'C', 1), ('B', 'F', 1), ('B', 'E', 1), ('C', 'D', 1), ('C', 'G', 1), ('D', 'G', 1), ('E', 'F', 1), ('G', 'H', 1)]
G.add_weighted_edges_from(edges)


import networkx as nx
import matplotlib.pyplot as plt


def draw_graph(G, selected_edges=None, coloring=None, show_all_edges=True):
    """
    Draw graph highlighting a bipartite subgraph solution.

    Parameters:
    - G: networkx graph
    - selected_edges: list of (u, v) edges kept in solution
    - coloring: dict {node: 0/1}
    - show_all_edges: if True, show removed edges faded; otherwise only show selected edges
    """

    pos = nx.spring_layout(G, seed=42)  # deterministic layout
    if coloring:
        node_colors = [
            'lightblue' if coloring[n] == 0 else 'lightgreen'
            for n in G.nodes()
        ]
    else:
        node_colors = 'skyblue'

    if selected_edges:
        selected_set = set(tuple(sorted(e)) for e in selected_edges)

        edge_colors = []
        edge_widths = []

        for u, v in G.edges():
            if tuple(sorted((u, v))) in selected_set:
                edge_colors.append('black')   # kept edges
                edge_widths.append(2.5)
            else:
                if show_all_edges:
                    edge_colors.append('lightgray')  # removed edges
                    edge_widths.append(1)
                else:
                    edge_colors.append('none')
                    edge_widths.append(0)
    else:
        edge_colors = 'gray'
        edge_widths = 1.5

    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=800,
        font_weight='bold',
        edge_color=edge_colors,
        width=edge_widths
    )

    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # if edge_labels:
    #     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Maximum Bipartite Subgraph Solution")
    plt.show()

print(nx.adjacency_matrix(G, nodelist=list(sorted(G.nodes))).toarray())
draw_graph(G)


# 1 A
def plot_normalized_degree(G):
    N = len(G.nodes)
    degree_sequence = pd.Series(sorted((d for n, d in G.degree()), reverse=True))
    counts = degree_sequence.value_counts()
    normalized_counts = counts.values/N

    plt.bar(counts.index, normalized_counts)
    plt.title("(Normalized) Degree histogram")
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.show()

# print(G.degree())
# plot_normalized_degree(G, N)

# 1 B
def dist_matrix(G):
    nodes = G.nodes
    print(list(sorted(nodes)))
    dist_matrix = nx.floyd_warshall_numpy(G, nodelist=list(sorted(nodes)))
    print(dist_matrix)

def semi_manual_dijkstra_stuff(G):
    N = len(G.nodes)
    nodes = list(G.nodes)
    
    closeness = {node: 0.0 for node in nodes}
    betweenness = {node: 0.0 for node in nodes}
    
    max_val = 0
    max_pairs = set()

    for s in nodes:
        # Dijkstra with paths 
        dists, paths = nx.single_source_dijkstra(G, s)
        
        # Closeness
        sum_dists = sum(dists.values())
        if sum_dists > 0:
            closeness[s] = (N - 1) / sum_dists
        
        # Diameter
        new_max_dist = max(dists.values())
        if new_max_dist > max_val:
            max_val = new_max_dist
            max_pairs = {frozenset([s, k]) for k, v in dists.items() if v == max_val}
        elif new_max_dist == max_val:
            for k, v in dists.items():
                if v == max_val:
                    max_pairs.add(frozenset([s, k]))

        # Betweeness
        for t, path in paths.items():
            if s == t:
                continue
            # A node is "between" s and t if it's in the path, 
            # excluding the start (s) and end (t) nodes.
            for internal_node in path[1:-1]:
                betweenness[internal_node] += 1

    norm_factor = (N - 1) * (N - 2)
    for node in betweenness:
        betweenness[node] /= norm_factor

    print(f"Manual Diameter: {max_val}")
    return {
        "closeness": closeness,
        "betweenness": betweenness,
        "diameter": max_val,
        "max_pairs": max_pairs
    }

def network_x_1b(G):
    print("Network X diameter")
    print(nx.diameter(G))
    print(nx.average_shortest_path_length(G, method='dijkstra'))

# dist_matrix(G)
# semi_manual_dijkstra_stuff(G)
# network_x_1b(G)


# 1 C
# clusters = nx.clustering(G)
# avg_cc = np.mean(list(clusters.values()))
# print(avg_cc)
# print(clusters)

# print(nx.transitivity(G))

# 1 D 
# print('Betweeness Centrality')
# print(nx.betweenness_centrality(G))
# print('Closeness Centrality')
# print(nx.closeness_centrality(G))

# print(semi_manual_dijkstra_stuff(G))


# 1 E
from ortools.sat.python import cp_model

def max_bipartite_subgraph_cp(G):
    model = cp_model.CpModel()

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]

    num_nodes = len(nodes)

    color = [model.NewBoolVar(f"color_{i}") for i in range(num_nodes)]
    use_edge = []

    for idx, (u, v) in enumerate(edges):
        e = model.NewBoolVar(f"use_edge_{idx}")
        use_edge.append(e)

        # If edge is used then endpoints must have different colors
        model.Add(color[u] != color[v]).OnlyEnforceIf(e)

    for i in range(num_nodes):
        incident_edges = [
            use_edge[idx]
            for idx, (u, v) in enumerate(edges)
            if u == i or v == i
        ]
        # no node can have less than 1 edge
        if incident_edges:
            model.Add(sum(incident_edges) >= 1)

    # Objective: maximize number of edges
    model.Maximize(sum(use_edge))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30

    result = solver.Solve(model)

    if result in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected_edges = [
            (idx_to_node[u], idx_to_node[v])
            for i, (u, v) in enumerate(edges)
            if solver.Value(use_edge[i]) == 1
        ]

        coloring = {
            idx_to_node[i]: solver.Value(color[i])
            for i in range(num_nodes)
        }

        return selected_edges, coloring
    else:
        return None, None


# selected_edges, coloring = max_bipartite_subgraph_cp(G)

# print("Selected edges:", selected_edges)
# print("Coloring:", coloring)
# draw_graph(G, selected_edges=selected_edges, coloring=coloring)
