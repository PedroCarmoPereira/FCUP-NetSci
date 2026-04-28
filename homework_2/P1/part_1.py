import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# So the idea is that F is a dead end/spider trap.
# The rest of the graph is well connected which boosts the other nodes betweeness and closeness
G1 = nx.DiGraph()

edges_1 = [
    ("A", "B"),
    ("B", "A"),
    ("B", "D"),
    ("D", "B"),
    ("B", "C"),
    ("C", "B"),
    ("C", "D"), 
    ("A", "C"),
    ("C", "A"),
    ("D", "E"),
    ("E", "F"),
    ("F", "F"),
]

G1.add_edges_from(edges_1)

pagerank = nx.pagerank(G1, alpha=0.85)

closeness = nx.closeness_centrality(G1)

betweenness = nx.betweenness_centrality(G1)

# 1
# print("PageRank:")
# for node, value in pagerank.items():
#     print(f"{node}: {value:.4f}")

# print("\nCloseness Centrality:")
# for node, value in closeness.items():
#     print(f"{node}: {value:.4f}")

# print("\nBetweenness Centrality:")
# for node, value in betweenness.items():
#     print(f"{node}: {value:.4f}")

# nx.draw(G1, with_labels=True)
# plt.show()

# 3
# https://github.com/melkael/pagerank_power_method/tree/master
# https://www.geeksforgeeks.org/python/page-rank-algorithm-implementation/
def my_pagerank(G, beta=0.85, eps=1e-6, debug=False):
    N = len(G.nodes)
    last_prs = { node: 0 for node in G.nodes}
    curr_prs = { node: 1/N for node in G.nodes}

    zip_sub = lambda keys, lp, cp: sum(abs(cp[k] - lp[k]) for k in keys)
    teleport = (1 - beta) / N
    i = 0
    while zip_sub(G.nodes, last_prs, curr_prs) > eps:
        last_prs = dict(curr_prs)
        for node in G.nodes:
            rank_sum = 0
            for neighbor in G.predecessors(node):
                if G.out_degree(neighbor) != 0:
                    rank_sum += last_prs[neighbor] / G.out_degree(neighbor)
            
            curr_prs[node] = teleport + (beta * rank_sum)
        i += 1
        if debug:
            print(f'ITER:{i}\n PRS:{curr_prs}')

    return curr_prs, i

# 4
G2 = nx.DiGraph()

edges_2 = [
    ("A", "B"),
    ("B", "D"),
    ("D", "B"),
    ("B", "C"),
    ("C", "D"),
    ("B", "E"), 
    ("B", "F"),
    ("E", "D"),
    ("F", "E"),
    ("F", "G"),
    ("G", "F"),
]

G2.add_edges_from(edges_2)

# prs, number_of_iters = my_pagerank(G2, debug=False)
# print(number_of_iters)
# print(prs)
# print(nx.pagerank(G2))

# nx.draw(G2, with_labels=True)
# plt.show()

# 5

start = 0
stop = 1
step = 0.05

def exp_5(G, start, stop, step):
    iterations = []
    node_ranks = {node: [] for node in G.nodes}
    betas = np.arange(start, stop + step, step)
    for b in betas:
        ranks, iters = my_pagerank(G, beta=b)
        iterations.append(iters)
        for node in G.nodes:
            node_ranks[node].append(ranks[node])

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot (a): Iterations vs Beta
    ax1.plot(betas, iterations, marker='o', color='teal')
    ax1.set_title("Convergence Speed vs. Beta")
    ax1.set_xlabel("Beta (Damping Factor)")
    ax1.set_ylabel("Iterations to Converge")
    ax1.grid(True)

    # Plot (b): Node Ranks vs Beta
    for node, values in node_ranks.items():
        ax1_plot = ax2.plot(betas, values, label=f"Node {node}")
    ax2.set_title("Node PageRank Values vs. Beta")
    ax2.set_xlabel("Beta (Damping Factor)")
    ax2.set_ylabel("PageRank Value")
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return iterations, node_ranks

exp_5(G2, start, stop, step)