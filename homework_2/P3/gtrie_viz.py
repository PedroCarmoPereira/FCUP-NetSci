import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def parse_and_draw_gtrie(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Split content to get the Motif Analysis section
    if "Motif Analysis Results" not in content:
        print("Could not find Motif Analysis section.")
        return
    
    motif_section = content.split("Motif Analysis Results")[-1]
    
    # Regex to find blocks of 0s and 1s that form the adjacency matrix
    # This looks for 4 rows of 4 digits (since your Subgraph Size is 4)
    pattern = r"((?:[01]{4}\n?){4})"
    matches = re.findall(pattern, motif_section)

    count = 1
    for match in matches:
        # Clean the string and create a list of lists (the matrix)
        rows = match.strip().split('\n')
        matrix = [[int(char) for char in row] for row in rows]
        
        # Create Graph from adjacency matrix
        # Note: Since your output says 'Directed: NO', we use Graph()
        G = nx.from_numpy_array(np.array(matrix))
        
        # Remove self-loops if any (diagonal of the matrix)
        G.remove_edges_from(nx.selfloop_edges(G))

        # Check if the subgraph actually has edges (Org_Freq > 0)
        # We only save if there is at least one occurrence or edge to show
        if G.number_of_edges() > 0:
            plt.figure(figsize=(4, 4))
            plt.title(f"Subgraph Structure {count}")
            
            # Simple circular layout for small subgraphs
            pos = nx.circular_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                    node_size=800, font_weight='bold')
            
            # Save to file
            file_name = f"{count}.png"
            plt.savefig(file_name)
            plt.close()
            print(f"Saved {file_name}")
            count += 1


if __name__ == "__main__":
    # Ensure your output file is named 'results.txt' or change this string
    parse_and_draw_gtrie("results_gtrie_4.txt")