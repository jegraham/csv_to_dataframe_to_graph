import pandas as pd
import math
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# Import the Data
filepathC = '/home/girl-boss/Downloads/May 4th Dataset Car and Tower  - Sheet1.csv'
df = pd.read_csv(filepathC)

print("Display Graph")

# Assign numeric IDs based on row index
car_ids = torch.arange(len(df), dtype=torch.float)
vertex = car_ids  # Use the IDs directly as the vertex tensor

edge = []

# Loop row by row
for index, row in df.iterrows():
    x1 = df.loc[index, 'X']
    y1 = df.loc[index, 'Y']

    if index == len(df) - 1:
        break  # exit for loop

    x2 = df.loc[index + 1, 'X']
    y2 = df.loc[index + 1, 'Y']

    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    #print(distance)

    # Convert the dataframe to the edge index matrix
    if distance <= 3000:
        new_edge = (index, index + 1)
        edge.append(new_edge)

G = nx.Graph()
G.add_nodes_from(vertex.numpy())
G.add_edges_from(edge)

# Visualize the graph
pos = nx.spring_layout(G)  # Define the layout of the graph
nx.draw(G, with_labels=True)
plt.savefig('graph.png')

# Convert the graph to an adjacency matrix
adj_matrix = nx.convert_matrix.to_numpy_array(G)

# Save the adjacency matrix to a file
np.savetxt('adjacency_matrix.csv', adj_matrix, delimiter=',')
