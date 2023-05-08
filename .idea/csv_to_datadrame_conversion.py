import pandas as pd
import math
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.transforms import NormalizeFeatures

# Import the Data
from google.colab import files
uploaded = files.upload()

file_name = next(iter(uploaded))
df = pd.read_csv(file_name)

print("Data Frame")
print(df)

# Convert 'Car ID' column to numeric values
df['Car ID'] = pd.to_numeric(df['Car ID'], errors='coerce')

# Assign numeric IDs based on 'Car ID' column
car_ids = torch.tensor(df['Car ID'].values, dtype=torch.float)
vertex = car_ids  # Use 'Car ID' as the vertex tensor

edge = []
x_list = []
y_list = []

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
        x_list.append(x1)
        y_list.append(y1)

data = Data(x=torch.tensor(x_list), y=torch.tensor(y_list), edge_index=torch.tensor(edge).t().contiguous())


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


# Apply feature normalization
transform = NormalizeFeatures()
data = transform(data)




# TEST PLOTTING GRID GRAPH
x = df['X']
y = df['Y']


plt.plot(x, y)
plt.grid(True)

# Add labels to the points based on a column from the CSV
labels = df['Car ID']
for i, label in enumerate(labels):
    plt.text(x[i], y[i], label)


plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('CSV Graph Output')

# Set the limits and tick marks for the x and y axes
x_padding = 5 # Set the padding for the x-axis
y_padding = 5 # Set the padding for the y-axis
plt.xlim(min(x) - x_padding, max(x) + x_padding)
plt.ylim(min(y) - y_padding, max(y) + y_padding)
x_ticks = range(int(min(x)) - x_padding, int(max(x)) + x_padding + 1)
y_ticks = range(int(min(y)) - y_padding, int(max(y)) + y_padding + 1)
plt.xticks(x_ticks)
plt.yticks(y_ticks)

# Add a perimeter or bounding box around the graph
x_range = max(x) - min(x) + 2*x_padding
y_range = max(y) - min(y) + 2*y_padding
rect = plt.Rectangle((min(x) + 20 - x_padding, min(y) + 20 - y_padding), x_range, y_range, fill=False, linewidth=2)
plt.gca().add_patch(rect)

# Show the plot
plt.show()

