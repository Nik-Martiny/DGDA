import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


NUM_NODES = 200

NUM_CLIENT = 80
INTERNAL_IDX = 120
WEB_IDX = 150
ROUTER_IDX = 170
IOT_IDX = 200

# This is going to be dynamically created
# Starts at node 0 and ends at node NUM_NODES - 1

all_nodes = range(NUM_NODES)

client_group = {} # 0-39%
internal_server_group = {}
web_server_group = {} # 60-74%
router_group = {} # 75-84%
IoT_group = {} # 85-NUM_NODES-1

# Create the networkX graph
# It is an undirected graph
G = nx.Graph()

G.add_nodes_from(all_nodes)



client_group = all_nodes[:NUM_CLIENT]
internal_server_group = all_nodes[NUM_CLIENT:INTERNAL_IDX]
web_server_group = all_nodes[INTERNAL_IDX:WEB_IDX]
router_group = all_nodes[WEB_IDX:ROUTER_IDX]
Iot_group = all_nodes[ROUTER_IDX:IOT_IDX]



# Start with the routers and switches

# Split the router group into two groups
# 150-154 <-- Take each index node and pair it with the index+10 node
# 155-159 <-- Same thing here

router_switch_pairs = {(router_group[n], router_group[n+10]) for n in range(int(len(router_group)/2))}


# Add these pairs as edges in the graph
G.add_edges_from(router_switch_pairs)

# Draw these in a simple graph

#nx.draw_spring(G, with_labels=True)
#plt.show()

# Add the edges for the star topology with the routers

# 0->1 1->2 2->3 3->4 4->0

router_group_1 = {router_group[i] for i in range(int(len(router_group)/4))}
router_group_2 = {router_group[i+len(router_group_1)*2] for i in range(int(len(router_group)/4))}


for n in router_group_2:
    print(n)
#print(intra_router_pairs)
