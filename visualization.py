import opendssdirect as dss
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

###############################################################################
# 1) Compile circuit
###############################################################################
dss.Command('Redirect "C:\Program Files\OpenDSS\IEEETestCases\IEEE118Bus\master_file.dss"')
dss.Solution.Solve()

###############################################################################
# 2) Gather bus voltages
###############################################################################
bus_voltage_dict = {}
for bus_name in dss.Circuit.AllBusNames():
    dss.Circuit.SetActiveBus(bus_name)
    bus_voltage_dict[bus_name] = dss.Bus.kVBase()

###############################################################################
# 3) Identify load buses
###############################################################################
load_buses = set()
for ld in dss.Loads.AllNames():
    dss.Loads.Name(ld)
    bus_names = dss.CktElement.BusNames()  # e.g. ["Bus123.1.2"]
    if bus_names:
        load_bus = bus_names[0].split('.')[0]
        load_buses.add(load_bus)

###############################################################################
# 4) Build main graph (Lines + Transformers)
###############################################################################
G = nx.Graph()

# Lines
for ln in dss.Lines.AllNames():
    dss.Lines.Name(ln)
    b1 = dss.Lines.Bus1().split('.')[0]
    b2 = dss.Lines.Bus2().split('.')[0]
    # Check if line is switch
    is_switch = bool(dss.Lines.IsSwitch())
    G.add_edge(b1, b2, is_switch=is_switch)

# Transformers
for xfmr in dss.Transformers.AllNames():
    dss.Transformers.Name(xfmr)
    bus_names = dss.CktElement.BusNames()
    if len(bus_names) >= 2:
        b1 = bus_names[0].split('.')[0]
        b2 = bus_names[1].split('.')[0]
        # Usually not a switch
        G.add_edge(b1, b2, is_switch=False)

###############################################################################
# 5) Define voltage categories
###############################################################################
def get_voltage_category(kv):
    """
    Example thresholds for HV, MV, LV1, LV2.
    Adjust to your system needs.
    """
    if kv >= 60:
        return "HV"
    elif kv >= 15:
        return "MV"
    elif kv > 6:
        return "LV2"   # ~0.48 kV
    elif kv > 0:
        return "LV1"   # ~0.208 kV
    else:
        return "Unknown"

# Assign node attributes
for node in G.nodes():
    kv = bus_voltage_dict.get(node, 0.0)
    cat = get_voltage_category(kv)
    G.nodes[node]["base_kv"]  = kv
    G.nodes[node]["category"] = cat
    G.nodes[node]["is_load"]  = (node in load_buses)

# Identify bridging nodes (edges that cross categories)
bridge_nodes = set()
for (u, v) in G.edges():
    if G.nodes[u]["category"] != G.nodes[v]["category"]:
        bridge_nodes.add(u)
        bridge_nodes.add(v)

for nd in G.nodes():
    G.nodes[nd]["bridge_node"] = (nd in bridge_nodes)

###############################################################################
# 6) Split into subgraphs: HV, MV, LV1, LV2, Unknown
###############################################################################
categories = ["HV", "MV", "LV1", "LV2", "Unknown"]
subgraphs = {c: nx.Graph() for c in categories}

# Populate subgraphs
for n in G.nodes():
    cat = G.nodes[n]["category"]
    subgraphs[cat].add_node(n)
    # Copy node attributes
    for attr, val in G.nodes[n].items():
        subgraphs[cat].nodes[n][attr] = val

# Add edges only if endpoints share the same category
for (u, v) in G.edges():
    c_u = G.nodes[u]["category"]
    c_v = G.nodes[v]["category"]
    if c_u == c_v:
        subgraphs[c_u].add_edge(u, v, **G[u][v])

###############################################################################
# 7) Plot each category, with special layout for LV1
###############################################################################
for cat, sg in subgraphs.items():
    if sg.number_of_nodes() == 0:
        continue

    # Choose a layout
    if cat == "LV1":
        # For a dense/meshed LV1 subgraph, Kamada-Kawai might be more "spread out"
        layout_func = lambda g: nx.kamada_kawai_layout(g, scale=3.0)
        fig_size = (10, 8)   # bigger figure
        node_size = 200
        label_size = 6
        edge_width = 2.0
    else:
        # Use dot layout for others (or pick any you like)
        layout_func = lambda g: graphviz_layout(g, prog="dot")
        fig_size = (10, 8)
        node_size = 400
        label_size = 6
        edge_width = 1.0

    plt.figure(figsize=fig_size, dpi=150)

    # Get positions
    pos = layout_func(sg) if cat == "LV1" else layout_func(sg)

    # Prepare node colors
    node_colors = []
    node_borders = []
    for n in sg.nodes():
        # Fill color for load vs. non-load
        node_colors.append("red" if sg.nodes[n]["is_load"] else "lightgray")
        # Outline bridging nodes
        node_borders.append("magenta" if sg.nodes[n]["bridge_node"] else "black")

    nx.draw_networkx_nodes(
        sg, pos,
        node_color=node_colors,
        edgecolors=node_borders,
        node_size=node_size,
    )
    nx.draw_networkx_labels(
        sg, pos,
        font_size=label_size
    )

    # Edge color for switch lines
    edge_colors = []
    for (u, v) in sg.edges():
        edge_colors.append("red" if sg[u][v]["is_switch"] else "gray")

    nx.draw_networkx_edges(
        sg, pos,
        edge_color=edge_colors,
        width = edge_width
    )

    plt.title(f"{cat} Subgraph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{cat}_subgraph.png")
    plt.show()
