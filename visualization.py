# import opendssdirect as dss
# import networkx as nx
# import matplotlib.pyplot as plt
# from networkx.drawing.nx_agraph import graphviz_layout
# import matplotlib as mpl
# import matplotlib.patches as mpatches
# import matplotlib.lines as mlines
import numpy as np
#
# ###############################################################################
# # 0) Global Figure and Style Settings
# ###############################################################################
# mpl.rcParams['figure.dpi'] = 300
# mpl.rcParams['savefig.dpi'] = 300
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.size'] = 10
#
# ###############################################################################
# # 1) Compile circuit
# ###############################################################################
# dss.Command('Redirect "Master.dss"')
# dss.Solution.Solve()
#
# ###############################################################################
# # 2) Gather bus voltages
# ###############################################################################
# bus_voltage_dict = {}
# for bus_name in dss.Circuit.AllBusNames():
#     dss.Circuit.SetActiveBus(bus_name)
#     bus_voltage_dict[bus_name] = dss.Bus.kVBase()
#
# ###############################################################################
# # 3) Identify load buses
# ###############################################################################
# load_buses = set()
# for ld in dss.Loads.AllNames():
#     dss.Loads.Name(ld)
#     bus_names = dss.CktElement.BusNames()  # e.g. ["Bus123.1.2"]
#     if bus_names:
#         load_bus = bus_names[0].split('.')[0]
#         load_buses.add(load_bus)
#
# ###############################################################################
# # 4) Build main graph (Lines + Transformers)
# ###############################################################################
# G = nx.Graph()
#
# # Lines
# for ln in dss.Lines.AllNames():
#     dss.Lines.Name(ln)
#     b1 = dss.Lines.Bus1().split('.')[0]
#     b2 = dss.Lines.Bus2().split('.')[0]
#     # Check if line is a switch
#     is_switch = bool(dss.Lines.IsSwitch())
#     G.add_edge(b1, b2, is_switch=is_switch)
#
# # Transformers
# for xfmr in dss.Transformers.AllNames():
#     dss.Transformers.Name(xfmr)
#     bus_names = dss.CktElement.BusNames()
#     if len(bus_names) >= 2:
#         b1 = bus_names[0].split('.')[0]
#         b2 = bus_names[1].split('.')[0]
#         # Typically not a switch
#         G.add_edge(b1, b2, is_switch=False)
#
# ###############################################################################
# # 5) Define voltage categories
# ###############################################################################
# def get_voltage_category(kv):
#     """
#     Categorize according to your new requirements:
#       - HV = 230 kV or higher
#       - MV = 13.8 kV or higher
#       - SN (Spot Network) = 0.48 kV or higher
#       - LV = 0.208 kV or higher
#       - Unknown otherwise
#     """
#     if kv >= 230.0/np.sqrt(3):
#         return "HV"
#     elif kv >= 13.8/np.sqrt(3):
#         return "MV"
#     elif kv >= 0.48/np.sqrt(3):
#         return "SN"     # Spot Network
#     elif kv >= 0.208/np.sqrt(3):
#         return "LV"     # Low Voltage
#     else:
#         return "Unknown"
#
# # Assign node attributes
# for node in G.nodes():
#     kv = bus_voltage_dict.get(node, 0.0)
#     cat = get_voltage_category(kv)
#     G.nodes[node]["base_kv"]  = kv
#     G.nodes[node]["category"] = cat
#     G.nodes[node]["is_load"]  = (node in load_buses)
#
# # Identify bridging nodes (those connecting different voltage categories)
# bridge_nodes = set()
# for (u, v) in G.edges():
#     if G.nodes[u]["category"] != G.nodes[v]["category"]:
#         bridge_nodes.add(u)
#         bridge_nodes.add(v)
#
# for nd in G.nodes():
#     G.nodes[nd]["bridge_node"] = (nd in bridge_nodes)
#
# ###############################################################################
# # 6) Split into subgraphs by category
# ###############################################################################
# categories = ["HV", "MV", "SN", "LV", "Unknown"]
# subgraphs = {c: nx.Graph() for c in categories}
#
# # Populate subgraphs
# for n in G.nodes():
#     cat = G.nodes[n]["category"]
#     subgraphs[cat].add_node(n)
#     # Copy node attributes
#     for attr, val in G.nodes[n].items():
#         subgraphs[cat].nodes[n][attr] = val
#
# # Add edges only if endpoints share the same category
# for (u, v) in G.edges():
#     c_u = G.nodes[u]["category"]
#     c_v = G.nodes[v]["category"]
#     if c_u == c_v:
#         subgraphs[c_u].add_edge(u, v, **G[u][v])
#
# ###############################################################################
# # 7) Plot each category
# ###############################################################################
# for cat, sg in subgraphs.items():
#     if sg.number_of_nodes() == 0:
#         continue
#
#     # Pick layout & styling for each category:
#     if cat in ["HV", "MV"]:
#         # Dot layout tends to give a hierarchical or radial look
#         layout_func = lambda g: graphviz_layout(g, prog="dot")
#         fig_size = (7, 5)
#         node_size = 300
#         edge_width = 1.0
#     elif cat in ["SN", "LV"]:
#         # Kamada-Kawai for a more spread out look in dense distribution networks
#         layout_func = lambda g: nx.kamada_kawai_layout(g, scale=3.0)
#         fig_size = (8, 6)
#         node_size = 250
#         edge_width = 1.5
#     else:
#         # Unknown category
#         layout_func = lambda g: nx.spring_layout(g, seed=42)
#         fig_size = (6, 4)
#         node_size = 200
#         edge_width = 1.0
#
#     plt.figure(figsize=fig_size)
#
#     # Compute positions
#     pos = layout_func(sg)
#
#     # Node colors
#     node_colors = []
#     node_borders = []
#     for n in sg.nodes():
#         if sg.nodes[n]["is_load"]:
#             # Light salmon for load
#             node_colors.append("#FCA082")
#         else:
#             # Light gray for non-load
#             node_colors.append("#F0F0F0")
#         # Blue outline for bridging
#         node_borders.append("blue" if sg.nodes[n]["bridge_node"] else "black")
#
#     nx.draw_networkx_nodes(
#         sg, pos,
#         node_color=node_colors,
#         edgecolors=node_borders,
#         node_size=node_size
#     )
#
#     # Edge colors
#     edge_colors = []
#     for (u, v) in sg.edges():
#         if sg[u][v]["is_switch"]:
#             # Switch lines in a muted blue
#             edge_colors.append("#3182BD")
#         else:
#             # Normal lines in dark gray
#             edge_colors.append("#636363")
#
#     nx.draw_networkx_edges(
#         sg, pos,
#         edge_color=edge_colors,
#         width=edge_width
#     )
#
#     # Labels with a white bounding box
#     nx.draw_networkx_labels(
#         sg, pos,
#         font_size=7,
#         font_family="serif",
#         font_color="black",
#         bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.3)
#     )
#
#     # Legend
#     load_patch = mpatches.Patch(facecolor="#FCA082", edgecolor="black", label="Load Bus")
#     non_load_patch = mpatches.Patch(facecolor="#F0F0F0", edgecolor="black", label="Non-Load Bus")
#     bridge_patch = mpatches.Patch(facecolor="none", edgecolor="blue", label="Bridging Node")
#     switch_line = mlines.Line2D([], [], color="#3182BD", linewidth=2, label="Switch Line")
#     normal_line = mlines.Line2D([], [], color="#636363", linewidth=2, label="Normal Line")
#
#     plt.legend(
#         handles=[load_patch, non_load_patch, bridge_patch, switch_line, normal_line],
#         loc="best",
#         fontsize=7,
#         frameon=True
#     )
#
#     plt.title(f"{cat} Subgraph", fontsize=12)
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(f"{cat}_subgraph.png", bbox_inches='tight')
#     plt.show()
import opendssdirect as dss
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

###############################################################################
# 0) Global Figure and Style Settings
###############################################################################
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10

###############################################################################
# 1) Compile circuit
###############################################################################
dss.Command('Redirect "Master.dss"')
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
    # Check if line is a switch
    is_switch = bool(dss.Lines.IsSwitch())
    G.add_edge(b1, b2, is_switch=is_switch)

# Transformers
for xfmr in dss.Transformers.AllNames():
    dss.Transformers.Name(xfmr)
    bus_names = dss.CktElement.BusNames()
    if len(bus_names) >= 2:
        b1 = bus_names[0].split('.')[0]
        b2 = bus_names[1].split('.')[0]
        # Typically not a switch
        G.add_edge(b1, b2, is_switch=False)

###############################################################################
# 5) Define voltage categories
###############################################################################
def get_voltage_category(kv):
    """
    Adjust these thresholds to your requirements:
      - HV = 230 kV or higher
      - MV = 13.8 kV or higher
      - SN (spot network) = 0.48 kV or higher
      - LV = 0.208 kV or higher
      - Unknown otherwise
    """
    if kv >= 230.0/np.sqrt(3):
        return "HV"
    elif kv >= 13.8/np.sqrt(3):
        return "MV"
    elif kv >= 0.48/np.sqrt(3):
        return "SN"     # Spot Network
    elif kv >= 0.208/np.sqrt(3):
        return "LV"     # Low Voltage
    else:
        return "Unknown"

# Assign node attributes
for node in G.nodes():
    kv = bus_voltage_dict.get(node, 0.0)
    cat = get_voltage_category(kv)
    G.nodes[node]["base_kv"]  = kv
    G.nodes[node]["category"] = cat
    G.nodes[node]["is_load"]  = (node in load_buses)

# Identify bridging nodes (those connecting different voltage categories)
bridge_nodes = set()
for (u, v) in G.edges():
    if G.nodes[u]["category"] != G.nodes[v]["category"]:
        bridge_nodes.add(u)
        bridge_nodes.add(v)

for nd in G.nodes():
    G.nodes[nd]["bridge_node"] = (nd in bridge_nodes)

###############################################################################
# 6) Split into subgraphs by category
###############################################################################
categories = ["HV", "MV", "SN", "LV", "Unknown"]
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
# 7) Plot each category
###############################################################################
###############################################################################
# 0) Global Figure and Style Settings
###############################################################################
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10

###############################################################################
# 1) (Optional) Compile Circuit, Solve, and Build Main Graph 'G'
###############################################################################
# ...
# [Your code that builds the main graph G and assigns node attributes:
#  "category" in {"HV", "MV", "LV", ...}, "is_load", "bridge_node", etc.]

# Example categories for demonstration:
# for node in G.nodes():
#     # Suppose we already have G.nodes[node]["category"] = "HV", "MV", or "LV"
#     # and G.nodes[node]["is_load"], G.nodes[node]["bridge_node"]

subgraphs = {"HV": nx.Graph(), "MV": nx.Graph(), "LV": nx.Graph()}

for n in G.nodes():
    cat = G.nodes[n]["category"]
    if cat in subgraphs:
        subgraphs[cat].add_node(n)
        # Copy node attributes
        for attr, val in G.nodes[n].items():
            subgraphs[cat].nodes[n][attr] = val

# Add edges only if endpoints share the same category
for (u, v) in G.edges():
    c_u = G.nodes[u]["category"]
    c_v = G.nodes[v]["category"]
    if c_u == c_v and c_u in subgraphs:
        subgraphs[c_u].add_edge(u, v, **G[u][v])

# Extract them for convenience
sg_HV = subgraphs["HV"]
sg_MV = subgraphs["MV"]
sg_LV = subgraphs["LV"]

###############################################################################
# 3) Set Up a Single Figure with 3 Subplots
###############################################################################
fig, (ax_hv, ax_mv, ax_lv) = plt.subplots(1, 3, figsize=(18, 6))

###############################################################################
# 4) Define a Helper to Plot Each Subgraph on a Given Axes
###############################################################################
def plot_subgraph(sg, ax, title, layout="dot"):
    """Draw 'sg' on Axes 'ax' using the specified layout and styling."""
    # Choose layout
    if layout == "dot":
        pos = graphviz_layout(sg, prog="dot")
        node_size = 300
        edge_width = 1.0
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(sg, scale=3.0)
        node_size = 250
        edge_width = 1.5
    else:
        # fallback
        pos = nx.spring_layout(sg, seed=42)
        node_size = 200
        edge_width = 1.0

    # Prepare node colors and borders
    node_colors = []
    node_borders = []
    for n in sg.nodes():
        if sg.nodes[n]["is_load"]:
            node_colors.append("#FCA082")  # Light salmon for load
        else:
            node_colors.append("#F0F0F0")  # Light gray for non-load
        # Blue outline for bridging nodes
        node_borders.append("blue" if sg.nodes[n]["bridge_node"] else "black")

    # Draw nodes
    nx.draw_networkx_nodes(
        sg, pos, ax=ax,
        node_color=node_colors,
        edgecolors=node_borders,
        node_size=node_size,
        linewidths=1.0
    )

    # Prepare edge colors
    edge_colors = []
    for (u, v) in sg.edges():
        if sg[u][v]["is_switch"]:
            edge_colors.append("#3182BD")  # Switch line (blue)
        else:
            edge_colors.append("#636363")  # Normal line (dark gray)

    # Draw edges
    nx.draw_networkx_edges(
        sg, pos, ax=ax,
        edge_color=edge_colors,
        width=edge_width
    )

    # Draw labels (no bounding box)
    nx.draw_networkx_labels(
        sg, pos, ax=ax,
        font_size=7,
        font_family="serif",
        font_color="black"
    )

    ax.set_title(title, fontsize=12)
    ax.axis("off")

###############################################################################
# 5) Plot HV, MV, LV in Each Subplot
###############################################################################
plot_subgraph(sg_HV, ax_hv,  title="HV Subgraph", layout="dot")
plot_subgraph(sg_MV, ax_mv,  title="MV Subgraph", layout="dot")
plot_subgraph(sg_LV, ax_lv,  title="LV Subgraph", layout="kamada")

###############################################################################
# 6) Use ONE Shared Legend for the Entire Figure
###############################################################################
# Prepare your legend handles just once
load_patch = mpatches.Patch(facecolor="#FCA082", edgecolor="black", label="Load Bus")
non_load_patch = mpatches.Patch(facecolor="#F0F0F0", edgecolor="black", label="Non-Load Bus")
bridge_patch = mpatches.Patch(facecolor="none", edgecolor="blue", label="Transformer Node")
switch_line = mlines.Line2D([], [], color="#3182BD", linewidth=2, label="Switch Line")
normal_line = mlines.Line2D([], [], color="#636363", linewidth=2, label="Normal Line")
legend_handles = [load_patch, non_load_patch, bridge_patch, switch_line, normal_line]

# Place the legend at the bottom (or top) – adjust as needed
# fig.legend(
#     handles=legend_handles,
#     loc="lower center",       # Could use "upper center" if you prefer
#     bbox_to_anchor=(0.5, 0.0),# adjust to move up/down
#     ncol=5,
#     fontsize=8,
#     frameon=True
# )
fig.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),  # Adjust these coordinates as needed
    ncol=1,
    fontsize=10,
    frameon=True
)

# Optionally, add dashed dividing lines between subplots in figure coords
# (May need to adjust x-positions depending on your layout)
line1 = mlines.Line2D([0.33, 0.33], [0.02, 0.98], transform=fig.transFigure,
                      color='black', linestyle='--', lw=1)
line2 = mlines.Line2D([0.66, 0.66], [0.02, 0.98], transform=fig.transFigure,
                      color='black', linestyle='--', lw=1)
fig.add_artist(line1)
fig.add_artist(line2)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for legend
plt.savefig("HV_MV_LV_combined.png", bbox_inches="tight")
plt.show()