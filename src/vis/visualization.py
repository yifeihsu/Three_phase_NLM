"""
Network visualization for the OpenDSS model.

- Compiles `data/Master.dss`
- Builds a graph with buses as nodes and lines/transformers as edges
- Colors buses by voltage category (HV/MV/SN/LV/Unknown)
- Highlights bridging nodes (between different categories)
- Saves a PNG to `results/vis_overview.png`
"""
from __future__ import annotations

import os
import math
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

try:
    from networkx.drawing.nx_agraph import graphviz_layout  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    graphviz_layout = None  # fallback handled below

import opendssdirect as dss

from config import DATA_DIR, RESULTS_DIR


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_voltage_category(kv_ln: float) -> str:
    """
    Categorize by LN base-kV using typical thresholds:
      - HV:  >= 230 kV LL  -> >= 230/sqrt(3) LN
      - MV:  >= 13.8 kV LL -> >= 13.8/sqrt(3) LN
      - SN:  >= 0.48 kV LL -> >= 0.48/sqrt(3) LN (spot network)
      - LV:  >= 0.208 kV LL-> >= 0.208/sqrt(3) LN
      - Unknown otherwise
    """
    if kv_ln is None:
        return "Unknown"
    hv = 230.0 / math.sqrt(3.0)
    mv = 13.8 / math.sqrt(3.0)
    sn = 0.48 / math.sqrt(3.0)
    lv = 0.208 / math.sqrt(3.0)
    if kv_ln >= hv:
        return "HV"
    if kv_ln >= mv:
        return "MV"
    if kv_ln >= sn:
        return "SN"
    if kv_ln >= lv:
        return "LV"
    return "Unknown"


def build_graph() -> nx.Graph:
    """Compile DSS, solve snapshot, and return a NetworkX graph with attributes."""
    master = os.fspath(DATA_DIR / "Master.dss")
    dss.Basic.ClearAll()
    dss.Command(f'compile "{master}"')
    dss.Command('set mode=snapshot')
    dss.Solution.Solve()

    buses = dss.Circuit.AllBusNames() or []

    # Voltage base per bus (LN kV)
    bus_kv_ln: Dict[str, float] = {}
    for b in buses:
        dss.Circuit.SetActiveBus(b)
        kv_ln = dss.Bus.kVBase() or 0.0
        # Some APIs may return None; guard with float
        try:
            bus_kv_ln[b] = float(kv_ln)
        except Exception:
            bus_kv_ln[b] = 0.0

    # Load buses
    load_buses = set()
    for name in dss.Loads.AllNames() or []:
        dss.Loads.Name(name)
        bus_names = dss.CktElement.BusNames() or []
        if bus_names:
            load_buses.add(bus_names[0].split('.')[0])

    G = nx.Graph()
    for b in buses:
        kv = bus_kv_ln.get(b, 0.0)
        G.add_node(
            b,
            base_kv_ln=kv,
            category=get_voltage_category(kv),
            is_load=(b in load_buses),
        )

    # Lines
    for ln in dss.Lines.AllNames() or []:
        dss.Lines.Name(ln)
        b1 = dss.Lines.Bus1().split('.')[0]
        b2 = dss.Lines.Bus2().split('.')[0]
        is_switch = bool(dss.Lines.IsSwitch())
        if b1 and b2:
            G.add_edge(b1, b2, is_switch=is_switch)

    # Transformers
    for xf in dss.Transformers.AllNames() or []:
        dss.Transformers.Name(xf)
        bus_names = dss.CktElement.BusNames() or []
        if len(bus_names) >= 2:
            b1 = bus_names[0].split('.')[0]
            b2 = bus_names[1].split('.')[0]
            if b1 and b2:
                G.add_edge(b1, b2, is_switch=False)

    # Mark bridging nodes (connecting different categories)
    bridge_nodes = set()
    for u, v in G.edges():
        if G.nodes[u].get("category") != G.nodes[v].get("category"):
            bridge_nodes.add(u)
            bridge_nodes.add(v)
    nx.set_node_attributes(G, {n: (n in bridge_nodes) for n in G.nodes()}, name="bridge_node")

    return G


def draw_overview(G: nx.Graph, save_path: str) -> None:
    """Render and save an overview figure of the network graph."""
    mpl.rcParams['figure.dpi'] = 200
    mpl.rcParams['savefig.dpi'] = 200
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 9

    # Choose layout
    if graphviz_layout is not None:
        try:
            pos = graphviz_layout(G, prog="sfdp")
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=2.0 / math.sqrt(max(1, len(G))))
    else:
        pos = nx.spring_layout(G, seed=42, k=2.0 / math.sqrt(max(1, len(G))))

    # Category colors
    cat_color = {
        "HV": "#4B0082",      # indigo
        "MV": "#1f77b4",      # blue
        "SN": "#ff7f0e",      # orange
        "LV": "#2ca02c",      # green
        "Unknown": "#7f7f7f",  # gray
    }

    node_colors = [cat_color.get(G.nodes[n].get("category"), "#7f7f7f") for n in G.nodes()]
    node_edges = ["#1f77b4" if G.nodes[n].get("bridge_node") else "black" for n in G.nodes()]
    node_sizes = [80 if G.nodes[n].get("is_load") else 40 for n in G.nodes()]

    # Edge colors: switches lighter
    edge_colors = ["#3182bd" if G.edges[e].get("is_switch") else "#636363" for e in G.edges()]
    edge_widths = [1.0 if G.edges[e].get("is_switch") else 1.8 for e in G.edges()]

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(
        G, pos, ax=ax,
        with_labels=False,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color=edge_colors,
        width=edge_widths,
        linewidths=0.7,
        edgecolors=node_edges,
    )

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_items = [
        Patch(facecolor=cat_color["HV"], edgecolor="black", label="HV"),
        Patch(facecolor=cat_color["MV"], edgecolor="black", label="MV"),
        Patch(facecolor=cat_color["SN"], edgecolor="black", label="SN"),
        Patch(facecolor=cat_color["LV"], edgecolor="black", label="LV"),
        Patch(facecolor=cat_color["Unknown"], edgecolor="black", label="Unknown"),
        Line2D([], [], color="#636363", lw=2, label="Line/Transformer"),
        Line2D([], [], color="#3182bd", lw=2, label="Switch"),
        Line2D([], [], color="#1f77b4", marker="o", linestyle="None", label="Bridging node"),
    ]
    ax.legend(handles=legend_items, loc="best", fontsize=8)
    ax.set_title("OpenDSS Network Overview")
    ax.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    G = build_graph()
    out = os.fspath(RESULTS_DIR / "vis_overview.png")
    draw_overview(G, out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

