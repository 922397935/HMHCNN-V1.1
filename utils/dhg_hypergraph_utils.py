import numpy as np
import networkx as nx
import statistics
from itertools import combinations, product
from collections import defaultdict
from scipy.cluster import hierarchy

from dhg import Hypergraph
from dhg import Graph
"""
用XGI计算线图，容易出现keyError，这里改成DHG包试试
XGI转换线图的代码确实有问题，无法接受
"""


def adjacent_values_dict(H):
    '''his next function computes de adjacent values of a hypergraph from
    its nodes and hyperedges. It returns two dictionaries:
    - degree_dict : it is a dictionary node : hyperdegree (a_{ii})
    - degree_i_j_dict: it is a dictionary node : hyperneighbors (a_{ij})

    Parameters
    ----------
    H : Hypergraph

    Returns
    -------
    hyperdegree_dict : dict
    aij_dict : dict
    '''

    # Get the hyperdegrees 节点：节点超度值
    hyperdeg_list = H.D_v._values().cpu().view(-1).numpy().tolist()
    node_list = H.v
    hyperdeg_dict = {node: degree for node, degree in zip(node_list, hyperdeg_list)}
    aij_dict = defaultdict(lambda: 0)  # dictionary with default value 0 for all possible entries

    for edge in H.e[0]:
        for i in edge:
            for j in edge:
                aij_dict[(i, j)] += 1

    return hyperdeg_dict, aij_dict


def derivatives_dict(hyperdeg_dict, aij_dict, verbose=False):
    '''Once we have the adjacency values of a hypergraph, we compute their
    derivative values. We set the infinity value as a 999.
    It returns a dictionary with edge (i,j) : dH/d{i,j}

    Parameters
    ----------
    hyperdeg_dict : dict
    aij_dict : dict
    verbose : bool, default False.

    Returns
    -------
    similar_dict : dict
    equivalent_nodes : list of tuples
    '''

    # Auxiliary function
    jaccard = lambda i, j: (hyperdeg_dict[i] + hyperdeg_dict[j] - 2 * aij_dict[(i, j)]) / aij_dict[(i, j)]

    # Initialize the returned variables
    similar_dict = {}
    equivalent_nodes = []

    # Iterate over each and every edge.
    for edge in aij_dict.keys():

        if aij_dict[edge] == 0:  # If equivalent -> infinite derivative

            equivalent_nodes.append(edge)
            if verbose:
                print(f'Nodes {edge[0]} and {edge[1]} are equivalent.')
            similar_dict[edge] = np.inf

        else:
            similar_dict[edge] = jaccard(*edge)  # Compute the derivative

    return similar_dict, equivalent_nodes


def derivative_graph(H):
    '''Given a dictionary of the dH/d{i,j} per edge,
    create the associated "derivative" graph and compute
    from it the derivative adjacency matrix. 

    Parameters
    ----------
    similar_dict : dict
    equivalent_nodes : dict
    threshold : None (default) or float or int

    Returns
    -------
    derivative_graph of hypergraph
    '''
    hyperdeg_dict, aij_dict = adjacent_values_dict(H)
    similar_dict, _ = derivatives_dict(hyperdeg_dict, aij_dict, verbose=False)


    # Create the adjacency graph
    G = nx.Graph()
    for (i, j), deriv in similar_dict.items():
        G.add_edge(i, j, weight=deriv)

    # Remove equivalent nodes from it
    for (i, j) in _:
        G.remove_node(j)

        # Sort nodes
    Gsort = nx.Graph()
    Gsort.add_nodes_from(sorted(G.nodes(data=True)))
    Gsort.add_edges_from(G.edges(data=True))

    return Gsort


from xgi.exception import XGIError

def to_line_graph(H, s=1, weights=None):
    """The s-line graph of the hypergraph.

    The s-line graph of the hypergraph `H` is the graph whose
    nodes correspond to each hyperedge in `H`, linked together
    if they share at least s vertices.

    Optional edge weights correspond to the size of the
    intersection between the hyperedges, optionally
    normalized by the size of the smaller hyperedge.

    Parameters
    ----------
    H : Hypergraph
        The hypergraph of interest
    s : int
        The intersection size to consider edges
        as connected, by default 1.
    weights : str or None
        Specify whether to return a weighted line graph. If None,
        returns an unweighted line graph. If 'absolute', includes
        edge weights corresponding to the size of intersection
        between hyperedges. If 'normalized', includes edge weights
        normalized by the size of the smaller hyperedge.

    Returns
    -------
    LG : networkx.Graph
         The line graph associated to the Hypergraph

    References
    ----------
    "Hypernetwork science via high-order hypergraph walks", by Sinan G. Aksoy, Cliff
    Joslyn, Carlos Ortiz Marrero, Brenda Praggastis & Emilie Purvine.
    https://doi.org/10.1140/epjds/s13688-020-00231-0

    """
    if weights not in [None, "absolute", "normalized"]:
        raise XGIError(
            f"{weights} not a valid weights option. Choices are "
            "None, 'absolute', and 'normalized'."
        )
    LG = nx.Graph()
    edge_dict = {index: edge for index, edge in zip(range(H.num_e), H.e[0])}
    edge_label_dict = {tuple(edge): index for index, edge in edge_dict.items()}
   
    LG.add_nodes_from(range(H.num_e))
    
    for edge1, edge2 in combinations(edge_label_dict.keys(), 2):
        edge1 = set(edge1)
        edge2 = set(edge2)
        # Check that the intersection size is larger than s
        intersection_size = len(edge1.intersection(edge2))
        if intersection_size >= s:
            if not weights:
                # Add unweighted edge
                LG.add_edge(
                    edge_label_dict[tuple(edge1)], edge_label_dict[tuple(edge2)]
                )
            else:
                # Compute the (normalized) weight
                weight = intersection_size
                if weights == "normalized":
                    weight /= min([len(edge1), len(edge2)])
                # Add edge with weight
                LG.add_edge(
                    edge_label_dict[tuple(edge1)],
                    edge_label_dict[tuple(edge2)],
                    weight=weight,
                )

    return LG


if __name__ == '__main__':
    import xgi
    H = Hypergraph(6, e_list=[[0, 1, 5], [1, 3], [2, 3, 4], [2 ,4]])
    # HG = xgi.Hypergraph([[0, 1, 5], [1, 3], [2, 3, 4], [2 ,4]])
    # dev_G = derivative_graph(H)
    # print(dev_G)
    LG = to_line_graph(H)
    print(LG)