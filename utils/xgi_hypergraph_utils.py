#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 基于XGI的导图
'''
Author: Guillermo Vera, Jinling Yang, Gonzalo Contreras-Aso
Title: Functions related to community partitions in hypergraphs with the derivative graph.
参考文章《Detecting communities in higher-order networks by using their derivative
graphs》
链接：https://github.com/LaComarca-Lab/HyperGraph-Communities
'''

import numpy as np
import networkx as nx
import statistics
from itertools import combinations, product
from collections import defaultdict
from scipy.cluster import hierarchy


def adjacent_values_dict(H):
    '''his next function computes de adjacent values of a hypergraph from
    its nodes and hyperedges. It returns two dictionaries:
    - degree_dict : it is a dictionary node : hyperdegree (a_{ii})
    - degree_i_j_dict: it is a dictionary node : hyperneighbors (a_{ij})

    Parameters
    ----------
    H : xgi.Hypergraph

    Returns
    -------
    hyperdegree_dict : dict
    aij_dict : dict
    '''

    # Get the hyperdegrees
    hyperdeg_dict = H.degree()

    aij_dict = defaultdict(lambda: 0)  # dictionary with default value 0 for all possible entries

    for edge in H.edges.members():
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

def derivative_community_matrix(similar_dict, equivalent_nodes, threshold=None):
    '''Given a dictionary of the dH/d{i,j} per edge,
    create the associated "derivative" graph and compute
    from it the derivative adjacency matrix. If a threshold is
    given, return the community matrix too.
    
    Parameters
    ----------
    similar_dict : dict
    equivalent_nodes : dict
    threshold : None (default) or float or int

    Returns
    -------
    derivative_matrix : np.array
    community_matrix : np.array
    '''

    # Check the variable threshold
    assert not threshold or isinstance(threshold, float) or isinstance(threshold, int)

    # Create the adjacency graph
    G = nx.Graph()
    for (i,j), deriv in similar_dict.items():
        G.add_edge(i,j, weight=deriv)

    # Remove equivalent nodes from it
    for (i,j) in equivalent_nodes:
        G.remove_node(j) 

    # Sort nodes 
    Gsort = nx.Graph()
    Gsort.add_nodes_from(sorted(G.nodes(data=True)))
    Gsort.add_edges_from(G.edges(data=True))

    # Compute the derivative adjacency matrix
    derivative_matrix = nx.to_numpy_array(Gsort)

    if not threshold:
        return derivative_matrix 
    
    # Compute the community adjacency matrix (filter values above threshold)
    community_matrix = np.where(derivative_matrix  < threshold, derivative_matrix, 0)

    return derivative_matrix, community_matrix



def means_of_a_matrix(matrix):
    '''A function to help us to calculate the harmonic mean, normal mean and
    standard deviation from the derivative values (not considering infinity and 0 values)

    Parameters
    ----------
    matrix : np.array

    Returns
    -------
    harmonic_mean : float
    normal_mean : float
    des_tipica : float
    '''
    n = len(matrix)
    Har = []
    M = []

    for i in range(n):
        for j in range(n-i):
            # The first case we want to exclude 0 values to be able to
            # compute the harmonic mean, otherwise would make error dividing by 0
            if i <= j + i + 1 and matrix[i][j] > 0 and matrix[i][j] != np.inf:
                Har.append(matrix[i][j])
                M.append(matrix[i][j])

    harmonic_mean = statistics.harmonic_mean(Har)
    normal_mean = statistics.mean(M)
    des_tipica = statistics.stdev(Har)

    return harmonic_mean, normal_mean, des_tipica


def derivative_list(H, factor=10.0):
    """Given an XGI Hypergraph H, compute its derivative list. The
    factor parameter multiplies the maximum similarity for nodes not related.

    Parameters
    ----------
    H : xgi.Hypergraph
    factor : float

    Returns
    -------
    derivatives : list
    """

    # Compute the necessary matrices and dictionaries
    hyperdeg_dict, aij_dict = adjacent_values_dict(H)
    similar_dict, _ = derivatives_dict(hyperdeg_dict, aij_dict, verbose=False)
    max_similarity = np.max(list(similar_dict.values()))
    
    # Compute the derivate list
    derivatives = []     # It will contain all the derivatives in a list
    for (i,j) in combinations(list(H.nodes),2):
        if (i,j) in similar_dict.keys():
            derivatives.append(similar_dict[(i,j)])
        elif (j,i) in similar_dict.keys():
            derivatives.append(similar_dict[(j,i)])
        else:
            derivatives.append(factor*max_similarity) 

    return derivatives

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

    edge_label_dict = {tuple(edge): index for index, edge in H._edge.items()}

    LG.add_nodes_from(H.edges)

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

