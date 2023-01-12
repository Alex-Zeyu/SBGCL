import sys

import networkx as nx
from networkx.algorithms.cycles import cycle_basis
import torch
import numpy as np
import os
from collections import defaultdict
#

def load_data(dataset_name):
    train_file_path = os.path.join('datasets', f'{dataset_name}_training.txt')
    val_file_path = os.path.join('datasets', f'{dataset_name}_val.txt')
    test_file_path = os.path.join('datasets', f'{dataset_name}_test.txt')

    train_edgelist = []
    with open(train_file_path) as f:
        for ind, line in enumerate(f):
            a, b, s = map(int, line.split('\t'))
            train_edgelist.append((a, b, s))

    val_edgelist = []
    with open(val_file_path) as f:
        for ind, line in enumerate(f):
            a, b, s = map(int, line.split('\t'))
            val_edgelist.append((a, b, s))

    test_edgelist = []
    with open(test_file_path) as f:
        for ind, line in enumerate(f):
            a, b, s = map(int, line.split('\t'))
            test_edgelist.append((a, b, s))

    return np.array(train_edgelist), np.array(val_edgelist), np.array(test_edgelist)

def load_new_data(dataset_name):
    train_file_path = os.path.join('new_datasets', f'{dataset_name}_training.txt')
    val_file_path = os.path.join('new_datasets', f'{dataset_name}_val.txt')
    test_file_path = os.path.join('new_datasets', f'{dataset_name}_test.txt')

    train_edgelist = []
    with open(train_file_path) as f:
        for ind, line in enumerate(f):
            a, b, s = map(int, line.split('\t'))
            train_edgelist.append((a, b, s))

    val_edgelist = []
    with open(val_file_path) as f:
        for ind, line in enumerate(f):
            a, b, s = map(int, line.split('\t'))
            val_edgelist.append((a, b, s))

    test_edgelist = []
    with open(test_file_path) as f:
        for ind, line in enumerate(f):
            a, b, s = map(int, line.split('\t'))
            test_edgelist.append((a, b, s))

    return np.array(train_edgelist), np.array(val_edgelist), np.array(test_edgelist)


def create_perspectives(edge_lists):
    # create edge indices for pespective 2
    edgelist_a_b_pos, edgelist_a_b_neg = defaultdict(list), defaultdict(list)
    edgelist_b_a_pos, edgelist_b_a_neg = defaultdict(list), defaultdict(list)
    edgelist_a_a_pos, edgelist_a_a_neg = defaultdict(list), defaultdict(list)
    edgelist_b_b_pos, edgelist_b_b_neg = defaultdict(list), defaultdict(list)

    for a, b, s in edge_lists:
        if s == 1:
            edgelist_a_b_pos[a].append(b)
            edgelist_b_a_pos[b].append(a)
        elif s == -1:
            edgelist_a_b_neg[a].append(b)
            edgelist_b_a_neg[b].append(a)
        else:
            print(a, b, s)
            raise Exception("s must be -1/1")

    edge_list_a_a = defaultdict(lambda: defaultdict(int))
    edge_list_b_b = defaultdict(lambda: defaultdict(int))

    # for a, b, s in edge_lists:
    #     for b2 in edgelist_a_b_pos[a]:
    #         edge_list_b_b[b][b2] += 1 * s * (len(edgelist_a_b_pos[a])+len(edgelist_a_b_neg[a]))
    #     for b2 in edgelist_a_b_neg[a]:
    #         edge_list_b_b[b][b2] -= 1 * s * (len(edgelist_a_b_pos[a])+len(edgelist_a_b_neg[a]))
    #     for a2 in edgelist_b_a_pos[b]:
    #         edge_list_a_a[a][a2] += 1 * s * (len(edgelist_b_a_pos[b]) + len(edgelist_b_a_neg[b]))
    #     for a2 in edgelist_b_a_neg[b]:
    #         edge_list_a_a[a][a2] -= 1 * s * (len(edgelist_b_a_pos[b]) + len(edgelist_b_a_neg[b]))

    # for a, b, s in edge_lists:
    #     for b2 in edgelist_a_b_pos[a]:
    #         edge_list_b_b[b][b2] += 1 * s * (1/(len(edgelist_a_b_pos[a])+len(edgelist_a_b_neg[a])))
    #     for b2 in edgelist_a_b_neg[a]:
    #         edge_list_b_b[b][b2] -= 1 * s * (1/(len(edgelist_a_b_pos[a])+len(edgelist_a_b_neg[a])))
    #     for a2 in edgelist_b_a_pos[b]:
    #         edge_list_a_a[a][a2] += 1 * s * (1/ (len(edgelist_b_a_pos[b]) + len(edgelist_b_a_neg[b])))
    #     for a2 in edgelist_b_a_neg[b]:
    #         edge_list_a_a[a][a2] -= 1 * s * (1/ (len(edgelist_b_a_pos[b]) + len(edgelist_b_a_neg[b])))

    for a, b, s in edge_lists:
        for b2 in edgelist_a_b_pos[a]:
            edge_list_b_b[b][b2] += 1 * s
        for b2 in edgelist_a_b_neg[a]:
            edge_list_b_b[b][b2] -= 1 * s
        for a2 in edgelist_b_a_pos[b]:
            edge_list_a_a[a][a2] += 1 * s
        for a2 in edgelist_b_a_neg[b]:
            edge_list_a_a[a][a2] -= 1 * s


    for a1 in edge_list_a_a:
        for a2 in edge_list_a_a[a1]:
            v = edge_list_a_a[a1][a2]
            if a1 == a2: continue
            if v > 0:
                edgelist_a_a_pos[a1].append(a2)
            elif v < 0:
                edgelist_a_a_neg[a1].append(a2)

    for b1 in edge_list_b_b:
        for b2 in edge_list_b_b[b1]:
            v = edge_list_b_b[b1][b2]
            if b1 == b2: continue
            if v > 0:
                edgelist_b_b_pos[b1].append(b2)
            elif v < 0:
                edgelist_b_b_neg[b1].append(b2)

    return edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,\
                    edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg


def calculate_balanced_index(edges):
    sign_dict = dict()
    G = nx.Graph()
    for src, dst, sign in edges.tolist():
        sign_dict[(src,dst)] = sign
        sign_dict[(dst,src)] = sign
        G.add_edge(src,dst)
    circles = cycle_basis(G)
    circles = [circle for circle in circles if len(circle)==3]
    circles_count = len(circles)
    balance_circles = []
    for tri in circles:
        if sign_dict[tri[0],tri[1]]*sign_dict[tri[1],tri[2]]*sign_dict[tri[2],tri[0]]>0:
            balance_circles.append(tri)
    return len(balance_circles) / circles_count

def unbalanced(edges):
    sign_dict = dict()
    G = nx.Graph()
    for src, dst, sign in edges.tolist():
        sign_dict[(src,dst)] = sign
        sign_dict[(dst,src)] = sign
        G.add_edge(src,dst)
    circles = cycle_basis(G)
    triangles = [circle for circle in circles if len(circle) == 3]
    butterflies = [circle for circle in circles if len(circle)==4]
    print(f'tri:{len(triangles)}, but: {len(butterflies)}')
    # circles_count = len(circles)
    unbalance_triangles = []
    unbalance_butterflies = []

    for tri in triangles:
        if sign_dict[tri[0],tri[1]]*sign_dict[tri[1],tri[2]]*sign_dict[tri[2],tri[0]]<0:
            unbalance_triangles.append(tri)

    for but in butterflies:
        if sign_dict[but[0],but[1]]*sign_dict[but[1],but[2]]*sign_dict[but[2],but[3]]*sign_dict[but[3], but[0]]<0:
            unbalance_butterflies.append(but)
    return len(unbalance_triangles), len(unbalance_butterflies),len(unbalance_triangles)/(len(triangles)+1), len(unbalance_butterflies)/len(butterflies)




def random_sign_pertubation(edges, ratio):
    '''
    Change the sign of the chosen edge
    :param edges: (pos_edge_index, neg_edge_index)
    :param ratio: edge pertubated edges
    :return: edges (modified_pos_edge_index, modified_neg_edge_index)
    '''
    pos_edge_index, neg_edge_index = edges

    pos_index_mask = torch.empty(pos_edge_index.shape[1]).bernoulli_(p=ratio).long().bool()
    neg_index_mask = torch.empty(neg_edge_index.shape[1]).bernoulli_(p=ratio).long().bool()

    modified_pos_edge_index = torch.cat([neg_edge_index[:,neg_index_mask],
                                         pos_edge_index[:,(pos_index_mask * -1 + 1).bool()]], dim=1)
    modified_neg_edge_index = torch.cat([pos_edge_index[:,pos_index_mask],
                                         neg_edge_index[:,(neg_index_mask * -1 + 1).bool()]], dim=1)

    return modified_pos_edge_index, modified_neg_edge_index


