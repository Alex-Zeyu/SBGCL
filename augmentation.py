import numpy as np
import torch
import dgl
import random
from collections import defaultdict
from torch_geometric.utils import negative_sampling

# two ways to augment, i.e., attribute perturbation and structure perturbation

def generate_mask(row, column, args):
    # 1 -- leave   0 -- drop
    arr_mask_ratio = np.random.uniform(0, 1, size=(row, column))
    arr_mask = np.ma.masked_array(arr_mask_ratio, mask=(arr_mask_ratio < args.mask_ratio)).filled(0)
    arr_mask = np.ma.masked_array(arr_mask, mask=(arr_mask >= args.mask_ratio)).filled(1)
    return arr_mask

def perturb_attr(feature, args):
    # generate noise g_attr (perturb node attribute)
    attr_noise = np.random.normal(loc=0, scale=0.1, size=(feature.shape[0], feature.shape[1]))
    attr_mask = generate_mask(args.mask_ratio, row=feature.shape[0], column=feature.shape[1])
    noise_feature = feature * attr_mask + (1 - attr_mask) * attr_noise
    return noise_feature.float()


def dict2array(edgelist: defaultdict):
    edges = []
    for node in edgelist:
        for neighbor in edgelist[node]:
            edges.append([node, neighbor])
    return np.array(edges).T


def perturb_stru_inter(edges, args):
    edgelist_a_b_pos, edgelist_a_b_neg = edges
    edgelist_a_b_pos = dict2array(edgelist_a_b_pos)
    edgelist_a_b_neg = dict2array(edgelist_a_b_neg)
    if args.augment == 'delete':
        mask_pos = generate_mask(row=1, column=edgelist_a_b_pos.shape[1], args=args).squeeze()
        mask_neg = generate_mask(row=1, column=edgelist_a_b_neg.shape[1], args=args).squeeze()
        return torch.from_numpy(edgelist_a_b_pos[:, mask_pos != 0]), torch.from_numpy(edgelist_a_b_neg[:, mask_neg != 0])
    if args.augment == 'flip':
        mask_pos = generate_mask(row=1, column=edgelist_a_b_pos.shape[1], args=args).squeeze()
        mask_neg = generate_mask(row=1, column=edgelist_a_b_neg.shape[1], args=args).squeeze()
        temp_pos = np.concatenate((edgelist_a_b_pos[:,mask_pos!=0], edgelist_a_b_neg[:,mask_neg==0]),axis=1)
        temp_neg = np.concatenate((edgelist_a_b_pos[:,mask_pos==0], edgelist_a_b_neg[:,mask_neg!=0]),axis=1)
        return torch.from_numpy(temp_pos), torch.from_numpy(temp_neg)
    if args.augment == 'add':
        edgelist_a_b_pos = torch.from_numpy(edgelist_a_b_pos)
        edgelist_a_b_neg = torch.from_numpy(edgelist_a_b_neg)

        temp = torch.cat([edgelist_a_b_pos, edgelist_a_b_neg], dim=1)

        pos_add_a_b = int(args.mask_ratio * edgelist_a_b_pos.shape[1])
        neg_add_a_b = int(args.mask_ratio * edgelist_a_b_neg.shape[1])

        edges_pos_add_a_b = negative_sampling(temp, num_neg_samples=pos_add_a_b)

        temp = torch.cat([edgelist_a_b_pos, edgelist_a_b_neg,temp], dim=1)
        edges_neg_add_a_b = negative_sampling(temp, num_neg_samples=neg_add_a_b)

        edgelist_a_b_pos = torch.cat([edgelist_a_b_pos, edges_pos_add_a_b], dim=1)
        edgelist_a_b_neg = torch.cat([edgelist_a_b_neg, edges_neg_add_a_b], dim=1)
        return edgelist_a_b_pos, edgelist_a_b_neg



def perturb_stru_intra(edges, args):
    edgelist_a_a_pos, edgelist_a_a_neg,\
    edgelist_b_b_pos, edgelist_b_b_neg = edges

    edgelist_a_a_pos = dict2array(edgelist_a_a_pos)
    edgelist_a_a_neg = dict2array(edgelist_a_a_neg)
    edgelist_b_b_pos = dict2array(edgelist_b_b_pos)
    edgelist_b_b_neg = dict2array(edgelist_b_b_neg)

    if args.augment == 'delete':
        mask_a_pos = generate_mask(row=1, column=edgelist_a_a_pos.shape[1], args= args).squeeze()
        mask_a_neg = generate_mask(row=1, column=edgelist_a_a_neg.shape[1], args= args).squeeze()
        mask_b_pos = generate_mask(row=1, column=edgelist_b_b_pos.shape[1], args= args).squeeze()
        mask_b_neg = generate_mask(row=1, column=edgelist_b_b_neg.shape[1], args= args).squeeze()

        temp_a_pos = edgelist_a_a_pos[:, mask_a_pos!=0]
        temp_a_neg = edgelist_a_a_neg[:, mask_a_neg!=0]
        temp_b_pos = edgelist_b_b_pos[:, mask_b_pos!=0]
        temp_b_neg = edgelist_b_b_neg[:, mask_b_neg!=0]
        return torch.from_numpy(temp_a_pos), torch.from_numpy(temp_a_neg), \
               torch.from_numpy(temp_b_pos), torch.from_numpy(temp_b_neg)

    if args.augment == 'flip':
        mask_a_pos = generate_mask(row=1, column=edgelist_a_a_pos.shape[1], args= args).squeeze()
        mask_a_neg = generate_mask(row=1, column=edgelist_a_a_neg.shape[1], args= args).squeeze()
        mask_b_pos = generate_mask(row=1, column=edgelist_b_b_pos.shape[1], args= args).squeeze()
        mask_b_neg = generate_mask(row=1, column=edgelist_b_b_neg.shape[1], args= args).squeeze()

        temp_a_pos = np.concatenate((edgelist_a_a_pos[:,mask_a_pos!=0], edgelist_a_a_neg[:,mask_a_neg==0]),axis=1)
        temp_a_neg = np.concatenate((edgelist_a_a_pos[:,mask_a_pos==0], edgelist_a_a_neg[:,mask_a_neg!=0]),axis=1)
        temp_b_pos = np.concatenate((edgelist_b_b_pos[:,mask_b_pos!=0], edgelist_b_b_neg[:,mask_b_neg==0]),axis=1)
        temp_b_neg = np.concatenate((edgelist_b_b_pos[:,mask_b_pos==0], edgelist_b_b_neg[:,mask_b_neg!=0]),axis=1)
        return torch.from_numpy(temp_a_pos), torch.from_numpy(temp_a_neg), \
               torch.from_numpy(temp_b_pos), torch.from_numpy(temp_b_neg)

    if args.augment == 'add':
        edgelist_a_a_pos = torch.from_numpy(edgelist_a_a_pos)
        edgelist_a_a_neg = torch.from_numpy(edgelist_a_a_neg)
        edgelist_b_b_pos = torch.from_numpy(edgelist_b_b_pos)
        edgelist_b_b_neg = torch.from_numpy(edgelist_b_b_neg)

        pos_add_a = int(args.mask_ratio * edgelist_a_a_pos.shape[1])
        neg_add_a = int(args.mask_ratio * edgelist_a_a_neg.shape[1])
        pos_add_b = int(args.mask_ratio * edgelist_b_b_pos.shape[1])
        neg_add_b = int(args.mask_ratio * edgelist_b_b_neg.shape[1])

        temp_a = torch.cat([edgelist_a_a_pos, edgelist_a_a_neg], dim=1)
        temp_b = torch.cat([edgelist_b_b_pos, edgelist_b_b_neg], dim=1)

        edges_pos_add_a = negative_sampling(temp_a, num_neg_samples=pos_add_a)
        temp_a = torch.cat([edges_pos_add_a, temp_a],dim=1)
        edges_neg_add_a = negative_sampling(temp_a, num_neg_samples=neg_add_a)


        edges_pos_add_b = negative_sampling(temp_b, num_neg_samples=pos_add_b)
        temp_b = torch.cat([edges_pos_add_b, temp_b], dim=1)
        edges_neg_add_b = negative_sampling(temp_b, num_neg_samples=neg_add_b)

        edgelist_a_a_pos = torch.cat([edgelist_a_a_pos, edges_pos_add_a], dim=1)
        edgelist_a_a_neg = torch.cat([edgelist_a_a_neg, edges_neg_add_a], dim=1)
        edgelist_b_b_pos = torch.cat([edgelist_b_b_pos, edges_pos_add_b], dim=1)
        edgelist_b_b_neg = torch.cat([edgelist_b_b_neg, edges_neg_add_b], dim=1)
        return edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg












