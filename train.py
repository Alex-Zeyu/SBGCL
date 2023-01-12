# import library
import torch
import numpy as np
import torch_geometric
from data_load import load_data, create_perspectives, random_sign_pertubation,load_new_data
from augmentation import perturb_stru_inter, perturb_stru_intra, dict2array
from sbgrl import SBGRL,test_and_val
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true', default=False,
                    help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default='review-1',
                    help='choose dataset')
parser.add_argument('--seed', type=int, default=2023,
                    help='Random seed.')
parser.add_argument('--mask_ratio', type=float, default=0.1,
                    help='random mask ratio')
parser.add_argument('--tau', type=float, default=0.05,
                    help='temperature parameter')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout parameter')
# parser.add_argument('--weight_decay', type=float, default=1e-5,
#                     help='Weight Decay')
parser.add_argument('--beta', type=float, default=5e-4,
                    help='control contribution of loss contrastive')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='control the contribution of inter and intra loss')
parser.add_argument('--augment', type=str, default='delete',
                    help='augment method')
parser.add_argument('--predictor', type=str, default='2-linear',
                    help='decoder method')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--dim_embs', type=int, default=16,
                    help='initial embedding size of node')
parser.add_argument('--epochs', type=int, default=300,
                    help='initial embedding size of node')


args = parser.parse_args()
print(args)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# torch.set_default_dtype(torch.float32)

torch_geometric.seed_everything(args.seed)

dataset = args.dataset

train_edgelist, val_edgelist, test_edgelist = load_data(dataset) # numpy.array [edges_n, 3]

edges = np.concatenate((train_edgelist, val_edgelist, test_edgelist), axis=0)
num_a = edges[:, 0].max() - edges[:, 0].min() + 1
num_b = edges[:, 1].max() - edges[:, 1].min() + 1

train_edgelist[:, 1] = train_edgelist[:, 1] + num_a
val_edgelist[:, 1] = val_edgelist[:, 1] + num_a
test_edgelist[:, 1] = test_edgelist[:, 1] + num_a



val_pos_mask = val_edgelist[:, 2] > 0
val_neg_mask = val_edgelist[:, 2] < 0

test_pos_mask = test_edgelist[:, 2] > 0
test_neg_mask = test_edgelist[:, 2] < 0

# create two perspectives a_b different set, a_a, b_b same set
edgelist_a_b_pos, edgelist_a_b_neg, \
edgelist_b_a_pos, edgelist_b_a_neg, \
edgelist_a_a_pos, edgelist_a_a_neg, \
edgelist_b_b_pos, edgelist_b_b_neg = create_perspectives(train_edgelist) # defaultdict(list)

train_pos_edges = torch.from_numpy(dict2array(edgelist_a_b_pos))
train_neg_edges = torch.from_numpy(dict2array(edgelist_a_b_neg))

# add random noise
# train_pos_edges, train_neg_edges =  random_sign_pertubation((train_pos_edges, train_neg_edges), ratio=0.1)

# train and test edges
val_pos_edges = torch.from_numpy(val_edgelist[val_pos_mask, 0:2].T) # [2, edges_n]
val_neg_edges = torch.from_numpy(val_edgelist[val_neg_mask, 0:2].T)
test_pos_edges = torch.from_numpy(test_edgelist[test_pos_mask, 0:2].T)
test_neg_edges = torch.from_numpy(test_edgelist[test_neg_mask, 0:2].T)

# augments
# Graph 1
edgelist_a_b_pos_1, edgelist_a_b_neg_1 = \
    perturb_stru_inter((edgelist_a_b_pos, edgelist_a_b_neg), args)
# pos_index_a_b_1 = to_undirected(edgelist_a_b_pos_1).to(device)
# neg_index_a_b_1 = to_undirected(edgelist_a_b_neg_1).to(device)

pos_index_a_b_1 = edgelist_a_b_pos_1.to(device)
neg_index_a_b_1 = edgelist_a_b_neg_1.to(device)

# Graph 2
edgelist_a_b_pos_2, edgelist_a_b_neg_2 = \
    perturb_stru_inter((edgelist_a_b_pos, edgelist_a_b_neg), args)
# pos_index_a_b_2 = to_undirected(edgelist_a_b_pos_2).to(device)
# neg_index_a_b_2 = to_undirected(edgelist_a_b_neg_2).to(device)

pos_index_a_b_2 = edgelist_a_b_pos_2.to(device)
neg_index_a_b_2 = edgelist_a_b_neg_2.to(device)

# Graph 3
edgelist_a_a_pos_1, edgelist_a_a_neg_1, \
edgelist_b_b_pos_1, edgelist_b_b_neg_1 = \
    perturb_stru_intra((edgelist_a_a_pos, edgelist_a_a_neg,
                        edgelist_b_b_pos, edgelist_b_b_neg), args)
# pos_index_a_1 = to_undirected(edgelist_a_a_pos_1).to(device)
# neg_index_a_1 = to_undirected(edgelist_a_a_neg_1).to(device)
# pos_index_b_1 = to_undirected(edgelist_b_b_pos_1).to(device)
# neg_index_b_1 = to_undirected(edgelist_b_b_neg_1).to(device)

pos_index_a_1 = edgelist_a_a_pos_1.to(device)
neg_index_a_1 = edgelist_a_a_neg_1.to(device)
pos_index_b_1 = edgelist_b_b_pos_1.to(device)
neg_index_b_1 = edgelist_b_b_neg_1.to(device)

# Graph 4
edgelist_a_a_pos_2, edgelist_a_a_neg_2, \
edgelist_b_b_pos_2, edgelist_b_b_neg_2 = \
    perturb_stru_intra((edgelist_a_a_pos, edgelist_a_a_neg,
                        edgelist_b_b_pos, edgelist_b_b_neg), args)
# pos_index_a_2 = to_undirected(edgelist_a_a_pos_2).to(device)
# neg_index_a_2 = to_undirected(edgelist_a_a_neg_2).to(device)
# pos_index_b_2 = to_undirected(edgelist_b_b_pos_2).to(device)
# neg_index_b_2 = to_undirected(edgelist_b_b_neg_2).to(device)

pos_index_a_2 = edgelist_a_a_pos_2.to(device)
neg_index_a_2 = edgelist_a_a_neg_2.to(device)
pos_index_b_2 = edgelist_b_b_pos_2.to(device)
neg_index_b_2 = edgelist_b_b_neg_2.to(device)

x = torch.rand(size=(num_a+num_b, args.dim_embs)).to(device)

model = SBGRL(args, num_a, num_b).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
cnt = 0

uids_train = torch.cat([train_pos_edges[0], train_neg_edges[0]]).long().to(device)
vids_train = torch.cat([train_pos_edges[1], train_neg_edges[1]]).long().to(device)
y_label_train = torch.cat([torch.ones(train_pos_edges.shape[1]), torch.zeros(train_neg_edges.shape[1])]).to(device)

uids_val = torch.cat([val_pos_edges[0], val_neg_edges[0]]).long().to(device)
vids_val = torch.cat([val_pos_edges[1], val_neg_edges[1]]).long().to(device)
y_label_val = torch.cat([torch.ones(val_pos_edges.shape[1]), torch.zeros(val_neg_edges.shape[1])]).to(device)

uids_test = torch.cat([test_pos_edges[0], test_neg_edges[0]]).long().to(device)
vids_test = torch.cat([test_pos_edges[1], test_neg_edges[1]]).long().to(device)
y_label_test = torch.cat([torch.ones(test_pos_edges.shape[1]), torch.zeros(test_neg_edges.shape[1])]).to(device)




res_best = {'val_auc': 0,'val_f1':0}

for epoch in range(args.epochs):

    edges = (pos_index_a_b_1, neg_index_a_b_1, pos_index_a_b_2, neg_index_a_b_2,
             pos_index_a_1, neg_index_a_1, pos_index_b_1, neg_index_b_1,
             pos_index_a_2, neg_index_a_2, pos_index_b_2, neg_index_b_2)

    x_pos_index_a_b_1, x_neg_index_a_b_1, \
    x_pos_index_a_b_2, x_neg_index_a_b_2, \
    x_pos_index_a_b_3, x_neg_index_a_b_3, \
    x_pos_index_a_b_4, x_neg_index_a_b_4= model(edges, x)

    x_after = x_pos_index_a_b_1, x_neg_index_a_b_1, \
               x_pos_index_a_b_2, x_neg_index_a_b_2, \
               x_pos_index_a_b_3, x_neg_index_a_b_3, \
               x_pos_index_a_b_4, x_neg_index_a_b_4

    loss_contrastive = model.computer_contrastive_loss(x_after)

    y_score = model.predict_combine(model.embs, uids_train, vids_train)

    loss_label = model.compute_label_loss(y_score, y_label_train)
    loss = args.beta * loss_contrastive + loss_label
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    res_cur = dict()
    cnt += 1
    model.eval()
    y_score_train = model.predict_combine(model.embs, uids_train, vids_train)
    res = test_and_val(y_score_train, y_label_train, mode='train', epoch=epoch )
    res_cur.update(res)
    y_score_val = model.predict_combine(model.embs, uids_val, vids_val)
    res = test_and_val(y_score_val, y_label_val, mode='val', epoch=epoch)
    res_cur.update(res)
    y_score_test = model.predict_combine(model.embs, uids_test, vids_test)
    res = test_and_val(y_score_test, y_label_test, mode='test', epoch=epoch)
    res_cur.update(res)
    print(res_cur)
    if res_cur['val_auc'] + res_cur['val_f1'] > res_best['val_auc']+res_best['val_f1']:
        res_best = res_cur
        print(res_best)


print('Done! Best Results:')
print_list = ['test_auc', 'test_f1', 'test_macro_f1', 'test_micro_f1']
for i in print_list:
    print(i, res_best[i], end=' ')




