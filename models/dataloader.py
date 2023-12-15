import torch
import pickle
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import networkx as nx
from torch.utils.data import DataLoader, Dataset
from torchnlp.encoders.text import StaticTokenizerEncoder
import random
import yaml  
import copy
from collections import defaultdict


class TimeDataset(Dataset):

    def __init__(self, data_rcds, x_tms_prob, data_traj, train=True):
        if train:
            self.data_rcds = data_rcds[:-50]
            self.data_traj = data_traj[:-50]
            self.x_prob = x_tms_prob[:-50]
            self.data_len = self.data_rcds.shape[0]
        else:
            self.data_rcds = data_rcds[-50:]
            self.data_traj = data_traj[-50:]
            self.x_prob = x_tms_prob[-50:]
            self.data_len = self.data_rcds.shape[0]
        # self.data_rcds = data_rcds 
        # self.data_traj = data_traj 
        # self.x_prob = x_tms_prob 
        # self.data_len = self.data_rcds.shape[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.data_rcds[idx], self.x_prob[idx], self.data_traj[idx]


def fake_data_for_train(configs):
    data_file = '../dataset/'
    data = pickle.load(open(data_file + "data_sim_4k.pkl", "rb"))
    # traj_nodes = pickle.load(open(data_file + "traj_nodes_769.pkl", "rb"))
    G = pickle.load(open(data_file + "longhua_1.8k.pkl", "rb"))
    
    res_traj = data['traj_y'] 
    res_traj_od = []
    cam_x = defaultdict(list)
    ans = copy.deepcopy(res_traj)
    for ix, seq in data['traj_y'].items():
        for item in data['cam_x'][ix]:
            if item in seq:
                break
        o_ix = seq.index(item)
        for item in data['cam_x'][ix][::-1]:
            if item in seq:
                break
        d_ix = len(seq) - seq[::-1].index(item)
        res_traj_od.append(['<s>'] + seq[o_ix: d_ix] + ['<\s>'])
    cam_x['node_token'] = [list(map(int, item)) for item in data['tklet'].values()]
    cam_x['node_token_wotklet'] = [list(map(int, item)) for item in data['cam_x'].values()]
    cam_x['recd_token'] = [list(map(int, np.array(item) + 3)) for item in data['recd_token'].values()]
    cam_x['cam_tms'] = [torch.tensor(list(map(int, item))) for item in data['cam_tms'].values()]

    cam_x['cam_x_high'] = [list(map(int, item)) for item in data['cam_x_high'].values()]
    cam_x['recd_token_high'] = [list(map(int, np.array(item) + 3)) for item in data['recd_token_high'].values()]
    return res_traj_od, cam_x, G


import torch.nn as nn
class RecordsEmbeding(nn.Module):


    def __init__(self, configs):
        super(RecordsEmbeding, self).__init__()
        rcd2vec = pd.read_pickle('../dataset/records_4w.pkl')
        rcd2vec = pd.DataFrame.from_dict(rcd2vec, orient='columns').car_feature.values
        rcd2vec = np.array(rcd2vec.tolist())
        self.node_embedding = nn.Embedding(rcd2vec.shape[0] + 3, 512)
        # self.embedding_weights(np.stack(used_records.car_feature.values))
        self.embedding_weights(rcd2vec)

    def embedding_weights(self, node_emb):
        self.node_embedding.weight.data[-node_emb.shape[0]:].copy_(
            torch.from_numpy(node_emb))
        self.node_embedding.weight.data.requires_grad = False

    def forward(self, x):
        return self.node_embedding(x).detach()


def data_loader(batch_size, configs):
    res_traj, cam_x, G = fake_data_for_train(configs)
    # res_traj, cam_x, df_records = real_data_for_test(configs)
    # data_file = configs['data_path']
    # G = pickle.load(open(data_file + configs['map_file'], "rb"))
    nodeid2token = StaticTokenizerEncoder(sample=[sorted(list(G.nodes()))],
                                        tokenize=lambda s: list(map(str, s)))

    x_pads_low, _ = nodeid2token.batch_encode(cam_x['node_token']) 
    x_pads_last, _ = nodeid2token.batch_encode(cam_x['node_token_wotklet']) 
    x_pads_last = torch.cat([x_pads_last, torch.zeros([x_pads_last.size(0), x_pads_low.size(1) - x_pads_last.size(1)])], dim=1)

    x_pads_high, _ = nodeid2token.batch_encode(cam_x['cam_x_high']) 
    x_pads_high = torch.cat([x_pads_high, torch.zeros([x_pads_high.size(0), x_pads_low.size(1) - x_pads_high.size(1)])], dim=1)

    x_pads = torch.cat([x_pads_last[:, None], x_pads_high[:, None], x_pads_low[:, None]], dim=1).int()
    y_pads, _ = nodeid2token.batch_encode(res_traj) 
    x_prob_low = pad_sequence([torch.tensor(i) for i in cam_x['recd_token']], batch_first=True, padding_value=0).int()[:, None] 
    x_prob_high = pad_sequence([torch.tensor(i) for i in cam_x['recd_token']], batch_first=True, padding_value=0).int()[:, None] 
    x_recd_low = pad_sequence([torch.tensor(i) for i in cam_x['recd_token']], batch_first=True, padding_value=0).int()[:, None] 
    x_recd_high = pad_sequence([torch.tensor(i) for i in cam_x['recd_token_high']], batch_first=True, padding_value=0).int()[:, None] 
    x_recd_high = torch.cat([x_recd_high, torch.zeros([x_recd_high.size(0), 1, x_recd_low.size(2) - x_recd_high.size(2)])], dim=2).long()
    
    deno_gt = [torch.tensor([1 if ix in res_traj[i] else 0 for ix in item]) for i, item in enumerate(cam_x['node_token_wotklet'])]
    deno_gt = pad_sequence(deno_gt, batch_first=True, padding_value=-1).int()[:, None] 
    x_prob = torch.cat([x_prob_low, x_prob_high, x_recd_low, x_recd_high, deno_gt], dim=1)
    train_set = TimeDataset(x_pads, x_prob, y_pads)

    res_traj, cam_x, _ = fake_data_for_train(configs)
    x_pads_low, _ = nodeid2token.batch_encode(cam_x['node_token']) 
    x_pads_last, _ = nodeid2token.batch_encode(cam_x['node_token_wotklet']) 
    x_pads_last = torch.cat([x_pads_last, torch.zeros([x_pads_last.size(0), x_pads_low.size(1) - x_pads_last.size(1)])], dim=1)

    x_pads_high, _ = nodeid2token.batch_encode(cam_x['cam_x_high']) 
    x_pads_high = torch.cat([x_pads_high, torch.zeros([x_pads_high.size(0), x_pads_low.size(1) - x_pads_high.size(1)])], dim=1)

    x_pads = torch.cat([x_pads_last[:, None], x_pads_high[:, None], x_pads_low[:, None]], dim=1).int()
    y_pads, _ = nodeid2token.batch_encode(res_traj)
    x_prob_low = pad_sequence([torch.tensor(i) for i in cam_x['recd_token']], batch_first=True, padding_value=0).int()[:, None] 
    x_prob_high = pad_sequence([torch.tensor(i) for i in cam_x['recd_token']], batch_first=True, padding_value=0).int()[:, None] 
    x_recd_low = pad_sequence([torch.tensor(i) for i in cam_x['recd_token']], batch_first=True, padding_value=0).int()[:, None] 
    x_recd_high = pad_sequence([torch.tensor(i) for i in cam_x['recd_token_high']], batch_first=True, padding_value=0).int()[:, None] 
    x_recd_high = torch.cat([x_recd_high, torch.zeros([x_recd_high.size(0), 1, x_recd_low.size(2) - x_recd_high.size(2)])], dim=2).long()
 
    deno_gt = [torch.tensor([1 if ix in res_traj[i] else 0 for ix in item]) for i, item in enumerate(cam_x['node_token_wotklet'])]
    deno_gt = pad_sequence(deno_gt, batch_first=True, padding_value=-1).int()[:, None] 
    x_prob = torch.cat([x_prob_low, x_prob_high, x_recd_low, x_recd_high, deno_gt], dim=1)
    test_set = TimeDataset(x_pads, x_prob, y_pads, train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size * 10,
                             shuffle=False)
    recd_emb = RecordsEmbeding(configs=configs)
    # torch.save(recd_emb.state_dict(), 'visual_emb.pth')
    # recd_emb.load_state_dict(torch.load('visual_emb.pth'))
    return train_loader, test_loader, nodeid2token, recd_emb


if __name__ == "__main__":
    data_loader(32)