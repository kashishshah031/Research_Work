import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

import data
from model import GraphVAE, GraphDecoderOnly
from data import GraphAdjSampler

CUDA = 2

LR_milestones = [500, 1000]

def build_model(args, max_num_nodes):
    out_dim = max_num_nodes * (max_num_nodes + 1) // 2
    if args.feature_type == 'id':
        input_dim = max_num_nodes
    elif args.feature_type == 'deg':
        input_dim = 1
    elif args.feature_type == 'struct':
        input_dim = 2
    model = GraphVAE(input_dim, 64, 256, max_num_nodes)
    return model

def train(args, dataloader, model):
    epoch = 1
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=args.lr)

    model.train()
    for epoch in range(5000):
        for batch_idx, data_batch in enumerate(dataloader):
            model.zero_grad()
            features = data_batch['features'].float()
            adj_input = data_batch['adj'].float()

            features = Variable(features).cuda()
            adj_input = Variable(adj_input).cuda()

            loss = model(features, adj_input)
            print('Epoch: ', epoch, ', Iter: ', batch_idx, ', Loss: ', loss)
            loss.backward()

            optimizer.step()
            scheduler.step()
            break

def masked_node_ce_loss(logits, target_onehot, node_mask):
    B, N, C = logits.shape
    target = target_onehot.argmax(dim=-1)  # [B,N]
    logits = logits.view(B * N, C)
    target = target.view(B * N)
    mask = node_mask.view(B * N)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.cross_entropy(logits[mask], target[mask], reduction='mean')

def masked_edge_ce_loss(edge_logits, adj_onehot, node_mask):
    # edge_logits: [B,N,N,Cb], adj_onehot: [B,N,N,Cb], node_mask: [B,N]
    B, N, _, Cb = edge_logits.shape
    valid = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)              # [B,N,N]
    triu = torch.triu(torch.ones(N, N, device=edge_logits.device), diagonal=1).bool().unsqueeze(0)
    mask = (valid & triu)                                                # [B,N,N]
    if mask.sum() == 0:
        return torch.tensor(0.0, device=edge_logits.device)
    # select masked pairs and compute CE over classes
    logits_sel = edge_logits[mask]                                       # [M,Cb]
    target_idx = adj_onehot.argmax(dim=-1)[mask]                         # [M]
    return F.cross_entropy(logits_sel, target_idx, reduction='mean')

def train_decoder(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    ds = data.ExternalLatentGraphDataset(args.pairs)

    num_val = max(1, int(len(ds) * args.val_ratio))
    num_train = len(ds) - num_val
    g = torch.Generator().manual_seed(args.seed)
    train_set, val_set = torch.utils.data.random_split(ds, [num_train, num_val], generator=g)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = GraphDecoderOnly(
        latent_dim=ds.latent_dim,
        max_num_nodes=ds.N_MAX,
        node_classes=ds.C_NODE,
        bond_classes=ds.C_BOND,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float('inf')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            H = batch['H'].to(device)
            X = batch['X'].to(device)               # [B,N,C_node]
            A = batch['A'].to(device)               # [B,N,N,C_bond]
            node_mask = batch['node_mask'].to(device)

            node_logits, edge_logits = model(H)
            node_loss = masked_node_ce_loss(node_logits, X, node_mask)
            edge_loss = masked_edge_ce_loss(edge_logits, A, node_mask)
            loss = args.w_node * node_loss + args.w_edge * edge_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        scheduler.step()

        # validation
        model.eval()
        with torch.no_grad():
            vals = []
            for batch in val_loader:
                H = batch['H'].to(device)
                X = batch['X'].to(device)
                A = batch['A'].to(device)
                node_mask = batch['node_mask'].to(device)
                node_logits, edge_logits = model(H)
                nl = masked_node_ce_loss(node_logits, X, node_mask)
                el = masked_edge_ce_loss(edge_logits, A, node_mask)
                vals.append((args.w_node * nl + args.w_edge * el).item())
            val_loss = sum(vals) / max(1, len(vals))
        print(f'Epoch {epoch:03d} | val_loss {val_loss:.4f}')
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model': model.state_dict(),
                'config': {
                    'latent_dim': ds.latent_dim,
                    'N_MAX': ds.N_MAX,
                    'C_NODE': ds.C_NODE,
                    'C_BOND': ds.C_BOND,
                    'hidden_dim': args.hidden_dim,
                    'n_layers': args.n_layers,
                    'dropout': args.dropout,
                }
            }, args.out)
            print(f'Saved best to {args.out}')

            
def arg_parse():
    parser = argparse.ArgumentParser(description='GraphVAE and Decoder-only training.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset',
            help='Input dataset.')

    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--max_num_nodes', dest='max_num_nodes', type=int,
            help='Predefined maximum number of nodes in train/test graphs. -1 if determined by \
                  training data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')

    # New: decoder-only path args
    parser.add_argument('--decoder_only', action='store_true',
            help='Use external MoleMCL latents and train decoder only.')
    parser.add_argument('--pairs', type=str, default='results/decoder_pairs.pt',
            help='Path to decoder_pairs.pt')
    parser.add_argument('--out', type=str, default='results/decoder_only.pt',
            help='Where to save the trained decoder')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--w_node', type=float, default=1.0)
    parser.add_argument('--w_edge', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')

    parser.set_defaults(dataset='grid',
                        feature_type='id',
                        lr=0.001,
                        batch_size=1,
                        num_workers=1,
                        max_num_nodes=-1)
    return parser.parse_args()

def main():
    prog_args = arg_parse()

    # New: early exit for decoder-only training
    if prog_args.decoder_only:
        train_decoder(prog_args)
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA)
    print('CUDA', CUDA)
    ### running log

    if prog_args.dataset == 'enzymes':
        graphs = data.Graph_load_batch(min_num_nodes=10, name='ENZYMES')
        num_graphs_raw = len(graphs)
    elif prog_args.dataset == 'grid':
        graphs = []
        for i in range(2,3):
            for j in range(2,3):
                graphs.append(nx.grid_2d_graph(i,j))
        num_graphs_raw = len(graphs)

    if prog_args.max_num_nodes == -1:
        max_num_nodes = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    else:
        max_num_nodes = prog_args.max_num_nodes
        # remove graphs with number of nodes greater than max_num_nodes
        graphs = [g for g in graphs if g.number_of_nodes() <= max_num_nodes]

    graphs_len = len(graphs)
    print('Number of graphs removed due to upper-limit of number of nodes: ',
            num_graphs_raw - graphs_len)
    graphs_test = graphs[int(0.8 * graphs_len):]
    #graphs_train = graphs[0:int(0.8*graphs_len)]
    graphs_train = graphs

    print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
    print('max number node: {}'.format(max_num_nodes))

    dataset = GraphAdjSampler(graphs_train, max_num_nodes, features=prog_args.feature_type)
    dataset_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=prog_args.batch_size,
            num_workers=prog_args.num_workers)
    model = build_model(prog_args, max_num_nodes).cuda()
    train(prog_args, dataset_loader, model)

if __name__ == '__main__':
    main()