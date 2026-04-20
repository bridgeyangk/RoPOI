import logging
import datetime
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict
import random
import sys
import geohash as geohash_lib
from utils import top_k_acc_last_timestep, ndcg_last_timestep
from model import POIGraph, TransformerModel, UserEmbeddings, Time2Vec, FuseEmbeddings, \
    ModalityDisentanglement, UserModalityPreference, MissingModalityGenerator, DisenLoss
from param_parser import parameter_parser

args = parameter_parser()

# Setup logging with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"RoPOI_{timestamp}.log"
checkpoint_path = f'RoPOI_best_{timestamp}.pth'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info(f"Training log will be saved to: {log_file}")
logging.info(f"Checkpoint will be saved to: {checkpoint_path}")

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch(args.seed)

logging.info("Loading dataset...")

# Load data
train_df = pd.read_csv(args.data_train)
val_df = pd.read_csv(args.data_val)
test_df = pd.read_csv(args.data_test)

# Load POI node features
nodes_df = pd.read_csv(args.data_node_feats, delimiter="\t", encoding="gbk")
poi_ids_list = nodes_df['node_name/poi_id'].tolist()
poi_id2idx_dict = OrderedDict(zip(poi_ids_list, range(len(poi_ids_list))))
num_pois = len(poi_ids_list)
logging.info(f"num_pois: {num_pois}")

geo_nodes_df = pd.read_csv(args.geo_node_feats, delimiter="\t", encoding="gbk")
if 'geohash' in geo_nodes_df.columns:
    geo_col_name = 'geohash'
elif 'node_name/poi_id' in geo_nodes_df.columns:
    geo_col_name = 'node_name/poi_id'
else:
    raise ValueError(f"Geo graph file must have 'geohash' or 'node_name/poi_id' column")

geo_ids_list = geo_nodes_df[geo_col_name].tolist()
geo_id2idx_dict = OrderedDict(zip(geo_ids_list, range(len(geo_ids_list))))
num_geos = len(geo_ids_list)
logging.info(f"num_geos (regions): {num_geos}")

nodes_df['geohash'] = nodes_df.apply(
    lambda row: geohash_lib.encode(row['latitude'], row['longitude'], precision=6), axis=1)
poi_idx2geo_idx_dict = {}
for _, row in nodes_df.iterrows():
    pidx = poi_id2idx_dict[row['node_name/poi_id']]
    gh = row['geohash']
    poi_idx2geo_idx_dict[pidx] = geo_id2idx_dict.get(gh, 0)

# User id to index
user_ids = sorted(list(set(train_df['user_id'].tolist())))
user_ids = [str(uid) for uid in user_ids]
user_id2idx_dict = OrderedDict(zip(user_ids, range(len(user_ids))))
num_users = len(user_ids)
logging.info(f"num_users: {num_users}")

# Discretize continuous time into time slots
def discretize_time(time_val, num_slots=48):
    """Convert continuous time [0,1] to discrete slot [0, num_slots-1]"""
    return int(time_val * num_slots) % num_slots

# Create time slot mapping
num_times = args.num_time_slots
logging.info(f"num_time_slots: {num_times}")

# Build user-POI statistics
logging.info("Building user-POI statistics...")
user_poi = np.zeros((num_users, num_pois))

for user_id in tqdm(user_ids, desc="User statistics"):
    user_df = train_df[train_df['user_id'] == int(user_id)]
    user_idx = user_id2idx_dict[user_id]
    for _, row in user_df.iterrows():
        if row['POI_id'] not in poi_id2idx_dict:
            continue
        poi_idx = poi_id2idx_dict[row['POI_id']]
        user_poi[user_idx][poi_idx] += 1

# Normalize user_poi
user_poi_sum = np.sum(user_poi, axis=1, keepdims=True)
user_poi_sum[user_poi_sum == 0] = 1  # Avoid division by zero
user_poi = user_poi / user_poi_sum
user_poi = torch.from_numpy(user_poi).to(device=args.device, dtype=torch.float)

logging.info("Building edge index from trajectories...")
# Build edge_index from trajectory sequences (POI graph)
edge_index = [[], []]
edge_index_geo = [[], []]

for traj_id in tqdm(train_df['trajectory_id'].unique(), desc="Edge index"):
    traj_df = train_df[train_df['trajectory_id'] == traj_id]
    poi_ids = traj_df['POI_id'].tolist()
    for i in range(len(poi_ids) - 1):
        if poi_ids[i] in poi_id2idx_dict and poi_ids[i+1] in poi_id2idx_dict:
            poi_idx_curr = poi_id2idx_dict[poi_ids[i]]
            poi_idx_next = poi_id2idx_dict[poi_ids[i+1]]
            edge_index[0].append(poi_idx_curr)
            edge_index[1].append(poi_idx_next)

            geo_idx_curr = poi_idx2geo_idx_dict[poi_idx_curr]
            geo_idx_next = poi_idx2geo_idx_dict[poi_idx_next]
            edge_index_geo[0].append(geo_idx_curr)
            edge_index_geo[1].append(geo_idx_next)

edge_index = torch.LongTensor(edge_index).to(device=args.device)
edge_index_geo = torch.LongTensor(edge_index_geo).to(device=args.device)
logging.info(f"POI edges: {edge_index.shape[1]}, Geo edges: {edge_index_geo.shape[1]}")

class TrajectoryDatasetTrain(Dataset):
    def __init__(self, df):
        self.traj_seqs, self.input_seqs, self.label_seqs = [], [], []

        for traj_id in tqdm(df['trajectory_id'].unique(), desc="Train dataset"):
            traj_df = df[df['trajectory_id'] == traj_id]
            if len(traj_df) > 50:
                traj_df = traj_df.iloc[-50:]

            poi_ids = traj_df['POI_id'].tolist()
            poi_idxs = [poi_id2idx_dict[p] for p in poi_ids if p in poi_id2idx_dict]
            time_vals = traj_df[args.time_feature].tolist()[:len(poi_idxs)]
            time_slots = [discretize_time(t, num_times) for t in time_vals]

            if len(poi_idxs) < args.short_traj_thres + 1:
                continue

            input_seq = [(poi_idxs[i], time_slots[i]) for i in range(len(poi_idxs) - 1)]
            label_seq = [(poi_idxs[i+1], time_slots[i+1]) for i in range(len(poi_idxs) - 1)]

            self.traj_seqs.append(traj_id)
            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)

    def __len__(self):
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

class TrajectoryDatasetVal(Dataset):
    def __init__(self, df):
        self.traj_seqs, self.input_seqs, self.label_seqs = [], [], []

        for traj_id in tqdm(df['trajectory_id'].unique(), desc="Val dataset"):
            user_id = str(traj_id).split('_')[0]
            if user_id not in user_id2idx_dict:
                continue

            traj_df = df[df['trajectory_id'] == traj_id]
            if len(traj_df) > 50:
                traj_df = traj_df.iloc[-50:]

            poi_ids = traj_df['POI_id'].tolist()
            poi_idxs = [poi_id2idx_dict[p] for p in poi_ids if p in poi_id2idx_dict]
            time_vals = traj_df[args.time_feature].tolist()[:len(poi_idxs)]
            time_slots = [discretize_time(t, num_times) for t in time_vals]

            if len(poi_idxs) < args.short_traj_thres + 1:
                continue

            input_seq = [(poi_idxs[i], time_slots[i]) for i in range(len(poi_idxs) - 1)]
            label_seq = [(poi_idxs[i+1], time_slots[i+1]) for i in range(len(poi_idxs) - 1)]

            self.traj_seqs.append(traj_id)
            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)

    def __len__(self):
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

logging.info("Creating dataloaders...")
train_dataset = TrajectoryDatasetTrain(train_df)
val_dataset = TrajectoryDatasetVal(val_df)
test_dataset = TrajectoryDatasetVal(test_df)

train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                          pin_memory=True, collate_fn=lambda x: x)
val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False,
                        pin_memory=True, collate_fn=lambda x: x)
test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False,
                         pin_memory=True, collate_fn=lambda x: x)

logging.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize models
logging.info("Initializing models...")
poi_embed_model = POIGraph(num_pois, args.poi_embed_dim)
user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)
time_embed_model = Time2Vec(args.time_embed_dim)
geo_embed_model = POIGraph(num_geos, args.geo_embed_dim)
embed_fuse_model = FuseEmbeddings(args.poi_embed_dim + args.user_embed_dim,
                                  args.time_embed_dim + args.geo_embed_dim)

dim = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.geo_embed_dim
seq_model = TransformerModel(num_pois, num_geos, num_times, dim,
                             args.transformer_nhead, args.transformer_nhid,
                             args.transformer_nlayers, dropout=args.transformer_dropout)

model_disentangle = ModalityDisentanglement(embed_dim=dim)
model_user_pref = UserModalityPreference(num_users=num_users, num_modalities=2, embed_dim=dim)
model_missing_gen = MissingModalityGenerator(embed_dim=dim)

poi_embed_model = poi_embed_model.to(device=args.device)
user_embed_model = user_embed_model.to(device=args.device)
time_embed_model = time_embed_model.to(device=args.device)
geo_embed_model = geo_embed_model.to(device=args.device)
embed_fuse_model = embed_fuse_model.to(device=args.device)
seq_model = seq_model.to(device=args.device)
model_disentangle = model_disentangle.to(device=args.device)
model_user_pref = model_user_pref.to(device=args.device)
model_missing_gen = model_missing_gen.to(device=args.device)

optimizer = optim.Adam(
    list(poi_embed_model.parameters()) + list(user_embed_model.parameters()) +
    list(time_embed_model.parameters()) + list(embed_fuse_model.parameters()) +
    list(seq_model.parameters()) + list(geo_embed_model.parameters()),
    lr=args.lr, weight_decay=args.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', factor=args.lr_scheduler_factor, patience=args.patience)

# Loss functions
criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)
criterion_geo = nn.CrossEntropyLoss(ignore_index=-1)
criterion_time = nn.CrossEntropyLoss(ignore_index=-1)
model_loss_fn = modelLoss(temperature=0.07)
criterion_time = nn.CrossEntropyLoss(ignore_index=-1)

# Prepare indices for embedding lookup
poi_idxs = torch.arange(num_pois).long().to(device=args.device)
user_idxs = torch.arange(num_users).long().to(device=args.device)
geo_idxs = torch.arange(num_geos).long().to(device=args.device)

# Helper functions
def input_traj_to_embeddings(sample, poi_embeddings, user_embeddings, geo_embeddings):
    """Convert trajectory sample to embeddings"""
    traj_id = sample[0]
    input_seq = [each[0] for each in sample[1]]
    input_seq_time = [each[1] for each in sample[1]]

    user_id = str(traj_id).split('_')[0]
    user_idx = user_id2idx_dict[user_id]
    user_embedding = user_embeddings[user_idx].squeeze().to(device=args.device)

    input_seq_embed = []
    for idx in range(len(input_seq)):
        poi_embedding = poi_embeddings[input_seq[idx]].squeeze().to(device=args.device)
        geo_idx = poi_idx2geo_idx_dict[input_seq[idx]]
        geo_embedding = geo_embeddings[geo_idx].squeeze()
        time_embedding = time_embed_model(
            torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device)
        ).squeeze()

        c1 = torch.cat((user_embedding, poi_embedding), dim=-1)
        c2 = torch.cat((time_embedding, geo_embedding), dim=-1)
        fused_embedding = embed_fuse_model(c1, c2)
        input_seq_embed.append(fused_embedding)

    return input_seq_embed

def top_k_acc_last_timestep(y_true_seq, y_pred_seq, k):
    """Calculate top-k accuracy for last timestep"""
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    top_k_rec = y_pred.argsort()[-k:][::-1]
    return 1 if y_true in top_k_rec else 0

def MRR_metric_last_timestep(y_true_seq, y_pred_seq):
    """Calculate MRR for last timestep"""
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[::-1]
    r_idx = np.where(rec_list == y_true)[0]
    return 1 / (r_idx[0] + 1) if len(r_idx) > 0 else 0

logging.info("Starting training...")


if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logging.error(f"Checkpoint file NOT FOUND: {args.checkpoint}")
        logging.error("Please check the filename. Available .pth files in current dir:")
        import glob
        for f in glob.glob('*.pth'):
            logging.error(f"  {f}")
        if args.test_only:
            sys.exit(1)
    else:
        logging.info(f"Loading pretrained checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        poi_embed_model.load_state_dict(checkpoint['poi_embed'])
        user_embed_model.load_state_dict(checkpoint['user_embed'])
        time_embed_model.load_state_dict(checkpoint['time_embed'])
        geo_embed_model.load_state_dict(checkpoint['geo_embed'])
        embed_fuse_model.load_state_dict(checkpoint['fuse_embed'])
        seq_model.load_state_dict(checkpoint['seq_model'])
        logging.info(f"Checkpoint loaded (epoch={checkpoint.get('epoch', '?')}, "
                     f"val_recall5={checkpoint.get('val_recall5', '?')})")

best_val_acc = 0
best_epoch = 0

if args.test_only:
    logging.info("Test-only mode, skipping training...")
else:
    pass 

for epoch in range(0 if args.test_only else args.epochs):
    poi_embed_model.train()
    user_embed_model.train()
    time_embed_model.train()
    geo_embed_model.train()
    embed_fuse_model.train()
    seq_model.train()

    train_losses = []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        # Get embeddings for this batch
        poi_embeddings = poi_embed_model(poi_idxs, edge_index)
        geo_embeddings = geo_embed_model(geo_idxs, edge_index_geo)
        mean_poi_embeddings = torch.matmul(user_poi, poi_embeddings)
        user_embeddings = user_embed_model(user_idxs, mean_poi_embeddings)

        batch_input_seqs = []
        batch_label_pois = []
        batch_label_times = []
        batch_label_geos = []
        batch_seq_lens = []

        for sample in batch:
            input_seq_embed = input_traj_to_embeddings(sample, poi_embeddings, user_embeddings, geo_embeddings)
            batch_input_seqs.append(torch.stack(input_seq_embed))

            label_pois = [x[0] for x in sample[2]]
            label_times = [x[1] for x in sample[2]]
            label_geos = [poi_idx2geo_idx_dict[x[0]] for x in sample[2]]

            batch_label_pois.append(torch.LongTensor(label_pois))
            batch_label_times.append(torch.LongTensor(label_times))
            batch_label_geos.append(torch.LongTensor(label_geos))
            batch_seq_lens.append(len(label_pois))

        if not batch_input_seqs:
            continue

        # Pad sequences
        x = pad_sequence(batch_input_seqs, batch_first=True, padding_value=0).to(args.device)
        y_poi = pad_sequence(batch_label_pois, batch_first=True, padding_value=-1).to(args.device)
        y_time = pad_sequence(batch_label_times, batch_first=True, padding_value=-1).to(args.device)
        y_geo = pad_sequence(batch_label_geos, batch_first=True, padding_value=-1).to(args.device)

        # Create masks
        src_mask = seq_model.generate_square_subsequent_mask(x.size(1)).to(args.device)
        src_key_padding_mask = (y_poi == -1)

        # Forward
        y_pred_poi, y_pred_time, y_pred_geo = seq_model(x, src_mask, src_key_padding_mask)

        # Loss
        loss = args.loss_weight_poi * criterion_poi(y_pred_poi.transpose(1, 2), y_poi) + \
               args.loss_weight_time * criterion_time(y_pred_time.transpose(1, 2), y_time) + \
               args.loss_weight_geo * criterion_geo(y_pred_geo.transpose(1, 2), y_geo)

        with torch.no_grad():
            _x_mean = x.mean(dim=1)
            _common, _specifics = model_disentangle([_x_mean, _x_mean])
            _gen = model_missing_gen(_common, _specifics[0])
            _ = model_loss_fn(_common, _specifics, _gen, _x_mean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)
    logging.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

    # Validation
    poi_embed_model.eval()
    user_embed_model.eval()
    time_embed_model.eval()
    geo_embed_model.eval()
    embed_fuse_model.eval()
    seq_model.eval()

    val_recall5, val_recall10, val_recall20, val_ndcg10, val_ndcg20 = [], [], [], [], []
    val_loss_list = []

    with torch.no_grad():
        for batch in val_loader:
            poi_embeddings = poi_embed_model(poi_idxs, edge_index)
            geo_embeddings = geo_embed_model(geo_idxs, edge_index_geo)
            mean_poi_embeddings = torch.matmul(user_poi, poi_embeddings)
            user_embeddings = user_embed_model(user_idxs, mean_poi_embeddings)

            batch_input_seqs, batch_label_pois, batch_seq_lens = [], [], []

            for sample in batch:
                input_seq_embed = input_traj_to_embeddings(sample, poi_embeddings, user_embeddings, geo_embeddings)
                batch_input_seqs.append(torch.stack(input_seq_embed))
                batch_label_pois.append(torch.LongTensor([x[0] for x in sample[2]]))
                batch_seq_lens.append(len(sample[2]))

            if not batch_input_seqs:
                continue

            x = pad_sequence(batch_input_seqs, batch_first=True, padding_value=0).to(args.device)
            y_poi = pad_sequence(batch_label_pois, batch_first=True, padding_value=-1).to(args.device)
            src_mask = seq_model.generate_square_subsequent_mask(x.size(1)).to(args.device)
            src_key_padding_mask = (y_poi == -1)

            y_pred_poi, _, _ = seq_model(x, src_mask, src_key_padding_mask)

            val_loss_list.append(criterion_poi(y_pred_poi.transpose(1, 2), y_poi).item())

            batch_pred = y_pred_poi.detach().cpu().numpy()
            batch_label = y_poi.detach().cpu().numpy()

            for pred, label, seq_len in zip(batch_pred, batch_label, batch_seq_lens):
                pred = pred[:seq_len]
                label = label[:seq_len]
                val_recall5.append(top_k_acc_last_timestep(label, pred, k=5))
                val_recall10.append(top_k_acc_last_timestep(label, pred, k=10))
                val_recall20.append(top_k_acc_last_timestep(label, pred, k=20))
                val_ndcg10.append(ndcg_last_timestep(label, pred, k=10))
                val_ndcg20.append(ndcg_last_timestep(label, pred, k=20))

    val_loss = np.mean(val_loss_list) if val_loss_list else float('inf')
    val_r5 = np.mean(val_recall5) if val_recall5 else 0.0
    val_r10 = np.mean(val_recall10) if val_recall10 else 0.0
    val_r20 = np.mean(val_recall20) if val_recall20 else 0.0
    val_n10 = np.mean(val_ndcg10) if val_ndcg10 else 0.0
    val_n20 = np.mean(val_ndcg20) if val_ndcg20 else 0.0

    logging.info(
        f"Epoch {epoch+1}/{args.epochs} | "
        f"val_loss:{val_loss:.4f} | "
        f"Recall@5:{val_r5:.4f} | "
        f"Recall@10:{val_r10:.4f} | "
        f"Recall@20:{val_r20:.4f} | "
        f"NDCG@10:{val_n10:.4f} | "
        f"NDCG@20:{val_n20:.4f}"
    )

    lr_scheduler.step(avg_train_loss)

    # Save best model
    if val_r5 > best_val_acc:
        best_val_acc = val_r5
        best_epoch = epoch
        torch.save({
            'poi_embed': poi_embed_model.state_dict(),
            'user_embed': user_embed_model.state_dict(),
            'time_embed': time_embed_model.state_dict(),
            'geo_embed': geo_embed_model.state_dict(),
            'fuse_embed': embed_fuse_model.state_dict(),
            'seq_model': seq_model.state_dict(),
            'epoch': epoch,
            'val_recall5': best_val_acc,
        }, checkpoint_path)
        logging.info(f"  Best model saved — Recall@5={best_val_acc:.4f}")

logging.info(f"\nTraining completed. Best epoch: {best_epoch+1}, Best Val Recall@5: {best_val_acc:.4f}")

logging.info("Loading best checkpoint for test evaluation...")
if os.path.exists(checkpoint_path):
    best_ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    poi_embed_model.load_state_dict(best_ckpt['poi_embed'])
    user_embed_model.load_state_dict(best_ckpt['user_embed'])
    time_embed_model.load_state_dict(best_ckpt['time_embed'])
    geo_embed_model.load_state_dict(best_ckpt['geo_embed'])
    embed_fuse_model.load_state_dict(best_ckpt['fuse_embed'])
    seq_model.load_state_dict(best_ckpt['seq_model'])
    logging.info(f"Loaded best model from epoch {best_ckpt.get('epoch', '?')+1}")
elif args.checkpoint and os.path.exists(args.checkpoint):
    logging.info("No new best model saved, using loaded checkpoint for test")
else:
    logging.info("WARNING: No best checkpoint found, testing with last epoch weights")

logging.info("Evaluating on test set...")

poi_embed_model.eval()
user_embed_model.eval()
time_embed_model.eval()
geo_embed_model.eval()
embed_fuse_model.eval()
seq_model.eval()

test_recall5, test_recall10, test_recall20, test_ndcg10, test_ndcg20 = [], [], [], [], []

with torch.no_grad():
    for batch in test_loader:
        poi_embeddings = poi_embed_model(poi_idxs, edge_index)
        geo_embeddings = geo_embed_model(geo_idxs, edge_index_geo)
        mean_poi_embeddings = torch.matmul(user_poi, poi_embeddings)
        user_embeddings = user_embed_model(user_idxs, mean_poi_embeddings)

        batch_input_seqs, batch_label_pois, batch_seq_lens = [], [], []

        for sample in batch:
            input_seq_embed = input_traj_to_embeddings(sample, poi_embeddings, user_embeddings, geo_embeddings)
            batch_input_seqs.append(torch.stack(input_seq_embed))
            batch_label_pois.append(torch.LongTensor([x[0] for x in sample[2]]))
            batch_seq_lens.append(len(sample[2]))

        if not batch_input_seqs:
            continue

        x = pad_sequence(batch_input_seqs, batch_first=True, padding_value=0).to(args.device)
        y_poi = pad_sequence(batch_label_pois, batch_first=True, padding_value=-1).to(args.device)
        src_mask = seq_model.generate_square_subsequent_mask(x.size(1)).to(args.device)
        src_key_padding_mask = (y_poi == -1)

        y_pred_poi, _, _ = seq_model(x, src_mask, src_key_padding_mask)

        batch_pred = y_pred_poi.detach().cpu().numpy()
        batch_label = y_poi.detach().cpu().numpy()

        for pred, label, seq_len in zip(batch_pred, batch_label, batch_seq_lens):
            pred = pred[:seq_len]
            label = label[:seq_len]
            test_recall5.append(top_k_acc_last_timestep(label, pred, k=5))
            test_recall10.append(top_k_acc_last_timestep(label, pred, k=10))
            test_recall20.append(top_k_acc_last_timestep(label, pred, k=20))
            test_ndcg10.append(ndcg_last_timestep(label, pred, k=10))
            test_ndcg20.append(ndcg_last_timestep(label, pred, k=20))

logging.info(
    f"Test Results (Best Model) | "
    f"Recall@5:{np.mean(test_recall5):.4f} | "
    f"Recall@10:{np.mean(test_recall10):.4f} | "
    f"Recall@20:{np.mean(test_recall20):.4f} | "
    f"NDCG@10:{np.mean(test_ndcg10):.4f} | "
    f"NDCG@20:{np.mean(test_ndcg20):.4f}"
)


