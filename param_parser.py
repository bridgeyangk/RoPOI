"""Parameter parser for RoPOI"""
import argparse
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parameter_parser():
    parser = argparse.ArgumentParser(description="Parameter parser for RoPOI")

    # Dataset paths
    parser.add_argument('--data_train', type=str, default='data/NYC/NYC_train.csv')
    parser.add_argument('--data_val', type=str, default='data/NYC/NYC_test.csv')
    parser.add_argument('--data_test', type=str, default='data/NYC/NYC_test.csv')
    parser.add_argument('--data_node_feats', type=str, default='data/NYC/graph_X.csv')
    parser.add_argument('--data_adj_mtx', type=str, default='data/NYC/graph_A.csv')
    parser.add_argument('--geo_node_feats', type=str, default='data/NYC/geo_graph_X.csv')
    parser.add_argument('--geo_adj_mtx', type=str, default='data/NYC/geo_graph_A.csv')

    # Data processing
    parser.add_argument('--time_feature', type=str, default='norm_in_day_time')
    parser.add_argument('--short_traj_thres', type=int, default=2)
    parser.add_argument('--num_time_slots', type=int, default=48,
                        help='Discretize continuous time into slots')

    # Model architecture
    parser.add_argument('--poi_embed_dim', type=int, default=128)
    parser.add_argument('--user_embed_dim', type=int, default=128)
    parser.add_argument('--geo_embed_dim', type=int, default=64)
    parser.add_argument('--time_embed_dim', type=int, default=64)
    parser.add_argument('--multimodal_embed_dim', type=int, default=32)
    parser.add_argument('--NLP_embedding_dim', type=int, default=768)
    parser.add_argument('--gcn_nhid', type=list, default=[32, 64])
    parser.add_argument('--gcn_dropout', type=float, default=0.7)
    parser.add_argument('--multimodal_dropout', type=float, default=0.5)
    parser.add_argument('--transformer_nhead', type=int, default=2)
    parser.add_argument('--transformer_nhid', type=int, default=1024)
    parser.add_argument('--transformer_nlayers', type=int, default=2)
    parser.add_argument('--transformer_dropout', type=float, default=0.5)

    # Training
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=device)

    # Loss weights
    parser.add_argument('--loss_weight_poi', type=float, default=1.0)
    parser.add_argument('--loss_weight_time', type=float, default=0.2)
    parser.add_argument('--loss_weight_geo', type=float, default=0.2)
    parser.add_argument('--loss_weight_disen', type=float, default=0.01)


    # Test mode
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='Only run test evaluation (no training)')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to checkpoint file for test_only or resume training')

    return parser.parse_args()
