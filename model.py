import math
import torch
from torch import nn
from torch.nn import Module
from torch_geometric.nn import GatedGraphConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class POIGraph(Module):
    def __init__(self, n_nodes, hidden_size, node_features=None):
        super(POIGraph, self).__init__()
        self.hidden_size = hidden_size
        self.n_nodes = n_nodes
        self.node_features = node_features

        if node_features is not None:
            # Use linear layer to map features to embedding space
            self.feature_proj = nn.Linear(node_features.shape[1], hidden_size)
        else:
            # Use learnable embedding
            self.embedding = nn.Embedding(self.n_nodes, self.hidden_size)

        self.ggnn = GatedGraphConv(self.hidden_size, num_layers=2)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, A):
        if self.node_features is not None:
            # Use node features
            hidden = self.feature_proj(self.node_features)
        else:
            # Use embedding lookup
            hidden = self.embedding(inputs)
        hidden = self.ggnn(hidden, A)
        return hidden

    def getembedding(self, inputs):
        if self.node_features is not None:
            return self.feature_proj(self.node_features)
        return self.embedding(inputs)
   
class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim,)
        self.linear_1 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def forward(self, user_idx, mean_poi_embeddings):
        embed = self.user_embedding(user_idx)
        embed = embed + mean_poi_embeddings
        embed = self.leaky_relu(self.linear_1(embed))
        return embed
    
class FuseEmbeddings(nn.Module):
    def __init__(self, embed_dim1, embed_dim2):
        super(FuseEmbeddings, self).__init__()
        embed_dim = embed_dim1 + embed_dim2
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, embed1, embed2):
        x = self.fuse_embed(torch.cat((embed1, embed2), -1)) 
        x = self.leaky_relu(x)
        return x

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class Time2Vec(nn.Module):
    def __init__(self, out_dim):
        super(Time2Vec, self).__init__()
        self.sin = SineActivation(1, out_dim)

    def forward(self, x):
        x = self.sin(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_regions, num_times, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first = True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, num_times)
        self.decoder_region = nn.Linear(embed_size, num_regions)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask, src_key_padding_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_region = self.decoder_region(x)
        return out_poi, out_time, out_region

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# GCN classes for multimodal embeddings
from torch.nn import Parameter
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()
        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)
        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)
        return x


class ModalGenModule(nn.Module):
    """
    Missing modality generation for image/review/meta POI features.
    Takes z_g (general embedding from DisenModule) as the shared representation,
    and uses cross-modal generators to generate missing modality features.
    """
    def __init__(self, modal_dim, embed_dim):
        super().__init__()
        # Project each raw modality into embed_dim
        self.img_proj  = nn.Linear(modal_dim, embed_dim)
        self.rev_proj  = nn.Linear(modal_dim, embed_dim)
        self.meta_proj = nn.Linear(modal_dim, embed_dim)

        # Cross-modal generators (from z_g to each modality's projected space)
        self.gen_img  = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Linear(embed_dim, embed_dim))
        self.gen_rev  = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Linear(embed_dim, embed_dim))
        self.gen_meta = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Linear(embed_dim, embed_dim))

        # Decoders: generate original raw feature from embed_dim
        self.dec_img  = nn.Linear(embed_dim, modal_dim)
        self.dec_rev  = nn.Linear(embed_dim, modal_dim)
        self.dec_meta = nn.Linear(embed_dim, modal_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, z_g, img_feat, rev_feat, meta_feat,
                missing_img, missing_rev, missing_meta):
        """
        Args:
            z_g: [N, embed_dim] general embedding from DisenModule
            *_feat: [N, modal_dim] raw modal features (mean-filled for missing)
            missing_*: BoolTensor [N] True where that modality is missing
        Returns:
            scalar loss
        """
        img_e  = self.img_proj(img_feat)
        rev_e  = self.rev_proj(rev_feat)
        meta_e = self.meta_proj(meta_feat)

        # Generation loss on non-missing items
        loss = torch.tensor(0.0, device=z_g.device)

        valid_img  = ~missing_img
        valid_rev  = ~missing_rev
        valid_meta = ~missing_meta

        if valid_img.any():
            img_gen = self.dec_img(self.gen_img(z_g[valid_img]))
            loss += F.mse_loss(img_gen, img_feat[valid_img].detach())
            # Generation consistency in projected space
            loss += F.mse_loss(self.gen_img(z_g[valid_img]), img_e[valid_img].detach())

        if valid_rev.any():
            rev_gen = self.dec_rev(self.gen_rev(z_g[valid_rev]))
            loss += F.mse_loss(rev_gen, rev_feat[valid_rev].detach())
            loss += F.mse_loss(self.gen_rev(z_g[valid_rev]), rev_e[valid_rev].detach())

        if valid_meta.any():
            meta_gen = self.dec_meta(self.gen_meta(z_g[valid_meta]))
            loss += F.mse_loss(meta_gen, meta_feat[valid_meta].detach())
            loss += F.mse_loss(self.gen_meta(z_g[valid_meta]), meta_e[valid_meta].detach())

        return loss


class DisenModule(nn.Module):
    def __init__(self, embed_dim, num_users, num_modalities=2, num_heads=2, temperature=0.07):
        super(DisenModule, self).__init__()
        self.temperature = temperature
        self.embed_dim = embed_dim

        self.general_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.general_proj = nn.Linear(embed_dim, embed_dim)


        self.specific_proj = nn.Linear(embed_dim, embed_dim)
        self.mu_general = nn.Linear(embed_dim, embed_dim)
        self.logvar_general = nn.Linear(embed_dim, embed_dim)
        self.mu_specific = nn.Linear(embed_dim, embed_dim)
        self.logvar_specific = nn.Linear(embed_dim, embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_dim, embed_dim)
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        stacked = torch.stack([x, x], dim=1)
        g_feat, _ = self.general_attn(stacked, stacked, stacked)
        g_feat = self.layer_norm(self.general_proj(g_feat.mean(dim=1)))
        mu_g, logvar_g = self.mu_general(g_feat), torch.clamp(self.logvar_general(g_feat), -10, 2)
        z_g = self.reparameterize(mu_g, logvar_g)

        s_feat = self.layer_norm(self.specific_proj(x) - g_feat)
        mu_s, logvar_s = self.mu_specific(s_feat), torch.clamp(self.logvar_specific(s_feat), -10, 2)
        z_s = self.reparameterize(mu_s, logvar_s)

        kl_g = -0.5 * (1 + logvar_g - mu_g.pow(2) - logvar_g.exp()).sum(dim=-1).mean()
        kl_s = -0.5 * (1 + logvar_s - mu_s.pow(2) - logvar_s.exp()).sum(dim=-1).mean()
        l_kl = kl_g + kl_s


        z_g_norm = F.normalize(z_g, dim=-1)
        z_s_norm = F.normalize(z_s, dim=-1)
        logits = torch.mm(z_g_norm, z_s_norm.t()) / self.temperature
        labels = torch.arange(x.size(0), device=x.device)
        l_infonce = F.cross_entropy(logits, labels)

        x_recon = self.decoder(torch.cat([z_g, z_s], dim=-1))
        l_recon = F.mse_loss(x_recon, x.detach())

        return 0.01*l_kl + l_infonce + l_recon
