import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear, Embedding
from torch_geometric.nn import MessagePassing, radius_graph
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from math import pi as PI

from utils.chem import BOND_TYPES
from ..common import MeanReadout, SumReadout, MultiLayerPerceptron


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AsymmetricSineCosineSmearing(Module):

    def __init__(self, num_basis=50):
        super().__init__()
        num_basis_k = num_basis // 2
        num_basis_l = num_basis - num_basis_k
        self.register_buffer('freq_k', torch.arange(1, num_basis_k + 1).float())
        self.register_buffer('freq_l', torch.arange(1, num_basis_l + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0) + self.freq_l.size(0)

    def forward(self, angle):
        # If we don't incorporate `cos`, the embedding of 0-deg and 180-deg will be the
        #  same, which is undesirable.
        s = torch.sin(angle.view(-1, 1) * self.freq_k.view(1, -1))  # (num_angles, num_basis_k)
        c = torch.cos(angle.view(-1, 1) * self.freq_l.view(1, -1))  # (num_angles, num_basis_l)
        return torch.cat([s, c], dim=-1)


class SymmetricCosineSmearing(Module):

    def __init__(self, num_basis=50):
        super().__init__()
        self.register_buffer('freq_k', torch.arange(1, num_basis + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0)

    def forward(self, angle):
        return torch.cos(angle.view(-1, 1) * self.freq_k.view(1, -1))  # (num_angles, num_basis)


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff, smooth):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.smooth = smooth

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_length, edge_attr):
        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)  # Modification: cutoff
        else:
            C = (edge_length <= self.cutoff).float()
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class CFConvAttention(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff, smooth):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.smooth = smooth

        self.att_lin = Linear(num_filters, num_filters)  # Transform node features
        self.attention = Linear(2 * num_filters, 1)  # Compute attention scores
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.att_lin.weight)
        torch.nn.init.xavier_uniform_(self.attention.weight)
        self.att_lin.bias.data.fill_(0)
        self.attention.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.lin1(x)
        x_transformed = F.relu(self.att_lin(x))

        # Compute attention coefficients
        row, col = edge_index
        alpha = self.attention(torch.cat([x_transformed[row], x_transformed[col]], dim=-1))
        alpha = F.leaky_relu(alpha)
        alpha = F.softmax(alpha, dim=1)

        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)  # Modification: cutoff
        else:
            C = (edge_length <= self.cutoff).float()
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.propagate(edge_index, x=x, alpha=alpha, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return alpha * x_j * W


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff, smooth):
        super(InteractionBlock, self).__init__()
        mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, mlp, cutoff, smooth)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class VAEEncoder(Module):
    def __init__(self, input_dim, latent_dim):
        super(VAEEncoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, input_dim // 2)
        self.fc_mu = torch.nn.Linear(input_dim // 2, latent_dim)
        self.fc_logvar = torch.nn.Linear(input_dim // 2, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class VAEDecoder(Module):
    def __init__(self, latent_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.fc1 = torch.nn.Linear(latent_dim, output_dim // 2)
        self.fc2 = torch.nn.Linear(output_dim // 2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SchNetEncoder(Module):

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, edge_channels=100, cutoff=10.0, smooth=False, have_attention=False, heads=1):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff

        self.embedding = Embedding(100, hidden_channels, max_norm=10.0)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff, smooth)
            self.interactions.append(block)
        self.have_attention = have_attention
        if self.have_attention:
            self.self_attention = torch.nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=heads)

    def forward(self, z, edge_index, edge_length, edge_attr, embed_node=True):
        if embed_node:
            assert z.dim() == 1 and z.dtype == torch.long
            h = self.embedding(z)
        else:
            h = z
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        h, _ = self.self_attention(h.unsqueeze(1), h.unsqueeze(1), h.unsqueeze(1))
        h = h.squeeze(1)

        return h

class SchNetVAE(Module):
    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, edge_channels=100, cutoff=10.0, smooth=False, latent_dim=50):
        super().__init__()

        self.hidden_channels = hidden_channels

        # Existing SchNetEncoder components
        self.schnet_encoder = SchNetEncoder(hidden_channels, num_filters,
                                            num_interactions, edge_channels, cutoff, smooth)

        self.latent_dim = latent_dim
        self.vae_encoder = VAEEncoder(hidden_channels, self.latent_dim)
        self.vae_decoder = VAEDecoder(self.latent_dim, hidden_channels)


    def forward(self, z, edge_index, edge_length, edge_attr, embed_node=True):
        h = self.schnet_encoder(z, edge_index, edge_length, edge_attr, embed_node)

        mu, logvar = self.vae_encoder(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        reconstructed_h = self.vae_decoder(z)

        return h, reconstructed_h, mu, logvar

    def vae_loss(self, reconstructed_h, h, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed_h, h)

        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_div
