import itertools
import sys
import numpy as np
import jax.numpy as jnp
from flax import nnx


class ProteinMPNN(nnx.Module):
    num_letters = 21

    def __init__(
        self,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        augment_eps=0.0,
        dropout=0.0,
        device=None,
        atom_context_num=0,
        model_type="protein_mpnn",
        ligand_mpnn_use_side_chain_context=False,
        *,
        rngs: nnx.Rngs,
    ):

        self.model_type = model_type
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        if self.model_type == "protein_mpnn" or self.model_type == "soluble_mpnn":
            self.features = ProteinFeatures(
                node_features=node_features,
                edge_features=edge_features,
                top_k=k_neighbors,
                augment_eps=augment_eps,
                rngs=rngs,
            )


class PositionalEncodings(nnx.Module):
    def __init__(self, num_embeddings, max_relative_feature=32, *, rngs: nnx.Rngs):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nnx.Linear(
            2 * max_relative_feature + 1 + 1, num_embeddings, rngs=rngs
        )

    def forward(self, offset, mask):
        d = jnp.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = nnx.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot)
        return E


class ProteinFeatures(nnx.Module):
    def __init__(
        self,
        edge_features=128,
        node_features=128,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=48,
        augment_eps=0.0,
        *,
        rngs: nnx.Rngs,
    ):

        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings, rngs=rngs)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nnx.Linear(
            edge_in, edge_features, use_bias=False, rngs=rngs
        )
        self.norm_edges = nnx.LayerNorm(edge_features, rngs=rngs)

    def _dist(self, X: jnp.ndarray, mask: jnp.ndarray, eps: float = 1e-6):
        """Computes distances to residues and finds the closest K residues

        Args:
            X (jnp.ndarray): B x n_residues x 3
            mask (jnp.ndarray): _description_
            eps (float, optional): _description_. Defaults to 1e-6.

        Returns:
            _type_: _description_
        """

        mask_2D = jnp.expand_dims(mask, 1) * jnp.expand_dims(mask, 2)
        dX = jnp.expand_dims(X, 1) - jnp.expand_dims(X, 2)
        D = mask_2D * jnp.sqrt(jnp.sum(dX**2, 3) + eps)
        D_max, _ = jnp.max(D, -1, keepdims=True)
        D_adjust = D + (1.0 - mask_2D) * D_max

        k = np.minimum(self.top_k, X.shape[1])
        idx = jnp.argsort(D_adjust, axis=-1)[..., :k]
        D_neighbors = jnp.take_along_axis(D_adjust, idx, axis=-1)

        return D_neighbors, idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, input_features):
        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]

        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return E, E_idx


# Gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = jnp.broadcast_to(
        neighbor_idx[..., None], (*neighbor_idx.shape, edges.shape[-1])
    )
    return jnp.take_along_axis(edges, neighbors, axis=2)


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    B, N, K = neighbor_idx.shape
    neighbors_flat = jnp.broadcast_to(
        neighbor_idx.reshape(B, N * K)[..., None], (B, N * K, nodes.shape[-1])
    )
    return jnp.take_along_axis(nodes, neighbors_flat, axis=1).reshape(
        B, N, K, nodes.shape[-1]
    )


def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features [B,K,C]
    idx_flat = jnp.broadcast_to(
        neighbor_idx[..., None], (*neighbor_idx.shape, nodes.shape[-1])
    )
    return jnp.take_along_axis(nodes, idx_flat, axis=1)


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return jnp.concatenate([h_neighbors, h_nodes], axis=-1)
