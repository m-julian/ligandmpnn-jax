from .proteinmpnn_model import (
    DecLayer,
    EncLayer,
    PositionalEncodings,
    ProteinFeatures,
    PositionWiseFeedForward,
    gather_edges,
    gather_nodes,
    cat_neighbors_nodes,
)
import numpy as np
import jax.numpy as jnp
from flax import nnx
import jax
import itertools
from .constants import SIDECHAIN_ATOM_TYPES, ELEMENT_LOOKUP_TABLE


class LingadMPNN(nnx.Module):
    num_letters = 21

    def __init__(
        self,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=21,
        k_neighbors=48,
        augment_eps=0.0,
        dropout=0.0,
        atom_context_num=0,
        ligand_mpnn_use_side_chain_context=False,
        *,
        rngs: nnx.Rngs,
    ):

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeaturesLigand(
            node_features,
            edge_features,
            top_k=k_neighbors,
            augment_eps=augment_eps,
            atom_context_num=atom_context_num,
            use_side_chains=ligand_mpnn_use_side_chain_context,
            rngs=rngs,
        )
        self.W_v = nnx.Linear(node_features, hidden_dim, use_bias=True, rngs=rngs)
        self.W_c = nnx.Linear(hidden_dim, hidden_dim, use_bias=True, rngs=rngs)

        self.W_nodes_y = nnx.Linear(hidden_dim, hidden_dim, use_bias=True, rngs=rngs)
        self.W_edges_y = nnx.Linear(hidden_dim, hidden_dim, use_bias=True, rngs=rngs)

        self.V_C = nnx.Linear(hidden_dim, hidden_dim, use_bias=False, rngs=rngs)
        self.V_C_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)

        self.context_encoder_layers = nnx.List(
            [
                DecLayer(hidden_dim, hidden_dim * 2, dropout=dropout, rngs=rngs)
                for _ in range(2)
            ]
        )

        self.y_context_encoder_layers = nnx.List(
            [DecLayerJ(hidden_dim, hidden_dim, dropout=dropout) for _ in range(2)]
        )


class DecLayerJ(nnx.Module):
    def __init__(
        self,
        num_hidden,
        num_in,
        dropout=0.1,
        scale=30,
        *,
        rngs: nnx.Rngs,
    ):
        super(DecLayerJ, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout, rngs=rngs)
        self.norm1 = nnx.LayerNorm(num_hidden, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_hidden, rngs=rngs)

        self.W1 = nnx.Linear(num_hidden + num_in, num_hidden, use_bias=True, rngs=rngs)
        self.W2 = nnx.Linear(num_hidden, num_hidden, use_bias=True, rngs=rngs)
        self.W3 = nnx.Linear(num_hidden, num_hidden, use_bias=True, rngs=rngs)
        self.act = nnx.gelu
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4, rngs=rngs)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(
            -1, -1, -1, h_E.size(-2), -1
        )  # the only difference
        h_EV = jnp.concat([h_V_expand, h_E], axis=-1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = jnp.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


# TODO: rewrite because code is duplicated across ProteinFeatures
# and ProteinFeaturesLigand
class ProteinFeaturesLigand(nnx.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
        atom_context_num=16,
        use_side_chains=False,
        *,
        rngs: nnx.Rngs,
    ):
        """Extract protein features"""

        self.use_side_chains = use_side_chains

        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.side_chain_atom_types = jnp.array(SIDECHAIN_ATOM_TYPES)
        self.periodic_table_features = jnp.array(ELEMENT_LOOKUP_TABLE)

        self.embeddings = PositionalEncodings(num_positional_embeddings, rngs=rngs)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nnx.Linear(
            edge_in, edge_features, use_bias=False, rngs=rngs
        )
        self.norm_edges = nnx.LayerNorm(edge_features, rngs=rngs)

        self.node_project_down = nnx.Linear(
            5 * num_rbf + 64 + 4, node_features, use_bias=True, rngs=rngs
        )
        self.norm_nodes = nnx.LayerNorm(node_features, rngs=rngs)

        self.type_linear = nnx.Linear(147, 64, rngs=rngs)

        self.y_nodes = nnx.Linear(147, node_features, use_bias=False, rngs=rngs)
        self.y_edges = nnx.Linear(num_rbf, node_features, use_bias=False, rngs=rngs)

        self.norm_y_edges = nnx.LayerNorm(node_features, rngs=rngs)
        self.norm_y_nodes = nnx.LayerNorm(node_features, rngs=rngs)

        self.atom_context_num = atom_context_num

    def _make_angle_features(self, A, B, C, Y):
        v1 = A - B
        v2 = C - B
        v1_norm = jnp.linalg.norm(v1, axis=-1, keepdims=True)
        e1 = v1 / jnp.maximum(v1_norm, 1e-12)

        e1_v2_dot = jnp.einsum("bli, bli -> bl", e1, v2)[..., None]
        u2 = v2 - e1 * e1_v2_dot

        u2_norm = jnp.linalg.norm(u2, axis=-1, keepdims=True)
        e2 = u2 / jnp.maximum(u2_norm, 1e-12)

        e3 = jnp.cross(e1, e2, axis=-1)
        R_residue = jnp.concat(
            (e1[:, :, :, None], e2[:, :, :, None], e3[:, :, :, None]), axis=-1
        )

        local_vectors = jnp.einsum(
            "blqp, blyq -> blyp", R_residue, Y - B[:, :, None, :]
        )

        rxy = jnp.sqrt(local_vectors[..., 0] ** 2 + local_vectors[..., 1] ** 2 + 1e-8)
        f1 = local_vectors[..., 0] / rxy
        f2 = local_vectors[..., 1] / rxy
        rxyz = jnp.linalg.norm(local_vectors, dim=-1) + 1e-8
        f3 = rxy / rxyz
        f4 = local_vectors[..., 2] / rxyz

        f = jnp.concat(
            [f1[..., None], f2[..., None], f3[..., None], f4[..., None]], axis=-1
        )
        return f

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = jnp.expand_dims(mask, 1) * jnp.expand_dims(mask, 2)
        dX = jnp.expand_dims.unsqueeze(X, 1) - jnp.expand_dims.unsqueeze(X, 2)
        D = mask_2D * jnp.sqrt(jnp.sum(dX**2, 3) + eps)
        D_max, _ = jnp.max(D, -1, keepdims=True)
        D_adjust = D + (1.0 - mask_2D) * D_max

        D_neighbors, E_idx = jax.lax.top_k(
            -D_adjust, int(jnp.minimum(self.top_k, X.shape[1]))
        )
        D_neighbors = -D_neighbors

        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.reshape([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = jnp.expand_dims(D, -1)
        RBF = jnp.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = jnp.sqrt(
            jnp.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, input_features, key: jax.Array):
        Y = input_features["Y"]
        Y_m = input_features["Y_m"]
        Y_t = input_features["Y_t"]
        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]

        if self.augment_eps > 0:
            key1, key2 = jax.random.split(key)
            X = X + self.augment_eps * jax.random.normal(key1, X.shape)
            Y = Y + self.augment_eps * jax.random.normal(key2, Y.shape)

        B, L, _, _ = X.shape

        # TODO: this has already been implemented in the proteinmpnn features
        # find a nice way to change these
        N = X[:, :, 0, :]
        Ca = X[:, :, 1, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        b = Ca - N
        c = C - Ca
        a = jnp.cross(b, c, axis=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca  # shift from CA

        D_neighbors, E_idx = self._dist(Ca, mask)

        # [B, L, K, 25*num_rbf]
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
        RBF_all = jnp.concat(tuple(RBF_all), axis=-1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        # [B, L, K]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]

        # find self vs non-self interaction
        d_chains = (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        # [B, L, K, num_positional_embeddings]
        E_positional = self.embeddings(offset, E_chains)
        # [B, L, K, num_positional_embeddings + 25*num_rbf]
        # [B, L, K, edge_in]
        E = jnp.concat((E_positional, RBF_all), axis=-1)
        # [B, L, K, edge_features]
        E = self.edge_embedding(E)
        # [B, L, K, edge_features], normalized
        E = self.norm_edges(E)

        if self.use_side_chains:
            xyz_37 = input_features["xyz_37"]
            xyz_37_m = input_features["xyz_37_m"]
            E_idx_sub = E_idx[:, :, :16]  # [B, L, 15]
            mask_residues = input_features["chain_mask"]
            xyz_37_m = xyz_37_m * (1 - mask_residues[:, :, None])
            R_m = gather_nodes(xyz_37_m[:, :, 5:], E_idx_sub)

            X_sidechain = xyz_37[:, :, 5:, :].view(B, L, -1)
            R = gather_nodes(X_sidechain, E_idx_sub).reshape(
                B, L, E_idx_sub.shape[2], -1, 3
            )
            R_t = jnp.tile(
                self.side_chain_atom_types[None, None, None, :],
                (B, L, E_idx_sub.shape[2], 1),
            )

            R = R.reshape(B, L, -1, 3)  # coordinates
            R_m = R_m.reshape(B, L, -1)  # mask
            R_t = R_t.reshape(B, L, -1)  # atom types

            # Ligand atom context
            Y = jnp.concat((R, Y), axis=2)  # [B, L, atoms, 3]
            Y_m = jnp.concat((R_m, Y_m), axis=2)  # [B, L, atoms]
            Y_t = jnp.concat((R_t, Y_t), axis=2)  # [B, L, atoms]

            Cb_Y_distances = jnp.sum((Cb[:, :, None, :] - Y) ** 2, -1)
            mask_Y = mask[:, :, None] * Y_m
            Cb_Y_distances_adjusted = Cb_Y_distances * mask_Y + (1.0 - mask_Y) * 10000.0
            _, E_idx_Y = jax.lax.top_k(-Cb_Y_distances_adjusted, self.atom_context_num)

            Y = Y[jnp.arange(B)[:, None, None], jnp.arange(L)[None, :, None], E_idx_Y]
            Y_t = Y_t[
                jnp.arange(B)[:, None, None], jnp.arange(L)[None, :, None], E_idx_Y
            ]
            Y_m = Y_m[
                jnp.arange(B)[:, None, None], jnp.arange(L)[None, :, None], E_idx_Y
            ]

        # group; 19 categories including 0
        Y_t_g = self.periodic_table_features[1][Y_t]
        # period; 8 categories including 0
        Y_t_p = self.periodic_table_features[2][Y_t]

        Y_t_g_1hot_ = nnx.one_hot(Y_t_g, 19)  # [B, L, M, 19]
        Y_t_p_1hot_ = nnx.one_hot(Y_t_p, 8)  # [B, L, M, 8]
        Y_t_1hot_ = nnx.one_hot(Y_t, 120)  # [B, L, M, 120]

        Y_t_1hot_ = jnp.concat(
            [Y_t_1hot_, Y_t_g_1hot_, Y_t_p_1hot_], axis=-1
        )  # [B, L, M, 147]
        Y_t_1hot = self.type_linear(Y_t_1hot_)

        D_N_Y = self._rbf(
            jnp.sqrt(jnp.sum((N[:, :, None, :] - Y) ** 2, -1) + 1e-6)
        )  # [B, L, M, num_bins]
        D_Ca_Y = self._rbf(jnp.sqrt(jnp.sum((Ca[:, :, None, :] - Y) ** 2, -1) + 1e-6))
        D_C_Y = self._rbf(jnp.sqrt(jnp.sum((C[:, :, None, :] - Y) ** 2, -1) + 1e-6))
        D_O_Y = self._rbf(jnp.sqrt(jnp.sum((O[:, :, None, :] - Y) ** 2, -1) + 1e-6))
        D_Cb_Y = self._rbf(jnp.sqrt(jnp.sum((Cb[:, :, None, :] - Y) ** 2, -1) + 1e-6))

        f_angles = self._make_angle_features(N, Ca, C, Y)  # [B, L, M, 4]

        D_all = jnp.concat(
            (D_N_Y, D_Ca_Y, D_C_Y, D_O_Y, D_Cb_Y, Y_t_1hot, f_angles), axis=-1
        )  # [B,L,M,5*num_bins+5]
        V = self.node_project_down(D_all)  # [B, L, M, node_features]
        V = self.norm_nodes(V)

        Y_edges = self._rbf(
            jnp.sqrt(
                jnp.sum((Y[:, :, :, None, :] - Y[:, :, None, :, :]) ** 2, -1) + 1e-6
            )
        )  # [B, L, M, M, num_bins]

        Y_edges = self.y_edges(Y_edges)
        Y_nodes = self.y_nodes(Y_t_1hot_)

        Y_edges = self.norm_y_edges(Y_edges)
        Y_nodes = self.norm_y_nodes(Y_nodes)

        return V, E, E_idx, Y_nodes, Y_edges, Y_m
