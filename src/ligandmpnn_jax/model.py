import numpy as np
import jax.numpy as jnp
from flax import nnx
import jax


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
        *,
        rngs: nnx.Rngs,
    ):

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.features = ProteinFeatures(
            node_features=node_features,
            edge_features=edge_features,
            top_k=k_neighbors,
            augment_eps=augment_eps,
            rngs=rngs,
        )

        self.W_v = nnx.Linear(node_features, hidden_dim, use_bias=True, rngs=rngs)
        self.W_c = nnx.Linear(hidden_dim, hidden_dim, use_bias=True, rngs=rngs)
        self.W_e = nnx.Linear(edge_features, hidden_dim, use_bias=True, rngs=rngs)

        self.encoder_layers = nnx.List(
            [
                EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout, rngs=rngs)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nnx.List(
            [
                DecLayer(hidden_dim, hidden_dim * 2, dropout=dropout, rngs=rngs)
                for _ in range(num_decoder_layers)
            ]
        )

    def encode(self, feature_dict):

        # xyz_37 = feature_dict["xyz_37"] #[B,L,37,3] - xyz coordinates for all atoms if needed
        # xyz_37_m = feature_dict["xyz_37_m"] #[B,L,37] - mask for all coords
        # X = feature_dict["X"] #[B,L,4,3] - backbone xyz coordinates for N,CA,C,O

        # [B,L] - integer protein sequence encoded using "restype_STRtoINT"
        S_true = feature_dict["S"]

        # [B,L] - mask for missing regions - should be removed! all ones most of the time
        mask = feature_dict["mask"]
        mask: jax.Array

        B, L = S_true.shape
        device = S_true.device

        E, E_idx = self.features(feature_dict)
        h_V = jnp.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask[..., None], E_idx).squeeze(-1)
        mask_attend = mask[..., None] * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        return h_V, h_E, E_idx

    def sample(self, feature_dict):
        # xyz_37 = feature_dict["xyz_37"] #[B,L,37,3] - xyz coordinates for all atoms if needed
        # xyz_37_m = feature_dict["xyz_37_m"] #[B,L,37] - mask for all coords
        # Y = feature_dict["Y"] #[B,L,num_context_atoms,3] - for ligandMPNN coords
        # Y_t = feature_dict["Y_t"] #[B,L,num_context_atoms] - element type
        # Y_m = feature_dict["Y_m"] #[B,L,num_context_atoms] - mask
        # X = feature_dict["X"] #[B,L,4,3] - backbone xyz coordinates for N,CA,C,O
        B_decoder = feature_dict["batch_size"]
        S_true = feature_dict[
            "S"
        ]  # [B,L] - integer proitein sequence encoded using "restype_STRtoINT"
        # R_idx = feature_dict["R_idx"] #[B,L] - primary sequence residue index
        mask = feature_dict[
            "mask"
        ]  # [B,L] - mask for missing regions - should be removed! all ones most of the time
        chain_mask = feature_dict[
            "chain_mask"
        ]  # [B,L] - mask for which residues need to be fixed; 0.0 - fixed; 1.0 - will be designed
        bias = feature_dict["bias"]  # [B,L,21] - amino acid bias per position
        # chain_labels = feature_dict["chain_labels"] #[B,L] - integer labels for chain letters
        randn = feature_dict[
            "randn"
        ]  # [B,L] - random numbers for decoding order; only the first entry is used since decoding within a batch needs to match for symmetry
        temperature = feature_dict[
            "temperature"
        ]  # float - sampling temperature; prob = softmax(logits/temperature)
        symmetry_list_of_lists = feature_dict[
            "symmetry_residues"
        ]  # [[0, 1, 14], [10,11,14,15], [20, 21]] #indices to select X over length - L
        symmetry_weights_list_of_lists = feature_dict[
            "symmetry_weights"
        ]  # [[1.0, 1.0, 1.0], [-2.0,1.1,0.2,1.1], [2.3, 1.1]]

        B, L = S_true.shape
        device = S_true.device

        h_V, h_E, E_idx = self.encode(feature_dict)

        print(h_V.shape)
        exit()

        chain_mask = mask * chain_mask  # update chain_M to include missing regions
        decoding_order = torch.argsort(
            (chain_mask + 0.0001) * (torch.abs(randn))
        )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        if len(symmetry_list_of_lists[0]) == 0 and len(symmetry_list_of_lists) == 1:
            E_idx = E_idx.repeat(B_decoder, 1, 1)
            permutation_matrix_reverse = torch.nn.functional.one_hot(
                decoding_order, num_classes=L
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(L, L, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([B, L, 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)

            # repeat for decoding
            S_true = S_true.repeat(B_decoder, 1)
            h_V = h_V.repeat(B_decoder, 1, 1)
            h_E = h_E.repeat(B_decoder, 1, 1, 1)
            chain_mask = chain_mask.repeat(B_decoder, 1)
            mask = mask.repeat(B_decoder, 1)
            bias = bias.repeat(B_decoder, 1, 1)

            all_probs = torch.zeros(
                (B_decoder, L, 20), device=device, dtype=torch.float32
            )
            all_log_probs = torch.zeros(
                (B_decoder, L, 21), device=device, dtype=torch.float32
            )
            h_S = torch.zeros_like(h_V, device=device)
            S = 20 * torch.ones((B_decoder, L), dtype=torch.int64, device=device)
            h_V_stack = [h_V] + [
                torch.zeros_like(h_V, device=device)
                for _ in range(len(self.decoder_layers))
            ]

            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
            h_EXV_encoder_fw = mask_fw * h_EXV_encoder

            for t_ in range(L):
                t = decoding_order[:, t_]  # [B]
                chain_mask_t = torch.gather(chain_mask, 1, t[:, None])[:, 0]  # [B]
                mask_t = torch.gather(mask, 1, t[:, None])[:, 0]  # [B]
                bias_t = torch.gather(bias, 1, t[:, None, None].repeat(1, 1, 21))[
                    :, 0, :
                ]  # [B,21]

                E_idx_t = torch.gather(
                    E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1])
                )
                h_E_t = torch.gather(
                    h_E,
                    1,
                    t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]),
                )
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(
                    h_EXV_encoder_fw,
                    1,
                    t[:, None, None, None].repeat(
                        1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]
                    ),
                )

                mask_bw_t = torch.gather(
                    mask_bw,
                    1,
                    t[:, None, None, None].repeat(
                        1, 1, mask_bw.shape[-2], mask_bw.shape[-1]
                    ),
                )

                for l, layer in enumerate(self.decoder_layers):
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(
                        h_V_stack[l],
                        1,
                        t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]),
                    )
                    h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l + 1].scatter_(
                        1,
                        t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                        layer(h_V_t, h_ESV_t, mask_V=mask_t),
                    )

                h_V_t = torch.gather(
                    h_V_stack[-1],
                    1,
                    t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]),
                )[:, 0]
                logits = self.W_out(h_V_t)  # [B,21]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [B,21]

                probs = torch.nn.functional.softmax(
                    (logits + bias_t) / temperature, dim=-1
                )  # [B,21]
                probs_sample = probs[:, :20] / torch.sum(
                    probs[:, :20], dim=-1, keepdim=True
                )  # hard omit X #[B,20]
                S_t = torch.multinomial(probs_sample, 1)[:, 0]  # [B]

                all_probs.scatter_(
                    1,
                    t[:, None, None].repeat(1, 1, 20),
                    (chain_mask_t[:, None, None] * probs_sample[:, None, :]).float(),
                )
                all_log_probs.scatter_(
                    1,
                    t[:, None, None].repeat(1, 1, 21),
                    (chain_mask_t[:, None, None] * log_probs[:, None, :]).float(),
                )
                S_true_t = torch.gather(S_true, 1, t[:, None])[:, 0]
                S_t = (S_t * chain_mask_t + S_true_t * (1.0 - chain_mask_t)).long()
                h_S.scatter_(
                    1,
                    t[:, None, None].repeat(1, 1, h_S.shape[-1]),
                    self.W_s(S_t)[:, None, :],
                )
                S.scatter_(1, t[:, None], S_t[:, None])

            output_dict = {
                "S": S,
                "sampling_probs": all_probs,
                "log_probs": all_log_probs,
                "decoding_order": decoding_order,
            }

        else:
            # weights for symmetric design
            symmetry_weights = torch.ones([L], device=device, dtype=torch.float32)
            for i1, item_list in enumerate(symmetry_list_of_lists):
                for i2, item in enumerate(item_list):
                    symmetry_weights[item] = symmetry_weights_list_of_lists[i1][i2]

            new_decoding_order = []
            for t_dec in list(decoding_order[0,].cpu().data.numpy()):
                if t_dec not in list(itertools.chain(*new_decoding_order)):
                    list_a = [item for item in symmetry_list_of_lists if t_dec in item]
                    if list_a:
                        new_decoding_order.append(list_a[0])
                    else:
                        new_decoding_order.append([t_dec])

            decoding_order = torch.tensor(
                list(itertools.chain(*new_decoding_order)), device=device
            )[None,].repeat(B, 1)

            permutation_matrix_reverse = torch.nn.functional.one_hot(
                decoding_order, num_classes=L
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(L, L, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([B, L, 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)

            # repeat for decoding
            S_true = S_true.repeat(B_decoder, 1)
            h_V = h_V.repeat(B_decoder, 1, 1)
            h_E = h_E.repeat(B_decoder, 1, 1, 1)
            E_idx = E_idx.repeat(B_decoder, 1, 1)
            mask_fw = mask_fw.repeat(B_decoder, 1, 1, 1)
            mask_bw = mask_bw.repeat(B_decoder, 1, 1, 1)
            chain_mask = chain_mask.repeat(B_decoder, 1)
            mask = mask.repeat(B_decoder, 1)
            bias = bias.repeat(B_decoder, 1, 1)

            all_probs = torch.zeros(
                (B_decoder, L, 20), device=device, dtype=torch.float32
            )
            all_log_probs = torch.zeros(
                (B_decoder, L, 21), device=device, dtype=torch.float32
            )
            h_S = torch.zeros_like(h_V, device=device)
            S = 20 * torch.ones((B_decoder, L), dtype=torch.int64, device=device)
            h_V_stack = [h_V] + [
                torch.zeros_like(h_V, device=device)
                for _ in range(len(self.decoder_layers))
            ]

            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
            h_EXV_encoder_fw = mask_fw * h_EXV_encoder

            for t_list in new_decoding_order:
                total_logits = 0.0
                for t in t_list:
                    chain_mask_t = chain_mask[:, t]  # [B]
                    mask_t = mask[:, t]  # [B]
                    bias_t = bias[:, t]  # [B, 21]

                    E_idx_t = E_idx[:, t : t + 1]
                    h_E_t = h_E[:, t : t + 1]
                    h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                    h_EXV_encoder_t = h_EXV_encoder_fw[:, t : t + 1]
                    for l, layer in enumerate(self.decoder_layers):
                        h_ESV_decoder_t = cat_neighbors_nodes(
                            h_V_stack[l], h_ES_t, E_idx_t
                        )
                        h_V_t = h_V_stack[l][:, t : t + 1]
                        h_ESV_t = (
                            mask_bw[:, t : t + 1] * h_ESV_decoder_t + h_EXV_encoder_t
                        )
                        h_V_stack[l + 1][:, t : t + 1, :] = layer(
                            h_V_t, h_ESV_t, mask_V=mask_t[:, None]
                        )

                    h_V_t = h_V_stack[-1][:, t]
                    logits = self.W_out(h_V_t)  # [B,21]
                    log_probs = torch.nn.functional.log_softmax(
                        logits, dim=-1
                    )  # [B,21]
                    all_log_probs[:, t] = (
                        chain_mask_t[:, None] * log_probs
                    ).float()  # [B,21]
                    total_logits += symmetry_weights[t] * logits

                probs = torch.nn.functional.softmax(
                    (total_logits + bias_t) / temperature, dim=-1
                )  # [B,21]
                probs_sample = probs[:, :20] / torch.sum(
                    probs[:, :20], dim=-1, keepdim=True
                )  # hard omit X #[B,20]
                S_t = torch.multinomial(probs_sample, 1)[:, 0]  # [B]
                for t in t_list:
                    chain_mask_t = chain_mask[:, t]  # [B]
                    all_probs[:, t] = (
                        chain_mask_t[:, None] * probs_sample
                    ).float()  # [B,20]
                    S_true_t = S_true[:, t]  # [B]
                    S_t = (S_t * chain_mask_t + S_true_t * (1.0 - chain_mask_t)).long()
                    h_S[:, t] = self.W_s(S_t)
                    S[:, t] = S_t

            output_dict = {
                "S": S,
                "sampling_probs": all_probs,
                "log_probs": all_log_probs,
                "decoding_order": decoding_order.repeat(B_decoder, 1),
            }
        return output_dict

    def single_aa_score(self, feature_dict, use_sequence: bool):
        """
        feature_dict - input features
        use_sequence - False using backbone info only
        """
        B_decoder = feature_dict["batch_size"]
        S_true_enc = feature_dict["S"]
        mask_enc = feature_dict["mask"]
        chain_mask_enc = feature_dict["chain_mask"]
        randn = feature_dict["randn"]
        B, L = S_true_enc.shape
        device = S_true_enc.device

        h_V_enc, h_E_enc, E_idx_enc = self.encode(feature_dict)
        log_probs_out = torch.zeros([B_decoder, L, 21], device=device).float()
        logits_out = torch.zeros([B_decoder, L, 21], device=device).float()
        decoding_order_out = torch.zeros([B_decoder, L, L], device=device).float()

        for idx in range(L):
            h_V = torch.clone(h_V_enc)
            E_idx = torch.clone(E_idx_enc)
            mask = torch.clone(mask_enc)
            S_true = torch.clone(S_true_enc)
            if not use_sequence:
                order_mask = torch.zeros(chain_mask_enc.shape[1], device=device).float()
                order_mask[idx] = 1.0
            else:
                order_mask = torch.ones(chain_mask_enc.shape[1], device=device).float()
                order_mask[idx] = 0.0
            decoding_order = torch.argsort(
                (order_mask + 0.0001) * (torch.abs(randn))
            )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            E_idx = E_idx.repeat(B_decoder, 1, 1)
            permutation_matrix_reverse = torch.nn.functional.one_hot(
                decoding_order, num_classes=L
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(L, L, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([B, L, 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)
            S_true = S_true.repeat(B_decoder, 1)
            h_V = h_V.repeat(B_decoder, 1, 1)
            h_E = h_E_enc.repeat(B_decoder, 1, 1, 1)
            mask = mask.repeat(B_decoder, 1)

            h_S = self.W_s(S_true)
            h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

            # Build encoder embeddings
            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see.
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            log_probs_out[:, idx, :] = log_probs[:, idx, :]
            logits_out[:, idx, :] = logits[:, idx, :]
            decoding_order_out[:, idx, :] = decoding_order

        output_dict = {
            "S": S_true,
            "log_probs": log_probs_out,
            "logits": logits_out,
            "decoding_order": decoding_order_out,
        }
        return output_dict

    def score(self, feature_dict, use_sequence: bool):
        B_decoder = feature_dict["batch_size"]
        S_true = feature_dict["S"]
        mask = feature_dict["mask"]
        chain_mask = feature_dict["chain_mask"]
        randn = feature_dict["randn"]
        symmetry_list_of_lists = feature_dict["symmetry_residues"]
        B, L = S_true.shape
        device = S_true.device

        h_V, h_E, E_idx = self.encode(feature_dict)

        chain_mask = mask * chain_mask  # update chain_M to include missing regions
        decoding_order = torch.argsort(
            (chain_mask + 0.0001) * (torch.abs(randn))
        )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        if len(symmetry_list_of_lists[0]) == 0 and len(symmetry_list_of_lists) == 1:
            E_idx = E_idx.repeat(B_decoder, 1, 1)
            permutation_matrix_reverse = torch.nn.functional.one_hot(
                decoding_order, num_classes=L
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(L, L, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([B, L, 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)
        else:
            new_decoding_order = []
            for t_dec in list(decoding_order[0,].cpu().data.numpy()):
                if t_dec not in list(itertools.chain(*new_decoding_order)):
                    list_a = [item for item in symmetry_list_of_lists if t_dec in item]
                    if list_a:
                        new_decoding_order.append(list_a[0])
                    else:
                        new_decoding_order.append([t_dec])

            decoding_order = torch.tensor(
                list(itertools.chain(*new_decoding_order)), device=device
            )[None,].repeat(B, 1)

            permutation_matrix_reverse = torch.nn.functional.one_hot(
                decoding_order, num_classes=L
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(L, L, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([B, L, 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)

            E_idx = E_idx.repeat(B_decoder, 1, 1)
            mask_fw = mask_fw.repeat(B_decoder, 1, 1, 1)
            mask_bw = mask_bw.repeat(B_decoder, 1, 1, 1)
            decoding_order = decoding_order.repeat(B_decoder, 1)

        S_true = S_true.repeat(B_decoder, 1)
        h_V = h_V.repeat(B_decoder, 1, 1)
        h_E = h_E.repeat(B_decoder, 1, 1, 1)
        mask = mask.repeat(B_decoder, 1)

        h_S = self.W_s(S_true)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        if not use_sequence:
            for layer in self.decoder_layers:
                h_V = layer(h_V, h_EXV_encoder_fw, mask)
        else:
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see.
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        output_dict = {
            "S": S_true,
            "log_probs": log_probs,
            "logits": logits,
            "decoding_order": decoding_order,
        }
        return output_dict


class PositionalEncodings(nnx.Module):
    def __init__(self, num_embeddings, max_relative_feature=32, *, rngs: nnx.Rngs):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nnx.Linear(
            2 * max_relative_feature + 1 + 1, num_embeddings, rngs=rngs
        )

    def __call__(self, offset, mask):
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

        self.rngs = rngs
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
        D_max = jnp.max(D, axis=-1, keepdims=True)
        D_adjust = D + (1.0 - mask_2D) * D_max

        k = np.minimum(self.top_k, X.shape[1])
        idx = jnp.argsort(D_adjust, axis=-1)[..., :k]
        D_neighbors = jnp.take_along_axis(D_adjust, idx, axis=-1)

        return D_neighbors, idx

    def _rbf(self, D):
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_mu = D_mu[None, None, None, :]
        D_sigma = (D_max - D_min) / D_count
        D_expand = D[..., None]
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

    def __call__(self, input_features):

        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]

        if self.augment_eps > 0:
            X = X + self.augment_eps * jax.random.normal(self.rngs.noise(), X.shape)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = jnp.cross(b, c, axis=-1)
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
        RBF_all = jnp.concat(tuple(RBF_all), axis=-1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).astype(
            jnp.int32
        )  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset, E_chains)
        E = jnp.concat((E_positional, RBF_all), axis=-1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return E, E_idx


class PositionWiseFeedForward(nnx.Module):
    def __init__(self, num_hidden, num_ff, *, rngs: nnx.Rngs):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nnx.Linear(num_hidden, num_ff, use_bias=True, rngs=rngs)
        self.W_out = nnx.Linear(num_ff, num_hidden, use_bias=True, rngs=rngs)
        self.act = jax.nn.gelu

    def __call__(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class EncLayer(nnx.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30, *, rngs: nnx.Rngs):
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nnx.Dropout(dropout)
        self.dropout2 = nnx.Dropout(dropout)
        self.dropout3 = nnx.Dropout(dropout)
        self.norm1 = nnx.LayerNorm(num_hidden, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_hidden, rngs=rngs)
        self.norm3 = nnx.LayerNorm(num_hidden, rngs=rngs)

        self.W1 = nnx.Linear(num_hidden + num_in, num_hidden, use_bias=True, rngs=rngs)
        self.W2 = nnx.Linear(num_hidden, num_hidden, use_bias=True, rngs=rngs)
        self.W3 = nnx.Linear(num_hidden, num_hidden, use_bias=True, rngs=rngs)
        self.W11 = nnx.Linear(num_hidden + num_in, num_hidden, use_bias=True, rngs=rngs)
        self.W12 = nnx.Linear(num_hidden, num_hidden, use_bias=True, rngs=rngs)
        self.W13 = nnx.Linear(num_hidden, num_hidden, use_bias=True, rngs=rngs)
        self.act = jax.nn.gelu
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4, rngs=rngs)

    def __call__(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        # TODO: add in
        # h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_V_expand = jnp.broadcast_to(
            h_V[..., None, :], (*h_V.shape[:2], h_EV.shape[-2], h_V.shape[-1])
        )
        h_EV = jnp.concatenate([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend[..., None] * h_message
        dh = jnp.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V[..., None]
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = jnp.broadcast_to(
            h_V[..., None, :], (*h_V.shape[:2], h_EV.shape[-2], h_V.shape[-1])
        )
        h_EV = jnp.concat([h_V_expand, h_EV], axis=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nnx.Module):
    def __init__(
        self,
        num_hidden,
        num_in,
        dropout=0.1,
        scale=30,
        *,
        rngs: nnx.Rngs,
    ):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout, rngs=rngs)
        self.norm1 = nnx.LayerNorm(num_hidden, use_bias=True, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_hidden, use_bias=True, rngs=rngs)

        self.W1 = nnx.Linear(num_hidden + num_in, num_hidden, use_bias=True, rngs=rngs)
        self.W2 = nnx.Linear(num_hidden, num_hidden, use_bias=True, rngs=rngs)
        self.W3 = nnx.Linear(num_hidden, num_hidden, use_bias=True, rngs=rngs)
        self.act = jax.nn.gelu
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4, rngs=rngs)

    def __call__(self, h_V, h_E, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        h_V_expand = jnp.broadcast_to(
            h_V[..., None, :], (*h_V.shape[:-1], h_E.shape[-2], h_V.shape[-1])
        )

        h_EV = jnp.concat([h_V_expand, h_E], axis=-1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend[..., None] * h_message
        dh = jnp.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V[..., None]
            h_V = mask_V * h_V
        return h_V


# Gather functions
def gather_edges(edges, neighbor_idx) -> jax.Array:
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = jnp.broadcast_to(
        neighbor_idx[..., None], (*neighbor_idx.shape, edges.shape[-1])
    )
    return jnp.take_along_axis(edges, neighbors, axis=2)


def gather_nodes(nodes, neighbor_idx) -> jax.Array:
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    B, N, K = neighbor_idx.shape
    neighbors_flat = jnp.broadcast_to(
        neighbor_idx.reshape(B, N * K)[..., None], (B, N * K, nodes.shape[-1])
    )
    return jnp.take_along_axis(nodes, neighbors_flat, axis=1).reshape(
        B, N, K, nodes.shape[-1]
    )


def gather_nodes_t(nodes, neighbor_idx) -> jax.Array:
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features [B,K,C]
    idx_flat = jnp.broadcast_to(
        neighbor_idx[..., None], (*neighbor_idx.shape, nodes.shape[-1])
    )
    return jnp.take_along_axis(nodes, idx_flat, axis=1)


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx) -> jax.Array:
    h_nodes = gather_nodes(h_nodes, E_idx)
    return jnp.concatenate([h_neighbors, h_nodes], axis=-1)
