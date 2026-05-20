import argparse
import copy
import json
import random
import numpy as np
from .data_utils import (
    featurize_proteinmpnn,
    get_score,
    get_seq_rec,
    parse_PDB,
    write_full_PDB,
)
from .constants import ALPHABET, RESTYPE_STRTOINT, ELEMENT_DICT, RESTYPE_1TO3
from .model import ProteinMPNN
from prody import writePDB
import jax
from .utils import protein_dict_to_serializable
from pathlib import Path
import orbax.checkpoint as ocp
from flax import nnx
import jax.numpy as jnp
from dataclasses import dataclass, field


@dataclass
class ProteinMPNNConfig:
    pdb_path: Path
    out_dir: Path
    seed: int
    batch_size: int
    temperature: float
    checkpoint_path: Path
    chains_to_design: set[str]
    fixed_residues: set[str]
    residues_to_design: set[str]
    parse_these_chains_only: tuple
    verbose: bool = False
    save_stats: bool = False
    atom_context_num: int = 1
    bias_AA: str = ""
    omit_AA: str = ""
    num_edges: int = 32
    noise_level: float = 0.3
    atom_context_num: int = 25
    symmetry_residues: str = ""
    symmetry_weights: str = ""

    @property
    def seqs_dir(self) -> Path:
        return self.out_dir / "seqs"

    @property
    def backbones_dir(self) -> Path:
        return self.out_dir / "backbones"

    @property
    def packed_dir(self) -> Path:
        return self.out_dir / "packed"

    @property
    def stats_dir(self) -> Path:
        return self.out_dir / "stats"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ProteinMPNNConfig":

        pdb_path = Path(args.pdb_path)
        fixed_residues = set(args.fixed_residues.split())
        residues_to_design = set(args.residues_to_design.split())

        assert fixed_residues.isdisjoint(residues_to_design), (
            "Residues cannot be both fixed and designated for design"
        )

        checkpoint_path = Path(args.checkpoint_protein_mpnn)
        assert checkpoint_path.exists(), (
            f"The model path {checkpoint_path} does not exist."
        )

        chains_to_design = (
            set(args.chains_to_design.split(",")) if args.chains_to_design else set()
        )
        parse_these_chains_only = (
            args.parse_these_chains_only.split(",")
            if args.parse_these_chains_only
            else ()
        )

        symmetry_residues = args.symmetry_residues if args.symmetry_residues else ""
        symmetry_weights = args.symmetry_weights if args.symmetry_weights else ""

        return cls(
            pdb_path=pdb_path,
            out_dir=Path(args.out_folder),
            seed=args.seed,
            batch_size=args.batch_size,
            temperature=args.temperature,
            save_stats=args.save_stats,
            checkpoint_path=checkpoint_path,
            bias_AA=args.bias_AA or "",
            omit_AA=args.omit_AA or "",
            chains_to_design=chains_to_design,
            parse_these_chains_only=parse_these_chains_only,
            fixed_residues=fixed_residues,
            residues_to_design=residues_to_design,
            verbose=args.verbose,
            symmetry_residues=symmetry_residues,
            symmetry_weights=symmetry_weights,
        )


def make_directory_structure(config: ProteinMPNNConfig):

    config.out_dir.mkdir(exist_ok=True, parents=True)
    config.seqs_dir.mkdir(exist_ok=True, parents=True)
    config.backbones_dir.mkdir(exist_ok=True, parents=True)
    config.packed_dir.mkdir(exist_ok=True, parents=True)
    if config.save_stats:
        config.stats_dir.mkdir(exist_ok=True, parents=True)


def proteinmpnn_predict(args: argparse.Namespace):
    """
    Inference function
    """

    config = ProteinMPNNConfig.from_args(args)

    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(config.checkpoint_path.resolve())
    config.num_edges = restored["num_edges"]

    make_directory_structure(config)

    jax_key = jax.random.key(config.seed)
    params_key, dropout_key = jax.random.split(jax_key, 2)

    rngs = nnx.Rngs(params=params_key, dropout=dropout_key)
    random.seed(config.seed)
    np.random.seed(config.seed)

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=config.num_edges,
        rngs=rngs,
    )

    # checkpointer = ocp.StandardCheckpointer()
    # _, abstract_state = nnx.split(model)
    # state = checkpointer.restore(config.checkpoint_path.absolute(), abstract_state)
    # nnx.update(model, state)

    bias_AA = jnp.zeros([21], dtype=jnp.float32)
    if config.bias_AA:
        for item in config.bias_AA.split(","):
            aa, val = item.split(":")
            bias_AA = bias_AA.at[RESTYPE_STRTOINT[aa]].set(float(val))

    omit_AA = jnp.array([AA in config.omit_AA for AA in ALPHABET], dtype=jnp.float32)

    parse_these_chains_only_list = config.parse_these_chains_only

    pdb = config.pdb_path
    if config.verbose:
        print(f"Designing protein from this path: {pdb}")

    protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
        str(pdb),
        chains=tuple(parse_these_chains_only_list),
        parse_all_atoms=False,
        parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy,
    )
    # set other atom bfactors to 0.0
    if other_atoms:
        other_bfactors = other_atoms.getBetas()  # type: ignore
        other_atoms.setBetas(other_bfactors * 0.0)  # type: ignore

    # make chain_letter + residue_idx + insertion_code mapping to integers
    R_idx_list = np.array(protein_dict["R_idx"]).tolist()
    chain_letters_list = protein_dict["chain_letters"]

    encoded_residues = [
        f"{(chain_letters_list[i])}{R_idx_item}{icodes[i]}"
        for i, R_idx_item in enumerate(R_idx_list)
    ]

    encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
    encoded_residue_dict_rev = dict(
        zip(list(range(len(encoded_residues))), encoded_residues)
    )

    bias_AA_per_residue = jnp.zeros([len(encoded_residues), 21], dtype=jnp.float32)
    omit_AA_per_residue = jnp.zeros([len(encoded_residues), 21], dtype=jnp.float32)

    # 1 if fixed
    fixed_positions = jnp.array(
        [int(item not in config.fixed_residues) for item in encoded_residues],
    )
    # 1 if to be designed
    positions_to_design = jnp.array(
        [int(item not in config.residues_to_design) for item in encoded_residues],
    )

    chains_to_design = (
        config.chains_to_design
        if config.chains_to_design
        else protein_dict["chain_letters"]
    )
    # 1 if the residue is in a chain to be designed
    chain_mask = jnp.array(
        [item in chains_to_design for item in protein_dict["chain_letters"]],
        dtype=np.int32,
    )

    other_data_dict = {}

    # 0 means residue is fixed, 1 if to be designed
    if config.residues_to_design:
        other_data_dict["chain_mask"] = chain_mask * (1 - positions_to_design)
    elif config.fixed_residues:
        other_data_dict["chain_mask"] = chain_mask * fixed_positions
    # otherwise the whole chain will be designed
    else:
        other_data_dict["chain_mask"] = chain_mask

    if config.verbose:
        PDB_residues_to_be_redesigned = [
            encoded_residue_dict_rev[item]
            for item in range(len(other_data_dict["chain_mask"]))
            if other_data_dict["chain_mask"][item] == 1
        ]
        PDB_residues_to_be_fixed = [
            encoded_residue_dict_rev[item]
            for item in range(len(other_data_dict["chain_mask"]))
            if other_data_dict["chain_mask"][item] == 0
        ]
        print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
        print("These residues will be fixed: ", PDB_residues_to_be_fixed)

    # specify which residues are linked
    if config.symmetry_residues:
        symmetry_residues_list_of_lists = [
            x.split(",") for x in config.symmetry_residues.split("|")
        ]
        remapped_symmetry_residues = []
        for t_list in symmetry_residues_list_of_lists:
            tmp_list = []
            for t in t_list:
                tmp_list.append(encoded_residue_dict[t])
            remapped_symmetry_residues.append(tmp_list)
    else:
        remapped_symmetry_residues = [[]]

    # specify linking weights
    if config.symmetry_weights:
        symmetry_weights = [
            [float(item) for item in x.split(",")]
            for x in args.symmetry_weights.split("|")
        ]
    else:
        symmetry_weights = [[]]

    if args.homo_oligomer:
        if args.verbose:
            print("Designing HOMO-OLIGOMER")
        chain_letters_set = list(set(chain_letters_list))
        reference_chain = chain_letters_set[0]
        lc = len(reference_chain)
        residue_indices = [
            item[lc:] for item in encoded_residues if item[:lc] == reference_chain
        ]
        remapped_symmetry_residues = []
        symmetry_weights = []
        for res in residue_indices:
            tmp_list = []
            tmp_w_list = []
            for chain in chain_letters_set:
                name = chain + res
                tmp_list.append(encoded_residue_dict[name])
                tmp_w_list.append(1 / len(chain_letters_set))
            remapped_symmetry_residues.append(tmp_list)
            symmetry_weights.append(tmp_w_list)

    other_data_dict["symmetry_residues"] = remapped_symmetry_residues
    other_data_dict["symmetry_weights"] = symmetry_weights

    feature_dict = featurize_proteinmpnn(
        protein_dict,
    )
    feature_dict["batch_size"] = args.batch_size
    B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
    # add additional keys to the feature dictionary
    feature_dict["temperature"] = args.temperature
    feature_dict["bias"] = (
        jnp.tile(-1e8 * omit_AA[None, None, :] + bias_AA, (1, L, 1))
        + bias_AA_per_residue[None]
        - 1e8 * omit_AA_per_residue[None]
    )

    # update the dictionary with any new keys produced here
    # i.e. for symmetry and chain mask
    feature_dict.update(other_data_dict)
    sampling_probs_list = []
    log_probs_list = []
    decoding_order_list = []
    S_list = []
    loss_list = []
    loss_per_residue_list = []
    loss_XY_list = []
    for _ in range(args.number_of_batches):
        jax_key, randn_key, sample_key = jax.random.split(jax_key, 3)
        feature_dict["randn"] = jax.random.normal(
            randn_key, shape=(feature_dict["batch_size"], feature_dict["mask"].shape[1])
        )

        output_dict = model.sample(feature_dict, key=sample_key)

        # compute confidence scores
        loss, loss_per_residue = get_score(
            output_dict["S"],
            output_dict["log_probs"],
            feature_dict["mask"] * feature_dict["chain_mask"],
        )
        if args.model_type == "ligand_mpnn":
            combined_mask = (
                feature_dict["mask"]
                * feature_dict["mask_XY"]
                * feature_dict["chain_mask"]
            )
        else:
            combined_mask = feature_dict["mask"] * feature_dict["chain_mask"]
        loss_XY, _ = get_score(
            output_dict["S"], output_dict["log_probs"], combined_mask
        )
        # -----
        S_list.append(output_dict["S"])
        log_probs_list.append(output_dict["log_probs"])
        sampling_probs_list.append(output_dict["sampling_probs"])
        decoding_order_list.append(output_dict["decoding_order"])
        loss_list.append(loss)
        loss_per_residue_list.append(loss_per_residue)
        loss_XY_list.append(loss_XY)
    S_stack = jnp.concat(S_list, axis=0)
    log_probs_stack = jnp.concat(log_probs_list, axis=0)
    sampling_probs_stack = jnp.concat(sampling_probs_list, axis=0)
    decoding_order_stack = jnp.concat(decoding_order_list, axis=0)
    loss_stack = jnp.concat(loss_list, axis=0)
    loss_per_residue_stack = jnp.concat(loss_per_residue_list, axis=0)
    loss_XY_stack = jnp.concat(loss_XY_list, axis=0)
    rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
    rec_stack = get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)

    native_seq = "".join(
        [restype_int_to_str[AA] for AA in feature_dict["S"][0].cpu().numpy()]
    )
    seq_np = np.array(list(native_seq))
    seq_out_str = []
    for mask in protein_dict["mask_c"]:
        seq_out_str += list(seq_np[mask.cpu().numpy()])
        seq_out_str += [args.fasta_seq_separation]
    seq_out_str = "".join(seq_out_str)[:-1]

    output_fasta = config.out_dir / "seqs" / f"{pdb.stem + args.file_ending + '.fa'}"
    output_backbones = config.out_dir + "/backbones/"
    output_packed = config.out_dir + "/packed/"
    output_stats_path = config.out_dir + "stats/" + name + args.file_ending + ".pt"

    out_dict = {}
    out_dict["generated_sequences"] = np.array(S_stack)
    out_dict["sampling_probs"] = np.array(sampling_probs_stack)
    out_dict["log_probs"] = np.array(log_probs_stack)
    out_dict["decoding_order"] = np.array(decoding_order_stack)
    out_dict["native_sequence"] = np.array(feature_dict["S"][0])
    out_dict["mask"] = np.array(feature_dict["mask"][0])
    out_dict["chain_mask"] = np.array(feature_dict["chain_mask"][0])
    out_dict["seed"] = config.seed
    out_dict["temperature"] = args.temperature
    if args.save_stats:
        np.save(output_stats_path, out_dict)

    if args.pack_side_chains:
        if args.verbose:
            print("Packing side chains...")
        feature_dict_ = featurize(
            protein_dict,
            cutoff_for_score=8.0,
            use_atom_context=args.pack_with_ligand_context,
            number_of_ligand_atoms=16,
            model_type="ligand_mpnn",
        )
        sc_feature_dict = copy.deepcopy(feature_dict_)
        B = args.batch_size
        for k, v in sc_feature_dict.items():
            if k != "S":
                try:
                    num_dim = len(v.shape)
                    if num_dim == 2:
                        sc_feature_dict[k] = v.repeat(B, 1)
                    elif num_dim == 3:
                        sc_feature_dict[k] = v.repeat(B, 1, 1)
                    elif num_dim == 4:
                        sc_feature_dict[k] = v.repeat(B, 1, 1, 1)
                    elif num_dim == 5:
                        sc_feature_dict[k] = v.repeat(B, 1, 1, 1, 1)
                except:
                    pass
        X_stack_list = []
        X_m_stack_list = []
        b_factor_stack_list = []
        for _ in range(args.number_of_packs_per_design):
            X_list = []
            X_m_list = []
            b_factor_list = []
            for c in range(args.number_of_batches):
                sc_feature_dict["S"] = S_list[c]
                sc_dict = pack_side_chains(
                    sc_feature_dict,
                    model_sc,
                    args.sc_num_denoising_steps,
                    args.sc_num_samples,
                    args.repack_everything,
                )
                X_list.append(sc_dict["X"])
                X_m_list.append(sc_dict["X_m"])
                b_factor_list.append(sc_dict["b_factors"])

            X_stack = torch.cat(X_list, 0)
            X_m_stack = torch.cat(X_m_list, 0)
            b_factor_stack = torch.cat(b_factor_list, 0)

            X_stack_list.append(X_stack)
            X_m_stack_list.append(X_m_stack)
            b_factor_stack_list.append(b_factor_stack)

    with open(output_fasta, "w") as f:
        f.write(
            ">{}, T={}, seed={}, num_res={}, num_ligand_res={}, use_ligand_context={}, ligand_cutoff_distance={}, batch_size={}, number_of_batches={}, model_path={}\n{}\n".format(
                name,
                args.temperature,
                seed,
                torch.sum(rec_mask).cpu().numpy(),
                torch.sum(combined_mask[:1]).cpu().numpy(),
                bool(args.ligand_mpnn_use_atom_context),
                float(args.ligand_mpnn_cutoff_for_score),
                args.batch_size,
                args.number_of_batches,
                checkpoint_path,
                seq_out_str,
            )
        )
        for ix in range(S_stack.shape[0]):
            ix_suffix = ix
            if not args.zero_indexed:
                ix_suffix += 1
            seq_rec_print = np.format_float_positional(
                rec_stack[ix].cpu().numpy(), unique=False, precision=4
            )
            loss_np = np.format_float_positional(
                np.exp(-loss_stack[ix].cpu().numpy()), unique=False, precision=4
            )
            loss_XY_np = np.format_float_positional(
                np.exp(-loss_XY_stack[ix].cpu().numpy()),
                unique=False,
                precision=4,
            )
            seq = "".join([restype_int_to_str[AA] for AA in S_stack[ix].cpu().numpy()])

            # write new sequences into PDB with backbone coordinates
            seq_prody = np.array([restype_1to3[AA] for AA in list(seq)])[None,].repeat(
                4, 1
            )
            bfactor_prody = (
                loss_per_residue_stack[ix].cpu().numpy()[None, :].repeat(4, 1)
            )
            backbone.setResnames(seq_prody)  # type: ignore
            backbone.setBetas(  # type: ignore
                np.exp(-bfactor_prody) * (bfactor_prody > 0.01).astype(np.float32)
            )
            if other_atoms:
                writePDB(
                    output_backbones
                    + name
                    + "_"
                    + str(ix_suffix)
                    + args.file_ending
                    + ".pdb",
                    backbone + other_atoms,
                )
            else:
                writePDB(
                    output_backbones
                    + name
                    + "_"
                    + str(ix_suffix)
                    + args.file_ending
                    + ".pdb",
                    backbone,
                )

            # TODO: add back in
            # write full PDB files
            # if args.pack_side_chains:
            #     for c_pack in range(args.number_of_packs_per_design):
            #         X_stack = X_stack_list[c_pack]
            #         X_m_stack = X_m_stack_list[c_pack]
            #         b_factor_stack = b_factor_stack_list[c_pack]
            #         write_full_PDB(
            #             output_packed
            #             + name
            #             + args.packed_suffix
            #             + "_"
            #             + str(ix_suffix)
            #             + "_"
            #             + str(c_pack + 1)
            #             + args.file_ending
            #             + ".pdb",
            #             X_stack[ix].cpu().numpy(),
            #             X_m_stack[ix].cpu().numpy(),
            #             b_factor_stack[ix].cpu().numpy(),
            #             feature_dict["R_idx_original"][0].cpu().numpy(),
            #             protein_dict["chain_letters"],
            #             S_stack[ix].cpu().numpy(),
            #             other_atoms=other_atoms,
            #             icodes=icodes,
            #             force_hetatm=args.force_hetatm,
            #         )
            # -----

            # write fasta lines
            seq_np = np.array(list(seq))
            seq_out_str = []
            for mask in protein_dict["mask_c"]:
                seq_out_str += list(seq_np[mask.cpu().numpy()])
                seq_out_str += [args.fasta_seq_separation]
            seq_out_str = "".join(seq_out_str)[:-1]
            if ix == S_stack.shape[0] - 1:
                # final 2 lines
                f.write(
                    ">{}, id={}, T={}, seed={}, overall_confidence={}, ligand_confidence={}, seq_rec={}\n{}".format(
                        name,
                        ix_suffix,
                        args.temperature,
                        seed,
                        loss_np,
                        loss_XY_np,
                        seq_rec_print,
                        seq_out_str,
                    )
                )
            else:
                f.write(
                    ">{}, id={}, T={}, seed={}, overall_confidence={}, ligand_confidence={}, seq_rec={}\n{}\n".format(
                        name,
                        ix_suffix,
                        args.temperature,
                        seed,
                        loss_np,
                        loss_XY_np,
                        seq_rec_print,
                        seq_out_str,
                    )
                )
