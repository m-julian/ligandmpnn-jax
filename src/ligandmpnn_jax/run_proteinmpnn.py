import argparse

# import json
import random
import numpy as np
from .data_utils import (
    featurize_proteinmpnn,
    get_score,
    get_seq_rec,
    parse_PDB,
)
from .constants import ALPHABET, RESTYPE_STRTOINT, RESTYPE_INTTOSTR, RESTYPE_1TO3
from prody import writePDB
import jax
from pathlib import Path
import orbax.checkpoint as ocp
from flax import nnx
import jax.numpy as jnp
from dataclasses import dataclass
from .utils import load_model, make_fasta_seq


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
    bias_AA: str = ""
    omit_AA: str = ""
    num_edges: int = 32
    symmetry_residues: str = ""
    symmetry_weights: str = ""
    fasta_separation: str = ":"

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

        checkpoint_path = Path(args.checkpoint_proteinmpnn)
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
        fasta_separation = (
            args.fasta_seq_separation if args.fasta_seq_separation else ":"
        )

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
            fasta_separation=fasta_separation,
        )


def make_directory_structure(config: ProteinMPNNConfig):
    config.out_dir.mkdir(exist_ok=True, parents=True)
    config.seqs_dir.mkdir(exist_ok=True, parents=True)
    config.backbones_dir.mkdir(exist_ok=True, parents=True)
    config.packed_dir.mkdir(exist_ok=True, parents=True)
    if config.save_stats:
        config.stats_dir.mkdir(exist_ok=True, parents=True)


def proteinmpnn_predict(args: argparse.Namespace):
    config = ProteinMPNNConfig.from_args(args)

    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(config.checkpoint_path.resolve())
    config.num_edges = int(restored["num_edges"])

    make_directory_structure(config)

    jax_key = jax.random.key(config.seed)
    params_key, dropout_key = jax.random.split(jax_key, 2)

    rngs = nnx.Rngs(params=params_key, dropout=dropout_key)
    random.seed(config.seed)
    np.random.seed(config.seed)

    model = load_model(config.checkpoint_path, rngs, config.num_edges)

    # TODO: store the vocab length in the config? Otherwise 21 occurs everywhere
    bias_AA = jnp.zeros([21], dtype=jnp.float32)
    if config.bias_AA:
        for item in config.bias_AA.split(","):
            aa, val = item.split(":")
            bias_AA = bias_AA.at[RESTYPE_STRTOINT[aa]].set(float(val))

    omit_AA = jnp.array([AA in config.omit_AA for AA in ALPHABET], dtype=jnp.float32)

    pdb = config.pdb_path
    if config.verbose:
        print(f"Designing protein from: {pdb}")

    protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
        str(pdb),
        chains=tuple(config.parse_these_chains_only),
        parse_all_atoms=False,
        parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy,
    )
    if other_atoms:
        other_atoms.setBetas(other_atoms.getBetas() * 0.0)  # type: ignore

    R_idx_list = np.array(protein_dict["R_idx"]).tolist()
    chain_letters_list = protein_dict["chain_letters"]

    encoded_residues = [
        f"{chain_letters_list[i]}{R_idx_item}{icodes[i]}"
        for i, R_idx_item in enumerate(R_idx_list)
    ]
    encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
    encoded_residue_dict_rev = {v: k for k, v in encoded_residue_dict.items()}

    bias_AA_per_residue = jnp.zeros([len(encoded_residues), 21], dtype=jnp.float32)
    omit_AA_per_residue = jnp.zeros([len(encoded_residues), 21], dtype=jnp.float32)

    fixed_positions = jnp.array(
        [int(item not in config.fixed_residues) for item in encoded_residues],
    )
    positions_to_design = jnp.array(
        [int(item not in config.residues_to_design) for item in encoded_residues],
    )

    chains_to_design = (
        config.chains_to_design
        if config.chains_to_design
        else protein_dict["chain_letters"]
    )
    chain_mask = jnp.array(
        [item in chains_to_design for item in protein_dict["chain_letters"]],
        dtype=np.int32,
    )

    other_data_dict = {}
    if config.residues_to_design:
        other_data_dict["chain_mask"] = chain_mask * (1 - positions_to_design)
    elif config.fixed_residues:
        other_data_dict["chain_mask"] = chain_mask * fixed_positions
    else:
        other_data_dict["chain_mask"] = chain_mask

    if config.verbose:
        redesigned = [
            encoded_residue_dict_rev[i]
            for i, m in enumerate(other_data_dict["chain_mask"])
            if m == 1
        ]
        fixed = [
            encoded_residue_dict_rev[i]
            for i, m in enumerate(other_data_dict["chain_mask"])
            if m == 0
        ]
        print("Redesigning:", redesigned)
        print("Fixed:", fixed)

    if config.symmetry_residues:
        symmetry_residues_list_of_lists = [
            x.split(",") for x in config.symmetry_residues.split("|")
        ]
        remapped_symmetry_residues = [
            [encoded_residue_dict[t] for t in t_list]
            for t_list in symmetry_residues_list_of_lists
        ]
    else:
        remapped_symmetry_residues = [[]]

    if config.symmetry_weights:
        symmetry_weights = [
            [float(item) for item in x.split(",")]
            for x in config.symmetry_weights.split("|")
        ]
    else:
        symmetry_weights = [[]]

    if args.homo_oligomer:
        if config.verbose:
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
            tmp_list = [
                encoded_residue_dict[chain + res] for chain in chain_letters_set
            ]
            remapped_symmetry_residues.append(tmp_list)
            symmetry_weights.append(
                [1 / len(chain_letters_set)] * len(chain_letters_set)
            )

    other_data_dict["symmetry_residues"] = remapped_symmetry_residues
    other_data_dict["symmetry_weights"] = symmetry_weights

    feature_dict = featurize_proteinmpnn(protein_dict)
    feature_dict["batch_size"] = args.batch_size
    B, L, _, _ = feature_dict["X"].shape
    feature_dict["temperature"] = args.temperature
    feature_dict["bias"] = (
        jnp.tile(-1e8 * omit_AA[None, None, :] + bias_AA, (1, L, 1))
        + bias_AA_per_residue[None]
        - 1e8 * omit_AA_per_residue[None]
    )
    feature_dict.update(other_data_dict)

    S_list = []
    log_probs_list = []
    sampling_probs_list = []
    decoding_order_list = []
    loss_list = []
    loss_per_residue_list = []
    loss_XY_list = []

    for _ in range(args.number_of_batches):
        jax_key, randn_key, sample_key = jax.random.split(jax_key, 3)
        feature_dict["randn"] = jax.random.normal(
            randn_key, shape=(feature_dict["batch_size"], feature_dict["mask"].shape[1])
        )

        output_dict = model.sample(feature_dict, key=sample_key)

        combined_mask = feature_dict["mask"] * feature_dict["chain_mask"]
        loss, loss_per_residue = get_score(
            output_dict["S"], output_dict["log_probs"], combined_mask
        )
        loss_XY, _ = get_score(
            output_dict["S"], output_dict["log_probs"], combined_mask
        )

        S_list.append(output_dict["S"])
        log_probs_list.append(output_dict["log_probs"])
        sampling_probs_list.append(output_dict["sampling_probs"])
        decoding_order_list.append(output_dict["decoding_order"])
        loss_list.append(loss)
        loss_per_residue_list.append(loss_per_residue)
        loss_XY_list.append(loss_XY)

    S_stack = jnp.concat(S_list, axis=0)
    # log_probs_stack = jnp.concat(log_probs_list, axis=0)
    # sampling_probs_stack = jnp.concat(sampling_probs_list, axis=0)
    # decoding_order_stack = jnp.concat(decoding_order_list, axis=0)
    loss_stack = jnp.concat(loss_list, axis=0)
    loss_per_residue_stack = jnp.concat(loss_per_residue_list, axis=0)
    loss_XY_stack = jnp.concat(loss_XY_list, axis=0)

    rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
    rec_stack = get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)

    name = pdb.stem
    native_seq_arr = np.array(feature_dict["S"][0])
    native_seq = "".join(RESTYPE_INTTOSTR[aa] for aa in native_seq_arr)
    seq_np = np.array(list(native_seq))

    output_fasta = config.seqs_dir / f"{name}{args.file_ending}.fa"
    with open(output_fasta, "w") as f:
        # header line with native sequence
        f.write(
            f">{name}, T={args.temperature}, seed={config.seed},"
            f" num_res={int(jnp.sum(rec_mask))}, batch_size={args.batch_size},"
            f" number_of_batches={args.number_of_batches},"
            f" model_path={config.checkpoint_path}\n"
        )
        f.write(
            make_fasta_seq(
                seq_np, protein_dict, fasta_separation=config.fasta_separation
            )
            + "\n"
        )

        for ix in range(S_stack.shape[0]):
            ix_suffix = ix if args.zero_indexed else ix + 1
            seq = "".join(RESTYPE_INTTOSTR[aa] for aa in np.array(S_stack[ix]))
            seq_rec_print = np.format_float_positional(
                float(rec_stack[ix]), unique=False, precision=4
            )
            loss_np = np.format_float_positional(
                float(jnp.exp(-loss_stack[ix])), unique=False, precision=4
            )
            loss_XY_np = np.format_float_positional(
                float(jnp.exp(-loss_XY_stack[ix])), unique=False, precision=4
            )

            # write backbone PDB with b-factors as per-residue confidence
            bfactor_prody = np.array(loss_per_residue_stack[ix])
            seq_prody = np.array([RESTYPE_1TO3[aa] for aa in list(seq)])[None].repeat(
                4, 0
            )
            backbone.setResnames(seq_prody.T.reshape(-1))  # type: ignore
            backbone.setBetas(  # type: ignore
                np.exp(-np.repeat(bfactor_prody, 4))
                * (np.repeat(bfactor_prody, 4) > 0.01).astype(np.float32)
            )
            bb_out = config.backbones_dir / f"{name}_{ix_suffix}{args.file_ending}.pdb"
            if other_atoms:
                writePDB(str(bb_out), backbone + other_atoms)
            else:
                writePDB(str(bb_out), backbone)

            newline = "\n" if ix < S_stack.shape[0] - 1 else ""
            f.write(
                f">{name}, id={ix_suffix}, T={args.temperature}, seed={config.seed},"
                f" overall_confidence={loss_np}, ligand_confidence={loss_XY_np},"
                f" seq_rec={seq_rec_print}\n"
                f"{make_fasta_seq(np.array(list(seq)), protein_dict, config.fasta_separation)}{newline}"
            )

    print(f"Wrote sequences to {output_fasta}")
