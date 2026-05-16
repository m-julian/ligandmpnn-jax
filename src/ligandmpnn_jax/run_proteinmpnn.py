import argparse
import copy
import json
import random
import numpy as np
from .data_utils import (
    featurize,
    get_score,
    get_seq_rec,
    parse_PDB,
    write_full_PDB,
)
from .constants import ALPHABET, RESTYPE_STRTOINT
from .model import ProteinMPNN
from prody import writePDB
import jax
from .utils import protein_dict_to_serializable
from pathlib import Path
import orbax.checkpoint as ocp
from flax import nnx
import jax.numpy as jnp
from dataclasses import dataclass, field
import logging

logger = logging.getLogger()


@dataclass
class ProteinMPNNConfig:
    pdb_paths: list[str]
    out_dir: Path
    seed: int
    batch_size: int
    temperature: float
    fixed_residues_multi: dict[str, list[str]]
    redesigned_residues_multi: dict[str, list[str]]
    checkpoint_path: Path
    save_stats: bool = False
    atom_context_num: int = 1
    bias_AA: str = ""
    bias_AA_per_residue_multi: dict[str, dict] = field(default_factory=dict)
    omit_AA_per_residue_multi: dict[str, dict] = field(default_factory=dict)
    omit_AA: str = ""
    verbose: bool = False
    num_edges: int = 32
    noise_level: float = 0.3
    atom_context_num: int = 25

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
        if args.pdb_path_multi:
            with open(args.pdb_path_multi) as fh:
                pdb_paths = list(json.load(fh))
        else:
            pdb_paths = [args.pdb_path]

        if args.fixed_residues_multi:
            with open(args.fixed_residues_multi) as fh:
                fixed_residues_multi = {k: v.split() for k, v in json.load(fh).items()}
        else:
            fixed = args.fixed_residues.split()
            fixed_residues_multi = {pdb: fixed for pdb in pdb_paths}

        if args.redesigned_residues_multi:
            with open(args.redesigned_residues_multi) as fh:
                redesigned_residues_multi = {
                    k: v.split() for k, v in json.load(fh).items()
                }
        else:
            redesigned = args.redesigned_residues.split()
            redesigned_residues_multi = {pdb: redesigned for pdb in pdb_paths}

        checkpoint_path = Path(args.checkpoint_protein_mpnn)
        assert checkpoint_path.exists(), (
            f"The model path {checkpoint_path} does not exist."
        )

        if args.bias_AA_per_residue_multi:
            with open(args.bias_AA_per_residue_multi) as fh:
                bias_AA_per_residue_multi = json.load(fh)
        elif args.bias_AA_per_residue:
            with open(args.bias_AA_per_residue) as fh:
                per_residue = json.load(fh)
            bias_AA_per_residue_multi = {pdb: per_residue for pdb in pdb_paths}
        else:
            bias_AA_per_residue_multi = {}

        if args.omit_AA_per_residue_multi:
            with open(args.omit_AA_per_residue_multi) as fh:
                omit_AA_per_residue_multi = json.load(fh)
        elif args.omit_AA_per_residue:
            with open(args.omit_AA_per_residue) as fh:
                per_residue = json.load(fh)
            omit_AA_per_residue_multi = {pdb: per_residue for pdb in pdb_paths}
        else:
            omit_AA_per_residue_multi = {}

        return cls(
            pdb_paths=pdb_paths,
            out_dir=Path(args.out_folder),
            seed=args.seed,
            batch_size=args.batch_size,
            temperature=args.temperature,
            fixed_residues_multi=fixed_residues_multi,
            redesigned_residues_multi=redesigned_residues_multi,
            save_stats=args.save_stats,
            checkpoint_path=checkpoint_path,
            bias_AA=args.bias_AA or "",
            bias_AA_per_residue_multi=bias_AA_per_residue_multi,
            omit_AA_per_residue_multi=omit_AA_per_residue_multi,
            omit_AA=args.omit_AA or "",
            verbose=args.verbose,
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

    # setup seeds
    jax_key = jax.random.key(config.seed)
    rngs = nnx.Rngs(params=jax_key, dropout=jax_key)
    random.seed(config.seed)
    np.random.seed(config.seed)
    device = jax.devices()[0]

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=config.num_edges,
        device=device,
        atom_context_num=config.atom_context_num,
        model_type=args.model_type,
        rngs=rngs,
    )

    # checkpointer = ocp.StandardCheckpointer()
    # _, abstract_state = nnx.split(model)
    # state = checkpointer.restore(config.checkpoint_path.absolute(), abstract_state)
    # nnx.update(model, state)

    bias_AA = jnp.zeros([21], device=device, dtype=jnp.float32)
    if config.bias_AA:
        for item in config.bias_AA.split(","):
            aa, val = item.split(":")
            bias_AA = bias_AA.at[RESTYPE_STRTOINT[aa]].set(float(val))

    omit_AA = jnp.array([AA in config.omit_AA for AA in ALPHABET], dtype=jnp.float32)

    if len(args.parse_these_chains_only) != 0:
        parse_these_chains_only_list = args.parse_these_chains_only.split(",")
    else:
        parse_these_chains_only_list = []

    # loop over PDB paths
    for pdb in config.pdb_paths:
        if args.verbose:
            logger.info("Designing protein from this path:", pdb)

        fixed_residues = config.fixed_residues_multi[pdb]
        redesigned_residues = config.redesigned_residues_multi[pdb]

        protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
            pdb,
            device=device,
            chains=parse_these_chains_only_list,
            parse_all_atoms=False,
            parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy,
        )

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

        bias_AA_per_residue = jnp.zeros(
            [len(encoded_residues), 21], device=device, dtype=jnp.float32
        )
        if config.bias_AA_per_residue_multi:
            bias_dict = config.bias_AA_per_residue_multi[pdb]
            for residue_name, v1 in bias_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid, v2 in v1.items():
                        if amino_acid in alphabet:
                            j1 = restype_str_to_int[amino_acid]
                            bias_AA_per_residue[i1, j1] = v2

        omit_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )
        if config.omit_AA_per_residue_multi:
            omit_dict = config.omit_AA_per_residue_multi[pdb]
            for residue_name, v1 in omit_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid in v1:
                        if amino_acid in alphabet:
                            j1 = restype_str_to_int[amino_acid]
                            omit_AA_per_residue[i1, j1] = 1.0

        fixed_positions = torch.tensor(
            [int(item not in fixed_residues) for item in encoded_residues],
            device=device,
        )
        redesigned_positions = torch.tensor(
            [int(item not in redesigned_residues) for item in encoded_residues],
            device=device,
        )

        # specify which residues are buried for checkpoint_per_residue_label_membrane_mpnn model
        if args.transmembrane_buried:
            buried_residues = [item for item in args.transmembrane_buried.split()]
            buried_positions = torch.tensor(
                [int(item in buried_residues) for item in encoded_residues],
                device=device,
            )
        else:
            buried_positions = torch.zeros_like(fixed_positions)

        if args.transmembrane_interface:
            interface_residues = [item for item in args.transmembrane_interface.split()]
            interface_positions = torch.tensor(
                [int(item in interface_residues) for item in encoded_residues],
                device=device,
            )
        else:
            interface_positions = torch.zeros_like(fixed_positions)
        protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
            1 - interface_positions
        ) + 1 * interface_positions * (1 - buried_positions)

        if args.model_type == "global_label_membrane_mpnn":
            protein_dict["membrane_per_residue_labels"] = (
                args.global_transmembrane_label + 0 * fixed_positions
            )
        if len(args.chains_to_design) != 0:
            chains_to_design_list = args.chains_to_design.split(",")
        else:
            chains_to_design_list = protein_dict["chain_letters"]

        chain_mask = torch.tensor(
            np.array(
                [
                    item in chains_to_design_list
                    for item in protein_dict["chain_letters"]
                ],
                dtype=np.int32,
            ),
            device=device,
        )

        # create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
        if redesigned_residues:
            protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
        elif fixed_residues:
            protein_dict["chain_mask"] = chain_mask * fixed_positions
        else:
            protein_dict["chain_mask"] = chain_mask

        if args.verbose:
            PDB_residues_to_be_redesigned = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 1
            ]
            PDB_residues_to_be_fixed = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 0
            ]
            print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
            print("These residues will be fixed: ", PDB_residues_to_be_fixed)

        # specify which residues are linked
        if args.symmetry_residues:
            symmetry_residues_list_of_lists = [
                x.split(",") for x in args.symmetry_residues.split("|")
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
        if args.symmetry_weights:
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

        # set other atom bfactors to 0.0
        if other_atoms:
            other_bfactors = other_atoms.getBetas()
            other_atoms.setBetas(other_bfactors * 0.0)

        # adjust input PDB name by dropping .pdb if it does exist
        name = pdb[pdb.rfind("/") + 1 :]
        if name[-4:] == ".pdb":
            name = name[:-4]

        with torch.no_grad():
            # run featurize to remap R_idx and add batch dimension
            if args.verbose:
                if "Y" in list(protein_dict):
                    atom_coords = protein_dict["Y"].cpu().numpy()
                    atom_types = list(protein_dict["Y_t"].cpu().numpy())
                    atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                    number_of_atoms_parsed = np.sum(atom_mask)
                else:
                    print("No ligand atoms parsed")
                    number_of_atoms_parsed = 0
                    atom_types = ""
                    atom_coords = []
                if number_of_atoms_parsed == 0:
                    print("No ligand atoms parsed")
                elif args.model_type == "ligand_mpnn":
                    print(
                        f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed}"
                    )
                    for i, atom_type in enumerate(atom_types):
                        print(
                            f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}"
                        )
            feature_dict = featurize(
                protein_dict,
                cutoff_for_score=args.ligand_mpnn_cutoff_for_score,
                use_atom_context=args.ligand_mpnn_use_atom_context,
                number_of_ligand_atoms=atom_context_num,
                model_type=args.model_type,
            )
            feature_dict["batch_size"] = args.batch_size
            B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
            # add additional keys to the feature dictionary
            feature_dict["temperature"] = args.temperature
            feature_dict["bias"] = (
                (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                + bias_AA_per_residue[None]
                - 1e8 * omit_AA_per_residue[None]
            )
            feature_dict["symmetry_residues"] = remapped_symmetry_residues
            feature_dict["symmetry_weights"] = symmetry_weights

            sampling_probs_list = []
            log_probs_list = []
            decoding_order_list = []
            S_list = []
            loss_list = []
            loss_per_residue_list = []
            loss_XY_list = []
            for _ in range(args.number_of_batches):
                feature_dict["randn"] = torch.randn(
                    [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                    device=device,
                )
                output_dict = model.sample(feature_dict)

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
            S_stack = torch.cat(S_list, 0)
            log_probs_stack = torch.cat(log_probs_list, 0)
            sampling_probs_stack = torch.cat(sampling_probs_list, 0)
            decoding_order_stack = torch.cat(decoding_order_list, 0)
            loss_stack = torch.cat(loss_list, 0)
            loss_per_residue_stack = torch.cat(loss_per_residue_list, 0)
            loss_XY_stack = torch.cat(loss_XY_list, 0)
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

            output_fasta = base_folder + "/seqs/" + name + args.file_ending + ".fa"
            output_backbones = base_folder + "/backbones/"
            output_packed = base_folder + "/packed/"
            output_stats_path = base_folder + "stats/" + name + args.file_ending + ".pt"

            out_dict = {}
            out_dict["generated_sequences"] = S_stack.cpu()
            out_dict["sampling_probs"] = sampling_probs_stack.cpu()
            out_dict["log_probs"] = log_probs_stack.cpu()
            out_dict["decoding_order"] = decoding_order_stack.cpu()
            out_dict["native_sequence"] = feature_dict["S"][0].cpu()
            out_dict["mask"] = feature_dict["mask"][0].cpu()
            out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu()
            out_dict["seed"] = seed
            out_dict["temperature"] = args.temperature
            if args.save_stats:
                torch.save(out_dict, output_stats_path)

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
                    seq = "".join(
                        [restype_int_to_str[AA] for AA in S_stack[ix].cpu().numpy()]
                    )

                    # write new sequences into PDB with backbone coordinates
                    seq_prody = np.array([restype_1to3[AA] for AA in list(seq)])[
                        None,
                    ].repeat(4, 1)
                    bfactor_prody = (
                        loss_per_residue_stack[ix].cpu().numpy()[None, :].repeat(4, 1)
                    )
                    backbone.setResnames(seq_prody)
                    backbone.setBetas(
                        np.exp(-bfactor_prody)
                        * (bfactor_prody > 0.01).astype(np.float32)
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

                    # write full PDB files
                    if args.pack_side_chains:
                        for c_pack in range(args.number_of_packs_per_design):
                            X_stack = X_stack_list[c_pack]
                            X_m_stack = X_m_stack_list[c_pack]
                            b_factor_stack = b_factor_stack_list[c_pack]
                            write_full_PDB(
                                output_packed
                                + name
                                + args.packed_suffix
                                + "_"
                                + str(ix_suffix)
                                + "_"
                                + str(c_pack + 1)
                                + args.file_ending
                                + ".pdb",
                                X_stack[ix].cpu().numpy(),
                                X_m_stack[ix].cpu().numpy(),
                                b_factor_stack[ix].cpu().numpy(),
                                feature_dict["R_idx_original"][0].cpu().numpy(),
                                protein_dict["chain_letters"],
                                S_stack[ix].cpu().numpy(),
                                other_atoms=other_atoms,
                                icodes=icodes,
                                force_hetatm=args.force_hetatm,
                            )
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
