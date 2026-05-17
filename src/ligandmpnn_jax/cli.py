import argparse
from .run_proteinmpnn import proteinmpnn_predict
import numpy as np


def add_common_args(parser: argparse.ArgumentParser) -> None:

    # select one or multiple PDB files to apply the models to
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pdb_path",
        type=str,
        help="Path to input PDB.",
    )
    group.add_argument(
        "--pdb_path_multi",
        type=str,
        help="Path to JSON listing PDB paths. {'/path/to/pdb': ''}",
    )

    parser.add_argument(
        "--out_folder",
        type=str,
        required=True,
        default="default",
        help="Output folder for sequences/backbones.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(np.random.randint(0, high=99999, size=1, dtype=int)[0]),
        help="RNG seed.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Sequences per forward pass."
    )
    parser.add_argument(
        "--number_of_batches", type=int, default=1, help="Number of batches to run."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Sampling temperature."
    )
    parser.add_argument("--verbose", action="store_true", default=False, help="Print progress information.")
    parser.add_argument(
        "--save_stats", type=int, default=0, help="Save output statistics."
    )
    parser.add_argument(
        "--fixed_residues",
        type=str,
        default="",
        help="Space-separated fixed residues, e.g. 'A12 A13 B2'.",
    )
    parser.add_argument(
        "--residues_to_design",
        type=str,
        default="",
        help="Space-separated residues to redesign; all others are fixed.",
    )
    parser.add_argument(
        "--bias_AA",
        type=str,
        default="",
        help="Per-AA generation bias, e.g. 'A:-1.0,P:2.3'.",
    )
    parser.add_argument(
        "--bias_AA_per_residue",
        type=str,
        default="",
        help="Path to per-residue bias JSON.",
    )
    parser.add_argument(
        "--bias_AA_per_residue_multi",
        type=str,
        default="",
        help="Path to per-PDB per-residue bias JSON.",
    )
    parser.add_argument(
        "--omit_AA", type=str, default="", help="AA letters to omit, e.g. 'ACG'."
    )
    parser.add_argument(
        "--omit_AA_per_residue",
        type=str,
        default="",
        help="Path to per-residue omit JSON.",
    )
    parser.add_argument(
        "--omit_AA_per_residue_multi",
        type=str,
        default="",
        help="Path to per-PDB per-residue omit JSON.",
    )
    parser.add_argument(
        "--chains_to_design",
        type=str,
        default="",
        help="Comma-separated chains to redesign, e.g. 'A,B'.",
    )
    parser.add_argument(
        "--parse_these_chains_only",
        type=str,
        default="",
        help="Comma-separated chains to parse, e.g. 'A,B'.",
    )
    parser.add_argument(
        "--symmetry_residues",
        type=str,
        default="",
        help="Linked residue groups, e.g. 'A12,A13|C2,C3'.",
    )
    parser.add_argument(
        "--symmetry_weights",
        type=str,
        default="",
        help="Weights matching --symmetry_residues, e.g. '1.0,1.0|-1.0,2.0'.",
    )
    parser.add_argument(
        "--homo_oligomer",
        type=int,
        default=0,
        help="1 - auto-set symmetry args for homooligomer design.",
    )
    parser.add_argument(
        "--file_ending",
        type=str,
        default="",
        help="Suffix appended to output filenames.",
    )
    parser.add_argument(
        "--zero_indexed",
        type=int,
        default=0,
        help="1 - start output PDB numbering from 0.",
    )
    parser.add_argument(
        "--fasta_seq_separation",
        type=str,
        default=":",
        help="Separator between chains in FASTA output.",
    )
    parser.add_argument(
        "--parse_atoms_with_zero_occupancy",
        type=int,
        default=0,
        help="1 - parse atoms with zero occupancy.",
    )
    parser.add_argument(
        "--ligand_mpnn_use_atom_context",
        type=int,
        default=1,
        help="1 - use ligand atom context in featurize, 0 - ignore it.",
    )
    parser.add_argument(
        "--ligand_mpnn_cutoff_for_score",
        type=float,
        default=8.0,
        help="Å cutoff between protein and context atoms for score reporting.",
    )
    parser.add_argument(
        "--pack_side_chains",
        type=int,
        default=0,
        help="1 - run side-chain packer after sequence design.",
    )
    parser.add_argument(
        "--checkpoint_path_sc",
        type=str,
        default="./model_params/ligandmpnn_sc_v_32_002_16.pt",
        help="Path to side-chain packer weights.",
    )
    parser.add_argument(
        "--number_of_packs_per_design",
        type=int,
        default=4,
        help="Independent side-chain packing samples per design.",
    )
    parser.add_argument(
        "--sc_num_denoising_steps",
        type=int,
        default=3,
        help="Packer denoising/recycling steps.",
    )
    parser.add_argument(
        "--sc_num_samples",
        type=int,
        default=16,
        help="Samples drawn from mixture distribution for packing.",
    )
    parser.add_argument(
        "--repack_everything",
        type=int,
        default=0,
        help="1 - repack all residues including fixed ones.",
    )
    parser.add_argument(
        "--force_hetatm",
        type=int,
        default=0,
        help="1 - write ligand atoms as HETATM after packing.",
    )
    parser.add_argument(
        "--packed_suffix",
        type=str,
        default="_packed",
        help="Suffix for packed PDB files.",
    )
    parser.add_argument(
        "--pack_with_ligand_context",
        type=int,
        default=1,
        help="1 - pack side chains using ligand context.",
    )


def build_protein_mpnn_parser(subparsers) -> None:
    """protein-mpnn: standard ProteinMPNN and SolubleMPNN."""
    parser = subparsers.add_parser(
        "protein-mpnn",
        help="Run ProteinMPNN or SolubleMPNN (no ligand/membrane context).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)

    parser.add_argument(
        "--model_type",
        type=str,
        default="protein_mpnn",
        choices=["protein_mpnn", "soluble_mpnn"],
        help="Model variant to use.",
    )
    parser.add_argument(
        "--checkpoint_protein_mpnn",
        type=str,
        default="./model_params/proteinmpnn_v_48_020.pt",
        help="Path to ProteinMPNN weights.",
    )
    parser.add_argument(
        "--checkpoint_soluble_mpnn",
        type=str,
        default="./model_params/solublempnn_v_48_020.pt",
        help="Path to SolubleMPNN weights.",
    )

    parser.set_defaults(func=run_protein_mpnn)


def build_ligand_mpnn_parser(subparsers) -> None:
    """ligand-mpnn: atomic-context-aware LigandMPNN."""
    parser = subparsers.add_parser(
        "ligand-mpnn",
        help="Run LigandMPNN with small-molecule / nucleotide / metal context.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)

    parser.set_defaults(model_type="ligand_mpnn")
    parser.add_argument(
        "--checkpoint_ligand_mpnn",
        type=str,
        default="./model_params/ligandmpnn_v_32_010_25.pt",
        help="Path to LigandMPNN weights.",
    )
    parser.add_argument(
        "--ligand_mpnn_use_side_chain_context",
        type=int,
        default=0,
        help="1 - use fixed-residue side chains as extra ligand context.",
    )

    parser.set_defaults(func=run_ligand_mpnn)


def build_membrane_mpnn_parser(subparsers) -> None:
    """membrane-mpnn: per-residue or global membrane label models."""
    parser = subparsers.add_parser(
        "membrane-mpnn",
        help="Run membrane ProteinMPNN (per-residue or global transmembrane label).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)

    parser.add_argument(
        "--model_type",
        type=str,
        default="per_residue_label_membrane_mpnn",
        choices=["per_residue_label_membrane_mpnn", "global_label_membrane_mpnn"],
        help="Membrane model variant.",
    )
    parser.add_argument(
        "--checkpoint_per_residue_label_membrane_mpnn",
        type=str,
        default="./model_params/per_residue_label_membrane_mpnn_v_48_020.pt",
        help="Path to per-residue membrane model weights.",
    )
    parser.add_argument(
        "--checkpoint_global_label_membrane_mpnn",
        type=str,
        default="./model_params/global_label_membrane_mpnn_v_48_020.pt",
        help="Path to global-label membrane model weights.",
    )
    parser.add_argument(
        "--transmembrane_buried",
        type=str,
        default="",
        help="Buried residues for per-residue model, e.g. 'A12 A13 B2'.",
    )
    parser.add_argument(
        "--transmembrane_interface",
        type=str,
        default="",
        help="Interface residues for per-residue model, e.g. 'A14 B25'.",
    )
    parser.add_argument(
        "--global_transmembrane_label",
        type=int,
        default=0,
        help="Global label for global-label model. 1 - transmembrane, 0 - soluble.",
    )

    parser.set_defaults(func=run_membrane_mpnn)


def run_protein_mpnn(args: argparse.Namespace):
    proteinmpnn_predict(args)


def run_ligand_mpnn(args: argparse.Namespace) -> None:
    pass


def run_membrane_mpnn(args: argparse.Namespace) -> None:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ligandmpnn",
        description="LigandMPNN-JAX: structure-based protein sequence design.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<model>")
    subparsers.required = True

    build_protein_mpnn_parser(subparsers)
    build_ligand_mpnn_parser(subparsers)
    build_membrane_mpnn_parser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
