import numpy as np
from prody import writePDB, confProDy, AtomGroup, parsePDB, Selection
from .constants import (
    ELEMENT_LIST,
    RESTYPE_3TO1,
    RESTYPE_1TO3,
    restype_name_to_atom14_names,
    RESTYPE_STRTOINT,
    RESTYPE_INTTOSTR,
    ATOM_ORDER,
    AA_ATOM_NAMES,
    AA_BACKBONE_ATOM_NAMES,
)
import jax.numpy as jnp
from flax import nnx
from typing import TypedDict, Any, cast
import jax

confProDy(verbosity="none")


def get_seq_rec(S: jnp.ndarray, S_pred: jnp.ndarray, mask: jnp.ndarray):
    """
    S : true sequence shape=[batch, length]
    S_pred : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]

    average : averaged sequence recovery shape=[batch]
    """
    match = S == S_pred
    average = jnp.sum(match * mask, axis=-1) / jnp.sum(mask, axis=-1)
    return average


def get_score(
    S: jnp.ndarray,
    log_probs: jnp.ndarray,
    mask: jnp.ndarray,
    noise: float = 1e-8,
    num_letters: int = 21,
):
    """
    S : true sequence shape=[batch, length]
    log_probs : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]

    average_loss : averaged categorical cross entropy (CCE) [batch]
    loss_per_resdue : per position CCE [batch, length]
    """
    S_one_hot = nnx.one_hot(S, num_letters)
    loss_per_residue = -(S_one_hot * log_probs).sum(-1)  # [B, L]
    average_loss = jnp.sum(loss_per_residue * mask, axis=-1) / (
        jnp.sum(mask, axis=-1) + noise
    )
    return average_loss, loss_per_residue


def write_full_PDB(
    save_path: str,
    X: np.ndarray,
    X_m: np.ndarray,
    b_factors: np.ndarray,
    R_idx: np.ndarray,
    chain_letters: np.ndarray,
    S: np.ndarray,
    other_atoms=None,
    icodes=None,
    force_hetatm=False,
):
    """
    14 because the largest aa (TRP) has 14 atoms
    excess elements are empty strings in amino acid does not have 14 atoms

    save_path : path where the PDB will be written to
    X : protein atom xyz coordinates shape=[length, 14, 3]
    X_m : protein atom mask shape=[length, 14]
    b_factors: shape=[length, 14]
    R_idx: protein residue indices shape=[length]
    chain_letters: protein chain letters shape=[length]
    S : protein amino acid sequence shape=[length]
    other_atoms: other atoms parsed by prody
    icodes: a list of insertion codes for the PDB; e.g. antibody loops
    """

    # the sequence list
    S_str = [RESTYPE_1TO3[AA] for AA in [RESTYPE_INTTOSTR[AA] for AA in S]]

    X_list = []
    b_factor_list = []
    atom_name_list = []
    element_name_list = []
    residue_name_list = []
    residue_number_list = []
    chain_id_list = []
    icodes_list = []
    for i, AA in enumerate(S_str):
        sel = X_m[i].astype(np.int32) == 1
        total = np.sum(sel)
        tmp = np.array(restype_name_to_atom14_names[AA])[sel]
        X_list.append(X[i][sel])
        b_factor_list.append(b_factors[i][sel])
        atom_name_list.append(tmp)
        element_name_list += [AA[:1] for AA in list(tmp)]
        residue_name_list += total * [AA]
        residue_number_list += total * [R_idx[i]]
        chain_id_list += total * [chain_letters[i]]
        if icodes:
            icodes_list += total * [icodes[i]]

    X_stack = np.concatenate(X_list, 0)
    b_factor_stack = float(np.concatenate(b_factor_list, 0))
    atom_name_stack = np.concatenate(atom_name_list, 0)

    protein = AtomGroup()
    protein.setCoords(X_stack)
    protein.setBetas(b_factor_stack)  # type: ignore
    protein.setNames(atom_name_stack)  # type: ignore
    protein.setResnames(residue_name_list)  # type: ignore
    protein.setElements(element_name_list)  # type: ignore
    protein.setOccupancies(np.ones([X_stack.shape[0]]))  # type: ignore
    protein.setResnums(residue_number_list)  # type: ignore
    protein.setChids(chain_id_list)  # type: ignore
    protein.setIcodes(icodes_list)  # type: ignore

    if other_atoms:
        other_atoms_g = AtomGroup()
        other_atoms_g.setCoords(other_atoms.getCoords())  # type: ignore
        other_atoms_g.setNames(other_atoms.getNames())  # type: ignore
        other_atoms_g.setResnames(other_atoms.getResnames())  # type: ignore
        other_atoms_g.setElements(other_atoms.getElements())  # type: ignore
        other_atoms_g.setOccupancies(other_atoms.getOccupancies())  # type: ignore
        other_atoms_g.setResnums(other_atoms.getResnums())  # type: ignore
        other_atoms_g.setChids(other_atoms.getChids())  # type: ignore
        if force_hetatm:
            other_atoms_g.setFlags("hetatm", other_atoms.getFlags("hetatm"))
        writePDB(save_path, protein + other_atoms_g)
    else:
        writePDB(save_path, protein)


def get_aligned_coordinates(protein_atoms: AtomGroup, CA_dict: dict, atom_name: str):
    """
    protein_atoms: prody atom group
    CA_dict: mapping between chain_residue_idx_icodes and integers
    atom_name: atom to be parsed; e.g. CA
    """
    atom_atoms = protein_atoms.select(f"name {atom_name}")

    if atom_atoms is not None:
        atom_coords = atom_atoms.getCoords()
        atom_resnums = atom_atoms.getResnums()
        atom_chain_ids = atom_atoms.getChids()
        atom_icodes = atom_atoms.getIcodes()

    atom_coords_ = np.zeros([len(CA_dict), 3], np.float32)
    atom_coords_m = np.zeros([len(CA_dict)], np.int32)
    if atom_atoms is not None:
        for i in range(len(atom_resnums)):
            code = atom_chain_ids[i] + "_" + str(atom_resnums[i]) + "_" + atom_icodes[i]
            if code in list(CA_dict):
                atom_coords_[CA_dict[code], :] = atom_coords[i]
                atom_coords_m[CA_dict[code]] = 1
    return atom_coords_, atom_coords_m


class InputDataDict(TypedDict):
    X: jax.Array
    mask: jax.Array
    Y: jax.Array
    Y_t: jax.Array
    Y_m: jax.Array
    R_idx: jax.Array
    chain_labels: jax.Array
    chain_letters: list[str]
    mask_c: list[jax.Array]
    chain_list: list[str]
    S: jax.Array
    xyz_37: jax.Array
    xyz_37_m: jax.Array


def parse_PDB(
    input_path: str,
    device: str = "cpu",
    chains: list = [],
    parse_all_atoms: bool = False,
    parse_atoms_with_zero_occupancy: bool = False,
) -> tuple[InputDataDict, Selection, Selection, Any, Selection]:
    """
    input_path : path for the input PDB
    device: device for the torch.Tensor
    chains: a list specifying which chains need to be parsed; e.g. ["A", "B"]
    parse_all_atoms: if False parse only N,CA,C,O otherwise all 37 atoms
    parse_atoms_with_zero_occupancy: if True atoms with zero occupancy will be parsed
    """

    element_dict = dict(zip(ELEMENT_LIST, range(1, len(ELEMENT_LIST))))

    if not parse_all_atoms:
        atom_types = AA_BACKBONE_ATOM_NAMES
    else:
        atom_types = AA_ATOM_NAMES

    atoms = parsePDB(input_path)
    if not parse_atoms_with_zero_occupancy:
        atoms = atoms.select("occupancy > 0")  # type: ignore
    if chains:
        str_out = ""
        for item in chains:
            str_out += " chain " + item + " or"
        atoms = atoms.select(str_out[1:-3])  # type: ignore

    protein_atoms = atoms.select("protein")  # type: ignore
    backbone: Selection = protein_atoms.select("backbone")  # type: ignore
    other_atoms: Selection = atoms.select("not protein and not water")  # type: ignore
    water_atoms = atoms.select("water")  # type: ignore

    CA_atoms = protein_atoms.select("name CA")
    CA_resnums = CA_atoms.getResnums()
    CA_chain_ids = CA_atoms.getChids()
    CA_icodes = CA_atoms.getIcodes()

    CA_dict = {}
    for i in range(len(CA_resnums)):
        code = CA_chain_ids[i] + "_" + str(CA_resnums[i]) + "_" + CA_icodes[i]
        CA_dict[code] = i

    xyz_37 = np.zeros([len(CA_dict), 37, 3], np.float32)
    xyz_37_m = np.zeros([len(CA_dict), 37], np.int32)
    for atom_name in atom_types:
        xyz, xyz_m = get_aligned_coordinates(protein_atoms, CA_dict, atom_name)
        xyz_37[:, ATOM_ORDER[atom_name], :] = xyz
        xyz_37_m[:, ATOM_ORDER[atom_name]] = xyz_m

    N = xyz_37[:, ATOM_ORDER["N"], :]
    CA = xyz_37[:, ATOM_ORDER["CA"], :]
    C = xyz_37[:, ATOM_ORDER["C"], :]
    O = xyz_37[:, ATOM_ORDER["O"], :]

    N_m = xyz_37_m[:, ATOM_ORDER["N"]]
    CA_m = xyz_37_m[:, ATOM_ORDER["CA"]]
    C_m = xyz_37_m[:, ATOM_ORDER["C"]]
    O_m = xyz_37_m[:, ATOM_ORDER["O"]]

    mask = N_m * CA_m * C_m * O_m  # must all 4 atoms exist

    b = CA - N
    c = C - CA
    a = np.cross(b, c, axis=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

    chain_labels = np.array(CA_atoms.getChindices(), dtype=np.int32)
    R_idx = np.array(CA_resnums, dtype=np.int32)
    S = CA_atoms.getResnames()
    S = [RESTYPE_3TO1[AA] if AA in list(RESTYPE_3TO1) else "X" for AA in list(S)]
    S = np.array([RESTYPE_STRTOINT[AA] for AA in list(S)], np.int32)
    X = np.concatenate([N[:, None], CA[:, None], C[:, None], O[:, None]], 1)

    try:
        Y = np.array(other_atoms.getCoords(), dtype=np.float32)
        Y_t = list(other_atoms.getElements())  # type: ignore
        Y_t = np.array(
            [
                element_dict[y_t.upper()] if y_t.upper() in ELEMENT_LIST else 0
                for y_t in Y_t
            ],
            dtype=np.int32,
        )
        Y_m = (Y_t != 1) * (Y_t != 0)

        Y = Y[Y_m, :]
        Y_t = Y_t[Y_m]
        Y_m = Y_m[Y_m]
    except:
        Y = np.zeros([1, 3], np.float32)
        Y_t = np.zeros([1], np.int32)
        Y_m = np.zeros([1], np.int32)

    output_dict: dict[str, Any] = {}
    output_dict["X"] = jnp.array(X, device=device, dtype=jnp.float32)
    output_dict["mask"] = jnp.array(mask, device=device, dtype=jnp.int32)
    output_dict["Y"] = jnp.array(Y, device=device, dtype=jnp.float32)
    output_dict["Y_t"] = jnp.array(Y_t, device=device, dtype=jnp.int32)
    output_dict["Y_m"] = jnp.array(Y_m, device=device, dtype=jnp.int32)

    output_dict["R_idx"] = jnp.array(R_idx, device=device, dtype=jnp.int32)
    output_dict["chain_labels"] = jnp.array(
        chain_labels, device=device, dtype=jnp.int32
    )

    output_dict["chain_letters"] = CA_chain_ids

    # chain_list = ["A", "B", "C"]
    # mask_c[0]  = [True,  True,  False, False, False, False]  # chain A
    # mask_c[1]  = [False, False, True,  True,  True,  False]  # chain B
    # mask_c[2]  = [False, False, False, False, False, True ]  # chain C
    mask_c = []
    chain_list = sorted(list(set(CA_chain_ids)))
    for chain in chain_list:
        mask_c.append(
            jnp.array(
                [chain == item for item in CA_chain_ids],
                device=device,
                dtype=bool,
            )
        )

    output_dict["mask_c"] = mask_c
    output_dict["chain_list"] = chain_list

    output_dict["S"] = jnp.array(S, device=device, dtype=jnp.int32)

    output_dict["xyz_37"] = jnp.array(xyz_37, device=device, dtype=jnp.float32)
    output_dict["xyz_37_m"] = jnp.array(xyz_37_m, device=device, dtype=jnp.int32)

    return (
        cast(InputDataDict, output_dict),
        cast(Selection, backbone),
        cast(Selection, other_atoms),
        cast(Any, CA_icodes),
        cast(Selection, water_atoms),
    )


def get_nearest_neighbours(CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms):
    device = CB.device
    mask_CBY = mask[:, None] * Y_m[None, :]  # [A,B]
    L2_AB = jnp.sum((CB[:, None, :] - Y[None, :, :]) ** 2, -1)
    L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * 1000.0

    nn_idx = jnp.argsort(L2_AB, -1)[:, :number_of_ligand_atoms]
    L2_AB_nn = jnp.take_along_axis(L2_AB, nn_idx, axis=1)
    D_AB_closest = jnp.sqrt(L2_AB_nn[:, 0])

    Y_tmp = Y[nn_idx]  # [A, k, 3]
    Y_t_tmp = Y_t[nn_idx]  # [A, k]
    Y_m_tmp = Y_m[nn_idx]  # [A, k]

    Y_out = jnp.zeros(
        [CB.shape[0], number_of_ligand_atoms, 3], dtype=jnp.float32, device=device
    )
    Y_t_out = jnp.zeros(
        [CB.shape[0], number_of_ligand_atoms], dtype=jnp.int32, device=device
    )
    Y_m_out = jnp.zeros(
        [CB.shape[0], number_of_ligand_atoms], dtype=jnp.int32, device=device
    )

    num_nn_update = Y_tmp.shape[1]
    Y_out = Y_out.at[:, :num_nn_update].set(Y_tmp)
    Y_t_out = Y_t_out.at[:, :num_nn_update].set(Y_t_tmp)
    Y_m_out = Y_m_out.at[:, :num_nn_update].set(Y_m_tmp)

    return Y_out, Y_t_out, Y_m_out, D_AB_closest


def featurize(
    input_dict: InputDataDict,
    cutoff_for_score: float = 8.0,
    use_atom_context: bool = True,
    number_of_ligand_atoms: int = 16,
    model_type: str = "protein_mpnn",
):
    output_dict = {}
    if model_type == "ligand_mpnn":
        mask = input_dict["mask"]
        Y = input_dict["Y"]
        Y_t = input_dict["Y_t"]
        Y_m = input_dict["Y_m"]
        N = input_dict["X"][:, 0, :]
        CA = input_dict["X"][:, 1, :]
        C = input_dict["X"][:, 2, :]
        b = CA - N
        c = C - CA
        a = jnp.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        Y, Y_t, Y_m, D_XY = get_nearest_neighbours(
            CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms
        )
        mask_XY = (D_XY < cutoff_for_score) * mask * Y_m[:, 0]
        output_dict["mask_XY"] = mask_XY[None,]

        output_dict["Y"] = Y[None,]
        output_dict["Y_t"] = Y_t[None,]
        output_dict["Y_m"] = Y_m[None,]
        if not use_atom_context:
            output_dict["Y_m"] = 0.0 * output_dict["Y_m"]

    R_idx_list = []
    count = 0
    R_idx_prev = -100000
    for R_idx in list(input_dict["R_idx"]):
        if R_idx_prev == R_idx:
            count += 1
        R_idx_list.append(R_idx + count)
        R_idx_prev = R_idx
    R_idx_renumbered = jnp.array(R_idx_list, device=R_idx.device)
    output_dict["R_idx"] = R_idx_renumbered[None,]
    output_dict["R_idx_original"] = input_dict["R_idx"][None,]
    output_dict["chain_labels"] = input_dict["chain_labels"][None,]
    output_dict["S"] = input_dict["S"][None,]
    output_dict["mask"] = input_dict["mask"][None,]

    output_dict["X"] = input_dict["X"][None,]

    if "xyz_37" in list(input_dict):
        output_dict["xyz_37"] = input_dict["xyz_37"][None,]
        output_dict["xyz_37_m"] = input_dict["xyz_37_m"][None,]

    return output_dict
