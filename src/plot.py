import gc
import logging
from pathlib import Path

import matplotlib
from matplotlib.collections import LineCollection
from tqdm import tqdm

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation

logger = logging.getLogger(__name__)


def plot_total_distance_graph(output, distance_list, frame_step: int = 1):

    total_distance_per_frame = np.sum(distance_list, axis=1)
    avg_distance = np.mean(total_distance_per_frame)

    num_frames = distance_list.shape[0]

    fig, ax = plt.subplots()
    x_vals = np.arange(num_frames) * frame_step
    y_vals = total_distance_per_frame

    ax.plot(x_vals, y_vals)

    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Cumulative Distance between residues (nm)")
    fig.suptitle("Total Distance between Heavy Atoms", fontsize=13, fontweight="bold")    
    ax.set_title(f"Average Cumulative Distance: {avg_distance:.3f} nm", fontsize=10, color="dimgray")
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")

    plt.close(fig)

    return Path(output)

def plot_displacement_heatmap(output, distance_list, residue_pairs, frame_step: int = 1, animate: bool = False):

    
    displacement_matrix = md.geometry.squareform(distance_list, residue_pairs)

    step_deltas = np.zeros_like(displacement_matrix)
    step_deltas[1:] = np.abs(np.diff(displacement_matrix, axis=0))


    cumulative_matrix = np.cumsum(step_deltas, axis=0)

    max_disp = np.max(cumulative_matrix) or 1.0     
    max_disp_per_frame = np.max(cumulative_matrix, axis=(1, 2))
    n_frames, n_res, _ = displacement_matrix.shape
    logger.debug("n_res: %s", n_res)

    final_frame_matrix = cumulative_matrix[-1]
    res_i, res_j = np.triu_indices(n_res, k=1)
    displacement_values = final_frame_matrix[res_i, res_j]

    valid_mask = displacement_values > 0
    
    if np.any(valid_mask):
        valid_res_i = res_i[valid_mask]
        valid_res_j = res_j[valid_mask]
        valid_displacements = displacement_values[valid_mask]

        sorted_indices = np.argsort(-valid_displacements)
        top_indices = sorted_indices[:10]

        logger.info("--- TOP 10 RESIDUE PAIRS BY CUMULATIVE DISPLACEMENT ---")
        for rank, idx in enumerate(top_indices, start=1):
            r1_idx = valid_res_i[idx]
            r2_idx = valid_res_j[idx]
            disp_nm = valid_displacements[idx]

            r1_name = f"Res {r1_idx + 1}"
            r2_name = f"Res {r2_idx + 1}"

            logger.info(f"{rank}: {r1_name} <-> {r2_name} | Total Disp: {disp_nm:.3f} nm")
        logger.info("-------------------------------------------------------")
    else:
        logger.info("--- NO RESIDUE DISPLACEMENT DETECTED ---")

    fig, ax = plt.subplots(figsize=(12,12))
    fig.suptitle("Residue Displacement", fontsize=16, fontweight="bold")

    initial_matrix = cumulative_matrix[0] if animate else final_frame_matrix

    im = ax.imshow(
        initial_matrix,
        cmap="coolwarm",
        vmin=0.0,
        vmax=max_disp,
        origin="lower",
        aspect="equal",
        interpolation="nearest"
    )

    cbar = fig.colorbar(im, ax=ax, label="Total Displacement (nm)")

    tick_step = max(1, int(np.ceil(n_res / 15)))
    
    if tick_step > 5:
        tick_step = max(5, round(tick_step / 5) * 5)

    ticks = np.arange(0, n_res, tick_step)
    tick_labels = ticks + 1

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(tick_labels, rotation=0) 

    if n_res <= 100:
        pixel_edges = np.arange(-0.5, n_res, 1)
        vlines = [[(x, -0.5), (x, n_res - 0.5)] for x in pixel_edges]
        hlines = [[(-0.5, y), (n_res - 0.5, y)] for y in pixel_edges]
        
        grid_collection = LineCollection(
            vlines + hlines,
            colors="#A0A0A0",
            linewidths=0.5,
            antialiaseds=False,
            zorder=2
        )
        ax.add_collection(grid_collection)

    ax.set_xlabel("Residue Index", fontweight="bold")
    ax.set_ylabel("Residue Index", fontweight="bold")

    base_path = Path(output)
    static_output = base_path.with_suffix(".png")    
    last_actual_t = (n_frames - 1) * frame_step
    final_title = f"Frame 0 -> Frame {last_actual_t} | Max Disp: {max_disp_per_frame[-1]:.3f} nm"


    if animate:
        video_output = base_path.with_suffix(".mp4")
        pbar = tqdm(total=n_frames, desc="Rendering Animation", unit="frame")
        title = ax.set_title("Frame 0 -> Frame 0 | Max Disp: 0.000 nm", fontweight="bold", fontsize=12)

        def update_frame(t: int):
            pbar.n = t + 1
            pbar.update(0)
            im.set_data(cumulative_matrix[t])
            actual_t = t * frame_step
            title.set_text(f"Frame 0 -> Frame {actual_t} | Max Disp: {max_disp_per_frame[t]:.3f} nm")
            return [im, title]

        plt.tight_layout()

        anim = FuncAnimation(
            
            fig,
            update_frame,
            frames=n_frames,
            interval=1000,
            blit=True,
        )
        anim.save(video_output, writer="ffmpeg", fps=10)
        pbar.close()

        im.set_data(final_frame_matrix)
        title.set_text(final_title)
        fig.savefig(static_output, dpi=300, bbox_inches="tight")

        plt.close(fig)
        gc.collect()
        return video_output, static_output

    else:
        ax.set_title(final_title, fontweight="bold", fontsize=12)
        plt.tight_layout()
        fig.savefig(static_output, dpi=300, bbox_inches="tight")


        plt.close(fig)
        gc.collect()
        return static_output


def print_hbonds(trajectory: md.Trajectory, freqfloat: float = 0.1, frame_step: int = 1):

    hbonds = md.baker_hubbard(trajectory, freq=freqfloat)
    label = lambda hbond : '%s -- %s' % (trajectory.topology.atom(hbond[0]), trajectory.topology.atom(hbond[2]))
    hbond_strings = [f"Hydrogen bonds present in > {freqfloat * 100:.1f}% of frames (freq > {freqfloat}):"]
    
    hbond_strings.extend([label(hbond) for hbond in hbonds])
    
    return "\n".join(hbond_strings)

def plot_hbonds_heatmap(output, trajectory: md.Trajectory, frame_step: int = 1) -> str:

    protein_indices = trajectory.topology.select("protein")
    trajectory = trajectory.atom_slice(protein_indices)
    
    if frame_step > 1:
        trajectory = trajectory[::frame_step]   
    all_hbonds = md.baker_hubbard(traj=trajectory, freq=0.0)

    donors = all_hbonds[:, 0]
    hydrogens = all_hbonds[:, 1]
    acceptors = all_hbonds[:, 2]

    ha_pairs = np.column_stack([hydrogens, acceptors])
    ha_distances = md.compute_distances(trajectory, ha_pairs)

    dha_triplets = np.column_stack([donors, hydrogens, acceptors])
    dha_angles = md.compute_angles(trajectory, dha_triplets)

    distance_mask = ha_distances <= 0.25
    angle_mask = dha_angles >= np.radians(120.0)

    hbond_presence_matrix = distance_mask & angle_mask

    da_pairs = np.column_stack([donors, acceptors])
    unique_da_pairs, inverse_idx = np.unique(da_pairs, axis=0, return_inverse=True)
    
    n_unique_da = len(unique_da_pairs)
    n_frames = hbond_presence_matrix.shape[0]
    da_presence_matrix = np.zeros((n_frames, n_unique_da), dtype=bool)
    
    for i in range(n_unique_da):
        da_presence_matrix[:, i] = np.any(hbond_presence_matrix[:, inverse_idx == i], axis=1)
        
    changes_per_frame = np.diff(da_presence_matrix, axis=0) != 0
    events_per_da = np.sum(changes_per_frame, axis=0)

    d_res = np.array([trajectory.topology.atom(pair[0]).residue.index for pair in unique_da_pairs], dtype=int)
    a_res = np.array([trajectory.topology.atom(pair[1]).residue.index for pair in unique_da_pairs], dtype=int)

    logger.debug(f"len(d_res)={len(d_res)}, len(a_res)={len(a_res)}")
    logger.debug(f"len(events_per_da)={len(events_per_da)}")

    top_10_indices = np.argsort(events_per_da)[-10:][::-1]
    
    logger.info("--- TOP 10 HYDROGEN BONDS BY TRANSITION EVENTS ---")
    for rank, idx in enumerate(top_10_indices, start=1):
        events = events_per_da[idx]
        
        d_atom = trajectory.topology.atom(donors[idx])
        a_atom = trajectory.topology.atom(acceptors[idx])
        
        d_name = f"{d_atom.residue.name}{d_atom.residue.resSeq}-{d_atom.name}"
        a_name = f"{a_atom.residue.name}{a_atom.residue.resSeq}-{a_atom.name}"
        
        logger.info(f"{rank}: {d_name} -> {a_name} | Events: {events}")
    logger.info("--------------------------------------------------")

    n_res = trajectory.topology.n_residues
    
    directed_matrix = np.zeros((n_res, n_res), dtype=float)
    np.add.at(directed_matrix, (d_res, a_res), events_per_da)
    
    logger.debug(f"directed_matrix max = {np.max(directed_matrix)}")
    
    event_matrix = directed_matrix + directed_matrix.T - np.diag(np.diag(directed_matrix))

    upper_tri = np.triu(event_matrix)
    
    flat_indices = np.argsort(upper_tri.ravel())[-10:][::-1]
    
    row_indices, col_indices = np.unravel_index(flat_indices, event_matrix.shape)

    valid_pairs = np.zeros((n_res, n_res), dtype=bool)
    valid_pairs[d_res, a_res] = True
    valid_pairs = valid_pairs | valid_pairs.T
    
    event_matrix[~valid_pairs] = np.nan
    
    logger.debug("--- TOP 10 RESIDUE PAIRS BY TOTAL HBOND TRANSITION EVENTS ---")
    for rank, (r_i, r_j) in enumerate(zip(row_indices, col_indices), start=1):
        total_events = upper_tri[r_i, r_j]
        
        if total_events > 0:
            res1 = trajectory.topology.residue(r_i)
            res2 = trajectory.topology.residue(r_j)
            
            res1_name = f"{res1.name}{res1.resSeq}"
            res2_name = f"{res2.name}{res2.resSeq}"
            
            logger.debug(f"{rank}: {res1_name} <-> {res2_name} | Total Events: {total_events}")
    logger.debug("-------------------------------------------------------")

    fig, ax = plt.subplots(figsize=(12,12))
    fig.suptitle("Hydrogen Bond Dynamics", fontsize=16, fontweight="bold")

    # cmap = plt.get_cmap("coolwarm").copy()
    # cmap.set_bad(color="lightgray")
    cmap = plt.get_cmap("viridis").with_extremes(bad="lightgray")

    im = ax.imshow(
        event_matrix,
        cmap=cmap,
        origin="lower",
        aspect="equal",
    )
    
    cbar = fig.colorbar(im, ax=ax, label="Total Transition Events")

    # ticks = np.arange(0, n_res, 5)
    # tick_labels = ticks + 1

    # ax.set_xticks(ticks)
    # ax.set_yticks(ticks)
    # ax.set_xticklabels(tick_labels, rotation=0)
    # ax.set_yticklabels(tick_labels, rotation=0)

    # pixel_edges = np.arange(-0.5, n_res, 1)
    # ax.set_xticks(pixel_edges, minor=True)
    # ax.set_yticks(pixel_edges, minor=True)

    # ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    # ax.tick_params(which="minor", bottom=False, left=False)

    tick_step = max(1, int(np.ceil(n_res / 15)))
    
    if tick_step > 5:
        tick_step = max(5, round(tick_step / 5) * 5)

    ticks = np.arange(0, n_res, tick_step)
    tick_labels = ticks + 1

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(tick_labels, rotation=0) 

    if n_res <= 100:
        pixel_edges = np.arange(-0.5, n_res, 1)
        vlines = [[(x, -0.5), (x, n_res - 0.5)] for x in pixel_edges]
        hlines = [[(-0.5, y), (n_res - 0.5, y)] for y in pixel_edges]
        
        grid_collection = LineCollection(
            vlines + hlines,
            colors="#A0A0A0",
            linewidths=0.5,
            antialiaseds=False,
            zorder=2
        )
        ax.add_collection(grid_collection)

    ax.set_xlabel("Residue Index", fontweight="bold")
    ax.set_ylabel("Residue Index", fontweight="bold")

    plt.tight_layout()
    fig.savefig(output, bbox_inches="tight")

    plt.close(fig)
    gc.collect()

    return Path(output)


def plot_hbonds_occupancy_heatmap(
    output: str,
    trajectory: md.Trajectory,
    frame_step: int = 1,
    as_fraction: bool = True,
) -> Path:

    protein_indices = trajectory.topology.select("protein")
    trajectory = trajectory.atom_slice(protein_indices)

    if frame_step > 1:
        trajectory = trajectory[::frame_step]

    n_frames = trajectory.n_frames
    n_res = trajectory.topology.n_residues

    all_hbonds = md.baker_hubbard(traj=trajectory, freq=0.0)

    if len(all_hbonds) == 0:
        logger.warning("No hydrogen bonds detected in trajectory.")
        return Path(output)

    donors = all_hbonds[:, 0]
    hydrogens = all_hbonds[:, 1]
    acceptors = all_hbonds[:, 2]

    ha_pairs = np.column_stack([hydrogens, acceptors])
    ha_distances = md.compute_distances(trajectory, ha_pairs)

    dha_triplets = np.column_stack([donors, hydrogens, acceptors])
    dha_angles = md.compute_angles(trajectory, dha_triplets)

    distance_mask = ha_distances <= 0.25
    angle_mask = dha_angles >= np.radians(120.0)
    hbond_presence_matrix = distance_mask & angle_mask

    da_pairs = np.column_stack([donors, acceptors])
    unique_da_pairs, inverse_idx = np.unique(
        da_pairs, axis=0, return_inverse=True
    )

    n_unique_da = len(unique_da_pairs)
    da_presence_matrix = np.zeros((n_frames, n_unique_da), dtype=bool)

    for i in range(n_unique_da):
        da_presence_matrix[:, i] = np.any(
            hbond_presence_matrix[:, inverse_idx == i], axis=1
        )

    active_frames_per_da = np.sum(da_presence_matrix, axis=0)

    d_res = np.array(
        [
            trajectory.topology.atom(pair[0]).residue.index
            for pair in unique_da_pairs
        ],
        dtype=int,
    )
    a_res = np.array(
        [
            trajectory.topology.atom(pair[1]).residue.index
            for pair in unique_da_pairs
        ],
        dtype=int,
    )

    top_10_indices = np.argsort(active_frames_per_da)[-10:][::-1]

    logger.info("--- TOP 10 HYDROGEN BONDS BY ACTIVE FRAMES ---")
    for rank, idx in enumerate(top_10_indices, start=1):
        frames_active = active_frames_per_da[idx]
        pct = (frames_active / n_frames) * 100

        d_atom = trajectory.topology.atom(unique_da_pairs[idx, 0])
        a_atom = trajectory.topology.atom(unique_da_pairs[idx, 1])

        d_name = f"{d_atom.residue.name}{d_atom.residue.resSeq}-{d_atom.name}"
        a_name = f"{a_atom.residue.name}{a_atom.residue.resSeq}-{a_atom.name}"

        logger.info(
            f"{rank}: {d_name} -> {a_name} | Active: {frames_active}/{n_frames} frames ({pct:.1f}%)"
        )
    logger.info("--------------------------------------------------")

    directed_matrix = np.zeros((n_res, n_res), dtype=float)
    np.add.at(directed_matrix, (d_res, a_res), active_frames_per_da)

    occupancy_matrix = (
        directed_matrix
        + directed_matrix.T
        - np.diag(np.diag(directed_matrix))
    )

    if as_fraction:
        occupancy_matrix = occupancy_matrix / n_frames

    valid_pairs = np.zeros((n_res, n_res), dtype=bool)
    valid_pairs[d_res, a_res] = True
    valid_pairs = valid_pairs | valid_pairs.T

    occupancy_matrix[~valid_pairs] = np.nan

    fig, ax = plt.subplots(figsize=(12, 12))
    title_str = (
        "Hydrogen Bond Occupancy (Fraction)"
        if as_fraction
        else "Hydrogen Bond Active Frames"
    )
    fig.suptitle(title_str, fontsize=16, fontweight="bold")

    cmap = plt.get_cmap("viridis").with_extremes(bad="lightgray")

    vmax_val = 1.0 if as_fraction else n_frames
    im = ax.imshow(
        occupancy_matrix,
        cmap=cmap,
        origin="lower",
        aspect="equal",
        vmin=0.0,
        vmax=vmax_val,
    )

    cbar_label = (
        "Occupancy Fraction" if as_fraction else "Total Active Frames"
    )
    fig.colorbar(im, ax=ax, label=cbar_label)

    tick_step = max(1, int(np.ceil(n_res / 15)))
    if tick_step > 5:
        tick_step = max(5, round(tick_step / 5) * 5)

    ticks = np.arange(0, n_res, tick_step)
    tick_labels = ticks + 1

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(
        tick_labels, rotation=45, ha="right", rotation_mode="anchor"
    )
    ax.set_yticklabels(tick_labels, rotation=0)

    if n_res <= 100:
        pixel_edges = np.arange(-0.5, n_res, 1)
        vlines = [[(x, -0.5), (x, n_res - 0.5)] for x in pixel_edges]
        hlines = [[(-0.5, y), (n_res - 0.5, y)] for y in pixel_edges]

        grid_collection = LineCollection(
            vlines + hlines,
            colors="#A0A0A0",
            linewidths=0.5,
            antialiaseds=False,
            zorder=2,
        )
        ax.add_collection(grid_collection)

    ax.set_xlabel("Residue Index", fontweight="bold")
    ax.set_ylabel("Residue Index", fontweight="bold")

    plt.tight_layout()
    output_path = Path(output)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)

    plt.close(fig)
    gc.collect()

    return output_path


def compute_salt_bridge_transitions(trajectory, distance_cutoff=0.4):
    top = trajectory.topology
    n_res = top.n_residues

    acidic_oxygens = top.select(
        "(resname ASP and name OD1 OD2) or "
        "(resname GLU and name OE1 OE2) or "
        "(resname CYM and name SG)"
    )
    basic_nitrogens = top.select(
        "(resname LYS and name NZ) or "
        "(resname ARG and name NH1 NH2 NE) or "
        "(resname HIP and name ND1 NE2)"
    )

    if len(acidic_oxygens) == 0 or len(basic_nitrogens) == 0:
        return np.full((n_res, n_res), np.nan)

    O_idx, N_idx = np.meshgrid(acidic_oxygens, basic_nitrogens, indexing='ij')
    atom_pairs = np.column_stack((O_idx.ravel(), N_idx.ravel()))

    atom_to_res = np.array([atom.residue.index for atom in top.atoms])
    res_pairs = atom_to_res[atom_pairs]

    valid_mask = res_pairs[:, 0] != res_pairs[:, 1]
    atom_pairs = atom_pairs[valid_mask]
    res_pairs = res_pairs[valid_mask]

    if len(atom_pairs) == 0:
        return np.full((n_res, n_res), np.nan)

    unique_res_pairs, inverse_indices = np.unique(res_pairs, axis=0, return_inverse=True)
    
    sort_idx = np.argsort(inverse_indices)
    sorted_inverse = inverse_indices[sort_idx]
    sorted_atom_pairs = atom_pairs[sort_idx]
    
    boundaries = np.searchsorted(sorted_inverse, np.arange(len(unique_res_pairs)))

    distances = md.compute_distances(trajectory, sorted_atom_pairs)
    is_bonded_atom = distances <= distance_cutoff

    res_pair_bonded = np.bitwise_or.reduceat(is_bonded_atom, boundaries, axis=1)

    transitions = (np.diff(res_pair_bonded.astype(np.int8), axis=0) != 0).sum(axis=0)


    target_resSeq_a = 12
    target_resSeq_b = 14

    idx_a = next((res.index for res in top.residues if res.resSeq == target_resSeq_a), None)
    idx_b = next((res.index for res in top.residues if res.resSeq == target_resSeq_b), None)

    if idx_a is not None and idx_b is not None:
        pair_mask = (
            ((unique_res_pairs[:, 0] == idx_a) & (unique_res_pairs[:, 1] == idx_b)) |
            ((unique_res_pairs[:, 0] == idx_b) & (unique_res_pairs[:, 1] == idx_a))
        )
        
        if np.any(pair_mask):
            pair_idx = np.where(pair_mask)[0][0]
            transition_frames = np.where(np.diff(res_pair_bonded[:, pair_idx].astype(np.int8)) != 0)[0] + 1
            logger.info(f"Transitions for {top.residue(idx_a)} <-> {top.residue(idx_b)} occurred at frames: {transition_frames.tolist()}")
        else:
            logger.info(f"No salt bridge recorded between resSeq {target_resSeq_a} and {target_resSeq_b}.")
    else:
        logger.warning(f"Could not find residues with resSeq {target_resSeq_a} or {target_resSeq_b} in topology.")

    ever_bonded = res_pair_bonded.any(axis=0)

    raw_matrix = np.zeros((n_res, n_res))
    valid_pairs_mask = np.zeros((n_res, n_res), dtype=bool)

    r1 = unique_res_pairs[:, 0]
    r2 = unique_res_pairs[:, 1]
    
    raw_matrix[r1, r2] = transitions
    valid_pairs_mask[r1, r2] = ever_bonded
    event_matrix = raw_matrix + raw_matrix.T
    valid_pairs = valid_pairs_mask | valid_pairs_mask.T
    event_matrix[~valid_pairs] = np.nan

    mask = np.triu(~np.isnan(event_matrix), k=1)
    res_i, res_j = np.where(mask)
    
    if len(res_i) > 0:
        events = event_matrix[res_i, res_j]
        
        sorted_indices = np.argsort(-events)
        logger.debug("--- ALL SALT BRIDGES BY TRANSITION EVENTS ---")
        for rank, idx in enumerate(sorted_indices, start=1):
            r1_idx = res_i[idx]
            r2_idx = res_j[idx]
            n_events = int(events[idx])
            
            res1 = top.residue(r1_idx)
            res2 = top.residue(r2_idx)
            
            r1_name = f"{res1.name}{res1.resSeq}"
            r2_name = f"{res2.name}{res2.resSeq}"
            
            logger.debug(f"{rank}: {r1_name} <-> {r2_name} | Events: {n_events}")
        logger.debug("--------------------------------------------------")

        top_indices = sorted_indices[:10]
        
        logger.info("--- TOP 10 SALT BRIDGES BY TRANSITION EVENTS ---")
        for rank, idx in enumerate(top_indices, start=1):
            r1_idx = res_i[idx]
            r2_idx = res_j[idx]
            n_events = int(events[idx])
            
            res1 = top.residue(r1_idx)
            res2 = top.residue(r2_idx)
            
            r1_name = f"{res1.name}{res1.resSeq}"
            r2_name = f"{res2.name}{res2.resSeq}"
            
            logger.info(f"{rank}: {r1_name} <-> {r2_name} | Events: {n_events}")
        logger.info("--------------------------------------------------")
    else:
        logger.info("--- NO SALT BRIDGE TRANSITIONS FOUND ---")

    return event_matrix

def plot_salt_bridge_transitions(trajectory, output, frame_step=1, distance_cutoff=0.4):

    protein_indices = trajectory.topology.select("protein")
    trajectory = trajectory.atom_slice(protein_indices)
    
    if frame_step > 1:
        trajectory = trajectory[::frame_step]
    event_matrix = compute_salt_bridge_transitions(trajectory, distance_cutoff)
    n_res = trajectory.topology.n_residues

    fig, ax = plt.subplots(figsize=(12,12))
    fig.suptitle("Salt Bridge Dynamics", fontsize=16, fontweight="bold")

    # cmap = plt.get_cmap("coolwarm").copy()
    # cmap.set_bad(color="lightgray")
    cmap = plt.get_cmap("plasma").with_extremes(bad="lightgray")


    im = ax.imshow(
        event_matrix,
        cmap=cmap,
        origin="lower",
        aspect="equal",
    )
    
    cbar = fig.colorbar(im, ax=ax, label="Total Transition Events")

    ticks = np.arange(0, n_res, 5)
    tick_labels = ticks + 1

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_yticklabels(tick_labels, rotation=0)

    # pixel_edges = np.arange(-0.5, n_res, 1)
    # ax.set_xticks(pixel_edges, minor=True)
    # ax.set_yticks(pixel_edges, minor=True)

    # ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    # ax.tick_params(which="minor", bottom=False, left=False)

    tick_step = max(1, int(np.ceil(n_res / 15)))
    
    if tick_step > 5:
        tick_step = max(5, round(tick_step / 5) * 5)

    ticks = np.arange(0, n_res, tick_step)
    tick_labels = ticks + 1

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(tick_labels, rotation=0) 

    if n_res <= 100:
        pixel_edges = np.arange(-0.5, n_res, 1)
        vlines = [[(x, -0.5), (x, n_res - 0.5)] for x in pixel_edges]
        hlines = [[(-0.5, y), (n_res - 0.5, y)] for y in pixel_edges]
        
        grid_collection = LineCollection(
            vlines + hlines,
            colors="#A0A0A0",
            linewidths=0.5,
            antialiaseds=False,
            zorder=2
        )
        ax.add_collection(grid_collection)

    ax.set_xlabel("Residue Index", fontweight="bold")
    ax.set_ylabel("Residue Index", fontweight="bold")

    plt.tight_layout()
    fig.savefig(output, bbox_inches="tight")

    plt.close(fig)
    gc.collect()

    return Path(output)