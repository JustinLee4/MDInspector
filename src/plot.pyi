import logging
from pathlib import Path

import matplotlib
from tqdm import tqdm

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation

def plot_total_distance_graph(output, distance_list, frame_step: int = 1) -> Path: ...

def plot_displacement_heatmap(output, distance_list, residue_pairs, frame_step: int = 1, animate: bool = False): ...

def print_hbonds(trajectory: md.Trajectory, freqfloat: float = 0.1, frame_step: int = 1) -> Path: ...

def plot_hbonds_heatmap(output, trajectory: md.Trajectory, frame_step: int = 1) -> Path: ...

def plot_hbonds_occupancy_heatmap(output: str, trajectory: md.Trajectory, frame_step: int = 1, as_fraction: bool = True) -> Path: ...

def compute_salt_bridge_transitions(trajectory, distance_cutoff=0.4): ...

def plot_salt_bridge_transitions(event_matrix, n_res, output): ...

def compute_salt_bridge_occupancy(trajectory: md.Trajectory, distance_cutoff: float = 0.4, as_fraction: bool = True) -> np.ndarray: ...

def plot_salt_bridge_occupancy_heatmap(trajectory: md.Trajectory, output: str, frame_step: int = 1, distance_cutoff: float = 0.4, as_fraction: bool = True) -> Path:...

