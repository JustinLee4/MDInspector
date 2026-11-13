# import MDAnalysis as mda
import mdtraj as md


def compute_residue_distance_heavy(traj: md.Trajectory):
    distance_matrix = md.compute_contacts(traj=traj, contacts='all', scheme='closest-heavy')
    return distance_matrix