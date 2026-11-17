import argparse
import logging
from datetime import datetime
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd

import src.plot as plot
import src.setup_logging as log
import src.traj_parser as traj_parser

#input handling

parser = argparse.ArgumentParser()

parser.add_argument("-f", "--input_file", help="Path to Topology (.gro) File", required=True)
parser.add_argument("-t", "--traj_file", help="Path to Trajectory (.xtc) File", required=True)
parser.add_argument("-o", "--output", help="Name of Output Folder", default="")
parser.add_argument("-s", "--step", help="Number of frames to skip between computing steps", default=1)
parser.add_argument("-a", "--animate", action="store_true", help="Render Animations")
args = parser.parse_args()


#housekeeping
BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output" / args.output
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = BASE_DIR / "logs"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

def banner(logger:logging.Logger, args, start_time):

    logger.info("-" * 70)
    logger.info("I hope this works ")
    logger.info("-" * 70)
    logger.info("Started at %s", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Input Structure  : %s", args.input_file)
    logger.info("Input Trajectory : %s", args.traj_file)
    logger.info("Output Directory : output/%s", args.output)
    logger.info("-" * 70)


def main():
    start_time = datetime.now()
            
    log.setup_logging(output_dir=OUTPUT_DIR)
    logger = logging.getLogger(__name__)

    banner(logger=logger, args=args, start_time=start_time)

    #load trajectory
    logger.info("Start!")

    top = md.load_topology(args.input_file)
    keep_indices = top.select("protein or resname CY1 ZN1")

    trajectory = md.load(args.traj_file, top=top, atom_indices=keep_indices, stride=args.step)
    logger.debug(trajectory)

    #distance between closest heavy atoms per residue
    distance_list, residue_pairs = traj_parser.compute_residue_distance_heavy(traj=trajectory)


    unique_residues = np.unique(residue_pairs)
    num_residues_counted = len(unique_residues)

    logger.debug("Total contact pairs evaluated: %s", residue_pairs.shape[0])
    logger.debug("Number of unique residues involved in contacts: %s", num_residues_counted)
    logger.debug("Residue index list: %s", unique_residues)

    #create graph on total distance between heavy atoms per residue
    logger.info("Writing total_distance_graph")
    plot.plot_total_distance_graph(output=(OUTPUT_DIR / "total_distance.png"),distance_list=distance_list, frame_step=args.step)

    logger.info("Writing displacement_heatmap")
    plot.plot_displacement_heatmap(output=(OUTPUT_DIR / "total_displacement_heatmap.mp4"), distance_list=distance_list, residue_pairs=residue_pairs, frame_step=args.step, animate=args.animate)


    #plot hbond heatmap
    logger.info("Writing hbond_heatmap")
    logger.debug(plot.print_hbonds(trajectory=trajectory, frame_step=1))

    plot.plot_hbonds_occupancy_heatmap(output= (OUTPUT_DIR / "hbond_occupancy.png"), trajectory= trajectory, frame_step = args.step, as_fraction=True)

    plot.plot_hbonds_heatmap(trajectory=trajectory, frame_step=1, output=OUTPUT_DIR / "hbonds.png")

    #plot salt bridge heatmap
    logger.info("Writing salt_bridge_heatmap")
    plot.plot_salt_bridge_transitions(trajectory=trajectory, frame_step=1, output=OUTPUT_DIR / "salt_bridge.png")

    plot.plot_salt_bridge_occupancy_heatmap(output= (OUTPUT_DIR / "salt_bridge_occupancy.png"), trajectory=trajectory, frame_step=args.step, as_fraction=True)



    #ending
    end_time = datetime.now()
    logger.info("Finished on %s", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    hours, remainder = divmod((end_time-start_time).seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info('Total elapsed time {:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds)))

if __name__ == "__main__":
    main()