import os

import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import itertools as it
from scattering.scattering import compute_van_hove, compute_partial_van_hove
from scattering.utils.utils import get_dt
from scattering.utils.features import find_local_maxima


def calc_partial_vhf(xtc,
                   gro,
                   md_chunk,
                   chunk_length=200,
                   skip=1,
                   r_range=(0, 0.8),
                   bin_width=0.001,
                   periodic=True, opt=True):
    """
    Wrapper around compute_van_hove that uses md.iterload to break up the trajec    trajectory into smaller chunks
    
    Parameters
    ----------
    trj: GROMACS XTC or TRR
    gro: GROMACS GRO file
    md_chunk: int
        Chunks of trajectory to loop over in md.iterload
    chunk_length: int, default=200
        Chunks of trajectory to compute VHF over
    skip: int, default=1
        Number of timesteps to skip
    """
    g_r_list = list()
    frame = md.load_frame(xtc, top=gro, index=1)
    names = list(set([i.name for i in frame.topology.atoms]))

    for elem1, elem2 in it.combinations_with_replacement(names[::-1], 2):
        for i, trj in enumerate(md.iterload(xtc, top=gro, chunk=md_chunk, skip=skip)):
            r, g_r_t = compute_partial_van_hove(trj=trj,
                                chunk_length=chunk_length,
                                selection1='name {}'.format(elem1),
                                selection2='name {}'.format(elem2),
                                r_range=r_range,
                                bin_width=bin_width,
                                periodic=periodic,
                                opt=opt)
            g_r_list.append(g_r_t)

        t = trj.time[:chunk_length]
        dt = get_dt(trj)
        g_r_t = np.mean(g_r_list, axis=0)

        # Save output to text files
        np.savetxt('vhf_{}_{}.txt'.format(elem1,elem2),
            g_r_t, header='# Van Hove Function, dt: {} fs, dr: {}'.format(
            dt,
            np.unique(np.round(np.diff(t), 6))[0],
        ))
        np.savetxt('r_{}_{}.txt'.format(elem1,elem2), r, header='# Times, ps')
        np.savetxt('t_{}_{}.txt'.format(elem1,elem2), t, header='# Positions, nm')
