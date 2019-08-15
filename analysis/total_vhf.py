import os

import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import mdtraj as md
from scattering.scattering import compute_van_hove, compute_van_hove_il
from scattering.utils.utils import get_dt
from scattering.utils.features import find_local_maxima

def calc_total_vhf(trj,
                   gro,
                   md_chunk,
                   vhf_txt='vhf.txt',
                   r_txt='r.txt',
                   t_txt='t.txt',
                   chunk_length=200,
                   skip=1,
                   water=False,
                   r_range=(0, 0.8),
                   bin_width=0.001,
                   periodic=True, op=True):
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
    for i, trj in enumerate(md.iterload(trj, top=gro, chunk=md_chunk, skip=skip)):
        r, t, g_r_t = compute_van_hove(trj=trj,
                            chunk_length=chunk_length,
                            water=water,
                            r_range=r_range,
                            bin_width=bin_width,
                            periodic=periodic,
                            opt=opt)
         g_r_list.append(g_r_t)

         if i == 0:
             global_t = t
             time = trj.time

    dt = get_dt(trj)
    g_r_t = np.mean(g_r_list, axis=0)

    # Save output to text files
    np.savetxt(vhf_txt,
        g_r_t, header='# Van Hove Function, dt: {} fs, dr: {}'.format(
        dt,
        np.unique(np.round(np.diff(time), 6))[0],
    ))
    np.savetxt(r_txt, r, header='# Times, ps')
    np.savetxt(t_txt, t, header='# Positions, nm')
