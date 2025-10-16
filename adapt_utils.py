import numpy as np
import copy
import multiprocessing as mp
import os
import itertools

from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value, construct_ups_state
from slowquant.unitary_coupled_cluster.util import iterate_t1, iterate_t2, iterate_t1_generalized, iterate_t2_generalized
from slowquant.unitary_coupled_cluster.operators import G1, G2


def pool(WF, so_ir, generalized):
    '''
    Defines the excitation pool and implements symmetry filter.

    Arguments
        WF: SlowQuant wave function object
        so_ir: list of irreducible representation labels for each spin orbital
        generalized: generalized pool or not?->Bool

    Returns
        pool_data: pool data with symmetry allowed excitations
    '''
    pool_data = {
    "excitation indeces": [],
    "excitation type": [],
    "excitation operator": []
    }

    ## GENERALIZED vs P-H excitation pool -> first layer always P-H since HF is always reference
    if generalized == True:

        ## Generate indeces for singly-excited operators
        for a, i in iterate_t1_generalized(WF.num_spin_orbs):
            pool_data["excitation indeces"].append((i, a))            
            pool_data["excitation type"].append("single")
            pool_data["excitation operator"].append(G1(i, a, True))

        ## Generate indeces for doubly-excited operators
        for a, i, b, j in iterate_t2_generalized(WF.num_spin_orbs):
            pool_data["excitation indeces"].append((i, j, a, b))
            pool_data["excitation type"].append("double")
            pool_data["excitation operator"].append(G2(i, j, a, b, True))

    else:

        ## Generate indeces for singly-excited operators
        for a, i in iterate_t1(WF.active_occ_spin_idx, WF.active_unocc_spin_idx):
            pool_data["excitation indeces"].append((i, a))            
            pool_data["excitation type"].append("single")
            pool_data["excitation operator"].append(G1(i, a, True))

        ## Generate indeces for doubly-excited operators
        for a, i, b, j in iterate_t2(WF.active_occ_spin_idx, WF.active_unocc_spin_idx):
            pool_data["excitation indeces"].append((i, j, a, b))
            pool_data["excitation type"].append("double")
            pool_data["excitation operator"].append(G2(i, j, a, b, True))

    # Pruning away symmetry-forbidden excitations

    for num, excitation in enumerate(pool_data["excitation indeces"]):

        if len(excitation) == 2:

            i, a = excitation

            if so_ir[i] != so_ir[a]:
                del pool_data["excitation indeces"][num]
                del pool_data["excitation type"][num]
                del pool_data["excitation operator"][num]

        if len(excitation) == 4:

            i, j, a, b = excitation

            if (so_ir[i] != so_ir[a] != so_ir[j] != so_ir[b]
                or so_ir[i] != so_ir[j] and so_ir[a] != so_ir[b]
                or so_ir[i] != so_ir[a] and so_ir[j] != so_ir[b]
                or so_ir[i] != so_ir[b] and so_ir[j] != so_ir[a]):

                del pool_data["excitation indeces"][num]
                del pool_data["excitation type"][num]
                del pool_data["excitation operator"][num]

    return pool_data

def gradient_evaluator(WF, H_ket, T):
    '''
    Calculate gradient - analytical evaluation

    Arguments
        WF: SlowQuant wavefunction object
        H_ket: propagated states
        T: excitation operator

    Returns
        gr: gradient
    ''' 
    gr = expectation_value(WF.ci_coeffs, [T], H_ket,
                            WF.ci_info, WF.thetas, WF.ups_layout)
    gr -= expectation_value(H_ket, [T], WF.ci_coeffs,
                        WF.ci_info, WF.thetas, WF.ups_layout)
        
    return gr


def gradient_parallel(WF, H_ket, pool_data):
    '''
    Parallelizes gradient calculation over the pool

    Arguments
        WF: SlowQuant wavefunction object
        H_ket: propagated states
        T: excitation operator

    Returns
        results: list of gradients over the pool
    '''
    with mp.Pool(processes=os.cpu_count()) as pool:
        results = pool.starmap(gradient_evaluator, [(WF, H_ket, T) for T in pool_data["excitation operator"]])

    return results