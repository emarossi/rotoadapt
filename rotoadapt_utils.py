import numpy as np
import copy
import multiprocessing as mp
import os
import itertools

from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value, construct_ups_state
from slowquant.unitary_coupled_cluster.util import iterate_t1, iterate_t2, iterate_t1_generalized, iterate_t2_generalized
from slowquant.unitary_coupled_cluster.operators import G1, G2

# Modules for Hamiltonian construction
from slowquant.molecularintegrals.integralfunctions import one_electron_integral_transform, two_electron_integral_transform
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import commutator
from slowquant.unitary_coupled_cluster.operators import hamiltonian_full_space

# Qiskit utils to get number of measurements
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

## POOL DEFINITION

def pool_SD(WF, so_ir, generalized, efficient):
    '''
    Defines the excitation pool.

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
    "excitation operator": [],
    "H fragments": [],
    "N_Paulis_H1": [],
    "N_Paulis_H4": [],
    "N_Paulis_H0": [],
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

    if efficient == True:

        print('SPLITTING HAMILTONIAN....')

        #define mapper
        mapper = JordanWignerMapper()

        for A in pool_data["excitation operator"]:
            
            H1, H4, H0 = H_split(WF.num_orbs, WF.h_mo, WF.g_mo, A)

            pool_data["N_Paulis_H1"].append(len(mapper.map(FermionicOp(H1.get_qiskit_form(WF.num_orbs), WF.num_spin_orbs)).paulis))
            pool_data["N_Paulis_H4"].append(len(mapper.map(FermionicOp(H4.get_qiskit_form(WF.num_orbs), WF.num_spin_orbs)).paulis))
            pool_data["N_Paulis_H0"].append(len(mapper.map(FermionicOp(H0.get_qiskit_form(WF.num_orbs), WF.num_spin_orbs)).paulis))
            pool_data["H fragments"].append((H1, H4, H0))


    return pool_data

def pool_D(WF, so_ir, generalized, efficient):
    '''
    Defines the excitation pool.

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
    "excitation operator": [],
    "H fragments": [],
    "N_Paulis_H1": [],
    "N_Paulis_H4": [],
    "N_Paulis_H0": [],
    }

    ## GENERALIZED vs P-H excitation pool -> first layer always P-H since HF is always reference
    if generalized == True:

        ## Generate indeces for doubly-excited operators
        for a, i, b, j in iterate_t2_generalized(WF.num_spin_orbs):
            pool_data["excitation indeces"].append((i, j, a, b))
            pool_data["excitation type"].append("double")
            pool_data["excitation operator"].append(G2(i, j, a, b, True))

    else:

        ## Generate indeces for doubly-excited operators
        for a, i, b, j in iterate_t2(WF.active_occ_spin_idx, WF.active_unocc_spin_idx):
            pool_data["excitation indeces"].append((i, j, a, b))
            pool_data["excitation type"].append("double")
            pool_data["excitation operator"].append(G2(i, j, a, b, True))

    if efficient == True:

        print('SPLITTING HAMILTONIAN....')

        #define mapper
        mapper = JordanWignerMapper()

        for A in pool_data["excitation operator"]:
            
            H1, H4, H0 = H_split(WF.num_orbs, WF.h_mo, WF.g_mo, A)

            pool_data["N_Paulis_H1"].append(len(mapper.map(FermionicOp(H1.get_qiskit_form(WF.num_orbs), WF.num_spin_orbs)).paulis))
            pool_data["N_Paulis_H4"].append(len(mapper.map(FermionicOp(H4.get_qiskit_form(WF.num_orbs), WF.num_spin_orbs)).paulis))
            pool_data["N_Paulis_H0"].append(len(mapper.map(FermionicOp(H0.get_qiskit_form(WF.num_orbs), WF.num_spin_orbs)).paulis))
            pool_data["H fragments"].append((H1, H4, H0))

    return pool_data

## ROTOSELECT 5 SHOTS

def energy_landscape(A, B, C, D, E, theta):
    '''
    Defines analytic form of energy landscape E(theta) -> similar to eq. 3 in arXiv:2409.05939v2 
    
    arguments
        A, B, C, D, E: coefficients from solution of system of equations
        theta: parameter of the nth gate being optimized

    returns
        energy at theta
    '''
    return A + B*np.cos(theta) + C*np.sin(theta) + D*np.cos(2*theta) + E*np.sin(2*theta)

def global_min_search(A, B, C, D, E):
    """
    Find the global minimum of energy landscape using the companion matrix method
    E(theta) = A + B*cos(theta) + C*sin(theta) + D*cos(2*theta) + E*sin(2*theta)

    Arguments
        A, B, C, D, E: energy landscape coefficients
    
    Returns
        theta_min: the angle (radians) giving the minimum energy
        E_min: the minimum energy
    """
    # Coefficients of derivative in terms of sin/cos
    # dE/dtheta = -B*sin(theta) + C*cos(theta) - 2*D*sin(2*theta) + 2*E*cos(2*theta)
    # Find zeros of derivative
    
    # Used Weierstrass substitution x = tan(theta/2)
    # sin(theta) = 2*x/(1+x^2), cos(theta) = (1-x^2)/(1+x^2)
    # sin(2theta) = 4*x*(1-x^2)/(1+x^2)^2, cos(2theta) = (1-6*x^2+x^4)/(1+x^2)^2
    # Multiply by (1+x^2)^2 to get quartic polynomial: p4*x^4 + p3*x^3 + p2*x^2 + p1*x + p0 = 0
    
    p4 = 2*E - C
    p3 = 8*D - 2*B
    p2 = -12*E
    p1 = -2*B - 8*D
    p0 = 2*E + C

    # Determine polynomial degree
    coeffs_array = np.array([p0, p1, p2, p3, p4])

    # Checking if polynomial is not zero -> can happen with generalized excitation ops
    if not np.all(coeffs_array == 0):

        tol = 1e-6

        def cmat_builder(coeffs):
            """
            Build the companion matrix for a normalized polynomial.
            
            Arguments
                coeffs: coeffs of monic polynomial (normalized to leading order) with minus sign (last = 1)
            
            Returns
                C: companion matrix 
            """
            deg = len(coeffs) - 1
            if deg < 1:
                return None
            C = np.zeros((deg, deg))
            C[1:, :-1] = np.eye(deg - 1)
            C[:, -1] = coeffs[:-1]

            return C

        # Checking the order of the polynomial -> getting corresponding companion matrix

        if np.abs(p4) > np.abs(p3)*tol:
            Cmat = cmat_builder([-p0/p4, -p1/p4, -p2/p4, -p3/p4, 1])
            
        elif np.abs(p3) > np.abs(p2)*tol:
            Cmat = cmat_builder([-p0/p3, -p1/p3, -p2/p3, 1])

        elif np.abs(p2) > np.abs(p1)*tol:
            Cmat = cmat_builder([-p0/p2, -p1/p2, 1])
            
        else:
            Cmat = cmat_builder([-p0/p1, 1])

        # Eigenvalues of Cmat are roots of the quartic polynomial = zeroes of derivative function
        roots = np.linalg.eigvals(Cmat)
    
        # Keep only real roots
        real_roots = roots[np.isreal(roots)].real
    
        # Convert x -> theta
        thetas = 2 * np.arctan(real_roots)
    
        # Evaluate energy at candidate points and find global minimum
        energies = energy_landscape(A, B, C, D, E, thetas)
        idx_min = np.argmin(energies)
        theta_min = thetas[idx_min]
        energy_min = energies[idx_min]

    else:
        theta_min = 0
        energy_min = 0
    
    return theta_min, energy_min


def optimizer(thetas, energies):
    '''
    Solves systems of equations for coefficients of energy landscape

    Arguments
        thetas: list of thetas where landscape is sampled
        energies: list of energy samples of landscape at thetas

    Returns
        theta_min: theta for minimum
        energy_min: energy minimum
    '''
    X = np.column_stack([np.ones_like(thetas), 
                         np.cos(thetas), 
                         np.sin(thetas), 
                         np.cos(2*thetas), 
                         np.sin(2*thetas)
                         ])
                
    coeffs = np.linalg.solve(X, energies)
    A, B, C, D, E = coeffs

    theta_min, energy_min = global_min_search(A, B, C, D, E)

    return theta_min, energy_min

def pool_evaluator(WF, pool_index, pool_data, E_prev):
    '''
    Extends ansatz with candidate unitary from pool
    Finds global minimum using companion matrix method
    
    Arguments
        WF: WF object from SlowQuant from previous iteration
        pool_index: index of considered unitary in pool list
        H: hamiltonian
        pool_data: dictionary with info on pool operators
    
    Returns
        E_min: absolute minimum of energy
        theta_min: theta at absolute minimum of energy
    '''
    excitation_pool = pool_data["excitation indeces"]
    excitation_pool_type = pool_data["excitation type"]

    # Updating last layer with the pool candidate
    WF.ups_layout.excitation_indices[-1] = excitation_pool[pool_index]
    WF.ups_layout.excitation_operator_type[-1] = excitation_pool_type[pool_index]

    # Energy measurements to build system of equations
    energies = [E_prev]
    thetas = [0.0]

    for l in range(1,5):
        current_thetas = WF.thetas
        current_thetas[-1] = (2*np.pi*l)/5
        WF.thetas = current_thetas
        energies.append(WF.energy_elec)
        thetas.append((2*np.pi*l)/5)

    Thetas = np.array(thetas)
    Energies = np.array(energies)

    # Find energy landscape and its global minimum
    theta_min, E_min = optimizer(Thetas, Energies)

    return theta_min, E_min

def measure_energy_theta(WF, H, thetas_list):
    '''
    Updates WF with new thetas + measures energy

    Arguments
        WF: WF object from SlowQuant from previous iteration
        H: hamiltonian
        thetas_list: thetas to pass to WF

    Return
        energy: energy estimate for given theta setting
    '''
    WF.thetas = thetas_list
    energy = float(expectation_value(WF.ci_coeffs, [H], WF.ci_coeffs, WF.ci_info, WF.thetas, WF.ups_layout))

    return energy

def rotoselect(WF, pool_data, cas_en, adapt_threshold = 1e-5):  # adapt_threshold for chemical accuracy
    '''
    Constructs ansatz iteratively using Rotoselect algorithm
    No parameter optimization after selection

    Arguments
        WF: wave function object from SlowQuant
        pool_data: dictionary with the data about the operator pool
        cas_en: CASSCF reference energy (used to estimate chemical accuracy)
        adapt_threshold: min energy reduction upon addition of new layer (optional, default is chemical accuracy)

    Returns
        WF: final wave function object from SlowQuant
        en_traj: energy optimization trajectory
        rdm1_traj: rdm1 at each layer    
    '''
    # Defining the excitation pool
    excitation_pool = pool_data["excitation indeces"]
    excitation_pool_type = pool_data["excitation type"]

    # Initialize previous energy and energy trajectory (here with HF energy)
    E_prev_adapt = float(WF.energy_elec)
    en_traj = [E_prev_adapt]
    rdm1_traj = [WF.rdm1]
    rdm2_traj = [WF.rdm2]

    converged = False

    while converged == False and WF.ups_layout.n_params <= 100:

        # Load new operator in the ansatz and initialize
        WF.ups_layout.n_params += 1
        WF.ups_layout.excitation_indices.append((0,0))
        WF.ups_layout.excitation_operator_type.append(" ")
        WF.ups_layout.grad_param_R[f"p{WF.ups_layout.n_params:09d}"] = 2
        WF.ups_layout.param_names.append(f"p{WF.ups_layout.n_params:09d}")
        WF._thetas.append(0.0)

        results = []

        # Looping through pool operator -> get the best ansatz
        for i in range(len(excitation_pool)):
            results.append(pool_evaluator(WF, i, pool_data, E_prev_adapt))

        # results = pool_parallel(WF, H, pool_data, E_prev_adapt)
        theta_pool, energy_pool = zip(*results)

        op_index = np.argmin(energy_pool)

        print('OPERATOR->', op_index)
        print(f'Theta {theta_pool[op_index]} - Energy {energy_pool[op_index]} - previous {E_prev_adapt}')
    
        # deltaE_adapt = np.abs(energy_pool[op_index]-E_prev_adapt) # with respect to previous layer
        deltaE_adapt = np.abs(cas_en-energy_pool[op_index]) # for chemical accuracy threshold


        if deltaE_adapt < adapt_threshold or WF.ups_layout.n_params >= 100:
            # Updating last layer with data from best operator
            WF.ups_layout.excitation_indices[-1] = excitation_pool[op_index]
            WF.ups_layout.excitation_operator_type[-1] = excitation_pool_type[op_index]
            thetas = WF.thetas
            thetas[-1] = theta_pool[op_index]
            WF.thetas = thetas

            # Appending energy and termination
            en_traj.append(float(energy_pool[op_index]))
            rdm1_traj.append(WF.rdm1)
            rdm2_traj.append(WF.rdm2)
            converged = True

            # Final printout
            print('----------------------')
            print(f'FINAL RESULT - Energy: {en_traj[-1]} - #Layers: {WF.ups_layout.n_params}')
            print('EXCITATION SEQUENCE')
            for i, (op_idx, op_type) in enumerate(zip(WF.ups_layout.excitation_indices, WF.ups_layout.excitation_operator_type)):
                print(f'LAYER {i}: {op_idx} | {op_type}')

        else:
            # Updating last layer with data from best operator
            WF.ups_layout.excitation_indices[-1] = excitation_pool[op_index]
            WF.ups_layout.excitation_operator_type[-1] = excitation_pool_type[op_index]
            thetas = WF.thetas
            thetas[-1] = theta_pool[op_index]
            WF.thetas = thetas
            
            # Appending energy to trajectory and updating 'previous' energy for next iteration
            print(f'RESULT at layer {WF.ups_layout.n_params} - Energy: {energy_pool[op_index]} - Previous: {E_prev_adapt} - Delta: {deltaE_adapt} - Theta: {theta_pool[op_index]}')
            en_traj.append(float(energy_pool[op_index]))
            rdm1_traj.append(WF.rdm1)
            rdm2_traj.append(WF.rdm2)            
            E_prev_adapt = float(energy_pool[op_index])  # with respect to previous layer

    return WF, en_traj, rdm1_traj, rdm2_traj

def rotoselect_opt(WF, pool_data, cas_en, po, oo, adapt_threshold = 1e-5):  # adapt_threshold for chemical accuracy
    '''
    Constructs ansatz iteratively using Rotoselect algorithm
    Full VQE optimization after selection

    Arguments
        WF: wave function object from SlowQuant
        pool_data: dictionary with the data about the operator pool
        cas_en: CASSCF reference energy (used to estimate chemical accuracy)
        adapt_threshold: min energy reduction upon addition of new layer (optional, default is chemical accuracy)
        po: optimize parameters of the unitary (boolean)
        oo: orbital optimization (boolean)

    Returns
        WF: final wave function object from SlowQuant
        en_traj: energy optimization trajectory
        rdm1_traj: rdm1 at each layer        
    '''
    # Defining the excitation pool
    excitation_pool = pool_data["excitation indeces"]
    excitation_pool_type = pool_data["excitation type"]

    # Initialize previous energy and energy trajectory (here with HF energy)
    E_prev_adapt = float(WF.energy_elec)
    en_traj = [E_prev_adapt]
    rdm1_traj = [WF.rdm1]
    rdm2_traj = [WF.rdm2]

    converged = False

    while converged == False and WF.ups_layout.n_params <= 100:

        # Load new operator slot in the ansatz
        WF.ups_layout.n_params += 1
        WF.ups_layout.excitation_indices.append((0,0))
        WF.ups_layout.excitation_operator_type.append(" ")
        WF.ups_layout.grad_param_R[f"p{WF.ups_layout.n_params:09d}"] = 2
        WF.ups_layout.param_names.append(f"p{WF.ups_layout.n_params:09d}")
        WF._thetas.append(0.0)

        results = []

        for i in range(len(excitation_pool)):
            results.append(pool_evaluator(WF, i, pool_data, E_prev_adapt))

        # results = pool_parallel(WF, H, pool_data, E_prev_adapt)
        theta_pool, energy_pool = zip(*results)
        op_index = np.argmin(energy_pool)

        print('OPERATOR->', op_index)
        print(f'Theta {theta_pool[op_index]} - Energy {energy_pool[op_index]} - previous {E_prev_adapt}')

        # deltaE_adapt = np.abs(energy_pool[op_index]-E_prev_adapt)  # with respect to previous layer
        deltaE_adapt = np.abs(cas_en-energy_pool[op_index])  # for chemical accuracy

        if deltaE_adapt < adapt_threshold or WF.ups_layout.n_params >= 100:  
            # Updating last layer with data from best operator
            WF.ups_layout.excitation_indices[-1] = excitation_pool[op_index]
            WF.ups_layout.excitation_operator_type[-1] = excitation_pool_type[op_index]
            thetas = WF.thetas
            thetas[-1] = theta_pool[op_index]
            WF.thetas = thetas

            # Appending energy and termination (before full optimization)
            en_traj.append(float(energy_pool[op_index]))
            rdm1_traj.append(WF.rdm1)
            rdm2_traj.append(WF.rdm2)
            converged = True
            print('----------------------')
            print(f'FINAL RESULT - Energy: {en_traj[-1]} - #Layers: {WF.ups_layout.n_params}')
            print('EXCITATION SEQUENCE')
            for i, (op_idx, op_type) in enumerate(zip(WF.ups_layout.excitation_indices, WF.ups_layout.excitation_operator_type)):
                print(f'LAYER {i}: {op_idx} | {op_type}')
        
        else:
            # Updating last layer with data from best operator
            WF.ups_layout.excitation_indices[-1] = excitation_pool[op_index]
            WF.ups_layout.excitation_operator_type[-1] = excitation_pool_type[op_index]
            thetas = WF.thetas
            thetas[-1] = theta_pool[op_index]
            WF.thetas = thetas

            # Running an optimization
            if po == True and oo == True:
                WF.run_wf_optimization_1step("bfgs", orbital_optimization=True)

            if po == True and oo == False:
                WF.run_wf_optimization_1step("bfgs", orbital_optimization=False)

            if po == False and oo == True:
                WF.run_orbital_optimization()

            # WF = rotosolve(WF)
            
            # Appending energy to trajectory and updating 'previous' energy for next iteration
            en_traj.append(WF.energy_elec)
            deltaE_adapt = np.abs(cas_en-en_traj[-1])            

            # Checking convergence to chemical accuracy
            if deltaE_adapt < adapt_threshold or WF.ups_layout.n_params >= 150:
                rdm1_traj.append(WF.rdm1)
                rdm2_traj.append(WF.rdm2)
                converged = True

                # Final printout
                print('----------------------')
                print(f'FINAL RESULT - Energy: {en_traj[-1]} - #Layers: {WF.ups_layout.n_params}')
                print('EXCITATION SEQUENCE')
                for i, (op_idx, op_type) in enumerate(zip(WF.ups_layout.excitation_indices, WF.ups_layout.excitation_operator_type)):
                    print(f'LAYER {i}: {op_idx} | {op_type}')

            else:
                rdm1_traj.append(WF.rdm1)
                rdm2_traj.append(WF.rdm2)
                print(f'RESULT at layer {WF.ups_layout.n_params} - Energy: {WF.energy_elec} - Previous: {E_prev_adapt} - Delta: {deltaE_adapt} - Theta: {theta_pool[op_index]}')
                E_prev_adapt = en_traj[-1]

    return WF, en_traj, rdm1_traj, rdm2_traj


## ROTOSELECT EFFICIENT

# Construct H operator and split it into H1 and H4

def a_op(spinless_idx: int, spin: str, dagger: bool) -> FermionicOperator:
    """Construct annihilation/creation operator.

    Args:
        spinless_idx: Spatial orbital index.
        spin: Alpha or beta spin.
        dagger: If creation operator.

    Returns:
        Annihilation/creation operator.
    """
    if spin not in ("alpha", "beta"):
        raise ValueError(f'spin must be "alpha" or "beta" got {spin}')
    idx = 2 * spinless_idx
    if spin == "beta":
        idx += 1
    return FermionicOperator({((idx, dagger),): 1})

def Epq_s(p: int, q: int) -> FermionicOperator:
    r"""Construct Fermi strings for one-electron excitation operator.

    .. math::
        \hat{O}_{n} = \hat{a}^\dagger_{p,\sigma}\hat{a}_{q,\tau}

    Args:
        p: Spatial orbital index.
        q: Spatial orbital index.

    Returns:
        O1, O2: Two one-electron Fermi strings for all excitations between spin-orbitals
    """
    O1 = a_op(p, "alpha", dagger=True) * a_op(q, "alpha", dagger=False)
    O2 = a_op(p, "beta", dagger=True) * a_op(q, "beta", dagger=False)

    return O1, O2

def Epqrs_s(p: int, q: int, r: int, s: int) -> FermionicOperator:
    r"""
    Construct Fermi strings for two-electron excitation operator -> 2.2.15 pink bible

    .. math::
        \hat{O}_{n} = hat{a}^\dagger_{p,\sigma}\hat{a}_{r,\tau}\hat{a}_{s,\tau}\hat{a}_{q,\tau}

    Args:
        p: Spatial orbital index.
        q: Spatial orbital index.
        r: Spatial orbital index.
        s: Spatial orbital index.

    Returns:
        O1, O2, O3, O4: Four two-electron Fermi strings for all excitations between spin-orbitals.
    """
    O1 = (a_op(p, "alpha", dagger=True) 
          * a_op(r, "alpha", dagger=True) 
          * a_op(s, "alpha", dagger=False)
          * a_op(q, "alpha", dagger=False)
          ) 

    O2 = (a_op(p, "beta", dagger=True) 
        * a_op(r, "alpha", dagger=True) 
        * a_op(s, "alpha", dagger=False)
        * a_op(q, "beta", dagger=False)
        )
                
    O3 = (a_op(p, "alpha", dagger=True) 
        * a_op(r, "beta", dagger=True) 
        * a_op(s, "beta", dagger=False)
        * a_op(q, "alpha", dagger=False)
        )
                
    O4 = (a_op(p, "beta", dagger=True) 
        * a_op(r, "beta", dagger=True) 
        * a_op(s, "beta", dagger=False)
        * a_op(q, "beta", dagger=False)
        )

    return O1, O2, O3, O4

def EvaluateSort(O: FermionicOperator, 
                A: FermionicOperator, 
                H1: FermionicOperator, 
                H4: FermionicOperator,
                H0: FermionicOperator, 
                intgr: float,
                num_part: int):  
    '''
    Divides H into \alpha=1 (H1) and \alpha=4 (H4) cases (PRA 111, 042825 (2025)) and \alpha=0
    1. Calculates [O,A] and A[O,A]A
    2. If A[O,A]A = 0 and [O,A] != 0 -> \alpha=1 -> add integr*O to H1
    3. If A[O,A]A = [O,A] != 0 -> \alpha=4 -> add integr*O to H4
    4. If A[O,A]A = [O,A] == 0 -> \alpha=0 -> add integr*O to H0

    Arguments:
        - O: Fermi string from H
        - A: anti-symmetric generator from pool
        - H1: Fermi strings from \alpha = 1 case
        - H4: Fermi strings from \alpha = 4 case
        - H0: Fermi strings from \alpha = 0 case
        - intgr: molecular integral
        - num_part: 1-particle or 2-particle part of H

    Returns:
        - H1: H1 with added Fermi string or input H1
        - H4: H4 with added Fermi string or input H4
    '''
    # commutator [O,A]
    comm = commutator(O, A)
    cc = comm.operator_count

    # product commutator A[O,A]A
    pc = (A*comm*A).operator_count

    print(O.operators_readable, A.operators_readable)
    print('[O,A]:', commutator(O, A).operators_readable)
    print('A[O,A]A:', (A*commutator(O, A)*A).operators_readable)

    # 1-particle part of H
    if num_part == 1:

        if cc == pc == {}:
            H0 += intgr * O
            print('alpha = 0')

        if cc == pc != {}:
            H4 += intgr * O
            print('alpha = 4')

        if pc != cc:
            H1 += intgr * O
            print('alpha = 1')        

    # 2-particle part of H
    if num_part == 2:

        if cc == pc == {}:
            H0 += 1/2 * intgr * O
            print('alpha = 0')

        if cc == pc != {}:
            H4 += 1/2 * intgr * O
            print('alpha = 4')

        if pc != cc:
            H1 += 1/2 * intgr * O
            print('alpha = 1')

    return H1, H4, H0

def H_split(nMO, h_mo, g_mo, A):
    '''
    Splits H into H1 and H4 via evaluated_sort
    
    Arguments:
        - nMO: number of molecular orbitals
        - h_mo: 1-electron integrals, molecular orbitals basis
        - g_mo: 2-electron integrals, molecular orbitals basis
        - A: anti-hermitian generator

    Returns:
        - H1, H4, H0: \alpha=1, \alpha=4 and \alpha=0 Hamiltonians
    '''
    H1 = FermionicOperator({})
    H4 = FermionicOperator({})
    H0 = FermionicOperator({})

    for p in range(nMO):
        for q in range(nMO):
            if abs(h_mo[p, q]) < 10**-14:
                continue

            # Generate 1-electron Fermi strings 
            O1, O2 = Epq_s(p, q)

            # Split Fermi strings into H1 or H4 
            H1, H4, H0 = EvaluateSort(O1, A, H1, H4, H0, h_mo[p, q], 1)
            H1, H4, H0 = EvaluateSort(O2, A, H1, H4, H0, h_mo[p, q], 1)

            print('---------')

    for p in range(nMO):
        for q in range(nMO):
            for r in range(nMO):
                for s in range(nMO):
                    if abs(g_mo[p, q, r, s]) < 10**-14:
                        continue

                    # Generate 2-electron Fermi strings 
                    O1, O2, O3, O4 = Epqrs_s(p, q, r, s)
                                
                    # Split Fermi strings into H1 or H4 
                    H1, H4, H0 = EvaluateSort(O1, A, H1, H4, H0, g_mo[p, q, r, s], 2)
                    H1, H4, H0 = EvaluateSort(O2, A, H1, H4, H0, g_mo[p, q, r, s], 2)
                    H1, H4, H0 = EvaluateSort(O3, A, H1, H4, H0, g_mo[p, q, r, s], 2)
                    H1, H4, H0 = EvaluateSort(O4, A, H1, H4, H0, g_mo[p, q, r, s], 2)

    return H1, H4, H0

def landscape1(A, B, C, theta):
    '''
    Energy landscape for \alpha = 1 case --> eq. 12 in PRA 111, 042825 (2025)
    
    Arguments:
        A, B, C: landscape coefficients
        theta: parameter of analytic landscape

    Returns:
        Analytic landscape function
    '''
    return A + B*np.sin(theta) + C*(1-np.cos(theta))

def landscape4(A, B, C, theta):
    '''
    Energy landscape for \alpha = 4 case --> eq. 13 in PRA 111, 042825 (2025)
    
    Arguments:
        A, B, C: landscape coefficients
        theta: parameter of analytic landscape

    Returns:
        Analytic landscape function
    '''
    return A + 1/2*B*np.sin(2*theta) + 1/2*C*np.sin(theta)**2

def landscape_tot(A1, B1, C1, A4, B4, C4, theta):
    '''
    Total energy landscape summing landscape1 and landscape4
    
    Arguments:
        A1, B1, C1: landscape1 coefficients
        A4, B4, C4: landscape4 coefficients
        theta: parameter of analytic landscape (common between landscape1 and landscape4)

    Returns:
        Total analytic landscape function
    '''
    return A1 + B1*np.sin(theta) + C1*(1-np.cos(theta)) + A4 + 1/2*B4*np.sin(2*theta) + 1/2*C4*np.sin(theta)**2

# Polynomial corresponding to the derivative -> find zeros = find min-max

def poly_min(B1, C1, B4, C4):
    '''
    Calculates coefficients of polynomial
        P4*x^4 + P3*x^3 + P2*x^2 +P1*x + P0
    corresponding to derivative of landscape_tot
    Obtained via substitutions: sin(\theta) = 2x/1+x^2 and cos(\theta) = 1-x^2/1+x^2
    
    Arguments:
        B1, C1, B4, C4: cofficients of landscape1 and lanscape4

    Returns:
        P4, P3, P2, P1, P0: coefficients of derivative polynomial
    
    '''
    P4 = B4 - B1
    P3 = 2*C1 - 2*C4
    P2 = -6*B4
    P1 = 2*C1 + 2*C4
    P0 = B1 + B4

    return [P4, P3, P2, P1, P0]

def rotoselect_efficient_opt(WF, pool_data, cas_en, po, oo, adapt_threshold = 1e-5):  # adapt_threshold for chemical accuracy

    # Defining the excitation pool
    excitation_pool = pool_data["excitation indeces"]
    excitation_pool_type = pool_data["excitation type"]

    # Initialize previous energy and energy trajectory (here with HF energy)
    E_prev_adapt = float(WF.energy_elec)
    en_traj = [E_prev_adapt]
    rdm1_traj = [WF.rdm1]
    rdm2_traj = [WF.rdm2]

    # Initialize thetas sample
    thetas_samples = np.array([0, np.pi/3, 3*np.pi/2])

    converged = False

    # Initialize the number of measurements over the pool
    N_Paulis_pool = 0

    while converged == False and WF.ups_layout.n_params <= 100:

        # Load new operator slot in the ansatz
        WF.ups_layout.n_params += 1
        WF.ups_layout.excitation_indices.append((0,0))
        WF.ups_layout.excitation_operator_type.append(" ")
        WF.ups_layout.grad_param_R[f"p{WF.ups_layout.n_params:09d}"] = 2
        WF.ups_layout.param_names.append(f"p{WF.ups_layout.n_params:09d}")
        WF._thetas.append(0.0)

        energy_pool = []
        theta_pool = []

        for i in range(len(excitation_pool)):
            
            H1, H4, H0 = pool_data["H fragments"][i]

            N_Paulis_H1 = pool_data["N_Paulis_H1"][i]
            N_Paulis_H4 = pool_data["N_Paulis_H4"][i]

            N_Paulis_H0 = pool_data["N_Paulis_H0"][i]

            E1 = []
            E4 = []

            WF.ups_layout.excitation_indices[-1] = excitation_pool[i]
            WF.ups_layout.excitation_operator_type[-1] = excitation_pool_type[i]

            # MEASUREMENT - 2 SHOTS + Small part H (either H1 or H4) - 2 shots if either #Paulis_H1 or #Paulis_H4 = 0
            for theta in thetas_samples:

                thetas = WF.thetas
                thetas[-1] = theta
                WF.thetas = thetas

                # Recycling previous measurement for \theta = 0
                if theta == 0:

                    if N_Paulis_H0 < N_Paulis_H4 and N_Paulis_H1 < N_Paulis_H4:

                        E0 = expectation_value(WF.ci_coeffs, [H0], WF.ci_coeffs, WF.ci_info)
                        N_Paulis_pool += N_Paulis_H0
                        E1.append(expectation_value(WF.ci_coeffs, [H1], WF.ci_coeffs, WF.ci_info))
                        N_Paulis_pool += N_Paulis_H1
                        E4.append(E_prev_adapt - E1[0] - E0)

                    if N_Paulis_H0 < N_Paulis_H1 and N_Paulis_H4 < N_Paulis_H1:

                        E0 = expectation_value(WF.ci_coeffs, [H0], WF.ci_coeffs, WF.ci_info)
                        N_Paulis_pool += N_Paulis_H0
                        E4.append(expectation_value(WF.ci_coeffs, [H4], WF.ci_coeffs, WF.ci_info))
                        N_Paulis_pool += N_Paulis_H4
                        E1.append(E_prev_adapt - E4[0] - E0)

                    if N_Paulis_H1 < N_Paulis_H0 and N_Paulis_H4 < N_Paulis_H0:

                        E1.append(expectation_value(WF.ci_coeffs, [H1], WF.ci_coeffs, WF.ci_info))
                        N_Paulis_pool += N_Paulis_H1
                        E4.append(expectation_value(WF.ci_coeffs, [H4], WF.ci_coeffs, WF.ci_info))
                        N_Paulis_pool += N_Paulis_H4
                        E0 = E_prev_adapt - E4[0] - E1[0]

                # Measure both H1 and H4 for next two \thetas
                else:
                    E1.append(expectation_value(WF.ci_coeffs, [H1], WF.ci_coeffs, WF.ci_info))
                    N_Paulis_pool += N_Paulis_H1
                    E4.append(expectation_value(WF.ci_coeffs, [H4], WF.ci_coeffs, WF.ci_info))
                    N_Paulis_pool += N_Paulis_H4

            E1 = np.array(E1) + np.repeat([E0], 3)
            E4 = np.array(E4)

            ## FIND LANDSCAPE MINIMUM

            # Getting landscape coefficients by solving system of equations
            X1 = np.column_stack([np.ones_like(thetas_samples),
                                np.sin(thetas_samples), 
                                1-np.cos(thetas_samples),
                                ])
                    
            A1, B1, C1 = np.linalg.solve(X1, E1)

            X4 = np.column_stack([np.ones_like(thetas_samples),
                                1/2*np.sin(2*thetas_samples),
                                1/2*np.sin(thetas_samples)**2,
                                ])
                    
            A4, B4, C4 = np.linalg.solve(X4, E4)

            # Find thetas for min and max energy points by finding zeros of derivative polynomial
            pol_coeffs = np.array(poly_min(B1, C1, B4, C4))

            # Run np.roots only when pol_coeffs is not array of zeros (crashes otherwise)
            if pol_coeffs.all() == 0:
                theta_min_max = [0]

            else:
                # Removing complex roots
                roots = np.roots(pol_coeffs)                
                theta_min_max = 2*np.arctan(roots[np.isclose(roots.imag, 0)].real)

                if np.any(theta_min_max.imag > 1e-12):
                    raise ValueError('Complex roots encountered')
            
            # Calculate min/max energy of total landscape
            E_min_max = landscape_tot(A1, B1, C1, A4, B4, C4, theta_min_max)      

            #Get energy minimum and corresponding theta
            E_min = np.min(E_min_max)
            theta_min = theta_min_max[np.argmin(E_min_max)]

            # Store energy scores and corresponding theta
            energy_pool.append(E_min)
            theta_pool.append(theta_min)

        op_index = np.argmin(energy_pool)

        print('OPERATOR->', op_index)
        print(f'Theta {theta_pool[op_index]} - Energy {energy_pool[op_index]} - previous {E_prev_adapt}')

        # deltaE_adapt = np.abs(energy_pool[op_index]-E_prev_adapt)  # with respect to previous layer
        deltaE_adapt = np.abs(cas_en-energy_pool[op_index])  # with respect to full CI

        if deltaE_adapt < adapt_threshold or WF.ups_layout.n_params >= 100:  
            # Updating last layer with data from best operator
            WF.ups_layout.excitation_indices[-1] = excitation_pool[op_index]
            WF.ups_layout.excitation_operator_type[-1] = excitation_pool_type[op_index]
            thetas = WF.thetas
            thetas[-1] = theta_pool[op_index]
            WF.thetas = thetas

            # Appending energy and termination (before full optimization)
            en_traj.append(float(energy_pool[op_index]))
            rdm1_traj.append(WF.rdm1)
            rdm2_traj.append(WF.rdm2)
            converged = True
            print('----------------------')
            print(f'FINAL RESULT - Energy: {en_traj[-1]} - #Layers: {WF.ups_layout.n_params}')
            print('EXCITATION SEQUENCE')
            for i, (op_idx, op_type) in enumerate(zip(WF.ups_layout.excitation_indices, WF.ups_layout.excitation_operator_type)):
                print(f'LAYER {i}: {op_idx} | {op_type}')
        
        else:
            # Updating last layer with data from best operator
            WF.ups_layout.excitation_indices[-1] = excitation_pool[op_index]
            WF.ups_layout.excitation_operator_type[-1] = excitation_pool_type[op_index]
            thetas = WF.thetas
            thetas[-1] = theta_pool[op_index]
            WF.thetas = thetas

            # Running an optimization
            if po == True and oo == True:
                WF.run_wf_optimization_1step("bfgs", orbital_optimization=True)

            if po == True and oo == False:
                WF.run_wf_optimization_1step("bfgs", orbital_optimization=False)

            if po == False and oo == True:
                WF.run_orbital_optimization()

            # WF = rotosolve(WF)
            
            # Appending energy to trajectory and updating 'previous' energy for next iteration
            en_traj.append(WF.energy_elec)
            deltaE_adapt = np.abs(cas_en-en_traj[-1])            

            # Checking convergence to chemical accuracy
            if deltaE_adapt < adapt_threshold or WF.ups_layout.n_params >= 100:
                rdm1_traj.append(WF.rdm1)
                rdm2_traj.append(WF.rdm2)
                converged = True

                # Final printout
                print('----------------------')
                print(f'FINAL RESULT - Energy: {en_traj[-1]} - #Layers: {WF.ups_layout.n_params}')
                print('EXCITATION SEQUENCE')
                for i, (op_idx, op_type) in enumerate(zip(WF.ups_layout.excitation_indices, WF.ups_layout.excitation_operator_type)):
                    print(f'LAYER {i}: {op_idx} | {op_type}')

            else:
                rdm1_traj.append(WF.rdm1)
                rdm2_traj.append(WF.rdm2)
                print(f'RESULT at layer {WF.ups_layout.n_params} - Energy: {WF.energy_elec} - Previous: {E_prev_adapt} - Delta: {deltaE_adapt} - Theta: {theta_pool[op_index]}')
                E_prev_adapt = en_traj[-1]

    return WF, en_traj, N_Paulis_pool, rdm1_traj, rdm2_traj

def rotoselect_efficient(WF, pool_data, cas_en, po, oo, adapt_threshold = 1e-5):  # adapt_threshold for chemical accuracy

    # Defining the excitation pool
    excitation_pool = pool_data["excitation indeces"]
    excitation_pool_type = pool_data["excitation type"]

    # Initialize previous energy and energy trajectory (here with HF energy)
    E_prev_adapt = float(WF.energy_elec)
    en_traj = [E_prev_adapt]
    rdm1_traj = [WF.rdm1]
    rdm2_traj = [WF.rdm2]

    # Initialize thetas sample
    thetas_samples = np.array([0, np.pi/3, 3*np.pi/2])

    converged = False

    # Initialize the number of measurements over the pool
    N_Paulis_pool = 0

    while converged == False and WF.ups_layout.n_params <= 100:

        # Load new operator slot in the ansatz
        WF.ups_layout.n_params += 1
        WF.ups_layout.excitation_indices.append((0,0))
        WF.ups_layout.excitation_operator_type.append(" ")
        WF.ups_layout.grad_param_R[f"p{WF.ups_layout.n_params:09d}"] = 2
        WF.ups_layout.param_names.append(f"p{WF.ups_layout.n_params:09d}")
        WF._thetas.append(0.0)

        energy_pool = []
        theta_pool = []

        for i in range(len(excitation_pool)):
            
            H1, H4, H0 = pool_data["H fragments"][i]
            N_Paulis_H1 = pool_data["N_Paulis_H1"][i]
            N_Paulis_H4 = pool_data["N_Paulis_H4"][i]
            N_Paulis_H0 = pool_data["N_Paulis_H0"][i]

            E1 = []
            E4 = []

            WF.ups_layout.excitation_indices[-1] = excitation_pool[i]
            WF.ups_layout.excitation_operator_type[-1] = excitation_pool_type[i]

            # MEASUREMENT - 2 SHOTS + Small part H (either H1 or H4) - 2 shots if either #Paulis_H1 or #Paulis_H4 = 0
            for theta in thetas_samples:

                thetas = WF.thetas
                thetas[-1] = theta
                WF.thetas = thetas

                # Recycling previous measurement for \theta = 0
                if theta == 0:

                    if N_Paulis_H0 < N_Paulis_H4 and N_Paulis_H1 < N_Paulis_H4:

                        E0 = expectation_value(WF.ci_coeffs, [H0], WF.ci_coeffs, WF.ci_info)
                        N_Paulis_pool += N_Paulis_H0
                        E1.append(expectation_value(WF.ci_coeffs, [H1], WF.ci_coeffs, WF.ci_info))
                        N_Paulis_pool += N_Paulis_H1
                        E4.append(E_prev_adapt - E1[0] - E0)

                    if N_Paulis_H0 < N_Paulis_H1 and N_Paulis_H4 < N_Paulis_H1:

                        E0 = expectation_value(WF.ci_coeffs, [H0], WF.ci_coeffs, WF.ci_info)
                        N_Paulis_pool += N_Paulis_H0
                        E4.append(expectation_value(WF.ci_coeffs, [H4], WF.ci_coeffs, WF.ci_info))
                        N_Paulis_pool += N_Paulis_H4
                        E1.append(E_prev_adapt - E4[0] - E0)

                    if N_Paulis_H1 < N_Paulis_H0 and N_Paulis_H4 < N_Paulis_H0:

                        E1.append(expectation_value(WF.ci_coeffs, [H1], WF.ci_coeffs, WF.ci_info))
                        N_Paulis_pool += N_Paulis_H1
                        E4.append(expectation_value(WF.ci_coeffs, [H4], WF.ci_coeffs, WF.ci_info))
                        N_Paulis_pool += N_Paulis_H4
                        E0 = E_prev_adapt - E4[0] - E1[0]

                # Measure both H1 and H4 for next two \thetas
                else:
                    E1.append(expectation_value(WF.ci_coeffs, [H1], WF.ci_coeffs, WF.ci_info))
                    N_Paulis_pool += N_Paulis_H1
                    E4.append(expectation_value(WF.ci_coeffs, [H4], WF.ci_coeffs, WF.ci_info))
                    N_Paulis_pool += N_Paulis_H4

            E1 = np.array(E1) + np.repeat([E0], 3)
            E4 = np.array(E4)

            ## FIND LANDSCAPE MINIMUM

            # Getting landscape coefficients by solving system of equations
            X1 = np.column_stack([np.ones_like(thetas_samples),
                                np.sin(thetas_samples), 
                                1-np.cos(thetas_samples),
                                ])
                    
            A1, B1, C1 = np.linalg.solve(X1, E1)

            X4 = np.column_stack([np.ones_like(thetas_samples),
                                1/2*np.sin(2*thetas_samples),
                                1/2*np.sin(thetas_samples)**2,
                                ])
                    
            A4, B4, C4 = np.linalg.solve(X4, E4)

            # Find thetas for min and max energy points by finding zeros of derivative polynomial
            pol_coeffs = np.array(poly_min(B1, C1, B4, C4))

            # Run np.roots only when pol_coeffs is not array of zeros (crashes otherwise)
            if pol_coeffs.all() == 0:
                theta_min_max = [0]

            else:
                # Removing complex roots
                roots = np.roots(pol_coeffs)                
                theta_min_max = 2*np.arctan(roots[np.isclose(roots.imag, 0)].real)

                if np.any(theta_min_max.imag > 1e-12):
                    raise ValueError('Complex roots encountered')
            
            # Calculate min/max energy of total landscape
            E_min_max = landscape_tot(A1, B1, C1, A4, B4, C4, theta_min_max)      

            #Get energy minimum and corresponding theta
            E_min = np.min(E_min_max)
            theta_min = theta_min_max[np.argmin(E_min_max)]

            # Store energy scores and corresponding theta
            energy_pool.append(E_min)
            theta_pool.append(theta_min)

        op_index = np.argmin(energy_pool)

        print('OPERATOR->', op_index)
        print(f'Theta {theta_pool[op_index]} - Energy {energy_pool[op_index]} - previous {E_prev_adapt}')
    
        # deltaE_adapt = np.abs(energy_pool[op_index]-E_prev_adapt) # with respect to previous layer
        deltaE_adapt = np.abs(cas_en-energy_pool[op_index]) # for chemical accuracy threshold

        if deltaE_adapt < adapt_threshold or WF.ups_layout.n_params >= 100:
            # Updating last layer with data from best operator
            WF.ups_layout.excitation_indices[-1] = excitation_pool[op_index]
            WF.ups_layout.excitation_operator_type[-1] = excitation_pool_type[op_index]
            thetas = WF.thetas
            thetas[-1] = theta_pool[op_index]
            WF.thetas = thetas

            # Appending energy and termination
            en_traj.append(float(energy_pool[op_index]))
            rdm1_traj.append(WF.rdm1)
            rdm2_traj.append(WF.rdm2)
            converged = True

            # Final printout
            print('----------------------')
            print(f'FINAL RESULT - Energy: {en_traj[-1]} - #Layers: {WF.ups_layout.n_params}')
            print('EXCITATION SEQUENCE')
            for i, (op_idx, op_type) in enumerate(zip(WF.ups_layout.excitation_indices, WF.ups_layout.excitation_operator_type)):
                print(f'LAYER {i}: {op_idx} | {op_type}')

        else:
            # Updating last layer with data from best operator
            WF.ups_layout.excitation_indices[-1] = excitation_pool[op_index]
            WF.ups_layout.excitation_operator_type[-1] = excitation_pool_type[op_index]
            thetas = WF.thetas
            thetas[-1] = theta_pool[op_index]
            WF.thetas = thetas
            
            # Appending energy to trajectory and updating 'previous' energy for next iteration
            print(f'RESULT at layer {WF.ups_layout.n_params} - Energy: {energy_pool[op_index]} - Previous: {E_prev_adapt} - Delta: {deltaE_adapt} - Theta: {theta_pool[op_index]}')
            en_traj.append(float(energy_pool[op_index]))
            rdm1_traj.append(WF.rdm1)
            rdm2_traj.append(WF.rdm2)            
            E_prev_adapt = float(energy_pool[op_index])  # with respect to previous layer

    return WF, en_traj, N_Paulis_pool, rdm1_traj, rdm2_traj

def rotosolve(WF, max_epochs = 1000, opt_threshold = 1e-6):
    '''
    Implements RotoSolve algorithm from arXiv:2409.05939

    Arguments
        WF: wave function object from SlowQuant
        max_epochs: max number of theta optimization cycles
        opt_threshold: energy change threshold for \theta variation

    Returns
        WF: final wave function object from SlowQuant
    '''

    converged = False
    current_thetas = WF.thetas
    E_prev = WF.energy_elec

    while converged == False:

        for epochs in range(max_epochs):

            E_prev_ep = E_prev

            # Looping over all thetas parameters
            for d in range(WF.ups_layout.n_params):

                # Energy measurements to build system of equations
                energies = [E_prev]
                thetas = [current_thetas[d]]

                for l in range(1,5):
                    current_thetas[d] = thetas[0] + (2*np.pi*l)/5
                    WF.thetas = current_thetas
                    energies.append(WF.energy_elec)
                    thetas.append(current_thetas[d])

                Thetas = np.array(thetas)
                Energies = np.array(energies)

                # Find energy landscape and its global minimum
                theta_min, E_min = optimizer(Thetas, Energies)

                # Assign minimum theta to current_thetas
                current_thetas[d] = theta_min
                E_prev = E_min

                # Updating thetas and adding OPT energy evaluations
                WF.thetas = current_thetas
                WF.num_energy_evals += 4

            deltaE = np.abs(E_prev_ep-E_prev)

            if deltaE <= opt_threshold:
                print(f'Layer: {WF.ups_layout.n_params} - CONVERGED at step: {epochs} - \u0394E: {deltaE}')
                converged = True
                break

            else:
                print(f'Layer: {WF.ups_layout.n_params} - convergence at step: {epochs} - \u0394E: {deltaE}')

    return WF