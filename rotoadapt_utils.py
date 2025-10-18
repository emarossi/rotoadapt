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
    Find the global minimum of energy landscape using the compation matrix method
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
            Cmat = cmat_builder([-p0/p2, 1])

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
        current_thetas[-1] = (2*np.pi*l)/5.5
        WF.thetas = current_thetas
        energies.append(WF.energy_elec)
        thetas.append((2*np.pi*l)/5.5)

    # WF.num_energy_evals += 4  # adding rotoselect energy evaluations

    Thetas = np.array(thetas)
    Energies = np.array(energies)

    # Find energy landscape and its global minimum
    theta_min, E_min = optimizer(Thetas, Energies)

    return theta_min, E_min


def pool_parallel(WF, H, pool_data, E_prev):
    '''
    Parallelizes energy estimations over the pool

    Arguments
        WF: WF object from SlowQuant from previous iteration
        H: hamiltonian
        pool_data: dictionary with info on pool operators

    Returns
        results: list of (theta_min, E_min) for each pool operator   
    '''
    excitation_pool = pool_data["excitation indeces"]
    pool_idx_array = np.arange(len(excitation_pool))

    # mp.set_start_method("spawn", force=True)

    with mp.Pool(processes=os.cpu_count()) as pool:
        results = pool.starmap(pool_evaluator, [(WF, pool_index, H, pool_data, E_prev) for pool_index in pool_idx_array])

    return results

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

def measurement_parallel_opt(WF, H, d):
    '''
    Parallelizes energy measurements for global minimum search

    Arguments
        WF: WF object from SlowQuant from previous iteration
        H: hamiltonian
        d: index of theta being optimized

    Returns
        thetas: list of thetas to construct companion matrix
        energies: list of energies for system in companion matrix method
    '''
    current_thetas = WF.thetas
    thetas_scan = [current_thetas[:d] + [current_thetas[d] + (2*np.pi*l)/5.5] + current_thetas[d+1:] for l in range(1,5)]
    thetas = [current_thetas[d] + (2*np.pi*l)/5.5 for l in range(1,5)]

    with mp.Pool(processes=os.cpu_count()) as pool:
            energies = pool.starmap(measure_energy_theta, [(WF, H, thetas_scan[i]) for i in range(0, len(thetas_scan))])
    
    return thetas, energies

def rotoadapt(WF, H, pool_data, max_epochs = 20, adapt_threshold = 1e-5, opt_threshold = 1e-5):
    '''
    Implements ADAPT ExcitationSolve algorithm from arXiv:2409.05939 (D.1, D.2)

    Arguments
        WF: wave function object from SlowQuant
        H: Hamiltonian operator from SlowQuant
        pool_data: dictionary with the data about the operator pool
        max_epochs: max number of theta optimization cycles
        adapt_threshold: min energy reduction upon addition of new layer
        opt_threshold: min energy reduction upon theta optimization

    Returns
        WF: final wave function object from SlowQuant
        en_traj: energy optimization trajectory    
    '''

    # Defining the excitation pool
    excitation_pool = pool_data["excitation indeces"]
    excitation_pool_type = pool_data["excitation type"]

    # Initialize previous energy and energy trajectory (here with HF energy)
    E_prev_adapt = float(expectation_value(WF.ci_coeffs, [H], WF.ci_coeffs, WF.ci_info, WF.thetas, WF.ups_layout))
    en_traj = [E_prev_adapt]

    num_measures = 0

    converged = False

    while converged == False:

        # Looping through pool operator -> get the best ansatz
        results = pool_parallel(WF, H, pool_data)
        theta_pool, energy_pool = zip(*results)
        op_index = np.argmin(energy_pool)

        # Update number of measurements
        num_measures += 5*len(excitation_pool)

        print('OPERATOR->', op_index)
        print(f'Theta {theta_pool[op_index]} - Energy {energy_pool[op_index]} - previous {E_prev_adapt}')

        deltaE_adapt = np.abs(energy_pool[op_index]-E_prev_adapt)

        # Updating WF with new operator
        WF.ups_layout.excitation_indices.append(np.array(excitation_pool[op_index])-WF.num_inactive_spin_orbs)
        WF.ups_layout.excitation_operator_type.append(excitation_pool_type[op_index])
        WF.ups_layout.n_params += 1
        WF._thetas.append(theta_pool[op_index])
        WF.ci_coeffs = construct_ups_state(WF.ci_coeffs, WF.ci_info, WF.thetas, WF.ups_layout)  

        if deltaE_adapt <= adapt_threshold:
            en_traj.append(float(energy_pool[op_index]))
            break

        # Optimization of all the thetas after adding new layer

        if WF.ups_layout.n_params == 1:
            E_min = energy_pool[op_index]
            deltaE_adapt = np.abs(E_prev_adapt-E_min)

        else:

            for epochs in range(max_epochs):

                for d in range(WF.ups_layout.n_params):

                    current_thetas = WF.thetas

                    if epochs == 0:
                        energies = [energy_pool[op_index]]
                        thetas = [theta_pool[op_index]]
                        E_prev_opt = energy_pool[op_index]

                    sample_thetas, sample_energies = measurement_parallel_opt(WF, H, d)

                    num_measures += 4

                    Thetas = np.array(thetas + sample_thetas)
                    Energies = np.array(energies + sample_energies)

                    theta_min, E_min = optimizer(Thetas, Energies)

                    current_thetas[d] = theta_min
                    WF.thetas = current_thetas
                    energies = [E_min]
                    thetas = [theta_min]

                deltaE_opt = np.abs(E_prev_opt-E_min)

                print(f'Layer {WF.ups_layout.n_params} - convergence at step {epochs}:', deltaE_opt)

                if deltaE_opt <= opt_threshold:
                    break
                else:
                    E_prev_opt = E_min


        deltaE_adapt = np.abs(E_min-E_prev_adapt)

        print(f'deltaE_adapt for layers {WF.ups_layout.n_params} parameters is: ', deltaE_adapt)

        if deltaE_adapt <= adapt_threshold:
            print(f'FINAL RESULT - Energy: {energy_pool[op_index]} - Previous: {E_prev_adapt} - Delta: {deltaE_adapt} - Theta: {theta_pool[op_index]}')
            en_traj.append(float(E_min))
            converged = True

        else:
            en_traj.append(float(E_min))
            E_prev_adapt = E_min
            print(f'energy trajectory at layer {WF.ups_layout.n_params} = {en_traj}')

    return WF, en_traj, num_measures

def rotoselect(WF, pool_data, cas_en, adapt_threshold = 1.6e-3):  # adapt_threshold for chemical accuracy
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

    converged = False

    while converged == False and WF.ups_layout.n_params <= 50:

        # Load new operator in the ansatz and initialize
        WF.ups_layout.excitation_indices.append((0,0))
        WF.ups_layout.excitation_operator_type.append(" ")
        WF.ups_layout.n_params += 1
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

        if deltaE_adapt < adapt_threshold:
            # Updating last layer with data from best operator
            WF.ups_layout.excitation_indices[-1] = excitation_pool[op_index]
            WF.ups_layout.excitation_operator_type[-1] = excitation_pool_type[op_index]
            thetas = WF.thetas
            thetas[-1] = theta_pool[op_index]
            WF.thetas = thetas

            # Appending energy and termination
            en_traj.append(float(energy_pool[op_index]))
            rdm1_traj.append(WF.rdm1)
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
            E_prev_adapt = float(energy_pool[op_index])  # with respect to previous layer

    return WF, en_traj, rdm1_traj

def rotoselect_opt(WF, pool_data, cas_en, adapt_threshold = 1.6e-3):  # adapt_threshold for chemical accuracy
    '''
    Constructs ansatz iteratively using Rotoselect algorithm
    Full VQE optimization after selection

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

    converged = False

    while converged == False:

        # Load new operator slot in the ansatz
        WF.ups_layout.excitation_indices.append((0,0))
        WF.ups_layout.excitation_operator_type.append(" ")
        WF.ups_layout.n_params += 1
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

        if deltaE_adapt < adapt_threshold:  
            # Updating last layer with data from best operator
            WF.ups_layout.excitation_indices[-1] = excitation_pool[op_index]
            WF.ups_layout.excitation_operator_type[-1] = excitation_pool_type[op_index]
            thetas = WF.thetas
            thetas[-1] = theta_pool[op_index]
            WF.thetas = thetas

            # Appending energy and termination (before full optimization)
            en_traj.append(float(energy_pool[op_index]))
            rdm1_traj.append(WF.rdm1)
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
            WF.run_wf_optimization_1step("slsqp", orbital_optimization=False)
            
            # Appending energy to trajectory and updating 'previous' energy for next iteration
            en_traj.append(WF.energy_elec)
            deltaE_adapt = np.abs(cas_en-en_traj[-1])            

            # Checking convergence to chemical accuracy
            if deltaE_adapt < adapt_threshold:
                rdm1_traj.append(WF.rdm1)
                converged = True

                # Final printout
                print('----------------------')
                print(f'FINAL RESULT - Energy: {en_traj[-1]} - #Layers: {WF.ups_layout.n_params}')
                print('EXCITATION SEQUENCE')
                for i, (op_idx, op_type) in enumerate(zip(WF.ups_layout.excitation_indices, WF.ups_layout.excitation_operator_type)):
                    print(f'LAYER {i}: {op_idx} | {op_type}')

            else:
                rdm1_traj.append(WF.rdm1)
                print(f'RESULT at layer {WF.ups_layout.n_params} - Energy: {WF.energy_elec} - Previous: {E_prev_adapt} - Delta: {deltaE_adapt} - Theta: {theta_pool[op_index]}')
                E_prev_adapt = en_traj[-1]

    return WF, en_traj, rdm1_traj