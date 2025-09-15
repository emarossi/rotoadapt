import numpy as np
import copy
import multiprocessing as mp
import os
import itertools

from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value, construct_ups_state

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
    
    Returns:
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
    
    # Build companion matrix -> divide by p4 to make the polynomial monic
    Cmat = np.array([
        [0, 0, 0, -p0/p4],
        [1, 0, 0, -p1/p4],
        [0, 1, 0, -p2/p4],
        [0, 0, 1, -p3/p4]
    ])
    
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

def pool_evaluator(WF, pool_index, H, pool_data):
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

    # getting a copy of WF and adding one unitary to its layout

    pool_WF = copy.deepcopy(WF)
    pool_WF.ups_layout.excitation_indices.append(np.array(excitation_pool[pool_index])-pool_WF.num_inactive_spin_orbs)
    pool_WF.ups_layout.excitation_operator_type.append(excitation_pool_type[pool_index])
    pool_WF.ups_layout.n_params += 1
    pool_WF._thetas.append(0.0)
    pool_WF.ci_coeffs = construct_ups_state(pool_WF.ci_coeffs, pool_WF.ci_info, pool_WF.thetas, pool_WF.ups_layout)

    energies = []
    thetas = []

    # global minimum with companion matrix method --> TO DO: parallelize

    for l in range(0,5):
        current_thetas = pool_WF.thetas
        current_thetas[-1] += (2*np.pi*l)/5.5
        thetas.append(current_thetas[-1])
        pool_WF.thetas = current_thetas
        energies.append(float(expectation_value(pool_WF.ci_coeffs, [H], pool_WF.ci_coeffs, pool_WF.ci_info, pool_WF.thetas, pool_WF.ups_layout)))

    Thetas = np.array(thetas)
    Energies = np.array(energies)

    theta_min, E_min = optimizer(Thetas, Energies)

    return theta_min, E_min


def pool_parallel(WF, H, pool_data):
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
        results = pool.starmap(pool_evaluator, [(WF, pool_index, H, pool_data) for pool_index in pool_idx_array])

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