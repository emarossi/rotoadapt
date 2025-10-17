import numpy as np
from pyscf import gto, scf
import matplotlib.pyplot as plt
import argparse
import os
import pickle

# Utilities
from slowquant.molecularintegrals.integralfunctions import one_electron_integral_transform, two_electron_integral_transform
from slowquant.unitary_coupled_cluster.util import iterate_t1, iterate_t2

# Operators
from slowquant.unitary_coupled_cluster.operators import G1, G2
from slowquant.unitary_coupled_cluster.operators import hamiltonian_0i_0a
from slowquant.unitary_coupled_cluster.operator_state_algebra import propagate_state, expectation_value, construct_ups_state

# Wave function ansatz - unitary product state
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS

# Functions for rotoadapt
import rotoadapt_utils

# Functions for rotoadapt
import rotoadapt_utils

## INPUT VARIABLES

# Create parser
parser = argparse.ArgumentParser(description="rodoadapt script - returns energy opt trajectory, WF object, #measurements")

# Add arguments
parser.add_argument("--mol", type=str, required = True, help="Molecule (H2O, LiH)")
parser.add_argument("--AS", type=int, nargs=2, required = True, help="Active space nEL nMO")
parser.add_argument("--gen", type=bool, default = False, help="Generalized excitation operators")
parser.add_argument("--full_opt", type=bool, default = False, help="full VQE optimization")

# Parse arguments
args = parser.parse_args()

molecule = args.mol  # molecule specifics via string
AS = args.AS  # active space (nEL, nMO)
gen = args.gen
full_opt = args.full_opt

# Getting path to current and parent folder
parent_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
results_folder = os.path.join(parent_folder, "rotoadapt/rotoadapt_analysis")

## DEFINE MOLECULE IN PYSCF

if molecule == 'H2O':
    # geometry = 'O 0.000000 0.000000 0.000000; H 0.960000 0.000000 0.000000; H -0.240365 0.929422 0.000000' #H2O equilibrium
    geometry = 'O 0.000000  0.000000  0.000000; H  1.068895  1.461020  0.000000; H 1.068895  -1.461020  0.000000' #H2O stretched (symmetric - 1.81 AA)
    geometry = 'O 0.000000  0.000000  0.000000; H  1.068895  1.461020  0.000000; H 1.068895  -1.461020  0.000000' #H2O stretched (symmetric - 1.81 AA)

if molecule == 'LiH':
    # geometry = 'H 0.000000 0.000000 0.000000; Li 1.595000 0.00000 0.000000' #LiH equilibrium
    geometry = 'H 0.000000 0.000000 0.000000; Li 3.0000 0.00000 0.000000' #LiH stretched

if molecule == 'N2':
    geometry = 'N 0.000000 0.000000 0.000000; N 2.0980 0.00000 0.000000' #N2 stretched

if molecule == 'BeH2':
    # geometry = 'Be 0.000000 0.000000 0.000000; H 1.34000 0.00000 0.000000; H -1.34000 0.00000 0.000000' #BeH2 equilibrium
    geometry = 'Be 0.000000 0.000000 0.000000; H 2.34000 0.00000 0.000000; H -2.34000 0.00000 0.000000' #BeH2 stretched


mol_obj = gto.Mole()
mol_obj.build(atom = geometry, basis = 'sto-3g', symmetry='c2v')
hf_obj = scf.RHF(mol_obj)
hf_obj.kernel()

# Getting the IR of the spin orbitals
so_ir = [int(mo_ir) for mo_ir in hf_obj.get_orbsym(hf_obj.mo_coeff) for _ in range(2)]

nEL = AS[0]
nMO = AS[1]

cas_obj = hf_obj.CASCI(nMO, nEL)
# cas_obj.conv_tol = 1e-14
# cas_obj.max_cycle_macro = 100
cas_obj.kernel()

cas_en = cas_obj.e_tot-mol_obj.enuc

print(f'Energy HF: {hf_obj.energy_tot()-mol_obj.enuc}, Energy CAS: {cas_en}')

# Getting integrals in MO basis

h_ao = mol_obj.intor('int1e_kin')+mol_obj.intor('int1e_nuc')
g_ao = mol_obj.intor('int2e')
mo_coeff = hf_obj.mo_coeff

## DEFINING SLOWQUANT OBJECT

# Define WaveFunction via SlowQuant
WF = WaveFunctionUPS(
        hf_obj.mol.nelectron,
        AS,  #(num_active_electrons, num_active_orbitals)
        hf_obj.mo_coeff,
        h_ao,
        g_ao,
        "fuccsd",
        include_active_kappa=True,
    )

# Add energy evaluation counter attribute
WF.num_energy_evals = 0

en_traj = [hf_obj.energy_tot()-mol_obj.enuc]
rdm1_traj = [WF.rdm1]

pool_data = rotoadapt_utils.pool(WF, so_ir, gen)

def do_adapt(WF, maxiter, epoch=1e-6 , orbital_opt: bool = False):
    '''Run Adapt VQE algorithm

    args:
        maxiter: max number of VQE iteration
        epoch: gradient variation threshold
        orbital_opt: enable orbital optimization
    '''
    
    nloop = 0

    # ADAPT ANSATZ + VQE
    for j in range(maxiter):

        #Energy Hamiltonian Fermionic operator
        Hamiltonian = hamiltonian_0i_0a(
            WF.h_mo,
            WF.g_mo,
            WF.num_inactive_orbs,
            WF.num_active_orbs,
        )

        #Apply operator to state -> obtain new state (list of operators, state, info on CI space, active space params)
        H_ket = propagate_state([Hamiltonian], WF.ci_coeffs, WF.ci_info, WF.thetas, WF.ups_layout)

        grad = []

        #GRADIENTS
        # grad = adapt_utils.gradient_parallel(WF, H_ket, pool_data)
        for T in pool_data["excitation operator"]:

            gr = expectation_value(WF.ci_coeffs,[T],  H_ket,
                                WF.ci_info, WF.thetas, WF.ups_layout)
            gr -= expectation_value(H_ket,[T],  WF.ci_coeffs,
                                WF.ci_info, WF.thetas, WF.ups_layout)
            grad.append(gr)
        
        print()
        print("------GP Printing Grad and Excitation Pool")
        print("------GP #################################")
        print(
                f"------GP{str("").center(72)}"
            )
        print(
                f"------GP{str("-----------------------------------------------------------------------------------").center(72)}"
            )
        print(
                f"------GP{str("Grad").center(27)} | {str("Excitation Pool indices").center(18)} | {str("Excitation Pool type").center(27)}"
            )
        for i in range(len(grad)):

            print(
                f"------GP{str(grad[i]).center(27)} | {str(pool_data["excitation indeces"][i]).center(18)} | {pool_data["excitation type"][i].center(27)}"
            )

        print()
        print("### Index of Max grad :: ", end=" ")
        print(np.argmax(np.abs(grad)))
        print("### Number of excitation gradient > %e :: "%(epoch), end=" ")
        print( (np.abs(grad) > epoch).sum())

        #Getting maximum gradient
        max_arg = np.argmax(np.abs(grad))

        #Check if gradient is smaller than smallest improvement (epoch)
        if(np.max(np.abs(grad)) < epoch):
            nloop = j
            break

        #Update ansatz with new excitation operator (corresponding to max gradient)
        WF.ups_layout.excitation_indices.append(np.array(pool_data["excitation indeces"][max_arg])-WF.num_inactive_spin_orbs) # rescale indeces to stay only in active space (?)
        WF.ups_layout.excitation_operator_type.append(pool_data["excitation type"][max_arg])
        #del excitation_pool[max_arg]
        #del excitation_pool_type[max_arg]
        WF.ups_layout.n_params += 1
        
        # add theta parameter for new operator
        WF._thetas.append(0.0)
        # np.append(WF._thetas, 0.0)

        # VQE optimization
        if full_opt == True:
            WF.run_wf_optimization_1step("slsqp", orbital_optimization=orbital_opt, opt_last=False) # full VQE optimization

        if full_opt == False:
            WF.run_wf_optimization_1step("slsqp", orbital_optimization=orbital_opt, opt_last=True) # Optimize only last unitary

        deltaE_adapt = np.abs(cas_en-WF.energy_elec)
        rdm1_traj.append(WF.rdm1)

        if deltaE_adapt < epoch:
            en_traj.append(WF.energy_elec)
            # Final printout
            print('----------------------')
            print(f'FINAL RESULT - Energy: {en_traj[-1]} - #Layers: {WF.ups_layout.n_params}')
            print('EXCITATION SEQUENCE')
            for i, (op_idx, op_type) in enumerate(zip(WF.ups_layout.excitation_indices, WF.ups_layout.excitation_operator_type)):
                print(f'LAYER {i}: {op_idx} | {op_type}')
            break

        else:
            en_traj.append(WF.energy_elec)
            print('#########CI COEFFS########')
            print(WF.ci_coeffs)
            print('#########################')

            print()
            print("------TP Printing the Optimised Theta")
            print("------TP ############################")

            print(
                    f"------TP {str("Thetas").center(27)} | {str("UPS Layout indices").center(18)} | {str("Excitation indices").center(18)} | {str("UPS Layout type").center(27)}"
            )
        # for i in range(len(self._thetas)):

            # print(
                # f"------TP {str(self._thetas[i]).center(27)} | {str(self.ups_layout.excitation_indices[i]).center(18)} |{str(self.ups_layout.excitation_indices[i] + self.num_inactive_spin_orbs ).center(18)} | {self.ups_layout.excitation_operator_type[i].center(27)}"
            # )
    print("------TP  Number of Loop", end = " ")
    print(nloop)

    return WF, en_traj, rdm1_traj

# Define epoch for chemical accuracy

epoch_ca = 1.6e-3

WF, en_traj, rdm1_traj = do_adapt(WF, epoch=epoch_ca, maxiter=30)

output = {'molecule': molecule,
          'ci_ref': cas_obj.e_tot-mol_obj.enuc, # CASCI reference energy
          'en_traj': np.array(en_traj), # array of electronic energie shape=(#layers)
          'rdm1_traj': rdm1_traj, # rdm1 over the whole trajectory WF object
          'num_measures': WF.num_energy_evals
          }

## OUTPUT ONLY LAST OPTIMIZATION

if full_opt == True:
    with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-stretch-GR.pkl'), 'wb') as f:
        pickle.dump(output, f)

if full_opt == False:
    with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-stretch-GR_last_opt.pkl'), 'wb') as f:
        pickle.dump(output, f)    
