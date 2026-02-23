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
from slowquant.unitary_coupled_cluster.operators import G1, G2, commutator
from slowquant.unitary_coupled_cluster.operators import hamiltonian_0i_0a, hamiltonian_full_space
from slowquant.unitary_coupled_cluster.operator_state_algebra import propagate_state, expectation_value, construct_ups_state

# Wave function ansatz - unitary product state
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS

# Qiskit utils to get number of measurements
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

# Functions for rotoadapt
import adapt_utils
from rotoadapt_utils import rotosolve

## INPUT VARIABLES

# Create parser
parser = argparse.ArgumentParser(description="adapt script - returns energy opt trajectory, WF object, #measurements")

# Add arguments
parser.add_argument("--mol", type=str, required = True, help="Molecule (H2O, LiH)")
parser.add_argument("--AS", type=int, nargs=2, required = True, help="Active space nEL nMO")
parser.add_argument("--gen", action="store_true", help="Generalized excitation operators")
parser.add_argument("--po", action="store_true", help="Unitary parameter optimization")
parser.add_argument("--oo", action="store_true", help="Orbital optimization")

# Parse arguments
args = parser.parse_args()

molecule = args.mol  # molecule specifics via string
AS = args.AS  # active space (nEL, nMO)
gen = args.gen
po = args.po
oo = args.oo

# Getting path to current and parent folder
parent_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
results_folder = os.path.join(parent_folder, "rotoadapt_analysis")
 
## DEFINE MOLECULE IN PYSCF

if molecule == 'H2O':
    geometry = 'O 0.000000 0.000000 0.000000; H 0.960000 0.000000 0.000000; H -0.240365 0.929422 0.000000' #H2O equilibrium
    # geometry = 'O 0.000000  0.000000  0.000000; H  1.068895  1.461020  0.000000; H 1.068895  -1.461020  0.000000' #H2O stretched (symmetric - 1.81 AA) 

if molecule == 'LiH':
    geometry = 'H 0.000000 0.000000 0.000000; Li 1.595000 0.00000 0.000000' #LiH equilibrium
    # geometry = 'H 0.000000 0.000000 0.000000; Li 3.0000 0.00000 0.000000' #LiH stretched

if molecule == 'BeH2':
    geometry = 'Be 0.000000 0.000000 0.000000; H 1.33376 0.000000 0.000000; H -1.33376 0.000000 0.000000' #BeH2 equilibrium
    # geometry = 'Be 0.000000 0.000000 0.000000; H 1.33376 0.000000 0.000000; H -1.33376 0.000000 0.000000' #BeH2 triangular

if molecule == 'N2':
    geometry = 'N 0.000000 0.000000 0.000000; N 2.0980 0.00000 0.000000' #N2 stretched

if molecule == 'H6': #stretched H6
    geometry = "H -7.500000 0.000000 0.000000; H -4.500000 0.000000 0.000000; H -1.500000 0.000000 0.000000; H 1.500000 0.000000 0.000000; H 4.500000 0.000000 0.000000; H 7.500000 0.000000 0.000000"

## BeH2 INSERTION PROBLEM
# if 'BeH2' in molecule:

#     def Be_ins_coords(x_Be):
#         '''
#         Generates coordinates for BeH2 insertion
#         Be moving along x, H2 moving along z according to:
#         z_H = -0.46*x_Be+2.54
        
#         Arguments
#             x_Be: x coordinate of the Be atom
        
#         Returns
#             Be_xyz+H2_xyz: string with xyz coordinates of BeH2
#         '''
#         if x_Be <= 4:
#             z_H = -0.46*x_Be+2.54
#         else:
#             z_H = 0.7
        
#         # Converting into Angstroms
#         x_Be *= 0.529177
#         z_H *= 0.529177

#         Be_xyz = f'Be {x_Be:.6f} 0.000000 0.000000; '
#         H2_xyz = f'H 0.000000 0.000000 {np.abs(z_H):.6f}; H 0.000000 0.000000 -{np.abs(z_H):.6f}'

#         return Be_xyz+H2_xyz

#     geometry = Be_ins_coords(float(molecule.split('-')[1].strip()))

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
cas_rdm1 = cas_obj.make_rdm1()

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
        "adapt",
        include_active_kappa=True,
    )

Hamiltonian = hamiltonian_full_space(
        WF.h_mo, 
        WF.g_mo, 
        WF.num_orbs
    )

en_traj = [hf_obj.energy_tot()-mol_obj.enuc]
rdm1_traj = [WF.rdm1]

if oo == True:
    pool_data = adapt_utils.pool_D(WF, so_ir, gen)
    print('D pool')

if oo == False:
    pool_data = adapt_utils.pool_SD(WF, so_ir, gen)
    print('SD pool')

pool_Ncomm_qubit = 0

#define mapper
mapper = JordanWignerMapper()

# Number of Pauli strings for Hamiltonian -> cost Hamiltonian evaluation
NHam_qubit = len(mapper.map(FermionicOp(Hamiltonian.get_qiskit_form(WF.num_orbs), WF.num_spin_orbs)).paulis)

# Sum pauli strings for each commutator in the pool -> cost gradient evaluation in the pool
for T in pool_data["excitation operator"]:

    comm = commutator(T, Hamiltonian)
    pool_Ncomm_qubit += len(mapper.map(FermionicOp(comm.get_qiskit_form(WF.num_orbs), WF.num_spin_orbs)).paulis)


def do_adapt(WF, maxiter, epoch=1e-6):
    '''Run Adapt VQE algorithm
    
    args:
        maxiter: max number of VQE iteration
        epoch: gradient variation threshold
        orbital_opt: enable orbital optimization
    '''
    
    Hamiltonian = hamiltonian_full_space(
        WF.h_mo, 
        WF.g_mo, 
        WF.num_orbs
    )

    nloop = 0

    # ADAPT ANSATZ + VQE
    for j in range(maxiter):

        if oo == True:
            print('Orbital optimization - rebuilding the H...')

            Hamiltonian = hamiltonian_full_space(
                WF.h_mo, 
                WF.g_mo, 
                WF.num_orbs
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
        WF.ups_layout.n_params += 1
        WF.ups_layout.excitation_indices.append(pool_data["excitation indeces"][max_arg]) # rescale indeces to stay only in active space (?)
        WF.ups_layout.excitation_operator_type.append(pool_data["excitation type"][max_arg])
        WF.ups_layout.grad_param_R[f"p{WF.ups_layout.n_params:09d}"] = 2
        WF.ups_layout.param_names.append(f"p{WF.ups_layout.n_params:09d}")
        WF._thetas.append(0.0)

        # VQE optimization
        # from rotoadapt_utils import rotosolve

        # GB-full
        if po == True and oo == False:
            WF.run_wf_optimization_1step("bfgs", orbital_optimization = False) # full VQE optimization
            # WF = rotosolve(WF)
        
        # oo-GB-full
        if po == True and oo == True:
            WF.run_wf_optimization_1step("bfgs", orbital_optimization = True) # full VQE optimization

        # GB-last
        if po == False and oo == False:
            WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization = False, opt_last=True) # Optimize only last unitary

        # oo-GB-last
        if po == False and oo == True:
            WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization = False, opt_last=True) # Optimize only last unitary
            WF.run_orbital_optimization()


        deltaE_adapt = np.abs(cas_en-WF.energy_elec)
        rdm1_traj.append(WF.rdm1)

        if deltaE_adapt < epoch or WF.ups_layout.n_params >= 150:
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
            print(f'RESULT - Energy: {en_traj[-1]} - #Layers: {WF.ups_layout.n_params} - DeltaE: {deltaE_adapt}')
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
epoch_ca = 1e-5

WF, en_traj, rdm1_traj = do_adapt(WF, epoch=epoch_ca, maxiter=1000)

# Total cost: cost_pool_evaluation * num_layers + cost VQE
cost_pool = int(pool_Ncomm_qubit)*WF.ups_layout.n_params
# cost_pool = int((WF.ups_layout.n_params)*4*len(pool_data['excitation indeces'])*NHam_qubit)
cost_VQE = int(WF.num_energy_evals*NHam_qubit)

num_en_evals = cost_pool + cost_VQE
print(f'COST POOL: {cost_pool} - COST VQE: {int(WF.num_energy_evals*NHam_qubit)}')


output = {'molecule': molecule,
         'ref_data': {'elec_en_ref': cas_obj.e_tot-mol_obj.enuc,
                       'nuc_en_ref': mol_obj.enuc,
                       'rdm1_ref': cas_rdm1,
                       }, # CASCI reference data
          'en_traj': np.array(en_traj), # array of electronic energie shape=(#layers)
          'rdm1_traj': rdm1_traj, # rdm1 over the whole trajectory WF object
          'num_en_evals': {'num_en_evals': num_en_evals,
                           'cost_pool': cost_pool,
                            'cost_VQE': cost_VQE
                            },  # optimization total cost
          'ansatz_data': {'num_layers': WF.ups_layout.n_params,
                          'excitation_idx': WF.ups_layout.excitation_indices,
                          'excitation_op_type': WF.ups_layout.excitation_operator_type,
                          'thetas': WF.thetas,
                          },
          }

## OUTPUT

if gen == True:

    # GB-full
    if po == True and oo == False:

        with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-eq-GB-full-gen.pkl'), 'wb') as f:
            pickle.dump(output, f)

        # with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-str-GB-full-gen.pkl'), 'wb') as f:
        #     pickle.dump(output, f)

    # GB-last
    elif po == False and oo == False:

        with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-eq-GB-last-gen.pkl'), 'wb') as f:
            pickle.dump(output, f)

        # with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-str-GB-last-gen.pkl'), 'wb') as f:
        #     pickle.dump(output, f)

    # oo-GB-last
    elif po == False and oo == True:

        with open(os.path.join(results_folder, f'oo-{molecule}-{nEL}_{nMO}-eq-GB-last-gen.pkl'), 'wb') as f:
            pickle.dump(output, f)

        # with open(os.path.join(results_folder, f'oo-{molecule}-{nEL}_{nMO}-str-GB-last-gen.pkl'), 'wb') as f:
        #     pickle.dump(output, f)

    # oo-GB-full
    elif po == True and oo == True:

        with open(os.path.join(results_folder, f'oo-{molecule}-{nEL}_{nMO}-eq-GB-full-gen.pkl'), 'wb') as f:
            pickle.dump(output, f)

        # with open(os.path.join(results_folder, f'oo-{molecule}-{nEL}_{nMO}-str-GB-full-gen.pkl'), 'wb') as f:
        #     pickle.dump(output, f)

else:

    if po == True and oo == False:

        with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-GB-full.pkl'), 'wb') as f:
            pickle.dump(output, f)

        # with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-str-GB-full.pkl'), 'wb') as f:
        #     pickle.dump(output, f)

    elif po == False and oo == False:

        with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-GB-last.pkl'), 'wb') as f:
            pickle.dump(output, f)

        # with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-str-GB-last.pkl'), 'wb') as f:
        #     pickle.dump(output, f)

    elif po == False and oo == True:

        with open(os.path.join(results_folder, f'oo-{molecule}-{nEL}_{nMO}-GB-last.pkl'), 'wb') as f:
            pickle.dump(output, f)

        # with open(os.path.join(results_folder, f'oo-{molecule}-{nEL}_{nMO}-str-GB-last.pkl'), 'wb') as f:
        #     pickle.dump(output, f)

    elif po == True and oo == True:

        with open(os.path.join(results_folder, f'oo-{molecule}-{nEL}_{nMO}-GB-full.pkl'), 'wb') as f:
            pickle.dump(output, f)

        # with open(os.path.join(results_folder, f'oo-{molecule}-{nEL}_{nMO}-str-GB-full.pkl'), 'wb') as f:
        #     pickle.dump(output, f)