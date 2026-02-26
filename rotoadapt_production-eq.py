import numpy as np
from pyscf import gto, scf
import argparse
import os
import pickle

# Utilities
from slowquant.molecularintegrals.integralfunctions import one_electron_integral_transform, two_electron_integral_transform
from slowquant.unitary_coupled_cluster.util import iterate_t1, iterate_t2
from slowquant.unitary_coupled_cluster.util import iterate_t1_generalized, iterate_t2_generalized


# Operators
from slowquant.unitary_coupled_cluster.operators import G1, G2
from slowquant.unitary_coupled_cluster.operators import hamiltonian_0i_0a, hamiltonian_full_space
from slowquant.unitary_coupled_cluster.operator_state_algebra import propagate_state, expectation_value, construct_ups_state

# Wave function ansatz - unitary product state
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS

# Qiskit utils to get number of measurements
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

# Functions for rotoadapt
import rotoadapt_utils

## INPUT VARIABLES

# Create parser
parser = argparse.ArgumentParser(description="rodoadapt script - returns energy opt trajectory, WF object, #measurements")

# Add arguments
parser.add_argument("--mol", type=str, required = True, help="Molecule (H2O, LiH, BeH2, H6)")
parser.add_argument("--AS", type=int, nargs=2, required = True, help="Active space nEL nMO")
parser.add_argument("--gen", action="store_true", help="Generalized excitation operators")
parser.add_argument("--po", action="store_true", help="unitary parameter optimization")
parser.add_argument("--oo", action="store_true", help="orbital optimization")
parser.add_argument("--eff", action="store_true", help="Rotoselect efficient")

# Parse arguments
args = parser.parse_args()

molecule = args.mol  # molecule specifics via string
AS = args.AS  # active space (nEL, nMO)
gen = args.gen
po = args.po
oo = args.oo
eff = args.eff

# Getting path to current and parent folder
parent_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
results_folder = os.path.join(parent_folder, "rotoadapt_analysis")
 
## DEFINE MOLECULE IN PYSCF

if molecule == 'H2O':
    geometry = 'O 0.000000 0.000000 0.000000; H 0.960000 0.000000 0.000000; H -0.240365 0.929422 0.000000' #H2O equilibrium

if molecule == 'LiH':
    geometry = 'H 0.000000 0.000000 0.000000; Li 1.595000 0.00000 0.000000' #LiH equilibrium

if molecule == 'BeH2':
    geometry = 'Be 0.000000 0.000000 0.000000; H 1.33376 0.000000 0.000000; H -1.33376 0.000000 0.000000' #BeH2 equilibrium

if molecule == 'N2':
    # geometry = 'N 0.000000 0.000000 0.000000; N 1.0980 0.00000 0.000000' #N2 equilibrium
    geometry = 'N 0.000000 0.000000 0.000000; N 2.0980 0.00000 0.000000' #N2 stretched

if molecule == 'H6': # stretched H6
    geometry = "H -7.500000 0.000000 0.000000; H -4.500000 0.000000 0.000000; H -1.500000 0.000000 0.000000; H 1.500000 0.000000 0.000000; H 4.500000 0.000000 0.000000; H 7.500000 0.000000 0.000000"

# # BeH2 INSERTION PROBLEM
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

h_mo = one_electron_integral_transform(mo_coeff,h_ao)
g_mo = two_electron_integral_transform(mo_coeff,g_ao)

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

#Energy Hamiltonian Fermionic operator
Hamiltonian = hamiltonian_full_space(
    WF.h_mo, 
    WF.g_mo, 
    WF.num_orbs
)

#define mapper
mapper = JordanWignerMapper()

# Number of Pauli strings for Hamiltonian -> cost Hamiltonian evaluation
NHam_qubit = len(mapper.map(FermionicOp(Hamiltonian.get_qiskit_form(WF.num_orbs), WF.num_spin_orbs)).paulis)

## DEFINE EXCITATION POOL -> dictionary with data

if oo == True:
    pool_data = rotoadapt_utils.pool_D(WF, so_ir, gen, eff)
    print('D pool')

if oo == False:
    pool_data = rotoadapt_utils.pool_SD(WF, so_ir, gen, eff)
    print('SD pool')


## CALCULATE ROTOADAPT

# Add Rotomeasurements

# RS-full
if po == True and oo == False and eff == False:

    WF, en_traj, rdm1_traj, rdm2_traj = rotoadapt_utils.rotoselect_opt(WF, pool_data, cas_en, True, False)    # rotoselect + full VQE optimzation
    
    # Count number of measurements (#layers * 4 * #op_pool * #Pauli_strings_H) + VQE cost
    cost_pool = int((WF.ups_layout.n_params)*4*len(pool_data['excitation indeces'])*NHam_qubit)

# RS
if po == False and oo == False and eff == False:

    WF, en_traj, rdm1_traj, rdm2_traj = rotoadapt_utils.rotoselect(WF, pool_data, cas_en)    # rotoselect + full VQE optimzation

    cost_pool = int((WF.ups_layout.n_params)*4*len(pool_data['excitation indeces'])*NHam_qubit)

# oo-RS
if po == False and oo == True and eff == False:

    WF, en_traj, rdm1_traj, rdm2_traj = rotoadapt_utils.rotoselect_opt(WF, pool_data, cas_en, False, True)    # rotoselect + full VQE optimzation

    cost_pool = int((WF.ups_layout.n_params)*4*len(pool_data['excitation indeces'])*NHam_qubit)   

# oo-RS-full
if po == True and oo == True and eff == False:

    WF, en_traj, rdm1_traj, rdm2_traj = rotoadapt_utils.rotoselect_opt(WF, pool_data, cas_en, True, True)    # rotoselect + full VQE optimzation
    
    # Count number of measurements (#layers * 4 * #op_pool * #Pauli_strings_H) + VQE cost
    cost_pool = int((WF.ups_layout.n_params)*4*len(pool_data['excitation indeces'])*NHam_qubit)

# RSe-full
if po == True and oo == False and eff == True:

    WF, en_traj, cost_pool, rdm1_traj, rdm2_traj = rotoadapt_utils.rotoselect_efficient_opt(WF, pool_data, cas_en, po, oo)

# RSe
if po == False and oo == False and eff == True:
    WF, en_traj, cost_pool, rdm1_traj, rdm2_traj = rotoadapt_utils.rotoselect_efficient(WF, pool_data, cas_en, po, oo)

# oo-RSe-full
if po == True and oo == True and eff == True:
    WF, en_traj, cost_pool, rdm1_traj, rdm2_traj = rotoadapt_utils.rotoselect_efficient_opt_oo(WF, pool_data, cas_en, po, oo)

# oo-RSe
if po == False and oo == True and eff == True:
    WF, en_traj, cost_pool, rdm1_traj, rdm2_traj = rotoadapt_utils.rotoselect_efficient_opt_oo(WF, pool_data, cas_en, po, oo)

cost_VQE = int(WF.num_energy_evals*NHam_qubit)

num_en_evals = cost_pool + cost_VQE

print(f'COST POOL: {cost_pool} - COST VQE: {cost_VQE}')

# SAVING RELEVANT OBJECTS

output = {'molecule': molecule,
          'ref_data': {'elec_en_ref': cas_obj.e_tot-mol_obj.enuc,
                       'nuc_en_ref': mol_obj.enuc,
                       'rdm1_ref': cas_rdm1,
                       }, # CASCI reference data
          'en_traj': np.array(en_traj), # array of electronic energie shape=(#layers)
          'rdm1_traj': rdm1_traj, # rdm1 over the whole trajectory WF object
          'rdm2_traj': rdm2_traj, # rdm1 over the whole trajectory WF object
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

if gen == True:

    # RS-full
    if po == True and oo == False and eff == False:

        with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-eq-RS-full-gen.pkl'), 'wb') as f:
            pickle.dump(output, f)

    # RS
    elif po == False and oo == False and eff == False:

        with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-eq-RS-gen.pkl'), 'wb') as f:
            pickle.dump(output, f)

    # oo-RS
    elif po == False and oo == True and eff == False:

        with open(os.path.join(results_folder, f'oo-{molecule}-{nEL}_{nMO}-eq-RS-gen.pkl'), 'wb') as f:
            pickle.dump(output, f)

    # oo-RS-full
    elif po == True and oo == True and eff == False:

        with open(os.path.join(results_folder, f'oo-{molecule}-{nEL}_{nMO}-eq-RS-full-gen.pkl'), 'wb') as f:
            pickle.dump(output, f)

    # RSe-full
    elif po == True and oo == False and eff == True:

        with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-eq-RSe-full-gen.pkl'), 'wb') as f:
            pickle.dump(output, f)

    # RSe
    elif po == False and oo == False and eff == True:

        with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-eq-RSe-gen.pkl'), 'wb') as f:
            pickle.dump(output, f)

    # oo-RSe-full
    elif po == True and oo == True and eff == True:

        with open(os.path.join(results_folder, f'oo-{molecule}-{nEL}_{nMO}-eq-RSe-full-gen.pkl'), 'wb') as f:
            pickle.dump(output, f)

    # oo-RSe
    elif po == False and oo == True and eff == True:

        with open(os.path.join(results_folder, f'oo-{molecule}-{nEL}_{nMO}-eq-RSe-gen.pkl'), 'wb') as f:
            pickle.dump(output, f)

else:

    if po == True and oo == False:

        with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-eq-RS-full.pkl'), 'wb') as f:
            pickle.dump(output, f)

    elif po == False and oo == False:

        with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-eq-RS.pkl'), 'wb') as f:
            pickle.dump(output, f)

    elif po == False and oo == True:

        with open(os.path.join(results_folder, f'oo-{molecule}-{nEL}_{nMO}-eq-RS.pkl'), 'wb') as f:
            pickle.dump(output, f)

    elif po == True and oo == True:

        with open(os.path.join(results_folder, f'oo-{molecule}-{nEL}_{nMO}-eq-RS-full.pkl'), 'wb') as f:
            pickle.dump(output, f)
