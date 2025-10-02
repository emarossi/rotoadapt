import numpy as np
from pyscf import gto, scf
import argparse
import os
import pickle
import pickle

# Utilities
from slowquant.molecularintegrals.integralfunctions import one_electron_integral_transform, two_electron_integral_transform
from slowquant.unitary_coupled_cluster.util import iterate_t1, iterate_t2
from slowquant.unitary_coupled_cluster.util import iterate_t1_generalized, iterate_t2_generalized

from slowquant.unitary_coupled_cluster.util import iterate_t1_generalized, iterate_t2_generalized

from slowquant.unitary_coupled_cluster.util import iterate_t1_generalized, iterate_t2_generalized


# Operators
from slowquant.unitary_coupled_cluster.operators import G1, G2
from slowquant.unitary_coupled_cluster.operators import hamiltonian_0i_0a
from slowquant.unitary_coupled_cluster.operator_state_algebra import propagate_state, expectation_value, construct_ups_state

# Wave function ansatz - unitary product state
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS

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

results_folder = os.path.join(parent_folder, "rotoadapt/rotoadapt_analysis")

## DEFINE MOLECULE IN PYSCF

if molecule == 'H2O':
    # geometry = 'O 0.000000 0.000000 0.000000; H 0.960000 0.000000 0.000000; H -0.240365 0.929422 0.000000' #H2O equilibrium
    geometry = 'O 0.000000  0.000000  0.000000; H  1.068895  1.461020  0.000000; H 1.068895  -1.461020  0.000000' #H2O stretched (symmetric - 1.81 AA)

if molecule == 'LiH':
    # geometry = 'H 0.000000 0.000000 0.000000; Li 1.595000 0.00000 0.000000' #LiH equilibrium
    geometry = 'H 0.000000 0.000000 0.000000; Li 3.0000 0.00000 0.000000' #LiH stretched

if molecule == 'N2':
    geometry = 'N 0.000000 0.000000 0.000000; N 1.0980 0.00000 0.000000' #N2 equilibrium
    # geometry = 'N 0.000000 0.000000 0.000000; N 2.0980 0.00000 0.000000' #N2 stretched

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
        "fuccsd",
        include_active_kappa=True,
    )

# Add energy evaluation counter attribute
WF.num_energy_evals = 0

# Define Hamiltonian
H = hamiltonian_0i_0a(
    WF.h_mo,
    WF.g_mo,
    WF.num_inactive_orbs,
    WF.num_active_orbs
)

## DEFINE EXCITATION POOL -> dictionary with data

pool_data = {
    "excitation indeces": [],
    "excitation type": [],
    "excitation operator": []
}

num_inactive_so = WF.num_inactive_spin_orbs # use it to rescale operator indeces to the active space

## EXCITATION WITH RESPECT TO HF REFERENCE

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

## CALCULATE ROTOADAPT

# Add Rotomeasurements

if full_opt == True:
    WF, en_traj, rdm1_traj = rotoadapt_utils.rotoselect_opt(WF, pool_data, cas_en)    # rotoselect + full VQE optimzation

if full_opt == False:
    WF, en_traj, rdm1_traj = rotoadapt_utils.rotoselect(WF, pool_data, cas_en)  # rotoselect - no optimization

# SAVING RELEVANT OBJECTS

import pickle

# Create pickleable WF object representation
wf_data = {
    'num_params': WF.ups_layout.n_params,
    'excitation_indices': [idx.tolist() if hasattr(idx, 'tolist') else idx for idx in WF.ups_layout.excitation_indices],
    'excitation_types': WF.ups_layout.excitation_operator_type,
    'thetas': WF.thetas.copy(),
    'final_energy': WF.energy_elec
}

output = {'molecule': molecule,
          'ci_ref': cas_obj.e_tot-mol_obj.enuc, # CASCI reference energy
          'ci_ref': cas_obj.e_tot-mol_obj.enuc, # CASCI reference energy
          'en_traj': np.array(en_traj), # array of electronic energie shape=(#layers)
          'wf_data': wf_data, # Essential WF information instead of full object
          'num_measures': WF.num_energy_evals
          }

# with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-stretch-RS_OPT.pkl'), 'wb') as f:
#     pickle.dump(output, f)

# Create results directory if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

with open(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-stretch-RS.pkl'), 'wb') as f:
    pickle.dump(output, f)
