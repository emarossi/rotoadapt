import numpy as np
from pyscf import gto, scf
import argparse

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

## INPUT VARIABLES

# Create parser
parser = argparse.ArgumentParser(description="rodoadapt script - returns energy opt trajectory, WF object, #measurements")

# Add arguments
parser.add_argument("--mol", type=str, required = True, help="Molecule (H2O, LiH)")
parser.add_argument("--AS", type=int, nargs=2, required = True, help="Active space nEL nMO")
parser.add_argument("--adapt_thr", type=float, default=5e-6, help="adapt layer threshold")
parser.add_argument("--opt_thr", type=float, default=1e-5, help="adapt optimization threshold")
parser.add_argument("--opt_max_iter", type=float, default=20, help="max number of optimization cycles")

# Parse arguments
args = parser.parse_args()

molecule = args.mol  # molecule specifics via string
AS = args.AS  # active space (nEL, nMO)
adapt_thr = args.adapt_thr
opt_thr = args.opt_thr
max_iter = args.opt_max_iter
 
## DEFINE MOLECULE IN PYSCF

if molecule == 'H2O':
    geometry = 'O 0.000000 0.000000 0.000000; H 0.960000 0.000000 0.000000; H -0.240365 0.929422 0.000000' #H2O equilibrium
    # geometry = 'O 0.000000  0.000000  0.000000; H  1.000000  0.000000  0.000000; H -0.250380  0.968150  0.000000' #H2O stretched    

if molecule == 'LiH':
    # geometry = 'H 0.000000 0.000000 0.000000; Li 1.595000 0.00000 0.000000' #LiH equilibrium
    geometry = 'H 0.000000 0.000000 0.000000; Li 3.0000 0.00000 0.000000' #LiH stretched


mol_obj = gto.Mole()
mol_obj.build(atom = geometry, basis = 'sto-3g')
hf_obj = scf.RHF(mol_obj)
hf_obj.kernel()

nEL = AS[0]
nMO = AS[1]

cas_obj = hf_obj.CASCI(nMO, nEL)
# cas_obj.conv_tol = 1e-14
# cas_obj.max_cycle_macro = 100
cas_obj.kernel()

print(f'Energy HF: {hf_obj.energy_tot()-mol_obj.enuc}, Energy CAS: {cas_obj.e_tot-mol_obj.enuc}')

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

## EXCITATION WITH RESPECT TO REFERENCE

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

# WF, en_traj, num_measures = rotoadapt_utils.rotoselect_opt(WF, H, pool_data, adapt_thr)
WF, en_traj, num_measures = rotoadapt_utils.rotoselect(WF, H, pool_data, adapt_thr)
# WF, en_traj, num_measures = rotoadapt_utils.rotoadapt(WF, H, pool_data, max_iter, adapt_thr, opt_thr)

## SAVING RELEVANT OBJECTS

import pickle

output = {'molecule': molecule,
          'num_metadata': {'adapt_thr': adapt_thr, 
                           'opt_thr': opt_thr, 
                           'opt_max_iter': max_iter},
          'ci_ref': cas_obj.e_tot-mol_obj.enuc, 
          'en_traj': np.array(en_traj), 
          'WF': WF,
          'num_measures': num_measures
          }

with open(f'gen-{molecule}-{nEL}_{nMO}-stretch-RS.pkl', 'wb') as f:
    pickle.dump(output, f)
