# Use the local version of SlowQuant
import sys
import importlib
# Clear any system-installed slowquant modules
for module in list(sys.modules.keys()):
    if module.startswith('slowquant'):
        del sys.modules[module]
# Add SlowQuant_copy to path FIRST
# Add SlowQuant_copy to path FIRST (relative to this script's location)
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, 'SlowQuant'))

import numpy as np
from pyscf import gto, scf
import matplotlib.pyplot as plt
import argparse
import os

# Qiskit imports for circuit-based approach
# Qiskit 2.x uses StatevectorSampler instead of Sampler
from qiskit.primitives import StatevectorSampler
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

# Utilities
from slowquant.molecularintegrals.integralfunctions import one_electron_integral_transform, two_electron_integral_transform
from slowquant.unitary_coupled_cluster.util import iterate_t1, iterate_t2

# Operators
from slowquant.unitary_coupled_cluster.operators import G1, G2
from slowquant.unitary_coupled_cluster.operators import hamiltonian_0i_0a

# Circuit-based wave function
from slowquant.qiskit_interface.circuit_wavefunction import WaveFunctionCircuit
from slowquant.qiskit_interface.interface import QuantumInterface

# Functions for rotoadapt
import adapt_utils

## INPUT VARIABLES

# Create parser
parser = argparse.ArgumentParser(description="Circuit-based ADAPT-VQE script")

# Add arguments
parser.add_argument("--mol", type=str, required=True, help="Molecule (H2O, LiH, N2)")
parser.add_argument("--AS", type=int, nargs=2, required=True, help="Active space nEL nMO")
parser.add_argument("--gen", type=bool, default=False, help="Generalized excitation operators")
parser.add_argument("--adapt_thr", type=float, default=5e-6, help="adapt layer threshold")
parser.add_argument("--opt_thr", type=float, default=1e-5, help="adapt optimization threshold")
parser.add_argument("--opt_max_iter", type=float, default=20, help="max number of optimization cycles")

# Parse arguments
args = parser.parse_args()

molecule = args.mol
AS = args.AS
gen = args.gen
adapt_thr = args.adapt_thr
opt_thr = args.opt_thr
max_iter = args.opt_max_iter

## DEFINE MOLECULE IN PYSCF

if molecule == 'H2O':
    geometry = 'O 0.000000  0.000000  0.000000; H  1.068895  1.461020  0.000000; H 1.068895  -1.461020  0.000000'

if molecule == 'LiH':
    geometry = 'H 0.000000 0.000000 0.000000; Li 3.0000 0.00000 0.000000'

if molecule == 'H4':
    geometry = 'H 0.000000 0.000000 0.000000; H 1.000000 0.000000 0.000000; H 2.000000 0.000000 0.000000; H 3.000000 0.000000 0.000000'

if molecule == 'N2':
    geometry = 'N 0.000000 0.000000 0.000000; N 2.0980 0.00000 0.000000'

mol_obj = gto.Mole()
mol_obj.build(atom=geometry, basis='sto-3g', symmetry='c2v')
hf_obj = scf.RHF(mol_obj)
hf_obj.kernel()

# Getting the IR of the spin orbitals
so_ir = [int(mo_ir) for mo_ir in hf_obj.get_orbsym(hf_obj.mo_coeff) for _ in range(2)]

nEL = AS[0]
nMO = AS[1]

cas_obj = hf_obj.CASCI(nMO, nEL)
cas_obj.kernel()

cas_en = cas_obj.e_tot - mol_obj.enuc

print(f'Energy HF: {hf_obj.energy_tot()-mol_obj.enuc}, Energy CAS: {cas_en}')

# Getting integrals in MO basis
h_ao = mol_obj.intor('int1e_kin') + mol_obj.intor('int1e_nuc')
g_ao = mol_obj.intor('int2e')
mo_coeff = hf_obj.mo_coeff

## DEFINING SLOWQUANT CIRCUIT-BASED OBJECT

sampler = StatevectorSampler()
mapper = JordanWignerMapper()

quantum_interface = QuantumInterface(
    primitive=sampler,
    ansatz="HF",
    mapper=mapper,
    ISA=False,
    shots=None,
)

WF = WaveFunctionCircuit(
    num_elec=hf_obj.mol.nelectron,
    cas=AS,
    mo_coeffs=hf_obj.mo_coeff,
    h_ao=h_ao,
    g_ao=g_ao,
    quantum_interface=quantum_interface,
    include_active_kappa=True,
)

en_traj = [hf_obj.energy_tot() - mol_obj.enuc]
pool_data = adapt_utils.pool(WF, so_ir, gen)

from slowquant.qiskit_interface.operators_circuits import single_excitation, double_excitation

adapt_selected_indices = []
adapt_selected_types = []

def rebuild_adapt_circuit(WF, selected_indices, selected_types):
    from qiskit_nature.second_q.circuit.library import HartreeFock
    
    num_orbs = WF.num_active_orbs
    mapper = WF.QI.mapper
    qc = HartreeFock(num_orbs, (0, 0), mapper)
    grad_param_R = {}
    
    for idx, (indices, exc_type) in enumerate(zip(selected_indices, selected_types)):
        theta = Parameter(f"theta_{idx}")
        
        if exc_type == "single":
            i, a = indices
            i_active = i - WF.num_inactive_spin_orbs
            a_active = a - WF.num_inactive_spin_orbs
            qc = single_excitation(i_active, a_active, num_orbs, qc, theta, mapper)
            grad_param_R[f"theta_{idx}"] = 2
        elif exc_type == "double":
            i, j, a, b = indices
            i_active = i - WF.num_inactive_spin_orbs
            j_active = j - WF.num_inactive_spin_orbs
            a_active = a - WF.num_inactive_spin_orbs
            b_active = b - WF.num_inactive_spin_orbs
            qc = double_excitation(i_active, j_active, a_active, b_active, num_orbs, qc, theta, mapper)
            grad_param_R[f"theta_{idx}"] = 2
    
    WF.QI.circuit = qc
    WF.QI.grad_param_R = grad_param_R
    WF.QI.param_names = [str(p) for p in qc.parameters]
    print(f"  Rebuilt circuit: {len(selected_indices)} operators, {qc.num_parameters} parameters")

def do_adapt(WF, maxiter, epoch=1e-6, orbital_opt: bool = False):
    global adapt_selected_indices, adapt_selected_types
    nloop = 0
    measurements_per_iteration = []
    cumulative_measurements = []
    total_measurements = 0
    
    for j in range(maxiter):
        Hamiltonian = hamiltonian_0i_0a(WF.h_mo, WF.g_mo, WF.num_inactive_orbs, WF.num_active_orbs)
        grad = []
        
        print(f"\nADAPT Step {j}: Calculating gradients for {len(pool_data['excitation operator'])} operators...")
        measurements_this_iter = 0
        for idx, T in enumerate(pool_data["excitation operator"]):
            commutator = Hamiltonian * T - T * Hamiltonian
            commutator_folded = commutator.get_folded_operator(
                WF.num_inactive_orbs, WF.num_active_orbs, WF.num_virtual_orbs
            )
            gr = WF.QI.quantum_expectation_value(commutator_folded, overwrite_shots=8192)
            grad.append(gr)
            measurements_this_iter += 8192
        
        max_arg = np.argmax(np.abs(grad))
        max_grad = np.max(np.abs(grad))
        
        total_measurements += measurements_this_iter
        measurements_per_iteration.append(measurements_this_iter)
        cumulative_measurements.append(total_measurements)
        
        print(f"  Maximum gradient: {max_grad:.6e} (operator {max_arg})")
        print(f"  Type: {pool_data['excitation type'][max_arg]}, Indices: {pool_data['excitation indeces'][max_arg]}")
        
        if max_grad < epoch:
            print(f"\nConverged! Gradient {max_grad:.6e} < threshold {epoch:.6e}")
            nloop = j
            break
        
        adapt_selected_indices.append(pool_data['excitation indeces'][max_arg])
        adapt_selected_types.append(pool_data['excitation type'][max_arg])

        # Save current optimized thetas BEFORE rebuilding circuit
        current_thetas = WF.thetas.copy() if WF.thetas else []

        print(f"  Adding operator to circuit...")
        rebuild_adapt_circuit(WF, adapt_selected_indices, adapt_selected_types)

        # Set thetas: previous optimized values + 0.0 for new operator
        WF.QI._parameters = current_thetas + [0.0]
        
        print(f"  Running VQE optimization...")
        WF.run_wf_optimization_1step("slsqp", orbital_optimization=orbital_opt)
        
        deltaE_adapt = np.abs(cas_en - WF.energy_elec)
        en_traj.append(WF.energy_elec)
        
        print(f"  Energy: {WF.energy_elec:.8f} Ha, Delta: {deltaE_adapt:.6e} Ha")
        
        if deltaE_adapt < epoch:
            print(f'\nFINAL - Energy: {WF.energy_elec:.8f} Ha, Delta: {deltaE_adapt:.6e} Ha')
            break
    
    print(f"\n{'='*60}")
    print(f"ADAPT-VQE completed: {nloop} iterations, {len(adapt_selected_indices)} operators")
    print(f"Final energy: {WF.energy_elec:.8f} Ha")
    print(f"Total measurements: {total_measurements:,}")
    print(f"{'='*60}\n")
    
    return WF, en_traj, measurements_per_iteration, cumulative_measurements

epoch_ca = 1.6e-3

print("\n" + "="*60)
print("Starting Circuit-Based ADAPT-VQE (Standard)")
print("="*60)
WF, en_traj, measurements_per_iter, cumulative_measurements = do_adapt(WF, epoch=epoch_ca, maxiter=15)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"HF Energy:      {hf_obj.energy_tot()-mol_obj.enuc:.8f} Ha")
print(f"CASCI Energy:   {cas_en:.8f} Ha")
print(f"ADAPT Energy:   {WF.energy_elec:.8f} Ha")
print(f"Energy Error:   {abs(WF.energy_elec - cas_en):.6e} Ha")
print(f"Circuit Depth:  {len(adapt_selected_indices)} operators")
print(f"Parameters:     {len(WF.thetas)}")
print("="*60)

## PLOT ENERGY ERROR VS TOTAL MEASUREMENTS
plt.figure(figsize=(8, 6))

# Calculate energy error (difference from CASCI reference)
# en_traj[0] is HF, en_traj[1:] are ADAPT iterations
en_error = np.abs(np.array(en_traj[1:]) - cas_en)

# Plot energy error vs cumulative measurements
plt.plot(cumulative_measurements[:len(en_error)], en_error, 'o-', label='ADAPT-VQE')
plt.axhline(y=epoch_ca, color='r', linestyle='--', label=f'Chemical accuracy ({epoch_ca})')

plt.xlabel('Total Measurements')
plt.ylabel('Energy Error (Hartree)')
plt.yscale('log')
plt.title(f'{molecule} ({nEL}e, {nMO}o) - ADAPT-VQE Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save figure
parent_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
results_folder = os.path.join(parent_folder, "rotoadapt")
os.makedirs(results_folder, exist_ok=True)
plt.savefig(os.path.join(results_folder, f'{molecule}-{nEL}_{nMO}-circuit-energy_vs_measurements_exact.png'), dpi=150)
plt.show()
