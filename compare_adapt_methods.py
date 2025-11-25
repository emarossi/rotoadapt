# Comparison script for BAI-based ADAPT-VQE vs Standard ADAPT-VQE
# Use the local version of SlowQuant
import sys
import importlib
# Clear any system-installed slowquant modules
for module in list(sys.modules.keys()):
    if module.startswith('slowquant'):
        del sys.modules[module]
# Add SlowQuant_copy to path FIRST
sys.path.insert(0, '/Users/rick/rotoadapt/SlowQuant_copy')

import numpy as np
from pyscf import gto, scf
import matplotlib.pyplot as plt
import argparse
import os
import copy

# Qiskit imports for circuit-based approach
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
from slowquant.qiskit_interface.operators_circuits import single_excitation, double_excitation

## INPUT VARIABLES

# Create parser
parser = argparse.ArgumentParser(description="Compare BAI-based vs Standard ADAPT-VQE")
parser.add_argument("--mol", type=str, required=True, help="Molecule (H2O, LiH, N2)")
parser.add_argument("--AS", type=int, nargs=2, required=True, help="Active space nEL nMO")
parser.add_argument("--gen", type=bool, default=False, help="Generalized excitation operators")
parser.add_argument("--adapt_thr", type=float, default=5e-6, help="adapt layer threshold")
parser.add_argument("--opt_thr", type=float, default=1e-5, help="adapt optimization threshold")
parser.add_argument("--opt_max_iter", type=float, default=20, help="max number of optimization cycles")
parser.add_argument("--save_plots", type=str, default=None, help="Directory to save plots (optional)")

# Parse arguments
args = parser.parse_args()

molecule = args.mol
AS = args.AS
gen = args.gen
adapt_thr = args.adapt_thr
opt_thr = args.opt_thr
max_iter = args.opt_max_iter
save_dir = args.save_plots

## DEFINE MOLECULE IN PYSCF

if molecule == 'H2O':
    geometry = 'O 0.000000  0.000000  0.000000; H  1.068895  1.461020  0.000000; H 1.068895  -1.461020  0.000000'
elif molecule == 'LiH':
    geometry = 'H 0.000000 0.000000 0.000000; Li 3.0000 0.00000 0.000000'
elif molecule == 'H4':
    geometry = 'H 0.000000 0.000000 0.000000; H 1.000000 0.000000 0.000000; H 2.000000 0.000000 0.000000; H 3.000000 0.000000 0.000000'
elif molecule == 'N2':
    geometry = 'N 0.000000 0.000000 0.000000; N 2.0980 0.00000 0.000000'

mol_obj = gto.Mole()
mol_obj.build(atom=geometry, basis='sto-3g', symmetry='c2v')
hf_obj = scf.RHF(mol_obj)
hf_obj.kernel()

# Getting the IR of the spin orbitals
so_ir = [int(mo_ir) for mo_ir in hf_obj.get_orbsym(hf_obj.mo_coeff) for _ in range(2)]

nEL = AS[0]
nMO = AS[1]

# Try to compute CASCI energy, but handle errors gracefully
casci_available = False
try:
    cas_obj = hf_obj.CASCI(nMO, nEL)
    cas_obj.kernel()
    cas_en = cas_obj.e_tot - mol_obj.enuc
    casci_available = True
    print(f'Energy HF: {hf_obj.energy_tot()-mol_obj.enuc}, Energy CAS: {cas_en}')
except (AssertionError, ValueError) as e:
    print(f'Warning: Could not compute CASCI energy ({e})')
    print(f'Energy HF: {hf_obj.energy_tot()-mol_obj.enuc}')
    print('Using HF energy as reference (CASCI will be set to HF energy)')
    cas_en = hf_obj.energy_tot() - mol_obj.enuc

# Getting integrals in MO basis
h_ao = mol_obj.intor('int1e_kin') + mol_obj.intor('int1e_nuc')
g_ao = mol_obj.intor('int2e')
mo_coeff = hf_obj.mo_coeff

# Shared function to rebuild circuit
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

# Standard ADAPT-VQE
def do_adapt_standard(WF, pool_data, cas_en, maxiter, epoch=1e-6, orbital_opt: bool = False):
    adapt_selected_indices = []
    adapt_selected_types = []
    nloop = 0
    measurements_per_iteration = []
    cumulative_measurements = []
    total_measurements = 0
    en_traj = [WF.energy_elec]
    
    for j in range(maxiter):
        Hamiltonian = hamiltonian_0i_0a(WF.h_mo, WF.g_mo, WF.num_inactive_orbs, WF.num_active_orbs)
        grad = []
        
        print(f"\n[Standard] ADAPT Step {j}: Calculating gradients for {len(pool_data['excitation operator'])} operators...")
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
        
        if max_grad < epoch:
            print(f"\n[Standard] Converged! Gradient {max_grad:.6e} < threshold {epoch:.6e}")
            nloop = j
            break
        
        adapt_selected_indices.append(pool_data['excitation indeces'][max_arg])
        adapt_selected_types.append(pool_data['excitation type'][max_arg])
        
        rebuild_adapt_circuit(WF, adapt_selected_indices, adapt_selected_types)
        
        current_thetas = WF.thetas
        WF.thetas = current_thetas + [0.0]
        
        WF.run_wf_optimization_1step("slsqp", orbital_optimization=orbital_opt)
        
        deltaE_adapt = np.abs(cas_en - WF.energy_elec)
        en_traj.append(WF.energy_elec)
        
        print(f"  [Standard] Energy: {WF.energy_elec:.8f} Ha, Delta: {deltaE_adapt:.6e} Ha, Measurements: {measurements_this_iter:,}")
        
        if deltaE_adapt < epoch:
            print(f'\n[Standard] FINAL - Energy: {WF.energy_elec:.8f} Ha, Delta: {deltaE_adapt:.6e} Ha')
            break
    
    print(f"\n[Standard] ADAPT-VQE completed: {nloop} iterations, {len(adapt_selected_indices)} operators")
    print(f"[Standard] Total measurements: {total_measurements:,}")
    
    return WF, en_traj, measurements_per_iteration, cumulative_measurements, adapt_selected_indices

# BAI-based ADAPT-VQE
def do_adapt_BAI(WF, pool_data, cas_en, maxiter, epoch=1e-6, orbital_opt: bool = False):
    adapt_selected_indices = []
    adapt_selected_types = []
    nloop = 0
    measurements_per_iteration = []
    cumulative_measurements = []
    total_measurements = 0
    en_traj = [WF.energy_elec]
    
    for j in range(maxiter):
        Hamiltonian = hamiltonian_0i_0a(WF.h_mo, WF.g_mo, WF.num_inactive_orbs, WF.num_active_orbs)
        rounds = 0
        active_arms = [i for i in range(len(pool_data['excitation operator']))]
        x = 0.005
        target_accuracy = 0.001
        total_candidates = len(pool_data['excitation operator'])
        temp_grad = np.zeros(total_candidates)
        measurements_this_iter = 0
        print(f"\n[BAI] ADAPT Step {j}: Starting with {len(pool_data['excitation operator'])} operators...")
        while len(active_arms) > 1 and rounds < 10:
            rounds += 1
            for idx in active_arms:
                T = pool_data["excitation operator"][idx] 
                commutator = Hamiltonian * T - T * Hamiltonian
                commutator_folded = commutator.get_folded_operator(
                    WF.num_inactive_orbs, WF.num_active_orbs, WF.num_virtual_orbs
                )
                gr = WF.QI.quantum_expectation_value(commutator_folded, overwrite_shots=512)
                temp_grad[idx] = (temp_grad[idx] * rounds + gr) / (rounds + 1)
                measurements_this_iter += 512

            temp_max_grad = np.max(np.abs(temp_grad))
            accuracy = x - (x - target_accuracy) * rounds / 10
            R = accuracy * 8
            temp_active_arms = []
            for idx in active_arms:
                if np.abs(temp_grad[idx]) + R >= temp_max_grad - R:
                    temp_active_arms.append(idx)
            
            active_arms = temp_active_arms
            print(f"  [BAI] Round {rounds}: {len(active_arms)} active operators remain.")

            max_arg = np.argmax(np.abs(temp_grad))
            max_grad = np.max(np.abs(temp_grad))
        
        total_measurements += measurements_this_iter
        measurements_per_iteration.append(measurements_this_iter)
        cumulative_measurements.append(total_measurements)

        print(f"  [BAI] Maximum gradient: {max_grad:.6e} (operator {max_arg})")
        
        if max_grad < epoch:
            print(f"\n[BAI] Converged! Gradient {max_grad:.6e} < threshold {epoch:.6e}")
            nloop = j
            break
        
        adapt_selected_indices.append(pool_data['excitation indeces'][max_arg])
        adapt_selected_types.append(pool_data['excitation type'][max_arg])
        
        rebuild_adapt_circuit(WF, adapt_selected_indices, adapt_selected_types)
        
        current_thetas = WF.thetas
        WF.thetas = current_thetas + [0.0]
        
        WF.run_wf_optimization_1step("slsqp", orbital_optimization=orbital_opt, maxiter=20)
        
        deltaE_adapt = np.abs(cas_en - WF.energy_elec)
        en_traj.append(WF.energy_elec)
        
        print(f"  [BAI] Energy: {WF.energy_elec:.8f} Ha, Delta: {deltaE_adapt:.6e} Ha, Measurements: {measurements_this_iter:,}")
        
        if deltaE_adapt < epoch:
            print(f'\n[BAI] FINAL - Energy: {WF.energy_elec:.8f} Ha, Delta: {deltaE_adapt:.6e} Ha')
            break
    
    print(f"\n[BAI] ADAPT-VQE completed: {nloop} iterations, {len(adapt_selected_indices)} operators")
    print(f"[BAI] Total measurements: {total_measurements:,}")
    
    return WF, en_traj, measurements_per_iteration, cumulative_measurements, adapt_selected_indices

# Initialize wave functions
def create_wavefunction():
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
    
    return WF

# Run both methods
print("\n" + "="*80)
print("COMPARISON: BAI-based ADAPT-VQE vs Standard ADAPT-VQE")
print("="*80)

epoch_ca = 1.6e-3

# Standard ADAPT-VQE
print("\n" + "="*80)
print("RUNNING STANDARD ADAPT-VQE")
print("="*80)
WF_standard = create_wavefunction()
pool_data_standard = adapt_utils.pool(WF_standard, so_ir, gen)
WF_standard, en_traj_standard, meas_per_iter_standard, cum_meas_standard, selected_standard = do_adapt_standard(
    WF_standard, pool_data_standard, cas_en, epoch=epoch_ca, maxiter=15
)

# BAI-based ADAPT-VQE
print("\n" + "="*80)
print("RUNNING BAI-based ADAPT-VQE")
print("="*80)
WF_BAI = create_wavefunction()
pool_data_BAI = adapt_utils.pool(WF_BAI, so_ir, gen)
WF_BAI, en_traj_BAI, meas_per_iter_BAI, cum_meas_BAI, selected_BAI = do_adapt_BAI(
    WF_BAI, pool_data_BAI, cas_en, epoch=epoch_ca, maxiter=15
)

# Print comparison summary
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)
print(f"Molecule: {molecule}, Active Space: {AS[0]} electrons, {AS[1]} orbitals")
print(f"Reference Energy: {cas_en:.8f} Ha", end="")
if casci_available:
    print(" (CASCI)")
else:
    print(" (HF - CASCI not available)")
print()
print("Standard ADAPT-VQE:")
print(f"  Final Energy:     {WF_standard.energy_elec:.8f} Ha")
print(f"  Energy Error:     {abs(WF_standard.energy_elec - cas_en):.6e} Ha")
print(f"  Iterations:       {len(selected_standard)}")
print(f"  Total Measurements: {sum(meas_per_iter_standard):,}")
print()
print("BAI-based ADAPT-VQE:")
print(f"  Final Energy:     {WF_BAI.energy_elec:.8f} Ha")
print(f"  Energy Error:     {abs(WF_BAI.energy_elec - cas_en):.6e} Ha")
print(f"  Iterations:       {len(selected_BAI)}")
print(f"  Total Measurements: {sum(meas_per_iter_BAI):,}")
print()
print("Comparison:")
print(f"  Energy Difference: {abs(WF_standard.energy_elec - WF_BAI.energy_elec):.6e} Ha")
print(f"  Measurement Ratio:  {sum(meas_per_iter_BAI) / sum(meas_per_iter_standard):.3f}x (BAI/Standard)")
print(f"  Measurement Savings: {(1 - sum(meas_per_iter_BAI) / sum(meas_per_iter_standard)) * 100:.1f}%")
print("="*80)

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'ADAPT-VQE Comparison: {molecule} ({AS[0]}e, {AS[1]}o)', fontsize=14, fontweight='bold')

# Energy trajectory vs iterations
ax1 = axes[0, 0]
iterations_standard = range(len(en_traj_standard))
iterations_BAI = range(len(en_traj_BAI))
ax1.plot(iterations_standard, en_traj_standard, 'o-', label='Standard ADAPT', linewidth=2, markersize=6)
ax1.plot(iterations_BAI, en_traj_BAI, 's-', label='BAI-based ADAPT', linewidth=2, markersize=6)
ax1.axhline(y=cas_en, color='r', linestyle='--', label='CASCI Energy', linewidth=1.5)
ax1.set_xlabel('ADAPT Iteration', fontsize=11)
ax1.set_ylabel('Energy (Ha)', fontsize=11)
ax1.set_title('Energy Convergence', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Energy trajectory vs cumulative measurements
ax2 = axes[0, 1]
# Pad cumulative measurements to match energy trajectory length (first point is HF, no measurements yet)
cum_meas_standard_plot = [0] + cum_meas_standard[:len(en_traj_standard)-1]
cum_meas_BAI_plot = [0] + cum_meas_BAI[:len(en_traj_BAI)-1]
# Ensure same length
min_len = min(len(cum_meas_standard_plot), len(en_traj_standard))
ax2.plot(cum_meas_standard_plot[:min_len], en_traj_standard[:min_len], 'o-', label='Standard ADAPT', linewidth=2, markersize=6)
min_len_BAI = min(len(cum_meas_BAI_plot), len(en_traj_BAI))
ax2.plot(cum_meas_BAI_plot[:min_len_BAI], en_traj_BAI[:min_len_BAI], 's-', label='BAI-based ADAPT', linewidth=2, markersize=6)
ax2.axhline(y=cas_en, color='r', linestyle='--', label='CASCI Energy', linewidth=1.5)
ax2.set_xlabel('Cumulative Measurements', fontsize=11)
ax2.set_ylabel('Energy (Ha)', fontsize=11)
ax2.set_title('Energy vs Measurements', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

# Measurements per iteration
ax3 = axes[1, 0]
iter_std_meas = range(len(meas_per_iter_standard))
iter_BAI_meas = range(len(meas_per_iter_BAI))
ax3.bar([i - 0.2 for i in iter_std_meas], meas_per_iter_standard, width=0.4, 
        label='Standard ADAPT', alpha=0.7, color='blue')
ax3.bar([i + 0.2 for i in iter_BAI_meas], meas_per_iter_BAI, width=0.4, 
        label='BAI-based ADAPT', alpha=0.7, color='orange')
ax3.set_xlabel('ADAPT Iteration', fontsize=11)
ax3.set_ylabel('Measurements per Iteration', fontsize=11)
ax3.set_title('Measurements per Iteration', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')
ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Cumulative measurements
ax4 = axes[1, 1]
iter_std_cum = range(len(cum_meas_standard))
iter_BAI_cum = range(len(cum_meas_BAI))
ax4.plot(iter_std_cum, cum_meas_standard, 'o-', label='Standard ADAPT', linewidth=2, markersize=6)
ax4.plot(iter_BAI_cum, cum_meas_BAI, 's-', label='BAI-based ADAPT', linewidth=2, markersize=6)
ax4.set_xlabel('ADAPT Iteration', fontsize=11)
ax4.set_ylabel('Cumulative Measurements', fontsize=11)
ax4.set_title('Cumulative Measurements', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()

# Save plots if requested
if save_dir:
    os.makedirs(save_dir, exist_ok=True)
    filename = f"adapt_comparison_{molecule}_{AS[0]}e_{AS[1]}o.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {filepath}")
else:
    plt.show()

print("\nComparison complete!")

