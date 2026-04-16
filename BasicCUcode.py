import cupy as cp
from cuquantum.pauliprop.experimental import (
    LibraryHandle,
    PauliExpansion,
    PauliExpansionOptions,
    PauliRotationGate,
    Truncation,
    get_num_packed_integers,
)

# Step 1: Create library handle
handle = LibraryHandle()

# Step 2: Prepare buffers for the initial observable
num_qubits = 4
ints_per_string = get_num_packed_integers(num_qubits)

xz_bits = cp.zeros((1000, 2 * ints_per_string), dtype=cp.uint64)
coeffs = cp.zeros((1000,), dtype=cp.float64)

# Encode Z_0 observable
xz_bits[0, ints_per_string] = 1  # Set Z bit for qubit 0
coeffs[0] = 1.0

# Step 3: Create a PauliExpansion
options = PauliExpansionOptions(blocking=True)
expansion = PauliExpansion(
    handle,
    num_qubits,
    num_terms=1,
    xz_bits=xz_bits,
    coeffs=coeffs,
    sort_order="little_endian_bitwise",
    options=options,
)

# Step 4: Apply gates (back-propagation through adjoint circuit)
gate = PauliRotationGate(angle=0.1, pauli_string="X", qubit_indices=(0,))
expansion = expansion.apply_gate(gate, adjoint=True)

# Step 5: Compute expectation value with zero state
result = expansion.trace_with_zero_state()
print(f"Expectation value: {result}")
