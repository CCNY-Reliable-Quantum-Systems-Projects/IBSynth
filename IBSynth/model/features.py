import torch
import numpy as np
import itertools

from bqskit.ir import Circuit

from IBSynth.utils import count_large_gates
from IBSynth.config import get_device


def get_block_fv(block: Circuit):
    return torch.tensor(
        [count_large_gates(block)], dtype=torch.float32, device=get_device()
    )


_I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _precompute_paulis(n_qubits):
    paulis = []
    for pauli_tuple in itertools.product([_I, _X, _Y, _Z], repeat=n_qubits):
        p = pauli_tuple[0]
        for i in range(1, n_qubits):
            p = np.kron(p, pauli_tuple[i])
        paulis.append(p)
    return np.array(paulis)


PAULI_CACHE_ARR = {
    1: _precompute_paulis(1),
    2: _precompute_paulis(2),
    3: _precompute_paulis(3),
}


def get_pauli_rank_batch(matrices, atol=1e-7):
    """Vectorized calculation of Pauli rank using Einstein Summation."""
    n_qubits = int(np.log2(matrices.shape[1]))
    if n_qubits not in PAULI_CACHE_ARR:
        return [0] * matrices.shape[0]

    paulis = PAULI_CACHE_ARR[n_qubits]
    n = matrices.shape[1]

    traces = np.einsum("iab,jba->ji", paulis, matrices)
    coeffs = traces / n
    ranks = np.sum(np.abs(coeffs) > atol, axis=1)
    return ranks.tolist()


def get_entanglement_entropy_batch(matrices, tol=1e-10):
    """Vectorized SVD calculation of Entanglement Entropy."""
    n_qubits = int(np.log2(matrices.shape[1]))
    if n_qubits > 12:
        return [0.0] * matrices.shape[0]

    states = matrices[:, :, 0]
    n_a = n_qubits // 2
    n_b = n_qubits - n_a

    reshaped_states = states.reshape(-1, 2**n_b, 2**n_a)  # Shape: (N, 4, 2)
    singular_values = np.linalg.svd(reshaped_states, compute_uv=False)
    probabilities = singular_values**2

    safe_probs = np.where(probabilities > tol, probabilities, 1.0)
    entropies = -np.sum(
        np.where(probabilities > tol, probabilities * np.log2(safe_probs), 0.0), axis=1
    )
    return entropies.tolist()


def get_schmidt_rank_batch(matrices, tol=1e-10):
    """Vectorized tensor reshaping and SVD for Schmidt rank."""
    N = matrices.shape[0]
    n_qubits = int(np.log2(matrices.shape[1]))
    d_a, d_b = 2 ** (n_qubits // 2), 2 ** (n_qubits - n_qubits // 2)

    u_tensor = matrices.reshape(N, d_b, d_a, d_b, d_a)
    reshuffled = u_tensor.transpose(0, 2, 4, 1, 3).reshape(N, d_a**2, d_b**2)

    singular_values = np.linalg.svd(reshuffled, compute_uv=False)
    ranks = np.sum(singular_values > tol, axis=1)
    return ranks.tolist()


def get_eigenphase_variance_batch(matrices):
    """Vectorized Eigenphase variance."""
    eigvals = np.linalg.eigvals(matrices)
    phases = np.angle(eigvals)
    variances = np.var(phases, axis=1)
    return variances.tolist()


def get_interaction_density_batch(circuits):
    """Native loop over circuits (Very fast since it requires no heavy math)."""
    densities = []
    for circ in circuits:
        edges = set()
        for op in circ:
            if len(op.location) == 2:
                edges.add(tuple(sorted(op.location)))
            elif len(op.location) > 2:
                for pair in itertools.combinations(op.location, 2):
                    edges.add(tuple(sorted(pair)))
        possible = (circ.num_qudits * (circ.num_qudits - 1)) / 2
        densities.append(len(edges) / possible if possible > 0 else 0.0)
    return densities
