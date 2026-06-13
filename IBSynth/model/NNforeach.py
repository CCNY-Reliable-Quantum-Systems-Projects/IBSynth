import time
import torch
import torch.nn as nn
import numpy as np

from typing import List, Callable
from einops import rearrange

from bqskit.ir import Circuit
from bqskit.utils.math import canonical_unitary
from bqskit.passes import NOOPPass
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.gates.parameterized.pauli import PauliGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.runtime import get_runtime
from bqskit.compiler.basepass import _sub_do_work, BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passdata import PassData
from bqskit.compiler.workflow import Workflow, WorkflowLike

from IBSynth.model.features import get_block_fv, get_eigenphase_variance_batch, get_entanglement_entropy_batch, get_interaction_density_batch, get_pauli_rank_batch, get_schmidt_rank_batch
from IBSynth.config import get_device
from IBSynth.model.classifier import BinaryClassifier
from IBSynth.model.autoencoder import UnitaryAutoencoder

def inference(
    subcircuits: List[Circuit],
    model: BinaryClassifier,
    autoencoder: UnitaryAutoencoder,
    scaler,
    threshold: float,
    canonical: bool = True,
) -> List[int]:
    
    device = get_device()

    model.eval()
    if autoencoder:
        autoencoder.eval()
        autoencoder.to(device)

    raw_matrices = []
    other_features_list = []
    for circ in subcircuits:
        mat = circ.get_unitary().numpy
        if canonical:
            mat = canonical_unitary(mat)
        raw_matrices.append(mat)
        other_features_list.append(get_block_fv(circ))
        
    matrices_np = np.array(raw_matrices)
    other_features_tensor = torch.stack(other_features_list).to(device)
    
    pauli_ranks = get_pauli_rank_batch(matrices_np)
    entropies = get_entanglement_entropy_batch(matrices_np)
    schmidt_ranks = get_schmidt_rank_batch(matrices_np)
    variances = get_eigenphase_variance_batch(matrices_np)
    densities = get_interaction_density_batch(subcircuits)
    
    heu_base_np = np.column_stack([
        pauli_ranks, entropies, schmidt_ranks, variances, densities
    ])
        

    with torch.no_grad():
        complex_tensor = torch.from_numpy(matrices_np).to(dtype=torch.cfloat, device=device)
        real_input = torch.stack([complex_tensor.real, complex_tensor.imag], dim=1).float()
        x_flat = rearrange(real_input, "b c h w -> b (c h w)")
        nerf_output = autoencoder.nerf(x_flat)
        encoded_unitary_np = autoencoder.encoder(nerf_output).cpu().numpy()
        
    other_features_np = other_features_tensor.cpu().numpy()
    feat_np = np.concatenate([encoded_unitary_np, heu_base_np, other_features_np], axis=1)
    
    scaled_feat = scaler.transform(feat_np)
    scaled_tensor = torch.from_numpy(scaled_feat).float().to(device)
    
    with torch.no_grad():
        outputs = model(scaled_tensor)
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
    probs = np.atleast_1d(probs)
    predictions = (probs > threshold).astype(int).tolist()
    
    return predictions


ReplaceFilterFn = Callable[[Circuit, Operation], bool]


class NNForEachBlockPass(BasePass):
    key = "NNForEachBlockPass_data"

    def __init__(
        self,
        loop_body: WorkflowLike,
        model: nn.Module = None,
        threshold: float = 0.5,
        autoencoder: nn.Module = None,
        scaler=None,
        canonical: bool = True,
        replace_filter: ReplaceFilterFn | str = "always",
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.autoencoder = autoencoder
        self.scaler = scaler
        self.canonical = canonical
        self.replace_filter = replace_filter
        self.workflow = Workflow(loop_body)

    def default_collection_filter(self, op: Operation) -> bool:
        return isinstance(
            op.gate, (CircuitGate, ConstantUnitaryGate, VariableUnitaryGate, PauliGate)
        )

    async def run(self, circuit: Circuit, data: PassData) -> None:
        t_start = time.perf_counter()
        replace_filter = (
            self.replace_filter
            if not isinstance(self.replace_filter, str)
            else lambda new, old: True
        )

        if self.key not in data:
            data[self.key] = []

        blocks = [
            (cycle, op)
            for cycle, op in circuit.operations_with_cycles()
            if self.default_collection_filter(op)
        ]
        if not blocks:
            data[self.key].append({})
            return

        model_ = data.model
        coupling_graph = data.connectivity

        subcircuits, block_datas = [], []
        subcirc_locs = []
        for i, (cycle, op) in enumerate(blocks):
            if isinstance(op.gate, CircuitGate):
                subcircuit = op.gate._circuit.copy()
                subcircuit.set_params(op.params)
            else:
                subcircuit = Circuit.from_operation(op)

            subradixes = [circuit.radixes[q] for q in op.location]
            subnumbering = {op.location[i]: i for i in range(len(op.location))}
            submodel = MachineModel(
                len(op.location),
                coupling_graph.get_subgraph(op.location, subnumbering),
                model_.gate_set,
                subradixes,
            )

            block_data = PassData(subcircuit)
            block_data["subnumbering"] = subnumbering
            block_data["model"] = submodel
            block_data["point"] = CircuitPoint(cycle, op.location[0])
            block_data.seed = data.seed

            subcircuits.append(subcircuit)
            block_datas.append(block_data)
            if len(op.location) == 3:
                subcirc_locs.append(list(op.location))

        t_overhead_setup = time.perf_counter() - t_start
        t_inf_start = time.perf_counter()

        # Vectorized ML Inference
        valid_subcircs = [sc for sc in subcircuits]
        part_decisions = []
        if valid_subcircs:
            part_decisions = inference(
                valid_subcircs,
                subcirc_locs,
                self.model,
                self.autoencoder,
                self.scaler,
                self.threshold,
                self.canonical,
            )

        t_inference = time.perf_counter() - t_inf_start

        workflows = [
            self.workflow if dec == 1 else NOOPPass() for dec in part_decisions
        ]

        t_synth_start = time.perf_counter()
        results = await get_runtime().map(
            _sub_do_work, workflows, subcircuits, block_datas
        )
        t_synthesis = time.perf_counter() - t_synth_start

        t_teardown_start = time.perf_counter()
        completed_subcircuits, completed_block_datas = zip(*results)

        points, ops = [], []
        error_sum = 0.0
        for i, (cycle, op) in enumerate(blocks):
            subcircuit = completed_subcircuits[i]
            block_data = completed_block_datas[i]
            if replace_filter(subcircuit, op):
                points.append(CircuitPoint(cycle, op.location[0]))
                ops.append(
                    Operation(
                        CircuitGate(subcircuit, True), op.location, subcircuit.params
                    )
                )
                error_sum += block_data.error

        circuit.batch_replace(points, ops)
        t_overhead_teardown = time.perf_counter() - t_teardown_start

        timing_data = {
            "overhead_time": t_overhead_setup + t_overhead_teardown,
            "inference_time": t_inference,
            "synthesis_time": t_synthesis,
            "total_pass_time": time.perf_counter() - t_start,
            "decisions": part_decisions,
        }
        data[self.key].append(timing_data)
        data.update_error_mul(error_sum)
