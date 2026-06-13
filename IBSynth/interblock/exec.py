import os
import time
import glob
import torch
import datetime
import pandas as pd
import numpy as np

from typing import List


from bqskit.ir import Circuit
from bqskit.compiler import Compiler
from bqskit.passes import (
    QuickPartitioner,
    ForEachBlockPass,
    QSearchSynthesisPass,
    WidthPredicate,
    IfThenElsePass,
    NOOPPass,
    SetRandomSeedPass,
)
from bqskit.compiler.basepass import BasePass

from IBSynth.interblock.core import interblock
from IBSynth.utils import count_large_gates, calculate_round_error
from IBSynth.interblock.inclusion import gen_incl_point_func, inclusion_all

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)


def calc_opt_circuit_under_error(
    org_circ: Circuit,
    res_circ: Circuit,
    error_cap: float = 1e-5,
    error_cur: float = 0.0,
):

    opt_res = 0
    block_data_list = []
    over_cap = False

    for idx, (org_block, syn_block) in enumerate(zip(org_circ, res_circ)):
        org_cnt = count_large_gates(org_block)
        syn_cnt = count_large_gates(syn_block)
        diff = org_cnt - syn_cnt
        opt_res += syn_cnt

        if diff > 0:
            block_error = float(
                org_block.get_unitary().get_distance_from(syn_block.get_unitary())
            )

            score = diff / max(block_error, 1e-16)

            block_data_list.append(
                {
                    "idx": idx,
                    "org_block": org_block,
                    "syn_block": syn_block,
                    "diff": diff,
                    "error": block_error,
                    "score": score,
                }
            )

    sorted_blocks = sorted(block_data_list, key=lambda b: b["score"], reverse=True)

    final_circuit = Circuit(res_circ.num_qudits)
    round_error = 0
    syn_idx = set()
    error_by_reduction = {}

    for b in sorted_blocks:
        temp_round_error = round_error + b["error"]
        temp_total_error = calculate_round_error(error_cur, temp_round_error)

        diff = b["diff"]
        if diff not in error_by_reduction:
            error_by_reduction[diff] = {
                "accepted_count": 0,
                "accepted_error": 0.0,
                "rejected_count": 0,
                "rejected_error": 0.0,
            }

        if temp_total_error > error_cap:
            over_cap = True
            error_by_reduction[diff]["rejected_count"] += 1
            error_by_reduction[diff]["rejected_error"] += b["error"]
        else:
            round_error = temp_round_error
            syn_idx.add(b["idx"])
            error_by_reduction[diff]["accepted_count"] += 1
            error_by_reduction[diff]["accepted_error"] += b["error"]

    for i, (org_block, syn_block) in enumerate(zip(org_circ, res_circ)):
        if i in syn_idx:
            final_circuit.append(syn_block)
        else:
            final_circuit.append(org_block)

    final_error = calculate_round_error(round_error, error_cur)

    return final_circuit, final_error, over_cap


def run_interblock(
    circuit,
    seed: int = 0,
    allowed_error: float = 1e-5,
    gate_red_perc: float = 0.01,
    workflow: List[BasePass] = [
        ForEachBlockPass(
            [
                IfThenElsePass(
                    WidthPredicate(2),
                    NOOPPass(),
                    QSearchSynthesisPass(success_threshold=1e-8),
                )
            ],
            calculate_error_bound=True,
            replace_filter="less-than-multi",
        ),
    ],
    compiler: Compiler | None = None,
):
    """Run interblock over a circuit with a capped error under specific seed."""
    data = {}
    n_rounds = 0

    workflow.insert(0, SetRandomSeedPass(seed))

    stime = time.time()

    starting_gate_count = count_large_gates(circuit)

    if not compiler:
        started_compiler = True
        compiler = Compiler()
    else:
        started_compiler = False

    parted_circuit = compiler.compile(circuit, [QuickPartitioner()])

    synth_part_circuit = compiler.compile(parted_circuit, workflow)

    n_rounds += 1

    synth_part_circuit, total_error, error_over = calc_opt_circuit_under_error(
        parted_circuit, synth_part_circuit, error_cap=allowed_error
    )

    opt_diff = starting_gate_count - count_large_gates(synth_part_circuit)

    opt_diff_small = opt_diff < (starting_gate_count * gate_red_perc)

    while not opt_diff_small and not error_over:
        synth_part_circuit_res, round_error, _, _, _ = interblock(
            parted_circuit=synth_part_circuit,
            inclusion_function=inclusion_all,
            compiler=compiler,
            return_error=True,
            workflow=workflow,
            current_error=total_error,
            error_bound=allowed_error,
        )

        opt_diff = count_large_gates(synth_part_circuit) - count_large_gates(
            synth_part_circuit_res
        )
        opt_diff_small = opt_diff < (starting_gate_count * gate_red_perc)

        total_error = calculate_round_error(total_error, round_error)
        if total_error > allowed_error:
            error_over = True

        synth_part_circuit = synth_part_circuit_res

        n_rounds += 1

    if started_compiler:
        compiler.close()

    synth_part_circuit.unfold_all()

    data["org_lgc"] = starting_gate_count
    data["res_lgc"] = count_large_gates(synth_part_circuit)

    data["total_time"] = time.time() - stime
    data["error"] = total_error
    data["n_rounds"] = n_rounds

    return synth_part_circuit, data
