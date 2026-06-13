"""Where the core interblock methods exist."""
from bqskit.ir import Circuit, CircuitPoint
from bqskit.compiler import Compiler
from bqskit.passes import (
    NOOPPass,
    QSearchSynthesisPass,
    IfThenElsePass,
    ForEachBlockPass,
    WidthPredicate,
    SetRandomSeedPass
)

from IBSynth.interblock.frame import resolve_all_frames, construct_frames_parallel
from IBSynth.utils import count_large_gates
import asyncio

def calculate_error(error1, error2):
    return 1 - ((1 - error1) * (1 - error2))

def interblock(
    parted_circuit: Circuit,
    inclusion_function,
    compiler: Compiler | None = None,
    return_error=True,
    workflow=None,
    current_error: float = 0.0,
    error_bound: float = 1E-3
):
    if workflow is None:
        # Basic workflow
        workflow = [
            SetRandomSeedPass(0),
            ForEachBlockPass(   # NEEDED
                [
                    IfThenElsePass(WidthPredicate(2), NOOPPass(), QSearchSynthesisPass())
                ],
                replace_filter="less-than-multi",
            )
        ]

    target_points = [
        point for point in parted_circuit._dag.keys() if inclusion_function(point)
    ]
    target_points = list(set(target_points))
    
    frames = asyncio.run(construct_frames_parallel(parted_circuit, target_points))
    
    frames = resolve_all_frames(frames, parted_circuit)

    target_points = [frame.target_point for frame in frames]

    interblocks_points = []
    original_interblocks = []
    interblock_frame_locations = [] 

    for i, target_point in enumerate(target_points):
        frame = frames[i]
        if len(frame.interblock_regions) == 0:
            interblocks_points.append(None)
            original_interblocks.append(None)
            interblock_frame_locations.append(None)
            continue

        frame_circuit = frame.circuit.copy()
        frame_circuit.set_params(frame.circuit.params)

        interblock_points = []
        interblocks = []
        frame_locs = [] 
        
        for interblock_region in frame.interblock_regions:
            if interblock_region is None or interblock_region.empty:
                interblock_points.append(None)
                interblocks.append(None)
                frame_locs.append(None)
                continue
                
            interblock_point = frame_circuit.fold(interblock_region)
            frame_interblock_op = frame_circuit.get_operation(interblock_point)
            
            frame_locs.append(list(frame_interblock_op.location))

            frame_interblock_circ = frame_interblock_op.gate._circuit.copy()
            frame_interblock_circ.set_params(frame_interblock_op.params)
            frame_interblock_circ.unfold_all()
            
            interblocks.append(frame_interblock_circ)
            interblock_points.append(interblock_point)

            frame_circuit.unfold_all()

        interblocks_points.append(interblock_points)
        original_interblocks.append(interblocks)
        interblock_frame_locations.append(frame_locs)
    
    compiler_provided = compiler is not None
    if not compiler_provided:
        compiler = Compiler()
            
    compile_jobs = {}
    for i in range(len(interblocks_points)):
        if interblocks_points[i] is None:
            continue
        for j in range(len(interblocks_points[i])):
            if interblocks_points[i][j] is None:
                continue
            
            orig_circ = original_interblocks[i][j]
            
            wrapped_circ = Circuit(orig_circ.num_qudits)
            wrapped_circ.append_circuit(orig_circ, list(range(orig_circ.num_qudits)), as_circuit_gate=True)
            
            job_id = compiler.submit(wrapped_circ, workflow)
            compile_jobs[(i, j)] = job_id

    synthesized_interblocks = {}
    for coord, job_id in compile_jobs.items():
        res_circ = compiler.result(job_id)
        res_circ.unfold_all() 
        synthesized_interblocks[coord] = res_circ
        
    if not compiler_provided:
        compiler.close()

    interblock_errors = []
    optimizaiton_res = []
    locations_in_lists = []
    
    for i in range(len(interblocks_points)):
        if interblocks_points[i] is None:
            continue
        for j in range(len(interblocks_points[i])):
            if interblocks_points[i][j] is None:
                continue
                
            org_interblock = original_interblocks[i][j]
            synth_interblock_circ = synthesized_interblocks[(i, j)]
            
            orig_count = count_large_gates(org_interblock)
            synth_count = count_large_gates(synth_interblock_circ)
            
            if synth_count >= orig_count:
                interblocks_points[i][j] = None
                continue
            
            interblock_errors.append(
                org_interblock.get_unitary().get_distance_from(synth_interblock_circ.get_unitary())
            )
            optimizaiton_res.append(orig_count - synth_count)
            locations_in_lists.append((i, j))

    curr_round_error = 0.0
    dont_continue = False
    for i, (diff, error, location_in_lists) in enumerate(sorted(zip(optimizaiton_res, interblock_errors, locations_in_lists), key=lambda x: x[0], reverse=True)):
        curr_round_error += error
        theoretical_error = calculate_error(current_error, curr_round_error)
        
        if theoretical_error > error_bound:
            interblocks_points[location_in_lists[0]][location_in_lists[1]] = None
            original_interblocks[location_in_lists[0]][location_in_lists[1]] = None
            dont_continue = True

    points_used = []
    final_circuit = Circuit(parted_circuit.num_qudits)
    
    all_claimed_macro_points = set()
    for frame in frames:
        all_claimed_macro_points.update(frame.block_points)
        
    block_to_ops = {}

    for target_point_idx, frame in enumerate(frames):
        if not frame.block_points:
            continue
            
        mapping = frame.qubit_mapping
        
        if len(frame.interblock_regions) > 0:
            interblock_points = interblocks_points[target_point_idx]
            
            for i, interblock_point in enumerate(interblock_points):
                if interblock_point is None or len(frame.interblock_regions) == 0:
                    continue
                if frame.interblock_regions[i] is None:
                    continue
                    
                interblock_res_circ = synthesized_interblocks[(target_point_idx, i)].copy()
                interblock_frame_loc = interblock_frame_locations[target_point_idx][i]
                
                frame.replace_region(
                    frame.interblock_regions[i],
                    interblock_res_circ,
                    interblock_frame_loc,
                )

        for pt in frame.circuit._dag.keys():
            op = frame.circuit.get_operation(pt)
            try:
                macro_point, internal_idx = frame._gate_registry[pt]
            except KeyError:
                macro_point, internal_idx = frame.target_point, pt[0]
                
            global_loc = [mapping[q] for q in op.location]
            
            if macro_point not in block_to_ops:
                block_to_ops[macro_point] = []
                
            block_to_ops[macro_point].append(
                (internal_idx, op.gate, global_loc, op.params)
            )

    # Make sure blocks are in order as they are in the circuit for correct reconstruction
    chronological_points = sorted(
        list(parted_circuit._dag.keys()), 
        key=lambda p: (p[0], p[1])
    )

    for macro_point in chronological_points:
        if macro_point in all_claimed_macro_points:
            if macro_point in block_to_ops:
                surviving_ops = sorted(block_to_ops[macro_point], key=lambda x: x[0])
                
                current_bundle = []
                
                def flush_bundle():
                    if not current_bundle: 
                        return
                    bundle_qubits = set()
                    for _, _, loc, _ in current_bundle:
                        bundle_qubits.update(loc)
                    bundle_qubits = sorted(list(bundle_qubits))
                    
                    if len(current_bundle) == 1 and type(current_bundle[0][1]).__name__ == 'CircuitGate':
                        _, gate, loc, params = current_bundle[0]
                        cycle = final_circuit.append_gate(gate, loc, params)
                        points_used.append(CircuitPoint(cycle, loc[0]))
                    else:
                        macro_circ = Circuit(len(bundle_qubits))
                        for _, gate, loc, params in current_bundle:
                            local_loc = [bundle_qubits.index(q) for q in loc]
                            macro_circ.append_gate(gate, local_loc, params)
                        
                        cycle = final_circuit.append_circuit(macro_circ, bundle_qubits, as_circuit_gate=True)
                        points_used.append(CircuitPoint(cycle, bundle_qubits[0]))
                    current_bundle.clear()

                for op_tuple in surviving_ops:
                    gate = op_tuple[1]
                    if type(gate).__name__ == 'CircuitGate':
                        flush_bundle() 
                        current_bundle.append(op_tuple)
                        flush_bundle() 
                    else:
                        current_bundle.append(op_tuple)
                        
                flush_bundle() 
                
        else:
            op = parted_circuit.get_operation(macro_point)
            cycle = final_circuit.append_gate(op.gate, op.location, op.params)
            points_used.append(CircuitPoint(cycle, op.location[0]))
    
    unfolded_final = final_circuit.copy()
    unfolded_final.unfold_all()
    unfolded_original = parted_circuit.copy()
    unfolded_original.unfold_all()

    if return_error:
        return final_circuit, sum(interblock_errors), points_used, len(synthesized_interblocks), dont_continue
    else:
        return final_circuit, points_used, len(synthesized_interblocks), dont_continue


