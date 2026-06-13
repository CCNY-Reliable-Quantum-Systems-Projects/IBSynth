from typing import List
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from bqskit.ir import Circuit, Operation, CircuitRegion
from bqskit.ir import CircuitPoint
from bqskit.ir.gates import CircuitGate

from IBSynth.utils import count_large_gates


def score_frame_interblock(circuit: Circuit, interblock_region: CircuitRegion):
    frame_interblock = circuit.copy()
    frame_interblock.set_params(circuit.params)
    frame_interblock.fold(interblock_region)
    total_score = 0
    for op in frame_interblock:
        if not isinstance(op.gate, CircuitGate):
            continue
        score = count_large_gates(op)
        total_score += score
    return total_score


class CircuitFrame:

    def __init__(
        self,
        block_points: List[CircuitPoint],
        blocks: List[Operation],
        target_point: CircuitPoint,
        interblock_size: int = 3,
    ):

        self._gate_registry: dict[CircuitPoint, CircuitPoint] = {}

        self.qubit_mapping = []
        for op in blocks:
            self.qubit_mapping += op.location
        self.qubit_mapping = list(set(sorted(self.qubit_mapping)))

        self.target_point = target_point

        self.num_qudits = len(self.qubit_mapping)
        self.interblock_size = interblock_size
        self.block_points = block_points

        (
            self.circuit,
            self.block_regions,
            self.interblock_regions,
            self.interblock_scores,
        ) = self.append_blocks_and_interblock(block_points, blocks, target_point)

    def append_blocks_and_interblock(
        self,
        block_points: List[CircuitPoint],
        blocks: List[Operation],
        target_point: CircuitPoint,
    ) -> None:
        """New interblock construction methodology (reducing number of surrounds needed)"""

        circuit = Circuit(len(self.qubit_mapping))

        block_regions = {}

        surround_points = []

        # Add blocks in order
        for block, block_point in zip(blocks, block_points):
            block_loc_frame = [
                self.qubit_mapping.index(qubit) for qubit in block.location
            ]

            block_regions[block_point] = []

            block_circ = block.gate._circuit.copy()
            block_circ.set_params(block.params)
            block_circ.unfold_all()

            for i, op in enumerate(block_circ):
                op_loc_frame = [block_loc_frame[i] for i in op.location]
                op_cycle = circuit.append_gate(op.gate, op_loc_frame, op.params)
                op_point = CircuitPoint(op_cycle, op_loc_frame[0])

                block_regions[block_point].append(op_point)

                self._gate_registry[op_point] = (block_point, i)

                if block_point == target_point:
                    if len(op.location) > 1:
                        if len(surround_points) == 0:
                            surround_points.append(op_point)
                            surround_points.append(op_point)
                        else:
                            surround_points[1] = op_point


            block_regions[block_point] = circuit.get_region(block_regions[block_point])

        surround_points = list(set(surround_points))

        interblock_regions = []
        interblock_scores = []

        while len(surround_points) != 0:
            surround_point = surround_points.pop(0)
            if surround_point is None:
                continue
            interblock_region = circuit.surround(surround_point, 3)

            if (
                interblock_region.num_qudits == self.interblock_size
                and circuit.is_valid_region(interblock_region)
                and interblock_region not in interblock_regions
            ):
                interblock_regions.append(interblock_region)
                interblock_region_score = score_frame_interblock(
                    circuit, interblock_region
                )
                interblock_scores.append(interblock_region_score)
            else:
                continue

            for i, region in enumerate(interblock_regions):
                if region == interblock_region or region is None:
                    continue

                if interblock_region.overlaps(region):
                    if interblock_scores[i] > interblock_scores[-1]:
                        interblock_regions.pop()
                        interblock_scores.pop()
                        break
                    else:
                        interblock_regions[i] = None
                        interblock_scores[i] = None

        interblock_regions = [
            interblock_region
            for interblock_region in interblock_regions
            if interblock_region is not None
        ]
        interblock_scores = [
            interblock_score
            for interblock_score in interblock_scores
            if interblock_score is not None
        ]

        return circuit, block_regions, interblock_regions, interblock_scores

    def reduce_frame(self):
        """
        Reduces the frame to only include blocks that overlap with the
        interblock (synthesis) region by leveraging the robust remove_points method.
        """

        points_to_remove = []

        # Identify all elementary gates (points) that belong to blocks
        # which DO NOT overlap with the interblock region.
        for block_point in self.block_points:
            # If a block's region does not overlap the interblock region,
            # mark all of its constituent gates for removal.
            block_region = self.block_regions[block_point]
            remove_block = True
            for interblock_region in self.interblock_regions:
                if interblock_region.overlaps(block_region):
                    remove_block = False

            if remove_block:
                points_to_remove += list(block_region.points)

        # Use our single, trusted remove_points method to do the actual removal
        # and metadata reconstruction. This avoids duplicating complex logic.
        self.remove_points(points_to_remove)

    def remove_points(self, remove_points: List[CircuitPoint]):
        """
        Rebuilds the frame's circuit, excluding the specified points, and
        correctly recalculates all associated metadata like regions and registries.

        ToDo: Make faster
        1. Can we purely remove the operations at those points in place? (not creating a new object?)
        2. If we must recreate a new object is there a faster way to do postprocessing?
        """
        if not remove_points:
            return

        new_circuit = Circuit(self.circuit.num_qudits)
        new_gate_registry = {}

        # Temporary storage for points belonging to each region after removal
        # The key is the original block_point, value is a list of new CircuitPoints
        temp_block_points_map = {bp: [] for bp in self.block_points}
        temp_interblocks_points = [[] for _ in self.interblock_regions]

        for point, _ in self.circuit._dag.items():
            if point in remove_points:
                continue

            registry_data = self._gate_registry.get(point)
            if registry_data is None:
                continue
                
            original_block_point, orig_internal_idx = registry_data

            op = self.circuit.get_operation(point)
            new_cycle_index = new_circuit.append(op)
            new_point = CircuitPoint(new_cycle_index, point[1])

            # Pass the full tuple to the new registry
            new_gate_registry[new_point] = (original_block_point, orig_internal_idx)
            temp_block_points_map[original_block_point].append(new_point)

            if len(self.interblock_regions) > 0:
                interblock_overlaps = [
                    interblock_region.overlaps(point)
                    for interblock_region in self.interblock_regions
                ]
                if any(interblock_overlaps):
                    interblock_idx_to_keep = 0
                    best_score = 0
                    for i, interblock_region in enumerate(self.interblock_regions):
                        if interblock_overlaps[i]:
                            if self.interblock_scores[i] > best_score:
                                interblock_idx_to_keep = i
                                best_score = self.interblock_scores[i]
                    temp_interblocks_points[interblock_idx_to_keep].append(new_point)

        # Now, commit the new state
        self.circuit = new_circuit
        self._gate_registry = new_gate_registry

        # Rebuild block_regions and block_points from scratch
        new_block_regions = {}
        new_block_points = []
        for block_point, points_list in temp_block_points_map.items():
            if points_list:  # Only keep blocks that still have gates
                new_block_points.append(block_point)
                new_block_regions[block_point] = self.circuit.get_region(points_list)

        self.block_points = new_block_points
        self.block_regions = new_block_regions

        # Rebuild interblock_region
        if temp_interblocks_points:
            self.interblock_regions = []
            self.interblock_scores = []
            for temp_interblock_points in temp_interblocks_points:
                try:
                    region = self.circuit.get_region(temp_interblock_points)
                except:
                    continue
                if region.empty:
                    continue
                score = score_frame_interblock(new_circuit, region)
                self.interblock_regions.append(region)
                self.interblock_scores.append(score)

        else:
            # If no interblock gates remain, the region becomes invalid
            self.interblock_regions = []
            self.interblock_scores = []


    def replace_region(self, region, subcircuit: Circuit, location):
        new_circuit = Circuit(self.circuit.num_qudits)
        subcircuit_added = False

        new_gate_registry = {}
        new_block_region_points = {block_point: [] for block_point in self.block_points}
        new_interblock_region_points = [[] for _ in self.interblock_regions]

        subcircuit_ready = {q: False for q in region.location}

        for point, _ in self.circuit._dag.items():
            block_point = self._gate_registry.get(point)
            op = self.circuit.get_operation(point)

            if region.overlaps(point):
                for q in op.location:
                    if q in subcircuit_ready:
                        subcircuit_ready[q] = True
                        
                if all(subcircuit_ready.values()) and not subcircuit_added:
                    cycle = new_circuit.append_circuit(
                        subcircuit, location, as_circuit_gate=True
                    )
                    if block_point:
                        new_pt = CircuitPoint(cycle, location[0])
                        new_gate_registry[new_pt] = block_point
                        if block_point not in new_block_region_points:
                            new_block_region_points[block_point] = []
                        new_block_region_points[block_point].append(new_pt)
                    subcircuit_added = True
                continue

            op_cycle = new_circuit.append(op)
            new_op_point = CircuitPoint(op_cycle, op.location[0])
            
            if block_point:
                new_gate_registry[new_op_point] = block_point
                if block_point not in new_block_region_points:
                    new_block_region_points[block_point] = []
                new_block_region_points[block_point].append(new_op_point)
                
            if len(self.interblock_regions) > 0:
                interblock_overlaps = [
                    (r is not None and r.overlaps(point)) 
                    for r in self.interblock_regions
                ]
                if any(interblock_overlaps):
                    interblock_idx_to_keep = 0
                    best_score = -1
                    for i, r in enumerate(self.interblock_regions):
                        if interblock_overlaps[i]:
                            if self.interblock_scores[i] > best_score:
                                interblock_idx_to_keep = i
                                best_score = self.interblock_scores[i]
                    new_interblock_region_points[interblock_idx_to_keep].append(new_op_point)

        self.circuit = new_circuit
        self._gate_registry = new_gate_registry
        
        new_block_regions = {}
        new_block_points = []
        for bp, op_points in new_block_region_points.items():
            if len(op_points) == 0:
                continue
            try:
                new_block_regions[bp] = new_circuit.get_region(op_points)
                new_block_points.append(bp)
            except:
                continue
        self.block_regions = new_block_regions
        self.block_points = new_block_points

        new_interblock_regions = []
        new_interblock_scores = []
        for i, op_points in enumerate(new_interblock_region_points):
            # Preserves array length to prevent IndexError in later rounds
            if len(op_points) == 0:
                new_interblock_regions.append(None)
                new_interblock_scores.append(0.0)
                continue
            try:
                new_interblock_region = new_circuit.get_region(op_points)
                new_interblock_score = score_frame_interblock(new_circuit, new_interblock_region)
                new_interblock_regions.append(new_interblock_region)
                new_interblock_scores.append(new_interblock_score)
            except:
                new_interblock_regions.append(None)
                new_interblock_scores.append(0.0)
                
        self.interblock_regions = new_interblock_regions
        self.interblock_scores = new_interblock_scores


async def construct_frames_parallel(
    parted_circuit: Circuit, target_points: list[CircuitPoint]
) -> List[CircuitFrame]:

    tasks = [(parted_circuit, tp) for tp in target_points]

    num_processes = cpu_count()

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_target_point, tasks)

    # Don't include None results
    interblock_frames = [frame for frame in results if frame is not None and len(frame.interblock_regions) > 0 and frame.circuit.num_qudits <= 7]

    return interblock_frames

def construct_frames(
    parted_circuit: Circuit, target_points: list[CircuitPoint], max_qubit = 7
) -> List[CircuitFrame]:

    tasks = [(parted_circuit, tp) for tp in target_points]
    
    results = [process_target_point(task) for task in tasks]

    interblock_frames = [frame for frame in results if frame is not None and len(frame.interblock_regions) > 0 and frame.circuit.num_qudits <= 7]

    return interblock_frames

def check_connection(
    parted_circuit: Circuit,
    start_point: CircuitPoint,
    interest_point: CircuitPoint
):
    try:
        parted_circuit.get_region([start_point, interest_point])
        return True
    except:
        return False
    
def check_big_connection(parted_circuit: Circuit, points: list):
    try:
        parted_circuit.get_region(points)
        return True
    except:
        return False


def process_target_point(args):
    """
    Worker function to find the best frame for a single target point.
    Takes a tuple of (parted_circuit, target_point) to work with Pool.map.
    """
    parted_circuit, target_point = args

    # Manage Predecessors
    prev_points = parted_circuit.prev(target_point)
    prev_points = [
        p for p in prev_points if check_connection(parted_circuit, target_point, p)
    ]

    # Manage Successors
    next_points = parted_circuit.next(target_point)
    next_points = [
        n for n in next_points if check_connection(parted_circuit, target_point, n)
    ]

    best_interblock_frame = None

    all_points = prev_points + next_points
    all_points.append(target_point)

    if not check_big_connection(parted_circuit, all_points):
        return None

    # Handle when target point has successors and predecessors
    if len(prev_points) > 0 and len(next_points) > 0:
        points = prev_points + [target_point] + next_points
        blocks = parted_circuit.get_operations(points)
        frame = CircuitFrame(points, blocks, target_point)
        if len(frame.interblock_regions) > 0:
            best_interblock_frame = frame

    # Handle if there are either no predecessors or successors
    elif len(prev_points) > 0:  # Only predecessors
        points = prev_points + [target_point]
        blocks = parted_circuit.get_operations(points)
        frame = CircuitFrame(points, blocks, target_point)
        if len(frame.interblock_regions) > 0:
            best_interblock_frame = frame

    elif len(next_points) > 0:  # Only successors
        points = [target_point] + next_points
        blocks = parted_circuit.get_operations(points)
        frame = CircuitFrame(points, blocks, target_point)
        if len(frame.interblock_regions) > 0:
            best_interblock_frame = frame

    if best_interblock_frame:
        best_interblock_frame.reduce_frame() # only do once save time
        return best_interblock_frame

    return None  # Return None if no frame was found


######################
## RESOLVING FRAMES ##
######################


def resolve_all_frames(frames: List[CircuitFrame], parted_circuit: Circuit):

    points_used = []
    for frame in frames:
        points_used += frame.block_points
    points_used = list(set(points_used))

    point_to_frames_map = defaultdict(list)

    for frame in frames:
        for point in frame.block_points:
            point_to_frames_map[point].append(frame)

    for point, frame_group in point_to_frames_map.items():
        resolve_overlapping_frames(point, frame_group, parted_circuit)

    return frames


def resolve_overlapping_frames(
    overlap_point: CircuitPoint, frames: List['CircuitFrame'], parted_circuit: Circuit
):
    """
    Resolves overlaps perfectly using the immutable _gate_registry IDs.
    Guarantees every gate is kept by exactly one frame.
    """
    overlap_block = parted_circuit.get_operation(overlap_point)
    overlap_subcircuit = overlap_block.gate._circuit.copy()
    overlap_subcircuit.set_params(overlap_block.params)

    removals_per_frame = {i: [] for i in range(len(frames))}

    for orig_idx, _ in enumerate(overlap_subcircuit):
        gate_universal_id = (overlap_point, orig_idx)

        candidate_frames = []
        for i, frame in enumerate(frames):
            found_point = None
            for pt, reg_id in frame._gate_registry.items():
                if reg_id == gate_universal_id:
                    found_point = pt
                    break

            if found_point is not None:
                in_interblock = False
                best_score = -1.0
                for j, ib_region in enumerate(frame.interblock_regions):
                    if ib_region.overlaps(found_point):
                        in_interblock = True
                        if frame.interblock_scores[j] > best_score:
                            best_score = frame.interblock_scores[j]

                candidate_frames.append({
                    'frame_idx': i,
                    'point': found_point,
                    'in_interblock': in_interblock,
                    'score': best_score,
                    'target_cycle': frame.target_point[0]
                })

        if not candidate_frames:
            continue
        if len(candidate_frames) == 1:
            continue

        winner_idx = None
        interblock_candidates = [c for c in candidate_frames if c['in_interblock']]

        if interblock_candidates:
            interblock_candidates.sort(key=lambda x: x['score'], reverse=True)
            winner_idx = interblock_candidates[0]['frame_idx']
        else:
            candidate_frames.sort(key=lambda x: x['target_cycle'])
            winner_idx = candidate_frames[0]['frame_idx']

        for c in candidate_frames:
            if c['frame_idx'] != winner_idx:
                loser_frame = frames[c['frame_idx']]
                
                full_op_points = [
                    pt for pt, reg_id in loser_frame._gate_registry.items() 
                    if reg_id == gate_universal_id
                ]
                removals_per_frame[c['frame_idx']].extend(full_op_points)

    for i, frame in enumerate(frames):
        if removals_per_frame[i]:
            unique_points_to_remove = list(set(removals_per_frame[i]))
            frame.remove_points(unique_points_to_remove)



def gate_in_circuit(
    circuit: Circuit, gate, gate_location: List[int], cycles
) -> CircuitPoint | None:
    for point in circuit._dag.keys():
        if point[0] < cycles[0] or point[0] > cycles[1]:
            continue

        op = circuit.get_operation(point)

        if gate == op.gate and gate_location == list(op.location):
            return point  # We found the exact gate we're looking for

    return None 
