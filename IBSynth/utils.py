

from bqskit.ir import Operation, Circuit



def calculate_round_error(error1, error2):
    """Triangle inequality calculation for error across rounds"""
    return error1 + error2

def count_large_gates(circuit_like: Operation | Circuit | list[Operation]):
    """Count the number of gates acting on 2 or more qubits in a bqskit quantum object"""
    
    if isinstance(circuit_like, Operation):
        operation = circuit_like.gate._circuit.copy()
        operation.unfold_all()
        num_large_gates = 0
        for gate in operation.gate_set:
            if gate.num_qudits == 1:
                continue
            num_large_gates += operation.count(gate)
    elif isinstance(circuit_like, Circuit):
        circuit = circuit_like.copy()
        circuit.unfold_all()
        num_large_gates = 0
        for gate in circuit.gate_set:
            if gate.num_qudits == 1:
                continue
            num_large_gates += circuit.count(gate)
    else:
        operations = circuit_like
        num_large_gates = 0
        for operation in operations:
            for gate in operation.gate._circuit.gate_set:
                if gate.num_qudits == 1:
                    continue
                num_large_gates += operation.gate._circuit.count(gate)

    return num_large_gates


def calc_parted_circ_dist(start_circ: Circuit, fin_circ: Circuit, req_indiv: bool = False):

    errors = []

    for op1, op2 in zip(start_circ, fin_circ):
        if op1 != op2:
            errors.append(op1.get_unitary().get_distance_from(op2.get_unitary()))
        else:
            errors.append(0.0)

    if req_indiv:
        return errors
    
    return sum(errors)

