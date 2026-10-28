from __future__ import annotations

import pytest
from pytket.circuit import Circuit, OpType

from pytket_dqc_cpp.circuits import Distribution, HypergraphCircuit
from pytket_dqc_cpp.networks import NISQNetwork
from pytket_dqc_cpp.placement import Placement
from pytket_dqc_cpp.utils.circuit_analysis import ConstraintException


def _build_bounded_memory_distribution() -> tuple[Distribution, int]:
    """Construct a distribution that violates bounded communication memory capacity.

    This reproduces the scenario in upstream `test_to_pytket_circuit_circ_with_embeddings_1`
    (around line 936 of tests/circuits_test.py), where upstream had:
        except ConstraintException as e:
            e.server = 3  # typo in upstream test; assignment rather than assertion
    """
    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 0.1234, [1, 2])
    circ.add_gate(OpType.CU1, 0.1234, [0, 2])
    circ.add_gate(OpType.CU1, 0.1234, [2, 3])
    circ.add_gate(OpType.CU1, 0.1234, [0, 3])
    circ.H(0).H(2).Rz(0.1234, 3)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.add_gate(OpType.CU1, 1.0, [0, 3])
    circ.add_gate(OpType.CU1, 1.0, [1, 2])
    circ.H(0).H(2).Rz(0.1234, 0)
    circ.add_gate(OpType.CU1, 0.1234, [0, 1])
    circ.add_gate(OpType.CU1, 0.1234, [0, 3])
    circ.add_gate(OpType.CU1, 1.0, [1, 2])

    placement = Placement(
        {
            0: 1,
            1: 1,
            2: 2,
            3: 4,
            4: 1,
            5: 2,
            6: 4,
            7: 4,
            8: 2,
            9: 4,
            10: 0,
            11: 1,
            12: 4,
            13: 3,
        }
    )

    hyp_circ = HypergraphCircuit(circ)
    hyp_circ.hyperedge_list = []
    hyp_circ.hyperedge_dict = {v: [] for v in hyp_circ.vertex_list}
    hyp_circ.vertex_neighbours = {v: set() for v in hyp_circ.vertex_list}

    hyp_circ.add_hyperedge([0, 11, 12])
    hyp_circ.add_hyperedge([0, 5, 7])
    hyp_circ.add_hyperedge([0, 8, 9])
    hyp_circ.add_hyperedge([1, 4, 10, 11, 13])
    hyp_circ.add_hyperedge([2, 4, 5, 6, 13])
    hyp_circ.add_hyperedge([2, 8, 10])
    hyp_circ.add_hyperedge([3, 6, 7, 9, 12])

    network = NISQNetwork(
        [[0, 1], [0, 2], [0, 3], [3, 4]],
        {0: [0], 1: [1, 2], 2: [3, 4], 3: [7], 4: [5, 6]},
        server_ebit_mem={0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
    )

    distribution = Distribution(hyp_circ, placement, network)
    assert distribution.is_valid()
    return distribution, 0


def test_constraint_exception_v_gate_parity():
    """Verify e.v_gate and e.server attribute parity with upstream pytket-dqc.

    In upstream pytket-dqc, ConstraintException provides:
      - e.server: The offending server index (int)
      - e.v_gate: The gate vertex id causing the violation (int)

    The C++ pybind11 binding directly translates dqc::ConstraintException into
    pytket_dqc_cpp.utils.circuit_analysis.ConstraintException with both attributes preserved.
    """
    distribution, expected_server = _build_bounded_memory_distribution()

    with pytest.raises(ConstraintException) as exc_info:
        distribution.to_pytket_circuit()

    exc = exc_info.value
    assert exc.server == expected_server
    assert exc.v_gate is not None
    assert exc.v_gate == 11


def test_constraint_exception_hint_parity():
    """Verify remediation hint message parity with upstream pytket-dqc.

    Upstream pytket-dqc appends:
      'to_pytket_circuit(allow_update=True)'
    to the ConstraintException message to guide the user.
    """
    distribution, _ = _build_bounded_memory_distribution()

    with pytest.raises(ConstraintException) as exc_info:
        distribution.to_pytket_circuit()

    exc = exc_info.value
    expected_hint = "to_pytket_circuit(allow_update=True)"
    assert expected_hint in str(exc), f"Expected guidance hint '{expected_hint}' in exception message: {str(exc)!r}"


def test_constraint_exception_non_home_parity():
    """Verify exact error message and attribute parity for non-home server constraint violations."""
    import pytket_dqc as ref

    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    network_kwargs = {
        "connections": [[0, 1], [1, 2]],
        "server_qubits": {0: [0], 1: [1], 2: [2, 3]},
        "server_ebit_mem": {0: 1, 1: 1, 2: 1},
    }
    placement_dict = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2}

    ref_net = ref.NISQNetwork(
        network_kwargs["connections"],
        network_kwargs["server_qubits"],
        server_ebit_mem=network_kwargs["server_ebit_mem"],
    )
    ref_dist = ref.Distribution(
        ref.HypergraphCircuit(circ), ref.Placement(placement_dict), ref_net
    )

    cpp_net = NISQNetwork(
        network_kwargs["connections"],
        network_kwargs["server_qubits"],
        server_ebit_mem=network_kwargs["server_ebit_mem"],
    )
    cpp_dist = Distribution(
        HypergraphCircuit(circ), Placement(placement_dict), cpp_net
    )

    with pytest.raises(ref.utils.circuit_analysis.ConstraintException) as ref_info:
        ref_dist.to_pytket_circuit(satisfy_bound=True, allow_update=True)

    with pytest.raises(ConstraintException) as cpp_info:
        cpp_dist.to_pytket_circuit(satisfy_bound=True, allow_update=True)

    ref_exc = ref_info.value
    cpp_exc = cpp_info.value

    # Check exact byte-for-byte message parity (including upstream's 'sinceit' and trailing guidance)
    assert str(cpp_exc) == str(ref_exc)
    assert "sinceit is placed on a non-home server" in str(cpp_exc)
    assert "Consider using an approach that does not produce evicted gates" in str(cpp_exc)
    assert cpp_exc.server == ref_exc.server
    assert getattr(cpp_exc, "v_gate", None) == getattr(ref_exc, "v_gate", None)

