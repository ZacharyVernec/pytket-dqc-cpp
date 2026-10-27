# Copyright 2023 Quantinuum and The University of Tokyo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from `pytket_dqc` (commit 1ac175e) for the `pytket_dqc_cpp` adapter.

"""Lazy verification helpers delegated to the reference implementation."""

from pytket import Circuit, Qubit

from .._delegation import reference_module, warn_once


def check_equivalence(
    circ1: Circuit, circ2: Circuit, qubit_mapping: dict[Qubit, Qubit]
) -> bool:
    """
    Use PyZX to check the two circuits are equivalent. This is done by
    concatenating the ZX diagram of `circ1` with the adjoint of the
    ZX diagram of `circ2` to see if the result is the identity.

    NOTE: PyZX can be installed via `pip install pyzx`. You will also need
    to install pytket-pyzx via `pip install pytket-pyzx`.

    NOTE: returning False does not guarantee that the circuits are different;
    it just means that PyZX could not prove that they are equivalent.
    If it returns True and the `qubit_mapping` covers all qubits of both
    circuits, then it is guaranteed that the circuits are equivalent.
    If some qubits are missing from `qubit_mapping` then there are
    post-selections/projections to state 0, which means that the test is not
    a formal proof -- we would need to test for projection to state 1 as well
    (and, in fact, all combinations of 0 and 1).
    Hence, in the latter case, we can only say "we have strong evidence that
    the circuits are equivalent".

    :param circ1: The first of the two circuits to be compared for equality
    :type circ1: Circuit
    :param circ2: The second of the two circuits to be compared for equality
    :type circ2: Circuit
    :param qubit_mapping: A mapping from qubits of `circ1` to qubits of
        `circ2`. If a qubit is not included in this dictionary it means that
        it ought to be treated as an ancilla (i.e. prepared and measured).
    :type qubit_mapping: dict[Qubit, Qubit]

    :return: Whether the circuits are proven to be equivalent.
    :rtype: bool
    """
    warn_once("check_equivalence")
    return reference_module("utils.verification").check_equivalence(
        circ1, circ2, qubit_mapping
    )


def to_pyzx(circuit: Circuit, mask: list[Qubit]) -> "zx.graph.base.BaseGraph":
    """Convert a circuit to a ZX diagram in PyZX. Every starting EJPP
    process and ending EJPP process is converted to a CX gate. All non-ancilla
    qubits should be in ``mask`` and no ancilla qubits should be in ``mask``.

    Note: Ancilla qubits are projected to state 0. This is not equivalent to
    and EJPP ending process, since what we really need is to discard the qubit
    so we should check for every combination of projections to 0 and 1 states.
    However, projecting to 0 is enough for our purposes to give strong evidence
    of circuit equality.

    :param circuit: The circuit to be converted to a ZX-diagram
    :type circuit: Circuit
    :param mask: The list of qubits that are not ancillas, ordered according
        to the intended order of wires in the output ZX-diagram.
    :type mask: list[Qubit]
    """
    warn_once("to_pyzx")
    return reference_module("utils.verification").to_pyzx(circuit, mask)
