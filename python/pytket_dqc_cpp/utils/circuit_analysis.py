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

"""Circuit-level checks performed by the compiled core."""

from pytket import Circuit

from .._bridge import core, serialize_circuit


class ConstraintException(Exception):
    """Raised when the communication memory constraint of a server is
    exceeded. Stores the offending server and gate vertex at which
    the constraint was violated.
    """

    def __init__(self, message, server):
        super().__init__(message)
        self.server = server
        self.v_gate = None


def all_cu1_local(circ: Circuit) -> bool:
    """Checks that all of the CU1 gates in the circuit are local."""
    qubits, commands = serialize_circuit(circ)
    return core.all_cu1_local(qubits, commands)


def ebit_memory_required(circ: Circuit) -> dict[int, int]:
    """Scan the circuit and find, for each server, the maximum number of
    ebits simultaneously linked to it. This corresponds to the minimum
    memory dedicated to ebits this circuit requires.

    :param circ: The circuit to be analysed
    :type circ: Circuit

    :return: A dictionary mapping servers to ebit memory required. In
        particular, dict[2] == 3 means that, at some point in the circuit,
        there are 3 ebits simultaneously sharing a qubit with server 2.
    :rtype: dict[int, int]
    """
    qubits, commands = serialize_circuit(circ)

    # Find all server IDs and initialise their ebit memory to 0
    # Scan the circuit for starting and ending EJPP processes and update
    #   the ebit memory requirement accordingly
    return {
        int(server_id): int(required)
        for server_id, required in core.ebit_memory_required(qubits, commands).items()
    }


# TODO: This is checked by parsing the name of the qubit.
# Is there a better way to do this?
def is_link_qubit(qubit) -> bool:
    # ``qubit_name`` follows either of these patterns:
    #     Workspace qubit: ['server', server_id+'['+qubit_id+']']
    #     Link qubit: ['server', server_id, 'link', 'register['+qubit_id+']']
    # Sanity check
    return core.is_link_qubit(str(qubit))


# TODO: The way the server ID is obtained is by parsing the name of
# the qubit. Is there a better way to access this information?
def get_server_id(qubit) -> int:
    # ``qubit_name`` follows either of these patterns:
    #     Workspace qubit: ['server', server_id+'['+qubit_id+']']
    #     Link qubit: ['server', server_id, 'link', 'register['+qubit_id+']']
    # Sanity check
    return core.get_server_id(str(qubit))


def ebit_cost(circ: Circuit) -> int:
    """Scan the circuit and return the number of ebits required to implement
    it.

    :param circ: The circuit to be analysed.
    :type circ: Circuit

    :return: The number of ebits consumed by the circuit.
    :rtype: int
    """
    qubits, commands = serialize_circuit(circ)
    return core.ebit_cost(qubits, commands)
