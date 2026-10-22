#pragma once

#include <string>
#include <vector>

namespace dqc {

// Serialized representation of a pytket `Command`, sufficient for the DQC
// algorithms that consume circuits. The Python adapter builds these from
// `Command.op.get_name()`, `Command.op.type` (as an int, see op_type.hpp),
// `Command.op.params` and `str(qubit)` for each qubit.
struct Command {
  std::string op_name;              // op.get_name()
  std::vector<std::string> qubits;  // str(qubit) for each qubit
  int op_type = -1;                 // pytket OpType enum value (see op_type.hpp)
  std::vector<double> params;       // op.params

  bool operator==(const Command& o) const {
    return op_type == o.op_type && params == o.params && qubits == o.qubits;
  }
  bool operator!=(const Command& o) const { return !(*this == o); }
};

// Serialized representation of a pytket `Circuit`.
struct SerializedCircuit {
  std::vector<std::string> qubits;  // str(qubit) for each circuit qubit
  std::vector<Command> commands;    // in circuit order
};

}  // namespace dqc
