#pragma once

namespace dqc {

// pytket `OpType` enum values, pinned to pytket 2.16.0 (see README
// "Environment pins"). The Python adapter serializes `cmd.op.type` as these
// integer values.
inline constexpr int kOpBarrier = 8;
inline constexpr int kOpZ = 22;
inline constexpr int kOpX = 23;
inline constexpr int kOpS = 25;
inline constexpr int kOpSdg = 26;
inline constexpr int kOpH = 33;
inline constexpr int kOpRx = 34;
inline constexpr int kOpRz = 36;
inline constexpr int kOpCX = 45;
inline constexpr int kOpCZ = 47;
inline constexpr int kOpCU1 = 58;
inline constexpr int kOpCustomGate = 99;

}  // namespace dqc
