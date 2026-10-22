#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

#include "dqc/hypergraph.hpp"
#include "dqc/op_type.hpp"
#include "dqc/pacman.hpp"
#include "dqc/placement.hpp"

using dqc::Hyperedge;
using dqc::HypergraphCircuit;
using dqc::PacMan;
using dqc::Packet;
using dqc::Placement;
using dqc::SerializedCircuit;
using dqc::Vertex;

namespace {

std::vector<int> indices(const std::vector<Packet>& ps) {
  std::vector<int> out;
  for (const Packet& p : ps) out.push_back(p.packet_index);
  return out;
}

std::vector<int> sorted_indices(const std::vector<Packet>& ps) {
  std::vector<int> out = indices(ps);
  std::sort(out.begin(), out.end());
  return out;
}

// `hopping_03` circuit: CU1, H, CU1, H, CU1 on qubit 0 (sandwich embedding).
HypergraphCircuit hopping_03_circuit() {
  SerializedCircuit c;
  c.qubits = {"q[0]", "q[1]", "q[2]", "q[3]", "q[4]", "q[5]"};
  c.commands = {
      {"CU1", {"q[0]", "q[2]"}, dqc::kOpCU1, {1.0}},
      {"H", {"q[0]"}, dqc::kOpH},
      {"CU1", {"q[0]", "q[3]"}, dqc::kOpCU1, {1.0}},
      {"H", {"q[0]"}, dqc::kOpH},
      {"CU1", {"q[0]", "q[2]"}, dqc::kOpCU1, {1.0}},
  };
  HypergraphCircuit h;
  h.from_circuit(c);
  return h;
}

Placement hopping_03_placement() {
  Placement p;
  for (int i = 0; i < 6; ++i) p.set(i, i / 2);
  p.set(6, 0);
  p.set(7, 0);
  p.set(8, 0);
  return p;
}

// `neighbouring_00` circuit: CU1, Rz, CU1 on qubit 0, with the first
// hyperedge split so the two gates become neighbouring packets.
HypergraphCircuit neighbouring_00_circuit() {
  SerializedCircuit c;
  c.qubits = {"q[0]", "q[1]", "q[2]", "q[3]", "q[4]", "q[5]"};
  c.commands = {
      {"CU1", {"q[0]", "q[2]"}, dqc::kOpCU1, {1.0}},
      {"Rz", {"q[0]"}, dqc::kOpRz, {0.3}},
      {"CU1", {"q[0]", "q[3]"}, dqc::kOpCU1, {1.0}},
  };
  HypergraphCircuit h;
  h.from_circuit(c);
  h.split_hyperedge(Hyperedge({0, 6, 7}, 1),
                    {Hyperedge({0, 7}, 1), Hyperedge({0, 6}, 1)});
  return h;
}

Placement neighbouring_00_placement() {
  Placement p;
  for (int i = 0; i < 6; ++i) p.set(i, i / 2);
  p.set(6, 0);
  p.set(7, 0);
  return p;
}

}  // namespace

TEST_CASE("pacman builds packets (hopping_03)") {
  const PacMan pm(hopping_03_circuit(), hopping_03_placement());

  const auto& pbq = pm.packets_by_qubit();
  REQUIRE(pbq.at(0).size() == 3);
  REQUIRE(pbq.at(1).empty());
  REQUIRE(pbq.at(2).size() == 1);
  REQUIRE(pbq.at(3).size() == 1);
  REQUIRE(pbq.at(4).empty());
  REQUIRE(pbq.at(5).empty());

  const Packet& p0 = pbq.at(0)[0];
  REQUIRE(p0.packet_index == 0);
  REQUIRE(p0.qubit_vertex == 0);
  REQUIRE(p0.connected_server_index == 1);
  REQUIRE(p0.gate_vertices == std::vector<Vertex>{6});

  const Packet& p1 = pbq.at(0)[1];
  REQUIRE(p1.packet_index == 1);
  REQUIRE(p1.connected_server_index == 1);
  REQUIRE(p1.gate_vertices == std::vector<Vertex>{7});

  const Packet& p2 = pbq.at(0)[2];
  REQUIRE(p2.packet_index == 2);
  REQUIRE(p2.connected_server_index == 1);
  REQUIRE(p2.gate_vertices == std::vector<Vertex>{8});

  const Packet& p3 = pbq.at(2)[0];
  REQUIRE(p3.packet_index == 3);
  REQUIRE(p3.qubit_vertex == 2);
  REQUIRE(p3.connected_server_index == 0);
  REQUIRE(p3.gate_vertices == (std::vector<Vertex>{6, 8}));

  const Packet& p4 = pbq.at(3)[0];
  REQUIRE(p4.packet_index == 4);
  REQUIRE(p4.connected_server_index == 0);
  REQUIRE(p4.gate_vertices == std::vector<Vertex>{7});
}

TEST_CASE("pacman identifies hopping (hopping_03)") {
  const PacMan pm(hopping_03_circuit(), hopping_03_placement());

  // No neighbouring packets; one hopping packet (0 -> 2) on qubit 0.
  for (const auto& [qv, nps] : pm.neighbouring_packets()) {
    (void)qv;
    REQUIRE(nps.empty());
  }
  REQUIRE(pm.hopping_packets().at(0).size() == 1);
  const auto& hp = pm.hopping_packets().at(0)[0];
  REQUIRE(hp.first.packet_index == 0);
  REQUIRE(hp.second.packet_index == 2);

  // Merged packets: qubit 0 -> [0,2] and [1]; qubits 2,3 -> singletons.
  const auto& mp = pm.merged_packets().at(0);
  REQUIRE(mp.size() == 2);
  REQUIRE(indices(mp[0].packets) == std::vector<int>({0, 2}));
  REQUIRE(indices(mp[1].packets) == std::vector<int>({1}));
  REQUIRE(indices(pm.merged_packets().at(2)[0].packets) ==
          std::vector<int>({3}));
  REQUIRE(indices(pm.merged_packets().at(3)[0].packets) ==
          std::vector<int>({4}));
}

TEST_CASE("pacman packet navigation helpers (hopping_03)") {
  const PacMan pm(hopping_03_circuit(), hopping_03_placement());
  const auto& pbq = pm.packets_by_qubit().at(0);

  // get_subsequent_hopping_packet: P0 -> P2, P1 -> None, P2 -> None.
  REQUIRE(pm.get_subsequent_hopping_packet(pbq[0])->packet_index == 2);
  REQUIRE(!pm.get_subsequent_hopping_packet(pbq[1]).has_value());
  REQUIRE(!pm.get_subsequent_hopping_packet(pbq[2]).has_value());

  // get_subsequent_neighbouring_packet: none in this circuit.
  REQUIRE(!pm.get_subsequent_neighbouring_packet(pbq[0]).has_value());

  // get_next_packet (same connected server): P0 -> P1 -> P2 -> None.
  REQUIRE(pm.get_next_packet(pbq[0])->packet_index == 1);
  REQUIRE(pm.get_next_packet(pbq[1])->packet_index == 2);
  REQUIRE(!pm.get_next_packet(pbq[2]).has_value());

  // get_all_packets is index-sorted; get_all_merged_packets covers all.
  const std::vector<Packet> all = pm.get_all_packets();
  REQUIRE(indices(all) == std::vector<int>({0, 1, 2, 3, 4}));
  REQUIRE(all[0].gate_vertices == std::vector<Vertex>{6});
  std::vector<std::vector<int>> all_merged;
  for (const auto& m : pm.get_all_merged_packets()) {
    all_merged.push_back(indices(m.packets));
  }
  REQUIRE(all_merged ==
          (std::vector<std::vector<int>>{{0, 2}, {1}, {3}, {4}}));
}

TEST_CASE("pacman connectivity and embedding (hopping_03)") {
  const PacMan pm(hopping_03_circuit(), hopping_03_placement());

  // get_connected_packets (compared as sorted index lists).
  const Packet& p0 = pm.packets_by_qubit().at(0)[0];
  const Packet& p1 = pm.packets_by_qubit().at(0)[1];
  const Packet& p2 = pm.packets_by_qubit().at(0)[2];
  const Packet& p3 = pm.packets_by_qubit().at(2)[0];
  const Packet& p4 = pm.packets_by_qubit().at(3)[0];
  REQUIRE(sorted_indices(pm.get_connected_packets(p0)) ==
          std::vector<int>({3}));
  REQUIRE(sorted_indices(pm.get_connected_packets(p1)) ==
          std::vector<int>({4}));
  REQUIRE(sorted_indices(pm.get_connected_packets(p2)) ==
          std::vector<int>({3}));
  REQUIRE(sorted_indices(pm.get_connected_packets(p3)) ==
          std::vector<int>({0, 2}));
  REQUIRE(sorted_indices(pm.get_connected_packets(p4)) ==
          std::vector<int>({1}));

  // get_embedded_packets for the (0,2) hopping -> P1.
  REQUIRE(sorted_indices(pm.get_embedded_packets({p0, p2})) ==
          std::vector<int>({1}));

  // is_packet_embedded: only P1 is embedded.
  REQUIRE(!pm.is_packet_embedded(p0));
  REQUIRE(pm.is_packet_embedded(p1));
  REQUIRE(!pm.is_packet_embedded(p2));

  // get_containing_merged_packet.
  REQUIRE(indices(pm.get_containing_merged_packet(p0).packets) ==
          std::vector<int>({0, 2}));
  REQUIRE(indices(pm.get_containing_merged_packet(p1).packets) ==
          std::vector<int>({1}));

  // get_split_packets of the merged packet {0,2} on conflict (0,2).
  const auto [left, right] =
      pm.get_split_packets(pm.get_containing_merged_packet(p0), {p0, p2});
  REQUIRE(indices(left.packets) == std::vector<int>({0}));
  REQUIRE(indices(right.packets) == std::vector<int>({2}));

  // are_neighbouring / are_hoppable on qubit-0 pairs.
  REQUIRE(!pm.are_neighbouring_packets(p0, p1));
  REQUIRE(!pm.are_neighbouring_packets(p0, p2));
  REQUIRE(pm.are_hoppable_packets(p0, p2));
  REQUIRE(!pm.are_hoppable_packets(p0, p1));
  REQUIRE(!pm.are_hoppable_packets(p1, p2));

  // get_hopping_packets_within over the full merged-packet set.
  const auto& mps = pm.merged_packets().at(0);
  std::vector<dqc::MergedPacket> mp_set(mps.begin(), mps.end());
  mp_set.push_back(pm.merged_packets().at(2)[0]);
  mp_set.push_back(pm.merged_packets().at(3)[0]);
  dqc::PySet<dqc::MergedPacket> within_set(&dqc::py_hash_merged,
                                           &dqc::py_eq_merged);
  for (const auto& m : mp_set) within_set.add(m);
  const auto within = pm.get_hopping_packets_within(within_set).iteration_order();
  REQUIRE(within.size() == 1);
  REQUIRE(within[0].first.packet_index == 0);
  REQUIRE(within[0].second.packet_index == 2);
}

TEST_CASE("pacman identifies neighbouring (neighbouring_00)") {
  const PacMan pm(neighbouring_00_circuit(), neighbouring_00_placement());

  const auto& pbq = pm.packets_by_qubit();
  REQUIRE(pbq.at(0).size() == 2);
  REQUIRE(pbq.at(0)[0].packet_index == 0);
  REQUIRE(pbq.at(0)[0].gate_vertices == std::vector<Vertex>{7});
  REQUIRE(pbq.at(0)[1].packet_index == 1);
  REQUIRE(pbq.at(0)[1].gate_vertices == std::vector<Vertex>{6});
  REQUIRE(pbq.at(2)[0].gate_vertices == std::vector<Vertex>{6});
  REQUIRE(pbq.at(3)[0].gate_vertices == std::vector<Vertex>{7});

  // Neighbouring packets on qubit 0: [[0, 1]].
  REQUIRE(pm.neighbouring_packets().at(0).size() == 1);
  REQUIRE(indices(pm.neighbouring_packets().at(0)[0]) ==
          std::vector<int>({0, 1}));

  // No hopping packets.
  for (const auto& [qv, hps] : pm.hopping_packets()) {
    (void)qv;
    REQUIRE(hps.empty());
  }

  // Merged: qubit 0 -> [[0,1]].
  REQUIRE(indices(pm.merged_packets().at(0)[0].packets) ==
          std::vector<int>({0, 1}));

  // get_subsequent_neighbouring_packet: P0 -> P1, P1 -> None.
  REQUIRE(pm.get_subsequent_neighbouring_packet(pbq.at(0)[0])->packet_index ==
          1);
  REQUIRE(!pm.get_subsequent_neighbouring_packet(pbq.at(0)[1]).has_value());

  // get_next_packet: P0 -> P1 (same server), P1 -> None.
  REQUIRE(pm.get_next_packet(pbq.at(0)[0])->packet_index == 1);
  REQUIRE(!pm.get_next_packet(pbq.at(0)[1]).has_value());

  // are_neighbouring_packets(P0, P1) is True.
  REQUIRE(pm.are_neighbouring_packets(pbq.at(0)[0], pbq.at(0)[1]));
}
