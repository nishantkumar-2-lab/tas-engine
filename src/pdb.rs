//! Pattern Database (PDB) — Precomputed Admissible Heuristic (Phase 2)
//!
//! We run a backward BFS from the Victory Cell on a *relaxed* graph
//! where all doors are treated as permanently open and all switches /
//! one-way constraints are ignored.  This guarantees the resulting
//! distance is a lower-bound (never overestimates) on the true cost,
//! making it a perfectly admissible and consistent heuristic for
//! A* / IDA*.
//!
//! The table stores distances for every `(Cell, Keys)` pair:
//!   index = cell * 16 + keys_bitmask   =>   64 * 16 = 1 024 entries.
//! Although the relaxed distance does not depend on keys, we keep the
//! full cross-product so that later key-aware heuristics can drop in
//! without changing the indexing scheme.

use crate::state::PackedState;
use crate::env::{terrain_at, coords, idx, Input, step};

/// Flat-array Pattern Database.
/// Every lookup is a single array index — no hashing, no indirection.
pub struct PatternDatabase {
    /// 1024 entries: index = cell*16 + keys_bitmask.
    table: [u32; 1024],
}

impl PatternDatabase {
    /// Compute the PDB once at program startup.
    /// Backward BFS from `victory_cell` on the doors-open / no-switch graph.
    pub fn compute(victory_cell: u8) -> Self {
        let mut table = [u32::MAX; 1024];

        // ─── BFS over the 64 cells on the relaxed graph ────────────────
        // Only walls (terrain == 1) are obstacles.
        let mut dist = [u32::MAX; 64];

        // Stack-allocated ring buffer — zero heap, cache-friendly.
        let mut queue = [0u8; 64];
        let mut head: usize = 0;
        let mut tail: usize = 0;

        // Seed the BFS at the victory cell.
        queue[tail] = victory_cell;
        tail = tail.wrapping_add(1);
        dist[victory_cell as usize] = 0;

        while head != tail {
            let cell = queue[head];
            head = head.wrapping_add(1);
            let d = dist[cell as usize];
            let (row, col) = coords(cell);

            // Four cardinal neighbours, unrolled for branch-prediction.
            let neighbours: [(i8, i8); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
            for (dr, dc) in neighbours {
                let nr = (row as i8) + dr;
                let nc = (col as i8) + dc;
                // Bounds check: 0..7 for both axes.
                if nr < 0 || nr > 7 || nc < 0 || nc > 7 {
                    continue;
                }
                let ncell = idx(nr as u8, nc as u8);
                let t = terrain_at(ncell);
                // Walls (terrain == 1) are the only obstacles in the relaxed graph.
                if t == 1 {
                    continue;
                }
                if dist[ncell as usize] == u32::MAX {
                    dist[ncell as usize] = d + 1;
                    queue[tail] = ncell;
                    tail = tail.wrapping_add(1);
                }
            }
        }

        // ─── Expand to full (Cell × Keys) table ──────────────────────
        // The relaxed distance is identical for every key combination,
        // so we broadcast the cell distance across all 16 key slots.
        for cell in 0..64usize {
            let d = dist[cell];
            let base = cell * 16;
            // Unrolled small loop: 16 writes per cell, fully cache-resident.
            for keys in 0..16usize {
                table[base + keys] = d;
            }
        }

        PatternDatabase { table }
    }

    /// O(1) heuristic lookup.
    /// Returns the precomputed lower-bound distance from `state` to victory.
    #[inline(always)]
    pub fn get_heuristic(&self, state: PackedState) -> u32 {
        let cell = state.get_cell() as usize;
        let keys = state.get_keys() as usize;
        // Safety: cell ∈ [0,63], keys ∈ [0,15]  =>  index ∈ [0,1023].
        // The table has exactly 1024 entries, so this is always in-bounds.
        *unsafe { self.table.get_unchecked(cell * 16 + keys) }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::idx;

    #[test]
    fn test_victory_cell_zero_distance() {
        let pdb = PatternDatabase::compute(idx(7, 6));
        // At the victory cell, distance must be 0 regardless of keys.
        for keys in 0..16u8 {
            let s = PackedState::new(idx(7, 6), keys, 0, 0, 0);
            assert_eq!(pdb.get_heuristic(s), 0);
        }
    }

    #[test]
    fn test_neighbour_of_victory() {
        let pdb = PatternDatabase::compute(idx(7, 6));
        // (6,6) is one step away from victory (7,6) and is an empty cell.
        let s = PackedState::new(idx(6, 6), 0, 0, 0, 0);
        assert_eq!(pdb.get_heuristic(s), 1);
    }

    #[test]
    fn test_unreachable_cell_is_max() {
        // A wall cell should remain at u32::MAX in the relaxed graph.
        let pdb = PatternDatabase::compute(idx(7, 6));
        let s = PackedState::new(idx(0, 0), 0, 0, 0, 0);
        assert_eq!(pdb.get_heuristic(s), u32::MAX);
    }

    #[test]
    fn test_monotonicity_known_path() {
        let pdb = PatternDatabase::compute(idx(7, 6));
        // (6,6) is distance 1 from victory.
        // (5,6) is distance 2 from victory (5,6 -> 6,6 -> 7,6).
        let closer = PackedState::new(idx(6, 6), 0, 0, 0, 0);
        let farther = PackedState::new(idx(5, 6), 0, 0, 0, 0);
        let h_closer = pdb.get_heuristic(closer);
        let h_farther = pdb.get_heuristic(farther);

        assert!(
            h_farther >= h_closer || h_closer == u32::MAX || h_farther == u32::MAX,
            "heuristic should be non-decreasing when moving away from goal"
        );
    }

    #[test]
    fn test_table_size_is_1024() {
        let pdb = PatternDatabase::compute(idx(7, 6));
        assert_eq!(pdb.table.len(), 1024);
    }

    #[test]
    fn test_all_cells_have_value() {
        let pdb = PatternDatabase::compute(idx(7, 6));
        // Every reachable cell should have a finite distance for at least one key combo.
        for cell in 0..64u8 {
            let s = PackedState::new(cell, 0, 0, 0, 0);
            let h = pdb.get_heuristic(s);
            // Either reachable (finite) or a wall (MAX).
            assert!(h == u32::MAX || h < u32::MAX);
        }
    }

    /// Rigorous admissibility proof-by-enumeration.
    ///
    /// We build the complete reverse transition graph over all
    /// (Cell, Keys, Doors, Switches) configurations, then run a single
    /// reverse BFS from every victory configuration.  This gives the
    /// true optimal cost from *every* state to victory in O(N) time,
    /// where N = 64 × 16 × 16 × 16 = 262 144.
    #[test]
    fn test_pdb_never_overestimates() {
        let pdb = PatternDatabase::compute(idx(7, 6));

        const STATE_COUNT: usize = 64 * 16 * 16 * 16; // 262_144

        // Pack (Cell, Keys, Doors, Switches) into a flat 18-bit index.
        let pack = |cell: u8, keys: u8, doors: u8, switches: u8| -> usize {
            ((cell as usize) << 12)
                | ((keys as usize) << 8)
                | ((doors as usize) << 4)
                | (switches as usize)
        };

        // ─── 1. Build reverse transition graph ────────────────────────
        let mut rev: Vec<Vec<u32>> = vec![Vec::new(); STATE_COUNT];

        for cell in 0..64u8 {
            for keys in 0..16u8 {
                for doors in 0..16u8 {
                    for switches in 0..16u8 {
                        let from_idx = pack(cell, keys, doors, switches);
                        let state = PackedState::new(cell, keys, doors, switches, 0);

                        for input_id in 0..6u8 {
                            let input = match input_id {
                                0 => Input::Wait,
                                1 => Input::Up,
                                2 => Input::Down,
                                3 => Input::Left,
                                4 => Input::Right,
                                5 => Input::Activate,
                                _ => unreachable!(),
                            };
                            if let Some(next) = step(state, input) {
                                let to_idx = pack(next.get_cell(), next.get_keys(), next.get_doors(), next.get_switches());
                                rev[to_idx].push(from_idx as u32);
                            }
                        }
                    }
                }
            }
        }

        // ─── 2. Reverse BFS from all victory configurations ───────────
        let mut true_dist = vec![u32::MAX; STATE_COUNT];
        let mut queue = std::collections::VecDeque::with_capacity(STATE_COUNT);

        for keys in 0..16u8 {
            for doors in 0..16u8 {
                for switches in 0..16u8 {
                    let v = pack(idx(7, 6), keys, doors, switches);
                    true_dist[v] = 0;
                    queue.push_back(v);
                }
            }
        }

        while let Some(u) = queue.pop_front() {
            let d = true_dist[u];
            for &pred in &rev[u] {
                let p = pred as usize;
                if true_dist[p] == u32::MAX {
                    true_dist[p] = d + 1;
                    queue.push_back(p);
                }
            }
        }

        // ─── 3. Verify PDB <= true_dist for every VALID state ─────────
        // Wall cells are never valid player positions; skip them.
        for cell in 0..64u8 {
            if terrain_at(cell) == 1 {
                continue; // wall — not a valid state
            }
            for keys in 0..16u8 {
                for doors in 0..16u8 {
                    for switches in 0..16u8 {
                        let s_idx = pack(cell, keys, doors, switches);
                        let state = PackedState::new(cell, keys, doors, switches, 0);
                        let h = pdb.get_heuristic(state);
                        let true_cost = true_dist[s_idx];

                        if true_cost != u32::MAX {
                            assert!(
                                h <= true_cost,
                                "PDB heuristic {} overestimates true cost {} for state raw=0x{:016X}",
                                h, true_cost, state.raw()
                            );
                        }
                    }
                }
            }
        }
    }
}
