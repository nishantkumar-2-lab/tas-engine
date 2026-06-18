//! Generic Pattern Database (Admissible Heuristic)
//!
//! Computes a backward BFS from the victory cell on a *relaxed* graph
//! (doors open, switches ignored) to build an exact distance table.
//! Lookup is O(1): `table[cell * 16 + keys]`.

use crate::state::{PackedState, LayoutConfig};
use crate::env::WorldConfig;

pub struct PatternDatabase {
    table: Vec<u32>,
    cell_count: usize,
}

impl PatternDatabase {
    /// Compute PDB from the given world configuration.
    pub fn compute(_layout: &LayoutConfig, world: &WorldConfig) -> Self {
        let cells = world.cell_count();
        let mut dist = vec![u32::MAX; cells];
        let mut queue = std::collections::VecDeque::with_capacity(cells);

        let victory = world.victory_cell as usize;
        dist[victory] = 0;
        queue.push_back(victory);

        while let Some(cell) = queue.pop_front() {
            let d = dist[cell];
            let (row, col) = world.coords(cell as u16);
            let deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)];

            for &(dr, dc) in &deltas {
                let pr = if dr < 0 { row.wrapping_sub(1) } else { row + (dr as u8) };
                let pc = if dc < 0 { col.wrapping_sub(1) } else { col + (dc as u8) };

                if dr < 0 && row == 0 { continue; }
                if dc < 0 && col == 0 { continue; }
                if pr >= world.height || pc >= world.width { continue; }

                let pred = world.idx(pr, pc) as usize;
                let rule = world.rule(world.terrain_at(pred as u16));
                if !rule.walkable { continue; }

                if dist[pred] == u32::MAX {
                    dist[pred] = d + 1;
                    queue.push_back(pred);
                }
            }
        }

        // Broadcast geometric distances to all 16 key combos per cell.
        let mut table = vec![u32::MAX; cells * 16];
        for cell in 0..cells {
            let d = dist[cell];
            for keys in 0..16usize {
                table[cell * 16 + keys] = d;
            }
        }

        PatternDatabase { table, cell_count: cells }
    }

    /// O(1) heuristic lookup.
    #[inline(always)]
    pub fn get_heuristic(&self, state: &PackedState, layout: &LayoutConfig) -> u32 {
        let cell = state.get_location(layout) as usize;
        let keys = state.get_keys(layout) as usize;
        let idx = cell * 16 + keys;
        // Safety: idx is always within bounds.
        unsafe { *self.table.get_unchecked(idx) }
    }

    pub fn cell_count(&self) -> usize {
        self.cell_count
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::default_world;

    #[test]
    fn test_victory_cell_zero_distance() {
        let layout = LayoutConfig::standard();
        let world = default_world();
        let pdb = PatternDatabase::compute(&layout, &world);
        let mut s = PackedState::zero();
        s.set_location(&layout, world.victory_cell);
        assert_eq!(pdb.get_heuristic(&s, &layout), 0);
    }

    #[test]
    fn test_unreachable_cell_is_max() {
        let layout = LayoutConfig::standard();
        let world = default_world();
        let pdb = PatternDatabase::compute(&layout, &world);
        let mut s = PackedState::zero();
        s.set_location(&layout, 0); // (0,0) is a wall in default maze
        assert_eq!(pdb.get_heuristic(&s, &layout), u32::MAX);
    }

    #[test]
    fn test_start_cell_heuristic() {
        let layout = LayoutConfig::standard();
        let world = default_world();
        let pdb = PatternDatabase::compute(&layout, &world);
        let mut s = PackedState::zero();
        s.set_location(&layout, world.start_cell);
        let h = pdb.get_heuristic(&s, &layout);
        println!("Start cell {} heuristic = {}", world.start_cell, h);
        assert_ne!(h, u32::MAX, "start cell must be reachable");
    }
}
