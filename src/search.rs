//! Generic IDA* Search Engine
//!
//! Iterative Deepening A* with a generational dominance filter.
//! The filter is a flat Vec<u16> sized to `cells × 16 × 16 × 16`
//! (cell, keys, doors, switches).  If this exceeds ~4 MB,
//! it falls back to an `FxHashMap` backed by pre-allocated buckets.

use crate::state::{PackedState, LayoutConfig};
use crate::env::{WorldConfig, INPUT_COUNT, step};
use crate::pdb::PatternDatabase;
use fxhash::FxHashMap;

// ─── Dominance Filter ─────────────────────────────────────────────────────

enum DominanceFilter {
    /// Flat array: index = ((cell << 12) | (keys << 8) | (doors << 4) | switches).
    /// Value = best tick count seen (0 = empty).
    Flat { data: Vec<u16> },
    /// Fallback for huge state spaces.
    Map { data: FxHashMap<u64, u16> },
}

impl DominanceFilter {
    fn new(cell_count: usize) -> Self {
        let slots = cell_count * 16 * 16 * 16;
        if slots <= 4_000_000 {
            DominanceFilter::Flat {
                data: vec![0; slots],
            }
        } else {
            DominanceFilter::Map {
                data: FxHashMap::default(),
            }
        }
    }

    fn clear(&mut self) {
        match self {
            DominanceFilter::Flat { data, .. } => {
                // Fast memset for cache-resident array (~0.5 MB for 64 cells).
                data.fill(0);
            }
            DominanceFilter::Map { data, .. } => {
                data.clear();
            }
        }
    }

    #[inline(always)]
    fn is_dominated(&self, state: &PackedState, layout: &LayoutConfig) -> bool {
        let cell = state.get_location(layout) as u64;
        let keys = state.get_keys(layout) as u64;
        let doors = state.get_doors(layout) as u64;
        let switches = state.get_switches(layout) as u64;
        let idx = (cell << 12) | (keys << 8) | (doors << 4) | switches;
        let ticks = state.get_ticks(layout);

        match self {
            DominanceFilter::Flat { data, .. } => {
                let entry = unsafe { *data.get_unchecked(idx as usize) };
                entry != 0 && entry <= ticks
            }
            DominanceFilter::Map { data, .. } => {
                match data.get(&idx) {
                    Some(&stored) => stored != 0 && stored <= ticks,
                    None => false,
                }
            }
        }
    }

    #[inline(always)]
    fn mark(&mut self, state: &PackedState, layout: &LayoutConfig) {
        let cell = state.get_location(layout) as u64;
        let keys = state.get_keys(layout) as u64;
        let doors = state.get_doors(layout) as u64;
        let switches = state.get_switches(layout) as u64;
        let idx = (cell << 12) | (keys << 8) | (doors << 4) | switches;
        let ticks = state.get_ticks(layout);

        match self {
            DominanceFilter::Flat { data, .. } => {
                unsafe { *data.get_unchecked_mut(idx as usize) = ticks; }
            }
            DominanceFilter::Map { data, .. } => {
                data.insert(idx, ticks);
            }
        }
    }
}

// ─── IDA* Solver ──────────────────────────────────────────────────────────

pub fn solve(
    start: PackedState,
    layout: &LayoutConfig,
    world: &WorldConfig,
    pdb: &PatternDatabase,
) -> Option<(Vec<u8>, u64)> {
    let h = pdb.get_heuristic(&start, layout);
    if h == u32::MAX {
        return None;
    }
    if h == 0 {
        return Some((Vec::new(), 0));
    }

    let mut bound = h as u64;
    let mut path: Vec<u8> = Vec::with_capacity(64);
    let mut nodes_evaluated: u64 = 0;
    let mut filter = DominanceFilter::new(world.cell_count());

    loop {
        filter.clear();
        match dfs(
            start,
            0,
            bound,
            layout,
            world,
            pdb,
            &mut path,
            &mut filter,
            &mut nodes_evaluated,
        ) {
            None => return Some((path.clone(), nodes_evaluated)),
            Some(b) => bound = b,
        }
    }
}

/// Recursive IDA* DFS.
/// Returns `None` when the goal is found (path is in `path`).
/// Returns `Some(min_excess)` when the smallest f-value that exceeded
/// the bound across this subtree is `min_excess`.
fn dfs(
    state: PackedState,
    g: u64,
    bound: u64,
    layout: &LayoutConfig,
    world: &WorldConfig,
    pdb: &PatternDatabase,
    path: &mut Vec<u8>,
    filter: &mut DominanceFilter,
    nodes: &mut u64,
) -> Option<u64> {
    let h = pdb.get_heuristic(&state, layout) as u64;
    let f = g + h;
    if f > bound {
        return Some(f);
    }
    if h == 0 {
        return None; // Goal found
    }

    if filter.is_dominated(&state, layout) {
        return Some(u64::MAX);
    }
    filter.mark(&state, layout);
    *nodes += 1;

    let mut min_excess = u64::MAX;

    for input in 0..INPUT_COUNT {
        if let Some(next) = step(state, input, layout, world) {
            let delta = (next.get_ticks(layout).saturating_sub(state.get_ticks(layout))) as u64;
            let next_g = g + delta.max(1);
            path.push(input);

            match dfs(next, next_g, bound, layout, world, pdb, path, filter, nodes) {
                None => return None, // Found in child
                Some(b) => {
                    if b < min_excess {
                        min_excess = b;
                    }
                }
            }

            path.pop();
        }
    }

    Some(min_excess)
}

// ─── Tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::default_world;

    #[test]
    fn test_solve_default_maze() {
        let layout = LayoutConfig::standard();
        let world = default_world();
        let pdb = PatternDatabase::compute(&layout, &world);
        let mut start = PackedState::zero();
        start.set_location(&layout, world.start_cell);
        let result = solve(start, &layout, &world, &pdb);
        assert!(result.is_some(), "IDA* must find a path");
        let (path, nodes) = result.unwrap();
        assert!(!path.is_empty());
        println!("Default maze: {} inputs, {} nodes", path.len(), nodes);
    }
}
