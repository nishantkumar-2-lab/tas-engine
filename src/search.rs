//! IDA* Search Engine (Phase 2)
//!
//! Iterative Deepening A* with:
//! - Pre-computed Pattern Database heuristic (O(1) lookup)
//! - Generational flat-array dominance filter (no heap per iteration)
//! - Stack-only recursion depth (path buffer lives on the caller's frame)
//!
//! The search minimizes the number of ticks (inputs) to reach the
//! victory cell.  Every input — even Wait — costs exactly one tick.

use crate::state::PackedState;
use crate::env::{terrain_at, Input, step};
use crate::pdb::PatternDatabase;

/// Solve the maze from `start` to the victory cell.
/// Returns `(path, nodes_evaluated)` or `None` if unsolvable.
pub fn solve(start: PackedState, pdb: &PatternDatabase) -> Option<(Vec<u8>, u64)> {
    let h0 = pdb.get_heuristic(start);
    if h0 == u32::MAX {
        return None; // start is walled off even in the relaxed graph
    }

    let mut threshold = h0;
    // Reusable path buffer — only one allocation for the entire search.
    let mut path = Vec::with_capacity(256);
    let mut nodes_evaluated: u64 = 0;

    // Dominance filter: 2^18 entries = 262 144 slots.
    // Each slot stores (generation, best_ticks_seen) for a
    // (Cell, Keys, Doors, Switches) configuration.
    const TABLE_SIZE: usize = 1 << 18; // 262_144
    let mut dom_gen = vec![0u32; TABLE_SIZE];
    let mut dom_ticks = vec![0u16; TABLE_SIZE];
    let mut current_gen: u32 = 1;

    loop {
        path.clear();
        let mut next_threshold = u32::MAX;

        let found = dfs(
            start,
            0,
            threshold,
            &mut path,
            pdb,
            &mut dom_gen,
            &mut dom_ticks,
            current_gen,
            &mut next_threshold,
            &mut nodes_evaluated,
        );

        if found {
            return Some((path, nodes_evaluated));
        }

        if next_threshold == u32::MAX {
            return None; // exhausted state space
        }

        threshold = next_threshold;
        current_gen += 1;
        if current_gen == 0 {
            // Generation counter wrapped — hard-reset the table.
            dom_gen.fill(0);
            current_gen = 1;
        }
    }
}

/// Recursive depth-first search for IDA*.
///
/// Returns `true` when the goal is found.  `next_threshold` is updated
/// with the smallest f-score that exceeded the current `threshold`.
#[inline(always)]
fn dfs(
    current: PackedState,
    g: u32,
    threshold: u32,
    path: &mut Vec<u8>,
    pdb: &PatternDatabase,
    dom_gen: &mut [u32],
    dom_ticks: &mut [u16],
    current_gen: u32,
    next_threshold: &mut u32,
    nodes_evaluated: &mut u64,
) -> bool {
    *nodes_evaluated += 1;

    // ─── Goal test ──────────────────────────────────────────────────────
    // Terrain::Victory == 18.
    if terrain_at(current.get_cell()) == 18 {
        return true;
    }

    let h = pdb.get_heuristic(current);
    let f = g.saturating_add(h);

    // ─── Threshold prune ────────────────────────────────────────────────
    if f > threshold {
        if f < *next_threshold {
            *next_threshold = f;
        }
        return false;
    }

    // ─── Dominance prune ────────────────────────────────────────────────
    // If we have already visited this (Cell, Keys, Doors, Switches) with
    // fewer or equal ticks, this path is strictly worse — kill it.
    let dom_idx = dominance_index(current);
    if dom_gen[dom_idx] == current_gen {
        if current.get_ticks() >= dom_ticks[dom_idx] {
            return false;
        }
        dom_ticks[dom_idx] = current.get_ticks();
    } else {
        dom_gen[dom_idx] = current_gen;
        dom_ticks[dom_idx] = current.get_ticks();
    }

    // ─── Expand children ────────────────────────────────────────────────
    // Order: movement inputs first (encourage progress), then Activate,
    // then Wait (discourage stalling).
    const INPUTS: [u8; 6] = [
        Input::Up as u8,
        Input::Down as u8,
        Input::Left as u8,
        Input::Right as u8,
        Input::Activate as u8,
        Input::Wait as u8,
    ];

    for &input_id in &INPUTS {
        let input = match input_id {
            0 => Input::Wait,
            1 => Input::Up,
            2 => Input::Down,
            3 => Input::Left,
            4 => Input::Right,
            5 => Input::Activate,
            _ => unreachable!(),
        };

        if let Some(next) = step(current, input) {
            path.push(input_id);
            if dfs(
                next,
                g.saturating_add(1),
                threshold,
                path,
                pdb,
                dom_gen,
                dom_ticks,
                current_gen,
                next_threshold,
                nodes_evaluated,
            ) {
                return true;
            }
            path.pop();
        }
    }

    false
}

/// Compute the flat-array index for the dominance filter.
/// Collapses the (Cell, Keys, Doors, Switches) tuple into 18 bits:
///   cell     : bits 17..12  (6 bits)
///   keys     : bits 11..8   (4 bits)
///   doors    : bits 7..4    (4 bits)
///   switches : bits 3..0    (4 bits)
#[inline(always)]
fn dominance_index(state: PackedState) -> usize {
    let cell = state.get_cell() as usize;
    let keys = state.get_keys() as usize;
    let doors = state.get_doors() as usize;
    let switches = state.get_switches() as usize;

    (cell << 12) | (keys << 8) | (doors << 4) | switches
}

// ─── Tests ─────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::idx;

    #[test]
    fn test_solve_already_at_victory() {
        let pdb = PatternDatabase::compute(idx(7, 6));
        let start = PackedState::new(idx(7, 6), 0, 0, 0, 0);
        let solution = solve(start, &pdb);
        assert!(solution.is_some());
        let (path, nodes) = solution.unwrap();
        assert!(path.is_empty());
        assert!(nodes > 0);
    }

    #[test]
    fn test_solve_one_step_from_victory() {
        let pdb = PatternDatabase::compute(idx(7, 6));
        let start = PackedState::new(idx(6, 6), 0, 0, 0, 0);
        let solution = solve(start, &pdb);
        assert!(solution.is_some());
        let (path, _nodes) = solution.unwrap();
        assert_eq!(path.len(), 1);
    }

    #[test]
    fn test_solve_full_maze() {
        let pdb = PatternDatabase::compute(idx(7, 6));
        let start = PackedState::new(idx(1, 1), 0, 0, 0, 0);
        let solution = solve(start, &pdb);
        assert!(solution.is_some(), "IDA* must find a path to victory");
        let (path, nodes) = solution.unwrap();
        assert!(!path.is_empty(), "path should contain at least one input");
        println!("Full maze solved: {} inputs, {} nodes evaluated", path.len(), nodes);
    }

    #[test]
    fn test_solve_hell_maze() {
        crate::env::select_hell_maze();
        let pdb = PatternDatabase::compute(idx(7, 3));
        let start = PackedState::new(idx(1, 3), 0, 0, 0, 0);
        let solution = solve(start, &pdb);
        assert!(solution.is_some(), "IDA* must solve the hell maze");
        let (path, nodes) = solution.unwrap();
        assert!(!path.is_empty());
        println!("Hell maze solved: {} inputs, {} nodes evaluated", path.len(), nodes);
        crate::env::select_default_maze();
    }
}
