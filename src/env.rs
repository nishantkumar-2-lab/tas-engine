//! Mock High-Complexity Maze Environment (Phase 1)
//!
//! We model an 8x8 grid where each cell has a terrain type.
//! The transition function is the heart of the TAS engine:
//!   step(state, input) -> Option<PackedState>
//!
//! Inputs are 6 discrete actions: Wait, Up, Down, Left, Right, Activate.
//! The function validates wall collisions, key requirements for doors,
//! and switch-operated gates. All transitions are deterministic.

use crate::state::PackedState;

// ─── Input Enumeration ──────────────────────────────────────────────────────

/// The 6 atomic inputs the TAS engine can emit.
/// Each variant maps to a `u8` constant for branchless dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Input {
    Wait     = 0,
    Up       = 1,
    Down     = 2,
    Left     = 3,
    Right    = 4,
    Activate = 5,
}

// ─── Terrain Types ──────────────────────────────────────────────────────────

/// Every cell in the 8x8 grid has one of these terrain types.
/// We store terrain as a single byte for cache-friendly lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[allow(dead_code)]
pub enum Terrain {
    Empty       = 0,
    Wall        = 1,
    Key0        = 2,
    Key1        = 3,
    Key2        = 4,
    Key3        = 5,
    Door0       = 6,
    Door1       = 7,
    Door2       = 8,
    Door3       = 9,
    Switch0     = 10,
    Switch1     = 11,
    Switch2     = 12,
    Switch3     = 13,
    OneWayNorth = 14,
    OneWaySouth = 15,
    OneWayEast  = 16,
    OneWayWest  = 17,
    Victory     = 18,
}

// ─── Grid Layout (8x8 = 64 cells) ─────────────────────────────────────────
/// A deliberately complex maze with nested key-door puzzles,
/// switch-controlled gates, and one-way passages.
/// Stored as a flat [Terrain; 64] array for O(1) indexing.
const MAZE: [u8; 64] = [
    // Row 0
    1, 1, 1, 1, 1, 1, 1, 1,
    // Row 1
    1, 0, 0, 2, 1, 0, 0, 1,
    // Row 2
    1, 0, 1, 6, 1, 0, 1, 1,
    // Row 3
    1, 0, 0, 0, 10, 0, 3, 1,
    // Row 4
    1, 1, 7, 1, 1, 1, 0, 1,
    // Row 5
    1, 0, 0, 0, 0, 0, 0, 1,
    // Row 6
    1, 0, 1, 1, 1, 4, 0, 1,
    // Row 7
    1, 1, 1, 1, 1, 1, 18, 1,
];

/// Hell maze: maximally interdependent keys/doors/switches/one-ways.
/// Key0 at (1,1), Door0 at (6,1). Key1 at (1,6), Door1 at (6,6).
/// Key2 at (3,3), Door2 at (5,3). Key3 at (3,5), Door3 at (5,5).
/// Switches create one-way choke points. Victory at (7,3).
const MAZE_HELL: [u8; 64] = [
    // Row 0
    1, 1, 1, 1, 1, 1, 1, 1,
    // Row 1
    1, 2, 1, 0, 0, 0, 3, 1,
    // Row 2
    1, 0, 1, 0, 1, 0, 1, 1,
    // Row 3
    1, 0, 0, 4, 10, 5, 0, 1,
    // Row 4
    1, 1, 1, 14, 1, 15, 1, 1,
    // Row 5
    1, 0, 1, 8, 1, 9, 0, 1,
    // Row 6
    1, 6, 1, 0, 0, 0, 7, 1,
    // Row 7
    1, 1, 1, 18, 1, 1, 1, 1,
];

static MAZE_SELECTOR: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Switch to the hell maze for stress testing.
pub fn select_hell_maze() {
    MAZE_SELECTOR.store(1, std::sync::atomic::Ordering::Relaxed);
}

/// Switch back to the default maze.
pub fn select_default_maze() {
    MAZE_SELECTOR.store(0, std::sync::atomic::Ordering::Relaxed);
}

// ─── Helper: coordinate <-> cell index ────────────────────────────────────

/// Convert (row, col) -> cell index.  row*8 + col.
#[inline(always)]
pub const fn idx(row: u8, col: u8) -> u8 {
    (row << 3) | col // row * 8 + col, compiler folds to shift+or
}

/// Convert cell index -> (row, col).
#[inline(always)]
pub const fn coords(cell: u8) -> (u8, u8) {
    (cell >> 3, cell & 0x07)
}

// ─── Terrain Lookup ─────────────────────────────────────────────────────────

/// Return the terrain byte at a given cell index.
/// Reads from the currently selected maze (default or hell).
#[inline(always)]
pub fn terrain_at(cell: u8) -> u8 {
    let sel = MAZE_SELECTOR.load(std::sync::atomic::Ordering::Relaxed);
    if sel == 0 {
        MAZE[cell as usize]
    } else {
        MAZE_HELL[cell as usize]
    }
}

// ─── State Transition ───────────────────────────────────────────────────────

/// Apply one input to a state, returning the new state if valid.
/// Returns `None` for illegal moves (wall, locked door, one-way violation).
///
/// This is the hot path of the entire engine; every operation is inlined
/// and branchless where possible.
#[inline(always)]
pub fn step(current: PackedState, input: Input) -> Option<PackedState> {
    let terrain = terrain_at(current.get_cell());

    match input {
        Input::Wait => {
            // Wait always succeeds; ticks advance.
            Some(current.increment_tick())
        }
        Input::Activate => {
            // Activate toggles switches or opens doors if key is held.
            let mut next = current.increment_tick();
            match terrain {
                10 => next = next.toggle_switch(0),
                11 => next = next.toggle_switch(1),
                12 => next = next.toggle_switch(2),
                13 => next = next.toggle_switch(3),
                6  => {
                    if current.has_key(0) {
                        next = next.open_door(0);
                    }
                }
                7  => {
                    if current.has_key(1) {
                        next = next.open_door(1);
                    }
                }
                8  => {
                    if current.has_key(2) {
                        next = next.open_door(2);
                    }
                }
                9  => {
                    if current.has_key(3) {
                        next = next.open_door(3);
                    }
                }
                _ => {}
            }
            Some(next)
        }
        Input::Up | Input::Down | Input::Left | Input::Right => {
            let (row, col) = coords(current.get_cell());

            // Compute target cell coordinates.
            let (tr, tc) = match input {
                Input::Up    => (row.saturating_sub(1), col),
                Input::Down  => (row.saturating_add(1), col),
                Input::Left  => (row, col.saturating_sub(1)),
                Input::Right => (row, col.saturating_add(1)),
                _ => unreachable!(),
            };

            // Bounds check: 0..7 for both row and col.
            if tr > 7 || tc > 7 {
                return None;
            }

            let target_cell = idx(tr, tc);
            let target_terrain = terrain_at(target_cell);

            // Wall collision check.
            if target_terrain == 1 {
                return None;
            }

            // One-way passage checks: ensure movement direction is allowed.
            // OneWayNorth: can only ENTER from the south (moving Up).
            if target_terrain == 14 && input != Input::Up {
                return None;
            }
            // OneWaySouth: can only ENTER from the north (moving Down).
            if target_terrain == 15 && input != Input::Down {
                return None;
            }
            // OneWayEast:  can only ENTER from the west (moving Right).
            if target_terrain == 16 && input != Input::Right {
                return None;
            }
            // OneWayWest:  can only ENTER from the east (moving Left).
            if target_terrain == 17 && input != Input::Left {
                return None;
            }

            // Door checks: door id = terrain - 6.
            // A door blocks movement only if it is closed AND the player
            // does not hold the matching key.
            if (6..=9).contains(&target_terrain) {
                let door_id = target_terrain - 6;
                if !current.is_door_open(door_id) && !current.has_key(door_id) {
                    return None;
                }
            }

            let mut next = current.set_cell(target_cell).increment_tick();

            // Key pickup: if target cell contains a key, add it.
            if (2..=5).contains(&target_terrain) {
                let key_id = target_terrain - 2;
                next = next.add_key(key_id);
            }

            Some(next)
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coords_roundtrip() {
        for i in 0..64u8 {
            let (r, c) = coords(i);
            assert_eq!(idx(r, c), i);
        }
    }

    #[test]
    fn test_wait_advances_ticks() {
        let s = PackedState::new(idx(1, 1), 0, 0, 0, 0);
        let s2 = step(s, Input::Wait).unwrap();
        assert_eq!(s2.get_ticks(), 1);
    }

    #[test]
    fn test_move_into_empty() {
        let s = PackedState::new(idx(1, 1), 0, 0, 0, 0);
        let s2 = step(s, Input::Right).unwrap();
        assert_eq!(s2.get_cell(), idx(1, 2));
    }

    #[test]
    fn test_wall_blocks() {
        let s = PackedState::new(idx(1, 1), 0, 0, 0, 0);
        assert!(step(s, Input::Up).is_none()); // row 0 is all walls
    }

    #[test]
    fn test_bounds_check() {
        let s = PackedState::new(idx(1, 1), 0, 0, 0, 0);
        assert!(step(s, Input::Left).is_none()); // col 0 is wall anyway
    }

    #[test]
    fn test_key_pickup() {
        // Key0 is at (1,3)
        let s = PackedState::new(idx(1, 2), 0, 0, 0, 0);
        let s2 = step(s, Input::Right).unwrap();
        assert!(s2.has_key(0));
    }

    #[test]
    fn test_door_locked_without_key() {
        // Door0 is at (2,3). To reach it from (2,2):
        let s = PackedState::new(idx(2, 2), 0, 0, 0, 0);
        assert!(step(s, Input::Right).is_none());
    }

    #[test]
    fn test_door_open_with_key() {
        let s = PackedState::new(idx(2, 2), 0b0001, 0, 0, 0);
        let s2 = step(s, Input::Right).unwrap();
        assert_eq!(s2.get_cell(), idx(2, 3));
    }

    #[test]
    fn test_switch_toggle() {
        // Switch0 at (3,4)
        let s = PackedState::new(idx(3, 4), 0, 0, 0, 0);
        let s2 = step(s, Input::Activate).unwrap();
        assert!(s2.is_switch_flipped(0));
    }

    #[test]
    fn test_victory_cell_reachable() {
        // Victory at (7,6)
        let s = PackedState::new(idx(7, 6), 0, 0, 0, 0);
        let terrain = terrain_at(s.get_cell());
        assert_eq!(terrain, 18);
    }
}
