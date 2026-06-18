//! Bit-Packed State Engine (Phase 1)
//! 
//! Every software state is compressed into a single `u64` to guarantee:
//! - O(1) state transitions via bitwise operations
//! - Cache-line alignment (8 bytes per state)
//! - Zero-cost serialization (identity function on u64)
//!
//! Layout:
//!   Bits 0..6    : Cell index (0-63 for 8x8 grid)
//!   Bits 6..10   : Key bitflags (4 keys)
//!   Bits 10..14  : Door bitflags (4 doors, open/closed)
//!   Bits 14..18  : Switch bitflags (4 switches)
//!   Bits 18..34  : Tick counter (16 bits, up to 65,535 frames)
//!   Bits 34..64  : Reserved for expansion

/// Newtype wrapper for a packed u64 state.
/// Using a newtype prevents accidental mixing with raw integers
/// and allows us to implement methods with zero overhead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedState(pub u64);

// ─── Bitfield Constants ─────────────────────────────────────────────────────
// Masks isolate specific bit-ranges; shifts align values into those ranges.

const CELL_MASK: u64      = 0x3F;        // 6 bits:  0b111111
const CELL_SHIFT: u64     = 0;

const KEY_MASK: u64       = 0x0F;        // 4 bits
const KEY_SHIFT: u64      = 6;

const DOOR_MASK: u64      = 0x0F;        // 4 bits
const DOOR_SHIFT: u64     = 10;

const SWITCH_MASK: u64    = 0x0F;        // 4 bits
const SWITCH_SHIFT: u64   = 14;

const TICK_MASK: u64      = 0xFFFF;      // 16 bits
const TICK_SHIFT: u64     = 18;

// ─── Constructor ────────────────────────────────────────────────────────────

impl PackedState {
    /// Build a state from its constituent fields.
    /// Every parameter is masked to guarantee it fits inside its bitfield,
    /// making this constructor branchless.
    #[inline(always)]
    pub const fn new(cell: u8, keys: u8, doors: u8, switches: u8, ticks: u16) -> Self {
        let mut raw: u64 = 0;
        raw |= ((cell as u64) & CELL_MASK) << CELL_SHIFT;
        raw |= ((keys as u64) & KEY_MASK) << KEY_SHIFT;
        raw |= ((doors as u64) & DOOR_MASK) << DOOR_SHIFT;
        raw |= ((switches as u64) & SWITCH_MASK) << SWITCH_SHIFT;
        raw |= ((ticks as u64) & TICK_MASK) << TICK_SHIFT;
        PackedState(raw)
    }

    // ─── Cell (position) ──────────────────────────────────────────────────

    /// Extract cell index (0..63).
    /// Uses mask-and-shift: `(raw >> shift) & mask`.
    /// Branchless, single ALU instruction on x86_64.
    #[inline(always)]
    pub const fn get_cell(self) -> u8 {
        ((self.0 >> CELL_SHIFT) & CELL_MASK) as u8
    }

    /// Replace cell index in-place.
    /// Clears old bits with `& !mask`, then ORs new value.
    #[inline(always)]
    pub const fn set_cell(self, cell: u8) -> Self {
        let cleared = self.0 & !(CELL_MASK << CELL_SHIFT);
        PackedState(cleared | (((cell as u64) & CELL_MASK) << CELL_SHIFT))
    }

    // ─── Keys ───────────────────────────────────────────────────────────────

    /// Extract the raw keys bitmask (0..15).
    /// Used by the Pattern Database for O(1) heuristic indexing.
    #[inline(always)]
    pub const fn get_keys(self) -> u8 {
        ((self.0 >> KEY_SHIFT) & KEY_MASK) as u8
    }

    /// Test if key `id` (0..3) is collected.
    /// `1 << id` creates the bit flag; AND with keys field tests it.
    #[inline(always)]
    pub const fn has_key(self, id: u8) -> bool {
        let keys = ((self.0 >> KEY_SHIFT) & KEY_MASK) as u8;
        (keys & (1 << id)) != 0
    }

    /// Add key `id` (0..3) to the keyring.
    /// ORs the flag into the keys field.
    #[inline(always)]
    pub const fn add_key(self, id: u8) -> Self {
        let flag = ((1u64 << id) & KEY_MASK) << KEY_SHIFT;
        PackedState(self.0 | flag)
    }

    // ─── Doors ──────────────────────────────────────────────────────────────

    /// Test if door `id` (0..3) is open.
    #[inline(always)]
    pub const fn is_door_open(self, id: u8) -> bool {
        let doors = ((self.0 >> DOOR_SHIFT) & DOOR_MASK) as u8;
        (doors & (1 << id)) != 0
    }

    /// Toggle door `id` (0..3) open/closed.
    /// XOR flips the bit without branches.
    #[inline(always)]
    pub const fn toggle_door(self, id: u8) -> Self {
        let flag = ((1u64 << id) & DOOR_MASK) << DOOR_SHIFT;
        PackedState(self.0 ^ flag)
    }

    /// Force door `id` open (idempotent).
    #[inline(always)]
    pub const fn open_door(self, id: u8) -> Self {
        let flag = ((1u64 << id) & DOOR_MASK) << DOOR_SHIFT;
        PackedState(self.0 | flag)
    }

    /// Extract the raw doors bitmask (0..15).
    #[inline(always)]
    pub const fn get_doors(self) -> u8 {
        ((self.0 >> DOOR_SHIFT) & DOOR_MASK) as u8
    }

    /// Force door `id` closed (idempotent).
    #[allow(dead_code)]
    #[inline(always)]
    pub const fn close_door(self, id: u8) -> Self {
        let flag = ((1u64 << id) & DOOR_MASK) << DOOR_SHIFT;
        PackedState(self.0 & !flag)
    }

    // ─── Switches ───────────────────────────────────────────────────────────

    /// Test if switch `id` (0..3) is flipped.
    #[inline(always)]
    pub const fn is_switch_flipped(self, id: u8) -> bool {
        let switches = ((self.0 >> SWITCH_SHIFT) & SWITCH_MASK) as u8;
        (switches & (1 << id)) != 0
    }

    /// Extract the raw switches bitmask (0..15).
    #[inline(always)]
    pub const fn get_switches(self) -> u8 {
        ((self.0 >> SWITCH_SHIFT) & SWITCH_MASK) as u8
    }

    /// Toggle switch `id` (0..3).
    #[inline(always)]
    pub const fn toggle_switch(self, id: u8) -> Self {
        let flag = ((1u64 << id) & SWITCH_MASK) << SWITCH_SHIFT;
        PackedState(self.0 ^ flag)
    }

    // ─── Ticks ──────────────────────────────────────────────────────────────

    /// Extract tick counter.
    #[inline(always)]
    pub const fn get_ticks(self) -> u16 {
        ((self.0 >> TICK_SHIFT) & TICK_MASK) as u16
    }

    /// Increment tick counter by 1.
    /// Wraps at 2^16-1 (saturates to prevent overflow into reserved bits).
    #[inline(always)]
    pub const fn increment_tick(self) -> Self {
        let current = ((self.0 >> TICK_SHIFT) & TICK_MASK) as u16;
        let next = current.saturating_add(1);
        let cleared = self.0 & !(TICK_MASK << TICK_SHIFT);
        PackedState(cleared | ((next as u64) << TICK_SHIFT))
    }

    // ─── Raw access ─────────────────────────────────────────────────────────

    /// Direct raw access for hashing, serialization, and closed-set lookups.
    #[inline(always)]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Reconstruct from raw u64 (useful for deserialization).
    #[inline(always)]
    pub const fn from_raw(raw: u64) -> Self {
        PackedState(raw)
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_getters() {
        let s = PackedState::new(42, 0b1010, 0b0101, 0b1111, 12345);
        assert_eq!(s.get_cell(), 42);
        assert_eq!(s.has_key(1), true);
        assert_eq!(s.has_key(0), false);
        assert_eq!(s.is_door_open(0), true);
        assert_eq!(s.is_door_open(1), false);
        assert_eq!(s.is_switch_flipped(0), true);
        assert_eq!(s.get_ticks(), 12345);
    }

    #[test]
    fn test_set_cell() {
        let s = PackedState::new(0, 0, 0, 0, 0);
        let s2 = s.set_cell(63);
        assert_eq!(s2.get_cell(), 63);
        // Original unchanged (Copy type)
        assert_eq!(s.get_cell(), 0);
    }

    #[test]
    fn test_add_key() {
        let s = PackedState::new(0, 0, 0, 0, 0);
        let s2 = s.add_key(2);
        assert!(s2.has_key(2));
        assert!(!s2.has_key(0));
    }

    #[test]
    fn test_toggle_door() {
        let s = PackedState::new(0, 0, 0, 0, 0);
        let s2 = s.toggle_door(1);
        assert!(s2.is_door_open(1));
        let s3 = s2.toggle_door(1);
        assert!(!s3.is_door_open(1));
    }

    #[test]
    fn test_increment_tick() {
        let s = PackedState::new(0, 0, 0, 0, 0);
        let s2 = s.increment_tick();
        assert_eq!(s2.get_ticks(), 1);
        let mut s3 = s;
        for _ in 0..100 {
            s3 = s3.increment_tick();
        }
        assert_eq!(s3.get_ticks(), 100);
    }

    #[test]
    fn test_saturation() {
        let s = PackedState::new(0, 0, 0, 0, 65534);
        let s2 = s.increment_tick();
        assert_eq!(s2.get_ticks(), 65535);
        let s3 = s2.increment_tick();
        assert_eq!(s3.get_ticks(), 65535); // saturates
    }

    #[test]
    fn test_all_bits_roundtrip() {
        // Ensure every field survives a full encode/decode cycle.
        let original = PackedState::new(63, 0x0F, 0x0F, 0x0F, 65535);
        let raw = original.raw();
        let recovered = PackedState::from_raw(raw);
        assert_eq!(original, recovered);
    }
}
