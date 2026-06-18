//! Universal Stack-Allocated State Engine
//!
//! `PackedState` is a fixed `[u8; 16]` array, making it `Copy`.
//! It lives entirely on the CPU stack/registers — zero heap.
//! A `LayoutConfig` descriptor tells the engine where each field lives.

use std::fmt;

/// Fixed-size state primitive. 16 bytes = 128 bits of addressable scratch.
/// Passed by value (Copy) — no indirection, no heap.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackedState {
    pub data: [u8; 16],
}

impl fmt::Debug for PackedState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PackedState({:02x?})", &self.data)
    }
}

/// Runtime descriptor for field placement inside `PackedState.data`.
/// All offsets are in bytes; masks are in bits within that byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutConfig {
    pub loc_offset: usize,
    pub loc_bytes: usize,
    pub keys_offset: usize,
    pub doors_offset: usize,
    pub switches_offset: usize,
    pub ticks_offset: usize,
    pub tick_cost_offset: usize,
}

impl LayoutConfig {
    /// Standard 8x8 layout:
    ///   loc       : bytes 0..1  (u16, up to 65k cells)
    ///   keys      : byte  2
    ///   doors     : byte  3
    ///   switches  : byte  4
    ///   ticks     : bytes 5..6 (u16)
    ///   tick_cost : byte  7 (accumulated cost for variable terrain)
    pub const fn standard() -> Self {
        LayoutConfig {
            loc_offset: 0,
            loc_bytes: 2,
            keys_offset: 2,
            doors_offset: 3,
            switches_offset: 4,
            ticks_offset: 5,
            tick_cost_offset: 7,
        }
    }
}

impl PackedState {
    /// Zero-initialised state.
    pub const fn zero() -> Self {
        PackedState { data: [0; 16] }
    }

    // ─── Location (multi-byte, little-endian) ─────────────────────────────

    #[inline(always)]
    pub fn get_location(&self, cfg: &LayoutConfig) -> u16 {
        let off = cfg.loc_offset;
        if cfg.loc_bytes == 1 {
            self.data[off] as u16
        } else {
            (self.data[off] as u16) | ((self.data[off + 1] as u16) << 8)
        }
    }

    #[inline(always)]
    pub fn set_location(&mut self, cfg: &LayoutConfig, val: u16) {
        let off = cfg.loc_offset;
        self.data[off] = (val & 0xFF) as u8;
        if cfg.loc_bytes > 1 {
            self.data[off + 1] = ((val >> 8) & 0xFF) as u8;
        }
    }

    // ─── Single-byte bitflags ─────────────────────────────────────────────

    #[inline(always)]
    pub fn get_keys(&self, cfg: &LayoutConfig) -> u8 {
        self.data[cfg.keys_offset]
    }

    #[inline(always)]
    pub fn set_keys(&mut self, cfg: &LayoutConfig, val: u8) {
        self.data[cfg.keys_offset] = val;
    }

    #[inline(always)]
    pub fn has_key(&self, cfg: &LayoutConfig, id: u8) -> bool {
        (self.data[cfg.keys_offset] & (1 << id)) != 0
    }

    #[inline(always)]
    pub fn add_key(&mut self, cfg: &LayoutConfig, id: u8) {
        self.data[cfg.keys_offset] |= 1 << id;
    }

    #[inline(always)]
    pub fn get_doors(&self, cfg: &LayoutConfig) -> u8 {
        self.data[cfg.doors_offset]
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn set_doors(&mut self, cfg: &LayoutConfig, val: u8) {
        self.data[cfg.doors_offset] = val;
    }

    #[inline(always)]
    pub fn is_door_open(&self, cfg: &LayoutConfig, id: u8) -> bool {
        (self.data[cfg.doors_offset] & (1 << id)) != 0
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn toggle_door(&mut self, cfg: &LayoutConfig, id: u8) {
        self.data[cfg.doors_offset] ^= 1 << id;
    }

    #[inline(always)]
    pub fn open_door(&mut self, cfg: &LayoutConfig, id: u8) {
        self.data[cfg.doors_offset] |= 1 << id;
    }

    #[inline(always)]
    pub fn get_switches(&self, cfg: &LayoutConfig) -> u8 {
        self.data[cfg.switches_offset]
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn set_switches(&mut self, cfg: &LayoutConfig, val: u8) {
        self.data[cfg.switches_offset] = val;
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn is_switch_flipped(&self, cfg: &LayoutConfig, id: u8) -> bool {
        (self.data[cfg.switches_offset] & (1 << id)) != 0
    }

    #[inline(always)]
    pub fn toggle_switch(&mut self, cfg: &LayoutConfig, id: u8) {
        self.data[cfg.switches_offset] ^= 1 << id;
    }

    // ─── Ticks (u16, little-endian) ────────────────────────────────────────

    #[inline(always)]
    pub fn get_ticks(&self, cfg: &LayoutConfig) -> u16 {
        let off = cfg.ticks_offset;
        (self.data[off] as u16) | ((self.data[off + 1] as u16) << 8)
    }

    #[inline(always)]
    pub fn set_ticks(&mut self, cfg: &LayoutConfig, val: u16) {
        let off = cfg.ticks_offset;
        self.data[off] = (val & 0xFF) as u8;
        self.data[off + 1] = ((val >> 8) & 0xFF) as u8;
    }

    #[inline(always)]
    pub fn increment_ticks(&mut self, cfg: &LayoutConfig, amount: u16) {
        let current = self.get_ticks(cfg);
        let next = current.saturating_add(amount);
        self.set_ticks(cfg, next);
    }

    // ─── Tick cost accumulator (variable terrain) ────────────────────────

    #[allow(dead_code)]
    #[inline(always)]
    pub fn get_tick_cost(&self, cfg: &LayoutConfig) -> u8 {
        self.data[cfg.tick_cost_offset]
    }

    #[inline(always)]
    pub fn add_tick_cost(&mut self, cfg: &LayoutConfig, amount: u8) {
        let current = self.data[cfg.tick_cost_offset];
        self.data[cfg.tick_cost_offset] = current.saturating_add(amount);
    }

    // ─── Floating-Point (f32) Coordinate Mapping ──────────────────────────
    /// Reads 4 bytes at `offset` as a little-endian f32.
    /// Zero bytes decode to 0.0.
    #[allow(dead_code)]
    #[inline(always)]
    pub fn get_f32(&self, offset: usize) -> f32 {
        let bytes = [self.data[offset], self.data[offset + 1],
                     self.data[offset + 2], self.data[offset + 3]];
        f32::from_le_bytes(bytes)
    }

    /// Writes a little-endian f32 into 4 bytes at `offset`.
    #[allow(dead_code)]
    #[inline(always)]
    pub fn set_f32(&mut self, offset: usize, val: f32) {
        let bytes = val.to_le_bytes();
        self.data[offset..offset + 4].copy_from_slice(&bytes);
    }

    /// Reads 4 bytes at `offset` as a little-endian u32.
    #[allow(dead_code)]
    #[inline(always)]
    pub fn get_u32(&self, offset: usize) -> u32 {
        (self.data[offset] as u32)
            | ((self.data[offset + 1] as u32) << 8)
            | ((self.data[offset + 2] as u32) << 16)
            | ((self.data[offset + 3] as u32) << 24)
    }

    /// Writes a little-endian u32 into 4 bytes at `offset`.
    #[allow(dead_code)]
    #[inline(always)]
    pub fn set_u32(&mut self, offset: usize, val: u32) {
        self.data[offset] = (val & 0xFF) as u8;
        self.data[offset + 1] = ((val >> 8) & 0xFF) as u8;
        self.data[offset + 2] = ((val >> 16) & 0xFF) as u8;
        self.data[offset + 3] = ((val >> 24) & 0xFF) as u8;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_location_roundtrip() {
        let cfg = LayoutConfig::standard();
        let mut s = PackedState::zero();
        s.set_location(&cfg, 42069);
        assert_eq!(s.get_location(&cfg), 42069);
    }

    #[test]
    fn test_keys_bitflags() {
        let cfg = LayoutConfig::standard();
        let mut s = PackedState::zero();
        s.add_key(&cfg, 2);
        assert!(s.has_key(&cfg, 2));
        assert!(!s.has_key(&cfg, 0));
    }

    #[test]
    fn test_doors_toggle() {
        let cfg = LayoutConfig::standard();
        let mut s = PackedState::zero();
        s.toggle_door(&cfg, 1);
        assert!(s.is_door_open(&cfg, 1));
        s.toggle_door(&cfg, 1);
        assert!(!s.is_door_open(&cfg, 1));
    }

    #[test]
    fn test_ticks_increment() {
        let cfg = LayoutConfig::standard();
        let mut s = PackedState::zero();
        s.increment_ticks(&cfg, 5);
        assert_eq!(s.get_ticks(&cfg), 5);
        s.increment_ticks(&cfg, 3);
        assert_eq!(s.get_ticks(&cfg), 8);
    }

    #[test]
    fn test_no_cross_byte_contamination() {
        let cfg = LayoutConfig::standard();
        let mut s = PackedState::zero();
        s.set_location(&cfg, 0xFFFF);
        s.set_keys(&cfg, 0xAB);
        s.set_doors(&cfg, 0xCD);
        s.set_switches(&cfg, 0xEF);
        s.set_ticks(&cfg, 0x1234);
        assert_eq!(s.get_location(&cfg), 0xFFFF);
        assert_eq!(s.get_keys(&cfg), 0xAB);
        assert_eq!(s.get_doors(&cfg), 0xCD);
        assert_eq!(s.get_switches(&cfg), 0xEF);
        assert_eq!(s.get_ticks(&cfg), 0x1234);
    }

    #[test]
    fn test_f32_roundtrip() {
        let mut s = PackedState::zero();
        s.set_f32(0, 3.14159);
        assert!((s.get_f32(0) - 3.14159).abs() < 0.0001);
    }

    #[test]
    fn test_u32_roundtrip() {
        let mut s = PackedState::zero();
        s.set_u32(4, 0xDEADBEEF);
        assert_eq!(s.get_u32(4), 0xDEADBEEF);
    }
}
