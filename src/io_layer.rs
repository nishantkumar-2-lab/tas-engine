//! External Memory & Input Abstraction Interface
//!
//! Decouples the search loop from the transition environment via two traits:
//!   - `StateProvider`: captures the current runtime state.
//!   - `InputInjector`: sends an input to the target software.
//!
//! Two backends are provided:
//!   A. `InternalSimulator` — wraps our existing `WorldConfig` + `step()`.
//!   B. `ExternalNativeHook` — structural outline for OS-level process memory
//!      read/write (platform-specific implementations fill the `read/write` fns).

use crate::state::{PackedState, LayoutConfig};
use crate::env::{WorldConfig, step};

// ─── Core Traits ──────────────────────────────────────────────────────────

/// Captures the current software state as a `[u8; 16]` snapshot.
/// In an internal simulator this is just a copy of the current `PackedState`.
/// In an external hook this reads target process memory.
#[allow(dead_code)]
pub trait StateProvider {
    /// Return the live state of the target software.
    fn capture_state(&mut self) -> PackedState;
}

/// Injects one discrete input into the target software.
/// In an internal simulator this calls `step()`.
/// In an external hook this writes to input registers or simulates hardware.
#[allow(dead_code)]
pub trait InputInjector {
    fn inject_input(&mut self, input_id: u8);
}

// ─── Backend A: Internal Simulator ────────────────────────────────────────

/// Wraps a `WorldConfig` and a mutable scratch state to provide
/// fully deterministic, high-throughput offline simulation.
#[allow(dead_code)]
pub struct InternalSimulator<'a> {
    pub layout: &'a LayoutConfig,
    pub world: &'a WorldConfig,
    pub state: PackedState,
}

impl<'a> InternalSimulator<'a> {
    #[allow(dead_code)]
    pub fn new(layout: &'a LayoutConfig, world: &'a WorldConfig, start: PackedState) -> Self {
        InternalSimulator { layout, world, state: start }
    }
}

impl<'a> StateProvider for InternalSimulator<'a> {
    #[inline(always)]
    fn capture_state(&mut self) -> PackedState {
        self.state
    }
}

impl<'a> InputInjector for InternalSimulator<'a> {
    #[inline(always)]
    fn inject_input(&mut self, input_id: u8) {
        if let Some(next) = step(self.state, input_id, self.layout, self.world) {
            self.state = next;
        }
    }
}

// ─── Backend B: External Native Hook (Structural Outline) ─────────────────

/// Platform-independent descriptor for a memory region inside a target process.
#[derive(Clone, Copy, Debug)]
pub struct MemoryRegion {
    #[allow(dead_code)]
    pub base_address: usize,
    #[allow(dead_code)]
    pub size: usize,
}

/// Low-latency process memory boundary adapter.
/// On Windows: uses `ReadProcessMemory` / `WriteProcessMemory`.
/// On Linux:   uses `/proc/<pid>/mem` or `process_vm_readv/writev`.
///
/// This struct holds the configuration; the actual OS calls live in
/// platform-specific `impl` blocks (or `unsafe` helper modules).
pub struct ExternalNativeHook {
    #[allow(dead_code)]
    pub process_id: u32,
    #[allow(dead_code)]
    pub state_region: MemoryRegion,
    #[allow(dead_code)]
    pub input_region: MemoryRegion,
    #[allow(dead_code)]
    pub layout: LayoutConfig,
    #[allow(dead_code)]
    pub cached_state: PackedState,
}

impl ExternalNativeHook {
    /// Create a new hook descriptor without attempting connection.
    #[allow(dead_code)]
    pub fn new(process_id: u32, state_region: MemoryRegion, input_region: MemoryRegion, layout: LayoutConfig) -> Self {
        ExternalNativeHook {
            process_id,
            state_region,
            input_region,
            layout,
            cached_state: PackedState::zero(),
        }
    }

    /// Platform-specific memory read.
    /// Windows: `ReadProcessMemory(handle, addr, buf, size, &mut read)`
    /// Linux:   `process_vm_readv(pid, local_iov, 1, remote_iov, 1, 0)`
    ///
    /// Returns `true` on success.
    #[allow(dead_code)]
    pub fn read_target_memory(&self, _offset: usize, _buf: &mut [u8]) -> bool {
        // OS-specific implementation goes here.
        false
    }

    /// Platform-specific memory write.
    /// Windows: `WriteProcessMemory(handle, addr, buf, size, &mut written)`
    /// Linux:   `process_vm_writev(pid, local_iov, 1, remote_iov, 1, 0)`
    ///
    /// Returns `true` on success.
    #[allow(dead_code)]
    pub fn write_target_memory(&self, _offset: usize, _buf: &[u8]) -> bool {
        // OS-specific implementation goes here.
        false
    }
}

impl StateProvider for ExternalNativeHook {
    fn capture_state(&mut self) -> PackedState {
        let mut buf = [0u8; 16];
        if self.read_target_memory(0, &mut buf) {
            self.cached_state = PackedState { data: buf };
        }
        self.cached_state
    }
}

impl InputInjector for ExternalNativeHook {
    fn inject_input(&mut self, input_id: u8) {
        let _ = self.write_target_memory(0, &[input_id]);
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::default_world;

    #[test]
    fn test_internal_simulator_capture() {
        let layout = LayoutConfig::standard();
        let world = default_world();
        let mut start = PackedState::zero();
        start.set_location(&layout, world.start_cell);
        let mut sim = InternalSimulator::new(&layout, &world, start);
        let s = sim.capture_state();
        assert_eq!(s.get_location(&layout), world.start_cell);
    }

    #[test]
    fn test_internal_simulator_inject() {
        let layout = LayoutConfig::standard();
        let world = default_world();
        let mut start = PackedState::zero();
        start.set_location(&layout, world.start_cell);
        let mut sim = InternalSimulator::new(&layout, &world, start);
        sim.inject_input(4); // Right
        let s = sim.capture_state();
        assert_ne!(s.get_location(&layout), world.start_cell);
    }
}
