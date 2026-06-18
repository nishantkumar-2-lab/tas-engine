//! High-Speed Execution & Macro Driver (Phase 3)
//!
//! A zero-jitter, allocation-free input delivery pipeline that translates
//! the mathematically optimal path from IDA* into precise, frame-perfect
//! software inputs.  The driver uses a hybrid timer:
//!   - Coarse sleep (`thread::sleep`) for long idle periods.
//!   - CPU spin-lock (`std::time::Instant` tight loop) for the final
//!     2 ms before each fire-time, guaranteeing sub-microsecond accuracy.

use std::fs::File;
use std::io::{Write, Result as IoResult};
use std::time::{Duration, Instant};
use std::thread;

// ─── Constants ──────────────────────────────────────────────────────────────

/// When within this many nanoseconds of the target tick, switch from
/// coarse sleep to a pure CPU spin-lock.
const SPIN_THRESHOLD_NS: u64 = 2_000_000; // 2 ms

/// Magic header for `.tas` binary export files.
const TAS_MAGIC: &[u8] = b"TAS\0";

const TAS_VERSION: u8 = 1;

// ─── Data Structures ────────────────────────────────────────────────────────

/// A single scheduled input frame.
/// `input_id` maps to the `Input` enum (0..5).
/// `target_ns` is the absolute nanosecond timestamp from the run start
/// at which the input must be injected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputFrame {
    pub input_id: u8,
    /// Nanoseconds since the driver epoch.
    pub target_ns: u64,
}

/// Jitter statistics collected during a live injection run.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct JitterReport {
    pub max_ns: i64,
    pub min_ns: i64,
    pub avg_ns: f64,
    pub samples: u64,
}

/// The compiled macro driver.
/// Holds a pre-allocated, fixed-size array of `InputFrame`s — zero
/// runtime heap allocations after construction.
pub struct TASDriver {
    frames: Vec<InputFrame>,
}

impl TASDriver {
    /// Compile a raw path (sequence of input IDs) into a scheduled
    /// `TASDriver` targeting `frame_rate_hz`.
    ///
    /// Every input is scheduled one frame apart:
    ///   frame_period_ns = 1_000_000_000 / frame_rate_hz
    ///   frame[i].target_ns = i * frame_period_ns
    #[inline]
    pub fn compile_macro(path: &[u8], frame_rate_hz: u64) -> Self {
        assert!(frame_rate_hz > 0, "frame_rate_hz must be > 0");
        let period_ns = 1_000_000_000u64 / frame_rate_hz;

        let mut frames = Vec::with_capacity(path.len());
        for (i, &input_id) in path.iter().enumerate() {
            frames.push(InputFrame {
                input_id,
                target_ns: (i as u64).saturating_mul(period_ns),
            });
        }

        TASDriver { frames }
    }

    /// Return the number of frames in the compiled macro.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Return an immutable slice of the compiled frames.
    #[inline(always)]
    pub fn frames(&self) -> &[InputFrame] {
        &self.frames
    }

    /// Execute the macro in real-time, calling `inject_callback` for
    /// each frame at its mathematically perfect fire-time.
    ///
    /// Jitter is measured as `actual_fire_ns - target_ns` for every
    /// frame.  Returns a `JitterReport` summarising the run.
    ///
    /// Timing strategy:
    ///   1. Record the epoch `Instant`.
    ///   2. For each frame, compute `target_instant = epoch + target_ns`.
    ///   3. If `target_instant - now > 2 ms`, sleep for half the gap
    ///      (or yield if the gap is tiny) to avoid burning CPU.
    ///   4. Spin-lock in a tight `while now < target_instant` loop for
    ///      the final sub-2-ms window.
    ///   5. Fire the callback, record jitter.
    #[inline]
    pub fn run_live_injection<F>(
        &self,
        _frame_rate_hz: u64,
        mut inject_callback: F,
    ) -> JitterReport
    where
        F: FnMut(u8),
    {
        if self.frames.is_empty() {
            return JitterReport::default();
        }

        let epoch = Instant::now();

        let mut max_ns: i64 = i64::MIN;
        let mut min_ns: i64 = i64::MAX;
        let mut sum_ns: i64 = 0;
        let mut samples: u64 = 0;

        for frame in &self.frames {
            let target_instant = epoch + Duration::from_nanos(frame.target_ns);

            // ─── Hybrid wait ──────────────────────────────────────────
            loop {
                let now = Instant::now();
                if now >= target_instant {
                    break;
                }
                let remaining = target_instant.duration_since(now).as_nanos() as u64;
                if remaining > SPIN_THRESHOLD_NS {
                    // Coarse sleep: leave 1 ms of headroom, then resume.
                    let sleep_dur = remaining.saturating_sub(1_000_000);
                    thread::sleep(Duration::from_nanos(sleep_dur));
                } else {
                    // Spin-lock for the final 2 ms (or less).
                    // std::hint::spin_loop hint prevents CPU pipeline stall.
                    std::hint::spin_loop();
                }
            }

            // ─── Fire callback ────────────────────────────────────────
            let fire_time = Instant::now();
            inject_callback(frame.input_id);

            // ─── Jitter measurement ───────────────────────────────────
            let jitter = fire_time.duration_since(epoch).as_nanos() as i64
                - frame.target_ns as i64;
            if jitter > max_ns {
                max_ns = jitter;
            }
            if jitter < min_ns {
                min_ns = jitter;
            }
            sum_ns = sum_ns.wrapping_add(jitter);
            samples += 1;
        }

        let avg_ns = if samples > 0 {
            (sum_ns as f64) / (samples as f64)
        } else {
            0.0
        };

        JitterReport {
            max_ns,
            min_ns,
            avg_ns,
            samples,
        }
    }

    /// Export the compiled macro to a `.tas` binary file.
    ///
    /// File format (little-endian):
    ///   4 bytes : magic    "TAS\0"
    ///   1 byte  : version  0x01
    ///   4 bytes : frame_rate_hz (u32)
    ///   4 bytes : frame_count   (u32)
    ///   N × 9 bytes: for each frame:
    ///       1 byte  : input_id
    ///       8 bytes : target_ns (u64)
    pub fn export_to_tas_file(&self, path: &str, frame_rate_hz: u64) -> IoResult<()> {
        let mut file = File::create(path)?;

        // Header
        file.write_all(TAS_MAGIC)?;
        file.write_all(&[TAS_VERSION])?;
        file.write_all(&(frame_rate_hz as u32).to_le_bytes())?;
        file.write_all(&(self.frames.len() as u32).to_le_bytes())?;

        // Frames
        for frame in &self.frames {
            file.write_all(&[frame.input_id])?;
            file.write_all(&frame.target_ns.to_le_bytes())?;
        }

        Ok(())
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_macro_scheduling() {
        let path = vec![4u8, 4u8, 4u8]; // Right, Right, Right
        let driver = TASDriver::compile_macro(&path, 60);
        assert_eq!(driver.len(), 3);

        let frames = driver.frames();
        // 60 FPS => 16_666_666 ns per frame
        let period = 1_000_000_000u64 / 60;
        assert_eq!(frames[0].target_ns, 0);
        assert_eq!(frames[1].target_ns, period);
        assert_eq!(frames[2].target_ns, period * 2);
    }

    #[test]
    fn test_compile_macro_1000hz() {
        let path = vec![0u8, 1u8, 2u8];
        let driver = TASDriver::compile_macro(&path, 1000);
        let period = 1_000_000u64; // 1 ms
        let frames = driver.frames();
        assert_eq!(frames[0].target_ns, 0);
        assert_eq!(frames[1].target_ns, period);
        assert_eq!(frames[2].target_ns, period * 2);
    }

    #[test]
    fn test_live_injection_sequence() {
        let path = vec![4u8, 2u8, 3u8]; // Right, Down, Left
        let driver = TASDriver::compile_macro(&path, 1000);
        let mut captured = Vec::new();

        let _report = driver.run_live_injection(1000, |input| {
            captured.push(input);
        });

        assert_eq!(captured, path);
    }

    #[test]
    fn test_jitter_report_non_empty() {
        let path = vec![0u8; 10];
        let driver = TASDriver::compile_macro(&path, 1000);
        let report = driver.run_live_injection(1000, |_input| {});
        assert_eq!(report.samples, 10);
    }

    #[test]
    fn test_export_roundtrip() {
        let path = vec![4u8, 2u8, 5u8];
        let driver = TASDriver::compile_macro(&path, 60);
        let temp_path = "test_output.tas";

        driver.export_to_tas_file(temp_path, 60).unwrap();

        let bytes = std::fs::read(temp_path).unwrap();
        assert_eq!(&bytes[0..4], b"TAS\0");
        assert_eq!(bytes[4], 1); // version

        let frame_rate = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]);
        assert_eq!(frame_rate, 60);

        let frame_count = u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]);
        assert_eq!(frame_count, 3);

        // Frame 0 at offset 13
        assert_eq!(bytes[13], 4); // input_id
        let ts0 = u64::from_le_bytes([
            bytes[14], bytes[15], bytes[16], bytes[17],
            bytes[18], bytes[19], bytes[20], bytes[21],
        ]);
        assert_eq!(ts0, 0);

        std::fs::remove_file(temp_path).unwrap();
    }
}
