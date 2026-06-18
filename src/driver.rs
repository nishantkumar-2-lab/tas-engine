//! High-Speed Execution & Macro Driver
//!
//! Zero-jitter input delivery pipeline. Hybrid timer:
//!   - Coarse sleep for gaps > 2 ms.
//!   - CPU spin-lock for the final sub-2-ms window.

use std::fs::File;
use std::io::{Write, Result as IoResult};
use std::time::{Duration, Instant};
use std::thread;

const SPIN_THRESHOLD_NS: u64 = 2_000_000;
#[allow(dead_code)]
const TAS_MAGIC: &[u8] = b"TAS\0";
#[allow(dead_code)]
const TAS_VERSION: u8 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputFrame {
    pub input_id: u8,
    pub target_ns: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct JitterReport {
    pub max_ns: i64,
    pub min_ns: i64,
    pub avg_ns: f64,
    pub samples: u64,
}

pub struct TASDriver {
    frames: Vec<InputFrame>,
}

impl TASDriver {
    #[inline]
    pub fn compile_macro(path: &[u8], frame_rate_hz: u64) -> Self {
        assert!(frame_rate_hz > 0);
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

    #[allow(dead_code)]
    #[inline(always)]
    pub fn len(&self) -> usize { self.frames.len() }
    #[allow(dead_code)]
    #[inline(always)]
    pub fn frames(&self) -> &[InputFrame] { &self.frames }

    #[inline]
    pub fn run_live_injection<F>(&self, _frame_rate_hz: u64, mut inject: F) -> JitterReport
    where F: FnMut(u8) {
        if self.frames.is_empty() { return JitterReport::default(); }
        let epoch = Instant::now();
        let mut max_ns = i64::MIN;
        let mut min_ns = i64::MAX;
        let mut sum_ns: i64 = 0;
        let mut samples = 0u64;

        for frame in &self.frames {
            let target = epoch + Duration::from_nanos(frame.target_ns);
            loop {
                let now = Instant::now();
                if now >= target { break; }
                let rem = target.duration_since(now).as_nanos() as u64;
                if rem > SPIN_THRESHOLD_NS {
                    thread::sleep(Duration::from_nanos(rem.saturating_sub(1_000_000)));
                } else {
                    std::hint::spin_loop();
                }
            }
            let fire = Instant::now();
            inject(frame.input_id);
            let jitter = fire.duration_since(epoch).as_nanos() as i64 - frame.target_ns as i64;
            max_ns = max_ns.max(jitter);
            min_ns = min_ns.min(jitter);
            sum_ns = sum_ns.wrapping_add(jitter);
            samples += 1;
        }
        let avg_ns = if samples > 0 { (sum_ns as f64) / (samples as f64) } else { 0.0 };
        JitterReport { max_ns, min_ns, avg_ns, samples }
    }

    /// Inject inputs synchronized to an external frame counter.
    /// Spins until `poll_counter()` returns a value > last_frame,
    /// then fires the next input immediately.
    /// This achieves single-cycle alignment with the target software.
    #[allow(dead_code)]
    #[inline]
    pub fn run_memory_sync_injection<F, P>(&self, mut inject: F, mut poll_counter: P) -> JitterReport
    where
        F: FnMut(u8),
        P: FnMut() -> u64,
    {
        if self.frames.is_empty() { return JitterReport::default(); }
        let epoch = Instant::now();
        let mut max_ns = i64::MIN;
        let mut min_ns = i64::MAX;
        let mut sum_ns: i64 = 0;
        let mut samples = 0u64;
        let mut last_frame = poll_counter();

        for frame in &self.frames {
            // Spin-lock until the target increments its frame counter.
            loop {
                let current = poll_counter();
                if current > last_frame {
                    last_frame = current;
                    break;
                }
                std::hint::spin_loop();
            }
            let fire = Instant::now();
            inject(frame.input_id);
            let jitter = fire.duration_since(epoch).as_nanos() as i64 - frame.target_ns as i64;
            max_ns = max_ns.max(jitter);
            min_ns = min_ns.min(jitter);
            sum_ns = sum_ns.wrapping_add(jitter);
            samples += 1;
        }
        let avg_ns = if samples > 0 { (sum_ns as f64) / (samples as f64) } else { 0.0 };
        JitterReport { max_ns, min_ns, avg_ns, samples }
    }

    #[allow(dead_code)]
    pub fn export_to_tas_file(&self, path: &str, frame_rate_hz: u64) -> IoResult<()> {
        let mut file = File::create(path)?;
        file.write_all(TAS_MAGIC)?;
        file.write_all(&[TAS_VERSION])?;
        file.write_all(&(frame_rate_hz as u32).to_le_bytes())?;
        file.write_all(&(self.frames.len() as u32).to_le_bytes())?;
        for frame in &self.frames {
            file.write_all(&[frame.input_id])?;
            file.write_all(&frame.target_ns.to_le_bytes())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_macro_scheduling() {
        let driver = TASDriver::compile_macro(&[4, 4, 4], 60);
        assert_eq!(driver.len(), 3);
        let period = 1_000_000_000u64 / 60;
        assert_eq!(driver.frames()[0].target_ns, 0);
        assert_eq!(driver.frames()[1].target_ns, period);
    }

    #[test]
    fn test_live_injection_sequence() {
        let driver = TASDriver::compile_macro(&[4, 2, 3], 1000);
        let mut captured = Vec::new();
        let _ = driver.run_live_injection(1000, |input| captured.push(input));
        assert_eq!(captured, vec![4, 2, 3]);
    }
}
