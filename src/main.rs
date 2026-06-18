mod state;
mod env;
mod pdb;
mod search;
mod driver;

use state::PackedState;
use env::{Input, step, terrain_at};
use pdb::PatternDatabase;
use search::solve;
use std::time::Instant;

#[allow(dead_code)]
fn input_name(id: u8) -> &'static str {
    match id {
        0 => "Wait",
        1 => "Up",
        2 => "Down",
        3 => "Left",
        4 => "Right",
        5 => "Activate",
        _ => "???",
    }
}

fn benchmark_maze(name: &str, victory: u8, start: PackedState) {
    println!("\n[{}]", name);
    let pdb = PatternDatabase::compute(victory);
    let t0 = Instant::now();
    let result = solve(start, &pdb);
    let dt = t0.elapsed().as_micros();

    match result {
        None => println!("  FAILED: unsolvable"),
        Some((path, nodes)) => {
            println!("  Path: {} inputs", path.len());
            println!("  Nodes: {}", nodes);
            println!("  Time: {} us", dt);
            if dt > 0 {
                println!("  Throughput: {:.2} M states/sec",
                    (nodes as f64) / (dt as f64));
            }
        }
    }
}

fn raw_throughput_benchmark() {
    println!("\n[Raw Throughput Micro-Benchmark]");
    const ITERATIONS: u64 = 5_000_000;
    let state = PackedState::new(env::idx(3, 3), 0b1111, 0b1111, 0b1111, 100);
    let inputs = [
        Input::Wait, Input::Up, Input::Down,
        Input::Left, Input::Right, Input::Activate,
    ];

    let t0 = Instant::now();
    let mut total: u64 = 0;
    let mut s = state;
    for _ in 0..ITERATIONS {
        for &input in &inputs {
            if let Some(next) = step(s, input) {
                s = next;
                total += 1;
            } else {
                total += 1;
            }
        }
    }
    let dt_us = t0.elapsed().as_micros().max(1);
    let per_sec = (total as f64) / (dt_us as f64) * 1_000_000.0;
    println!("  Evaluations: {}", total);
    println!("  Time: {} us", dt_us);
    println!("  Throughput: {:.0} states/sec", per_sec);
    if per_sec > 10_000_000.0 {
        println!("  Verdict: >10M TARGET MET");
    } else {
        println!("  Verdict: BELOW TARGET (build with RUSTFLAGS=\"-C target-cpu=native\")");
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      TAS ENGINE — HACKATHON VICTORY DASHBOARD              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // ─── Phase 2: Default Maze ────────────────────────────────────────
    env::select_default_maze();
    benchmark_maze("Phase 2 — Default Maze", env::idx(7, 6),
        PackedState::new(env::idx(1, 1), 0, 0, 0, 0));

    // ─── Phase 4: Hell Maze ─────────────────────────────────────────
    env::select_hell_maze();
    benchmark_maze("Phase 4 — Hell Maze (Stress)", env::idx(7, 3),
        PackedState::new(env::idx(1, 3), 0, 0, 0, 0));
    env::select_default_maze();

    // ─── Phase 4: Raw Throughput ─────────────────────────────────────
    raw_throughput_benchmark();

    // ─── Phase 3: Driver Jitter ─────────────────────────────────────
    println!("\n[Phase 3 — Driver Jitter at 1000 Hz]");
    let driver = driver::TASDriver::compile_macro(&[4, 2, 4, 2, 4, 2], 1000);
    let report = driver.run_live_injection(1000, |_input| {});
    println!("  Frames: {}", report.samples);
    println!("  Max jitter: {} ns", report.max_ns);
    println!("  Min jitter: {} ns", report.min_ns);
    println!("  Avg jitter: {:.2} ns", report.avg_ns);
    println!("  Verdict: {}", if report.avg_ns.abs() < 1000.0 {
        "SUB-MICROSECOND (WIN)"
    } else {
        ">1 us"
    });

    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Build tip: RUSTFLAGS=\"-C target-cpu=native\" cargo run --release");
    println!("══════════════════════════════════════════════════════════════");
}

// ─── Integration Tests ─────────────────────────────────────────────────────
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_roundtrip_serialization() {
        let s = PackedState::new(42, 0b1010, 0b0101, 0b1111, 12345);
        let raw = s.raw();
        let recovered = PackedState::from_raw(raw);
        assert_eq!(s, recovered);
    }

    #[test]
    fn test_short_valid_path() {
        let start = PackedState::new(env::idx(1, 1), 0, 0, 0, 0);
        let s1 = step(start, Input::Right).unwrap();
        let s2 = step(s1, Input::Right).unwrap();
        assert!(s2.has_key(0));
        assert_eq!(s2.get_ticks(), 2);
    }

    #[test]
    fn test_victory_cell_exists() {
        assert_eq!(terrain_at(env::idx(7, 6)), 18);
    }

    #[test]
    fn test_wall_is_impenetrable() {
        let start = PackedState::new(env::idx(1, 1), 0, 0, 0, 0);
        // All four directions from (1,1):
        // Up -> (0,1) wall
        assert!(step(start, Input::Up).is_none());
        // Down -> (2,1) empty (valid)
        assert!(step(start, Input::Down).is_some());
        // Left -> (1,0) wall
        assert!(step(start, Input::Left).is_none());
        // Right -> (1,2) empty (valid)
        assert!(step(start, Input::Right).is_some());
    }

    #[test]
    fn test_key0_opens_door0() {
        // Path: (1,1) -> Right -> (1,2) -> Right -> (1,3) [Key0]
        // Then Down -> (2,3) [Door0] requires Key0.
        let mut s = PackedState::new(env::idx(1, 1), 0, 0, 0, 0);
        s = step(s, Input::Right).unwrap();
        s = step(s, Input::Right).unwrap();
        assert!(s.has_key(0));
        s = step(s, Input::Down).unwrap();
        assert_eq!(s.get_cell(), env::idx(2, 3));
    }
}
