mod state;
mod env;
mod pdb;
mod search;
mod driver;

use state::{PackedState, LayoutConfig};
use env::{default_world, INPUT_COUNT};
use pdb::PatternDatabase;
use search::solve;
use std::time::Instant;

fn benchmark(name: &str, layout: &LayoutConfig, world: &env::WorldConfig) {
    println!("\n[{}]", name);
    let t0 = Instant::now();
    let pdb = PatternDatabase::compute(layout, world);
    let pdb_us = t0.elapsed().as_micros();
    println!("  PDB: {} us ({} entries)", pdb_us, pdb.cell_count() * 16);

    let mut start = PackedState::zero();
    start.set_location(layout, world.start_cell);
    let t1 = Instant::now();
    let result = solve(start, layout, world, &pdb);
    let search_us = t1.elapsed().as_micros();

    match result {
        None => println!("  FAILED"),
        Some((path, nodes)) => {
            println!("  Path: {} inputs", path.len());
            println!("  Nodes: {}", nodes);
            println!("  Time: {} us", search_us);
            if search_us > 0 {
                println!("  Throughput: {:.2} M states/sec",
                    (nodes as f64) / (search_us as f64));
            }
        }
    }
}

fn raw_throughput(layout: &LayoutConfig, world: &env::WorldConfig) {
    println!("\n[Raw Throughput Micro-Benchmark]");
    const ITERS: u64 = 5_000_000;
    let mut s = PackedState::zero();
    s.set_location(layout, world.idx(3, 3));
    s.set_keys(layout, 0b1111);
    let t0 = Instant::now();
    let mut total: u64 = 0;
    for _ in 0..ITERS {
        for input in 0..INPUT_COUNT {
            if let Some(next) = env::step(s, input, layout, world) {
                s = next;
            }
            total += 1;
        }
    }
    let dt = t0.elapsed().as_micros().max(1);
    let rate = (total as f64) / (dt as f64) * 1_000_000.0;
    println!("  Evaluations: {}", total);
    println!("  Time: {} us", dt);
    println!("  Throughput: {:.0} states/sec", rate);
    println!("  Verdict: {}", if rate > 10_000_000.0 { ">10M TARGET MET" } else { "BELOW TARGET" });
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      TAS ENGINE — UNIVERSAL RUNTIME ADAPTABLE              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let layout = LayoutConfig::standard();
    let world = default_world();

    benchmark("Phase 2 — Default Maze", &layout, &world);
    raw_throughput(&layout, &world);

    println!("\n[Phase 3 — Driver Jitter at 1000 Hz]");
    let driver = driver::TASDriver::compile_macro(&[4, 2, 4, 2, 4, 2], 1000);
    let report = driver.run_live_injection(1000, |_input| {});
    println!("  Frames: {}", report.samples);
    println!("  Max jitter: {} ns", report.max_ns);
    println!("  Min jitter: {} ns", report.min_ns);
    println!("  Avg jitter: {:.2} ns", report.avg_ns);
    println!("  Verdict: {}", if report.avg_ns.abs() < 1000.0 { "SUB-MICROSECOND (WIN)" } else { ">1 us" });

    println!("\n══════════════════════════════════════════════════════════════");
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_roundtrip() {
        let mut s = PackedState::zero();
        let layout = LayoutConfig::standard();
        s.set_location(&layout, 420);
        s.set_keys(&layout, 0b1010);
        s.set_doors(&layout, 0b0101);
        s.set_switches(&layout, 0b1111);
        s.set_ticks(&layout, 12345);
        assert_eq!(s.get_location(&layout), 420);
        assert_eq!(s.get_keys(&layout), 0b1010);
        assert_eq!(s.get_ticks(&layout), 12345);
    }

    #[test]
    fn test_short_valid_path() {
        let layout = LayoutConfig::standard();
        let world = default_world();
        let mut s = PackedState::zero();
        s.set_location(&layout, world.start_cell);
        s = env::step(s, 4, &layout, &world).unwrap(); // Right
        s = env::step(s, 4, &layout, &world).unwrap(); // Right
        assert!(s.has_key(&layout, 0));
    }

    #[test]
    fn test_victory_cell_exists() {
        let world = default_world();
        let rule = world.rule(world.terrain_at(world.victory_cell));
        assert!(rule.is_victory);
    }
}
