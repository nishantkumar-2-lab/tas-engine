mod state;
mod env;
mod pdb;
mod search;
mod driver;
mod spatial;
mod io_layer;

use state::{PackedState, LayoutConfig};
use env::{default_world, INPUT_COUNT};
use pdb::PatternDatabase;
use search::{solve, plan_sequence, RollingHorizonConfig};
use spatial::SpatialQuantizer;
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

fn demo_continuous_physics() {
    println!("\n[Archetype B — Continuous 2D Physics Space]");
    // Float space: 8.0 x 8.0 units, quantized to 8x8 grid (1.0 per cell).
    let quant = SpatialQuantizer::new_2d(0.0, 0.0, 1.0, 8, 8);
    let mut s = PackedState::zero();
    s.set_f32(0, 0.5);  // X at byte 0
    s.set_f32(4, 0.5);  // Y at byte 4

    let cell = quant.quantize(s.get_f32(0), s.get_f32(4), 0.0);
    println!("  Start f32: ({:.2}, {:.2}) -> Cell {}", s.get_f32(0), s.get_f32(4), cell);

    // Simulate 3 physics steps moving right.
    for _ in 0..3 {
        let x = s.get_f32(0) + 1.0;
        s.set_f32(0, x);
    }
    let end_cell = quant.quantize(s.get_f32(0), s.get_f32(4), 0.0);
    println!("  After 3x Right: ({:.2}, {:.2}) -> Cell {}", s.get_f32(0), s.get_f32(4), end_cell);
    println!("  Spatial quantizer cells: {}", quant.cell_count());
}

fn demo_rolling_horizon() {
    println!("\n[Archetype C — Non-Deterministic / Rolling-Horizon Replan]");
    let layout = LayoutConfig::standard();
    let world = default_world();
    let pdb = PatternDatabase::compute(&layout, &world);

    let mut state = PackedState::zero();
    state.set_location(&layout, world.start_cell);

    let rh = RollingHorizonConfig { lookahead_depth: 4, node_budget: 50_000 };

    // Simulate 3 replan cycles.
    for cycle in 0..3 {
        let t0 = Instant::now();
        let result = plan_sequence(state, &layout, &world, &pdb, &rh);
        let dt = t0.elapsed().as_micros();
        match result {
            None => { println!("  Cycle {}: No path", cycle); break; }
            Some((path, nodes)) => {
                println!("  Cycle {}: {} inputs, {} nodes, {} us", cycle, path.len(), nodes, dt);
                // Execute first input, then re-plan (simulating real-time).
                if !path.is_empty() {
                    if let Some(next) = env::step(state, path[0], &layout, &world) {
                        state = next;
                    }
                }
            }
        }
    }
    println!("  Final state cell: {}", state.get_location(&layout));
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   TAS ENGINE — MODULAR HYBRID SPEEDRUNNING FRAMEWORK       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let layout = LayoutConfig::standard();
    let world = default_world();

    benchmark("Archetype A — Discrete Grid", &layout, &world);
    raw_throughput(&layout, &world);
    demo_continuous_physics();
    demo_rolling_horizon();

    println!("\n[Phase 3 — Driver Jitter at 1000 Hz]");
    let driver = driver::TASDriver::compile_macro(&[4, 2, 4, 2, 4, 2], 1000);
    let report = driver.run_live_injection(1000, |_input| {});
    println!("  Frames: {}", report.samples);
    println!("  Max jitter: {} ns", report.max_ns);
    println!("  Min jitter: {} ns", report.min_ns);
    println!("  Avg jitter: {:.2} ns", report.avg_ns);
    println!("  Verdict: {}", if report.avg_ns.abs() < 1000.0 { "SUB-MICROSECOND (WIN)" } else { ">1 us" });

    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Discrete   | Continuous | Non-Deterministic | External Hook");
    println!("══════════════════════════════════════════════════════════════");
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
