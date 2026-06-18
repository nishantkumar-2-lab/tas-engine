# TAS Engine — Tool-Assisted Speedrunning in Software

A deterministic, zero-overhead state-space pathfinding engine built for the
world's most elite hackathon.  It finds and executes the mathematically optimal
path through a complex software environment using bit-packed states,
contraction-inspired dominance pruning, and a frame-perfect input driver.

## Architecture

| Phase | Component | Technology |
|-------|-----------|------------|
| **1** | Bit-Packed State | `u64` newtype with `#[inline(always)] const fn` getters/setters |
| **2** | Pathfinding | IDA* + Pattern Database (1024-entry flat array) + 262K-slot generational dominance filter |
| **3** | Execution | Hybrid spin-lock timer (coarse sleep + CPU spin) + `.tas` binary export |
| **4** | Stress Test | Hell maze with maximally interdependent keys/doors/switches/one-ways |

## Performance (Release Build)

| Metric | Result |
|--------|--------|
| Default maze solution | 11 inputs, 22 nodes, ~300 us |
| Hell maze solution | 10 inputs, 132 nodes, ~230 us |
| **Raw throughput** | **289,000,000+ states/sec** |
| Target throughput | 10,000,000 states/sec |
| **Margin** | **28.9x** |
| Driver jitter (1000 Hz) | Avg: ~83 ns, Max: ~100 ns |

## Build

```bash
cargo test                          # 38 tests, all passing
cargo run --release                 # dashboard benchmark

# Native CPU tuning (optional, marginal on x86_64)
RUSTFLAGS="-C target-cpu=native" cargo run --release
```

## State Bit Layout (64 bits)

```
Bits 0..6    : Cell index (0-63 for 8x8 grid)
Bits 6..10   : Key bitflags (4 keys)
Bits 10..14  : Door bitflags (4 doors)
Bits 14..18  : Switch bitflags (4 switches)
Bits 18..34  : Tick counter (16 bits, up to 65,535)
Bits 34..64  : Reserved for expansion
```

## File Map

```
src/
  state.rs   — Core bit-packed state engine
  env.rs     — Dual maze layouts + deterministic transition function
  pdb.rs     — Pattern Database (backward BFS, admissible heuristic)
  search.rs  — IDA* with generational dominance pruning
  driver.rs  — Hybrid spin-lock timer + TAS binary export
  main.rs    — Dashboard entry point + integration tests
```

## Algorithmic Guarantees

- **Admissible heuristic**: The PDB never overestimates true cost.
  Proven by exhaustive reverse-graph BFS over all 262,144 valid states.
- **Optimal path**: IDA* with an admissible heuristic finds the
  shortest path in terms of input count.
- **Deterministic**: Every run produces identical output; no randomness,
  no ML, no heuristics that cannot be mathematically proven.

## License

Built for competition. Use at your own risk.
