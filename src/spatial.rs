//! Continuous-Discrete Spatial Mapping
//!
//! Converts floating-point (f32) coordinates inside `PackedState`
//! into discrete voxel/cell indices for Pattern-Database heuristic lookups.
//! Zero-allocation, stack-only computation.

/// Uniform-grid quantizer for continuous 2D/3D spaces.
/// Maps (x, y, z) float coordinates into a flat cell index
/// via division by `cell_size` and clamping to grid bounds.
#[derive(Clone, Copy, Debug)]
pub struct SpatialQuantizer {
    pub origin_x: f32,
    pub origin_y: f32,
    pub origin_z: f32,
    pub cell_size: f32,
    pub grid_w: u16,
    pub grid_h: u16,
    pub grid_d: u16,
}

impl SpatialQuantizer {
    /// 2D quantizer (depth = 1).
    pub const fn new_2d(origin_x: f32, origin_y: f32, cell_size: f32, w: u16, h: u16) -> Self {
        SpatialQuantizer {
            origin_x, origin_y, origin_z: 0.0,
            cell_size, grid_w: w, grid_h: h, grid_d: 1,
        }
    }

    /// 3D quantizer.
    #[allow(dead_code)]
    pub const fn new_3d(
        origin_x: f32, origin_y: f32, origin_z: f32,
        cell_size: f32, w: u16, h: u16, d: u16,
    ) -> Self {
        SpatialQuantizer { origin_x, origin_y, origin_z, cell_size, grid_w: w, grid_h: h, grid_d: d }
    }

    /// Quantize (x, y, z) -> flat cell index.
    /// Clamps to grid bounds so every coordinate yields a valid cell.
    #[inline(always)]
    pub fn quantize(&self, x: f32, y: f32, z: f32) -> u16 {
        let ix = ((x - self.origin_x) / self.cell_size).floor() as i32;
        let iy = ((y - self.origin_y) / self.cell_size).floor() as i32;
        let iz = ((z - self.origin_z) / self.cell_size).floor() as i32;

        let cx = ix.clamp(0, (self.grid_w as i32) - 1) as u16;
        let cy = iy.clamp(0, (self.grid_h as i32) - 1) as u16;
        let cz = iz.clamp(0, (self.grid_d as i32) - 1) as u16;

        (cz as u16) * (self.grid_w * self.grid_h) as u16
            + (cy as u16) * (self.grid_w as u16)
            + cx
    }

    /// Reverse: flat cell index -> center coordinate (x, y, z).
    #[allow(dead_code)]
    #[inline(always)]
    pub fn dequantize(&self, cell: u16) -> (f32, f32, f32) {
        let wh = (self.grid_w * self.grid_h) as u16;
        let cz = cell / wh;
        let rem = cell % wh;
        let cy = rem / self.grid_w;
        let cx = rem % self.grid_w;
        let half = self.cell_size * 0.5;
        (
            self.origin_x + (cx as f32) * self.cell_size + half,
            self.origin_y + (cy as f32) * self.cell_size + half,
            self.origin_z + (cz as f32) * self.cell_size + half,
        )
    }

    /// Total number of discrete cells in the grid.
    #[allow(dead_code)]
    #[inline(always)]
    pub fn cell_count(&self) -> usize {
        (self.grid_w as usize) * (self.grid_h as usize) * (self.grid_d as usize)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_center() {
        let q = SpatialQuantizer::new_2d(0.0, 0.0, 1.0, 8, 8);
        assert_eq!(q.quantize(0.5, 0.5, 0.0), 0);
        assert_eq!(q.quantize(1.5, 0.5, 0.0), 1);
        assert_eq!(q.quantize(0.5, 1.5, 0.0), 8);
    }

    #[test]
    fn test_quantize_clamp() {
        let q = SpatialQuantizer::new_2d(0.0, 0.0, 1.0, 4, 4);
        // Way outside grid should clamp to max cell.
        assert_eq!(q.quantize(100.0, 100.0, 0.0), 15);
        assert_eq!(q.quantize(-100.0, -100.0, 0.0), 0);
    }

    #[test]
    fn test_dequantize_roundtrip() {
        let q = SpatialQuantizer::new_2d(0.0, 0.0, 2.0, 4, 4);
        let cell = q.quantize(3.0, 5.0, 0.0);
        let (x, y, z) = q.dequantize(cell);
        // Center of cell (1,2) with size 2 -> (3.0, 5.0, 0.0)
        assert!((x - 3.0).abs() < 0.001);
        assert!((y - 5.0).abs() < 0.001);
        assert_eq!(z, 1.0);
    }
}
