/// Fused forward-fill → diff → scatter-add pipeline for combining multi-frequency
/// position DataFrames. Works on column-major (Fortran-order) flattened arrays,
/// matching pandas' native memory layout to avoid copies.
///
/// For each input df: reads values once per column, computes trades inline, and
/// scatter-adds directly into the result — no intermediate allocations.

/// Combine multiple position DataFrames via trades + cumsum.
///
/// # Arguments
/// * `target_ts` - sorted target timestamps (union of all inputs), i64 nanoseconds
/// * `dfs` - list of DfInput structs
///   - timestamps: sorted i64 nanosecond timestamps for this df
///   - values: column-major (F-order) f64 values (n_rows * n_cols), may contain NaN
///     Layout: [col0_row0, col0_row1, ..., col0_rowN, col1_row0, ...]
///   - n_rows, n_cols: shape
///   - col_indices: maps each source column to a target column index
/// * `n_target_cols` - total number of columns in result
/// * `is_mean` - if true, divide by count of active dfs per cell
/// * `max_ffill_ns` - a cell is forward-filled for at most this long after its last
///   non-NaN print. Past that (strictly later than last print + max_ffill) the df's
///   position is closed with a flatten trade on the first target row beyond the limit
///   and contributes 0 (and, for mean, leaves the active count) until it prints again.
///   None = forward-fill indefinitely.
///
/// # Returns
/// (result_flat, first_valid_row, count_starts_flat) where:
/// * result_flat: row-major f64 array (n_target * n_target_cols)
/// * first_valid_row: per-column first valid row index (n_target_cols)
/// * count_starts_flat: row-major f64 (n_target * n_target_cols), only used for mean
pub fn combine_positions_core(
    target_ts: &[i64],
    dfs: &[DfInput],
    n_target_cols: usize,
    is_mean: bool,
    max_ffill_ns: Option<i64>,
) -> (Vec<f64>, Vec<usize>, Vec<f64>) {
    let n_target = target_ts.len();
    let mut result = vec![0.0f64; n_target * n_target_cols];
    let mut first_valid_row = vec![n_target; n_target_cols];
    let mut count_starts = if is_mean {
        vec![0.0f64; n_target * n_target_cols]
    } else {
        Vec::new()
    };

    for df in dfs {
        let ts = &df.timestamps;
        let vals = &df.values;
        let n_rows = df.n_rows;
        let n_src_cols = df.n_cols;
        let col_indices = &df.col_indices;
        let full_target = n_rows == n_target;

        // For the general path, precompute the target row mapping once per df.
        // Uses a linear merge scan O(n_target + n_rows) instead of
        // binary search per row O(n_rows * log(n_target)).
        let row_map = if !full_target {
            let mut map = vec![0usize; n_rows];
            let mut ti = 0usize;
            for r in 0..n_rows {
                while ti < n_target && target_ts[ti] < ts[r] {
                    ti += 1;
                }
                map[r] = ti;
            }
            map
        } else {
            Vec::new()
        };

        // Process each source column with fused ffill → diff → scatter.
        // Column-major layout: column src_c starts at vals[src_c * n_rows],
        // with consecutive row values contiguous in memory (cache-friendly).
        for src_c in 0..n_src_cols {
            let dst_c = col_indices[src_c] as usize;
            let col_base = src_c * n_rows; // start of this column in F-order
            let col = &vals[col_base..col_base + n_rows];

            // Find first non-NaN in this column
            let first_valid = match col.iter().position(|v| !v.is_nan()) {
                Some(f) => f,
                None => continue, // all-NaN column, skip
            };

            // Track first valid row in target
            let target_row_first = if full_target {
                first_valid
            } else {
                row_map[first_valid]
            };
            if target_row_first < first_valid_row[dst_c] {
                first_valid_row[dst_c] = target_row_first;
            }

            let mut ctx = ColumnCtx {
                target_ts,
                n_target_cols,
                dst_c,
                is_mean,
                max_ffill_ns,
                result: &mut result,
                count_starts: &mut count_starts,
            };
            if full_target {
                // Fast path: df covers the full target — direct indexed write
                ctx.process_column(col, ts, first_valid, |r| r);
            } else {
                // General path: use precomputed row_map for O(1) lookups
                ctx.process_column(col, ts, first_valid, |r| row_map[r]);
            }
        }
    }

    // Cumsum per column to recover positions
    for c in 0..n_target_cols {
        let mut acc = 0.0f64;
        for r in 0..n_target {
            acc += result[r * n_target_cols + c];
            result[r * n_target_cols + c] = acc;
        }
    }

    // NaN before first contribution per column
    for c in 0..n_target_cols {
        let fv = first_valid_row[c];
        for r in 0..fv.min(n_target) {
            result[r * n_target_cols + c] = f64::NAN;
        }
        if fv >= n_target {
            for r in 0..n_target {
                result[r * n_target_cols + c] = f64::NAN;
            }
        }
    }

    // Mean: cumsum count_starts, divide result. A cell whose count is 0 after the column's
    // first contribution is one every df closed under max_ffill: it stays at the (zero)
    // sum rather than turning NaN.
    if is_mean {
        for c in 0..n_target_cols {
            let mut count_acc = 0.0f64;
            let fv = first_valid_row[c];
            for r in 0..n_target {
                let idx = r * n_target_cols + c;
                count_acc += count_starts[idx];
                if count_acc > 0.0 {
                    result[idx] /= count_acc;
                } else if r < fv {
                    result[idx] = f64::NAN;
                }
            }
        }
    }

    (result, first_valid_row, count_starts)
}

/// Per-column scatter state shared by the full-target and general paths.
struct ColumnCtx<'a> {
    target_ts: &'a [i64],
    n_target_cols: usize,
    dst_c: usize,
    is_mean: bool,
    max_ffill_ns: Option<i64>,
    result: &'a mut Vec<f64>,
    count_starts: &'a mut Vec<f64>,
}

impl<'a> ColumnCtx<'a> {
    #[inline]
    fn add_trade(&mut self, target_row: usize, trade: f64) {
        if trade != 0.0 {
            self.result[target_row * self.n_target_cols + self.dst_c] += trade;
        }
    }

    #[inline]
    fn add_count(&mut self, target_row: usize, delta: f64) {
        if self.is_mean {
            self.count_starts[target_row * self.n_target_cols + self.dst_c] += delta;
        }
    }

    /// First target row strictly later than `last_print_ts + max_ffill`, if any.
    #[inline]
    fn close_row(&self, last_print_ts: i64) -> Option<usize> {
        let cap = self.max_ffill_ns?;
        let limit = last_print_ts.saturating_add(cap);
        let row = self.target_ts.partition_point(|&t| t <= limit);
        if row < self.target_ts.len() { Some(row) } else { None }
    }

    /// Fused ffill → diff → scatter for one source column. `col` is the column's values,
    /// `ts` its timestamps, `first_valid` the first non-NaN row, `to_target` maps a source
    /// row to its target row.
    fn process_column<F: Fn(usize) -> usize>(&mut self, col: &[f64], ts: &[i64], first_valid: usize, to_target: F) {
        let n_rows = col.len();
        let first_row = to_target(first_valid);
        self.add_count(first_row, 1.0);
        let mut prev = col[first_valid];
        self.add_trade(first_row, prev);
        let mut last_print_ts = ts[first_valid];
        let mut closed = false; // flattened by max_ffill since the last print

        for r in (first_valid + 1)..n_rows {
            let v = col[r]; // sequential memory access
            let is_print = !v.is_nan();
            // The forward fill ran past max_ffill before this row: close on the first target
            // row beyond the limit (at or before this row's own target row).
            if let Some(cap) = self.max_ffill_ns {
                if !closed && ts[r].saturating_sub(last_print_ts) > cap {
                    if let Some(row) = self.close_row(last_print_ts) {
                        self.add_trade(row, -prev);
                        self.add_count(row, -1.0);
                        prev = 0.0;
                        closed = true;
                    }
                }
            }
            if !is_print {
                continue; // ffill: current == prev, no trade
            }
            let target_row = to_target(r);
            if closed {
                self.add_count(target_row, 1.0);
                closed = false;
            }
            self.add_trade(target_row, v - prev);
            prev = v;
            last_print_ts = ts[r];
        }

        // The df ends before the target does: close once the fill runs past max_ffill.
        if !closed {
            if let Some(row) = self.close_row(last_print_ts) {
                self.add_trade(row, -prev);
                self.add_count(row, -1.0);
            }
        }
    }
}

pub struct DfInput {
    pub timestamps: Vec<i64>,
    pub values: Vec<f64>,     // column-major (F-order), n_rows * n_cols
    pub n_rows: usize,
    pub n_cols: usize,
    pub col_indices: Vec<u32>, // maps src col → target col index
}
