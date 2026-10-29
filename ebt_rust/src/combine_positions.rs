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

            // Find first non-NaN in this column
            let mut first_valid: Option<usize> = None;
            for r in 0..n_rows {
                let v = vals[col_base + r];
                if !v.is_nan() {
                    first_valid = Some(r);
                    break;
                }
            }
            let first_valid = match first_valid {
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
            if is_mean {
                count_starts[target_row_first * n_target_cols + dst_c] += 1.0;
            }

            // Fused ffill → diff → scatter
            let mut prev = 0.0f64;
            let first_val = vals[col_base + first_valid];

            if full_target {
                // Fast path: df covers the full target — direct indexed write
                result[first_valid * n_target_cols + dst_c] += first_val;
                prev = first_val;

                for r in (first_valid + 1)..n_rows {
                    let v = vals[col_base + r]; // sequential memory access
                    let current = if v.is_nan() { prev } else { v };
                    let trade = current - prev;
                    if trade != 0.0 {
                        result[r * n_target_cols + dst_c] += trade;
                    }
                    prev = current;
                }
            } else {
                // General path: use precomputed row_map for O(1) lookups
                let target_row = row_map[first_valid];
                result[target_row * n_target_cols + dst_c] += first_val;
                prev = first_val;

                for r in (first_valid + 1)..n_rows {
                    let v = vals[col_base + r]; // sequential memory access
                    let current = if v.is_nan() { prev } else { v };
                    let trade = current - prev;
                    if trade != 0.0 {
                        let target_row = row_map[r];
                        result[target_row * n_target_cols + dst_c] += trade;
                    }
                    prev = current;
                }
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

    // Mean: cumsum count_starts, divide result
    if is_mean {
        for c in 0..n_target_cols {
            let mut count_acc = 0.0f64;
            for r in 0..n_target {
                let idx = r * n_target_cols + c;
                count_acc += count_starts[idx];
                if count_acc > 0.0 {
                    result[idx] /= count_acc;
                } else {
                    result[idx] = f64::NAN;
                }
            }
        }
    }

    (result, first_valid_row, count_starts)
}

pub struct DfInput {
    pub timestamps: Vec<i64>,
    pub values: Vec<f64>,     // column-major (F-order), n_rows * n_cols
    pub n_rows: usize,
    pub n_cols: usize,
    pub col_indices: Vec<u32>, // maps src col → target col index
}
