/// VWAP execution simulator.
///
/// Simulates realistic execution where position changes are ramped over a
/// VWAP window weighted by volume. When signals reverse mid-execution,
/// outstanding orders naturally net, reducing realized turnover.

/// Precompute forward-looking volume sums using a sliding window — O(n).
///
/// fwd_vol[t] = sum of volume[j] where timestamps[j] in [timestamps[t], timestamps[t] + window_ns)
fn compute_fwd_vol(timestamps: &[i64], volume: &[f64], window_ns: i64) -> Vec<f64> {
    let n = timestamps.len();
    let mut fwd_vol = vec![0.0f64; n];
    let mut right: usize = 0;
    let mut sum = 0.0f64;

    for left in 0..n {
        while right < n && timestamps[right] < timestamps[left] + window_ns {
            sum += volume[right];
            right += 1;
        }
        fwd_vol[left] = sum;
        sum -= volume[left];
    }

    fwd_vol
}

/// Simulate VWAP execution for a single asset.
///
/// Time is divided into non-overlapping windows of `window_ns` duration.
/// Within each window, execution ramps toward the current target weighted
/// by volume. At each timestep:
///
///   remaining_delta = target[t] - actual[t-1]
///   frac = volume[t] / remaining_volume_in_window
///   actual[t] = actual[t-1] + remaining_delta * frac
///
/// When the target changes mid-window, the remaining execution adjusts
/// from the current actual position — this is where netting happens.
/// At the last bar of a window (remaining_vol = volume[t]), frac = 1.0
/// and the position snaps to the current target.
fn simulate_vwap_1d(
    timestamps: &[i64], // nanosecond timestamps
    target: &[f64],
    volume: &[f64],
    window_ns: i64,
) -> Vec<f64> {
    let n = timestamps.len();
    let mut actual = vec![0.0f64; n];

    let fwd_vol = compute_fwd_vol(timestamps, volume, window_ns);

    let mut prev = 0.0f64;
    let mut window_start_ts: i64 = i64::MIN;
    let mut remaining_vol = 0.0f64;

    for t in 0..n {
        let tgt = target[t];
        if tgt.is_nan() {
            // No target update: hold. The bar's volume still passes, so the open window
            // has that much less volume left in which to finish the ramp (otherwise the
            // last bar of the window would not snap to the target).
            actual[t] = prev;
            if window_start_ts != i64::MIN && timestamps[t] < window_start_ts + window_ns {
                remaining_vol = (remaining_vol - volume[t]).max(0.0);
            }
            continue;
        }

        // Start new window if first bar or current window has elapsed
        if window_start_ts == i64::MIN || timestamps[t] >= window_start_ts + window_ns {
            window_start_ts = timestamps[t];
            remaining_vol = fwd_vol[t];
        }

        let delta = tgt - prev;
        if remaining_vol > 0.0 {
            let frac = volume[t] / remaining_vol;
            actual[t] = prev + delta * frac;
            remaining_vol -= volume[t];
            if remaining_vol < 0.0 {
                remaining_vol = 0.0;
            }
        } else {
            actual[t] = tgt; // no volume → snap to target
        }
        prev = actual[t];
    }

    actual
}

/// Column-major matrix version: simulate VWAP execution for multiple assets.
///
/// Input arrays are flattened column-major: data[col * n_rows + row].
/// Returns flattened column-major result of same shape.
pub fn simulate_vwap_matrix(
    timestamps: &[i64],
    target_data: &[f64],
    volume_data: &[f64],
    n_rows: usize,
    n_cols: usize,
    window_ns: i64,
) -> Vec<f64> {
    let mut result = vec![0.0f64; n_rows * n_cols];

    for col in 0..n_cols {
        let offset = col * n_rows;
        let target_col = &target_data[offset..offset + n_rows];
        let volume_col = &volume_data[offset..offset + n_rows];
        let out = simulate_vwap_1d(timestamps, target_col, volume_col, window_ns);
        result[offset..offset + n_rows].copy_from_slice(&out);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ns(minutes: i64) -> i64 {
        minutes * 60_000_000_000
    }

    #[test]
    fn test_constant_target_uniform_volume() {
        // With constant target and uniform volume, reaches target in exactly W periods.
        let timestamps: Vec<i64> = (0..10).map(|i| ns(i)).collect();
        let target = vec![1.0; 10];
        let volume = vec![100.0; 10];
        let window_ns = ns(3);

        let actual = simulate_vwap_1d(&timestamps, &target, &volume, window_ns);

        let tol = 1e-10;
        assert!((actual[0] - 1.0 / 3.0).abs() < tol, "t=0: {}", actual[0]);
        assert!((actual[1] - 2.0 / 3.0).abs() < tol, "t=1: {}", actual[1]);
        assert!((actual[2] - 1.0).abs() < tol, "t=2: {}", actual[2]);
        // Stays at target after window completes
        assert!((actual[5] - 1.0).abs() < tol);
    }

    #[test]
    fn test_mid_execution_netting() {
        // Signal flips at t=1 (after 1/3 of buy executed) → netting effect
        let timestamps: Vec<i64> = (0..6).map(|i| ns(i)).collect();
        let target = vec![1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let volume = vec![100.0; 6];
        let window_ns = ns(3);

        let actual = simulate_vwap_1d(&timestamps, &target, &volume, window_ns);

        let tol = 1e-10;
        // t=0: buy 1/3 of 1.0
        assert!((actual[0] - 1.0 / 3.0).abs() < tol);
        // t=1: remaining_vol = 200 (2 bars left in window).
        //   delta = -1 - 1/3 = -4/3. frac = 100/200 = 1/2.
        //   trade = -4/3 * 1/2 = -2/3. actual = 1/3 - 2/3 = -1/3.
        assert!((actual[1] - (-1.0 / 3.0)).abs() < tol, "t=1: {}", actual[1]);
        // t=2: remaining_vol = 100. frac = 1. Snaps to target.
        assert!((actual[2] - (-1.0)).abs() < tol, "t=2: {}", actual[2]);

        // Actual turnover
        let mut turnover = actual[0].abs();
        for t in 1..6 {
            turnover += (actual[t] - actual[t - 1]).abs();
        }
        // Instant turnover: |1| + |-1 - 1| = 3.0
        assert!(turnover < 3.0, "turnover {} should be < 3.0", turnover);
        // Expected: |1/3| + |2/3| + |2/3| = 5/3 ≈ 1.67
        assert!((turnover - 5.0 / 3.0).abs() < tol);
    }

    #[test]
    fn test_nonuniform_volume() {
        // High volume first bar → most execution happens there
        let timestamps: Vec<i64> = (0..3).map(|i| ns(i)).collect();
        let target = vec![1.0; 3];
        let volume = vec![300.0, 50.0, 50.0];
        let window_ns = ns(3);

        let actual = simulate_vwap_1d(&timestamps, &target, &volume, window_ns);

        let tol = 1e-10;
        // t=0: 300/400 = 75%
        assert!((actual[0] - 0.75).abs() < tol);
        // t=2: reaches 1.0 (last bar in window, frac=1)
        assert!((actual[2] - 1.0).abs() < tol);
    }

    #[test]
    fn test_irregular_timestamps() {
        // Gap in timestamps — window is time-based
        let timestamps: Vec<i64> = vec![ns(0), ns(1), ns(2), ns(10), ns(11)];
        let target = vec![1.0; 5];
        let volume = vec![100.0; 5];
        let window_ns = ns(3);

        let actual = simulate_vwap_1d(&timestamps, &target, &volume, window_ns);

        let tol = 1e-10;
        // First window [0, 3min): bars 0,1,2
        assert!((actual[0] - 1.0 / 3.0).abs() < tol);
        assert!((actual[1] - 2.0 / 3.0).abs() < tol);
        assert!((actual[2] - 1.0).abs() < tol);
        // Bar at 10min starts new window (gap > 3min)
        // Already at target, so no change
        assert!((actual[3] - 1.0).abs() < tol);
        assert!((actual[4] - 1.0).abs() < tol);
    }

    #[test]
    fn test_window_boundary_new_order() {
        // Window elapses, then new target → starts new execution
        let timestamps: Vec<i64> = (0..6).map(|i| ns(i)).collect();
        // First window: buy 1.0. Second window (starts t=3): target changes to 0.0.
        let target = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let volume = vec![100.0; 6];
        let window_ns = ns(3);

        let actual = simulate_vwap_1d(&timestamps, &target, &volume, window_ns);

        let tol = 1e-10;
        // First window: ramp to 1.0
        assert!((actual[2] - 1.0).abs() < tol);
        // Second window: ramp from 1.0 to 0.0
        assert!((actual[3] - 2.0 / 3.0).abs() < tol, "t=3: {}", actual[3]);
        assert!((actual[4] - 1.0 / 3.0).abs() < tol, "t=4: {}", actual[4]);
        assert!((actual[5] - 0.0).abs() < tol, "t=5: {}", actual[5]);
    }

    #[test]
    fn test_nan_target_holds_position() {
        let timestamps: Vec<i64> = (0..4).map(|i| ns(i)).collect();
        let target = vec![1.0, f64::NAN, f64::NAN, 1.0];
        let volume = vec![100.0; 4];
        let window_ns = ns(3);

        let actual = simulate_vwap_1d(&timestamps, &target, &volume, window_ns);

        let tol = 1e-10;
        // t=0: partial execution
        assert!((actual[0] - 1.0 / 3.0).abs() < tol);
        // t=1,2: NaN target → hold
        assert!((actual[1] - 1.0 / 3.0).abs() < tol);
        assert!((actual[2] - 1.0 / 3.0).abs() < tol);
        // t=3: new window, continue toward 1.0
        // fwd_vol[3] = volume[3] = 100 (only 1 bar in window). frac = 1.
        assert!((actual[3] - 1.0).abs() < tol);
    }

    #[test]
    fn test_nan_target_mid_window_still_reaches_target() {
        // NaN in the middle of a window: hold on that bar, but the window still ends on
        // schedule and the last bar snaps to the target.
        let timestamps: Vec<i64> = (0..3).map(|i| ns(i)).collect();
        let target = vec![1.0, f64::NAN, 1.0];
        let volume = vec![100.0; 3];
        let actual = simulate_vwap_1d(&timestamps, &target, &volume, ns(3));
        let tol = 1e-10;
        assert!((actual[0] - 1.0 / 3.0).abs() < tol);
        assert!((actual[1] - 1.0 / 3.0).abs() < tol, "NaN bar holds: {}", actual[1]);
        assert!((actual[2] - 1.0).abs() < tol, "last bar of the window snaps to target: {}", actual[2]);
    }

    #[test]
    fn test_single_bar_window_is_passthrough() {
        // When window <= bar interval, actual should exactly equal target.
        // This is the identity property: instant execution = no ramp.
        let n = 20;
        let timestamps: Vec<i64> = (0..n).map(|i| ns(i * 5)).collect(); // 5-min bars
        let target: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.7).sin()) // oscillating signal
            .collect();
        let volume = vec![100.0; n as usize];
        let window_ns = ns(5); // window = bar interval

        let actual = simulate_vwap_1d(&timestamps, &target, &volume, window_ns);

        let tol = 1e-10;
        for t in 0..n as usize {
            assert!(
                (actual[t] - target[t]).abs() < tol,
                "t={}: actual={} != target={}",
                t, actual[t], target[t]
            );
        }
    }

    #[test]
    fn test_zero_volume_snaps() {
        let timestamps: Vec<i64> = (0..3).map(|i| ns(i)).collect();
        let target = vec![1.0; 3];
        let volume = vec![0.0, 0.0, 0.0];
        let window_ns = ns(3);

        let actual = simulate_vwap_1d(&timestamps, &target, &volume, window_ns);

        // No volume → snap to target immediately
        assert_eq!(actual[0], 1.0);
    }
}
