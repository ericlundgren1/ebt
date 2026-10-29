//! Ensemble outlier detection for trade data.
//!
//! Design notes:
//!
//! * **RollingStats**: O(1) mean/std via running sum/sum_sq; O(1) percentile
//!   access via a sorted `Vec<f64>` maintained with binary-search insert/remove.
//! * **AdaptiveLevelFilter**: tau is updated on *both* accept and reject paths
//!   (Robbins–Monro convergence).  Median of last 10 accepted prices
//!   computed inline with a fixed-size array sort.
//! * **OutlierDetector**: rolling stats updated with all rows; stats computed
//!   once per row, reused for z-score and IQR.
//! * Detectors are streaming, row-by-row processors (see `rows`); the public
//!   entry points run them over in-memory arrays.

use std::collections::VecDeque;

use crate::rows::{
    RowData, RowProcessor,
    QuoteRowData, QuoteRowProcessor,
    OhlcvRowData, OhlcvRowProcessor,
};

// ---------------------------------------------------------------------------
// Rolling Statistics — O(1) mean/std, O(1) percentile
// ---------------------------------------------------------------------------

/// Cached statistics snapshot (stack-allocated, no heap alloc).
struct Stats {
    mean: f64,
    std: f64,
    q25: f64,
    q75: f64,
    iqr: f64,
}

/// Rolling window of prices with incremental mean/std and a sorted buffer
/// for O(1) percentile look-up.
struct RollingStats {
    window_size: usize,
    buffer: VecDeque<f64>,
    sorted: Vec<f64>,
    sum: f64,
    sum_sq: f64,
}

impl RollingStats {
    fn new(window_size: usize) -> Self {
        Self {
            window_size,
            buffer: VecDeque::with_capacity(window_size),
            sorted: Vec::with_capacity(window_size),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Add a price to the rolling window.  O(log w) search + O(w) memcpy for
    /// the sorted buffer (w = window_size, typically 1000 — microseconds).
    fn update(&mut self, price: f64) {
        // Evict oldest if full
        if self.buffer.len() == self.window_size {
            let evicted = self.buffer.pop_front().unwrap();
            self.sum -= evicted;
            self.sum_sq -= evicted * evicted;
            // Remove from sorted vec
            let pos = self
                .sorted
                .binary_search_by(|a| a.partial_cmp(&evicted).unwrap_or(std::cmp::Ordering::Equal))
                .expect("evicted value must be in sorted vec");
            self.sorted.remove(pos);
        }
        // Add new
        self.buffer.push_back(price);
        self.sum += price;
        self.sum_sq += price * price;
        // Insert into sorted vec at correct position
        let pos = self
            .sorted
            .binary_search_by(|a| a.partial_cmp(&price).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(|x| x);
        self.sorted.insert(pos, price);
    }

    /// Clear all data from the rolling window (used after regime shifts).
    fn reset(&mut self) {
        self.buffer.clear();
        self.sorted.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }

    /// Returns `None` if fewer than 10 observations in the window.
    fn get_stats(&self) -> Option<Stats> {
        let n = self.buffer.len();
        if n < 10 {
            return None;
        }
        let nf = n as f64;
        let mean = self.sum / nf;
        let variance = (self.sum_sq / nf) - mean * mean;
        let std = if variance > 0.0 { variance.sqrt() } else { 0.0 };
        let q25 = percentile_linear(&self.sorted, 0.25);
        let q75 = percentile_linear(&self.sorted, 0.75);
        let iqr = q75 - q25;
        Some(Stats {
            mean,
            std,
            q25,
            q75,
            iqr,
        })
    }
}

/// Linear-interpolation percentile on an already-sorted slice (matches numpy default).
#[inline]
fn percentile_linear(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    debug_assert!(n > 0);
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    if hi >= n {
        return sorted[n - 1];
    }
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ---------------------------------------------------------------------------
// Adaptive Level Filter
// ---------------------------------------------------------------------------

struct AdaptiveLevelFilter {
    // Parameters
    tau_min: f64,
    tau_max: f64,
    learning_rate: f64,
    var_alpha: f64,
    accepted_max: usize,
    // Regime shift parameters
    n_switch: usize,
    cand_kappa: f64,
    max_cand_age_events: usize,
    max_cand_age_seconds: f64,
    cand_beta: f64,
    initial_tau: f64,
    // Gap widening
    gap_widen_threshold_sec: f64,
    // Scale floor: absolute (log units) and in learned ticks relative to the level
    min_scale: f64,
    min_scale_ticks: f64,
    tick_est: Option<f64>,
    // Candidate clustering tolerance floor (log units): a regime shift's new level is
    // unknown, so clustering keeps a wider tolerance than the acceptance band.
    cand_min_scale: f64,
    // Scale floor used until the tick size has been learned (log units)
    unknown_tick_scale: f64,
    // A regime switch needs its confirmations to span at least this long (seconds)
    min_switch_seconds: f64,
    // State
    accepted_prices: VecDeque<f64>,
    current_level: Option<f64>,
    current_time_ns: Option<i64>,
    variance_rate: f64,
    tau: f64,
    n_accepted: usize,
    // Candidate tracking state
    candidate_level: Option<f64>,
    candidate_n_confirmed: usize,
    candidate_start_ns: Option<i64>,
    candidate_age_events: usize,
    candidate_last_price: Option<f64>,
    candidate_last_confirm_ns: Option<i64>,
    min_cand_confirm_gap_ns: i64,
    last_step_regime_switch: bool,
}

impl AdaptiveLevelFilter {
    fn new(gap_widen_threshold_sec: f64) -> Self {
        Self {
            tau_min: 3.0,
            tau_max: 8.0,
            learning_rate: 0.01,
            var_alpha: 0.05,
            accepted_max: 500,
            // Regime shift parameters
            n_switch: 3,
            cand_kappa: 2.5,
            max_cand_age_events: 30,
            max_cand_age_seconds: 300.0,
            cand_beta: 0.35,
            initial_tau: 4.0,
            // Gap widening
            gap_widen_threshold_sec,
            min_scale: 0.0,
            min_scale_ticks: 3.0,
            tick_est: None,
            cand_min_scale: 0.005,
            unknown_tick_scale: 0.005,
            min_switch_seconds: 0.0,
            // State
            accepted_prices: VecDeque::with_capacity(500),
            current_level: None,
            current_time_ns: None,
            variance_rate: 1e-4,
            tau: 4.0,
            n_accepted: 0,
            // Candidate tracking state
            candidate_level: None,
            candidate_n_confirmed: 0,
            candidate_start_ns: None,
            candidate_age_events: 0,
            candidate_last_price: None,
            candidate_last_confirm_ns: None,
            min_cand_confirm_gap_ns: 1_000_000_000, // 1 second
            last_step_regime_switch: false,
        }
    }

    #[inline]
    fn update_variance(&mut self, price_change: f64, time_delta: f64) {
        let td = time_delta.max(0.001).min(300.0);
        let r2 = (price_change * price_change) / td;
        let r2 = r2.min(self.variance_rate * 10.0).min(1e-3);
        self.variance_rate = (1.0 - self.var_alpha) * self.variance_rate + self.var_alpha * r2;
        self.variance_rate = self.variance_rate.max(1e-8).min(1e-2);
    }

    #[inline]
    fn get_scale(&self, time_delta: f64) -> f64 {
        let td = time_delta.max(0.001).min(300.0);
        let td_sat = 60.0 * (1.0 - (-td / 60.0).exp());
        let scale = (self.variance_rate * td_sat + 1e-8).sqrt();
        // Floor: `min_scale` (absolute) and `min_scale_ticks` learned ticks relative to
        // the level, so the band can't collapse below a few ticks on quiet stretches
        // but is otherwise set by the learned variance, not a fixed percentage.
        // Until a tick has been learned (the series hasn't moved yet) the old fixed
        // floor applies, so a long run of identical prices can't shrink the band to
        // nothing and reject the first legitimate tick move.
        let tick_floor = match (self.tick_est, self.current_level) {
            (Some(t), Some(l)) if l > 0.0 => self.min_scale_ticks * t / l,
            _ => self.unknown_tick_scale,
        };
        scale.max(self.min_scale).max(tick_floor)
    }

    /// Push a price into the bounded accepted_prices deque; learns the tick size as
    /// the smallest non-zero move between consecutive accepted prices (causal).
    #[inline]
    fn push_accepted(&mut self, price: f64) {
        if let Some(&last) = self.accepted_prices.back() {
            let d = (price - last).abs();
            if d > 0.0 && d.is_finite() {
                self.tick_est = Some(self.tick_est.map_or(d, |t| t.min(d)));
            }
        }
        if self.accepted_prices.len() == self.accepted_max {
            self.accepted_prices.pop_front();
        }
        self.accepted_prices.push_back(price);
    }

    /// Invalidate the current candidate (reset all candidate state).
    #[inline]
    fn invalidate_candidate(&mut self) {
        self.candidate_level = None;
        self.candidate_n_confirmed = 0;
        self.candidate_start_ns = None;
        self.candidate_age_events = 0;
        self.candidate_last_price = None;
        self.candidate_last_confirm_ns = None;
    }

    /// Check if the current candidate has expired by age or time.
    #[inline]
    fn candidate_expired(&self, time_ns: i64) -> bool {
        if self.candidate_level.is_none() {
            return true;
        }
        if self.min_switch_seconds <= 0.0 && self.candidate_age_events >= self.max_cand_age_events {
            return true;
        }
        if let Some(start_ns) = self.candidate_start_ns {
            let elapsed_sec = (time_ns - start_ns) as f64 / 1e9;
            if elapsed_sec > self.max_cand_age_seconds.max(2.0 * self.min_switch_seconds) {
                return true;
            }
        }
        false
    }

    /// Process one observation.  Returns `true` if accepted.
    #[inline]
    fn step(&mut self, time_ns: i64, price: f64) -> bool {
        // Bootstrap
        if self.current_level.is_none() {
            self.current_level = Some(price);
            self.current_time_ns = Some(time_ns);
            self.push_accepted(price);
            self.n_accepted = 1;
            return true;
        }

        let level = self.current_level.unwrap();
        let prev_ns = self.current_time_ns.unwrap();
        let time_delta = ((time_ns - prev_ns) as f64 / 1e9).max(0.001);

        // Widen tau after extended gaps to avoid rejecting legitimate post-gap prices
        if time_delta > self.gap_widen_threshold_sec {
            let gap_factor = ((time_delta - self.gap_widen_threshold_sec) / (24.0 * 3600.0)).min(1.0);
            self.tau = self.tau + gap_factor * (self.tau_max - self.tau);
        }

        let scale = self.get_scale(time_delta);

        // Standardised deviation in log-space
        let price_change = (price / level).ln();
        let u = price_change / scale;
        let abs_u = u.abs();

        self.last_step_regime_switch = false;

        if abs_u <= self.tau {
            // --- ACCEPT ---
            // Invalidate any pending regime-shift candidate. An accepted
            // point proves the current level is still alive — rejected
            // outliers near the candidate are noise (e.g. stale quotes),
            // not evidence of a genuine regime shift. Real regime shifts
            // produce ONLY rejections with no interleaved accepts.
            self.invalidate_candidate();

            // Update level: median of last 10 accepted, or EMA
            if self.accepted_prices.len() >= 10 {
                let start = self.accepted_prices.len() - 10;
                let mut recent = [0.0_f64; 10];
                for (i, &p) in self.accepted_prices.iter().skip(start).enumerate() {
                    recent[i] = p;
                }
                recent.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                self.current_level = Some((recent[4] + recent[5]) / 2.0);
            } else {
                let alpha = 0.1;
                self.current_level = Some((1.0 - alpha) * level + alpha * price);
            }

            self.current_time_ns = Some(time_ns);
            self.push_accepted(price);
            self.n_accepted += 1;
            self.update_variance(price_change, time_delta);

            // Robbins–Monro tau update (accepted: indicator = 1.0)
            if self.n_accepted > 10 {
                self.tau += self.learning_rate * (0.999 - 1.0);
                self.tau = self.tau.max(self.tau_min).min(self.tau_max);
            }
            true
        } else {
            // --- REJECT ---
            // Also update tau on rejection (indicator = 0.0)
            if self.n_accepted > 10 {
                self.tau += self.learning_rate * 0.999;
                self.tau = self.tau.max(self.tau_min).min(self.tau_max);
            }

            // --- Regime shift candidate tracking ---
            let log_price = price.ln();

            // Stale quote detection — two checks:
            // 1. Same price as last confirmation (exact duplicate)
            // 2. Too close in time to last confirmation (rapid-fire stale burst)
            // Stale quotes from a frozen bid/ask interleave with live quotes
            // within the same millisecond. Real regime shifts have confirmations
            // spread over seconds.
            let is_stale_price = if let Some(last_p) = self.candidate_last_price {
                let max_val = price.abs().max(last_p.abs());
                max_val > 1e-15 && (price - last_p).abs() / max_val < 1e-10
            } else {
                false
            };
            let is_too_fast = if let Some(last_ns) = self.candidate_last_confirm_ns {
                (time_ns - last_ns) < self.min_cand_confirm_gap_ns
            } else {
                false
            };
            let is_stale = is_stale_price || is_too_fast;

            if self.candidate_expired(time_ns) {
                // No candidate or expired → start new candidate at this point
                self.candidate_level = Some(log_price);
                self.candidate_n_confirmed = 1;
                self.candidate_start_ns = Some(time_ns);
                self.candidate_age_events = 1;
                self.candidate_last_price = Some(price);
                self.candidate_last_confirm_ns = Some(time_ns);
            } else {
                // We have a live candidate — check if this point clusters near it
                let cand_log = self.candidate_level.unwrap();
                let cand_scale = scale.max(self.cand_min_scale);
                let u_c = (log_price - cand_log) / cand_scale;

                if u_c.abs() <= self.cand_kappa {
                    // Point clusters near candidate
                    self.candidate_age_events += 1;

                    // EMA update of candidate center regardless of stale status
                    let new_cand = (1.0 - self.cand_beta) * cand_log + self.cand_beta * log_price;
                    self.candidate_level = Some(new_cand);

                    if !is_stale {
                        // Fresh, temporally-separated confirmation
                        self.candidate_n_confirmed += 1;
                        self.candidate_last_price = Some(price);
                        self.candidate_last_confirm_ns = Some(time_ns);

                        // Check if we have enough confirmations, spanning long enough → regime switch
                        let cand_age_sec = self.candidate_start_ns
                            .map_or(0.0, |s| (time_ns - s) as f64 / 1e9);
                        if self.candidate_n_confirmed >= self.n_switch
                            && cand_age_sec >= self.min_switch_seconds
                        {
                            let new_level = self.candidate_level.unwrap().exp();
                            self.current_level = Some(new_level);
                            self.current_time_ns = Some(time_ns);
                            // After regime switch, be permissive — the candidate
                            // EMA may be imprecise. Tau tightens via Robbins-Monro.
                            self.tau = self.tau_max;
                            // Clear accepted_prices buffer so the median-of-10
                            // level update doesn't pull back to the old level.
                            self.accepted_prices.clear();
                            self.push_accepted(price);
                            self.n_accepted += 1;
                            self.invalidate_candidate();
                            self.last_step_regime_switch = true;
                            return true; // Accept via regime switch
                        }
                    }
                } else {
                    // Point is far from candidate too → reset candidate to this new point
                    self.candidate_level = Some(log_price);
                    self.candidate_n_confirmed = 1;
                    self.candidate_start_ns = Some(time_ns);
                    self.candidate_age_events = 1;
                    self.candidate_last_price = Some(price);
                    self.candidate_last_confirm_ns = Some(time_ns);
                }
            }

            false
        }
    }
}

// ---------------------------------------------------------------------------
// Outlier Detector Configuration
// ---------------------------------------------------------------------------

pub struct OutlierConfig {
    pub use_adaptive_level: bool,
    pub use_rolling_zscore: bool,
    pub use_iqr: bool,
    pub use_volume_check: bool,
    pub use_time_gap_check: bool,
    pub use_price_jump_check: bool,
    pub zscore_threshold: f64,
    pub iqr_multiplier: f64,
    pub min_votes_to_reject: usize,
    pub window_size: usize,
    /// When true, z-score and IQR count as a single "statistical" vote (max +1).
    /// Prevents correlated tests from double-counting on tight-spread instruments.
    pub merge_stat_votes: bool,
    /// After a time gap longer than this (seconds), widen the adaptive filter's
    /// tau toward tau_max to avoid rejecting legitimate post-gap prices.
    /// Default: 14400.0 (4 hours).
    pub gap_widen_threshold_sec: f64,
    /// During warmup (fewer than this many accepted points), increase the
    /// effective min_votes_to_reject by 1 to prevent aggressive false positives
    /// on sparse/young contracts.  Default: 50.
    pub warmup_threshold: usize,
    /// When true, regime-shift detection overrides the vote threshold
    /// (a regime switch forces acceptance regardless of other votes).
    /// Default: true.
    pub regime_override: bool,
    /// When true, only accepted prices enter the rolling statistics window,
    /// and the window is reset after a regime shift.
    /// When false, ALL prices update the window before voting.
    /// Default: true.
    pub accepted_only_window: bool,
    /// Floor of the adaptive level filter's scale, in log-price units. Default 0.0:
    /// the band is set by the learned variance and `min_scale_ticks`.
    pub min_scale: f64,
    /// Floor of the level filter's scale in learned ticks (smallest non-zero move
    /// between accepted prices) relative to the level. Default 3.
    pub min_scale_ticks: f64,
    /// Confirmations (≥1 s apart) needed to accept a new price level. Default 3.
    pub n_switch: usize,
    /// A regime switch additionally needs its confirmations to span at least this
    /// many seconds, so a burst of bad prints within a second can't become a regime.
    /// Default 0 (off).
    pub min_switch_seconds: f64,
    /// While fewer than `warmup_threshold` points are accepted the vote threshold is
    /// relaxed; a price further than this fraction from the last accepted price is
    /// rejected outright instead. Default 0.10.
    pub warmup_hard_limit: f64,
}

impl Default for OutlierConfig {
    fn default() -> Self {
        Self {
            use_adaptive_level: true,
            use_rolling_zscore: true,
            use_iqr: true,
            use_volume_check: true,
            use_time_gap_check: true,
            use_price_jump_check: true,
            zscore_threshold: 5.0,
            iqr_multiplier: 5.0,
            min_votes_to_reject: 2,
            window_size: 1000,
            merge_stat_votes: true,
            gap_widen_threshold_sec: 14400.0, // 4 hours
            warmup_threshold: 50,
            regime_override: true,
            accepted_only_window: true,
            min_scale: 0.0,
            min_scale_ticks: 3.0,
            n_switch: 3,
            min_switch_seconds: 0.0,
            warmup_hard_limit: 0.10,
        }
    }
}

// ---------------------------------------------------------------------------
// Debug Buffers — optional sidecar for diagnostic output
// ---------------------------------------------------------------------------

/// Pre-allocated column buffers for debug output.  One `Vec<f64>` per debug
/// column.  When debug is disabled these are never allocated.
pub struct DebugBuffers {
    pub level: Vec<f64>,
    pub scale: Vec<f64>,
    pub u: Vec<f64>,
    pub tau: Vec<f64>,
    pub variance_rate: Vec<f64>,
    pub zscore: Vec<f64>,
    pub iqr_lower: Vec<f64>,
    pub iqr_upper: Vec<f64>,
    pub price_change_pct: Vec<f64>,
    pub time_gap_seconds: Vec<f64>,
    pub volume_change: Vec<f64>,
    pub votes_against: Vec<f64>,
    pub cand_level: Vec<f64>,
    pub cand_n_confirmed: Vec<f64>,
    pub regime_switch: Vec<f64>,
}

impl DebugBuffers {
    /// Create pre-allocated debug buffers for `n` rows.
    fn with_capacity(n: usize) -> Self {
        Self {
            level: Vec::with_capacity(n),
            scale: Vec::with_capacity(n),
            u: Vec::with_capacity(n),
            tau: Vec::with_capacity(n),
            variance_rate: Vec::with_capacity(n),
            zscore: Vec::with_capacity(n),
            iqr_lower: Vec::with_capacity(n),
            iqr_upper: Vec::with_capacity(n),
            price_change_pct: Vec::with_capacity(n),
            time_gap_seconds: Vec::with_capacity(n),
            volume_change: Vec::with_capacity(n),
            votes_against: Vec::with_capacity(n),
            cand_level: Vec::with_capacity(n),
            cand_n_confirmed: Vec::with_capacity(n),
            regime_switch: Vec::with_capacity(n),
        }
    }

    /// Push NaN into every column (used for null rows).
    #[inline]
    fn push_nan(&mut self) {
        self.level.push(f64::NAN);
        self.scale.push(f64::NAN);
        self.u.push(f64::NAN);
        self.tau.push(f64::NAN);
        self.variance_rate.push(f64::NAN);
        self.zscore.push(f64::NAN);
        self.iqr_lower.push(f64::NAN);
        self.iqr_upper.push(f64::NAN);
        self.price_change_pct.push(f64::NAN);
        self.time_gap_seconds.push(f64::NAN);
        self.volume_change.push(f64::NAN);
        self.votes_against.push(f64::NAN);
        self.cand_level.push(f64::NAN);
        self.cand_n_confirmed.push(f64::NAN);
        self.regime_switch.push(f64::NAN);
    }
}

/// Result containing both the accepted mask and the debug buffers.
pub struct OutlierDebugResult {
    pub accepted: Vec<bool>,
    pub debug: DebugBuffers,
}

// ---------------------------------------------------------------------------
// Ensemble Outlier Detector
// ---------------------------------------------------------------------------

pub struct OutlierDetector {
    config: OutlierConfig,
    adaptive_filter: Option<AdaptiveLevelFilter>,
    rolling_stats: Option<RollingStats>,
    last_price: Option<f64>,
    last_time_ns: Option<i64>,
    last_volume: Option<f64>,
    debug_buffers: Option<DebugBuffers>,
    total_accepted: usize,
}

impl OutlierDetector {
    pub fn new(config: OutlierConfig) -> Self {
        let adaptive_filter = if config.use_adaptive_level {
            let mut f = AdaptiveLevelFilter::new(config.gap_widen_threshold_sec);
            f.min_scale = config.min_scale;
            f.min_scale_ticks = config.min_scale_ticks;
            f.n_switch = config.n_switch;
            f.min_switch_seconds = config.min_switch_seconds;
            Some(f)
        } else {
            None
        };
        let rolling_stats = if config.use_rolling_zscore || config.use_iqr {
            Some(RollingStats::new(config.window_size))
        } else {
            None
        };
        Self {
            config,
            adaptive_filter,
            rolling_stats,
            last_price: None,
            last_time_ns: None,
            last_volume: None,
            debug_buffers: None,
            total_accepted: 0,
        }
    }

    /// Enable debug output, pre-allocating buffers for `total_rows` rows.
    pub fn enable_debug(&mut self, total_rows: usize) {
        self.debug_buffers = Some(DebugBuffers::with_capacity(total_rows));
    }

    /// Move the debug buffers out of the detector (consuming them).
    pub fn take_debug_buffers(&mut self) -> Option<DebugBuffers> {
        self.debug_buffers.take()
    }

    /// Returns `true` if the observation is accepted (not an outlier).
    #[inline]
    fn detect(&mut self, time_ns: i64, price: f64, volume: f64) -> bool {
        let debug = self.debug_buffers.is_some();

        // Debug intermediates — initialised to NaN; overwritten by each method
        // that actually fires.  Only read when `debug` is true.
        let mut d_level = f64::NAN;
        let mut d_scale = f64::NAN;
        let mut d_u = f64::NAN;
        let mut d_tau = f64::NAN;
        let mut d_variance_rate = f64::NAN;
        let mut d_zscore = f64::NAN;
        let mut d_iqr_lower = f64::NAN;
        let mut d_iqr_upper = f64::NAN;
        let mut d_price_change_pct = f64::NAN;
        let mut d_time_gap_seconds = f64::NAN;
        let mut d_volume_change = f64::NAN;
        let mut d_cand_level = f64::NAN;
        let mut d_cand_n_confirmed = f64::NAN;
        let mut d_regime_switch = 0.0_f64;

        // Basic validation
        if !price.is_finite() || price <= 0.0 {
            if debug {
                if let Some(ref mut db) = self.debug_buffers {
                    db.level.push(d_level);
                    db.scale.push(d_scale);
                    db.u.push(d_u);
                    db.tau.push(d_tau);
                    db.variance_rate.push(d_variance_rate);
                    db.zscore.push(d_zscore);
                    db.iqr_lower.push(d_iqr_lower);
                    db.iqr_upper.push(d_iqr_upper);
                    db.price_change_pct.push(d_price_change_pct);
                    db.time_gap_seconds.push(d_time_gap_seconds);
                    db.volume_change.push(d_volume_change);
                    db.votes_against.push(f64::NAN);
                    db.cand_level.push(d_cand_level);
                    db.cand_n_confirmed.push(d_cand_n_confirmed);
                    db.regime_switch.push(d_regime_switch);
                }
            }
            return false;
        }

        let mut votes_against: usize = 0;
        let mut regime_switch_active = false;

        // Method 1: Adaptive level filter
        if self.config.use_adaptive_level {
            if let Some(ref mut filter) = self.adaptive_filter {
                // Capture state BEFORE step for debug
                if debug {
                    d_level = filter.current_level.unwrap_or(f64::NAN);
                    d_tau = filter.tau;
                    d_variance_rate = filter.variance_rate;
                    if let (Some(level), Some(prev_ns)) = (filter.current_level, filter.current_time_ns) {
                        let time_delta = ((time_ns - prev_ns) as f64 / 1e9).max(0.001);
                        let scale = filter.get_scale(time_delta);
                        let price_change = (price / level).ln();
                        d_scale = scale;
                        d_u = price_change / scale;
                    }
                }
                if !filter.step(time_ns, price) {
                    votes_against += 1;
                }
                regime_switch_active = filter.last_step_regime_switch;
                // Capture updated state after step
                if debug {
                    d_tau = filter.tau;
                    d_variance_rate = filter.variance_rate;
                    d_cand_level = filter.candidate_level.map(|l| l.exp()).unwrap_or(f64::NAN);
                    d_cand_n_confirmed = filter.candidate_n_confirmed as f64;
                    d_regime_switch = if filter.last_step_regime_switch { 1.0 } else { 0.0 };
                }
            }
        }

        // accepted_only_window == false: update rolling stats with ALL rows before voting
        if !self.config.accepted_only_window {
            if self.config.use_rolling_zscore || self.config.use_iqr {
                if let Some(ref mut rs) = self.rolling_stats {
                    rs.update(price);
                }
            }
        }

        // Compute stats ONCE (reused by methods 2 and 3)
        let cached_stats = if self.config.use_rolling_zscore || self.config.use_iqr {
            self.rolling_stats.as_ref().and_then(|s| s.get_stats())
        } else {
            None
        };

        // Methods 2 & 3: Rolling Z-score and IQR
        // When merge_stat_votes is true, these correlated statistical tests
        // contribute at most 1 combined vote instead of 2.
        {
            let mut zscore_fired = false;
            let mut iqr_fired = false;

            if self.config.use_rolling_zscore {
                if let Some(ref stats) = cached_stats {
                    if stats.std > 0.0 {
                        let zscore = ((price - stats.mean) / stats.std).abs();
                        if debug {
                            d_zscore = zscore;
                        }
                        if zscore > self.config.zscore_threshold {
                            zscore_fired = true;
                        }
                    }
                }
            }

            if self.config.use_iqr {
                if let Some(ref stats) = cached_stats {
                    let lower = stats.q25 - self.config.iqr_multiplier * stats.iqr;
                    let upper = stats.q75 + self.config.iqr_multiplier * stats.iqr;
                    if debug {
                        d_iqr_lower = lower;
                        d_iqr_upper = upper;
                    }
                    if price < lower || price > upper {
                        iqr_fired = true;
                    }
                }
            }

            if self.config.merge_stat_votes {
                // Merged: at most 1 vote from the statistical tests
                if zscore_fired || iqr_fired {
                    votes_against += 1;
                }
            } else {
                // Unmerged: each test votes independently
                if zscore_fired {
                    votes_against += 1;
                }
                if iqr_fired {
                    votes_against += 1;
                }
            }
        }

        // Method 4: Price jump check
        if self.config.use_price_jump_check {
            if let Some(last_p) = self.last_price {
                let pct = ((price - last_p) / last_p).abs();
                if debug {
                    d_price_change_pct = pct;
                }
                if pct > 0.10 {
                    if let Some(last_ns) = self.last_time_ns {
                        let gap_hours = (time_ns - last_ns) as f64 / 1e9 / 3600.0;
                        if gap_hours < 24.0 {
                            votes_against += 1;
                        }
                    }
                }
            }
        }

        // Method 5: Time gap check
        if self.config.use_time_gap_check {
            if let Some(last_ns) = self.last_time_ns {
                let gap_sec = (time_ns - last_ns) as f64 / 1e9;
                if debug {
                    d_time_gap_seconds = gap_sec;
                }
                if gap_sec < 0.0 {
                    votes_against += 1;
                } else if gap_sec > 604800.0 {  // 7 days
                    if let Some(last_p) = self.last_price {
                        if last_p > 0.0 && ((price - last_p) / last_p).abs() < 0.000001 {  // essentially exact match
                            votes_against += 1;
                        }
                    }
                }
            }
        }

        // Method 6: Volume consistency (supporting evidence only)
        if self.config.use_volume_check && volume.is_finite() && volume > 0.0 {
            if let Some(last_vol) = self.last_volume {
                if last_vol > 0.0 {
                    let vol_change = ((volume - last_vol) / last_vol.max(1e-10)).abs();
                    if debug {
                        d_volume_change = vol_change;
                    }
                    if vol_change > 1000.0 && votes_against > 0 {
                        votes_against += 1;
                    }
                }
            }
        }

        // Decision: regime-shift override forces acceptance regardless of votes
        let effective_threshold = if self.total_accepted < self.config.warmup_threshold {
            self.config.min_votes_to_reject + 1
        } else {
            self.config.min_votes_to_reject
        };
        // During warm-up the vote threshold is relaxed, so a gross deviation from the
        // last accepted price is rejected outright instead of needing three votes.
        let warmup_hard_reject = self.total_accepted < self.config.warmup_threshold
            && self.last_price.map_or(false, |lp| ((price - lp) / lp).abs() > self.config.warmup_hard_limit);
        let is_accepted = !warmup_hard_reject
            && (if self.config.regime_override {
                regime_switch_active || votes_against < effective_threshold
            } else {
                votes_against < effective_threshold
            });

        if is_accepted {
            self.last_price = Some(price);
            self.last_time_ns = Some(time_ns);
            if volume.is_finite() && volume > 0.0 {
                self.last_volume = Some(volume);
            }
            self.total_accepted += 1;

            // Only accepted prices enter the rolling window (prevents outlier contamination)
            if self.config.accepted_only_window {
                if self.config.use_rolling_zscore || self.config.use_iqr {
                    if let Some(ref mut rs) = self.rolling_stats {
                        rs.update(price);
                    }
                }
            }
        }

        // After a regime switch, reset the rolling window so z-score/IQR
        // re-calibrate on new-regime data instead of cascading false positives
        if regime_switch_active && self.config.accepted_only_window {
            if let Some(ref mut rs) = self.rolling_stats {
                rs.reset();
            }
        }

        // Push debug values (single branch per row, perfectly predicted)
        if debug {
            if let Some(ref mut db) = self.debug_buffers {
                db.level.push(d_level);
                db.scale.push(d_scale);
                db.u.push(d_u);
                db.tau.push(d_tau);
                db.variance_rate.push(d_variance_rate);
                db.zscore.push(d_zscore);
                db.iqr_lower.push(d_iqr_lower);
                db.iqr_upper.push(d_iqr_upper);
                db.price_change_pct.push(d_price_change_pct);
                db.time_gap_seconds.push(d_time_gap_seconds);
                db.volume_change.push(d_volume_change);
                db.votes_against.push(votes_against as f64);
                db.cand_level.push(d_cand_level);
                db.cand_n_confirmed.push(d_cand_n_confirmed);
                db.regime_switch.push(d_regime_switch);
            }
        }

        is_accepted
    }
}

// ---------------------------------------------------------------------------
// RowProcessor implementation for OutlierDetector
// ---------------------------------------------------------------------------

impl RowProcessor for OutlierDetector {
    type Output = bool;

    #[inline(always)]
    fn process_row(&mut self, row: &RowData) -> bool {
        self.detect(row.ts, row.price, row.volume)
    }

    fn null_value() -> bool {
        false
    }

    #[inline]
    fn on_null(&mut self) {
        if let Some(ref mut db) = self.debug_buffers {
            db.push_nan();
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run ensemble outlier detection over trade arrays.
///
/// Rows where `valid[i]` is false are treated as missing: they are marked not
/// accepted and don't update the detector state.
pub fn detect_outliers_arrays(
    time_ns: &[i64],
    price: &[f64],
    volume: &[f64],
    valid: &[bool],
    config: OutlierConfig,
) -> Vec<bool> {
    let mut detector = OutlierDetector::new(config);
    run_trades(&mut detector, time_ns, price, volume, valid)
}

/// Same as `detect_outliers_arrays` but also returns the debug columns.
pub fn detect_outliers_arrays_debug(
    time_ns: &[i64],
    price: &[f64],
    volume: &[f64],
    valid: &[bool],
    config: OutlierConfig,
) -> OutlierDebugResult {
    let mut detector = OutlierDetector::new(config);
    detector.enable_debug(time_ns.len());
    let accepted = run_trades(&mut detector, time_ns, price, volume, valid);
    let debug = detector
        .take_debug_buffers()
        .expect("debug buffers were enabled");
    OutlierDebugResult { accepted, debug }
}

fn run_trades(
    detector: &mut OutlierDetector,
    time_ns: &[i64],
    price: &[f64],
    volume: &[f64],
    valid: &[bool],
) -> Vec<bool> {
    let mut out = Vec::with_capacity(time_ns.len());
    for i in 0..time_ns.len() {
        if valid[i] {
            out.push(detector.process_row(&RowData {
                ts: time_ns[i],
                price: price[i],
                volume: volume[i],
            }));
        } else {
            out.push(OutlierDetector::null_value());
            detector.on_null();
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Spread Filter — upper-bound-only sanity check for bid-ask spreads
// ---------------------------------------------------------------------------

struct SpreadFilter {
    // Online quantile (Robbins-Monro)
    quant_q: f64,
    quant_tau: f64,
    quant_lr0: f64,
    quant_decay: f64,
    quant_n: u64,
    quant_tau_min: f64,
    quant_tau_max: f64,
    // Robust variance rate EWMA
    variance_rate: f64,
    var_alpha: f64,
    var_cap_mult: f64,
    var_cap_absolute: f64,
    // Level tracking
    spread_level: Option<f64>,
    last_time_ns: Option<i64>,
    // Scale parameters
    dt_saturation_sec: f64,
    scale_max: f64,
    /// Absolute floor of the scale (relative-spread units). Default 0.0.
    min_scale: f64,
    /// Floor set by the caller from the learned tick size (ticks / mid).
    tick_floor: f64,
    // Self-recovery: consecutive rejection counter
    consecutive_rejections: usize,
    max_consecutive_before_reset: usize,
}

impl SpreadFilter {
    fn new() -> Self {
        Self {
            quant_q: 0.999,
            quant_tau: 20.0,
            quant_lr0: 0.02,
            quant_decay: 0.7,
            quant_n: 0,
            quant_tau_min: 3.0,
            quant_tau_max: 20.0,
            variance_rate: 1e-6,
            var_alpha: 0.02,
            var_cap_mult: 25.0,
            var_cap_absolute: 1e-2,
            spread_level: None,
            last_time_ns: None,
            dt_saturation_sec: 60.0,
            scale_max: 0.02,
            min_scale: 0.0,
            tick_floor: 0.0,
            consecutive_rejections: 0,
            max_consecutive_before_reset: 5,
        }
    }

    /// Reset the spread filter to a new level (e.g. after a mid-price regime
    /// switch that invalidates the old relative-spread baseline).
    #[inline]
    fn reset_to(&mut self, spread: f64, time_ns: i64) {
        self.spread_level = Some(spread);
        self.last_time_ns = Some(time_ns);
        // Keep quant_tau and quant_n — the quantile estimate is still useful
        // Reset variance rate to a generous initial value so the scale is wide
        self.variance_rate = 1e-3;
        self.consecutive_rejections = 0;
    }

    #[inline]
    fn get_scale(&self, dt_sec: f64) -> f64 {
        let dt_saturated = self.dt_saturation_sec
            * (1.0 - (-dt_sec / self.dt_saturation_sec).exp());
        let scale_from_dt = (self.variance_rate * dt_saturated + 1e-12).sqrt();
        // Floor from the learned tick size (and an optional absolute floor), not a
        // fixed fraction of the mid: the band can't collapse below a few ticks but is
        // otherwise set by the learned spread variance.
        scale_from_dt.max(self.min_scale).max(self.tick_floor).min(self.scale_max)
    }

    /// Update the online quantile (Robbins-Monro) with observed |u|.
    #[inline]
    fn update_quantile(&mut self, abs_u: f64) {
        self.quant_n += 1;
        let lr = self.quant_lr0 / (self.quant_n as f64).powf(self.quant_decay);
        let indicator = if abs_u <= self.quant_tau { 1.0 } else { 0.0 };
        self.quant_tau += lr * (self.quant_q - indicator);
        self.quant_tau = self.quant_tau.max(self.quant_tau_min).min(self.quant_tau_max);
    }

    /// Update variance rate EWMA with capped r2.
    #[inline]
    fn update_variance(&mut self, dx: f64, dt_sec: f64) {
        let dt_capped = dt_sec.max(0.01).min(1.0);
        let r2 = (dx * dx) / dt_capped;
        let cap_relative = (self.variance_rate * self.var_cap_mult).max(1e-12 * self.var_cap_mult);
        let r2 = r2.min(cap_relative).min(self.var_cap_absolute);
        self.variance_rate = (1.0 - self.var_alpha) * self.variance_rate + self.var_alpha * r2;
        self.variance_rate = self.variance_rate.max(1e-12).min(1e2);
    }

    /// Process one spread observation. Returns (accepted, SpreadDebugInfo).
    ///
    /// Upper-bound only: small spreads (u <= tau) are always OK.
    #[inline]
    fn step(&mut self, time_ns: i64, spread: f64) -> (bool, SpreadDebugInfo) {
        // Bootstrap
        if self.spread_level.is_none() {
            self.spread_level = Some(spread);
            self.last_time_ns = Some(time_ns);
            let scale = self.get_scale(1e-3);
            let upper_bound = (spread + self.quant_tau * scale).max(1e-6);
            return (true, SpreadDebugInfo {
                spread_level: spread,
                spread_scale: scale,
                spread_u: 0.0,
                spread_tau: self.quant_tau,
                spread_variance_rate: self.variance_rate,
                spread_upper_bound: upper_bound,
            });
        }

        let level = self.spread_level.unwrap();
        let prev_ns = self.last_time_ns.unwrap();
        let dt_sec = ((time_ns - prev_ns) as f64 / 1e9).max(1e-3);
        let scale = self.get_scale(dt_sec);
        let u = (spread - level) / scale;
        let upper_bound = (level + self.quant_tau * scale).max(1e-6);

        let debug = SpreadDebugInfo {
            spread_level: level,
            spread_scale: scale,
            spread_u: u,
            spread_tau: self.quant_tau,
            spread_variance_rate: self.variance_rate,
            spread_upper_bound: upper_bound,
        };

        // Upper-bound only: accept if u <= tau (small spreads always OK)
        if u <= self.quant_tau {
            let dx = spread - level;
            // Cap dx: allow level to track within 2x scale or 5% of level
            let level_prop_cap = level.abs() * 0.05;
            let abs_cap = level_prop_cap.max(0.005);
            let dx_capped = dx.max(-2.0 * scale).min(2.0 * scale);
            let dx_capped = dx_capped.max(-abs_cap).min(abs_cap);
            // Update level (ensure positive)
            self.spread_level = Some((level + dx_capped).max(1e-6));
            self.last_time_ns = Some(time_ns);
            self.update_variance(dx_capped, dt_sec);
            self.update_quantile(u.abs());
            self.consecutive_rejections = 0;
            (true, debug)
        } else {
            self.consecutive_rejections += 1;
            // Self-recovery: if too many consecutive rejections, the level
            // is stuck — force-reset to current observation.
            if self.consecutive_rejections >= self.max_consecutive_before_reset {
                self.spread_level = Some(spread);
                self.last_time_ns = Some(time_ns);
                self.variance_rate = 1e-3;
                self.consecutive_rejections = 0;
                // Re-compute debug info for the now-accepted observation
                let new_debug = SpreadDebugInfo {
                    spread_level: spread,
                    spread_scale: scale,
                    spread_u: 0.0,
                    spread_tau: self.quant_tau,
                    spread_variance_rate: self.variance_rate,
                    spread_upper_bound: (spread + self.quant_tau * scale).max(1e-6),
                };
                (true, new_debug)
            } else {
                // Update quantile even on rejection so tau can widen
                self.update_quantile(u.abs());
                // Pull level toward rejected observation so it can
                // track legitimate spread-level changes.
                let dx = spread - level;
                let slow_step = dx * 0.5; // 50% pull per rejection
                self.spread_level = Some((level + slow_step).max(1e-6));
                // Also update variance with the observed deviation
                let td = dt_sec.max(0.01).min(1.0);
                let r2 = (dx * dx) / td;
                let r2 = r2.min(self.variance_rate * 3.0).min(self.var_cap_absolute);
                let alpha_rej = self.var_alpha * 0.5;
                self.variance_rate = (1.0 - alpha_rej) * self.variance_rate + alpha_rej * r2;
                self.variance_rate = self.variance_rate.max(1e-12).min(1e2);
                (false, debug)
            }
        }
    }
}

/// Debug info from a single SpreadFilter step.
struct SpreadDebugInfo {
    spread_level: f64,
    spread_scale: f64,
    spread_u: f64,
    spread_tau: f64,
    spread_variance_rate: f64,
    spread_upper_bound: f64,
}

// ---------------------------------------------------------------------------
// Quote Outlier Detection — two-gate cascade (mid-price + spread)
// ---------------------------------------------------------------------------

pub struct QuoteOutlierConfig {
    pub gap_widen_threshold_sec: f64,
    pub mid_tau_min: f64,
    pub mid_tau_max: f64,
    pub mid_n_switch: usize,
    pub spread_max_consecutive_reject: usize,
    pub max_spread: f64,
    pub warmup_count: usize,
    /// Accept crossed quotes (bid > ask) by swapping the sides, rejecting only
    /// crosses above 5%. Default false: a crossed quote is invalid, like high < low.
    pub allow_crossed: bool,
    /// Mid-price level filter scale floors; see `OutlierConfig::min_scale{,_ticks}`.
    pub mid_min_scale: f64,
    pub mid_min_scale_ticks: f64,
    /// See `OutlierConfig::min_switch_seconds`.
    pub mid_min_switch_seconds: f64,
    /// Spread filter scale floors (relative-spread units / learned ticks over mid).
    pub spread_min_scale: f64,
    pub spread_min_scale_ticks: f64,
    /// During warm-up a mid further than this fraction from the accepted level is
    /// rejected outright. Default 0.10.
    pub warmup_hard_limit: f64,
}

impl Default for QuoteOutlierConfig {
    fn default() -> Self {
        Self {
            gap_widen_threshold_sec: 14400.0,
            mid_tau_min: 5.0,
            mid_tau_max: 8.0,
            mid_n_switch: 2,
            spread_max_consecutive_reject: 5,
            max_spread: 0.50,
            warmup_count: 10,
            allow_crossed: false,
            mid_min_scale: 0.0,
            mid_min_scale_ticks: 3.0,
            mid_min_switch_seconds: 0.0,
            spread_min_scale: 0.0,
            spread_min_scale_ticks: 3.0,
            warmup_hard_limit: 0.10,
        }
    }
}

/// Pre-allocated column buffers for quote debug output.
pub struct QuoteDebugBuffers {
    // Mid-price columns
    pub mid_level: Vec<f64>,
    pub mid_scale: Vec<f64>,
    pub mid_u: Vec<f64>,
    pub mid_tau: Vec<f64>,
    pub mid_variance_rate: Vec<f64>,
    pub mid_cand_level: Vec<f64>,
    pub mid_cand_n_confirmed: Vec<f64>,
    pub mid_regime_switch: Vec<f64>,
    // Spread columns
    pub spread_level: Vec<f64>,
    pub spread_scale: Vec<f64>,
    pub spread_u: Vec<f64>,
    pub spread_tau: Vec<f64>,
    pub spread_variance_rate: Vec<f64>,
    pub spread_upper_bound: Vec<f64>,
    // Meta
    pub gate: Vec<f64>,
}

impl QuoteDebugBuffers {
    fn with_capacity(n: usize) -> Self {
        Self {
            mid_level: Vec::with_capacity(n),
            mid_scale: Vec::with_capacity(n),
            mid_u: Vec::with_capacity(n),
            mid_tau: Vec::with_capacity(n),
            mid_variance_rate: Vec::with_capacity(n),
            mid_cand_level: Vec::with_capacity(n),
            mid_cand_n_confirmed: Vec::with_capacity(n),
            mid_regime_switch: Vec::with_capacity(n),
            spread_level: Vec::with_capacity(n),
            spread_scale: Vec::with_capacity(n),
            spread_u: Vec::with_capacity(n),
            spread_tau: Vec::with_capacity(n),
            spread_variance_rate: Vec::with_capacity(n),
            spread_upper_bound: Vec::with_capacity(n),
            gate: Vec::with_capacity(n),
        }
    }

    #[inline]
    fn push_nan(&mut self) {
        self.mid_level.push(f64::NAN);
        self.mid_scale.push(f64::NAN);
        self.mid_u.push(f64::NAN);
        self.mid_tau.push(f64::NAN);
        self.mid_variance_rate.push(f64::NAN);
        self.mid_cand_level.push(f64::NAN);
        self.mid_cand_n_confirmed.push(f64::NAN);
        self.mid_regime_switch.push(f64::NAN);
        self.spread_level.push(f64::NAN);
        self.spread_scale.push(f64::NAN);
        self.spread_u.push(f64::NAN);
        self.spread_tau.push(f64::NAN);
        self.spread_variance_rate.push(f64::NAN);
        self.spread_upper_bound.push(f64::NAN);
        self.gate.push(0.0);
    }
}

pub struct QuoteDebugResult {
    pub accepted: Vec<bool>,
    pub debug: QuoteDebugBuffers,
}

pub struct QuoteOutlierDetector {
    mid_filter: AdaptiveLevelFilter,
    spread_filter: SpreadFilter,
    max_spread: f64,
    warmup_count: usize,
    n_accepted: usize,
    allow_crossed: bool,
    warmup_hard_limit: f64,
    spread_min_scale_ticks: f64,
    // Learned tick size: smallest non-zero move of an accepted bid or ask (causal)
    tick_est: Option<f64>,
    last_bid: Option<f64>,
    last_ask: Option<f64>,
    debug_buffers: Option<QuoteDebugBuffers>,
}

impl QuoteOutlierDetector {
    pub fn new(config: QuoteOutlierConfig) -> Self {
        let mut mid_filter = AdaptiveLevelFilter::new(config.gap_widen_threshold_sec);
        mid_filter.tau_min = config.mid_tau_min;
        mid_filter.tau_max = config.mid_tau_max;
        mid_filter.tau = config.mid_tau_min.max(mid_filter.initial_tau);
        mid_filter.initial_tau = config.mid_tau_min.max(mid_filter.initial_tau);
        mid_filter.n_switch = config.mid_n_switch;
        mid_filter.min_scale = config.mid_min_scale;
        mid_filter.min_scale_ticks = config.mid_min_scale_ticks;
        mid_filter.min_switch_seconds = config.mid_min_switch_seconds;
        let mut spread_filter = SpreadFilter::new();
        spread_filter.max_consecutive_before_reset = config.spread_max_consecutive_reject;
        spread_filter.min_scale = config.spread_min_scale;
        Self {
            mid_filter,
            spread_filter,
            max_spread: config.max_spread,
            warmup_count: config.warmup_count,
            n_accepted: 0,
            allow_crossed: config.allow_crossed,
            warmup_hard_limit: config.warmup_hard_limit,
            spread_min_scale_ticks: config.spread_min_scale_ticks,
            tick_est: None,
            last_bid: None,
            last_ask: None,
            debug_buffers: None,
        }
    }

    pub fn enable_debug(&mut self, total_rows: usize) {
        self.debug_buffers = Some(QuoteDebugBuffers::with_capacity(total_rows));
    }

    pub fn take_debug_buffers(&mut self) -> Option<QuoteDebugBuffers> {
        self.debug_buffers.take()
    }

    /// Smallest non-zero move of an accepted bid or ask so far (causal tick estimate).
    #[inline]
    fn learn_tick(&mut self, bid: f64, ask: f64) {
        for (cur, last) in [(bid, self.last_bid), (ask, self.last_ask)] {
            if let Some(l) = last {
                let d = (cur - l).abs();
                if d > 0.0 && d.is_finite() {
                    self.tick_est = Some(self.tick_est.map_or(d, |t| t.min(d)));
                }
            }
        }
        self.last_bid = Some(bid);
        self.last_ask = Some(ask);
    }

    /// Returns `true` if the quote is accepted (not an outlier).
    #[inline]
    fn detect_quote(&mut self, time_ns: i64, bid: f64, ask: f64) -> bool {
        let debug = self.debug_buffers.is_some();

        // Hard validation
        if !bid.is_finite() || !ask.is_finite() || bid <= 0.0 || ask <= 0.0 {
            if debug {
                if let Some(ref mut db) = self.debug_buffers {
                    db.push_nan();
                }
            }
            return false;
        }

        // A crossed quote (bid > ask) is a stale or bad side: invalid, like high < low
        // for a bar. With `allow_crossed` the sides are swapped and only a cross above
        // 5% is rejected (micro-crosses from latency are then tolerated).
        if bid > ask && (!self.allow_crossed || bid > ask * 1.05) {
            if debug {
                if let Some(ref mut db) = self.debug_buffers {
                    db.push_nan();
                }
            }
            return false;
        }
        let (bid_c, ask_c) = if bid > ask { (ask, bid) } else { (bid, ask) };
        let mid = (bid_c + ask_c) / 2.0;
        let spread = (ask_c - bid_c) / mid;

        // During warm-up the gates are lenient, so a mid far from the accepted level
        // is rejected outright instead.
        if self.n_accepted < self.warmup_count {
            if let Some(level) = self.mid_filter.current_level {
                if level > 0.0 && ((mid / level).ln()).abs() > self.warmup_hard_limit {
                    if debug {
                        if let Some(ref mut db) = self.debug_buffers {
                            db.push_nan();
                        }
                    }
                    return false;
                }
            }
        }
        // Until a tick has been learned the spread filter keeps its old fixed floor.
        self.spread_filter.tick_floor = match self.tick_est {
            Some(t) if mid > 0.0 => self.spread_min_scale_ticks * t / mid,
            _ => 0.002,
        };

        // Hard max spread check: reject clearly bad data
        if spread > self.max_spread {
            if debug {
                if let Some(ref mut db) = self.debug_buffers {
                    db.push_nan();
                }
            }
            return false;
        }

        // One-sided staleness check: if one side of the quote is far from
        // the accepted mid level while the other is close, the far side is
        // frozen/stale. Wide-spread instruments have both sides
        // roughly equally far from mid, so they pass this check.
        if let Some(level) = self.mid_filter.current_level {
            let bid_dev = ((bid_c / level).ln()).abs();
            let ask_dev = ((ask_c / level).ln()).abs();
            let max_dev = bid_dev.max(ask_dev);
            let min_dev = bid_dev.min(ask_dev);
            if max_dev > 0.02 && (min_dev < 1e-10 || max_dev / min_dev > 20.0) {
                if debug {
                    if let Some(ref mut db) = self.debug_buffers {
                        db.push_nan();
                    }
                }
                return false;
            }
        }

        // Capture mid-filter state BEFORE step for debug
        let mut d_mid_level = f64::NAN;
        let mut d_mid_scale = f64::NAN;
        let mut d_mid_u = f64::NAN;
        let mut d_mid_tau = f64::NAN;
        let mut d_mid_variance_rate = f64::NAN;
        let mut d_mid_cand_level = f64::NAN;
        let mut d_mid_cand_n_confirmed = f64::NAN;
        let mut d_mid_regime_switch = 0.0_f64;

        if debug {
            d_mid_level = self.mid_filter.current_level.unwrap_or(f64::NAN);
            d_mid_tau = self.mid_filter.tau;
            d_mid_variance_rate = self.mid_filter.variance_rate;
            if let (Some(level), Some(prev_ns)) = (self.mid_filter.current_level, self.mid_filter.current_time_ns) {
                let time_delta = ((time_ns - prev_ns) as f64 / 1e9).max(0.001);
                let scale = self.mid_filter.get_scale(time_delta);
                let price_change = (mid / level).ln();
                d_mid_scale = scale;
                d_mid_u = price_change / scale;
            }
        }

        // Gate 1: mid-price filter
        let mid_accepted = self.mid_filter.step(time_ns, mid);
        let regime_switched = self.mid_filter.last_step_regime_switch;

        if debug {
            d_mid_tau = self.mid_filter.tau;
            d_mid_variance_rate = self.mid_filter.variance_rate;
            d_mid_cand_level = self.mid_filter.candidate_level.map(|l| l.exp()).unwrap_or(f64::NAN);
            d_mid_cand_n_confirmed = self.mid_filter.candidate_n_confirmed as f64;
            d_mid_regime_switch = if regime_switched { 1.0 } else { 0.0 };
        }

        if !mid_accepted {
            if debug {
                if let Some(ref mut db) = self.debug_buffers {
                    db.mid_level.push(d_mid_level);
                    db.mid_scale.push(d_mid_scale);
                    db.mid_u.push(d_mid_u);
                    db.mid_tau.push(d_mid_tau);
                    db.mid_variance_rate.push(d_mid_variance_rate);
                    db.mid_cand_level.push(d_mid_cand_level);
                    db.mid_cand_n_confirmed.push(d_mid_cand_n_confirmed);
                    db.mid_regime_switch.push(d_mid_regime_switch);
                    db.spread_level.push(f64::NAN);
                    db.spread_scale.push(f64::NAN);
                    db.spread_u.push(f64::NAN);
                    db.spread_tau.push(f64::NAN);
                    db.spread_variance_rate.push(f64::NAN);
                    db.spread_upper_bound.push(f64::NAN);
                    db.gate.push(1.0); // mid_reject
                }
            }
            return false;
        }

        // After a mid-price regime switch, the relative spread baseline is
        // invalid (spread = (ask-bid)/mid, and mid just changed fundamentally).
        // Reset the spread filter to the current observation.
        if regime_switched {
            self.spread_filter.reset_to(spread, time_ns);
        }

        // During warmup, skip the spread gate entirely — let the spread
        // filter calibrate on real data before it starts rejecting.
        if self.n_accepted < self.warmup_count {
            // Still feed the observation to the spread filter for calibration
            let (_, spread_dbg) = self.spread_filter.step(time_ns, spread);
            self.n_accepted += 1;
            self.learn_tick(bid_c, ask_c);

            if debug {
                if let Some(ref mut db) = self.debug_buffers {
                    db.mid_level.push(d_mid_level);
                    db.mid_scale.push(d_mid_scale);
                    db.mid_u.push(d_mid_u);
                    db.mid_tau.push(d_mid_tau);
                    db.mid_variance_rate.push(d_mid_variance_rate);
                    db.mid_cand_level.push(d_mid_cand_level);
                    db.mid_cand_n_confirmed.push(d_mid_cand_n_confirmed);
                    db.mid_regime_switch.push(d_mid_regime_switch);
                    db.spread_level.push(spread_dbg.spread_level);
                    db.spread_scale.push(spread_dbg.spread_scale);
                    db.spread_u.push(spread_dbg.spread_u);
                    db.spread_tau.push(spread_dbg.spread_tau);
                    db.spread_variance_rate.push(spread_dbg.spread_variance_rate);
                    db.spread_upper_bound.push(spread_dbg.spread_upper_bound);
                    db.gate.push(3.0); // accepted (warmup bypass)
                }
            }
            return true;
        }

        // Gate 2: spread filter — feed data for calibration but don't
        // reject. Spread anomalies don't imply bad mid-prices.
        let (_, spread_dbg) = self.spread_filter.step(time_ns, spread);
        self.n_accepted += 1;
        self.learn_tick(bid_c, ask_c);

        if debug {
            if let Some(ref mut db) = self.debug_buffers {
                db.mid_level.push(d_mid_level);
                db.mid_scale.push(d_mid_scale);
                db.mid_u.push(d_mid_u);
                db.mid_tau.push(d_mid_tau);
                db.mid_variance_rate.push(d_mid_variance_rate);
                db.mid_cand_level.push(d_mid_cand_level);
                db.mid_cand_n_confirmed.push(d_mid_cand_n_confirmed);
                db.mid_regime_switch.push(d_mid_regime_switch);
                db.spread_level.push(spread_dbg.spread_level);
                db.spread_scale.push(spread_dbg.spread_scale);
                db.spread_u.push(spread_dbg.spread_u);
                db.spread_tau.push(spread_dbg.spread_tau);
                db.spread_variance_rate.push(spread_dbg.spread_variance_rate);
                db.spread_upper_bound.push(spread_dbg.spread_upper_bound);
                db.gate.push(3.0); // accepted
            }
        }

        true
    }
}

// ---------------------------------------------------------------------------
// QuoteRowProcessor implementation for QuoteOutlierDetector
// ---------------------------------------------------------------------------

impl QuoteRowProcessor for QuoteOutlierDetector {
    type Output = bool;

    #[inline(always)]
    fn process_row(&mut self, row: &QuoteRowData) -> bool {
        self.detect_quote(row.ts, row.bid_price, row.ask_price)
    }

    fn null_value() -> bool {
        false
    }

    #[inline]
    fn on_null(&mut self) {
        if let Some(ref mut db) = self.debug_buffers {
            db.push_nan();
        }
    }
}

// ---------------------------------------------------------------------------
// Retrospective refinement pass
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Quote Public API
// ---------------------------------------------------------------------------

/// Run quote outlier detection (mid + spread cascade) over quote arrays.
///
/// Rows where `valid[i]` is false are treated as missing.
pub fn detect_outliers_quotes_arrays(
    time_ns: &[i64],
    bid: &[f64],
    ask: &[f64],
    valid: &[bool],
    config: QuoteOutlierConfig,
) -> Vec<bool> {
    let mut detector = QuoteOutlierDetector::new(config);
    run_quotes(&mut detector, time_ns, bid, ask, valid)
}

/// Same as `detect_outliers_quotes_arrays` but also returns the debug columns.
pub fn detect_outliers_quotes_arrays_debug(
    time_ns: &[i64],
    bid: &[f64],
    ask: &[f64],
    valid: &[bool],
    config: QuoteOutlierConfig,
) -> QuoteDebugResult {
    let mut detector = QuoteOutlierDetector::new(config);
    detector.enable_debug(time_ns.len());
    let accepted = run_quotes(&mut detector, time_ns, bid, ask, valid);
    let debug = detector
        .take_debug_buffers()
        .expect("debug buffers were enabled");
    QuoteDebugResult { accepted, debug }
}

fn run_quotes(
    detector: &mut QuoteOutlierDetector,
    time_ns: &[i64],
    bid: &[f64],
    ask: &[f64],
    valid: &[bool],
) -> Vec<bool> {
    let mut out = Vec::with_capacity(time_ns.len());
    for i in 0..time_ns.len() {
        if valid[i] {
            out.push(detector.process_row(&QuoteRowData {
                ts: time_ns[i],
                bid_price: bid[i],
                ask_price: ask[i],
            }));
        } else {
            out.push(QuoteOutlierDetector::null_value());
            detector.on_null();
        }
    }
    out
}

// ---------------------------------------------------------------------------
// OHLCV Bar Outlier Detection
// ---------------------------------------------------------------------------

/// Pre-allocated column buffers for OHLCV bar debug output.
pub struct OhlcvDebugBuffers {
    pub bar_open: Vec<f64>,
    pub bar_high: Vec<f64>,
    pub bar_low: Vec<f64>,
    pub bar_close: Vec<f64>,
    pub bar_volume: Vec<f64>,
    pub bar_invalid_reason: Vec<f64>,
}

impl OhlcvDebugBuffers {
    fn with_capacity(n: usize) -> Self {
        Self {
            bar_open: Vec::with_capacity(n),
            bar_high: Vec::with_capacity(n),
            bar_low: Vec::with_capacity(n),
            bar_close: Vec::with_capacity(n),
            bar_volume: Vec::with_capacity(n),
            bar_invalid_reason: Vec::with_capacity(n),
        }
    }

    #[inline]
    fn push_bar(&mut self, reason: f64, row: &OhlcvRowData) {
        self.bar_open.push(row.open);
        self.bar_high.push(row.high);
        self.bar_low.push(row.low);
        self.bar_close.push(row.close);
        self.bar_volume.push(row.volume);
        self.bar_invalid_reason.push(reason);
    }

    #[inline]
    fn push_nan(&mut self) {
        self.bar_open.push(f64::NAN);
        self.bar_high.push(f64::NAN);
        self.bar_low.push(f64::NAN);
        self.bar_close.push(f64::NAN);
        self.bar_volume.push(f64::NAN);
        self.bar_invalid_reason.push(f64::NAN);
    }
}

/// Result containing both the accepted mask and debug buffers for OHLCV detection.
pub struct OhlcvDebugResult {
    pub accepted: Vec<bool>,
    pub inner_debug: DebugBuffers,
    pub ohlcv_debug: OhlcvDebugBuffers,
}

/// OHLCV bar outlier detector.
///
/// Wraps the full ensemble `OutlierDetector` (6 methods on close price)
/// and adds bar-consistency pre-validation:
/// - Non-finite OHLC prices → hard-reject (reason=1)
/// - high < low → hard-reject (reason=2)
/// Note: zero/negative prices are valid (e.g. settlement-priced or spread instruments).
pub struct OhlcvOutlierDetector {
    inner: OutlierDetector,
    debug_buffers: Option<OhlcvDebugBuffers>,
}

impl OhlcvOutlierDetector {
    pub fn new(config: OutlierConfig) -> Self {
        Self {
            inner: OutlierDetector::new(config),
            debug_buffers: None,
        }
    }

    pub fn enable_debug(&mut self, total_rows: usize) {
        self.inner.enable_debug(total_rows);
        self.debug_buffers = Some(OhlcvDebugBuffers::with_capacity(total_rows));
    }

    pub fn take_debug_buffers(&mut self) -> Option<(DebugBuffers, OhlcvDebugBuffers)> {
        let inner_db = self.inner.take_debug_buffers()?;
        let ohlcv_db = self.debug_buffers.take()?;
        Some((inner_db, ohlcv_db))
    }

    /// Returns `true` if the bar is accepted (not an outlier).
    #[inline]
    fn detect_bar(&mut self, row: &OhlcvRowData) -> bool {
        let debug = self.debug_buffers.is_some();

        // --- Bar consistency pre-validation ---
        // Reason codes: 0=ok, 1=non-finite, 2=high<low
        // Note: zero/negative prices are valid (e.g. settlement-priced or spread instruments).
        let (bar_invalid, reason) = if !row.open.is_finite()
            || !row.high.is_finite()
            || !row.low.is_finite()
            || !row.close.is_finite()
        {
            (true, 1.0)
        } else if row.high < row.low {
            (true, 2.0)
        } else {
            (false, 0.0)
        };

        if bar_invalid {
            // Push NaN into inner detector's debug buffers for alignment
            self.inner.on_null();
            if debug {
                if let Some(ref mut db) = self.debug_buffers {
                    db.push_bar(reason, row);
                }
            }
            return false;
        }

        // --- Feed close price through the inner ensemble detector ---
        let accepted = self.inner.detect(row.ts, row.close, row.volume);

        if debug {
            if let Some(ref mut db) = self.debug_buffers {
                db.push_bar(reason, row);
            }
        }

        accepted
    }
}

// ---------------------------------------------------------------------------
// OhlcvRowProcessor implementation for OhlcvOutlierDetector
// ---------------------------------------------------------------------------

impl OhlcvRowProcessor for OhlcvOutlierDetector {
    type Output = bool;

    #[inline(always)]
    fn process_row(&mut self, row: &OhlcvRowData) -> bool {
        self.detect_bar(row)
    }

    fn null_value() -> bool {
        false
    }

    #[inline]
    fn on_null(&mut self) {
        self.inner.on_null();
        if let Some(ref mut db) = self.debug_buffers {
            db.push_nan();
        }
    }
}

// ---------------------------------------------------------------------------
// OHLCV Public API
// ---------------------------------------------------------------------------

/// Run OHLCV bar outlier detection over bar arrays.
///
/// Rows where `valid[i]` is false are treated as missing.
pub fn detect_outliers_ohlcv_arrays(
    time_ns: &[i64],
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    valid: &[bool],
    config: OutlierConfig,
) -> Vec<bool> {
    let mut detector = OhlcvOutlierDetector::new(config);
    run_ohlcv(&mut detector, time_ns, open, high, low, close, volume, valid)
}

/// Same as `detect_outliers_ohlcv_arrays` but also returns the debug columns.
pub fn detect_outliers_ohlcv_arrays_debug(
    time_ns: &[i64],
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    valid: &[bool],
    config: OutlierConfig,
) -> OhlcvDebugResult {
    let mut detector = OhlcvOutlierDetector::new(config);
    detector.enable_debug(time_ns.len());
    let accepted = run_ohlcv(&mut detector, time_ns, open, high, low, close, volume, valid);
    let (inner_debug, ohlcv_debug) = detector
        .take_debug_buffers()
        .expect("debug buffers were enabled");
    OhlcvDebugResult { accepted, inner_debug, ohlcv_debug }
}

#[allow(clippy::too_many_arguments)]
fn run_ohlcv(
    detector: &mut OhlcvOutlierDetector,
    time_ns: &[i64],
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    valid: &[bool],
) -> Vec<bool> {
    let mut out = Vec::with_capacity(time_ns.len());
    for i in 0..time_ns.len() {
        if valid[i] {
            out.push(detector.process_row(&OhlcvRowData {
                ts: time_ns[i],
                open: open[i],
                high: high[i],
                low: low[i],
                close: close[i],
                volume: volume[i],
            }));
        } else {
            out.push(OhlcvOutlierDetector::null_value());
            detector.on_null();
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_stats_incremental() {
        let mut rs = RollingStats::new(100);
        for i in 1..=100 {
            rs.update(i as f64);
        }
        let stats = rs.get_stats().unwrap();
        // mean of 1..=100 = 50.5
        assert!(
            (stats.mean - 50.5).abs() < 0.01,
            "mean was {}",
            stats.mean
        );
        assert!(stats.std > 0.0);
        // median ≈ 50.5
        let median = percentile_linear(&rs.sorted, 0.5);
        assert!(
            (median - 50.5).abs() < 1.0,
            "median was {}",
            median
        );
    }

    #[test]
    fn test_rolling_stats_eviction() {
        let mut rs = RollingStats::new(5);
        for i in 1..=10 {
            rs.update(i as f64);
        }
        // Window should be [6,7,8,9,10]
        assert_eq!(rs.buffer.len(), 5);
        assert_eq!(rs.sorted.len(), 5);
        assert_eq!(rs.sorted, vec![6.0, 7.0, 8.0, 9.0, 10.0]);
        let expected_sum: f64 = (6..=10).map(|x| x as f64).sum();
        assert!((rs.sum - expected_sum).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_filter_bootstrap() {
        let mut f = AdaptiveLevelFilter::new(14400.0);
        assert!(f.step(1_000_000_000, 100.0)); // bootstrap always accepts
        assert_eq!(f.n_accepted, 1);
        assert_eq!(f.current_level, Some(100.0));
    }

    #[test]
    fn test_adaptive_filter_tau_two_way() {
        let mut f = AdaptiveLevelFilter::new(14400.0);
        let base_ns: i64 = 1_000_000_000_000;
        let one_sec_ns: i64 = 1_000_000_000;

        // Bootstrap + 20 normal observations to get past n_accepted > 10
        f.step(base_ns, 100.0);
        for i in 1..=20 {
            f.step(base_ns + i * one_sec_ns, 100.0 + 0.001 * i as f64);
        }
        let tau_after_accepts = f.tau;

        // Now inject a far-away price that gets rejected
        let rejected = f.step(base_ns + 21 * one_sec_ns, 200.0);
        assert!(!rejected, "200.0 should be rejected when level ≈ 100");
        // tau should have increased on rejection
        assert!(
            f.tau > tau_after_accepts,
            "tau should increase on rejection: {} vs {}",
            f.tau,
            tau_after_accepts
        );
    }

    /// Helper: run the detector on a sequence of (time_ns, price, volume) tuples.
    fn run_detector(data: &[(i64, f64, f64)], config: OutlierConfig) -> Vec<bool> {
        let mut detector = OutlierDetector::new(config);
        data.iter()
            .map(|&(t, p, v)| detector.detect(t, p, v))
            .collect()
    }

    #[test]
    fn test_regime_shift_detection() {
        // 100 trades at price 100, then jump to ~200.
        // The 3rd point at the new level should be accepted via regime switch.
        //
        // We isolate the adaptive level filter (disable other methods) because
        // rolling z-score/IQR have no regime-shift logic and would keep voting
        // against the new level until the window fills.
        //
        // Prices at the new level vary slightly (as in a real market) to avoid
        // being flagged as stale quotes.
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64)> = Vec::new();
        // 100 normal trades at price 100
        for i in 0..100 {
            data.push((base + i * one_sec, 100.0, 10.0));
        }
        // Jump to ~200 — slight variation to reflect real tick movement
        for i in 0..10 {
            data.push((base + (100 + i) * one_sec, 200.0 + 0.1 * i as f64, 10.0));
        }

        let mut config = OutlierConfig::default();
        config.use_rolling_zscore = false;
        config.use_iqr = false;
        config.use_volume_check = false;
        config.use_time_gap_check = false;
        config.use_price_jump_check = false;
        config.min_votes_to_reject = 1;

        let results = run_detector(&data, config);

        // All 100 trades at price=100 should be accepted
        assert!(
            results[..100].iter().all(|&b| b),
            "All normal trades should be accepted"
        );

        // First 1-2 trades at 200 may be rejected (before regime shift kicks in)
        // But by the n_switch-th point (3rd), regime switch should accept
        let accepted_at_200: Vec<usize> = results[100..]
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect();

        assert!(
            !accepted_at_200.is_empty(),
            "Regime shift should eventually accept the new level"
        );
        // The first accepted point at the new level should be at most at index n_switch (3)
        assert!(
            accepted_at_200[0] <= 3,
            "First accepted at new level should be by the 3rd point, was at index {}",
            accepted_at_200[0]
        );
    }

    #[test]
    fn test_single_outlier_rejected() {
        // Normal trades, then a 10x spike, then normal again.
        // The spike should be rejected.
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64)> = Vec::new();
        // 50 normal trades at ~100
        for i in 0..50 {
            data.push((base + i * one_sec, 100.0 + 0.01 * i as f64, 10.0));
        }
        // Single spike at 10x
        data.push((base + 50 * one_sec, 1000.0, 10.0));
        // Return to normal
        for i in 51..60 {
            data.push((base + i * one_sec, 100.0 + 0.01 * i as f64, 10.0));
        }

        let results = run_detector(&data, OutlierConfig::default());

        // The spike at index 50 should be rejected
        assert!(
            !results[50],
            "10x price spike should be rejected"
        );
        // Normal trades before and after should be accepted
        assert!(
            results[10..50].iter().all(|&b| b),
            "Normal trades before spike should be accepted"
        );
        assert!(
            results[51..].iter().all(|&b| b),
            "Normal trades after spike should be accepted"
        );
    }

    #[test]
    fn test_vote_threshold() {
        // With min_votes_to_reject=2, a point flagged by only 1 method should
        // be accepted.  With min_votes_to_reject=1, it should be rejected.
        //
        // Strategy: use only adaptive level filter (disable others) so exactly
        // 1 vote is possible, then toggle the threshold.
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64)> = Vec::new();
        // 20 normal trades
        for i in 0..20 {
            data.push((base + i * one_sec, 100.0, 10.0));
        }
        // A moderately deviant price that the adaptive filter rejects but
        // rolling stats (with only 20 points of history) won't flag
        data.push((base + 20 * one_sec, 130.0, 10.0));

        // Config with only adaptive level filter enabled → max 1 vote from
        // this method. With min_votes=2, the point should be accepted.
        let mut config_2votes = OutlierConfig::default();
        config_2votes.use_rolling_zscore = false;
        config_2votes.use_iqr = false;
        config_2votes.use_volume_check = false;
        config_2votes.use_time_gap_check = false;
        config_2votes.use_price_jump_check = false;
        config_2votes.min_votes_to_reject = 2;
        config_2votes.warmup_threshold = 0;

        let results_2 = run_detector(&data, config_2votes);
        // With only 1 method active and threshold=2, even outliers pass
        assert!(
            results_2[20],
            "With min_votes=2 and only 1 method active, deviant point should be accepted"
        );

        // Same config but with min_votes=1 → should reject
        let mut config_1vote = OutlierConfig::default();
        config_1vote.use_rolling_zscore = false;
        config_1vote.use_iqr = false;
        config_1vote.use_volume_check = false;
        config_1vote.use_time_gap_check = false;
        config_1vote.use_price_jump_check = false;
        config_1vote.min_votes_to_reject = 1;
        config_1vote.warmup_threshold = 0;

        let results_1 = run_detector(&data, config_1vote);
        assert!(
            !results_1[20],
            "With min_votes=1 and adaptive filter rejecting, deviant point should be rejected"
        );
    }

    #[test]
    fn test_merged_stat_votes_no_double_count() {
        // 1000 stable trades at price 100, then a moderate deviation that
        // triggers both z-score and IQR but NOT the adaptive filter.
        //
        // With merge_stat_votes=true (default): statistical tests give 1 vote,
        // total = 1 < min_votes_to_reject(2) → ACCEPTED.
        //
        // With merge_stat_votes=false: z-score gives 1, IQR gives 1,
        // total = 2 >= min_votes_to_reject(2) → REJECTED.
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64)> = Vec::new();
        // 1000 stable trades at price ~100.0 with tiny noise so std > 0
        for i in 0..1000 {
            let noise = 0.0001 * ((i as f64 * 0.3).sin());
            data.push((base + i * one_sec, 100.0 + noise, 10.0));
        }
        // A large deviation: triggers both z-score (>5σ) and IQR.
        // With tiny noise, std ≈ 0.00007, so a 1.0 move is ~14000σ.
        // IQR ≈ 0.0001, so 101.0 is far outside bounds.
        data.push((base + 1000 * one_sec, 101.0, 10.0));

        // Config: only z-score + IQR enabled (disable adaptive, price_jump, etc.)
        // so we isolate the statistical double-count behaviour.
        let mut config_merged = OutlierConfig::default();
        config_merged.use_adaptive_level = false;
        config_merged.use_volume_check = false;
        config_merged.use_time_gap_check = false;
        config_merged.use_price_jump_check = false;
        config_merged.merge_stat_votes = true;  // default
        config_merged.min_votes_to_reject = 2;

        let results_merged = run_detector(&data, config_merged);
        assert!(
            results_merged[1000],
            "With merge_stat_votes=true, stat tests give 1 vote → accepted (1 < 2)"
        );

        // Same but with merge disabled → 2 votes → rejected
        let mut config_basic = OutlierConfig::default();
        config_basic.use_adaptive_level = false;
        config_basic.use_volume_check = false;
        config_basic.use_time_gap_check = false;
        config_basic.use_price_jump_check = false;
        config_basic.merge_stat_votes = false;
        config_basic.min_votes_to_reject = 2;

        let results_basic = run_detector(&data, config_basic);
        assert!(
            !results_basic[1000],
            "With merge_stat_votes=false, stat tests give 2 votes → rejected (2 >= 2)"
        );
    }

    #[test]
    fn test_weekend_gap_accepted() {
        // 100 trades at price 100, then a 48-hour gap (weekend), then trades at 99.5.
        // Should be accepted — legitimate post-gap price within normal range.
        // The adaptive filter widens tau after a gap.
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64)> = Vec::new();
        // 100 normal trades at price 100, 1 second apart
        for i in 0..100 {
            data.push((base + i * one_sec, 100.0, 10.0));
        }
        // 48-hour gap (weekend), then trades at 99.5 (0.5% move — perfectly normal)
        let weekend_gap_ns: i64 = 48 * 3600 * one_sec;
        for i in 0..5 {
            data.push((base + 100 * one_sec + weekend_gap_ns + i * one_sec, 99.5, 10.0));
        }

        let results = run_detector(&data, OutlierConfig::default());

        // All pre-gap trades should be accepted
        assert!(
            results[..100].iter().all(|&b| b),
            "All normal trades before gap should be accepted"
        );
        // Post-gap trades at 99.5 should be accepted (within normal range)
        let post_gap_accepted: Vec<usize> = results[100..]
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect();
        assert!(
            !post_gap_accepted.is_empty(),
            "Post-weekend trades at 99.5 (0.5% move) should be accepted, not rejected as outliers"
        );
        // The first post-gap trade should be accepted thanks to tau widening
        assert!(
            results[100],
            "First post-weekend trade should be accepted (tau widened after 48h gap)"
        );
    }

    #[test]
    fn test_illiquid_stale_accepted() {
        // 20 trades at price 100 with 1-hour spacing, then a 3-day gap, then trade at 100.05.
        // Should be accepted — legitimate illiquid instrument.
        // The stale voter doesn't fire on a 3-day gap (threshold is 7 days).
        let one_sec: i64 = 1_000_000_000;
        let one_hour: i64 = 3600 * one_sec;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64)> = Vec::new();
        // 20 trades at price 100, 1 hour apart
        for i in 0..20 {
            data.push((base + i * one_hour, 100.0, 10.0));
        }
        // 3-day gap, then trade at 100.05 (tiny move — legitimate)
        let three_days_ns: i64 = 3 * 24 * 3600 * one_sec;
        data.push((base + 20 * one_hour + three_days_ns, 100.05, 10.0));

        let results = run_detector(&data, OutlierConfig::default());

        // The post-gap trade should be accepted (3 days < 7 day stale threshold)
        assert!(
            results[20],
            "Trade after 3-day gap with tiny price move should be accepted (stale threshold is 7 days)"
        );
    }

    #[test]
    fn test_genuine_stale_rebroadcast_rejected() {
        // 100 trades at price 100, then a 10-day gap, then trade at exactly 100.0.
        // Should be rejected — suspiciously long gap with identical price.
        // The stale-price safety net still applies on long gaps.
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64)> = Vec::new();
        // 100 normal trades at price 100, 1 second apart
        for i in 0..100 {
            data.push((base + i * one_sec, 100.0, 10.0));
        }
        // 10-day gap, then trade at exactly 100.0 (suspicious stale rebroadcast)
        let ten_days_ns: i64 = 10 * 24 * 3600 * one_sec;
        data.push((base + 100 * one_sec + ten_days_ns, 100.0, 10.0));

        // Use config with min_votes=1 and only time_gap + adaptive enabled
        // to ensure the stale voter can single-handedly reject
        let mut config = OutlierConfig::default();
        config.min_votes_to_reject = 1;
        config.use_rolling_zscore = false;
        config.use_iqr = false;
        config.use_volume_check = false;
        config.use_price_jump_check = false;

        let results = run_detector(&data, config);

        assert!(
            !results[100],
            "Trade after 10-day gap at exactly the same price should be rejected as stale rebroadcast"
        );
    }

    #[test]
    fn test_stale_quotes_dont_trigger_regime_switch() {
        // 100 trades at price 100, then repeated stale quotes at price 50.
        // The stale quotes should NOT trigger a regime switch because they
        // all have the exact same price (no tick variation = not real market).
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64)> = Vec::new();
        for i in 0..100 {
            data.push((base + i * one_sec, 100.0, 10.0));
        }
        // 20 stale quotes at exactly 50.0 — all identical
        for i in 0..20 {
            data.push((base + (100 + i) * one_sec, 50.0, 10.0));
        }
        // Return to true price — these should all be accepted
        for i in 0..10 {
            data.push((base + (120 + i) * one_sec, 100.0 + 0.01 * i as f64, 10.0));
        }

        let mut config = OutlierConfig::default();
        config.use_rolling_zscore = false;
        config.use_iqr = false;
        config.use_volume_check = false;
        config.use_time_gap_check = false;
        config.use_price_jump_check = false;
        config.min_votes_to_reject = 1;
        let results = run_detector(&data, config);

        // All 20 stale quotes should be rejected (never accepted)
        assert!(
            results[100..120].iter().all(|&b| !b),
            "Stale quotes at identical price should all be rejected"
        );
        // Return to true price should be accepted
        assert!(
            results[120..130].iter().all(|&b| b),
            "True prices after stale period should be accepted (no regime switch happened)"
        );
    }

    #[test]
    fn test_interleaved_stale_quotes_no_regime_switch() {
        // Simulate a pattern seen in real quote data: a frozen stale ask
        // interleaved with the true ask at a much lower level.
        let ms: i64 = 1_000_000; // 1 millisecond
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64)> = Vec::new();
        // 100 normal trades to establish level at 100
        for i in 0..100 {
            data.push((base + i * 1_000_000_000, 100.0 + 0.01 * (i % 5) as f64, 10.0));
        }
        // 200 interleaved: true at ~100 alternating with stale at ~50
        // Stale quotes have slightly varying prices (e.g. from different
        // normal-side ticks) so they're not exact duplicates
        let t_start = base + 100 * 1_000_000_000;
        for i in 0..200 {
            if i % 2 == 0 {
                // True quote
                data.push((t_start + i * ms, 100.0 + 0.01 * (i % 7) as f64, 10.0));
            } else {
                // Stale quote — slightly varying (50.0, 50.01, 50.02, ...)
                data.push((t_start + i * ms, 50.0 + 0.01 * (i % 10) as f64, 10.0));
            }
        }
        // Return to only true prices
        for i in 0..20 {
            data.push((t_start + (200 + i) * 1_000_000_000, 100.0 + 0.01 * i as f64, 10.0));
        }

        let mut config = OutlierConfig::default();
        config.use_rolling_zscore = false;
        config.use_iqr = false;
        config.use_volume_check = false;
        config.use_time_gap_check = false;
        config.use_price_jump_check = false;
        config.min_votes_to_reject = 1;
        let results = run_detector(&data, config);

        // True quotes (even indices in the interleaved section) should be accepted
        let true_accepted: usize = (100..300)
            .step_by(2)
            .filter(|&i| results[i])
            .count();
        assert!(
            true_accepted >= 90,
            "Most true quotes during interleaving should be accepted, got {}/100",
            true_accepted
        );
        // Return-to-normal should all be accepted (no regime switch happened)
        assert!(
            results[300..320].iter().all(|&b| b),
            "All post-interleaving normal quotes should be accepted"
        );
    }

    // -----------------------------------------------------------------------
    // Regime shifts, warm-up and level-filter behaviour
    // -----------------------------------------------------------------------

    #[test]
    fn test_regime_switch_override_with_all_voters() {
        // 100 trades at 100, then 10 at ~200, ALL voters enabled.
        // The 3rd new-level point must be accepted (regime switch overrides
        // z-score + price_jump votes).
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64)> = Vec::new();
        for i in 0..100 {
            data.push((base + i * one_sec, 100.0, 10.0));
        }
        for i in 0..10 {
            data.push((base + (100 + i) * one_sec, 200.0 + 0.1 * i as f64, 10.0));
        }

        // All voters enabled (default config), warmup disabled so it doesn't interfere
        let mut config = OutlierConfig::default();
        config.warmup_threshold = 0;
        let results = run_detector(&data, config);

        // All 100 normal trades accepted
        assert!(
            results[..100].iter().all(|&b| b),
            "All normal trades should be accepted"
        );

        // At least by the 3rd point at 200, regime switch should force acceptance
        let accepted_at_200: Vec<usize> = results[100..]
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect();

        assert!(
            !accepted_at_200.is_empty(),
            "Regime switch should force acceptance at the new level even with all voters"
        );
        assert!(
            accepted_at_200[0] <= 3,
            "First accepted at new level should be by the 3rd point (regime switch override), was at index {}",
            accepted_at_200[0]
        );

        // After regime switch, subsequent points at 200 should also be accepted
        // (window was reset, so z-score/IQR don't have old-regime data)
        let post_switch_accepted = results[100 + accepted_at_200[0]..110]
            .iter()
            .filter(|&&b| b)
            .count();
        assert!(
            post_switch_accepted >= 5,
            "Most post-switch trades at new level should be accepted (window reset), got {}/{}",
            post_switch_accepted,
            110 - (100 + accepted_at_200[0])
        );
    }

    #[test]
    fn test_no_override_for_single_outlier() {
        // Normal trades, then a 10x spike (single point), then normal again.
        // Regime switch should NOT fire for a single outlier.
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64)> = Vec::new();
        for i in 0..100 {
            data.push((base + i * one_sec, 100.0 + 0.01 * i as f64, 10.0));
        }
        // Single spike
        data.push((base + 100 * one_sec, 1000.0, 10.0));
        // Return to normal
        for i in 101..110 {
            data.push((base + i * one_sec, 100.0 + 0.01 * i as f64, 10.0));
        }

        let mut config = OutlierConfig::default();
        config.warmup_threshold = 0;
        let results = run_detector(&data, config);

        assert!(
            !results[100],
            "Single 10x spike should be rejected (regime switch needs 3 consecutive points)"
        );
        assert!(
            results[101..].iter().all(|&b| b),
            "Normal trades after spike should be accepted"
        );
    }

    #[test]
    fn test_clean_window_no_contamination() {
        // After rejected outlier burst, window mean/std should be unaffected
        // because only accepted prices enter the window.
        // Use 2 spikes (not 3+) to avoid triggering regime switch (n_switch=3).
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64)> = Vec::new();
        // 100 normal trades at 100
        for i in 0..100 {
            data.push((base + i * one_sec, 100.0, 10.0));
        }
        // 2 outlier spikes at 1000 (rejected, won't trigger regime switch)
        for i in 0..2 {
            data.push((base + (100 + i) * one_sec, 1000.0, 10.0));
        }
        // Return to normal — these should be accepted if window is clean
        for i in 0..10 {
            data.push((base + (102 + i) * one_sec, 100.0, 10.0));
        }

        let mut config = OutlierConfig::default();
        config.warmup_threshold = 0;

        let mut detector = OutlierDetector::new(config);
        let results: Vec<bool> = data.iter()
            .map(|&(t, p, v)| detector.detect(t, p, v))
            .collect();

        // The 2 spikes should be rejected
        assert!(
            results[100..102].iter().all(|&b| !b),
            "Both 1000-price spikes should be rejected"
        );

        // Post-spike normal trades should all be accepted (clean window)
        assert!(
            results[102..112].iter().all(|&b| b),
            "Normal trades after rejected spikes should be accepted (window not contaminated)"
        );

        // Verify the rolling window wasn't contaminated: its mean should be ~100, not shifted
        if let Some(ref rs) = detector.rolling_stats {
            if let Some(stats) = rs.get_stats() {
                assert!(
                    (stats.mean - 100.0).abs() < 1.0,
                    "Rolling window mean should be ~100 (not contaminated), got {}",
                    stats.mean
                );
            }
        }
    }

    #[test]
    fn test_window_reset_after_regime_shift() {
        // 100 trades at 100, then regime shift to ~200. After the switch,
        // 50 trades at 200 should all be accepted (window reset → z-score/IQR
        // inactive until re-calibrated on new-regime data).
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64)> = Vec::new();
        for i in 0..100 {
            data.push((base + i * one_sec, 100.0, 10.0));
        }
        // Jump to ~200 — regime switch fires on ~3rd point
        // Slight variation (±0.05) to reflect real tick movement
        for i in 0..53 {
            data.push((base + (100 + i) * one_sec, 200.0 + 0.05 * (i as f64 % 3.0 - 1.0), 10.0));
        }

        let mut config = OutlierConfig::default();
        config.warmup_threshold = 0;
        let results = run_detector(&data, config);

        // Find the regime switch point (first accepted at new level)
        let first_accepted_200 = results[100..]
            .iter()
            .position(|&b| b)
            .expect("Regime switch should accept at least one point at 200");

        // All 50 trades after the regime switch point should be accepted
        let post_switch = &results[100 + first_accepted_200..153];
        let accepted_count = post_switch.iter().filter(|&&b| b).count();
        assert!(
            accepted_count == post_switch.len(),
            "All {} post-regime-switch trades at 200 should be accepted (window reset), but only {} were",
            post_switch.len(),
            accepted_count
        );
    }

    #[test]
    fn test_warmup_lenient_on_sparse_data() {
        // 30 trades with mild variation, no rejections during warmup
        // (effective threshold = min_votes_to_reject + 1 = 3).
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64)> = Vec::new();
        for i in 0..30 {
            // Mild variation: 100 ± 0.5
            let price = 100.0 + 0.5 * ((i as f64 * 0.7).sin());
            data.push((base + i * one_sec, price, 10.0));
        }

        let mut config = OutlierConfig::default();
        config.warmup_threshold = 50; // default — all 30 points are in warmup

        let results = run_detector(&data, config);

        // During warmup, effective threshold is 3 (not 2), so mild variations
        // that might get 2 votes should still be accepted
        let rejected = results.iter().filter(|&&b| !b).count();
        assert_eq!(
            rejected, 0,
            "No rejections expected during warmup with mild variation, got {}",
            rejected
        );
    }

    #[test]
    fn test_warmup_still_rejects_extreme() {
        // 10x spike during warmup should still be rejected (gets 3+ votes).
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64)> = Vec::new();
        // 20 normal trades
        for i in 0..20 {
            data.push((base + i * one_sec, 100.0, 10.0));
        }
        // Extreme spike at 10x
        data.push((base + 20 * one_sec, 1000.0, 10.0));

        let mut config = OutlierConfig::default();
        config.warmup_threshold = 50; // still in warmup at trade 20

        let results = run_detector(&data, config);

        assert!(
            !results[20],
            "10x spike during warmup should still be rejected (gets 3+ votes from adaptive + stat + price_jump)"
        );
    }

    // -----------------------------------------------------------------------
    // Quote outlier detection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_spread_filter_bootstrap() {
        let mut sf = SpreadFilter::new();
        let (accepted, dbg) = sf.step(1_000_000_000, 0.0005);
        assert!(accepted, "Bootstrap should always accept");
        assert_eq!(sf.spread_level, Some(0.0005));
        assert!((dbg.spread_u - 0.0).abs() < 1e-10, "Bootstrap u should be 0");
    }

    #[test]
    fn test_spread_filter_upper_bound_only() {
        let mut sf = SpreadFilter::new();
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        // Bootstrap with a normal spread
        sf.step(base, 0.0005);

        // Very small spread (u < 0) should always be accepted
        let (accepted, dbg) = sf.step(base + one_sec, 0.0001);
        assert!(accepted, "Smaller spread should always be accepted (upper-bound only)");
        assert!(dbg.spread_u < 0.0, "u should be negative for smaller spread");

        // Huge spread should be rejected
        let (accepted, _) = sf.step(base + 2 * one_sec, 100.0);
        assert!(!accepted, "Absurdly large spread should be rejected");
    }

    #[test]
    fn test_quote_hard_validation() {
        let mut detector = QuoteOutlierDetector::new(QuoteOutlierConfig::default());

        // Crossed (bid > ask): invalid, like high < low for a bar
        assert!(!detector.detect_quote(1_000_000_000, 100.0, 99.0));
        assert!(!detector.detect_quote(1_500_000_000, 110.0, 100.0));
        // Invalid: negative bid
        assert!(!detector.detect_quote(2_000_000_000, -1.0, 100.0));
        // Invalid: NaN
        assert!(!detector.detect_quote(3_000_000_000, f64::NAN, 100.0));
        // Invalid: zero bid
        assert!(!detector.detect_quote(4_000_000_000, 0.0, 100.0));
        // Valid; locked (bid == ask) is valid too
        assert!(detector.detect_quote(5_000_000_000, 99.0, 100.0));
        assert!(detector.detect_quote(6_000_000_000, 100.0, 100.0));

        // allow_crossed: micro-cross swapped and tolerated, extreme cross still rejected
        let mut cfg = QuoteOutlierConfig::default();
        cfg.allow_crossed = true;
        let mut lenient = QuoteOutlierDetector::new(cfg);
        assert!(lenient.detect_quote(1_000_000_000, 100.0, 99.0));
        assert!(!lenient.detect_quote(1_500_000_000, 110.0, 100.0));
    }

    #[test]
    fn test_quote_two_gate_cascade() {
        // Gate 1 (mid) rejects a mid-price outlier; gate 2 (spread) never runs.
        // Then gate 1 passes but gate 2 rejects a huge spread.
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut detector = QuoteOutlierDetector::new(QuoteOutlierConfig::default());

        // Bootstrap with normal quote
        assert!(detector.detect_quote(base, 100.0, 100.001));

        // Feed a few more normal quotes to build state
        for i in 1..20 {
            assert!(detector.detect_quote(base + i * one_sec, 100.0, 100.001));
        }

        // Mid-price outlier: bid=200, ask=200.001 → mid=200 is far from level ≈ 100
        let mid_outlier = detector.detect_quote(base + 20 * one_sec, 200.0, 200.001);
        assert!(!mid_outlier, "Mid-price outlier should be rejected by gate 1");

        // Normal mid but absurd spread: bid=100, ask=200
        let spread_outlier = detector.detect_quote(base + 21 * one_sec, 100.0, 200.0);
        assert!(!spread_outlier, "Huge spread should be rejected by gate 2");
    }

    #[test]
    fn test_quote_normal_data_accepted() {
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut detector = QuoteOutlierDetector::new(QuoteOutlierConfig::default());

        let mut accepted_count = 0;
        let total = 1000;
        for i in 0..total {
            let noise = 0.0001 * ((i as f64 * 0.3).sin());
            let bid = 100.0 + noise;
            let ask = bid + 0.001; // typical tight spread
            if detector.detect_quote(base + (i as i64) * one_sec, bid, ask) {
                accepted_count += 1;
            }
        }

        let acceptance_rate = accepted_count as f64 / total as f64;
        assert!(
            acceptance_rate > 0.99,
            "Normal quote data should have >99% acceptance rate, got {:.2}%",
            acceptance_rate * 100.0
        );
    }

    // -----------------------------------------------------------------------
    // OHLCV bar outlier detection tests
    // -----------------------------------------------------------------------

    /// Helper: run the OHLCV detector on a sequence of (time_ns, open, high, low, close, volume).
    fn run_ohlcv_detector(
        data: &[(i64, f64, f64, f64, f64, f64)],
        config: OutlierConfig,
    ) -> Vec<bool> {
        let mut detector = OhlcvOutlierDetector::new(config);
        data.iter()
            .map(|&(t, o, h, l, c, v)| {
                let row = OhlcvRowData {
                    ts: t,
                    open: o,
                    high: h,
                    low: l,
                    close: c,
                    volume: v,
                };
                detector.detect_bar(&row)
            })
            .collect()
    }

    #[test]
    fn test_ohlcv_bar_consistency_rejects() {
        let base: i64 = 1_000_000_000_000;
        let one_sec: i64 = 1_000_000_000;

        // Bootstrap with a valid bar
        let mut data = vec![(base, 100.0, 101.0, 99.0, 100.5, 10.0)];

        // Non-finite close → rejected (reason=1)
        data.push((base + one_sec, 100.0, 101.0, 99.0, f64::NAN, 10.0));
        // Non-finite high → rejected (reason=1)
        data.push((base + 2 * one_sec, 100.0, f64::INFINITY, 99.0, 100.5, 10.0));
        // high < low → rejected (reason=2)
        data.push((base + 3 * one_sec, 100.0, 98.0, 102.0, 100.5, 10.0));
        // Valid bar → accepted
        data.push((base + 4 * one_sec, 100.0, 101.0, 99.0, 100.5, 10.0));

        let results = run_ohlcv_detector(&data, OutlierConfig::default());

        assert!(results[0], "Valid bootstrap bar should be accepted");
        assert!(!results[1], "NaN close should be rejected");
        assert!(!results[2], "Infinity high should be rejected");
        assert!(!results[3], "high < low should be rejected");
        assert!(results[4], "Valid bar should be accepted");
    }

    #[test]
    fn test_ohlcv_zero_and_negative_prices_accepted() {
        // Settlement-priced and spread instruments can have zero/negative prices.
        // Verify that zero and negative prices do NOT trigger hard-rejects
        // (bar_invalid_reason must be 0 for all bars with valid OHLC structure).
        let base: i64 = 1_000_000_000_000;
        let one_sec: i64 = 1_000_000_000;
        let n = 50usize;

        let mut detector = OhlcvOutlierDetector::new(OutlierConfig::default());
        detector.enable_debug(n);
        for i in 0..n {
            // Close drifts from -0.50 to -0.01 (all negative)
            let c = -0.5 + 0.01 * i as f64;
            let row = OhlcvRowData {
                ts: base + (i as i64) * one_sec,
                open: c - 0.1,
                high: c + 0.2,
                low: c - 0.2,
                close: c,
                volume: 10.0,
            };
            detector.detect_bar(&row);
        }

        let (_inner_debug, ohlcv_debug) = detector.take_debug_buffers().unwrap();
        let reasons = &ohlcv_debug.bar_invalid_reason;

        // The key invariant: no bar should be hard-rejected for zero/negative price
        for (i, &reason) in reasons.iter().enumerate() {
            assert!(
                reason == 0.0,
                "Bar {} has bar_invalid_reason={}, expected 0 (no hard-reject for negative prices)",
                i, reason,
            );
        }
    }

    #[test]
    fn test_ohlcv_close_outlier_detected() {
        // 50 normal bars, then 1 bar with a 10x close spike, then normal again.
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64, f64, f64, f64)> = Vec::new();
        // 50 normal bars at close ~100
        for i in 0..50 {
            let c = 100.0 + 0.01 * i as f64;
            data.push((base + i * one_sec, c - 0.1, c + 0.2, c - 0.2, c, 10.0));
        }
        // Single spike: close at 1000
        data.push((base + 50 * one_sec, 999.0, 1001.0, 998.0, 1000.0, 10.0));
        // Return to normal
        for i in 51..60 {
            let c = 100.0 + 0.01 * i as f64;
            data.push((base + i * one_sec, c - 0.1, c + 0.2, c - 0.2, c, 10.0));
        }

        let results = run_ohlcv_detector(&data, OutlierConfig::default());

        assert!(
            !results[50],
            "10x close price spike should be rejected"
        );
        assert!(
            results[10..50].iter().all(|&b| b),
            "Normal bars before spike should be accepted"
        );
        assert!(
            results[51..].iter().all(|&b| b),
            "Normal bars after spike should be accepted"
        );
    }

    #[test]
    fn test_ohlcv_regime_shift() {
        // 100 bars at close=100, then jump to close=200.
        // Regime switch should accept by the 3rd point.
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64, f64, f64, f64)> = Vec::new();
        for i in 0..100 {
            data.push((base + i * one_sec, 99.5, 100.5, 99.0, 100.0, 10.0));
        }
        for i in 0..10 {
            let c = 200.0 + 0.1 * i as f64;
            data.push((base + (100 + i) * one_sec, c - 0.5, c + 0.5, c - 1.0, c, 10.0));
        }

        let mut config = OutlierConfig::default();
        config.use_rolling_zscore = false;
        config.use_iqr = false;
        config.use_volume_check = false;
        config.use_time_gap_check = false;
        config.use_price_jump_check = false;
        config.min_votes_to_reject = 1;

        let results = run_ohlcv_detector(&data, config);

        assert!(
            results[..100].iter().all(|&b| b),
            "All normal bars should be accepted"
        );

        let accepted_at_200: Vec<usize> = results[100..]
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect();

        assert!(
            !accepted_at_200.is_empty(),
            "Regime shift should eventually accept the new level"
        );
        assert!(
            accepted_at_200[0] <= 3,
            "First accepted at new level should be by the 3rd bar, was at index {}",
            accepted_at_200[0]
        );
    }

    #[test]
    fn test_ohlcv_vote_threshold() {
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut data: Vec<(i64, f64, f64, f64, f64, f64)> = Vec::new();
        for i in 0..20 {
            data.push((base + i * one_sec, 99.5, 100.5, 99.0, 100.0, 10.0));
        }
        // Moderately deviant close
        data.push((base + 20 * one_sec, 129.5, 130.5, 129.0, 130.0, 10.0));

        // Only adaptive level filter → max 1 vote. min_votes=2 → accepted.
        let mut config_2 = OutlierConfig::default();
        config_2.use_rolling_zscore = false;
        config_2.use_iqr = false;
        config_2.use_volume_check = false;
        config_2.use_time_gap_check = false;
        config_2.use_price_jump_check = false;
        config_2.min_votes_to_reject = 2;
        config_2.warmup_threshold = 0;

        let results_2 = run_ohlcv_detector(&data, config_2);
        assert!(
            results_2[20],
            "With min_votes=2 and only 1 method active, deviant bar should be accepted"
        );

        // Same but min_votes=1 → rejected.
        let mut config_1 = OutlierConfig::default();
        config_1.use_rolling_zscore = false;
        config_1.use_iqr = false;
        config_1.use_volume_check = false;
        config_1.use_time_gap_check = false;
        config_1.use_price_jump_check = false;
        config_1.min_votes_to_reject = 1;
        config_1.warmup_threshold = 0;

        let results_1 = run_ohlcv_detector(&data, config_1);
        assert!(
            !results_1[20],
            "With min_votes=1 and adaptive filter rejecting, deviant bar should be rejected"
        );
    }

    #[test]
    fn test_ohlcv_debug_alignment() {
        let one_sec: i64 = 1_000_000_000;
        let base: i64 = 1_000_000_000_000;

        let mut detector = OhlcvOutlierDetector::new(OutlierConfig::default());
        detector.enable_debug(25);

        // 20 normal bars + 2 invalid bars + 3 normal bars
        for i in 0..20 {
            let c = 100.0 + 0.01 * i as f64;
            let row = OhlcvRowData {
                ts: base + i * one_sec,
                open: c - 0.1, high: c + 0.2, low: c - 0.2, close: c, volume: 10.0,
            };
            detector.detect_bar(&row);
        }
        // Invalid bar: high < low
        let row = OhlcvRowData {
            ts: base + 20 * one_sec,
            open: 100.0, high: 98.0, low: 102.0, close: 100.0, volume: 10.0,
        };
        detector.detect_bar(&row);
        // Invalid bar: NaN
        let row = OhlcvRowData {
            ts: base + 21 * one_sec,
            open: f64::NAN, high: 101.0, low: 99.0, close: 100.0, volume: 10.0,
        };
        detector.detect_bar(&row);
        // 3 more normal bars
        for i in 22..25 {
            let c = 100.0 + 0.01 * i as f64;
            let row = OhlcvRowData {
                ts: base + i * one_sec,
                open: c - 0.1, high: c + 0.2, low: c - 0.2, close: c, volume: 10.0,
            };
            detector.detect_bar(&row);
        }

        let (inner_db, ohlcv_db) = detector.take_debug_buffers()
            .expect("debug buffers were enabled");

        let n = 25;
        // Inner detector debug: 15 columns all length n
        assert_eq!(inner_db.level.len(), n, "inner level length");
        assert_eq!(inner_db.scale.len(), n, "inner scale length");
        assert_eq!(inner_db.u.len(), n, "inner u length");
        assert_eq!(inner_db.tau.len(), n, "inner tau length");
        assert_eq!(inner_db.variance_rate.len(), n, "inner variance_rate length");
        assert_eq!(inner_db.zscore.len(), n, "inner zscore length");
        assert_eq!(inner_db.iqr_lower.len(), n, "inner iqr_lower length");
        assert_eq!(inner_db.iqr_upper.len(), n, "inner iqr_upper length");
        assert_eq!(inner_db.price_change_pct.len(), n, "inner price_change_pct length");
        assert_eq!(inner_db.time_gap_seconds.len(), n, "inner time_gap_seconds length");
        assert_eq!(inner_db.volume_change.len(), n, "inner volume_change length");
        assert_eq!(inner_db.votes_against.len(), n, "inner votes_against length");
        assert_eq!(inner_db.cand_level.len(), n, "inner cand_level length");
        assert_eq!(inner_db.cand_n_confirmed.len(), n, "inner cand_n_confirmed length");
        assert_eq!(inner_db.regime_switch.len(), n, "inner regime_switch length");

        // OHLCV debug: 6 columns all length n
        assert_eq!(ohlcv_db.bar_open.len(), n, "bar_open length");
        assert_eq!(ohlcv_db.bar_high.len(), n, "bar_high length");
        assert_eq!(ohlcv_db.bar_low.len(), n, "bar_low length");
        assert_eq!(ohlcv_db.bar_close.len(), n, "bar_close length");
        assert_eq!(ohlcv_db.bar_volume.len(), n, "bar_volume length");
        assert_eq!(ohlcv_db.bar_invalid_reason.len(), n, "bar_invalid_reason length");

        // Invalid bars should have appropriate reason codes
        assert_eq!(ohlcv_db.bar_invalid_reason[20], 2.0, "high<low should have reason=2");
        assert_eq!(ohlcv_db.bar_invalid_reason[21], 1.0, "NaN should have reason=1");
        // Valid bars should have reason=0
        assert_eq!(ohlcv_db.bar_invalid_reason[0], 0.0, "valid bar should have reason=0");
    }
}
