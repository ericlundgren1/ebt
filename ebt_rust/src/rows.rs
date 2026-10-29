//! Row types and per-row processor traits shared by the streaming detectors.
//!
//! A processor sees one row at a time, in order, and keeps its own state. Rows
//! with a missing value get `null_value()` and an `on_null()` call instead, so
//! output and debug buffers stay aligned with the input.

/// Trade row. Timestamps are in nanoseconds.
pub struct RowData {
    pub ts: i64,
    pub price: f64,
    pub volume: f64,
}

/// Row-by-row processor for trades.
pub trait RowProcessor {
    type Output: Copy;

    /// Process a single row and return the computed value.
    /// Called once per non-null row in sequence.
    fn process_row(&mut self, row: &RowData) -> Self::Output;

    /// Value to emit for null rows (any column is null).
    fn null_value() -> Self::Output;

    /// Called when a null row is encountered, after `null_value()` has been pushed.
    /// Override to keep sidecar buffers aligned with the output vector.
    fn on_null(&mut self) {}
}

/// Quote row. Timestamps are in nanoseconds.
pub struct QuoteRowData {
    pub ts: i64,
    pub bid_price: f64,
    pub ask_price: f64,
}

/// Row-by-row processor for quotes.
pub trait QuoteRowProcessor {
    type Output: Copy;

    fn process_row(&mut self, row: &QuoteRowData) -> Self::Output;
    fn null_value() -> Self::Output;
    fn on_null(&mut self) {}
}

/// OHLCV bar row. Timestamps are in nanoseconds.
pub struct OhlcvRowData {
    pub ts: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Row-by-row processor for OHLCV bars.
pub trait OhlcvRowProcessor {
    type Output: Copy;

    fn process_row(&mut self, row: &OhlcvRowData) -> Self::Output;
    fn null_value() -> Self::Output;
    fn on_null(&mut self) {}
}
