mod comparer;
mod hamming;
mod jaccard;
mod jaro;
mod matcher;
mod utils;

mod rustyfloof;

#[cfg(feature = "python")]
pub use rustyfloof::_rustyfloof;
