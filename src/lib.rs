mod comparer;
mod hamming;
mod jaro;
mod levenshtein;
mod matcher;
mod metaphone;
mod phonetic;
mod set_based;
mod utils;

mod rustyfloof;

#[cfg(feature = "python")]
pub use rustyfloof::_rustyfloof;
