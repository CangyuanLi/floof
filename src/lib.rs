mod comparer;
mod hamming;
mod jaccard;
mod jaro;
mod levenshtein;
mod matcher;
mod metaphone;
mod soundex;
mod utils;

mod rustyfloof;

#[cfg(feature = "python")]
pub use rustyfloof::_rustyfloof;
