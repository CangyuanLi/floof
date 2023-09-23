mod comparer;
mod hamming;
mod jaccard;
mod jaro;
mod levenshtein;
mod matcher;
mod metaphone;
mod nysiis;
mod phonetic;
mod utils;

mod rustyfloof;

#[cfg(feature = "python")]
pub use rustyfloof::_rustyfloof;
