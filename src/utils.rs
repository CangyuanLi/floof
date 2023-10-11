use indicatif::ProgressStyle;
use pyo3::exceptions::PyException;
use pyo3::pyclass;
use smallvec::SmallVec;
use thiserror::Error;

/// A custom Error for this crate, mainly so we can convert easily from a Rust error to
/// a Python expection.
#[derive(Error, Debug)]
pub enum FloofError {
    #[error(transparent)]
    ThreadPoolError(#[from] rayon::ThreadPoolBuildError),
}

impl std::convert::From<FloofError> for pyo3::PyErr {
    fn from(err: FloofError) -> pyo3::PyErr {
        match err {
            FloofError::ThreadPoolError(_) => PyException::new_err(err.to_string()),
        }
    }
}

const VEC_SIZE: usize = 32;

pub type FastVec<T> = SmallVec<[T; VEC_SIZE]>;
pub type SimilarityFunc = fn(&str, &str) -> f64;
pub type SimilarityFuncSlice<T> = fn(&[T], &[T]) -> f64;

pub trait HasLength {
    fn len(&self) -> usize;
}

impl<T> HasLength for FastVec<T> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> HasLength for &[T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

#[pyclass]
#[derive(Debug)]
pub struct Score {
    pub similarity: f64,
    pub str1: String,
    pub str2: String,
}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.similarity).total_cmp(&other.similarity)
    }
}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Score {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

impl Eq for Score {}

pub type ScoreTuple = (f64, String, String);

impl From<Score> for ScoreTuple {
    fn from(e: Score) -> ScoreTuple {
        let Score {
            similarity,
            str1,
            str2,
        } = e;
        (similarity, str1, str2)
    }
}

const PROGRESS_TEMPLATE: &str =
    "{percent}%|{wide_bar}| {human_pos}/{human_len} [{elapsed_precise}<{eta_precise}, {per_sec}]";

pub fn create_progress_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(PROGRESS_TEMPLATE)
        .unwrap()
}

/// Returns a f64 that is the distance scaled by the max length of the two strings.
/// The formula is (max_len - dist) / max_len.
///
/// # Arguments
///
/// * `distance` - The edit distance between two strings, e.g. a Hamming distance
/// * `len1` - The length of the first string
/// * `len2` - The length of the second string
#[inline]
pub fn distance_to_similarity(distance: usize, len1: usize, len2: usize) -> f64 {
    let max_len = std::cmp::max(len1, len2);
    let max_len = max_len as f64;
    let dist = distance as f64;

    (max_len - dist) / max_len
}

/// Returns a Result where the error is a Rayon::ThreadPoolBuildError (converted to a
/// Floof::ThreadPoolError) and the success is a thread pool with the requested number
/// of threads.
///
/// # Arguments
///
/// * `num_threads` - Number of threads in the pool
pub fn create_rayon_pool(num_threads: usize) -> Result<rayon::ThreadPool, FloofError> {
    match rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
    {
        Err(e) => Err(e.into()),
        Ok(pool) => Ok(pool),
    }
}
