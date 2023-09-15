use pyo3::exceptions::PyException;
use smallvec::SmallVec;
use thiserror::Error;

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

pub fn distance_to_similarity(distance: usize, len1: usize, len2: usize) -> f64 {
    let max_len = std::cmp::max(len1, len2);
    let max_len = max_len as f64;
    let dist = distance as f64;

    (max_len - dist) / max_len
}

pub fn create_rayon_pool(num_threads: usize) -> Result<rayon::ThreadPool, FloofError> {
    match rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
    {
        Err(e) => Err(e.into()),
        Ok(pool) => Ok(pool),
    }
}
