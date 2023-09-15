use crate::comparer::{fuzzycompare, fuzzycompare_sequential};
use crate::hamming as _hamming;
use crate::jaccard as _jaccard;
use crate::utils;
use phf::phf_map;
use pyo3::exceptions::*;
use pyo3::prelude::*;

static FUNC_MAPPER: phf::Map<&str, fn(&str, &str) -> f64> = phf_map! {
    "hamming" => _hamming::hamming,
    "hamming_ascii" => _hamming::hamming_ascii,
    "jaccard" =>_jaccard::jaccard,
    "jaccard_ascii" => _jaccard::jaccard_ascii,
    // "levenshtein" => fn(),
    // "levenshtein_ascii" => fn(),
    // "damerau_levenshtein" => fn(),
    // "damerau_levenshtein_ascii" => fn(),
    // "jaro" => fn(),
    // "jaro_ascii" => fn(),
    // "jarowinkler" => fn(),
    // "jarowinkler_ascii" => fn(),
};

#[pyfunction]
fn hamming(s1: &str, s2: &str) -> PyResult<f64> {
    Ok(_hamming::hamming(s1, s2))
}

#[pyfunction]
fn hamming_ascii(s1: &str, s2: &str) -> PyResult<f64> {
    Ok(_hamming::hamming_ascii(s1, s2))
}

#[pyfunction]
fn _compare(
    arr1: Vec<&str>,
    arr2: Vec<&str>,
    func_name: &str,
    n_jobs: usize,
) -> PyResult<Vec<f64>> {
    let func = FUNC_MAPPER.get(func_name);
    let err = Err(PyKeyError::new_err(func_name.to_string()));

    if n_jobs == 0 {
        match func {
            None => err,
            Some(f) => Ok(fuzzycompare(&arr1, &arr2, *f)),
        }
    } else if n_jobs == 1 {
        match func {
            None => err,
            Some(f) => Ok(fuzzycompare_sequential(&arr1, &arr2, *f)),
        }
    } else {
        match func {
            None => err,
            Some(f) => {
                Ok(utils::create_rayon_pool(n_jobs)?.install(|| fuzzycompare(&arr1, &arr2, *f)))
            }
        }
    }
}
