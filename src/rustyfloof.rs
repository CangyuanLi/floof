use crate::comparer::{fuzzycompare, fuzzycompare_sequential};
use crate::hamming as _hamming;
use crate::jaccard as _jaccard;
use crate::jaro as _jaro;
use crate::matcher::{fuzzymatch, fuzzymatch_sequential};
use crate::utils;
use phf::phf_map;
use pyo3::exceptions::*;
use pyo3::prelude::*;

static FUNC_MAPPER: phf::Map<&str, fn(&str, &str) -> f64> = phf_map! {
    "hamming" => _hamming::hamming,
    "hamming_ascii" => _hamming::hamming_ascii,
    "jaccard" =>_jaccard::jaccard,
    "jaccard_ascii" => _jaccard::jaccard_ascii,
    "jaro" => _jaro::jaro,
    "jaro_ascii" => _jaro::jaro_ascii,
    "jaro_winkler" => _jaro::jaro_winkler,
    "jaro_winkler_ascii" => _jaro::jaro_winkler_ascii,
};

#[pyfunction]
#[pyo3(signature = (s1, s2, ascii_only=false))]
fn hamming(s1: &str, s2: &str, ascii_only: bool) -> PyResult<f64> {
    if ascii_only {
        Ok(_hamming::hamming_ascii(s1, s2))
    } else {
        Ok(_hamming::hamming(s1, s2))
    }
}

#[pyfunction]
#[pyo3(signature = (s1, s2, ascii_only=false))]
fn jaccard(s1: &str, s2: &str, ascii_only: bool) -> PyResult<f64> {
    if ascii_only {
        Ok(_jaccard::jaccard_ascii(s1, s2))
    } else {
        Ok(_jaccard::jaccard(s1, s2))
    }
}

#[pyfunction]
#[pyo3(signature = (s1, s2, ascii_only=false))]
fn jaro(s1: &str, s2: &str, ascii_only: bool) -> PyResult<f64> {
    if ascii_only {
        Ok(_jaro::jaro_ascii(s1, s2))
    } else {
        Ok(_jaro::jaro(s1, s2))
    }
}

#[pyfunction]
#[pyo3(signature = (s1, s2, ascii_only=false))]
fn jaro_winkler(s1: &str, s2: &str, ascii_only: bool) -> PyResult<f64> {
    if ascii_only {
        Ok(_jaro::jaro_winkler_ascii(s1, s2))
    } else {
        Ok(_jaro::jaro_winkler(s1, s2))
    }
}

#[pyfunction]
#[pyo3(signature = (arr1, arr2, func_name, n_jobs=0))]
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

#[pyfunction]
#[pyo3(signature = (arr1, arr2, func_name, k_matches=5, threshold=0.0, n_jobs=0, quiet=false))]
pub fn _match(
    arr1: Vec<&str>,
    arr2: Vec<&str>,
    func_name: &str,
    k_matches: usize,
    threshold: f64,
    n_jobs: usize,
    quiet: bool,
) -> PyResult<Vec<utils::ScoreTuple>> {
    let func = FUNC_MAPPER.get(func_name);
    let err = Err(PyKeyError::new_err(func_name.to_string()));

    if n_jobs == 0 {
        match func {
            None => err,
            Some(f) => Ok(fuzzymatch(&arr1, &arr2, *f, k_matches, threshold, quiet)),
        }
    } else if n_jobs == 1 {
        match func {
            None => err,
            Some(f) => Ok(fuzzymatch_sequential(
                &arr1, &arr2, *f, k_matches, threshold, quiet,
            )),
        }
    } else {
        match func {
            None => err,
            Some(f) => Ok(utils::create_rayon_pool(n_jobs)?
                .install(|| fuzzymatch(&arr1, &arr2, *f, k_matches, threshold, quiet))),
        }
    }
}

#[pymodule]
pub fn _rustyfloof(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hamming, m)?)?;
    m.add_function(wrap_pyfunction!(jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(jaro, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler, m)?)?;
    m.add_function(wrap_pyfunction!(_compare, m)?)?;
    m.add_function(wrap_pyfunction!(_match, m)?)?;

    Ok(())
}
