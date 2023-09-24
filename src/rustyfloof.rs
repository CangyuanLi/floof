use crate::comparer::{fuzzycompare, fuzzycompare_sequential};
use crate::hamming as _hamming;
use crate::jaro as _jaro;
use crate::levenshtein as _levenshtein;
use crate::matcher::{
    fuzzymatch, fuzzymatch_sequential, fuzzymatch_slice, fuzzymatch_slice_sequential,
};
use crate::phonetic as _phonetic;
use crate::set_based as _set_based;
use crate::utils;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use unicode_segmentation::UnicodeSegmentation;

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
        Ok(_set_based::jaccard_ascii(s1, s2))
    } else {
        Ok(_set_based::jaccard(s1, s2))
    }
}

#[pyfunction]
#[pyo3(signature = (s1, s2, ascii_only=false))]
fn sorensen_dice(s1: &str, s2: &str, ascii_only: bool) -> PyResult<f64> {
    if ascii_only {
        Ok(_set_based::sorensen_dice_ascii(s1, s2))
    } else {
        Ok(_set_based::sorensen_dice(s1, s2))
    }
}

#[pyfunction]
#[pyo3(signature = (s1, s2, ascii_only=false))]
fn cosine(s1: &str, s2: &str, ascii_only: bool) -> PyResult<f64> {
    if ascii_only {
        Ok(_set_based::cosine_ascii(s1, s2))
    } else {
        Ok(_set_based::cosine(s1, s2))
    }
}

#[pyfunction]
#[pyo3(signature = (s1, s2, ascii_only=false))]
fn bag(s1: &str, s2: &str, ascii_only: bool) -> PyResult<f64> {
    if ascii_only {
        Ok(_set_based::bag_ascii(s1, s2))
    } else {
        Ok(_set_based::bag(s1, s2))
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
#[pyo3(signature = (s1, s2, ascii_only=false))]
fn levenshtein(s1: &str, s2: &str, ascii_only: bool) -> PyResult<f64> {
    if ascii_only {
        Ok(_levenshtein::levenshtein_ascii(s1, s2))
    } else {
        Ok(_levenshtein::levenshtein(s1, s2))
    }
}

#[pyfunction]
#[pyo3(signature = (s1, s2, ascii_only=false))]
fn damerau_levenshtein(s1: &str, s2: &str, ascii_only: bool) -> PyResult<f64> {
    if ascii_only {
        Ok(_levenshtein::damerau_levenshtein_ascii(s1, s2))
    } else {
        Ok(_levenshtein::damerau_levenshtein(s1, s2))
    }
}

#[pyfunction]
#[pyo3(signature = (s1, s2, ascii_only=false))]
fn osa(s1: &str, s2: &str, ascii_only: bool) -> PyResult<f64> {
    if ascii_only {
        Ok(_levenshtein::osa_ascii(s1, s2))
    } else {
        Ok(_levenshtein::osa(s1, s2))
    }
}

#[pyfunction]
#[pyo3(signature = (s1, s2, ascii_only=false))]
fn soundex(s1: &str, s2: &str, ascii_only: bool) -> PyResult<f64> {
    if ascii_only {
        Ok(_phonetic::soundex_ascii(s1, s2))
    } else {
        Ok(_phonetic::soundex(s1, s2))
    }
}

#[pyfunction]
fn _extract_graphemes(arr: Vec<&str>) -> Vec<(&str, Vec<&str>)> {
    arr.iter()
        .map(|s| {
            let s = *s;
            let us: Vec<&str> = UnicodeSegmentation::graphemes(s, true).collect();

            (s, us)
        })
        .collect()
}

#[pyfunction]
fn _extract_bytes(slice: Vec<&str>) -> Vec<(&str, &[u8])> {
    slice
        .iter()
        .map(|s| {
            let s = *s;
            let bytes = s.as_bytes();
            (s, bytes)
        })
        .collect()
}

#[allow(clippy::let_and_return)]
fn func_dispatcher(func_name: &str) -> utils::SimilarityFunc {
    let func = match func_name {
        "hamming" => _hamming::hamming,
        "hamming_ascii" => _hamming::hamming_ascii,
        "jaccard" => _set_based::jaccard,
        "jaccard_ascii" => _set_based::jaccard_ascii,
        "sorensen_dice" => _set_based::sorensen_dice,
        "sorensen_dice_ascii" => _set_based::sorensen_dice_ascii,
        "cosine" => _set_based::cosine,
        "cosine_ascii" => _set_based::cosine_ascii,
        "bag" => _set_based::bag,
        "bag_ascii" => _set_based::bag_ascii,
        "jaro" => _jaro::jaro,
        "jaro_ascii" => _jaro::jaro_ascii,
        "jaro_winkler" => _jaro::jaro_winkler,
        "jaro_winkler_ascii" => _jaro::jaro_winkler_ascii,
        "levenshtein" => _levenshtein::levenshtein,
        "levenshtein_ascii" => _levenshtein::levenshtein_ascii,
        "damerau_levenshtein" => _levenshtein::damerau_levenshtein,
        "damerau_levenshtein_ascii" => _levenshtein::damerau_levenshtein_ascii,
        "osa" => _levenshtein::osa,
        "osa_ascii" => _levenshtein::osa_ascii,
        "soundex" => _phonetic::soundex,
        "soundex_ascii" => _phonetic::soundex_ascii,
        _ => panic!("{func_name} is not a valid function"),
    };

    func
}

#[pyfunction]
#[pyo3(signature = (arr1, arr2, func_name, n_jobs=0))]
fn _compare(
    arr1: Vec<&str>,
    arr2: Vec<&str>,
    func_name: &str,
    n_jobs: usize,
) -> PyResult<Vec<f64>> {
    let func = func_dispatcher(func_name);

    let arr1 = arr1.as_slice();
    let arr2 = arr2.as_slice();

    if n_jobs == 0 {
        Ok(fuzzycompare(arr1, arr2, func))
    } else if n_jobs == 1 {
        Ok(fuzzycompare_sequential(arr1, arr2, func))
    } else {
        Ok(utils::create_rayon_pool(n_jobs)?.install(|| fuzzycompare(arr1, arr2, func)))
    }
}

#[pyfunction]
#[pyo3(signature = (arr1, arr2, func_name, k_matches=5, threshold=0.0, n_jobs=0, quiet=false))]
fn _match(
    arr1: Vec<&str>,
    arr2: Vec<&str>,
    func_name: &str,
    k_matches: usize,
    threshold: f64,
    n_jobs: usize,
    quiet: bool,
) -> PyResult<Vec<utils::ScoreTuple>> {
    let func = func_dispatcher(func_name);
    let arr1 = arr1.as_slice();
    let arr2 = arr2.as_slice();

    if n_jobs == 0 {
        Ok(fuzzymatch(arr1, arr2, func, k_matches, threshold, quiet))
    } else if n_jobs == 1 {
        Ok(fuzzymatch_sequential(
            arr1, arr2, func, k_matches, threshold, quiet,
        ))
    } else {
        Ok(utils::create_rayon_pool(n_jobs)?
            .install(|| fuzzymatch(arr1, arr2, func, k_matches, threshold, quiet)))
    }
}

fn match_slice_core<T: PartialEq + Sync + Eq + std::hash::Hash>(
    slice1: &[(&str, &[T])],
    slice2: &[(&str, &[T])],
    func: fn(&[T], &[T]) -> f64,
    k_matches: usize,
    threshold: f64,
    n_jobs: usize,
    quiet: bool,
) -> PyResult<Vec<utils::ScoreTuple>> {
    if n_jobs == 0 {
        Ok(fuzzymatch_slice(
            slice1, slice2, func, k_matches, threshold, quiet,
        ))
    } else if n_jobs == 1 {
        Ok(fuzzymatch_slice_sequential(
            slice1, slice2, func, k_matches, threshold, quiet,
        ))
    } else {
        Ok(utils::create_rayon_pool(n_jobs)?
            .install(|| fuzzymatch_slice(slice1, slice2, func, k_matches, threshold, quiet)))
    }
}

#[pyfunction]
#[pyo3(signature = (processed_arr1, processed_arr2, func_name, k_matches=5, threshold=0.0, n_jobs=0, quiet=false))]
fn _match_slice(
    processed_arr1: Vec<(&str, Vec<&str>)>,
    processed_arr2: Vec<(&str, Vec<&str>)>,
    func_name: &str,
    k_matches: usize,
    threshold: f64,
    n_jobs: usize,
    quiet: bool,
) -> PyResult<Vec<utils::ScoreTuple>> {
    let func = match func_name {
        "hamming_similarity" => _hamming::hamming_similarity,
        "jaro_similarity" => _jaro::jaro_similarity,
        "jaro_winkler_similarity" => _jaro::jaro_winkler_similarity,
        "levenshtein_similarity" => _levenshtein::levenshtein_similarity,
        "damerau_levenshtein_similarity" => _levenshtein::damerau_levenshtein_similarity,
        "osa_similarity" => _levenshtein::osa_similarity,
        "bag_similarity" => _set_based::bag_similarity,
        _ => return Err(PyKeyError::new_err(func_name.to_string())),
    };

    let processed_arr1: Vec<(&str, &[&str])> = processed_arr1
        .iter()
        .map(|(x, y)| (*x, y.as_slice()))
        .collect();
    let processed_arr1 = processed_arr1.as_slice();

    let processed_arr2: Vec<(&str, &[&str])> = processed_arr2
        .iter()
        .map(|(x, y)| (*x, y.as_slice()))
        .collect();
    let processed_arr2 = processed_arr2.as_slice();

    match_slice_core(
        processed_arr1,
        processed_arr2,
        func,
        k_matches,
        threshold,
        n_jobs,
        quiet,
    )
}

#[pyfunction]
#[pyo3(signature = (processed_arr1, processed_arr2, func_name, k_matches=5, threshold=0.0, n_jobs=0, quiet=false))]
fn _match_slice_ascii(
    processed_arr1: Vec<(&str, &[u8])>,
    processed_arr2: Vec<(&str, &[u8])>,
    func_name: &str,
    k_matches: usize,
    threshold: f64,
    n_jobs: usize,
    quiet: bool,
) -> PyResult<Vec<utils::ScoreTuple>> {
    let func = match func_name {
        "hamming_similarity" => _hamming::hamming_similarity,
        "jaro_similarity" => _jaro::jaro_similarity,
        "jaro_winkler_similarity" => _jaro::jaro_winkler_similarity,
        "levenshtein_similarity" => _levenshtein::levenshtein_similarity,
        "damerau_levenshtein_similarity" => _levenshtein::damerau_levenshtein_similarity,
        "damerau_levenshtein_similarity_ascii" => {
            _levenshtein::damerau_levenshtein_similarity_ascii
        }
        "osa_similarity" => _levenshtein::osa_similarity,
        "soundex_similarity" => _phonetic::soundex_similarity,
        "bag_similarity" => _set_based::bag_similarity,
        _ => return Err(PyKeyError::new_err(func_name.to_string())),
    };

    let processed_arr1 = processed_arr1.as_slice();
    let processed_arr2 = processed_arr2.as_slice();

    match_slice_core(
        processed_arr1,
        processed_arr2,
        func,
        k_matches,
        threshold,
        n_jobs,
        quiet,
    )
}

#[pymodule]
pub fn _rustyfloof(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hamming, m)?)?;
    m.add_function(wrap_pyfunction!(jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(sorensen_dice, m)?)?;
    m.add_function(wrap_pyfunction!(cosine, m)?)?;
    m.add_function(wrap_pyfunction!(bag, m)?)?;
    m.add_function(wrap_pyfunction!(jaro, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(osa, m)?)?;
    m.add_function(wrap_pyfunction!(soundex, m)?)?;
    m.add_function(wrap_pyfunction!(_extract_graphemes, m)?)?;
    m.add_function(wrap_pyfunction!(_extract_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(_compare, m)?)?;
    m.add_function(wrap_pyfunction!(_match, m)?)?;
    m.add_function(wrap_pyfunction!(_match_slice, m)?)?;
    m.add_function(wrap_pyfunction!(_match_slice_ascii, m)?)?;

    Ok(())
}
