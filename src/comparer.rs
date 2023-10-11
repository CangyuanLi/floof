use rayon::prelude::*;

use crate::utils;
use indicatif::{ParallelProgressIterator, ProgressIterator};

pub fn fuzzycompare_slice<T: PartialEq + Sync>(
    slice1: &[&[T]],
    slice2: &[&[T]],
    func: utils::SimilarityFuncSlice<T>,
    quiet: bool,
) -> Vec<f64> {
    let iter = slice1.par_iter().zip(slice2).map(|(x, y)| func(x, y));

    if quiet {
        return iter.collect::<Vec<f64>>();
    }

    let res: Vec<f64> = iter
        .progress_with_style(utils::create_progress_style())
        .collect();

    res
}

pub fn fuzzycompare_slice_sequential<T: PartialEq>(
    slice1: &[&[T]],
    slice2: &[&[T]],
    func: utils::SimilarityFuncSlice<T>,
    quiet: bool,
) -> Vec<f64> {
    let iter = slice1.iter().zip(slice2).map(|(x, y)| func(x, y));

    if quiet {
        return iter.collect::<Vec<f64>>();
    }

    let res: Vec<f64> = iter
        .progress_with_style(utils::create_progress_style())
        .collect();

    res
}

pub fn fuzzycompare(
    arr1: &[&str],
    arr2: &[&str],
    func: utils::SimilarityFunc,
    quiet: bool,
) -> Vec<f64> {
    let iter = arr1.par_iter().zip(arr2).map(|(s1, s2)| func(s1, s2));

    if quiet {
        return iter.collect::<Vec<f64>>();
    }

    let res: Vec<f64> = iter
        .progress_with_style(utils::create_progress_style())
        .collect();

    res
}

pub fn fuzzycompare_sequential(
    arr1: &[&str],
    arr2: &[&str],
    func: utils::SimilarityFunc,
    quiet: bool,
) -> Vec<f64> {
    let iter = arr1.iter().zip(arr2).map(|(s1, s2)| func(s1, s2));

    if quiet {
        return iter.collect::<Vec<f64>>();
    }

    let res: Vec<f64> = iter
        .progress_with_style(utils::create_progress_style())
        .collect();

    res
}
