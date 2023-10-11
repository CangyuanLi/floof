use crate::utils;
use indicatif::{ParallelProgressIterator, ProgressIterator};
use min_max_heap::MinMaxHeap;
use rayon::prelude::*;

#[allow(clippy::redundant_closure)]
fn get_matches_slice<T: PartialEq>(
    s1: &(&str, &[T]),
    arr2: &[(&str, &[T])],
    func: utils::SimilarityFuncSlice<T>,
    k_matches: usize,
    threshold: f64,
) -> Vec<utils::ScoreTuple> {
    let mut heap = MinMaxHeap::with_capacity(k_matches);
    let s1_str = s1.0;
    let s1_repr = s1.1;
    for (s2, s2_repr) in arr2 {
        let score = utils::Score {
            similarity: func(s1_repr, s2_repr),
            str1: s1_str.to_string(),
            str2: s2.to_string(),
        };

        if score.similarity < threshold {
            continue;
        }

        if heap.len() < k_matches {
            heap.push(score);
        } else if &score > heap.peek_min().unwrap() {
            heap.push_pop_min(score);
        }
    }

    let res: Vec<utils::ScoreTuple> = heap
        .into_iter()
        .map(|elem| utils::ScoreTuple::from(elem))
        .collect();

    res
}

#[allow(clippy::redundant_closure)]
fn get_matches_slice_all<T: PartialEq>(
    s1: &(&str, &[T]),
    arr2: &[(&str, &[T])],
    func: utils::SimilarityFuncSlice<T>,
    threshold: f64,
) -> Vec<utils::ScoreTuple> {
    let mut res = Vec::with_capacity(arr2.len());
    let s1_str = s1.0;
    let s1_repr = s1.1;
    for (s2, s2_repr) in arr2 {
        let similarity = func(s1_repr, s2_repr);

        if similarity < threshold {
            continue;
        }

        let score = (similarity, s1_str.to_string(), s2.to_string());

        res.push(score);
    }

    res
}

pub fn fuzzymatch_slice_all<T: PartialEq + Sync>(
    arr1: &[(&str, &[T])],
    arr2: &[(&str, &[T])],
    func: utils::SimilarityFuncSlice<T>,
    threshold: f64,
    quiet: bool,
) -> Vec<utils::ScoreTuple> {
    let iter = arr1.par_iter();

    if quiet {
        let res: Vec<utils::ScoreTuple> = iter
            .flat_map(|s1| get_matches_slice_all(s1, arr2, func, threshold))
            .collect();

        return res;
    }

    let res: Vec<utils::ScoreTuple> = iter
        .progress_with_style(utils::create_progress_style())
        .flat_map(|s1| get_matches_slice_all(s1, arr2, func, threshold))
        .collect();

    res
}

pub fn fuzzymatch_slice_all_sequential<T: PartialEq>(
    arr1: &[(&str, &[T])],
    arr2: &[(&str, &[T])],
    func: utils::SimilarityFuncSlice<T>,
    threshold: f64,
    quiet: bool,
) -> Vec<utils::ScoreTuple> {
    let iter = arr1.iter();

    if quiet {
        let res: Vec<utils::ScoreTuple> = iter
            .flat_map(|s1| get_matches_slice_all(s1, arr2, func, threshold))
            .collect();

        return res;
    }

    let res: Vec<utils::ScoreTuple> = iter
        .progress_with_style(utils::create_progress_style())
        .flat_map(|s1| get_matches_slice_all(s1, arr2, func, threshold))
        .collect();

    res
}

pub fn fuzzymatch_slice<T: PartialEq + Sync>(
    arr1: &[(&str, &[T])],
    arr2: &[(&str, &[T])],
    func: utils::SimilarityFuncSlice<T>,
    k_matches: usize,
    threshold: f64,
    quiet: bool,
) -> Vec<utils::ScoreTuple> {
    let iter = arr1.par_iter();

    if quiet {
        let res: Vec<utils::ScoreTuple> = iter
            .flat_map(|s1| get_matches_slice(s1, arr2, func, k_matches, threshold))
            .collect();

        return res;
    }

    let res: Vec<utils::ScoreTuple> = iter
        .progress_with_style(utils::create_progress_style())
        .flat_map(|s1| get_matches_slice(s1, arr2, func, k_matches, threshold))
        .collect();

    res
}

pub fn fuzzymatch_slice_sequential<T: PartialEq + Sync>(
    arr1: &[(&str, &[T])],
    arr2: &[(&str, &[T])],
    func: utils::SimilarityFuncSlice<T>,
    k_matches: usize,
    threshold: f64,
    quiet: bool,
) -> Vec<utils::ScoreTuple> {
    let iter = arr1.iter();

    if quiet {
        let res: Vec<utils::ScoreTuple> = iter
            .flat_map(|s1| get_matches_slice(s1, arr2, func, k_matches, threshold))
            .collect();

        return res;
    }

    let res: Vec<utils::ScoreTuple> = iter
        .progress_with_style(utils::create_progress_style())
        .flat_map(|s1| get_matches_slice(s1, arr2, func, k_matches, threshold))
        .collect();

    res
}

fn get_matches(
    s1: &str,
    arr2: &[&str],
    func: utils::SimilarityFunc,
    k_matches: usize,
    threshold: f64,
) -> Vec<utils::ScoreTuple> {
    let mut heap = MinMaxHeap::with_capacity(k_matches);
    for s2 in arr2 {
        let score = utils::Score {
            similarity: func(s1, s2),
            str1: s1.to_string(),
            str2: s2.to_string(),
        };

        if score.similarity < threshold {
            continue;
        }

        if heap.len() < k_matches {
            heap.push(score);
        } else if &score > heap.peek_min().unwrap() {
            heap.push_pop_min(score);
        }
    }

    let mut res: Vec<utils::ScoreTuple> = Vec::with_capacity(heap.len());
    for elem in heap {
        res.push(<utils::ScoreTuple>::from(elem));
    }

    res
}

pub fn fuzzymatch(
    arr1: &[&str],
    arr2: &[&str],
    func: utils::SimilarityFunc,
    k_matches: usize,
    threshold: f64,
    quiet: bool,
) -> Vec<utils::ScoreTuple> {
    let iter = arr1.par_iter();

    if quiet {
        let res: Vec<utils::ScoreTuple> = iter
            .flat_map(|s1| get_matches(s1, arr2, func, k_matches, threshold))
            .collect();

        return res;
    }

    let res: Vec<utils::ScoreTuple> = iter
        .progress_with_style(utils::create_progress_style())
        .flat_map(|s1| get_matches(s1, arr2, func, k_matches, threshold))
        .collect();

    res
}

pub fn fuzzymatch_sequential(
    arr1: &[&str],
    arr2: &[&str],
    func: utils::SimilarityFunc,
    k_matches: usize,
    threshold: f64,
    quiet: bool,
) -> Vec<utils::ScoreTuple> {
    let iter = arr1.iter();

    if quiet {
        let res: Vec<utils::ScoreTuple> = iter
            .flat_map(|s1| get_matches(s1, arr2, func, k_matches, threshold))
            .collect();

        return res;
    }

    let res: Vec<utils::ScoreTuple> = iter
        .progress_with_style(utils::create_progress_style())
        .flat_map(|s1| get_matches(s1, arr2, func, k_matches, threshold))
        .collect();

    res
}
