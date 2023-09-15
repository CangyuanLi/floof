use crate::utils;
use min_max_heap::MinMaxHeap;
use rayon::prelude::*;

#[derive(Debug)]
pub struct Score(f64, String, String);

impl Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.0).total_cmp(&other.0)
    }
}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Score {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Score {}

fn get_matches(
    s1: &str,
    arr2: &[&str],
    func: utils::SimilarityFunc,
    k_matches: usize,
) -> Vec<Score> {
    let mut heap = MinMaxHeap::with_capacity(k_matches);
    for s2 in arr2 {
        let score = Score(func(s1, s2), s1.to_string(), s2.to_string());

        if heap.len() < k_matches {
            heap.push(score);
        } else if &score > heap.peek_min().unwrap() {
            heap.push_pop_min(score);
        }
    }

    let mut res: Vec<Score> = Vec::with_capacity(heap.len());
    for elem in heap {
        res.push(elem);
    }

    res
}

pub fn fuzzymatch(
    arr1: &[&str],
    arr2: &[&str],
    func: utils::SimilarityFunc,
    k_matches: usize,
) -> Vec<Score> {
    let res: Vec<Score> = arr1
        .par_iter()
        .flat_map(|s1| get_matches(s1, arr2, func, k_matches))
        .collect();

    res
}

pub fn fuzzymatch_sequential(
    arr1: &[&str],
    arr2: &[&str],
    func: utils::SimilarityFunc,
    k_matches: usize,
) -> Vec<Score> {
    let res: Vec<Score> = arr1
        .iter()
        .flat_map(|s1| get_matches(s1, arr2, func, k_matches))
        .collect();

    res
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hamming::{hamming, hamming_ascii};

    #[test]
    fn test_match() {
        let arr1 = ["abc", "def", "a;dlkfj", "asldkj;f", "ab"];
        let arr2 = ["abc", "a;sdklfj", "weuifh", "cjfkj", "abdef"];
        let k_matches = 2;
        dbg!(fuzzymatch(&arr1, &arr2, hamming_ascii, k_matches));
        dbg!(fuzzymatch(&arr1, &arr2, hamming, k_matches));
    }
}
