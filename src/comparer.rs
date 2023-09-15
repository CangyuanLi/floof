use rayon::prelude::*;

use crate::utils;

pub fn fuzzycompare(arr1: &[&str], arr2: &[&str], func: utils::SimilarityFunc) -> Vec<f64> {
    assert_eq!(arr1.len(), arr2.len());

    let res: Vec<f64> = arr1
        .into_par_iter()
        .zip(arr2)
        .map(|(s1, s2)| func(s1, s2))
        .collect();

    res
}

pub fn fuzzycompare_sequential(
    arr1: &[&str],
    arr2: &[&str],
    func: utils::SimilarityFunc,
) -> Vec<f64> {
    assert_eq!(arr1.len(), arr2.len());

    let res: Vec<f64> = arr1.iter().zip(arr2).map(|(s1, s2)| func(s1, s2)).collect();

    res
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hamming::{hamming, hamming_ascii};

    #[test]
    fn test_compare() {
        let arr1 = ["abc", "def", "a;dlkfj", "asldkj;f", "ab"];
        let arr2 = ["abc", "a;sdklfj", "weuifh", "cjfkj", "abdef"];
        dbg!(fuzzycompare(&arr1, &arr2, hamming_ascii));
        dbg!(fuzzycompare(&arr1, &arr2, hamming));
        dbg!(fuzzycompare_sequential(&arr1, &arr2, hamming));
    }
}
