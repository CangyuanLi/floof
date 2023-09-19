use crate::utils;
use unicode_segmentation::UnicodeSegmentation;

pub fn levenshtein_similarity<T: PartialEq>(slice1: &[T], slice2: &[T]) -> f64 {
    let len2 = slice2.len();
    let mut cache: utils::FastVec<usize> = (1..len2 + 1).collect();
    let mut levenshtein_distance = 0;
    let mut prev;

    for (i, x) in slice1.iter().enumerate() {
        levenshtein_distance += 1;
        prev = i;

        for (j, y) in slice2.iter().enumerate() {
            let cost = if x == y { 0usize } else { 1usize };
            let distance = prev + cost;
            prev = cache[j];
            levenshtein_distance =
                std::cmp::min(levenshtein_distance + 1, std::cmp::min(distance, prev + 1));
            cache[j] = levenshtein_distance;
        }
    }

    utils::distance_to_similarity(levenshtein_distance, slice1.len(), len2)
}

pub fn levenshtein(s1: &str, s2: &str) -> f64 {
    let us1: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s1, true).collect();
    let us2: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s2, true).collect();

    levenshtein_similarity(&us1, &us2)
}

pub fn levenshtein_ascii(s1: &str, s2: &str) -> f64 {
    levenshtein_similarity(s1.as_bytes(), s2.as_bytes())
}
