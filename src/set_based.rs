use crate::utils;
use counter::Counter;
use std::hash::Hash;
use unicode_segmentation::UnicodeSegmentation;

// Jaccard

pub fn jaccard_similarity<T: Eq + Hash>(slice1: &[T], slice2: &[T]) -> f64 {
    let set1: ahash::HashSet<_> = slice1.iter().collect();
    let set2: ahash::HashSet<_> = slice2.iter().collect();

    let intersection_size = set1.intersection(&set2).count();
    let union_size = set1.len() + set2.len() - intersection_size;

    if union_size == 0 {
        return 0.0;
    }

    intersection_size as f64 / union_size as f64
}

pub fn jaccard_ascii(s1: &str, s2: &str) -> f64 {
    jaccard_similarity(s1.as_bytes(), s2.as_bytes())
}

pub fn jaccard(s1: &str, s2: &str) -> f64 {
    let us1: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s1, true).collect();
    let us2: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s2, true).collect();

    jaccard_similarity(&us1, &us2)
}

// Sorensen-Dice

pub fn sorensen_dice_similarity<T: Eq + Hash>(slice1: &[T], slice2: &[T]) -> f64 {
    let jaccard_sim = jaccard_similarity(slice1, slice2);

    (2.0 * jaccard_sim) / (1.0 + jaccard_sim)
}

pub fn sorensen_dice_ascii(s1: &str, s2: &str) -> f64 {
    sorensen_dice_similarity(s1.as_bytes(), s2.as_bytes())
}

pub fn sorensen_dice(s1: &str, s2: &str) -> f64 {
    let us1: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s1, true).collect();
    let us2: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s2, true).collect();

    sorensen_dice_similarity(&us1, &us2)
}

// Cosine distance (Otsuka-Ochiai)

pub fn cosine_similarity<T: Eq + Hash>(slice1: &[T], slice2: &[T]) -> f64 {
    let set1: ahash::HashSet<_> = slice1.iter().collect();
    let set2: ahash::HashSet<_> = slice2.iter().collect();

    let intersection_size = set1.intersection(&set2).count();
    let denom = ((set1.len() * set2.len()) as f64).sqrt();

    intersection_size as f64 / denom
}

pub fn cosine(s1: &str, s2: &str) -> f64 {
    let set1: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s1, true).collect();
    let set2: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s2, true).collect();

    cosine_similarity(&set1, &set2)
}

pub fn cosine_ascii(s1: &str, s2: &str) -> f64 {
    cosine_similarity(s1.as_bytes(), s2.as_bytes())
}

// Bag

pub fn bag_similarity<T: Eq + Hash>(slice1: &[T], slice2: &[T]) -> f64 {
    let len1 = slice1.len();
    let len2 = slice2.len();

    let bag1: Counter<&T> = slice1.iter().collect();
    let bag2: Counter<&T> = slice2.iter().collect();

    let size1: usize = (bag1.clone() - bag2.clone()).values().sum();
    let size2: usize = (bag2 - bag1).values().sum();

    let max_size = std::cmp::max(size1, size2);

    utils::distance_to_similarity(max_size, len1, len2)
}

pub fn bag(s1: &str, s2: &str) -> f64 {
    let us1: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s1, true).collect();
    let us2: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s2, true).collect();

    bag_similarity(&us1, &us2)
}

pub fn bag_ascii(s1: &str, s2: &str) -> f64 {
    bag_similarity(s1.as_bytes(), s2.as_bytes())
}

// Overlap coefficient

pub fn overlap_similarity<T: Eq + Hash>(slice1: &[T], slice2: &[T]) -> f64 {
    let set1: ahash::AHashSet<_> = slice1.iter().collect();
    let set2: ahash::AHashSet<_> = slice2.iter().collect();

    let intersection_size = set1.intersection(&set2).count();

    intersection_size as f64 / std::cmp::min(set1.len(), set2.len()) as f64
}

pub fn overlap(s1: &str, s2: &str) -> f64 {
    let us1: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s1, true).collect();
    let us2: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s2, true).collect();

    overlap_similarity(&us1, &us2)
}

pub fn overlap_ascii(s1: &str, s2: &str) -> f64 {
    overlap_similarity(s1.as_bytes(), s2.as_bytes())
}

// Tverskey Index

pub fn tversky_similarity<T: Eq + Hash>(slice1: &[T], slice2: &[T]) -> f64 {
    let set1: ahash::AHashSet<_> = slice1.iter().collect();
    let set2: ahash::AHashSet<_> = slice2.iter().collect();

    let intersection_size = set1.intersection(&set2).count() as f64;
    let set1m2_size = set1.difference(&set2).count() as f64;
    let set2m1_size = set2.difference(&set1).count() as f64;

    intersection_size / (intersection_size + 0.5 * set1m2_size + 0.5 * set2m1_size)
}

pub fn tversky(s1: &str, s2: &str) -> f64 {
    let us1: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s1, true).collect();
    let us2: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s2, true).collect();

    tversky_similarity(&us1, &us2)
}

pub fn tversky_ascii(s1: &str, s2: &str) -> f64 {
    tversky_similarity(s1.as_bytes(), s2.as_bytes())
}
