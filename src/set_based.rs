use std::hash::Hash;
use unicode_segmentation::UnicodeSegmentation;

// Jaccard

fn jaccard_similarity<T: PartialEq + Hash + Eq>(
    set1: &ahash::HashSet<T>,
    set2: &ahash::HashSet<T>,
) -> f64 {
    let intersection_size = set1.intersection(set2).count();
    let union_size = set1.len() + set2.len() - intersection_size;

    if union_size == 0 {
        return 0.0;
    }

    intersection_size as f64 / union_size as f64
}

pub fn jaccard_ascii(s1: &str, s2: &str) -> f64 {
    let set1: ahash::HashSet<u8> = s1.bytes().collect();
    let set2: ahash::HashSet<u8> = s2.bytes().collect();

    jaccard_similarity(&set1, &set2)
}

pub fn jaccard(s1: &str, s2: &str) -> f64 {
    let set1: ahash::HashSet<&str> = UnicodeSegmentation::graphemes(s1, true).collect();
    let set2: ahash::HashSet<&str> = UnicodeSegmentation::graphemes(s2, true).collect();

    jaccard_similarity(&set1, &set2)
}

// Sorensen-Dice

fn sorensen_dice_similarity<T: PartialEq + Hash + Eq>(
    set1: &ahash::HashSet<T>,
    set2: &ahash::HashSet<T>,
) -> f64 {
    let jaccard_sim = jaccard_similarity(set1, set2);

    (2.0 * jaccard_sim) / (1.0 + jaccard_sim)
}

pub fn sorensen_dice_ascii(s1: &str, s2: &str) -> f64 {
    let set1: ahash::HashSet<u8> = s1.bytes().collect();
    let set2: ahash::HashSet<u8> = s2.bytes().collect();

    sorensen_dice_similarity(&set1, &set2)
}

pub fn sorensen_dice(s1: &str, s2: &str) -> f64 {
    let set1: ahash::HashSet<&str> = UnicodeSegmentation::graphemes(s1, true).collect();
    let set2: ahash::HashSet<&str> = UnicodeSegmentation::graphemes(s2, true).collect();

    sorensen_dice_similarity(&set1, &set2)
}

// Cosine distance (Otsuka-Ochiai)

pub fn cosine_similarity<T: PartialEq + Hash + Eq>(
    set1: &ahash::HashSet<T>,
    set2: &ahash::HashSet<T>,
) -> f64 {
    let intersection_size = set1.intersection(set2).count();
    let denom = ((set1.len() * set2.len()) as f64).sqrt();

    intersection_size as f64 / denom
}

pub fn cosine(s1: &str, s2: &str) -> f64 {
    let set1: ahash::HashSet<&str> = UnicodeSegmentation::graphemes(s1, true).collect();
    let set2: ahash::HashSet<&str> = UnicodeSegmentation::graphemes(s2, true).collect();

    cosine_similarity(&set1, &set2)
}

pub fn cosine_ascii(s1: &str, s2: &str) -> f64 {
    let set1: ahash::HashSet<u8> = s1.bytes().collect();
    let set2: ahash::HashSet<u8> = s2.bytes().collect();

    cosine_similarity(&set1, &set2)
}
