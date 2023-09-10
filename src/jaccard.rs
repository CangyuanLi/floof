use std::hash::Hash;

use ahash;
use unicode_segmentation::UnicodeSegmentation;

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
