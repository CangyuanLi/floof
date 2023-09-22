use crate::utils;
use smallvec::smallvec;
use unicode_segmentation::UnicodeSegmentation;

macro_rules! min {
    ($x: expr) => ($x);
    ($x: expr, $($z: expr),+) => (::std::cmp::min($x, min!($($z),*)));
}

// Levenshtein

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
            levenshtein_distance = min!(levenshtein_distance + 1, distance, prev + 1);
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

// Optimal String Alignment

pub fn osa_similarity<T: PartialEq>(slice1: &[T], slice2: &[T]) -> f64 {
    let len1 = slice1.len();
    let len2 = slice2.len();

    let end_idx2 = len2 + 1;

    let mut prev_two_distances: utils::FastVec<usize> = (0..end_idx2).collect();
    let mut prev_distances: utils::FastVec<usize> = (0..end_idx2).collect();
    let mut curr_distances: utils::FastVec<usize> = smallvec![0];

    for (i, x) in slice1.iter().enumerate() {
        curr_distances[0] = i + 1;

        for (j, y) in slice2.iter().enumerate() {
            let cost = if x == y { 0 } else { 1 };

            let deletion_cost = curr_distances[j] + 1;
            let insertion_cost = prev_distances[j + 1] + 1;
            let substitution_cost = prev_distances[j] + cost;

            curr_distances[j + 1] = min!(deletion_cost, insertion_cost, substitution_cost);

            if i > 0 && j > 0 && x != y && x == &slice2[j - 1] && y == &slice1[i - 1] {
                curr_distances[j + 1] =
                    std::cmp::min(curr_distances[j + 1], prev_two_distances[j - 1] + 1);
            }
        }

        prev_two_distances.clone_from(&prev_distances);
        prev_distances.clone_from(&curr_distances);
    }

    let osa_distance = curr_distances[len2];

    utils::distance_to_similarity(osa_distance, len1, len2)
}

pub fn osa(s1: &str, s2: &str) -> f64 {
    let us1: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s1, true).collect();
    let us2: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s2, true).collect();

    osa_similarity(&us1, &us2)
}

pub fn osa_ascii(s1: &str, s2: &str) -> f64 {
    osa_similarity(s1.as_bytes(), s2.as_bytes())
}

// Damerau-Levenshtein

/// Maps an index from a matrix to a 1D-array
#[inline]
fn map_index(i: usize, j: usize, width: usize) -> usize {
    j * width + i
}

pub fn damerau_levenshtein_similarity<T: Eq + std::hash::Hash>(slice1: &[T], slice2: &[T]) -> f64 {
    let len1 = slice1.len();
    let len2 = slice2.len();

    // Need inclusive ranges all over the place, but ..= syntax may lead to worse
    // performance, see
    // https://www.reddit.com/r/rust/comments/15tvuio/why_isnt_the_for_loop_optimized_better_in_this/
    let end_idx1 = len1 + 1;
    let end_idx2 = len2 + 1;

    let max_dist = len1 + len2;
    let width = len1 + 2;

    // In the Wikipedia pseudocode, this is a matrix. Use one vector for performance.
    let mut distances: Vec<usize> = vec![0; width * (len2 + 2)];
    distances[0] = max_dist;

    for i in 0..end_idx1 {
        distances[map_index(i + 1, 0, width)] = max_dist;
        distances[map_index(i + 1, 0, width)] = i;
    }

    for j in 0..end_idx2 {
        distances[map_index(0, j + 1, width)] = max_dist;
        distances[map_index(1, j + 1, width)] = j;
    }

    let mut items = ahash::AHashMap::with_capacity(std::cmp::max(len1, len2));
    for i in 1..end_idx1 {
        let mut db = 0;

        for j in 1..end_idx2 {
            let k = match items.get(&slice2[j - 1]) {
                Some(&value) => value,
                None => 0,
            };

            let insertion_cost = distances[map_index(i, j + 1, width)] + 1;
            let deletion_cost = distances[map_index(i + 1, j, width)] + 1;
            let transposition_cost =
                distances[map_index(k, db, width)] + (i - k - 1) + 1 + (j - db - 1);

            let mut substitution_cost = distances[map_index(i, j, width)] + 1;
            if slice1[i - 1] == slice2[j - 1] {
                db = j;
                substitution_cost -= 1;
            }

            distances[map_index(i + 1, j + 1, width)] = min!(
                insertion_cost,
                deletion_cost,
                transposition_cost,
                substitution_cost
            );
        }

        items.insert(&slice1[i - 1], i);
    }

    let damerau_levenshtein_distance = distances[map_index(end_idx1, end_idx2, width)];

    utils::distance_to_similarity(damerau_levenshtein_distance, len1, len2)
}

/// Implements the exact same algorithm as `damerau_levenshtein_similarity`, but uses
/// an array instead of a HashMap. We can do this because the ASCII character set maps
/// to an array of length 256. This should yield better performance.
///
/// TODO: Benchmark the exact performance increase.
pub fn damerau_levenshtein_similarity_ascii(slice1: &[u8], slice2: &[u8]) -> f64 {
    let len1 = slice1.len();
    let len2 = slice2.len();

    // Need inclusive ranges all over the place, but ..= syntax may lead to worse
    // performance, see
    // https://www.reddit.com/r/rust/comments/15tvuio/why_isnt_the_for_loop_optimized_better_in_this/
    let end_idx1 = len1 + 1;
    let end_idx2 = len2 + 1;

    let max_dist = len1 + len2;
    let width = len1 + 2;

    // In the Wikipedia pseudocode, this is a matrix. Use one vector for performance.
    let mut distances: Vec<usize> = vec![0; width * (len2 + 2)];
    distances[0] = max_dist;

    for i in 0..end_idx1 {
        distances[map_index(i + 1, 0, width)] = max_dist;
        distances[map_index(i + 1, 0, width)] = i;
    }

    for j in 0..end_idx2 {
        distances[map_index(0, j + 1, width)] = max_dist;
        distances[map_index(1, j + 1, width)] = j;
    }

    let mut items = [0usize; 256];

    for i in 1..end_idx1 {
        let mut db = 0;

        for j in 1..end_idx2 {
            let k = items[slice2[j - 1] as usize];

            let insertion_cost = distances[map_index(i, j + 1, width)] + 1;
            let deletion_cost = distances[map_index(i + 1, j, width)] + 1;
            let transposition_cost =
                distances[map_index(k, db, width)] + (i - k - 1) + 1 + (j - db - 1);

            let mut substitution_cost = distances[map_index(i, j, width)] + 1;
            if slice1[i - 1] == slice2[j - 1] {
                db = j;
                substitution_cost -= 1;
            }

            distances[map_index(i + 1, j + 1, width)] = min!(
                insertion_cost,
                deletion_cost,
                transposition_cost,
                substitution_cost
            );
        }

        items[slice1[i - 1] as usize] = i;
    }

    let damerau_levenshtein_distance = distances[map_index(end_idx1, end_idx2, width)];

    utils::distance_to_similarity(damerau_levenshtein_distance, len1, len2)
}

pub fn damerau_levenshtein(s1: &str, s2: &str) -> f64 {
    let us1: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s1, true).collect();
    let us2: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s2, true).collect();

    damerau_levenshtein_similarity(&us1, &us2)
}

pub fn damerau_levenshtein_ascii(s1: &str, s2: &str) -> f64 {
    damerau_levenshtein_similarity_ascii(s1.as_bytes(), s2.as_bytes())
}
