use crate::utils;
use smallvec::smallvec;
use unicode_segmentation::UnicodeSegmentation;

const DEFAULT_JARO_WINKLER_WEIGHT: f64 = 0.1;

fn find_common_prefix_len<T: PartialEq>(slice1: &[T], slice2: &[T]) -> f64 {
    slice1
        .iter()
        .zip(slice2.iter())
        .take(4)
        .take_while(|(ch1, ch2)| ch1 == ch2)
        .count() as f64
}

#[allow(clippy::let_and_return)]
pub fn jaro_similarity<T: PartialEq>(slice1: &[T], slice2: &[T]) -> f64 {
    let len1 = slice1.len();
    let len2 = slice2.len();

    // Values from iter1 and iter2 are considered matching if they match AND are not
    // more than `search_range` characters apart
    let search_range = (std::cmp::max(len1, len2) / 2).saturating_sub(1);

    let mut matches1: utils::FastVec<bool> = smallvec![false; len1];
    let mut matches2: utils::FastVec<bool> = smallvec![false; len2];

    let mut matches = 0;

    for (i, x) in slice1.iter().enumerate() {
        let start = i.saturating_sub(search_range);
        let end = std::cmp::min(i + search_range, len2 - 1) + 1;

        if start >= end {
            continue;
        }

        for j in start..end {
            if matches2[j] || x != &slice2[j] {
                continue;
            }

            matches1[i] = true;
            matches2[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // transposition
    let matched1 =
        std::iter::zip(slice1, matches1).filter_map(|(x, m)| if m { Some(x) } else { None });
    let matched2 =
        std::iter::zip(slice2, matches2).filter_map(|(y, m)| if m { Some(y) } else { None });
    let transpositions = std::iter::zip(matched1, matched2)
        .filter(|(x, y)| x != y)
        .count();

    let matches = matches as f64;
    let transpositions = transpositions as f64;
    let len1 = len1 as f64;
    let len2 = len2 as f64;

    let jaro_sim =
        (matches / len1 + matches / len2 + ((matches - transpositions / 2.0) / matches)) / 3.0;

    jaro_sim
}

pub fn jaro_winkler_similarity<T: PartialEq>(slice1: &[T], slice2: &[T]) -> f64 {
    let jaro_sim = jaro_similarity(slice1, slice2);
    let common_prefix_len = find_common_prefix_len(slice1, slice2);

    jaro_sim + common_prefix_len * DEFAULT_JARO_WINKLER_WEIGHT * (1.0 - jaro_sim)
}

pub fn jaro(s1: &str, s2: &str) -> f64 {
    let us1: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s1, true).collect();
    let us2: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s2, true).collect();

    jaro_similarity(&us1, &us2)
}

pub fn jaro_ascii(s1: &str, s2: &str) -> f64 {
    jaro_similarity(s1.as_bytes(), s2.as_bytes())
}

pub fn jaro_winkler(s1: &str, s2: &str) -> f64 {
    let us1: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s1, true).collect();
    let us2: utils::FastVec<&str> = UnicodeSegmentation::graphemes(s2, true).collect();

    jaro_winkler_similarity(&us1, &us2)
}

pub fn jaro_winkler_ascii(s1: &str, s2: &str) -> f64 {
    let s1 = s1.as_bytes();
    let s2 = s2.as_bytes();

    jaro_winkler_similarity(s1, s2)
}
