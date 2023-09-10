use crate::utils;
use unicode_segmentation::UnicodeSegmentation;

fn hamming_similarity<T>(iter1: T, iter2: T) -> f64
where
    T: utils::HasLength + IntoIterator,
    T::Item: PartialEq,
{
    let len1 = iter1.len();
    let len2 = iter2.len();
    let mut dist = len1 - len2;
    for (x, y) in std::iter::zip(iter1, iter2) {
        if x != y {
            dist += 1;
        }
    }

    utils::distance_to_similarity(dist, len1, len2)
}

pub fn hamming(s1: &str, s2: &str) -> f64 {
    let us1 = UnicodeSegmentation::graphemes(s1, true).collect::<utils::FastVec<&str>>();
    let us2 = UnicodeSegmentation::graphemes(s2, true).collect::<utils::FastVec<&str>>();

    hamming_similarity(us1, us2)
}

pub fn hamming_ascii(s1: &str, s2: &str) -> f64 {
    let s1 = s1.as_bytes();
    let s2 = s2.as_bytes();

    hamming_similarity(s1, s2)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_hamming() {
        let s1 = "abc";
        let s2 = "def";
        dbg!(hamming(s1, s2));
    }

    #[test]
    fn test_hamming_ascii() {
        let s1 = "abc";
        let s2 = "def";
        dbg!(hamming_ascii(s1, s2));
    }
}
