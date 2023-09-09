use smallvec::SmallVec;

const VEC_SIZE: usize = 32;

pub type FastVec<T> = SmallVec<[T; VEC_SIZE]>;

pub trait HasLength {
    fn len(&self) -> usize;
}

impl<T> HasLength for FastVec<T> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> HasLength for &[T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

pub fn distance_to_percent(distance: usize, len1: usize, len2: usize) -> f64 {
    let max_len = std::cmp::max(len1, len2);
    let max_len = max_len as f64;
    let dist = distance as f64;

    (max_len - dist) / max_len
}
