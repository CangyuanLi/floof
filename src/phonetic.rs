use crate::hamming;
use deunicode::AsciiChars;
use smallvec::SmallVec;

const A: u8 = 65;
const B: u8 = 66;
const C: u8 = 67;
const D: u8 = 68;
const E: u8 = 69;
const F: u8 = 70;
const G: u8 = 71;
const H: u8 = 72;
const I: u8 = 73;
const J: u8 = 74;
const K: u8 = 75;
const L: u8 = 76;
const M: u8 = 77;
const N: u8 = 78;
const O: u8 = 79;
const P: u8 = 80;
const Q: u8 = 81;
const R: u8 = 82;
const S: u8 = 83;
const T: u8 = 84;
const U: u8 = 85;
const V: u8 = 86;
const W: u8 = 87;
const X: u8 = 88;
const Y: u8 = 89;
const Z: u8 = 90;

// Soundex

// #[inline]
// fn char_to_digit(c: char) -> char {
//     match c {
//         'B' | 'F' | 'P' | 'V' => '1',
//         'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => '2',
//         'D' | 'T' => '3',
//         'L' => '4',
//         'M' | 'N' => '5',
//         'R' => '6',
//         _ => '*', // * is a sentinel value that says this isn't in our vocab
//     }
// }

#[inline]
fn byte_to_digit(b: u8) -> u8 {
    match b {
        B | F | P | V => 1,
        C | G | J | K | Q | S | X | Z => 2,
        D | T => 3,
        L => 4,
        M | N => 5,
        R => 6,
        _ => 0,
    }
}

fn get_soundex_code(slice: &[u8]) -> SmallVec<[u8; 4]> {
    // 1. Retain the first letter

    let mut soundex_code: SmallVec<[u8; 4]> = SmallVec::with_capacity(4);
    soundex_code.push(slice[0]);

    // 2. Replace consonants with digits (after the first letter)
    let mut prev_b = slice[0];
    for b in slice.iter().skip(1) {
        let c = *b;
        let digit = byte_to_digit(c);

        if digit != 0 {
            if digit != prev_b {
                soundex_code.push(digit)
            }
            prev_b = digit;
        } else if c != H && c != W {
            prev_b = 0;
        }

        if soundex_code.len() == 4 {
            break;
        }
    }

    while soundex_code.len() < 4 {
        soundex_code.push(0);
    }

    soundex_code
}

fn process_str(s: &str) -> String {
    s.to_uppercase().ascii_chars().flatten().collect::<String>()
}

fn byte_to_char(b: u8) -> String {
    if b >= A {
        char::from(b).to_string()
    } else {
        b.to_string()
    }
}

pub fn soundex_code(s: &str) -> String {
    get_soundex_code(process_str(s).as_bytes())
        .into_iter()
        .map(byte_to_char)
        .collect()
}

pub fn soundex_code_ascii(s: &str) -> String {
    get_soundex_code(s.as_bytes())
        .into_iter()
        .map(byte_to_char)
        .collect()
}

pub fn soundex_similarity(slice1: &[u8], slice2: &[u8]) -> f64 {
    hamming::hamming_similarity(
        get_soundex_code(slice1).as_slice(),
        get_soundex_code(slice2).as_slice(),
    )
}

pub fn soundex(s1: &str, s2: &str) -> f64 {
    let s1 = process_str(s1);
    let s2 = process_str(s2);

    soundex_similarity(s1.as_bytes(), s2.as_bytes())
}

pub fn soundex_ascii(s1: &str, s2: &str) -> f64 {
    let s1 = s1.to_ascii_uppercase();
    let s2 = s2.to_ascii_uppercase();

    soundex_similarity(s1.as_bytes(), s2.as_bytes())
}

// NYSIIS

// fn is_vowel(b: u8) -> bool {
//     matches!(b, A | E | I | O | U)
// }

// fn get_nysiis_code(slice: &[u8]) -> &[u8] {
//     let mut vec = utils::FastVec::from(slice);

//     // prefixes
//     if vec.starts_with(&[M, A, C]) {
//         vec[1] = C;
//     } else if vec.starts_with(&[K, N]) {
//         vec[0] = N;
//     } else if vec[0] == K {
//         vec[0] = C;
//     } else if vec.starts_with(&[P, H]) || vec.starts_with(&[P, F]) {
//         vec[0] = F;
//         vec[1] = F;
//     } else if vec.starts_with(&[S, C, H]) {
//         vec[1] = S;
//         vec[2] = S;
//     }

//     // suffixes
//     if vec.ends_with(&[E, E]) || vec.ends_with(&[I, E]) {
//         vec.pop();
//         vec.pop();
//         vec.push(I);
//     } else if vec.ends_with(&[D, T])
//         || vec.ends_with(&[R, T])
//         || vec.ends_with(&[R, D])
//         || vec.ends_with(&[N, T])
//         || vec.ends_with(&[N, D])
//     {
//         vec.pop();
//         vec.pop();
//         vec.push(D);
//     }

//     // start building out the code
//     let nysiis_code = utils::FastVec::new();
//     nysiis_code.push(slice[0]);

//     let mut i = 1;
//     while i < vec.len() {
//         let b = vec[i];

//         if i + 1 < slice.len() && b == E && vec[i + 1] == V {
//             vec[i] = A;
//             vec[i + 1] = F;
//             i += 1;
//         } else if is_vowel(b) {
//             vec[i] = A;
//         } else if b == Q {
//             vec[i] = G;
//         } else if b == Z {
//             vec[i] = S;
//         } else if b == M {
//             vec[i] = N;
//         } else if b == K {
//             if i + 1 < slice.len() && vec[i + 1] == N {
//                 vec[i] = N;
//             } else {
//                 vec[i] = C;
//             }
//         } else if b == S && vec[i + 1] == C && vec[i + 2] == H {
//             vec[i + 1] = S;
//             vec[i + 2] = H;
//             i += 2; // skip ahead 3, but we always inc by 1 at end of the loop
//         } else if b == P && vec[i + 1] == H {
//             vec[i] = F;
//             vec[i + 1] = F;
//             i += 1; // skip ahead 2, but we always inc by 1 at end of the loop
//         } else if b == H && (!is_vowel(vec[i - 1]) || !is_vowel(vec[i + 1])) {
//             // If the current position is the letter 'H' and either the preceding or
//             // following letter is not a vowel (AEIOU) then replace the current position
//             // with the preceding letter
//             if is_vowel(vec[i - 1]) {
//                 vec[i] = A;
//             } else {
//                 vec[i] = vec[i - 1];
//             }
//         } else if b == W && is_vowel(vec[i - 1]) {
//             vec[i] = vec[i - 1];
//         }

//         if

//         i += 1;
//     }

//     &[1, 2, 3]
// }
