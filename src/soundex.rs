use crate::hamming;
use deunicode::AsciiChars;
use smallvec::SmallVec;

const B: u8 = 66;
const C: u8 = 67;
const D: u8 = 68;
const F: u8 = 70;
const G: u8 = 71;
const H: u8 = 72;
const J: u8 = 74;
const K: u8 = 75;
const L: u8 = 76;
const M: u8 = 77;
const N: u8 = 78;
const P: u8 = 80;
const Q: u8 = 81;
const R: u8 = 82;
const S: u8 = 83;
const T: u8 = 84;
const V: u8 = 86;
const W: u8 = 87;
const X: u8 = 88;
const Z: u8 = 90;

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
