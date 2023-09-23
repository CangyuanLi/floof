use unicode_segmentation::UnicodeSegmentation;

struct Metaphone {
    len: usize,
    encoded_len: usize,
    primary: String,
    secondary: String,
    curr_idx: usize,
    end_idx: usize,
}

fn is_vowel(char: &str) -> bool {
    matches!(
        char,
        "A" | "E"
            | "I"
            | "O"
            | "U"
            | "Y"
            | "À"
            | "Á"
            | "Â"
            | "Ã"
            | "Ä"
            | "Å"
            | "Æ"
            | "È"
            | "É"
            | "Ê"
            | "Ë"
            | "Ì"
            | "Í"
            | "Î"
            | "Ï"
            | "Ò"
            | "Ó"
            | "Ô"
            | "Õ"
            | "Ö"
            | ""
            | "Ø"
            | "Ù"
            | "Ú"
            | "Û"
            | "Ü"
            | "Ý"
            | ""
    )
}

fn is_vowel_ascii(char: &str) -> bool {
    matches!(char, "A" | "E" | "I" | "O" | "U" | "Y")
}

impl Metaphone {
    fn string_at(self, start_idx: usize, length: usize, compare_strings: &[&str]) -> bool {
        if (start_idx > self.len) || ((start_idx + length - 1) > (self.len - 1)) {
            return false;
        }

        true
    }
}

pub fn metaphone_slice(slice: &[&str]) {}
