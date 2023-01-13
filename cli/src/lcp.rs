pub fn lcp_one_to_one<'a, 'b>(one1: &'a str, one2: &'b str) -> &'a str {
    let mut chars1 = one1.chars();
    let mut chars2 = one2.chars();
    let mut match_len = 0;
    loop {
        match (chars1.next(), chars2.next()) {
            (Some(c1), Some(c2)) if c1 == c2 => {
                match_len += 1;
            }
            _ => break,
        }
    }
    &one1[..match_len]
}

pub fn lcp_many_to_many<'a>(elements: Vec<&'a str>) -> &'a str {
    if elements.is_empty() {
        return "";
    }

    let mut chars: Vec<std::str::Chars> = elements.iter().map(|elem| elem.chars()).collect();
    let mut match_len = 0;
    'outer: loop {
        match chars[0].next() {
            Some(c1) => {
                for elem_chars in chars[1..].iter_mut() {
                    match elem_chars.next() {
                        Some(c2) if c1 == c2 => (),
                        _ => break 'outer,
                    }
                }
                match_len += 1;
            }
            None => break,
        }
    }

    &elements[0][..match_len]
}
