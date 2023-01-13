use crate::lcp::lcp_one_to_one;
use dyn_clone::DynClone;

pub type SuggestResult = (TokenMatch, Vec<Suggestion>);

pub trait Suggest: DynClone {
    // Mutates internal state only when match is full
    fn update(&mut self, token: &str) -> SuggestResult;

    fn consume_input(&mut self, input: &str) -> SuggestResult {
        let (mut token_match, mut suggestions) = (TokenMatch::Full, Vec::new());
        for token in tokenize(input) {
            if let TokenMatch::Full = token_match {
                (token_match, suggestions) = self.update(token);
            } else {
                return NO_SUGGESTION;
            }
        }
        (token_match, suggestions)
    }

    fn reset(&mut self);
}
dyn_clone::clone_trait_object!(Suggest);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Suggestion {
    previous: String,
    current: String,
    leftover: String,
    autocomplete: bool,
}

impl Suggestion {
    pub fn new(previous: String, current: String, leftover: String, autocomplete: bool) -> Self {
        Self {
            previous,
            current,
            leftover,
            autocomplete,
        }
    }

    #[inline]
    pub fn first_token(current: String, leftover: String, autocomplete: bool) -> Self {
        Self::new(String::new(), current, leftover, autocomplete)
    }

    #[inline]
    pub fn last_token(previous: String, current: String, autocomplete: bool) -> Self {
        Self::new(previous, current, String::new(), autocomplete)
    }

    #[inline]
    pub fn single_token(current: String, autocomplete: bool) -> Self {
        Self::new(String::new(), current, String::new(), autocomplete)
    }

    pub fn get_autocompletion(&self) -> Option<&str> {
        if self.autocomplete {
            Some(&self.current)
        } else {
            None
        }
    }
}

impl ToString for Suggestion {
    fn to_string(&self) -> String {
        let maybe_space_previous = if self.previous.is_empty() { "" } else { " " };
        let maybe_space_leftover = if self.leftover.is_empty() { "" } else { " " };
        format!(
            "{}{}{}{}{}",
            self.previous, maybe_space_previous, self.current, maybe_space_leftover, self.leftover
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenMatch {
    None,
    Partial,
    Full,
}

impl TokenMatch {
    pub fn max(matches: &[Self]) -> Self {
        matches.iter().fold(TokenMatch::None, |acc, m| match m {
            TokenMatch::Full => TokenMatch::Full,
            TokenMatch::Partial => {
                if let TokenMatch::Full = acc {
                    acc
                } else {
                    *m
                }
            }
            TokenMatch::None => acc,
        })
    }
}

pub const NO_SUGGESTION: SuggestResult = (TokenMatch::None, Vec::new());

pub fn end_of_input(token: &str, suggestions: Vec<Suggestion>) -> SuggestResult {
    if token.is_empty() {
        (TokenMatch::None, suggestions)
    } else {
        NO_SUGGESTION
    }
}

pub fn lcp_match(kw: &str, token: &str) -> TokenMatch {
    if token.len() > kw.len() {
        return TokenMatch::None;
    }

    let lcp_len = lcp_one_to_one(kw, token).len();
    if lcp_len == kw.len() {
        TokenMatch::Full
    } else if lcp_len == token.len() {
        TokenMatch::Partial
    } else {
        TokenMatch::None
    }
}

pub fn tokenize(input: &str) -> Vec<&str> {
    let mut tokens: Vec<&str> = input.split_whitespace().collect();
    if input.is_empty() || input.chars().last().unwrap().is_whitespace() {
        tokens.push("");
    }
    tokens
}

#[derive(Clone)]
pub struct OneOpaqueArgCmd<'a> {
    kw: &'a str,
    arg_name: &'a str,
    state: OneOpaqueArgCmdState,
}

#[derive(Clone)]
enum OneOpaqueArgCmdState {
    /// Matching the command keyword
    MatchKw,
    /// Matching the command argument
    MatchArg,
    /// Nothing left to match
    End,
}

impl<'a> OneOpaqueArgCmd<'a> {
    pub fn new(kw: &'a str, arg_name: &'a str) -> Self {
        Self {
            kw,
            arg_name,
            state: OneOpaqueArgCmdState::MatchKw,
        }
    }

    #[inline]
    fn format_arg(&self) -> String {
        format!("<{}>", self.arg_name)
    }

    #[inline]
    fn get_kw_suggestion(&self) -> Suggestion {
        Suggestion::first_token(self.kw.to_string(), self.format_arg(), true)
    }

    #[inline]
    fn get_arg_suggestion(&self) -> Suggestion {
        Suggestion::last_token(self.kw.to_string(), self.format_arg(), false)
    }
}

impl<'a> Suggest for OneOpaqueArgCmd<'a> {
    fn update(&mut self, token: &str) -> SuggestResult {
        match &self.state {
            OneOpaqueArgCmdState::MatchKw => match lcp_match(&self.kw, token) {
                TokenMatch::Full => {
                    self.state = OneOpaqueArgCmdState::MatchArg;
                    (TokenMatch::Full, vec![self.get_kw_suggestion()])
                }
                TokenMatch::Partial => (TokenMatch::Partial, vec![self.get_kw_suggestion()]),
                TokenMatch::None => NO_SUGGESTION,
            },
            OneOpaqueArgCmdState::MatchArg => {
                self.state = OneOpaqueArgCmdState::End;
                (TokenMatch::Full, vec![self.get_arg_suggestion()])
            }
            OneOpaqueArgCmdState::End => end_of_input(token, vec![self.get_arg_suggestion()]),
        }
    }

    fn reset(&mut self) {
        self.state = OneOpaqueArgCmdState::MatchKw
    }
}

#[derive(Clone)]
pub struct SingleKwCmd {
    kw: String,
    matched_kw: bool,
}

impl SingleKwCmd {
    pub fn new(kw: String) -> Self {
        Self {
            kw,
            matched_kw: false,
        }
    }

    #[inline]
    fn get_suggestion(&self) -> Suggestion {
        Suggestion::single_token(self.kw.to_string(), true)
    }
}

impl Suggest for SingleKwCmd {
    fn update(&mut self, token: &str) -> SuggestResult {
        if self.matched_kw {
            return end_of_input(token, vec![self.get_suggestion()]);
        }
        match lcp_match(&self.kw, token) {
            TokenMatch::Full => {
                self.matched_kw = true;
                (TokenMatch::Full, vec![self.get_suggestion()])
            }
            TokenMatch::Partial => (TokenMatch::Partial, vec![self.get_suggestion()]),
            TokenMatch::None => end_of_input(token, vec![self.get_suggestion()]),
        }
    }

    fn reset(&mut self) {
        self.matched_kw = false;
    }
}

#[derive(Clone)]
pub struct OneChoiceArgCmd<'a> {
    kw: &'a str,
    choices: &'a [&'a str],
    state: OneChoiceArgCmdState<'a>,
}

#[derive(Clone)]
enum OneChoiceArgCmdState<'a> {
    /// Matching the command keyword
    MatchKw,
    /// Matching the choice
    MatchChoice,
    /// Nothing left to match
    End(&'a str),
}

impl<'a> OneChoiceArgCmd<'a> {
    pub fn new(kw: &'a str, choices: &'a [&'a str]) -> Self {
        Self {
            kw,
            choices,
            state: OneChoiceArgCmdState::MatchKw,
        }
    }

    #[inline]
    fn get_choice_suggestion(&self, choice: &str) -> Suggestion {
        Suggestion::last_token(self.kw.to_string(), choice.to_string(), true)
    }
}

impl<'a> Suggest for OneChoiceArgCmd<'a> {
    fn update(&mut self, token: &str) -> SuggestResult {
        match &self.state {
            OneChoiceArgCmdState::MatchKw => match lcp_match(self.kw, token) {
                TokenMatch::Full => {
                    self.state = OneChoiceArgCmdState::MatchChoice;
                    (
                        TokenMatch::Full,
                        self.choices
                            .iter()
                            .map(|c| {
                                Suggestion::first_token(self.kw.to_string(), c.to_string(), true)
                            })
                            .collect(),
                    )
                }
                TokenMatch::Partial => (
                    TokenMatch::Partial,
                    vec![Suggestion::first_token(
                        self.kw.to_string(),
                        format!("<{}>", self.choices.join("|")),
                        true,
                    )],
                ),
                TokenMatch::None => NO_SUGGESTION,
            },
            OneChoiceArgCmdState::MatchChoice => {
                let matched_choices: Vec<(TokenMatch, &str)> = self
                    .choices
                    .iter()
                    .map(|c| (lcp_match(c, token), *c))
                    .filter(|(token_match, _)| match token_match {
                        TokenMatch::None => false,
                        _ => true,
                    })
                    .collect();

                if matched_choices.is_empty() {
                    NO_SUGGESTION
                } else if matched_choices.len() == 1 {
                    let choice = matched_choices[0].1;
                    self.state = OneChoiceArgCmdState::End(choice);
                    (
                        matched_choices[0].0,
                        vec![self.get_choice_suggestion(choice)],
                    )
                } else {
                    let matches: Vec<TokenMatch> =
                        matched_choices.iter().map(|(m, _)| *m).collect();
                    (
                        TokenMatch::max(&matches),
                        matched_choices
                            .iter()
                            .map(|(_, c)| self.get_choice_suggestion(c))
                            .collect(),
                    )
                }
            }
            OneChoiceArgCmdState::End(choice) => {
                end_of_input(token, vec![self.get_choice_suggestion(*choice)])
            }
        }
    }

    fn reset(&mut self) {
        self.state = OneChoiceArgCmdState::MatchKw;
    }
}

#[cfg(test)]
mod test {
    use super::{tokenize, SingleKwCmd, Suggest, Suggestion, TokenMatch};

    #[test]
    fn test_tokenize() {
        assert_eq!(
            tokenize(""),
            vec!["".to_string()],
            "Single empty token must be generated for empty input"
        );
        assert_eq!(
            tokenize(" "),
            vec!["".to_string()],
            "Input with trailing whitespace(s) must be parsed as single empty token"
        );
        assert_eq!(
            tokenize("test"),
            vec!["test".to_string()],
            "Input without whitespaces must be parsed as single token"
        );
    }

    #[test]
    fn test_consume_input() {
        let cmd = SingleKwCmd::new("test_cmd".to_string());
        let update_result = cmd.clone().update("");
        let consume_result = cmd.clone().consume_input("");
        assert_update_result_eq(update_result, consume_result)
    }

    #[test]
    fn test_single_wk_cmd() {
        let cmd_kw = "test_cmd";
        let full_cmd_suggestion = Suggestion::single_token(cmd_kw.to_string(), true);
        let cmd = SingleKwCmd::new(cmd_kw.to_string());

        // Check result with no input
        let (tm, sugs) = cmd.clone().consume_input("");
        assert_match(TokenMatch::Partial, tm);
        assert_single_suggestion(&full_cmd_suggestion, &sugs);

        // Check result with matching partial input
        let (tm, sugs) = cmd.clone().consume_input("test_");
        assert_match(TokenMatch::Partial, tm);
        assert_single_suggestion(&full_cmd_suggestion, &sugs);

        // Check result with non-matching partial input
        let (tm, sugs) = cmd.clone().consume_input("test_x");
        assert_match(TokenMatch::None, tm);
        assert_no_suggestion(&sugs);

        // Check result with full input
        let (tm, sugs) = cmd.clone().consume_input("test_cmd");
        assert_match(TokenMatch::Full, tm);
        assert_single_suggestion(&full_cmd_suggestion, &sugs);

        // Check result with more than full input
        let (tm, sugs) = cmd.clone().consume_input("test_cmd_x");
        assert_match(TokenMatch::None, tm);
        assert_no_suggestion(&sugs);

        // Check result with full input and unmatched token
        let (tm, sugs) = cmd.clone().consume_input("test_cmd x");
        assert_match(TokenMatch::None, tm);
        assert_no_suggestion(&sugs)
    }

    fn assert_update_result_eq(
        expected: (TokenMatch, Vec<Suggestion>),
        got: (TokenMatch, Vec<Suggestion>),
    ) {
        assert_match(expected.0, got.0);
        assert_eq!(expected.1, got.1);
    }

    #[inline]
    fn assert_match(expected: TokenMatch, got: TokenMatch) {
        assert_eq!(expected, got);
    }

    #[inline]
    fn assert_single_suggestion(expected: &Suggestion, got: &Vec<Suggestion>) {
        assert_eq!(1, got.len());
        assert_eq!(expected, &got[0]);
    }

    #[inline]
    fn assert_no_suggestion(got: &Vec<Suggestion>) {
        assert_eq!(0, got.len());
    }
}
