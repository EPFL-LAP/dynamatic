use std::collections::HashMap;

use crate::{
    lcp::lcp_many_to_many,
    suggest::{tokenize, Suggest, Suggestion},
};
use inquire::{autocompletion::Replacement, Autocomplete, CustomUserError};

#[derive(Clone)]
pub struct CommandCompleter {
    suggesters: Vec<Box<dyn Suggest>>,
    suggestions: Vec<Suggestion>,
    autocompletions: HashMap<String, Option<String>>,
}

impl CommandCompleter {
    pub fn new(suggesters: Vec<Box<dyn Suggest>>) -> Self {
        Self {
            suggesters,
            suggestions: Vec::new(),
            autocompletions: HashMap::new(),
        }
    }
}

impl Autocomplete for CommandCompleter {
    fn get_suggestions(&mut self, input: &str) -> Result<Vec<String>, CustomUserError> {
        // Reset all suggesters
        self.suggesters
            .iter_mut()
            .for_each(|suggester| suggester.reset());

        // Make all suggesters consume the entire input and update suggesions
        self.suggestions = self
            .suggesters
            .iter_mut()
            .map(|s| s.consume_input(input).1)
            .flatten()
            .collect();

        // Update autocompletions (mapping between displayed suggestion and
        // string to autocomplete to on selection)
        self.autocompletions.clear();
        for suggestion in self.suggestions.iter() {
            self.autocompletions.insert(
                suggestion.to_string(),
                match suggestion.get_autocompletion() {
                    Some(completion) => Some(completion.to_string()),
                    None => None,
                },
            );
        }

        Ok(self
            .suggestions
            .iter()
            .map(|suggestion| suggestion.to_string())
            .collect())
    }

    fn get_completion(
        &mut self,
        input: &str,
        highlighted_suggestion: Option<String>,
    ) -> Result<Replacement, CustomUserError> {
        let last_token = {
            let tokens = tokenize(input);
            if tokens.is_empty() {
                ""
            } else {
                tokens[tokens.len() - 1]
            }
        };

        let process_auto = |auto: &Option<String>, is_full_match: bool| match auto {
            Some(completion) => {
                let idx = input.len() - last_token.len();
                let maybe_space = if is_full_match { " " } else { "" };
                format!("{}{}{}", &input[0..idx], completion, maybe_space)
            }
            None => input.to_string(),
        };

        Ok(match highlighted_suggestion {
            Some(suggestion) => Some(process_auto(
                self.autocompletions.get(&suggestion).unwrap(),
                true,
            )),
            None => {
                // The autocompletion is the longest common prefix of all possible completions
                let completions: Vec<&str> = self
                    .autocompletions
                    .values()
                    .filter(|comp| if let Some(_) = comp { true } else { false })
                    .map(|comp| {
                        if let Some(auto) = comp {
                            &auto[..]
                        } else {
                            panic!("unreachable")
                        }
                    })
                    .collect();

                if completions.is_empty() {
                    Replacement::None
                } else {
                    let lcp = lcp_many_to_many(completions.iter().map(|c| &**c).collect());
                    Replacement::Some(process_auto(&Some(lcp.to_string()), completions.len() == 1))
                }
            }
        })
    }
}
