use std::{error::Error, fmt::Display};

use crate::suggest::Suggest;
use dyn_clone::DynClone;
use inquire::{error::InquireResult, InquireError};

pub trait Command<State>: DynClone
where
    State: Clone,
{
    fn execute(&self, tokens: &[&str], state: &mut State) -> ExecuteResult;

    fn get_suggester(&self) -> Box<dyn Suggest>;

    fn check_keyword(&self, tokens: &[&str], kw: &str) -> Result<(), ExecuteError> {
        if tokens.is_empty() || tokens[0] != kw {
            return Err(ExecuteError::ErrKeyword);
        }
        Ok(())
    }

    fn check_num_args(&self, tokens: &[&str], expected: usize) -> Result<(), ExecuteError> {
        let n_args = tokens.len() - 1;
        if expected != n_args {
            return Err(ExecuteError::ErrArgument(format!(
                "Incorrect number of arguments. Expected {} but got {}",
                expected, n_args
            )));
        }
        return Ok(());
    }
}

pub type ExecuteResult = Result<CommandEffect, ExecuteError>;

pub enum CommandEffect {
    State(Option<String>),
    Callback(fn() -> ()),
    Run,
    Exit,
}

pub enum ExecuteError {
    ErrKeyword,
    ErrArgument(String),
}

#[derive(Debug)]
pub enum PromptError {
    NoMatch,
    InvalidArgs(String),
    UserEscape,
    UserExit,
    LibError(String),
}

impl Display for PromptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Self::NoMatch => "input does not match any known command".to_string(),
            Self::InvalidArgs(msg) => format!("incorrect arguments provided\n{}", &msg),
            Self::UserEscape => "user pressed escape".to_string(),
            Self::UserExit => "user pressed Ctrl+c".to_string(),
            Self::LibError(err) => format!("there was a library error.\n{err}"),
        };
        write!(f, "{msg}")
    }
}

impl Error for PromptError {}

pub fn handle_prompt_result<'a, State>(
    prompt: &InquireResult<String>,
    commands: &'a [Box<dyn Command<State>>],
    state: &mut State,
) -> Result<CommandEffect, PromptError>
where
    State: Clone,
{
    match prompt {
        Ok(input) => {
            let tokens: Vec<&str> = input.split_whitespace().collect();

            // Go through the list of commands to see if any matches
            for cmd in commands {
                match cmd.execute(&tokens, state) {
                    Ok(effect) => {
                        return Ok(effect);
                    }
                    Err(exe_error) => match exe_error {
                        ExecuteError::ErrKeyword => (),
                        ExecuteError::ErrArgument(msg) => {
                            return Err(PromptError::InvalidArgs(msg));
                        }
                    },
                }
            }
            Err(PromptError::NoMatch)
        }
        Err(err) => Err(match err {
            InquireError::OperationCanceled => PromptError::UserEscape,
            InquireError::OperationInterrupted => PromptError::UserExit,
            inquire_err => PromptError::LibError(format!("{inquire_err}")),
        }),
    }
}
