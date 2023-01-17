mod command;
mod compile;
mod completer;
mod lcp;
mod pipeline;
mod suggest;

use std::process::exit;

use command::{handle_prompt_result, Command, CommandEffect, ExecuteResult, PromptError};
use compile::compile;
use completer::CommandCompleter;
use inquire::Text;
use suggest::{SingleKwCmd, Suggest};

fn manage() {}

fn main() {
    let commands: [Box<dyn Command<()>>; 2] = [
        Box::new(MainCmd::new("compile".to_string(), compile)),
        Box::new(MainCmd::new("manage".to_string(), manage)),
    ];

    let completer = CommandCompleter::new(commands.iter().map(|cmd| cmd.get_suggester()).collect());
    loop {
        let prompt = Text::new("[dynamatic++]")
            .with_autocomplete(completer.clone())
            .prompt();
        match handle_prompt_result(&prompt, &commands, &mut ()) {
            Ok(effect) => match effect {
                CommandEffect::Callback(callback) => callback(),
                CommandEffect::Exit => break,
                _ => (),
            },
            Err(err) => match err {
                PromptError::UserEscape => break,
                PromptError::UserExit => exit(0),
                other_err => println!("{other_err}"),
            },
        }
    }
}

#[derive(Clone)]
pub struct MainCmd {
    kw: String,
    callback: fn() -> (),
}

impl MainCmd {
    pub fn new(kw: String, callback: fn() -> ()) -> Self {
        Self { kw, callback }
    }
}

impl Command<()> for MainCmd {
    fn execute(&self, tokens: &[&str], _state: &mut ()) -> ExecuteResult {
        Self::check_keyword(&self, tokens, &self.kw)?;
        Ok(CommandEffect::Callback(self.callback))
    }

    fn get_suggester(&self) -> Box<dyn Suggest> {
        Box::new(SingleKwCmd::new(&self.kw))
    }
}
