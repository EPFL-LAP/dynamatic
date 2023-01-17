use crate::{
    command::{
        handle_prompt_result, Command, CommandEffect, ExecuteError, ExecuteResult, PromptError,
    },
    completer::CommandCompleter,
    pipeline::{CompileStep, Dialect, PipelineState, DIALECTS},
    suggest::{
        lcp_match, OneChoiceArgCmd, OneOpaqueArgCmd, SingleKwCmd, Suggest, SuggestResult,
        Suggestion, TokenMatch, NO_SUGGESTION,
    },
};
use inquire::Text;
use std::{fmt::Display, process::exit, vec};

pub fn compile() {
    let mut state = CompileState::new();
    loop {
        let commands = get_compile_commands(&state);
        let completer =
            CommandCompleter::new(commands.iter().map(|cmd| cmd.get_suggester()).collect());
        let prompt = Text::new("[compile]")
            .with_help_message(&state.to_string())
            .with_autocomplete(completer)
            .prompt();

        match handle_prompt_result(&prompt, &commands, &mut state) {
            Ok(effect) => match effect {
                CommandEffect::State(opt_msg) => {
                    if let Some(msg) = opt_msg {
                        println!("{msg}");
                    }
                }
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
struct CompileState {
    source_filepath: Option<String>,
    pipeline: PipelineState,
}

dyn_clone::clone_trait_object!(Command<CompileState>);

impl CompileState {
    fn new() -> Self {
        Self {
            source_filepath: None,
            pipeline: PipelineState::new(Dialect::Source, Dialect::Handshake),
        }
    }
}

impl Display for CompileState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let src_name = match &self.source_filepath {
            Some(fp) => fp.clone(),
            None => String::new(),
        };

        write!(f, "\nSource file: {}\n{}", src_name, self.pipeline)
    }
}

fn get_compile_commands(state: &CompileState) -> Vec<Box<dyn Command<CompileState>>> {
    let mut root_suggestions: Vec<Box<dyn Command<CompileState>>> = vec![
        Box::new(OptCmd::new(&state.pipeline)),
        Box::new(SourceCmd::new()),
        Box::new(PipelineEndpointCmd::new(true)),
        Box::new(PipelineEndpointCmd::new(false)),
        Box::new(PipelineConfigCmd::new(true)),
        Box::new(PipelineConfigCmd::new(false)),
    ];
    if let Some(_) = state.source_filepath {
        root_suggestions.push(Box::new(RunCmd::new()))
    }
    root_suggestions
}

/// Command to add optimization passes at different levels of the compile pipeline.
/// Syntax:
///     opt <compile_step_name>[ <pass_name>]+
/// Possible <compile_step_name>'s depend on the current pipeline_state
/// pass_name is opaque
#[derive(Clone)]
struct OptCmd {
    steps: Vec<String>,
    state: OptCmdState,
}

#[derive(Clone)]
enum OptCmdState {
    /// Matching the command keyword
    MatchKw,
    // Matching the compile step
    MatchStep,
    // Matching any number of passes
    MatchPass(String, Vec<String>),
}

impl OptCmd {
    const KW: &str = "opt";
    const STEP: &str = "<compile_step_name>";
    const PASSES: &str = "<pass_name>";

    fn new(pipeline_state: &PipelineState) -> Self {
        Self {
            state: OptCmdState::MatchKw,
            steps: pipeline_state
                .get_steps()
                .keys()
                .map(|step| step.get_short_name())
                .collect(),
        }
    }

    #[inline]
    fn get_kw_suggestion(&self) -> Suggestion {
        Suggestion::first_token(Self::KW, &format!("{} {}", Self::STEP, Self::PASSES), true)
    }

    #[inline]
    fn get_step_suggestion(&self, step: &str) -> Suggestion {
        Suggestion::new(Self::KW, step, Self::PASSES, true)
    }
}

impl Suggest for OptCmd {
    fn update(&mut self, token: &str) -> SuggestResult {
        match &mut self.state {
            OptCmdState::MatchKw => match lcp_match("opt", token) {
                TokenMatch::Full => {
                    self.state = OptCmdState::MatchStep;
                    (TokenMatch::Full, vec![self.get_kw_suggestion()])
                }
                TokenMatch::Partial => (TokenMatch::Partial, vec![self.get_kw_suggestion()]),
                TokenMatch::None => NO_SUGGESTION,
            },
            OptCmdState::MatchStep => {
                let matched_steps: Vec<(TokenMatch, &String)> = self
                    .steps
                    .iter()
                    .map(|s| (lcp_match(s, token), s))
                    .filter(|(token_match, _)| match token_match {
                        TokenMatch::None => false,
                        _ => true,
                    })
                    .collect();

                if matched_steps.is_empty() {
                    NO_SUGGESTION
                } else if matched_steps.len() == 1 {
                    let step = matched_steps[0].1;
                    self.state = OptCmdState::MatchPass(step.clone(), Vec::new());
                    (matched_steps[0].0, vec![self.get_step_suggestion(&step)])
                } else {
                    let matches: Vec<TokenMatch> = matched_steps.iter().map(|(m, _)| *m).collect();
                    let max_match = TokenMatch::max(&matches);
                    if let TokenMatch::Full = max_match {
                        self.state = OptCmdState::MatchPass(token.to_string(), Vec::new());
                    }
                    (
                        max_match,
                        matched_steps
                            .iter()
                            .map(|(_, o)| self.get_step_suggestion(&o))
                            .collect(),
                    )
                }
            }
            OptCmdState::MatchPass(step, passes) => {
                let sugg_previous = {
                    let maybe_space = if passes.is_empty() { "" } else { " " };
                    format!("{} {}{}{}", Self::KW, step, maybe_space, passes.join(" "))
                };

                passes.push(token.to_string());
                (
                    TokenMatch::Full,
                    vec![Suggestion::last_token(&sugg_previous, Self::PASSES, false)],
                )
            }
        }
    }

    fn reset(&mut self) {
        self.state = OptCmdState::MatchKw;
    }
}

impl Command<CompileState> for OptCmd {
    fn execute(&self, tokens: &[&str], state: &mut CompileState) -> ExecuteResult {
        Self::check_keyword(&self, tokens, Self::KW)?;

        // Check whether there are enough arguments
        if tokens.len() < 2 {
            return Err(ExecuteError::ErrArgument(format!(
                "Expected at least 1 argument, got {}",
                tokens.len() - 1
            )));
        }

        // Check whether the step name is correct
        let legal_steps: Vec<String> = state
            .pipeline
            .get_steps()
            .keys()
            .map(|step| step.get_short_name())
            .collect();
        if let None = legal_steps.iter().find(|step| step == &tokens[1]) {
            return Err(ExecuteError::ErrArgument(format!(
                "Illegal step name provided. Expected one of {{{}}}, but got \"{}\"",
                legal_steps.join(", "),
                tokens[1]
            )));
        }

        // Command is correct, update the optimization step with remaining tokens
        state.pipeline.configure_step(
            CompileStep::from_string(tokens[1]).unwrap(),
            Vec::from_iter(tokens[2..].into_iter().map(|tok| tok.to_string())),
        );

        Ok(CommandEffect::State(Some(format!(
            "Updated compile step \"{}\" to run with arguments \"{}\"",
            tokens[1],
            tokens[2..].join(" ")
        ))))
    }

    fn get_suggester(&self) -> Box<dyn Suggest> {
        let mut suggester = self.clone();
        suggester.reset();
        Box::new(suggester)
    }
}

#[derive(Clone)]
struct SourceCmd {}

impl SourceCmd {
    pub const KW: &str = "source";

    pub fn new() -> Self {
        Self {}
    }
}

impl Command<CompileState> for SourceCmd {
    fn execute(&self, tokens: &[&str], state: &mut CompileState) -> ExecuteResult {
        Self::check_keyword(&self, tokens, Self::KW)?;
        Self::check_num_args(&self, tokens, 1)?;

        // Update source with new filepath
        state.source_filepath = Some(tokens[1].to_string());
        Ok(CommandEffect::State(Some(format!(
            "Set source filepath to {}",
            tokens[1].to_string()
        ))))
    }

    fn get_suggester(&self) -> Box<dyn Suggest> {
        Box::new(OneOpaqueArgCmd::new(Self::KW, "src_file"))
    }
}

#[derive(Clone)]
struct PipelineEndpointCmd {
    is_source: bool,
    kw: &'static str,
}

impl PipelineEndpointCmd {
    pub fn new(is_source: bool) -> Self {
        Self {
            is_source,
            kw: if is_source { "from" } else { "to" },
        }
    }
}

impl Command<CompileState> for PipelineEndpointCmd {
    fn execute(&self, tokens: &[&str], state: &mut CompileState) -> ExecuteResult {
        Self::check_keyword(&self, tokens, self.kw)?;
        Self::check_num_args(&self, tokens, 1)?;

        // Attempt to parse the dialect name and modify the pipeline
        match Dialect::from_string(tokens[1]) {
            Some(dialect) => Ok(CommandEffect::State(if self.is_source {
                state.pipeline.set_source(dialect);
                Some(format!("Updated source dialect to {}", dialect))
            } else {
                state.pipeline.set_destination(dialect);
                Some(format!("Updated destination dialect to {}", dialect))
            })),
            None => {
                let names: Vec<&str> = DIALECTS.iter().map(|info| info.get_name()).collect();
                Err(ExecuteError::ErrArgument(format!(
                    "Illegal dialect name provided. Expected one of {{{}}}, but got {}",
                    names.join(", "),
                    tokens[1],
                )))
            }
        }
    }

    fn get_suggester(&self) -> Box<dyn Suggest> {
        Box::new(OneChoiceArgCmd::new(
            self.kw,
            &DIALECTS
                .iter()
                .map(|info| info.get_name())
                .collect::<Vec<&str>>(),
        ))
    }
}

#[derive(Clone)]
pub struct PipelineConfigCmd {
    is_load: bool,
    kw: &'static str,
}

impl PipelineConfigCmd {
    pub fn new(is_load: bool) -> Self {
        Self {
            is_load,
            kw: if is_load {
                "load_pipeline"
            } else {
                "save_pipeline"
            },
        }
    }
}

impl Command<CompileState> for PipelineConfigCmd {
    fn execute(&self, tokens: &[&str], _state: &mut CompileState) -> ExecuteResult {
        Self::check_keyword(&self, tokens, self.kw)?;
        Self::check_num_args(&self, tokens, 1)?;

        // TODO: does nothing for now, needs to be implemented
        Ok(CommandEffect::State(if self.is_load {
            Some(format!("Loaded pipeline configuration from {}", tokens[1]))
        } else {
            Some(format!("Saved pipeline configuration to {}", tokens[1]))
        }))
    }

    fn get_suggester(&self) -> Box<dyn Suggest> {
        Box::new(OneOpaqueArgCmd::new(self.kw, "config_file"))
    }
}

#[derive(Clone)]
pub struct RunCmd {}

impl RunCmd {
    pub const KW: &str = "run";

    pub fn new() -> Self {
        Self {}
    }
}

impl Command<CompileState> for RunCmd {
    fn execute(&self, tokens: &[&str], _state: &mut CompileState) -> ExecuteResult {
        Self::check_keyword(&self, tokens, Self::KW)?;
        Self::check_num_args(&self, tokens, 0)?;

        // Create script from pipeline configuration

        // TODO: does nothing for now, needs to be implemented
        Ok(CommandEffect::State(Some("Done!".to_string())))
    }

    fn get_suggester(&self) -> Box<dyn Suggest> {
        Box::new(SingleKwCmd::new(Self::KW))
    }
}
