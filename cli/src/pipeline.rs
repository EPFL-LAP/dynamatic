use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::{format, Display},
    path::Path,
};

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum Dialect {
    Source = 0,
    Scf = 1,
    Affine = 2,
    Std = 3,
    Handshake = 4,
}

impl Dialect {
    pub const N_DIALECTS: usize = 5;

    pub fn from_string(input: &str) -> Option<Self> {
        if let Some(idx) = DIALECTS.iter().position(|info| info.name == input) {
            Some(DIALECTS[idx].dialect)
        } else {
            None
        }
    }

    pub fn get_steps_to(&self, dst: &Self) -> Vec<CompileStep> {
        let from_idx = *self as usize;
        match *dst as usize {
            to_idx if to_idx < from_idx => Vec::new(),
            to_idx if to_idx == from_idx => vec![CompileStep::Transformation(*self)],
            to_idx => {
                let mut conversions = Vec::new();

                for idx in from_idx..to_idx {
                    if idx != 0 {
                        // No transformation on C/C++ (source dialect)
                        conversions.push(CompileStep::Transformation(DIALECTS[idx].dialect));
                    }
                    conversions.push(CompileStep::Lowering(DIALECTS[idx].dialect))
                }
                conversions.push(CompileStep::Transformation(DIALECTS[to_idx].dialect));
                conversions
            }
        }
    }

    pub fn get_lower(&self) -> Option<Self> {
        let idx = *self as usize;
        if idx == Self::N_DIALECTS - 1 {
            None
        } else {
            Some(DIALECTS[idx + 1].dialect)
        }
    }

    #[inline]
    pub fn get_info(&self) -> &DialectInfo {
        &DIALECTS[*self as usize]
    }
}

impl Display for Dialect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", DIALECTS[*self as usize].name)
    }
}

impl PartialOrd for Dialect {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let idx_self = *self as usize;
        let idx_other = *other as usize;
        if idx_self < idx_other {
            Some(Ordering::Less)
        } else if idx_self == idx_other {
            Some(Ordering::Equal)
        } else {
            Some(Ordering::Greater)
        }
    }
}

impl Ord for Dialect {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum CompileStep {
    Transformation(Dialect),
    Lowering(Dialect),
}

impl CompileStep {
    pub fn get_short_name(&self) -> String {
        match self {
            Self::Transformation(d) => d.get_info().name.to_string(),
            Self::Lowering(d) => {
                format!(
                    "{}_to_{}",
                    d.get_info().name,
                    d.get_lower().unwrap().get_info().name
                )
            }
        }
    }

    pub fn from_string(input: &str) -> Option<Self> {
        match input.find("_to_") {
            Some(idx) => {
                let src = &input[..idx];
                let dst = &input[(idx + 4)..];
                if let Some(idx_src) = DIALECTS.iter().position(|info| info.name == src) {
                    if let Some(idx_dst) = DIALECTS.iter().position(|info| info.name == dst) {
                        if idx_src == idx_dst - 1 {
                            return Some(CompileStep::Lowering(DIALECTS[idx_src].dialect));
                        }
                    }
                }
                None
            }
            None => Dialect::from_string(input).map_or(None, |d| Some(Self::Transformation(d))),
        }
    }

    #[inline]
    pub fn get_info(&self) -> &CompileStepInfo {
        &COMPILE_STEPS[match self {
            Self::Transformation(d) => 2 * ((*d as usize) - 1) + 1,
            Self::Lowering(d) => 2 * (*d as usize),
        }]
    }
}

impl Display for CompileStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transformation(d) => write!(f, "{}", d),
            Self::Lowering(d) => write!(f, "{} -> {}", d, d.get_lower().unwrap()),
        }
    }
}

impl PartialOrd for CompileStep {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self {
            Self::Transformation(t1) => match other {
                Self::Transformation(t2) => t1.partial_cmp(t2),
                Self::Lowering(_) => Some(Ordering::Less),
            },
            Self::Lowering(l1) => match other {
                Self::Transformation(_) => Some(Ordering::Greater),
                Self::Lowering(l2) => l1.partial_cmp(l2),
            },
        }
    }
}

impl Ord for CompileStep {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Clone)]
pub struct PipelineState {
    source: Dialect,
    destination: Dialect,
    steps: Vec<CompileStep>,
    args: HashMap<CompileStep, Vec<String>>,
}

impl PipelineState {
    pub fn new(source: Dialect, destination: Dialect) -> Self {
        let steps = source.get_steps_to(&destination);
        let args = Self::generate_args(&steps);
        Self {
            source,
            destination,
            steps,
            args,
        }
    }

    pub fn get_steps(&self) -> &[CompileStep] {
        &self.steps
    }

    pub fn get_args(&self, step: &CompileStep) -> Option<&Vec<String>> {
        self.args.get(step)
    }

    pub fn set_source(&mut self, source: Dialect) {
        self.source = source;
        self.update_steps();
    }

    pub fn set_destination(&mut self, destination: Dialect) {
        self.destination = destination;
        self.update_steps();
    }

    pub fn configure_step(&mut self, step: CompileStep, args: Vec<String>) -> Vec<String> {
        if self.args.contains_key(&step) {
            self.args.insert(step, args).unwrap()
        } else {
            panic!(
                "Compile step \"{}\" is invalid in pipeline from {} to {}",
                step, self.source, self.destination
            )
        }
    }

    fn update_steps(&mut self) {
        let mut new_steps_opt = HashMap::new();
        self.steps = self.source.get_steps_to(&self.destination);
        for s in self.steps.iter() {
            match self.args.get(&s) {
                Some(value) => new_steps_opt.insert(*s, value.clone()),
                None => new_steps_opt.insert(*s, Vec::new()),
            };
        }
        self.args = new_steps_opt
    }

    fn generate_args(steps: &[CompileStep]) -> HashMap<CompileStep, Vec<String>> {
        let mut args = HashMap::new();
        for s in steps {
            args.insert(*s, Vec::new());
        }
        args
    }
}

impl Display for PipelineState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut steps: Vec<&CompileStep> = self.args.keys().collect();
        steps.sort();

        write!(
            f,
            "Source type: {}\nDestination type: {}\nCompile steps:\n",
            self.source, self.destination
        )?;
        for s in steps {
            write!(f, "\t{}: {}\n", s, self.args.get(s).unwrap().join(" "))?;
        }
        Ok(())
    }
}

pub struct DialectInfo {
    pub dialect: Dialect,
    pub name: &'static str,
}

pub const DIALECTS: [DialectInfo; Dialect::N_DIALECTS] = [
    DialectInfo {
        dialect: Dialect::Source,
        name: "c_cpp",
    },
    DialectInfo {
        dialect: Dialect::Scf,
        name: "scf",
    },
    DialectInfo {
        dialect: Dialect::Affine,
        name: "affine",
    },
    DialectInfo {
        dialect: Dialect::Std,
        name: "std",
    },
    DialectInfo {
        dialect: Dialect::Handshake,
        name: "handshake",
    },
];

pub struct CompileStepInfo {
    pub step: CompileStep,
    pub binary: &'static str,
    pub args: &'static [&'static str],
    pub get_output_file: fn(&str) -> String,
}

impl CompileStepInfo {
    #[inline]
    pub fn get_step(&self) -> &CompileStep {
        &self.step
    }

    #[inline]
    pub fn get_binary(&self) -> &'static str {
        &self.binary
    }

    #[inline]
    pub fn get_args(&self) -> &'static [&'static str] {
        &self.args
    }
}

pub const COMPILE_STEPS: [CompileStepInfo; 8] = [
    CompileStepInfo {
        step: CompileStep::Lowering(Dialect::Source),
        binary: "cgeist",
        args: &["-S", "-O3"],
        get_output_file: |input| gen_filename(input, "scf"),
    },
    CompileStepInfo {
        step: CompileStep::Transformation(Dialect::Scf),
        binary: "polygeist-opt",
        args: &[],
        get_output_file: |input| gen_filename(input, "scf_opt"),
    },
    CompileStepInfo {
        step: CompileStep::Lowering(Dialect::Scf),
        binary: "polygeist-opt",
        args: &["-raise-scf-to-affine"],
        get_output_file: |input| gen_filename(input, "affine"),
    },
    CompileStepInfo {
        step: CompileStep::Transformation(Dialect::Affine),
        binary: "polygeist-opt",
        args: &[],
        get_output_file: |input| gen_filename(input, "affine_opt"),
    },
    CompileStepInfo {
        step: CompileStep::Lowering(Dialect::Affine),
        binary: "mlir-opt",
        args: &["-lower-affine", "-convert-scf-to-cf"],
        get_output_file: |input| gen_filename(input, "std"),
    },
    CompileStepInfo {
        step: CompileStep::Transformation(Dialect::Std),
        binary: "mlir-opt",
        args: &[],
        get_output_file: |input| gen_filename(input, "std_opt"),
    },
    CompileStepInfo {
        step: CompileStep::Lowering(Dialect::Std),
        binary: "circt-opt",
        args: &[
            "--flatten-memref",
            "--flatten-memref-calls",
            "--lower-std-to-handshake",
        ],
        get_output_file: |input| gen_filename(input, "handshake"),
    },
    CompileStepInfo {
        step: CompileStep::Transformation(Dialect::Handshake),
        binary: "circt-opt",
        args: &[],
        get_output_file: |input| gen_filename(input, "handshake_opt"),
    },
];

fn gen_filename(input: &str, step: &str) -> String {
    format!(
        "{}.{step}.mlir",
        Path::new(input)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string()
    )
}
