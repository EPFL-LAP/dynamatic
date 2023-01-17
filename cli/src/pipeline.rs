use std::{cmp::Ordering, collections::HashMap, fmt::Display};

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum Dialect {
    Source = 0,
    Scf = 1,
    Affine = 2,
    Std = 3,
    Handshake = 4,
}

impl Dialect {
    pub fn get_steps_to(&self, dst: &Self) -> Vec<CompileStep> {
        let from_level = *self as usize;
        let to_level = *dst as usize;
        match to_level {
            to if to < from_level => Vec::new(),
            to if to == from_level => vec![CompileStep::Transformation(*self)],
            _ => {
                let mut conversions = Vec::new();

                for idx in from_level..to_level {
                    conversions.push(CompileStep::Transformation(DIALECTS[idx].dialect));
                    conversions.push(CompileStep::Conversion(
                        DIALECTS[idx].dialect,
                        DIALECTS[idx + 1].dialect,
                    ))
                }
                conversions.push(CompileStep::Transformation(DIALECTS[to_level].dialect));
                conversions
            }
        }
    }

    pub fn get_short_name(&self) -> &'static str {
        DIALECTS[*self as usize].name
    }

    pub fn from_string(input: &str) -> Option<Self> {
        if let Some(idx) = DIALECTS.iter().position(|info| info.name == input) {
            Some(DIALECTS[idx].dialect)
        } else {
            None
        }
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
    Conversion(Dialect, Dialect),
}

impl CompileStep {
    pub fn get_short_name(&self) -> String {
        match self {
            Self::Transformation(d) => d.get_short_name().to_string(),
            Self::Conversion(from, to) => {
                format!("{}_to_{}", from.get_short_name(), to.get_short_name())
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
                            return Some(CompileStep::Conversion(
                                DIALECTS[idx_src].dialect,
                                DIALECTS[idx_dst].dialect,
                            ));
                        }
                    }
                }
                None
            }
            None => Dialect::from_string(input).map_or(None, |d| Some(Self::Transformation(d))),
        }
    }
}

impl Display for CompileStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transformation(d) => write!(f, "{}", d),
            Self::Conversion(from, to) => write!(f, "{} -> {}", from, to),
        }
    }
}

impl PartialOrd for CompileStep {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self {
            Self::Transformation(t1) => match other {
                Self::Transformation(t2) => t1.partial_cmp(t2),
                Self::Conversion(_, _) => Some(Ordering::Less),
            },
            Self::Conversion(c1_from, _) => match other {
                Self::Transformation(_) => Some(Ordering::Greater),
                Self::Conversion(c2_from, _) => c1_from.partial_cmp(c2_from),
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
    steps: HashMap<CompileStep, Vec<String>>,
}

impl PipelineState {
    pub fn new(source: Dialect, destination: Dialect) -> Self {
        Self {
            source,
            destination,
            steps: Self::generate_steps(source, destination),
        }
    }

    pub fn get_steps(&self) -> &HashMap<CompileStep, Vec<String>> {
        &self.steps
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
        if self.steps.contains_key(&step) {
            self.steps.insert(step, args).unwrap()
        } else {
            panic!(
                "Compile step \"{}\" is invalid in pipeline from {} to {}",
                step, self.source, self.destination
            )
        }
    }

    fn update_steps(&mut self) {
        let mut new_steps_opt = HashMap::new();
        for step in self.source.get_steps_to(&self.destination).into_iter() {
            match self.steps.get(&step) {
                Some(value) => new_steps_opt.insert(step, value.clone()),
                None => new_steps_opt.insert(step, Vec::new()),
            };
        }
        self.steps = new_steps_opt
    }

    fn generate_steps(source: Dialect, destination: Dialect) -> HashMap<CompileStep, Vec<String>> {
        let mut steps_opt = HashMap::new();
        for step in source.get_steps_to(&destination).into_iter() {
            steps_opt.insert(step, Vec::new());
        }
        steps_opt
    }
}

impl Display for PipelineState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut steps: Vec<&CompileStep> = self.steps.keys().collect();
        steps.sort();

        write!(
            f,
            "Source type: {}\nDestination type: {}\nCompile steps:\n",
            self.source, self.destination
        )?;
        for s in steps {
            write!(f, "\t{}: {}\n", s, self.steps.get(s).unwrap().join(" "))?;
        }
        Ok(())
    }
}

pub struct DialectInfo {
    dialect: Dialect,
    name: &'static str,
}

impl DialectInfo {
    #[inline]
    pub fn get_dialect(&self) -> &Dialect {
        &self.dialect
    }

    #[inline]
    pub fn get_name(&self) -> &'static str {
        &self.name
    }
}

pub const DIALECTS: [DialectInfo; 5] = [
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
    pub args: &'static str,
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
    pub fn get_args(&self) -> &'static str {
        &self.args
    }
}

pub const COMPILE_STEPS: [CompileStepInfo; 8] = [
    CompileStepInfo {
        step: CompileStep::Conversion(Dialect::Source, Dialect::Scf),
        binary: "cgeist",
        args: "-S -O3",
    },
    CompileStepInfo {
        step: CompileStep::Transformation(Dialect::Scf),
        binary: "polygeist-opt",
        args: "",
    },
    CompileStepInfo {
        step: CompileStep::Conversion(Dialect::Scf, Dialect::Affine),
        binary: "polygeist-opt",
        args: "-raise-scf-to-affine",
    },
    CompileStepInfo {
        step: CompileStep::Transformation(Dialect::Affine),
        binary: "polygeist-opt",
        args: "",
    },
    CompileStepInfo {
        step: CompileStep::Conversion(Dialect::Affine, Dialect::Std),
        binary: "mlir-opt",
        args: "-lower-affine -convert-scf-to-cf",
    },
    CompileStepInfo {
        step: CompileStep::Transformation(Dialect::Std),
        binary: "mlir-opt",
        args: "",
    },
    CompileStepInfo {
        step: CompileStep::Conversion(Dialect::Std, Dialect::Handshake),
        binary: "circt-opt",
        args: "--flatten-memref --flatten-memref-calls --lower-std-to-handshake",
    },
    CompileStepInfo {
        step: CompileStep::Transformation(Dialect::Handshake),
        binary: "circt-opt",
        args: "",
    },
];
