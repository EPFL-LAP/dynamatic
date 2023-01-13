use std::{cmp::Ordering, collections::HashMap, fmt::Display};

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum Dialect {
    Source,
    Scf,
    Affine,
    Std,
    Handshake,
}

impl Dialect {
    pub fn get_steps_to(self, dst: Self) -> Vec<CompileStep> {
        let from_level = self.get_idx();
        let to_level = dst.get_idx();
        match to_level {
            to if to < from_level => Vec::new(),
            to if to == from_level => vec![CompileStep::Transformation(self)],
            _ => {
                let mut conversions = Vec::new();

                for idx in from_level..to_level {
                    conversions.push(CompileStep::Transformation(DIALECT_ORDER[idx]));
                    conversions.push(CompileStep::Conversion(
                        DIALECT_ORDER[idx],
                        DIALECT_ORDER[idx + 1],
                    ))
                }
                conversions.push(CompileStep::Transformation(DIALECT_ORDER[to_level]));
                conversions
            }
        }
    }

    pub fn get_short_name(&self) -> &'static str {
        DIALECT_NAMES[self.get_idx()]
    }

    pub fn from_string(input: &str) -> Option<Self> {
        if let Some(idx) = DIALECT_NAMES.iter().position(|dialect| *dialect == input) {
            Some(DIALECT_ORDER[idx])
        } else {
            None
        }
    }

    fn get_idx(&self) -> usize {
        DIALECT_ORDER.iter().position(|d| d == self).unwrap()
    }
}

impl Display for Dialect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dialect_str = match &self {
            Self::Source => "C/C++",
            Self::Scf => "SCF",
            Self::Affine => "Affine",
            Self::Std => "Standard",
            Self::Handshake => "Handshake",
        };
        write!(f, "{}", dialect_str)
    }
}

impl PartialOrd for Dialect {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let idx1 = self.get_idx();
        let idx2 = other.get_idx();
        if idx1 < idx2 {
            Some(Ordering::Less)
        } else if idx1 == idx2 {
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

pub const DIALECT_ORDER: [Dialect; 5] = [
    Dialect::Source,
    Dialect::Scf,
    Dialect::Affine,
    Dialect::Std,
    Dialect::Handshake,
];

pub const DIALECT_NAMES: [&str; 5] = ["c_cpp", "scf", "affine", "std", "handshake"];

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
                if let Some(idx_src) = DIALECT_NAMES.iter().position(|dialect| *dialect == src) {
                    if let Some(idx_dst) = DIALECT_NAMES.iter().position(|dialect| *dialect == dst)
                    {
                        if idx_src == idx_dst - 1 {
                            return Some(CompileStep::Conversion(
                                DIALECT_ORDER[idx_src],
                                DIALECT_ORDER[idx_dst],
                            ));
                        }
                    }
                }
                None
            }
            None => {
                if let Some(idx) = DIALECT_NAMES.iter().position(|dialect| *dialect == input) {
                    Some(CompileStep::Transformation(DIALECT_ORDER[idx]))
                } else {
                    None
                }
            }
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
        for step in self.source.get_steps_to(self.destination).into_iter() {
            match self.steps.get(&step) {
                Some(value) => new_steps_opt.insert(step, value.clone()),
                None => new_steps_opt.insert(step, Vec::new()),
            };
        }
        self.steps = new_steps_opt
    }

    fn generate_steps(source: Dialect, destination: Dialect) -> HashMap<CompileStep, Vec<String>> {
        let mut steps_opt = HashMap::new();
        for step in source.get_steps_to(destination).into_iter() {
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
