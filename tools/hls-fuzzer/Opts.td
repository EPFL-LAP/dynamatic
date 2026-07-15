include "llvm/Option/OptParser.td"

class F<string letter, string help> : Flag<["-"], letter>, HelpText<help>;

def target : Separate<["--"], "target">, MetaVarName<"<target>">;
def num_threads : JoinedOrSeparate<["-"], "j">,
                  MetaVarName<"<num-threads>">,
                  HelpText<"Number of threads to use">;
def num_programs : Separate<["--"], "num-programs">,
                   MetaVarName<"<num-programs>">,
                   HelpText<"Exit after this many programs have been generated and "
                            "verified (default: run until interrupted)">;
def n : JoinedOrSeparate<["-"], "n">, Alias<num_programs>,
       MetaVarName<"<num-programs>">,
       HelpText<"Alias for --num-programs">;
def inplace : Flag<["--"], "inplace">,
              HelpText<"Generate and verify every program on a thread of this "
                       "process rather than in a process of its own. Fuzzing "
                       "stops as soon as one of them crashes">;
// Internal option used by the fuzzer to re-execute itself for every program it
// generates and verifies, isolating it in a process of its own. Hidden from the
// help text as it is not meant to be used directly; '--inplace' is what does
// the same in this process.
def single_program : Separate<["--"], "single-program">, MetaVarName<"<dir>">,
                     Flags<[HelpHidden]>;
def help : Flag<["--"], "help">, HelpText<"displays this help text">;

// Bare '--statistics' enables reporting of all statistics, while
// '--statistics=<a>,<b>,...' restricts reporting to the named statistics.
def statistics_flag : Flag<["--"], "statistics">,
                      HelpText<"Report all statistics gathered during fuzzing">;
def statistics : CommaJoined<["--"], "statistics=">,
                 MetaVarName<"<statistics>">,
                 HelpText<"Report only the given comma-separated statistics">;

def json_output : Separate<["--"], "json-output">,
                  MetaVarName<"<file>">,
                  HelpText<"Write the test count, bug count, testing rate and "
                           "all requested statistics to <file> as JSON instead "
                           "of reporting them on the console">;

def grp_oracle : OptionGroup<"Oracle options">;

def functional : Flag<["--"], "functional">,
                 HelpText<"Switch to functional testing">,
                 Group<grp_oracle>;
def non_functional : Flag<["--"], "non-functional">,
                     HelpText<"Switch to non-functional testing">,
                     Group<grp_oracle>;
