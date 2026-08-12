include "llvm/Option/OptParser.td"

class F<string letter, string help> : Flag<["-"], letter>, HelpText<help>;

// Options that influence the program that is generated and how it is verified.
// Every option in this group (directly or through a nested group) is captured
// in a reproducer and replayed by '--reproduce', so adding a new option here is
// all that is needed for it to become part of reproducers.
def grp_target : OptionGroup<"Target options">;

def target : Separate<["--"], "target">, MetaVarName<"<target>">,
             Group<grp_target>;
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
def single_process : Flag<["--"], "single-process">,
              HelpText<"Generate and verify every program on a thread of this "
                       "process rather than in a process of its own. Fuzzing "
                       "stops as soon as one of them crashes">;
// Internal option used by the fuzzer to re-execute itself for every program it
// generates and verifies, isolating it in a process of its own. Hidden from the
// help text as it is not meant to be used directly; '--single-process' is what
// does the same in this process.
def run_as_worker_subprocess : Separate<["--"], "run-as-worker-subprocess">,
                               MetaVarName<"<dir>">,
                               Flags<[HelpHidden]>;
// Internal option that generates a program but skips its verification. Only the
// generation is exercised, which lets tests check that a program can be
// regenerated from its reproducer without running the heavy verification flow.
// Hidden as it is not meant for normal use. Deliberately not a target option:
// it does not influence the generated program and is therefore not part of a
// reproducer.
def no_verify : Flag<["--"], "no-verify">, Flags<[HelpHidden]>;
def reproduce : Separate<["--"], "reproduce">, MetaVarName<"<dir>">,
                HelpText<"Regenerate and verify, in this very process, the exact "
                         "program described by the 'reproducer.json' in <dir>. "
                         "Used to reproduce and debug a crash of the generation "
                         "or verification">;
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

def grp_oracle : OptionGroup<"Oracle options">, Group<grp_target>;

def functional : Flag<["--"], "functional">,
                 HelpText<"Switch to functional testing">,
                 Group<grp_oracle>;
def non_functional : Flag<["--"], "non-functional">,
                     HelpText<"Switch to non-functional testing">,
                     Group<grp_oracle>;
