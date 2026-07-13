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
def help : Flag<["--"], "help">, HelpText<"displays this help text">;

// Bare '--statistics' enables reporting of all statistics, while
// '--statistics=<a>,<b>,...' restricts reporting to the named statistics.
def statistics_flag : Flag<["--"], "statistics">,
                      HelpText<"Report all statistics gathered during fuzzing">;
def statistics : CommaJoined<["--"], "statistics=">,
                 MetaVarName<"<statistics>">,
                 HelpText<"Report only the given comma-separated statistics">;

def grp_oracle : OptionGroup<"Oracle options">;

def functional : Flag<["--"], "functional">,
                 HelpText<"Switch to functional testing">,
                 Group<grp_oracle>;
def non_functional : Flag<["--"], "non-functional">,
                     HelpText<"Switch to non-functional testing">,
                     Group<grp_oracle>;
