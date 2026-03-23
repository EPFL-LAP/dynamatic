include "llvm/Option/OptParser.td"

class F<string letter, string help> : Flag<["-"], letter>, HelpText<help>;

def target : Separate<["--"], "target">, MetaVarName<"<target>">;
def num_threads : JoinedOrSeparate<["-"], "j">,
                  MetaVarName<"<num-threads>">,
                  HelpText<"Number of threads to use">;
def help : Flag<["--"], "help">, HelpText<"displays this help text">;

def grp_oracle : OptionGroup<"Oracle options">;

def functional : Flag<["--"], "functional">,
                 HelpText<"Switch to functional testing">,
                 Group<grp_oracle>;
def non_functional : Flag<["--"], "non-functional">,
                     HelpText<"Switch to non-functional testing">,
                     Group<grp_oracle>;
