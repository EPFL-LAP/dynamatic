include "llvm/Option/OptParser.td"

class F<string letter, string help> : Flag<["-"], letter>, HelpText<help>;

def target : Separate<["--"], "target">, MetaVarName<"<target>">;
def num_threads : JoinedOrSeparate<["-"], "j">,
    MetaVarName<"<num-threads>">,
    HelpText<"Number of threads to use">;
def help : Flag<["--"], "help">, HelpText<"displays this help text">;
