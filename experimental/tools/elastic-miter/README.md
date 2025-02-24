# elastic-miter
### A tool to compare two MLIR circuits in the Handshake dialect


This compares two MLIR circuits in the handshake dialect by constructing an elastic-miter circuits. (TODO add link to paper).

TODO how much should I explain how the elastic-miter works here?

<!-- This creates an elastic-miter module in the Handshake dialect using two MLIR files as input. Each file must contain exactly one module, and each module must include a single handshake.func. The generated miter MLIR file and JSON config file will be saved in the specified output directory. -->


### Usage
By default the tool uses NuSMV for the verification. The NuSMV needs to be reachable on the path. (TODO talk about modified NuSMV here).



```bash
elastic-miter --lhs=<lhs-file-path> --rhs=<lhs-file-path> -o <out-dir> [--loop=<string>] [--loop_strict=<string>] [--seq_length=<string>] [--token_limit=<string>] [--cex]
```


#### Command-line options

```bash
--lhs=<string>	     Specify the left-hand side (LHS) input file
--rhs=<string>       Specify the right-hand side (RHS) input file
-o <string>	         Specify the output directory
--loop=<string>
```


#### Sequence constraints

Since the compared circuits are usually part of a larger circuit, they do not need to be equivalent under all circumstances, but rather only under specific *contexts*. These context are modeled using constraints on the sequence generators. 

##### Sequence Length Relation

``dsad``
##### Loop Condition
The number of tokens in the
input with the index dataSequence is equivalent to the number of false tokens
at the output with the index controlSequence. If lastFalse is set, the last
token in the controlSequence needs to be false.

``dsad``

##### Token Limit
The number of tokens at the input
// with index inputSequence can only be up to "limit" higher than the number of
// tokens at the ouput with the index outputSequence.
``dsad``

#### Example
TODO
```bash
OUT_DIR="experimental/tools/elastic-miter/out"
REWRITES="experimental/test/tools/elastic-miter/rewrites"
elastic-miter --lhs=$REWRITES/b_lhs.mlir --rhs=$REWRITES/b_rhs.mlir -o $OUT_DIR --seq_length="0+1=3" --seq_length="0=2" --loop_strict=0,1
```

