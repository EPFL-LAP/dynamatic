# elastic-miter: A tool to compare two MLIR circuits in the Handshake dialect

The tool compares two MLIR circuits in the handshake dialect by constructing an Elastic-Miter circuit, formally verfiying their (non-)equivalence.

## Citation

Ayatallah Elakhras, Jiahui Xu, Martin Erhart, Paolo Ienne, and Lana Josipović. 2025. ElasticMiter: Formally Verified Dataflow Circuit Rewrites. In *Proceedings of the 30th ACM International Conference*

## Setup

By default the tool uses NuSMV for the verification. The official version only supports printing 2^16 state spaces. To circumvent this problem, we provide a modified binary supporting 2^24 states.

Setting the `--enable-leq-binaries` flag when building Dynamatic enables the automatic download of the modified binary:

```
$ ./build.sh --enable-leq-binaries -f
```

Don't forget to add the `-f` flag to force CMake to re-run if you've already built Dynamatic.

Currently, the conversion to SMV requires the [dot2smv](https://github.com/Jiahui17/dot2smv) converter (soon to be replaced by a dedicated SMV backend). It is also downloaded when `--enable-leq-binaries` is used.

The binary will be located at `bin/elastic-miter`.

## Usage

The tool supports following options:
```bash
$ ./bin/elastic-miter --lhs=<lhs-file-path> --rhs=<lhs-file-path> -o <out-dir> [--loop=<string>] [--loop_strict=<string>] [--seq_length=<string>] [--token_limit=<string>] [--cex]
```

Below is a description of the command-line options:

```
--lhs=<string>          Specify the left-hand side (LHS) input file
--rhs=<string>          Specify the right-hand side (RHS) input file
-o <string>             Specify the output directory
--loop=<string>         Specify a Loop Condition sequence contraint (explained later). Can be used multiple times.
--loop_strict=<string>  Specify a Strict Loop Condition sequence contraint. Can be used multiple times.
--seq_length=<string>   Specify a Sequence Length Relation constraint (explained later). Can be used multiple times.
--token_limit=<string>  Specify a Token Limit constraint (explained later). Can be used multiple times.
--cex                   Enable counterexamples.
```


## Sequence constraints

Since the compared circuits are usually part of a larger circuit, they do not need to be equivalent under all circumstances, but rather only under specific *contexts*. These context are modeled using constraints on the sequence generators. 


### Sequence Length Relation
A Sequence Length Relation constraint controls the relative length of the input sequences.
The constraint has the form of an arithmetic equation. The number in the equation will be replaced the respective input with the index of the number.  
Example:  
`--seq_length="0+1=2"` will ensure that the inputs with index 0 and index 1 together produce as many tokens as the input with index 2.


### Loop Condition
A Loop Condition sequence contraint has the form `<dataSequence>,<controlSequence>`.
The number of tokens in the input with the index `dataSequence` is equivalent to the number of false tokens at the output with the index controlSequence.  
Example:  
`--loop="0,1"`

### Token Limit
A Token Limit constraint has the form `<inputSequence>,<outputSequence>,<limit>`.
At any point in time, the number of tokens which are created at the input with index `inputSequence` can only be up to `limit` higher than the number of tokens reaching the output with the index `outputSequence`.  
Example:  
`--token_limit="1,1,2"` ensures that there are only two tokens in the circuit which enter at the input with index 1 and leave at the output with index 1.

## Examples

Example rewrites are available in the `rewrites` folder. You can test all of them using the `prove-rewrites.sh` script:

```bash
$ experimental/tools/elastic-miter/prove-rewrites.sh
```

Alternatively, you can directly invoke the `elastic-miter` binary as follows:

```bash
$ OUT_DIR="experimental/tools/elastic-miter/out"
$ REWRITES="experimental/test/tools/elastic-miter/rewrites"
$ ./bin/elastic-miter --lhs=$REWRITES/b_lhs.mlir --rhs=$REWRITES/b_rhs.mlir -o $OUT_DIR --seq_length="0+1=3" --seq_length="0=2" --loop_strict=0,1
```
