# elastic-miter
### A tool to create an elastic-miter circuit


This creates an elastic-miter module in the Handshake dialect using two MLIR files as input. Each file must contain exactly one module, and each module must include a single handshake.func. The generated miter MLIR file and JSON config file will be saved in the specified output directory.


#### Usage

elastic-miter --lhs=\<lhs-file-path\> --rhs=\<lhs-file-path\> -o \<out-dir\> --bufferSlots=\<nr-of-slots\>

##### Command-line options

```bash
--lhs=<string>	     Specify the left-hand side (LHS) input file
--rhs=<string>       Specify the right-hand side (RHS) input file
-o <string>	         Specify the output directory
--bufferSlots=<uint> Specify the number of slots in the decouling buffers
```


##### Example

```bash
elastic-miter --lhs=experimental/tools/elastic-miter-generator/rewrites/a_lhs.mlir --rhs=experimental/tools/elastic-miter-generator/rewrites/a_rhs.mlir -o experimental/tools/elastic-miter-generator/out/comp --bufferSlots 4
```

