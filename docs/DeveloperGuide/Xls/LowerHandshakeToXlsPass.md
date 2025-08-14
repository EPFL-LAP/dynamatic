# Lower Handshake to XLS

## Overview

The experimental `--lower-handshake-to-xls` pass is and
exploratory/proof-of-concept alternative backend for Dynamatic, that converts a 
handshake function into a network of XLS "procs" connected by XLS channels.

This network can then be elaborated, converted to XLS IR, and synthesized into
Verilog.

The rough flow is as follows:
```bash
# Convert final handshake to XLS MLIR:
dynamatic-xls-opt --lower-handshake-to-xls handshake_export.mlir > sprocs.mlir

# Elaborate XLS MLIR:
xls_opt --elaborate-procs --instantiate-eprocs --symbol-dce sprocs.mlir >  procs.mlir

# Convert XLS MLIR to XLS IR:
xls_translate --mlir-xls-to-xls procs.mlir --main-function="NAME_OF_TOP_PROC_IN_PROCS_MLIR" > proc.ir

# Optimize:
opt_main proc.ir > proc.opt.ir

# Codegen:
codegen_main proc.opt.ir \
    --multi_proc \
    --delay_model=asap7 \
    --pipeline_stages=1 \
    --reset="rst" \
    --materialize_internal_fifos \
    --flop_inputs=true \
    --flop_inputs_kind=zerolatency \
    --flop_outputs=true \
    --flop_outputs_kind=zerolatency \
    --use_system_verilog=false > final.v
```


Note that the XLS MLIR dialect features a higher-level representation of XLS procs than the
normal XLS IR, called "structural procs" or "sprocs". These make it much simpler to define
and manipulate hierarchical networks of procs. The `--lower-handshake-to-xls` pass
emits such sprocs, requiring `xls_opt`'s `--elaborate-procs` and `--instantiate-eprocs`
to convert the MLIR into a form that can be translated to XLS IR.

## Implementation

The pass is roughly similar structure to the RTL export of Dynamatic. Since there
are no parametric procs in XLS IR, a C++-based code emitter generates
proc definitions of all required handshake unit parametertrizations. These are
then instantiated and connect in a top proc using XLS channels.

Buffers are not converted to XLS procs, but rather modify the properties of the 
XLS channels they are replaced by.

## Limitations

Note that this is not intended as a working Dynamatic backend, but rather as an
exploration of XLS inter-op. Only a subset of handshake ops are supported, and
the code is not well tested.

XLS also does not provide fine enough per-proc pipelining control to guarantee
that all procs behave equivalent to the verilog/VHDL implementations in terms 
of latency and transparency.

Dynamatic's `LazyForkOp` **cannot** be represented as an XLS proc, since the later
does not allow a proc to check if an output is ready without sending.

XLS supports floating point operations, but currently no floating point
handhsake units are converted: In XLS, at the IR level, there is no notion of
floating point arithmetic, and all floating point operations are implemented
using a large network of integer/basic ops by the DSLX frontend. This makes
writing the parametric emitter for these ops not any more difficult, but 
certainly much more verbose and annoying.

## Known Issues

Blows up if an SSA value is used before it is defined, making loops impossible:

```mlir
module {
  handshake.func @foo() -> (!handshake.channel<i3>) attributes {argNames = [], resNames = ["out0"]} {
    %1 = constant %0 {handshake.bb = 1 : ui32, value = 3 : i3} : <>, <i3>
    %0 = source {handshake.bb = 1 : ui32} : <>
    end {handshake.bb = 1 : ui32} %1 : <i3>
  }
}
```

(Did I mention this was a half-backed proof of concept?)
