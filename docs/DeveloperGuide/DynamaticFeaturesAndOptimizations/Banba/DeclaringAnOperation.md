# Declaring an Operation

Banba is an LSQ generator. It describes an LSQ in its own MLIR dialect, `banba`, before lowering it to hardware. This page explains how an operation of that dialect is declared, using the `lsq` operation as the example.

All the files mentioned on this page are in `tools/backend/lsq-generator-banba`.

## What an Operation Is

An operation of the `banba` dialect is a hardware component. Its inputs and outputs are the ports of the component.

The `lsq` operation is the outermost component: the whole LSQ. Its inputs and outputs are the LSQ's ports to the circuit and its channels to the memory.

## Where Operations Are Declared

Operations are declared in `BanbaOps.td`, in [TableGen](https://mlir.llvm.org/docs/DefiningDialects/Operations/). From each declaration, the build generates a C++ class.

This is the declaration of the `lsq` operation, showing only its name and some of its inputs and outputs:

```tablegen
def LsqOp : Op<Banba_Dialect, "lsq"> {
  let arguments = (ins
      ClockType:$clk,
      I1:$rst,
      ArrayI1:$group_init_valid,
      ArrayInt:$ld_port_addr,
      ...
      Int:$wr_resp_id);
  let results = (outs
      ArrayI1:$group_init_ready,
      ...
      I1:$empty);
}
```

The first line gives the operation its name in the IR, `banba.lsq`, and the name of its C++ class, `banba::LsqOp`.

## Inputs and Outputs

`arguments` lists the inputs and `results` lists the outputs. Each line is a kind of type, followed by the name of the input or output.

The kinds of type are declared at the top of `BanbaOps.td`:

| Kind | Meaning |
|---|---|
| `I1` | A single bit. |
| `Int` | An integer of any width. |
| `ArrayInt` | A CIRCT hardware array of integers. |
| `ArrayI1` | A CIRCT hardware array of single bits. |
| `ClockType` | CIRCT's clock type. |

An array is used for a set of ports of the same kind. For example, `ld_port_addr` is one array that holds one address per load port.

These kinds are deliberately loose: they say that an input is an integer, but not how wide it is. The exact widths and lengths depend on the LSQ's configuration.
