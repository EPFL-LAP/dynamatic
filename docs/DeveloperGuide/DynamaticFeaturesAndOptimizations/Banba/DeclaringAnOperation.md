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

## Type Verification

The types of an operation's inputs and outputs are verified in two ways.

### By Kind

The declaration gives each input and output a kind of type, such as `Int` or `ArrayI1`. MLIR checks these kinds automatically.

The kinds of type are declared at the top of `BanbaOps.td`:

| Kind | Meaning |
|---|---|
| `I1` | A single bit. |
| `Int` | An integer of any width. |
| `ArrayInt` | A CIRCT hardware array of integers. |
| `ArrayI1` | A CIRCT hardware array of single bits. |
| `ClockType` | CIRCT's clock type. |

An array is used for a set of ports of the same kind. For example, `ld_port_addr` is one array that holds one address per load port.

A kind is loose. For example, `Int` says that an input is an integer, but not how wide it is.

### Against the Config

The exact widths and lengths depend on the LSQ's config. For example, `ld_port_addr` has one address per load port, and the width of an address comes from the config.

Banba defines simple functionality for enforcing the types of an operation based on the config. An operation declares that it uses this functionality by listing the `ExpectedTypes` interface and the `VerifyTypesByConfig` trait. The trait performs the check, each time the IR is verified.

```tablegen
def LsqOp : Op<Banba_Dialect, "lsq",
    [DeclareOpInterfaceMethods<ExpectedTypes>, VerifyTypesByConfig]> {
```

To fulfill the contract the interface specifies, the operation must provide a definition of the `getExpectedOperandTypes` and `getExpectedResultTypes` functions. Each returns one type per input, or per output, in the order in which the declaration lists them.

```tablegen
  let extraClassDefinition = [{
    SmallVector<Type> $cppClass::getExpectedOperandTypes(ConfigAttr config) {
      ...
    }
    SmallVector<Type> $cppClass::getExpectedResultTypes(ConfigAttr config) {
      ...
    }
  }];
```

Banba's `Types` class provides useful helpers for implementing this functionality concisely.

```cpp
/// Builds the types of the operations' inputs and outputs
class Types {
public:
  explicit Types(MLIRContext *context) : context(context) {}

  /// CIRCT's clock type
  Type Clock() const { return ClockType::get(context); }

  /// An integer of the given width
  Type Int(unsigned width) const { return IntegerType::get(context, width); }

  /// A CIRCT hardware array holding `length` integers of the given width
  Type Array(unsigned length, unsigned width) const {
    return circt::hw::ArrayType::get(IntegerType::get(context, width), length);
  }

private:
  MLIRContext *context;
};
```

For example, this is the start of the `lsq` operation's `getExpectedOperandTypes`:

```cpp
SmallVector<Type> $cppClass::getExpectedOperandTypes(ConfigAttr config) {
  Types t(config.getContext());
  unsigned numGroups = config.getNumBBs();
  unsigned numLoadPorts = config.getNumLoadPorts();
  unsigned addrWidth = config.getAddrWidth();
  ...
  return {t.Clock(),                          // clk
          t.Int(1),                           // rst
          t.Array(numGroups, 1),              // group_init_valid
          t.Array(numLoadPorts, addrWidth),   // ld_port_addr
          t.Array(numLoadPorts, 1),           // ld_port_addr_valid
          ...};
}
```

## Region Construction

Banba operations correspond to sub-modules. Each operation is responsible for adding its own internal sub-operations and wiring them together, as well as possibly adding some direct RTL logic.

An operation declares that it does so using the `ConstructsRegion` interface. To meet the contract specified, it must then provide a definition of the `constructRegion` function.

```tablegen
def LsqOp : Op<Banba_Dialect, "lsq",
    [DeclareOpInterfaceMethods<ConstructsRegion>]> {
```

```cpp
void LsqOp::constructRegion() {
  ...
}
```
