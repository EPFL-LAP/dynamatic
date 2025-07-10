# Backend

This document describes the interconnected behavior of our RTL backend and of the JSON-formatted RTL configuration file, which together bridge the gap between MLIR and synthesizable RTL. There are two main sections in this document.

1. [Design](#design) | Provides an overview of the backend's design and its underlying rationale.
2. [RTL configuration](#rtl-configuration) | Describes the expected JSON format for RTL configuration files.
3. [Matching logic](#matching-logic) | Explains the logic that the backend uses to parse the configuration file and determine the mapping between MLIR and RTL.

## Design

The RTL backend's role is to transform a semi-abstract in-MLIR representation of a dataflow circuit into a specific RTL implementation that matches the behavior that the IR expresses. As such, the backend does not alter the semantics of its input circuit; rather, its task is two-fold.

1. To emit synthesizable RTL modules that implement each operation of the input IR.
2. To emit the "glue" RTL that connects all the RTL modules together to implement the entire circuit.

The first subtask is by far the most complex to implement in a flexible and robust way, whereas the second subtask is easily achievable once we know how to instantiate each of the RTL module we need. As such this design section heavily focuses on how our RTL backend fulfills the first one's requirements. The [next section](#rtl-configuration) indirectly touches on both subtasks by describing how RTL configuration files dictate RTL emission.

Formally, the RTL backend is a sequence of two transformations handled by two separate binaries. This process's starting point is the fully optimized and buffered Handshake-level IR produced by our numerous transformation and optimization passes.

1. In a first step, Handshake operations are converted to *HW* (read "hardware") operations; *HW* is a "lower-level" MLIR dialect whose structure closely ressembles that of RTL code. This is achieved by running the [`HandshakeToHW` conversion pass](#handshake-to-hw) using Dynamatic's optimizer (`dynamatic-opt`). In addition to performing the lowering of Handshake operations, the conversion pass also adds information to the IR that tells the second step which standard RTL modules the circuit uses.
2. In the second step, the HW-level IR emitted by the first step goes through our RTL emitter (`export-vhdl`), which produces synthesizable RTL.

### Handshake to HW

The `HandshakeToHW` conversion pass may appear unnecessary at first glance; one could imagine going directly from Handshake-level IR to RTL without any intermediate IR transformation. While this would certainly be possible, we argue that the resulting backend would become quite complex for no discernable advantage. Having the conversion pass as a kind of pre-processing step to the actual RTL emission allow us to separate concerns in an elegant way, yielding two manageable pieces of software that, while intrinsically linked, are technically independent.  

In particular, the conversion pass offloads multiple IR analysis/transformation steps from the RTL emission logic and is able to emit a valid (HW-level) IR that showcases the result of these transformations in a convenient way. The ability to observe the close-to-RTL in-MLIR representation of the circuit before emitting the actual RTL makes debugging significantly easier, as one can see precisely what circuit will be emitted (identical IO ports, module names, etc.); this would be impossible or at least cumbersome had these transformations happened purely in-memory. Importantly, the conversion pass

1. makes memory interfaces (i.e., their respective signal bundle in the top-level RTL module) explicit,
2. identifies precisely the set of standard RTL modules we will need in the final circuit, and
3. associate a port name to each SSA value use and each SSA result and store it inside the IR to make the RTL emitter's job as minimal as possible.

#### Making memory interfaces explicit

IR at the Handshake level still links MLIR operations representing memory interfaces (e.g., LSQ) inside dataflow circuits to their (implicitly represented) backing memories using the standard `mlir::MemRefType` type, which abstracts the underlying IO that will eventually connect the two together. For example, a Handshake function operating on a single 32-bit-wide integer array of size 64 has the following signature (control signals omitted).

```mlir
handshake.func @func(%mem: memref<64xi32>) -> none { ... }
```

The conversion pass would lower this Handshake function (`handshake::FuncOp`) to an equivalent HW module (`hw::HwModuleOp`) with a signature that makes all the signals connecting the memory interface to its memory explicit (the following snippet omits control signals for brevity).

```mlir
hw.module @func(in %mem_loadData : i32, out mem_loadEn : i1, out mem_loadAddr : i32,
                out mem_storeEn : i1, out mem_storeAddr : i32, out mem_storeData : i32) { ... }
```

> [!NOTE]
> Note that unlike in the Handshake function, the HW module's inputs *and* outputs are between parentheses.

The single `memref`-typed `mem` argument to the Handshake function is replaced by one module input (`mem_loadData`) and 5 module outputs (`mem_loadEn`, `mem_loadAddr`, `mem_storeEn`, `mem_storeAddr`, and `mem_storeData`) that all have simple types immediately lowerable to RTL. The interface's actual specification (i.e., the composition of the signal bundle that the `memref` lowers to) is a separate concern; shown here is Dynamatic's current memory interface, but it could in practice be any signal bundle that fits one's needs.

#### Identifying necessary modules

In the general case, every MLIR operation inside a Handshake function in the input IR ends up being emitted as an instantiation of a specific RTL module. The mapping between these MLIR operations and the eventual RTL instantiations being one-to-one, this part of the conversion is relatively trivial to implement and think about. One less trivial matter, however, is determining what those instances should be *of*. In other words, which RTL modules need to be instantiated and therefore need be part of the final RTL design.

Consider the following `handshake::MuxOp` operation, which represents a regular dataflow multiplexer taking any strictly positive number of data inputs and a select input to dictate which of the data inputs should be forwarded to the single output.

```mlir
%result1 = handshake.mux %select [%data1, %data2] : i1, i32
```

This particular multiplexer has 2 data inputs whose data bus is 32-bit wide, and a 1-bit wide select input (1 bit is enough to select between 2 inputs). Now consider this second multiplexer which, despite having the same identified characteristics, has different data inputs.

```mlir
// Previous multipexer
%result1 = handshake.mux %select [%data1, %data2] : i1, i32

// New one with same characteristics
// - 2 data inputs
// - 32-bit data bus
// - 1-bit select bus
%result2 = handshake.mux %select [%data3, %data4] : i1, i32
```

As mentioned before, each of these two multiplexers would be emitted as a separate instantiation of a specific RTL module. However, it remains to determine whether these two instantiations would be of *the same RTL module*. In that particular example, both multiplexer modules (whether they were different or identical) would have the same top-level IO. Indeed, the three characteristics we previously identified (number of data inputs, data bus width, select bus width) completely characterize the multiplexer's RTL interface (their gate-level implementation could of course be different).

Predictably, not all multiplexers will have the same RTL interface. Consider the following multiplexer with 16-bit data buses.

```mlir
// Previous multipexers with
// - 2 data inputs
// - 32-bit data bus
// - 1-bit select bus
%result1 = handshake.mux %select [%data1, %data2] : i1, i32
%result2 = handshake.mux %select [%data3, %data4] : i1, i32

// This multiplexer has 16-bit data buses instead of 32
%result3 = handshake.mux %select [%data5, %data6] : i1, i16
```

It should be clear that there is not, at least in the general case, a clear correspondance between Handshake operation types (e.g., `handshake::MuxOp`) and the interface of the RTL module they will eventually being emitted as. Two MLIR operations of the same type may be emitted as two RTL instances of the same RTL module, or as two RTL instances of different RTL modules. The conversion pass needs a way to identify its concrete RTL module needs based on its input IR.

We introduce the concept of *RTL parameter* to formalize this mapping between MLIR operations and RTL modules. The general idea is, during conversion of each Handshake-level MLIR operation to an `hw::InstanceOp`---the HW dialect's operation that represents RTL instances---to identify the "intrinsic structural characteristics" of each operation and add to the IR an operation that will instruct the RTL emitter to emit a matching RTL module. We call these "intrinsic structural characteristics" *RTL parameters*, and we encode them as attributes to `hw::HWModuleExternOp` operations, which as their name suggest represent external RTL modules that are needed by the main module's implementation.

Consider an input Handshake function containing the three multiplexers we previously described (all other operations omitted).

```mlir
handshake.func @func(...) -> ... {
  ...
  // 2 data inputs, 32-bit data, 1-bit select
  %result1 = handshake.mux %select [%data1, %data2] : i1, i32
  %result2 = handshake.mux %select [%data3, %data4] : i1, i32
  // 2 data inputs, 16-bit data, 1-bit select
  %result3 = handshake.mux %select [%data5, %data6] : i1, i16
  ...
}
```

The conversion pass would lower this Handshake function to something that looks like the following (details omitted for brevity).

```mlir
// RTL module directly corresponding to the input Handshake function.
// This is the "glue" RTL that connects everything together.
hw.module @func(...) {
  ...
  // 2 data inputs, **32-bit data**, 1-bit select
  %result1 = hw.instance @mux_32 "mux1" (%select, %data1, %data2) -> channel<i32>
  %result1 = hw.instance @mux_32 "mux2" (%select, %data3, %data4) -> channel<i32>

  // 2 data inputs, **16-bit data**, 1-bit select
  %result1 = hw.instance @mux_16 "mux3" (%select, %data5, %data6) -> channel<i16>
  ...
}

// RTL module corresponding to the mux variant with **32-bit data**.
// The RTL emitter will need to *concretize* an RTL implementation for this module.
hw.module.extern @mux_32( in channel<i1>, in channel<i32>,
                          in channel<i32>, out channel<i32>) attributes {
  hw.name = "handshake.mux", 
  hw.parameters = {SIZE = 2 : ui32, DATA_WIDTH = 32 : ui32, SELECT_WIDTH = 1 : ui32}
}


// RTL module corresponding to the mux variant with **16-bit data**.
// The RTL emitter will need to *concretize* an RTL implementation for this module.
hw.module.extern @mux_16( in channel<i1>, in channel<i16>,
                          in channel<i16>, out channel<i16>) attributes {
  hw.name = "handshake.mux", 
  hw.parameters = {SIZE = 2 : ui32, DATA_WIDTH = 16 : ui32, SELECT_WIDTH = 1 : ui32}
}
```

Observe that while each multiplexer maps directly to a `hw.instance` (`hw::InstanceOp`) operation, the conversion pass only produces two external RTL modules (`hw.module.extern`): one for the multiplexer variant with 32-bit data, and one for the variant with 16-bit data. These `hw.module.extern` (`hw::HWModuleExternOp`) operations encode two important pieces of information in dedicated MLIR attributes.

- `hw.name` is the canonical name of the MLIR operation from which the RTL module originates, here the Handshake-level multiplexer `handshake.mux`.
- `hw.parameters` is a dictionary mapping each of the multiplexers's RTL parameter to a specific value.

Importantly, each input operation type defines the set of RTL parameters which characterizes it. As we just saw, for multiplexers these are the number of data inputs (`SIZE`), the data-bus width (`DATA_WIDTH`), and the select-bus width (`SELECT_WIDTH`). The conversion pass will generate one external module definition for each unique combination of RTL name and parameter values dervied from the input IR. These are the RTL modules that the second part of the backend, the RTL emitter, will need to derive an implementation for so that they can be instantiated from the main RTL module. We call this step *concretization* and explain its underlying logic in the [RTL emission](#rtl-emission) subsection.

> [!IMPORTANT]
> While the pass itself sets RTL parameters purely according to each operation's structural characteristics, nothing prevents passes up the pipeline to already set arbitrary RTL parameters on MLIR operations. The `HandshakeToHW` conversion pass treats RTL parameters already present in the input IR transparently by considering them on the same level as the structural parameters it itself sets (unless there is a name conflict, in which case it emits a warning). It is then up to the backend's [RTL configuration](#rtl-configuration) to recognize these "extra RTL parameters" and act accordingly (they may be ignored if nothing is done, resulting in a "regular" RTL module being concretized, see the [matching logic](#matching-logic)). For example, a pass up the pipeline may wish to distinguish between two different RTL implementations (say, `A` and `B`) of `handshake.mux` operations in order to gain performance. Such a pass could already tag these operations with an RTL parameter (e.g., `hw.parameters = {IMPLEMENTATION = "A"}`) to carry that information down the pipeline and, with proper support in the backend's RTL configuration, concretize and instantiate the intended RTL module.

#### Port names

At the Handshake level, the input and output ports of MLIR operations (in MLIR jargon, their operands and results) do not have names. In keeping with the objective of the `HandshakeToHW` conversion pass to lower the IR to a close-to-RTL representation, the pass associates a port name to each input and output port of each HW-level instance and (external) module operation. These port names will end up as-is in the emitted RTL design (unless explicitly modified by the RTL configuration, see JSON options [`io-kind`](#io-kind), [`io-signals`](#io-signals), and [`io-map`](#io-map)). They are derived through a mix of means depending on the specific input MLIR operation type.

### RTL emission

The RTL emitter picks up the IR that comes out of the `HandshakeToHW` conversion pass and turns it into a synthesizable RTL design. Importantly, the emitter takes as additional argument a list of JSON-formatted RTL configuration files which describe the set of parameterized RTL components it can conretize and instantiate; the [next section](#rtl-configuration) covers in details the configuration file's expected syntax, including all of its options.

After parsing RTL configuration files, the emitter attempts to match each `hw.module.extern` (`hw::HWModuleExternOp`) operation in its input IR to entries in the configuration files using the `hw.name` and `hw.parameters` attributes; the [last section](#matching-logic) describes the matching logic in details. If a matching RTL component is found, then the emitter *concretizes* the RTL module implementation that corresponds to the `hw.module.extern` operation into the final RTL design. This concretization may be as simple as copying a generic RTL implementation of a component to the output directory, or require running an arbitrarily complex RTL generator that will generate a specific implementation of the component that depends on the specific RTL parameter values. RTL configuration files dictate the concretization method for each RTL component they declare. If any `hw.module.extern` operation finds no match in the RTL configuration, RTL emission fails.

Circling back to the multiplexer example, it is possible to define [a single generic RTL multiplexer implementation](../../experimental/data/vhdl/handshake/mux.vhd) that is able to implement all possible combinations of RTL parameter values. Assuming an appropriate RTL configuration, the RTL emitter would simply copy that known generic RTL implementation to the final RTL design if its input IR contained any `hw.module.extern` operation with name `handshake.mux` and valid value for each of the three RTL parameters.

Emitting each `hw.module` (`hw::hwModuleOp`) and `hw.instance` (`hw::InstanceOp`) operation to RTL is relatively straightforward once all external modules are concretized. This translation is almost one-to-one, requires little work, and is HDL-independent beyond syntactic concerns.

## RTL configuration

An RTL configuration file is made up of a list of JSON objects which each describe a parameterized RTL component along with

1. a method to [retrieve a concrete implementation](#concretization-methods) of the RTL component for each valid combination of [parameters](#parameters-format) (a step we call *concretization*),
2. a list of [timing models](#models-format) for the component, each optionally constrained by specific RTL parameter values, and
3. a list of [options](#options).

### Component description format

Each JSON object describing an RTL component should specify a mandatory `name` key and optional `parameters` and `models` keys.

```json
{
  "name": "<name-of-the-corresponding-mlir-op>",
  "parameters": [],
  "models": []
}
```

- The `name` key must map to a string that identifies the RTL component the entry corresponds to. For RTL components mapping one-to-one with an MLIR operation, this would typically be the canonical MLIR operation name. For example. for a mux it would be `handshake.mux`.
- The `parameters` key must map to a list of JSON objects, each describing a parameter of the RTL component one must provide to derive a concrete implementation of the component. For example, for a mux these parameters would be the number of data inputs (`SIZE`), the data bus width on all data inputs (`DATA_WIDTH`), and the data bus width of the select signal (`SELECT_WIDTH`). The ["`parameters` format" section](#parameters-format) describes the expected and recognized keys in each JSON object. If the `parameters` key is omitted, it is assumed to be an empty list.
- The `models` key must map to a list of JSON objects, each containing the path to a file containing a timing model for the RTL component. RTL component parameters generally have an influence on a component's timing model; therefore, it is often useful to specify multiple timing models for various combinations of parameters, along with a generic unconstrained fallback model to catch all remaining combinations. To support such behavior, each model in the list may optionally define constraints on the RTL parameters (using a similar syntax as during parameter description) to restrict the applicability of the model to specific conretizations of the component for which the constraints are verified. For example, for a mux we could have a specific timing model when the mux has exactly two data inputs (`SIZE == 2`) and control-only data inputs (`DATA_WIDTH == 0`), and a second fallback model for all remaining parameter combinations. The ["`models` format" section](#models-format) describes the expected and recognized keys in each JSON object. If the `models` key is omitted, it is assumed to be an empty list.

The mux example described above would look like the following in JSON.

```json
{
  "name": "handshake.mux",
  "parameters": [
      { "name": "SIZE", "type": "unsigned", "lb": 2 },
      { "name": "DATA_WIDTH", "type": "unsigned", "ub": 64 },
      { "name": "SELECT_WIDTH", "type": "unsigned", "range": [1, 6] }
    ],
  "models": [
    {
      "constraints": [
        { "parameter": "SIZE", "eq": 2 },
        { "parameter": "DATA_WIDTH", "eq": 0 }
      ],
      "path": "/path/to/model/for/control-mux-with-2-inputs.sdf"
    },
    { "path": "/path/to/model/for/any-mux.sdf" }
  ]
}
```

### Concretization methods

Finally, each RTL component description must indicate whether the component must be concretized simply by replacing generic entity parameters during instantiation (implying that the component already has a generic RTL implementation with the same number of parameters as declared in the JSON entry), or by generating the component on-demand for specific parameter values using an arbitray generator.

- For the former, one would define the `generic` key, which must map to the filepath of the generic RTL implementation on disk.
- For the latter, one would define the `generator` key, which must map to a shell command that, when ran, creates the implementation of the component at a specific filesystem location.

Exactly one of the two keys must exist for any given component (i.e., a component is either a generic or generated on-demand).
  
> [!IMPORTANT]
> The string value associated to the `generic` and `generator` key supports *parameter substitution*; if it contains the name of component parameters prefixed by a `$` symbol (shell-like syntax), these will be replaced by explicit parameter values during component concretization. Additionally, the backend provides a couple extra *backend parameters* during component concretization which hold meta-information useful during generation but not linked to any component's specific implementation. Backend parameters have reserved names and are substituted with explicit values just like regular component parameters. The ["special parameters" section](#backend-parameters) lists all special parameters.
>
> Parameter substitution is key for generated components, whose shell command must contain the explicit parameter values to generate the matching RTL implementation on request, but is often useful in other contexts too. When the backend supports parameter substitution for a particular JSON field, we explicitly indicate it in this specification.

#### Generic

If the mux were to be defined generically, the JSON would look like the following (`parameters` and `models` values ommited for brevity).

```json
{
  "name": "handshake.mux",
  "generic": "$DYNAMATIC/data/vhdl/handshake/mux.vhd"
}
```

When concretizing a generic component, the backend simply needs to copy and paste the generic implementation into the final RTL design. During component instantiation, explicit parameter values are provided for each instance of the generic component, in the order in which they are defined in the `parameters` key-value pair. Note that `$DYNAMATIC` is a [backend parameter](#backend-parameters) which indicates the path to Dynamatic's top-level directory.

#### Generator

If the mux needed to be generated for each parameter combination, the JSON would look like the following (`parameters` and `models` values ommited for brevity).

```json
{
  "name": "handshake.mux",
  "generator": "/path/to/mux/generator $SIZE $DATA_WIDTH $SELECT_WIDTH --output \"$OUTPUT_DIR\" --name $MODULE_NAME"
}
```

When concretizing a generated component, the backend opaquely issues the provided shell command, replacing known parameter names prefixed by `$` with their actual values (e.g., for the mux, `$SIZE`, `$DATA_WIDTH`, and `$SELECT_WIDTH` would be replaced by their corresponding parameter values). Note that `$OUTPUT_DIR` and `$MODULE_NAME` are [backend parameters](#backend-parameters) which indicate, respectively, the path to the directory where the generator must create a file containing the component's RTL implementation, and the name of the main RTL module that the backend expects the generator to create.

#### Per-parameter concretization method

In some situations, it may be desirable to override the backend's concretization-method-dependent behavior on a per-parameter basis. For example, specific RTL parameters of a *generic* component may be useful for matching purposes (see [matching logic](#matching-logic)) but absent in the generic implementation of the RTL module. Conversely, a component *generator* may produce "partially generic" RTL modules requiring specific RTL parameters during instantiation.

All parameters support the `generic` key which, when present, must map to a boolean indicating whether the parameter should be provided as a generic parameter to instances of the concretized RTL module, regardless of the component's concretization method. The backend follows the behavior dictated by the component's concretization method for all RTL parameters that do not specify the `generic` key.

### `parameters` format

Each JSON object describing an RTL component parameter must contain two mandatory keys.

```json
{
  "parameters": [
    { "name": "<parameter-name>", "type": "<parameter-type>" },
    { "name": "<other-parameter-name>", "type": "<other-parameter-type>" },
  ]
}
```

- The `name` key must map to string that uniquely identifies the component parameter. Only alphanumeric characters, dashes, and underscores are allowed in parameter names.
- The `type` key must map to a string denoting the parameter's datatype. Currently supported values are
  - `unsigned` for an unsigned integer and
  - `string` for an arbitrary sequence of characters.

Depending on the parameter type, additional key-value pairs constraining the set of allowed values are recognized.

#### unsigned

Unsigned parameters can be range-restricted (by default, any value greater than or equal to 0 is accepted) using the `lb`, `ub`, and `range` key-value pairs, which are all inclusive. Exact matches are possible using the `eq` key-value pair. Finally, `ne` allows to check for differences.

```json
{
  "parameters": [
    { "name": "BETWEEN_2_AND_64", "type": "unsigned", "lb": 2, "ub": 64 }, 
    { "name": "SHORT_BETWEEN_2_AND_64", "type": "unsigned", "range": [2, 64] }, 
    { "name": "EXACTLY_4", "type": "unsigned", "eq": 4 }, 
    { "name": "DIFFERENT_THAN_2", "type": "unsigned", "ne": 2 }, 
  ]
}
```

#### string

For string parameters, only exact matches/differences are currently supported with `eq` and `ne`.

```json
{
  "parameters": [
    { "name": "EXACTLY_MY_STRING", "type": "string", "eq": "MY_STRING" }, 
    { "name": "NOT_THIS_OTHER_STRING", "type": "string", "ne": "THIS_OTHER_STRING" }, 
  ]
}
```

#### Backend parameters

During component concretization, the backend injects extra *backend parameters* that are available for parameter substitution in addition to the parameters of the component being concretized. These parameters have reserved names which cannot be used by user-declared parameters in the RTL configuration file. All backend parameters are listed below.

- `DYNAMATIC`: path to Dynamatic's top-level directory (without a trailing slash).
- `OUTPUT_DIR`: path to output directory where the component is expected to be concretized (without a trailing slash). This is only really meaningful for generated components, for which it tells the generator the direcotry in which to create the VHDL (`.vhd`) or Verilog (`.v`) file containing the component's RTL implementation. Generators can assume that the directory already exists.
- `MODULE_NAME`: RTL module name (or "entity" in VHDL jargon) that the backend will use to instantiate the component from RTL. Concretization must result in a module of this name being created inside the output directory. Since module names are unique within the context of each execution of the backend, generators may assume that they can create without conflict a file named `$MODULE_NAME.<extension>` inside the output directory to store the generated RTL implementation; in other words, a safe output path is `"$OUTPUT_DIR/$MODULE_NAME.<extension>"` (note the quotes around the path to handle potential spaces inside the output directory's path correctly). This parameter is controllable from the RTL configuration file itsel, see the [relevant option](#module-name).

### `models` format

Each JSON object describing a timing model must contain the `path` key, indicating the path to a timing model for the component.

```json
{
  "models": [
    { "path": "/path/to/model.sdf" },
    { "path": "/path/to/other-model.sdf" },
  ]
}
```

Additionally, each object can contain the `constraints` key, which must map to a list of JSON objects describing a constraint on a specific component parameter which restricts the applicability of the timing model. The expected format matches closely that of the `parameters` array. Each entry in the list of constraints must reference a parameter name under the `name` key to denote the parameter being constrained. Then, for the associated parameter type, the same constraint-setting key-value pairs as during parameter definition are available to constrain the set of values for which the timing model should match.

The following example shows a component with two parameters and two timing models. One which restricts the set of possible values for both parameters, and an unconstrained fallback model which will be selected if the parameter values do not satisfy the first model's constraints (`components` and concretization method fields ommited for brevity).

```json
{
  "parameters": [
    { "name": "UNSIGNED_PARAM", "type": "unsigned" },
    { "name": "OTHER_UNSIGNED_PARAM", "type": "unsigned" },
    { "name": "STRING_PARAM", "type": "string" }
  ],
  "models": [
    { 
      "constraints": [
        { "name": "UNSIGNED_PARAM", "lb": 4 },
        { "name": "STRING_PARAM", "eq": "THIS_STRING" },
      ],
      "path": "/path/to/model-with-constraints" 
    },
    {
      "path": "/path/to/fallback/model.sdf"
    }
  ]
}
```

### Options

Each RTL component description recognizes a number of options that may be helpful in certain situations. These each have a dedicated key name which must exist at the component description's top-level and map to a JSON element of the valid type (depending on the specific option). See examples in each subsection.

#### `dependencies`

Components may indicate a list of other components they depend on (e.g., which define RTL module(s) that they instantiate within their own module's implementation) via their name. When concretizing a component with dependencies, the backend will look for components within the RTL configuration whose name matches each of the dependencies and attempt to concretize them along the original component. The backend is able to recursively concretize dependencies's dependencies and ensures that any dependency is concretized only a single time, even if it appears in the dependency list of multiple components in the current backend execution. This system allows to indirectly concretize "supporting" (i.e., depended on) RTL components used within the implementation of multiple "real" (i.e., corresponding to MLIR operations) RTL components seamlessly and without code duplication.

The `dependencies` option, when present, must map to a list of strings representing RTL component names within the configuration file. The list is assumed to be empty when omitted. In the following example, attempting to concretize the `handshake.mux` component will make the backend concretize the `first_dependency` and `second_dependency` components as well (some JSON content omitted for brevity).

```json
[
  {
    "name": "handshake.mux",
    "generic": "$DYNAMATIC/data/vhdl/handshake/mux.vhd",
    "dependencies": ["first_dependency", "second_dependency"]
  },
  { 
    "name": "first_dependency",
    "generic": "/path/to/first/dependency.vhd",
  },
  { 
    "name": "second_dependency",
    "generic": "/path/to/second/dependency.vhd",
  }
]
```

At the moment the dependency management system is relatively barebone; only parameter-less components can appear in dependencies since there is no existing mechanism to transfer the original component's parameters to the component it depends on (therefore, any dependency with at least one parameter will fail to match due to the lack of parameters provided during dependency resolution, see [matching logic](#matching-logic)).

#### `module-name`

> [!NOTE]
> The `module-name` option supports parameter substitution.

During RTL emission, the backend associates a module name to each RTL component concretization to uniquely identify it with respect to

1. differently named RTL components, and to
2. other concretizations of the same RTL component with different RTL parameter values.

By default, the backend derives a unique module name for each concretization using the following logic.

- For generic components, the module name is set to be the filename part of the filepath, without the file extension. For the example given in the [generic section](#generic) which associates the string `$DYNAMATIC/data/vhdl/handshake/mux.vhd` to the `generic` key, the derived module name would simply be `mux`.
- For generated components, the module name is provided by the backend logic itself, and is in general derived from the specific RTL parameter values associated to the concretization.

The [`MODULE_NAME` backend parameter](#backend-parameters) stores, for each component concretization, the associated module name. This allows JSON values supporting parameter substitution to include the name of the RTL module they are expected to generate during concretization.

> [!WARNING]
> The backend uses module names to determine whether different component concretizations should be identical. When an RTL component is selected for concretization and the derived module name is identical to a previously  concretized component, then the current component will be assumed to be identical to the previous one and therefore will not be concretized anew. This makes sense when considering that each module name indicates the actual name of the RTL module (Verilog `module` keyword or VHDL `entity` keyword) that the backend expects the concretization step to bring into the "current workspace" (i.e., to implement in a file inside the output directory). Multiple modules with the same name would cause name clashes, making the resulting RTL ambiguous.

The `module-name`, when present, must map to a string which overrides the default module name for the component. In the following example, the generic `handshake.mux` component would normally get asssigned the `mux` module name by default, but if the actual RTL module inside the file was named `a_different_mux_name` we could indicate this using the option as follows (some JSON content omitted for brevity).

```json
{
  "name": "handshake.mux",
  "generic": "$DYNAMATIC/data/vhdl/handshake/mux.vhd",
  "module-name": "a_different_mux_name"
}
```

#### `arch-name`

> [!NOTE]
> The `arch-name` option supports parameter substitution.

The internal implementation of VHDL entities is contained in so-called "architectures". Because there may be multiple such architectures for a single entity, each of them maps to a unique name inside the VHDL implementation. Instantiating a VHDL entitiy requires that one specifies the chosen architecure by name in addition to the entity name itself. By default, the backend assumes that the architecture to choose when instantiating VHDL entities is called "arch".

The `arch-name` option, when present, must map to a string which overrides the default architecture name for the component. If the architecture of our usual `handshake.mux` example was named `a_different_arch_name` then we could indicate this using the option as follow (some JSON content omitted for brevity).

```json
{
  "name": "handshake.mux",
  "generic": "$DYNAMATIC/data/vhdl/handshake/mux.vhd",
  "arch-name": "a_different_arch_name"
}
```

#### `use-json-config`

> [!NOTE]
> The `use-json-config` option supports parameter substitution.

When an RTL component is very complex and/or heavily parameterized (e.g., the LSQ), it may be cumbersome or impossible to specify all of its parameters using our rather simple RTL typed parameter system. Such components may provide the `use-json-config` option which, when present, must map to a string indicating the path to a file in which the backend can JSON-serialize all RTL parameters associated to the concretization. This file can then be deserialized from a component generator to get back all generation parameters easily. Consequentlt, this option does not really make sense for generic components.

Below is an example of how you would use such a parameter for generating an LSQ by first having the backend serialize all its RTL parameters to a JSON file.

```json
{
  "name": "handshake.lsq",
  "generic": "/my/lsq/generator --config \"$OUTPUT_DIR/$MODULE_NAME.json\"",
  "use-json-config": "$OUTPUT_DIR/$MODULE_NAME.json"
}
```

#### `hdl`

The `hdl` option, when present, must map to a string indicating the hardware description language (HDL) in which the concretized component is written. Possible values are `vhdl` (default), or `verilog`. If the `handshake.mux` component was written in Verilog, we would explictly specify it as follows.

```json
{
  "name": "handshake.mux",
  "generic": "$DYNAMATIC/data/vhdl/handshake/mux.vhd",
  "hdl": "verilog"
}
```

#### `io-kind`

The `io-kind` option, when present, must map to a string indicating the naming convention to use for the module's ports that logically belong to arrays of bitvectors. This matters when instantiating the associated RTL component because the backend must know how to name each of the individual bitvectors to do the port mapping.

- Generic RTL modules may have to use something akin to an array of bitvectors to represent such variable-sized ports. In this case, each individual bitvector's name will be formed from the base port name and a numeric index into the array it represents. This `io-kind` is called `hierarchical` (default).
- RTL generators, like Chisel, may flatten such arrays into separate bitvectors. In this case, each individual bitvector's name will be formed from the base port name along with a textual suffix indicating the logical port index. This `io-kind` is called `flat`.

Let's take the example of a multiplexer implementation with a configurable number of data inputs. Its VHDL implementation could follow any of the two conventions.

With `hierarchical` IO, the component's JSON description (some content omitted for brevity) and RTL implementation would look like the following.

```json
{
  "name": "handshake.mux",
  "generic": "$DYNAMATIC/data/vhdl/handshake/mux.vhd",
  "io-kind": "hierarchical"
}
```

```vhdl
entity mux is
  generic (SIZE : integer; DATA_WIDTH : integer);
  ports (
    -- all other IO omitted for brevity
    dataInputs : in array(SIZE) of std_logic_vector(DATA_WIDTH - 1 downto 0)
  );
end entity;
```

If we were to concretize a multiplexer with 2 inputs and 32-bit datawidth using the above generic component, we would need to name its data inputs `dataInputs(0)` and `dataInputs(1)` during instantiation. However, if we were to use a generator to concretize this specific multiplexer implementation, the component's JSON description (some content omitted for brevity) and RTL implementation would most likely look like the following.

```json
{
  "name": "handshake.mux",
  "generator": "/my/mux/generator $SIZE $DATA_WIDTH $SELECT_WIDTH",
  "io-kind": "flat"
}
```

```vhdl
entity mux is
  ports (
    -- all other IO omitted for brevity
    dataInputs_0 : in std_logic_vector(31 downto 0);
    dataInputs_1 : in std_logic_vector(31 downto 0)
  );
end entity;
```

We would need to name its data inputs `dataInputs_0` and `dataInputs_1` during instantiation in this case.

In both cases, the base name `dataInputs` is part of the specification of `handshake.mux`, the matching MLIR operation. Within the IR, these ports are always named following the `flat` convention: `dataInputs_0` and `dataInputs_1`. During RTL emission, they will be converted to the first hierarchical form by default, or left as is if the `io-kind` is explicitly set to `flat`.

#### `io-signals`

The backend has naming convention when it comes to signals part of the same dataflow channel. By default, if the channel name is `channel_name`, then all signal names will start with the channel name and be suffixed by a specific (possibly empty) string.

- the data bus has no suffix (`channel_name`),
- the valid wire has a `_valid` suffix (`channel_name_valid`), and
- the ready wire has a `_ready` suffix (`channel_name_ready`).

This matters when instantiating the associated RTL component because the backend must know how to name each of the individual signals to do the port mapping.

The `io-signals` option, when present, must map to a JSON object made up of key/string-value pairs where the key indicates a specific signal within a dataflow channel and the value indicates the suffix to use instead of the default one. Recognized keys are `data`, `valid`, and `ready`.

For example, the `handshake.mux` component could modify its empty-by-default data signal suffix to `_bits` to match Chisel's conventions.  

```json
{
  "name": "handshake.mux",
  "generator": "/my/chisel/mux/generator $SIZE $DATA_WIDTH $SELECT_WIDTH",
  "io-signals": { "data": "_bits" }
}
```

#### `io-map`

The backend determines the port name of each RTL module's signal using the operand/result names encoded in HW-level IR, which themselves come from the `handshake::NamedIOInterface` interface for Handshake operations, and from custom logic for operations from other dialects. In some cases, however, the concretized RTL implementation of a component may not match these conventions and it may be unpractical to modify the RTL to make it agree with MLIR port names.

The `io-map` option, when present, must map to a list of JSON objects each made up of a single key/string-value pair indicating how to map MLIR port names matching the key to RTL port names encoded by the value. If the option is absent, the list is assumed to be empty. For each MLIR port name, the list of remappings is evaluated in definition order, stopping at the first MLIR port name matching the key. When no remapping matches, the MLIR and RTL port names are understood to be identical.

Remappings support a very simplified form of regular expression matching where, for each JSON object, either the key or both the key and value may contain a single wildcard `*` character. In the key, any possible empty sequence of characters can be matched to the wildcard. If the value also contains a wildcard, then the wildcard-matched characters in the MLIR port name will be copied at the wildcard's position in the RTL port name.

For example, if the `handshake.mux` components's RTL implementation prefixed all its signal names with the `io_` string and named its selector channel input `io_select` instead of `index` (the MLIR operation's convention), then we could leverage the `io-map` option to make the two work together without modifying any C++ or RTL code.

```json
{
  "name": "handshake.mux",
  "generator": "/my/chisel/mux/generator $SIZE $DATA_WIDTH $SELECT_WIDTH",
  "io-map": [
    { "index": "io_select" },
    { "*": "io_*" },
  ]
}
```

> [!WARNING]
> The backend performs port name remapping before adding [signal-specific suffixes](#io-signals) to port names and before taking into account the [IO kind](#io-kind) for logical port arrays.

## Matching logic

As mentionned, a large part of the RTL emitter's job is to concretize an RTL module for each `hw.module.extern` (`hw::HWModuleExternOp`) operation present in the input IR. It does so by querying the RTL configuration it parsed from RTL configuration files for possible matches. This section gives some pointers as to how the matching logic work.

Upon encountering a `hw.module.extern` operation, the RTL emitter creates an *RTL request* which it then sends to the RTL configuration. The request looks for the `hw.name` and `hw.parameters` attributes attached to the operation to determine, respectively, the name of the RTL component that the operation corresponds to and the mapping between RTL parameter name and value. Upon reception of the RTL request, the RTL configuration iterates over all of its known components *in parsing order* to try to find a potential match. The order of evaluation of RTL components parsed from the same JSON file is the same as the order of top-level objects in the file. If the RTL configuration was parsed from multiple files, it evaluates files in the order in which they were provided as arguments to the RTL emitter. The RTL configuration stops at the first successful match, if there is any.

A successful match between an RTL request and an RTL component requires a combination of two factors.

1. The name of the RTL component and the name associated to the RTL request must be *exactly* the same.
2. The name of every RTL parameter that the component declares must be part of the parameter name-to-value mapping associated to the RTL request. Furthermore, the value of that parameter must satisfy any [constraints associated to the RTL parameter's type](#parameters-format).

> [!IMPORTANT]
> A successful match does not require the second factor's reciprocal. If the RTL request contains a name-to-value parameter mapping whose name is not a known RTL parameter according to the RTL component's definition, then the match will still be successful. This allows to easily define "fallback" behaviors in advanced use cases. A specific RTL component may have "extra RTL parameters" that allows compiler passes to configure the underlying RTL implementation of this component to a very fine degree. However, we do not want to force the default compilation flow (which may not care for this level of control) to specify these RTL parameters in every request for the component. We need to be able to match requests specifying all parameters (including the extra ones) to the RTL component offering fine control while still being able to match requests only specifying the regular "structural" parameters to the "basic" RTL component. This can be achieved by declaring the RTL component *twice* in the configuration files, once with the extra parameters and once without. As long as RTL configuration evaluates the former component first (see evaluation order above), we will get the desired "fallback" behavior while benefiting from the extra control on-demand.

If the RTL configuration finds a match, it returns the associated component to the RTL emitter which then concretizes the RTL module (along any dependency) inside the circuit's final RTL design.
