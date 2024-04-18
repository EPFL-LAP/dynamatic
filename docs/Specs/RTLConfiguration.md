# RTL Configuration

This document describes the interconnected behavior of our RTL backend and of the JSON-formatted RTL configuration file, which together bridge the gap between MLIR and synthesizable RTL. There are two main sections in this document.

1. [JSON Format](#json-format) | Describes the expected JSON format for the RTL configuration file.
2. [Matching logic](#matching-logic) | Explains the logic that the backend uses to parse the configuration file and determine the mapping between MLIR and RTL.

## JSON Format

The RTL configuration file is made up of a list of JSON objects which each describe a parameterized RTL component along with

1. a method to retrieve a concrete implementation of the RTL component for each valid combination of parameters (a step we call *concretization*), and
2. a list of timing models for the component, each optionally constrained by specific RTL parameter values.  


### Component description format

Each JSON object describing an RTL component must contain three mandatory keys.

```json
{
  "component": "<name-of-the-corresponding-mlir-op>",
  "parameters": [],
  "models": []
}
```

- The `component` key must map to a string value mapping to the name of the MLIR operation that the RTL component corresponds to. For example. for a mux it would be `handshake.mux`.
- The `parameters` key must map to a list of JSON objects, each describing a parameter of the RTL component one must provide to derive a concrete implementation of the component. For example, for a mux these parameters would be the number of data inputs (`SIZE`), the data bus width on all data inputs (`DATA_WIDTH`), and the data bus width of the select signal (`SELECT_WIDTH`). The [`"parameters"` section below](#parameters-format) describes the expected and recognized keys in each JSON object.
- The `models` key must map to a list of JSON objects, each containing the path to a file containing a timing model for the RTL component. RTL component parameters generally have an influence on a component's timing model; therefore, it is often useful to specify multiple timing models for various combinations of parameters, along with a generic unconstrained fallback model to catch all remaining combinations. To support such behavior, each model in the list may optionally define constraints on the RTL parameters (using a similar syntax as during parameter description) to restrict the applicability of the model to specific conretizations of the component for which the constraints are verified. For example, for a mux we could have a specific timing model when the mux has exactly two data inputs (`SIZE == 2`) and control-only data inputs (`DATA_WIDTH == 0`), and a second fallback model for all remaining parameter combinations. The [`"models"` section below](#models-format) describes the expected and recognized keys in each JSON object.

The mux example described above would look like the following in JSON.

```json
{
  "component": "handshake.mux",
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
      "path": "/path/to/model/for/control-mux-with-2-inputs"
    },
    { "path": "/path/to/model/for/any-mux" }
  ]
}
```

### Concretization methods

Finally, each RTL component description must indicate whether the component must be concretized simply by replacing generic entity parameters during instantiation (implying that the component already has a generic RTL implementation with the same number of parameters as declared in the JSON entry), or by generating the component on-demand for specific parameter values using an arbitray generator.

- For the former, one would define the `generic` key, which must map to the filepath of the generic RTL implementation on disk.
- For the latter, one would define the `generator` key, which must map to a shell command that, when ran, creates the implementation of the component at a specific filesystem location.

Note that exactly one of the two keys must exist for any given component (i.e., a component is either a generic or generated on-demand).

#### Generic

If the mux were to be defined generically, the JSON would look like the following (`parameters` and `models` values ommited for brevity).

```json
{
  "component": "handshake.mux",
  "parameters": [],
  "models": [],
  "generic": "/path/to/generic/mux.vhd"
}
```

When concretizing a generic component, the backend simply needs to copy and paste the generic implementation into the final RTL design. During component instantiation, explicit parameter values are provided for each instance of the generic component, in the order in which they are defined in the `parameters` key-value pair.

#### Generator

If the mux needed to be generated for each parameter combination, the JSON would look like the following (`parameters` and `models` values ommited for brevity).

```json
{
  "component": "handshake.mux",
  "parameters": [],
  "models": [],
  "generator": "/path/to/mux/generator $SIZE $DATA_WIDTH $SELECT_WIDTH --target $OUTPUT_PATH --name $COMPONENT_NAME"
}
```

When concretizing a generated component, the backend opaquely issues the provided shell command, replacing known parameter names prefixed by `$` with their actual values (e.g., for the mux, `$SIZE`, `$DATA_WIDTH`, and `$SELECT_WIDTH` would be replaced by their corresponding parameter values). Additionally, the backend has a couple reserved parameter names for "generator parameters" that serve to create the final RTL design and ensure that there are no conflicts between multiple concretizations of a single generated RTL component.

- `OUTPUT_PATH` is the filepath at which the generator must store the generated RTL component.
- `COMPONENT_NAME` is the name that the generator must give to the generated RTL entity.

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

Unsigned parameters can be range-restricted (by default, any value greater than or equal to 0 is accepted) using the `lb`, `rb`, and `range` key-value pairs, which are all inclusive. Exact matches are possible using the `eq` key-value pair. Finally, `ne` allows to check for differences.

```json
{
  "parameters": [
    { "name": "BETWEEN_2_AND_64", "type": "unsigned", "lb": 2, "rb": 64 }, 
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

### `models` format

Each JSON object describing a timing model must contain the `path` key, indicating the path to a timing model for the component.

```json
{
  "models": [
    { "path": "/path/to/model" },
    { "path": "/path/to/other-model" },
  ]
}
```

Additionally, each object can contain the `constraints` key, which must map to a list of JSON objects describing a constraint on a specific component parameter which restricts the applicability of the timing model. The expected format matches closely that of the `parameters` array. Each entry in the list of constraints must reference a parameter name under the `name` key to denote the parameter being constrained. Then, for the associated parameter type, the same constraint-setting key-value pairs as during parameter definition are available to constrain the set of values for which the timing model should match.

The following example shows a component with two parameters and two timing models. One which restricts the set of possible values for both parameters, and an unconstrained fallback model which will be selected if the parameter values do not satisfy the first model's constraints (`components` and concretization method fields ommited for brevity).

```json
{
  "parameters": [
    { "name": "UNSIGNED_PARAM", "type": "unsigned" },
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
      "path": "/path/to/fallback/model"
    }
  ]
}
```

## Matching logic

*To come...*
