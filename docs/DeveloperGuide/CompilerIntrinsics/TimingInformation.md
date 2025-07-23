# Timing Information and its handling in Dynamatic
This document explains how Dynamatic stores and uses timing information for hardware operators, providing both conceptual understanding and implementation guidance.

## What is Timing Information?
Each operator in a hardware circuit is characterized by two fundamental timing properties:

- **Latency**: The number of clock cycles an operator requires to produce a valid output after receiving valid input, assuming the output is ready to accept the result. This latency is always an integer and corresponds to the number of pipeline stages (i.e., registers) the data passes through.

- **Delay**: The combinational delay along a path — i.e., the time it takes for a signal to propagate through combinational logic, without being interrupted by clocked elements (registers). Delay is measured in physical time units (e.g., nanoseconds).


We classify combinational delays into two categories:

- **Intra-port delays**: Combinational delays from an input port to an output port with no intervening registers. These represent purely combinational paths through an operator. 

- **Port2Reg delays**: Combinational delays either from an input port to the first register stage, or from the last register stage to an output port. These capture the logic surrounding the sequential boundaries of an operator.

- **Reg2Reg delay** :Combinational delays from one register stage to the next register stage within a single pipelined operation, representing the longest logic path between these sequential elements.

This difference is a key distinction between **pipelined** and **non-pipelined** operations. Consider the following graph :

![image](https://github.com/user-attachments/assets/c6d2961c-f4ba-47ed-a7a0-6201314f9530)



In the pipelined case (i.e., when latency > 0), registers are placed along the paths between input and output ports. As a result, these paths no longer have any intra-port delays, since there are no purely combinational routes connecting inputs directly to outputs. However, port2reg delays still exist on these paths — capturing the combinational delays between an input port and the first register stage, and between the last register stage and an output port. In the figure, the inport and outport delays illustrate these port2reg delays.


In the non-pipelined case, there are no registers on the path connecting the input to output port. For this reason, there are no port2reg delays and the only delay present is the intra-port delay (comb logic delay).


In the previous example, we assumed there is only one input and one output port. However, there can be multiple ones and of different types. We can differentiate input and output port into 4 types:
- **DATA (D)** representing the data signal.
- **CONDITION (C)** representing the condition signal.
- **VALID (V)** representing the valid signal of the handshake communication.
- **READY (R)** representing the ready signal of the handshake communication.


The combinational delays can connect ports of the same or different types. The ones of different types supported for now are the following ones: VR (valid to ready), CV (control to valid), CR (control to ready), VC (valid to control), and VD (valid to data).


**Note** : The current code does not seem to use the information related to inport and outport delays. Furthermore all the port delays are 0 for all listed components. We assume this is the intended behaviour for now. We welcome a change to this documentation if the code structure changes. 



## Where Timing Data is Stored

All timing information lives in the [components JSON file](https://github.com/EPFL-LAP/dynamatic/blob/main/data/components.json). Here's what a typical entry looks like:

```json
{
  "handshake.addi": {
    "latency": {
      "64":{
        "2.3": 8,
        "4.2": 4
    },
    "delay": {
      "data": {
        "32": 2.287,
        "64": 2.767
      },
      "valid": {
        "1": 1.397
      },
      "ready": {
        "1": 1.4
      },
      "VR": 1.409,
      "CV": 0,
      "CR": 0,
      "VC": 0,
      "VD": 0
    },
    "inport": { /* port-specific delays, structured like the delay set above */ },
    "outport": { /* port-specific delays, structured like the delay set above  */ }
  }
}
```

The JSON object encodes the following timing information:
- `latency`: A dictionary mapping bitwidths to timing features of multiple implementation of the component available for that bitwidth. This is done as a map listing all existing implementations, providing their internal combinational delay as key and their latency as value.
- `delays`: A dictionary describing intra-port delays — i.e., combinational delays between input and output ports with no intervening registers (in nanoseconds).
- `inport`: A dictionary specifying port2reg delays from an input port to the first register stage (in nanoseconds).
- `outport`: A dictionary specifying port2reg delays from the last register stage to an output port (in nanoseconds).


The delays dictionary is structured as follows:

- It includes three special keys: "data", "valid", and "ready". Each of these maps to a nested dictionary that captures intra-port delays between ports of the same type. In these nested dictionaries, the keys are bitwidths and the values are the corresponding delay values.

- Additional keys in the delays dictionary represent intra-port delays between different port types (e.g., from "valid" to "data"), and their values are the corresponding delay amounts.

The inport and outport dictionaries follow the same structure as the delays dictionary, capturing combinational delays between ports and registers instead of port-to-port paths.

The delay information can be computed using a [characterization script](https://github.com/EPFL-LAP/dynamatic/tree/main/tools/backend/synth-characterization/run-characterization.py). More information about the script are present in [this doc](../Specs/TimingCharacterization.md).

The latest version of these delays has been computed using Vivado 2019.1.

## How Timing Information is Used

Timing data is primarily used during **buffer placement**, which inserts buffers in the dataflow circuit. While basic buffer placement (i.e., `on-merges`) ignores timing, the advanced MILP algorithms (fpga20 and flp22) rely heavily on this information to optimize circuit performance and area.

Timing information (especially reg2reg delays) is also used in the **backend**, in order to generate appropriate RTL units which meet speed requirements. 

# Implementation Overview

In this section, we present the data structures used to store timing information, along with the code that extracts this information from the JSON and populates those structures.

## Core Data Structures

The timing system uses the following core data structures:

- **[TimingDatabase](https://github.com/EPFL-LAP/dynamatic/blob/main/include/dynamatic/Support/TimingModels.h#L221)**: IR-level timing container
  - Contains the timing data for the entire IR.
  - Stores multiple `TimingModel` instances (one per operation).
  - Provides accessor methods to retrieve timing information.
  - Gets populated from the JSON file during buffer placement passes.

- **[TimingModel](https://github.com/EPFL-LAP/dynamatic/blob/main/include/dynamatic/Support/TimingModels.h#L151)**: Per-operation timing data container
  - Encapsulates all timing data for a single operation (latencies and delays).
  - Uses `BitwidthDepMetric` structure to represent bitwidth-dependent values (see below).
  - Contains nested `PortModel` structures for port2reg delay information. 

- **[PortModel](https://github.com/KillianMcCourt/dynamatic/blob/pr1-clean/include/dynamatic/Support/TimingModels.h#L152)** : Port2reg delay values container

  - There are two objects of this class in the Timing Model class for input port and output port.

  - This structure contains three fields : data, valid and ready delays. The first one is represented using the `BitwidthDepMetric` structure.

- **[BitwidthDepMetric](https://github.com/EPFL-LAP/dynamatic/blob/main/include/dynamatic/Support/TimingModels.h#L46)**: Bitwidth-dependent timing map
  - Maps bitwidths to timing information. This information can for instance be integers, or complex structures, like maps.
  - Supports queries like `getCeilMetric(bitwidth)` to return the timing value for the closest equal or greater supported bitwidth.


- **[DelayDepMetric](https://github.com/EPFL-LAP/dynamatic/blob/main/include/dynamatic/Support/TimingModels.h#L46)**: Bitwidth-dependent timing map
  - Maps delays to timing values (e.g., for delay 3.5ns → 9 cycles)
  - Supports queries like `getDelayMetric(targetCP)` to return the timing value for the highest listed delay that remains smaller than the targetCP.


## Loading Timing Data from JSON
Before detailing the process, an introduction of the main functions involved is required :

- ```fromJSON((const ljson::Value &jsonValue, T &target, ljson::Path path)``` : this is the primary function used, with a number of overloads for various T object types. These overloads are, in order :  [first called](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L373) on the TimingDatabase, then [on every](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L330) TimingModel inside the Database, then on individual fields(example for BitwidthDepMetric [here](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L373) ); PortModels also have a dedicated [overload](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L396) .

- [```deserializenested((ArrayRef<std::string> keys, const ljson::Object *object, T &out, ljson::Path path)```](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L256): this function is called by the TimingModel fromJSON. It calls the ```fromJSON(*value, out, currentPath)```   for the inidividual fields, by iterating across the path provided by the ```TimingModel```-level ```fromJSON```. Therefore, it handles the deserialisation of said fields, by passing back the object deserialized. 

The process follows these steps:

1. **Initialization**  
   Create an empty `TimingDatabase`, and call the initialization `readFromJSON` on it. This function:

   1.1 **File Reading**  
   Loads the entire contents of the `components.json` into a string, and then parses it as a JSON.

   1.2 **Begin Extraction**  
   We then call `fromJSON` on the `TimingDatabase` and the parsed JSON to begin the deserialization process.

2. **Deserialization**  
   The [`TimingDatabase`](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L439) `fromJSON` overload iterates over the JSON object, where each key represents an operation name and the values are the timing information. For every found operation, it will :

   2.1 **Create a `TimingModel`** instance.

   2.2 **Call `fromJSON` on that `TimingModel` and the parsed JSON**. This `fromJSON` contains a list of timing characteristics that are to be filled. For each it uses predefined string arrays as nested key paths, for example :`Data delays: {"delay", "data"}`.

   - 2.2.1 **For each field and it's nested key path**, it will call [`deserializeNested`](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L268). This function validates that each step in the path exists and is the correct type (object vs value) exists.  
   - 
   - 2.2.2 This in turn **calls the appropriate `fromJSON` and writes the result back into the field**. For example,for [`BitwidthDepMetric<double>`](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L295), the fromJSON parses integer bitwidth keys and their associated timing values, writing results back into the `TimingModel`which made the request.

   2.3 Once every key listed in 2.2 has been handled, we **Write back the `TimingModel` into the database.**
  
Once deserialisation is done for **all operators**, the database will contain the full information of the JSON. 

##  Core Functions of Data Structures

### TimingDatabase

The TimingDatabase provides several core methods:

1. **[bool insertTimingModel(StringRef name, TimingModel &model)](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L183)**: inserts the timing model `model` with the key `name` in the TimingDatabase.

2. **[TimingModel* getModel(OperationName opName)](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L183)**: returns the TimingModel of operation with name `opName`.

3. **[TimingModel* getModel(Operation* op)](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L183)**: returns the TimingModel of operation `op`.

4.  **[LogicalResult getLatency(Operation *op, SignalType signalType, double &latency)](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L114)**: queries the latency of a certain operation `op` for output port of type `signalType` and it saves the latency as unsigned cycle count in the `latency` variable.

5. **[LogicalResult getInternalDelay(Operation *op, SignalType signalType, double &delay)](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L143)**: queries the reg2reg internal delay of a certain operation `op` for output port of type `signalType` and it saves the delay as a double (in nanoseconds) in the `delay` variable.

6. **[LogicalResult getPortDelay(Operation *op, SignalType signalType, double &delay)](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L161)**: queries the port2reg delay of a certain operation `op` for input/output port of type `signalType` and it saves the delay as a double (in nanoseconds) in the `delay` variable.

7. **[LogicalResult getTotalDelay(Operation *op, SignalType signalType, double &delay)](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L183)**: queries the total delay of a certain operation `op` for output port of type `signalType` and it saves the delay as a double (in nanoseconds) in the `delay` variable.

The LogicalResult or boolean types of these functions represent the successful or unsuccessful execution of the function.

The functions 4-7 automatically handle bitwidth lookup and return the appropriate timing value for the requested operation and signal type.

### TimingModel

The TimingModel provides several core methods:


1. **[LogicalResult getTotalDataDelay(unsigned bitwidth, double &delay)]()**: queries the total data delay at a certain bitwidth `bitwidth` and it saves the delay as a double (in nanoseconds) in the `delay` variable.

2. **[double getTotalValidDelay()]()**: returns the total valid delay as a double (in nanoseconds).

3. **[double getTotalReadyDelay()]()**: returns the total ready delay as a double (in nanoseconds).

4. **[bool fromJSON(const llvm::json::Value &jsonValue, TimingModel &model, llvm::json::Path path)]()**: extracts the TimingModel information from the JSON fragment `jsonValue` located at the specified path `path` relative to the root of the full JSON structure, and stores it in the variable `model`. 

5. **[bool fromJSON(const llvm::json::Value &jsonValue, TimingModel::PortModel &model, llvm::json::Path path)]()**: extracts the PortModel information from the JSON fragment `jsonValue` located at the specified path `path` relative to the root of the full JSON structure, and stores it in the variable `model`. 

The LogicalResult or boolean types of these functions represent the successful or unsuccessful execution of the function.

### BitwidthDepMetric

The main function of BitwidthDepMetric is the following:

1. **[LogicalResult getCeilMetric(unsigned bitwidth, M &metric)]()**: queries the metric with the smallest key among the ones with a key bigger than `bitwidth` and saves the metric in the variable `metric`.

### DelayDepMetric

The functions of BitwidthDepMetric are the following:  

1. **LogicalResult [getDelayCeilMetric(double targetPeriod, M &metric)](https://github.com/EPFL-LAP/dynamatic/blob/doc_branch_2/include/dynamatic/Support/TimingModels.h#L109)**: finds the highest delay that does not exceed the targetPeriod and returns the corresponding metric value. This selects the fastest implementation that still meets timing constraints. If no suitable delay is found, falls back to the lowest available delay with a critical warning.

2. **LogicalResult [getDelayCeilValue(double targetPeriod, double &delay)](https://github.com/EPFL-LAP/dynamatic/blob/doc_branch_2/include/dynamatic/Support/TimingModels.h#L142)**: similar to getDelayCeilMetric but returns the delay value itself rather than the associated metric. Finds the highest delay that is less than or equal to targetPeriod, or falls back to the minimum delay if no suitable option exists.w


# Timing Information in the IRs

Timing information is generally used immediately upon being obtained, for instance latency is obtained for the MILP solver during the buffer placement stage. However, the reg2eg internal delay must be made available in the backend to select the correct implementation to instantiate, but depends on targetCP which isn't known in the backend. 

Therefore, internal delay is added as an attribute to arithmetic ops in the IR at the end of the buffer placement stage, and is represented in the hardware IR. The value given is chosen using getDelayCeilValue, ensuring the choice passed into the IR is the same one that was made at any other point with getDelayCeilMetric. 

Sample code of the attribute :

in handhsake IR :

```
    %57 = addf %56, %54 {...
internal_delay = "3_649333"} : <f32>
```

in hardware IR : 

```
hw.module.extern @handshake_addf_0(... INTERNAL_DELAY = "3_649333"}}

```





# Timing Information in FloPoCo units - Current architecture naming standard

FloPoCo units are identified by the triplet **{operator name, bitwidth, measured internal delay}** which serves to uniquely identify them. The "measured internal delay" refers to the reg2reg delay obtained from Vivado's post-place-and-route timing analysis, which provides the actual achieved delay rather than the target specification

We use different VHDL architectures to differentiate between different implementations of the same operator. Each floating point wrapper file (addf.vhd, mulf.vhd, etc.) contains a separate architecture for each FloPoCo implementation, identified by a bitwidth-delay pair added as a suffix to "arch" to form a unique name. The legacy Dynamatic backend supports this approach by allowing an "arch-name" to be specified, which we leverage to select the appropriate architecture for each operator implementation

Both the operator specific wrappers and the shared flopoco reference file are generated by the [seperate unit module generator](https://github.com/ETHZ-DYNAMO/flopoco-to-dataflow-converter). See its own documentation for further details.

Consider the following example from [addf.vhd](https://github.com/EPFL-LAP/dynamatic/blob/doc_branch_2/data/vhdl/arith/flopoco/addf.vhd), which shows how all architectures are present inside the file, but distinguished by architecture name:

```
architecture arch_64_5_091333 of addf is
    
...

        operator : entity work.FloatingPointAdder_64_5_091333(arch)
        port map (
            clk   => clk,
            ce_1 => oehb_ready,
            ce_2 => oehb_ready,
            ce_3 => oehb_ready,
            ce_4 => oehb_ready,
            ce_5 => oehb_ready,
            ce_6 => oehb_ready,
            ce_7 => oehb_ready,
            X     => ip_lhs,
            Y     => ip_rhs,
            R     => ip_result
        );
end architecture;

architecture arch_64_9_068000 of addf is

...
        operator : entity work.FloatingPointAdder_64_9_068000(arch)
        port map (
            clk   => clk,
            ce_1 => oehb_ready,
            ce_2 => oehb_ready,
            X     => ip_lhs,
            Y     => ip_rhs,
            R     => ip_result
        );
end architecture;
```



Therefore, the desired version of the operator is used, based on the timing information passed through the hardware IR's INTERNAL_DELAY field and the operation bitwidth.

**Note :  usage of the dedicated flopco unit module is reccomended to ensure consistent data between the json used for timing information and the backend**.

