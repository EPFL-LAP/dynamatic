# Timing Information and its handling in Dynamatic



This document explains how Dynamatic stores and uses timing information for hardware operators, providing both conceptual understanding and implementation guidance.



## What is Timing Information?



Each operator in a hardware circuit is characterized by two fundamental timing properties:



- **Latency**: The number of clock cycles an operator requires to produce a valid output after receiving valid input, assuming the output is ready to accept the result. This latency is always an integer and corresponds to the number of pipeline stages (i.e., registers) the data passes through.



- **Delay**: The combinational delay along a path — i.e., the time it takes for a signal to propagate through combinational logic, without being interrupted by clocked elements (registers). Delay is measured in physical time units (e.g., nanoseconds).





We classify combinational delays into two categories:



- **Intra-port delays**: Combinational delays from an input port to an output port with no intervening registers. These represent purely combinational paths through an operator.



- **Port2Reg delays**: Combinational delays either from an input port to the first register stage, or from the last register stage to an output port. These capture the logic surrounding the sequential boundaries of an operator.



This difference is a key distinction between **pipelined** and **non-pipelined** operations. Consider the following graph :


![image](https://github.com/user-attachments/assets/9f6f8608-7a80-4d91-a92b-bc0cb5dcf3af)




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

      "64": 0.0

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

- `latency`: A dictionary mapping bitwidths to the latency (in clock cycles) of the component at that bitwidth.

- `delays`: A dictionary describing intra-port delays — i.e., combinational delays between input and output ports with no intervening registers (in nanoseconds).

- `inport`: A dictionary specifying port2reg delays from an input port to the first register stage (in nanoseconds).

- `outport`: A dictionary specifying port2reg delays from the last register stage to an output port (in nanoseconds).





The delays dictionary is structured as follows:



- It includes three special keys: "data", "valid", and "ready". Each of these maps to a nested dictionary that captures intra-port delays between ports of the same type. In these nested dictionaries, the keys are bitwidths and the values are the corresponding delay values.



- Additional keys in the delays dictionary represent intra-port delays between different port types (e.g., from "valid" to "data"), and their values are the corresponding delay amounts.



The inport and outport dictionaries follow the same structure as the delays dictionary, capturing combinational delays between ports and registers instead of port-to-port paths.



## How Timing Information is Used



Timing data is primarily used during **buffer placement**, which inserts buffers in the dataflow circuit. While basic buffer placement (i.e., `on-merges`) ignores timing, the advanced MILP algorithms (fpga20 and flp22) rely heavily on this information to optimize circuit performance and area.



# Implementation Overview



In this section, we present the data structures used to store timing information, along with the code that extracts this information from the JSON and populates those structures.



## Core Data Structures



The timing system uses a two-level hierarchy:



**[TimingDatabase](https://github.com/EPFL-LAP/dynamatic/blob/main/include/dynamatic/Support/TimingModels.h#L174)**: IR-level timing container

- Contains the timing data for the entire IR.

- Stores multiple `TimingModel` instances (one per operation).

- Provides accessor methods to retrieve timing information.

- Gets populated from the JSON file during buffer placement passes.



**[TimingModel](https://github.com/EPFL-LAP/dynamatic/blob/main/include/dynamatic/Support/TimingModels.h#L103)**: Per-operation timing data container

- Encapsulates all timing data for a single operation (latencies and delays).

- Uses `BitwidthDepMetric` structure to represent bitwidth-dependent values (see below).

- Contains nested `PortModel` structures for port2reg delay information. 


**[PortModel](https://github.com/KillianMcCourt/dynamatic/blob/pr1-clean/include/dynamatic/Support/TimingModels.h#L152)** : Nested structure which serves to store relevevant port-to-register delays. A timing model will generally have two, for inport and outport.

-this structure contains three fields : data, valid and ready delays.



Since many timing characteristics depend on operand bitwidth, a dedicated structure is used:



**[BitwidthDepMetric](https://github.com/EPFL-LAP/dynamatic/blob/main/include/dynamatic/Support/TimingModels.h#L46)**: Bitwidth-dependent timing map

- Maps bitwidths to timing values (e.g., for latency 32-bit → 9 cycles)

- Supports queries like `getCeilMetric(bitwidth)` to return the timing value for the closest equal or greater supported bitwidth.







## Loading Timing Data from JSON

Before detailing the process, an introduction of the main functions involved is required :

```fromJSON((const ljson::Value &jsonValue, T &target, ljson::Path path)``` : this is the primary function used, with a number of overloads for various T object types. These overloads are, in order :  [first called](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L373) on the TimingDatabase, then [on every](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L330) TimingModel inside the Database, then on individual fields(example for BitwidthDepMetric [here](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L373) ); PortModels also have a dedicated[overload](https://github.com/KillianMcCourt/dynamatic/blob/pr1-clean/lib/Support/TimingModels.cpp#L426) .


[```deserializenested((ArrayRef<std::string> keys, const ljson::Object *object, T &out, ljson::Path path)```](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L256): this function is called by the TimingModel fromJSON. It calls the ```fromJSON(*value, out, currentPath)```   for the inidividual fields, by iterating across the path provided by the ```TimingModel```-level ```fromJSON```. Therefore, it handles the deserialisation of said fields, by passing back the object deserialized. 


The process follows these steps:


1. **Initialization**: Create empty TimingDatabase

2. **File Reading**: Parse the entire components.json file

3. **Data Extraction**: For each operator in the JSON:

   - Create a TimingModel instance

   - Extract latency, delay, and port timing data, with appropriate calls to a fromJSON function . 
   - Handle bitwidth-dependent values appropriately

   - Insert the completed model into the TimingDatabase
  

The JSON parsing handles the nested structure automatically, converting string keys to bitwidths and organizing delay values by signal type. 



##  Core Functions of Data structures



The TimingDatabase provides several getter methods, all of the LogicalResult type (returns a boolean success or failure state, and passes desired information back as a reference) :



- **[getLatency(Operation *op, SignalType signalType, double &latency)](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L114)**: passes back latency as unsigned cycle count for an operation of a given signal type.

- **[getInternalDelay(Operation *op, SignalType signalType, double &latency)](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L143)**: passes back processing delay as a double (in microseconds), excluding ports.

- **[getPortDelay(Operation *op, SignalType signalType, double &latency)](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L161)**: passes back delay as a double for specific input/output ports  

- **[getTotalDelay(Operation *op, SignalType signalType, double &latency)](https://github.com/EPFL-LAP/dynamatic/blob/main/lib/Support/TimingModels.cpp#L183)**: passes back the sum (as a double) of all relevant delays for an operation path, by adding internal and port delays.



These methods automatically handle bitwidth lookup and return the appropriate timing value for the requested operation and signal type.



**TODO: Should add functions per data structure** (only the public ones that can be used from users of these data structures)



---



This timing system ensures Dynamatic can generate hardware that meets timing requirements while providing the flexibility to optimize for different bitwidths and operation types.