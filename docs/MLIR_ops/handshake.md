# Dataflow units
## TEMPLATE
---------------------
### type here
#### aka *type here*  

**Description:**  
//type here 

**Example:**  
  ```mlir
// type here
  ```
---
```
### handshake.//type here
- //type here
```
---  
----------------------
## Handshake dialect function
#### aka *func* 

**Description:**  
The func operation represents a handshaked function.
This is almost exactly like a standard FuncOp, except that it has
some extra verification conditions. In particular, each Value must
only have a single use.

**Example:**  
  ```mlir
// type here
  ```
---
>//type here
---  
## OpAsmOpInterface Methods
### 1. Module instantiate operation
#### aka *instance*  

**Description:**  
The instance operation represents the instantiation of a module.  This
is similar to a function call, except that different instances of the
same module are guaranteed to have their own distinct state.
The instantiated module is encoded as a symbol reference attribute named
"module". An instance operation takes a control input as its last argument
and returns a control output as its last result.

**Example:**  
  ```mlir
%2:2 = handshake.instance @my_add(%0, %1, %ctrl) : (f32, f32, none) -> (f32, none)
  ```
---
```
### handshake.instance
- datawidth of module:
- number of control inputs:
- number of control outputs:
- datawidth:
```
---
### 2. Handshake dialect return
#### aka *return*  

**Description:**  
The return operation represents a handshaked
function.  This is almost exactly like a standard ReturnOp, except
that it exists in a handshake.func.  It has the same operands as
standard ReturnOp which it replaces and an additional control -
only operand(exit point of control - only network).

**Example:**  
  ```mlir
// type here
  ```
---
```
### handshake.return
- number of inputs:
- datawidth:
```
---
### 3. Buffer operation
#### aka *buffer*  

**Description:**  
The buffer operation represents a buffer operation. $slots
must be an unsigned integer larger than 0. $bufferType=BufferTypeEnum::seq indicates a
nontransparent buffer, while $bufferType=BufferTypeEnum::fifo indicates a transparent
buffer.

An 'initValues' attribute containing a list of integer values may be provided.
The list must be of the same length as the number of slots. This will
initialize the buffer with the given values upon reset.
For now, only sequential buffers are allowed to have initial values.
@todo: How to support different init types? these have to be stored (and
retrieved) as attributes, hence they must be of a known type.

**Example:**  
  ```mlir
%2:2 = handshake.buffer @my_buffer_in_ui0_out_none(%in0: !esi.channel<i0>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i0>)
  ```
---
```
### handshake.buffer
- operand datawidth:
- buffer type:
- number of slots / initValues:
```
---
### 4. Fork operation
#### aka *fork*  

**Description:**  
The fork operation represents a fork operation.  A
single input is replicated to N outputs and distributed to each
output as soon as the corresponding successor is available.

**Example:**  
  ```mlir
%1:2 = fork [2] %0 : i32
  ```
---
```
### handshake.fork
- number of outputs:
- datawidth:
```
---
### 5. Lazy fork operation
#### aka *lazy_fork*  

**Description:**  
The lazy_fork operation represents a lazy fork operation.
A single input is replicated to N outputs and distributed to each
output when all successors are available.

**Example:**  
  ```mlir
%1:2 = lazy_fork [2] %0 : i32
  ```
---
```
### handshake.lazy_fork
- number of outputs:
- datawidth:
```
---
### 6. Merge operation
#### aka *merge*  

**Description:**  
The merge operation represents a (nondeterministic)
merge operation. Any input is propagated to the single output. The
number of inputs corresponds to the number of predecessor
blocks. 

**Example:**  
  ```mlir
%0 = merge %a, %b, %c : i32
  ```
---
```
### handshake.merge
- number of inputs:
- datawidth:
```
---
### 7. Mux operation
#### aka *mux*  

**Description:**  
The mux operation represents a(deterministic) merge operation.
Operands: select, data0, data1, data2, ...

The 'select' operand is received from ControlMerge of the same
block and it represents the index of the data operand that the mux
should propagate to its single output.  The number of data inputs
corresponds to the number of predecessor blocks.

The mux operation is intended solely for control+dataflow selection.
For purely dataflow selection, use the 'select' operation instead. 

**Example:**  
  ```mlir
  %0 = mux %select [%data0, %data1, %data2] {attributes}: index, i32
  ```
---
```
### handshake.mux
- number of inputs:
- datawidth:
```
---
### 8. Control merge operation
#### aka *control_merge*  

**Description:**  
The control_merge operation represents a
(nondeterministic) control merge.  Any input is propagated to the
first output and the index of the propagated input is sent to the
second output.  The number of inputs corresponds to the number of
predecessor blocks. 

**Example:**  
  ```mlir
%0, %idx = control_merge %a, %b, %c {attributes} : i32, index
  ```
---
```
### handshake.control_merge
- number of inputs:
- datawidth:
```
---
### 9. Branch operation
#### aka *br*  

**Description:**  
The branch operation represents an unconditional
branch.  The single data input is propagated to the single
successor.  The input must be triggered by some predecessor to
avoid continous triggering of a successor block. 

**Example:**  
  ```mlir
%1 = br %0 : i32
  ```
---
```
### handshake.br
- datawidth:
```
---
### 10. Conditional branch operation
#### aka *cond_br*  

**Description:**  
The cbranch operation represents a conditional
branch.  The data input is propagated to one of the two outputs
based on the condition input.
**Example:**  
  ```mlir
%true, %false = conditional_branch %cond, %data : i32
  ```
---
```
### handshake.cond_br
- condition datawidth:
- datawidth:
```
---
### 11. Sink operation
#### aka *sink*  

**Description:**  
The sink operation discards any data that arrives at its
input.The sink has no successors and it can continuously consume data. 

**Example:**  
  ```mlir
sink %data : i32
  ```
---
```
### handshake.sink
- datawidth:
```
---
### 12. Source operation
#### aka *source*  

**Description:**  
The source operation represents continuous token
source.  The source continously sets a 'valid' signal which the
successor can consume at any point in time. 

**Example:**  
  ```mlir
// type here
  ```
---
```
### handshake.source
- datawidth:
```
---
### 13. Never operation
#### aka *never*  

**Description:**  
The never operation represents disconnected data
source. The source never sets any 'valid' signal which will
never trigger the successor at any point in time. 

**Example:**  
  ```mlir
// type here
  ```
---
```
### handshake.never
- datawidth:
```
---
### 14. Constant operation
#### aka *constant*  

**Description:**  
The const has a constant value. When triggered by its
single `ctrl` input, it sends the constant value to its single
successor. 

**Example:**  
  ```mlir
%0 = constant %ctrl {value = 42 : i32} : i32
  ```
---
```
### handshake.constant
- constant value:
- datawidth:
```
---
### 15. Memory
#### aka *memory*  

**Description:**  
Each MemoryOp represents an independent memory or memory region (BRAM or external memory).
It receives memory access requests from load and store operations. For every request,
it returns data (for load) and a data-less token indicating completion.
The memory op represents a flat, unidimensional memory.
Operands: all stores (stdata1, staddr1, stdata2, staddr2, ...), then all loads (ldaddr1, ldaddr2,...)
Outputs: all load outputs, ordered the same as
load data (lddata1, lddata2, ...), followed by all none outputs,
ordered as operands (stnone1, stnone2,...ldnone1, ldnone2,...)

**Example:**  
  ```mlir
- 
  ```
---
```
### handshake.memory
- number of inputs:
- number of outputs:
- datawidth:
```
---
### 16. External memory
#### aka *extmemory*  

**Description:**  
An ExternalMemoryOp represents a wrapper around a memref input to a
handshake function. The semantics of the load/store operands are identical
to what is decribed for MemoryOp. The only difference is that the first
operand to this operand is a memref value.
Upon lowering to FIRRTL, a handshake interface will be created in the
top-level component for each load- and store which connected to this memory. 

**Example:**  
  ```mlir
handshake.func @main(%i: index, %v: i32, %mem : memref<10xi32>, %ctrl: none) -> none {
  %stCtrl = extmemory[ld = 0, st = 1](%mem : memref<10xi32>)(%vout, %addr) {id = 0 : i32} : (i32, index) -> (none)
  %vout, %addr = store(%v, %i, %ctrl) : (i32, index, none) -> (i32, index)
  ...
}
  ```
---
```
### handshake.extmemory
- number of inputs:
- number of outputs:
- datawidth:
```
---
### 17. Load operation
#### aka *load*  

**Description:**  
Load memory port, sends load requests to MemoryOp. From dataflow
predecessor, receives address indices and a control-only value
which signals completion of all previous memory accesses which
target the same memory.  When all inputs are received, the load
sends the address indices to MemoryOp. When the MemoryOp returns
a piece of data, the load sends it to its dataflow successor.

Operands: address indices (from predecessor), data (from MemoryOp), control-only input.
Results: data (to successor), address indices (to MemoryOp).

**Example:**  
  ```mlir
%dataToSucc, %addr1ToMem, %addr2ToMem = load [%addr1, %addr2] %dataFromMem, %ctrl : i8, i16, index
  ```
---
```
### handshake.load
- number of inputs:
- number of outputs:
- datawidth:
```
---
### 18. Store operation
#### aka *store*  

**Description:**  
Store memory port, sends store requests to MemoryOp. From dataflow
predecessors, receives address indices, data, and a control-only
value which signals completion of all previous memory accesses
which target the same memory.  When all inputs are received, the
store sends the address and data to MemoryOp.

Operands: address indices, data, control-only input.
Results: data and address indices (sent to MemoryOp).
Types: data type followed by address type. 

**Example:**  
  ```mlir
%dataToMem, %addrToMem = store [%addr1, %addr2] %dataFromPred , %ctrl : i8, i16, index
  ```
---
```
### handshake.store
- number of inputs:
- number of outputs:
- datawidth:
```
---
### 19. Join operation
#### aka *join*  

**Description:**  
A control-only synchronizer.  Produces a valid output when all
inputs become available.

**Example:**  
  ```mlir
%0 = join %a, %b, %c : i32, i1, none
  ```
---
```
### handshake.join
- number of inputs:
- datawidth:
```
---
### 20. Sync operation
#### aka *sync*  

**Description:**  
Synchronizes an arbitrary set of inputs. Synchronization implies applying
join semantics in between all in- and output ports.

**Example:**  
  ```mlir
%aSynced, %bSynced, %cSynced = sync %a, %b, %c : i32, i1, none
  ```
---
```
### handshake.sync
- number of inputs:
- number of outputs:
- datawidth:
```
---
### 21. Unpacks a tuple
#### aka *unpack*  

**Description:**  
The unpack operation assigns each value of a tuple to a separate
value for further processing. The number of results corresponds
to the number of tuple elements.
Similar to fork, each output is distributed as soon as the corresponding
successor is ready. 

**Example:**  
  ```mlir
%a, %b = handshake.unpack %tuple {attributes} : tuple<i32, i64>
  ```
---
```
### handshake.unpack
- number of outputs:
- datawidth:
```
---
### 22. Packs a tuple
#### aka *pack*  

**Description:**  
The pack operation constructs a tuple from separate values.
The number of operands corresponds to the number of tuple elements.
Similar to join, the output is ready when all inputs are ready.

**Example:**  
  ```mlir
%tuple = handshake.pack %a, %b {attributes} : tuple<i32, i64>
  ```
---
```
### handshake.pack
- number of inputs:
- datawidth:
```
---
## Dynamatic handshake operations
### 1. Memory controller
#### aka *mem_controller*  

**Description:**  
  Each MemoryControllerOp represents an interface to an externally defined 
  unidimensional memory (i.e., it wraps a memref input to a handshake 
  function). It receives control signals from each basic block containing
  store operations referencing the wrapped memref; the formers are fed to the
  operation through constants indicating the number of stores the basic block
  will make to the referenced memory region. It also receives memory access
  requests from load (addr) and store (addr+data) operations. It returns a
  value (data) for each load request as well as a control signal to indicate
  basic block completion to the enclosing function's end operation.

  The operation also contains an attribute (memOps) specifying the set of
  memory operations on the referenced memory region. Each element in the
  top-level list of attributes is itself an attribute list whose elements
  identify load and store accesses performed within a single basic block,
  in program order. The total number of basic blocks, load operations and
  store operations referencing the memory region can be derived from this
  attribute.

  Finally, the operation contains a unique numer identifying the interface
  (id).

  The order of operands is  
  1. Wrapped memref
  2. For each basic block referencing the memory region:
        a. Control signal fed through a constant indicating the number of store
    operations in the block (if there is at least one store in the block)
        b. Load/Store access requests from within the block, in program order

  The order of results is
  1. Load results, in program order (i.e., in load access requests order)
  2. Control signal indicating completion

**Example:**  
  ```mlir
%ldData1, %ldData2, %ctrl = mem_controller[%mem : memref<16xi32>]
    (%bb1, %stAddr1, %stData1, %ldAddr1, %ldAddr2, %bb3, %stAddr2, %stData2) :
    {accesses = [
      [#handshake<AccessType Store>, #handshake<AccessType Load>], 
      [#handshake<AccessType Load>],
      [#handshake<AccessType Store>],
    ], id = 0 : i32}
(i32, index, i32, index, index, i32, index, i32) -> (i32, i32, none)
  ```
---
```
### handshake.mem_controller
- number of inputs:
- number of outputs:
- datawidth:
``` 
---  
### 2. Load operation
#### aka *d_load*  
**Description:**
Represents a load memory port which sends load requests to a memory
interface. It receives an address from a dataflow predecessor and a data
value from the memory interface that eventually holds the loaded data. It
returns an address that is sent to the memory interface as well as a data
value that is sent to its dataflow successor.

The order of operands is
1. address (from predecessor)
2. data value (from memory interface)

The order of results is
1. address (to memory interface).
2. data value (to successor)

**Example:**  
```mlir
%dataToSucc, %addrToMem = d_load [%addr] %dataFromMem : i32, index
```
---
```
### handshake.d_load
- datawidth:
```
---  
### 3. Store operation
#### aka *d_store*

**Description:**
Represents a store memory port which sends store requests to a memory
interface. It receives an address and a data value to store in memory from
its dataflow predecessors. It returns the value to store and an address,
both of which are sent to the memory interface.

The order of operands is
1. address (from predecessor)
2. data value (from predecessor)

The order of results is
1. address (to memory interface).
2. data value (to memory interface)

**Example:**
```mlir
%dataToMem, %addrToMem = d_store [%addr] %dataToStore : i32, index
```
---
```
### handshake.d_store
- datawidth:
```
---  
### 4. Handshake-level return
#### aka *d_return*

**Description:**
This return operation represents the handshake equivalent of a 
func::ReturnOp (i.e., the latter is converted to the former one-to-one,
preserving operands). This operation also outputs a list of values that
correspond one-to-one to its own operands (i.e., same type and semantic
value).

**Example:**
```mlir
%out1, %out2 = d_return %in1, %in2 : i32, i64
```
---
```
### handshake.d_return
- number of inputs/outputs:
- datawidth:
```
---  
### 5. Function endpoint
#### aka *end*

**Description:**
This operation is the terminator of every handshake function in Dynamatic.
Its operands are the merged results of all return operations in the function
(or simply the forwarded results of a single return) followed by the output
control signals of all memory interfaces in the function.

**Example:**
```mlir
end %res1, %res2, %ctrl1, %ctrl2 : i32, i64, none, none
```
---
```
### handshake.end
- number of inputs:
- datawidth:
```
---  