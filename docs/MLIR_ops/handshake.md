**TEMPLATE** header consists of a unit structure template - in case you want to create a new one, for example.  
**Mutual schemes** include all units grouped with parametres they need in format [scheme] - [list of units] 
**Dataflow units** and **Arithmetic units** correspond to dataflow (@handshake) and arithmetic (@arith) units with detailed description.

**TO DO** 
1. Add examples, where they're missing
2. After writing a code to add corresponding names (like handshake_fork_3_i32 etc)
3. Delete [no need] units & change numeration?
4. Maybe it's worth uniting the same structures for handshake and arith
5. Check this file for mistakes
6. Maybe make this file look better

# 0. TEMPLATE
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
# 1. Mutual schemes
## 1. ###handshake.[name]
### buffer (2.3)
```
### handshake.buffer
- datawidth:
- buffer type:
- number of slots:
- number of outputs
```
### num_outs & dtw
```mlir
- number of outputs:
- datawidth:
```
2 units with this parametres:
- fork (2.4)
- lazy_fork (2.5)
### num_inps & dtw
```mlir
- number of inputs:
- datawidth:
```
3 units with this parametres:
- merge (2.6)
- mux (2.7)
- control_merge (2.8)
### dtw
```mlir
- datawidth:
```
6 units with this parametres:
- br (2.9)
- cond_br (2.10)
- sink (2.11)
- source (2.12)
- d_load (3.2)
- d_store (3.3)
### constant (2.14)
```mlir
- constant value:
- datawidth:
```
### num_inps & array_dtws
```mlir
- number of inputs:
- array of datawidths:
```
3 units with this parametres:
- join (2.19)
- d_return (3.4)
- end (3.5)
### sync (2.20)
```mlir
- number of inputs:
- number of outputs:
- datawidth:
```
### mem_controller (3.1)
``` mlir
### handshake.mem_controller
- datawidth:
- number of inputs:
- array of input types (bb/load/store):
- number of store/load requests:
- array of request types (load/store):
- number of outputs:
``` 
## 2. ###arith.[name]
### 1. datawidth
``` mlir
- datawidth:
```
29 units with this parametres:
- addf (3.1)
- addi (3.2)
- andi (3.3)
- bitcast (3.4)
- ceildivsi (3.5)
- ceildivui (3.6)
- divf (3.9)
- divsi (3.10)
- divui (3.11)
- floordivsi (3.15)
- maxf (3.18)
- maxsi (3.19)
- maxui (3.20)
- minf (3.21)
- minsi (3.22)
- minui (3.23)
- mulf (3.24)
- muli (3.25)
- negf (3.26)
- ori (3.27)
- remf (3.28)
- remsi (3.29)
- remui (3.30)
- shli (3.32)
- shrsi (3.33)
- shrui (3.34)
- subf (3.36)
- subi (3.37)
- xori (3.41)
### 2. predicate & datawidth
``` mlir
- predicate:
- datawidth:
```
3 units with this parametres:
- cmpf (3.7)
- cmpi (3.8)
- select (3.31)
### 3. input dtw & output dtw
``` mlir
- input datawidth:
- output datawidth:
```
9 units with this parametres:
- extf (3.12)
- extsi (3.13)
- extui (3.14)
- fptosi (3.16)
- fptoui (3.17)
- sitofp (3.35)
- truncf (3.38)
- trunci (3.39)
- uitofp (3.40)
# 2. Dataflow units
## 1. [NO NEED] Handshake dialect function
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
## 2. OpAsmOpInterface Methods
### 1. [NO NEED] Module instantiate operation
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
### 2. [NO NEED] Handshake dialect return
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
- datawidth:
- buffer type:
- number of slots:
- number of outputs
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
### 13. [NO NEED] Never operation
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
### 15. [NO NEED] Memory
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
### 16. [NO NEED] External memory
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
### 17. [NO NEED] Load operation
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
### 18. [NO NEED] Store operation
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
- array of datawidths:
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
### 21. [NO NEED] Unpacks a tuple
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
### 22. [NO NEED] Packs a tuple
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
## 3. Dynamatic handshake operations
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
- datawidth:
- number of inputs:
- array of input types (bb/load/store):
- number of store/load requests:
- array of request types (load/store):
- number of outputs:
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
- array of datawidths:
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
- array of datawidths:
```
---  
# 3. Arithmetic units
### 1. Floating point addition operation
#### aka *addf*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.addf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

The addf operation takes two operands and returns one result, each of these is required to be the same type. This type may be a floating point scalar type, a vector whose element type is a floating point type, or a floating point tensor.

**Example:**  
  ```mlir
// Scalar addition.
%a = arith.addf %b, %c : f64

// SIMD vector addition, e.g. for Intel SSE.
%f = arith.addf %g, %h : vector<4xf32>

// Tensor addition.
%x = arith.addf %y, %z : tensor<4x?xbf16>
  ```
---
```
### arith.addf
- datawidth:
```
--- 
### 2. Integer addition operation
#### aka *addi*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.addi` $lhs `,` $rhs attr-dict `:` type($result)
```

The addi operation takes two operands and returns one result, each of these is required to be the same type. This type may be an integer scalar type, a vector whose element type is integer, or a tensor of integers. It has no standard attributes.

**Example:**  
  ```mlir
// Scalar addition.
%a = arith.addi %b, %c : i64

// SIMD vector element-wise addition, e.g. for Intel SSE.
%f = arith.addi %g, %h : vector<4xi32>

// Tensor element-wise addition.
%x = arith.addi %y, %z : tensor<4x?xi8>
  ```
---
```
### arith.addi
- datawidth:
```
--- 
### 3. Integer binary and
#### aka *andi*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.andi` $lhs `,` $rhs attr-dict `:` type($result)
```

The andi operation takes two operands and returns one result, each of these is required to be the same type. This type may be an integer scalar type, a vector whose element type is integer, or a tensor of integers. It has no standard attributes.

**Example:**  
  ```mlir
// Scalar integer bitwise and.
%a = arith.andi %b, %c : i64

// SIMD vector element-wise bitwise integer and.
%f = arith.andi %g, %h : vector<4xi32>

// Tensor element-wise bitwise integer and.
%x = arith.andi %y, %z : tensor<4x?xi8>
  ```
---
```
### arith.andi
- datawidth:
```
--- 
### 4. Bitcast between values of equal bit width
#### aka *bitcast*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.bitcast` $in attr-dict `:` type($in) `to` type($out)
```

Bitcast an integer or floating point value to an integer or floating point value of equal bit width. When operating on vectors, casts elementwise.

Note that this implements a logical bitcast independent of target endianness. This allows constant folding without target information and is consitent with the bitcast constant folders in LLVM (see https://github.com/llvm/llvm-project/blob/18c19414eb/llvm/lib/IR/ConstantFold.cpp#L168) For targets where the source and target type have the same endianness (which is the standard), this cast will also change no bits at runtime, but it may still require an operation, for example if the machine has different floating point and integer register files. For targets that have a different endianness for the source and target types (e.g. float is big-endian and integer is little-endian) a proper lowering would add operations to swap the order of words in addition to the bitcast.

**Example:**  
  ```mlir
// type here
  ```
---
```
### arith.bitcast
- datawidth:
```
--- 
### 5. Signed ceil integer division operation
#### aka *ceildivsi*  

**Description:**  
Syntax:
```mlir
operation ::= `arith.ceildivsi` $lhs `,` $rhs attr-dict `:` type($result)
```
Signed integer division. Rounds towards positive infinity, i.e. 7 / -2 = -3. 

**Example:**  
  ```mlir
// Scalar signed integer division.
%a = arith.ceildivsi %b, %c : i64
  ```
---
```
### arith.ceildivsi
- datawidth:
```
--- 
### 6. Unsigned ceil integer division operation
#### aka *ceildivsi*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.ceildivui` $lhs `,` $rhs attr-dict `:` type($result)
```

Unsigned integer division. Rounds towards positive infinity. Treats the leading bit as the most significant, i.e. for i16 given two’s complement representation, 6 / -2 = 6 / (2^16 - 2) = 1. 

**Example:**  
  ```mlir
// Scalar unsigned integer division.
%a = arith.ceildivui %b, %c : i64
  ```
---
```
### arith.ceildivui
- datawidth:
```
--- 
### 7. Floating-point comparison operation
#### aka *cmpf*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.cmpf` $predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)
```

The cmpf operation compares its two operands according to the float comparison rules and the predicate specified by the respective attribute. The predicate defines the type of comparison: (un)orderedness, (in)equality and signed less/greater than (or equal to) as well as predicates that are always true or false. The operands must have the same type, and this type must be a float type, or a vector or tensor thereof. The result is an i1, or a vector/tensor thereof having the same shape as the inputs. Unlike cmpi, the operands are always treated as signed. The u prefix indicates unordered comparison, not unsigned comparison, so “une” means unordered or not equal. For the sake of readability by humans, custom assembly form for the operation uses a string-typed attribute for the predicate. The value of this attribute corresponds to lower-cased name of the predicate constant, e.g., “one” means “ordered not equal”. The string representation of the attribute is merely a syntactic sugar and is converted to an integer attribute by the parser.

**Example:**  
  ```mlir
%r1 = arith.cmpf oeq, %0, %1 : f32
%r2 = arith.cmpf ult, %0, %1 : tensor<42x42xf64>
%r3 = "arith.cmpf"(%0, %1) {predicate: 0} : (f8, f8) -> i1
  ```
---
```
### arith.cmpf
- predicate:
- datawidth:
```
--- 
### 8. Integer comparison operation
#### aka *cmpi*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.cmpi` $predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)
```

The cmpi operation is a generic comparison for integer-like types. Its two arguments can be integers, vectors or tensors thereof as long as their types match. The operation produces an i1 for the former case, a vector or a tensor of i1 with the same shape as inputs in the other cases.

Its first argument is an attribute that defines which type of comparison is performed. The following comparisons are supported:

    equal (mnemonic: "eq"; integer value: 0)
    not equal (mnemonic: "ne"; integer value: 1)
    signed less than (mnemonic: "slt"; integer value: 2)
    signed less than or equal (mnemonic: "sle"; integer value: 3)
    signed greater than (mnemonic: "sgt"; integer value: 4)
    signed greater than or equal (mnemonic: "sge"; integer value: 5)
    unsigned less than (mnemonic: "ult"; integer value: 6)
    unsigned less than or equal (mnemonic: "ule"; integer value: 7)
    unsigned greater than (mnemonic: "ugt"; integer value: 8)
    unsigned greater than or equal (mnemonic: "uge"; integer value: 9)

The result is 1 if the comparison is true and 0 otherwise. For vector or tensor operands, the comparison is performed elementwise and the element of the result indicates whether the comparison is true for the operand elements with the same indices as those of the result.

**Example:**  
  ```mlir
// Custom form of scalar "signed less than" comparison.
%x = arith.cmpi slt, %lhs, %rhs : i32

// Generic form of the same operation.
%x = "arith.cmpi"(%lhs, %rhs) {predicate = 2 : i64} : (i32, i32) -> i1

// Custom form of vector equality comparison.
%x = arith.cmpi eq, %lhs, %rhs : vector<4xi64>

// Generic form of the same operation.
%x = "arith.cmpi"(%lhs, %rhs) {predicate = 0 : i64}
    : (vector<4xi64>, vector<4xi64>) -> vector<4xi1>
  ```
---
```
### arith.cmpi
- predicate:
- datawidth:
```
--- 
### 9. Floating point division operation
#### aka *divf*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.divf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

**Example:**  
  ```mlir
// type here
  ```
---
```
### arith.divf
- datawidth:
```
--- 
### 10. Signed integer division operation
#### aka *divsi*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.divsi` $lhs `,` $rhs attr-dict `:` type($result)
```

Signed integer division. Rounds towards zero. Treats the leading bit as sign, i.e. 6 / -2 = -3.

**Example:**  
  ```mlir
// Scalar signed integer division.
%a = arith.divsi %b, %c : i64

// SIMD vector element-wise division.
%f = arith.divsi %g, %h : vector<4xi32>

// Tensor element-wise integer division.
%x = arith.divsi %y, %z : tensor<4x?xi8>
  ```
---
```
### arith.divsi
- datawidth:
```
--- 
### 11. Unsigned integer division operation
#### aka *divui*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.divui` $lhs `,` $rhs attr-dict `:` type($result)
```

Unsigned integer division. Rounds towards zero. Treats the leading bit as the most significant, i.e. for i16 given two’s complement representation, 6 / -2 = 6 / (2^16 - 2) = 0. 

**Example:**  
  ```mlir
// Scalar unsigned integer division.
%a = arith.divui %b, %c : i64

// SIMD vector element-wise division.
%f = arith.divui %g, %h : vector<4xi32>

// Tensor element-wise integer division.
%x = arith.divui %y, %z : tensor<4x?xi8>

  ```
---
```
### arith.divui
- datawidth:
```
--- 
### 12. Cast from floating-point to wider floating-point
#### aka *extf*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.extf` $in attr-dict `:` type($in) `to` type($out)
```

Cast a floating-point value to a larger floating-point-typed value. The destination type must to be strictly wider than the source type. When operating on vectors, casts elementwise. 

**Example:**  
  ```mlir
// type here
  ```
---
```
### arith.extf
- input datawidth:
- output datawidth:
```
--- 
### 13. Integer sign extension operation
#### aka *extsi*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.extsi` $in attr-dict `:` type($in) `to` type($out)
```

The integer sign extension operation takes an integer input of width M and an integer destination type of width N. The destination bit-width must be larger than the input bit-width (N > M). The top-most (N - M) bits of the output are filled with copies of the most-significant bit of the input. 

**Example:**  
  ```mlir
%1 = arith.constant 5 : i3      // %1 is 0b101
%2 = arith.extsi %1 : i3 to i6  // %2 is 0b111101
%3 = arith.constant 2 : i3      // %3 is 0b010
%4 = arith.extsi %3 : i3 to i6  // %4 is 0b000010

%5 = arith.extsi %0 : vector<2 x i32> to vector<2 x i64>
  ```
---
```
### arith.extsi
- input datawidth:
- output datawidth:
```
--- 
### 14. Integer zero extension operation
#### aka *extui*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.extui` $in attr-dict `:` type($in) `to` type($out)
```

The integer zero extension operation takes an integer input of width M and an integer destination type of width N. The destination bit-width must be larger than the input bit-width (N > M). The top-most (N - M) bits of the output are filled with zeros.

**Example:**  
  ```mlir
  %1 = arith.constant 5 : i3      // %1 is 0b101
  %2 = arith.extui %1 : i3 to i6  // %2 is 0b000101
  %3 = arith.constant 2 : i3      // %3 is 0b010
  %4 = arith.extui %3 : i3 to i6  // %4 is 0b000010

  %5 = arith.extui %0 : vector<2 x i32> to vector<2 x i64>
  ```
---
```
### arith.extui
- input datawidth:
- output datawidth:
```
--- 
### 15. Signed floor integer division operation
#### aka *floordivsi*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.floordivsi` $lhs `,` $rhs attr-dict `:` type($result)
```

Signed integer division. Rounds towards negative infinity, i.e. 5 / -2 = -3.

**Example:**  
  ```mlir
// Scalar signed integer division.
%a = arith.floordivsi %b, %c : i64
  ```
---
```
### arith.floordivsi
- datawidth:
```
--- 
### 16. Cast from floating-point type to integer type
#### aka *fptosi*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.fptosi` $in attr-dict `:` type($in) `to` type($out)
```

Cast from a value interpreted as floating-point to the nearest (rounding towards zero) signed integer value. When operating on vectors, casts elementwise. 

**Example:**  
  ```mlir
// type here
  ```
---
```
### arith.fptosi
- input datawidth:
- output datawidth:
```
--- 
### 17. Cast from floating-point type to unsigned integer type
#### aka *fptoui*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.fptoui` $in attr-dict `:` type($in) `to` type($out)
```

Cast from a value interpreted as floating-point to the nearest (rounding towards zero) unsigned integer value. When operating on vectors, casts elementwise.

**Example:**  
  ```mlir
// type here
  ```
---
```
### arith.fptoui
- input datawidth:
- output datawidth:
```
--- 
### 18. Floating-point maximum operation
#### aka *maxf*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.maxf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```
Syntax:

```mlir
operation ::= ssa-id `=` `arith.maxf` ssa-use `,` ssa-use `:` type
```

Returns the maximum of the two arguments, treating -0.0 as less than +0.0. If one of the arguments is NaN, then the result is also NaN.

**Example:**  
  ```mlir
// Scalar floating-point maximum.
%a = arith.maxf %b, %c : f64
  ```
---
```
### arith.maxf
- datawidth:
```
--- 
### 19. Signed integer maximum operation
#### aka *maxsi*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.maxsi` $lhs `,` $rhs attr-dict `:` type($result)
```

**Example:**  
  ```mlir
// type here
  ```
---
```
### arith.maxsi
- datawidth:
```
--- 
### 20. Unsigned integer maximum operation
#### aka *maxui*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.maxui` $lhs `,` $rhs attr-dict `:` type($result)
```

**Example:**  
  ```mlir
// type here
  ```
---
```
### arith.maxui
- datawidth:
```
--- 
### 21. Floating-point minimum operation
#### aka *minf*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.minf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```
Syntax:

```mlir
operation ::= ssa-id `=` `arith.minf` ssa-use `,` ssa-use `:` type
```

Returns the minimum of the two arguments, treating -0.0 as less than +0.0. If one of the arguments is NaN, then the result is also NaN.

**Example:**  
  ```mlir
// Scalar floating-point minimum.
%a = arith.minf %b, %c : f64
  ```
---
```
### arith.minf
- datawidth:
```
--- 
### 22. Signed integer minimum operation
#### aka *minsi*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.minsi` $lhs `,` $rhs attr-dict `:` type($result)
 ```

**Example:**  
  ```mlir
// type here
  ```
---
```
### arith.minsi
- datawidth:
```
--- 
### 23. Unsigned integer minimum operation
#### aka *minui*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.minui` $lhs `,` $rhs attr-dict `:` type($result)
```

**Example:**  
  ```mlir
// type here
  ```
---
```
### arith.minui
- datawidth:
```
--- 
### 24. Floating point multiplication operation
#### aka *mulf*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.mulf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

The mulf operation takes two operands and returns one result, each of these is required to be the same type. This type may be a floating point scalar type, a vector whose element type is a floating point type, or a floating point tensor.

**Example:**  
  ```mlir
// Scalar multiplication.
%a = arith.mulf %b, %c : f64

// SIMD pointwise vector multiplication, e.g. for Intel SSE.
%f = arith.mulf %g, %h : vector<4xf32>

// Tensor pointwise multiplication.
%x = arith.mulf %y, %z : tensor<4x?xbf16>
  ```
---
```
### arith.mulf
- datawidth:
```
--- 
### 25. Integer multiplication operation
#### aka *muli*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.muli` $lhs `,` $rhs attr-dict `:` type($result)
```

**Example:**  
  ```mlir
// type here
  ```
---
```
### arith.muli
- datawidth:
```
--- 
### 26. Floating point negation
#### aka *negf*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.negf` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

The negf operation computes the negation of a given value. It takes one operand and returns one result of the same type. This type may be a float scalar type, a vector whose element type is float, or a tensor of floats. It has no standard attributes.

**Example:**  
  ```mlir
// Scalar negation value.
%a = arith.negf %b : f64

// SIMD vector element-wise negation value.
%f = arith.negf %g : vector<4xf32>

// Tensor element-wise negation value.
%x = arith.negf %y : tensor<4x?xf8>
  ```
---
```
### arith.negf
- datawidth:
```
--- 
### 27. Integer binary or
#### aka *ori*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.ori` $lhs `,` $rhs attr-dict `:` type($result)
```

The ori operation takes two operands and returns one result, each of these is required to be the same type. This type may be an integer scalar type, a vector whose element type is integer, or a tensor of integers. It has no standard attributes.

**Example:**  
  ```mlir
// Scalar integer bitwise or.
%a = arith.ori %b, %c : i64

// SIMD vector element-wise bitwise integer or.
%f = arith.ori %g, %h : vector<4xi32>

// Tensor element-wise bitwise integer or.
%x = arith.ori %y, %z : tensor<4x?xi8>
  ```
---
```
### arith.ori
- datawidth:
```
--- 
### 28. Floating point division remainder operation
#### aka *remf*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.remf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

**Example:**  
  ```mlir
// type here
  ```
---
```
### arith.remf
- datawidth:
```
--- 
### 29. Signed integer division remainder operation
#### aka *remsi*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.remsi` $lhs `,` $rhs attr-dict `:` type($result)
```

**Example:**  
  ```mlir
// Scalar signed integer division remainder.
%a = arith.remsi %b, %c : i64

// SIMD vector element-wise division remainder.
%f = arith.remsi %g, %h : vector<4xi32>

// Tensor element-wise integer division remainder.
%x = arith.remsi %y, %z : tensor<4x?xi8>
  ```
---
```
### arith.remsi
- datawidth:
```
--- 
### 30. Unsigned integer division remainder operation
#### aka *remui*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.remui` $lhs `,` $rhs attr-dict `:` type($result)
```

Unsigned integer division remainder. Treats the leading bit as the most significant, i.e. for i16, 6 % -2 = 6 % (2^16 - 2) = 6. 

**Example:**  
  ```mlir
// Scalar unsigned integer division remainder.
%a = arith.remui %b, %c : i64

// SIMD vector element-wise division remainder.
%f = arith.remui %g, %h : vector<4xi32>

// Tensor element-wise integer division remainder.
%x = arith.remui %y, %z : tensor<4x?xi8>

  ```
---
```
### arith.remui
- datawidth:
```
--- 
### 31. Select operation
#### aka *select*  

**Description:**  
The arith.select operation chooses one value based on a binary condition supplied as its first operand. If the value of the first operand is 1, the second operand is chosen, otherwise the third operand is chosen. The second and the third operand must have the same type.

The operation applies to vectors and tensors elementwise given the shape of all operands is identical. The choice is made for each element individually based on the value at the same position as the element in the condition operand. If an i1 is provided as the condition, the entire vector or tensor is chosen. 

**Example:**  
  ```mlir
// Custom form of scalar selection.
%x = arith.select %cond, %true, %false : i32

// Generic form of the same operation.
%x = "arith.select"(%cond, %true, %false) : (i1, i32, i32) -> i32

// Element-wise vector selection.
%vx = arith.select %vcond, %vtrue, %vfalse : vector<42xi1>, vector<42xf32>

// Full vector selection.
%vx = arith.select %cond, %vtrue, %vfalse : vector<42xf32>
  ```
---
```
### arith.select
- predicate:
- datawidth:
```
--- 
### 32. Integer left-shift
#### aka *shli*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.shli` $lhs `,` $rhs attr-dict `:` type($result)
```

The shli operation shifts an integer value to the left by a variable amount. The low order bits are filled with zeros. 

**Example:**  
  ```mlir
%1 = arith.constant 5 : i8                 // %1 is 0b00000101
%2 = arith.constant 3 : i8
%3 = arith.shli %1, %2 : (i8, i8) -> i8    // %3 is 0b00101000
  ```
---
```
### arith.shli
- datawidth:
```
--- 
### 33. Signed integer right-shift
#### aka *shrsi*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.shrsi` $lhs `,` $rhs attr-dict `:` type($result)
```

The shrsi operation shifts an integer value to the right by a variable amount. The integer is interpreted as signed. The high order bits in the output are filled with copies of the most-significant bit of the shifted value (which means that the sign of the value is preserved). 

**Example:**  
  ```mlir
%1 = arith.constant 160 : i8               // %1 is 0b10100000
%2 = arith.constant 3 : i8
%3 = arith.shrsi %1, %2 : (i8, i8) -> i8   // %3 is 0b11110100
%4 = arith.constant 96 : i8                   // %4 is 0b01100000
%5 = arith.shrsi %4, %2 : (i8, i8) -> i8   // %5 is 0b00001100
  ```
---
```
### arith.shrsi
- datawidth:
```
--- 
### 34. Unsigned integer right-shift
#### aka *shrui*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.shrui` $lhs `,` $rhs attr-dict `:` type($result)
```

The shrui operation shifts an integer value to the right by a variable amount. The integer is interpreted as unsigned. The high order bits are always filled with zeros.

**Example:**  
  ```mlir
%1 = arith.constant 160 : i8               // %1 is 0b10100000
%2 = arith.constant 3 : i8
%3 = arith.shrui %1, %2 : (i8, i8) -> i8   // %3 is 0b00010100
  ```
---
```
### arith.shrui
- datawidth:
```
--- 
### 35. Cast from integer type to floating-point
#### aka *sitofp*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.sitofp` $in attr-dict `:` type($in) `to` type($out)
```

Cast from a value interpreted as a signed integer to the corresponding floating-point value. If the value cannot be exactly represented, it is rounded using the default rounding mode. When operating on vectors, casts elementwise.

**Example:**  
  ```mlir
// type here
  ```
---
```
### arith.sitofp
- input datawidth:
- output datawidth:
```
--- 
### 36. Floating point subtraction operation
#### aka *subf*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.subf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

The subf operation takes two operands and returns one result, each of these is required to be the same type. This type may be a floating point scalar type, a vector whose element type is a floating point type, or a floating point tensor.

**Example:**  
  ```mlir
// Scalar subtraction.
%a = arith.subf %b, %c : f64

// SIMD vector subtraction, e.g. for Intel SSE.
%f = arith.subf %g, %h : vector<4xf32>

// Tensor subtraction.
%x = arith.subf %y, %z : tensor<4x?xbf16>
  ```
---
```
### arith.subf
- datawidth:
```
--- 
### 37. Integer subtraction operation
#### aka *subi*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.subi` $lhs `,` $rhs attr-dict `:` type($result)
```

**Example:**  
  ```mlir
// type here
  ```
---
```
### arith.subi
- datawidth:
```
--- 
### 38. Cast from floating-point to narrower floating-point
#### aka *truncf*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.truncf` $in attr-dict `:` type($in) `to` type($out)
```

Truncate a floating-point value to a smaller floating-point-typed value. The destination type must be strictly narrower than the source type. If the value cannot be exactly represented, it is rounded using the default rounding mode. When operating on vectors, casts elementwise.

**Example:**  
  ```mlir
// type here
  ```
---
```
### arith.truncf
- input datawidth:
- output datawidth:
```
--- 
### 39. Integer truncation operation
#### aka *trunci*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.trunci` $in attr-dict `:` type($in) `to` type($out)
```

The integer truncation operation takes an integer input of width M and an integer destination type of width N. The destination bit-width must be smaller than the input bit-width (N < M). The top-most (N - M) bits of the input are discarded.

**Example:**  
  ```mlir
  %1 = arith.constant 21 : i5     // %1 is 0b10101
  %2 = arith.trunci %1 : i5 to i4 // %2 is 0b0101
  %3 = arith.trunci %1 : i5 to i3 // %3 is 0b101

  %5 = arith.trunci %0 : vector<2 x i32> to vector<2 x i16>
  ```
---
```
### arith.trunci
- input datawidth:
- output datawidth:
```
--- 
### 40. Cast from unsigned integer type to floating-point
#### aka *uitofp*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.uitofp` $in attr-dict `:` type($in) `to` type($out)
```

Cast from a value interpreted as unsigned integer to the corresponding floating-point value. If the value cannot be exactly represented, it is rounded using the default rounding mode. When operating on vectors, casts elementwise.

**Example:**  
  ```mlir
// type here
  ```
---
```
### arith.uitofp
- input datawidth:
- output datawidth:
```
--- 
### 41. Integer binary xor
#### aka *xori*  

**Description:**  
Syntax:

```mlir
operation ::= `arith.xori` $lhs `,` $rhs attr-dict `:` type($result)
```

The xori operation takes two operands and returns one result, each of these is required to be the same type. This type may be an integer scalar type, a vector whose element type is integer, or a tensor of integers. It has no standard attributes.

**Example:**  
  ```mlir
// Scalar integer bitwise xor.
%a = arith.xori %b, %c : i64

// SIMD vector element-wise bitwise integer xor.
%f = arith.xori %g, %h : vector<4xi32>

// Tensor element-wise bitwise integer xor.
%x = arith.xori %y, %z : tensor<4x?xi8>
  ```
---
```
### arith.xori
- datawidth
```
--- 