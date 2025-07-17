[INFO] Compiled cf to handshake with FTD
[INFO] Using FPGA'23 for LSQ connection'
[PROD_CONS_MEM_DEP] Dependency from [^bb2] to [^bb4]
[PROD_CONS_MEM_DEP] Dependency from [^bb4] to [^bb2]
[PROD_CONS_MEM_DEP] Dependency from [^bb3] to [^bb4]
[PROD_CONS_MEM_DEP] Dependency from [^bb4] to [^bb3]
[MEM_GROUP] Group for [^bb3]; predecessors = {^bb4, }; successors = {^bb4, } 
[MEM_GROUP] Group for [^bb4]; predecessors = {^bb3, ^bb2, }; successors = {^bb3, ^bb2, } 
[MEM_GROUP] Group for [^bb2]; predecessors = {^bb4, }; successors = {^bb4, } 


[MEM_GROUP] Group for [^bb3]; predecessors = {^bb4, }; successors = {^bb4, } 
[MEM_GROUP] Group for [^bb4]; predecessors = {^bb3, ^bb2, }; successors = {^bb3, ^bb2, } 
[MEM_GROUP] Group for [^bb2]; predecessors = {^bb4, }; successors = {^bb4, } 

=== Lazy Forks per Block ===
Block: ^bb2
Fork Operation:
%22:2 = "handshake.lazy_fork"(%arg3) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
----
Block: ^bb3
Fork Operation:
%31:2 = "handshake.lazy_fork"(%arg3) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
----
Block: ^bb4
Fork Operation:
%38:2 = "handshake.lazy_fork"(%arg3) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
----

=== Lazy Forks per Block ===
Block: ^bb2
Fork Operation:
%26:2 = "handshake.lazy_fork"(%6) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
----
Block: ^bb3
Fork Operation:
%35:2 = "handshake.lazy_fork"(%9) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
----
Block: ^bb4
Fork Operation:
%43:2 = "handshake.lazy_fork"(%42) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
----

=== Merge/Select Network Created ===
%6 = "handshake.merge"(%43#0, %6, %6, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
%7 = "handshake.merge"(%26#0, %26#0, %26#0, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
%8 = "handshake.merge"(%35#0, %35#0, %8, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
%9 = "handshake.merge"(%43#0, %9, %9, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
%14 = "handshake.merge"(%13, %24) {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
%16:2 = "handshake.control_merge"(%5, %34#1, %41#1, %54) {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> (!handshake.control<>, !handshake.channel<i2>)
%27:2 = "handshake.control_merge"(%25#0) {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
%36:2 = "handshake.control_merge"(%34#0) {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
%45:2 = "handshake.control_merge"(%41#0) {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
%55:2 = "handshake.control_merge"(%25#1) {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)


=== Debug Print: Operand Producers in Function ===

-- Block #0 --
Operation: %0:4 = "handshake.lsq"(%arg0, %arg2, %26#1, %32#0, %35#1, %39#0, %43#1, %50#0, %52#0, %52#1, %55#0) {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq1"} : (memref<10xi32>, !handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
  Operand #0: null (function argument or constant?)
  Operand #1: null (function argument or constant?)
  Operand #2: Produced by: %26:2 = "handshake.lazy_fork"(%6) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #3: Produced by: %32:2 = "handshake.load"(%44#0, %0#0) {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>) (from different block → possible phi needed)
  Operand #4: Produced by: %35:2 = "handshake.lazy_fork"(%9) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #5: Produced by: %39:2 = "handshake.load"(%44#0, %0#1) {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>) (from different block → possible phi needed)
  Operand #6: Produced by: %43:2 = "handshake.lazy_fork"(%42) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #7: Produced by: %50:2 = "handshake.load"(%44#0, %0#2) {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>) (from different block → possible phi needed)
  Operand #8: Produced by: %52:2 = "handshake.store"(%44#0, %51) {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>) (from different block → possible phi needed)
  Operand #9: Produced by: %52:2 = "handshake.store"(%44#0, %51) {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>) (from different block → possible phi needed)
  Operand #10: Produced by: %55:2 = "handshake.control_merge"(%25#1) {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>) (from different block → possible phi needed)
----
Operation: %1 = "handshake.source"() {handshake.bb = 0 : ui32, handshake.name = "source0"} : () -> !handshake.control<>
----
Operation: %2 = "handshake.constant"(%1) {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
  Operand #0: Produced by: %1 = "handshake.source"() {handshake.bb = 0 : ui32, handshake.name = "source0"} : () -> !handshake.control<>
----
Operation: %3 = "handshake.source"() {handshake.bb = 0 : ui32, handshake.name = "source1"} : () -> !handshake.control<>
----
Operation: %4 = "handshake.constant"(%3) {handshake.bb = 0 : ui32, handshake.name = "constant4", value = true} : (!handshake.control<>) -> !handshake.channel<i1>
  Operand #0: Produced by: %3 = "handshake.source"() {handshake.bb = 0 : ui32, handshake.name = "source1"} : () -> !handshake.control<>
----
Operation: %5 = "handshake.br"(%arg3) {handshake.bb = 0 : ui32, handshake.name = "br1"} : (!handshake.control<>) -> !handshake.control<>
  Operand #0: null (function argument or constant?)
----
Operation: "cf.br"()[^bb1] : () -> ()
----

-- Block #1 --
Operation: %6 = "handshake.merge"(%43#0, %6, %6, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #0: Produced by: %43:2 = "handshake.lazy_fork"(%42) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #1: Produced by: %6 = "handshake.merge"(%43#0, %6, %6, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #2: Produced by: %6 = "handshake.merge"(%43#0, %6, %6, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #3: null (function argument or constant?)
----
Operation: %7 = "handshake.merge"(%26#0, %26#0, %26#0, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #0: Produced by: %26:2 = "handshake.lazy_fork"(%6) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #1: Produced by: %26:2 = "handshake.lazy_fork"(%6) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #2: Produced by: %26:2 = "handshake.lazy_fork"(%6) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #3: null (function argument or constant?)
----
Operation: %8 = "handshake.merge"(%35#0, %35#0, %8, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #0: Produced by: %35:2 = "handshake.lazy_fork"(%9) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #1: Produced by: %35:2 = "handshake.lazy_fork"(%9) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #2: Produced by: %8 = "handshake.merge"(%35#0, %35#0, %8, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #3: null (function argument or constant?)
----
Operation: %9 = "handshake.merge"(%43#0, %9, %9, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #0: Produced by: %43:2 = "handshake.lazy_fork"(%42) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #1: Produced by: %9 = "handshake.merge"(%43#0, %9, %9, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #2: Produced by: %9 = "handshake.merge"(%43#0, %9, %9, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #3: null (function argument or constant?)
----
Operation: %10:2 = "handshake.cond_br"(%24, %19) {handshake.bb = 1 : ui32, handshake.name = "cond_br14"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #0: Produced by: %24 = "handshake.andi"(%23, %21) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #1: Produced by: %19 = "handshake.mux"(%33, %22, %20) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
----
Operation: %11:2 = "handshake.cond_br"(%24, %17) {handshake.bb = 1 : ui32, handshake.name = "cond_br15"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> (!handshake.channel<i1>, !handshake.channel<i1>)
  Operand #0: Produced by: %24 = "handshake.andi"(%23, %21) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #1: Produced by: %17 = "handshake.mux"(%33, %29, %18) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux1"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
----
Operation: %12:2 = "handshake.cond_br"(%24, %15) {handshake.bb = 1 : ui32, handshake.name = "cond_br16"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #0: Produced by: %24 = "handshake.andi"(%23, %21) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #1: Produced by: %15 = "handshake.mux"(%14, %arg1, %12#0) {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
----
Operation: %13 = "handshake.constant"(%arg3) {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : (!handshake.control<>) -> !handshake.channel<i1>
  Operand #0: null (function argument or constant?)
----
Operation: %14 = "handshake.merge"(%13, %24) {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #0: Produced by: %13 = "handshake.constant"(%arg3) {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : (!handshake.control<>) -> !handshake.channel<i1>
  Operand #1: Produced by: %24 = "handshake.andi"(%23, %21) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
----
Operation: %15 = "handshake.mux"(%14, %arg1, %12#0) {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #0: Produced by: %14 = "handshake.merge"(%13, %24) {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #1: null (function argument or constant?)
  Operand #2: Produced by: %12:2 = "handshake.cond_br"(%24, %15) {handshake.bb = 1 : ui32, handshake.name = "cond_br16"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
----
Operation: %16:2 = "handshake.control_merge"(%5, %34#1, %41#1, %54) {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> (!handshake.control<>, !handshake.channel<i2>)
  Operand #0: Produced by: %5 = "handshake.br"(%arg3) {handshake.bb = 0 : ui32, handshake.name = "br1"} : (!handshake.control<>) -> !handshake.control<> (from different block → possible phi needed)
  Operand #1: Produced by: %34:2 = "handshake.cond_br"(%33, %27#0) {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #2: Produced by: %41:2 = "handshake.cond_br"(%40, %36#0) {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #3: Produced by: %54 = "handshake.br"(%45#0) {handshake.bb = 4 : ui32, handshake.name = "br2"} : (!handshake.control<>) -> !handshake.control<> (from different block → possible phi needed)
----
Operation: %17 = "handshake.mux"(%33, %29, %18) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux1"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #0: Produced by: %33 = "handshake.cmpi"(%32#1, %31) {handshake.bb = 2 : ui32, handshake.name = "cmpi1", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #1: Produced by: %29 = "handshake.constant"(%28) {handshake.bb = 2 : ui32, handshake.name = "constant5", value = false} : (!handshake.control<>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #2: Produced by: %18 = "handshake.mux"(%40, %40, %47) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
----
Operation: %18 = "handshake.mux"(%40, %40, %47) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #0: Produced by: %40 = "handshake.cmpi"(%39#1, %38) {handshake.bb = 3 : ui32, handshake.name = "cmpi2", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #1: Produced by: %40 = "handshake.cmpi"(%39#1, %38) {handshake.bb = 3 : ui32, handshake.name = "cmpi2", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #2: Produced by: %47 = "handshake.constant"(%46) {handshake.bb = 4 : ui32, handshake.name = "constant8", value = true} : (!handshake.control<>) -> !handshake.channel<i1> (from different block → possible phi needed)
----
Operation: %19 = "handshake.mux"(%33, %22, %20) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #0: Produced by: %33 = "handshake.cmpi"(%32#1, %31) {handshake.bb = 2 : ui32, handshake.name = "cmpi1", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #1: Produced by: %22 = "handshake.mux"(%14, %2, %10#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #2: Produced by: %20 = "handshake.mux"(%40, %22, %53) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
----
Operation: %20 = "handshake.mux"(%40, %22, %53) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #0: Produced by: %40 = "handshake.cmpi"(%39#1, %38) {handshake.bb = 3 : ui32, handshake.name = "cmpi2", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #1: Produced by: %22 = "handshake.mux"(%14, %2, %10#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #2: Produced by: %53 = "handshake.addi"(%44#0, %49) {handshake.bb = 4 : ui32, handshake.name = "addi1"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32> (from different block → possible phi needed)
----
Operation: %21 = "handshake.mux"(%14, %4, %11#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #0: Produced by: %14 = "handshake.merge"(%13, %24) {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #1: Produced by: %4 = "handshake.constant"(%3) {handshake.bb = 0 : ui32, handshake.name = "constant4", value = true} : (!handshake.control<>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #2: Produced by: %11:2 = "handshake.cond_br"(%24, %17) {handshake.bb = 1 : ui32, handshake.name = "cond_br15"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> (!handshake.channel<i1>, !handshake.channel<i1>)
----
Operation: %22 = "handshake.mux"(%14, %2, %10#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #0: Produced by: %14 = "handshake.merge"(%13, %24) {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #1: Produced by: %2 = "handshake.constant"(%1) {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : (!handshake.control<>) -> !handshake.channel<i32> (from different block → possible phi needed)
  Operand #2: Produced by: %10:2 = "handshake.cond_br"(%24, %19) {handshake.bb = 1 : ui32, handshake.name = "cond_br14"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
----
Operation: %23 = "handshake.cmpi"(%22, %15) {handshake.bb = 1 : ui32, handshake.name = "cmpi0", predicate = 2 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
  Operand #0: Produced by: %22 = "handshake.mux"(%14, %2, %10#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #1: Produced by: %15 = "handshake.mux"(%14, %arg1, %12#0) {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
----
Operation: %24 = "handshake.andi"(%23, %21) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #0: Produced by: %23 = "handshake.cmpi"(%22, %15) {handshake.bb = 1 : ui32, handshake.name = "cmpi0", predicate = 2 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
  Operand #1: Produced by: %21 = "handshake.mux"(%14, %4, %11#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
----
Operation: %25:2 = "handshake.cond_br"(%24, %16#0) {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
  Operand #0: Produced by: %24 = "handshake.andi"(%23, %21) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #1: Produced by: %16:2 = "handshake.control_merge"(%5, %34#1, %41#1, %54) {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> (!handshake.control<>, !handshake.channel<i2>)
----
Operation: "cf.cond_br"(%24)[^bb2, ^bb5] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!handshake.channel<i1>) -> ()
  Operand #0: Produced by: %24 = "handshake.andi"(%23, %21) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
----

-- Block #2 --
Operation: %26:2 = "handshake.lazy_fork"(%6) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
  Operand #0: Produced by: %6 = "handshake.merge"(%43#0, %6, %6, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<> (from different block → possible phi needed)
----
Operation: %27:2 = "handshake.control_merge"(%25#0) {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
  Operand #0: Produced by: %25:2 = "handshake.cond_br"(%24, %16#0) {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
----
Operation: %28 = "handshake.source"() {handshake.bb = 2 : ui32, handshake.name = "source3"} : () -> !handshake.control<>
----
Operation: %29 = "handshake.constant"(%28) {handshake.bb = 2 : ui32, handshake.name = "constant5", value = false} : (!handshake.control<>) -> !handshake.channel<i1>
  Operand #0: Produced by: %28 = "handshake.source"() {handshake.bb = 2 : ui32, handshake.name = "source3"} : () -> !handshake.control<>
----
Operation: %30 = "handshake.source"() {handshake.bb = 2 : ui32, handshake.name = "source4"} : () -> !handshake.control<>
----
Operation: %31 = "handshake.constant"(%30) {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -1 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
  Operand #0: Produced by: %30 = "handshake.source"() {handshake.bb = 2 : ui32, handshake.name = "source4"} : () -> !handshake.control<>
----
Operation: %32:2 = "handshake.load"(%44#0, %0#0) {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #0: Produced by: %44:2 = "handshake.cond_br"(%24, %22) {handshake.bb = 4 : ui32, handshake.name = "cond_br17"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>) (from different block → possible phi needed)
  Operand #1: Produced by: %0:4 = "handshake.lsq"(%arg0, %arg2, %26#1, %32#0, %35#1, %39#0, %43#1, %50#0, %52#0, %52#1, %55#0) {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq1"} : (memref<10xi32>, !handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) (from different block → possible phi needed)
----
Operation: %33 = "handshake.cmpi"(%32#1, %31) {handshake.bb = 2 : ui32, handshake.name = "cmpi1", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
  Operand #0: Produced by: %32:2 = "handshake.load"(%44#0, %0#0) {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #1: Produced by: %31 = "handshake.constant"(%30) {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -1 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
----
Operation: %34:2 = "handshake.cond_br"(%33, %27#0) {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
  Operand #0: Produced by: %33 = "handshake.cmpi"(%32#1, %31) {handshake.bb = 2 : ui32, handshake.name = "cmpi1", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
  Operand #1: Produced by: %27:2 = "handshake.control_merge"(%25#0) {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
----
Operation: "cf.cond_br"(%33)[^bb3, ^bb1] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!handshake.channel<i1>) -> ()
  Operand #0: Produced by: %33 = "handshake.cmpi"(%32#1, %31) {handshake.bb = 2 : ui32, handshake.name = "cmpi1", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
----

-- Block #3 --
Operation: %35:2 = "handshake.lazy_fork"(%9) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
  Operand #0: Produced by: %9 = "handshake.merge"(%43#0, %9, %9, %arg3) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<> (from different block → possible phi needed)
----
Operation: %36:2 = "handshake.control_merge"(%34#0) {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
  Operand #0: Produced by: %34:2 = "handshake.cond_br"(%33, %27#0) {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
----
Operation: %37 = "handshake.source"() {handshake.bb = 3 : ui32, handshake.name = "source5"} : () -> !handshake.control<>
----
Operation: %38 = "handshake.constant"(%37) {handshake.bb = 3 : ui32, handshake.name = "constant7", value = 0 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
  Operand #0: Produced by: %37 = "handshake.source"() {handshake.bb = 3 : ui32, handshake.name = "source5"} : () -> !handshake.control<>
----
Operation: %39:2 = "handshake.load"(%44#0, %0#1) {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #0: Produced by: %44:2 = "handshake.cond_br"(%24, %22) {handshake.bb = 4 : ui32, handshake.name = "cond_br17"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>) (from different block → possible phi needed)
  Operand #1: Produced by: %0:4 = "handshake.lsq"(%arg0, %arg2, %26#1, %32#0, %35#1, %39#0, %43#1, %50#0, %52#0, %52#1, %55#0) {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq1"} : (memref<10xi32>, !handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) (from different block → possible phi needed)
----
Operation: %40 = "handshake.cmpi"(%39#1, %38) {handshake.bb = 3 : ui32, handshake.name = "cmpi2", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
  Operand #0: Produced by: %39:2 = "handshake.load"(%44#0, %0#1) {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #1: Produced by: %38 = "handshake.constant"(%37) {handshake.bb = 3 : ui32, handshake.name = "constant7", value = 0 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
----
Operation: %41:2 = "handshake.cond_br"(%40, %36#0) {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
  Operand #0: Produced by: %40 = "handshake.cmpi"(%39#1, %38) {handshake.bb = 3 : ui32, handshake.name = "cmpi2", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
  Operand #1: Produced by: %36:2 = "handshake.control_merge"(%34#0) {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
----
Operation: "cf.cond_br"(%40)[^bb4, ^bb1] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!handshake.channel<i1>) -> ()
  Operand #0: Produced by: %40 = "handshake.cmpi"(%39#1, %38) {handshake.bb = 3 : ui32, handshake.name = "cmpi2", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
----

-- Block #4 --
Operation: %42 = "handshake.join"(%35#0, %26#0) : (!handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #0: Produced by: %35:2 = "handshake.lazy_fork"(%9) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #1: Produced by: %26:2 = "handshake.lazy_fork"(%6) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
----
Operation: %43:2 = "handshake.lazy_fork"(%42) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
  Operand #0: Produced by: %42 = "handshake.join"(%35#0, %26#0) : (!handshake.control<>, !handshake.control<>) -> !handshake.control<>
----
Operation: %44:2 = "handshake.cond_br"(%24, %22) {handshake.bb = 4 : ui32, handshake.name = "cond_br17"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #0: Produced by: %24 = "handshake.andi"(%23, %21) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #1: Produced by: %22 = "handshake.mux"(%14, %2, %10#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32> (from different block → possible phi needed)
----
Operation: %45:2 = "handshake.control_merge"(%41#0) {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
  Operand #0: Produced by: %41:2 = "handshake.cond_br"(%40, %36#0) {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
----
Operation: %46 = "handshake.source"() {handshake.bb = 4 : ui32, handshake.name = "source6"} : () -> !handshake.control<>
----
Operation: %47 = "handshake.constant"(%46) {handshake.bb = 4 : ui32, handshake.name = "constant8", value = true} : (!handshake.control<>) -> !handshake.channel<i1>
  Operand #0: Produced by: %46 = "handshake.source"() {handshake.bb = 4 : ui32, handshake.name = "source6"} : () -> !handshake.control<>
----
Operation: %48 = "handshake.source"() {handshake.bb = 4 : ui32, handshake.name = "source7"} : () -> !handshake.control<>
----
Operation: %49 = "handshake.constant"(%48) {handshake.bb = 4 : ui32, handshake.name = "constant9", value = 1 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
  Operand #0: Produced by: %48 = "handshake.source"() {handshake.bb = 4 : ui32, handshake.name = "source7"} : () -> !handshake.control<>
----
Operation: %50:2 = "handshake.load"(%44#0, %0#2) {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #0: Produced by: %44:2 = "handshake.cond_br"(%24, %22) {handshake.bb = 4 : ui32, handshake.name = "cond_br17"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #1: Produced by: %0:4 = "handshake.lsq"(%arg0, %arg2, %26#1, %32#0, %35#1, %39#0, %43#1, %50#0, %52#0, %52#1, %55#0) {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq1"} : (memref<10xi32>, !handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) (from different block → possible phi needed)
----
Operation: %51 = "handshake.addi"(%50#1, %49) {handshake.bb = 4 : ui32, handshake.name = "addi0"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #0: Produced by: %50:2 = "handshake.load"(%44#0, %0#2) {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #1: Produced by: %49 = "handshake.constant"(%48) {handshake.bb = 4 : ui32, handshake.name = "constant9", value = 1 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
----
Operation: %52:2 = "handshake.store"(%44#0, %51) {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #0: Produced by: %44:2 = "handshake.cond_br"(%24, %22) {handshake.bb = 4 : ui32, handshake.name = "cond_br17"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #1: Produced by: %51 = "handshake.addi"(%50#1, %49) {handshake.bb = 4 : ui32, handshake.name = "addi0"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
----
Operation: %53 = "handshake.addi"(%44#0, %49) {handshake.bb = 4 : ui32, handshake.name = "addi1"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #0: Produced by: %44:2 = "handshake.cond_br"(%24, %22) {handshake.bb = 4 : ui32, handshake.name = "cond_br17"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #1: Produced by: %49 = "handshake.constant"(%48) {handshake.bb = 4 : ui32, handshake.name = "constant9", value = 1 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
----
Operation: %54 = "handshake.br"(%45#0) {handshake.bb = 4 : ui32, handshake.name = "br2"} : (!handshake.control<>) -> !handshake.control<>
  Operand #0: Produced by: %45:2 = "handshake.control_merge"(%41#0) {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
----
Operation: "cf.br"()[^bb1] : () -> ()
----

-- Block #5 --
Operation: %55:2 = "handshake.control_merge"(%25#1) {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
  Operand #0: Produced by: %25:2 = "handshake.cond_br"(%24, %16#0) {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
----
Operation: "handshake.end"(%0#3, %arg3) {handshake.bb = 5 : ui32, handshake.name = "end0"} : (!handshake.control<>, !handshake.control<>) -> ()
  Operand #0: Produced by: %0:4 = "handshake.lsq"(%arg0, %arg2, %26#1, %32#0, %35#1, %39#0, %43#1, %50#0, %52#0, %52#1, %55#0) {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq1"} : (memref<10xi32>, !handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) (from different block → possible phi needed)
  Operand #1: null (function argument or constant?)
----
Operation: "func.return"() : () -> ()