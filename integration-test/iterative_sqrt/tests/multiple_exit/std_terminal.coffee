[INFO] Using FPGA'23 for LSQ connection'
[PROD_CONS_MEM_DEP] Dependency from [^bb2] to [^bb4]
[PROD_CONS_MEM_DEP] Dependency from [^bb4] to [^bb2]
[PROD_CONS_MEM_DEP] Dependency from [^bb3] to [^bb4]
[PROD_CONS_MEM_DEP] Dependency from [^bb4] to [^bb3]
[MEM_GROUP] Group for [^bb2]; predecessors = {^bb4, }; successors = {^bb4, } 
[MEM_GROUP] Group for [^bb4]; predecessors = {^bb2, ^bb3, }; successors = {^bb2, ^bb3, } 
[MEM_GROUP] Group for [^bb3]; predecessors = {^bb4, }; successors = {^bb4, } 


[MEM_GROUP] Group for [^bb2]; predecessors = {^bb4, }; successors = {^bb4, } 
[MEM_GROUP] Group for [^bb4]; predecessors = {^bb2, ^bb3, }; successors = {^bb2, ^bb3, } 
[MEM_GROUP] Group for [^bb3]; predecessors = {^bb4, }; successors = {^bb4, } 

=== Lazy Forks per Block ===
Block: ^bb3
Fork Operation:
%38:2 = "handshake.lazy_fork"(%arg2) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
----
Block: ^bb4
Fork Operation:
%52:2 = "handshake.lazy_fork"(%arg2) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
----
Block: ^bb2
Fork Operation:
%29:2 = "handshake.lazy_fork"(%arg2) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
----


[][][]In createPhiNetwork val size is =2



[][][]In createPhiNetwork val size is =2



[][][]In createPhiNetwork val size is =2



[][][]In createPhiNetwork val size is =2


=== Lazy Forks per Block ===
Block: ^bb3
Fork Operation:
%42:2 = "handshake.lazy_fork"(%7) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
----
Block: ^bb4
Fork Operation:
%57:2 = "handshake.lazy_fork"(%56) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
----
Block: ^bb2
Fork Operation:
%33:2 = "handshake.lazy_fork"(%10) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
----

=== Merge/Select Network Created ===
%7 = "handshake.merge"(%57#0, %7, %7, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
%8 = "handshake.merge"(%42#0, %42#0, %8, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
%9 = "handshake.merge"(%33#0, %33#0, %33#0, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
%10 = "handshake.merge"(%57#0, %10, %10, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
%15:2 = "handshake.control_merge"(%6, %41#1, %55#1, %68) {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> (!handshake.control<>, !handshake.channel<i2>)
%24 = "handshake.merge"(%22, %31) {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
%34:2 = "handshake.control_merge"(%32#0) {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
%45:2 = "handshake.control_merge"(%41#0) {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
%54 = "handshake.select"(%51, %47, %43#0) {handshake.bb = 3 : ui32, handshake.name = "select0"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
%59:2 = "handshake.control_merge"(%55#0) {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
%69:2 = "handshake.control_merge"(%32#1) {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
%72 = "handshake.select"(%44#1, %71, %43#1) {handshake.bb = 5 : ui32, handshake.name = "select1"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>


=== Debug Print: Operand Producers in Function ===

-- Block #0 --
Operation: %0:4 = "handshake.lsq"(%arg0, %arg1, %33#1, %39#0, %42#1, %50#0, %57#1, %64#0, %66#0, %66#1, %69#0) {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq1"} : (memref<10xi32>, !handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
  Operand #0: null (function argument or constant?)
  Operand #1: null (function argument or constant?)
  Operand #2: Produced by: %33:2 = "handshake.lazy_fork"(%10) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #3: Produced by: %39:2 = "handshake.load"(%58#0, %0#0) {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>) (from different block → possible phi needed)
  Operand #4: Produced by: %42:2 = "handshake.lazy_fork"(%7) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #5: Produced by: %50:2 = "handshake.load"(%58#0, %0#1) {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>) (from different block → possible phi needed)
  Operand #6: Produced by: %57:2 = "handshake.lazy_fork"(%56) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #7: Produced by: %64:2 = "handshake.load"(%58#0, %0#2) {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>) (from different block → possible phi needed)
  Operand #8: Produced by: %66:2 = "handshake.store"(%58#0, %65) {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>) (from different block → possible phi needed)
  Operand #9: Produced by: %66:2 = "handshake.store"(%58#0, %65) {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>) (from different block → possible phi needed)
  Operand #10: Produced by: %69:2 = "handshake.control_merge"(%32#1) {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>) (from different block → possible phi needed)
----
Operation: %1 = "handshake.source"() {handshake.bb = 0 : ui32, handshake.name = "source0"} : () -> !handshake.control<>
----
Operation: %2 = "handshake.constant"(%1) {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
  Operand #0: Produced by: %1 = "handshake.source"() {handshake.bb = 0 : ui32, handshake.name = "source0"} : () -> !handshake.control<>
----
Operation: %3 = "handshake.source"() {handshake.bb = 0 : ui32, handshake.name = "source1"} : () -> !handshake.control<>
----
Operation: %4 = "handshake.constant"(%3) {handshake.bb = 0 : ui32, handshake.name = "constant6", value = true} : (!handshake.control<>) -> !handshake.channel<i1>
  Operand #0: Produced by: %3 = "handshake.source"() {handshake.bb = 0 : ui32, handshake.name = "source1"} : () -> !handshake.control<>
----
Operation: %5 = "handshake.constant"(%arg2) {handshake.bb = 0 : ui32, handshake.name = "mlir.undef0", value = 0 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
  Operand #0: null (function argument or constant?)
----
Operation: %6 = "handshake.br"(%arg2) {handshake.bb = 0 : ui32, handshake.name = "br1"} : (!handshake.control<>) -> !handshake.control<>
  Operand #0: null (function argument or constant?)
----
Operation: "cf.br"()[^bb1] : () -> ()
----

-- Block #1 --
Operation: %7 = "handshake.merge"(%57#0, %7, %7, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #0: Produced by: %57:2 = "handshake.lazy_fork"(%56) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #1: Produced by: %7 = "handshake.merge"(%57#0, %7, %7, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #2: Produced by: %7 = "handshake.merge"(%57#0, %7, %7, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #3: null (function argument or constant?)
----
Operation: %8 = "handshake.merge"(%42#0, %42#0, %8, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #0: Produced by: %42:2 = "handshake.lazy_fork"(%7) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #1: Produced by: %42:2 = "handshake.lazy_fork"(%7) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #2: Produced by: %8 = "handshake.merge"(%42#0, %42#0, %8, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #3: null (function argument or constant?)
----
Operation: %9 = "handshake.merge"(%33#0, %33#0, %33#0, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #0: Produced by: %33:2 = "handshake.lazy_fork"(%10) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #1: Produced by: %33:2 = "handshake.lazy_fork"(%10) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #2: Produced by: %33:2 = "handshake.lazy_fork"(%10) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #3: null (function argument or constant?)
----
Operation: %10 = "handshake.merge"(%57#0, %10, %10, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #0: Produced by: %57:2 = "handshake.lazy_fork"(%56) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #1: Produced by: %10 = "handshake.merge"(%57#0, %10, %10, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #2: Produced by: %10 = "handshake.merge"(%57#0, %10, %10, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #3: null (function argument or constant?)
----
Operation: %11:2 = "handshake.cond_br"(%31, %20) {handshake.bb = 1 : ui32, handshake.name = "cond_br25"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #0: Produced by: %31 = "handshake.andi"(%30, %23) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #1: Produced by: %20 = "handshake.mux"(%40, %27, %21) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
----
Operation: %12:2 = "handshake.cond_br"(%31, %19) {handshake.bb = 1 : ui32, handshake.name = "cond_br26"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> (!handshake.channel<i1>, !handshake.channel<i1>)
  Operand #0: Produced by: %31 = "handshake.andi"(%30, %23) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #1: Produced by: %19 = "handshake.mux"(%40, %26, %53) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
----
Operation: %13:2 = "handshake.cond_br"(%31, %18) {handshake.bb = 1 : ui32, handshake.name = "cond_br27"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #0: Produced by: %31 = "handshake.andi"(%30, %23) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #1: Produced by: %18 = "handshake.mux"(%40, %25, %54) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
----
Operation: %14:2 = "handshake.cond_br"(%31, %16) {handshake.bb = 1 : ui32, handshake.name = "cond_br28"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> (!handshake.channel<i1>, !handshake.channel<i1>)
  Operand #0: Produced by: %31 = "handshake.andi"(%30, %23) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #1: Produced by: %16 = "handshake.mux"(%40, %36, %17) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux0"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
----
Operation: %15:2 = "handshake.control_merge"(%6, %41#1, %55#1, %68) {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> (!handshake.control<>, !handshake.channel<i2>)
  Operand #0: Produced by: %6 = "handshake.br"(%arg2) {handshake.bb = 0 : ui32, handshake.name = "br1"} : (!handshake.control<>) -> !handshake.control<> (from different block → possible phi needed)
  Operand #1: Produced by: %41:2 = "handshake.cond_br"(%40, %34#0) {handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #2: Produced by: %55:2 = "handshake.cond_br"(%52, %45#0) {handshake.bb = 3 : ui32, handshake.name = "cond_br23"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #3: Produced by: %68 = "handshake.br"(%59#0) {handshake.bb = 4 : ui32, handshake.name = "br2"} : (!handshake.control<>) -> !handshake.control<> (from different block → possible phi needed)
----
Operation: %16 = "handshake.mux"(%40, %36, %17) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux0"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #0: Produced by: %40 = "handshake.cmpi"(%39#1, %38) {handshake.bb = 2 : ui32, handshake.name = "cmpi1", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #1: Produced by: %36 = "handshake.constant"(%35) {handshake.bb = 2 : ui32, handshake.name = "constant8", value = false} : (!handshake.control<>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #2: Produced by: %17 = "handshake.mux"(%52, %52, %61) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux1"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
----
Operation: %17 = "handshake.mux"(%52, %52, %61) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux1"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #0: Produced by: %52 = "handshake.cmpi"(%50#1, %49) {handshake.bb = 3 : ui32, handshake.name = "cmpi3", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #1: Produced by: %52 = "handshake.cmpi"(%50#1, %49) {handshake.bb = 3 : ui32, handshake.name = "cmpi3", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #2: Produced by: %61 = "handshake.constant"(%60) {handshake.bb = 4 : ui32, handshake.name = "constant12", value = true} : (!handshake.control<>) -> !handshake.channel<i1> (from different block → possible phi needed)
----
Operation: %18 = "handshake.mux"(%40, %25, %54) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #0: Produced by: %40 = "handshake.cmpi"(%39#1, %38) {handshake.bb = 2 : ui32, handshake.name = "cmpi1", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #1: Produced by: %25 = "handshake.mux"(%24, %5, %13#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #2: Produced by: %54 = "handshake.select"(%51, %47, %43#0) {handshake.bb = 3 : ui32, handshake.name = "select0"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32> (from different block → possible phi needed)
----
Operation: %19 = "handshake.mux"(%40, %26, %53) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #0: Produced by: %40 = "handshake.cmpi"(%39#1, %38) {handshake.bb = 2 : ui32, handshake.name = "cmpi1", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #1: Produced by: %26 = "handshake.mux"(%24, %4, %12#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #2: Produced by: %53 = "handshake.andi"(%52, %44#0) {handshake.bb = 3 : ui32, handshake.name = "andi1"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1> (from different block → possible phi needed)
----
Operation: %20 = "handshake.mux"(%40, %27, %21) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #0: Produced by: %40 = "handshake.cmpi"(%39#1, %38) {handshake.bb = 2 : ui32, handshake.name = "cmpi1", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #1: Produced by: %27 = "handshake.mux"(%24, %2, %11#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #2: Produced by: %21 = "handshake.mux"(%52, %27, %67) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
----
Operation: %21 = "handshake.mux"(%52, %27, %67) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #0: Produced by: %52 = "handshake.cmpi"(%50#1, %49) {handshake.bb = 3 : ui32, handshake.name = "cmpi3", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #1: Produced by: %27 = "handshake.mux"(%24, %2, %11#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #2: Produced by: %67 = "handshake.addi"(%58#0, %63) {handshake.bb = 4 : ui32, handshake.name = "addi1"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32> (from different block → possible phi needed)
----
Operation: %22 = "handshake.constant"(%arg2) {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : (!handshake.control<>) -> !handshake.channel<i1>
  Operand #0: null (function argument or constant?)
----
Operation: %23 = "handshake.mux"(%24, %4, %14#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #0: Produced by: %24 = "handshake.merge"(%22, %31) {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #1: Produced by: %4 = "handshake.constant"(%3) {handshake.bb = 0 : ui32, handshake.name = "constant6", value = true} : (!handshake.control<>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #2: Produced by: %14:2 = "handshake.cond_br"(%31, %16) {handshake.bb = 1 : ui32, handshake.name = "cond_br28"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> (!handshake.channel<i1>, !handshake.channel<i1>)
----
Operation: %24 = "handshake.merge"(%22, %31) {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #0: Produced by: %22 = "handshake.constant"(%arg2) {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : (!handshake.control<>) -> !handshake.channel<i1>
  Operand #1: Produced by: %31 = "handshake.andi"(%30, %23) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
----
Operation: %25 = "handshake.mux"(%24, %5, %13#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #0: Produced by: %24 = "handshake.merge"(%22, %31) {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #1: Produced by: %5 = "handshake.constant"(%arg2) {handshake.bb = 0 : ui32, handshake.name = "mlir.undef0", value = 0 : i32} : (!handshake.control<>) -> !handshake.channel<i32> (from different block → possible phi needed)
  Operand #2: Produced by: %13:2 = "handshake.cond_br"(%31, %18) {handshake.bb = 1 : ui32, handshake.name = "cond_br27"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
----
Operation: %26 = "handshake.mux"(%24, %4, %12#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #0: Produced by: %24 = "handshake.merge"(%22, %31) {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #1: Produced by: %4 = "handshake.constant"(%3) {handshake.bb = 0 : ui32, handshake.name = "constant6", value = true} : (!handshake.control<>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #2: Produced by: %12:2 = "handshake.cond_br"(%31, %19) {handshake.bb = 1 : ui32, handshake.name = "cond_br26"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> (!handshake.channel<i1>, !handshake.channel<i1>)
----
Operation: %27 = "handshake.mux"(%24, %2, %11#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #0: Produced by: %24 = "handshake.merge"(%22, %31) {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #1: Produced by: %2 = "handshake.constant"(%1) {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : (!handshake.control<>) -> !handshake.channel<i32> (from different block → possible phi needed)
  Operand #2: Produced by: %11:2 = "handshake.cond_br"(%31, %20) {handshake.bb = 1 : ui32, handshake.name = "cond_br25"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
----
Operation: %28 = "handshake.source"() {handshake.bb = 1 : ui32, handshake.name = "source2"} : () -> !handshake.control<>
----
Operation: %29 = "handshake.constant"(%28) {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 10 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
  Operand #0: Produced by: %28 = "handshake.source"() {handshake.bb = 1 : ui32, handshake.name = "source2"} : () -> !handshake.control<>
----
Operation: %30 = "handshake.cmpi"(%27, %29) {handshake.bb = 1 : ui32, handshake.name = "cmpi0", predicate = 2 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
  Operand #0: Produced by: %27 = "handshake.mux"(%24, %2, %11#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #1: Produced by: %29 = "handshake.constant"(%28) {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 10 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
----
Operation: %31 = "handshake.andi"(%30, %23) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #0: Produced by: %30 = "handshake.cmpi"(%27, %29) {handshake.bb = 1 : ui32, handshake.name = "cmpi0", predicate = 2 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
  Operand #1: Produced by: %23 = "handshake.mux"(%24, %4, %14#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
----
Operation: %32:2 = "handshake.cond_br"(%31, %15#0) {handshake.bb = 1 : ui32, handshake.name = "cond_br9"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
  Operand #0: Produced by: %31 = "handshake.andi"(%30, %23) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #1: Produced by: %15:2 = "handshake.control_merge"(%6, %41#1, %55#1, %68) {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> (!handshake.control<>, !handshake.channel<i2>)
----
Operation: "cf.cond_br"(%31)[^bb2, ^bb5] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!handshake.channel<i1>) -> ()
  Operand #0: Produced by: %31 = "handshake.andi"(%30, %23) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
----

-- Block #2 --
Operation: %33:2 = "handshake.lazy_fork"(%10) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
  Operand #0: Produced by: %10 = "handshake.merge"(%57#0, %10, %10, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<> (from different block → possible phi needed)
----
Operation: %34:2 = "handshake.control_merge"(%32#0) {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
  Operand #0: Produced by: %32:2 = "handshake.cond_br"(%31, %15#0) {handshake.bb = 1 : ui32, handshake.name = "cond_br9"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
----
Operation: %35 = "handshake.source"() {handshake.bb = 2 : ui32, handshake.name = "source6"} : () -> !handshake.control<>
----
Operation: %36 = "handshake.constant"(%35) {handshake.bb = 2 : ui32, handshake.name = "constant8", value = false} : (!handshake.control<>) -> !handshake.channel<i1>
  Operand #0: Produced by: %35 = "handshake.source"() {handshake.bb = 2 : ui32, handshake.name = "source6"} : () -> !handshake.control<>
----
Operation: %37 = "handshake.source"() {handshake.bb = 2 : ui32, handshake.name = "source7"} : () -> !handshake.control<>
----
Operation: %38 = "handshake.constant"(%37) {handshake.bb = 2 : ui32, handshake.name = "constant9", value = -1 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
  Operand #0: Produced by: %37 = "handshake.source"() {handshake.bb = 2 : ui32, handshake.name = "source7"} : () -> !handshake.control<>
----
Operation: %39:2 = "handshake.load"(%58#0, %0#0) {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #0: Produced by: %58:2 = "handshake.cond_br"(%31, %27) {handshake.bb = 4 : ui32, handshake.name = "cond_br31"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>) (from different block → possible phi needed)
  Operand #1: Produced by: %0:4 = "handshake.lsq"(%arg0, %arg1, %33#1, %39#0, %42#1, %50#0, %57#1, %64#0, %66#0, %66#1, %69#0) {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq1"} : (memref<10xi32>, !handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) (from different block → possible phi needed)
----
Operation: %40 = "handshake.cmpi"(%39#1, %38) {handshake.bb = 2 : ui32, handshake.name = "cmpi1", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
  Operand #0: Produced by: %39:2 = "handshake.load"(%58#0, %0#0) {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #1: Produced by: %38 = "handshake.constant"(%37) {handshake.bb = 2 : ui32, handshake.name = "constant9", value = -1 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
----
Operation: %41:2 = "handshake.cond_br"(%40, %34#0) {handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
  Operand #0: Produced by: %40 = "handshake.cmpi"(%39#1, %38) {handshake.bb = 2 : ui32, handshake.name = "cmpi1", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
  Operand #1: Produced by: %34:2 = "handshake.control_merge"(%32#0) {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
----
Operation: "cf.cond_br"(%40)[^bb3, ^bb1] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!handshake.channel<i1>) -> ()
  Operand #0: Produced by: %40 = "handshake.cmpi"(%39#1, %38) {handshake.bb = 2 : ui32, handshake.name = "cmpi1", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
----

-- Block #3 --
Operation: %42:2 = "handshake.lazy_fork"(%7) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
  Operand #0: Produced by: %7 = "handshake.merge"(%57#0, %7, %7, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<> (from different block → possible phi needed)
----
Operation: %43:2 = "handshake.cond_br"(%31, %25) {handshake.bb = 3 : ui32, handshake.name = "cond_br29"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #0: Produced by: %31 = "handshake.andi"(%30, %23) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #1: Produced by: %25 = "handshake.mux"(%24, %5, %13#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32> (from different block → possible phi needed)
----
Operation: %44:2 = "handshake.cond_br"(%31, %26) {handshake.bb = 3 : ui32, handshake.name = "cond_br30"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> (!handshake.channel<i1>, !handshake.channel<i1>)
  Operand #0: Produced by: %31 = "handshake.andi"(%30, %23) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #1: Produced by: %26 = "handshake.mux"(%24, %4, %12#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1> (from different block → possible phi needed)
----
Operation: %45:2 = "handshake.control_merge"(%41#0) {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
  Operand #0: Produced by: %41:2 = "handshake.cond_br"(%40, %34#0) {handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
----
Operation: %46 = "handshake.source"() {handshake.bb = 3 : ui32, handshake.name = "source8"} : () -> !handshake.control<>
----
Operation: %47 = "handshake.constant"(%46) {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 1 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
  Operand #0: Produced by: %46 = "handshake.source"() {handshake.bb = 3 : ui32, handshake.name = "source8"} : () -> !handshake.control<>
----
Operation: %48 = "handshake.source"() {handshake.bb = 3 : ui32, handshake.name = "source9"} : () -> !handshake.control<>
----
Operation: %49 = "handshake.constant"(%48) {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 0 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
  Operand #0: Produced by: %48 = "handshake.source"() {handshake.bb = 3 : ui32, handshake.name = "source9"} : () -> !handshake.control<>
----
Operation: %50:2 = "handshake.load"(%58#0, %0#1) {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #0: Produced by: %58:2 = "handshake.cond_br"(%31, %27) {handshake.bb = 4 : ui32, handshake.name = "cond_br31"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>) (from different block → possible phi needed)
  Operand #1: Produced by: %0:4 = "handshake.lsq"(%arg0, %arg1, %33#1, %39#0, %42#1, %50#0, %57#1, %64#0, %66#0, %66#1, %69#0) {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq1"} : (memref<10xi32>, !handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) (from different block → possible phi needed)
----
Operation: %51 = "handshake.cmpi"(%50#1, %49) {handshake.bb = 3 : ui32, handshake.name = "cmpi2", predicate = 0 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
  Operand #0: Produced by: %50:2 = "handshake.load"(%58#0, %0#1) {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #1: Produced by: %49 = "handshake.constant"(%48) {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 0 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
----
Operation: %52 = "handshake.cmpi"(%50#1, %49) {handshake.bb = 3 : ui32, handshake.name = "cmpi3", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
  Operand #0: Produced by: %50:2 = "handshake.load"(%58#0, %0#1) {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #1: Produced by: %49 = "handshake.constant"(%48) {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 0 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
----
Operation: %53 = "handshake.andi"(%52, %44#0) {handshake.bb = 3 : ui32, handshake.name = "andi1"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1>
  Operand #0: Produced by: %52 = "handshake.cmpi"(%50#1, %49) {handshake.bb = 3 : ui32, handshake.name = "cmpi3", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
  Operand #1: Produced by: %44:2 = "handshake.cond_br"(%31, %26) {handshake.bb = 3 : ui32, handshake.name = "cond_br30"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> (!handshake.channel<i1>, !handshake.channel<i1>)
----
Operation: %54 = "handshake.select"(%51, %47, %43#0) {handshake.bb = 3 : ui32, handshake.name = "select0"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #0: Produced by: %51 = "handshake.cmpi"(%50#1, %49) {handshake.bb = 3 : ui32, handshake.name = "cmpi2", predicate = 0 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
  Operand #1: Produced by: %47 = "handshake.constant"(%46) {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 1 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
  Operand #2: Produced by: %43:2 = "handshake.cond_br"(%31, %25) {handshake.bb = 3 : ui32, handshake.name = "cond_br29"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
----
Operation: %55:2 = "handshake.cond_br"(%52, %45#0) {handshake.bb = 3 : ui32, handshake.name = "cond_br23"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
  Operand #0: Produced by: %52 = "handshake.cmpi"(%50#1, %49) {handshake.bb = 3 : ui32, handshake.name = "cmpi3", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
  Operand #1: Produced by: %45:2 = "handshake.control_merge"(%41#0) {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
----
Operation: "cf.cond_br"(%52)[^bb4, ^bb1] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!handshake.channel<i1>) -> ()
  Operand #0: Produced by: %52 = "handshake.cmpi"(%50#1, %49) {handshake.bb = 3 : ui32, handshake.name = "cmpi3", predicate = 1 : i64} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i1>
----

-- Block #4 --
Operation: %56 = "handshake.join"(%33#0, %42#0) : (!handshake.control<>, !handshake.control<>) -> !handshake.control<>
  Operand #0: Produced by: %33:2 = "handshake.lazy_fork"(%10) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
  Operand #1: Produced by: %42:2 = "handshake.lazy_fork"(%7) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
----
Operation: %57:2 = "handshake.lazy_fork"(%56) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)
  Operand #0: Produced by: %56 = "handshake.join"(%33#0, %42#0) : (!handshake.control<>, !handshake.control<>) -> !handshake.control<>
----
Operation: %58:2 = "handshake.cond_br"(%31, %27) {handshake.bb = 4 : ui32, handshake.name = "cond_br31"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #0: Produced by: %31 = "handshake.andi"(%30, %23) {handshake.bb = 1 : ui32, handshake.name = "andi0"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> !handshake.channel<i1> (from different block → possible phi needed)
  Operand #1: Produced by: %27 = "handshake.mux"(%24, %2, %11#0) {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32> (from different block → possible phi needed)
----
Operation: %59:2 = "handshake.control_merge"(%55#0) {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
  Operand #0: Produced by: %55:2 = "handshake.cond_br"(%52, %45#0) {handshake.bb = 3 : ui32, handshake.name = "cond_br23"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
----
Operation: %60 = "handshake.source"() {handshake.bb = 4 : ui32, handshake.name = "source10"} : () -> !handshake.control<>
----
Operation: %61 = "handshake.constant"(%60) {handshake.bb = 4 : ui32, handshake.name = "constant12", value = true} : (!handshake.control<>) -> !handshake.channel<i1>
  Operand #0: Produced by: %60 = "handshake.source"() {handshake.bb = 4 : ui32, handshake.name = "source10"} : () -> !handshake.control<>
----
Operation: %62 = "handshake.source"() {handshake.bb = 4 : ui32, handshake.name = "source11"} : () -> !handshake.control<>
----
Operation: %63 = "handshake.constant"(%62) {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 1 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
  Operand #0: Produced by: %62 = "handshake.source"() {handshake.bb = 4 : ui32, handshake.name = "source11"} : () -> !handshake.control<>
----
Operation: %64:2 = "handshake.load"(%58#0, %0#2) {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #0: Produced by: %58:2 = "handshake.cond_br"(%31, %27) {handshake.bb = 4 : ui32, handshake.name = "cond_br31"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #1: Produced by: %0:4 = "handshake.lsq"(%arg0, %arg1, %33#1, %39#0, %42#1, %50#0, %57#1, %64#0, %66#0, %66#1, %69#0) {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq1"} : (memref<10xi32>, !handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) (from different block → possible phi needed)
----
Operation: %65 = "handshake.addi"(%64#1, %63) {handshake.bb = 4 : ui32, handshake.name = "addi0"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #0: Produced by: %64:2 = "handshake.load"(%58#0, %0#2) {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #1: Produced by: %63 = "handshake.constant"(%62) {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 1 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
----
Operation: %66:2 = "handshake.store"(%58#0, %65) {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #0: Produced by: %58:2 = "handshake.cond_br"(%31, %27) {handshake.bb = 4 : ui32, handshake.name = "cond_br31"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #1: Produced by: %65 = "handshake.addi"(%64#1, %63) {handshake.bb = 4 : ui32, handshake.name = "addi0"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
----
Operation: %67 = "handshake.addi"(%58#0, %63) {handshake.bb = 4 : ui32, handshake.name = "addi1"} : (!handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #0: Produced by: %58:2 = "handshake.cond_br"(%31, %27) {handshake.bb = 4 : ui32, handshake.name = "cond_br31"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
  Operand #1: Produced by: %63 = "handshake.constant"(%62) {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 1 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
----
Operation: %68 = "handshake.br"(%59#0) {handshake.bb = 4 : ui32, handshake.name = "br2"} : (!handshake.control<>) -> !handshake.control<>
  Operand #0: Produced by: %59:2 = "handshake.control_merge"(%55#0) {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
----
Operation: "cf.br"()[^bb1] : () -> ()
----

-- Block #5 --
Operation: %69:2 = "handshake.control_merge"(%32#1) {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : (!handshake.control<>) -> (!handshake.control<>, !handshake.channel<i1>)
  Operand #0: Produced by: %32:2 = "handshake.cond_br"(%31, %15#0) {handshake.bb = 1 : ui32, handshake.name = "cond_br9"} : (!handshake.channel<i1>, !handshake.control<>) -> (!handshake.control<>, !handshake.control<>) (from different block → possible phi needed)
----
Operation: %70 = "handshake.source"() {handshake.bb = 5 : ui32, handshake.name = "source12"} : () -> !handshake.control<>
----
Operation: %71 = "handshake.constant"(%70) {handshake.bb = 5 : ui32, handshake.name = "constant14", value = 2 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
  Operand #0: Produced by: %70 = "handshake.source"() {handshake.bb = 5 : ui32, handshake.name = "source12"} : () -> !handshake.control<>
----
Operation: %72 = "handshake.select"(%44#1, %71, %43#1) {handshake.bb = 5 : ui32, handshake.name = "select1"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #0: Produced by: %44:2 = "handshake.cond_br"(%31, %26) {handshake.bb = 3 : ui32, handshake.name = "cond_br30"} : (!handshake.channel<i1>, !handshake.channel<i1>) -> (!handshake.channel<i1>, !handshake.channel<i1>) (from different block → possible phi needed)
  Operand #1: Produced by: %71 = "handshake.constant"(%70) {handshake.bb = 5 : ui32, handshake.name = "constant14", value = 2 : i32} : (!handshake.control<>) -> !handshake.channel<i32>
  Operand #2: Produced by: %43:2 = "handshake.cond_br"(%31, %25) {handshake.bb = 3 : ui32, handshake.name = "cond_br29"} : (!handshake.channel<i1>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>) (from different block → possible phi needed)
----
Operation: "handshake.end"(%72, %0#3, %arg2) {handshake.bb = 5 : ui32, handshake.name = "end0"} : (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) -> ()
  Operand #0: Produced by: %72 = "handshake.select"(%44#1, %71, %43#1) {handshake.bb = 5 : ui32, handshake.name = "select1"} : (!handshake.channel<i1>, !handshake.channel<i32>, !handshake.channel<i32>) -> !handshake.channel<i32>
  Operand #1: Produced by: %0:4 = "handshake.lsq"(%arg0, %arg1, %33#1, %39#0, %42#1, %50#0, %57#1, %64#0, %66#0, %66#1, %69#0) {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq1"} : (memref<10xi32>, !handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) (from different block → possible phi needed)
  Operand #2: null (function argument or constant?)
----
Operation: "func.return"() : () -> ()
----


                convertSSAToGSAMerges:



Initial phi

[GSA] Block ^bb1 arg 0 type PHI_1
[GSA]    VALUE  : %57:2 = "handshake.lazy_fork"(%56) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)   (^bb4)
[GSA]    VALUE  : %7 = "handshake.merge"(%57#0, %7, %7, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>        (^bb1)
[GSA]    VALUE  : <block argument> of type '!handshake.control<>' at index: 2   (^bb0)
BB1

arg_num=0       number of my operands = 3         > 1 operands
Operands from:  BB4     BB1     BB0     one init operand
2 loop operands loopPhi id= 2


After Mu Placement

[GSA] Block ^bb1 arg 0 type MU_1 condition ^bb1
[GSA]    VALUE  : <block argument> of type '!handshake.control<>' at index: 2   (^bb0)
[GSA]    GATE   : PHI_2 (^bb1)
[GSA] Block ^bb1 arg 0 type PHI_2
[GSA]    VALUE  : %57:2 = "handshake.lazy_fork"(%56) : (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>)   (^bb4)
[GSA]    VALUE  : %7 = "handshake.merge"(%57#0, %7, %7, %arg2) {nphi} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) -> !handshake.control<>        (^bb1)
phi 2,  operand Size= 2:
Operands from:  BB1,    BB4,
****Common Dominator: ^bb1    
phi 2,  operand: 1:
1 , 
phi 2,  operand: 4:
shallalalllallallal
[PHI_ID] = 2
        Operand BB1      condition: 
1 

        Operand BB4      condition: 
0 

All Conditions: 0
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
Stack dump:
0.      Program arguments: /home/pourghei/dynamatic/bin/dynamatic-opt /home/pourghei/dynamatic/integration-test/iterative_sqrt/tests/multiple_exit/out/comp/handshake.mlir --handshake-analyze-lsq-usage --handshake-replace-memory-interfaces --handshake-straight-to-queue --handshake-combine-steering-logic
 #0 0x000059785683fe7d llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) /home/pourghei/dynamatic/polygeist/llvm-project/llvm/lib/Support/Unix/Signals.inc:723:11
 #1 0x000059785684036b PrintStackTraceSignalHandler(void*) /home/pourghei/dynamatic/polygeist/llvm-project/llvm/lib/Support/Unix/Signals.inc:798:1
 #2 0x000059785683e396 llvm::sys::RunSignalHandlers() /home/pourghei/dynamatic/polygeist/llvm-project/llvm/lib/Support/Signals.cpp:105:5
 #3 0x0000597856840b85 SignalHandler(int) /home/pourghei/dynamatic/polygeist/llvm-project/llvm/lib/Support/Unix/Signals.inc:413:1
 #4 0x000072be66442520 (/lib/x86_64-linux-gnu/libc.so.6+0x42520)
 #5 0x000072be664969fc __pthread_kill_implementation ./nptl/./nptl/pthread_kill.c:44:76
 #6 0x000072be664969fc __pthread_kill_internal ./nptl/./nptl/pthread_kill.c:78:10
 #7 0x000072be664969fc pthread_kill ./nptl/./nptl/pthread_kill.c:89:10
 #8 0x000072be66442476 gsignal ./signal/../sysdeps/posix/raise.c:27:6
 #9 0x000072be664287f3 abort ./stdlib/./stdlib/abort.c:81:7
#10 0x00005978566fe679 (/home/pourghei/dynamatic/bin/dynamatic-opt+0x1837679)
#11 0x0000597857ae089d std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >::front() /usr/bin/../lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/bits/stl_queue.h:218:2
#12 0x0000597857add1b7 dynamatic::experimental::gsa::GSAAnalysis::expandGammaTree(std::vector<std::pair<dynamatic::experimental::boolean::BoolExpression*, dynamatic::experimental::gsa::GateInput*>, std::allocator<std::pair<dynamatic::experimental::boolean::BoolExpression*, dynamatic::experimental::gsa::GateInput*> > >&, std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, dynamatic::experimental::gsa::Gate*, dynamatic::experimental::ftd::BlockIndexing const&) /home/pourghei/dynamatic/experimental/lib/Analysis/GSAAnalysis.cpp:159:29
#13 0x0000597857adbe83 dynamatic::experimental::gsa::GSAAnalysis::convertPhiToGamma(mlir::Region&, dynamatic::experimental::ftd::BlockIndexing const&) /home/pourghei/dynamatic/experimental/lib/Analysis/GSAAnalysis.cpp:560:11
#14 0x0000597857ada575 dynamatic::experimental::gsa::GSAAnalysis::convertSSAToGSAMerges(dynamatic::handshake::MergeOp&, mlir::Region&) /home/pourghei/dynamatic/experimental/lib/Analysis/GSAAnalysis.cpp:97:1
#15 0x0000597857ada21c dynamatic::experimental::gsa::GSAAnalysis::GSAAnalysis(dynamatic::handshake::MergeOp&, mlir::Region&) /home/pourghei/dynamatic/experimental/lib/Analysis/GSAAnalysis.cpp:37:1
#16 0x0000597856cf448d dynamatic::experimental::ftd::replaceMergeToGSA(dynamatic::handshake::FuncOp&, mlir::PatternRewriter&) /home/pourghei/dynamatic/experimental/lib/Support/FtdImplementation.cpp:1294:33
#17 0x0000597856d5351a (anonymous namespace)::applyStraightToQueue(dynamatic::handshake::FuncOp, mlir::MLIRContext*) /home/pourghei/dynamatic/experimental/lib/Transforms/HandshakeStraightToQueue.cpp:581:14
#18 0x0000597856d5229f (anonymous namespace)::HandshakeStraightToQueuePass::runDynamaticPass() /home/pourghei/dynamatic/experimental/lib/Transforms/HandshakeStraightToQueue.cpp:615:18
#19 0x000059785684ad96 dynamatic::DynamaticPass::runOnOperation() /home/pourghei/dynamatic/include/dynamatic/Support/DynamaticPass.h:76:5
#20 0x00005978581edc7b mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_7::operator()() const /home/pourghei/dynamatic/polygeist/llvm-project/mlir/lib/Pass/Pass.cpp:0:17
#21 0x00005978581edc15 void llvm::function_ref<void ()>::callback_fn<mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_7>(long) /home/pourghei/dynamatic/polygeist/llvm-project/llvm/include/llvm/ADT/STLFunctionalExtras.h:45:5
#22 0x0000597856757c59 llvm::function_ref<void ()>::operator()() const /home/pourghei/dynamatic/polygeist/llvm-project/llvm/include/llvm/ADT/STLFunctionalExtras.h:68:5
#23 0x00005978581f0d6d void mlir::MLIRContext::executeAction<mlir::PassExecutionAction, mlir::Pass&>(llvm::function_ref<void ()>, llvm::ArrayRef<mlir::IRUnit>, mlir::Pass&) /home/pourghei/dynamatic/polygeist/llvm-project/mlir/include/mlir/IR/MLIRContext.h:276:3
#24 0x00005978581e9133 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) /home/pourghei/dynamatic/polygeist/llvm-project/mlir/lib/Pass/Pass.cpp:509:17
#25 0x00005978581e96e4 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) /home/pourghei/dynamatic/polygeist/llvm-project/mlir/lib/Pass/Pass.cpp:569:16
#26 0x00005978581eb1cc mlir::PassManager::runPasses(mlir::Operation*, mlir::AnalysisManager) /home/pourghei/dynamatic/polygeist/llvm-project/mlir/lib/Pass/Pass.cpp:880:10
#27 0x00005978581eb0ef mlir::PassManager::run(mlir::Operation*) /home/pourghei/dynamatic/polygeist/llvm-project/mlir/lib/Pass/Pass.cpp:860:60
#28 0x00005978578f6bf2 performActions(llvm::raw_ostream&, std::shared_ptr<llvm::SourceMgr> const&, mlir::MLIRContext*, mlir::MlirOptMainConfig const&) /home/pourghei/dynamatic/polygeist/llvm-project/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:376:17
#29 0x00005978578f6829 processBuffer(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, mlir::MlirOptMainConfig const&, mlir::DialectRegistry&, llvm::ThreadPool*) /home/pourghei/dynamatic/polygeist/llvm-project/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:431:12
#30 0x00005978578f661b mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&)::$_2::operator()(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&) const /home/pourghei/dynamatic/polygeist/llvm-project/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:472:12
#31 0x00005978578f659d mlir::LogicalResult llvm::function_ref<mlir::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>::callback_fn<mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&)::$_2>(long, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&) /home/pourghei/dynamatic/polygeist/llvm-project/llvm/include/llvm/ADT/STLFunctionalExtras.h:45:12
#32 0x000059785794a599 llvm::function_ref<mlir::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>::operator()(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&) const /home/pourghei/dynamatic/polygeist/llvm-project/llvm/include/llvm/ADT/STLFunctionalExtras.h:68:12
#33 0x0000597857949b75 mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::function_ref<mlir::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>, llvm::raw_ostream&, bool, bool) /home/pourghei/dynamatic/polygeist/llvm-project/mlir/lib/Support/ToolUtilities.cpp:28:12
#34 0x00005978578f33d9 mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&) /home/pourghei/dynamatic/polygeist/llvm-project/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:475:10
#35 0x00005978578f38ae mlir::MlirOptMain(int, char**, llvm::StringRef, mlir::DialectRegistry&) /home/pourghei/dynamatic/polygeist/llvm-project/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:531:14
#36 0x00005978566fcaac main /home/pourghei/dynamatic/tools/dynamatic-opt/dynamatic-opt.cpp:87:23
#37 0x000072be66429d90 __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:58:16
#38 0x000072be66429e40 call_init ./csu/../csu/libc-start.c:128:20
#39 0x000072be66429e40 __libc_start_main ./csu/../csu/libc-start.c:379:5
#40 0x00005978566fc915 _start (/home/pourghei/dynamatic/bin/dynamatic-opt+0x1835915)
/home/pourghei/dynamatic/tools/dynamatic/scripts/compile.sh: line 185: 38616 Aborted                 "$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE" --handshake-analyze-lsq-usage --handshake-replace-memory-interfaces --handshake-straight-to-queue --handshake-combine-steering-logic > "$F_HANDSHAKE_SQ"
[FATAL] Failed to apply Straight to the Queue
dynamatic> write-hdl
/home/pourghei/dynamatic/bin/export-rtl: could not open input file '/home/pourghei/dynamatic/integration-test/iterative_sqrt/tests/multiple_exit/out/comp/hw.mlir': No such file or directory
[FATAL] Failed to export RTL (vhdl)
dynamatic> simulate
[INFO] Built kernel for IO gen.
[INFO] Ran kernel for IO gen.
[INFO] Launching Modelsim simulation
[FATAL] Simulation failed
dynamatic> exit

Goodbye!