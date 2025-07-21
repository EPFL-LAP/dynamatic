module {
  handshake.func @sup_mux_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["loopLiveIn", "oldContinue", "iterLiveOut"], resNames = ["iterLiveIn"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_loopLiveIn"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_oldContinue"} : <i1>
    %2 = ndwire %arg2 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_iterLiveOut"} : <i1>
    %3 = ndwire %7 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_iterLiveIn"} : <i1>
    %4:2 = fork [2] %1 {handshake.bb = 1 : ui32, handshake.name = "vm_fork_1"} : <i1>
    %5 = passer %2[%4#0] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>
    %6 = init %4#1 {handshake.bb = 1 : ui32, handshake.name = "oldInit", initToken = 0 : ui1} : <i1>
    %7 = mux %6 [%0, %5] {handshake.bb = 1 : ui32, handshake.name = "data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %3 : <i1>
  }
}
