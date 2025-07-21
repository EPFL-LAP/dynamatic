module {
  handshake.func @introduceResolver_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["loopContinue", "confirmSpec_backedge"], resNames = ["confirmSpec"]} {
    %0 = source {handshake.bb = 0 : ui32, handshake.name = "backedge_source_confirmSpec"} : <>
    %1 = ndconstant %0 {handshake.bb = 0 : ui32, handshake.name = "backedge_constant_confirmSpec"} : <>, <i1>
    %2:2 = fork [2] %1 {handshake.bb = 0 : ui32, handshake.name = "backedge_lf_start_confirmSpec"} : <i1>
    sink %arg1 {handshake.bb = 0 : ui32, handshake.name = "backedge_sink_start_confirmSpec"} : <i1>
    %3:2 = lazy_fork [2] %7 {handshake.bb = 3 : ui32, handshake.name = "backedge_lf_end_confirmSpec"} : <i1>
    %4 = cmpi eq, %2#1, %3#1 {handshake.bb = 3 : ui32, handshake.name = "backedge_eq_confirmSpec"} : <i1>
    sink %4 {handshake.bb = 3 : ui32, handshake.name = "backedge_sink_end_confirmSpec"} : <i1>
    %5 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_loopContinue"} : <i1>
    %6 = ndwire %2#0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_confirmSpec_backedge"} : <i1>
    %7 = ndwire %12 {handshake.bb = 3 : ui32, handshake.name = "ndw_out_confirmSpec"} : <i1>
    %8:2 = fork [2] %5 {handshake.bb = 1 : ui32, handshake.name = "ctx_fork"} : <i1>
    %9:2 = fork [2] %6 {handshake.bb = 1 : ui32, handshake.name = "ctx_fork_cs"} : <i1>
    %10 = passer %8#0[%9#0] {handshake.bb = 1 : ui32, handshake.name = "ctx_passer"} : <i1>, <i1>
    %11 = spec_v2_nd_speculator %10 {handshake.bb = 1 : ui32, handshake.name = "ndspec"} : <i1>
    sink %9#1 {handshake.bb = 2 : ui32, handshake.name = "vm_sink_2"} : <i1>
    %12 = spec_v2_resolver %8#1, %11 {handshake.bb = 2 : ui32, handshake.name = "resolver"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %3#0 : <i1>
  }
}
