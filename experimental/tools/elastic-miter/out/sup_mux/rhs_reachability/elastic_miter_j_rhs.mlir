module {
  handshake.func @j_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["In1", "In2", "Ctrl"], resNames = ["Out1"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_In1"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_In2"} : <i1>
    %2 = ndwire %arg2 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_Ctrl"} : <i1>
    %3 = ndwire %8 {handshake.bb = 3 : ui32, handshake.name = "ndw_out_Out1"} : <i1>
    %4 = spec_v2_repeating_init %2 {handshake.bb = 1 : ui32, handshake.name = "ri"} : <i1>
    %5:2 = fork [2] %4 {handshake.bb = 1 : ui32, handshake.name = "fork_data_mux"} : <i1>
    %6 = init %5#0 {handshake.bb = 1 : ui32, handshake.name = "init_buffer_ctrl", hw.parameters = {INITIAL_TOKEN = false, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %7 = mux %6 [%1, %0] {handshake.bb = 1 : ui32, handshake.name = "data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %8 = passer %7[%5#1] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %3 : <i1>
  }
}
