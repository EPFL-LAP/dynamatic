module {
  handshake.func @resolver_lhs(%actual: !handshake.channel<i1>, %generated: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["Actual", "Generated"], resNames = ["Confirm"]} {
    %actual_passer = passer %actual [%confirm_fork#0] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>
    %actual_buf1 = buffer %actual_passer {handshake.bb = 1 : ui32, handshake.name = "buf1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %actual_buf2 = buffer %actual_buf1 {handshake.bb = 1 : ui32, handshake.name = "buf2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %actual_ri = spec_v2_repeating_init %actual_buf2 {handshake.bb = 1 : ui32, handshake.name = "ri"} : <i1>
    %confirm = spec_v2_interpolator %actual_ri, %generated {handshake.bb = 1 : ui32, handshake.name = "interpolator"} : <i1>
    %confirm_fork:2 = fork [2] %confirm {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %confirm_fork#1 : <i1>
  }
}
