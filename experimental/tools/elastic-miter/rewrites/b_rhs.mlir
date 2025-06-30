module {
  handshake.func @b_rhs(%x: !handshake.channel<i1>, %y: !handshake.channel<i1>, %d: !handshake.control<>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.control<>) attributes {argNames = ["Xd_in", "Yd_in", "Dd_in", "Cd_in"], resNames = ["A_out", "B_out"]} {
    %a = mux %index [%x, %y] {handshake.bb = 1 : ui32, handshake.name = "mux"}  : <i1>, [<i1>, <i1>] to <i1>
    %index = init %c {handshake.bb = 1 : ui32, handshake.name = "init_buffer", hw.parameters = {INITIAL_TOKEN = 0 : i1, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %a, %d : <i1>, <>
  }
}
