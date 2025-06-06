module {
  handshake.func @b_rhs(%x: !handshake.channel<i32>, %y: !handshake.channel<i32>, %d: !handshake.control<>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["Xd", "Yd","Dd", "Cd"], resNames = ["A", "B"]} {
    %a = mux %index [%x, %y] {handshake.bb = 1 : ui32, handshake.name = "mux"}  : <i1>, [<i32>, <i32>] to <i32>
    %index = init %c {handshake.bb = 1 : ui32, handshake.name = "init_buffer", hw.parameters = {INITIAL_TOKEN = 0 : i1, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %a, %d : <i32>, <>
  }
}