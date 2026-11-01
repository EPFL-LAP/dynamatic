module {
  handshake.func @b_rhs(%Xd: <i32>, %Yd: <i32>, %Dd: <>, %Cd: <i1>) -> (A: <i32>, B: <>) {
    %a = mux %index [%Xd, %Yd] {handshake.bb = 1 : ui32, handshake.name = "mux"}  : <i1>, [<i32>, <i32>] to <i32>
    %index = init %Cd {handshake.bb = 1 : ui32, handshake.name = "init_buffer", hw.parameters = {INITIAL_TOKEN = 0 : i1, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %a, %Dd : <i32>, <>
  }
}