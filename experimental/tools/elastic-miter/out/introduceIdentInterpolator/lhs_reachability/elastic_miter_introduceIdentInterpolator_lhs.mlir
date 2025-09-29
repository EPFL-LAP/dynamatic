module {
  handshake.func @introduceIdentInterpolator_lhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in"], resNames = ["B_out"]} {
    %0 = not %arg0 {handshake.name = "not1"} : <i1>
    %1 = not %0 {handshake.name = "not2"} : <i1>
    end {handshake.name = "end0"} %1 : <i1>
  }
}
