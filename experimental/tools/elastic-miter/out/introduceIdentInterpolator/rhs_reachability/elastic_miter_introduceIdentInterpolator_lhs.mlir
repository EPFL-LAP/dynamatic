module {
  handshake.func @introduceIdentInterpolator_lhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in"], resNames = ["B_out"]} {
    %0:2 = fork [2] %arg0 {handshake.name = "a_fork"} : <i1>
    %1 = spec_v2_interpolator %0#0, %0#1 {handshake.name = "interpolate"} : <i1>
    end {handshake.name = "end0"} %1 : <i1>
  }
}
