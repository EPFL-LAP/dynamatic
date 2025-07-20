module {
  handshake.func @switchSuppressCtrl_ctx(%val: !handshake.channel<i1>, %a: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["val", "A_in"], resNames = ["val", "A_in", "val2"]} {
    %val_forked:3 = fork [3] %val {handshake.name = "fork"} : <i1>
    %val2 = spec_v2_interpolator %val_forked#0, %val_forked#1 {handshake.name = "interpolator"} : <i1>
    end {handshake.name = "end0"} %val_forked#2, %a, %val2 : <i1>, <i1>, <i1>
  }
}
