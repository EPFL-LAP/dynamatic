module {
  handshake.func @extension3_lhs(%arg: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["arg"], resNames = ["res"]} {
    %arg_init = init %arg {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    end {handshake.name = "end0"} %arg_init : <i1>
  }
}
