module {
  handshake.func @extension5_lhs(%arg: !handshake.channel<i1>, %ctrl: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["arg", "ctrl"], resNames = ["res"]} {
    %arg_passed = passer %arg [%ctrl] {handshake.name = "passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %arg_passed : <i1>
  }
}
