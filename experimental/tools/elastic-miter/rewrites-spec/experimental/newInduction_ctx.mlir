module {
  handshake.func @newInduction_ctx(%A_in: !handshake.channel<i1>, %C_in: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in", "C_in"], resNames = ["A_in", "C_in", "Constant"]} {
    %constant = smv_constant {handshake.name = "smv_constant"} : <i1>
    end {handshake.name = "end0"} %A_in, %C_in, %constant : <i1>, <i1>, <i1>
  }
}
