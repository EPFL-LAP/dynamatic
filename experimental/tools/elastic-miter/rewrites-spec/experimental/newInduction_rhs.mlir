module {
  handshake.func @newInduction_rhs(%A_in: !handshake.channel<i1>, %C_in: !handshake.channel<i1>, %Constant: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in", "C_in", "Constant"], resNames = ["B_out"]} {
    %constant_incr = smv_increment %Constant {handshake.name = "smv_incr"} : <i1>
    %false_duplicated = specv2_n_false_duplicator [%constant_incr] %C_in {handshake.name = "n_fd"} : [<i1>] <i1>
    %passed = passer %A_in [%false_duplicated] {handshake.name = "passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %passed : <i1>
  }
}
