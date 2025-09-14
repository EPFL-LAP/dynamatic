module {
  handshake.func @false_duplicator_lhs(%in: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["ins"], resNames = ["out"]} {
    %constant = smv_constant {handshake.name = "smv_constant"} : <i1>
    %out = specv2_n_false_duplicator [%constant] %in {handshake.name = "n_fd"} : [<i1>] <i1>
    end {handshake.name = "end0"} %out : <i1>
  }
}