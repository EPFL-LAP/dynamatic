module {
  handshake.func @ri_fork_lhs(%a: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["B_out", "C_out"]} {
    %a_ri = spec_v2_repeating_init %a {handshake.name = "ri", initToken = 1 : ui1} : <i1>
    %out:2 = fork [2] %a_ri {handshake.name = "fork"} : <i1>
    end {handshake.name = "end0"} %out#0, %out#1 : <i1>, <i1>
  }
}
