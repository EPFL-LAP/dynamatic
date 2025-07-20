module {
  handshake.func @ri_fork_rhs(%a: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["B_out", "C_out"]} {
    %out:2 = fork [2] %a {handshake.name = "fork"} : <i1>
    %b1 = spec_v2_repeating_init %out#0 {handshake.name = "ri1", initToken = 1 : ui1} : <i1>
    %b2 = spec_v2_repeating_init %out#1 {handshake.name = "ri2", initToken = 1 : ui1} : <i1>
    end {handshake.name = "end0"} %b1, %b2 : <i1>, <i1>
  }
}
