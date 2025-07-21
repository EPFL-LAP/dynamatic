module {
  handshake.func @ri_fork_rhs(%a: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["B_out", "C_out"]} {
    %out:2 = fork [2] %a {handshake.name = "fork"} : <i1>
    %a_buf1 = buffer %out#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "buffer1"} : <i1>
    %b1 = spec_v2_repeating_init %a_buf1 {handshake.name = "ri1", initToken = 1 : ui1} : <i1>
    %a_buf2 = buffer %out#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "buffer2"} : <i1>
    %b2 = spec_v2_repeating_init %a_buf2 {handshake.name = "ri2", initToken = 1 : ui1} : <i1>
    end {handshake.name = "end0"} %b1, %b2 : <i1>, <i1>
  }
}
