module {
  handshake.func @ri_fork_rhs(%a: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["B_out", "C_out"]} {
    %out:2 = fork [2] %a {handshake.name = "fork"} : <i1>
    %b1 = spec_v2_repeating_init %out#0 {handshake.name = "ri1", initToken = 1 : ui1} : <i1>
    %b_buf1 = buffer %b1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "buffer1", debugCounter = false} : <i1>
    %b2 = spec_v2_repeating_init %out#1 {handshake.name = "ri2", initToken = 1 : ui1} : <i1>
    %b_buf2 = buffer %b2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "buffer2", debugCounter = false} : <i1>
    end {handshake.name = "end0"} %b_buf1, %b_buf2 : <i1>, <i1>
  }
}
