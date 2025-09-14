module {
  handshake.func @repeating_init_rhs(%in: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["ins"], resNames = ["out"]} {
    %a = spec_v2_repeating_init %in {handshake.name = "ri1", initToken = 1 : ui1} : <i1>
    %a_buf = buffer %a, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "b1", debugCounter = false} : <i1>
    %b = spec_v2_repeating_init %a_buf {handshake.name = "ri2", initToken = 1 : ui1} : <i1>
    %b_buf = buffer %b, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "b2", debugCounter = false} : <i1>
    %c = spec_v2_repeating_init %b_buf {handshake.name = "ri3", initToken = 1 : ui1} : <i1>
    %c_buf = buffer %c, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "b3", debugCounter = false} : <i1>
    end {handshake.name = "end0"} %c_buf : <i1>
  }
}