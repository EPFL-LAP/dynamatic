module {
  handshake.func @repeating_init_lhs(%in: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["ins"], resNames = ["out"]} {
    %constant = smv_constant {handshake.name = "smv_constant"} : <i1>
    %buf = buffer %in, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "b", debugCounter = false} : <i1>
    %out = specv2_n_repeating_inits [%constant] %buf {handshake.name = "n_ri"} : [<i1>] <i1>
    end {handshake.name = "end0"} %out : <i1>
  }
}