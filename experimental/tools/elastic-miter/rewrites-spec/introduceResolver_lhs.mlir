module {
  handshake.func @introduceResolver_lhs(%loopContinue: !handshake.channel<i1>, %specLoopContinue: !handshake.channel<i1>, %confirmSpec_backedge: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["loopContinue", "specLoopContinue", "confirmSpec_backedge"], resNames = ["confirmSpec"]} {
    %lc_passer = passer %loopContinue [%confirmSpec_backedge] {handshake.name = "passer"} : <i1>, <i1>
    %lc_ri = spec_v2_repeating_init %lc_passer {handshake.name = "ri", initToken = 1 : ui1} : <i1>
    %lc_buf = buffer %lc_ri, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "buffer"} : <i1>
    %confirmSpec = spec_v2_interpolator %lc_buf, %specLoopContinue {handshake.name = "interpolate"} : <i1>
    end {handshake.name = "end0"} %confirmSpec : <i1>
  }
}