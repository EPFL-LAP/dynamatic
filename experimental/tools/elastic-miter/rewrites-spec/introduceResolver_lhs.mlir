module {
  handshake.func @introduceResolver_lhs(%loopContinue: !handshake.channel<i1>, %specLoopContinue: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["loopContinue", "specLoopContinue"], resNames = ["confirmSpec"]} {
    %lc_passer = passer %loopContinue [%confirmSpec_forked#0] {handshake.name = "passer"} : <i1>, <i1>
    %lc_buf = buffer %lc_passer, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "buffer"} : <i1>
    %lc_ri = spec_v2_repeating_init %lc_buf {handshake.name = "ri", initialToken = 1 : ui1} : <i1>
    %confirmSpec = spec_v2_interpolator %lc_ri, %specLoopContinue {handshake.name = "interpolate"} : <i1>
    %confirmSpec_forked:2 = fork [2] %confirmSpec {handshake.name = "fork"} : <i1>
    end {handshake.name = "end0"} %confirmSpec_forked#1 : <i1>
  }
}