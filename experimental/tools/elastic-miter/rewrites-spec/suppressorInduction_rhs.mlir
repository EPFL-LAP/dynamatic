module {
  handshake.func @suppressorInduction_rhs(%short: !handshake.channel<i1>, %oldLong: !handshake.channel<i1>, %a: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["short", "oldLong", "A_in"], resNames = ["B_out"]} {
    %long = spec_v2_repeating_init %oldLong {handshake.name = "ri", initToken = 1 : ui1} : <i1>
    %long_buffered = buffer %long, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "buffer"} : <i1>
    %interpolate = spec_v2_interpolator %short, %long_buffered {handshake.name = "interpolate"} : <i1>
    %passer1 = passer %a [%interpolate] {handshake.name = "passer"} : <i1>, <i1>
    end {handshake.name = "end"} %passer1 : <i1>
  }
}