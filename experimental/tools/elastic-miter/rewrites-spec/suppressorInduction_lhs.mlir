module {
  handshake.func @suppressorInduction_lhs(%short: !handshake.channel<i1>, %oldLong: !handshake.channel<i1>, %a: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["short", "oldLong", "A_in"], resNames = ["B_out"]} {
    %interpolate = spec_v2_interpolator %short, %oldLong {handshake.name = "interpolate"} : <i1>
    %oldLong_buffered = buffer %oldLong, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "buffer"} : <i1>
    %long = spec_v2_repeating_init %oldLong_buffered {handshake.name = "ri", initToken = 1 : ui1} : <i1>
    %passer1 = passer %a [%long] {handshake.name = "passer1"} : <i1>, <i1>
    %passer2 = passer %passer1 [%interpolate] {handshake.name = "passer2"} : <i1>, <i1>
    end {handshake.name = "end0"} %passer2 : <i1>
  }
}