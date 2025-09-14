module {
  handshake.func @newInduction_lhs(%A_in: !handshake.channel<i1>, %C_in: !handshake.channel<i1>, %Constant: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in", "C_in", "Constant"], resNames = ["B_out"]} {
    %constant_incr = smv_increment %Constant {handshake.name = "smv_incr"} : <i1>
    %repeating_inited = specv2_n_repeating_inits [%constant_incr] %C_in {handshake.name = "n_ri"} : [<i1>] <i1>
    %buffered = buffer %repeating_inited, bufferType = FIFO_BREAK_NONE, numSlots = 32 {handshake.name = "buffer", debugCounter = false} : <i1>
    %false_duplicated = specv2_n_false_duplicator [%Constant] %C_in {handshake.name = "n_fd"} : [<i1>] <i1>
    %passed_1 = passer %A_in [%buffered] {handshake.name = "passer1"} : <i1>, <i1>
    %passed_2 = passer %passed_1 [%false_duplicated] {handshake.name = "passer2"} : <i1>, <i1>
    end {handshake.name = "end0"} %passed_2 : <i1>
  }
}
