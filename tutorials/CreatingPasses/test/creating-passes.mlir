module {
  handshake.func @eraseSingleInputMerge(%start: !handshake.control<>, ...) -> !handshake.control<> attributes {argNames = ["start"], resNames = ["end"]} {
    %mergeStart = merge %start : <>
    end %mergeStart : <>
  }

  handshake.func @downgradeIndexLessControlMerge(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i32>, %start: none) -> !handshake.channel<i32> attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
    %cmergeRes, %cmergeIdx = control_merge [%arg0, %arg1] : [<i32>, <i32>] to <i32>, <i1>
    end %cmergeRes : <i32>
  }

  handshake.func @isMyArgZero(%arg0: !handshake.channel<i32>, %start: !handshake.control<>) -> !handshake.channel<i1> attributes {argNames = ["arg0", "start"], resNames = ["out0"]} {
  // ^bb0:
    %mergeArg0 = merge %arg0 : <i32>
    %mergeStart = merge %start : <>
    %zero = constant %mergeStart {value = 0 : i32} : <>, <i32>
    %isArg0EqZero = cmpi eq, %mergeArg0, %zero : <i32>
    %startIsZero, %startIsNotZero = cond_br %isArg0EqZero, %mergeStart : <i1>, <>
  // ^bb1:
    %cmergeRes1, %cmergeIdx1 = control_merge [%startIsZero]: [<>] to <>, <i1>
    %arg0isZero = constant %cmergeRes1 {value = 1 : i1} : <>, <i1>
    %res1 = br %arg0isZero : <i1>
  // ^bb2:
    %cmergeRes2, %cmergeIdx2 = control_merge [%startIsNotZero] : [<>] to <>, <i1>
    %arg0isNotZero = constant %cmergeRes2 {value = 0 : i1} : <>, <i1>
    %res2 = br %arg0isNotZero : <i1>
  // ^bb3:
    %cmergeRes3, %cmergeIdx3 = control_merge [%cmergeRes1, %cmergeRes2] : [<>, <>] to <>, <i1>
    %res = mux %cmergeIdx3 [%res1, %res2] : <i1>, [<i1>, <i1>] to <i1>
    end %res : <i1>
  }
}

