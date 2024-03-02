module {
  handshake.func @eraseSingleInputMerge(%start: none) -> none attributes {argNames = ["start"], resNames = ["out0"]} {
    %mergeStart = merge %start : none
    %returnVal = return %mergeStart : none
    end %returnVal : none
  }

  handshake.func @downgradeIndexLessControlMerge(%arg0: i32, %arg1: i32, %start: none) -> i32 attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
    %cmergeRes, %cmergeIdx = control_merge %arg0, %arg1 : i32, index
    %returnVal = return %cmergeRes : i32
    end %returnVal : i32
  }

  handshake.func @isMyArgZero(%arg0: i32, %start: none) -> i1 attributes {argNames = ["arg0", "start"], resNames = ["out0"]} {
  // ^bb0:
    %mergeArg0 = merge %arg0 : i32
    %mergeStart = merge %start : none
    %zero = constant %mergeStart {value = 0 : i32} : i32
    %isArg0EqZero = arith.cmpi eq, %mergeArg0, %zero : i32
    %startIsZero, %startIsNotZero = cond_br %isArg0EqZero, %mergeStart : none
  // ^bb1:
    %cmergeRes1, %cmergeIdx1 = control_merge %startIsZero : none, index
    %arg0isZero = constant %cmergeRes1 {value = 1 : i1} : i1
    %res1 = br %arg0isZero : i1
  // ^bb2:
    %cmergeRes2, %cmergeIdx2 = control_merge %startIsNotZero : none, index
    %arg0isNotZero = constant %cmergeRes2 {value = 0 : i1} : i1
    %res2 = br %arg0isNotZero : i1
  // ^bb3:
    %cmergeRes3, %cmergeIdx3 = control_merge %cmergeRes1, %cmergeRes2 : none, index
    %res = mux %cmergeIdx3 [%res1, %res2] : index, i1
    %returnVal = return %res : i1
    end %returnVal : i1
  }
}

