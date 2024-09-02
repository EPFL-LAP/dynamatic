// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --handshake-prepare-for-legacy --remove-operation-names %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @convertUncondBranchesDoNothing(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                                   %[[VAL_2:.*]]: none, ...) -> i32 attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = cond_br %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_5:.*]] = return %[[VAL_3]] : i32
// CHECK:           end %[[VAL_5]] : i32
// CHECK:         }
handshake.func @convertUncondBranchesDoNothing(%arg0: i1, %arg1: i32, %start: none) -> i32 {
  %trueResult, %falseResult = cond_br %arg0, %arg1 : i32
  %ret = return %trueResult : i32
  end %ret : i32
}

// -----

// CHECK-LABEL:   handshake.func @convertUncondBranchesStart(
// CHECK-SAME:                                               %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                               %[[VAL_2:.*]]: none, ...) -> i32 attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_2]] {handshake.bb = 0 : ui32, value = true} : i1
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = cond_br %[[VAL_3]], %[[VAL_0]] {handshake.bb = 0 : ui32} : i32
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_3]], %[[VAL_1]] {handshake.bb = 0 : ui32} : i32
// CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_4]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_9:.*]] = return %[[VAL_8]] : i32
// CHECK:           end %[[VAL_9]] : i32
// CHECK:         }
handshake.func @convertUncondBranchesStart(%arg0: i32, %arg1: i32, %start: none) -> i32 {
  %0 = br %arg0 {handshake.bb = 0 : ui32} : i32
  %1 = br %arg1 {handshake.bb = 0 : ui32}: i32
  %2 = arith.addi %0, %1: i32
  %ret = return %2 : i32
  end %ret : i32
}

// -----

// CHECK-LABEL:   handshake.func @convertUncondBranchesCMerge(
// CHECK-SAME:                                                %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                                %[[VAL_1:.*]]: none, ...) -> i32 attributes {argNames = ["arg0", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = control_merge %[[VAL_1]], %[[VAL_1]] {handshake.bb = 1 : ui32} : none, i32
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_2]] {handshake.bb = 1 : ui32, value = true} : i1
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = cond_br %[[VAL_4]], %[[VAL_0]] {handshake.bb = 1 : ui32} : i32
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = cond_br %[[VAL_4]], %[[VAL_3]] {handshake.bb = 1 : ui32} : i32
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_5]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_10:.*]] = return %[[VAL_9]] : i32
// CHECK:           end %[[VAL_10]] : i32
// CHECK:         }
handshake.func @convertUncondBranchesCMerge(%arg0: i32, %start: none) -> i32 {
  %ctrl, %idx = control_merge %start, %start {handshake.bb = 1 : ui32} : none, i32
  %0 = br %arg0 {handshake.bb = 1 : ui32} : i32
  %1 = br %idx {handshake.bb = 1 : ui32} : i32
  %2 = arith.addi %0, %1: i32
  %ret = return %2 : i32
  end %ret : i32
}


// -----

// CHECK-LABEL:   handshake.func @convertUncondBranchesOutOfBlocks(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                                     %[[VAL_2:.*]]: none, ...) -> i32 attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = source
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_3]] {value = true} : i1
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = cond_br %[[VAL_4]], %[[VAL_0]] {handshake.bb = 1 : ui32} : i32
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = cond_br %[[VAL_4]], %[[VAL_1]] {handshake.bb = 2 : ui32} : i32
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_5]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_10:.*]] = return %[[VAL_9]] : i32
// CHECK:           end %[[VAL_10]] : i32
// CHECK:         }
handshake.func @convertUncondBranchesOutOfBlocks(%arg0: i32, %arg1: i32, %start: none) -> i32 {
  %0 = br %arg0 {handshake.bb = 1 : ui32} : i32
  %1 = br %arg1 {handshake.bb = 2 : ui32} : i32
  %2 = arith.addi %0, %1: i32
  %ret = return %2 : i32
  end %ret : i32
}

// -----

// CHECK-LABEL:   handshake.func @convertUncondBranchesAllCases(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                                  %[[VAL_2:.*]]: none, ...) -> i32 attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = source
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_3]] {value = true} : i1
// CHECK:           %[[VAL_5:.*]] = constant %[[VAL_2]] {handshake.bb = 0 : ui32, value = true} : i1
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_5]], %[[VAL_0]] {handshake.bb = 0 : ui32} : i32
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = control_merge %[[VAL_2]], %[[VAL_2]] {handshake.bb = 1 : ui32} : none, i32
// CHECK:           %[[VAL_10:.*]] = constant %[[VAL_8]] {handshake.bb = 1 : ui32, value = true} : i1
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = cond_br %[[VAL_10]], %[[VAL_9]] {handshake.bb = 1 : ui32} : i32
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = cond_br %[[VAL_4]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_6]], %[[VAL_11]] : i32
// CHECK:           %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_13]] : i32
// CHECK:           %[[VAL_17:.*]] = return %[[VAL_16]] : i32
// CHECK:           end %[[VAL_17]] : i32
// CHECK:         }

handshake.func @convertUncondBranchesAllCases(%arg0: i32, %arg1: i32, %start: none) -> i32 {
  %0 = br %arg0 {handshake.bb = 0 : ui32} : i32
  %ctrl, %idx = control_merge %start, %start {handshake.bb = 1 : ui32} : none, i32
  %1 = br %idx {handshake.bb = 1 : ui32} : i32
  %2 = br %arg1 : i32
  %3 = arith.addi %0, %1: i32
  %4 = arith.addi %3, %2: i32
  %ret = return %4 : i32
  end %ret : i32
}


// -----

// CHECK-LABEL:   handshake.func @simplifyCMergesDoNothing(
// CHECK-SAME:                                             %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                             %[[VAL_1:.*]]: none, ...) -> index attributes {argNames = ["arg0", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = cond_br %[[VAL_0]], %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = control_merge %[[VAL_2]], %[[VAL_3]] : none, index
// CHECK:           %[[VAL_6:.*]] = return %[[VAL_5]] : index
// CHECK:           end %[[VAL_6]] : index
// CHECK:         }
handshake.func @simplifyCMergesDoNothing(%arg0: i1, %start: none) -> index {
  %trueResult, %falseResult = cond_br %arg0, %start : none
  %ctrl, %idx = control_merge %trueResult, %falseResult : none, index
  %ret = return %idx : index
  end %ret : index
}

// -----

// CHECK-LABEL:   handshake.func @simplifyCMergesIndexUnused(
// CHECK-SAME:                                               %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                               %[[VAL_1:.*]]: none, ...) -> none attributes {argNames = ["arg0", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = cond_br %[[VAL_0]], %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_2]], %[[VAL_3]] : none
// CHECK:           %[[VAL_5:.*]] = return %[[VAL_4]] : none
// CHECK:           end %[[VAL_5]] : none
// CHECK:         }
handshake.func @simplifyCMergesIndexUnused(%arg0: i1, %start: none) -> none {
  %trueResult, %falseResult = cond_br %arg0, %start : none
  %ctrl, %idx = control_merge %trueResult, %falseResult : none, index
  %ret = return %ctrl : none
  end %ret : none
}

// -----

// CHECK-LABEL:   handshake.func @simplifyCMergesSingleInputAndIndexUsed(
// CHECK-SAME:                                                           %[[VAL_0:.*]]: none, ...) -> (none, index) attributes {argNames = ["start"], resNames = ["out0", "out1"]} {
// CHECK:           %[[VAL_1:.*]] = merge %[[VAL_0]] : none
// CHECK:           %[[VAL_2:.*]] = constant %[[VAL_1]] {value = 0 : index} : index
// CHECK:           %[[VAL_3:.*]]:2 = return %[[VAL_1]], %[[VAL_2]] : none, index
// CHECK:           end %[[VAL_3]]#0, %[[VAL_3]]#1 : none, index
// CHECK:         }
handshake.func @simplifyCMergesSingleInputAndIndexUsed(%start: none) -> (none, index) {
  %ctrl, %idx = control_merge %start : none, index
  %ctrlRet, %idxRet = return %ctrl, %idx : none, index
  end %ctrlRet, %idxRet : none, index
}


// -----

// CHECK-LABEL:   handshake.func @simplifyCMergesSingleInput(
// CHECK-SAME:                                               %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = merge %[[VAL_0]] : none
// CHECK:           %[[VAL_2:.*]] = return %[[VAL_1]] : none
// CHECK:           end %[[VAL_2]] : none
// CHECK:         }
handshake.func @simplifyCMergesSingleInput(%start: none) -> none {
  %ctrl, %idx = control_merge %start : none, index
  %ret = return %ctrl : none
  end %ret : none
}

// -----

// CHECK-LABEL:   handshake.func @eraseEntryBlockMergesDoNothing(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = merge %[[VAL_0]], %[[VAL_0]] {handshake.bb = 0 : ui32} : none
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_1]] : none
// CHECK:           %[[VAL_3:.*]] = return %[[VAL_2]] : none
// CHECK:           end %[[VAL_3]] : none
// CHECK:         }
handshake.func @eraseEntryBlockMergesDoNothing(%start: none) -> none {
  %0 = merge %start, %start {handshake.bb = 0 : ui32} : none
  %1 = merge %0 : none
  %ret = return %1 : none
  end %ret : none
}

// -----

// CHECK-LABEL:   handshake.func @eraseEntryBlockMerges(
// CHECK-SAME:                                          %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_1:.*]] = return %[[VAL_0]] : none
// CHECK:           end %[[VAL_1]] : none
// CHECK:         }
handshake.func @eraseEntryBlockMerges(%start: none) -> none {
  %0 = merge %start {handshake.bb = 0 : ui32} : none
  %ret = return %0 : none
  end %ret : none
}