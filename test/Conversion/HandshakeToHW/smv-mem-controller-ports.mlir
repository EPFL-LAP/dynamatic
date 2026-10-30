// RUN: dynamatic-opt --lower-handshake-to-hw %s | FileCheck %s

// The SMV memory controller is generated with the port order of
// SMV_INPUT_SYMBOLS, and export-rtl instantiates it with the indexed ports of
// the same name grouped into arrays, in the order of their first port: all
// ctrl data signals come before the ctrl valid signals, even though the ports
// of the two groups (a control port and a store each) are interleaved

// CHECK: hw.module.extern @handshake_mem_controller_0
// CHECK-SAME: SMV_INPUT_SYMBOLS = "loadData, memStart_valid, ldAddr_0, ldAddr_0_valid, ctrl_0, ctrl_1, ctrl_0_valid, ctrl_1_valid, stAddr_0, stAddr_1, stAddr_0_valid, stAddr_1_valid, stData_0, stData_1, stData_0_valid, stData_1_valid, ctrlEnd_valid, ldData_0_ready, memEnd_ready"
handshake.func @twoStoreGroups(%mem: memref<4xi32>, %memStart: !handshake.control<>, %ldAddr: !handshake.channel<i2>, %ctrl0: !handshake.channel<i32>, %stAddr0: !handshake.channel<i2>, %stData0: !handshake.channel<i32>, %ctrl1: !handshake.channel<i32>, %stAddr1: !handshake.channel<i2>, %stData1: !handshake.channel<i32>, %ctrlEnd: !handshake.control<>, %start: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["mem", "memStart", "ldAddr", "ctrl0", "stAddr0", "stData0", "ctrl1", "stAddr1", "stData1", "ctrlEnd", "start"], resNames = ["out0", "memEnd", "end"]} {
  %ldData, %memEnd = mem_controller[%mem : memref<4xi32>] %memStart (%ldAddrToMC, %ctrl0, %stAddrToMC0, %stDataToMC0, %ctrl1, %stAddrToMC1, %stDataToMC1) %ctrlEnd {connectedBlocks = [0 : i32, 1 : i32, 2 : i32], handshake.name = "mem_controller0"} : (!handshake.channel<i2>, !handshake.channel<i32>, !handshake.channel<i2>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i2>, !handshake.channel<i32>) -> !handshake.channel<i32>
  %ldAddrToMC, %ldDataOut = load[%ldAddr] %ldData {handshake.bb = 0 : ui32, handshake.name = "load0"} : <i2>, <i32>, <i2>, <i32>
  %stAddrToMC0, %stDataToMC0 = store[%stAddr0] %stData0 {handshake.bb = 1 : ui32, handshake.name = "store0"} : <i2>, <i32>, <i2>, <i32>
  %stAddrToMC1, %stDataToMC1 = store[%stAddr1] %stData1 {handshake.bb = 2 : ui32, handshake.name = "store1"} : <i2>, <i32>, <i2>, <i32>
  end {handshake.bb = 2 : ui32, handshake.name = "end0"} %ldDataOut, %memEnd, %start : <i32>, <>, <>
}
