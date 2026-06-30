// RUN: hls-fuzzer-check-bitwidth %s 8

handshake.func @test2(%arg0: !handshake.channel<i32>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["var0", "start"], resNames = ["out0", "end"]} {
  // Allow forks of function arguments since they may be used in multiple truncations of different bitwidths.
  %1:3 = fork [3] %arg0 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i32>

  // Random example program that is done under bitwidth of 8 follows.
  %2 = trunci %1#0 {handshake.bb = 0 : ui32, handshake.name = "trunci0"} : <i32> to <i8>
  %3 = trunci %1#2 {handshake.bb = 0 : ui32, handshake.name = "trunci2"} : <i32> to <i1>
  %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi0"} : <i1> to <i8>
  %5 = ori %4, %2 {handshake.bb = 0 : ui32, handshake.name = "ori2"} : <i8>
  %6 = extui %5 {handshake.bb = 0 : ui32, handshake.name = "extui8"} : <i8> to <i32>
  end {handshake.bb = 0 : ui32, handshake.name = "end0"} %6, %arg2 : <i32>, <>
}
