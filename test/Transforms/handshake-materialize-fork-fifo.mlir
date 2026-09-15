// RUN: dynamatic-opt --handshake-materialize="forkFifoSize=4" --remove-operation-names %s | FileCheck %s

// CHECK-LABEL: handshake.func @bufferForkOutputs(
// CHECK: %[[FORK:.*]]:2 = fork [2] %{{.*}} : <i32>
// CHECK: %[[BUF0:.*]] = buffer %[[FORK]]#0, bufferType = FIFO_BREAK_NONE, numSlots = 4
// CHECK: %[[BUF1:.*]] = buffer %[[FORK]]#1, bufferType = FIFO_BREAK_NONE, numSlots = 4
// CHECK: %{{.*}} = addi %[[BUF0]], %[[BUF1]] : <i32>
handshake.func @bufferForkOutputs(%arg0: !handshake.channel<i32>, %start: !handshake.control<>) -> !handshake.channel<i32> {
  %sum = addi %arg0, %arg0 : <i32>
  end %sum : <i32>
}
