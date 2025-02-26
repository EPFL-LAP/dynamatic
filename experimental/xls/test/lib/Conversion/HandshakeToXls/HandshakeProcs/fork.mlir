// RUN: dynamatic-opt --lower-handshake-to-xls %s 2>&1 | FileCheck %s

// CHECK-LABEL:   xls.sproc @fork_42_x3(
// CHECK-SAME:                          %[[V_IN:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i42, in>,
// CHECK-SAME:                          %[[OUT0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i42, out>,
// CHECK-SAME:                          %[[OUT1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i42, out>,
// CHECK-SAME:                          %[[OUT2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i42, out>) {

// CHECK:           next (%[[V_IN]]: !xls.schan<i42, in>, %[[OUT0]]: !xls.schan<i42, out>, %[[OUT1]]: !xls.schan<i42, out>, %[[OUT2]]: !xls.schan<i42, out>) zeroinitializer {
// CHECK:             %[[TOK0:.*]] = xls.after_all  : !xls.token
// CHECK:             %[[TOK1:.*]], %[[DATA:.*]] = xls.sblocking_receive %[[TOK0]], %[[V_IN]] : (!xls.token, !xls.schan<i42, in>) -> (!xls.token, i42)
// CHECK:             %{{.*}} = xls.ssend %[[TOK1]], %[[DATA]], %[[OUT0]] : (!xls.token, i42, !xls.schan<i42, out>) -> !xls.token
// CHECK:             %{{.*}} = xls.ssend %[[TOK1]], %[[DATA]], %[[OUT1]] : (!xls.token, i42, !xls.schan<i42, out>) -> !xls.token
// CHECK:             %{{.*}} = xls.ssend %[[TOK1]], %[[DATA]], %[[OUT2]] : (!xls.token, i42, !xls.schan<i42, out>) -> !xls.token
// CHECK:             xls.yield
// CHECK:           }
// CHECK:         }

module {
  handshake.func @proc_fork(%arg0: !handshake.channel<i42>) -> (!handshake.channel<i42>, !handshake.channel<i42>, !handshake.channel<i42>) {
    %0:3 = fork [3] %arg0 : <i42>
    end %0#0, %0#1, %0#2 : <i42>, <i42>, <i42>
  }
}
