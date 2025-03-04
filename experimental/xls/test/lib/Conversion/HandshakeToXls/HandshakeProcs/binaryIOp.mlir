// RUN: dynamatic-opt --lower-handshake-to-xls %s 2>&1 | FileCheck %s

// CHECK-LABEL:   xls.sproc @xori_32(
// CHECK-SAME:                       %[[LHS:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i32, in>,
// CHECK-SAME:                       %[[RHS:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i32, in>,
// CHECK-SAME:                       %[[OUT:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i32, out>) {

// CHECK:           next (%[[LHS]]: !xls.schan<i32, in>, %[[RHS]]: !xls.schan<i32, in>, %[[OUT]]: !xls.schan<i32, out>) zeroinitializer {
// CHECK:             %[[TOK0:.*]] = xls.after_all  : !xls.token
// CHECK:             %[[TOK1:.*]], %[[DATA_LHS:.*]] = xls.sblocking_receive %[[TOK0]], %[[LHS]] : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:             %[[TOK2:.*]], %[[DATA_RHS:.*]] = xls.sblocking_receive %[[TOK0]], %[[RHS]] : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:             %[[DATA_RESULT:.*]] = xls.xor %[[DATA_LHS]], %[[DATA_RHS]] : i32
// CHECK:             %[[TOK3:.*]] = xls.after_all %[[TOK1]], %[[TOK2]] : !xls.token
// CHECK:             %{{.*}} = xls.ssend %[[TOK3]], %[[DATA_RESULT]], %[[OUT]] : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
// CHECK:             xls.yield
// CHECK:           }
// CHECK:         }

module {
  handshake.func @proc_binIop(%lhs: !handshake.channel<i32>, %rhs: !handshake.channel<i32>) -> (!handshake.channel<i32>) {
    %0 = handshake.xori %lhs, %rhs: <i32>
    end %0: <i32>
  }
}
