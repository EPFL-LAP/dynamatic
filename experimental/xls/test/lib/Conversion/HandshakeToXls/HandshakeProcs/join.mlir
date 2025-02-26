// RUN: dynamatic-opt --lower-handshake-to-xls %s 2>&1 | FileCheck %s

// CHECK-LABEL:   xls.sproc @join_x3(
// CHECK-SAME:                         %[[IN0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i0, in>,
// CHECK-SAME:                         %[[IN1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i0, in>,
// CHECK-SAME:                         %[[IN2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i0, in>,
// CHECK-SAME:                         %[[OUT:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i0, out>) {

// CHECK:           next (%[[IN0]]: !xls.schan<i0, in>, %[[IN1]]: !xls.schan<i0, in>, %[[IN2]]: !xls.schan<i0, in>, %[[OUT]]: !xls.schan<i0, out>) zeroinitializer {
// CHECK:             %[[TOK:.*]] = xls.after_all  : !xls.token
// CHECK:             %[[TRUE:.*]] = "xls.constant_scalar"() <{value = true}> : () -> i1

// CHECK:             %[[TOK0:.*]], %[[IN_DATA0:.*]], %[[DID_RX0:.*]] = xls.snonblocking_receive %[[TOK]], %[[IN0]], %[[TRUE]] : (!xls.token, !xls.schan<i0, in>, i1) -> (!xls.token, i0, i1)
// CHECK:             %{{.*}} = xls.ssend %[[TOK0]], %[[IN_DATA0]], %[[OUT]], %[[DID_RX0]] : (!xls.token, i0, !xls.schan<i0, out>, i1) -> !xls.token
// CHECK:             %[[DID_NOT_RX0:.*]] = xls.not %[[DID_RX0]] : i1
// CHECK:             %[[ALL_PREV_DID_NOT_RX0:.*]] = xls.and %[[DID_NOT_RX0]], %[[TRUE]] : i1

// CHECK:             %[[TOK1:.*]], %[[IN_DATA1:.*]], %[[DID_RX1:.*]] = xls.snonblocking_receive %[[TOK0]], %[[IN1]], %[[ALL_PREV_DID_NOT_RX0]] : (!xls.token, !xls.schan<i0, in>, i1) -> (!xls.token, i0, i1)
// CHECK:             %{{.*}} = xls.ssend %[[TOK1]], %[[IN_DATA1]], %[[OUT]], %[[DID_RX1]] : (!xls.token, i0, !xls.schan<i0, out>, i1) -> !xls.token
// CHECK:             %[[DID_NOT_RX1:.*]] = xls.not %[[DID_RX1]] : i1
// CHECK:             %[[ALL_PREV_DID_NOT_RX1:.*]] = xls.and %[[DID_NOT_RX1]], %[[ALL_PREV_DID_NOT_RX0]] : i1

// CHECK:             %[[TOK2:.*]], %[[IN_DATA2:.*]], %[[DID_RX2:.*]] = xls.snonblocking_receive %[[TOK1]], %[[IN2]], %[[ALL_PREV_DID_NOT_RX1]] : (!xls.token, !xls.schan<i0, in>, i1) -> (!xls.token, i0, i1)
// CHECK:             %{{.*}} = xls.ssend %[[TOK2]], %[[IN_DATA2]], %[[OUT]], %[[DID_RX2]] : (!xls.token, i0, !xls.schan<i0, out>, i1) -> !xls.token

// CHECK:             xls.yield
// CHECK:           }
// CHECK:         }


module {
  handshake.func @proc_join(%arg0: !handshake.control<>, %arg1: !handshake.control<>, %arg2: !handshake.control<>) -> (!handshake.control<>) {
    %0 = join %arg0, %arg1, %arg2 : <>
    end %0 : <>
  }
}
