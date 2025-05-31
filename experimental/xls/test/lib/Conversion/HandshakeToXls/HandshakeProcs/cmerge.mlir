// RUN: dynamatic-opt --lower-handshake-to-xls %s 2>&1 | FileCheck %s

// CHECK-LABEL:   xls.sproc @cmerge_33_idx2_x3(
// CHECK-SAME:                                 %[[IN0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i33, in>,
// CHECK-SAME:                                 %[[IN1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i33, in>,
// CHECK-SAME:                                 %[[IN2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i33, in>,
// CHECK-SAME:                                 %[[OUT:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i33, out>,
// CHECK-SAME:                                 %[[OUT_IDX:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i2, out>) {

// CHECK:           next (%[[IN0]]: !xls.schan<i33, in>, %[[IN1]]: !xls.schan<i33, in>, %[[IN2]]: !xls.schan<i33, in>, %[[OUT]]: !xls.schan<i33, out>, %[[OUT_IDX]]: !xls.schan<i2, out>) zeroinitializer {
// CHECK:             %[[TOK:.*]] = xls.after_all  : !xls.token
// CHECK:             %[[TRUE:.*]] = "xls.constant_scalar"() <{value = true}> : () -> i1

// CHECK:             %[[TOK0:.*]], %[[DATA0:.*]], %[[DID_RX0:.*]] = xls.snonblocking_receive %[[TOK]], %[[IN0]], %[[TRUE]] : (!xls.token, !xls.schan<i33, in>, i1) -> (!xls.token, i33, i1)
// CHECK:             %[[IDX0:.*]] = "xls.constant_scalar"() <{value = 0 : i2}> : () -> i2
// CHECK:             %{{.*}} = xls.ssend %[[TOK0]], %[[DATA0]], %[[OUT]], %[[DID_RX0]] : (!xls.token, i33, !xls.schan<i33, out>, i1) -> !xls.token
// CHECK:             %{{.*}} = xls.ssend %[[TOK0]], %[[IDX0]], %[[OUT_IDX]], %[[DID_RX0]] : (!xls.token, i2, !xls.schan<i2, out>, i1) -> !xls.token

// CHECK:             %[[DID_NOT_RX0:.*]] = xls.not %[[DID_RX0]] : i1
// CHECK:             %[[NO_PREV_DID_RX0:.*]] = xls.and %[[DID_NOT_RX0]], %[[TRUE]] : i1
// CHECK:             %[[TOK1:.*]], %[[DATA1:.*]], %[[DID_RX1:.*]] = xls.snonblocking_receive %[[TOK]], %[[IN0]], %[[NO_PREV_DID_RX0]] : (!xls.token, !xls.schan<i33, in>, i1) -> (!xls.token, i33, i1)
// CHECK:             %[[IDX1:.*]] = "xls.constant_scalar"() <{value = 0 : i2}> : () -> i2
// CHECK:             %{{.*}} = xls.ssend %[[TOK1]], %[[DATA1]], %[[OUT]], %[[DID_RX1]] : (!xls.token, i33, !xls.schan<i33, out>, i1) -> !xls.token
// CHECK:             %{{.*}} = xls.ssend %[[TOK1]], %[[IDX1]], %[[OUT_IDX]], %[[DID_RX1]] : (!xls.token, i2, !xls.schan<i2, out>, i1) -> !xls.token

// CHECK:             %[[DID_NOT_RX1:.*]] = xls.not %[[DID_RX1]] : i1
// CHECK:             %[[NO_PREV_DID_RX1:.*]] = xls.and %[[DID_NOT_RX1]], %[[NO_PREV_DID_RX0]] : i1
// CHECK:             %[[TOK2:.*]], %[[DATA2:.*]], %[[DID_RX2:.*]] = xls.snonblocking_receive %[[TOK]], %[[IN1]], %[[NO_PREV_DID_RX1]] : (!xls.token, !xls.schan<i33, in>, i1) -> (!xls.token, i33, i1)
// CHECK:             %[[IDX2:.*]] = "xls.constant_scalar"() <{value = 1 : i2}> : () -> i2
// CHECK:             %{{.*}} = xls.ssend %[[TOK2]], %[[DATA2]], %[[OUT]], %[[DID_RX2]] : (!xls.token, i33, !xls.schan<i33, out>, i1) -> !xls.token
// CHECK:             %{{.*}} = xls.ssend %[[TOK2]], %[[IDX2]], %[[OUT_IDX]], %[[DID_RX2]] : (!xls.token, i2, !xls.schan<i2, out>, i1) -> !xls.token

// CHECK:             %[[DID_NOT_RX2:.*]] = xls.not %[[DID_RX2]] : i1
// CHECK:             %[[NO_PREV_DID_RX2:.*]] = xls.and %[[DID_NOT_RX2]], %[[NO_PREV_DID_RX1]] : i1
// CHECK:             %[[TOK3:.*]], %[[DATA3:.*]], %[[DID_RX3:.*]] = xls.snonblocking_receive %[[TOK]], %[[IN2]], %[[NO_PREV_DID_RX2]] : (!xls.token, !xls.schan<i33, in>, i1) -> (!xls.token, i33, i1)
// CHECK:             %[[IDX3:.*]] = "xls.constant_scalar"() <{value = -2 : i2}> : () -> i2
// CHECK:             %{{.*}} = xls.ssend %[[TOK3]], %[[DATA3]], %[[OUT]], %[[DID_RX3]] : (!xls.token, i33, !xls.schan<i33, out>, i1) -> !xls.token
// CHECK:             %{{.*}} = xls.ssend %[[TOK3]], %[[IDX3]], %[[OUT_IDX]], %[[DID_RX3]] : (!xls.token, i2, !xls.schan<i2, out>, i1) -> !xls.token

// CHECK:             xls.yield
// CHECK:           }
// CHECK:         }

module {
  handshake.func @proc_cmerge(%a: !handshake.channel<i33>, %b: !handshake.channel<i33>, %c: !handshake.channel<i33>) -> (!handshake.channel<i33>, !handshake.channel<i2>) {
    %0, %1 = control_merge [%a, %b, %c] : [<i33>, <i33>, <i33>] to <i33>, <i2>
    end %0, %1 : <i33>, <i2> }
}
