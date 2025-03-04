// RUN: dynamatic-opt --lower-handshake-to-xls %s 2>&1 | FileCheck %s

// CHECK-LABEL:   xls.sproc @cbranch_11(
// CHECK-SAME:                          %[[IN_COND:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i1, in>,
// CHECK-SAME:                          %[[IN_DATA:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i11, in>,
// CHECK-SAME:                          %[[OUT_TRUE:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i11, out>,
// CHECK-SAME:                          %[[OUT_FALSE:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i11, out>) {

// CHECK:           next (%[[IN_COND]]: !xls.schan<i1, in>, %[[IN_DATA]]: !xls.schan<i11, in>, %[[OUT_TRUE]]: !xls.schan<i11, out>, %[[OUT_FALSE]]: !xls.schan<i11, out>) zeroinitializer {
// CHECK:             %[[TOK:.*]] = xls.after_all  : !xls.token
// CHECK:             %[[TOK_COND:.*]], %[[COND:.*]] = xls.sblocking_receive %[[TOK]], %[[IN_COND]] : (!xls.token, !xls.schan<i1, in>) -> (!xls.token, i1)
// CHECK:             %[[TOK_DATA:.*]], %[[DATA:.*]] = xls.sblocking_receive %[[TOK]], %[[IN_DATA]] : (!xls.token, !xls.schan<i11, in>) -> (!xls.token, i11)
// CHECK:             %[[TOK_FINAL:.*]] = xls.after_all %[[TOK_COND]], %[[TOK_DATA]] : !xls.token
// CHECK:             %[[NOT_COND:.*]] = xls.not %[[COND]] : i1
// CHECK:             %{{.*}} = xls.ssend %[[TOK_FINAL]], %[[DATA]], %[[OUT_TRUE]], %[[COND]] : (!xls.token, i11, !xls.schan<i11, out>, i1) -> !xls.token
// CHECK:             %{{.*}} = xls.ssend %[[TOK_FINAL]], %[[DATA]], %[[OUT_FALSE]], %[[NOT_COND]] : (!xls.token, i11, !xls.schan<i11, out>, i1) -> !xls.token
// CHECK:             xls.yield
// CHECK:           }
// CHECK:         }

module {
  handshake.func @proc_cbranch(%cond: !handshake.channel<i1>, %in: !handshake.channel<i11>) -> (!handshake.channel<i11>, !handshake.channel<i11>) {
    %true, %false = cond_br %cond, %in : !handshake.channel<i1>, !handshake.channel<i11>
    end %true, %false : <i11>, <i11> }
}
