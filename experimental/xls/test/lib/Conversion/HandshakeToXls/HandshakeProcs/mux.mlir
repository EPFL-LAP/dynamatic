// RUN: dynamatic-opt --lower-handshake-to-xls %s 2>&1 | FileCheck %s

// CHECK-LABEL:   xls.sproc @mux_5_sel2_x3(
// CHECK-SAME:                             %[[IN_SEL:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i2, in>,
// CHECK-SAME:                             %[[IN0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i5, in>,
// CHECK-SAME:                             %[[IN1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i5, in>,
// CHECK-SAME:                             %[[IN2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i5, in>,
// CHECK-SAME:                             %[[OUT:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i5, out>) {

// CHECK:           next (%[[IN_SEL]]: !xls.schan<i2, in>, %[[IN0]]: !xls.schan<i5, in>, %[[IN1]]: !xls.schan<i5, in>, %[[IN2]]: !xls.schan<i5, in>, %[[OUT]]: !xls.schan<i5, out>) zeroinitializer {
// CHECK:             %[[TOK:.*]] = xls.after_all  : !xls.token

// CHECK:             %{{.*}}, %[[DATA_SEL:.*]] = xls.sblocking_receive %[[TOK]], %[[IN_SEL]] : (!xls.token, !xls.schan<i2, in>) -> (!xls.token, i2)

// CHECK:             %[[IDX0:.*]] = "xls.constant_scalar"() <{value = 0 : ui2}> : () -> i2
// CHECK:             %[[IDX0_MATCH:.*]] = xls.eq %[[DATA_SEL]], %[[IDX0]] : (i2, i2) -> i1
// CHECK:             %[[TOK0:.*]], %[[DATA_IN0:.*]] = xls.sblocking_receive %[[TOK]], %[[IN0]], %[[IDX0_MATCH]] : (!xls.token, !xls.schan<i5, in>, i1) -> (!xls.token, i5)
// CHECK:             %{{.*}} = xls.ssend %[[TOK0]], %[[DATA_IN0]], %[[OUT]], %[[IDX0_MATCH]] : (!xls.token, i5, !xls.schan<i5, out>, i1) -> !xls.token

// CHECK:             %[[IDX1:.*]] = "xls.constant_scalar"() <{value = 1 : ui2}> : () -> i2
// CHECK:             %[[IDX1_MATCH:.*]] = xls.eq %[[DATA_SEL]], %[[IDX1]] : (i2, i2) -> i1
// CHECK:             %[[TOK1:.*]], %[[DATA_IN1:.*]] = xls.sblocking_receive %[[TOK]], %[[IN1]], %[[IDX1_MATCH]] : (!xls.token, !xls.schan<i5, in>, i1) -> (!xls.token, i5)
// CHECK:             %{{.*}} = xls.ssend %[[TOK1]], %[[DATA_IN1]], %[[OUT]], %[[IDX1_MATCH]] : (!xls.token, i5, !xls.schan<i5, out>, i1) -> !xls.token

// CHECK:             %[[IDX2:.*]] = "xls.constant_scalar"() <{value = 2 : ui2}> : () -> i2
// CHECK:             %[[IDX2_MATCH:.*]] = xls.eq %[[DATA_SEL]], %[[IDX2]] : (i2, i2) -> i1
// CHECK:             %[[TOK2:.*]], %[[DATA_IN2:.*]] = xls.sblocking_receive %[[TOK]], %[[IN2]], %[[IDX2_MATCH]] : (!xls.token, !xls.schan<i5, in>, i1) -> (!xls.token, i5)
// CHECK:             %{{.*}} = xls.ssend %[[TOK2]], %[[DATA_IN2]], %[[OUT]], %[[IDX2_MATCH]] : (!xls.token, i5, !xls.schan<i5, out>, i1) -> !xls.token

// CHECK:             xls.yield
// CHECK:           }
// CHECK:         }

module {
  handshake.func @proc_mux(%sel: !handshake.channel<i2>, %inp0: !handshake.channel<i5>, %inp1: !handshake.channel<i5>, %inp2: !handshake.channel<i5> ) -> (!handshake.channel<i5>) {
    %res = mux %sel [%inp0, %inp1, %inp2] : <i2>, [<i5>, <i5>, <i5>] to <i5>
    end %res : <i5>
  }
}
