// RUN: dynamatic-opt --lower-handshake-to-xls %s 2>&1 | FileCheck %s

// CHECK-LABEL:   xls.sproc @constant_val0xff_66(
// CHECK-SAME:                                   %[[IN:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i0, in>,
// CHECK-SAME:                                   %[[OUT:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i66, out>) {

// CHECK:           next (%[[IN]]: !xls.schan<i0, in>, %[[OUT]]: !xls.schan<i66, out>) zeroinitializer {
// CHECK:             %[[TOK0:.*]] = xls.after_all  : !xls.token
// CHECK:             %[[TOK1:.*]], %[[CTRL_IN:.*]] = xls.sblocking_receive %[[TOK0]], %[[IN]] : (!xls.token, !xls.schan<i0, in>) -> (!xls.token, i0)
// CHECK:             %[[DATA:.*]] = "xls.constant_scalar"() <{value = 255 : i66}> : () -> i66
// CHECK:             %{{.*}} = xls.ssend %[[TOK1]], %[[DATA]], %[[OUT]] : (!xls.token, i66, !xls.schan<i66, out>) -> !xls.token
// CHECK:             xls.yield
// CHECK:           }
// CHECK:         }

module {
  handshake.func @proc_constant(%in: !handshake.control<>) -> (!handshake.channel<i66>) {
    %1 = constant %in {value = 255 : i66} : <>, <i66>
    end %1: <i66>
  }
}
