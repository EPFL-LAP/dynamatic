// RUN: dynamatic-opt --lower-handshake-to-xls %s 2>&1 | FileCheck %s

// CHECK-LABEL:   xls.sproc @select_7(
// CHECK-SAME:                        %[[SEL:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i1, in>,
// CHECK-SAME:                        %[[INP_TRUE:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i7, in>,
// CHECK-SAME:                        %[[INP_FALSE:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i7, in>,
// CHECK-SAME:                        %[[INP_RESULT:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !xls.schan<i7, out>) {

// CHECK:           next (%[[SEL]]: !xls.schan<i1, in>, %[[INP_TRUE]]: !xls.schan<i7, in>, %[[INP_FALSE]]: !xls.schan<i7, in>, %[[INP_RESULT]]: !xls.schan<i7, out>) zeroinitializer {
// CHECK:             %[[TOK:.*]] = xls.after_all  : !xls.token
// CHECK:             %[[TOK0:.*]], %[[DATA_SEL:.*]] = xls.sblocking_receive %[[TOK]], %[[SEL]] : (!xls.token, !xls.schan<i1, in>) -> (!xls.token, i1)
// CHECK:             %[[TOK1:.*]], %[[DATA_TRUE:.*]] = xls.sblocking_receive %[[TOK]], %[[INP_TRUE]] : (!xls.token, !xls.schan<i7, in>) -> (!xls.token, i7)
// CHECK:             %[[TOK2:.*]], %[[DATA_FALSE:.*]] = xls.sblocking_receive %[[TOK]], %[[INP_FALSE]] : (!xls.token, !xls.schan<i7, in>) -> (!xls.token, i7)
// CHECK:             %[[TOK_END:.*]] = xls.after_all %[[TOK0]], %[[TOK1]], %[[TOK2]] : !xls.token
// CHECK:             %[[DATA_SELECTED:.*]] = xls.sel %[[DATA_SEL]] in {{\[}}%[[DATA_FALSE]], %[[DATA_TRUE]]] : (i1, [i7, i7]) -> i7
// CHECK:             %{{.*}} = xls.ssend %[[TOK_END]], %[[DATA_SELECTED]], %[[INP_RESULT]] : (!xls.token, i7, !xls.schan<i7, out>) -> !xls.token
// CHECK:             xls.yield
// CHECK:           }
// CHECK:         }

module {
  handshake.func @proc_select(%sel: !handshake.channel<i1>, %if_true: !handshake.channel<i7>, %if_false: !handshake.channel<i7>) -> (!handshake.channel<i7>) {
    %res = select %sel [%if_true, %if_false] : <i1>, <i7>
    end %res : <i7>
  }
}
