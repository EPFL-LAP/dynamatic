// RUN: dynamatic-opt %s --split-input-file --verify-diagnostics --handshake-rigidification

handshake.func @foo(%arg0 : !handshake.channel<i32>) -> (!handshake.channel<i32>) {
  %1 = handshake.buffer %arg0 : <i32>
  %end = handshake.buffer %1 : <i32>
  end %end : !handshake.channel<i32>
}