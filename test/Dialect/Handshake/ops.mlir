// RUN: dynamatic-opt %s --split-input-file --verify-diagnostics

handshake.func @unbundleControl(%ctrl: !handshake.control) -> !handshake.control {
  %valid = unbundle %ctrl [%ready] : (!handshake.control, i1) -> i1
  %ctrlAgain, %ready = bundle %valid : (i1) -> (!handshake.control, i1)
  end %ctrlAgain : !handshake.control
}

// -----

handshake.func @unbundleChannelSimple(%channel: !handshake.channel<i32>) -> (i1, i32) {
  %ctrl, %data = unbundle %channel : (!handshake.channel<i32>) -> (!handshake.control, i32)
  %valid = unbundle %ctrl [%ready] : (!handshake.control, i1) -> (i1)
  %ctrlAgain, %ready = bundle %valid : (i1) -> (!handshake.control, i1)
  %channelAgain = bundle %ctrlAgain, %data : (!handshake.control, i32) -> (!handshake.channel<i32>)
  end %channelAgain : !handshake.channel<i32>
}

// -----

handshake.func @unbundleChannelComplex(%channel: !handshake.channel<i32, [e1: i1 (U), e2: f16, e3: index (U)]>) -> !handshake.channel<i32, [e1: i1 (U), e2: f16, e3: index (U)]> {
  %ctrl, %data, %e2 = unbundle %channel [%e1, %e3] : (!handshake.channel<i32, [e1: i1 (U), e2: f16, e3: index (U)]>, i1, index) -> (!handshake.control, i32, f16)
  %channelAgain, %e1, %e3 = bundle %ctrl, %data, %e2 : (!handshake.control, i32, f16) -> (!handshake.channel<i32, [e1: i1 (U), e2: f16, e3: index (U)]>, i1, index)
  end %channelAgain : !handshake.channel<i32, [e1: i1 (U), e2: f16, e3: index (U)]>
}
