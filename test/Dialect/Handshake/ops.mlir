// RUN: dynamatic-opt %s --split-input-file --verify-diagnostics

handshake.func @unbundleControl(%ctrl: <>) -> (<>) {
  %valid = unbundle %ctrl [%ready] : <> to _
  %ctrlAgain, %ready = bundle %valid : _ to <>
  end %ctrlAgain : !handshake.control<>
}

// -----

handshake.func @unbundleChannelSimple(%channel: <i32>) -> (<i32>) {
  %ctrl, %data = unbundle %channel : <i32> to _
  %valid = unbundle %ctrl [%ready] : <> to _
  %ctrlAgain, %ready = bundle %valid : _ to <>
  %channelAgain = bundle %ctrlAgain, %data : _ to <i32>
  end %channelAgain : !handshake.channel<i32>
}

// -----

handshake.func @unbundleChannelComplex(%channel: <i32, [e1: i1 (U), e2: f16, e3: i32 (U)]>) -> (<i32, [e1: i1 (U), e2: f16, e3: i32 (U)]>) {
  %ctrl, %data, %e2 = unbundle %channel [%e1, %e3] : <i32, [e1: i1 (U), e2: f16, e3: i32 (U)]> to _
  %channelAgain, %e1, %e3 = bundle %ctrl, %data, %e2 : _ to <i32, [e1: i1 (U), e2: f16, e3: i32 (U)]>
  end %channelAgain : !handshake.channel<i32, [e1: i1 (U), e2: f16, e3: i32 (U)]>
}

// -----

handshake.func @sourceAndConstantWithExtraSignal(%ctrl: <>) -> (<>) {
  %ctrlWithExtraSignal = source : <[test: i2]>
  %valueWithExtraSignal = constant %ctrlWithExtraSignal {value = 100 : i32} : <[test: i2]>, <i32, [test: i2]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @loadWithExtraSignal(%ctrl: <>, %addr: <i32, [test: i2]>, %ldData: <i32>) -> (<>) {
  %ldAddrToMem, %ldDataToSucc = load [%addr] %ldData : <i32, [test: i2]>, <i32>, <i32>, <i32, [test: i2]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @muxWithExtraSignal(%ctrl: <>, %sel: <i2, [spec: i1, tag: i2]>, %data1: <i32, [spec: i1, tag: i2]>, %data2: <i32, [spec: i1, tag: i2]>, %data3: <i32, [spec: i1, tag: i2]>, %data4: <i32, [spec: i1, tag: i2]>) -> (<>) {
  %data = mux %sel [%data1, %data2, %data3, %data4] : <i2, [spec: i1, tag: i2]>, [<i32, [spec: i1, tag: i2]>, <i32, [spec: i1, tag: i2]>, <i32, [spec: i1, tag: i2]>, <i32, [spec: i1, tag: i2]>] to <i32, [spec: i1, tag: i2]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @cmergeWithExtraSignal(%ctrl: <>, %data1: <[spec: i1]>, %data2: <[spec: i1]>, %data3: <[spec: i1]>, %data4: <[spec: i1]>) -> (<>) {
  %data, %idx = control_merge [%data1, %data2, %data3, %data4] : [<[spec: i1]>, <[spec: i1]>, <[spec: i1]>, <[spec: i1]>] to <[spec: i1]>, <i2, [spec: i1]>
  end %ctrl : !handshake.control<>
}
