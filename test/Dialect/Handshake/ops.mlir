// RUN: dynamatic-opt %s --split-input-file --verify-diagnostics

handshake.func @unbundleControl(%ctrl: !handshake.control<>) -> !handshake.control<> {
  %valid = unbundle %ctrl [%ready] : <> to _
  %ctrlAgain, %ready = bundle %valid : _ to <>
  end %ctrlAgain : !handshake.control<>
}

// -----

handshake.func @unbundleChannelSimple(%channel: !handshake.channel<i32>) -> !handshake.channel<i32> {
  %ctrl, %data = unbundle %channel : <i32> to _
  %valid = unbundle %ctrl [%ready] : <> to _
  %ctrlAgain, %ready = bundle %valid : _ to <>
  %channelAgain = bundle %ctrlAgain, %data : _ to <i32>
  end %channelAgain : !handshake.channel<i32>
}

// -----

handshake.func @unbundleChannelComplex(%channel: !handshake.channel<i32, [e1: i1 (U), e2: f16, e3: i32 (U)]>) -> !handshake.channel<i32, [e1: i1 (U), e2: f16, e3: i32 (U)]> {
  %ctrl, %data, %e2 = unbundle %channel [%e1, %e3] : <i32, [e1: i1 (U), e2: f16, e3: i32 (U)]> to _
  %channelAgain, %e1, %e3 = bundle %ctrl, %data, %e2 : _ to <i32, [e1: i1 (U), e2: f16, e3: i32 (U)]>
  end %channelAgain : !handshake.channel<i32, [e1: i1 (U), e2: f16, e3: i32 (U)]>
}

// -----

handshake.func @reshapeChannelIntoData(%channel: !handshake.channel<f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]>) -> !handshake.channel<f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]> {
  %reshaped = reshape [MergeData] %channel : <f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]> -> <i42, [mergedUp: i8 (U)]>
  %backToOriginal = reshape [SplitData] %reshaped : <i42, [mergedUp: i8 (U)]> -> <f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]>
  end %backToOriginal : !handshake.channel<f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]>
}

// -----

handshake.func @reshapeChannelIntoDataNoExtraDown(%channel: !handshake.channel<f32, [up1: i4 (U)]>) -> !handshake.channel<f32, [up1: i4 (U)]> {
  %reshaped = reshape [MergeData] %channel : <f32, [up1: i4 (U)]> -> <f32, [mergedUp: i4 (U)]>
  %backToOriginal = reshape [SplitData] %reshaped : <f32, [mergedUp: i4 (U)]> -> <f32, [up1: i4 (U)]>
  end %backToOriginal : !handshake.channel<f32, [up1: i4 (U)]>
}

// -----

handshake.func @reshapeChannelIntoExtra(%channel: !handshake.channel<f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]>) -> !handshake.channel<f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]> {
  %reshaped = reshape [MergeExtra] %channel : <f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]> -> <f32, [mergedDown: i10, mergedUp: i8 (U)]>
  %backToOriginal = reshape [SplitExtra] %reshaped : <f32, [mergedDown: i10, mergedUp: i8 (U)]> -> <f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]>
  end %backToOriginal : !handshake.channel<f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]>
}

// -----

handshake.func @sourceAndConstantWithExtraSignal(%ctrl : !handshake.control<>) -> !handshake.control<> {
  %ctrlWithExtraSignal = source : <[test: i2]>
  %valueWithExtraSignal = constant %ctrlWithExtraSignal {value = 100 : i32} : <[test: i2]>, <i32, [test: i2]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @loadWithExtraSignal(%ctrl : !handshake.control<>, %addr : !handshake.channel<i32, [test: i2]>, %ldData : !handshake.channel<i32>) -> !handshake.control<> {
  %ldAddrToMem, %ldDataToSucc = load [%addr] %ldData : <i32, [test: i2]>, <i32>, <i32>, <i32, [test: i2]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @muxWithExtraSignal(
    %ctrl : !handshake.control<>,
    %sel : !handshake.channel<i2>,
    %data1 : !handshake.channel<i32>,
    %data2 : !handshake.channel<i32, [e1: i2]>,
    %data3 : !handshake.channel<i32, [e3: i6]>,
    %data4 : !handshake.channel<i32, [e1: i2, e2: i4]>) -> !handshake.control<> {
  %data = mux %sel [%data1, %data2, %data3, %data4] : <i2>, [<i32>, <i32, [e1: i2]>, <i32, [e3: i6]>, <i32, [e1: i2, e2: i4]>] to <i32, [e1: i2, e3: i6, e2: i4]>
  end %ctrl : !handshake.control<>
}

// ----

handshake.func @cmergeWithExtraSignal(
    %ctrl : !handshake.control<>,
    %data1 : !handshake.channel<i32>,
    %data2 : !handshake.channel<i32, [e1: i2]>,
    %data3 : !handshake.channel<i32, [e3: i6]>,
    %data4 : !handshake.channel<i32, [e1: i2, e2: i4]>) -> !handshake.control<> {
  %data, %idx = control_merge %data1, %data2, %data3, %data4 : [<i32>, <i32, [e1: i2]>, <i32, [e3: i6]>, <i32, [e1: i2, e2: i4]>] to <i32, [e1: i2, e3: i6, e2: i4]>, <i2>
  end %ctrl : !handshake.control<>
}
