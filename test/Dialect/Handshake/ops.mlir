// RUN: dynamatic-opt %s --split-input-file --verify-diagnostics

handshake.func @unbundleControl(%ctrl: !handshake.control<>) -> !handshake.control<> {
  %valid = unbundle %ctrl [%ready] : (!handshake.control<>, i1) -> i1
  %ctrlAgain, %ready = bundle %valid : (i1) -> (!handshake.control<>, i1)
  end %ctrlAgain : !handshake.control<>
}

// -----

handshake.func @unbundleChannelSimple(%channel: !handshake.channel<i32>) -> (i1, i32) {
  %ctrl, %data = unbundle %channel : (!handshake.channel<i32>) -> (!handshake.control<>, i32)
  %valid = unbundle %ctrl [%ready] : (!handshake.control<>, i1) -> (i1)
  %ctrlAgain, %ready = bundle %valid : (i1) -> (!handshake.control<>, i1)
  %channelAgain = bundle %ctrlAgain, %data : (!handshake.control<>, i32) -> (!handshake.channel<i32>)
  end %channelAgain : !handshake.channel<i32>
}

// -----

handshake.func @unbundleChannelComplex(%channel: !handshake.channel<i32, [e1: i1 (U), e2: f16, e3: i32 (U)]>) -> !handshake.channel<i32, [e1: i1 (U), e2: f16, e3: i32 (U)]> {
  %ctrl, %data, %e2 = unbundle %channel [%e1, %e3] : (!handshake.channel<i32, [e1: i1 (U), e2: f16, e3: i32 (U)]>, i1, i32) -> (!handshake.control<>, i32, f16)
  %channelAgain, %e1, %e3 = bundle %ctrl, %data, %e2 : (!handshake.control<>, i32, f16) -> (!handshake.channel<i32, [e1: i1 (U), e2: f16, e3: i32 (U)]>, i1, i32)
  end %channelAgain : !handshake.channel<i32, [e1: i1 (U), e2: f16, e3: i32 (U)]>
}

// -----

handshake.func @reshapeChannelIntoData(%channel: !handshake.channel<f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]>) -> !handshake.channel<f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]> {
  %reshaped = reshape [MergeData] %channel : (!handshake.channel<f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]>) -> (!handshake.channel<i42, [mergedUp: i8 (U)]>)
  %backToOriginal = reshape [SplitData] %reshaped : (!handshake.channel<i42, [mergedUp: i8 (U)]>) -> (!handshake.channel<f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]>)
  end %backToOriginal : !handshake.channel<f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]>
}

// -----

handshake.func @reshapeChannelIntoDataNoExtraDown(%channel: !handshake.channel<f32, [up1: i4 (U)]>) -> !handshake.channel<f32, [up1: i4 (U)]> {
  %reshaped = reshape [MergeData] %channel : (!handshake.channel<f32, [up1: i4 (U)]>) -> (!handshake.channel<f32, [mergedUp: i4 (U)]>)
  %backToOriginal = reshape [SplitData] %reshaped : (!handshake.channel<f32, [mergedUp: i4 (U)]>) -> (!handshake.channel<f32, [up1: i4 (U)]>)
  end %backToOriginal : !handshake.channel<f32, [up1: i4 (U)]>
}

// -----

handshake.func @reshapeChannelIntoExtra(%channel: !handshake.channel<f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]>) -> !handshake.channel<f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]> {
  %reshaped = reshape [MergeExtra] %channel : (!handshake.channel<f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]>) -> (!handshake.channel<f32, [mergedDown: i10, mergedUp: i8 (U)]>)
  %backToOriginal = reshape [SplitExtra] %reshaped : (!handshake.channel<f32, [mergedDown: i10, mergedUp: i8 (U)]>) -> (!handshake.channel<f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]>)
  end %backToOriginal : !handshake.channel<f32, [down1: i2, up1: i4 (U), up2: i4 (U), down2: i8]>
}
