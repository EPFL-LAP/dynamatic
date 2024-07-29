// RUN: dynamatic-opt %s --split-input-file --verify-diagnostics

// expected-error @below {{failed to parse ChannelType parameter 'dataType' which must be IntegerType or FloatType}}
handshake.func @invalidDataType(%arg0: !handshake.channel<!handshake.control<>>) -> !handshake.control<>

// -----

// expected-error @below {{failed to parse extra signal type which must be IntegerType or FloatType}}
// expected-error @below {{failed to parse ChannelType parameter 'extraSignals' which is to be a ArrayRef<ExtraSignal>}}
handshake.func @invalidExtraType(%arg0: !handshake.channel<i32, [extra: !handshake.control<>]>) -> !handshake.control<> 

// -----

// expected-error @below {{duplicated extra signal name, signal names must be unique}}
// expected-error @below {{failed to parse ChannelType parameter 'extraSignals' which is to be a ArrayRef<ExtraSignal>}}
handshake.func @duplicateExtraNames(%arg0: !handshake.channel<i32, [extra: i16, extra: f16]>) -> !handshake.control<>

// -----

handshake.func @invalidUnbundleControlMissingReady(%ctrl: !handshake.control<>) {
  // expected-error @below {{expected single i1 operand for ready}}
  %valid = unbundle %ctrl : (!handshake.control<>) -> i1  
  end
}

// -----

handshake.func @invalidUnbundleControlBadReady(%ctrl: !handshake.control<>, %badReady: i2) {
  // expected-error @below {{expected single i1 operand for ready}}
  %valid = unbundle %ctrl [%badReady] : (!handshake.control<>, i2) -> i1 
  end
}

// -----

handshake.func @invalidUnbundleControlMissingValid(%ctrl: !handshake.control<>, %ready: i1) {
  // expected-error @below {{expected single i1 result for valid}}
  unbundle %ctrl [%ready] : (!handshake.control<>, i1) -> () 
  end
}

// -----

handshake.func @invalidUnbundleControlBadValid(%ctrl: !handshake.control<>, %ready: i1) {
  // expected-error @below {{expected single i1 result for valid}}
  %badValid = unbundle %ctrl [%ready] : (!handshake.control<>, i1) -> i2
  end
}

// -----

handshake.func @invalidUnbundleChannelNotEnoughResults(%channel: !handshake.channel<i32>) {
  // expected-error @below {{not enough results, unbundling a !handshake.channel should produce at least two results}}
  %badCtrl = unbundle %channel : (!handshake.channel<i32>) -> (i32) 
  end
}

// -----

handshake.func @invalidUnbundleChannelInvalidControl(%channel: !handshake.channel<i32>) {
  // expected-error @below {{type mistmatch between expected !handshake.control type and operation's first result ('i32'}}
  %badCtrl, %badData = unbundle %channel : (!handshake.channel<i32>) -> (i32, i16) 
  end
}

// -----

handshake.func @invalidUnbundleChannelInvalidData(%channel: !handshake.channel<i32>) -> !handshake.control<> {
  // expected-error @below {{type mismatch between channel's data type ('i32') and operation's second result ('i16')}}
  %ctrl, %badData = unbundle %channel : (!handshake.channel<i32>) -> (!handshake.control<>, i16) 
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidUnbundleChannelMisingUp(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>) -> !handshake.control<> {
  // expected-error @below {{not enough operands, no value for extra signal 'extraUp'}}
  %ctrl, %data, %extraDown = unbundle %channel : (!handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>) -> (!handshake.control<>, i32, i4) 
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidUnbundleChannelBadUp(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, %badExtraUp: i2) -> !handshake.control<> {
  // expected-error @below {{type mismatch between extra signal 'extraUp' ('i1') and 1-th operand ('i2')}}
  %ctrl, %data, %extraDown = unbundle %channel [%badExtraUp] : (!handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, i2) -> (!handshake.control<>, i32, i4) 
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidUnbundleChannelExtraUp(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, %extraUp: i1, %otherExtraUp: i1) -> !handshake.control<> {
  // expected-error @below {{too many extra upstream values provided, expected 1 but got 2}}
  %ctrl, %data, %extraDown = unbundle %channel [%extraUp, %otherExtraUp] : (!handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, i1, i1) -> (!handshake.control<>, i32, i4) 
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidUnbundleChannelMisingDown(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, %extraUp: i1) -> !handshake.control<> {
  // expected-error @below {{not enough results, no value for extra signal 'extraDown'}}
  %ctrl, %data = unbundle %channel [%extraUp] : (!handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, i1) -> (!handshake.control<>, i32) 
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidUnbundleChannelBadDown(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, %extraUp: i1) -> !handshake.control<> {
  // expected-error @below {{type mismatch between extra signal 'extraDown' ('i4') and 2-th result ('i2')}}
  %ctrl, %data, %badExtraDown = unbundle %channel [%extraUp] : (!handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, i1) -> (!handshake.control<>, i32, i2) 
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidUnbundleChannelExtraDown(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, %extraUp: i1) -> !handshake.control<> {
  // expected-error @below {{too many extra downstream values provided, expected 1 but got 2}}
  %ctrl, %data, %extraDown, %badExtraDown = unbundle %channel [%extraUp] : (!handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, i1) -> (!handshake.control<>, i32, i4, i2) 
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidReshapeMergeDataExtraDown(%channel: !handshake.channel<f32, [down: i2]>) -> !handshake.channel<f32, [down: i2]> {
  // expected-error @below {{too many extra downstream signals in the destination type, expected 0 but got 1}}
  %reshaped = reshape [MergeData] %channel : (!handshake.channel<f32, [down: i2]>) -> (!handshake.channel<f32, [down: i2]>)
  end %reshaped : !handshake.channel<f32, [down: i2]>
}

// -----

handshake.func @invalidReshapeMergeDataManyUp(%channel: !handshake.channel<f32, [up1: i1 (U), up2: i2 (U)]>) -> !handshake.channel<f32, [up1: i1 (U), up2: i2 (U)]> {
  // expected-error @below {{merged channel type should have at most one uptream signal, but got 2}}
  %reshaped = reshape [MergeData] %channel : (!handshake.channel<f32, [up1: i1 (U), up2: i2 (U)]>) -> (!handshake.channel<f32, [up1: i1 (U), up2: i2 (U)]>)
  end %reshaped : !handshake.channel<f32, [up1: i1 (U), up2: i2 (U)]>
}

// -----

handshake.func @invalidReshapeMergeDataUpName(%channel: !handshake.channel<f32, [up1: i1 (U)]>) -> !handshake.channel<f32, [up1: i1 (U)]> {
  // expected-error @below {{invalid name for merged extra uptream signal, expected 'mergedUp' but got 'up1'}}
  %reshaped = reshape [MergeData] %channel : (!handshake.channel<f32, [up1: i1 (U)]>) -> (!handshake.channel<f32, [up1: i1 (U)]>)
  end %reshaped : !handshake.channel<f32, [up1: i1 (U)]>
}

// -----

handshake.func @invalidReshapeMergeDataUpWidth(%channel: !handshake.channel<f32, [up1: i1 (U), up2: i2 (U)]>) -> !handshake.channel<f32, [mergedUp: i4 (U)]> {
  // expected-error @below {{invalid bitwidth for merged extra uptream signal, expected 3 but got 4}}
  %reshaped = reshape [MergeData] %channel : (!handshake.channel<f32, [up1: i1 (U), up2: i2 (U)]>) -> (!handshake.channel<f32, [mergedUp: i4 (U)]>)
  end %reshaped : !handshake.channel<f32, [mergedUp: i4 (U)]>
}

// -----

handshake.func @invalidReshapeMergeDataUpType(%channel: !handshake.channel<f32, [up1: f16 (U), up2: f16 (U)]>) -> !handshake.channel<f32, [mergedUp: f32 (U)]> {
  // expected-error @below {{invalid type for merged extra uptream signal, expected IntegerType but got 'f32'}}
  %reshaped = reshape [MergeData] %channel : (!handshake.channel<f32, [up1: f16 (U), up2: f16 (U)]>) -> (!handshake.channel<f32, [mergedUp: f32 (U)]>)
  end %reshaped : !handshake.channel<f32, [mergedUp: f32 (U)]>
}

// -----

handshake.func @invalidReshapeMergeDataWidth(%channel: !handshake.channel<f32, [down1: i16, down2: i16]>) -> !handshake.channel<i65> {
  // expected-error @below {{invalid merged data type bitwidth, expected 64 but got 65}}
  %reshaped = reshape [MergeData] %channel : (!handshake.channel<f32, [down1: i16, down2: i16]>) -> (!handshake.channel<i65>)
  end %reshaped : !handshake.channel<i65>
}

// -----

handshake.func @invalidReshapeMergeDataWidth(%channel: !handshake.channel<f32, [down1: i16, down2: i16]>) -> !handshake.channel<i65> {
  // expected-error @below {{invalid merged data type, expected merged IntegerType but got 'f64'}}
  %reshaped = reshape [MergeData] %channel : (!handshake.channel<f32, [down1: i16, down2: i16]>) -> (!handshake.channel<f64>)
  end %reshaped : !handshake.channel<f64>
}

// -----

handshake.func @invalidReshapeMergeDataNoChange(%channel: !handshake.channel<f32>) -> !handshake.channel<i32> {
  // expected-error @below {{invalid destination data type, expected source data type 'f32' but got 'i32'}}
  %reshaped = reshape [MergeData] %channel : (!handshake.channel<f32>) -> (!handshake.channel<i32>)
  end %reshaped : !handshake.channel<i32>
}

// -----

handshake.func @invalidReshapeMergeExtraSameData(%channel: !handshake.channel<f32, [down1: i16, down2: i16]>) -> !handshake.channel<i32, [mergedDown: i32]> {
  // expected-error @below {{reshaping in this mode should not change the data type, expected 'f32' but got 'i32'}}
  %reshaped = reshape [MergeExtra] %channel : (!handshake.channel<f32, [down1: i16, down2: i16]>) -> (!handshake.channel<i32, [mergedDown: i32]>)
  end %reshaped : !handshake.channel<i32, [mergedDown: i32]>
}
