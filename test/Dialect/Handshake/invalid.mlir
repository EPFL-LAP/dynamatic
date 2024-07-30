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
  // expected-error @below {{custom op 'handshake.unbundle' 1 operands present, but expected 2}}
  %valid = unbundle %ctrl : <> to _  
  end
}

// -----

// expected-note @below {{prior use here}}
handshake.func @invalidUnbundleControlBadReady(%ctrl: !handshake.control<>, %badReady: i2) {
  // expected-error @below {{use of value '%badReady' expects different type than prior uses: 'i1' vs 'i2'}}
  %valid = unbundle %ctrl [%badReady] : <> to _ 
  end
}

// -----

handshake.func @invalidUnbundleChannelNotEnoughResults(%channel: !handshake.channel<i32>) {
  // expected-error @below {{operation defines 2 results but was provided 1 to bind}}
  %badCtrl = unbundle %channel : <i32> to _ 
  end
}

// -----

handshake.func @invalidUnbundleChannelMisingUp(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>) -> !handshake.control<> {
  // expected-error @below {{custom op 'handshake.unbundle' 1 operands present, but expected 2}}
  %ctrl, %data, %extraDown = unbundle %channel : <i32, [extraUp: i1 (U), extraDown: i4]> to _ 
  end %ctrl : !handshake.control<>
}

// -----

// expected-note @below {{prior use here}}
handshake.func @invalidUnbundleChannelBadUp(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, %badExtraUp: i2) -> !handshake.control<> {
  // expected-error @below {{use of value '%badExtraUp' expects different type than prior uses: 'i1' vs 'i2'}}
  %ctrl, %data, %extraDown = unbundle %channel [%badExtraUp] : <i32, [extraUp: i1 (U), extraDown: i4]> to _ 
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidUnbundleChannelExtraUp(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, %extraUp: i1, %otherExtraUp: i1) -> !handshake.control<> {
  // expected-error @below {{custom op 'handshake.unbundle' 3 operands present, but expected 2}}
  %ctrl, %data, %extraDown = unbundle %channel [%extraUp, %otherExtraUp] : <i32, [extraUp: i1 (U), extraDown: i4]> to _
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidUnbundleChannelMisingDown(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, %extraUp: i1) -> !handshake.control<> {
  // expected-error @below {{operation defines 3 results but was provided 2 to bind}}
  %ctrl, %data = unbundle %channel [%extraUp] : <i32, [extraUp: i1 (U), extraDown: i4]> to _ 
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidUnbundleChannelExtraDown(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, %extraUp: i1) -> !handshake.control<> {
  // expected-error @below {{operation defines 3 results but was provided 4 to bind}}
  %ctrl, %data, %extraDown, %badExtraDown = unbundle %channel [%extraUp] : <i32, [extraUp: i1 (U), extraDown: i4]> to _ 
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidReshapeMergeDataExtraDown(%channel: !handshake.channel<f32, [down: i2]>) -> !handshake.channel<f32, [down: i2]> {
  // expected-error @below {{too many extra downstream signals in the destination type, expected 0 but got 1}}
  %reshaped = reshape [MergeData] %channel : <f32, [down: i2]> -> <f32, [down: i2]>
  end %reshaped : !handshake.channel<f32, [down: i2]>
}

// -----

handshake.func @invalidReshapeMergeDataManyUp(%channel: !handshake.channel<f32, [up1: i1 (U), up2: i2 (U)]>) -> !handshake.channel<f32, [up1: i1 (U), up2: i2 (U)]> {
  // expected-error @below {{merged channel type should have at most one uptream signal, but got 2}}
  %reshaped = reshape [MergeData] %channel : <f32, [up1: i1 (U), up2: i2 (U)]> -> <f32, [up1: i1 (U), up2: i2 (U)]>
  end %reshaped : !handshake.channel<f32, [up1: i1 (U), up2: i2 (U)]>
}

// -----

handshake.func @invalidReshapeMergeDataUpName(%channel: !handshake.channel<f32, [up1: i1 (U)]>) -> !handshake.channel<f32, [up1: i1 (U)]> {
  // expected-error @below {{invalid name for merged extra uptream signal, expected 'mergedUp' but got 'up1'}}
  %reshaped = reshape [MergeData] %channel : <f32, [up1: i1 (U)]> -> <f32, [up1: i1 (U)]>
  end %reshaped : !handshake.channel<f32, [up1: i1 (U)]>
}

// -----

handshake.func @invalidReshapeMergeDataUpWidth(%channel: !handshake.channel<f32, [up1: i1 (U), up2: i2 (U)]>) -> !handshake.channel<f32, [mergedUp: i4 (U)]> {
  // expected-error @below {{invalid bitwidth for merged extra uptream signal, expected 3 but got 4}}
  %reshaped = reshape [MergeData] %channel : <f32, [up1: i1 (U), up2: i2 (U)]> -> <f32, [mergedUp: i4 (U)]>
  end %reshaped : !handshake.channel<f32, [mergedUp: i4 (U)]>
}

// -----

handshake.func @invalidReshapeMergeDataUpType(%channel: !handshake.channel<f32, [up1: f16 (U), up2: f16 (U)]>) -> !handshake.channel<f32, [mergedUp: f32 (U)]> {
  // expected-error @below {{invalid type for merged extra uptream signal, expected IntegerType but got 'f32'}}
  %reshaped = reshape [MergeData] %channel : <f32, [up1: f16 (U), up2: f16 (U)]> -> <f32, [mergedUp: f32 (U)]>
  end %reshaped : !handshake.channel<f32, [mergedUp: f32 (U)]>
}

// -----

handshake.func @invalidReshapeMergeDataWidth(%channel: !handshake.channel<f32, [down1: i16, down2: i16]>) -> !handshake.channel<i65> {
  // expected-error @below {{invalid merged data type bitwidth, expected 64 but got 65}}
  %reshaped = reshape [MergeData] %channel : <f32, [down1: i16, down2: i16]> -> <i65>
  end %reshaped : !handshake.channel<i65>
}

// -----

handshake.func @invalidReshapeMergeDataWidth(%channel: !handshake.channel<f32, [down1: i16, down2: i16]>) -> !handshake.channel<i65> {
  // expected-error @below {{invalid merged data type, expected merged IntegerType but got 'f64'}}
  %reshaped = reshape [MergeData] %channel : <f32, [down1: i16, down2: i16]> -> <f64>
  end %reshaped : !handshake.channel<f64>
}

// -----

handshake.func @invalidReshapeMergeDataNoChange(%channel: !handshake.channel<f32>) -> !handshake.channel<i32> {
  // expected-error @below {{invalid destination data type, expected source data type 'f32' but got 'i32'}}
  %reshaped = reshape [MergeData] %channel : <f32> -> <i32>
  end %reshaped : !handshake.channel<i32>
}

// -----

handshake.func @invalidReshapeMergeExtraSameData(%channel: !handshake.channel<f32, [down1: i16, down2: i16]>) -> !handshake.channel<i32, [mergedDown: i32]> {
  // expected-error @below {{reshaping in this mode should not change the data type, expected 'f32' but got 'i32'}}
  %reshaped = reshape [MergeExtra] %channel : <f32, [down1: i16, down2: i16]> -> <i32, [mergedDown: i32]>
  end %reshaped : !handshake.channel<i32, [mergedDown: i32]>
}
