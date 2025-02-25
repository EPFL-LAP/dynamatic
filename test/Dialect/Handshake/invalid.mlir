// RUN: dynamatic-opt %s --split-input-file --verify-diagnostics

// expected-error @below {{expected data type to be IntegerType or FloatType, but got 'index'}}
handshake.func @invalidDataType(%arg0: !handshake.channel<index>) 

// -----

// expected-error @below {{expected data type to have strictly positive bitwidth, but got 'i0'}}
handshake.func @invalidZeroWidthDataType(%arg0: !handshake.channel<i0>) 

// -----

// expected-error @below {{expected extra signal type to be IntegerType or FloatType, but extra' has type 'index'}}
handshake.func @invalidExtraType(%arg0: !handshake.channel<i32, [extra: index]>)  

// -----

// expected-error @below {{expected extra signal type to be IntegerType or FloatType, but extra' has type 'index'}}
handshake.func @invalidExtraTypeControl(%arg0: !handshake.control<[extra: index]>)  

// -----

// expected-error @below {{expected all signal names to be unique but 'extra' appears more than once}}
handshake.func @duplicateExtraNames(%arg0: !handshake.channel<i32, [extra: i16, extra: f16]>) 

// -----

// expected-error @below {{expected all signal names to be unique but 'extra' appears more than once}}
handshake.func @duplicateExtraNamesControl(%arg0: !handshake.control<[extra: i16, extra: f16]>) 

// -----

// expected-error @below {{'valid' is a reserved name, it cannot be used as an extra signal name}}
handshake.func @reservedExtraSignalName(%arg0: !handshake.channel<i32, [valid: i1]>)  

// -----

// expected-error @below {{'valid' is a reserved name, it cannot be used as an extra signal name}}
handshake.func @reservedExtraSignalNameControl(%arg0: !handshake.control<[valid: i1]>)  

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

// -----

handshake.func @invalidSourceAndConstantWithExtraSignal(%ctrl : !handshake.control<>) -> !handshake.control<> {
  %ctrlWithExtraSignal = source : <[test: i2]>
  // expected-error @below {{'handshake.constant' op failed to verify that all of {ctrl, result} have same extra signals}}
  %valueWithoutExtraSignal = constant %ctrlWithExtraSignal {value = 100 : i32} : <[test: i2]>, <i32>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidMuxWithDifferentDataTypesOutput(
    %ctrl : !handshake.control<>,
    %sel : !handshake.channel<i2>,
    %data1 : !handshake.channel<i32>,
    %data2 : !handshake.channel<i32, [e1: i2]>,
    %data3 : !handshake.channel<i32, [e3: i6]>,
    %data4 : !handshake.channel<i32, [e1: i2, e2: i4]>) -> !handshake.control<> {
  // expected-error @below {{'handshake.mux' op failed to verify that the operands inside the variadic name and result should all have the same type}}
  %data = mux %sel [%data1, %data2, %data3, %data4] : <i2>, [<i32>, <i32, [e1: i2]>, <i32, [e3: i6]>, <i32, [e1: i2, e2: i4]>] to <i1, [e1: i2, e3: i6, e2: i4]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidMuxWithDifferentDataTypesVariadic(
    %ctrl : !handshake.control<>,
    %sel : !handshake.channel<i2>,
    %data1 : !handshake.channel<i32>,
    %data2 : !handshake.channel<i32, [e1: i2]>,
    %data3 : !handshake.channel<i32, [e3: i6]>,
    %data4 : !handshake.channel<i1, [e1: i2, e2: i4]>) -> !handshake.control<> {
  // expected-error @below {{'handshake.mux' op failed to verify that the operands inside the variadic name and result should all have the same type}}
  %data = mux %sel [%data1, %data2, %data3, %data4] : <i2>, [<i32>, <i32, [e1: i2]>, <i32, [e3: i6]>, <i1, [e1: i2, e2: i4]>] to <i32, [e1: i2, e3: i6, e2: i4]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidMuxWithMoreExtraSignal(
    %ctrl : !handshake.control<>,
    %sel : !handshake.channel<i2>,
    %data1 : !handshake.channel<i32>,
    %data2 : !handshake.channel<i32, [e1: i2]>,
    %data3 : !handshake.channel<i32, [e3: i6]>,
    %data4 : !handshake.channel<i32, [e1: i2, e2: i4]>) -> !handshake.control<> {
  // expected-error @below {{'handshake.mux' op failed to verify that the extra signals of inputs and output should satisfy "Merging Relationship of Extra Signals"}}
  %data = mux %sel [%data1, %data2, %data3, %data4] : <i2>, [<i32>, <i32, [e1: i2]>, <i32, [e3: i6]>, <i32, [e1: i2, e2: i4]>] to <i32, [e1: i2, e3: i6, e2: i4, e4: i8]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidMuxWithLessExtraSignal(
    %ctrl : !handshake.control<>,
    %sel : !handshake.channel<i2>,
    %data1 : !handshake.channel<i32>,
    %data2 : !handshake.channel<i32, [e1: i2]>,
    %data3 : !handshake.channel<i32, [e3: i6]>,
    %data4 : !handshake.channel<i32, [e1: i2, e2: i4]>) -> !handshake.control<> {
  // expected-error @below {{'handshake.mux' op failed to verify that the extra signals of inputs and output should satisfy "Merging Relationship of Extra Signals"}}
  %data = mux %sel [%data1, %data2, %data3, %data4] : <i2>, [<i32>, <i32, [e1: i2]>, <i32, [e3: i6]>, <i32, [e1: i2, e2: i4]>] to <i32, [e1: i2, e3: i6]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidMuxWithConflictingExtraSignal(
    %ctrl : !handshake.control<>,
    %sel : !handshake.channel<i2>,
    %data1 : !handshake.channel<i32>,
    %data2 : !handshake.channel<i32, [e1: i2]>,
    %data3 : !handshake.channel<i32, [e2: i6]>,
    %data4 : !handshake.channel<i32, [e1: i2, e2: i4]>) -> !handshake.control<> {
  // expected-error @below {{'handshake.mux' op failed to verify that the extra signals of inputs and output should satisfy "Merging Relationship of Extra Signals"}}
  %data = mux %sel [%data1, %data2, %data3, %data4] : <i2>, [<i32>, <i32, [e1: i2]>, <i32, [e2: i6]>, <i32, [e1: i2, e2: i4]>] to <i32, [e1: i2, e2: i6]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidMuxWithNoDataInputs(
    %ctrl : !handshake.control<>,
    %sel : !handshake.channel<i2>) -> !handshake.control<> {
  // expected-error @below {{'handshake.mux' op failed to verify that the variadic dataOperands should have at least one element}}
  %data = mux %sel [] : <i2>, [] to <i32>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidSpecCommitWithDifferentExtraSignals(
    %ctrl : !handshake.channel<i1>,
    %dataIn : !handshake.channel<i32, [a: i1, spec: i1]>) {
  // expected-error @below {{'handshake.spec_commit' op failed to verify that all of {dataIn, dataOut} should have the same extra signals except for spec}}
  %dataOut = spec_commit[%ctrl] %dataIn : !handshake.channel<i32, [a: i1, spec: i1]>, !handshake.channel<i32>, <i1>
  end
}

// -----

handshake.func @invalidSpecCommitWithInvalidSpec(
    %ctrl : !handshake.channel<i1>,
    %dataIn : !handshake.channel<i32, [a: i1, spec: i2]>) {
  // expected-error @below {{'handshake.spec_commit' op failed to verify that should have a valid spec tag as an extra signal}}
  %dataOut = spec_commit[%ctrl] %dataIn : !handshake.channel<i32, [a: i1, spec: i2]>, !handshake.channel<i32, [a: i1]>, <i1>
  end
}

// -----

handshake.func @invalidSpecCommitWithDataOutSpec(
    %ctrl : !handshake.channel<i1>,
    %dataIn : !handshake.channel<i32, [a: i1, spec: i1]>) {
  // expected-error @below {{'handshake.spec_commit' op failed to verify that shouldn't have a spec tag as an extra signal}}
  %dataOut = spec_commit[%ctrl] %dataIn : !handshake.channel<i32, [a: i1, spec: i1]>, !handshake.channel<i32, [a: i1, spec: i2]>, <i1>
  end
}

// -----

handshake.func @invalidSpecCommitWithInvalidCtrl(
    %ctrl : !handshake.channel<i2>,
    %dataIn : !handshake.channel<i32, [spec: i1]>) {
  // expected-error @below {{'handshake.spec_commit' op failed to verify that ctrl should be of ChannelType carrying IntegerType data of width 1}}
  %dataOut = spec_commit[%ctrl] %dataIn : !handshake.channel<i32, [spec: i1]>, !handshake.channel<i32>, <i2>
  end
}
