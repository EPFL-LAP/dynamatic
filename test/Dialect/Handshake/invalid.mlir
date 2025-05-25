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

handshake.func @invalidSourceAndConstantWithExtraSignal(%ctrl : !handshake.control<>) -> !handshake.control<> {
  %ctrlWithExtraSignal = source : <[test: i2]>
  // expected-error @below {{'handshake.constant' op failed to verify that all of {ctrl, result} have same extra signals}}
  %valueWithoutExtraSignal = constant %ctrlWithExtraSignal {value = 100 : i32} : <[test: i2]>, <i32>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidMuxWithDifferentDataTypesOutput(
    %ctrl : !handshake.control<>,
    %sel : !handshake.channel<i2, [spec: i1]>,
    %data1 : !handshake.channel<i32, [spec: i1]>,
    %data2 : !handshake.channel<i32, [spec: i1]>,
    %data3 : !handshake.channel<i32, [spec: i1]>,
    %data4 : !handshake.channel<i32, [spec: i1]>) -> !handshake.control<> {
  // expected-error @below {{'handshake.mux' op failed to verify that all of {dataOperands, result} should have the same data type}}
  %data = mux %sel [%data1, %data2, %data3, %data4] : <i2, [spec: i1]>, [<i32, [spec: i1]>, <i32, [spec: i1]>, <i32, [spec: i1]>, <i32, [spec: i1]>] to <i1, [spec: i1]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidMuxWithDifferentDataTypesVariadic(
    %ctrl : !handshake.control<>,
    %sel : !handshake.channel<i2, [spec: i1]>,
    %data1 : !handshake.channel<i32, [spec: i1]>,
    %data2 : !handshake.channel<i32, [spec: i1]>,
    %data3 : !handshake.channel<i32, [spec: i1]>,
    %data4 : !handshake.channel<i1, [spec: i1]>) -> !handshake.control<> {
  // expected-error @below {{'handshake.mux' op failed to verify that all of {dataOperands, result} should have the same data type}}
  %data = mux %sel [%data1, %data2, %data3, %data4] : <i2, [spec: i1]>, [<i32, [spec: i1]>, <i32, [spec: i1]>, <i32, [spec: i1]>, <i1, [spec: i1]>] to <i32, [spec: i1]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidMuxWithDifferentExtraSignals(
    %ctrl : !handshake.control<>,
    %sel : !handshake.channel<i2, [spec: i1]>,
    %data1 : !handshake.channel<i32, [spec: i1]>,
    %data2 : !handshake.channel<i32, [spec: i1]>,
    %data3 : !handshake.channel<i32, [spec: i1, tag: i8]>,
    %data4 : !handshake.channel<i32, [spec: i1]>) -> !handshake.control<> {
  // expected-error @below {{'handshake.mux' op failed to verify that all of {dataOperands, result, selectOperand} should have the same extra signals}}
  %data = mux %sel [%data1, %data2, %data3, %data4] : <i2, [spec: i1]>, [<i32, [spec: i1]>, <i32, [spec: i1]>, <i32, [spec: i1, tag: i8]>, <i32, [spec: i1]>] to <i32, [spec: i1]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidMuxWithConflictingExtraSignals(
    %ctrl : !handshake.control<>,
    %sel : !handshake.channel<i2, [spec: i1]>,
    %data1 : !handshake.channel<i32, [spec: i1]>,
    %data2 : !handshake.channel<i32, [spec: i1]>,
    %data3 : !handshake.channel<i32, [spec: i2]>,
    %data4 : !handshake.channel<i32, [spec: i1]>) -> !handshake.control<> {
  // expected-error @below {{'handshake.mux' op failed to verify that all of {dataOperands, result, selectOperand} should have the same extra signals}}
  %data = mux %sel [%data1, %data2, %data3, %data4] : <i2, [spec: i1]>, [<i32, [spec: i1]>, <i32, [spec: i1]>, <i32, [spec: i2]>, <i32, [spec: i1]>] to <i32, [spec: i1]>
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

// -----

handshake.func @invalidSpecCommitWithInvalidDataTypes(
    %ctrl : !handshake.channel<i1>,
    %dataIn : !handshake.channel<i32, [spec: i1]>) {
  // expected-error @below {{'handshake.spec_commit' op failed to verify that all of {dataIn, dataOut} have same data type}}
  %dataOut = spec_commit[%ctrl] %dataIn : !handshake.channel<i32, [spec: i1]>, !handshake.control<>, <i1>
  end
}
