// RUN: dynamatic-opt %s --split-input-file --verify-diagnostics

// expected-error @below {{expected data type to be IntegerType or FloatType, but got 'index'}}
handshake.func @invalidDataType(%arg0: <index>) 

// -----

// expected-error @below {{expected data type to have strictly positive bitwidth, but got 'i0'}}
handshake.func @invalidZeroWidthDataType(%arg0: <i0>) 

// -----

// expected-error @below {{expected extra signal type to be IntegerType or FloatType, but extra' has type 'index'}}
handshake.func @invalidExtraType(%arg0: <i32, [extra: index]>)  

// -----

// expected-error @below {{expected extra signal type to be IntegerType or FloatType, but extra' has type 'index'}}
handshake.func @invalidExtraTypeControl(%arg0: <[extra: index]>)  

// -----

// expected-error @below {{expected all signal names to be unique but 'extra' appears more than once}}
handshake.func @duplicateExtraNames(%arg0: <i32, [extra: i16, extra: f16]>) 

// -----

// expected-error @below {{expected all signal names to be unique but 'extra' appears more than once}}
handshake.func @duplicateExtraNamesControl(%arg0: <[extra: i16, extra: f16]>) 

// -----

// expected-error @below {{'valid' is a reserved name, it cannot be used as an extra signal name}}
handshake.func @reservedExtraSignalName(%arg0: <i32, [valid: i1]>)  

// -----

// expected-error @below {{'valid' is a reserved name, it cannot be used as an extra signal name}}
handshake.func @reservedExtraSignalNameControl(%arg0: <[valid: i1]>)  

// -----

handshake.func @invalidUnbundleControlMissingReady(%ctrl: <>) {
  // expected-error @below {{custom op 'handshake.unbundle' number of operands and types do not match: got 1 operands and 2 types}}
  %valid = unbundle %ctrl : <> to _  
  end
}

// -----

// expected-note @below {{prior use here}}
handshake.func @invalidUnbundleControlBadReady(%ctrl: <>, %badReady: i2) {
  // expected-error @below {{use of value '%badReady' expects different type than prior uses: 'i1' vs 'i2'}}
  %valid = unbundle %ctrl [%badReady] : <> to _ 
  end
}

// -----

handshake.func @invalidUnbundleChannelNotEnoughResults(%channel: <i32>) {
  // expected-error @below {{operation defines 2 results but was provided 1 to bind}}
  %badCtrl = unbundle %channel : <i32> to _ 
  end
}

// -----

handshake.func @invalidUnbundleChannelMisingUp(%channel: <i32, [extraUp: i1 (U), extraDown: i4]>) -> (<>) {
  // expected-error @below {{custom op 'handshake.unbundle' number of operands and types do not match: got 1 operands and 2 types}}
  %ctrl, %data, %extraDown = unbundle %channel : <i32, [extraUp: i1 (U), extraDown: i4]> to _ 
  end %ctrl : !handshake.control<>
}

// -----

// expected-note @below {{prior use here}}
handshake.func @invalidUnbundleChannelBadUp(%channel: <i32, [extraUp: i1 (U), extraDown: i4]>, %badExtraUp: i2) -> (<>) {
  // expected-error @below {{use of value '%badExtraUp' expects different type than prior uses: 'i1' vs 'i2'}}
  %ctrl, %data, %extraDown = unbundle %channel [%badExtraUp] : <i32, [extraUp: i1 (U), extraDown: i4]> to _ 
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidUnbundleChannelExtraUp(%channel: <i32, [extraUp: i1 (U), extraDown: i4]>, %extraUp: i1, %otherExtraUp: i1) -> (<>) {
  // expected-error @below {{custom op 'handshake.unbundle' number of operands and types do not match: got 3 operands and 2 types}}
  %ctrl, %data, %extraDown = unbundle %channel [%extraUp, %otherExtraUp] : <i32, [extraUp: i1 (U), extraDown: i4]> to _
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidUnbundleChannelMisingDown(%channel: <i32, [extraUp: i1 (U), extraDown: i4]>, %extraUp: i1) -> (<>) {
  // expected-error @below {{operation defines 3 results but was provided 2 to bind}}
  %ctrl, %data = unbundle %channel [%extraUp] : <i32, [extraUp: i1 (U), extraDown: i4]> to _ 
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidUnbundleChannelExtraDown(%channel: <i32, [extraUp: i1 (U), extraDown: i4]>, %extraUp: i1) -> (<>) {
  // expected-error @below {{operation defines 3 results but was provided 4 to bind}}
  %ctrl, %data, %extraDown, %badExtraDown = unbundle %channel [%extraUp] : <i32, [extraUp: i1 (U), extraDown: i4]> to _ 
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidSourceAndConstantWithExtraSignal(%ctrl: <>) -> (<>) {
  %ctrlWithExtraSignal = source : <[test: i2]>
  // expected-error @below {{'handshake.constant' op failed to verify that all of {ctrl, result} have same extra signals}}
  %valueWithoutExtraSignal = constant %ctrlWithExtraSignal {value = 100 : i32} : <[test: i2]>, <i32>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidMuxWithDifferentDataTypesOutput(%ctrl: <>, %sel: <i2, [spec: i1]>, %data1: <i32, [spec: i1]>, %data2: <i32, [spec: i1]>, %data3: <i32, [spec: i1]>, %data4: <i32, [spec: i1]>) -> (<>) {
  // expected-error @below {{'handshake.mux' op failed to verify that all of {dataOperands, result} should have the same data type}}
  %data = mux %sel [%data1, %data2, %data3, %data4] : <i2, [spec: i1]>, [<i32, [spec: i1]>, <i32, [spec: i1]>, <i32, [spec: i1]>, <i32, [spec: i1]>] to <i1, [spec: i1]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidMuxWithDifferentDataTypesVariadic(%ctrl: <>, %sel: <i2, [spec: i1]>, %data1: <i32, [spec: i1]>, %data2: <i32, [spec: i1]>, %data3: <i32, [spec: i1]>, %data4: <i1, [spec: i1]>) -> (<>) {
  // expected-error @below {{'handshake.mux' op failed to verify that all of {dataOperands, result} should have the same data type}}
  %data = mux %sel [%data1, %data2, %data3, %data4] : <i2, [spec: i1]>, [<i32, [spec: i1]>, <i32, [spec: i1]>, <i32, [spec: i1]>, <i1, [spec: i1]>] to <i32, [spec: i1]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidMuxWithDifferentExtraSignals(%ctrl: <>, %sel: <i2, [spec: i1]>, %data1: <i32, [spec: i1]>, %data2: <i32, [spec: i1]>, %data3: <i32, [spec: i1, tag: i8]>, %data4: <i32, [spec: i1]>) -> (<>) {
  // expected-error @below {{'handshake.mux' op failed to verify that all of {dataOperands, result, selectOperand} should have the same extra signals}}
  %data = mux %sel [%data1, %data2, %data3, %data4] : <i2, [spec: i1]>, [<i32, [spec: i1]>, <i32, [spec: i1]>, <i32, [spec: i1, tag: i8]>, <i32, [spec: i1]>] to <i32, [spec: i1]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidMuxWithConflictingExtraSignals(%ctrl: <>, %sel: <i2, [spec: i1]>, %data1: <i32, [spec: i1]>, %data2: <i32, [spec: i1]>, %data3: <i32, [spec: i2]>, %data4: <i32, [spec: i1]>) -> (<>) {
  // expected-error @below {{'handshake.mux' op failed to verify that all of {dataOperands, result, selectOperand} should have the same extra signals}}
  %data = mux %sel [%data1, %data2, %data3, %data4] : <i2, [spec: i1]>, [<i32, [spec: i1]>, <i32, [spec: i1]>, <i32, [spec: i2]>, <i32, [spec: i1]>] to <i32, [spec: i1]>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidMuxWithNoDataInputs(%ctrl: <>, %sel: <i2>) -> (<>) {
  // expected-error @below {{'handshake.mux' op failed to verify that the variadic dataOperands should have at least one element}}
  %data = mux %sel [] : <i2>, [] to <i32>
  end %ctrl : !handshake.control<>
}

// -----

handshake.func @invalidSpecCommitWithDifferentExtraSignals(%ctrl: <i1>, %dataIn: <i32, [a: i1, spec: i1]>) {
  // expected-error @below {{'handshake.spec_commit' op failed to verify that all of {dataIn, dataOut} should have the same extra signals except for spec}}
  %dataOut = spec_commit[%ctrl] %dataIn : !handshake.channel<i32, [a: i1, spec: i1]>, !handshake.channel<i32>, <i1>
  end
}

// -----

handshake.func @invalidSpecCommitWithInvalidSpec(%ctrl: <i1>, %dataIn: <i32, [a: i1, spec: i2]>) {
  // expected-error @below {{'handshake.spec_commit' op failed to verify that should have a valid spec tag as an extra signal}}
  %dataOut = spec_commit[%ctrl] %dataIn : !handshake.channel<i32, [a: i1, spec: i2]>, !handshake.channel<i32, [a: i1]>, <i1>
  end
}

// -----

handshake.func @invalidSpecCommitWithDataOutSpec(%ctrl: <i1>, %dataIn: <i32, [a: i1, spec: i1]>) {
  // expected-error @below {{'handshake.spec_commit' op failed to verify that shouldn't have a spec tag as an extra signal}}
  %dataOut = spec_commit[%ctrl] %dataIn : !handshake.channel<i32, [a: i1, spec: i1]>, !handshake.channel<i32, [a: i1, spec: i2]>, <i1>
  end
}

// -----

handshake.func @invalidSpecCommitWithInvalidCtrl(%ctrl: <i2>, %dataIn: <i32, [spec: i1]>) {
  // expected-error @below {{'handshake.spec_commit' op failed to verify that ctrl should be of ChannelType carrying IntegerType data of width 1}}
  %dataOut = spec_commit[%ctrl] %dataIn : !handshake.channel<i32, [spec: i1]>, !handshake.channel<i32>, <i2>
  end
}

// -----

handshake.func @invalidSpecCommitWithInvalidDataTypes(%ctrl: <i1>, %dataIn: <i32, [spec: i1]>) {
  // expected-error @below {{'handshake.spec_commit' op failed to verify that all of {dataIn, dataOut} have same data type}}
  %dataOut = spec_commit[%ctrl] %dataIn : !handshake.channel<i32, [spec: i1]>, !handshake.control<>, <i1>
  end
}
