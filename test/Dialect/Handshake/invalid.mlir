// RUN: dynamatic-opt %s --split-input-file --verify-diagnostics

// expected-error @below {{failed to parse ChannelType parameter 'dataType' which must be `IndexType`, `IntegerType`, or `FloatType`}}
handshake.func @invalidDataType(%arg0: !handshake.channel<!handshake.control>) -> !handshake.control

// -----

// expected-error @below {{failed to parse extra signal type which must be `IndexType`, `IntegerType`, or `FloatType`}}
// expected-error @below {{failed to parse ChannelType parameter 'extraSignals' which is to be a `ArrayRef<ExtraSignal>`}}
handshake.func @invalidExtraType(%arg0: !handshake.channel<i32, [extra: !handshake.control]>) -> !handshake.control 

// -----

// expected-error @below {{duplicated extra signal name, signal names must be unique}}
// expected-error @below {{failed to parse ChannelType parameter 'extraSignals' which is to be a `ArrayRef<ExtraSignal>`}}
handshake.func @duplicateExtraNames(%arg0: !handshake.channel<i32, [extra: i16, extra: f16]>) -> !handshake.control

// -----

handshake.func @invalidUnbundleControlMissingReady(%ctrl: !handshake.control) -> i1 {
  // expected-error @below {{expected single i1 operand for ready}}
  %valid = unbundle %ctrl : (!handshake.control) -> i1  
  end %valid : i1
}

// -----

handshake.func @invalidUnbundleControlBadReady(%ctrl: !handshake.control, %badReady: i2) -> i1 {
  // expected-error @below {{expected single i1 operand for ready}}
  %valid = unbundle %ctrl [%badReady] : (!handshake.control, i2) -> i1 
  end %valid : i1
}

// -----

handshake.func @invalidUnbundleControlBadValid(%ctrl: !handshake.control, %ready: i1) -> i2 {
  // expected-error @below {{expected single i1 result for valid}}
  %badValid = unbundle %ctrl [%ready] : (!handshake.control, i1) -> i2
  end %badValid : i2
}

// -----

handshake.func @invalidUnbundleControlBadReady(%ctrl: !handshake.control, %ready: i1) -> i2 {
  // expected-error @below {{expected single i1 result for valid}}
  %badValid = unbundle %ctrl [%ready] : (!handshake.control, i1) -> i2 
  end %badValid : i2
}

// -----

handshake.func @invalidUnbundleChannelNotEnoughResults(%channel: !handshake.channel<i32>) -> i32 {
  // expected-error @below {{not enough results, unbundling a !handshake.channel should produce at least two results}}
  %badCtrl = unbundle %channel : (!handshake.channel<i32>) -> (i32) 
  end %badCtrl : i32
}

// -----

handshake.func @invalidUnbundleChannelInvalidControl(%channel: !handshake.channel<i32>) -> i32 {
  // expected-error @below {{type mistmatch between expected !handshake.control type and operation's first result ('i32'}}
  %badCtrl, %badData = unbundle %channel : (!handshake.channel<i32>) -> (i32, i16) 
  end %badCtrl : i32
}

// -----

handshake.func @invalidUnbundleChannelInvalidData(%channel: !handshake.channel<i32>) -> !handshake.control {
  // expected-error @below {{type mismatch between channel's data type ('i32') and operation's second result ('i16')}}
  %ctrl, %badData = unbundle %channel : (!handshake.channel<i32>) -> (!handshake.control, i16) 
  end %ctrl : !handshake.control
}

// -----

handshake.func @invalidUnbundleChannelMisingUp(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>) -> !handshake.control {
  // expected-error @below {{not enough operands, no value for extra signal 'extraUp'}}
  %ctrl, %data, %extraDown = unbundle %channel : (!handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>) -> (!handshake.control, i32, i4) 
  end %ctrl : !handshake.control
}

// -----

handshake.func @invalidUnbundleChannelBadUp(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, %badExtraUp: i2) -> !handshake.control {
  // expected-error @below {{type mismatch between extra signal 'extraUp' ('i1') and 1-th operand ('i2')}}
  %ctrl, %data, %extraDown = unbundle %channel [%badExtraUp] : (!handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, i2) -> (!handshake.control, i32, i4) 
  end %ctrl : !handshake.control
}

// -----

handshake.func @invalidUnbundleChannelExtraUp(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, %extraUp: i1, %otherExtraUp: i1) -> !handshake.control {
  // expected-error @below {{too many extra upstream values provided, expected 1 but got 2}}
  %ctrl, %data, %extraDown = unbundle %channel [%extraUp, %otherExtraUp] : (!handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, i1, i1) -> (!handshake.control, i32, i4) 
  end %ctrl : !handshake.control
}

// -----

handshake.func @invalidUnbundleChannelMisingDown(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, %extraUp: i1) -> !handshake.control {
  // expected-error @below {{not enough results, no value for extra signal 'extraDown'}}
  %ctrl, %data = unbundle %channel [%extraUp] : (!handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, i1) -> (!handshake.control, i32) 
  end %ctrl : !handshake.control
}

// -----

handshake.func @invalidUnbundleChannelBadDown(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, %extraUp: i1) -> !handshake.control {
  // expected-error @below {{type mismatch between extra signal 'extraDown' ('i4') and 2-th result ('i2')}}
  %ctrl, %data, %badExtraDown = unbundle %channel [%extraUp] : (!handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, i1) -> (!handshake.control, i32, i2) 
  end %ctrl : !handshake.control
}

// -----

handshake.func @invalidUnbundleChannelExtraDown(%channel: !handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, %extraUp: i1) -> !handshake.control {
  // expected-error @below {{too many extra downstream values provided, expected 1 but got 2}}
  %ctrl, %data, %extraDown, %badExtraDown = unbundle %channel [%extraUp] : (!handshake.channel<i32, [extraUp: i1 (U), extraDown: i4]>, i1) -> (!handshake.control, i32, i4, i2) 
  end %ctrl : !handshake.control
}
