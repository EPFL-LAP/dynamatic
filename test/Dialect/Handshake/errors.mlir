// RUN: dynamatic-opt %s --split-input-file --verify-diagnostics

// expected-error @below {{failed to parse ChannelType parameter 'dataType' which must be `IndexType`, `IntegerType`, or `FloatType`}}
handshake.func private @invalidDataType(%arg0: !handshake.channel<!handshake.control>) -> !handshake.control; 

// -----

// expected-error @below {{failed to parse extra signal type which must be `IndexType`, `IntegerType`, or `FloatType`}}
// expected-error @below {{failed to parse ChannelType parameter 'extraSignals' which is to be a `ArrayRef<ExtraSignal>`}}
handshake.func private @invalidExtraType(%arg0: !handshake.channel<i32, [extra: !handshake.control]>) -> !handshake.control; 

// -----

// expected-error @below {{duplicated extra signal name, signal names must be unique}}
// expected-error @below {{failed to parse ChannelType parameter 'extraSignals' which is to be a `ArrayRef<ExtraSignal>`}}
handshake.func private @duplicateExtraNames(%arg0: !handshake.channel<i32, [extra: i16, extra: f16]>) -> !handshake.control; 
