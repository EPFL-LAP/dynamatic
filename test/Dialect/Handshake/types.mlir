// RUN: dynamatic-opt %s --split-input-file --verify-diagnostics

handshake.func @simpleControl(%arg0: !handshake.control) -> !handshake.control

// -----

handshake.func @simpleChannel(%arg0: !handshake.channel<i32>) -> !handshake.control

// -----

handshake.func @simpleChannelWithDownExtra(%arg0: !handshake.channel<i32, [extra: i1]>) -> !handshake.control

// -----

handshake.func @simpleChannelWithUpExtra(%arg0: !handshake.channel<i32, [extra: i1 (U)]>) -> !handshake.control

// -----

handshake.func @simpleChannelWithDownAndUpExtra(%arg0: !handshake.channel<i32, [extraDown: i1, extraUp: i1 (U)]>) -> !handshake.control

// -----

handshake.func @validDataAndExtraTypes(%arg0: !handshake.channel<f32, [idx: index]>) -> !handshake.control
