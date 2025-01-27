// RUN: dynamatic-opt %s --split-input-file --verify-diagnostics

handshake.func @simpleControl(%arg0: !handshake.control<>) -> !handshake.control<>

// -----

handshake.func @simpleChannel(%arg0: !handshake.channel<i32>) -> !handshake.control<>

// -----

handshake.func @simpleControlWithDownExtra(%arg0: !handshake.control<[extra: i1]>) -> !handshake.control<>

// -----

handshake.func @simpleControlWithUpExtra(%arg0: !handshake.control<[extra: i1 (U)]>) -> !handshake.control<>

// -----

handshake.func @simpleControlWithDownAndUpExtra(%arg0: !handshake.control<[extraDown: i1, extraUp: i1 (U)]>) -> !handshake.control<>

// -----

handshake.func @simpleChannelWithDownExtra(%arg0: !handshake.channel<i32, [extra: i1]>) -> !handshake.control<>

// -----

handshake.func @simpleChannelWithUpExtra(%arg0: !handshake.channel<i32, [extra: i1 (U)]>) -> !handshake.control<>

// -----

handshake.func @simpleChannelWithDownAndUpExtra(%arg0: !handshake.channel<i32, [extraDown: i1, extraUp: i1 (U)]>) -> !handshake.control<>

// -----

handshake.func @validDataAndExtraTypes(%arg0: !handshake.channel<f32, [idx: i32]>) -> !handshake.control<>
