// RUN: dynamatic-opt %s --split-input-file --verify-diagnostics

handshake.func @simpleControl(%arg0: <>) -> (<>)

// -----

handshake.func @simpleChannel(%arg0: <i32>) -> (<>)

// -----

handshake.func @simpleControlWithDownExtra(%arg0: <[extra: i1]>) -> (<>)

// -----

handshake.func @simpleControlWithUpExtra(%arg0: <[extra: i1 (U)]>) -> (<>)

// -----

handshake.func @simpleControlWithDownAndUpExtra(%arg0: <[extraDown: i1, extraUp: i1 (U)]>) -> (<>)

// -----

handshake.func @simpleChannelWithDownExtra(%arg0: <i32, [extra: i1]>) -> (<>)

// -----

handshake.func @simpleChannelWithUpExtra(%arg0: <i32, [extra: i1 (U)]>) -> (<>)

// -----

handshake.func @simpleChannelWithDownAndUpExtra(%arg0: <i32, [extraDown: i1, extraUp: i1 (U)]>) -> (<>)

// -----

handshake.func @validDataAndExtraTypes(%arg0: <f32, [idx: i32]>) -> (<>)
