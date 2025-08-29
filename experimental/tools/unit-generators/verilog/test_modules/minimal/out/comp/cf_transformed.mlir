module {
  func.func @minimal(%arg0: i32 {handshake.arg_name = "x"}) -> i32 {
    return {handshake.name = "return0"} %arg0 : i32
  }
}

