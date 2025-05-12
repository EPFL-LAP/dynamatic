module {
  func.func @matching(%arg0: i32 {handshake.arg_name = "num_edges"}) -> i32 {
    return {handshake.name = "return0"} %arg0 : i32
  }
}

