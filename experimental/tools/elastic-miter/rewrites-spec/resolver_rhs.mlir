module {
  handshake.func @resolver_rhs(%actual: !handshake.channel<i1>, %generated: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["Actual", "Generated"], resNames = ["Confirm"]} {
    %confirm = spec_v2_resolver %actual, %generated {handshake.bb = 1 : ui32, handshake.name = "resolver"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %confirm : <i1>
  }
}
