from generators.support.arith_binary import generate_arith_binary


def generate_divui(name, params):

    latency = params["latency"]
    # FIXME: The latency of the long division depends on the bitwidth, but it
    # was hardcoded to 35 in the performance model.
    #
    # Here, we use the actual latency of the Vitis IP (see below for the
    # formula). This wouldn't change most of our benchmarks in the integration
    # tests (they use 32-bit division in any case), but we need to remember to
    # change the timing model to reflect this.
    bitwidth = params["bitwidth"]
    # The long division algorithm in the Vitis IP needs:
    # 2 cycles from the input/output regs
    # 1 cycle from the input reg of the division unit
    # BITWIDTH number for the actual division.
    latency = bitwidth + 2 + 1

    extra_signals = params.get("extra_signals", None)

    body = f"""
    divui_vitis_hls_wrapper_U1 : entity work.divui_vitis_hls_wrapper
    generic map({bitwidth}, {bitwidth}, {bitwidth})
    port map(
      clk   => clk,
      reset => rst,
      ce    => valid_buffer_ready,
      din0  => lhs,
      din1  => rhs,
      dout  => result
    );
    """

    return generate_arith_binary(
        name=name,
        handshake_op="divui",
        bitwidth=bitwidth,
        body=body,
        latency=latency,
        extra_signals=extra_signals
    )
