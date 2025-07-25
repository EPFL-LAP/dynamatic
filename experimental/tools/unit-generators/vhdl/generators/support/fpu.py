from generators.support.arith2 import generate_arith2


def generate_fpu_wrapper(name, params, core_unit, mod_type):
    impl = params["fpu_impl"]

    internal_delay = params["internal_delay"]
    latency = params["latency"]

    # only used for flopoco
    is_double = params.get("is_double", None)

    if impl == "flopoco":
        if is_double is None:
            raise ValueError(f"is_double was missing for generating a flopoco {mod_type}")

        bitwidth = 64 if is_double else 32
        signals = _get_flopoco_signals(bitwidth)
        body = _get_flopoco_body(
            core_unit,
            bitwidth,
            internal_delay,
            latency)
    elif impl == "vivado":
        signals = _get_vivado_signals()
        body = _get_vivado_body(mod_type)
        bitwidth = 32
    else:
        raise ValueError(f"Invalid FPU implementation: {impl}")

    return generate_arith2(
        name=name,
        modType=mod_type,
        bitwidth=bitwidth,
        signals=signals,
        body=body,
        latency=latency,
        extra_signals=params.get("extra_signals", None)
    )


def _get_flopoco_signals(bitwidth):
    return f"""
  --intermediate input signals for float conversion
  signal ip_lhs, ip_rhs : std_logic_vector({bitwidth + 2} - 1 downto 0);

  --intermidiate output signal(s) for float conversion
  signal ip_result : std_logic_vector({bitwidth + 2} - 1 downto 0);
  """


def _get_flopoco_body(core_unit, bitwidth, internal_delay, latency):
    clock_enables = "\n".join(
        [f"        ce_{i} => one_slot_break_dv_ready," for i in range(1, latency + 1)]
    )
    clock_enables = clock_enables.lstrip()

    return f"""
    ieee2nfloat_0: entity work.InputIEEE_{bitwidth}bit(arch)
            port map (
                --input
                X =>lhs,
                --output
                R => ip_lhs
            );

    ieee2nfloat_1: entity work.InputIEEE_{bitwidth}bit(arch)
            port map (
                --input
                X => rhs,
                --output
                R => ip_rhs
            );

    operator : entity work.{core_unit}_{bitwidth}_{internal_delay}(arch)
    port map (
        clk   => clk,
        {clock_enables}
        X     => ip_lhs,
        Y     => ip_rhs,
        R     => ip_result
    );

    nfloat2ieee : entity work.OutputIEEE_{bitwidth}bit(arch)
        port map (
            --input
            X => ip_result,
            --ouput
            R => result
        );
    """


def _get_vivado_signals():
    return ""


def _get_vivado_body(mod_type):
    return f"""
  {mod_type}_vitis_hls_wrapper_U1 : entity work.{mod_type}_vitis_hls_wrapper
    port map(
      clk   => clk,
      reset => rst,
      ce    => one_slot_break_dv_ready,
      din0  => lhs,
      din1  => rhs,
      dout  => result
    );
"""
