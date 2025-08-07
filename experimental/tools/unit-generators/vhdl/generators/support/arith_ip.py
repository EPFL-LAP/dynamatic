from generators.support.arith2 import generate_arith2


def generate_flopoco_ip_wrapper(name,
                                mod_type,
                                core_unit,
                                latency,
                                is_double,
                                internal_delay,
                                extra_signals):
    bitwidth = 64 if is_double else 32

    signals = f"""
  --intermediate input signals for float conversion
  signal ip_lhs, ip_rhs : std_logic_vector({bitwidth + 2} - 1 downto 0);

  --intermidiate output signal(s) for float conversion
  signal ip_result : std_logic_vector({bitwidth + 2} - 1 downto 0);
  """

    clock_enables = "\n".join(
        [f"        ce_{i} => one_slot_break_dv_ready," for i in range(1, latency + 1)]
    )
    clock_enables = clock_enables.lstrip()

    body = f"""
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

    return generate_arith2(
        name=name,
        modType=mod_type,
        bitwidth=bitwidth,
        signals=signals,
        body=body,
        latency=latency,
        extra_signals=extra_signals
    )


def generate_vivado_ip_wrapper(name,
                               mod_type,
                               latency,
                               extra_signals):

    body = f"""
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

    return generate_arith2(
        name=name,
        modType=mod_type,
        bitwidth=32,
        body=body,
        latency=latency,
        extra_signals=extra_signals
    )
