from generators.support.arith_binary import generate_arith_binary
from generators.support.unary import generate_unary


def generate_flopoco_ip_wrapper(name,
                                handshake_op,
                                core_unit,
                                latency,
                                is_double,
                                internal_delay,
                                extra_signals):
    """
    Generates boilerplate VHDL entity and handshaking code for arithmetic units which
    are wrappers for IP arithmetic cores, such as floating point units, or integer division.

    For flopoco units, it adds the necessary conversions, and instantiates the unit from
    flopoco_ip_cores, with a number of clock enable signals based on the operation's latency
    (The op latency must be provided when calling the backed, no latencies are hardcoded).

    For vitis ips, the unit is instantiated from vitis_ip_wrappers. These are only wrappers,
    and so these units can only be used if the propetiary IP cores are also present in simulation.
    The backend does not handle this.

    Args:
        name: Unique name based on MLIR op name (e.g. adder0).
        handshake_op: What kind of handshake op this RTL entity corresponds to. Only used in comments.
        core_unit: What is the name of this unit in the flopoco ip cores
        latency: Operation latency, used to add clock enables to flopoco units.
        is_double: Flag to specify either 32 or 64 bit units.
        internal_delay: internal delay of the unit, currently used to identify the desired flopoco unit.
        bitwidth: Unit bitwidth (if input/output are the same).
        extra_signals: Extra signals on input/output channels, from IR.


    Returns:
        VHDL code as a string.
    """

    bitwidth = 64 if is_double else 32

    signals = f"""
  --intermediate input signals for float conversion
  signal ip_lhs, ip_rhs : std_logic_vector({bitwidth + 2} - 1 downto 0);

  --intermidiate output signal(s) for float conversion
  signal ip_result : std_logic_vector({bitwidth + 2} - 1 downto 0);
  """

    clock_enables = "\n".join(
        [f"        ce_{i} => valid_buffer_ready," for i in range(1, latency + 1)]
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

    return generate_arith_binary(
        name=name,
        handshake_op=handshake_op,
        bitwidth=bitwidth,
        signals=signals,
        body=body,
        latency=latency,
        extra_signals=extra_signals
    )


def generate_vivado_ip_wrapper(name,
                               handshake_op,
                               latency,
                               extra_signals):

    body = f"""
  {handshake_op}_vitis_hls_wrapper_U1 : entity work.{handshake_op}_vitis_hls_wrapper
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
        handshake_op=handshake_op,
        bitwidth=32,
        body=body,
        latency=latency,
        extra_signals=extra_signals
    )


def generate_vitis_hls_unary_ip_wrapper(name,
                                        handshake_op,
                                        bitwidth,
                                        latency,
                                        ip_latency,
                                        extra_signals):
    """
    Generates the handshake unit of a unary arithmetic operation implemented
    with an IP core generated with Vitis HLS. The IP core is instantiated
    through its wrapper `{handshake_op}_vitis_hls_wrapper` (see
    data/vhdl/support/{handshake_op}_vitis_hls.vhd), whose pipeline advances
    when its clock enable is high.

    Args:
        name: Unique name based on MLIR op name (e.g. sqrtf0).
        handshake_op: Name of the handshake operation (e.g. sqrtf).
        bitwidth: Bitwidth of the input and the output.
        latency: Operation latency, from the timing model.
        ip_latency: Latency of the IP core, which the timing model must match.
        extra_signals: Extra signals on input/output channels, from IR.

    Returns:
        VHDL code as a string.
    """
    if latency != ip_latency:
        raise ValueError(
            f"{handshake_op}: the latency from the timing model ({latency}) "
            f"does not match the latency of the IP core ({ip_latency})")

    body = f"""
  {handshake_op}_vitis_hls_wrapper_U1 : entity work.{handshake_op}_vitis_hls_wrapper
    port map(
      clk   => clk,
      reset => rst,
      ce    => valid_buffer_ready,
      din0  => ins,
      dout  => outs
    );
    """

    return generate_unary(
        name=name,
        handshake_op=handshake_op,
        bitwidth=bitwidth,
        body=body,
        latency=latency,
        extra_signals=extra_signals
    )
