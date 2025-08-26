from generators.support.signal_manager import generate_unary_signal_manager
from generators.support.utils import ExtraSignals
from generators.support.utils import ExtraSignals
from generators.handshake.buffer import generate_valid_propagation_buffer


def generate_unary(
    name: str,
    handshake_op: str,
    body: str,
    extra_signals: ExtraSignals,
    signals: str = "",
    dependencies: str = "",
    latency: int = 0,
    bitwidth: int = None,
    input_bitwidth: int = None,
    output_bitwidth: int = None,
) -> str:
    """
    Generates boilerplate VHDL entity and handshaking code for unary units
    (units with one input and one output).

    If latency = 0:
      Output ready is directly forwarded up from input ready.
      Output valid is directly forwarded down from input valid.
    Else:
      Handshaking signals are passed through a valid propagation buffer.
      Which is either a one slot break dv, or a shift register buffer

    Args:
        name: Unique name based on MLIR op name (e.g. adder0).
        handshake_op: Which handshake operation this module corresponds to, used only in comments
        signals: Local signal declarations used in body.
        body: VHDL body of the unit, excluding handshaking.
        dependencies: Dependencies, excluding handshaking.
        latency:
        bitwidth: Unit bitwidth (if input/output are the same).
        input_bitwidth: Input bitwidth (used if asymmetric).
        output_bitwidth: Output bitwidth (used if asymmetric).
        extra_signals: Extra signals on input/output channels, from IR.


    Returns:
        VHDL code as a string.
    """
    if bitwidth is not None:
        if input_bitwidth is not None or output_bitwidth is not None:
            raise RuntimeError("If bitwidth is specified, input and output bitwidth must not be specified")

        input_bitwidth = bitwidth
        output_bitwidth = bitwidth

    elif input_bitwidth is None or output_bitwidth is None:
        raise RuntimeError("If bitwidth is not specified, both input and output bitwidth must be specified")

    def generate_inner(name): return _generate_unary(
        name,
        handshake_op,
        input_bitwidth,
        output_bitwidth,
        signals,
        body,
        dependencies,
        latency
    )

    def generate(): return generate_inner(name)

    if extra_signals:
        return generate_unary_signal_manager(
            name,
            input_bitwidth,
            output_bitwidth,
            extra_signals,
            generate_inner,
            latency
        )
    else:
        return generate()


def _generate_unary(
        name,
        handshake_op,
        input_bitwidth,
        output_bitwidth,
        signals,
        body,
        dependencies,
        latency
):

    # all unary units have the same entity
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.float_pkg.all;

-- Entity of {handshake_op}
entity {name} is
  port(
    clk: in std_logic;
    rst: in std_logic;
    -- input channel
    ins: in std_logic_vector({input_bitwidth} - 1 downto 0);
    ins_valid: in std_logic;
    ins_ready: out std_logic;
    -- output channel
    outs: out std_logic_vector({output_bitwidth} - 1 downto 0);
    outs_valid: out std_logic;
    outs_ready: in std_logic
  );
end entity;
"""
    signals = signals.lstrip()
    body = body.lstrip()

    # but the architecture differs depending
    # on the latency

    # Handshaking is directly forwarded
    if latency == 0:
        architecture = f"""
-- Architecture of {handshake_op}
architecture arch of {name} is
  {signals}
begin

  {body}

  -- combinatorial unit forwards handshaking
  outs_valid <= ins_valid;
  ins_ready <= outs_ready;

end architecture;
"""
    # otherwise, we need a buffer to propagate the valid
    else:
        valid_buffer_name = f"{name}_valid_buffer"
        dependencies += generate_valid_propagation_buffer(valid_buffer_name, latency)

        architecture = f"""
-- Architecture of {handshake_op}
architecture arch of {name} is
  {signals}
  signal valid_buffer_ready : std_logic;
begin
  valid_buffer : entity work.{valid_buffer_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      -- input channel from "ins"
      ins_valid  => ins_valid,
      ins_ready  => valid_buffer_ready,
      -- output channel to "outs"
      outs_valid => outs_valid,
      outs_ready => outs_ready
  );

  -- expose to allow use as a clock enable signal
  ins_ready <= valid_buffer_ready;

  {body}

end architecture;
"""

    return dependencies + entity + architecture
