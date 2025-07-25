from generators.support.signal_manager import generate_unary_signal_manager
from generators.support.utils import ExtraSignals
from generators.handshake.buffers.one_slot_break_dv import generate_one_slot_break_dv
from generators.support.delay_buffer import generate_delay_buffer

def generate_arith1(
    name: str,
    modType: str,
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

    If latency = 1:
      Handshaking signals are passed through a 1-slot BREAK_DV buffer.

    If latency > 1:
      Handshaking signals are passed through a shift register with
      latency - 1 slots, and then a 1-slot BREAK_DV buffer.

    Args:
        name: Unique name based on MLIR op name (e.g. buffer0).
        modType: More specific name, used in comments only.
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

    def generate_inner(name): return _generate_arith1(
        name,
        modType,
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


def _generate_arith1(
        name,
        modType,
        input_bitwidth,
        output_bitwidth,
        signals,
        body,
        dependencies,
        latency
):

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of {modType}
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

    if latency == 0:
        architecture = f"""
-- Architecture of {modType}
architecture arch of {name} is
  {signals}
begin

  {body}

  -- combinatorial unit forwards handshaking
  outs_valid <= ins_valid;
  ins_ready <= outs_ready;

end architecture;
"""
    elif latency == 1:
        one_slot_break_dv_name = f"{name}_one_slot_break_dv"

        dependencies += generate_one_slot_break_dv(one_slot_break_dv_name, {"bitwidth": 0})

        architecture = f"""
-- Architecture of {modType}
architecture arch of {name} is
  {signals}
begin
  one_slot_break_dv : entity work.{one_slot_break_dv_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      -- input channel from "ins"
      ins_valid  => ins_valid,
      ins_ready  => one_slot_break_dv_ready,
      -- output channel to "outs"
      outs_ready => outs_ready,
      outs_valid => outs_valid
  );

  {body}

end architecture;
"""
    else:
        one_slot_break_dv_name = f"{name}_one_slot_break_dv"
        buff_name = f"{name}_buff"

        dependencies += generate_one_slot_break_dv(one_slot_break_dv_name, {"bitwidth": 0})
        dependencies += generate_delay_buffer(
            buff_name,
            {"slots": latency - 1})

        architecture = f"""
-- Architecture of {modType}
architecture arch of {name} is
  {signals}
begin
  buff : entity work.{buff_name}(arch)
    port map(
      clk,
      rst,
      -- input channel from "ins"
      -- (without ready)
      valid_in  => ins_valid,
      -- output channel to one_slot_break_dv
      valid_out => buff_valid,
      ready_in   => one_slot_break_dv_ready
    );


  one_slot_break_dv : entity work.{one_slot_break_dv_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      -- input channel from buffer
      ins_valid  => buff_valid,
      ins_ready  => one_slot_break_dv_ready,
      -- output channel to "outs"
      outs_ready => outs_ready,
      outs_valid => outs_valid
  );

  -- input channel from "ins" to one_slot_break_dv
  -- (ready only)
  ins_ready <= one_slot_break_dv_ready;

  {body}

end architecture;
"""

    return dependencies + entity + architecture
