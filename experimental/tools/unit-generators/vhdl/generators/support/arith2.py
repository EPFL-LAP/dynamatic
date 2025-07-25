from generators.handshake.join import generate_join
from generators.handshake.buffers.one_slot_break_dv import generate_one_slot_break_dv
from generators.support.delay_buffer import generate_delay_buffer
from generators.support.signal_manager import generate_arith2_signal_manager
from generators.support.utils import ExtraSignals


def generate_arith2(
    name: str,
    modType: str,
    extra_signals: ExtraSignals,
    body: str,
    signals: str = "",
    dependencies: str = "",
    latency: int = 0,
    bitwidth: int = None,
    lhs_bitwidth: int = None,
    rhs_bitwidth: int = None,
    output_bitwidth: int = None,
):

    if bitwidth is not None:
        if lhs_bitwidth is not None or \
                rhs_bitwidth is not None or \
                output_bitwidth is not None:
            raise RuntimeError("If bitwidth is specified, lhs, rhs, and output bitwidth must not be specified")

        input_bitwidth = bitwidth
        output_bitwidth = bitwidth

    elif input_bitwidth is None or output_bitwidth is None:
        raise RuntimeError("If bitwidth is not specified, lhs, rhs, and output bitwidth must all be specified")

    def generate_inner(name): return _generate_arith2(
        name,
        modType,
        lhs_bitwidth,
        rhs_bitwidth,
        output_bitwidth,
        signals,
        body,
        latency,
        dependencies
    )

    def generate(): return generate_inner(name)

    if extra_signals:
        return generate_arith2_signal_manager(
            name,
            bitwidth,
            extra_signals,
            generate_inner,
            latency
        )
    else:
        return generate()

# Generate the actual unit


def _generate_arith2(
        name,
        modType,
        lhs_bitwidth,
        rhs_bitwidth,
        output_bitwidth,
        signals,
        body,
        latency,
        dependencies,
):

    join_name = f"{name}_join"
    dependencies += generate_join(join_name, {"size": 2})

    # all 2 input arithmetic units have the same entity
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.float_pkg.all;

-- Entity of {modType}
entity {name} is
  port(
    -- inputs
    clk: in std_logic;
    rst: in std_logic;
    lhs: in std_logic_vector({lhs_bitwidth} - 1 downto 0);
    lhs_valid: in std_logic;
    rhs: in std_logic_vector({rhs_bitwidth} - 1 downto 0);
    rhs_valid: in std_logic;
    result_ready: in std_logic;
    -- outputs
    result : out std_loigc_vector({output_bitwidth} - 1 downto 0);
    result_valid: out std_logic;
    lhs_ready: out std_logic;
    rhs_ready: out std_logic
  );
end entity;
"""
    signals = signals.lstrip()
    body = body.lstrip()

    # but the architecture differs depending
    # on the latency

    # Handshaking handled by a join
    if latency == 0:
        architecture = f"""
-- Architecture of {modType}
architecture arch of {name} is
  {signals}
begin
  join_inputs: entity work.{join_name}(arch)
  join_inputs : entity work.{join_name}(arch)
    port map(
      -- input channel from "lhs"
      ins_valid(0) => lhs_valid,
      ins_ready(0) => lhs_ready,
      -- input channel from "rhs"
      ins_valid(1) => rhs_valid,
      ins_ready(1) => rhs_ready
      -- output channel to "result"
      outs_valid   => result_valid,
      outs_ready   => result_ready,
    );

  {body}

end architecture;
"""
    # with latency 1,
    # we need an one_slot_break_dv to store the valid
    elif latency == 1:
        one_slot_break_dv_name = f"{name}_one_slot_break_dv"
        dependencies += generate_one_slot_break_dv(one_slot_break_dv_name, {"bitwidth": 0})

        architecture = f"""
-- Architecture of {modType}
architecture arch of {name} is
	signal join_valid, one_slot_break_dv_valid, one_slot_break_dv_ready : std_logic;
  {signals}
begin

  join_inputs : entity work.{join_name}(arch)
    port map(
      -- input channel from "lhs"
      ins_valid(0) => lhs_valid,
      ins_ready(0) => lhs_ready,
      -- input channel from "rhs"
      ins_valid(1) => rhs_valid,
      ins_ready(1) => rhs_ready
      -- output channel to one_slot_break_dv
      outs_valid   => join_valid,
      outs_ready   => one_slot_break_dv_ready,
    );

  one_slot_break_dv : entity work.{one_slot_break_dv_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      -- input channel from join
      ins_valid  => join_valid,
      ins_ready  => one_slot_break_dv_ready
      -- output channel to "result"
      outs_ready => result_ready,
      outs_valid => result_valid,
    );

  {body}

end architecture;
"""
    # with latency >1,
    # we need a delay buffer to propagate the valids
    # with the same latency as the unit
    # and we need an one_slot_break_dv to store the final valid
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
  signal join_valid                         : std_logic;
  signal buff_valid, one_slot_break_dv_valid, one_slot_break_dv_ready : std_logic;
  {signals}
begin
  join_inputs : entity work.{join_name}(arch)
    port map(
      -- input channel from "lhs"
      ins_valid(0) => lhs_valid,
      ins_ready(0) => lhs_ready,
      -- input channel from "rhs"
      ins_valid(1) => rhs_valid,
      ins_ready(1) => rhs_ready,
      -- output channel to buffer, using one_slot_break_dv ready
      outs_valid   => join_valid,
      outs_ready   => one_slot_break_dv_ready
    );

  buff : entity work.{buff_name}(arch)
    port map(
      clk,
      rst,
      -- input channel from join
      valid_in  => join_valid,
      -- output channel to one_slot_break_dv
      valid_out => buff_valid,
      read_in   => one_slot_break_dv_ready
    );

  one_slot_break_dv : entity work.{one_slot_break_dv_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      -- input channel from buffer
      ins_valid  => buff_valid,
      ins_ready  => one_slot_break_dv_ready,
      -- output channel to "result"
      outs_ready => result_ready,
      outs_valid => result_valid
    );

  {body}

end architecture;
"""

    return dependencies + entity + architecture
