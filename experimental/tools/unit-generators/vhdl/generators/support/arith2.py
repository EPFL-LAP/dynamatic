from generators.handshake.join import generate_join
from generators.support.signal_manager import generate_arith2_signal_manager
from generators.support.utils import ExtraSignals
from generators.handshake.buffer import generate_valid_propagation_buffer

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

        lhs_bitwidth = bitwidth
        rhs_bitwidth = bitwidth
        output_bitwidth = bitwidth

    elif lhs_bitwidth is None or rhs_bitwidth is None or output_bitwidth is None:
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
    clk: in std_logic;
    rst: in std_logic;
    -- input channel lhs
    lhs: in std_logic_vector({lhs_bitwidth} - 1 downto 0);
    lhs_valid: in std_logic;
    lhs_ready: out std_logic;
    -- input channel rhs
    rhs: in std_logic_vector({rhs_bitwidth} - 1 downto 0);
    rhs_valid: in std_logic;
    rhs_ready: out std_logic;
    -- output channel result
    result : out std_logic_vector({output_bitwidth} - 1 downto 0);
    result_valid: out std_logic;
    result_ready: in std_logic
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
  join_inputs : entity work.{join_name}(arch)
    port map(
      -- input valids
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      -- input readys
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready,
      -- output channel to "result"
      outs_valid   => result_valid,
      outs_ready   => result_ready
    );

  {body}

end architecture;
"""
    # otherwise, we need a buffer to propagate the valid
    else:
        valid_buffer_name = f"{name}_valid_buffer"
        dependencies += generate_valid_propagation_buffer(valid_buffer_name, latency)

        architecture = f"""
-- Architecture of {modType}
architecture arch of {name} is
	signal join_valid, valid_buffer_ready : std_logic;
  {signals}
begin

  join_inputs : entity work.{join_name}(arch)
    port map(
      -- input valids
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      -- input readys
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready,
      -- output channel to valid_buffer
      outs_valid   => join_valid,
      outs_ready   => valid_buffer_ready
    );

  valid_buffer : entity work.{valid_buffer_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      -- input channel from join
      ins_valid  => join_valid,
      ins_ready  => valid_buffer_ready,
      -- output channel to "result"
      outs_ready => result_ready,
      outs_valid => result_valid
    );

  {body}

end architecture;
"""

    return dependencies + entity + architecture
