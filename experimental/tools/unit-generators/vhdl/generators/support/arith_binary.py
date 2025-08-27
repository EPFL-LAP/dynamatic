from generators.handshake.join import generate_join
from generators.support.signal_manager import generate_arith_binary_signal_manager
from generators.support.utils import ExtraSignals
from generators.handshake.buffer import generate_valid_propagation_buffer

def generate_arith_binary(
    name: str,
    handshake_op: str,
    extra_signals: ExtraSignals,
    body: str,
    signals: str = "",
    dependencies: str = "",
    bitwidth: int = None,
    input_bitwidth: int = None,
    output_bitwidth: int = None,
    latency: int = 0,
):
    """
    Generates boilerplate VHDL entity and handshaking code for two input arithmetic units
    (though it could be used for any operation with two data-carrying inputs 
    used to generate a data output, 
    but it depends where we draw the line of what "arithmetic" means)

    If latency = 0:
      Output ready is directly forwarded up from input ready.
      Output valid is directly forwarded down from input valid.
    Else:
      Handshaking signals are passed through a valid propagation buffer.
      Which is either a one slot break dv, or a shift register buffer

    Args:
        name: Unique name based on MLIR op name (e.g. adder0).
        handshake_op: What kind of handshake op this RTL entity corresponds to. Only used in comments.
        extra_signals: Extra signals on input/output channels, from IR.
        body: VHDL body of the unit, excluding handshaking.
        signals: Local signal declarations used in body.
        dependencies: Dependencies, excluding handshaking.
        bitwidth: Unit bitwidth (if input/output are the same).
        input_bitwidth: Input bitwidth, if input and output bitwidth differ, e.g. cmpf
        output_bitwidth: Output bitwidth, if input and output bitwidth differ, e.g. cmpf
        latency:

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


    # generate inner function takes a name parameter
    # since the top level name is used for the signal manager wrapper
    #
    # the signal manager wrapper will make a new name for the inner unit
    def generate_inner(name): return _generate_arith_binary(
        name,
        handshake_op,
        input_bitwidth,
        output_bitwidth,
        signals,
        body,
        latency,
        dependencies
    )


    # if no signal manager,
    # the unit uses the top level name
    def generate(): return generate_inner(name)

    if extra_signals:
        return generate_arith_binary_signal_manager(
            name,
            input_bitwidth,
            output_bitwidth,
            extra_signals,
            generate_inner,
            latency
        )
    else:
        return generate()

def _generate_arith_binary(
        name,
        handshake_op,
        input_bitwidth,
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

-- Entity of {handshake_op}
entity {name} is
  port(
    clk: in std_logic;
    rst: in std_logic;
    -- input channel lhs
    lhs: in std_logic_vector({input_bitwidth} - 1 downto 0);
    lhs_valid: in std_logic;
    lhs_ready: out std_logic;
    -- input channel rhs
    rhs: in std_logic_vector({input_bitwidth} - 1 downto 0);
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
-- Architecture of {handshake_op}
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
-- Architecture of {handshake_op}
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
