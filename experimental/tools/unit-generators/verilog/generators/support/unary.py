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
    Generates boilerplate Verilog entity and handshaking code for unary units
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
        Verilog code as a string.
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

    # if no signal manager,
    # the unit uses the top level name
    def generate(): return generate_inner(name)

    if extra_signals:
        return generate_unary_signal_manager(
            name,
            extra_signals,
            generate_inner,
            latency,
            None,
            input_bitwidth,
            output_bitwidth
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
// Module of {handshake_op}
module {name} (
    input wire clk,
    input wire rst,
    // input channel
    input wire [{input_bitwidth - 1}:0] ins,
    input wire ins_valid,
    output wire ins_ready,
    // output channel
    output wire [{output_bitwidth - 1}:0] outs,
    output wire outs_valid,
    input wire outs_ready
  );
"""
    signals = signals.lstrip()
    body = body.lstrip()

    # but the architecture differs depending
    # on the latency

    # Handshaking is directly forwarded
    if latency == 0:
        architecture = f"""
  // Signals
  {signals}

  // Body
  {body}

  // combinatorial unit forwards handshaking
  assign outs_valid = ins_valid;
  assign ins_ready = outs_ready;

endmodule
"""
    # otherwise, we need a buffer to propagate the valid
    else:
        valid_buffer_name = f"{name}_valid_buffer"
        dependencies += generate_valid_propagation_buffer(valid_buffer_name, latency)

        architecture = f"""
  // Signals
  {signals}
  wire valid_buffer_ready;
  {valid_buffer_name} valid_buffer (
    .clk        (clk),
    .rst        (rst),
    // input channel from "ins"
    .ins_valid  (ins_valid),
    .ins_ready  (valid_buffer_ready),
    // output channel to "outs"
    .outs_valid (outs_valid),
    .outs_ready (outs_ready)
  );

  // expose to allow use as a clock enable signal
  assign ins_ready = valid_buffer_ready;

  {body}

end architecture;
"""

    return dependencies + entity + architecture
