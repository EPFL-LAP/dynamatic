from generators.handshake.join import generate_join
from generators.support.utils import ExtraSignals
from generators.support.delay_buffer import generate_delay_buffer
from generators.handshake.buffer import generate_valid_propagation_buffer


def generate_arith_binary(
    name: str,
    handshake_op: str,
    extra_signals: ExtraSignals,
    op_body: str,
    dependencies: str = "",
    latency: int = 0,
    bitwidth: int = None,
    input_bitwidth: int = None,
    output_bitwidth: int = None,
):

    if bitwidth is not None:
        if (input_bitwidth is not None) or (output_bitwidth is not None):
            raise RuntimeError(
                "If bitwidth is specified, input and output bitwidth must not be specified")
        input_bitwidth = bitwidth
        output_bitwidth = bitwidth
    else:
        if (input_bitwidth is None) or (output_bitwidth is None):
            raise RuntimeError(
                "If bitwidth is not specified, input and output bitwidth must all be specified")

    def generate_inner(name): return _generate_arith_binary(
        name,
        handshake_op,
        input_bitwidth,
        output_bitwidth,
        op_body,
        latency,
        dependencies)


    return generate_inner(name)


def _generate_arith_binary(
        name,
        handshake_op,
        input_bitwidth,
        output_bitwidth,
        op_body,
        latency,
        dependencies):

    # all 2 input arithmetic units have the same entity
    entity = f"""
// Module of {handshake_op}
module {name}(
  // inputs
  input  clk,
  input  rst,
  input  [{input_bitwidth} - 1 : 0] lhs,
  input  lhs_valid,
  input  [{input_bitwidth} - 1 : 0] rhs,
  input  rhs_valid,
  input  result_ready,
  // outputs
  output [{output_bitwidth} - 1 : 0] result,
  output result_valid,
  output lhs_ready,
  output rhs_ready
);
"""

    # but the architecture differs depending
    # on the latency

    # Handshaking handled by a join

    if latency == 0:  # --------------------------------------------------------------------------------------------------------------------------------
        join_name = f"{name}_join"
        dependencies += generate_join(join_name, {"size": 2})
        architecture = f"""
  // Instantiate the join node
  {join_name} join_inputs (
    .ins_valid  ({{rhs_valid, lhs_valid}}),
    .outs_ready (result_ready             ),
    .ins_ready  ({{rhs_ready, lhs_ready}}  ),
    .outs_valid (result_valid             )
  );

{op_body}

endmodule
"""
    else:  # --------------------------------------------------------------------------------------------------------------------------------
        # with latency >= 1

        join_name = f"{name}_join"
        dependencies += generate_join(join_name, {"size": 2})
        valid_buffer_name = f"{name}_valid_buffer"
        dependencies += generate_valid_propagation_buffer(
            valid_buffer_name, latency)

        architecture = f"""
  wire join_valid;
  wire valid_buffer_ready;

  // Instantiate the join node
  {join_name} join_inputs (
    .ins_valid  ({{rhs_valid, lhs_valid}}),
    .outs_ready (valid_buffer_ready             ),
    .ins_ready  ({{rhs_ready, lhs_ready}}  ),
    .outs_valid (join_valid             )
  );

  // valid buffer
  {valid_buffer_name} valid_buffer (
    .clk(clk),
    .rst(rst),
    .ins_valid(join_valid),
    .ins_ready(valid_buffer_ready),
    .outs_valid(result_valid),
    .outs_ready(result_ready)
  );

{op_body}

endmodule
"""
    # --------------------------------------------------------------------------------------------------------------------------------
    return dependencies + entity + architecture
