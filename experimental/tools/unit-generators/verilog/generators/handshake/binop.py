from generators.handshake.join import generate_join
from generators.support.utils import ExtraSignals
from generators.handshake.dataless.dataless_oehb import generate_dataless_oehb
from generators.support.delay_buffer import generate_delay_buffer

def generate_binop(
  name:str,
  handshake_op:str,
  extra_signals:ExtraSignals,
  op_body:str,
  dependencies:str = "",
  latency:int = 0,
  bitwidth: int = None,
  lhs_bitwidth: int = None,
  rhs_bitwidth: int = None,
  output_bitwidth: int = None,
):

  if bitwidth is not None:
    if (lhs_bitwidth is not None) or (rhs_bitwidth is not None) or (output_bitwidth is not None):
      raise RuntimeError("If bitwidth is specified, lhs, rhs, and output bitwidth must not be specified")
    lhs_bitwidth = bitwidth
    rhs_bitwidth = bitwidth
    output_bitwidth = bitwidth
  else:
    if (lhs_bitwidth is None) or (rhs_bitwidth is None) or (output_bitwidth is None):
      raise RuntimeError("If bitwidth is not specified, lhs, rhs, and output bitwidth must all be specified")

  def generate_inner(name):return _generate_arith2(
    name,
    handshake_op,
    lhs_bitwidth,
    rhs_bitwidth,
    output_bitwidth,
    op_body,
    latency,
    dependencies)

  if(extra_signals):
    raise RuntimeError("No support for extra signals added yet.")#TODO (Pass generate_inner)
  else:
    return generate_inner(name)


def _generate_arith2(
        name,
        handshake_op,
        lhs_bitwidth,
        rhs_bitwidth,
        output_bitwidth,
        op_body,
        latency,
        dependencies):

  header = "`timescale 1ns/1ps\n"

  # all 2 input arithmetic units have the same entity
  entity = f"""
//{handshake_op} Module
module {name}(
  // inputs
  input  clk,
  input  rst,
  input  [{lhs_bitwidth} - 1 : 0] lhs,
  input  lhs_valid,
  input  [{rhs_bitwidth} - 1 : 0] rhs,
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

  if latency == 0: #--------------------------------------------------------------------------------------------------------------------------------
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
  elif latency == 1: #--------------------------------------------------------------------------------------------------------------------------------
    # with latency 1,
    # we need an one_slot_break_dv to store the valid
    join_name = f"{name}_join"
    dependencies += generate_join(join_name, {"size": 2})
    oehb_name = f"{name}_oehb"
    dependencies += generate_dataless_oehb(oehb_name, {"bitwidth": 0, "latency": latency})

    architecture = f"""
  wire join_valid;
  wire oehb_ready;

  // Instantiate the join node
  {join_name} #(
    .SIZE(2)
  ) join_inputs (
    .ins_valid  ({{rhs_valid, lhs_valid}}),
    .outs_ready (oehb_ready             ),
    .ins_ready  ({{rhs_ready, lhs_ready}}  ),
    .outs_valid (join_valid             )
  );

  {oehb_name} oehb_inst (
    .clk(clk),
    .rst(rst),
    .ins_valid(join_valid),
    .ins_ready(oehb_ready),
    .outs_valid(result_valid),
    .outs_ready(result_ready)
  );

{op_body}

endmodule
"""
  else: #--------------------------------------------------------------------------------------------------------------------------------
    # with latency >1,
    # we need a delay buffer to propagate the valids
    # with the same latency as the unit
    # and we need an one_slot_break_dv to store the final valid
    
    join_name = f"{name}_join"
    dependencies += generate_join(join_name, {"size": 2})
    delay_buffer_name = f"{name}_delay_buffer"
    dependencies += generate_delay_buffer(delay_buffer_name, {"bitwidth": 0, "latency": latency - 1})
    oehb_name = f"{name}_oehb"
    dependencies += generate_dataless_oehb(oehb_name, {"bitwidth": 0, "latency": latency})

    architecture = f"""
  wire join_valid;
  wire oehb_ready;
  wire buff_valid;

  // Instantiate the join node
  {join_name} #(
    .SIZE(2)
  ) join_inputs (
    .ins_valid  ({{rhs_valid, lhs_valid}}),
    .outs_ready (oehb_ready             ),
    .ins_ready  ({{rhs_ready, lhs_ready}}  ),
    .outs_valid (join_valid             )
  );

  {delay_buffer_name} #(
    .SIZE( {latency} - 1)
  ) buff (
    .clk(clk),
    .rst(rst),
    .valid_in(join_valid),
    .ready_in(oehb_ready),
    .valid_out(buff_valid)
  );

  {oehb_name} oehb_inst (
    .clk(clk),
    .rst(rst),
    .ins_valid(buff_valid),
    .ins_ready(oehb_ready),
    .outs_valid(result_valid),
    .outs_ready(result_ready)
  );

{op_body}

endmodule
"""
  #--------------------------------------------------------------------------------------------------------------------------------
  return header + dependencies + entity + architecture