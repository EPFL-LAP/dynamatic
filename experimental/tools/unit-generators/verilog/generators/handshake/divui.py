from generators.support.delay_buffer import generate_delay_buffer
from generators.handshake.join import generate_join
def generate_divui(name, params):

    bitwidth = params["bitwidth"]
    latency = params["latency"]
    extra_signals = params.get("extra_signals", None)
    assert(latency == 36)

    join_name = name + "_join"
    join = generate_join(join_name, {"size": 2})

    delay_buffer_name = name + "_delay_buffer"
    delay_buffer = generate_delay_buffer(delay_buffer_name, { "size": 35 })

    divui_body = f"""
// Module of divui
module {name}(
  // inputs
  input  clk,
  input  rst,
  input  [{bitwidth} - 1 : 0] lhs,
  input  lhs_valid,
  input  [{bitwidth} - 1 : 0] rhs,
  input  rhs_valid,
  input  result_ready,
  // outputs
  output [{bitwidth} - 1 : 0] result,
  output result_valid,
  output lhs_ready,
  output rhs_ready
);

  wire join_valid;

  // Instantiate the join node
  {join_name} join_inputs (
    .ins_valid  ({{rhs_valid, lhs_valid}}),
    .outs_ready (result_ready             ),
    .ins_ready  ({{rhs_ready, lhs_ready}}  ),
    .outs_valid (join_valid             )
  );

  array_RAM_udiv_32ns_32ns_32_36_1 #(
    .ID(1),
    .NUM_STAGE(36),
    .din0_TYPE(32),
    .din1_TYPE(32),
    .dout_TYPE(32)
  ) array_RAM_udiv_32ns_32ns_32_36_1_U1 (
    .clk(clk),
    .reset(rst),
    .ce(result_ready),
    .din0(lhs),
    .din1(rhs),
    .dout(result)
  );

  {delay_buffer_name} buff (
    .clk(clk),
    .rst(rst),
    .valid_in(join_valid),
    .ready_in(result_ready),
    .valid_out(result_valid)
  );


endmodule
"""



    return join + delay_buffer + divui_body
