def generate_fork(name, params):
    # Number of output ports
    size = params["size"]

    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if bitwidth == 0:
        return _generate_fork_dataless(name, size)
    else:
        return _generate_fork(name, size, bitwidth)


def _generate_fork(name, size, bitwidth):

    fork_dataless_name = name + "_fork_dataless"

    fork_dataless = _generate_fork_dataless(fork_dataless_name, size)
    fork = f"""
// Module of fork
module {name}(
  input  clk,
  input  rst,
  // Input Channel
  input  [{bitwidth} - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Output Channel
  output [{size} * ({bitwidth}) - 1 : 0] outs,
  output [{size} - 1 : 0] outs_valid,
  input  [{size} - 1 : 0] outs_ready
);

  {fork_dataless_name} control (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (ins_valid  ),
    .ins_ready  (ins_ready  ),
    .outs_valid (outs_valid ),
    .outs_ready (outs_ready )
  );

  // Broadcast the input data to all output channels
  assign outs = {{{size}{{ins}}}};

endmodule
"""

    return fork_dataless + fork


def _generate_fork_dataless(name, size):

    eager_fork_register_block_name = name + "_eager_fork_register_block"
    eager_fork_register_block = f"""
// Module of eager_fork_register_block
module {eager_fork_register_block_name} (
  input clk,
  input rst,

  input ins_valid,  // Input Channel
  input outs_ready,  // Output Channel
  input backpressure,

  output outs_valid,
  output blockStop
);
  reg transmitValue = 1;
  wire keepValue;

  assign keepValue = ~outs_ready & transmitValue;

  always @(posedge clk) begin
    if (rst) begin
      transmitValue <= 1;
    end else begin
      transmitValue <= keepValue | ~backpressure;
    end
  end

  assign outs_valid = transmitValue & ins_valid;
  assign blockStop = keepValue;

endmodule
"""

    fork_dataless = f"""
// Module of fork_dataless

module {name}(
  input  clk,
  input  rst,
  // Input Channel
  input  ins_valid,
  output ins_ready,
  // Output Channel
  output [{size} - 1 : 0] outs_valid,
  input  [{size} - 1: 0] outs_ready
);
  // Internal Signal Definition
  wire [{size} - 1 : 0] blockStopArray;
  wire anyBlockStop;
  wire backpressure;

  assign anyBlockStop = |blockStopArray;

  assign ins_ready = ~anyBlockStop;
  assign backpressure = ins_valid & anyBlockStop;

  // Define generate variable
  genvar gen;

  generate
    for (gen = {size} - 1; gen >= 0; gen = gen - 1) begin: regBlock
      {eager_fork_register_block_name} regblock (
        .clk           (clk                ),
        .rst           (rst                ),
        .ins_valid     (ins_valid          ),
        .outs_ready    (outs_ready[gen]    ),
        .backpressure  (backpressure       ),
        .outs_valid    (outs_valid[gen]    ),
        .blockStop     (blockStopArray[gen])
      );
    end
  endgenerate


endmodule

"""
    return eager_fork_register_block + fork_dataless
    