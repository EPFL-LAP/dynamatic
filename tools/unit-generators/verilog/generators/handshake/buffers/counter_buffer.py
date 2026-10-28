def generate_counter_buffer(name, params):
    bitwidth = params["bitwidth"]
    dv_latency = int(params["dv_latency"])

    if bitwidth == 0:
        return _generate_counter_buffer_dataless(name, dv_latency)
    else:
        return _generate_counter_buffer(name, dv_latency, bitwidth)


def _counter_width(dv_latency):
    """Minimum bit-width needed to represent values 0..dv_latency-1."""
    if dv_latency <= 1:
        return 1
    return (dv_latency - 1).bit_length()


def _generate_counter_buffer_dataless(name, dv_latency):
    cw = _counter_width(dv_latency)

    return f"""
module {name}(
  input  clk,
  input  rst,
  input  ins_valid,
  input  outs_ready,
  output outs_valid,
  output ins_ready
);

  reg [{cw - 1}:0] counter;
  reg busy;

  wire done;
  assign done = busy & (counter == {cw}'d0);

  always @(posedge clk) begin
    if (rst) begin
      busy    <= 1'b0;
      counter <= {cw}'d0;
    end else begin
      if (!busy) begin
        if (ins_valid) begin
          busy    <= 1'b1;
          counter <= {cw}'d{dv_latency - 1};
        end
      end else if (counter > {cw}'d0) begin
        counter <= counter - {cw}'d1;
      end else if (outs_ready) begin
        if (ins_valid) begin
          counter <= {cw}'d{dv_latency - 1};
        end else begin
          busy <= 1'b0;
        end
      end
    end
  end

  assign outs_valid = done;
  assign ins_ready  = ~busy | (done & outs_ready);

endmodule
"""


def _generate_counter_buffer(name, dv_latency, bitwidth):
    cw = _counter_width(dv_latency)
    inner_name = f"{name}_inner"

    dependencies = _generate_counter_buffer_dataless(inner_name, dv_latency)

    body = f"""
module {name}(
  input  clk,
  input  rst,
  // Input channel
  input  [{bitwidth} - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Output channel
  output [{bitwidth} - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);

  wire inputReady;
  reg [{bitwidth} - 1 : 0] data_reg;

  {inner_name} control (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (ins_valid  ),
    .ins_ready  (inputReady ),
    .outs_valid (outs_valid ),
    .outs_ready (outs_ready )
  );

  always @(posedge clk) begin
    if (inputReady & ins_valid) begin
      data_reg <= ins;
    end
  end

  assign ins_ready = inputReady;
  assign outs      = data_reg;

endmodule
"""

    return dependencies + body
