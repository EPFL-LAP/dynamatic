from generators.support.elastic_fifo_inner import generate_elastic_fifo_inner
def generate_tfifo(name, params):
    num_slots = params["num_slots"]
    data_type = params["data_type"]

    verilog_header = "`timescale 1ns/1ps\n"
    elastic_fifo_inner_name = "elastic_fifo_inner"
    verilog_elastic_fifo_inner = generate_elastic_fifo_inner(elastic_fifo_inner_name, {"num_slots": num_slots, "data_type": data_type})

    verilog_tfifo_body = f"""
// tfifo Module
module {name} #(
  parameter NUM_SLOTS = {num_slots},
  parameter DATA_TYPE = {data_type}
)(
  input  clk,
  input  rst,
  input  [DATA_TYPE - 1 : 0] ins,
  input  ins_valid,
  input  outs_ready,
  output [DATA_TYPE - 1 : 0] outs,
  output outs_valid,
  output ins_ready
);
  wire mux_sel;
  wire fifo_valid, fifo_ready;
  wire fifo_pvalid, fifo_nready;
  wire [DATA_TYPE - 1 : 0] fifo_in, fifo_out;

  // Dataout assignment
  assign outs = mux_sel ? fifo_out : ins;

  assign outs_valid = ins_valid || fifo_valid;
  assign ins_ready = fifo_ready || outs_ready;
  assign fifo_pvalid = ins_valid && (!outs_ready || fifo_valid);
  assign mux_sel = fifo_valid;

  assign fifo_nready = outs_ready;
  assign fifo_in = ins;

  {elastic_fifo_inner_name} #(
    .NUM_SLOTS  (NUM_SLOTS ), 
    .DATA_TYPE (DATA_TYPE)
  ) fifo (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins        (fifo_in    ),
    .ins_valid  (fifo_pvalid),
    .outs_ready (fifo_nready),
    .outs       (fifo_out   ),
    .outs_valid (fifo_valid ),
    .ins_ready  (fifo_ready )
  );
endmodule

"""

    return verilog_header + verilog_elastic_fifo_inner + verilog_tfifo_body