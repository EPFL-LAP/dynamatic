from generators.support.dataless.dataless_elastic_fifo_inner import generate_dataless_elastic_fifo_inner
def generate_dataless_tfifo(name, params):
    num_slots = params["num_slots"]

    header = "`timescale 1ns/1ps\n"

    elastic_fifo_inner_dataless_name = "elastic_fifo_inner_dataless"
    elastic_fifo_inner_dataless = generate_dataless_elastic_fifo_inner(elastic_fifo_inner_dataless_name, {"num_slots": num_slots})

    dataless_tfifo_body = f"""

// Module of dataless_tfifo
module {name}(
  input  clk,
  input  rst,
  // Input channel
  input  ins_valid,
  output ins_ready,
  // Output channel
  output outs_valid,
  input  outs_ready
);
  wire fifo_valid, fifo_ready;
  wire fifo_pvalid, fifo_nready;

  assign outs_valid = ins_valid || fifo_valid;
  assign ins_ready = fifo_ready || outs_ready;
  assign fifo_pvalid = ins_valid && (!outs_ready || fifo_valid);
  assign fifo_nready = outs_ready;

  {elastic_fifo_inner_dataless_name} fifo (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (fifo_pvalid),
    .outs_ready (fifo_nready),
    .outs_valid (fifo_valid ),
    .ins_ready  (fifo_ready )
  );

endmodule

"""

    return header + elastic_fifo_inner_dataless + dataless_tfifo_body