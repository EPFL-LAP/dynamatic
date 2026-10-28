from generators.handshake.buffers.fifo_break_dv import generate_fifo_break_dv


def generate_fifo_break_none(name, params):
    bitwidth = params["bitwidth"]
    num_slots = params["num_slots"]

    if bitwidth == 0:
        return _generate_fifo_break_none_dataless(name, {"num_slots": num_slots})
    else:
        return _generate_fifo_break_none(name, {"num_slots": num_slots, "bitwidth": bitwidth})


def _generate_fifo_break_none(name, params):
    num_slots = params["num_slots"]
    bitwidth = params["bitwidth"]

    fifo_break_dv_name = name + "_fifo_break_dv"
    fifo_break_dv = generate_fifo_break_dv(
        fifo_break_dv_name, {"num_slots": num_slots, "bitwidth": bitwidth})

    fifo_break_none_body = f"""
// Module of fifo_break_none
module {name}(
  input  clk,
  input  rst,
  input  [{bitwidth} - 1 : 0] ins,
  input  ins_valid,
  input  outs_ready,
  output [{bitwidth} - 1 : 0] outs,
  output outs_valid,
  output ins_ready
);
  wire mux_sel;
  wire fifo_valid, fifo_ready;
  wire fifo_pvalid, fifo_nready;
  wire [{bitwidth} - 1 : 0] fifo_in, fifo_out;

  // Dataout assignment
  assign outs = mux_sel ? fifo_out : ins;

  assign outs_valid = ins_valid || fifo_valid;
  assign ins_ready = fifo_ready || outs_ready;
  assign fifo_pvalid = ins_valid && (!outs_ready || fifo_valid);
  assign mux_sel = fifo_valid;

  assign fifo_nready = outs_ready;
  assign fifo_in = ins;

  {fifo_break_dv_name} fifo (
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

    return fifo_break_dv + fifo_break_none_body


def _generate_fifo_break_none_dataless(name, params):
    num_slots = params["num_slots"]

    fifo_break_dv_name_dataless = name + "_fifo_break_dv_dataless"
    fifo_break_dv_dataless = generate_fifo_break_dv(
        fifo_break_dv_name_dataless, {"num_slots": num_slots, "bitwidth": 0})

    fifo_break_none_dataless_body = f"""

// Module of fifo_break_none_dataless
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

  {fifo_break_dv_name_dataless} fifo (
    .clk        (clk        ),
    .rst        (rst        ),
    .ins_valid  (fifo_pvalid),
    .outs_ready (fifo_nready),
    .outs_valid (fifo_valid ),
    .ins_ready  (fifo_ready )
  );

endmodule

"""

    return fifo_break_dv_dataless + fifo_break_none_dataless_body
