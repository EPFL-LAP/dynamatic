from generators.handshake.dataless.tehb import generate_tehb as generate_dataless_tehb

def generate_tehb(name, params):
    data_type = params["data_type"]
    if(data_type == 0):
        return generate_dataless_tehb(name, params)

    verilog_header = "`timescale 1ns/1ps\n"

    tehb_name = name + "_dataless_tehb"
    verilog_dataless_tehb = generate_dataless_tehb(tehb_name, params)

    verilog_tehb_body = f"""
// Module of tehb
module {name} #(
	parameter DATA_TYPE = {data_type}
)(
	input  clk,
	input  rst,
  // Input Channel
	input  [DATA_TYPE - 1 : 0] ins,
	input  ins_valid,
  output ins_ready,
  // Output Channel
  output [DATA_TYPE - 1 : 0]	outs,
  output outs_valid,
	input  outs_ready
);
	// Signal Definition
	wire regEnable, regNotFull;
	reg [DATA_TYPE - 1 : 0] dataReg = 0;

	// Instantiate control logic part
	{tehb_name} control (
		.clk		    (clk	     ),
		.rst		    (rst	     ),
		.ins_valid	(ins_valid ),
    .ins_ready	(regNotFull),
		.outs_valid	(outs_valid),
		.outs_ready	(outs_ready)
	);

	assign regEnable = regNotFull & ins_valid & ~outs_ready;

	always @(posedge clk) begin
		if (rst) begin
			dataReg <= 0;
		end else if (regEnable) begin
			dataReg <= ins;
		end
	end

	// Output Assignment
	assign outs = regNotFull ? ins : dataReg;

	assign ins_ready = regNotFull;

endmodule

"""
    return verilog_header + verilog_dataless_tehb + verilog_tehb_body