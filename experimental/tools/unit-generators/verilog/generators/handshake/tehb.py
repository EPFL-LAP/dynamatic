def generate_tehb(name, params):
    data_type = params["data_type"]
    if(data_type == 0):
        return generate_dataless_tehb(name, {})

    header = "`timescale 1ns/1ps\n"

    tehb_name = name + "_dataless_tehb"
    dataless_tehb = generate_dataless_tehb(tehb_name, {})

    tehb_body = f"""
// Module of tehb
module {name}(
	input  clk,
	input  rst,
  // Input Channel
	input  [{data_type} - 1 : 0] ins,
	input  ins_valid,
  output ins_ready,
  // Output Channel
  output [{data_type} - 1 : 0]	outs,
  output outs_valid,
	input  outs_ready
);
	// Signal Definition
	wire regEnable, regNotFull;
	reg [{data_type} - 1 : 0] dataReg = 0;

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
    return header + dataless_tehb + tehb_body

def generate_dataless_tehb(name, params):
    return f"""
`timescale 1ns/1ps
// Module of tehb
module {name} (
	input  clk,
	input  rst,
  // Input Channel
	input  ins_valid,
  output ins_ready,
  // Output Channel
  output outs_valid,	
	input  outs_ready
);
	reg fullReg = 0;
	
	always @(posedge clk) begin
		if (rst) begin
			fullReg <= 0;
		end else begin
			fullReg <= (ins_valid | fullReg) & ~outs_ready;
		end
	end

	assign ins_ready = ~fullReg;
	assign outs_valid = ins_valid | fullReg;

endmodule
"""