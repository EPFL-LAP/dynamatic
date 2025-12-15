`timescale 1ns/1ps
module divui #(
  parameter DATA_TYPE = 32
)(
  // inputs
  input  clk,
  input  rst,
  input  [DATA_TYPE - 1 : 0] lhs,
  input  lhs_valid,
  input  [DATA_TYPE - 1 : 0] rhs,
  input  rhs_valid,
  input  result_ready,
  // outputs
  output [DATA_TYPE - 1 : 0] result,
  output result_valid,
  output lhs_ready,
  output rhs_ready
);

  wire join_valid;
  wire oehb_ready;
  wire buff_valid;

  // Instantiate the join node
  join_type #(
    .SIZE(2)
  ) join_inputs (
    .ins_valid ({rhs_valid, lhs_valid}),
    .outs_ready (oehb_ready),
    .ins_ready ({rhs_ready, lhs_ready}),
    .outs_valid (join_valid)
  );

  divui_vitis_hls_wrapper ip (
      .clk(clk),
      .reset(rst),
      .din0(lhs),
      .din1(rhs),
      .ce(oehb_ready),
      .dout(result)
  );

  oehb_dataless oehb_inst (
    .clk(clk),
    .rst(rst),
    .ins_valid(buff_valid),
    .ins_ready(oehb_ready),
    .outs_valid(result_valid),
    .outs_ready(result_ready)
  ); 

  delay_buffer #(
    .SIZE(34)
  ) buff (
    .clk(clk),
    .rst(rst),
    .valid_in(join_valid),
    .ready_in(oehb_ready),
    .valid_out(buff_valid)
  );

endmodule

// [START This part is translated from the VHDL file using an AI tool]

module divui_vitis_hls_wrapper(
    input  wire clk,
    input  wire reset,
    input  wire ce,
    input  wire [31:0] din0,
    input  wire [31:0] din1,
    output wire [31:0] dout
);

    wire [31:0] sig_remd;

    dynamatic_units_6ns_udiv_32ns_32ns_32_36_1_div #(
        .in0_WIDTH(32),
        .in1_WIDTH(32),
        .out_WIDTH(32)
    ) div_inst (
        .clk(clk),
        .reset(reset),
        .ce(ce),
        .dividend(din0),
        .divisor(din1),
        .quot(dout),
        .remd(sig_remd)
    );
endmodule

module dynamatic_units_6ns_udiv_32ns_32ns_32_36_1_div #(
    parameter in0_WIDTH = 32,
    parameter in1_WIDTH = 32,
    parameter out_WIDTH = 32
)(
    input  wire clk,
    input  wire reset,
    input  wire ce,
    input  wire [in0_WIDTH-1:0] dividend,
    input  wire [in1_WIDTH-1:0] divisor,
    output reg  [out_WIDTH-1:0] quot,
    output reg  [out_WIDTH-1:0] remd
);

    wire [out_WIDTH-1:0] quot_u;
    wire [out_WIDTH-1:0] remd_u;

    reg  [in0_WIDTH-1:0] dividend0;
    reg  [in1_WIDTH-1:0] divisor0;

    dynamatic_units_6ns_udiv_32ns_32ns_32_36_1_div_u #(
        .in0_WIDTH(in0_WIDTH),
        .in1_WIDTH(in1_WIDTH),
        .out_WIDTH(out_WIDTH)
    ) u_div (
        .clk(clk),
        .reset(reset),
        .ce(ce),
        .dividend(dividend0),
        .divisor(divisor0),
        .quot(quot_u),
        .remd(remd_u)
    );

    // Input registers (latency = 1)
    always @(posedge clk) begin
        if (ce) begin
            dividend0 <= dividend;
            divisor0  <= divisor;
        end
    end

    // Output registers (latency = 1)
    always @(posedge clk) begin
        if (ce) begin
            quot <= quot_u;
            remd <= remd_u;
        end
    end
endmodule

module dynamatic_units_6ns_udiv_32ns_32ns_32_36_1_div_u #(
    parameter in0_WIDTH = 32,
    parameter in1_WIDTH = 32,
    parameter out_WIDTH = 32
)(
    input  wire clk,
    input  wire reset,
    input  wire ce,
    input  wire [in0_WIDTH-1:0] dividend,
    input  wire [in1_WIDTH-1:0] divisor,
    output wire [out_WIDTH-1:0] quot,
    output wire [out_WIDTH-1:0] remd
);

    // max(in0_WIDTH, in1_WIDTH)
    localparam cal_WIDTH = (in0_WIDTH > in1_WIDTH) ? in0_WIDTH : in1_WIDTH;

    // Internal pipeline arrays
    reg [in0_WIDTH-1:0] dividend_tmp [0:in0_WIDTH];
    reg [in1_WIDTH-1:0] divisor_tmp  [0:in0_WIDTH];
    reg [in0_WIDTH-1:0] remd_tmp     [0:in0_WIDTH];
    wire [in0_WIDTH-1:0] comb_tmp    [0:in0_WIDTH-1];
    wire [cal_WIDTH:0]   cal_tmp     [0:in0_WIDTH-1];

    // Output assignment
    assign quot = dividend_tmp[in0_WIDTH][out_WIDTH-1:0];
    assign remd = remd_tmp[in0_WIDTH][out_WIDTH-1:0];

    // Input buffer (latency = 1)
    always @(posedge clk) begin
        if (ce) begin
            dividend_tmp[0] <= dividend;
            divisor_tmp[0]  <= divisor;
            remd_tmp[0] <= {in0_WIDTH{1'b0}};
        end
    end

    // Main division pipeline
    genvar i;
    generate
        for (i = 0; i < in0_WIDTH; i = i + 1) begin : run_proc

            assign comb_tmp[i] = { remd_tmp[i][in0_WIDTH-2:0], dividend_tmp[i][in0_WIDTH-1] };
            assign cal_tmp[i]  = {1'b0, comb_tmp[i]} - {1'b0, divisor_tmp[i]};

            always @(posedge clk) begin
                if (ce) begin
                    dividend_tmp[i+1] <= { dividend_tmp[i][in0_WIDTH-2:0], ~cal_tmp[i][cal_WIDTH] };
                    divisor_tmp[i+1]  <= divisor_tmp[i];

                    if (cal_tmp[i][cal_WIDTH])
                        remd_tmp[i+1] <= comb_tmp[i];
                    else
                        remd_tmp[i+1] <= cal_tmp[i][in0_WIDTH-1:0];
                end
            end
        end
    endgenerate
endmodule

// [END This part is translated from the VHDL file using an AI tool]
