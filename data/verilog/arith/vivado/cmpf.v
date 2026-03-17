`timescale 1ns/1ps


module ENTITY_NAME #(
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
  output result,
  output result_valid,
  output lhs_ready,
  output rhs_ready
);

  //assert(DATA_TYPE == 32) else $error("ENTITY_NAME currently only supports 32-bit floating point operands");

  wire constant_one = 1'b1;
  wire [ DATA_TYPE + 1 :0] ip_lhs, ip_rhs;
  wire [ 7 :0] ip_result;
  reg [7:0] opcode;
  wire join_valid, oehb_ready, buff_valid;
  wire [7:0] result_tmp;

  reg [10*8 : 0] cmp_predicate = "COMPARATOR";

  // Instantiate the join node
  join_type #(
    .SIZE(2)
  ) join_inputs (
    .ins_valid  ({rhs_valid, lhs_valid}),
    .outs_ready (oehb_ready             ),
    .ins_ready  ({rhs_ready, lhs_ready}  ),
    .outs_valid (join_valid             )
  );

  localparam [7:0]
    AP_OEQ = 8'b00010100,
    AP_OGT = 8'b00100100,
    AP_OGE = 8'b00110100,
    AP_OLT = 8'b00001100,
    AP_OLE = 8'b00011100,
    AP_ONE = 8'b00101100,
    AP_UNO = 8'b00000100;


  // Compare the two strings
  reg [10*8:0] oeq_type = "OEQ";
  reg [10*8:0] ogt_type = "OGT";
  reg [10*8:0] oge_type = "OGE";
  reg [10*8:0] olt_type = "OLT";
  reg [10*8:0] ole_type = "OLE";
  reg [10*8:0] one_type = "ONE";
  reg [10*8:0] uno_type = "UNO";

  always @(*) begin
    if(cmp_predicate == oeq_type) begin
      opcode = AP_OEQ; // 8'b00010100
    end
    else if (cmp_predicate == ogt_type) begin
      opcode = AP_OGT; // 8'b00100100
    end
    else if (cmp_predicate == oge_type) begin
      opcode = AP_OGE; // 8'b00110100
    end
    else if (cmp_predicate == olt_type) begin
      opcode = AP_OLT; // 8'b00001100
    end
    else if (cmp_predicate == ole_type) begin
      opcode = AP_OLE; // 8'b00011100
    end
    else if (cmp_predicate == one_type) begin
      opcode = AP_ONE; // 8'b00101100
    end
    else if (cmp_predicate == uno_type) begin
      opcode = AP_UNO; // 8'b00000100
    end else begin
      $fatal("Unsupported comparator predicate"); 
    end
  end

  oehb #(
    .DATA_TYPE(8)
  ) oehb_lhs (
    .clk(clk),
    .rst(rst),
    .ins(ip_result),
    .ins_valid(buff_valid),
    .ins_ready(oehb_ready),
    .outs(result_tmp),
    .outs_valid(result_valid),
    .outs_ready(result_ready)
  ); 
    
  cmpf_vitis_hls_single_precision_lat_0 fcmp_inst (
    .s_axis_a_tvalid      ( join_valid ),
    .s_axis_a_tdata       ( lhs ),
    .s_axis_b_tvalid      ( join_valid ),
    .s_axis_b_tdata       ( rhs ),
    .s_axis_operation_tvalid ( join_valid ),
    .s_axis_operation_tdata  ( opcode ),
    .m_axis_result_tvalid ( buff_valid ),
    .m_axis_result_tdata  ( ip_result )
  );

  assign result = result_tmp[0];




endmodule