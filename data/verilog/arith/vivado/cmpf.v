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

  reg [7:0] opcode;
  wire ip_unordered, ip_result;
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
    AP_OEQ = 8'b00000001,
    AP_OGT = 8'b00000010,
    AP_OGE = 8'b00000011,
    AP_OLT = 8'b00000100,
    AP_OLE = 8'b00000101,
    AP_ONE = 8'b00000110,
    AP_UNO = 8'b00001000;


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
      opcode = AP_OEQ; // 5'b00001
    end
    else if (cmp_predicate == ogt_type) begin
      opcode = AP_OGT; // 5'b00010
    end
    else if (cmp_predicate == oge_type) begin
      opcode = AP_OGE; // 5'b00011
    end
    else if (cmp_predicate == olt_type) begin
      opcode = AP_OLT; // 5'b00100
    end
    else if (cmp_predicate == ole_type) begin
      opcode = AP_OLE; // 5'b00101
    end
    else if (cmp_predicate == one_type) begin
      opcode = AP_ONE; // 5'b00110
    end
    else if (cmp_predicate == uno_type) begin
      opcode = AP_UNO; // 5'b01000
    end else begin
      $fatal("Unsupported comparator predicate"); 
    end
  end

  oehb_dataless oehb_lhs (
    .clk(clk),
    .rst(rst),
    .ins_valid(join_valid),
    .ins_ready(oehb_ready),
    .outs_valid(buff_valid),
    .outs_ready(result_ready)
  ); 
    
  cmpf_vitis_hls_single_precision_lat_0 fcmp_inst (
    .s_axis_a_tvalid      ( join_valid ),
    .s_axis_a_tdata       ( lhs ),
    .s_axis_b_tvalid      ( join_valid ),
    .s_axis_b_tdata       ( rhs ),
    .s_axis_operation_tvalid ( join_valid ),
    .s_axis_operation_tdata  ( opcode ),
    .m_axis_result_tvalid ( result_valid ),
    .m_axis_result_tdata  ( result_tmp )
  );

  assign result = result_tmp[0];




endmodule