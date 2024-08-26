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
  output [DATA_TYPE - 1 : 0] result,
  output result_valid,
  output lhs_ready,
  output rhs_ready
);

  //assert(DATA_TYPE == 32) else $error("ENTITY_NAME currently only supports 32-bit floating point operands");

  wire constant_one = 1'b1;
  wire [ DATA_TYPE + 1 :0] ip_lhs, ip_rhs;

  wire ip_unordered, ip_result;

  reg [10*8 : 0] cmp_predicate = "COMPARATOR";

  // Instantiate the join node
  join_type #(
    .SIZE(2)
  ) join_inputs (
    .ins_valid  ({rhs_valid, lhs_valid}),
    .outs_ready (result_ready             ),
    .ins_ready  ({rhs_ready, lhs_ready}  ),
    .outs_valid (result_valid             )
  );

  ieee2nfloat_lhs InputIEEE_32bit (
    .X(lhs),
    .R(ip_lhs)
  );

  ieee2nfloat_rhs InputIEEE_32bit (
    .X(rhs),
    .R(ip_rhs)
  );

  // function to compare the equivalence of 2 strings
  function automatic bit compare_strings;
    input [10*8:0] str1;  
    input [10*8:0] str2;  
    integer i;
    begin
        // Initialize result as equal
        compare_strings = 1'b1;

        // Compare characters one by one
        for (i = 0; i < 10; i = i + 1) begin
            // If any character is different, set result to 0 and break
            if (str1[8*(i+1) -: 8] !== str2[8*(i+1) -: 8]) begin
                compare_strings = 1'b0;
                break;
            end
        end
    end
  endfunction

  // Compare the two strings
  reg [10*8:0] oeq_type = "OEQ";
  assign is_oeq = compare_strings(cmp_predicate, oeq_type);
  reg [10*8:0] ueq_type = "UEQ";
  assign is_ueq = compare_strings(cmp_predicate, ueq_type);
  reg [10*8:0] ogt_type = "OGT";
  assign is_ogt = compare_strings(cmp_predicate, ogt_type);
  reg [10*8:0] ugt_type = "UGT";
  assign is_ugt = compare_strings(cmp_predicate, ugt_type);
  reg [10*8:0] oge_type = "OGE";
  assign is_oge = compare_strings(cmp_predicate, oge_type);
  reg [10*8:0] uge_type = "UGE";
  assign is_uge = compare_strings(cmp_predicate, uge_type);
  reg [10*8:0] olt_type = "OLT";
  assign is_olt = compare_strings(cmp_predicate, olt_type);
  reg [10*8:0] ult_type = "ULT";
  assign is_ult = compare_strings(cmp_predicate, ult_type);
  reg [10*8:0] ole_type = "OLE";
  assign is_ole = compare_strings(cmp_predicate, ole_type);
  reg [10*8:0] ule_type = "ULE";
  assign is_ule = compare_strings(cmp_predicate, ule_type);
  reg [10*8:0] one_type = "ONE";
  assign is_une = compare_strings(cmp_predicate, one_type);
  reg [10*8:0] une_type = "UNE";
  assign is_une = compare_strings(cmp_predicate, une_type);
  reg [10*8:0] ord_type = "ORD";
  assign is_ord = compare_strings(cmp_predicate, ord_type);
  reg [10*8:0] uno_type = "UNO";
  assign is_uno = compare_strings(cmp_predicate, uno_type);

  initial begin
    if (is_oeq || is_ueq || is_one || is_uno || is_ord || is_uno) begin
      FloatingPointComparatorEQ operator (
        .clk(clk),
        .ce(constant_one),
        .X(ip_lhs),
        .Y(ip_rhs),
        .unordered(ip_unordered),
        .XeqY(ip_result)
      );
    end 
    if (is_ogt || is_ugt) begin
      FloatingPointComparatorGT operator (
        .clk(clk),
        .ce(constant_one),
        .X(ip_lhs),
        .Y(ip_rhs),
        .unordered(ip_unordered),
        .XgtY(ip_result)
      );
    end 
    if (is_oge || is_uge) begin
      FloatingPointComparatorGE operator (
        .clk(clk),
        .ce(constant_one),
        .X(ip_lhs),
        .Y(ip_rhs),
        .unordered(ip_unordered),
        .XgeY(ip_result)
      );
    end
    if (is_olt || is_ult) begin
      FloatingPointComparatorLT operator (
        .clk(clk),
        .ce(constant_one),
        .X(ip_lhs),
        .Y(ip_rhs),
        .unordered(ip_unordered),
        .XltY(ip_result)
      );
    end
    if (is_ole || is_ule) begin
      FloatingPointComparatorLE operator (
        .clk(clk),
        .ce(constant_one),
        .X(ip_lhs),
        .Y(ip_rhs),
        .unordered(ip_unordered),
        .XleY(ip_result)
      );
    end
    if( !is_oeq && !is_ueq && !is_ogt && !is_ugt && !is_oge && !is_uge && !is_olt && !is_ult && !is_ole && !is_ule && !is_one && !is_une && !is_ord && !is_uno) begin
      $error("COMPARATOR is an invalid predicate!");
    end
  end

  initial begin
    if (is_oeq || is_ogt || is_oge || is_olt || is_ole ) begin
      assign result = !ip_unordered & ip_result;
    end
    if (is_one) begin
      assign result = !ip_unordered & !ip_result;
    end
    if (is_ord) begin
      assign result = !ip_unordered;
    end
    if (is_ueq || is_ugt || is_uge || is_ult || is_ule ) begin
      assign result = ip_result | ip_unordered;
    end
    if (is_une) begin
      assign result = !ip_result | ip_unordered;
    end
    if (is_uno) begin
      assign result = ip_unordered;
    end

  end


endmodule