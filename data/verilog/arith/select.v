`timescale 1ns/1ps

module selector #(
    parameter DATA_TYPE = 8
)(
    // Inputs
    input  wire                     clk,
    input  wire                     rst,
    input  wire [0:0]               condition,
    input  wire                     condition_valid,
    input  wire [DATA_TYPE-1:0]     trueValue,
    input  wire                     trueValue_valid,
    input  wire [DATA_TYPE-1:0]     falseValue,
    input  wire                     falseValue_valid,
    input  wire                     result_ready,
    // Outputs
    output wire [DATA_TYPE-1:0]     result,
    output wire                     result_valid,
    output wire                     condition_ready,
    output wire                     trueValue_ready,
    output wire                     falseValue_ready
);

    // Parameters
    localparam discard_depth = 4;
    localparam counter_width = $clog2(discard_depth);

    // Internal signals
    reg [counter_width-1:0] num_token_to_discard_true  = 0;
    reg [counter_width-1:0] num_token_to_discard_false = 0;

    wire can_propagate_true;
    wire can_propagate_false;
    wire can_discard_true;
    wire can_discard_false;
    wire still_need_to_discard_true;
    wire still_need_to_discard_false;

    // ----------------------
    // Internal signal logic
    // ----------------------
    assign can_discard_true  = (trueValue_valid || (num_token_to_discard_true < discard_depth)) ? 1'b1 : 1'b0;
    assign can_discard_false = (falseValue_valid || (num_token_to_discard_false < discard_depth)) ? 1'b1 : 1'b0;

    assign can_propagate_true  = (condition_valid && condition[0] && trueValue_valid &&
                                 (num_token_to_discard_true == 0) && can_discard_false) ? 1'b1 : 1'b0;

    assign can_propagate_false = (condition_valid && !condition[0] && falseValue_valid &&
                                 (num_token_to_discard_false == 0) && can_discard_true) ? 1'b1 : 1'b0;

    assign still_need_to_discard_true  = (num_token_to_discard_true > 0) ? 1'b1 : 1'b0;
    assign still_need_to_discard_false = (num_token_to_discard_false > 0) ? 1'b1 : 1'b0;

    // ----------------------
    // Handshake signals
    // ----------------------
    assign result_valid     = can_propagate_true || can_propagate_false;
    assign result           = condition[0] ? trueValue : falseValue;

    assign trueValue_ready  = (~trueValue_valid) || (result_valid && result_ready) || still_need_to_discard_true;
    assign falseValue_ready = (~falseValue_valid) || (result_valid && result_ready) || still_need_to_discard_false;
    assign condition_ready  = (~condition_valid) || (result_valid && result_ready);

    // ----------------------
    // Discard counters
    // ----------------------
    always @(posedge clk) begin
        if (rst) begin
            num_token_to_discard_true  <= 0;
            num_token_to_discard_false <= 0;
        end else begin
            // True counter update
            if (result_valid && result_ready && !trueValue_valid)
                num_token_to_discard_true <= num_token_to_discard_true + 1;
            else if (still_need_to_discard_true && trueValue_valid)
                num_token_to_discard_true <= num_token_to_discard_true - 1;

            // False counter update
            if (result_valid && result_ready && !falseValue_valid)
                num_token_to_discard_false <= num_token_to_discard_false + 1;
            else if (still_need_to_discard_false && falseValue_valid)
                num_token_to_discard_false <= num_token_to_discard_false - 1;
        end
    end

endmodule
