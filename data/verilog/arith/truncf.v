module double_to_single (
    input  wire [63:0] ins,
    output wire [31:0] outs
);

    // Extract IEEE-754 double fields
    wire        sign_d;
    wire [10:0] exp_d;
    wire [51:0] frac_d;

    assign sign_d = ins[63];
    assign exp_d  = ins[62:52];
    assign frac_d = ins[51:0];

    // Convert to single precision
    wire        sign_s;
    wire [7:0]  exp_s;
    wire [22:0] frac_s;

    assign sign_s = sign_d;

    // Handle normal numbers: adjust exponent from double to single
    // double bias = 1023, single bias = 127
    assign exp_s = (exp_d == 11'h7FF) ? 8'hFF : // Inf/NaN
                   (exp_d == 11'h0)   ? 8'h00 : // Zero/subnormal
                                        (exp_d - 11'd1023 + 8'd127);

    // Truncate fraction: 52 → 23 bits
    assign frac_s = frac_d[51:29];

    assign outs = {sign_s, exp_s, frac_s};

endmodule

module truncf #(
    parameter INPUT_TYPE  = 64,
    parameter OUTPUT_TYPE = 32
)(
    input  wire clk,
    input  wire rst,

    // input channel
    input  wire [INPUT_TYPE-1:0] ins,
    input  wire ins_valid,
    output wire ins_ready,

    // output channel
    output wire [OUTPUT_TYPE-1:0] outs,
    output wire outs_valid,
    input  wire outs_ready
);

    initial begin
        if (INPUT_TYPE != 64)
            $fatal("truncf only supports double to single conversion!");
        if (OUTPUT_TYPE != 32)
            $fatal("truncf only supports double to single conversion!!");
    end

    double_to_single u_conv (
        .ins  (ins),
        .outs (outs)
    );

    // Valid/ready handshake
    assign outs_valid = ins_valid;
    assign ins_ready  = outs_ready;

endmodule

