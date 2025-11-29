module single_to_double (
    input  wire [31:0] ins,
    output wire [63:0] outs
);

    // IEEE-754 single format
    wire        sign_s;
    wire [7:0]  exp_s;
    wire [22:0] frac_s;

    assign sign_s = ins[31];
    assign exp_s  = ins[30:23];
    assign frac_s = ins[22:0];

    // IEEE-754 double format fields
    wire        sign_d;
    wire [10:0] exp_d;
    wire [51:0] frac_d;

    // Convert exponent:
    // exponent_bias_single = 127
    // exponent_bias_double = 1023
    // => exp_d = exp_s - 127 + 1023 = exp_s + 896
    assign sign_d = sign_s;

    assign exp_d  =
        (exp_s == 8'b0) ? 11'b0 :               // zero / subnormal → zero / subnormal
        (exp_s == 8'hFF) ? 11'h7FF :            // Inf/NaN
                           (exp_s + 11'd896);   // normal conversion

    // Convert fraction:
    // 23-bit → 52-bit by shifting left
    assign frac_d =
        (exp_s == 8'h00) ? {frac_s, 29'b0} :    // Subnormal: shift differently
                           {frac_s, 29'b0};     // Normal case

    assign outs = {sign_d, exp_d, frac_d};

endmodule


module extf #(
    parameter INPUT_TYPE  = 32,
    parameter OUTPUT_TYPE = 64
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
        if (INPUT_TYPE != 32)
            $fatal("extf only supports single to double conversion!");
        if (OUTPUT_TYPE != 64)
            $fatal("extf only supports single to double conversion!!");
    end
    
    // Instantiate converter
    single_to_double u_conv (
        .ins  (ins),
        .outs (outs)
    );

    // simple pass-through handshake
    assign outs_valid = ins_valid;
    assign ins_ready  = outs_ready;

endmodule
