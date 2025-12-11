module fptoui #(
    parameter DATA_TYPE = 32
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire [DATA_TYPE-1:0]    ins,
    input  wire                    ins_valid,
    input  wire                    outs_ready,

    output wire [DATA_TYPE-1:0]    outs,
    output wire                    outs_valid,
    output wire                    ins_ready
);

    // -----------------------------------------------------
    // Parameter check
    // -----------------------------------------------------
    initial begin
        if (DATA_TYPE != 32)
            $fatal("fptoui only supports 32-bit floating point inputs and outputs.");
    end

    // -----------------------------------------------------
    // Float32 to Signed Integer Conversion (IEEE-754)
    // -----------------------------------------------------

    wire        sign;
    wire [7:0]  exp;
    wire [22:0] frac;

    assign sign = ins[31];
    assign exp  = ins[30:23];
    assign frac = ins[22:0];
    reg signed [31:0] magnitude;

    // true exponent (unbiased)
    wire signed [9:0] e = exp - 127;

    // determine mantissa (with implicit leading 1 for normals)
    wire [23:0] mantissa = (exp == 0) ? {1'b0, frac} : {1'b1, frac};

    // compute shift amount (positive to left shift, negative to right shift)
    wire signed [9:0] shift = e - 23;

    reg signed [31:0] int_value;

    always @(*) begin
        // Special case: zero
        if (exp == 0 && frac == 0) begin
            int_value = 0;
        end
        // Overflow cases: exponent too large
        else if (e > 30) begin
            int_value = sign ? 32'h8000_0000 : 32'h7FFF_FFFF;
        end
        // |value| < 1 => truncates to zero
        else if (e < 0) begin
            int_value = 0;
        end 
        else begin
            // Shift mantissa safely
            if (shift >= 0)
                magnitude = mantissa << shift;
            else
                magnitude = mantissa >> (-shift);

            // Apply sign
            int_value = sign ? -magnitude : magnitude;
        end
    end


    wire [31:0] converted = int_value;

    // -----------------------------------------------------
    // 5-stage pipeline
    // -----------------------------------------------------
    reg [31:0] q0, q1, q2, q3, q4;
    wire       buff_valid;
    wire       oehb_ready;

    always @(posedge clk) begin
        if (rst) begin
            q0 <= 0;
            q1 <= 0;
            q2 <= 0;
            q3 <= 0;
            q4 <= 0;
        end else if (oehb_ready) begin
            q0 <= converted;
            q1 <= q0;
            q2 <= q1;
            q3 <= q2;
            q4 <= q3;
        end
    end

    assign outs = q4;

    // -----------------------------------------------------
    // Delay buffer (LATENCY-1 = 4)
    // -----------------------------------------------------
    delay_buffer #(
        .SIZE(4)
    ) delay_buffer_inst (
        .clk       (clk),
        .rst       (rst),
        .valid_in  (ins_valid),
        .ready_in  (oehb_ready),
        .valid_out (buff_valid)
    );

    // -----------------------------------------------------
    // Output OEHB 
    // -----------------------------------------------------
    wire [31:0] oehb_dataOut;
    wire [31:0] oehb_dataIn = q4;

    oehb #(
        .DATA_TYPE(DATA_TYPE)
    ) oehb_inst (
        .clk        (clk),
        .rst        (rst),
        .ins_valid  (buff_valid),
        .outs_ready (outs_ready),
        .outs_valid (outs_valid),
        .ins_ready  (oehb_ready),
        .ins        (oehb_dataIn),
        .outs       (oehb_dataOut)
    );

    assign ins_ready = oehb_ready;

endmodule
