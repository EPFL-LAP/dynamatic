module uitofp #(
    parameter DATA_TYPE = 32
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire [DATA_TYPE-1:0]    ins,
    input  wire                    ins_valid,
    input  wire                    outs_ready,

    output wire [31:0]             outs,
    output wire                    outs_valid,
    output wire                    ins_ready
);

    // -----------------------------------------------------
    // Parameters
    // -----------------------------------------------------
    localparam LATENCY = 5;

    // -----------------------------------------------------
    // Unsigned int -> Float32 Conversion
    // -----------------------------------------------------
    wire [31:0] abs_val = ins;

    // Leading Zero Count
    integer i;
    reg [5:0] lzc;
    always @(*) begin
        lzc = 0;
        for (i = 31; i >= 0; i = i - 1) begin
            if (abs_val[i] == 1'b1) begin
                lzc = 31 - i;
                i = -1;
            end
        end
    end

    // Normalized mantissa
    wire [31:0] shifted = abs_val << lzc;

    // Exponent
    wire [7:0] exp = (abs_val == 0) ? 8'b0 : (31 - lzc) + 8'd127;

    // Fraction (bits 30:8)
    wire [22:0] frac = shifted[30:8];

    // Sign bit for unsigned input is always 0
    wire [31:0] converted = (abs_val == 0) ? 32'b0 : {1'b0, exp, frac};

    // -----------------------------------------------------
    // 5-stage pipeline
    // -----------------------------------------------------
    reg [31:0] q0, q1, q2, q3, q4;
    wire       buff_valid;
    wire       oehb_ready;

    always @(posedge clk) begin
        if (rst) begin
            q0 <= 32'b0;
            q1 <= 32'b0;
            q2 <= 32'b0;
            q3 <= 32'b0;
            q4 <= 32'b0;
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
    // Delay buffer (LATENCY - 1 = 4)
    // -----------------------------------------------------
    delay_buffer #(
        .SIZE(LATENCY - 1)
    ) delay_buffer_inst (
        .clk       (clk),
        .rst       (rst),
        .ins_valid (ins_valid),
        .outs_ready(oehb_ready),
        .outs_valid(buff_valid)
    );

    // -----------------------------------------------------
    // Output OEHB 
    // -----------------------------------------------------
    wire [0:0] oehb_dataOut;
    wire [0:0] oehb_dataIn = q4[0];

    oehb #(
        .DATA_TYPE(1)
    ) oehb_inst (
        .clk        (clk),
        .rst        (rst),
        .ins_valid  (buff_valid),
        .outs_ready (outs_ready),
        .outs_valid (outs_valid),
        .ins_ready  (ins_ready),
        .ins        (oehb_dataIn),
        .outs       (oehb_dataOut)
    );

endmodule
