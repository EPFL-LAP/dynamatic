module sitofp #(
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

    // -----------------------------------------------
    // Parameter check
    // -----------------------------------------------
    initial begin
        if (DATA_TYPE != 32)
            $fatal("sitofp only supports 32-bit floating point outputs.");
    end

    // -----------------------------------------------
    // Integer to Float conversion 
    // -----------------------------------------------
    wire        sign;
    wire [31:0] abs_val;
    assign sign    = ins[31];
    assign abs_val = sign ? (~ins + 1'b1) : ins;

    // Leading zero count
    integer i;
    reg [5:0] lzc;
    always @(*) begin
        lzc = 0;
        for (i = 31; i >= 0; i = i - 1)
            if (abs_val[i] == 1'b1) begin
                lzc = 31 - i;
                i = -1;
            end
    end

    // Normalization
    wire [31:0] shifted = abs_val << lzc;

    // Exponent = (31 - lzc) + bias
    wire [7:0] exp = (abs_val == 0) ? 8'b0 :
                     (31 - lzc) + 8'd127;

    // Fraction = bits 30:8 of normalized significand
    wire [22:0] frac = shifted[30:8];

    // Final IEEE-754 float
    wire [31:0] converted;
    assign converted = (abs_val == 0) ?
                       32'b0 :
                       {sign, exp, frac};

    // -----------------------------------------------
    // 5-stage pipeline
    // -----------------------------------------------
    reg [31:0] q0, q1, q2, q3, q4;
    wire       oehb_ready;
    wire       buff_valid;

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

    // Output of the pipeline
    assign outs = q4;

    // -----------------------------------------------
    // Delay buffer 
    // -----------------------------------------------
    delay_buffer #(
        .SIZE(4)
    ) delay_buffer_inst (
        .clk       (clk),
        .rst       (rst),
        .valid_in  (ins_valid),
        .ready_in  (oehb_ready),
        .valid_out (buff_valid)
    );

    // -----------------------------------------------
    // Output OEHB
    // -----------------------------------------------
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
