module tb_join #(
    parameter SIZE = 4
)(
    // inputs
    input  wire [SIZE-1:0] ins_valid,
    input  wire            outs_ready,

    // outputs
    output wire            outs_valid,
    output reg  [SIZE-1:0] ins_ready
);

    // outs_valid is 1 when all ins_valid bits are 1
    assign outs_valid = (ins_valid == {SIZE{1'b1}}) ? 1'b1 : 1'b0;

    integer i, j;
    reg [SIZE-1:0] singlePValid;

    always @(*) begin
        // Compute singlePValid[i]
        for (i = 0; i < SIZE; i = i + 1) begin
            singlePValid[i] = 1'b1;
            for (j = 0; j < SIZE; j = j + 1) begin
                if (i != j)
                    singlePValid[i] = singlePValid[i] & ins_valid[j];
            end
        end

        // ins_ready[i] = singlePValid[i] AND outs_ready
        for (i = 0; i < SIZE; i = i + 1) begin
            ins_ready[i] = singlePValid[i] & outs_ready;
        end
    end

endmodule