`timescale 1ns/1ps
module mux #(
  parameter SIZE = 2,
  parameter DATA_TYPE = 32,
  parameter SELECT_TYPE = 2
)(
  input  clk,
  input  rst,
  // Data input channels
  input  [(SIZE * DATA_TYPE) - 1 : 0] ins, 
  input  [SIZE - 1 : 0] ins_valid,
  output reg [SIZE - 1 : 0] ins_ready,
  // Index input channel
  input  [SELECT_TYPE - 1 : 0] index,
  input  index_valid,
  output index_ready,
  // Output channel
  output [DATA_TYPE - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);
  wire [DATA_TYPE - 1 : 0] tehb_ins;
  wire tehb_ins_ready;
  wire tehb_ins_valid;

  reg [DATA_TYPE - 1 : 0] selectedData;
  reg selectedData_valid;

  integer i;
  always @(*) begin
    if(rst) begin
      selectedData_valid = 0;
      for (i = DATA_TYPE - 1; i >= 0; i = i - 1) begin
        selectedData[i] = 0;
      end
      for (i = SIZE - 1; i >= 0; i = i - 1) begin
        ins_ready[i] = 0;
      end
    end else begin
      selectedData = ins[0 * DATA_TYPE +: DATA_TYPE];
      selectedData_valid = 0;

      for (i = SIZE - 1; i >= 0; i = i - 1) begin
        if (((i[SELECT_TYPE - 1 : 0] == index) & index_valid & tehb_ins_ready & ins_valid[i]) | ~ins_valid[i]) begin
          ins_ready[i] = 1;
        end else begin
          ins_ready[i] = 0;
        end

        if (index == i[SELECT_TYPE - 1 : 0] && index_valid && ins_valid[i]) begin
          selectedData = ins[i * DATA_TYPE +: DATA_TYPE];
          selectedData_valid = 1;
        end
      end
    end 

  end

  assign index_ready = ~index_valid | (selectedData_valid & tehb_ins_ready);
  assign tehb_ins = selectedData;
  assign tehb_ins_valid = selectedData_valid;

  // Instantiate the tehb module
  tehb #(
    .DATA_TYPE(DATA_TYPE)
  ) tehb_inst (
    .clk        (clk           ),
    .rst        (rst           ),
    .ins        (tehb_ins      ),
    .ins_valid  (tehb_ins_valid),
    .ins_ready  (tehb_ins_ready),
    .outs       (outs          ),
    .outs_valid (outs_valid    ),
    .outs_ready (outs_ready    )
  );

endmodule
