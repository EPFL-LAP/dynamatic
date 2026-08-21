(* use_dsp = "no" *) 
module systolic_pe
#(
  parameter DATA_TYPE = 8,
  parameter BURST_TYPE = 16
)
(
  input wire clk,
  input rst,
  // Burst size information
  input wire [BURST_TYPE-1 : 0] burst_len,
  input wire burst_len_valid,
  output wire burst_len_ready,
  // West input data and control
  input wire [DATA_TYPE-1 : 0] west_input,
  input wire west_input_valid,
  output wire west_input_ready,
  // North input data and control
  input wire [DATA_TYPE-1 : 0] north_input,
  input wire north_input_valid,
  output wire north_input_ready,
  // East output data and control
  output wire [DATA_TYPE-1 : 0] east_output,
  output wire east_output_valid,
  input wire east_output_ready,
  // South output data and control
  output wire [DATA_TYPE-1 : 0] south_output,
  output wire south_output_valid,
  input wire south_output_ready,
  // Final result data and control
  output wire [DATA_TYPE-1 : 0] out,
  output wire out_valid,
  input wire out_ready
);

  reg started;
  reg done_reg;
  
  reg [DATA_TYPE-1 : 0] west_input_reg;
  reg [DATA_TYPE-1 : 0] north_input_reg;

  reg [DATA_TYPE-1 : 0] output_reg;
  reg [DATA_TYPE-1 : 0] product;

  reg [BURST_TYPE-1 : 0] compute_cycle;
  reg [BURST_TYPE-1 : 0] burst_len_reg;
  reg saved_burst_len;

  wire tmp_west_input_ready, tmp_north_input_ready;
  wire result_valid;
  wire result_valid_internal;

  wire reading_data;

  // Instantiate the join node to synchronize west and north inputs
  // and output when to perform the MAC operation
  join_type #(
    .SIZE(2)
  ) join_inputs (
    .ins_valid  ({west_input_valid, north_input_valid}),
    .outs_ready (result_ready             ),
    .ins_ready  ({tmp_west_input_ready, tmp_north_input_ready}  ),
    .outs_valid (result_valid_internal      )
  );

  // Compute result valid when reading data
  always @(posedge clk) begin
    if (rst == 1'b1) begin
      west_input_reg <= 0;
      north_input_reg <= 0;
      output_reg <= 0;
    end
    else if (reading_data && started) begin
      west_input_reg <= west_input;
      north_input_reg <= north_input;
      product = west_input * north_input; //MAC in single cycle
      output_reg <= output_reg + product;
    end
  end

  // Save burst length info
  always @(posedge clk) begin
    if (rst == 1'b1) begin
      burst_len_reg <= 0;
      saved_burst_len <= 1'b0;
    end
    else if (burst_len_valid && ~saved_burst_len) begin
      burst_len_reg <= burst_len;
      saved_burst_len <= 1'b1;
    end
  end

  // Count the number of elements accumulated
  always @(posedge clk)  begin
    if (rst == 1'b1) begin
      compute_cycle <= 0;
      done_reg <= 1'b0;
      started <= 1'b0;
    end
    else begin
      if (reading_data && !started) begin
        started <= 1'b1;
        output_reg <= 0;
        west_input_reg <= 0;
        north_input_reg <= 0;
        compute_cycle <= 0;
      end
      if (reading_data && started) begin
        if (compute_cycle == burst_len_reg - 1) begin
          done_reg <= 1'b1;
          saved_burst_len <= 1'b0;
          started <= 1'b0;
          burst_len_reg <= 0;
        end else begin
          compute_cycle <= compute_cycle + 1;
        end
      end
      else
        done_reg <= 1'b0;
    end
  end
  
  assign out = output_reg;
  assign out_valid = done_reg;
  assign east_output = west_input;
  assign east_output_valid = reading_data && started;
  assign south_output = north_input;
  assign south_output_valid = reading_data && started;
  // The PE is ready only when the next PEs are ready to accept data
  assign result_ready = out_ready & east_output_ready & south_output_ready;
  // West and north inputs are ready only when burst length has been saved
  // and next PEs are ready to accept data
  assign west_input_ready = saved_burst_len & tmp_west_input_ready & started;
  assign north_input_ready = saved_burst_len & tmp_north_input_ready & started;
  // Once a burst info has been saved, there is no need to accept new burst info
  assign burst_len_ready = ~saved_burst_len;
  // Reading data when both inputs are valid and burst length is saved
  assign reading_data = result_valid_internal & saved_burst_len;

endmodule


module systolic_unit
#(
  parameter DATA_TYPE = 8,
  parameter BURST_TYPE = 16
)
(
  input wire clk,
  input rst,
  input wire [(BURST_TYPE + 2*DATA_TYPE)-1 : 0] ins,
  input wire [2:0] ins_valid,
  output wire [2:0] ins_ready,
  output wire [(3*DATA_TYPE)-1 : 0] outs,
  output wire [2:0] outs_valid,
  input wire [2:0] outs_ready
);

  // Burst size information
  wire [BURST_TYPE-1 : 0] burst_len;
  wire burst_len_valid;
  wire burst_len_ready;
  // West data and control
  wire [DATA_TYPE-1 : 0] west_input;
  wire west_input_valid;
  wire west_input_ready;
  // North data and control
  wire [DATA_TYPE-1 : 0] north_input; 
  wire north_input_valid;
  wire north_input_ready;
  // East data and control
  wire [DATA_TYPE-1 : 0] east_output;
  wire east_output_valid;
  wire east_output_ready;
  // South data and control
  wire [DATA_TYPE-1 : 0] south_output;
  wire south_output_valid;
  wire south_output_ready;
  // Final result data and control
  wire [DATA_TYPE-1 : 0] out;
  wire out_valid;
  wire out_ready;

  // West data and control
  assign west_input = ins[BURST_TYPE-1 : 0];
  assign west_input_valid = ins_valid[0];
  assign ins_ready[0] = west_input_ready;
  // Burst size information
  assign north_input = ins[BURST_TYPE + DATA_TYPE -1 : BURST_TYPE];
  assign north_input_valid= ins_valid[1];
  assign ins_ready[1] = north_input_ready;
  // North data and control
  assign burst_len  = ins[BURST_TYPE + 2*DATA_TYPE -1 : BURST_TYPE + DATA_TYPE];
  assign burst_len_valid = ins_valid[2];
  assign ins_ready[2] = burst_len_ready;
  // Final result data and control
  assign  outs[DATA_TYPE-1 : 0] = out;
  assign  outs_valid[0] = out_valid;
  assign out_ready = outs_ready[0];
  // East data and control  
  assign  outs[2*DATA_TYPE-1 : DATA_TYPE] = east_output;
  assign  outs_valid[1] = east_output_valid;
  assign east_output_ready = outs_ready[1];
  // South data and control
  assign outs[3*DATA_TYPE-1 : 2*DATA_TYPE] = south_output;
  assign outs_valid[2] = south_output_valid;
  assign south_output_ready = outs_ready[2];



  wire [DATA_TYPE-1 : 0] tmp_west_input;
  wire tmp_west_input_valid, tmp_west_input_ready;
  wire [DATA_TYPE-1 : 0] tmp_north_input;
  wire tmp_north_input_valid, tmp_north_input_ready;
  wire [DATA_TYPE-1 : 0] tmp_east_output;
  wire tmp_east_output_valid, tmp_east_output_ready;
  wire [DATA_TYPE-1 : 0] tmp_south_output;
  wire tmp_south_output_valid, tmp_south_output_ready;
  wire [DATA_TYPE-1 : 0] tmp_result;
  wire tmp_result_valid, tmp_result_ready;

  // FIFO between west input and PE
  tfifo #(
    .DATA_TYPE(DATA_TYPE),
    .NUM_SLOTS(BURST_TYPE)
  ) west_fifo (
    .clk        (clk            ),
    .rst        (rst            ),
    .ins        (west_input         ),
    .ins_valid  (west_input_valid   ),
    .ins_ready  (west_input_ready   ),
    .outs       (tmp_west_input        ),
    .outs_valid (tmp_west_input_valid  ),
    .outs_ready (tmp_west_input_ready  )
  );

  // FIFO between north input and PE
  tfifo #(
    .DATA_TYPE(DATA_TYPE),
    .NUM_SLOTS(BURST_TYPE)
  ) north_fifo (
    .clk        (clk            ),
    .rst        (rst            ),
    .ins        (north_input         ),
    .ins_valid  (north_input_valid   ),
    .ins_ready  (north_input_ready   ),
    .outs       (tmp_north_input        ),
    .outs_valid (tmp_north_input_valid  ),
    .outs_ready (tmp_north_input_ready  )
  );

  systolic_pe #(
    .DATA_TYPE (DATA_TYPE),
    .BURST_TYPE(BURST_TYPE)
  ) pe (
    .clk                (clk                   ),
    .rst                (rst                   ),
    .burst_len         (burst_len            ),
    .burst_len_valid   (burst_len_valid      ),
    .burst_len_ready   (burst_len_ready      ),
    .west_input        (tmp_west_input       ),
    .west_input_valid  (tmp_west_input_valid ),
    .west_input_ready  (tmp_west_input_ready ),
    .north_input       (tmp_north_input      ),
    .north_input_valid (tmp_north_input_valid),
    .north_input_ready (tmp_north_input_ready),
    .east_output       (tmp_east_output      ),
    .east_output_valid (tmp_east_output_valid),
    .east_output_ready (tmp_east_output_ready),
    .south_output      (tmp_south_output     ),
    .south_output_valid(tmp_south_output_valid),
    .south_output_ready(tmp_south_output_ready),
    .out               (tmp_result            ),
    .out_valid         (tmp_result_valid      ),
    .out_ready         (tmp_result_ready      )
  );


  // OEHB between PE east output and systolic unit east output
  oehb #(
    .DATA_TYPE(DATA_TYPE)
  ) east_oehb (
    .clk        (clk                ),
    .rst        (rst                ),
    .ins        (tmp_east_output        ),
    .ins_valid  (tmp_east_output_valid  ),
    .ins_ready  (tmp_east_output_ready  ),
    .outs       (east_output          ),
    .outs_valid (east_output_valid    ),
    .outs_ready (east_output_ready    )
  );

  // OEHB between PE south output and systolic unit south output
  oehb #(
    .DATA_TYPE(DATA_TYPE)
  ) south_oehb (
    .clk        (clk                ),
    .rst        (rst                ),
    .ins        (tmp_south_output       ),
    .ins_valid  (tmp_south_output_valid ),
    .ins_ready  (tmp_south_output_ready ),
    .outs       (south_output         ),
    .outs_valid (south_output_valid   ),
    .outs_ready (south_output_ready   )
  );

  // OEHB between PE result output and systolic unit result output
  oehb #(
    .DATA_TYPE(DATA_TYPE)
  ) result_oehb (
    .clk        (clk                ),
    .rst        (rst                ),
    .ins        (tmp_result            ),
    .ins_valid  (tmp_result_valid      ),
    .ins_ready  (tmp_result_ready      ),
    .outs       (out                  ),
    .outs_valid (out_valid            ),
    .outs_ready (out_ready            )
  );

endmodule