`timescale 1ns/1ps
module ndwire_dataless (
  input  clk,
  input  rst,
  // Input channel
  input  ins_valid,
  output ins_ready,
  // Output channel
  output outs_valid,
  input  outs_ready
);

  typedef enum logic {SLEEPING, RUNNING} nd_state_t;

  // This is the source of non-determism.
  // It needs to be set to a primary input in a formal tool
  // If the formal tools does not implicitly treat undriven signals
  // like primary inputs this needs to be done explicitly.
  nd_state_t nd_next_state;

  nd_state_t state, next_state;


  always @* begin
    // If the wire is sleeping it can always switch to the running state.
    // If (ins_valid && outs_ready) we either have a transaction
    // and can freely choose the state again.
    if(state == SLEEPING) begin
      next_state = nd_next_state;
    end else if (ins_valid && outs_ready) begin
      next_state = nd_next_state;
    end else begin
      next_state = state;
    end
  end

  // The initialization of the state is non-deterministic
  always @(posedge clk or posedge rst) begin
    if (rst) begin
      state <= nd_next_state;
    end else begin
      state <= next_state;
    end
  end


  assign ins_ready  = outs_ready && (state == RUNNING);
  assign outs_valid = ins_valid  && (state == RUNNING);

endmodule
