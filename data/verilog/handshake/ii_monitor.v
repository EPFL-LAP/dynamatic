`timescale 1ns/1ps

module ii_monitor #(
  parameter INDEX_WIDTH     = 1,
  parameter LOOP_BACK_INDEX = 0,
  // Nesting depth of the measured loop (1 for a top-level loop) and deepest
  // depth reachable from it through its own descendants (equal to LOOP_DEPTH
  // when the loop is innermost).
  parameter LOOP_DEPTH      = 1,
  parameter LOOP_MAX_DEPTH  = 1
) (
  input  wire                       clk,
  input  wire                       rst,
  // Index channel of the dominating control_merge (all ports read-only).
  input  wire [INDEX_WIDTH - 1 : 0] index,
  input  wire                       index_valid,
  input  wire                       index_ready,
  // An exit channel of the loop (all ports read-only).
  input  wire                       exit_valid,
  input  wire                       exit_ready
);

  integer ii_cycle;
  integer ii_start;
  integer ii_last;
  integer ii_iters;
  reg     ii_measuring;

  // Report a finished measurement window. The iteration count is always
  // printed (including for loops that ran 0 or 1 times); the II is only
  // printed when more than one iteration was seen (otherwise there is no
  // interval to measure and "n/a" is printed instead).
  task report_window(input integer iters, input integer start_c,
                     input integer last_c);
    begin
      if (iters > 1)
        $display("II_INSTRUMENT: loop=%m depth=%0d/%0d II=%f iterations=%0d",
                 LOOP_DEPTH, LOOP_MAX_DEPTH,
                 (last_c - start_c) * 1.0 / (iters - 1), iters);
      else
        $display("II_INSTRUMENT: loop=%m depth=%0d/%0d II=n/a iterations=%0d",
                 LOOP_DEPTH, LOOP_MAX_DEPTH, iters);
    end
  endtask

  always @(posedge clk) begin : ii_monitor_proc
    if (rst) begin
      ii_cycle     = 0;
      ii_measuring = 0;
      ii_start     = 0;
      ii_last      = 0;
      ii_iters     = 0;
    end else begin
      ii_cycle = ii_cycle + 1;

      // Observe the control_merge index channel.
      if (index_valid && index_ready) begin
        if (index != LOOP_BACK_INDEX[INDEX_WIDTH - 1 : 0]) begin
          // Activation from outside the loop (first entry or re-entry).
          if (ii_measuring)
            report_window(ii_iters, ii_start, ii_last);
          ii_measuring = 1;
          ii_start     = ii_cycle;
          ii_last      = ii_cycle;
          ii_iters     = 1;
        end else begin
          // Activation from the loop back-edge.
          if (ii_measuring) begin
            ii_last  = ii_cycle;
            ii_iters = ii_iters + 1;
          end
        end
      end

      // Observe the loop exit channel.
      if (exit_valid && exit_ready) begin
        if (ii_measuring)
          report_window(ii_iters, ii_start, ii_last);
        ii_measuring = 0;
      end
    end
  end

endmodule
