
def generate_select(name, parameters):
    bitwidth = parameters["bitwidth"]
    # Maximum number of tokens that may be owed to be killed on one data input,
    # i.e., how far one data input may run ahead of the other. With 0, the
    # select waits for the condition and both data inputs (join).
    antitoken_depth = parameters.get("antitoken_depth", 1)
    if antitoken_depth < 0:
        raise ValueError(
            f"antitoken_depth must be non-negative, got {antitoken_depth}")
    if antitoken_depth == 0:
        return _generate_join_select(name, bitwidth)
    return _generate_select(name, bitwidth, antitoken_depth)


def _generate_select(name, bitwidth, antitoken_depth):
    cnt_bitwidth = antitoken_depth.bit_length()

    selector = f"""
// Module of select
// trueValue is selected when condition = 1, falseValue when condition = 0.
// When the selector fires with one value while the token of the other value has
// not arrived yet, that token is owed to be killed when it arrives. Kills are
// owed to at most one of trueValue and falseValue at a time: a condition
// selecting a value waits at the head of the condition channel until all kills
// owed to that value are done, so the next token of that value is the one it
// selects.
//   NORMAL     : no kill owed (cnt = 0)
//   KILL_TRUE  : cnt trueValue tokens owed to be killed, falseValue may run ahead
//   KILL_FALSE : cnt falseValue tokens owed to be killed, trueValue may run ahead
module {name}(
  // inputs
  input  clk,
  input  rst,
  input  condition,
  input  condition_valid,
  input  [{bitwidth}-1 : 0] trueValue,
  input  trueValue_valid,
  input  [{bitwidth}-1 : 0] falseValue,
  input  falseValue_valid,
  input  result_ready,
  // outputs
  output  [{bitwidth}-1 : 0] result,
  output  result_valid,
  output  condition_ready,
  output  trueValue_ready,
  output  falseValue_ready
);

  localparam NORMAL = 2'd0, KILL_TRUE = 2'd1, KILL_FALSE = 2'd2;

  reg [1 : 0] state = NORMAL;
  reg [{cnt_bitwidth}-1 : 0] cnt = 0;

  wire killTrueOwed, killFalseOwed, full;
  wire selTrue, selFalse, canTrue, canFalse, fireTrue, fireFalse, discardTrue, discardFalse;

  assign killTrueOwed = (state == KILL_TRUE);
  assign killFalseOwed = (state == KILL_FALSE);
  assign full = (cnt == {antitoken_depth});

  assign selTrue = condition_valid & condition;
  assign selFalse = condition_valid & !condition;

  // Select a value if its next token is not owed a kill, and if firing does not
  // add a kill to the other value when that one is full
  assign canTrue = selTrue & trueValue_valid & !killTrueOwed &
                   !(killFalseOwed & full & !falseValue_valid);
  assign canFalse = selFalse & falseValue_valid & !killFalseOwed &
                    !(killTrueOwed & full & !trueValue_valid);

  assign fireTrue = canTrue & result_ready;
  assign fireFalse = canFalse & result_ready;

  // Kill an arriving token if a kill is owed to its value, or if the other value
  // is selected in the same cycle (normal join)
  assign discardTrue = trueValue_valid & (killTrueOwed | fireFalse);
  assign discardFalse = falseValue_valid & (killFalseOwed | fireTrue);

  assign result_valid = canTrue | canFalse;
  assign trueValue_ready = !trueValue_valid | fireTrue | discardTrue;
  assign falseValue_ready = !falseValue_valid | fireFalse | discardFalse;
  assign condition_ready = !condition_valid | fireTrue | fireFalse;

  assign result = condition ? trueValue : falseValue;

  // State and counter update:
  //   - NORMAL -> KILL_TRUE with cnt = 1 when falseValue is selected and no
  //     trueValue token is present to kill in the same cycle.
  //   - KILL_TRUE: cnt + 1 when falseValue is selected and no trueValue token
  //     is present, cnt - 1 when a trueValue token is killed and falseValue is
  //     not selected, unchanged otherwise. Back to NORMAL when cnt reaches 0.
  //   - NORMAL -> KILL_FALSE and KILL_FALSE are symmetric.
  always @(posedge clk) begin
    if (rst) begin
      state <= NORMAL;
      cnt <= 0;
    end else begin
      case (state)
        NORMAL: begin
          if (fireFalse & !trueValue_valid) begin
            state <= KILL_TRUE;
            cnt <= 1;
          end else if (fireTrue & !falseValue_valid) begin
            state <= KILL_FALSE;
            cnt <= 1;
          end
        end
        KILL_TRUE: begin
          if (fireFalse & !trueValue_valid) begin
            cnt <= cnt + 1;
          end else if (!fireFalse & trueValue_valid) begin
            cnt <= cnt - 1;
            // cnt still holds the old value: the last owed token is killed
            // and cnt becomes 0 together with the move to NORMAL
            if (cnt == 1)
              state <= NORMAL;
          end
        end
        KILL_FALSE: begin
          if (fireTrue & !falseValue_valid) begin
            cnt <= cnt + 1;
          end else if (!fireTrue & falseValue_valid) begin
            cnt <= cnt - 1;
            // cnt still holds the old value: the last owed token is killed
            // and cnt becomes 0 together with the move to NORMAL
            if (cnt == 1)
              state <= NORMAL;
          end
        end
        default: begin
          state <= NORMAL;
          cnt <= 0;
        end
      endcase
    end
  end

endmodule
"""
    return selector


def _generate_join_select(name, bitwidth):
    return f"""
// Module of select with antitoken depth 0
// The selector waits for the condition and both data inputs, and consumes all
// three when the result is produced (join): no token is killed afterwards.
module {name}(
  // inputs
  input  clk,
  input  rst,
  input  condition,
  input  condition_valid,
  input  [{bitwidth}-1 : 0] trueValue,
  input  trueValue_valid,
  input  [{bitwidth}-1 : 0] falseValue,
  input  falseValue_valid,
  input  result_ready,
  // outputs
  output  [{bitwidth}-1 : 0] result,
  output  result_valid,
  output  condition_ready,
  output  trueValue_ready,
  output  falseValue_ready
);

  wire allValid, fire;

  assign allValid = condition_valid & trueValue_valid & falseValue_valid;
  assign fire = allValid & result_ready;

  assign result_valid = allValid;
  assign trueValue_ready = !trueValue_valid | fire;
  assign falseValue_ready = !falseValue_valid | fire;
  assign condition_ready = !condition_valid | fire;

  assign result = condition ? trueValue : falseValue;

endmodule
"""
