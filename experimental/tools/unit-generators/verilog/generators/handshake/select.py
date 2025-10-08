from generators.support.signal_manager.utils.concat import ConcatLayout
from generators.support.signal_manager.utils.entity import generate_entity
from generators.support.signal_manager.utils.generation import generate_concat_and_handshake, generate_slice_and_handshake, generate_signal_wise_forwarding
from generators.support.signal_manager.utils.internal_signal import create_internal_channel_decl


def generate_select(name, parameters):
    bitwidth = parameters["bitwidth"]
    extra_signals = parameters["extra_signals"]

    if extra_signals:
        return _generate_select_signal_manager(name, bitwidth, extra_signals)
    else:
        return _generate_select(name, bitwidth)


def _generate_select(name, bitwidth):

    antitoken_module_name = name + "_antitokens"

    antitokens = f"""
// Module of {antitoken_module_name} - Antitoken generation logic for handshake select
module {antitoken_module_name} (
  // inputs
  input  clk,
  input  reset,
  input  pvalid1,
  input  pvalid0,
  input  generate_at1,
  input  generate_at0,
  // outputs
  output  kill1,
  output  kill0,
  output  stop_valid
);

  wire reg_in0;
  wire reg_in1;
  reg reg_out0 = 1'b0;
  reg reg_out1 = 1'b0;

  always @(posedge clk) begin
    if (reset) begin
      reg_out0 <= 1'b0;
      reg_out1 <= 1'b0;
    end else begin
      reg_out0 <= reg_in0;
      reg_out1 <= reg_in1;
    end
  end

  assign reg_in0 = !pvalid0 & (generate_at0 | reg_out0);
  assign reg_in1 = !pvalid1 & (generate_at1 | reg_out1);

  assign stop_valid = reg_out0 | reg_out1;

  assign kill0 = generate_at0 | reg_out0;
  assign kill1 = generate_at1 | reg_out1;

endmodule
"""

    selector = f"""
// Module of select
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

  wire ee, validInternal, kill0, kill1, antitokenStop, g0, g1;

  // condition and one input
  assign ee = condition_valid & ( ( !condition & falseValue_valid) | (condition & trueValue_valid) );
  // propagate ee if not stopped antitoken
  assign validInternal = ee & !antitokenStop;

  assign g0 = !trueValue_valid & validInternal & result_ready;
  assign g1 = !falseValue_valid & validInternal & result_ready;

  assign result_valid = validInternal;
  assign trueValue_ready = (validInternal & result_ready) | kill0; // normal join or antitoken
  assign falseValue_ready = (validInternal & result_ready) | kill1; // normal join or antitoken
  assign condition_ready = (validInternal & result_ready); // normal join

  assign result = condition ? trueValue : falseValue;

  {antitoken_module_name} antitokens (
    .clk(clk),
    .reset(rst),
    .pvalid0(trueValue_valid),
    .pvalid1(falseValue_valid),
    .generate_at0(g0),
    .generate_at1(g1),
    .kill0(kill0),
    .kill1(kill1),
    .stop_valid(antitokenStop)
  );

endmodule
"""
    return antitokens + selector


def _generate_concat(bitwidth: int, concat_layout: ConcatLayout):
    concat_decls = []
    concat_assignments = []

    # Declare trueValue_inner and falseValue_inner channels
    # Example:
    # wire [32:0] trueValue_inner;
    # wire trueValue_inner_valid;
    # wire trueValue_inner_ready;
    concat_decls.extend(create_internal_channel_decl({
        "name": "trueValue_inner",
        "bitwidth": bitwidth + concat_layout.total_bitwidth
    }))
    # Example:
    # wire [32:0] falseValue_inner;
    # wire falseValue_inner_valid;
    # wire falseValue_inner_ready;
    concat_decls.extend(create_internal_channel_decl({
        "name": "falseValue_inner",
        "bitwidth": bitwidth + concat_layout.total_bitwidth
    }))

    # Concatenate trueValue data and extra signals to create trueValue_inner
    # Example:
    # assign trueValue_inner[32-1:0] = trueValue;
    # assign trueValue_inner[32]   = trueValue_spec;
    # assign trueValue_inner_valid  = trueValue_valid;
    # assign trueValue_ready         = trueValue_inner_ready;
    concat_assignments.extend(generate_concat_and_handshake(
        "trueValue", bitwidth, "trueValue_inner", concat_layout))

    # Concatenate falseValue data and extra signals to create falseValue_inner
    # Example:
    # assign falseValue_inner[32-1:0] = falseValue;
    # assign falseValue_inner[32:32]   = falseValue_spec;
    # assign falseValue_inner_valid  = falseValue_valid;
    # assign falseValue_ready         = falseValue_inner_ready;
    concat_assignments.extend(generate_concat_and_handshake(
        "falseValue", bitwidth, "falseValue_inner", concat_layout))

    return concat_assignments, concat_decls


def _generate_slice(bitwidth: int, concat_layout: ConcatLayout):
    slice_decls = []
    slice_assignments = []

    # Declare both result_inner_concat and result_inner channels
    # Example:
    # wire [32:0] result_inner_concat;
    # wire result_inner_concat_valid;
    # wire result_inner_concat_ready;
    slice_decls.extend(create_internal_channel_decl({
        "name": "result_inner_concat",
        "bitwidth": bitwidth + concat_layout.total_bitwidth
    }))
    # Example:
    # wire [31:0] result_inner;
    # wire result_inner_valid;
    # wire result_inner_ready;
    # wire [0:0] result_inner_spec;
    slice_decls.extend(create_internal_channel_decl({
        "name": "result_inner",
        "bitwidth": bitwidth,
        "extra_signals": concat_layout.extra_signals
    }))

    # Slice result_inner_concat to create result_inner data and extra signals
    # Example:
    # assign result_inner       = result_inner_concat[32-1:0];
    # assign result_inner_spec  = result_inner_concat[32];
    # assign result_inner_valid = result_inner_concat_valid;
    # assign result_inner_concat_ready = result_inner_ready;
    slice_assignments.extend(generate_slice_and_handshake(
        "result_inner_concat", "result_inner", bitwidth, concat_layout))

    return slice_assignments, slice_decls


def _generate_select_signal_manager(name, bitwidth, extra_signals):
    # Layout info for how extra signals are packed into one std_logic_vector
    concat_layout = ConcatLayout(extra_signals)
    extra_signals_total_bitwidth = concat_layout.total_bitwidth

    inner_name = f"{name}_inner"
    inner = _generate_select(inner_name, bitwidth +
                             extra_signals_total_bitwidth)

    entity = generate_entity(name, [{
        "name": "condition",
        "bitwidth": 1,
        "extra_signals": extra_signals
    }, {
        "name": "trueValue",
        "bitwidth": bitwidth,
        "extra_signals": extra_signals
    }, {
        "name": "falseValue",
        "bitwidth": bitwidth,
        "extra_signals": extra_signals
    }], [{
        "name": "result",
        "bitwidth": bitwidth,
        "extra_signals": extra_signals
    }])

    concat_assignments, concat_decls = _generate_concat(
        bitwidth, concat_layout)
    slice_assignments, slice_decls = _generate_slice(
        bitwidth, concat_layout)

    forwarding_assignments = []
    # Signal-wise forwarding of extra signals from condition and result_inner to result
    # Example: result_spec <= condition_spec or result_inner_spec;
    for signal_name in extra_signals:
        forwarding_assignments.extend(generate_signal_wise_forwarding(
            ["condition", "result_inner"], ["result"], signal_name))

    architecture = f"""
  // Declarations
  {"\n  ".join(concat_decls)}
  {"\n  ".join(slice_decls)}

  // Concatenate extra signals
  {"\n  ".join(concat_assignments)}
  {"\n  ".join(slice_assignments)}

  // Forwarding logic
  {"\n  ".join(forwarding_assignments)}

  assign result             = result_inner;
  assign result_valid       = result_inner_valid;
  assign result_inner_ready = result_ready;

  // Inner module instance
  {inner_name} inner (
      .clk(clk),
      .rst(rst),
      .condition(condition),
      .condition_valid(condition_valid),
      .condition_ready(condition_ready),
      .trueValue(trueValue_inner),
      .trueValue_valid(trueValue_inner_valid),
      .trueValue_ready(trueValue_inner_ready),
      .falseValue(falseValue_inner),
      .falseValue_valid(falseValue_inner_valid),
      .falseValue_ready(falseValue_inner_ready),
      .result(result_inner_concat),
      .result_valid(result_inner_concat_valid),
      .result_ready(result_inner_concat_ready)
  );

endmodule
"""

    return inner + entity + architecture
