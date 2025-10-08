from generators.handshake.fork import generate_fork
from generators.handshake.buffers.one_slot_break_r import generate_one_slot_break_r
from generators.handshake.merge import generate_merge
from generators.support.signal_manager.utils.entity import generate_entity
from generators.support.signal_manager.utils.forwarding import get_default_extra_signal_value
from generators.support.signal_manager.utils.concat import ConcatLayout
from generators.support.signal_manager.utils.generation import generate_concat_and_handshake, generate_slice_and_handshake
from generators.support.signal_manager.utils.types import ExtraSignals


def generate_control_merge(name, params):
    # Number of data input ports
    size = params["size"]

    data_bitwidth = params["data_bitwidth"]
    index_bitwidth = params["index_bitwidth"]

    # e.g., {"tag0": 8, "spec": 1}
    extra_signals = params["extra_signals"]

    if extra_signals:
        return _generate_control_merge_signal_manager(name, size, index_bitwidth, data_bitwidth, extra_signals)
    elif data_bitwidth == 0:
        return _generate_control_merge_dataless(name, size, index_bitwidth)
    else:
        return _generate_control_merge(name, size, index_bitwidth, data_bitwidth)


def _generate_control_merge(name, size, data_bitwidth, index_bitwidth):
    control_merge_dataless_name = name + "_control_merge_dataless"
    control_merge_dataless = _generate_control_merge_dataless(
        control_merge_dataless_name, size, index_bitwidth)

    control_merge_body = f"""
// Module of control_merge
  module {name}(
    input  clk,
    input  rst,
    // Input Channels
    input  [{size} * ({data_bitwidth}) - 1 : 0] ins,
    input  [{size} - 1 : 0] ins_valid,
    output [{size} - 1 : 0] ins_ready,
    // Data Output Channel
    output [{data_bitwidth} - 1 : 0] outs,
    output outs_valid,
    input  outs_ready,
    // Index Output Channel
    output [{index_bitwidth} - 1 : 0] index,
    output index_valid,
    input  index_ready
  );
    wire [{index_bitwidth} - 1 : 0] index_internal;

    {control_merge_dataless_name} control (
      .clk          (clk            ),
      .rst          (rst            ),
      .ins_valid    (ins_valid      ),
      .ins_ready    (ins_ready      ),
      .outs_valid   (outs_valid     ),
      .outs_ready   (outs_ready     ),
      .index        (index_internal ),
      .index_valid  (index_valid    ),
      .index_ready  (index_ready    )
    );

    assign index = index_internal;

    assign outs = ins[index_internal * {data_bitwidth} +: {data_bitwidth}];
  endmodule

"""

    return control_merge_dataless + control_merge_body


def _generate_control_merge_dataless(name, size, index_bitwidth):

    merge_dataless_name = name + "_merge_dataless"
    merge_dataless = generate_merge(
        merge_dataless_name, {"size": size, "bitwidth": 0})

    one_slot_break_r_name = name + "_one_slot_break_r"
    one_slot_break_r = generate_one_slot_break_r(
        one_slot_break_r_name, {"bitwidth": index_bitwidth})

    fork_dataless_name = name + "_fork_dataless"
    fork_dataless = generate_fork(
        fork_dataless_name, {"size": 2, "bitwidth": 0})

    controll_merge_dataless_body = f"""
// Module of control_merge_dataless
module {name}(
  input  clk,
  input  rst,
  // Input Channels, default 2 inputs
  input  [{size} - 1 : 0] ins_valid,
  output [{size} - 1 : 0] ins_ready,  
  // Data Output Channel
  output outs_valid,
  input  outs_ready,            
  // Index output Channel
  output [{index_bitwidth} - 1 : 0] index,
  output index_valid,
  input  index_ready
);
  wire dataAvailable;
  wire readyToFork;
  wire one_slot_break_rOut_valid;
  wire one_slot_break_rOut_ready;

  reg [{index_bitwidth} - 1 : 0] index_one_slot_break_r;
  integer i;
  reg found;
  always @(ins_valid) begin
    index_one_slot_break_r = {{{index_bitwidth}{{1'b0}}}};
    found = 1'b0;

    for (i = 0; i < {size}; i = i + 1) begin
      if (!found && ins_valid[i]) begin
        index_one_slot_break_r = i[{index_bitwidth} - 1 : 0];
        found = 1'b1; // Set flag to indicate the value has been found
      end
    end
  end

  // Instantiate Merge_dataless
  {merge_dataless_name} merge_ins (
    .clk        (clk          ),
    .rst        (rst          ),
    .ins_valid  (ins_valid    ),
    .ins_ready  (ins_ready    ),
    .outs_valid (dataAvailable),
    .outs_ready (one_slot_break_rOut_ready)
  );

  // Instantiate one_slot_break_r
  {one_slot_break_r_name} one_slot_break_r (
    .clk        (clk          ),
    .rst        (rst          ),
    .ins        (index_one_slot_break_r   ),
    .ins_valid  (dataAvailable),
    .ins_ready  (one_slot_break_rOut_ready),
    .outs       (index        ),
    .outs_valid (one_slot_break_rOut_valid),
    .outs_ready (readyToFork  )
  );

  // Instantiate Fork_dataless
  {fork_dataless_name} fork_dataless (
    .clk        (clk                      ),
    .rst        (rst                      ),
    .ins_valid  (one_slot_break_rOut_valid            ),
    .ins_ready  (readyToFork              ),
    .outs_valid ({{index_valid, outs_valid}}),
    .outs_ready ({{index_ready, outs_ready}})
  );

endmodule
"""

    return merge_dataless + one_slot_break_r + fork_dataless + controll_merge_dataless_body

# TODO: Update CMerge's type constraints and remove this function


def _generate_index_extra_signal_assignments(index_name: str, index_extra_signals: ExtraSignals) -> str:
    """
    Generate VHDL assignments for extra signals on the index port (cmerge).

    Example:
      - assign index_tag0 = "0";
    """

    # TODO: Extra signals on the index port are not tested
    index_extra_signals_list = []
    for signal_name in index_extra_signals:
        index_extra_signals_list.append(
            f"assign {index_name}_{signal_name} = {get_default_extra_signal_value(signal_name)};")
    return "\n  ".join(index_extra_signals_list)


def _generate_control_merge_signal_manager(name, size, index_bitwidth, data_bitwidth, extra_signals):
    # Generate signal manager entity
    entity = generate_entity(
        name,
        [{
            "name": "ins",
            "bitwidth": data_bitwidth,
            "size": size,
            "extra_signals": extra_signals
        }],
        [{
            "name": "index",
            "bitwidth": index_bitwidth,
            # TODO: Extra signals for index port are not tested
            "extra_signals": extra_signals
        }, {
            "name": "outs",
            "bitwidth": data_bitwidth,
            "extra_signals": extra_signals
        }])

    # Layout info for how extra signals are packed into one std_logic_vector
    concat_layout = ConcatLayout(extra_signals)
    extra_signals_bitwidth = concat_layout.total_bitwidth

    inner_name = f"{name}_inner"
    inner = _generate_control_merge(
        inner_name, size, index_bitwidth, extra_signals_bitwidth + data_bitwidth)

    assignments = []

    # Concatenate ins data and extra signals to create ins_inner
    assignments.extend(generate_concat_and_handshake(
        "ins", data_bitwidth, "ins_inner", concat_layout, size))

    # Slice outs_inner data to create outs data and extra signals
    assignments.extend(generate_slice_and_handshake(
        "outs_inner", "outs", data_bitwidth, concat_layout))

    # Assign index extra signals (TODO: Remove this)
    index_extra_signal_assignments = _generate_index_extra_signal_assignments(
        "index", extra_signals)

    architecture = f"""
  // Concat/slice data and extra signals
  {"\n  ".join(assignments)}

  // Assign index extra signals
  {index_extra_signal_assignments}

  // Inner module instance
  {inner_name} inner (
      .clk(clk),
      .rst(rst),
      .ins(ins_inner),
      .ins_valid(ins_inner_valid),
      .ins_ready(ins_inner_ready),
      .outs(outs_inner),
      .outs_valid(outs_inner_valid),
      .outs_ready(outs_inner_ready),
      .index(index),
      .index_valid(index_valid),
      .index_ready(index_ready)
  );

endmodule
"""

    return inner + entity + architecture
