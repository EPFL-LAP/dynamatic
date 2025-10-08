from generators.support.signal_manager.utils.entity import generate_entity
from generators.support.signal_manager.utils.concat import ConcatLayout
from generators.support.signal_manager.utils.generation import generate_concat_and_handshake, generate_slice_and_handshake, generate_signal_wise_forwarding
from generators.support.signal_manager.utils.internal_signal import create_internal_channel_decl
from generators.support.signal_manager.utils.types import ExtraSignals


def generate_mux(name, params):
    # Number of data input ports
    size = params["size"]

    data_bitwidth = params["data_bitwidth"]
    index_bitwidth = params["index_bitwidth"]

    # e.g., {"tag0": 8, "spec": 1}
    extra_signals = params["extra_signals"]

    if extra_signals:
        return _generate_mux_signal_manager(name, size, index_bitwidth, data_bitwidth, extra_signals)
    elif data_bitwidth == 0:
        return _generate_mux_dataless(name, size, index_bitwidth)
    else:
        return _generate_mux(name, size, index_bitwidth, data_bitwidth)


def _generate_mux(name, size, index_bitwidth, data_bitwidth):

    _mux_dataless = f"""
// Module of smux
module {name}(
  input  clk,
  input  rst,
  // Data input channels
  input  [({size} * {data_bitwidth}) - 1 : 0] ins, 
  input  [{size} - 1 : 0] ins_valid,
  output reg [{size} - 1 : 0] ins_ready,
  // Index input channel
  input  [{index_bitwidth} - 1 : 0] index,
  input  index_valid,
  output index_ready,
  // Output channel
  output [{data_bitwidth} - 1 : 0] outs,
  output outs_valid,
  input  outs_ready
);
  reg [{data_bitwidth} - 1 : 0] selectedData;
  reg selectedData_valid;

  integer i;
  always @(*) begin
    if(rst) begin
      selectedData_valid = 0;
      for (i = {data_bitwidth} - 1; i >= 0; i = i - 1) begin
        selectedData[i] = 0;
      end
      for (i = {size} - 1; i >= 0; i = i - 1) begin
        ins_ready[i] = 0;
      end
    end else begin
      selectedData = ins[0 * {data_bitwidth} +: {data_bitwidth}];
      selectedData_valid = 0;

      for (i = {size} - 1; i >= 0; i = i - 1) begin
        if (((i[{index_bitwidth} - 1 : 0] == index) & index_valid & outs_ready & ins_valid[i]) | ~ins_valid[i]) begin
          ins_ready[i] = 1;
        end else begin
          ins_ready[i] = 0;
        end

        if (index == i[{index_bitwidth} - 1 : 0] && index_valid && ins_valid[i]) begin
          selectedData = ins[i * {data_bitwidth} +: {data_bitwidth}];
          selectedData_valid = 1;
        end
      end
    end 

  end

  assign index_ready = ~index_valid | (selectedData_valid & outs_ready);
  assign outs = selectedData;
  assign outs_valid = selectedData_valid;

endmodule
"""
    return _mux_dataless


def _generate_mux_dataless(name, size, index_bitwidth):

    _mux_dataless = f"""
// Module of_mux_dataless
module {name}(
  input  clk,
  input  rst,
  // Data input channels
  input  [{size} - 1 : 0] ins_valid,
  output reg [{size} - 1 : 0] ins_ready,
  // Index input channel
  input  [{index_bitwidth} - 1 : 0] index,
  input  index_valid,
  output index_ready,
  // Output channel
  output outs_valid,
  input  outs_ready
);
  reg selectedData_valid;
  integer i;

  always @(*) begin
    selectedData_valid = 0;

    for (i = {size} - 1; i >= 0 ; i = i - 1) begin
      if (((i[{index_bitwidth} - 1 : 0] == index) & index_valid & ins_valid[i] & outs_ready) | ~ins_valid[i])
        ins_ready[i] = 1;
      else
        ins_ready[i] = 0;

      if (index == i[{index_bitwidth} - 1 : 0] && index_valid && ins_valid[i]) begin
        selectedData_valid = 1;
      end    
    end
  end

  assign outs_valid = selectedData_valid;
  assign index_ready = ~index_valid | (selectedData_valid & outs_ready);

endmodule
"""

    return _mux_dataless


def _generate_concat(data_bitwidth: int, concat_layout: ConcatLayout, size: int) -> tuple[str, str]:
    concat_decls = []
    concat_assignments = []

    # Declare ins_inner channel
    # Example:
    # wire [32:0] ins_inner [1:0];
    # wire [1:0] ins_inner_valid;
    # wire [1:0] ins_inner_ready;
    concat_decls.extend(create_internal_channel_decl({
        "name": "ins_inner",
        "bitwidth": data_bitwidth + concat_layout.total_bitwidth,
        "size": size
    }))

    # Concatenate ins data and extra signals to create ins_inner
    # Example:
    # assign ins_inner[0][31:0] = ins[0];
    # assign ins_inner[0][32]   = ins_0_spec;
    # assign ins_inner[1][31:0] = ins[1];
    # assign ins_inner[1][32]   = ins_1_spec;
    # assign ins_inner_valid     = ins_valid;
    # assign ins_ready           = ins_inner_ready;
    concat_assignments.extend(generate_concat_and_handshake(
        "ins", data_bitwidth, "ins_inner", concat_layout, size))

    return "\n  ".join(concat_assignments), "\n  ".join(concat_decls)


def _generate_slice(data_bitwidth: int, concat_layout: ConcatLayout) -> tuple[str, str]:
    slice_decls = []
    slice_assignments = []

    # Declare both outs_inner_concat and outs_inner channels
    # Example:
    # wire [32:0] outs_inner_concat;
    # wire outs_inner_concat_valid;
    # wire outs_inner_concat_ready;
    slice_decls.extend(create_internal_channel_decl({
        "name": "outs_inner_concat",
        "bitwidth": data_bitwidth + concat_layout.total_bitwidth
    }))

    # Example:
    # wire [31:0] outs_inner;
    # wire outs_inner_valid;
    # wire outs_inner_ready;
    # wire [0:0] outs_inner_spec;
    slice_decls.extend(create_internal_channel_decl({
        "name": "outs_inner",
        "bitwidth": data_bitwidth,
        "extra_signals": concat_layout.extra_signals
    }))

    # Slice outs_inner_concat to create outs_inner data and extra signals
    # Example:
    # assign outs_inner       = outs_inner_concat[32-1:0];
    # assign outs_inner_spec  = outs_inner_concat[32];
    # assign outs_inner_valid = outs_inner_concat_valid;
    # assign outs_inner_concat_ready = outs_inner_ready;

    slice_assignments.extend(generate_slice_and_handshake(
        "outs_inner_concat", "outs_inner", data_bitwidth, concat_layout))

    return "\n  ".join(slice_assignments), "\n  ".join(slice_decls)


def _generate_forwarding(extra_signals: ExtraSignals) -> str:
    forwarding_assignments = []

    # Signal-wise forwarding of extra signals from ins_inner and outs_inner to outs
    # Example:
    # outs_spec <= index_spec or outs_inner_spec;
    for signal_name in extra_signals:
        forwarding_assignments.extend(generate_signal_wise_forwarding(
            ["index", "outs_inner"], ["outs"], signal_name))

    return "\n  ".join(forwarding_assignments)


def _generate_mux_signal_manager(name, size, index_bitwidth, data_bitwidth, extra_signals):
    # Generate signal manager entity
    entity = generate_entity(
        name,
        [{
            "name": "ins",
            "bitwidth": data_bitwidth,
            "size": size,
            "extra_signals": extra_signals
        }, {
            "name": "index",
            "bitwidth": index_bitwidth,
            # TODO: Extra signals for index port are not tested
            "extra_signals": extra_signals
        }],
        [{
            "name": "outs",
            "bitwidth": data_bitwidth,
            "extra_signals": extra_signals
        }]
    )

    # Layout info for how extra signals are packed into one std_logic_vector
    concat_layout = ConcatLayout(extra_signals)
    extra_signals_bitwidth = concat_layout.total_bitwidth

    inner_name = f"{name}_inner"
    inner = _generate_mux(inner_name, size, index_bitwidth,
                          extra_signals_bitwidth + data_bitwidth)

    concat_assignments, concat_decls = _generate_concat(
        data_bitwidth, concat_layout, size)
    slice_assignments, slice_decls = _generate_slice(
        data_bitwidth, concat_layout)
    forwarding_assignments = _generate_forwarding(extra_signals)

    architecture = f"""
  // Declarations
  {concat_decls}
  {slice_decls}

  // Concatenate data and extra signals
  {concat_assignments}
  {slice_assignments}

  // Forwarding logic
  {forwarding_assignments}

  // Connect inner outputs to module outputs
  assign outs       = outs_inner;
  assign outs_valid = outs_inner_valid;
  assign outs_inner_ready = outs_ready;

  // Inner module instance
  {inner_name} inner (
      .clk(clk),
      .rst(rst),
      .ins(ins_inner),
      .ins_valid(ins_inner_valid),
      .ins_ready(ins_inner_ready),
      .index(index),
      .index_valid(index_valid),
      .index_ready(index_ready),
      .outs(outs_inner_concat),
      .outs_valid(outs_inner_concat_valid),
      .outs_ready(outs_inner_concat_ready)
  );
endmodule
"""

    return inner + entity + architecture
