from generators.support.signal_manager.utils.entity import generate_entity
from generators.support.signal_manager.utils.concat import ConcatLayout
from generators.support.signal_manager.utils.generation import generate_concat, generate_slice

from generators.handshake.buffers.one_slot_break_r import generate_one_slot_break_r
from generators.handshake.buffers.fifo_break_dv import generate_fifo_break_dv


def generate_load(name, params):
    addr_bitwidth = params["addr_bitwidth"]
    data_bitwidth = params["data_bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_load_signal_manager(name, data_bitwidth, addr_bitwidth, extra_signals)
    else:
        return _generate_load(name, data_bitwidth, addr_bitwidth)


def _generate_load(name, data_bitwidth, addr_bitwidth):

    one_slot_break_r_address_name = name + "_one_slot_break_r_address"
    one_slot_break_r_address = generate_one_slot_break_r(
        one_slot_break_r_address_name, {"bitwidth": addr_bitwidth})

    one_slot_break_r_data_name = name + "_one_slot_break_r_data"
    one_slot_break_r_data = generate_one_slot_break_r(
        one_slot_break_r_data_name, {"bitwidth": data_bitwidth})

    load_body = f"""

// Module of Load
module {name} #(
  parameter data_bitwidth = {data_bitwidth},
  parameter addr_bitwidth = {addr_bitwidth}
)(
  input clk,
  input rst,
  // Address from Circuit Channel
  input  [{addr_bitwidth} - 1 : 0] addrIn,
  input  addrIn_valid,
  output addrIn_ready,
  // Address to Interface Channel
  output [{addr_bitwidth} - 1 : 0] addrOut,
  output addrOut_valid,
  input  addrOut_ready,
  // Data from Interface Channel
  input  [{data_bitwidth} - 1 : 0] dataFromMem,
  input  dataFromMem_valid,
  output dataFromMem_ready,
  // Data from Memory Channel
  output [{data_bitwidth} - 1 : 0] dataOut,
  output dataOut_valid,
  input  dataOut_ready
);
  {one_slot_break_r_address_name} addr_one_slot_break_r (
    .clk        (clk            ),
    .rst        (rst            ),
    .ins        (addrIn         ),
    .ins_valid  (addrIn_valid   ),
    .ins_ready  (addrIn_ready   ),
    .outs       (addrOut        ),
    .outs_valid (addrOut_valid  ),
    .outs_ready (addrOut_ready  )
  );

  {one_slot_break_r_data_name} data_one_slot_break_r (
    .clk        (clk                ),
    .rst        (rst                ),
    .ins        (dataFromMem        ),
    .ins_valid  (dataFromMem_valid  ),
    .ins_ready  (dataFromMem_ready  ),
    .outs       (dataOut            ),
    .outs_valid (dataOut_valid      ),
    .outs_ready (dataOut_ready      )
  );

endmodule
"""

    return one_slot_break_r_address + one_slot_break_r_data + load_body


def _generate_load_signal_manager(name, data_bitwidth, addr_bitwidth, extra_signals):
    # Concatenate extra signals and store them in a dedicated FIFO

    # Get concatenation details for extra signals
    concat_layout = ConcatLayout(extra_signals)
    extra_signals_total_bitwidth = concat_layout.total_bitwidth

    inner_name = f"{name}_inner"
    inner = _generate_load(inner_name, data_bitwidth, addr_bitwidth)

    # Generate fifo_break_dv to store extra signals for in-flight memory requests
    fifo_break_dv_name = f"{name}_fifo_break_dv"
    fifo_break_dv = generate_fifo_break_dv(fifo_break_dv_name, {
        "bitwidth": extra_signals_total_bitwidth,
        "num_slots": 1  # Assume LoadOp is connected to a memory controller
    })

    entity = generate_entity(name, [{
        "name": "addrIn",
        "bitwidth": addr_bitwidth,
        "extra_signals": extra_signals
    }, {
        "name": "dataFromMem",
        "bitwidth": data_bitwidth,
        "extra_signals": {}
    }], [{
        "name": "addrOut",
        "bitwidth": addr_bitwidth,
        "extra_signals": {}
    }, {
        "name": "dataOut",
        "bitwidth": data_bitwidth,
        "extra_signals": extra_signals
    }])

    assignments = []

    # Concatenate addrIn extra signals to create signals_pre_buffer
    assignments.extend(generate_concat(
        "addrIn", 0, "signals_pre_buffer", concat_layout))

    # Slice signals_post_buffer to create dataOut data and extra signals
    assignments.extend(generate_slice(
        "signals_post_buffer", "dataOut", 0, concat_layout))

    architecture = f"""
  wire [{concat_layout.total_bitwidth}-1:0] signals_pre_buffer;
  wire [{concat_layout.total_bitwidth}-1:0] signals_post_buffer;
  wire transfer_in;
  wire transfer_out;

  // Transfer signal assignments
  assign transfer_in  = addrIn_valid & addrIn_ready;
  assign transfer_out = dataOut_valid & dataOut_ready;

  // Concat/slice extra signals
  {"\n  ".join(assignments)}

  // Buffer to store extra signals for in-flight memory requests
  // LoadOp is assumed to be connected to a memory controller
  // Use fifo_break_dv with latency 1 (MC latency)
  {fifo_break_dv_name} fifo_break_dv (
      .clk(clk),
      .rst(rst),
      .ins(signals_pre_buffer),
      .ins_valid(transfer_in),
      .ins_ready(),
      .outs(signals_post_buffer),
      .outs_valid(),
      .outs_ready(transfer_out)
  );

  // Inner module instance
  {inner_name} inner (
      .clk(clk),
      .rst(rst),
      .addrIn(addrIn),
      .addrIn_valid(addrIn_valid),
      .addrIn_ready(addrIn_ready),
      .addrOut(addrOut),
      .addrOut_valid(addrOut_valid),
      .addrOut_ready(addrOut_ready),
      .dataFromMem(dataFromMem),
      .dataFromMem_valid(dataFromMem_valid),
      .dataFromMem_ready(dataFromMem_ready),
      .dataOut(dataOut),
      .dataOut_valid(dataOut_valid),
      .dataOut_ready(dataOut_ready)
  );

endmodule
"""

    return inner + fifo_break_dv + entity + architecture
