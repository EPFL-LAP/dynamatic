
from generators.handshake.buffers.one_slot_break_r import generate_one_slot_break_r
from generators.handshake.buffers.fifo_break_dv import generate_fifo_break_dv


def generate_load(name, params):
    addr_bitwidth = params["addr_bitwidth"]
    data_bitwidth = params["data_bitwidth"]
    
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
