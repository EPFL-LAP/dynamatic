from generators.handshake.tehb import generate_tehb

def generate_load(name, params):
    data_type = params["data_type"]
    addr_type = params["addr_type"]

    header = "`timescale 1ns/1ps\n"

    tehb_address_name = name + "_tehb_address"
    tehb_address = generate_tehb(tehb_address_name, { "data_type": addr_type })

    tehb_data_name = name + "_tehb_data"
    tehb_data = generate_tehb(tehb_data_name, { "data_type": data_type })

    load_body = f"""

// Module of Load
module {name} #(
  parameter DATA_TYPE = {data_type},
  parameter ADDR_TYPE = {addr_type}
)(
  input clk,
  input rst,
  // Address from Circuit Channel
  input  [{addr_type} - 1 : 0] addrIn,
  input  addrIn_valid,
  output addrIn_ready,
  // Address to Interface Channel
  output [{addr_type} - 1 : 0] addrOut,
  output addrOut_valid,
  input  addrOut_ready,
  // Data from Interface Channel
  input  [{data_type} - 1 : 0] dataFromMem,
  input  dataFromMem_valid,
  output dataFromMem_ready,
  // Data from Memory Channel
  output [{data_type} - 1 : 0] dataOut,
  output dataOut_valid,
  input  dataOut_ready
);
  {tehb_address_name} addr_tehb (
    .clk        (clk            ),
    .rst        (rst            ),
    .ins        (addrIn         ),
    .ins_valid  (addrIn_valid   ),
    .ins_ready  (addrIn_ready   ),
    .outs       (addrOut        ),
    .outs_valid (addrOut_valid  ),
    .outs_ready (addrOut_ready  )
  );

  {tehb_data_name} data_tehb (
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

    return header + tehb_address + tehb_data +load_body