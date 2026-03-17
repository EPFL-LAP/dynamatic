from typing import List
import argparse
from sys import argv, stderr


RAM_VERILOG_TEMPLATE = """
`timescale 1ns/1ps
module MODULE_NAME (
  clk,
  rst,
  loadEn,
  loadAddr,
  storeEn,
  storeAddr,
  storeData,
  loadData
);
input clk;
input rst;
input loadEn;
input [ADDR_WIDTH - 1 : 0] loadAddr;
input storeEn;
input [ADDR_WIDTH - 1 : 0] storeAddr;
input [DATA_WIDTH - 1 : 0] storeData;
output [DATA_WIDTH - 1 : 0] loadData;
reg [DATA_WIDTH - 1 : 0] load_data_reg;
reg [DATA_WIDTH - 1 : 0] ram [SIZE - 1 : 0];
initial
begin
INITIAL_BLOCK
end
always@(posedge clk) begin
  if (loadEn) begin
    load_data_reg <= ram[loadAddr];
  end
end

always@(posedge clk) begin
  if (storeEn) begin
    ram[storeAddr] <= storeData;
  end
end
assign loadData = load_data_reg;
endmodule
"""

RAM_VHDL_TEMPLATE = """
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity MODULE_NAME is
  port (
    clk       : in std_logic;
    rst       : in std_logic;
    -- from circuit (mem_controller / LSQ)
    loadEn    : in std_logic;
    loadAddr  : in std_logic_vector(ADDR_WIDTH - 1 downto 0);
    storeEn   : in std_logic;
    storeAddr : in std_logic_vector(ADDR_WIDTH - 1 downto 0);
    storeData : in std_logic_vector(DATA_WIDTH - 1 downto 0);
    -- to circuit (mem_controller / LSQ)
    loadData  : out std_logic_vector(DATA_WIDTH - 1 downto 0)
  );
end entity;

architecture arch of MODULE_NAME is
  type ram_type is array (0 to SIZE - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
  INITIAL_BLOCK
begin
  read_proc : process(clk)
  begin
    if (rising_edge(clk)) then
      if (loadEn = '1') then
        loadData <= ram(to_integer(unsigned(loadAddr)));
      end if;
    end if;
  end process;

  write_proc : process(clk)
  begin
    if (rising_edge(clk)) then
      if (storeEn = '1') then
        ram(to_integer(unsigned(storeAddr))) <= storeData;
      end if;
    end if;
  end process;
end architecture;
"""


# Returns the 2's complement binary representation of integer `n` with the given
# `bitwidth`.
def to_twos_complement(n, bitwidth, addr):
    if n < 0:
        n = (1 << bitwidth) + n
    if n >= (1 << bitwidth) or n < 0:
        raise ValueError(
            f"""
            The memory cannot be correctly instantiated, since the value
            {n} at address {addr} doesn't fit in {bitwidth} bits.
            """)
    return format(n, f'0{bitwidth}b')


def gen_ram(
    module_name: str,
    hdl: str,
    data_width: int,
    addr_width: int,
    size: int,
    init_vals: List[str],
) -> str:

    init_strings = []
    init_str = ""

    assert len(init_vals) <= int(size)

    if hdl == "verilog":
        for id_, val in enumerate(init_vals):
            if (not val.lstrip("+-").isdigit()):
                raise ValueError(
                    f"Initial value {val} at address {id_} is not a number!")
            val_2s_complement = to_twos_complement(
                int(val), int(data_width), id_)
            init_strings.append(
                "ram[" + str(id_) + "] = " + str(data_width)
                + "'b" + val_2s_complement + ";")
        # If some elements are not initialized, fill in the rest with zeroes.
        if len(init_vals) < int(size):
            for _ in range(int(size) - len(init_vals)):
                init_strings.append(
                    "ram[" + str(id_) + "] = " + data_width + "'b0;")
        init_str = "\n".join(init_strings)
    elif hdl == "vhdl":
        init_strings = ["signal ram : ram_type := ("]
        init_items = []

        for id_, val in enumerate(init_vals):
            if (not val.lstrip("+-").isdigit()):
                raise ValueError(
                    f"Initial value {val} at address {id_} is not a number!")
            val_2s_complement = to_twos_complement(
                int(val), int(data_width), id_)
            init_items.append("\"" + val_2s_complement + "\"")

        # If some elements are not initialized, fill in the rest with zeroes.
        if len(init_vals) < int(size):
            for _ in range(int(size) - len(init_vals)):
                init_items.append("\"" + f"{0:0{data_width}b}" + "\"")
        init_strings.append(",\n".join(init_items))
        init_strings.append(");")
        init_str = "\n".join(init_strings)

        if (init_items == []):
            init_str = "signal ram : ram_type\n"
    else:
        raise ValueError("Unknown HDL type!")

    return (
        (RAM_VERILOG_TEMPLATE if hdl == "verilog" else RAM_VHDL_TEMPLATE).replace(
            "MODULE_NAME", module_name)
        .replace("DATA_WIDTH", str(data_width))
        .replace("ADDR_WIDTH", str(addr_width))
        .replace("SIZE", str(size))
        .replace("INITIAL_BLOCK", init_str)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a RAM HDL module."
    )

    parser.add_argument("--module-name", required=True,
                        help="Name of the generated module")
    parser.add_argument("--hdl", required=True,
                        help="Target HDL (e.g., verilog, vhdl)")
    parser.add_argument("--output", required=True,
                        help="Path to output HDL file")
    parser.add_argument("--data-width", type=int,
                        required=True, help="Data bus width")
    parser.add_argument("--addr-width", type=int,
                        required=True, help="Address bus width")
    parser.add_argument("--size", type=int, required=True,
                        help="Number of memory elements")
    parser.add_argument("--values", 
                        help="List of initial values (comma separated, e.g. --values \"1,2,3,4\")",
                        )

    args = parser.parse_args()

    with open(args.output, "w") as f:
        f.write(
            gen_ram(
                args.module_name,
                args.hdl,
                args.data_width,
                args.addr_width,
                args.size,
                args.values.split(","),
            )
        )
