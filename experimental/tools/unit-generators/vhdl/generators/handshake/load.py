import ast

from generators.support.utils import VhdlScalarType
from generators.handshake.tehb import generate_tehb
from generators.handshake.tfifo import generate_tfifo

def generate_load(name, params):
  port_types = ast.literal_eval(params["port_types"])

  # Ports communicating with the elastic circuit have the complete and same extra signals
  data_type = VhdlScalarType(port_types["dataOut"])
  addr_type = VhdlScalarType(port_types["addrIn"])

  return _generate_load(name, data_type.bitwidth, addr_type.bitwidth)

def _generate_load(name, data_bitwidth, addr_bitwidth):
  addr_tehb_name = f"{name}_addr_tehb"
  data_tehb_name = f"{name}_data_tehb"

  dependencies = \
    generate_tehb(addr_tehb_name, {
      "port_types": str({
        "ins": f"!handshake.channel<i{addr_bitwidth}>",
        "outs": f"!handshake.channel<i{addr_bitwidth}>"
      })
    }) + \
    generate_tehb(data_tehb_name, {
      "port_types": str({
        "ins": f"!handshake.channel<i{data_bitwidth}>",
        "outs": f"!handshake.channel<i{data_bitwidth}>"
      })
    })

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of load
entity {name} is
  port (
    clk, rst : in std_logic;
    -- address from circuit channel
    addrIn       : in  std_logic_vector({addr_bitwidth} - 1 downto 0);
    addrIn_valid : in  std_logic;
    addrIn_ready : out std_logic;
    -- address to interface channel
    addrOut       : out std_logic_vector({addr_bitwidth} - 1 downto 0);
    addrOut_valid : out std_logic;
    addrOut_ready : in  std_logic;
    -- data from interface channel
    dataFromMem       : in  std_logic_vector({data_bitwidth} - 1 downto 0);
    dataFromMem_valid : in  std_logic;
    dataFromMem_ready : out std_logic;
    -- data from memory channel
    dataOut       : out std_logic_vector({data_bitwidth} - 1 downto 0);
    dataOut_valid : out std_logic;
    dataOut_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of load
architecture arch of {name} is
begin
  addr_tehb : entity work.{addr_tehb_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      -- input channel
      ins       => addrIn,
      ins_valid => addrIn_valid,
      ins_ready => addrIn_ready,
      -- output channel
      outs       => addrOut,
      outs_valid => addrOut_valid,
      outs_ready => addrOut_ready
    );

  data_tehb : entity work.{data_tehb_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      -- input channel
      ins       => dataFromMem,
      ins_valid => dataFromMem_valid,
      ins_ready => dataFromMem_ready,
      -- output channel
      outs       => dataOut,
      outs_valid => dataOut_valid,
      outs_ready => dataOut_ready
    );
end architecture;
"""

  return dependencies + entity + architecture
