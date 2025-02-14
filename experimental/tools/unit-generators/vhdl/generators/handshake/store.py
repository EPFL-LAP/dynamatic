import ast

from generators.support.utils import VhdlScalarType, generate_extra_signal_ports, ExtraSignalMapping, generate_ins_concat_statements_dataless, generate_outs_concat_statements_dataless

def generate_store(name, params):
  port_types = ast.literal_eval(params["port_types"])

  # Ports communicating with the elastic circuit have the complete and same extra signals
  data_type = VhdlScalarType(port_types["dataIn"])
  addr_type = VhdlScalarType(port_types["addrIn"])

  if data_type.has_extra_signals():
    return _generate_store_signal_manager(name, data_type, addr_type)
  else:
    return _generate_store(name, data_type.bitwidth, addr_type.bitwidth)

def _generate_store(name, data_bitwidth, addr_bitwidth):
  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of store
entity {name} is
  port (
    clk, rst : in std_logic;
    -- data from circuit channel
    dataIn       : in  std_logic_vector({data_bitwidth} - 1 downto 0);
    dataIn_valid : in  std_logic;
    dataIn_ready : out std_logic;
    -- address from circuit channel
    addrIn       : in  std_logic_vector({addr_bitwidth} - 1 downto 0);
    addrIn_valid : in  std_logic;
    addrIn_ready : out std_logic;
    -- data to interface channel
    dataToMem       : out std_logic_vector({data_bitwidth} - 1 downto 0);
    dataToMem_valid : out std_logic;
    dataToMem_ready : in  std_logic;
    -- address to interface channel
    addrOut       : out std_logic_vector({addr_bitwidth} - 1 downto 0);
    addrOut_valid : out std_logic;
    addrOut_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of store
architecture arch of {name} is
begin
  -- data
  dataToMem       <= dataIn;
  dataToMem_valid <= dataIn_valid;
  dataIn_ready    <= dataToMem_ready;
  -- addr
  addrOut         <= addrIn;
  addrOut_valid   <= addrIn_valid;
  addrIn_ready    <= addrOut_ready;
end architecture;
"""

  return entity + architecture

def _generate_store_signal_manager(name, data_type, addr_type):
  inner_name = f"{name}_inner"

  data_bitwidth = data_type.bitwidth
  addr_bitwidth = addr_type.bitwidth

  dependencies = _generate_store(inner_name, data_bitwidth, addr_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of store signal manager
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- data from circuit channel
    dataIn       : in  std_logic_vector({data_bitwidth} - 1 downto 0);
    dataIn_valid : in  std_logic;
    dataIn_ready : out std_logic;
    -- address from circuit channel
    addrIn       : in  std_logic_vector({addr_bitwidth} - 1 downto 0);
    addrIn_valid : in  std_logic;
    addrIn_ready : out std_logic;
    -- data to interface channel
    dataToMem       : out std_logic_vector({data_bitwidth} - 1 downto 0);
    dataToMem_valid : out std_logic;
    dataToMem_ready : in  std_logic;
    -- address to interface channel
    addrOut       : out std_logic_vector({addr_bitwidth} - 1 downto 0);
    addrOut_valid : out std_logic;
    addrOut_ready : in  std_logic
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports([
    ("addrIn", "in"),
    ("dataIn", "out")
  ], data_type.extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of store signal manager
architecture arch of {name} is
begin
  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      dataIn => dataIn,
      dataIn_valid => dataIn_valid,
      dataIn_ready => dataIn_ready,
      addrIn => addrIn,
      addrIn_valid => addrIn_valid,
      addrIn_ready => addrIn_ready_inner,
      dataToMem => dataToMem,
      dataToMem_valid => dataToMem_valid,
      dataToMem_ready => dataToMem_ready,
      addrOut => addrOut,
      addrOut_valid => addrOut_valid,
      addrOut_ready => addrOut_ready
    );
end architecture;
"""

  return dependencies + entity + architecture
