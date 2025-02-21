from generators.support.mc_support import generate_read_memory_arbiter
from generators.handshake.mem_controller_loadless import generate_mem_controller_loadless
from generators.handshake.mem_controller_storeless import generate_mem_controller_storeless

def generate_mem_controller(name, params):
  num_controls = params["num_controls"]
  num_loads = params["num_loads"]
  num_stores = params["num_stores"]

  if num_controls == 0 and num_loads > 0 and num_stores == 0:
    return generate_mem_controller_storeless(name, params)
  elif num_controls > 0 and num_loads == 0 and num_stores > 0:
    return generate_mem_controller_loadless(name, params)
  elif num_controls > 0 and num_loads > 0 and num_stores > 0:
    return _generate_mem_controller(name, params)
  raise ValueError("Invalid configuration for mem_controller")

def _generate_mem_controller(name, params):
  loadless_name = f"{name}_loadless"
  read_arbiter_name = f"{name}_read_arbiter"

  num_controls = params["num_controls"]
  num_loads = params["num_loads"]
  num_stores = params["num_stores"]
  port_types = params["port_types"]
  data_bitwidth = int(port_types["loadData"][1:])
  addr_bitwidth = int(port_types["loadAddr"][1:])

  dependencies = generate_mem_controller_loadless(loadless_name, params) + \
    generate_read_memory_arbiter(read_arbiter_name, {
      "arbiter_size": num_loads,
      "addr_bitwidth": addr_bitwidth,
      "data_bitwidth": data_bitwidth,
    })

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of mem_controller
entity {name} is
  port (
    clk, rst : in std_logic;
    -- start input control
    memStart_valid : in  std_logic;
    memStart_ready : out std_logic;
    -- end output control
    memEnd_valid : out std_logic;
    memEnd_ready : in  std_logic;
    -- "no more requests" input control
    ctrlEnd_valid : in  std_logic;
    ctrlEnd_ready : out std_logic;
    -- control input channels
    ctrl       : in  data_array({num_controls} - 1 downto 0)(32 - 1 downto 0);
    ctrl_valid : in  std_logic_vector({num_controls} - 1 downto 0);
    ctrl_ready : out std_logic_vector({num_controls} - 1 downto 0);
    -- load address input channels
    ldAddr       : in  data_array({num_loads} - 1 downto 0)({addr_bitwidth} - 1 downto 0);
    ldAddr_valid : in  std_logic_vector({num_loads} - 1 downto 0);
    ldAddr_ready : out std_logic_vector({num_loads} - 1 downto 0);
    -- load data output channels
    ldData       : out data_array({num_loads} - 1 downto 0)({data_bitwidth} - 1 downto 0);
    ldData_valid : out std_logic_vector({num_loads} - 1 downto 0);
    ldData_ready : in  std_logic_vector({num_loads} - 1 downto 0);
    -- store address input channels
    stAddr       : in  data_array({num_stores} - 1 downto 0)({addr_bitwidth} - 1 downto 0);
    stAddr_valid : in  std_logic_vector({num_stores} - 1 downto 0);
    stAddr_ready : out std_logic_vector({num_stores} - 1 downto 0);
    -- store data input channels
    stData       : in  data_array({num_stores} - 1 downto 0)({data_bitwidth} - 1 downto 0);
    stData_valid : in  std_logic_vector({num_stores} - 1 downto 0);
    stData_ready : out std_logic_vector({num_stores} - 1 downto 0);
    -- interface to dual-port BRAM
    loadData  : in  std_logic_vector({data_bitwidth} - 1 downto 0);
    loadEn    : out std_logic;
    loadAddr  : out std_logic_vector({addr_bitwidth} - 1 downto 0);
    storeEn   : out std_logic;
    storeAddr : out std_logic_vector({addr_bitwidth} - 1 downto 0);
    storeData : out std_logic_vector({data_bitwidth} - 1 downto 0)
  );
end entity;
"""

  architecture = f"""
-- Architecture of mem_controller
architecture arch of {name} is
  signal dropLoadAddr : std_logic_vector({addr_bitwidth} - 1 downto 0);
  signal dropLoadData : std_logic_vector({data_bitwidth} - 1 downto 0);
  signal dropLoadEn   : std_logic;
begin

  stores : entity work.{loadless_name}
    port map(
      clk            => clk,
      rst            => rst,
      memStart_valid => memStart_valid,
      memStart_ready => memStart_ready,
      memEnd_valid   => memEnd_valid,
      memEnd_ready   => memEnd_ready,
      ctrlEnd_valid  => ctrlEnd_valid,
      ctrlEnd_ready  => ctrlEnd_ready,
      ctrl           => ctrl,
      ctrl_valid     => ctrl_valid,
      ctrl_ready     => ctrl_ready,
      stAddr         => stAddr,
      stAddr_valid   => stAddr_valid,
      stAddr_ready   => stAddr_ready,
      stData         => stData,
      stData_valid   => stData_valid,
      stData_ready   => stData_ready,
      loadData       => dropLoadData,
      loadEn         => dropLoadEn,
      loadAddr       => dropLoadAddr,
      storeEn        => storeEn,
      storeAddr      => storeAddr,
      storeData      => storeData
    );

  read_arbiter : entity work.{read_arbiter_name}
    port map(
      rst              => rst,
      clk              => clk,
      pValid           => ldAddr_valid,
      ready            => ldAddr_ready,
      address_in       => ldAddr,
      nReady           => ldData_ready,
      valid            => ldData_valid,
      data_out         => ldData,
      read_enable      => loadEn,
      read_address     => loadAddr,
      data_from_memory => loadData
    );
end architecture;
"""

  return dependencies + entity + architecture
