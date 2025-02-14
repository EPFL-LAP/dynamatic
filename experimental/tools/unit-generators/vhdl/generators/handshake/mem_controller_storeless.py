import ast

from generators.support.mc_support import generate_read_memory_arbiter, generate_mc_control
from generators.support.array import generate_2d_array

def generate_mem_controller_storeless(name, params):
  num_loads = int(params["num_loads"])
  port_types = ast.literal_eval(params["port_types"])
  data_bitwidth = int(port_types["loadData"][1:])
  addr_bitwidth = int(port_types["loadAddr"][1:])

  read_arbiter_name = f"{name}_read_arbiter"
  control_name = f"{name}_control"
  data_array_name = f"{name}_data_array"
  addr_array_name = f"{name}_addr_array"

  dependencies = generate_mc_control(control_name) + \
    generate_read_memory_arbiter(read_arbiter_name, num_loads, addr_bitwidth, data_bitwidth) + \
    generate_2d_array(data_array_name, num_loads, data_bitwidth) + \
    generate_2d_array(addr_array_name, num_loads, addr_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.{data_array_name}.all;
use work.{addr_array_name}.all;

-- Entity of mem_controller_storeless
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
    -- load address input channels
    ldAddr       : in  {addr_array_name};
    ldAddr_valid : in  std_logic_vector({num_loads} - 1 downto 0);
    ldAddr_ready : out std_logic_vector({num_loads} - 1 downto 0);
    -- load data output channels
    ldData       : out {data_array_name};
    ldData_valid : out std_logic_vector({num_loads} - 1 downto 0);
    ldData_ready : in  std_logic_vector({num_loads} - 1 downto 0);
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
-- Architecture of mem_controller_storeless
architecture arch of {name} is
  signal allRequestsDone : std_logic;
begin
  -- no stores will ever be issued
  storeAddr <= (others => '0');
  storeData <= (others => '0');
  storeEn   <= '0';

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

  -- NOTE: (lucas-rami) In addition to making sure there are no stores pending,
  -- we should also check that there are no loads pending as well. To achieve 
  -- this the control signals could simply start indicating the total number
  -- of accesses in the block instead of just the number of stores.
  allRequestsDone <= '1';

  control : entity work.{control_name}
    port map(
      rst             => rst,
      clk             => clk,
      memStart_valid  => memStart_valid,
      memStart_ready  => memStart_ready,
      memEnd_valid    => memEnd_valid,
      memEnd_ready    => memEnd_ready,
      ctrlEnd_valid   => ctrlEnd_valid,
      ctrlEnd_ready   => ctrlEnd_ready,
      allRequestsDone => allRequestsDone
    );

end architecture;
"""

  return dependencies + entity + architecture
