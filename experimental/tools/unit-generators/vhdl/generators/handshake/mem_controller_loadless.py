import ast

from generators.support.mc_support import generate_write_memory_arbiter, generate_mc_control
from generators.support.array import generate_2d_array

def generate_mem_controller_loadless(name, params):
  num_controls = int(params["num_controls"])
  num_stores = int(params["num_stores"])
  port_types = ast.literal_eval(params["port_types"])
  data_bitwidth = int(port_types["loadData"][1:])
  addr_bitwidth = int(port_types["loadAddr"][1:])

  write_arbiter_name = f"{name}_write_arbiter"
  control_name = f"{name}_control"
  data_array_name = f"{name}_data_array"
  addr_array_name = f"{name}_addr_array"
  ctrl_array_name = f"{name}_ctrl_array"

  dependencies = generate_mc_control(control_name) + \
    generate_write_memory_arbiter(write_arbiter_name, num_stores, addr_bitwidth, data_bitwidth) + \
    generate_2d_array(data_array_name, num_stores, data_bitwidth) + \
    generate_2d_array(addr_array_name, num_stores, addr_bitwidth) + \
    generate_2d_array(ctrl_array_name, num_controls, 32)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.{data_array_name}.all;
use work.{addr_array_name}.all;
use work.{ctrl_array_name}.all;

-- Entity of mem_controller_loadless
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
    ctrl       : in  {ctrl_array_name};
    ctrl_valid : in  std_logic_vector({num_controls} - 1 downto 0);
    ctrl_ready : out std_logic_vector({num_controls} - 1 downto 0);
    -- store address input channels
    stAddr       : in  {addr_array_name};
    stAddr_valid : in  std_logic_vector({num_stores} - 1 downto 0);
    stAddr_ready : out std_logic_vector({num_stores} - 1 downto 0);
    -- store data input channels
    stData       : in  {data_array_name};
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
-- Architecture of mem_controller_loadless

-- Terminology:
-- Access ports    : circuit to memory_controller;
-- Interface ports : memory_controller to memory_interface (e.g., BRAM/AXI);

architecture arch of {name} is
  
  -- TODO: The size of this counter should be configurable
  signal remainingStores                    : std_logic_vector(31 downto 0);
  
  -- Indicating the store interface port that there is a valid store request
  -- (currently not used).
  signal interface_port_valid               : std_logic_vector({num_stores} - 1 downto 0);

  -- Indicating a store port has both a valid data and a valid address.
  signal store_access_port_complete_request : std_logic_vector({num_stores} - 1 downto 0);

  -- Indicating the store port is selected by the arbiter.
  signal store_access_port_selected         : std_logic_vector({num_stores} - 1 downto 0);
  signal allRequestsDone                    : std_logic;


  constant zeroStore : std_logic_vector(31 downto 0)               := (others => '0');
  constant zeroCtrl  : std_logic_vector({num_controls} - 1 downto 0) := (others => '0');

begin
  loadEn   <= '0';
  loadAddr <= (others => '0');

  -- A store request is complete if both address and data are valid.
  store_access_port_complete_request <= stAddr_valid and stData_valid;

  write_arbiter : entity work.{write_arbiter_name}
    port map(
      rst            => rst,
      clk            => clk,
      pValid         => store_access_port_complete_request,
      ready          => store_access_port_selected,
      address_in     => stAddr,
      data_in        => stData,
      nReady         => (others => '1'),
      valid          => interface_port_valid,
      write_enable   => storeEn,
      write_address  => storeAddr,
      data_to_memory => storeData
    );

  stData_ready <= store_access_port_selected;
  stAddr_ready <= store_access_port_selected;
  ctrl_ready   <= (others => '1');

  count_stores : process (clk)
    variable counter : std_logic_vector(31 downto 0);
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        counter := (31 downto 0 => '0');
      else
        for i in 0 to {num_controls} - 1 loop
          if ctrl_valid(i) then
            counter := std_logic_vector(unsigned(counter) + unsigned(ctrl(i)));
          end if;
        end loop;
        if storeEn then
          counter := std_logic_vector(unsigned(counter) - 1);
        end if;
      end if;
      remainingStores <= counter;
    end if;
  end process;

  -- NOTE: (lucas-rami) In addition to making sure there are no stores pending,
  -- we should also check that there are no loads pending as well. To achieve 
  -- this the control signals could simply start indicating the total number
  -- of accesses in the block instead of just the number of stores.
  allRequestsDone <= '1' when (remainingStores = zeroStore) and (ctrl_valid = zeroCtrl) else '0';

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
