from generators.support.mc_support import generate_mc_control, generate_read_memory_arbiter, generate_write_memory_arbiter


def generate_mem_controller(name, params):
    num_controls = params["num_controls"]
    num_loads = params["num_loads"]
    num_stores = params["num_stores"]
    data_bitwidth = params["data_bitwidth"]
    addr_bitwidth = params["addr_bitwidth"]

    if num_controls == 0 and num_loads > 0 and num_stores == 0:
        return _generate_mem_controller_storeless(name, num_loads, addr_bitwidth, data_bitwidth)
    elif num_controls > 0 and num_loads == 0 and num_stores > 0:
        return _generate_mem_controller_loadless(name, num_controls, num_stores, addr_bitwidth, data_bitwidth)
    elif num_controls > 0 and num_loads > 0 and num_stores > 0:
        return _generate_mem_controller_mixed(name, num_controls, num_loads, num_stores, addr_bitwidth, data_bitwidth)
    raise ValueError("Invalid configuration for mem_controller")


def _generate_mem_controller_mixed(name, num_controls, num_loads, num_stores, addr_bitwidth, data_bitwidth):
    loadless_name = f"{name}_loadless"
    read_arbiter_name = f"{name}_read_arbiter"

    dependencies = _generate_mem_controller_loadless(loadless_name, num_controls, num_stores, addr_bitwidth, data_bitwidth) + \
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


def _generate_mem_controller_storeless(name, num_loads, addr_bitwidth, data_bitwidth):
    read_arbiter_name = f"{name}_read_arbiter"
    control_name = f"{name}_control"

    dependencies = generate_mc_control(control_name) + \
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
    ldAddr       : in  data_array({num_loads} - 1 downto 0)({addr_bitwidth} - 1 downto 0);
    ldAddr_valid : in  std_logic_vector({num_loads} - 1 downto 0);
    ldAddr_ready : out std_logic_vector({num_loads} - 1 downto 0);
    -- load data output channels
    ldData       : out data_array({num_loads} - 1 downto 0)({data_bitwidth} - 1 downto 0);
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


def _generate_mem_controller_loadless(name, num_controls, num_stores, addr_bitwidth, data_bitwidth):
    write_arbiter_name = f"{name}_write_arbiter"
    control_name = f"{name}_control"

    dependencies = generate_mc_control(control_name) + \
        generate_write_memory_arbiter(write_arbiter_name, {
            "arbiter_size": num_stores,
            "addr_bitwidth": addr_bitwidth,
            "data_bitwidth": data_bitwidth,
        })

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

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
    ctrl       : in  data_array({num_controls} - 1 downto 0)(32 - 1 downto 0);
    ctrl_valid : in  std_logic_vector({num_controls} - 1 downto 0);
    ctrl_ready : out std_logic_vector({num_controls} - 1 downto 0);
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
