-- handshake_mem_controller_0 : mem_controller({'num_controls': 1, 'num_loads': 0, 'num_stores': 1, 'port_types': {'loadData': 'i32', 'memStart': '!handshake.control<>', 'ctrl_0': '!handshake.channel<i32>', 'stAddr_0': '!handshake.channel<i5>', 'stData_0': '!handshake.channel<i32>', 'ctrlEnd': '!handshake.control<>', 'memEnd': '!handshake.control<>', 'loadEn': 'i1', 'loadAddr': 'i5', 'storeEn': 'i1', 'storeAddr': 'i5', 'storeData': 'i32'}, 'addr_bitwidth': 5, 'data_bitwidth': 32})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of mc_control
entity handshake_mem_controller_0_control is
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
    -- all requests completed
    allRequestsDone : in std_logic
  );
end entity;

-- Architecture of mc_control
architecture arch of handshake_mem_controller_0_control is
begin
  process (clk) begin
    if rising_edge(clk) then
      if (rst = '1') then
        memStart_ready <= '1';
        memEnd_valid   <= '0';
        ctrlEnd_ready  <= '0';
      else
        memStart_ready <= memStart_ready;
        memEnd_valid   <= memEnd_valid;
        ctrlEnd_ready  <= ctrlEnd_ready;
        -- determine when the memory has completed all requests
        if ctrlEnd_valid and allRequestsDone then
          memEnd_valid  <= '1';
          ctrlEnd_ready <= '1';
        end if;
        -- acknowledge the 'ctrlEnd' control
        if ctrlEnd_valid and ctrlEnd_ready then
          ctrlEnd_ready <= '0';
        end if;
        -- determine when the memory is idle
        if memStart_valid and memStart_ready then
          memStart_ready <= '0';
        end if;
        if memEnd_valid and memEnd_ready then
          memStart_ready <= '1';
          memEnd_valid   <= '0';
        end if;
      end if;
    end if;
  end process;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of write_priority
entity handshake_mem_controller_0_write_arbiter_priority is
  port (
    req          : in  std_logic_vector(1 - 1 downto 0);
    data_ready   : in  std_logic_vector(1 - 1 downto 0);
    priority_out : out std_logic_vector(1 - 1 downto 0)
  );
end entity;

-- Architecture of write_priority
architecture arch of handshake_mem_controller_0_write_arbiter_priority is

begin

  process (data_ready, req)
    variable prio_req : std_logic;

  begin
    -- the first index I such that (req(I) and data_ready(I) = '1') is '1', others are '0'
    priority_out(0) <= req(0) and data_ready(0);
    prio_req := req(0) and data_ready(0);

    for I in 1 to 1 - 1 loop
      priority_out(I) <= (not prio_req) and req(I) and data_ready(I);
      prio_req := prio_req or (req(I) and data_ready(I));
    end loop;
  end process;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of write_address_mux
entity handshake_mem_controller_0_write_arbiter_addressing is
  port (
    sel      : in  std_logic_vector(1 - 1 downto 0);
    addr_in  : in  data_array(1 - 1 downto 0)(5 - 1 downto 0);
    addr_out : out std_logic_vector(5 - 1 downto 0)
  );
end entity;

-- Architecture of write_address_mux
architecture arch of handshake_mem_controller_0_write_arbiter_addressing is
begin
  process (sel, addr_in)
    variable addr_out_var : std_logic_vector(5 - 1 downto 0);
  begin
    addr_out_var := (others => '0');
    for I in 0 to 1 - 1 loop
      if (sel(I) = '1') then
        addr_out_var := addr_in(I);
      end if;
    end loop;
    addr_out <= addr_out_var;
  end process;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of write_address_ready
entity handshake_mem_controller_0_write_arbiter_addressReady is
  port (
    sel    : in  std_logic_vector(1 - 1 downto 0);
    nReady : in  std_logic_vector(1 - 1 downto 0);
    ready  : out std_logic_vector(1 - 1 downto 0)
  );

end entity;

-- Architecture of write_address_ready
architecture arch of handshake_mem_controller_0_write_arbiter_addressReady is

begin

  GEN1 : for I in 0 to 1 - 1 generate
    ready(I) <= nReady(I) and sel(I);
  end generate GEN1;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of write_data_signals
entity handshake_mem_controller_0_write_arbiter_data is
  port (
    rst        : in  std_logic;
    clk        : in  std_logic;
    sel        : in  std_logic_vector(1 - 1 downto 0);
    write_data : out std_logic_vector(32 - 1 downto 0);
    in_data    : in  data_array(1 - 1 downto 0)(32 - 1 downto 0);
    valid      : out std_logic_vector(1 - 1 downto 0)
  );

end entity;

-- Architecture of write_data_signals
architecture arch of handshake_mem_controller_0_write_arbiter_data is

begin

  process (sel, in_data)
    variable data_out_var : std_logic_vector(32 - 1 downto 0);
  begin
    data_out_var := (others => '0');

    for I in 0 to 1 - 1 loop
      if (sel(I) = '1') then
        data_out_var := in_data(I);
      end if;
    end loop;
    write_data <= data_out_var;
  end process;

  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        for I in 0 to 1 - 1 loop
          valid(I) <= '0';
        end loop;
      else
        for I in 0 to 1 - 1 loop
          valid(I) <= sel(I);
        end loop;
      end if;
    end if;
  end process;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of write_memory_arbiter
entity handshake_mem_controller_0_write_arbiter is
  port (
    rst : in std_logic;
    clk : in std_logic;
    --- interface to previous
    pValid     : in  std_logic_vector(1 - 1 downto 0); --write requests
    ready      : out std_logic_vector(1 - 1 downto 0); -- ready
    address_in : in  data_array(1 - 1 downto 0)(5 - 1 downto 0);
    data_in    : in  data_array(1 - 1 downto 0)(32 - 1 downto 0); -- data from previous that want to write

    ---interface to next
    nReady : in  std_logic_vector(1 - 1 downto 0); -- next component can continue after write
    valid  : out std_logic_vector(1 - 1 downto 0); --sending write confirmation to next component

    ---interface to memory
    write_enable   : out std_logic;
    enable         : out std_logic;
    write_address  : out std_logic_vector(5 - 1 downto 0);
    data_to_memory : out std_logic_vector(32 - 1 downto 0)
  );

end entity;

-- Architecture of write_memory_arbiter
architecture arch of handshake_mem_controller_0_write_arbiter is
  signal priorityOut : std_logic_vector(1 - 1 downto 0);

begin

  priority : entity work.handshake_mem_controller_0_write_arbiter_priority
    port map(
      req          => pValid,
      data_ready   => nReady,
      priority_out => priorityOut
    );

  addressing : entity work.handshake_mem_controller_0_write_arbiter_addressing
    port map(
      sel      => priorityOut,
      addr_in  => address_in,
      addr_out => write_address
    );

  addressReady : entity work.handshake_mem_controller_0_write_arbiter_addressReady
    port map(
      sel    => priorityOut,
      nReady => nReady,
      ready  => ready
    );
  data : entity work.handshake_mem_controller_0_write_arbiter_data
    port map(
      rst        => rst,
      clk        => clk,
      sel        => priorityOut,
      write_data => data_to_memory,
      in_data    => data_in,
      valid      => valid
    );

  process (priorityOut) is
    variable write_en_var : std_logic;
  begin
    write_en_var := '0';
    for I in 0 to 1 - 1 loop
      write_en_var := write_en_var or priorityOut(I);
    end loop;
    write_enable <= write_en_var;
    enable       <= write_en_var;
  end process;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of mem_controller_loadless
entity handshake_mem_controller_0 is
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
    ctrl       : in  data_array(1 - 1 downto 0)(32 - 1 downto 0);
    ctrl_valid : in  std_logic_vector(1 - 1 downto 0);
    ctrl_ready : out std_logic_vector(1 - 1 downto 0);
    -- store address input channels
    stAddr       : in  data_array(1 - 1 downto 0)(5 - 1 downto 0);
    stAddr_valid : in  std_logic_vector(1 - 1 downto 0);
    stAddr_ready : out std_logic_vector(1 - 1 downto 0);
    -- store data input channels
    stData       : in  data_array(1 - 1 downto 0)(32 - 1 downto 0);
    stData_valid : in  std_logic_vector(1 - 1 downto 0);
    stData_ready : out std_logic_vector(1 - 1 downto 0);
    -- interface to dual-port BRAM
    loadData  : in  std_logic_vector(32 - 1 downto 0);
    loadEn    : out std_logic;
    loadAddr  : out std_logic_vector(5 - 1 downto 0);
    storeEn   : out std_logic;
    storeAddr : out std_logic_vector(5 - 1 downto 0);
    storeData : out std_logic_vector(32 - 1 downto 0)
  );
end entity;

-- Architecture of mem_controller_loadless

-- Terminology:
-- Access ports    : circuit to memory_controller;
-- Interface ports : memory_controller to memory_interface (e.g., BRAM/AXI);

architecture arch of handshake_mem_controller_0 is
  
  -- TODO: The size of this counter should be configurable
  signal remainingStores                    : std_logic_vector(31 downto 0);
  
  -- Indicating the store interface port that there is a valid store request
  -- (currently not used).
  signal interface_port_valid               : std_logic_vector(1 - 1 downto 0);

  -- Indicating a store port has both a valid data and a valid address.
  signal store_access_port_complete_request : std_logic_vector(1 - 1 downto 0);

  -- Indicating the store port is selected by the arbiter.
  signal store_access_port_selected         : std_logic_vector(1 - 1 downto 0);
  signal allRequestsDone                    : std_logic;


  constant zeroStore : std_logic_vector(31 downto 0)               := (others => '0');
  constant zeroCtrl  : std_logic_vector(1 - 1 downto 0) := (others => '0');

begin
  loadEn   <= '0';
  loadAddr <= (others => '0');

  -- A store request is complete if both address and data are valid.
  store_access_port_complete_request <= stAddr_valid and stData_valid;

  write_arbiter : entity work.handshake_mem_controller_0_write_arbiter
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
        for i in 0 to 1 - 1 loop
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

  control : entity work.handshake_mem_controller_0_control
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

