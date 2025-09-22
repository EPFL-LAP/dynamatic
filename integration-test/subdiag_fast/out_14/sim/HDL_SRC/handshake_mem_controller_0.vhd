-- handshake_mem_controller_0 : mem_controller({'num_controls': 0, 'num_loads': 1, 'num_stores': 0, 'addr_bitwidth': 10, 'data_bitwidth': 32})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of mc_control
entity handshake_mem_controller_0_control is
  port (
    clk : in std_logic;
    rst : in std_logic;
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

-- Entity of read_priority
entity handshake_mem_controller_0_read_arbiter_priority is
  port (
    req          : in  std_logic_vector(1 - 1 downto 0); -- read requests (pValid signals)
    data_ready   : in  std_logic_vector(1 - 1 downto 0); -- ready from next
    priority_out : out std_logic_vector(1 - 1 downto 0) -- priority function output
  );
end entity;

-- Architecture of read_priority
architecture arch of handshake_mem_controller_0_read_arbiter_priority is
begin
  process (req, data_ready)
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

-- Entity of read_address_mux
entity handshake_mem_controller_0_read_arbiter_addressing is
  port (
    sel      : in  std_logic_vector(1 - 1 downto 0);
    addr_in  : in  data_array(1 - 1 downto 0)(10 - 1 downto 0);
    addr_out : out std_logic_vector(10 - 1 downto 0)
  );
end entity;

-- Architecture of read_address_mux
architecture arch of handshake_mem_controller_0_read_arbiter_addressing is
begin
  process (sel, addr_in)
    variable addr_out_var : std_logic_vector(10 - 1 downto 0);
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

-- Entity of read_address_ready
entity handshake_mem_controller_0_read_arbiter_addressReady is
  port (
    sel    : in  std_logic_vector(1 - 1 downto 0);
    nReady : in  std_logic_vector(1 - 1 downto 0);
    ready  : out std_logic_vector(1 - 1 downto 0)
  );
end entity;

-- Architecture of read_address_ready
architecture arch of handshake_mem_controller_0_read_arbiter_addressReady is
begin
  GEN1 : for I in 0 to 1 - 1 generate
    ready(I) <= nReady(I) and sel(I);
  end generate GEN1;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of read_data_signals
entity handshake_mem_controller_0_read_arbiter_data is
  port (
    rst       : in  std_logic;
    clk       : in  std_logic;
    sel       : in  std_logic_vector(1 - 1 downto 0);
    read_data : in  std_logic_vector(32 - 1 downto 0);
    out_data  : out data_array(1 - 1 downto 0)(32 - 1 downto 0);
    valid     : out std_logic_vector(1 - 1 downto 0);
    nReady    : in  std_logic_vector(1 - 1 downto 0)
  );
end entity;

-- Architecture of read_data_signals
architecture arch of handshake_mem_controller_0_read_arbiter_data is
  signal sel_prev : std_logic_vector(1 - 1 downto 0);
  signal out_reg  : data_array(1 - 1 downto 0)(32 - 1 downto 0);
begin

  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        for I in 0 to 1 - 1 loop
          valid(I)    <= '0';
          sel_prev(I) <= '0';
        end loop;
      else
        for I in 0 to 1 - 1 loop
          sel_prev(I) <= sel(I);
          if (sel(I) = '1') then
            valid(I) <= '1'; --or not nReady(I); -- just sel(I) ??
            --sel_prev(I) <= '1';
          else
            if (nReady(I) = '1') then
              valid(I) <= '0';
              ---sel_prev(I) <= '0';
            end if;
          end if;
        end loop;
      end if;
    end if;
  end process;

  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        for I in 0 to 1 - 1 loop
          out_reg(I) <= (others => '0');
        end loop;
      else
        for I in 0 to 1 - 1 loop
          if (sel_prev(I) = '1') then
            out_reg(I) <= read_data;
          end if;
        end loop;
      end if;
    end if;
  end process;

  process (read_data, sel_prev, out_reg) is
  begin
    for I in 0 to 1 - 1 loop
      if (sel_prev(I) = '1') then
        out_data(I) <= read_data;
      else
        out_data(I) <= out_reg(I);
      end if;
    end loop;
  end process;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of read_memory_arbiter
entity handshake_mem_controller_0_read_arbiter is
  port (
    rst : in std_logic;
    clk : in std_logic;
    --- interface to previous
    pValid     : in  std_logic_vector(1 - 1 downto 0); -- read requests
    ready      : out std_logic_vector(1 - 1 downto 0); -- ready to process read
    address_in : in  data_array(1 - 1 downto 0)(10 - 1 downto 0);
    ---interface to next
    nReady   : in  std_logic_vector(1 - 1 downto 0); -- next component can accept data
    valid    : out std_logic_vector(1 - 1 downto 0); -- sending data to next component
    data_out : out data_array(1 - 1 downto 0)(32 - 1 downto 0); -- data to next components

    ---interface to memory
    read_enable      : out std_logic;
    read_address     : out std_logic_vector(10 - 1 downto 0);
    data_from_memory : in  std_logic_vector(32 - 1 downto 0));

end entity;

-- Architecture of read_memory_arbiter
architecture arch of handshake_mem_controller_0_read_arbiter is
  signal priorityOut : std_logic_vector(1 - 1 downto 0);

begin

  priority : entity work.handshake_mem_controller_0_read_arbiter_priority
    port map(
      req          => pValid,
      data_ready   => nReady,
      priority_out => priorityOut
    );

  addressing : entity work.handshake_mem_controller_0_read_arbiter_addressing
    port map(
      sel      => priorityOut,
      addr_in  => address_in,
      addr_out => read_address
    );

  addressReady : entity work.handshake_mem_controller_0_read_arbiter_addressReady
    port map(
      sel    => priorityOut,
      nReady => nReady,
      ready  => ready
    );

  data : entity work.handshake_mem_controller_0_read_arbiter_data
    port map(
      rst       => rst,
      clk       => clk,
      sel       => priorityOut,
      read_data => data_from_memory,
      out_data  => data_out,
      valid     => valid,
      nReady    => nReady
    );

  process (priorityOut) is
    variable read_en_var : std_logic;
  begin
    read_en_var := '0';
    for I in 0 to 1 - 1 loop
      read_en_var := read_en_var or priorityOut(I);
    end loop;
    read_enable <= read_en_var;
  end process;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of mem_controller_storeless
entity handshake_mem_controller_0 is
  port (
    clk : in std_logic;
    rst : in std_logic;
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
    ldAddr       : in  data_array(1 - 1 downto 0)(10 - 1 downto 0);
    ldAddr_valid : in  std_logic_vector(1 - 1 downto 0);
    ldAddr_ready : out std_logic_vector(1 - 1 downto 0);
    -- load data output channels
    ldData       : out data_array(1 - 1 downto 0)(32 - 1 downto 0);
    ldData_valid : out std_logic_vector(1 - 1 downto 0);
    ldData_ready : in  std_logic_vector(1 - 1 downto 0);
    -- interface to dual-port BRAM
    loadData  : in  std_logic_vector(32 - 1 downto 0);
    loadEn    : out std_logic;
    loadAddr  : out std_logic_vector(10 - 1 downto 0);
    storeEn   : out std_logic;
    storeAddr : out std_logic_vector(10 - 1 downto 0);
    storeData : out std_logic_vector(32 - 1 downto 0)
  );
end entity;

-- Architecture of mem_controller_storeless
architecture arch of handshake_mem_controller_0 is
  signal allRequestsDone : std_logic;
begin
  -- no stores will ever be issued
  storeAddr <= (others => '0');
  storeData <= (others => '0');
  storeEn   <= '0';

  read_arbiter : entity work.handshake_mem_controller_0_read_arbiter
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

