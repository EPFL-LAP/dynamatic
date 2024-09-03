
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity read_address_mux is
  generic (
    ARBITER_SIZE : natural;
    ADDR_TYPE   : natural
  );
  port (
    sel      : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
    addr_in  : in  data_array(ARBITER_SIZE - 1 downto 0)(ADDR_TYPE - 1 downto 0);
    addr_out : out std_logic_vector(ADDR_TYPE - 1 downto 0)
  );
end entity;

architecture arch of read_address_mux is

begin
  process (sel, addr_in)
    variable addr_out_var : std_logic_vector(ADDR_TYPE - 1 downto 0);
  begin
    addr_out_var := (others => '0');
    for I in 0 to ARBITER_SIZE - 1 loop
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

entity read_address_ready is
  generic (
    ARBITER_SIZE : natural
  );
  port (
    sel    : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
    nReady : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
    ready  : out std_logic_vector(ARBITER_SIZE - 1 downto 0)
  );
end entity;

architecture arch of read_address_ready is
begin
  GEN1 : for I in 0 to ARBITER_SIZE - 1 generate
    ready(I) <= nReady(I) and sel(I);
  end generate GEN1;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity read_data_signals is
  generic (
    ARBITER_SIZE : natural;
    DATA_TYPE   : natural
  );
  port (
    rst       : in  std_logic;
    clk       : in  std_logic;
    sel       : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
    read_data : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    out_data  : out data_array(ARBITER_SIZE - 1 downto 0)(DATA_TYPE - 1 downto 0);
    valid     : out std_logic_vector(ARBITER_SIZE - 1 downto 0);
    nReady    : in  std_logic_vector(ARBITER_SIZE - 1 downto 0)
  );
end entity;

architecture arch of read_data_signals is
  signal sel_prev : std_logic_vector(ARBITER_SIZE - 1 downto 0);
  signal out_reg  : data_array(ARBITER_SIZE - 1 downto 0)(DATA_TYPE - 1 downto 0);
begin

  process (clk, rst) is
  begin
    if (rst = '1') then
      for I in 0 to ARBITER_SIZE - 1 loop
        valid(I)    <= '0';
        sel_prev(I) <= '0';
      end loop;
    elsif (rising_edge(clk)) then
      for I in 0 to ARBITER_SIZE - 1 loop
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
  end process;

  process (clk, rst) is
  begin
    if (rising_edge(clk)) then
      for I in 0 to ARBITER_SIZE - 1 loop
        if (sel_prev(I) = '1') then
          out_reg(I) <= read_data;
        end if;
      end loop;
    end if;
  end process;

  process (read_data, sel_prev, out_reg) is
  begin
    for I in 0 to ARBITER_SIZE - 1 loop
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

entity read_priority is
  generic (
    ARBITER_SIZE : natural
  );
  port (
    req          : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); -- read requests (pValid signals)
    data_ready   : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); -- ready from next
    priority_out : out std_logic_vector(ARBITER_SIZE - 1 downto 0) -- priority function output
  );
end entity;

architecture arch of read_priority is

begin
  process (req, data_ready)
    variable prio_req : std_logic;
  begin
    -- the first index I such that (req(I) and data_ready(I) = '1') is '1', others are '0'
    priority_out(0) <= req(0) and data_ready(0);
    prio_req := req(0) and data_ready(0);
    for I in 1 to ARBITER_SIZE - 1 loop
      priority_out(I) <= (not prio_req) and req(I) and data_ready(I);
      prio_req := prio_req or (req(I) and data_ready(I));
    end loop;
  end process;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity read_memory_arbiter is
  generic (
    ARBITER_SIZE : natural := 2;
    ADDR_TYPE   : natural := 32;
    DATA_TYPE   : natural := 32
  );
  port (
    rst : in std_logic;
    clk : in std_logic;
    --- interface to previous
    pValid     : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); -- read requests
    ready      : out std_logic_vector(ARBITER_SIZE - 1 downto 0); -- ready to process read
    address_in : in  data_array(ARBITER_SIZE - 1 downto 0)(ADDR_TYPE - 1 downto 0);
    ---interface to next
    nReady   : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); -- next component can accept data
    valid    : out std_logic_vector(ARBITER_SIZE - 1 downto 0); -- sending data to next component
    data_out : out data_array(ARBITER_SIZE - 1 downto 0)(DATA_TYPE - 1 downto 0); -- data to next components

    ---interface to memory
    read_enable      : out std_logic;
    read_address     : out std_logic_vector(ADDR_TYPE - 1 downto 0);
    data_from_memory : in  std_logic_vector(DATA_TYPE - 1 downto 0));

end entity;

architecture arch of read_memory_arbiter is
  signal priorityOut : std_logic_vector(ARBITER_SIZE - 1 downto 0);

begin

  priority : entity work.read_priority
    generic map(
      ARBITER_SIZE => ARBITER_SIZE
    )
    port map(
      req          => pValid,
      data_ready   => nReady,
      priority_out => priorityOut
    );

  addressing : entity work.read_address_mux
    generic map(
      ARBITER_SIZE => ARBITER_SIZE,
      ADDR_TYPE   => ADDR_TYPE
    )
    port map(
      sel      => priorityOut,
      addr_in  => address_in,
      addr_out => read_address
    );

  adderssReady : entity work.read_address_ready
    generic map(
      ARBITER_SIZE => ARBITER_SIZE
    )
    port map(
      sel    => priorityOut,
      nReady => nReady,
      ready  => ready
    );

  data : entity work.read_data_signals
    generic map(
      ARBITER_SIZE => ARBITER_SIZE,
      DATA_TYPE   => DATA_TYPE
    )
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
    for I in 0 to ARBITER_SIZE - 1 loop
      read_en_var := read_en_var or priorityOut(I);
    end loop;
    read_enable <= read_en_var;
  end process;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity write_address_mux is
  generic (
    ARBITER_SIZE : natural;
    ADDR_TYPE   : natural
  );
  port (
    sel      : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
    addr_in  : in  data_array(ARBITER_SIZE - 1 downto 0)(ADDR_TYPE - 1 downto 0);
    addr_out : out std_logic_vector(ADDR_TYPE - 1 downto 0)
  );
end entity;

architecture arch of write_address_mux is

begin
  process (sel, addr_in)
    variable addr_out_var : std_logic_vector(ADDR_TYPE - 1 downto 0);
  begin
    addr_out_var := (others => '0');
    for I in 0 to ARBITER_SIZE - 1 loop
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

entity write_address_ready is
  generic (
    ARBITER_SIZE : natural
  );
  port (
    sel    : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
    nReady : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
    ready  : out std_logic_vector(ARBITER_SIZE - 1 downto 0)
  );

end entity;

architecture arch of write_address_ready is

begin

  GEN1 : for I in 0 to ARBITER_SIZE - 1 generate
    ready(I) <= nReady(I) and sel(I);
  end generate GEN1;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity write_data_signals is
  generic (
    ARBITER_SIZE : natural;
    DATA_TYPE   : natural
  );
  port (
    rst        : in  std_logic;
    clk        : in  std_logic;
    sel        : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
    write_data : out std_logic_vector(DATA_TYPE - 1 downto 0);
    in_data    : in  data_array(ARBITER_SIZE - 1 downto 0)(DATA_TYPE - 1 downto 0);
    valid      : out std_logic_vector(ARBITER_SIZE - 1 downto 0)
  );

end entity;

architecture arch of write_data_signals is

begin

  process (sel, in_data)
    variable data_out_var : std_logic_vector(DATA_TYPE - 1 downto 0);
  begin
    data_out_var := (others => '0');

    for I in 0 to ARBITER_SIZE - 1 loop
      if (sel(I) = '1') then
        data_out_var := in_data(I);
      end if;
    end loop;
    write_data <= data_out_var;
  end process;

  process (clk, rst) is
  begin
    if (rst = '1') then
      for I in 0 to ARBITER_SIZE - 1 loop
        valid(I) <= '0';
      end loop;

    elsif (rising_edge(clk)) then
      for I in 0 to ARBITER_SIZE - 1 loop
        valid(I) <= sel(I);
      end loop;
    end if;
  end process;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity write_priority is
  generic (
    ARBITER_SIZE : natural
  );
  port (
    req          : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
    data_ready   : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
    priority_out : out std_logic_vector(ARBITER_SIZE - 1 downto 0)
  );
end entity;

architecture arch of write_priority is

begin

  process (data_ready, req)
    variable prio_req : std_logic;

  begin
    -- the first index I such that (req(I) and data_ready(I) = '1') is '1', others are '0'
    priority_out(0) <= req(0) and data_ready(0);
    prio_req := req(0) and data_ready(0);

    for I in 1 to ARBITER_SIZE - 1 loop
      priority_out(I) <= (not prio_req) and req(I) and data_ready(I);
      prio_req := prio_req or (req(I) and data_ready(I));
    end loop;
  end process;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity write_memory_arbiter is
  generic (
    ARBITER_SIZE : natural := 2;
    ADDR_TYPE   : natural := 32;
    DATA_TYPE   : natural := 32
  );
  port (
    rst : in std_logic;
    clk : in std_logic;
    --- interface to previous
    pValid     : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); --write requests
    ready      : out std_logic_vector(ARBITER_SIZE - 1 downto 0); -- ready
    address_in : in  data_array(ARBITER_SIZE - 1 downto 0)(ADDR_TYPE - 1 downto 0);
    data_in    : in  data_array(ARBITER_SIZE - 1 downto 0)(DATA_TYPE - 1 downto 0); -- data from previous that want to write

    ---interface to next
    nReady : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); -- next component can continue after write
    valid  : out std_logic_vector(ARBITER_SIZE - 1 downto 0); --sending write confirmation to next component

    ---interface to memory
    write_enable   : out std_logic;
    enable         : out std_logic;
    write_address  : out std_logic_vector(ADDR_TYPE - 1 downto 0);
    data_to_memory : out std_logic_vector(DATA_TYPE - 1 downto 0)
  );

end entity;

architecture arch of write_memory_arbiter is
  signal priorityOut : std_logic_vector(ARBITER_SIZE - 1 downto 0);

begin

  priority : entity work.write_priority
    generic map(
      ARBITER_SIZE => ARBITER_SIZE
    )
    port map(
      req          => pValid,
      data_ready   => nReady,
      priority_out => priorityOut
    );

  addressing : entity work.write_address_mux
    generic map(
      ARBITER_SIZE => ARBITER_SIZE,
      ADDR_TYPE   => ADDR_TYPE
    )
    port map(
      sel      => priorityOut,
      addr_in  => address_in,
      addr_out => write_address
    );

  addressReady : entity work.write_address_ready
    generic map(
      ARBITER_SIZE => ARBITER_SIZE
    )
    port map(
      sel    => priorityOut,
      nReady => nReady,
      ready  => ready
    );
  data : entity work.write_data_signals
    generic map(
      ARBITER_SIZE => ARBITER_SIZE,
      DATA_TYPE   => DATA_TYPE
    )
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
    for I in 0 to ARBITER_SIZE - 1 loop
      write_en_var := write_en_var or priorityOut(I);
    end loop;
    write_enable <= write_en_var;
    enable       <= write_en_var;
  end process;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mc_control is
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

architecture arch of mc_control is
begin
  process (rst, clk)
  begin
    if rst then
      memStart_ready <= '1';
      memEnd_valid   <= '0';
      ctrlEnd_ready  <= '0';
    elsif rising_edge(clk) then
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
  end process;
end architecture;
