library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity write_memory_arbiter is
  generic (
    ARBITER_SIZE : natural := 2;
    ADDR_WIDTH   : natural := 32;
    DATA_WIDTH   : natural := 32
  );
  port (
    rst : in std_logic;
    clk : in std_logic;
    --- interface to previous
    pValid     : in std_logic_vector(ARBITER_SIZE - 1 downto 0);  --write requests
    ready      : out std_logic_vector(ARBITER_SIZE - 1 downto 0); -- ready
    address_in : in data_array(ARBITER_SIZE - 1 downto 0)(ADDR_WIDTH - 1 downto 0);
    data_in    : in data_array(ARBITER_SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0); -- data from previous that want to write

    ---interface to next
    nReady : in std_logic_vector(ARBITER_SIZE - 1 downto 0);  -- next component can continue after write
    valid  : out std_logic_vector(ARBITER_SIZE - 1 downto 0); --sending write confirmation to next component

    ---interface to memory
    write_enable   : out std_logic;
    enable         : out std_logic;
    write_address  : out std_logic_vector(ADDR_WIDTH - 1 downto 0);
    data_to_memory : out std_logic_vector(DATA_WIDTH - 1 downto 0)
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
      ADDR_WIDTH   => ADDR_WIDTH
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
      DATA_WIDTH   => DATA_WIDTH
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
