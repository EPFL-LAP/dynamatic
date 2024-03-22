library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity read_memory_arbiter is
  generic (
    ARBITER_SIZE : natural := 2;
    ADDR_WIDTH   : natural := 32;
    DATA_WIDTH   : natural := 32
  );
  port (
    rst : in std_logic;
    clk : in std_logic;
    --- interface to previous
    pValid     : in std_logic_vector(ARBITER_SIZE - 1 downto 0);  -- read requests
    ready      : out std_logic_vector(ARBITER_SIZE - 1 downto 0); -- ready to process read
    address_in : in data_array(ARBITER_SIZE - 1 downto 0)(ADDR_WIDTH - 1 downto 0);
    ---interface to next
    nReady   : in std_logic_vector(ARBITER_SIZE - 1 downto 0);                     -- next component can accept data
    valid    : out std_logic_vector(ARBITER_SIZE - 1 downto 0);                    -- sending data to next component
    data_out : out data_array(ARBITER_SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0); -- data to next components

    ---interface to memory
    read_enable      : out std_logic;
    read_address     : out std_logic_vector(ADDR_WIDTH - 1 downto 0);
    data_from_memory : in std_logic_vector(DATA_WIDTH - 1 downto 0));

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
      ADDR_WIDTH   => ADDR_WIDTH
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
      DATA_WIDTH   => DATA_WIDTH
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
