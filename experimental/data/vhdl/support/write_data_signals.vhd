library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity write_data_signals is
  generic (
    ARBITER_SIZE : natural;
    DATA_WIDTH   : natural
  );
  port (
    rst        : in std_logic;
    clk        : in std_logic;
    sel        : in std_logic_vector(ARBITER_SIZE - 1 downto 0);
    write_data : out std_logic_vector(DATA_WIDTH - 1 downto 0);
    in_data    : in data_array(ARBITER_SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0);
    valid      : out std_logic_vector(ARBITER_SIZE - 1 downto 0)
  );

end entity;

architecture arch of write_data_signals is

begin

  process (sel, in_data)
    variable data_out_var : std_logic_vector(DATA_WIDTH - 1 downto 0);
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
