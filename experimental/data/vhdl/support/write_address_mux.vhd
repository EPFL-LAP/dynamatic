library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity write_address_mux is
  generic (
    ARBITER_SIZE : natural;
    ADDR_WIDTH   : natural
  );
  port (
    sel      : in std_logic_vector(ARBITER_SIZE - 1 downto 0);
    addr_in  : in data_array(ARBITER_SIZE - 1 downto 0)(ADDR_WIDTH - 1 downto 0);
    addr_out : out std_logic_vector(ADDR_WIDTH - 1 downto 0)
  );
end entity;

architecture arch of write_address_mux is

begin
  process (sel, addr_in)
    variable addr_out_var : std_logic_vector(ADDR_WIDTH - 1 downto 0);
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
