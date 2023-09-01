library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity read_data_signals is
  generic (
    ARBITER_SIZE : natural;
    DATA_WIDTH   : natural
  );
  port (
    rst       : in std_logic;
    clk       : in std_logic;
    sel       : in std_logic_vector(ARBITER_SIZE - 1 downto 0);
    read_data : in std_logic_vector(DATA_WIDTH - 1 downto 0);
    out_data  : out data_array(ARBITER_SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0);
    valid     : out std_logic_vector(ARBITER_SIZE - 1 downto 0);
    nReady    : in std_logic_vector(ARBITER_SIZE - 1 downto 0)
  );
end entity;

architecture arch of read_data_signals is
  signal sel_prev : std_logic_vector(ARBITER_SIZE - 1 downto 0);
  signal out_reg  : data_array(ARBITER_SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0);
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
