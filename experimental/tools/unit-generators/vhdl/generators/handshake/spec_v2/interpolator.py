def generate_spec_v2_interpolator(name, _):
    return f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of interpolator
entity {name} is
  port (
    clk, rst : in std_logic;
    short : in std_logic_vector(0 downto 0);
    short_valid : in std_logic;
    short_ready : out std_logic;
    long : in std_logic_vector(0 downto 0);
    long_valid : in std_logic;
    long_ready : out std_logic;
    result : out std_logic_vector(0 downto 0);
    result_valid : out std_logic;
    result_ready : in std_logic
  );
end entity;

-- Architecture of interpolator
architecture arch of {name} is
  signal interpolate : std_logic;
  signal transfer : std_logic;
begin
  transfer <= long_valid and result_ready when interpolate else
              short_valid and long_valid and result_ready;
  process(clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        interpolate <= '0';
      else
        if transfer then
          if not interpolate and not short(0) and long(0) then
            interpolate <= '1';
          elsif interpolate and not long(0) then
            interpolate <= '0';
          end if;
        end if;
      end if;
    end if;
  end process;

  result <= "0" when interpolate else short;
  result_valid <= long_valid when interpolate else short_valid and long_valid;
  short_ready <= '0' when interpolate else long_valid and result_ready;
  long_ready <= result_ready when interpolate else short_valid and result_ready;
end architecture;
"""
