def generate_spec_v2_repeating_init(name, _):
    return f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of repeating_init
entity {name} is
  port (
    clk, rst : in std_logic;
    ins : in std_logic_vector(0 downto 0);
    ins_valid : in std_logic;
    ins_ready : out std_logic;
    outs : out std_logic_vector(0 downto 0);
    outs_valid : out std_logic;
    outs_ready : in std_logic
  );
end entity;

-- Architecture of repeating_init
architecture arch of {name} is
  signal emit_init : std_logic;
begin
  process(clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        emit_init <= '1';
      else
        if outs_valid and outs_ready then
          emit_init <= not outs(0);
        end if;
      end if;
    end if;
  end process;
  outs <= "1" when emit_init else ins;
  outs_valid <= emit_init or ins_valid;
  ins_ready <= not emit_init and outs_ready;
end architecture;
"""
