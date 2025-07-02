def generate_init(name, _):
    return f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of init
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(0 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(0 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of init
architecture arch of {name} is
  signal reg : std_logic_vector(0 downto 0);
  signal full : std_logic;
  signal enable : std_logic;
begin
  enable <= ins_ready and ins_valid and not outs_ready;
  outs <= reg when full else ins;
  outs_valid <= ins_valid or full;
  ins_ready <= not full;

  reg_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        reg <= (others => '0');
      else
        if enable then
          reg <= ins;
        end if;
      end if;
    end if;
  end process;

  full_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        full <= '1';
      else
        full <= outs_valid and not outs_ready;
      end if;
    end if;
  end process;
end architecture;
"""
