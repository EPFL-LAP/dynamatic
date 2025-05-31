def generate_eager_fork_register_block(name):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;

-- Entity of eager_fork_register_block
entity {name} is
  port (
    clk, rst : in std_logic;
    -- inputs
    ins_valid    : in std_logic;
    outs_ready   : in std_logic;
    backpressure : in std_logic;
    -- outputs
    outs_valid : out std_logic;
    blockStop  : out std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of eager_fork_register_block
architecture arch of {name} is
  signal transmitValue, keepValue : std_logic;
begin
  keepValue <= (not outs_ready) and transmitValue;

  process (clk)
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        transmitValue <= '1';
      else
        transmitValue <= keepValue or (not backpressure);
      end if;
    end if;
  end process;

  outs_valid <= transmitValue and ins_valid;
  blockStop  <= keepValue;
end architecture;
"""

    return entity + architecture
