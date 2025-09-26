-- handshake_sink_2 : sink({'bitwidth': 8, 'extra_signals': {'spec': 1}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of sink
entity handshake_sink_2_inner is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(8 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic
  );
end entity;

-- Architecture of sink
architecture arch of handshake_sink_2_inner is
begin
  ins_ready <= '1';
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of signal manager
entity handshake_sink_2 is
  port(
    clk : in std_logic;
    rst : in std_logic;
    ins : in std_logic_vector(8 - 1 downto 0);
    ins_valid : in std_logic;
    ins_ready : out std_logic;
    ins_spec : in std_logic_vector(1 - 1 downto 0)
  );
end entity;

-- Architecture of signal manager (default)
architecture arch of handshake_sink_2 is
begin
  -- Forward extra signals to output channels
  

  inner : entity work.handshake_sink_2_inner(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins,
      ins_valid => ins_valid,
      ins_ready => ins_ready
    );
end architecture;

