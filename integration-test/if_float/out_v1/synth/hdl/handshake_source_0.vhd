-- handshake_source_0 : source({'extra_signals': {'spec': 1}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of source
entity handshake_source_0_inner is
  port (
    clk, rst   : in std_logic;
    -- inputs
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic
  );
end entity;

-- Architecture of source
architecture arch of handshake_source_0_inner is
begin
  outs_valid <= '1';
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of signal manager
entity handshake_source_0 is
  port(
    clk : in std_logic;
    rst : in std_logic;
    outs_valid : out std_logic;
    outs_ready : in std_logic;
    outs_spec : out std_logic_vector(1 - 1 downto 0)
  );
end entity;

-- Architecture of signal manager (default)
architecture arch of handshake_source_0 is
begin
  -- Forward extra signals to output channels
  outs_spec <= "0";

  inner : entity work.handshake_source_0_inner(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;

