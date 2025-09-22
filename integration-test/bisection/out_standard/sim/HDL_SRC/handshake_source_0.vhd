-- handshake_source_0 : source({'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of source
entity handshake_source_0 is
  port (
    clk, rst   : in std_logic;
    -- inputs
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic
  );
end entity;

-- Architecture of source
architecture arch of handshake_source_0 is
begin
  outs_valid <= '1';
end architecture;

