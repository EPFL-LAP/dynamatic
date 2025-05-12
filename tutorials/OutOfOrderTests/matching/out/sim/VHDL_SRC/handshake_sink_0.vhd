-- handshake_sink_0 : sink({'port_types': {'ins': '!handshake.channel<i32>'}, 'bitwidth': 32, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of sink
entity handshake_sink_0 is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(32 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic
  );
end entity;

-- Architecture of sink
architecture arch of handshake_sink_0 is
begin
  ins_ready <= '1';
end architecture;

