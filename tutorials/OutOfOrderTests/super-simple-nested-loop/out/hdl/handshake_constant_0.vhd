-- handshake_constant_0 : constant({'value': '0', 'port_types': {'ctrl': '!handshake.control<>', 'outs': '!handshake.channel<i1>'}, 'bitwidth': 1, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of constant
entity handshake_constant_0 is
  port (
    clk, rst : in std_logic;
    -- input channel
    ctrl_valid : in  std_logic;
    ctrl_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(1 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of constant
architecture arch of handshake_constant_0 is
begin
  outs       <= "0";
  outs_valid <= ctrl_valid;
  ctrl_ready <= outs_ready;
end architecture;

