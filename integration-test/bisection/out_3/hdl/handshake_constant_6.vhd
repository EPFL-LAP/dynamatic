-- handshake_constant_6 : constant({'value': '01', 'bitwidth': 2, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of constant
entity handshake_constant_6 is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channel
    ctrl_valid : in  std_logic;
    ctrl_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(2 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of constant
architecture arch of handshake_constant_6 is
begin
  outs       <= "01";
  outs_valid <= ctrl_valid;
  ctrl_ready <= outs_ready;
end architecture;

