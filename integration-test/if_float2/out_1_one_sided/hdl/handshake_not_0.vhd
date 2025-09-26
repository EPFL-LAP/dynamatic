-- handshake_not_0 : not({'bitwidth': 1, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of not
entity handshake_not_0 is
  port (
    clk, rst : in std_logic;
    ins : in std_logic_vector(1 - 1 downto 0);
    ins_valid : in std_logic;
    ins_ready : out std_logic;
    outs : out std_logic_vector(1 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in std_logic
  );
end entity;

-- Architecture of not
architecture arch of handshake_not_0 is
begin
  outs <= not ins;
  outs_valid <= ins_valid;
  ins_ready <= outs_ready;
end architecture;

