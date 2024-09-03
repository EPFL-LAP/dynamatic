library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity sink_dataless is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic
  );
end entity;

architecture arch of sink_dataless is
begin
  ins_ready <= '1';
end architecture;
