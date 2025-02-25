library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity source is
  port (
    -- inputs
    clk, rst   : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic
  );
end entity;

architecture arch of source is
begin
  outs_valid <= '1';
end arch;
