library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity sink is
  generic (
    BITWIDTH : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(BITWIDTH - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic

  );
end entity;

architecture arch of sink is
begin
  ins_ready <= '1';
end arch;
