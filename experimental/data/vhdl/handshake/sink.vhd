library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.customTypes.all;
entity sink is

  generic (
    BITWIDTH : integer
  );

  port (
    -- inputs
    clk       : in std_logic;
    rst       : in std_logic;
    ins_valid : in std_logic;
    -- outputs
    ins_ready : out std_logic);
end sink;

architecture arch of sink is

begin

  ins_ready <= '1';

end arch;
