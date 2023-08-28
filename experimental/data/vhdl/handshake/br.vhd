library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity br is generic (BITWIDTH : integer);
port (
  -- inputs
  clk       : in std_logic;
  rst       : in std_logic;
  ins       : in std_logic_vector(BITWIDTH - 1 downto 0);
  ins_valid : in std_logic;
  ins_ready : out std_logic
  -- outputs
  outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
  outs_valid : out std_logic;
  outs_ready : in std_logic);
end br;

architecture arch of br is
begin
  outs       <= ins;
  outs_valid <= ins_valid;
  ins_ready  <= outs_ready;
end arch;
