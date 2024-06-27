library ieee;
use ieee.std_logic_1164.all;

entity branch_simple is
  port (
    -- inputs
    condition  : in std_logic;
    valid      : in std_logic;
    outs_ready : in std_logic_vector(1 downto 0);
    -- outputs
    ins_valid : out std_logic_vector(1 downto 0);
    ins_ready : out std_logic
  );
end branch_simple;

architecture arch of branch_simple is
begin
  ins_valid(1) <= (not condition) and valid;
  ins_valid(0) <= condition and valid;
  ins_ready    <= (outs_ready(1) and not condition) or (outs_ready(0) and condition);
end arch;
