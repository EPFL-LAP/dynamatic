library ieee;
use ieee.std_logic_1164.all;

entity branch_simple is
  port (
    -- inputs
    condition      : in std_logic;
    valid          : in std_logic;
    trueOut_ready  : in std_logic;
    falseOut_ready : in std_logic;
    -- outputs
    trueOut_valid  : out std_logic;
    falseOut_valid : out std_logic;
    ins_ready : out std_logic
  );
end branch_simple;

architecture arch of branch_simple is
begin
  falseOut_valid <= (not condition) and valid;
  trueOut_valid  <= condition and valid;
  ins_ready      <= (falseOut_ready and not condition) or (trueOut_ready and condition);
end arch;
