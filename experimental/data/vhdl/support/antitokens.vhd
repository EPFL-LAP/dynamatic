library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity antitokens is port (
  clk, reset                 : in std_logic;
  pvalid1, pvalid0           : in std_logic;
  kill1, kill0               : out std_logic;
  generate_at1, generate_at0 : in std_logic;
  stop_valid                 : out std_logic);

end antitokens;

architecture arch of antitokens is

  signal reg_in0, reg_in1, reg_out0, reg_out1 : std_logic;

begin

  reg0 : process (clk, reset, reg_in0)
  begin
    if (reset = '1') then
      reg_out0 <= '0';
    else
      if (rising_edge(clk)) then
        reg_out0 <= reg_in0;
      end if;
    end if;
  end process reg0;

  reg1 : process (clk, reset, reg_in1)
  begin
    if (reset = '1') then
      reg_out1 <= '0';
    else
      if (rising_edge(clk)) then
        reg_out1 <= reg_in1;
      end if;
    end if;
  end process reg1;

  reg_in0 <= not pvalid0 and (generate_at0 or reg_out0);
  reg_in1 <= not pvalid1 and (generate_at1 or reg_out1);

  stop_valid <= reg_out0 or reg_out1;

  kill0 <= generate_at0 or reg_out0;
  kill1 <= generate_at1 or reg_out1;
end arch;
