-----------------------------------------------------------------------
-- select, version 0.0
-----------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity antitokens is port(
  clk, reset : in std_logic;
  pvalid1, pvalid0 : in std_logic;
  kill1, kill0 : out std_logic;
  generate_at1, generate_at0 : in std_logic;
  stop_valid : out std_logic);

end antitokens;


architecture arch of antitokens is

signal reg_in0, reg_in1, reg_out0, reg_out1 : std_logic;
  
begin

  reg0 : process(clk, reset, reg_in0)
    begin
      if(reset='1') then
        reg_out0 <= '0'; 
      else
        if(rising_edge(clk))then
          reg_out0 <= reg_in0;
        end if;
      end if;
    end process reg0;

  reg1 : process(clk, reset, reg_in1)
    begin
      if(reset='1') then
        reg_out1 <= '0'; 
      else
        if(rising_edge(clk))then
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

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
USE work.customTypes.all;

entity select is 
Generic (
 BITWIDTH: integer
);
-- llvm select: operand(0) is condition, operand(1) is true, operand(2) is false
-- here, dataInArray(0) is true, dataInArray(1) is false operand
port (
    clk, rst :      in std_logic;
    dataInArray :   in data_array (1 downto 0)(BITWIDTH - 1 downto 0);
    dataOutArray :  out data_array (0 downto 0)(BITWIDTH - 1 downto 0);    
    pValidArray :   in std_logic_vector(2 downto 0);
    nReadyArray :   in std_logic_vector(0 downto 0);
    validArray :    out std_logic_vector(0 downto 0);
    readyArray :    out std_logic_vector(2 downto 0);
    condition: in data_array (0 downto 0)(0 downto 0)
  );

  end select;

architecture arch of select is
    signal  ee, validInternal : std_logic;
    signal  kill0, kill1 : std_logic;
  signal  antitokenStop:  std_logic;
  signal  g0, g1: std_logic;


    begin
      
      ee <= pValidArray(0) and ((not condition(0)(0) and pValidArray(2)) or (condition(0)(0) and pValidArray(1))); --condition and one input
      validInternal <= ee and not antitokenStop; -- propagate ee if not stopped by antitoken

      g0 <= not pValidArray(1) and validInternal and nReadyArray(0);
      g1 <= not pValidArray(2) and validInternal and nReadyArray(0);

      validArray(0) <= validInternal;
      readyArray(1) <= (not pValidArray(1)) or (validInternal and nReadyArray(0)) or kill0; -- normal join or antitoken
      readyArray(2) <= (not pValidArray(2)) or (validInternal and nReadyArray(0)) or kill1; --normal join or antitoken
      readyArray(0) <= (not pValidArray(0)) or (validInternal and nReadyArray(0)); --like normal join

      dataOutArray(0) <= dataInArray(1) when (condition(0)(0) = '0') else dataInArray(0);

      Antitokens: entity work.antitokens
        port map ( clk, rst, 
        pValidArray(2), pValidArray(1), 
              kill1, kill0,
              g1, g0, 
              antitokenStop);
    end arch;