----------------------------------------------------------------------- 
-- ret, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity ret_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
  clk, rst : in std_logic; 
  dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(0 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(0 downto 0));
end entity;

architecture arch of ret_op is

begin 

tehb: entity work.TEHB(arch) generic map (1, 1, DATA_SIZE_IN, DATA_SIZE_IN)
        port map (
        --inputs
            clk => clk, 
            rst => rst, 
            pValidArray(0)  => pValidArray(0), 
            nReadyArray(0) => nReadyArray(0),    
            validArray(0) => validArray(0), 
        --outputs
            readyArray(0) => readyArray(0),   
            dataInArray(0) => dataInArray(0),
            dataOutArray(0) => dataOutArray(0)
        );

end architecture;

----------------------------------------------------------------------- 
-- int add, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity add_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port (
  clk, rst : in std_logic; 
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of add_op is

signal join_valid : STD_LOGIC;

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= std_logic_vector(unsigned(dataInArray(0)) + unsigned (dataInArray(1)));
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- int sub, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity sub_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port (
  clk, rst : in std_logic; 
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of sub_op is

signal join_valid : STD_LOGIC;

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= std_logic_vector(unsigned(dataInArray(0)) - unsigned (dataInArray(1)));
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- int mul, version 0.0
-----------------------------------------------------------------------




-----------------------------------------------------------------------
-- logic and, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity and_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port (
  clk, rst : in std_logic; 
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of and_op is

signal join_valid : STD_LOGIC;

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= dataInArray(0) and dataInArray(1);
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- logic or, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity or_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port (
  clk, rst : in std_logic; 
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of or_op is

signal join_valid : STD_LOGIC;

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= dataInArray(0) or dataInArray(1);
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- logic xor, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity xor_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port (
  clk, rst : in std_logic; 
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of xor_op is

signal join_valid : STD_LOGIC;

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= dataInArray(0) xor dataInArray(1);
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- sext, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity sext_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
  clk, rst : in std_logic; 
  dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(0 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(0 downto 0));
end entity;

architecture arch of sext_op is

    signal join_valid : STD_LOGIC;

begin 

    dataOutArray(0)<= std_logic_vector(IEEE.numeric_std.resize(signed(dataInArray(0)),DATA_SIZE_OUT));
    validArray <= pValidArray;
    readyArray(0) <= nReadyArray(0);

end architecture;

-----------------------------------------------------------------------
-- zext, version 0.0
-- TODO check signed/unsigned
-----------------------------------------------------------------------
Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity zext_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
  clk, rst : in std_logic; 
  dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(0 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(0 downto 0));
end entity;

architecture arch of zext_op is

    signal join_valid : STD_LOGIC;

begin 

    dataOutArray(0)<= std_logic_vector(IEEE.numeric_std.resize(unsigned(dataInArray(0)),DATA_SIZE_OUT));
    validArray <= pValidArray;
    readyArray(0) <= nReadyArray(0);

end architecture;
-----------------------------------------------------------------------
-- shl, version 0.0
-----------------------------------------------------------------------
Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity shl_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
  clk, rst : in std_logic; 
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of shl_op is

    signal join_valid : STD_LOGIC;

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= std_logic_vector(shift_left(unsigned(dataInArray(0)),to_integer(unsigned('0' & dataInArray(1)(DATA_SIZE_IN-2 downto 0)))));
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- ashr, version 0.0
-----------------------------------------------------------------------
Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity ashr_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
  clk, rst : in std_logic; 
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of ashr_op is

    signal join_valid : STD_LOGIC;

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= std_logic_vector(shift_right(signed(dataInArray(0)),to_integer(unsigned('0' & dataInArray(1)(DATA_SIZE_IN-2 downto 0)))));
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- lshr, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity lshr_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
  clk, rst : in std_logic; 
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of lshr_op is

    signal join_valid : STD_LOGIC;

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= std_logic_vector(shift_right(unsigned(dataInArray(0)),to_integer(unsigned('0' & dataInArray(1)(DATA_SIZE_IN-2 downto 0)))));
    validArray(0) <= join_valid;


end architecture;

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

entity select_op is 
Generic (
 INPUTS : integer ; OUTPUTS : integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
-- llvm select: operand(0) is condition, operand(1) is true, operand(2) is false
-- here, dataInArray(0) is true, dataInArray(1) is false operand
port (
    clk, rst :      in std_logic;
    dataInArray :   in data_array (1 downto 0)(DATA_SIZE_IN - 1 downto 0);
    dataOutArray :  out data_array (0 downto 0)(DATA_SIZE_OUT - 1 downto 0);    
    pValidArray :   in std_logic_vector(2 downto 0);
    nReadyArray :   in std_logic_vector(0 downto 0);
    validArray :    out std_logic_vector(0 downto 0);
    readyArray :    out std_logic_vector(2 downto 0);
    condition: in data_array (0 downto 0)(0 downto 0)
  );

  end select_op;

architecture arch of select_op is
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


-----------------------------------------------------------------------
-- icmp eq, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity icmp_eq_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of icmp_eq_op is
--eq: yields true if the operands are equal, false otherwise. No sign interpretation is necessary or performed.
    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= one when (dataInArray(0) = dataInArray(1) ) else zero;
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- icmp ne, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity icmp_ne_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of icmp_ne_op is
--ne: yields true if the operands are unequal, false otherwise. No sign interpretation is necessary or performed.
    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= zero when (dataInArray(0) = dataInArray(1) ) else one;
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- icmp ugt, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity icmp_ugt_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of icmp_ugt_op is
--ugt: interprets the operands as unsigned values and yields true if op1 is greater than op2.
    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= one when (unsigned(dataInArray(0)) > unsigned(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- icmp uge, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity icmp_uge_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of icmp_uge_op is
--uge: interprets the operands as unsigned values and yields true if op1 is greater than or equal to op2.
    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= one when (unsigned(dataInArray(0)) >= unsigned(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- icmp sgt, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity icmp_sgt_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of icmp_sgt_op is
-- sgt: interprets the operands as signed values and yields true if op1 is greater than op2.
    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";
    signal res: std_logic_vector (0 downto 0);
    signal nready_tmp : std_logic;
begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= one when (signed(dataInArray(0)) > signed(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- icmp sge, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity icmp_sge_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of icmp_sge_op is
-- sge: interprets the operands as signed values and yields true if op1 is greater than or equal to op2.
    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";
    signal res: std_logic_vector (0 downto 0);
    signal nready_tmp : std_logic;
begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= one when (signed(dataInArray(0)) >= signed(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- icmp ult, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity icmp_ult_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of icmp_ult_op is
--ugt: interprets the operands as unsigned values and yields true if op1 is greater than op2.
    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= one when (unsigned(dataInArray(0)) < unsigned(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- icmp ule, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity icmp_ule_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of icmp_ule_op is
-- ule: interprets the operands as unsigned values and yields true if op1 is less than or equal to op2.
    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= one when (unsigned(dataInArray(0)) <= unsigned(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- icmp slt, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity icmp_slt_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of icmp_slt_op is
-- slt: interprets the operands as signed values and yields true if op1 is less than op2.
    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= one when (signed(dataInArray(0)) < signed(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- icmp sle, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity icmp_sle_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of icmp_sle_op is
-- slt: interprets the operands as signed values and yields true if op1 is less than op2.
    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= one when (signed(dataInArray(0)) <= signed(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;

----------------------------------------------------------------------- 
-- getelementptr, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;
entity getelementptr_op is
Generic ( INPUTS: integer; OUTPUTS : Integer; INPUT_SIZE: Integer; OUTPUT_SIZE : Integer; CONST_SIZE : Integer);

    -- component inputs: i, j, k,... dimx, dimy, dimz
    -- inputs: total number of inputs
    -- outputs: total number of outputs
    -- input/output size: bitwidths
    -- const_size: number of dimensions (dimx, ..)

port(
  clk : IN STD_LOGIC;
  rst : IN STD_LOGIC;
  pValidArray : IN std_logic_vector(INPUTS - 1 downto 0);
  nReadyArray : IN STD_LOGIC_VECTOR(0 downto 0);
  validArray : OUT STD_LOGIC_VECTOR(0 downto 0);
  readyArray : OUT std_logic_vector(INPUTS -1 downto 0);
  dataInArray: IN data_array(INPUTS-1 downto 0)(INPUT_SIZE - 1 DOWNTO 0);
  dataOutArray : OUT data_array(0 downto 0)(OUTPUT_SIZE - 1 DOWNTO 0));

attribute use_dsp48 : string;
attribute use_dsp48 of getelementptr_op : entity is "no";

end entity;


architecture arch of  getelementptr_op is

    signal join_valid : STD_LOGIC;


begin 

    -- join only for variable inputs
    join_write_temp:   entity work.join(arch) generic map(INPUTS - CONST_SIZE)
            port map( pValidArray(INPUTS - CONST_SIZE - 1 downto 0),  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray(INPUTS - CONST_SIZE - 1 downto 0));   --readyarray 

    readyArray (INPUTS -1 downto INPUTS - CONST_SIZE) <=  (others => '1');

    validArray(0) <= join_valid;

    -- convert index [i][j][k] or array[dimX][dimY][dimZ] into index [i * dimY*dimZ + j * dimZ + k] 
    process(dataInArray)
        variable tmp_data_out  : unsigned(INPUT_SIZE - 1 downto 0);
        variable tmp_const  : integer;
        variable tmp_mul  : integer;

    begin
        tmp_data_out  := (others => '0');

        for I in 0 to INPUTS - CONST_SIZE - 1 loop
          tmp_const  := 1;
          for J in INPUTS - CONST_SIZE + I to INPUTS - 1 loop
              tmp_const  := tmp_const * to_integer(unsigned(dataInArray(J)));
          end loop;
          tmp_mul := to_integer(unsigned(dataInArray(I))) * tmp_const; 
          tmp_data_out  := tmp_data_out + to_unsigned(tmp_mul, 32);
        end loop;
    dataOutArray(0)  <= std_logic_vector(resize(tmp_data_out, OUTPUT_SIZE));

    end process;

end architecture;


-----------------------------------------------------------------------
-- fcmp oeq, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;

use work.customTypes.all;

entity fcmp_oeq_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of fcmp_oeq_op is

    --Interface to vivado component
    component array_RAM_fcmp_32cud is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 2;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 1
        );
        port (
            clk    : in  std_logic;
            reset  : in  std_logic;
            ce     : in  std_logic;
            din0   : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1   : in  std_logic_vector(din1_WIDTH-1 downto 0);
            opcode : in  std_logic_vector(4 downto 0);
            dout   : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal join_valid : STD_LOGIC;
    constant alu_opcode : std_logic_vector(4 downto 0) := "00001";

begin 
    
    --TODO check with lana
    dataOutArray(0)(DATA_SIZE_OUT - 1 downto 1) <= (others => '0');

    array_RAM_fcmp_32ns_32ns_1_2_1_u1 : component array_RAM_fcmp_32cud 
    generic map (
        ID => 1,
        NUM_STAGE => 2,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 1)
    port map (
        clk => clk,
        reset => rst,
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        ce => nReadyArray(0),
        opcode => alu_opcode,
        dout(0) => dataOutArray(0)(0)
    );

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
    generic map(1)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));

end architecture;

-----------------------------------------------------------------------
-- fcmp ogt, version 0.0
-----------------------------------------------------------------------
Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fcmp_ogt_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of fcmp_ogt_op is

    --Interface to vivado component
    component array_RAM_fcmp_32cud is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 2;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 1
        );
        port (
            clk    : in  std_logic;
            reset  : in  std_logic;
            ce     : in  std_logic;
            din0   : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1   : in  std_logic_vector(din1_WIDTH-1 downto 0);
            opcode : in  std_logic_vector(4 downto 0);
            dout   : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal join_valid : STD_LOGIC;
    constant alu_opcode : std_logic_vector(4 downto 0) := "00010";

begin 

    --TODO check with lana
    dataOutArray(0)(DATA_SIZE_OUT - 1 downto 1) <= (others => '0');


    array_RAM_fcmp_32ns_32ns_1_2_1_u1 : component array_RAM_fcmp_32cud 
    generic map (
        ID => 1,
        NUM_STAGE => 2,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 1)
    port map (
        clk => clk,
        reset => rst,
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        ce => nReadyArray(0),
        opcode => alu_opcode,
        dout(0) => dataOutArray(0)(0));

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
    generic map(1)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));

end architecture;
-----------------------------------------------------------------------
-- fcmp oge, version 0.0
-- TODO
-----------------------------------------------------------------------
Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fcmp_oge_op is
    Generic (
    INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port(
        clk, rst : in std_logic; 
        dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
        dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
        pValidArray : in std_logic_vector(1 downto 0);
        nReadyArray : in std_logic_vector(0 downto 0);
        validArray : out std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(1 downto 0));
end entity;
    
architecture arch of fcmp_oge_op is

    --Interface to vivado component
    component array_RAM_fcmp_32cud is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 2;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 1
        );
        port (
            clk    : in  std_logic;
            reset  : in  std_logic;
            ce     : in  std_logic;
            din0   : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1   : in  std_logic_vector(din1_WIDTH-1 downto 0);
            opcode : in  std_logic_vector(4 downto 0);
            dout   : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal join_valid : STD_LOGIC;
    constant alu_opcode : std_logic_vector(4 downto 0) := "00011";

begin 

    --TODO check with lana
    dataOutArray(0)(DATA_SIZE_OUT - 1 downto 1) <= (others => '0');


    array_RAM_fcmp_32ns_32ns_1_2_1_u1 : component array_RAM_fcmp_32cud 
    generic map (
        ID => 1,
        NUM_STAGE => 2,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 1)
    port map (
        clk => clk,
        reset => rst,
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        ce => nReadyArray(0),
        opcode => alu_opcode,
        dout(0) => dataOutArray(0)(0));

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
    generic map(1)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));


end architecture;

-----------------------------------------------------------------------
-- fcmp olt, version 0.0
-- TODO
-----------------------------------------------------------------------
Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fcmp_olt_op is
    Generic (
    INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port(
        clk, rst : in std_logic; 
        dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
        dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
        pValidArray : in std_logic_vector(1 downto 0);
        nReadyArray : in std_logic_vector(0 downto 0);
        validArray : out std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(1 downto 0));
end entity;
    
architecture arch of fcmp_olt_op is

    --Interface to vivado component
    component array_RAM_fcmp_32cud is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 2;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 1
        );
        port (
            clk    : in  std_logic;
            reset  : in  std_logic;
            ce     : in  std_logic;
            din0   : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1   : in  std_logic_vector(din1_WIDTH-1 downto 0);
            opcode : in  std_logic_vector(4 downto 0);
            dout   : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal join_valid : STD_LOGIC;
    constant alu_opcode : std_logic_vector(4 downto 0) := "00100";

begin 

    --TODO check with lana
    dataOutArray(0)(DATA_SIZE_OUT - 1 downto 1) <= (others => '0');


    array_RAM_fcmp_32ns_32ns_1_2_1_u1 : component array_RAM_fcmp_32cud 
    generic map (
        ID => 1,
        NUM_STAGE => 2,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 1)
    port map (
        clk => clk,
        reset => rst,
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        ce => nReadyArray(0),
        opcode => alu_opcode,
        dout(0) => dataOutArray(0)(0));

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
    generic map(1)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));

end architecture;


-----------------------------------------------------------------------
-- fcmp ole, version 0.0
-- TODO
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;
entity fcmp_ole_op is
    Generic (
    INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port(
        clk, rst : in std_logic; 
        dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
        dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
        pValidArray : in std_logic_vector(1 downto 0);
        nReadyArray : in std_logic_vector(0 downto 0);
        validArray : out std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(1 downto 0));
    end entity;
    
    architecture arch of fcmp_ole_op is
    
        --Interface to vivado component
        component array_RAM_fcmp_32cud is
            generic (
                ID         : integer := 1;
                NUM_STAGE  : integer := 2;
                din0_WIDTH : integer := 32;
                din1_WIDTH : integer := 32;
                dout_WIDTH : integer := 1
            );
            port (
                clk    : in  std_logic;
                reset  : in  std_logic;
                ce     : in  std_logic;
                din0   : in  std_logic_vector(din0_WIDTH-1 downto 0);
                din1   : in  std_logic_vector(din1_WIDTH-1 downto 0);
                opcode : in  std_logic_vector(4 downto 0);
                dout   : out std_logic_vector(dout_WIDTH-1 downto 0)
            );
        end component;
    
        signal join_valid : STD_LOGIC;
        constant alu_opcode : std_logic_vector(4 downto 0) := "00101";
    
    begin 
    
    --TODO check with lana
    dataOutArray(0)(DATA_SIZE_OUT - 1 downto 1) <= (others => '0');


    array_RAM_fcmp_32ns_32ns_1_2_1_u1 : component array_RAM_fcmp_32cud 
    generic map (
        ID => 1,
        NUM_STAGE => 2,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 1)
    port map (
        clk => clk,
        reset => rst,
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        ce => nReadyArray(0),
        opcode => alu_opcode,
        dout(0) => dataOutArray(0)(0));

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
    generic map(1)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));

    
    end architecture;
-----------------------------------------------------------------------
-- fcmp one, version 0.0
-- TODO
-----------------------------------------------------------------------
Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fcmp_one_op is
    Generic (
    INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port(
        clk, rst : in std_logic; 
        dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
        dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
        pValidArray : in std_logic_vector(1 downto 0);
        nReadyArray : in std_logic_vector(0 downto 0);
        validArray : out std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(1 downto 0));
    end entity;
    
    architecture arch of fcmp_one_op is
    
        --Interface to vivado component
        component array_RAM_fcmp_32cud is
            generic (
                ID         : integer := 1;
                NUM_STAGE  : integer := 2;
                din0_WIDTH : integer := 32;
                din1_WIDTH : integer := 32;
                dout_WIDTH : integer := 1
            );
            port (
                clk    : in  std_logic;
                reset  : in  std_logic;
                ce     : in  std_logic;
                din0   : in  std_logic_vector(din0_WIDTH-1 downto 0);
                din1   : in  std_logic_vector(din1_WIDTH-1 downto 0);
                opcode : in  std_logic_vector(4 downto 0);
                dout   : out std_logic_vector(dout_WIDTH-1 downto 0)
            );
        end component;
    
        signal join_valid : STD_LOGIC;
        constant alu_opcode : std_logic_vector(4 downto 0) := "00110";
    
    begin 
    
        --TODO check with lana
    dataOutArray(0)(DATA_SIZE_OUT - 1 downto 1) <= (others => '0');


    array_RAM_fcmp_32ns_32ns_1_2_1_u1 : component array_RAM_fcmp_32cud 
    generic map (
        ID => 1,
        NUM_STAGE => 2,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 1)
    port map (
        clk => clk,
        reset => rst,
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        ce => nReadyArray(0),
        opcode => alu_opcode,
        dout(0) => dataOutArray(0)(0));

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
    generic map(1)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));

    
    end architecture;

-----------------------------------------------------------------------
-- fcmp ord, version 0.0
-- TODO
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fcmp_ord_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of fcmp_ord_op is

    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    dataOutArray(0) <= one when (signed(dataInArray(0)) <= signed(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- fcmp uno, version 0.0
-- TODO
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fcmp_uno_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of fcmp_uno_op is

    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    dataOutArray(0) <= one when (signed(dataInArray(0)) <= signed(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- fcmp ueq, version 0.0
-- TODO
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fcmp_ueq_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of fcmp_ueq_op is

    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    dataOutArray(0) <= one when (signed(dataInArray(0)) <= signed(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- fcmp uge, version 0.0
-- TODO
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fcmp_uge_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of fcmp_uge_op is

    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    dataOutArray(0) <= one when (signed(dataInArray(0)) <= signed(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- fcmp ult, version 0.0
-- TODO
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fcmp_ult_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of fcmp_ult_op is

    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    dataOutArray(0) <= one when (signed(dataInArray(0)) <= signed(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- fcmp ule, version 0.0
-- TODO
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fcmp_ule is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of fcmp_ule is

    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    dataOutArray(0) <= one when (signed(dataInArray(0)) <= signed(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- fcmp une, version 0.0
-- TODO
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fcmp_une is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of fcmp_une is

    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    dataOutArray(0) <= one when (signed(dataInArray(0)) <= signed(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;

-----------------------------------------------------------------------
-- fcmp ugt, version 0.0
-- TODO
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fcmp_ugt is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of fcmp_ugt is

    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    dataOutArray(0) <= one when (signed(dataInArray(0)) <= signed(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;

----------------------------------------------------------------------- 
-- float add, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fadd_op is
Generic (
 INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port (
  clk : IN STD_LOGIC;
  rst : IN STD_LOGIC;
  pValidArray : IN std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : OUT std_logic_vector(1 downto 0);
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;

architecture arch of fadd_op is

    -- Interface to Vivado component
    component array_RAM_fadd_32bkb is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 10;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 32
        );
        port (
            clk   : in  std_logic;
            reset : in  std_logic;
            ce    : in  std_logic;
            din0  : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1  : in  std_logic_vector(din1_WIDTH-1 downto 0);
            dout  : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal join_valid : STD_LOGIC;

    signal buff_valid, oehb_valid, oehb_ready : STD_LOGIC;
    signal oehb_dataOut, oehb_datain : std_logic_vector(0 downto 0);
    
    begin 
  

        join: entity work.join(arch) generic map(2)
        port map( pValidArray,  
                oehb_ready,                        
                join_valid,                  
                readyArray);   

        buff: entity work.delay_buffer(arch) generic map(8)
        port map(clk,
                rst,
                join_valid,
                oehb_ready,
                buff_valid);

        oehb: entity work.OEHB(arch) generic map (1, 1, 1, 1)
                port map (
                --inputspValidArray
                    clk => clk, 
                    rst => rst, 
                    pValidArray(0)  => buff_valid, -- real or speculatef condition (determined by merge1)
                    nReadyArray(0) => nReadyArray(0),    
                    validArray(0) => validArray(0), 
                --outputs
                    readyArray(0) => oehb_ready,   
                    dataInArray(0) => oehb_datain,
                    dataOutArray(0) => oehb_dataOut
                );

    
        array_RAM_fadd_32ns_32ns_32_10_full_dsp_1_U1 :  component array_RAM_fadd_32bkb
        port map (
            clk   => clk,
            reset => rst,
            ce    => oehb_ready,
            din0  => dataInArray(0),
            din1  => dataInArray(1),
            dout  => dataOutArray(0));
    
end architecture;



----------------------------------------------------------------------- 
-- float sub, version 0.0
-----------------------------------------------------------------------
Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fsub_op is
Generic (
 INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port (
  clk : IN STD_LOGIC;
  rst : IN STD_LOGIC;
  pValidArray : IN std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : OUT std_logic_vector(1 downto 0);
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;

architecture arch of fsub_op is

    -- Interface to Vivado component
    component array_RAM_fsub_32bkb is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 10;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 32
        );
        port (
            clk   : in  std_logic;
            reset : in  std_logic;
            ce    : in  std_logic;
            din0  : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1  : in  std_logic_vector(din1_WIDTH-1 downto 0);
            dout  : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal join_valid : STD_LOGIC;

    signal buff_valid, oehb_valid, oehb_ready : STD_LOGIC;
    signal oehb_dataOut, oehb_datain : std_logic_vector(0 downto 0);
    
begin 
    
        join: entity work.join(arch) generic map(2)
        port map( pValidArray,  
                oehb_ready,                        
                join_valid,                  
                readyArray);   

        buff: entity work.delay_buffer(arch) generic map(8)
        port map(clk,
                rst,
                join_valid,
                oehb_ready,
                buff_valid);

        oehb: entity work.OEHB(arch) generic map (1, 1, 1, 1)
                port map (
                --inputspValidArray
                    clk => clk, 
                    rst => rst, 
                    pValidArray(0)  => buff_valid, -- real or speculatef condition (determined by merge1)
                    nReadyArray(0) => nReadyArray(0),    
                    validArray(0) => validArray(0), 
                --outputs
                    readyArray(0) => oehb_ready,   
                    dataInArray(0) => oehb_datain,
                    dataOutArray(0) => oehb_dataOut
                );

    array_RAM_fsub_32ns_32ns_32_10_full_dsp_1_U1 :  component array_RAM_fsub_32bkb
    port map (
        clk   => clk,
        reset => rst,
        ce    => oehb_ready,
        din0  => dataInArray(0),
        din1  => dataInArray(1),
        dout  => dataOutArray(0));

end architecture;

----------------------------------------------------------------------- 
-- float mul, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fmul_op is
Generic (
 INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port (
  clk : IN STD_LOGIC;
  rst : IN STD_LOGIC;
  pValidArray : IN std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : OUT std_logic_vector(1 downto 0);
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;

architecture arch of fmul_op is

    -- Interface to Vivado component
    component array_RAM_fmul_32cud is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 6;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 32
        );
        port (
            clk   : in  std_logic;
            reset : in  std_logic;
            ce    : in  std_logic;
            din0  : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1  : in  std_logic_vector(din1_WIDTH-1 downto 0);
            dout  : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

        signal join_valid : STD_LOGIC;

    signal buff_valid, oehb_valid, oehb_ready : STD_LOGIC;
    signal oehb_dataOut, oehb_datain : std_logic_vector(0 downto 0);
    
begin 
    
        join: entity work.join(arch) generic map(2)
        port map( pValidArray,  
                oehb_ready,                        
                join_valid,                  
                readyArray);   

        buff: entity work.delay_buffer(arch) generic map(4)
        port map(clk,
                rst,
                join_valid,
                oehb_ready,
                buff_valid);

        oehb: entity work.OEHB(arch) generic map (1, 1, 1, 1)
                port map (
                --inputspValidArray
                    clk => clk, 
                    rst => rst, 
                    pValidArray(0)  => buff_valid, -- real or speculatef condition (determined by merge1)
                    nReadyArray(0) => nReadyArray(0),    
                    validArray(0) => validArray(0), 
                --outputs
                    readyArray(0) => oehb_ready,   
                    dataInArray(0) => oehb_datain,
                    dataOutArray(0) => oehb_dataOut
                );

    array_RAM_fmul_32ns_32ns_32_6_max_dsp_1_U1 :  component array_RAM_fmul_32cud
    port map (
        clk   => clk,
        reset => rst,
        ce    => oehb_ready, 
        din0  => dataInArray(0),
        din1  => dataInArray(1),
        dout  => dataOutArray(0));

end architecture;

----------------------------------------------------------------------- 
-- float neg, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fneg_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port (
  clk, rst : in std_logic; 
  dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(0 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(0 downto 0));
end entity;

architecture arch of fneg_op is

    constant msb_mask : std_logic_vector(31 downto 0) := (31 => '1', others => '0');

begin 

    dataOutArray(0) <= dataInArray(0) xor msb_mask;
    validArray(0) <= pValidArray(0);
    readyArray(0) <= nReadyArray(0);


end architecture;
----------------------------------------------------------------------- 
-- unsigned int division, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity udiv_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
 clk, rst : in std_logic; 
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of udiv_op is

    -- Interface to Vivado component
    component array_RAM_udiv_32ns_32ns_32_36_1 is
        generic (
            ID : INTEGER;
            NUM_STAGE : INTEGER;
            din0_WIDTH : INTEGER;
            din1_WIDTH : INTEGER;
            dout_WIDTH : INTEGER);
        port (
            clk : IN STD_LOGIC;
            reset : IN STD_LOGIC;
            ce : IN STD_LOGIC;
            din0 : IN STD_LOGIC_VECTOR(din0_WIDTH - 1 DOWNTO 0);
            din1 : IN STD_LOGIC_VECTOR(din1_WIDTH - 1 DOWNTO 0);
            dout : OUT STD_LOGIC_VECTOR(dout_WIDTH - 1 DOWNTO 0));
    end component;

    signal join_valid : STD_LOGIC;

begin 
    array_RAM_udiv_32ns_32ns_32_36_1_U1 : component array_RAM_udiv_32ns_32ns_32_36_1
    generic map (
        ID => 1,
        NUM_STAGE => 36,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 32)
    port map (
        clk => clk,
        reset => rst,
        ce => nReadyArray(0),
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        dout => dataOutArray(0));

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray

    buff: entity work.delay_buffer(arch) 
    generic map(35)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));

end architecture;

----------------------------------------------------------------------- 
-- signed int division, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity sdiv_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
 clk, rst : in std_logic; 
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of sdiv_op is

    -- Interface to Vivado component
    component array_RAM_sdiv_32ns_32ns_32_36_1 is
        generic (
            ID : INTEGER;
            NUM_STAGE : INTEGER;
            din0_WIDTH : INTEGER;
            din1_WIDTH : INTEGER;
            dout_WIDTH : INTEGER);
        port (
            clk : IN STD_LOGIC;
            reset : IN STD_LOGIC;
            ce : IN STD_LOGIC;
            din0 : IN STD_LOGIC_VECTOR(din0_WIDTH - 1 DOWNTO 0);
            din1 : IN STD_LOGIC_VECTOR(din1_WIDTH - 1 DOWNTO 0);
            dout : OUT STD_LOGIC_VECTOR(dout_WIDTH - 1 DOWNTO 0));
    end component;

    signal join_valid : STD_LOGIC;

begin 
    array_RAM_sdiv_32ns_32ns_32_36_1_U1 : component array_RAM_sdiv_32ns_32ns_32_36_1
    generic map (
        ID => 1,
        NUM_STAGE => 36,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 32)
    port map (
        clk => clk,
        reset => rst,
        ce => nReadyArray(0),
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        dout => dataOutArray(0));

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray

    buff: entity work.delay_buffer(arch) 
    generic map(35)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));

end architecture;

----------------------------------------------------------------------- 
-- float division, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fdiv_op is
Generic (
 INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port (
  clk : IN STD_LOGIC;
  rst : IN STD_LOGIC;
  pValidArray : IN std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : OUT std_logic_vector(1 downto 0);
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;

architecture arch of fdiv_op is

    -- Interface to Vivado component
    component array_RAM_fdiv_32ns_32ns_32_30_1 is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 30;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 32
        );
        port (
            clk   : in  std_logic;
            reset : in  std_logic;
            ce    : in  std_logic;
            din0  : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1  : in  std_logic_vector(din1_WIDTH-1 downto 0);
            dout  : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal join_valid : STD_LOGIC;
    signal buff_valid, oehb_valid, oehb_ready : STD_LOGIC;
    signal oehb_dataOut, oehb_datain : std_logic_vector(0 downto 0);
    
begin 
    
    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      oehb_ready,     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
        generic map(28)
        port map(clk,
                 rst,
                 join_valid,
                 oehb_ready,
                 buff_valid);

    oehb: entity work.OEHB(arch) generic map (1, 1, 1, 1)
        port map (
        --inputspValidArray
            clk => clk, 
            rst => rst, 
            pValidArray(0)  => buff_valid, -- real or speculatef condition (determined by merge1)
            nReadyArray(0) => nReadyArray(0),    
            validArray(0) => validArray(0), 
        --outputs
            readyArray(0) => oehb_ready,   
            dataInArray(0) => oehb_datain,
            dataOutArray(0) => oehb_dataOut
        );

    array_RAM_fdiv_32ns_32ns_32_30_1_U1 :  component array_RAM_fdiv_32ns_32ns_32_30_1
    port map (
        clk   => clk,
        reset => rst,
        ce    => oehb_ready,
        din0  => dataInArray(0),
        din1  => dataInArray(1),
        dout  => dataOutArray(0));

end architecture;

----------------------------------------------------------------------- 
-- srem, remainder of signed division, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity srem_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
 clk, rst : in std_logic; 
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of srem_op is
    --Interface to vivado component
    component array_RAM_srem_32ns_32ns_32_36_1 is
        generic (
            ID : INTEGER;
            NUM_STAGE : INTEGER;
            din0_WIDTH : INTEGER;
            din1_WIDTH : INTEGER;
            dout_WIDTH : INTEGER);
        port (
            clk : IN STD_LOGIC;
            reset : IN STD_LOGIC;
            ce : IN STD_LOGIC;
            din0 : IN STD_LOGIC_VECTOR(din0_WIDTH - 1 DOWNTO 0);
            din1 : IN STD_LOGIC_VECTOR(din1_WIDTH - 1 DOWNTO 0);
            dout : OUT STD_LOGIC_VECTOR(dout_WIDTH - 1 DOWNTO 0));
    end component;

    signal join_valid : STD_LOGIC;

begin 

array_RAM_srem_32ns_32ns_32_36_1_U1 : component array_RAM_srem_32ns_32ns_32_36_1
    generic map (
        ID => 1,
        NUM_STAGE => 36,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 32)
    port map (
        clk => clk,
        reset => rst,
        ce => nReadyArray(0),
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        dout => dataOutArray(0));

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray

    buff: entity work.delay_buffer(arch) 
    generic map(35)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));

end architecture;

----------------------------------------------------------------------- 
-- urem, remainder of unsigned division, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity urem_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
 clk, rst : in std_logic; 
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of urem_op is

    --Interface to vivado component
    component array_RAM_urem_32ns_32ns_32_36_1 is
        generic (
            ID : INTEGER;
            NUM_STAGE : INTEGER;
            din0_WIDTH : INTEGER;
            din1_WIDTH : INTEGER;
            dout_WIDTH : INTEGER);
        port (
            clk : IN STD_LOGIC;
            reset : IN STD_LOGIC;
            ce : IN STD_LOGIC;
            din0 : IN STD_LOGIC_VECTOR(din0_WIDTH - 1 DOWNTO 0);
            din1 : IN STD_LOGIC_VECTOR(din1_WIDTH - 1 DOWNTO 0);
            dout : OUT STD_LOGIC_VECTOR(dout_WIDTH - 1 DOWNTO 0));
    end component;
    signal join_valid : STD_LOGIC;

begin 

    array_RAM_urem_32ns_32ns_32_36_1_U1 : component array_RAM_urem_32ns_32ns_32_36_1
    generic map (
        ID => 1,
        NUM_STAGE => 36,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 32)
    port map (
        clk => clk,
        reset => rst,
        ce => nReadyArray(0),
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        dout => dataOutArray(0));

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray

    buff: entity work.delay_buffer(arch) 
    generic map(35)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));

end architecture;

----------------------------------------------------------------------- 
-- frem, remainder of float division, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity frem_op is
Generic (
  INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
 clk, rst : in std_logic; 
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
  pValidArray : in std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of frem_op is

    signal join_valid : STD_LOGIC;

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    dataOutArray(0) <= std_logic_vector(unsigned(dataInArray(0)) + unsigned (dataInArray(1)));
    validArray(0) <= join_valid;

end architecture;

-------------------
--sinf
----------------
Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity sinf_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(0 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(0 downto 0);
      dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of sinf_op is

    -- Interface to Vivado component
    component sin_or_cos_float_s is
    port (
        ap_clk : IN STD_LOGIC;
        ap_rst : IN STD_LOGIC;
        ap_start : IN STD_LOGIC;
        ap_done : OUT STD_LOGIC;
        ap_idle : OUT STD_LOGIC;
        ap_ready : OUT STD_LOGIC;
        t_in : IN STD_LOGIC_VECTOR (31 downto 0);
        ap_return : OUT STD_LOGIC_VECTOR (31 downto 0) );
    end component;
    
    signal idle : std_logic;
    signal component_ready : std_logic;

begin 
    
    sin_or_cos_float_s_U1 : component sin_or_cos_float_s
    port map(
        ap_clk  => clk,
        ap_rst =>  rst,
        ap_start => pValidArray(0),
        ap_done => validArray(0),
        ap_idle => idle,
        ap_ready => component_ready,
        t_in => dataInArray(0),
        ap_return => dataOutArray(0)
    );
    
    readyArray(0) <= idle and nReadyArray(0);
        
end architecture;

-------------------
--cosf
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity cosf_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(0 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(0 downto 0);
      dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of cosf_op is

    -- Interface to Vivado component
    component sin_or_cos_float_s is
    port (
        ap_clk : IN STD_LOGIC;
        ap_rst : IN STD_LOGIC;
        ap_start : IN STD_LOGIC;
        ap_done : OUT STD_LOGIC;
        ap_idle : OUT STD_LOGIC;
        ap_ready : OUT STD_LOGIC;
        t_in : IN STD_LOGIC_VECTOR (31 downto 0);
        ap_return : OUT STD_LOGIC_VECTOR (31 downto 0) );
    end component;
    
    signal idle : std_logic;
    signal component_ready : std_logic;

begin 
    
    sin_or_cos_float_s_U1 : component sin_or_cos_float_s
    port map(
        ap_clk  => clk,
        ap_rst =>  rst,
        ap_start => pValidArray(0),
        ap_done => validArray(0),
        ap_idle => idle,
        ap_ready => component_ready,
        t_in => dataInArray(0),
        ap_return => dataOutArray(0)
    );
    
    readyArray(0) <= idle and nReadyArray(0);
        
end architecture;

-------------------
--sqrtf
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity sqrtf_op is
Generic (
 INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port (
  clk : IN STD_LOGIC;
  rst : IN STD_LOGIC;
  pValidArray : IN std_logic_vector(0 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : OUT std_logic_vector(0 downto 0);
  dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;

architecture arch of sqrtf_op is

    -- Interface to Vivado component
    component array_RAM_fsqrt_32ns_32ns_32_28_1 is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 28;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 32
        );
        port (
            clk   : in  std_logic;
            reset : in  std_logic;
            ce    : in  std_logic;
            din0  : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1  : in  std_logic_vector(din1_WIDTH-1 downto 0);
            dout  : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    constant zeroes : std_logic_vector(31 downto 0) := (others => '0');

begin 

    buff: entity work.delay_buffer(arch) 
        generic map(27)
        port map(clk,
                 rst,
                 pValidArray(0),
                 nReadyArray(0),
                 validArray(0));

    array_RAM_fsqrt_32ns_32ns_32_28_1_U1 :  component array_RAM_fsqrt_32ns_32ns_32_28_1
    port map (
        clk   => clk,
        reset => rst,
        ce    => nReadyArray(0),
        din0  => zeroes,
        din1  => dataInArray(0),
        dout  => dataOutArray(0));

    readyArray(0) <= nReadyArray(0);
    
end architecture;

-------------------
--expf
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity expf_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(0 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(0 downto 0);
      dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of expf_op is

    -- Interface to Vivado component
    component array_RAM_fexp_32ns_32ns_32_18_full_dsp_1 is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 18;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 32
        );
        port (
            clk   : in  std_logic;
            reset : in  std_logic;
            ce    : in  std_logic;
            din0  : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1  : in  std_logic_vector(din1_WIDTH-1 downto 0);
            dout  : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal zeroes : std_logic_vector(31 downto 0) := (others => '0');

begin 
    
    array_RAM_fexp_32ns_32ns_32_18_full_dsp_1_U1 : component array_RAM_fexp_32ns_32ns_32_18_full_dsp_1
    port map(
        clk   => clk,
        reset => rst,
        ce    => nReadyArray(0),
        din0  => zeroes,
        din1  => dataInArray(0),
        dout  => dataOutArray(0)
    );

    buff: entity work.delay_buffer(arch) 
    generic map(17)
    port map(clk,
             rst,
             pValidArray(0),
             nReadyArray(0),
             validArray(0));

    readyArray(0) <= nReadyArray(0);
        
end architecture;

-------------------
--exp2f
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity exp2f_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(0 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(0 downto 0);
      dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of exp2f_op is

    -- Interface to Vivado component
    component exp2_generic_float_s is
        port (
            ap_clk : IN STD_LOGIC;
            ap_rst : IN STD_LOGIC;
            ap_start : IN STD_LOGIC;
            ap_done : OUT STD_LOGIC;
            ap_idle : OUT STD_LOGIC;
            ap_ready : OUT STD_LOGIC;
            x : IN STD_LOGIC_VECTOR (31 downto 0);
            ap_return : OUT STD_LOGIC_VECTOR (31 downto 0) );
    end component;
    
    signal idle : std_logic;
    signal component_ready : std_logic;

begin 
    
    exp2_generic_float_s_U1 : component exp2_generic_float_s
    port map(
        ap_clk  => clk,
        ap_rst =>  rst,
        ap_start => pValidArray(0),
        ap_done => validArray(0),
        ap_idle => idle,
        ap_ready => component_ready,
        x => dataInArray(0),
        ap_return => dataOutArray(0)
    );
    
    readyArray(0) <= idle and nReadyArray(0);
        
end architecture;

-------------------
--logf
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity logf_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(0 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(0 downto 0);
      dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of logf_op is

    -- Interface to Vivado component
    component array_RAM_flog_32ns_32ns_32_19_full_dsp_1 is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 19;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 32
        );
        port (
            clk   : in  std_logic;
            reset : in  std_logic;
            ce    : in  std_logic;
            din0  : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1  : in  std_logic_vector(din1_WIDTH-1 downto 0);
            dout  : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal zeroes : std_logic_vector(31 downto 0) := (others => '0');

begin 
    
    array_RAM_flog_32ns_32ns_32_19_full_dsp_1_U1 : component array_RAM_flog_32ns_32ns_32_19_full_dsp_1
    port map(
        clk   => clk,
        reset => rst,
        ce    => nReadyArray(0),
        din0  => zeroes,
        din1  => dataInArray(0),
        dout  => dataOutArray(0)
    );

    buff: entity work.delay_buffer(arch) 
    generic map(18)
    port map(clk,
             rst,
             pValidArray(0),
             nReadyArray(0),
             validArray(0)
    );

    readyArray(0) <= nReadyArray(0);
        
end architecture;

-------------------
--log2f
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity log2f_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(0 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(0 downto 0);
      dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of log2f_op is

    -- Interface to Vivado component
    component log_generic_float_s is
        port (
            ap_clk : IN STD_LOGIC;
            ap_rst : IN STD_LOGIC;
            ap_start : IN STD_LOGIC;
            ap_done : OUT STD_LOGIC;
            ap_idle : OUT STD_LOGIC;
            ap_ready : OUT STD_LOGIC;
            base_r : IN STD_LOGIC_VECTOR (31 downto 0);
            ap_return : OUT STD_LOGIC_VECTOR (31 downto 0) );
    end component;

    component fmul_op is
        Generic (
         INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
        );
        port (
          clk : IN STD_LOGIC;
          rst : IN STD_LOGIC;
          pValidArray : IN std_logic_vector(1 downto 0);
          nReadyArray : in std_logic_vector(0 downto 0);
          validArray : out std_logic_vector(0 downto 0);
          readyArray : OUT std_logic_vector(1 downto 0);
          dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
          dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
    end component;

    constant constant_factor : std_logic_vector(31 downto 0) := "00111111101110001010101000111011";
    signal idle : std_logic;
    signal component_ready : std_logic;
    signal log_valid : std_logic;
    signal mul_ready : std_logic;
    signal log_dout : std_logic_vector(31 downto 0);
    signal dont_care : std_logic;

begin 
    
    log_generic_float_s_U1 : component log_generic_float_s
    port map(
        ap_clk  => clk,
        ap_rst =>  rst,
        ap_start => pValidArray(0),
        ap_done => log_valid,
        ap_idle => idle,
        ap_ready => component_ready,
        base_r => dataInArray(0),
        ap_return => log_dout
    );
    
    fmul_U1 : component fmul_op
    generic map(1,1,32,32)
    port map(
     clk => clk,
     rst => rst,
     pValidArray(0) => log_valid,
     pValidArray(1) => '1',
     nReadyArray => nReadyArray,
     validArray => validArray,
     readyArray(0) => mul_ready,
     readyArray(1) => dont_care,
     dataInArray(0) => log_dout,
     dataInArray(1) => constant_factor,
     dataOutArray => dataOutArray
    );

    readyArray(0) <= idle and mul_ready;
        
end architecture;

-------------------
--log10f
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity log10f_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(0 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(0 downto 0);
      dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of log10f_op is

    component logf_op is
        Generic (
         INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
        );
        port (
          clk : IN STD_LOGIC;
          rst : IN STD_LOGIC;
          pValidArray : IN std_logic_vector(0 downto 0);
          nReadyArray : in std_logic_vector(0 downto 0);
          validArray : out std_logic_vector(0 downto 0);
          readyArray : OUT std_logic_vector(0 downto 0);
          dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
          dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
    end component;

    component fmul_op is
        Generic (
         INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
        );
        port (
          clk : IN STD_LOGIC;
          rst : IN STD_LOGIC;
          pValidArray : IN std_logic_vector(1 downto 0);
          nReadyArray : in std_logic_vector(0 downto 0);
          validArray : out std_logic_vector(0 downto 0);
          readyArray : OUT std_logic_vector(1 downto 0);
          dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
          dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
    end component;

    constant constant_factor : std_logic_vector(31 downto 0) := "00111110110111100101101111011001";
    signal component_ready : std_logic;
    signal log_valid : std_logic;
    signal mul_ready : std_logic;
    signal log_dout : std_logic_vector(31 downto 0);
    signal dont_care : std_logic;

begin 
    
    logf_u1 : component logf_op
    generic map(1,1,32,32)
    port map(
        clk => clk,
        rst => rst,
        pValidArray => pValidArray,
        nReadyArray(0) => mul_ready, 
        validArray(0) => log_valid,
        readyArray => readyArray,
        dataInArray => dataInArray, 
        dataOutArray(0) => log_dout
    );
    
    fmul_U1 : component fmul_op
    generic map(2,1,32,32)
    port map(
     clk => clk,
     rst => rst,
     pValidArray(0) => log_valid,
     pValidArray(1) => '1',
     nReadyArray => nReadyArray,
     validArray => validArray,
     readyArray(0) => mul_ready,
     readyArray(1) => dont_care,
     dataInArray(0) => log_dout,
     dataInArray(1) => constant_factor,
     dataOutArray => dataOutArray
    );
        
end architecture;

-------------------
--fabsf
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fabsf_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(0 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(0 downto 0);
      dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of fabsf_op is

begin 
    readyArray <= nReadyArray;
    validArray <= pValidArray;
    dataOutArray(0) <= '0' & dataInArray(0)(DATA_SIZE_IN - 2 downto 0);
        
end architecture;

-------------------
--trunc_op
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity trunc_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(0 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(0 downto 0);
      dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of trunc_op is
begin
    
    dataOutArray(0)<= dataInArray(0)(DATA_SIZE_OUT - 1 downto 0);
    validArray <= pValidArray;
    readyArray(0) <= nReadyArray(0);

end architecture;

-------------------
--floorf_op
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity floorf_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(0 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(0 downto 0);
      dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of floorf_op is

    component my_floorf is
        port (
            ap_clk : IN STD_LOGIC;
            ap_rst : IN STD_LOGIC;
            ap_start : IN STD_LOGIC;
            ap_done : OUT STD_LOGIC;
            ap_idle : OUT STD_LOGIC;
            ap_ready : OUT STD_LOGIC;
            a : IN STD_LOGIC_VECTOR (31 downto 0);
            ap_return : OUT STD_LOGIC_VECTOR (31 downto 0) 
        );
    end component;

    signal idle : std_logic;
    signal component_ready : std_logic;

begin 

    my_floorf_U1 : component my_floorf
    port map(
        ap_clk => clk,
        ap_rst => rst,
        ap_start => pValidArray(0),
        ap_done => validArray(0),
        ap_idle => idle,
        ap_ready => component_ready,
        a => dataInArray(0),
        ap_return => dataOutArray(0) 
    );

    readyArray(0) <= idle and nReadyArray(0);
        
end architecture;

-------------------
--ceilf_op
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity ceilf_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(0 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(0 downto 0);
      dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of ceilf_op is

    component my_ceilf is
        port (
            ap_clk : IN STD_LOGIC;
            ap_rst : IN STD_LOGIC;
            ap_start : IN STD_LOGIC;
            ap_done : OUT STD_LOGIC;
            ap_idle : OUT STD_LOGIC;
            ap_ready : OUT STD_LOGIC;
            a : IN STD_LOGIC_VECTOR (31 downto 0);
            ap_return : OUT STD_LOGIC_VECTOR (31 downto 0) 
        );
    end component;

    signal idle : std_logic;
    signal component_ready : std_logic;

begin 

    my_ceilf_U1 : component my_ceilf
    port map(
        ap_clk => clk,
        ap_rst => rst,
        ap_start => pValidArray(0),
        ap_done => validArray(0),
        ap_idle => idle,
        ap_ready => component_ready,
        a => dataInArray(0),
        ap_return => dataOutArray(0) 
    );

    readyArray(0) <= idle and nReadyArray(0);
        
end architecture;

-------------------
--roundf_op
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity roundf_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(0 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(0 downto 0);
      dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of roundf_op is

    component my_roundf is
        port (
            ap_clk : IN STD_LOGIC;
            ap_rst : IN STD_LOGIC;
            ap_start : IN STD_LOGIC;
            ap_done : OUT STD_LOGIC;
            ap_idle : OUT STD_LOGIC;
            ap_ready : OUT STD_LOGIC;
            a : IN STD_LOGIC_VECTOR (31 downto 0);
            ap_return : OUT STD_LOGIC_VECTOR (31 downto 0) 
        );
    end component;

    signal idle : std_logic;
    signal component_ready : std_logic;

begin 

    my_roundf_U1 : component my_roundf
    port map(
        ap_clk => clk,
        ap_rst => rst,
        ap_start => pValidArray(0),
        ap_done => validArray(0),
        ap_idle => idle,
        ap_ready => component_ready,
        a => dataInArray(0),
        ap_return => dataOutArray(0) 
    );

    readyArray(0) <= idle and nReadyArray(0);
        
end architecture;

-------------------
--fminf_op
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fminf_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(1 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(1 downto 0);
      dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of fminf_op is

    component my_fminf is
        port (
            ap_clk : IN STD_LOGIC;
            ap_rst : IN STD_LOGIC;
            a : IN STD_LOGIC_VECTOR (31 downto 0);
            b : IN STD_LOGIC_VECTOR (31 downto 0);
            ap_return : OUT STD_LOGIC_VECTOR (31 downto 0) );
        end component;

    signal join_valid : std_logic;

begin 

    my_trunc_U1 : component my_fminf
    port map(
        ap_clk => clk,
        ap_rst => rst,
        a => dataInArray(0),
        b => dataInArray(1),
        ap_return => dataOutArray(0)
    );

    join_write_temp:   entity work.join(arch) generic map(2)
    port map( pValidArray,  --pValidArray
              nReadyArray(0),     --nready                    
              join_valid,         --valid          
              readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
    generic map(1)
    port map(clk,
         rst,
         join_valid,
         nReadyArray(0),
         validArray(0));
        
end architecture;

-------------------
--fmaxf_op
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fmaxf_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(1 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(1 downto 0);
      dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of fmaxf_op is

    component my_fmaxf is
        port (
            ap_clk : IN STD_LOGIC;
            ap_rst : IN STD_LOGIC;
            a : IN STD_LOGIC_VECTOR (31 downto 0);
            b : IN STD_LOGIC_VECTOR (31 downto 0);
            ap_return : OUT STD_LOGIC_VECTOR (31 downto 0) );
        end component;

    signal join_valid : std_logic;

begin 

    my_trunc_U1 : component my_fmaxf
    port map(
        ap_clk => clk,
        ap_rst => rst,
        a => dataInArray(0),
        b => dataInArray(1),
        ap_return => dataOutArray(0)
    );

    join_write_temp:   entity work.join(arch) generic map(2)
    port map( pValidArray,  --pValidArray
              nReadyArray(0),     --nready                    
              join_valid,         --valid          
              readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
    generic map(1)
    port map(clk,
         rst,
         join_valid,
         nReadyArray(0),
         validArray(0));
        
end architecture;

-------------------
--powf_op
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity powf_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(1 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(1 downto 0);
      dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of powf_op is

    -- Interface to Vivado component
    component pow_generic_float_s is
        port (
            ap_clk : IN STD_LOGIC;
            ap_rst : IN STD_LOGIC;
            ap_start : IN STD_LOGIC;
            ap_done : OUT STD_LOGIC;
            ap_idle : OUT STD_LOGIC;
            ap_ready : OUT STD_LOGIC;
            base_r : IN STD_LOGIC_VECTOR (31 downto 0);
            exp : IN STD_LOGIC_VECTOR (31 downto 0);
            ap_return : OUT STD_LOGIC_VECTOR (31 downto 0) );
    end component;
    
    signal idle : std_logic;
    signal component_ready : std_logic;
    signal join_valid : std_logic;
    signal ready_intermediary : std_logic;

begin 
    
    pow_generic_float_s_U1 : component pow_generic_float_s
    port map(
        ap_clk => clk,
        ap_rst => rst,
        ap_start => join_valid,
        ap_done => validArray(0),
        ap_idle => idle,
        ap_ready => component_ready,
        base_r => dataInArray(0),
        exp => dataInArray(1),
        ap_return => dataOutArray(0)
    );

    join_write_temp:   entity work.join(arch) generic map(2)
    port map( pValidArray,  --pValidArray
              ready_intermediary,     --nready                    
              join_valid,         --valid          
              readyArray);   --readyarray 
    
    ready_intermediary <= idle and nReadyArray(0);
        
end architecture;


-------------------
--fabsf_op
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fabsf_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(0 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(0 downto 0);
      dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of fabsf_op is

begin 

    readyArray <= nReadyArray;
    validArray <= pValidArray;
    dataOutArray(0) <= '0' & dataInArray(0)(DATA_SIZE_IN - 2 downto 0);
        
end architecture;

-------------------
--copysignf_op
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity copysignf_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(1 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(1 downto 0);
      dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of copysignf_op is
    signal join_valid : std_logic;
begin
    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= dataInArray(1)(DATA_SIZE_IN - 1) & dataInArray(0)(DATA_SIZE_IN - 2 downto 0);
    validArray(0) <= join_valid;
        
end architecture;


Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity sitofp_op is
    Generic (
     INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(0 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(0 downto 0);
      dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;
    
architecture arch of sitofp_op is

    component array_RAM_sitofp_32ns_32_6_1 IS
    generic (
        ID : INTEGER;
        NUM_STAGE : INTEGER;
        din0_WIDTH : INTEGER;
        dout_WIDTH : INTEGER );
    port (
        clk : IN STD_LOGIC;
        reset : IN STD_LOGIC;
        din0 : IN STD_LOGIC_VECTOR (31 downto 0);
        ce : IN STD_LOGIC;
        dout : OUT STD_LOGIC_VECTOR (31 downto 0) );
    end component;

begin 

    buff: entity work.delay_buffer(arch) 
    generic map(5)
    port map(clk,
             rst,
             pValidArray(0),
             nReadyArray(0),
             validArray(0));

    array_RAM_sitofp_32ns_32_6_1_U1 :  component array_RAM_sitofp_32ns_32_6_1
    generic map (
        ID => 1,
        NUM_STAGE => 6,
        din0_WIDTH => 32,
        dout_WIDTH => 32)
    port map (
        clk   => clk,
        reset => rst,
        din0  => dataInArray(0),
        ce    => nReadyArray(0),
        dout  => dataOutArray(0));

    readyArray(0) <= nReadyArray(0);
        
end architecture;
