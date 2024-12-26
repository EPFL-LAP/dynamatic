--------------------------------------------------------------------------------
--                RightShifterSticky24_by_max_26_Freq450_uid4
-- VHDL generated for Kintex7 @ 450MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca (2008-2011), Florent de Dinechin (2008-2019)
--------------------------------------------------------------------------------
-- Pipeline depth: 3 cycles
-- Clock period (ns): 2.22222
-- Target frequency (MHz): 450
-- Input signals: X S
-- Output signals: R Sticky

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity RightShifterSticky24_by_max_26_Freq450_uid4 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(23 downto 0);
          S : in  std_logic_vector(4 downto 0);
          R : out  std_logic_vector(25 downto 0);
          Sticky : out  std_logic   );
end entity;

architecture arch of RightShifterSticky24_by_max_26_Freq450_uid4 is
signal ps, ps_d1, ps_d2, ps_d3 :  std_logic_vector(4 downto 0);
signal Xpadded :  std_logic_vector(25 downto 0);
signal level5 :  std_logic_vector(25 downto 0);
signal stk4, stk4_d1 :  std_logic;
signal level4, level4_d1 :  std_logic_vector(25 downto 0);
signal stk3, stk3_d1 :  std_logic;
signal level3, level3_d1, level3_d2 :  std_logic_vector(25 downto 0);
signal stk2 :  std_logic;
signal level2, level2_d1, level2_d2 :  std_logic_vector(25 downto 0);
signal stk1, stk1_d1 :  std_logic;
signal level1, level1_d1, level1_d2, level1_d3 :  std_logic_vector(25 downto 0);
signal stk0 :  std_logic;
signal level0 :  std_logic_vector(25 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               ps_d1 <=  ps;
               ps_d2 <=  ps_d1;
               ps_d3 <=  ps_d2;
               stk4_d1 <=  stk4;
               level4_d1 <=  level4;
               stk3_d1 <=  stk3;
               level3_d1 <=  level3;
               level3_d2 <=  level3_d1;
               level2_d1 <=  level2;
               level2_d2 <=  level2_d1;
               stk1_d1 <=  stk1;
               level1_d1 <=  level1;
               level1_d2 <=  level1_d1;
               level1_d3 <=  level1_d2;
            end if;
         end if;
      end process;
   ps<= S;
   Xpadded <= X&(1 downto 0 => '0');
   level5<= Xpadded;
   stk4 <= '1' when (level5(15 downto 0)/="0000000000000000" and ps(4)='1')   else '0';
   level4 <=  level5 when  ps(4)='0'    else (15 downto 0 => '0') & level5(25 downto 16);
   stk3 <= '1' when (level4_d1(7 downto 0)/="00000000" and ps_d1(3)='1') or stk4_d1 ='1'   else '0';
   level3 <=  level4 when  ps(3)='0'    else (7 downto 0 => '0') & level4(25 downto 8);
   stk2 <= '1' when (level3_d2(3 downto 0)/="0000" and ps_d2(2)='1') or stk3_d1 ='1'   else '0';
   level2 <=  level3 when  ps(2)='0'    else (3 downto 0 => '0') & level3(25 downto 4);
   stk1 <= '1' when (level2_d2(1 downto 0)/="00" and ps_d2(1)='1') or stk2 ='1'   else '0';
   level1 <=  level2 when  ps(1)='0'    else (1 downto 0 => '0') & level2(25 downto 2);
   stk0 <= '1' when (level1_d3(0 downto 0)/="0" and ps_d3(0)='1') or stk1_d1 ='1'   else '0';
   level0 <=  level1 when  ps(0)='0'    else (0 downto 0 => '0') & level1(25 downto 1);
   R <= level0;
   Sticky <= stk0;
end architecture;

--------------------------------------------------------------------------------
--                          IntAdder_27_Freq450_uid6
-- VHDL generated for Kintex7 @ 450MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2016)
--------------------------------------------------------------------------------
-- Pipeline depth: 1 cycles
-- Clock period (ns): 2.22222
-- Target frequency (MHz): 450
-- Input signals: X Y Cin
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntAdder_27_Freq450_uid6 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(26 downto 0);
          Y : in  std_logic_vector(26 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(26 downto 0)   );
end entity;

architecture arch of IntAdder_27_Freq450_uid6 is
signal Cin_0, Cin_0_d1 :  std_logic;
signal X_0, X_0_d1, X_0_d2, X_0_d3, X_0_d4, X_0_d5 :  std_logic_vector(11 downto 0);
signal Y_0, Y_0_d1, Y_0_d2, Y_0_d3 :  std_logic_vector(11 downto 0);
signal S_0 :  std_logic_vector(11 downto 0);
signal R_0 :  std_logic_vector(10 downto 0);
signal Cin_1 :  std_logic;
signal X_1, X_1_d1, X_1_d2, X_1_d3, X_1_d4, X_1_d5 :  std_logic_vector(16 downto 0);
signal Y_1, Y_1_d1, Y_1_d2, Y_1_d3 :  std_logic_vector(16 downto 0);
signal S_1 :  std_logic_vector(16 downto 0);
signal R_1 :  std_logic_vector(15 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               Cin_0_d1 <=  Cin_0;
               X_0_d1 <=  X_0;
               X_0_d2 <=  X_0_d1;
               X_0_d3 <=  X_0_d2;
               X_0_d4 <=  X_0_d3;
               X_0_d5 <=  X_0_d4;
               Y_0_d1 <=  Y_0;
               Y_0_d2 <=  Y_0_d1;
               Y_0_d3 <=  Y_0_d2;
               X_1_d1 <=  X_1;
               X_1_d2 <=  X_1_d1;
               X_1_d3 <=  X_1_d2;
               X_1_d4 <=  X_1_d3;
               X_1_d5 <=  X_1_d4;
               Y_1_d1 <=  Y_1;
               Y_1_d2 <=  Y_1_d1;
               Y_1_d3 <=  Y_1_d2;
            end if;
         end if;
      end process;
   Cin_0 <= Cin;
   X_0 <= '0' & X(10 downto 0);
   Y_0 <= '0' & Y(10 downto 0);
   S_0 <= X_0_d5 + Y_0_d3 + Cin_0_d1;
   R_0 <= S_0(10 downto 0);
   Cin_1 <= S_0(11);
   X_1 <= '0' & X(26 downto 11);
   Y_1 <= '0' & Y(26 downto 11);
   S_1 <= X_1_d5 + Y_1_d3 + Cin_1;
   R_1 <= S_1(15 downto 0);
   R <= R_1 & R_0 ;
end architecture;

--------------------------------------------------------------------------------
--                     Normalizer_Z_28_28_28_Freq450_uid8
-- VHDL generated for Kintex7 @ 450MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin, (2007-2020)
--------------------------------------------------------------------------------
-- Pipeline depth: 3 cycles
-- Clock period (ns): 2.22222
-- Target frequency (MHz): 450
-- Input signals: X
-- Output signals: Count R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity Normalizer_Z_28_28_28_Freq450_uid8 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(27 downto 0);
          Count : out  std_logic_vector(4 downto 0);
          R : out  std_logic_vector(27 downto 0)   );
end entity;

architecture arch of Normalizer_Z_28_28_28_Freq450_uid8 is
signal level5, level5_d1 :  std_logic_vector(27 downto 0);
signal count4, count4_d1, count4_d2, count4_d3 :  std_logic;
signal level4 :  std_logic_vector(27 downto 0);
signal count3, count3_d1, count3_d2 :  std_logic;
signal level3, level3_d1 :  std_logic_vector(27 downto 0);
signal count2, count2_d1 :  std_logic;
signal level2 :  std_logic_vector(27 downto 0);
signal count1, count1_d1 :  std_logic;
signal level1, level1_d1 :  std_logic_vector(27 downto 0);
signal count0 :  std_logic;
signal level0 :  std_logic_vector(27 downto 0);
signal sCount :  std_logic_vector(4 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               level5_d1 <=  level5;
               count4_d1 <=  count4;
               count4_d2 <=  count4_d1;
               count4_d3 <=  count4_d2;
               count3_d1 <=  count3;
               count3_d2 <=  count3_d1;
               level3_d1 <=  level3;
               count2_d1 <=  count2;
               count1_d1 <=  count1;
               level1_d1 <=  level1;
            end if;
         end if;
      end process;
   level5 <= X ;
   count4<= '1' when level5(27 downto 12) = (27 downto 12=>'0') else '0';
   level4<= level5_d1(27 downto 0) when count4_d1='0' else level5_d1(11 downto 0) & (15 downto 0 => '0');

   count3<= '1' when level4(27 downto 20) = (27 downto 20=>'0') else '0';
   level3<= level4(27 downto 0) when count3='0' else level4(19 downto 0) & (7 downto 0 => '0');

   count2<= '1' when level3_d1(27 downto 24) = (27 downto 24=>'0') else '0';
   level2<= level3_d1(27 downto 0) when count2='0' else level3_d1(23 downto 0) & (3 downto 0 => '0');

   count1<= '1' when level2(27 downto 26) = (27 downto 26=>'0') else '0';
   level1<= level2(27 downto 0) when count1='0' else level2(25 downto 0) & (1 downto 0 => '0');

   count0<= '1' when level1_d1(27 downto 27) = (27 downto 27=>'0') else '0';
   level0<= level1_d1(27 downto 0) when count0='0' else level1_d1(26 downto 0) & (0 downto 0 => '0');

   R <= level0;
   sCount <= count4_d3 & count3_d2 & count2_d1 & count1_d1 & count0;
   Count <= sCount;
end architecture;

--------------------------------------------------------------------------------
--                         IntAdder_34_Freq450_uid11
-- VHDL generated for Kintex7 @ 450MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2016)
--------------------------------------------------------------------------------
-- Pipeline depth: 1 cycles
-- Clock period (ns): 2.22222
-- Target frequency (MHz): 450
-- Input signals: X Y Cin
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntAdder_34_Freq450_uid11 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(33 downto 0);
          Y : in  std_logic_vector(33 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(33 downto 0)   );
end entity;

architecture arch of IntAdder_34_Freq450_uid11 is
signal Cin_1, Cin_1_d1 :  std_logic;
signal X_1, X_1_d1 :  std_logic_vector(34 downto 0);
signal Y_1, Y_1_d1, Y_1_d2, Y_1_d3, Y_1_d4, Y_1_d5, Y_1_d6, Y_1_d7, Y_1_d8, Y_1_d9 :  std_logic_vector(34 downto 0);
signal S_1 :  std_logic_vector(34 downto 0);
signal R_1 :  std_logic_vector(33 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               Cin_1_d1 <=  Cin_1;
               X_1_d1 <=  X_1;
               Y_1_d1 <=  Y_1;
               Y_1_d2 <=  Y_1_d1;
               Y_1_d3 <=  Y_1_d2;
               Y_1_d4 <=  Y_1_d3;
               Y_1_d5 <=  Y_1_d4;
               Y_1_d6 <=  Y_1_d5;
               Y_1_d7 <=  Y_1_d6;
               Y_1_d8 <=  Y_1_d7;
               Y_1_d9 <=  Y_1_d8;
            end if;
         end if;
      end process;
   Cin_1 <= Cin;
   X_1 <= '0' & X(33 downto 0);
   Y_1 <= '0' & Y(33 downto 0);
   S_1 <= X_1_d1 + Y_1_d9 + Cin_1_d1;
   R_1 <= S_1(33 downto 0);
   R <= R_1 ;
end architecture;

--------------------------------------------------------------------------------
--                             FloatingPointAdder
--                         (FPAdd_8_23_Freq450_uid2)
-- VHDL generated for Kintex7 @ 450MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin, Bogdan Pasca (2010-2017)
--------------------------------------------------------------------------------
-- Pipeline depth: 9 cycles
-- Clock period (ns): 2.22222
-- Target frequency (MHz): 450
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity FloatingPointAdder is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(8+23+2 downto 0);
          Y : in  std_logic_vector(8+23+2 downto 0);
          R : out  std_logic_vector(8+23+2 downto 0)   );
end entity;

architecture arch of FloatingPointAdder is
   component RightShifterSticky24_by_max_26_Freq450_uid4 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(23 downto 0);
             S : in  std_logic_vector(4 downto 0);
             R : out  std_logic_vector(25 downto 0);
             Sticky : out  std_logic   );
   end component;

   component IntAdder_27_Freq450_uid6 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(26 downto 0);
             Y : in  std_logic_vector(26 downto 0);
             Cin : in  std_logic;
             R : out  std_logic_vector(26 downto 0)   );
   end component;

   component Normalizer_Z_28_28_28_Freq450_uid8 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(27 downto 0);
             Count : out  std_logic_vector(4 downto 0);
             R : out  std_logic_vector(27 downto 0)   );
   end component;

   component IntAdder_34_Freq450_uid11 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(33 downto 0);
             Y : in  std_logic_vector(33 downto 0);
             Cin : in  std_logic;
             R : out  std_logic_vector(33 downto 0)   );
   end component;

signal excExpFracX :  std_logic_vector(32 downto 0);
signal excExpFracY :  std_logic_vector(32 downto 0);
signal swap :  std_logic;
signal eXmeY :  std_logic_vector(7 downto 0);
signal eYmeX :  std_logic_vector(7 downto 0);
signal expDiff, expDiff_d1 :  std_logic_vector(7 downto 0);
signal newX :  std_logic_vector(33 downto 0);
signal newY, newY_d1 :  std_logic_vector(33 downto 0);
signal expX, expX_d1 :  std_logic_vector(7 downto 0);
signal excX :  std_logic_vector(1 downto 0);
signal excY, excY_d1 :  std_logic_vector(1 downto 0);
signal signX, signX_d1 :  std_logic;
signal signY, signY_d1 :  std_logic;
signal EffSub, EffSub_d1, EffSub_d2, EffSub_d3, EffSub_d4, EffSub_d5, EffSub_d6, EffSub_d7, EffSub_d8 :  std_logic;
signal sXsYExnXY, sXsYExnXY_d1 :  std_logic_vector(5 downto 0);
signal sdExnXY :  std_logic_vector(3 downto 0);
signal fracY :  std_logic_vector(23 downto 0);
signal excRt, excRt_d1, excRt_d2, excRt_d3, excRt_d4, excRt_d5, excRt_d6, excRt_d7, excRt_d8 :  std_logic_vector(1 downto 0);
signal signR, signR_d1, signR_d2, signR_d3, signR_d4, signR_d5, signR_d6, signR_d7 :  std_logic;
signal shiftedOut :  std_logic;
signal shiftVal :  std_logic_vector(4 downto 0);
signal shiftedFracY :  std_logic_vector(25 downto 0);
signal sticky, sticky_d1 :  std_logic;
signal fracYpad, fracYpad_d1 :  std_logic_vector(26 downto 0);
signal EffSubVector, EffSubVector_d1 :  std_logic_vector(26 downto 0);
signal fracYpadXorOp :  std_logic_vector(26 downto 0);
signal fracXpad :  std_logic_vector(26 downto 0);
signal cInSigAdd :  std_logic;
signal fracAddResult :  std_logic_vector(26 downto 0);
signal fracSticky :  std_logic_vector(27 downto 0);
signal nZerosNew :  std_logic_vector(4 downto 0);
signal shiftedFrac :  std_logic_vector(27 downto 0);
signal extendedExpInc, extendedExpInc_d1, extendedExpInc_d2, extendedExpInc_d3, extendedExpInc_d4, extendedExpInc_d5, extendedExpInc_d6, extendedExpInc_d7 :  std_logic_vector(8 downto 0);
signal updatedExp :  std_logic_vector(9 downto 0);
signal eqdiffsign, eqdiffsign_d1 :  std_logic;
signal expFrac :  std_logic_vector(33 downto 0);
signal stk :  std_logic;
signal rnd :  std_logic;
signal lsb :  std_logic;
signal needToRound :  std_logic;
signal RoundedExpFrac :  std_logic_vector(33 downto 0);
signal upExc :  std_logic_vector(1 downto 0);
signal fracR :  std_logic_vector(22 downto 0);
signal expR :  std_logic_vector(7 downto 0);
signal exExpExc :  std_logic_vector(3 downto 0);
signal excRt2 :  std_logic_vector(1 downto 0);
signal excR :  std_logic_vector(1 downto 0);
signal signR2, signR2_d1 :  std_logic;
signal computedR :  std_logic_vector(33 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               expDiff_d1 <=  expDiff;
               newY_d1 <=  newY;
               expX_d1 <=  expX;
               excY_d1 <=  excY;
               signX_d1 <=  signX;
               signY_d1 <=  signY;
               EffSub_d1 <=  EffSub;
               EffSub_d2 <=  EffSub_d1;
               EffSub_d3 <=  EffSub_d2;
               EffSub_d4 <=  EffSub_d3;
               EffSub_d5 <=  EffSub_d4;
               EffSub_d6 <=  EffSub_d5;
               EffSub_d7 <=  EffSub_d6;
               EffSub_d8 <=  EffSub_d7;
               sXsYExnXY_d1 <=  sXsYExnXY;
               excRt_d1 <=  excRt;
               excRt_d2 <=  excRt_d1;
               excRt_d3 <=  excRt_d2;
               excRt_d4 <=  excRt_d3;
               excRt_d5 <=  excRt_d4;
               excRt_d6 <=  excRt_d5;
               excRt_d7 <=  excRt_d6;
               excRt_d8 <=  excRt_d7;
               signR_d1 <=  signR;
               signR_d2 <=  signR_d1;
               signR_d3 <=  signR_d2;
               signR_d4 <=  signR_d3;
               signR_d5 <=  signR_d4;
               signR_d6 <=  signR_d5;
               signR_d7 <=  signR_d6;
               sticky_d1 <=  sticky;
               fracYpad_d1 <=  fracYpad;
               EffSubVector_d1 <=  EffSubVector;
               extendedExpInc_d1 <=  extendedExpInc;
               extendedExpInc_d2 <=  extendedExpInc_d1;
               extendedExpInc_d3 <=  extendedExpInc_d2;
               extendedExpInc_d4 <=  extendedExpInc_d3;
               extendedExpInc_d5 <=  extendedExpInc_d4;
               extendedExpInc_d6 <=  extendedExpInc_d5;
               extendedExpInc_d7 <=  extendedExpInc_d6;
               eqdiffsign_d1 <=  eqdiffsign;
               signR2_d1 <=  signR2;
            end if;
         end if;
      end process;
   excExpFracX <= X(33 downto 32) & X(30 downto 0);
   excExpFracY <= Y(33 downto 32) & Y(30 downto 0);
   swap <= '1' when excExpFracX < excExpFracY else '0';
   -- exponent difference
   eXmeY <= (X(30 downto 23)) - (Y(30 downto 23));
   eYmeX <= (Y(30 downto 23)) - (X(30 downto 23));
   expDiff <= eXmeY when swap = '0' else eYmeX;
   -- input swap so that |X|>|Y|
   newX <= X when swap = '0' else Y;
   newY <= Y when swap = '0' else X;
   -- now we decompose the inputs into their sign, exponent, fraction
   expX<= newX(30 downto 23);
   excX<= newX(33 downto 32);
   excY<= newY(33 downto 32);
   signX<= newX(31);
   signY<= newY(31);
   EffSub <= signX_d1 xor signY_d1;
   sXsYExnXY <= signX & signY & excX & excY;
   sdExnXY <= excX & excY;
   fracY <= "000000000000000000000000" when excY_d1="00" else ('1' & newY_d1(22 downto 0));
   -- Exception management logic
   with sXsYExnXY_d1  select  
   excRt <= "00" when "000000"|"010000"|"100000"|"110000",
      "01" when "000101"|"010101"|"100101"|"110101"|"000100"|"010100"|"100100"|"110100"|"000001"|"010001"|"100001"|"110001",
      "10" when "111010"|"001010"|"001000"|"011000"|"101000"|"111000"|"000010"|"010010"|"100010"|"110010"|"001001"|"011001"|"101001"|"111001"|"000110"|"010110"|"100110"|"110110", 
      "11" when others;
   signR<= '0' when (sXsYExnXY_d1="100000" or sXsYExnXY_d1="010000") else signX_d1;
   shiftedOut <= '1' when (expDiff_d1 > 25) else '0';
   shiftVal <= expDiff_d1(4 downto 0) when shiftedOut='0' else CONV_STD_LOGIC_VECTOR(26,5);
   RightShifterComponent: RightShifterSticky24_by_max_26_Freq450_uid4
      port map ( clk  => clk,
                 ce => ce,
                 S => shiftVal,
                 X => fracY,
                 R => shiftedFracY,
                 Sticky => sticky);
   fracYpad <= "0" & shiftedFracY;
   EffSubVector <= (26 downto 0 => EffSub);
   fracYpadXorOp <= fracYpad_d1 xor EffSubVector_d1;
   fracXpad <= "01" & (newX(22 downto 0)) & "00";
   cInSigAdd <= EffSub_d3 and not sticky; -- if we subtract and the sticky was one, some of the negated sticky bits would have absorbed this carry 
   fracAdder: IntAdder_27_Freq450_uid6
      port map ( clk  => clk,
                 ce => ce,
                 Cin => cInSigAdd,
                 X => fracXpad,
                 Y => fracYpadXorOp,
                 R => fracAddResult);
   fracSticky<= fracAddResult & sticky_d1; 
   LZCAndShifter: Normalizer_Z_28_28_28_Freq450_uid8
      port map ( clk  => clk,
                 ce => ce,
                 X => fracSticky,
                 Count => nZerosNew,
                 R => shiftedFrac);
   extendedExpInc<= ("0" & expX_d1) + '1';
   updatedExp <= ("0" &extendedExpInc_d7) - ("00000" & nZerosNew);
   eqdiffsign <= '1' when nZerosNew="11111" else '0';
   expFrac<= updatedExp & shiftedFrac(26 downto 3);
   stk<= shiftedFrac(2) or shiftedFrac(1) or shiftedFrac(0);
   rnd<= shiftedFrac(3);
   lsb<= shiftedFrac(4);
   needToRound<= '1' when (rnd='1' and stk='1') or (rnd='1' and stk='0' and lsb='1')
  else '0';
   roundingAdder: IntAdder_34_Freq450_uid11
      port map ( clk  => clk,
                 ce => ce,
                 Cin => needToRound,
                 X => expFrac,
                 Y => "0000000000000000000000000000000000",
                 R => RoundedExpFrac);
   -- possible update to exception bits
   upExc <= RoundedExpFrac(33 downto 32);
   fracR <= RoundedExpFrac(23 downto 1);
   expR <= RoundedExpFrac(31 downto 24);
   exExpExc <= upExc & excRt_d8;
   with exExpExc  select  
   excRt2<= "00" when "0000"|"0100"|"1000"|"1100"|"1001"|"1101",
      "01" when "0001",
      "10" when "0010"|"0110"|"1010"|"1110"|"0101",
      "11" when others;
   excR <= "00" when (eqdiffsign_d1='1' and EffSub_d8='1'  and not(excRt_d8="11")) else excRt2;
   signR2 <= '0' when (eqdiffsign='1' and EffSub_d7='1') else signR_d7;
   computedR <= excR & signR2_d1 & expR & fracR;
   R <= computedR;
end architecture;



--------------------------------------------------------------------------------
--                        DSPBlock_17x24_Freq711_uid9
-- VHDL generated for Kintex7 @ 711MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 1.40647
-- Target frequency (MHz): 711
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity DSPBlock_17x24_Freq711_uid9 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(16 downto 0);
          Y : in  std_logic_vector(23 downto 0);
          R : out  std_logic_vector(40 downto 0)   );
end entity;

architecture arch of DSPBlock_17x24_Freq711_uid9 is
signal Mint :  std_logic_vector(40 downto 0);
signal M :  std_logic_vector(40 downto 0);
signal Rtmp :  std_logic_vector(40 downto 0);
begin
   Mint <= std_logic_vector(unsigned(X) * unsigned(Y)); -- multiplier
   M <= Mint(40 downto 0);
   Rtmp <= M;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                        DSPBlock_7x24_Freq711_uid11
-- VHDL generated for Kintex7 @ 711MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 1.40647
-- Target frequency (MHz): 711
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity DSPBlock_7x24_Freq711_uid11 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(6 downto 0);
          Y : in  std_logic_vector(23 downto 0);
          R : out  std_logic_vector(30 downto 0)   );
end entity;

architecture arch of DSPBlock_7x24_Freq711_uid11 is
signal Mint :  std_logic_vector(30 downto 0);
signal M :  std_logic_vector(30 downto 0);
signal Rtmp :  std_logic_vector(30 downto 0);
begin
   Mint <= std_logic_vector(unsigned(X) * unsigned(Y)); -- multiplier
   M <= Mint(30 downto 0);
   Rtmp <= M;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                         IntAdder_32_Freq711_uid14
-- VHDL generated for Kintex7 @ 711MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2016)
--------------------------------------------------------------------------------
-- Pipeline depth: 1 cycles
-- Clock period (ns): 1.40647
-- Target frequency (MHz): 711
-- Input signals: X Y Cin
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntAdder_32_Freq711_uid14 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(31 downto 0);
          Y : in  std_logic_vector(31 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(31 downto 0)   );
end entity;

architecture arch of IntAdder_32_Freq711_uid14 is
signal Cin_0, Cin_0_d1 :  std_logic;
signal X_0, X_0_d1 :  std_logic_vector(19 downto 0);
signal Y_0, Y_0_d1 :  std_logic_vector(19 downto 0);
signal S_0 :  std_logic_vector(19 downto 0);
signal R_0 :  std_logic_vector(18 downto 0);
signal Cin_1 :  std_logic;
signal X_1, X_1_d1 :  std_logic_vector(13 downto 0);
signal Y_1, Y_1_d1 :  std_logic_vector(13 downto 0);
signal S_1 :  std_logic_vector(13 downto 0);
signal R_1 :  std_logic_vector(12 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               Cin_0_d1 <=  Cin_0;
               X_0_d1 <=  X_0;
               Y_0_d1 <=  Y_0;
               X_1_d1 <=  X_1;
               Y_1_d1 <=  Y_1;
            end if;
         end if;
      end process;
   Cin_0 <= Cin;
   X_0 <= '0' & X(18 downto 0);
   Y_0 <= '0' & Y(18 downto 0);
   S_0 <= X_0_d1 + Y_0_d1 + Cin_0_d1;
   R_0 <= S_0(18 downto 0);
   Cin_1 <= S_0(19);
   X_1 <= '0' & X(31 downto 19);
   Y_1 <= '0' & Y(31 downto 19);
   S_1 <= X_1_d1 + Y_1_d1 + Cin_1;
   R_1 <= S_1(12 downto 0);
   R <= R_1 & R_0 ;
end architecture;

--------------------------------------------------------------------------------
--                         IntMultiplier_Freq711_uid5
-- VHDL generated for Kintex7 @ 711MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Martin Kumm, Florent de Dinechin, Kinga Illyes, Bogdan Popa, Bogdan Pasca, 2012
--------------------------------------------------------------------------------
-- Pipeline depth: 1 cycles
-- Clock period (ns): 1.40647
-- Target frequency (MHz): 711
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity IntMultiplier_Freq711_uid5 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(23 downto 0);
          Y : in  std_logic_vector(23 downto 0);
          R : out  std_logic_vector(47 downto 0)   );
end entity;

architecture arch of IntMultiplier_Freq711_uid5 is
   component DSPBlock_17x24_Freq711_uid9 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(16 downto 0);
             Y : in  std_logic_vector(23 downto 0);
             R : out  std_logic_vector(40 downto 0)   );
   end component;

   component DSPBlock_7x24_Freq711_uid11 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(6 downto 0);
             Y : in  std_logic_vector(23 downto 0);
             R : out  std_logic_vector(30 downto 0)   );
   end component;

   component IntAdder_32_Freq711_uid14 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(31 downto 0);
             Y : in  std_logic_vector(31 downto 0);
             Cin : in  std_logic;
             R : out  std_logic_vector(31 downto 0)   );
   end component;

signal XX_m6 :  std_logic_vector(23 downto 0);
signal YY_m6 :  std_logic_vector(23 downto 0);
signal tile_0_X :  std_logic_vector(16 downto 0);
signal tile_0_Y :  std_logic_vector(23 downto 0);
signal tile_0_output :  std_logic_vector(40 downto 0);
signal tile_0_filtered_output :  unsigned(40-0 downto 0);
signal bh7_w0_0 :  std_logic;
signal bh7_w1_0 :  std_logic;
signal bh7_w2_0 :  std_logic;
signal bh7_w3_0 :  std_logic;
signal bh7_w4_0 :  std_logic;
signal bh7_w5_0 :  std_logic;
signal bh7_w6_0 :  std_logic;
signal bh7_w7_0 :  std_logic;
signal bh7_w8_0 :  std_logic;
signal bh7_w9_0 :  std_logic;
signal bh7_w10_0 :  std_logic;
signal bh7_w11_0 :  std_logic;
signal bh7_w12_0 :  std_logic;
signal bh7_w13_0 :  std_logic;
signal bh7_w14_0 :  std_logic;
signal bh7_w15_0 :  std_logic;
signal bh7_w16_0 :  std_logic;
signal bh7_w17_0 :  std_logic;
signal bh7_w18_0 :  std_logic;
signal bh7_w19_0 :  std_logic;
signal bh7_w20_0 :  std_logic;
signal bh7_w21_0 :  std_logic;
signal bh7_w22_0 :  std_logic;
signal bh7_w23_0 :  std_logic;
signal bh7_w24_0 :  std_logic;
signal bh7_w25_0 :  std_logic;
signal bh7_w26_0 :  std_logic;
signal bh7_w27_0 :  std_logic;
signal bh7_w28_0 :  std_logic;
signal bh7_w29_0 :  std_logic;
signal bh7_w30_0 :  std_logic;
signal bh7_w31_0 :  std_logic;
signal bh7_w32_0 :  std_logic;
signal bh7_w33_0 :  std_logic;
signal bh7_w34_0 :  std_logic;
signal bh7_w35_0 :  std_logic;
signal bh7_w36_0 :  std_logic;
signal bh7_w37_0 :  std_logic;
signal bh7_w38_0 :  std_logic;
signal bh7_w39_0 :  std_logic;
signal bh7_w40_0 :  std_logic;
signal tile_1_X :  std_logic_vector(6 downto 0);
signal tile_1_Y :  std_logic_vector(23 downto 0);
signal tile_1_output :  std_logic_vector(30 downto 0);
signal tile_1_filtered_output :  unsigned(30-0 downto 0);
signal bh7_w17_1 :  std_logic;
signal bh7_w18_1 :  std_logic;
signal bh7_w19_1 :  std_logic;
signal bh7_w20_1 :  std_logic;
signal bh7_w21_1 :  std_logic;
signal bh7_w22_1 :  std_logic;
signal bh7_w23_1 :  std_logic;
signal bh7_w24_1 :  std_logic;
signal bh7_w25_1 :  std_logic;
signal bh7_w26_1 :  std_logic;
signal bh7_w27_1 :  std_logic;
signal bh7_w28_1 :  std_logic;
signal bh7_w29_1 :  std_logic;
signal bh7_w30_1 :  std_logic;
signal bh7_w31_1 :  std_logic;
signal bh7_w32_1 :  std_logic;
signal bh7_w33_1 :  std_logic;
signal bh7_w34_1 :  std_logic;
signal bh7_w35_1 :  std_logic;
signal bh7_w36_1 :  std_logic;
signal bh7_w37_1 :  std_logic;
signal bh7_w38_1 :  std_logic;
signal bh7_w39_1 :  std_logic;
signal bh7_w40_1 :  std_logic;
signal bh7_w41_0 :  std_logic;
signal bh7_w42_0 :  std_logic;
signal bh7_w43_0 :  std_logic;
signal bh7_w44_0 :  std_logic;
signal bh7_w45_0 :  std_logic;
signal bh7_w46_0 :  std_logic;
signal bh7_w47_0 :  std_logic;
signal tmp_bitheapResult_bh7_16, tmp_bitheapResult_bh7_16_d1 :  std_logic_vector(16 downto 0);
signal bitheapFinalAdd_bh7_In0 :  std_logic_vector(31 downto 0);
signal bitheapFinalAdd_bh7_In1 :  std_logic_vector(31 downto 0);
signal bitheapFinalAdd_bh7_Cin :  std_logic;
signal bitheapFinalAdd_bh7_Out :  std_logic_vector(31 downto 0);
signal bitheapResult_bh7 :  std_logic_vector(47 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               tmp_bitheapResult_bh7_16_d1 <=  tmp_bitheapResult_bh7_16;
            end if;
         end if;
      end process;
   XX_m6 <= X ;
   YY_m6 <= Y ;
   tile_0_X <= X(16 downto 0);
   tile_0_Y <= Y(23 downto 0);
   tile_0_mult: DSPBlock_17x24_Freq711_uid9
      port map ( clk  => clk,
                 ce => ce,
                 X => tile_0_X,
                 Y => tile_0_Y,
                 R => tile_0_output);

   tile_0_filtered_output <= unsigned(tile_0_output(40 downto 0));
   bh7_w0_0 <= tile_0_filtered_output(0);
   bh7_w1_0 <= tile_0_filtered_output(1);
   bh7_w2_0 <= tile_0_filtered_output(2);
   bh7_w3_0 <= tile_0_filtered_output(3);
   bh7_w4_0 <= tile_0_filtered_output(4);
   bh7_w5_0 <= tile_0_filtered_output(5);
   bh7_w6_0 <= tile_0_filtered_output(6);
   bh7_w7_0 <= tile_0_filtered_output(7);
   bh7_w8_0 <= tile_0_filtered_output(8);
   bh7_w9_0 <= tile_0_filtered_output(9);
   bh7_w10_0 <= tile_0_filtered_output(10);
   bh7_w11_0 <= tile_0_filtered_output(11);
   bh7_w12_0 <= tile_0_filtered_output(12);
   bh7_w13_0 <= tile_0_filtered_output(13);
   bh7_w14_0 <= tile_0_filtered_output(14);
   bh7_w15_0 <= tile_0_filtered_output(15);
   bh7_w16_0 <= tile_0_filtered_output(16);
   bh7_w17_0 <= tile_0_filtered_output(17);
   bh7_w18_0 <= tile_0_filtered_output(18);
   bh7_w19_0 <= tile_0_filtered_output(19);
   bh7_w20_0 <= tile_0_filtered_output(20);
   bh7_w21_0 <= tile_0_filtered_output(21);
   bh7_w22_0 <= tile_0_filtered_output(22);
   bh7_w23_0 <= tile_0_filtered_output(23);
   bh7_w24_0 <= tile_0_filtered_output(24);
   bh7_w25_0 <= tile_0_filtered_output(25);
   bh7_w26_0 <= tile_0_filtered_output(26);
   bh7_w27_0 <= tile_0_filtered_output(27);
   bh7_w28_0 <= tile_0_filtered_output(28);
   bh7_w29_0 <= tile_0_filtered_output(29);
   bh7_w30_0 <= tile_0_filtered_output(30);
   bh7_w31_0 <= tile_0_filtered_output(31);
   bh7_w32_0 <= tile_0_filtered_output(32);
   bh7_w33_0 <= tile_0_filtered_output(33);
   bh7_w34_0 <= tile_0_filtered_output(34);
   bh7_w35_0 <= tile_0_filtered_output(35);
   bh7_w36_0 <= tile_0_filtered_output(36);
   bh7_w37_0 <= tile_0_filtered_output(37);
   bh7_w38_0 <= tile_0_filtered_output(38);
   bh7_w39_0 <= tile_0_filtered_output(39);
   bh7_w40_0 <= tile_0_filtered_output(40);
   tile_1_X <= X(23 downto 17);
   tile_1_Y <= Y(23 downto 0);
   tile_1_mult: DSPBlock_7x24_Freq711_uid11
      port map ( clk  => clk,
                 ce => ce,
                 X => tile_1_X,
                 Y => tile_1_Y,
                 R => tile_1_output);

   tile_1_filtered_output <= unsigned(tile_1_output(30 downto 0));
   bh7_w17_1 <= tile_1_filtered_output(0);
   bh7_w18_1 <= tile_1_filtered_output(1);
   bh7_w19_1 <= tile_1_filtered_output(2);
   bh7_w20_1 <= tile_1_filtered_output(3);
   bh7_w21_1 <= tile_1_filtered_output(4);
   bh7_w22_1 <= tile_1_filtered_output(5);
   bh7_w23_1 <= tile_1_filtered_output(6);
   bh7_w24_1 <= tile_1_filtered_output(7);
   bh7_w25_1 <= tile_1_filtered_output(8);
   bh7_w26_1 <= tile_1_filtered_output(9);
   bh7_w27_1 <= tile_1_filtered_output(10);
   bh7_w28_1 <= tile_1_filtered_output(11);
   bh7_w29_1 <= tile_1_filtered_output(12);
   bh7_w30_1 <= tile_1_filtered_output(13);
   bh7_w31_1 <= tile_1_filtered_output(14);
   bh7_w32_1 <= tile_1_filtered_output(15);
   bh7_w33_1 <= tile_1_filtered_output(16);
   bh7_w34_1 <= tile_1_filtered_output(17);
   bh7_w35_1 <= tile_1_filtered_output(18);
   bh7_w36_1 <= tile_1_filtered_output(19);
   bh7_w37_1 <= tile_1_filtered_output(20);
   bh7_w38_1 <= tile_1_filtered_output(21);
   bh7_w39_1 <= tile_1_filtered_output(22);
   bh7_w40_1 <= tile_1_filtered_output(23);
   bh7_w41_0 <= tile_1_filtered_output(24);
   bh7_w42_0 <= tile_1_filtered_output(25);
   bh7_w43_0 <= tile_1_filtered_output(26);
   bh7_w44_0 <= tile_1_filtered_output(27);
   bh7_w45_0 <= tile_1_filtered_output(28);
   bh7_w46_0 <= tile_1_filtered_output(29);
   bh7_w47_0 <= tile_1_filtered_output(30);

   -- Adding the constant bits 
      -- All the constant bits are zero, nothing to add

   tmp_bitheapResult_bh7_16 <= bh7_w16_0 & bh7_w15_0 & bh7_w14_0 & bh7_w13_0 & bh7_w12_0 & bh7_w11_0 & bh7_w10_0 & bh7_w9_0 & bh7_w8_0 & bh7_w7_0 & bh7_w6_0 & bh7_w5_0 & bh7_w4_0 & bh7_w3_0 & bh7_w2_0 & bh7_w1_0 & bh7_w0_0;

   bitheapFinalAdd_bh7_In0 <= "0" & bh7_w47_0 & bh7_w46_0 & bh7_w45_0 & bh7_w44_0 & bh7_w43_0 & bh7_w42_0 & bh7_w41_0 & bh7_w40_0 & bh7_w39_0 & bh7_w38_0 & bh7_w37_0 & bh7_w36_0 & bh7_w35_0 & bh7_w34_0 & bh7_w33_0 & bh7_w32_0 & bh7_w31_0 & bh7_w30_0 & bh7_w29_0 & bh7_w28_0 & bh7_w27_0 & bh7_w26_0 & bh7_w25_0 & bh7_w24_0 & bh7_w23_0 & bh7_w22_0 & bh7_w21_0 & bh7_w20_0 & bh7_w19_0 & bh7_w18_0 & bh7_w17_0;
   bitheapFinalAdd_bh7_In1 <= "0" & "0" & "0" & "0" & "0" & "0" & "0" & "0" & bh7_w40_1 & bh7_w39_1 & bh7_w38_1 & bh7_w37_1 & bh7_w36_1 & bh7_w35_1 & bh7_w34_1 & bh7_w33_1 & bh7_w32_1 & bh7_w31_1 & bh7_w30_1 & bh7_w29_1 & bh7_w28_1 & bh7_w27_1 & bh7_w26_1 & bh7_w25_1 & bh7_w24_1 & bh7_w23_1 & bh7_w22_1 & bh7_w21_1 & bh7_w20_1 & bh7_w19_1 & bh7_w18_1 & bh7_w17_1;
   bitheapFinalAdd_bh7_Cin <= '0';

   bitheapFinalAdd_bh7: IntAdder_32_Freq711_uid14
      port map ( clk  => clk,
                 ce => ce,
                 Cin => bitheapFinalAdd_bh7_Cin,
                 X => bitheapFinalAdd_bh7_In0,
                 Y => bitheapFinalAdd_bh7_In1,
                 R => bitheapFinalAdd_bh7_Out);
   bitheapResult_bh7 <= bitheapFinalAdd_bh7_Out(30 downto 0) & tmp_bitheapResult_bh7_16_d1;
   R <= bitheapResult_bh7(47 downto 0);
end architecture;

--------------------------------------------------------------------------------
--                         IntAdder_33_Freq711_uid17
-- VHDL generated for Kintex7 @ 711MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2016)
--------------------------------------------------------------------------------
-- Pipeline depth: 2 cycles
-- Clock period (ns): 1.40647
-- Target frequency (MHz): 711
-- Input signals: X Y Cin
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntAdder_33_Freq711_uid17 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(32 downto 0);
          Y : in  std_logic_vector(32 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(32 downto 0)   );
end entity;

architecture arch of IntAdder_33_Freq711_uid17 is
signal Cin_1, Cin_1_d1 :  std_logic;
signal X_1, X_1_d1 :  std_logic_vector(19 downto 0);
signal Y_1, Y_1_d1, Y_1_d2, Y_1_d3 :  std_logic_vector(19 downto 0);
signal S_1 :  std_logic_vector(19 downto 0);
signal R_1, R_1_d1 :  std_logic_vector(18 downto 0);
signal Cin_2, Cin_2_d1 :  std_logic;
signal X_2, X_2_d1, X_2_d2 :  std_logic_vector(14 downto 0);
signal Y_2, Y_2_d1, Y_2_d2, Y_2_d3, Y_2_d4 :  std_logic_vector(14 downto 0);
signal S_2 :  std_logic_vector(14 downto 0);
signal R_2 :  std_logic_vector(13 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               Cin_1_d1 <=  Cin_1;
               X_1_d1 <=  X_1;
               Y_1_d1 <=  Y_1;
               Y_1_d2 <=  Y_1_d1;
               Y_1_d3 <=  Y_1_d2;
               R_1_d1 <=  R_1;
               Cin_2_d1 <=  Cin_2;
               X_2_d1 <=  X_2;
               X_2_d2 <=  X_2_d1;
               Y_2_d1 <=  Y_2;
               Y_2_d2 <=  Y_2_d1;
               Y_2_d3 <=  Y_2_d2;
               Y_2_d4 <=  Y_2_d3;
            end if;
         end if;
      end process;
   Cin_1 <= Cin;
   X_1 <= '0' & X(18 downto 0);
   Y_1 <= '0' & Y(18 downto 0);
   S_1 <= X_1_d1 + Y_1_d3 + Cin_1_d1;
   R_1 <= S_1(18 downto 0);
   Cin_2 <= S_1(19);
   X_2 <= '0' & X(32 downto 19);
   Y_2 <= '0' & Y(32 downto 19);
   S_2 <= X_2_d2 + Y_2_d4 + Cin_2_d1;
   R_2 <= S_2(13 downto 0);
   R <= R_2 & R_1_d1 ;
end architecture;

--------------------------------------------------------------------------------
--                          FloatingPointMultiplier
--                      (FPMult_8_23_uid2_Freq711_uid3)
-- VHDL generated for Kintex7 @ 711MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin 2008-2021
--------------------------------------------------------------------------------
-- Pipeline depth: 4 cycles
-- Clock period (ns): 1.40647
-- Target frequency (MHz): 711
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity FloatingPointMultiplier is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(8+23+2 downto 0);
          Y : in  std_logic_vector(8+23+2 downto 0);
          R : out  std_logic_vector(8+23+2 downto 0)   );
end entity;

architecture arch of FloatingPointMultiplier is
   component IntMultiplier_Freq711_uid5 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(23 downto 0);
             Y : in  std_logic_vector(23 downto 0);
             R : out  std_logic_vector(47 downto 0)   );
   end component;

   component IntAdder_33_Freq711_uid17 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(32 downto 0);
             Y : in  std_logic_vector(32 downto 0);
             Cin : in  std_logic;
             R : out  std_logic_vector(32 downto 0)   );
   end component;

signal sign, sign_d1, sign_d2, sign_d3, sign_d4 :  std_logic;
signal expX :  std_logic_vector(7 downto 0);
signal expY :  std_logic_vector(7 downto 0);
signal expSumPreSub, expSumPreSub_d1 :  std_logic_vector(9 downto 0);
signal bias, bias_d1 :  std_logic_vector(9 downto 0);
signal expSum :  std_logic_vector(9 downto 0);
signal sigX :  std_logic_vector(23 downto 0);
signal sigY :  std_logic_vector(23 downto 0);
signal sigProd, sigProd_d1 :  std_logic_vector(47 downto 0);
signal excSel :  std_logic_vector(3 downto 0);
signal exc, exc_d1, exc_d2, exc_d3, exc_d4 :  std_logic_vector(1 downto 0);
signal norm, norm_d1 :  std_logic;
signal expPostNorm, expPostNorm_d1 :  std_logic_vector(9 downto 0);
signal sigProdExt :  std_logic_vector(47 downto 0);
signal expSig :  std_logic_vector(32 downto 0);
signal sticky :  std_logic;
signal guard :  std_logic;
signal round :  std_logic;
signal expSigPostRound :  std_logic_vector(32 downto 0);
signal excPostNorm :  std_logic_vector(1 downto 0);
signal finalExc :  std_logic_vector(1 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               sign_d1 <=  sign;
               sign_d2 <=  sign_d1;
               sign_d3 <=  sign_d2;
               sign_d4 <=  sign_d3;
               expSumPreSub_d1 <=  expSumPreSub;
               bias_d1 <=  bias;
               sigProd_d1 <=  sigProd;
               exc_d1 <=  exc;
               exc_d2 <=  exc_d1;
               exc_d3 <=  exc_d2;
               exc_d4 <=  exc_d3;
               norm_d1 <=  norm;
               expPostNorm_d1 <=  expPostNorm;
            end if;
         end if;
      end process;
   sign <= X(31) xor Y(31);
   expX <= X(30 downto 23);
   expY <= Y(30 downto 23);
   expSumPreSub <= ("00" & expX) + ("00" & expY);
   bias <= CONV_STD_LOGIC_VECTOR(127,10);
   expSum <= expSumPreSub_d1 - bias_d1;
   sigX <= "1" & X(22 downto 0);
   sigY <= "1" & Y(22 downto 0);
   SignificandMultiplication: IntMultiplier_Freq711_uid5
      port map ( clk  => clk,
                 ce => ce,
                 X => sigX,
                 Y => sigY,
                 R => sigProd);
   excSel <= X(33 downto 32) & Y(33 downto 32);
   with excSel  select  
   exc <= "00" when  "0000" | "0001" | "0100", 
          "01" when "0101",
          "10" when "0110" | "1001" | "1010" ,
          "11" when others;
   norm <= sigProd(47);
   -- exponent update
   expPostNorm <= expSum + ("000000000" & norm);
   -- significand normalization shift
   sigProdExt <= sigProd_d1(46 downto 0) & "0" when norm_d1='1' else
                         sigProd_d1(45 downto 0) & "00";
   expSig <= expPostNorm_d1 & sigProdExt(47 downto 25);
   sticky <= sigProdExt(24);
   guard <= '0' when sigProdExt(23 downto 0)="000000000000000000000000" else '1';
   round <= sticky and ( (guard and not(sigProdExt(25))) or (sigProdExt(25) ))  ;
   RoundingAdder: IntAdder_33_Freq711_uid17
      port map ( clk  => clk,
                 ce => ce,
                 Cin => round,
                 X => expSig,
                 Y => "000000000000000000000000000000000",
                 R => expSigPostRound);
   with expSigPostRound(32 downto 31)  select 
   excPostNorm <=  "01"  when  "00",
                               "10"             when "01", 
                               "00"             when "11"|"10",
                               "11"             when others;
   with exc_d4  select  
   finalExc <= exc_d4 when  "11"|"10"|"00",
                       excPostNorm when others; 
   R <= finalExc & sign_d4 & expSigPostRound(30 downto 0);
end architecture;



--------------------------------------------------------------------------------
--                          selFunction_Freq630_uid4
-- VHDL generated for Kintex7 @ 630MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin, Bogdan Pasca (2007-2022)
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): 1.5873
-- Target frequency (MHz): 630
-- Input signals: X
-- Output signals: Y

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity selFunction_Freq630_uid4 is
    port (X : in  std_logic_vector(8 downto 0);
          Y : out  std_logic_vector(2 downto 0)   );
end entity;

architecture arch of selFunction_Freq630_uid4 is
signal Y0 :  std_logic_vector(2 downto 0);
attribute ram_extract: string;
attribute ram_style: string;
attribute ram_extract of Y0: signal is "yes";
attribute ram_style of Y0: signal is "distributed";
signal Y1 :  std_logic_vector(2 downto 0);
begin
   with X  select  Y0 <= 
      "000" when "000000000",
      "000" when "000000001",
      "000" when "000000010",
      "000" when "000000011",
      "000" when "000000100",
      "000" when "000000101",
      "000" when "000000110",
      "000" when "000000111",
      "000" when "000001000",
      "000" when "000001001",
      "000" when "000001010",
      "000" when "000001011",
      "000" when "000001100",
      "000" when "000001101",
      "000" when "000001110",
      "000" when "000001111",
      "001" when "000010000",
      "000" when "000010001",
      "000" when "000010010",
      "000" when "000010011",
      "000" when "000010100",
      "000" when "000010101",
      "000" when "000010110",
      "000" when "000010111",
      "001" when "000011000",
      "001" when "000011001",
      "001" when "000011010",
      "001" when "000011011",
      "000" when "000011100",
      "000" when "000011101",
      "000" when "000011110",
      "000" when "000011111",
      "001" when "000100000",
      "001" when "000100001",
      "001" when "000100010",
      "001" when "000100011",
      "001" when "000100100",
      "001" when "000100101",
      "001" when "000100110",
      "000" when "000100111",
      "001" when "000101000",
      "001" when "000101001",
      "001" when "000101010",
      "001" when "000101011",
      "001" when "000101100",
      "001" when "000101101",
      "001" when "000101110",
      "001" when "000101111",
      "010" when "000110000",
      "001" when "000110001",
      "001" when "000110010",
      "001" when "000110011",
      "001" when "000110100",
      "001" when "000110101",
      "001" when "000110110",
      "001" when "000110111",
      "010" when "000111000",
      "010" when "000111001",
      "001" when "000111010",
      "001" when "000111011",
      "001" when "000111100",
      "001" when "000111101",
      "001" when "000111110",
      "001" when "000111111",
      "010" when "001000000",
      "010" when "001000001",
      "010" when "001000010",
      "001" when "001000011",
      "001" when "001000100",
      "001" when "001000101",
      "001" when "001000110",
      "001" when "001000111",
      "010" when "001001000",
      "010" when "001001001",
      "010" when "001001010",
      "010" when "001001011",
      "001" when "001001100",
      "001" when "001001101",
      "001" when "001001110",
      "001" when "001001111",
      "010" when "001010000",
      "010" when "001010001",
      "010" when "001010010",
      "010" when "001010011",
      "010" when "001010100",
      "010" when "001010101",
      "001" when "001010110",
      "001" when "001010111",
      "010" when "001011000",
      "010" when "001011001",
      "010" when "001011010",
      "010" when "001011011",
      "010" when "001011100",
      "010" when "001011101",
      "010" when "001011110",
      "001" when "001011111",
      "010" when "001100000",
      "010" when "001100001",
      "010" when "001100010",
      "010" when "001100011",
      "010" when "001100100",
      "010" when "001100101",
      "010" when "001100110",
      "010" when "001100111",
      "010" when "001101000",
      "010" when "001101001",
      "010" when "001101010",
      "010" when "001101011",
      "010" when "001101100",
      "010" when "001101101",
      "010" when "001101110",
      "010" when "001101111",
      "010" when "001110000",
      "010" when "001110001",
      "010" when "001110010",
      "010" when "001110011",
      "010" when "001110100",
      "010" when "001110101",
      "010" when "001110110",
      "010" when "001110111",
      "010" when "001111000",
      "010" when "001111001",
      "010" when "001111010",
      "010" when "001111011",
      "010" when "001111100",
      "010" when "001111101",
      "010" when "001111110",
      "010" when "001111111",
      "010" when "010000000",
      "010" when "010000001",
      "010" when "010000010",
      "010" when "010000011",
      "010" when "010000100",
      "010" when "010000101",
      "010" when "010000110",
      "010" when "010000111",
      "010" when "010001000",
      "010" when "010001001",
      "010" when "010001010",
      "010" when "010001011",
      "010" when "010001100",
      "010" when "010001101",
      "010" when "010001110",
      "010" when "010001111",
      "010" when "010010000",
      "010" when "010010001",
      "010" when "010010010",
      "010" when "010010011",
      "010" when "010010100",
      "010" when "010010101",
      "010" when "010010110",
      "010" when "010010111",
      "010" when "010011000",
      "010" when "010011001",
      "010" when "010011010",
      "010" when "010011011",
      "010" when "010011100",
      "010" when "010011101",
      "010" when "010011110",
      "010" when "010011111",
      "010" when "010100000",
      "010" when "010100001",
      "010" when "010100010",
      "010" when "010100011",
      "010" when "010100100",
      "010" when "010100101",
      "010" when "010100110",
      "010" when "010100111",
      "010" when "010101000",
      "010" when "010101001",
      "010" when "010101010",
      "010" when "010101011",
      "010" when "010101100",
      "010" when "010101101",
      "010" when "010101110",
      "010" when "010101111",
      "010" when "010110000",
      "010" when "010110001",
      "010" when "010110010",
      "010" when "010110011",
      "010" when "010110100",
      "010" when "010110101",
      "010" when "010110110",
      "010" when "010110111",
      "010" when "010111000",
      "010" when "010111001",
      "010" when "010111010",
      "010" when "010111011",
      "010" when "010111100",
      "010" when "010111101",
      "010" when "010111110",
      "010" when "010111111",
      "010" when "011000000",
      "010" when "011000001",
      "010" when "011000010",
      "010" when "011000011",
      "010" when "011000100",
      "010" when "011000101",
      "010" when "011000110",
      "010" when "011000111",
      "010" when "011001000",
      "010" when "011001001",
      "010" when "011001010",
      "010" when "011001011",
      "010" when "011001100",
      "010" when "011001101",
      "010" when "011001110",
      "010" when "011001111",
      "010" when "011010000",
      "010" when "011010001",
      "010" when "011010010",
      "010" when "011010011",
      "010" when "011010100",
      "010" when "011010101",
      "010" when "011010110",
      "010" when "011010111",
      "010" when "011011000",
      "010" when "011011001",
      "010" when "011011010",
      "010" when "011011011",
      "010" when "011011100",
      "010" when "011011101",
      "010" when "011011110",
      "010" when "011011111",
      "010" when "011100000",
      "010" when "011100001",
      "010" when "011100010",
      "010" when "011100011",
      "010" when "011100100",
      "010" when "011100101",
      "010" when "011100110",
      "010" when "011100111",
      "010" when "011101000",
      "010" when "011101001",
      "010" when "011101010",
      "010" when "011101011",
      "010" when "011101100",
      "010" when "011101101",
      "010" when "011101110",
      "010" when "011101111",
      "010" when "011110000",
      "010" when "011110001",
      "010" when "011110010",
      "010" when "011110011",
      "010" when "011110100",
      "010" when "011110101",
      "010" when "011110110",
      "010" when "011110111",
      "010" when "011111000",
      "010" when "011111001",
      "010" when "011111010",
      "010" when "011111011",
      "010" when "011111100",
      "010" when "011111101",
      "010" when "011111110",
      "010" when "011111111",
      "110" when "100000000",
      "110" when "100000001",
      "110" when "100000010",
      "110" when "100000011",
      "110" when "100000100",
      "110" when "100000101",
      "110" when "100000110",
      "110" when "100000111",
      "110" when "100001000",
      "110" when "100001001",
      "110" when "100001010",
      "110" when "100001011",
      "110" when "100001100",
      "110" when "100001101",
      "110" when "100001110",
      "110" when "100001111",
      "110" when "100010000",
      "110" when "100010001",
      "110" when "100010010",
      "110" when "100010011",
      "110" when "100010100",
      "110" when "100010101",
      "110" when "100010110",
      "110" when "100010111",
      "110" when "100011000",
      "110" when "100011001",
      "110" when "100011010",
      "110" when "100011011",
      "110" when "100011100",
      "110" when "100011101",
      "110" when "100011110",
      "110" when "100011111",
      "110" when "100100000",
      "110" when "100100001",
      "110" when "100100010",
      "110" when "100100011",
      "110" when "100100100",
      "110" when "100100101",
      "110" when "100100110",
      "110" when "100100111",
      "110" when "100101000",
      "110" when "100101001",
      "110" when "100101010",
      "110" when "100101011",
      "110" when "100101100",
      "110" when "100101101",
      "110" when "100101110",
      "110" when "100101111",
      "110" when "100110000",
      "110" when "100110001",
      "110" when "100110010",
      "110" when "100110011",
      "110" when "100110100",
      "110" when "100110101",
      "110" when "100110110",
      "110" when "100110111",
      "110" when "100111000",
      "110" when "100111001",
      "110" when "100111010",
      "110" when "100111011",
      "110" when "100111100",
      "110" when "100111101",
      "110" when "100111110",
      "110" when "100111111",
      "110" when "101000000",
      "110" when "101000001",
      "110" when "101000010",
      "110" when "101000011",
      "110" when "101000100",
      "110" when "101000101",
      "110" when "101000110",
      "110" when "101000111",
      "110" when "101001000",
      "110" when "101001001",
      "110" when "101001010",
      "110" when "101001011",
      "110" when "101001100",
      "110" when "101001101",
      "110" when "101001110",
      "110" when "101001111",
      "110" when "101010000",
      "110" when "101010001",
      "110" when "101010010",
      "110" when "101010011",
      "110" when "101010100",
      "110" when "101010101",
      "110" when "101010110",
      "110" when "101010111",
      "110" when "101011000",
      "110" when "101011001",
      "110" when "101011010",
      "110" when "101011011",
      "110" when "101011100",
      "110" when "101011101",
      "110" when "101011110",
      "110" when "101011111",
      "110" when "101100000",
      "110" when "101100001",
      "110" when "101100010",
      "110" when "101100011",
      "110" when "101100100",
      "110" when "101100101",
      "110" when "101100110",
      "110" when "101100111",
      "110" when "101101000",
      "110" when "101101001",
      "110" when "101101010",
      "110" when "101101011",
      "110" when "101101100",
      "110" when "101101101",
      "110" when "101101110",
      "110" when "101101111",
      "110" when "101110000",
      "110" when "101110001",
      "110" when "101110010",
      "110" when "101110011",
      "110" when "101110100",
      "110" when "101110101",
      "110" when "101110110",
      "110" when "101110111",
      "110" when "101111000",
      "110" when "101111001",
      "110" when "101111010",
      "110" when "101111011",
      "110" when "101111100",
      "110" when "101111101",
      "110" when "101111110",
      "110" when "101111111",
      "110" when "110000000",
      "110" when "110000001",
      "110" when "110000010",
      "110" when "110000011",
      "110" when "110000100",
      "110" when "110000101",
      "110" when "110000110",
      "110" when "110000111",
      "110" when "110001000",
      "110" when "110001001",
      "110" when "110001010",
      "110" when "110001011",
      "110" when "110001100",
      "110" when "110001101",
      "110" when "110001110",
      "110" when "110001111",
      "110" when "110010000",
      "110" when "110010001",
      "110" when "110010010",
      "110" when "110010011",
      "110" when "110010100",
      "110" when "110010101",
      "110" when "110010110",
      "110" when "110010111",
      "110" when "110011000",
      "110" when "110011001",
      "110" when "110011010",
      "110" when "110011011",
      "110" when "110011100",
      "110" when "110011101",
      "110" when "110011110",
      "110" when "110011111",
      "110" when "110100000",
      "110" when "110100001",
      "110" when "110100010",
      "110" when "110100011",
      "110" when "110100100",
      "110" when "110100101",
      "110" when "110100110",
      "110" when "110100111",
      "110" when "110101000",
      "110" when "110101001",
      "110" when "110101010",
      "110" when "110101011",
      "110" when "110101100",
      "110" when "110101101",
      "110" when "110101110",
      "111" when "110101111",
      "110" when "110110000",
      "110" when "110110001",
      "110" when "110110010",
      "110" when "110110011",
      "110" when "110110100",
      "111" when "110110101",
      "111" when "110110110",
      "111" when "110110111",
      "110" when "110111000",
      "110" when "110111001",
      "110" when "110111010",
      "110" when "110111011",
      "111" when "110111100",
      "111" when "110111101",
      "111" when "110111110",
      "111" when "110111111",
      "110" when "111000000",
      "110" when "111000001",
      "111" when "111000010",
      "111" when "111000011",
      "111" when "111000100",
      "111" when "111000101",
      "111" when "111000110",
      "111" when "111000111",
      "110" when "111001000",
      "111" when "111001001",
      "111" when "111001010",
      "111" when "111001011",
      "111" when "111001100",
      "111" when "111001101",
      "111" when "111001110",
      "111" when "111001111",
      "111" when "111010000",
      "111" when "111010001",
      "111" when "111010010",
      "111" when "111010011",
      "111" when "111010100",
      "111" when "111010101",
      "111" when "111010110",
      "111" when "111010111",
      "111" when "111011000",
      "111" when "111011001",
      "111" when "111011010",
      "111" when "111011011",
      "111" when "111011100",
      "111" when "111011101",
      "111" when "111011110",
      "111" when "111011111",
      "111" when "111100000",
      "111" when "111100001",
      "111" when "111100010",
      "111" when "111100011",
      "111" when "111100100",
      "111" when "111100101",
      "111" when "111100110",
      "111" when "111100111",
      "111" when "111101000",
      "111" when "111101001",
      "111" when "111101010",
      "111" when "111101011",
      "000" when "111101100",
      "000" when "111101101",
      "000" when "111101110",
      "000" when "111101111",
      "000" when "111110000",
      "000" when "111110001",
      "000" when "111110010",
      "000" when "111110011",
      "000" when "111110100",
      "000" when "111110101",
      "000" when "111110110",
      "000" when "111110111",
      "000" when "111111000",
      "000" when "111111001",
      "000" when "111111010",
      "000" when "111111011",
      "000" when "111111100",
      "000" when "111111101",
      "000" when "111111110",
      "000" when "111111111",
      "---" when others;
   Y1 <= Y0; -- for the possible blockram register
   Y <= Y1;
end architecture;

--------------------------------------------------------------------------------
--                            FloatingPointDivider
--                         (FPDiv_8_23_Freq630_uid2)
-- VHDL generated for Kintex7 @ 630MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Maxime Christ, Florent de Dinechin (2015)
--------------------------------------------------------------------------------
-- Pipeline depth: 20 cycles
-- Clock period (ns): 1.5873
-- Target frequency (MHz): 630
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity FloatingPointDivider is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(8+23+2 downto 0);
          Y : in  std_logic_vector(8+23+2 downto 0);
          R : out  std_logic_vector(8+23+2 downto 0)   );
end entity;

architecture arch of FloatingPointDivider is
   component selFunction_Freq630_uid4 is
      port ( X : in  std_logic_vector(8 downto 0);
             Y : out  std_logic_vector(2 downto 0)   );
   end component;

signal fX :  std_logic_vector(23 downto 0);
signal fY :  std_logic_vector(23 downto 0);
signal expR0, expR0_d1, expR0_d2, expR0_d3, expR0_d4, expR0_d5, expR0_d6, expR0_d7, expR0_d8, expR0_d9, expR0_d10, expR0_d11, expR0_d12, expR0_d13, expR0_d14, expR0_d15, expR0_d16, expR0_d17, expR0_d18, expR0_d19, expR0_d20 :  std_logic_vector(9 downto 0);
signal sR, sR_d1, sR_d2, sR_d3, sR_d4, sR_d5, sR_d6, sR_d7, sR_d8, sR_d9, sR_d10, sR_d11, sR_d12, sR_d13, sR_d14, sR_d15, sR_d16, sR_d17, sR_d18, sR_d19, sR_d20 :  std_logic;
signal exnXY :  std_logic_vector(3 downto 0);
signal exnR0, exnR0_d1, exnR0_d2, exnR0_d3, exnR0_d4, exnR0_d5, exnR0_d6, exnR0_d7, exnR0_d8, exnR0_d9, exnR0_d10, exnR0_d11, exnR0_d12, exnR0_d13, exnR0_d14, exnR0_d15, exnR0_d16, exnR0_d17, exnR0_d18, exnR0_d19, exnR0_d20 :  std_logic_vector(1 downto 0);
signal D, D_d1, D_d2, D_d3, D_d4, D_d5, D_d6, D_d7, D_d8, D_d9, D_d10, D_d11, D_d12, D_d13, D_d14, D_d15, D_d16, D_d17 :  std_logic_vector(23 downto 0);
signal psX :  std_logic_vector(24 downto 0);
signal betaw14, betaw14_d1 :  std_logic_vector(26 downto 0);
signal sel14 :  std_logic_vector(8 downto 0);
signal q14, q14_d1 :  std_logic_vector(2 downto 0);
signal q14_copy5 :  std_logic_vector(2 downto 0);
signal absq14D, absq14D_d1 :  std_logic_vector(26 downto 0);
signal w13 :  std_logic_vector(26 downto 0);
signal betaw13, betaw13_d1 :  std_logic_vector(26 downto 0);
signal sel13 :  std_logic_vector(8 downto 0);
signal q13, q13_d1 :  std_logic_vector(2 downto 0);
signal q13_copy6 :  std_logic_vector(2 downto 0);
signal absq13D, absq13D_d1 :  std_logic_vector(26 downto 0);
signal w12 :  std_logic_vector(26 downto 0);
signal betaw12, betaw12_d1, betaw12_d2 :  std_logic_vector(26 downto 0);
signal sel12 :  std_logic_vector(8 downto 0);
signal q12, q12_d1 :  std_logic_vector(2 downto 0);
signal q12_copy7, q12_copy7_d1 :  std_logic_vector(2 downto 0);
signal absq12D, absq12D_d1 :  std_logic_vector(26 downto 0);
signal w11 :  std_logic_vector(26 downto 0);
signal betaw11, betaw11_d1 :  std_logic_vector(26 downto 0);
signal sel11 :  std_logic_vector(8 downto 0);
signal q11, q11_d1 :  std_logic_vector(2 downto 0);
signal q11_copy8 :  std_logic_vector(2 downto 0);
signal absq11D, absq11D_d1 :  std_logic_vector(26 downto 0);
signal w10 :  std_logic_vector(26 downto 0);
signal betaw10, betaw10_d1 :  std_logic_vector(26 downto 0);
signal sel10 :  std_logic_vector(8 downto 0);
signal q10, q10_d1 :  std_logic_vector(2 downto 0);
signal q10_copy9 :  std_logic_vector(2 downto 0);
signal absq10D, absq10D_d1 :  std_logic_vector(26 downto 0);
signal w9 :  std_logic_vector(26 downto 0);
signal betaw9, betaw9_d1, betaw9_d2 :  std_logic_vector(26 downto 0);
signal sel9 :  std_logic_vector(8 downto 0);
signal q9, q9_d1 :  std_logic_vector(2 downto 0);
signal q9_copy10, q9_copy10_d1 :  std_logic_vector(2 downto 0);
signal absq9D, absq9D_d1 :  std_logic_vector(26 downto 0);
signal w8 :  std_logic_vector(26 downto 0);
signal betaw8, betaw8_d1 :  std_logic_vector(26 downto 0);
signal sel8 :  std_logic_vector(8 downto 0);
signal q8, q8_d1 :  std_logic_vector(2 downto 0);
signal q8_copy11 :  std_logic_vector(2 downto 0);
signal absq8D, absq8D_d1 :  std_logic_vector(26 downto 0);
signal w7 :  std_logic_vector(26 downto 0);
signal betaw7, betaw7_d1 :  std_logic_vector(26 downto 0);
signal sel7 :  std_logic_vector(8 downto 0);
signal q7, q7_d1 :  std_logic_vector(2 downto 0);
signal q7_copy12 :  std_logic_vector(2 downto 0);
signal absq7D, absq7D_d1 :  std_logic_vector(26 downto 0);
signal w6 :  std_logic_vector(26 downto 0);
signal betaw6, betaw6_d1, betaw6_d2 :  std_logic_vector(26 downto 0);
signal sel6 :  std_logic_vector(8 downto 0);
signal q6, q6_d1 :  std_logic_vector(2 downto 0);
signal q6_copy13, q6_copy13_d1 :  std_logic_vector(2 downto 0);
signal absq6D, absq6D_d1 :  std_logic_vector(26 downto 0);
signal w5 :  std_logic_vector(26 downto 0);
signal betaw5, betaw5_d1 :  std_logic_vector(26 downto 0);
signal sel5 :  std_logic_vector(8 downto 0);
signal q5, q5_d1 :  std_logic_vector(2 downto 0);
signal q5_copy14 :  std_logic_vector(2 downto 0);
signal absq5D, absq5D_d1 :  std_logic_vector(26 downto 0);
signal w4 :  std_logic_vector(26 downto 0);
signal betaw4, betaw4_d1 :  std_logic_vector(26 downto 0);
signal sel4 :  std_logic_vector(8 downto 0);
signal q4, q4_d1 :  std_logic_vector(2 downto 0);
signal q4_copy15 :  std_logic_vector(2 downto 0);
signal absq4D, absq4D_d1 :  std_logic_vector(26 downto 0);
signal w3 :  std_logic_vector(26 downto 0);
signal betaw3, betaw3_d1, betaw3_d2 :  std_logic_vector(26 downto 0);
signal sel3 :  std_logic_vector(8 downto 0);
signal q3, q3_d1 :  std_logic_vector(2 downto 0);
signal q3_copy16, q3_copy16_d1 :  std_logic_vector(2 downto 0);
signal absq3D, absq3D_d1 :  std_logic_vector(26 downto 0);
signal w2 :  std_logic_vector(26 downto 0);
signal betaw2, betaw2_d1 :  std_logic_vector(26 downto 0);
signal sel2 :  std_logic_vector(8 downto 0);
signal q2, q2_d1 :  std_logic_vector(2 downto 0);
signal q2_copy17 :  std_logic_vector(2 downto 0);
signal absq2D, absq2D_d1 :  std_logic_vector(26 downto 0);
signal w1 :  std_logic_vector(26 downto 0);
signal betaw1, betaw1_d1 :  std_logic_vector(26 downto 0);
signal sel1 :  std_logic_vector(8 downto 0);
signal q1, q1_d1 :  std_logic_vector(2 downto 0);
signal q1_copy18 :  std_logic_vector(2 downto 0);
signal absq1D, absq1D_d1 :  std_logic_vector(26 downto 0);
signal w0 :  std_logic_vector(26 downto 0);
signal wfinal :  std_logic_vector(24 downto 0);
signal qM0 :  std_logic;
signal qP14, qP14_d1, qP14_d2, qP14_d3, qP14_d4, qP14_d5, qP14_d6, qP14_d7, qP14_d8, qP14_d9, qP14_d10, qP14_d11, qP14_d12, qP14_d13, qP14_d14, qP14_d15, qP14_d16, qP14_d17 :  std_logic_vector(1 downto 0);
signal qM14, qM14_d1, qM14_d2, qM14_d3, qM14_d4, qM14_d5, qM14_d6, qM14_d7, qM14_d8, qM14_d9, qM14_d10, qM14_d11, qM14_d12, qM14_d13, qM14_d14, qM14_d15, qM14_d16, qM14_d17, qM14_d18 :  std_logic_vector(1 downto 0);
signal qP13, qP13_d1, qP13_d2, qP13_d3, qP13_d4, qP13_d5, qP13_d6, qP13_d7, qP13_d8, qP13_d9, qP13_d10, qP13_d11, qP13_d12, qP13_d13, qP13_d14, qP13_d15, qP13_d16 :  std_logic_vector(1 downto 0);
signal qM13, qM13_d1, qM13_d2, qM13_d3, qM13_d4, qM13_d5, qM13_d6, qM13_d7, qM13_d8, qM13_d9, qM13_d10, qM13_d11, qM13_d12, qM13_d13, qM13_d14, qM13_d15, qM13_d16, qM13_d17 :  std_logic_vector(1 downto 0);
signal qP12, qP12_d1, qP12_d2, qP12_d3, qP12_d4, qP12_d5, qP12_d6, qP12_d7, qP12_d8, qP12_d9, qP12_d10, qP12_d11, qP12_d12, qP12_d13, qP12_d14 :  std_logic_vector(1 downto 0);
signal qM12, qM12_d1, qM12_d2, qM12_d3, qM12_d4, qM12_d5, qM12_d6, qM12_d7, qM12_d8, qM12_d9, qM12_d10, qM12_d11, qM12_d12, qM12_d13, qM12_d14, qM12_d15 :  std_logic_vector(1 downto 0);
signal qP11, qP11_d1, qP11_d2, qP11_d3, qP11_d4, qP11_d5, qP11_d6, qP11_d7, qP11_d8, qP11_d9, qP11_d10, qP11_d11, qP11_d12, qP11_d13 :  std_logic_vector(1 downto 0);
signal qM11, qM11_d1, qM11_d2, qM11_d3, qM11_d4, qM11_d5, qM11_d6, qM11_d7, qM11_d8, qM11_d9, qM11_d10, qM11_d11, qM11_d12, qM11_d13, qM11_d14 :  std_logic_vector(1 downto 0);
signal qP10, qP10_d1, qP10_d2, qP10_d3, qP10_d4, qP10_d5, qP10_d6, qP10_d7, qP10_d8, qP10_d9, qP10_d10, qP10_d11, qP10_d12 :  std_logic_vector(1 downto 0);
signal qM10, qM10_d1, qM10_d2, qM10_d3, qM10_d4, qM10_d5, qM10_d6, qM10_d7, qM10_d8, qM10_d9, qM10_d10, qM10_d11, qM10_d12, qM10_d13 :  std_logic_vector(1 downto 0);
signal qP9, qP9_d1, qP9_d2, qP9_d3, qP9_d4, qP9_d5, qP9_d6, qP9_d7, qP9_d8, qP9_d9, qP9_d10 :  std_logic_vector(1 downto 0);
signal qM9, qM9_d1, qM9_d2, qM9_d3, qM9_d4, qM9_d5, qM9_d6, qM9_d7, qM9_d8, qM9_d9, qM9_d10, qM9_d11 :  std_logic_vector(1 downto 0);
signal qP8, qP8_d1, qP8_d2, qP8_d3, qP8_d4, qP8_d5, qP8_d6, qP8_d7, qP8_d8, qP8_d9 :  std_logic_vector(1 downto 0);
signal qM8, qM8_d1, qM8_d2, qM8_d3, qM8_d4, qM8_d5, qM8_d6, qM8_d7, qM8_d8, qM8_d9, qM8_d10 :  std_logic_vector(1 downto 0);
signal qP7, qP7_d1, qP7_d2, qP7_d3, qP7_d4, qP7_d5, qP7_d6, qP7_d7, qP7_d8 :  std_logic_vector(1 downto 0);
signal qM7, qM7_d1, qM7_d2, qM7_d3, qM7_d4, qM7_d5, qM7_d6, qM7_d7, qM7_d8, qM7_d9 :  std_logic_vector(1 downto 0);
signal qP6, qP6_d1, qP6_d2, qP6_d3, qP6_d4, qP6_d5, qP6_d6 :  std_logic_vector(1 downto 0);
signal qM6, qM6_d1, qM6_d2, qM6_d3, qM6_d4, qM6_d5, qM6_d6, qM6_d7 :  std_logic_vector(1 downto 0);
signal qP5, qP5_d1, qP5_d2, qP5_d3, qP5_d4, qP5_d5 :  std_logic_vector(1 downto 0);
signal qM5, qM5_d1, qM5_d2, qM5_d3, qM5_d4, qM5_d5, qM5_d6 :  std_logic_vector(1 downto 0);
signal qP4, qP4_d1, qP4_d2, qP4_d3, qP4_d4 :  std_logic_vector(1 downto 0);
signal qM4, qM4_d1, qM4_d2, qM4_d3, qM4_d4, qM4_d5 :  std_logic_vector(1 downto 0);
signal qP3, qP3_d1, qP3_d2 :  std_logic_vector(1 downto 0);
signal qM3, qM3_d1, qM3_d2, qM3_d3 :  std_logic_vector(1 downto 0);
signal qP2, qP2_d1 :  std_logic_vector(1 downto 0);
signal qM2, qM2_d1, qM2_d2 :  std_logic_vector(1 downto 0);
signal qP1 :  std_logic_vector(1 downto 0);
signal qM1, qM1_d1 :  std_logic_vector(1 downto 0);
signal qP, qP_d1, qP_d2 :  std_logic_vector(27 downto 0);
signal qM, qM_d1 :  std_logic_vector(27 downto 0);
signal quotient :  std_logic_vector(27 downto 0);
signal mR, mR_d1 :  std_logic_vector(25 downto 0);
signal fRnorm, fRnorm_d1 :  std_logic_vector(23 downto 0);
signal round, round_d1 :  std_logic;
signal expR1 :  std_logic_vector(9 downto 0);
signal expfrac :  std_logic_vector(32 downto 0);
signal expfracR :  std_logic_vector(32 downto 0);
signal exnR :  std_logic_vector(1 downto 0);
signal exnRfinal :  std_logic_vector(1 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               expR0_d1 <=  expR0;
               expR0_d2 <=  expR0_d1;
               expR0_d3 <=  expR0_d2;
               expR0_d4 <=  expR0_d3;
               expR0_d5 <=  expR0_d4;
               expR0_d6 <=  expR0_d5;
               expR0_d7 <=  expR0_d6;
               expR0_d8 <=  expR0_d7;
               expR0_d9 <=  expR0_d8;
               expR0_d10 <=  expR0_d9;
               expR0_d11 <=  expR0_d10;
               expR0_d12 <=  expR0_d11;
               expR0_d13 <=  expR0_d12;
               expR0_d14 <=  expR0_d13;
               expR0_d15 <=  expR0_d14;
               expR0_d16 <=  expR0_d15;
               expR0_d17 <=  expR0_d16;
               expR0_d18 <=  expR0_d17;
               expR0_d19 <=  expR0_d18;
               expR0_d20 <=  expR0_d19;
               sR_d1 <=  sR;
               sR_d2 <=  sR_d1;
               sR_d3 <=  sR_d2;
               sR_d4 <=  sR_d3;
               sR_d5 <=  sR_d4;
               sR_d6 <=  sR_d5;
               sR_d7 <=  sR_d6;
               sR_d8 <=  sR_d7;
               sR_d9 <=  sR_d8;
               sR_d10 <=  sR_d9;
               sR_d11 <=  sR_d10;
               sR_d12 <=  sR_d11;
               sR_d13 <=  sR_d12;
               sR_d14 <=  sR_d13;
               sR_d15 <=  sR_d14;
               sR_d16 <=  sR_d15;
               sR_d17 <=  sR_d16;
               sR_d18 <=  sR_d17;
               sR_d19 <=  sR_d18;
               sR_d20 <=  sR_d19;
               exnR0_d1 <=  exnR0;
               exnR0_d2 <=  exnR0_d1;
               exnR0_d3 <=  exnR0_d2;
               exnR0_d4 <=  exnR0_d3;
               exnR0_d5 <=  exnR0_d4;
               exnR0_d6 <=  exnR0_d5;
               exnR0_d7 <=  exnR0_d6;
               exnR0_d8 <=  exnR0_d7;
               exnR0_d9 <=  exnR0_d8;
               exnR0_d10 <=  exnR0_d9;
               exnR0_d11 <=  exnR0_d10;
               exnR0_d12 <=  exnR0_d11;
               exnR0_d13 <=  exnR0_d12;
               exnR0_d14 <=  exnR0_d13;
               exnR0_d15 <=  exnR0_d14;
               exnR0_d16 <=  exnR0_d15;
               exnR0_d17 <=  exnR0_d16;
               exnR0_d18 <=  exnR0_d17;
               exnR0_d19 <=  exnR0_d18;
               exnR0_d20 <=  exnR0_d19;
               D_d1 <=  D;
               D_d2 <=  D_d1;
               D_d3 <=  D_d2;
               D_d4 <=  D_d3;
               D_d5 <=  D_d4;
               D_d6 <=  D_d5;
               D_d7 <=  D_d6;
               D_d8 <=  D_d7;
               D_d9 <=  D_d8;
               D_d10 <=  D_d9;
               D_d11 <=  D_d10;
               D_d12 <=  D_d11;
               D_d13 <=  D_d12;
               D_d14 <=  D_d13;
               D_d15 <=  D_d14;
               D_d16 <=  D_d15;
               D_d17 <=  D_d16;
               betaw14_d1 <=  betaw14;
               q14_d1 <=  q14;
               absq14D_d1 <=  absq14D;
               betaw13_d1 <=  betaw13;
               q13_d1 <=  q13;
               absq13D_d1 <=  absq13D;
               betaw12_d1 <=  betaw12;
               betaw12_d2 <=  betaw12_d1;
               q12_d1 <=  q12;
               q12_copy7_d1 <=  q12_copy7;
               absq12D_d1 <=  absq12D;
               betaw11_d1 <=  betaw11;
               q11_d1 <=  q11;
               absq11D_d1 <=  absq11D;
               betaw10_d1 <=  betaw10;
               q10_d1 <=  q10;
               absq10D_d1 <=  absq10D;
               betaw9_d1 <=  betaw9;
               betaw9_d2 <=  betaw9_d1;
               q9_d1 <=  q9;
               q9_copy10_d1 <=  q9_copy10;
               absq9D_d1 <=  absq9D;
               betaw8_d1 <=  betaw8;
               q8_d1 <=  q8;
               absq8D_d1 <=  absq8D;
               betaw7_d1 <=  betaw7;
               q7_d1 <=  q7;
               absq7D_d1 <=  absq7D;
               betaw6_d1 <=  betaw6;
               betaw6_d2 <=  betaw6_d1;
               q6_d1 <=  q6;
               q6_copy13_d1 <=  q6_copy13;
               absq6D_d1 <=  absq6D;
               betaw5_d1 <=  betaw5;
               q5_d1 <=  q5;
               absq5D_d1 <=  absq5D;
               betaw4_d1 <=  betaw4;
               q4_d1 <=  q4;
               absq4D_d1 <=  absq4D;
               betaw3_d1 <=  betaw3;
               betaw3_d2 <=  betaw3_d1;
               q3_d1 <=  q3;
               q3_copy16_d1 <=  q3_copy16;
               absq3D_d1 <=  absq3D;
               betaw2_d1 <=  betaw2;
               q2_d1 <=  q2;
               absq2D_d1 <=  absq2D;
               betaw1_d1 <=  betaw1;
               q1_d1 <=  q1;
               absq1D_d1 <=  absq1D;
               qP14_d1 <=  qP14;
               qP14_d2 <=  qP14_d1;
               qP14_d3 <=  qP14_d2;
               qP14_d4 <=  qP14_d3;
               qP14_d5 <=  qP14_d4;
               qP14_d6 <=  qP14_d5;
               qP14_d7 <=  qP14_d6;
               qP14_d8 <=  qP14_d7;
               qP14_d9 <=  qP14_d8;
               qP14_d10 <=  qP14_d9;
               qP14_d11 <=  qP14_d10;
               qP14_d12 <=  qP14_d11;
               qP14_d13 <=  qP14_d12;
               qP14_d14 <=  qP14_d13;
               qP14_d15 <=  qP14_d14;
               qP14_d16 <=  qP14_d15;
               qP14_d17 <=  qP14_d16;
               qM14_d1 <=  qM14;
               qM14_d2 <=  qM14_d1;
               qM14_d3 <=  qM14_d2;
               qM14_d4 <=  qM14_d3;
               qM14_d5 <=  qM14_d4;
               qM14_d6 <=  qM14_d5;
               qM14_d7 <=  qM14_d6;
               qM14_d8 <=  qM14_d7;
               qM14_d9 <=  qM14_d8;
               qM14_d10 <=  qM14_d9;
               qM14_d11 <=  qM14_d10;
               qM14_d12 <=  qM14_d11;
               qM14_d13 <=  qM14_d12;
               qM14_d14 <=  qM14_d13;
               qM14_d15 <=  qM14_d14;
               qM14_d16 <=  qM14_d15;
               qM14_d17 <=  qM14_d16;
               qM14_d18 <=  qM14_d17;
               qP13_d1 <=  qP13;
               qP13_d2 <=  qP13_d1;
               qP13_d3 <=  qP13_d2;
               qP13_d4 <=  qP13_d3;
               qP13_d5 <=  qP13_d4;
               qP13_d6 <=  qP13_d5;
               qP13_d7 <=  qP13_d6;
               qP13_d8 <=  qP13_d7;
               qP13_d9 <=  qP13_d8;
               qP13_d10 <=  qP13_d9;
               qP13_d11 <=  qP13_d10;
               qP13_d12 <=  qP13_d11;
               qP13_d13 <=  qP13_d12;
               qP13_d14 <=  qP13_d13;
               qP13_d15 <=  qP13_d14;
               qP13_d16 <=  qP13_d15;
               qM13_d1 <=  qM13;
               qM13_d2 <=  qM13_d1;
               qM13_d3 <=  qM13_d2;
               qM13_d4 <=  qM13_d3;
               qM13_d5 <=  qM13_d4;
               qM13_d6 <=  qM13_d5;
               qM13_d7 <=  qM13_d6;
               qM13_d8 <=  qM13_d7;
               qM13_d9 <=  qM13_d8;
               qM13_d10 <=  qM13_d9;
               qM13_d11 <=  qM13_d10;
               qM13_d12 <=  qM13_d11;
               qM13_d13 <=  qM13_d12;
               qM13_d14 <=  qM13_d13;
               qM13_d15 <=  qM13_d14;
               qM13_d16 <=  qM13_d15;
               qM13_d17 <=  qM13_d16;
               qP12_d1 <=  qP12;
               qP12_d2 <=  qP12_d1;
               qP12_d3 <=  qP12_d2;
               qP12_d4 <=  qP12_d3;
               qP12_d5 <=  qP12_d4;
               qP12_d6 <=  qP12_d5;
               qP12_d7 <=  qP12_d6;
               qP12_d8 <=  qP12_d7;
               qP12_d9 <=  qP12_d8;
               qP12_d10 <=  qP12_d9;
               qP12_d11 <=  qP12_d10;
               qP12_d12 <=  qP12_d11;
               qP12_d13 <=  qP12_d12;
               qP12_d14 <=  qP12_d13;
               qM12_d1 <=  qM12;
               qM12_d2 <=  qM12_d1;
               qM12_d3 <=  qM12_d2;
               qM12_d4 <=  qM12_d3;
               qM12_d5 <=  qM12_d4;
               qM12_d6 <=  qM12_d5;
               qM12_d7 <=  qM12_d6;
               qM12_d8 <=  qM12_d7;
               qM12_d9 <=  qM12_d8;
               qM12_d10 <=  qM12_d9;
               qM12_d11 <=  qM12_d10;
               qM12_d12 <=  qM12_d11;
               qM12_d13 <=  qM12_d12;
               qM12_d14 <=  qM12_d13;
               qM12_d15 <=  qM12_d14;
               qP11_d1 <=  qP11;
               qP11_d2 <=  qP11_d1;
               qP11_d3 <=  qP11_d2;
               qP11_d4 <=  qP11_d3;
               qP11_d5 <=  qP11_d4;
               qP11_d6 <=  qP11_d5;
               qP11_d7 <=  qP11_d6;
               qP11_d8 <=  qP11_d7;
               qP11_d9 <=  qP11_d8;
               qP11_d10 <=  qP11_d9;
               qP11_d11 <=  qP11_d10;
               qP11_d12 <=  qP11_d11;
               qP11_d13 <=  qP11_d12;
               qM11_d1 <=  qM11;
               qM11_d2 <=  qM11_d1;
               qM11_d3 <=  qM11_d2;
               qM11_d4 <=  qM11_d3;
               qM11_d5 <=  qM11_d4;
               qM11_d6 <=  qM11_d5;
               qM11_d7 <=  qM11_d6;
               qM11_d8 <=  qM11_d7;
               qM11_d9 <=  qM11_d8;
               qM11_d10 <=  qM11_d9;
               qM11_d11 <=  qM11_d10;
               qM11_d12 <=  qM11_d11;
               qM11_d13 <=  qM11_d12;
               qM11_d14 <=  qM11_d13;
               qP10_d1 <=  qP10;
               qP10_d2 <=  qP10_d1;
               qP10_d3 <=  qP10_d2;
               qP10_d4 <=  qP10_d3;
               qP10_d5 <=  qP10_d4;
               qP10_d6 <=  qP10_d5;
               qP10_d7 <=  qP10_d6;
               qP10_d8 <=  qP10_d7;
               qP10_d9 <=  qP10_d8;
               qP10_d10 <=  qP10_d9;
               qP10_d11 <=  qP10_d10;
               qP10_d12 <=  qP10_d11;
               qM10_d1 <=  qM10;
               qM10_d2 <=  qM10_d1;
               qM10_d3 <=  qM10_d2;
               qM10_d4 <=  qM10_d3;
               qM10_d5 <=  qM10_d4;
               qM10_d6 <=  qM10_d5;
               qM10_d7 <=  qM10_d6;
               qM10_d8 <=  qM10_d7;
               qM10_d9 <=  qM10_d8;
               qM10_d10 <=  qM10_d9;
               qM10_d11 <=  qM10_d10;
               qM10_d12 <=  qM10_d11;
               qM10_d13 <=  qM10_d12;
               qP9_d1 <=  qP9;
               qP9_d2 <=  qP9_d1;
               qP9_d3 <=  qP9_d2;
               qP9_d4 <=  qP9_d3;
               qP9_d5 <=  qP9_d4;
               qP9_d6 <=  qP9_d5;
               qP9_d7 <=  qP9_d6;
               qP9_d8 <=  qP9_d7;
               qP9_d9 <=  qP9_d8;
               qP9_d10 <=  qP9_d9;
               qM9_d1 <=  qM9;
               qM9_d2 <=  qM9_d1;
               qM9_d3 <=  qM9_d2;
               qM9_d4 <=  qM9_d3;
               qM9_d5 <=  qM9_d4;
               qM9_d6 <=  qM9_d5;
               qM9_d7 <=  qM9_d6;
               qM9_d8 <=  qM9_d7;
               qM9_d9 <=  qM9_d8;
               qM9_d10 <=  qM9_d9;
               qM9_d11 <=  qM9_d10;
               qP8_d1 <=  qP8;
               qP8_d2 <=  qP8_d1;
               qP8_d3 <=  qP8_d2;
               qP8_d4 <=  qP8_d3;
               qP8_d5 <=  qP8_d4;
               qP8_d6 <=  qP8_d5;
               qP8_d7 <=  qP8_d6;
               qP8_d8 <=  qP8_d7;
               qP8_d9 <=  qP8_d8;
               qM8_d1 <=  qM8;
               qM8_d2 <=  qM8_d1;
               qM8_d3 <=  qM8_d2;
               qM8_d4 <=  qM8_d3;
               qM8_d5 <=  qM8_d4;
               qM8_d6 <=  qM8_d5;
               qM8_d7 <=  qM8_d6;
               qM8_d8 <=  qM8_d7;
               qM8_d9 <=  qM8_d8;
               qM8_d10 <=  qM8_d9;
               qP7_d1 <=  qP7;
               qP7_d2 <=  qP7_d1;
               qP7_d3 <=  qP7_d2;
               qP7_d4 <=  qP7_d3;
               qP7_d5 <=  qP7_d4;
               qP7_d6 <=  qP7_d5;
               qP7_d7 <=  qP7_d6;
               qP7_d8 <=  qP7_d7;
               qM7_d1 <=  qM7;
               qM7_d2 <=  qM7_d1;
               qM7_d3 <=  qM7_d2;
               qM7_d4 <=  qM7_d3;
               qM7_d5 <=  qM7_d4;
               qM7_d6 <=  qM7_d5;
               qM7_d7 <=  qM7_d6;
               qM7_d8 <=  qM7_d7;
               qM7_d9 <=  qM7_d8;
               qP6_d1 <=  qP6;
               qP6_d2 <=  qP6_d1;
               qP6_d3 <=  qP6_d2;
               qP6_d4 <=  qP6_d3;
               qP6_d5 <=  qP6_d4;
               qP6_d6 <=  qP6_d5;
               qM6_d1 <=  qM6;
               qM6_d2 <=  qM6_d1;
               qM6_d3 <=  qM6_d2;
               qM6_d4 <=  qM6_d3;
               qM6_d5 <=  qM6_d4;
               qM6_d6 <=  qM6_d5;
               qM6_d7 <=  qM6_d6;
               qP5_d1 <=  qP5;
               qP5_d2 <=  qP5_d1;
               qP5_d3 <=  qP5_d2;
               qP5_d4 <=  qP5_d3;
               qP5_d5 <=  qP5_d4;
               qM5_d1 <=  qM5;
               qM5_d2 <=  qM5_d1;
               qM5_d3 <=  qM5_d2;
               qM5_d4 <=  qM5_d3;
               qM5_d5 <=  qM5_d4;
               qM5_d6 <=  qM5_d5;
               qP4_d1 <=  qP4;
               qP4_d2 <=  qP4_d1;
               qP4_d3 <=  qP4_d2;
               qP4_d4 <=  qP4_d3;
               qM4_d1 <=  qM4;
               qM4_d2 <=  qM4_d1;
               qM4_d3 <=  qM4_d2;
               qM4_d4 <=  qM4_d3;
               qM4_d5 <=  qM4_d4;
               qP3_d1 <=  qP3;
               qP3_d2 <=  qP3_d1;
               qM3_d1 <=  qM3;
               qM3_d2 <=  qM3_d1;
               qM3_d3 <=  qM3_d2;
               qP2_d1 <=  qP2;
               qM2_d1 <=  qM2;
               qM2_d2 <=  qM2_d1;
               qM1_d1 <=  qM1;
               qP_d1 <=  qP;
               qP_d2 <=  qP_d1;
               qM_d1 <=  qM;
               mR_d1 <=  mR;
               fRnorm_d1 <=  fRnorm;
               round_d1 <=  round;
            end if;
         end if;
      end process;
   fX <= "1" & X(22 downto 0);
   fY <= "1" & Y(22 downto 0);
   -- exponent difference, sign and exception combination computed early, to have fewer bits to pipeline
   expR0 <= ("00" & X(30 downto 23)) - ("00" & Y(30 downto 23));
   sR <= X(31) xor Y(31);
   -- early exception handling 
   exnXY <= X(33 downto 32) & Y(33 downto 32);
   with exnXY  select 
      exnR0 <= 
         "01"	 when "0101",										-- normal
         "00"	 when "0001" | "0010" | "0110", -- zero
         "10"	 when "0100" | "1000" | "1001", -- overflow
         "11"	 when others;										-- NaN
   D <= fY ;
   psX <= "0" & fX ;
   betaw14 <=  "00" & psX;
   sel14 <= betaw14(26 downto 21) & D(22 downto 20);
   SelFunctionTable14: selFunction_Freq630_uid4
      port map ( X => sel14,
                 Y => q14_copy5);
   q14 <= q14_copy5; -- output copy to hold a pipeline register if needed

   with q14  select 
      absq14D <= 
         "000" & D						 when "001" | "111", -- mult by 1
         "00" & D & "0"			   when "010" | "110", -- mult by 2
         (26 downto 0 => '0')	 when others;        -- mult by 0

   with q14_d1(2)  select 
   w13<= betaw14_d1 - absq14D_d1 when '0',
         betaw14_d1 + absq14D_d1 when others;

   betaw13 <= w13(24 downto 0) & "00"; -- multiplication by the radix
   sel13 <= betaw13(26 downto 21) & D_d1(22 downto 20);
   SelFunctionTable13: selFunction_Freq630_uid4
      port map ( X => sel13,
                 Y => q13_copy6);
   q13 <= q13_copy6; -- output copy to hold a pipeline register if needed

   with q13  select 
      absq13D <= 
         "000" & D_d1						 when "001" | "111", -- mult by 1
         "00" & D_d1 & "0"			   when "010" | "110", -- mult by 2
         (26 downto 0 => '0')	 when others;        -- mult by 0

   with q13_d1(2)  select 
   w12<= betaw13_d1 - absq13D_d1 when '0',
         betaw13_d1 + absq13D_d1 when others;

   betaw12 <= w12(24 downto 0) & "00"; -- multiplication by the radix
   sel12 <= betaw12(26 downto 21) & D_d2(22 downto 20);
   SelFunctionTable12: selFunction_Freq630_uid4
      port map ( X => sel12,
                 Y => q12_copy7);
   q12 <= q12_copy7_d1; -- output copy to hold a pipeline register if needed

   with q12  select 
      absq12D <= 
         "000" & D_d3						 when "001" | "111", -- mult by 1
         "00" & D_d3 & "0"			   when "010" | "110", -- mult by 2
         (26 downto 0 => '0')	 when others;        -- mult by 0

   with q12_d1(2)  select 
   w11<= betaw12_d2 - absq12D_d1 when '0',
         betaw12_d2 + absq12D_d1 when others;

   betaw11 <= w11(24 downto 0) & "00"; -- multiplication by the radix
   sel11 <= betaw11(26 downto 21) & D_d4(22 downto 20);
   SelFunctionTable11: selFunction_Freq630_uid4
      port map ( X => sel11,
                 Y => q11_copy8);
   q11 <= q11_copy8; -- output copy to hold a pipeline register if needed

   with q11  select 
      absq11D <= 
         "000" & D_d4						 when "001" | "111", -- mult by 1
         "00" & D_d4 & "0"			   when "010" | "110", -- mult by 2
         (26 downto 0 => '0')	 when others;        -- mult by 0

   with q11_d1(2)  select 
   w10<= betaw11_d1 - absq11D_d1 when '0',
         betaw11_d1 + absq11D_d1 when others;

   betaw10 <= w10(24 downto 0) & "00"; -- multiplication by the radix
   sel10 <= betaw10(26 downto 21) & D_d5(22 downto 20);
   SelFunctionTable10: selFunction_Freq630_uid4
      port map ( X => sel10,
                 Y => q10_copy9);
   q10 <= q10_copy9; -- output copy to hold a pipeline register if needed

   with q10  select 
      absq10D <= 
         "000" & D_d5						 when "001" | "111", -- mult by 1
         "00" & D_d5 & "0"			   when "010" | "110", -- mult by 2
         (26 downto 0 => '0')	 when others;        -- mult by 0

   with q10_d1(2)  select 
   w9<= betaw10_d1 - absq10D_d1 when '0',
         betaw10_d1 + absq10D_d1 when others;

   betaw9 <= w9(24 downto 0) & "00"; -- multiplication by the radix
   sel9 <= betaw9(26 downto 21) & D_d6(22 downto 20);
   SelFunctionTable9: selFunction_Freq630_uid4
      port map ( X => sel9,
                 Y => q9_copy10);
   q9 <= q9_copy10_d1; -- output copy to hold a pipeline register if needed

   with q9  select 
      absq9D <= 
         "000" & D_d7						 when "001" | "111", -- mult by 1
         "00" & D_d7 & "0"			   when "010" | "110", -- mult by 2
         (26 downto 0 => '0')	 when others;        -- mult by 0

   with q9_d1(2)  select 
   w8<= betaw9_d2 - absq9D_d1 when '0',
         betaw9_d2 + absq9D_d1 when others;

   betaw8 <= w8(24 downto 0) & "00"; -- multiplication by the radix
   sel8 <= betaw8(26 downto 21) & D_d8(22 downto 20);
   SelFunctionTable8: selFunction_Freq630_uid4
      port map ( X => sel8,
                 Y => q8_copy11);
   q8 <= q8_copy11; -- output copy to hold a pipeline register if needed

   with q8  select 
      absq8D <= 
         "000" & D_d8						 when "001" | "111", -- mult by 1
         "00" & D_d8 & "0"			   when "010" | "110", -- mult by 2
         (26 downto 0 => '0')	 when others;        -- mult by 0

   with q8_d1(2)  select 
   w7<= betaw8_d1 - absq8D_d1 when '0',
         betaw8_d1 + absq8D_d1 when others;

   betaw7 <= w7(24 downto 0) & "00"; -- multiplication by the radix
   sel7 <= betaw7(26 downto 21) & D_d9(22 downto 20);
   SelFunctionTable7: selFunction_Freq630_uid4
      port map ( X => sel7,
                 Y => q7_copy12);
   q7 <= q7_copy12; -- output copy to hold a pipeline register if needed

   with q7  select 
      absq7D <= 
         "000" & D_d9						 when "001" | "111", -- mult by 1
         "00" & D_d9 & "0"			   when "010" | "110", -- mult by 2
         (26 downto 0 => '0')	 when others;        -- mult by 0

   with q7_d1(2)  select 
   w6<= betaw7_d1 - absq7D_d1 when '0',
         betaw7_d1 + absq7D_d1 when others;

   betaw6 <= w6(24 downto 0) & "00"; -- multiplication by the radix
   sel6 <= betaw6(26 downto 21) & D_d10(22 downto 20);
   SelFunctionTable6: selFunction_Freq630_uid4
      port map ( X => sel6,
                 Y => q6_copy13);
   q6 <= q6_copy13_d1; -- output copy to hold a pipeline register if needed

   with q6  select 
      absq6D <= 
         "000" & D_d11						 when "001" | "111", -- mult by 1
         "00" & D_d11 & "0"			   when "010" | "110", -- mult by 2
         (26 downto 0 => '0')	 when others;        -- mult by 0

   with q6_d1(2)  select 
   w5<= betaw6_d2 - absq6D_d1 when '0',
         betaw6_d2 + absq6D_d1 when others;

   betaw5 <= w5(24 downto 0) & "00"; -- multiplication by the radix
   sel5 <= betaw5(26 downto 21) & D_d12(22 downto 20);
   SelFunctionTable5: selFunction_Freq630_uid4
      port map ( X => sel5,
                 Y => q5_copy14);
   q5 <= q5_copy14; -- output copy to hold a pipeline register if needed

   with q5  select 
      absq5D <= 
         "000" & D_d12						 when "001" | "111", -- mult by 1
         "00" & D_d12 & "0"			   when "010" | "110", -- mult by 2
         (26 downto 0 => '0')	 when others;        -- mult by 0

   with q5_d1(2)  select 
   w4<= betaw5_d1 - absq5D_d1 when '0',
         betaw5_d1 + absq5D_d1 when others;

   betaw4 <= w4(24 downto 0) & "00"; -- multiplication by the radix
   sel4 <= betaw4(26 downto 21) & D_d13(22 downto 20);
   SelFunctionTable4: selFunction_Freq630_uid4
      port map ( X => sel4,
                 Y => q4_copy15);
   q4 <= q4_copy15; -- output copy to hold a pipeline register if needed

   with q4  select 
      absq4D <= 
         "000" & D_d13						 when "001" | "111", -- mult by 1
         "00" & D_d13 & "0"			   when "010" | "110", -- mult by 2
         (26 downto 0 => '0')	 when others;        -- mult by 0

   with q4_d1(2)  select 
   w3<= betaw4_d1 - absq4D_d1 when '0',
         betaw4_d1 + absq4D_d1 when others;

   betaw3 <= w3(24 downto 0) & "00"; -- multiplication by the radix
   sel3 <= betaw3(26 downto 21) & D_d14(22 downto 20);
   SelFunctionTable3: selFunction_Freq630_uid4
      port map ( X => sel3,
                 Y => q3_copy16);
   q3 <= q3_copy16_d1; -- output copy to hold a pipeline register if needed

   with q3  select 
      absq3D <= 
         "000" & D_d15						 when "001" | "111", -- mult by 1
         "00" & D_d15 & "0"			   when "010" | "110", -- mult by 2
         (26 downto 0 => '0')	 when others;        -- mult by 0

   with q3_d1(2)  select 
   w2<= betaw3_d2 - absq3D_d1 when '0',
         betaw3_d2 + absq3D_d1 when others;

   betaw2 <= w2(24 downto 0) & "00"; -- multiplication by the radix
   sel2 <= betaw2(26 downto 21) & D_d16(22 downto 20);
   SelFunctionTable2: selFunction_Freq630_uid4
      port map ( X => sel2,
                 Y => q2_copy17);
   q2 <= q2_copy17; -- output copy to hold a pipeline register if needed

   with q2  select 
      absq2D <= 
         "000" & D_d16						 when "001" | "111", -- mult by 1
         "00" & D_d16 & "0"			   when "010" | "110", -- mult by 2
         (26 downto 0 => '0')	 when others;        -- mult by 0

   with q2_d1(2)  select 
   w1<= betaw2_d1 - absq2D_d1 when '0',
         betaw2_d1 + absq2D_d1 when others;

   betaw1 <= w1(24 downto 0) & "00"; -- multiplication by the radix
   sel1 <= betaw1(26 downto 21) & D_d17(22 downto 20);
   SelFunctionTable1: selFunction_Freq630_uid4
      port map ( X => sel1,
                 Y => q1_copy18);
   q1 <= q1_copy18; -- output copy to hold a pipeline register if needed

   with q1  select 
      absq1D <= 
         "000" & D_d17						 when "001" | "111", -- mult by 1
         "00" & D_d17 & "0"			   when "010" | "110", -- mult by 2
         (26 downto 0 => '0')	 when others;        -- mult by 0

   with q1_d1(2)  select 
   w0<= betaw1_d1 - absq1D_d1 when '0',
         betaw1_d1 + absq1D_d1 when others;

   wfinal <= w0(24 downto 0);
   qM0 <= wfinal(24); -- rounding bit is the sign of the remainder
   qP14 <=      q14(1 downto 0);
   qM14 <=      q14(2) & "0";
   qP13 <=      q13(1 downto 0);
   qM13 <=      q13(2) & "0";
   qP12 <=      q12(1 downto 0);
   qM12 <=      q12(2) & "0";
   qP11 <=      q11(1 downto 0);
   qM11 <=      q11(2) & "0";
   qP10 <=      q10(1 downto 0);
   qM10 <=      q10(2) & "0";
   qP9 <=      q9(1 downto 0);
   qM9 <=      q9(2) & "0";
   qP8 <=      q8(1 downto 0);
   qM8 <=      q8(2) & "0";
   qP7 <=      q7(1 downto 0);
   qM7 <=      q7(2) & "0";
   qP6 <=      q6(1 downto 0);
   qM6 <=      q6(2) & "0";
   qP5 <=      q5(1 downto 0);
   qM5 <=      q5(2) & "0";
   qP4 <=      q4(1 downto 0);
   qM4 <=      q4(2) & "0";
   qP3 <=      q3(1 downto 0);
   qM3 <=      q3(2) & "0";
   qP2 <=      q2(1 downto 0);
   qM2 <=      q2(2) & "0";
   qP1 <=      q1(1 downto 0);
   qM1 <=      q1(2) & "0";
   qP <= qP14_d17 & qP13_d16 & qP12_d14 & qP11_d13 & qP10_d12 & qP9_d10 & qP8_d9 & qP7_d8 & qP6_d6 & qP5_d5 & qP4_d4 & qP3_d2 & qP2_d1 & qP1;
   qM <= qM14_d18(0) & qM13_d17 & qM12_d15 & qM11_d14 & qM10_d13 & qM9_d11 & qM8_d10 & qM7_d9 & qM6_d7 & qM5_d6 & qM4_d5 & qM3_d3 & qM2_d2 & qM1_d1 & qM0;
   quotient <= qP_d2 - qM_d1;
   -- We need a mR in (0, -wf-2) format: 1+wF fraction bits, 1 round bit, and 1 guard bit for the normalisation,
   -- quotient is the truncation of the exact quotient to at least 2^(-wF-2) bits
   -- now discarding its possible known MSB zeroes, and dropping the possible extra LSB bit (due to radix 4) 
   mR <= quotient(26 downto 1); 
   -- normalisation
   fRnorm <=    mR(24 downto 1)  when mR(25)= '1'
           else mR(23 downto 0);  -- now fRnorm is a (-1, -wF-1) fraction
   round <= fRnorm(0); 
   expR1 <= expR0_d20 + ("000" & (6 downto 1 => '1') & mR_d1(25)); -- add back bias
   -- final rounding
   expfrac <= expR1 & fRnorm_d1(23 downto 1) ;
   expfracR <= expfrac + ((32 downto 1 => '0') & round_d1);
   exnR <=      "00"  when expfracR(32) = '1'   -- underflow
           else "10"  when  expfracR(32 downto 31) =  "01" -- overflow
           else "01";      -- 00, normal case
   with exnR0_d20  select 
      exnRfinal <= 
         exnR   when "01", -- normal
         exnR0_d20  when others;
   R <= exnRfinal & sR_d20 & expfracR(30 downto 0);
end architecture;



--------------------------------------------------------------------------------
--                RightShifterSticky24_by_max_26_Freq450_uid4
-- VHDL generated for Kintex7 @ 450MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca (2008-2011), Florent de Dinechin (2008-2019)
--------------------------------------------------------------------------------
-- Pipeline depth: 3 cycles
-- Clock period (ns): 2.22222
-- Target frequency (MHz): 450
-- Input signals: X S
-- Output signals: R Sticky

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity RightShifterSticky24_by_max_26_Freq450_uid4 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(23 downto 0);
          S : in  std_logic_vector(4 downto 0);
          R : out  std_logic_vector(25 downto 0);
          Sticky : out  std_logic   );
end entity;

architecture arch of RightShifterSticky24_by_max_26_Freq450_uid4 is
signal ps, ps_d1, ps_d2, ps_d3 :  std_logic_vector(4 downto 0);
signal Xpadded :  std_logic_vector(25 downto 0);
signal level5 :  std_logic_vector(25 downto 0);
signal stk4, stk4_d1 :  std_logic;
signal level4, level4_d1 :  std_logic_vector(25 downto 0);
signal stk3, stk3_d1 :  std_logic;
signal level3, level3_d1, level3_d2 :  std_logic_vector(25 downto 0);
signal stk2 :  std_logic;
signal level2, level2_d1, level2_d2 :  std_logic_vector(25 downto 0);
signal stk1, stk1_d1 :  std_logic;
signal level1, level1_d1, level1_d2, level1_d3 :  std_logic_vector(25 downto 0);
signal stk0 :  std_logic;
signal level0 :  std_logic_vector(25 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               ps_d1 <=  ps;
               ps_d2 <=  ps_d1;
               ps_d3 <=  ps_d2;
               stk4_d1 <=  stk4;
               level4_d1 <=  level4;
               stk3_d1 <=  stk3;
               level3_d1 <=  level3;
               level3_d2 <=  level3_d1;
               level2_d1 <=  level2;
               level2_d2 <=  level2_d1;
               stk1_d1 <=  stk1;
               level1_d1 <=  level1;
               level1_d2 <=  level1_d1;
               level1_d3 <=  level1_d2;
            end if;
         end if;
      end process;
   ps<= S;
   Xpadded <= X&(1 downto 0 => '0');
   level5<= Xpadded;
   stk4 <= '1' when (level5(15 downto 0)/="0000000000000000" and ps(4)='1')   else '0';
   level4 <=  level5 when  ps(4)='0'    else (15 downto 0 => '0') & level5(25 downto 16);
   stk3 <= '1' when (level4_d1(7 downto 0)/="00000000" and ps_d1(3)='1') or stk4_d1 ='1'   else '0';
   level3 <=  level4 when  ps(3)='0'    else (7 downto 0 => '0') & level4(25 downto 8);
   stk2 <= '1' when (level3_d2(3 downto 0)/="0000" and ps_d2(2)='1') or stk3_d1 ='1'   else '0';
   level2 <=  level3 when  ps(2)='0'    else (3 downto 0 => '0') & level3(25 downto 4);
   stk1 <= '1' when (level2_d2(1 downto 0)/="00" and ps_d2(1)='1') or stk2 ='1'   else '0';
   level1 <=  level2 when  ps(1)='0'    else (1 downto 0 => '0') & level2(25 downto 2);
   stk0 <= '1' when (level1_d3(0 downto 0)/="0" and ps_d3(0)='1') or stk1_d1 ='1'   else '0';
   level0 <=  level1 when  ps(0)='0'    else (0 downto 0 => '0') & level1(25 downto 1);
   R <= level0;
   Sticky <= stk0;
end architecture;

--------------------------------------------------------------------------------
--                          IntAdder_27_Freq450_uid6
-- VHDL generated for Kintex7 @ 450MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2016)
--------------------------------------------------------------------------------
-- Pipeline depth: 1 cycles
-- Clock period (ns): 2.22222
-- Target frequency (MHz): 450
-- Input signals: X Y Cin
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntAdder_27_Freq450_uid6 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(26 downto 0);
          Y : in  std_logic_vector(26 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(26 downto 0)   );
end entity;

architecture arch of IntAdder_27_Freq450_uid6 is
signal Cin_0, Cin_0_d1 :  std_logic;
signal X_0, X_0_d1, X_0_d2, X_0_d3, X_0_d4, X_0_d5 :  std_logic_vector(11 downto 0);
signal Y_0, Y_0_d1, Y_0_d2, Y_0_d3 :  std_logic_vector(11 downto 0);
signal S_0 :  std_logic_vector(11 downto 0);
signal R_0 :  std_logic_vector(10 downto 0);
signal Cin_1 :  std_logic;
signal X_1, X_1_d1, X_1_d2, X_1_d3, X_1_d4, X_1_d5 :  std_logic_vector(16 downto 0);
signal Y_1, Y_1_d1, Y_1_d2, Y_1_d3 :  std_logic_vector(16 downto 0);
signal S_1 :  std_logic_vector(16 downto 0);
signal R_1 :  std_logic_vector(15 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               Cin_0_d1 <=  Cin_0;
               X_0_d1 <=  X_0;
               X_0_d2 <=  X_0_d1;
               X_0_d3 <=  X_0_d2;
               X_0_d4 <=  X_0_d3;
               X_0_d5 <=  X_0_d4;
               Y_0_d1 <=  Y_0;
               Y_0_d2 <=  Y_0_d1;
               Y_0_d3 <=  Y_0_d2;
               X_1_d1 <=  X_1;
               X_1_d2 <=  X_1_d1;
               X_1_d3 <=  X_1_d2;
               X_1_d4 <=  X_1_d3;
               X_1_d5 <=  X_1_d4;
               Y_1_d1 <=  Y_1;
               Y_1_d2 <=  Y_1_d1;
               Y_1_d3 <=  Y_1_d2;
            end if;
         end if;
      end process;
   Cin_0 <= Cin;
   X_0 <= '0' & X(10 downto 0);
   Y_0 <= '0' & Y(10 downto 0);
   S_0 <= X_0_d5 + Y_0_d3 + Cin_0_d1;
   R_0 <= S_0(10 downto 0);
   Cin_1 <= S_0(11);
   X_1 <= '0' & X(26 downto 11);
   Y_1 <= '0' & Y(26 downto 11);
   S_1 <= X_1_d5 + Y_1_d3 + Cin_1;
   R_1 <= S_1(15 downto 0);
   R <= R_1 & R_0 ;
end architecture;

--------------------------------------------------------------------------------
--                     Normalizer_Z_28_28_28_Freq450_uid8
-- VHDL generated for Kintex7 @ 450MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin, (2007-2020)
--------------------------------------------------------------------------------
-- Pipeline depth: 3 cycles
-- Clock period (ns): 2.22222
-- Target frequency (MHz): 450
-- Input signals: X
-- Output signals: Count R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity Normalizer_Z_28_28_28_Freq450_uid8 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(27 downto 0);
          Count : out  std_logic_vector(4 downto 0);
          R : out  std_logic_vector(27 downto 0)   );
end entity;

architecture arch of Normalizer_Z_28_28_28_Freq450_uid8 is
signal level5, level5_d1 :  std_logic_vector(27 downto 0);
signal count4, count4_d1, count4_d2, count4_d3 :  std_logic;
signal level4 :  std_logic_vector(27 downto 0);
signal count3, count3_d1, count3_d2 :  std_logic;
signal level3, level3_d1 :  std_logic_vector(27 downto 0);
signal count2, count2_d1 :  std_logic;
signal level2 :  std_logic_vector(27 downto 0);
signal count1, count1_d1 :  std_logic;
signal level1, level1_d1 :  std_logic_vector(27 downto 0);
signal count0 :  std_logic;
signal level0 :  std_logic_vector(27 downto 0);
signal sCount :  std_logic_vector(4 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               level5_d1 <=  level5;
               count4_d1 <=  count4;
               count4_d2 <=  count4_d1;
               count4_d3 <=  count4_d2;
               count3_d1 <=  count3;
               count3_d2 <=  count3_d1;
               level3_d1 <=  level3;
               count2_d1 <=  count2;
               count1_d1 <=  count1;
               level1_d1 <=  level1;
            end if;
         end if;
      end process;
   level5 <= X ;
   count4<= '1' when level5(27 downto 12) = (27 downto 12=>'0') else '0';
   level4<= level5_d1(27 downto 0) when count4_d1='0' else level5_d1(11 downto 0) & (15 downto 0 => '0');

   count3<= '1' when level4(27 downto 20) = (27 downto 20=>'0') else '0';
   level3<= level4(27 downto 0) when count3='0' else level4(19 downto 0) & (7 downto 0 => '0');

   count2<= '1' when level3_d1(27 downto 24) = (27 downto 24=>'0') else '0';
   level2<= level3_d1(27 downto 0) when count2='0' else level3_d1(23 downto 0) & (3 downto 0 => '0');

   count1<= '1' when level2(27 downto 26) = (27 downto 26=>'0') else '0';
   level1<= level2(27 downto 0) when count1='0' else level2(25 downto 0) & (1 downto 0 => '0');

   count0<= '1' when level1_d1(27 downto 27) = (27 downto 27=>'0') else '0';
   level0<= level1_d1(27 downto 0) when count0='0' else level1_d1(26 downto 0) & (0 downto 0 => '0');

   R <= level0;
   sCount <= count4_d3 & count3_d2 & count2_d1 & count1_d1 & count0;
   Count <= sCount;
end architecture;

--------------------------------------------------------------------------------
--                         IntAdder_34_Freq450_uid11
-- VHDL generated for Kintex7 @ 450MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2016)
--------------------------------------------------------------------------------
-- Pipeline depth: 1 cycles
-- Clock period (ns): 2.22222
-- Target frequency (MHz): 450
-- Input signals: X Y Cin
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntAdder_34_Freq450_uid11 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(33 downto 0);
          Y : in  std_logic_vector(33 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(33 downto 0)   );
end entity;

architecture arch of IntAdder_34_Freq450_uid11 is
signal Cin_1, Cin_1_d1 :  std_logic;
signal X_1, X_1_d1 :  std_logic_vector(34 downto 0);
signal Y_1, Y_1_d1, Y_1_d2, Y_1_d3, Y_1_d4, Y_1_d5, Y_1_d6, Y_1_d7, Y_1_d8, Y_1_d9 :  std_logic_vector(34 downto 0);
signal S_1 :  std_logic_vector(34 downto 0);
signal R_1 :  std_logic_vector(33 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               Cin_1_d1 <=  Cin_1;
               X_1_d1 <=  X_1;
               Y_1_d1 <=  Y_1;
               Y_1_d2 <=  Y_1_d1;
               Y_1_d3 <=  Y_1_d2;
               Y_1_d4 <=  Y_1_d3;
               Y_1_d5 <=  Y_1_d4;
               Y_1_d6 <=  Y_1_d5;
               Y_1_d7 <=  Y_1_d6;
               Y_1_d8 <=  Y_1_d7;
               Y_1_d9 <=  Y_1_d8;
            end if;
         end if;
      end process;
   Cin_1 <= Cin;
   X_1 <= '0' & X(33 downto 0);
   Y_1 <= '0' & Y(33 downto 0);
   S_1 <= X_1_d1 + Y_1_d9 + Cin_1_d1;
   R_1 <= S_1(33 downto 0);
   R <= R_1 ;
end architecture;

--------------------------------------------------------------------------------
--                     IntComparator_31_010_Freq500_uid4
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2021)
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: XeqY

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntComparator_31_010_Freq500_uid4 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(30 downto 0);
          Y : in  std_logic_vector(30 downto 0);
          XeqY : out  std_logic   );
end entity;

architecture arch of IntComparator_31_010_Freq500_uid4 is
signal XeqYi :  std_logic;
begin
   XeqYi <= '1' when X=Y else '0';
   XeqY <= XeqYi;
end architecture;

--------------------------------------------------------------------------------
--                         FloatingPointComparatorEQ
--                      (FPComparator_8_23_Freq500_uid2)
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2021)
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: unordered XeqY

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity FloatingPointComparatorEQ is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(8+23+2 downto 0);
          Y : in  std_logic_vector(8+23+2 downto 0);
          unordered : out  std_logic;
          XeqY : out  std_logic   );
end entity;

architecture arch of FloatingPointComparatorEQ is
   component IntComparator_31_010_Freq500_uid4 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(30 downto 0);
             Y : in  std_logic_vector(30 downto 0);
             XeqY : out  std_logic   );
   end component;

signal excX :  std_logic_vector(1 downto 0);
signal excY :  std_logic_vector(1 downto 0);
signal signX :  std_logic;
signal signY :  std_logic;
signal ExpFracX :  std_logic_vector(30 downto 0);
signal ExpFracY :  std_logic_vector(30 downto 0);
signal isZeroX :  std_logic;
signal isZeroY :  std_logic;
signal isNormalX :  std_logic;
signal isNormalY :  std_logic;
signal isInfX :  std_logic;
signal isInfY :  std_logic;
signal isNaNX :  std_logic;
signal isNaNY :  std_logic;
signal negativeX :  std_logic;
signal positiveX :  std_logic;
signal negativeY :  std_logic;
signal positiveY :  std_logic;
signal ExpFracXeqExpFracY :  std_logic;
signal sameSign :  std_logic;
signal XeqYNum :  std_logic;
signal unorderedR :  std_logic;
signal XeqYR :  std_logic;
begin
   excX <= X(33 downto 32);
   excY <= Y(33 downto 32);
   signX <= X(31);
   signY <= Y(31);
   ExpFracX <= X(30 downto 0);
   ExpFracY <= Y(30 downto 0);
   -- Comparing (as integers) excX & ExpFracX with excY & ExpFracY would almost work 
   --  since indeed inf>normal>0	
   -- However we wouldn't capture infinity equality in cases when the infinities have different ExpFracs (who knows)...	 
   -- Besides, expliciting the isXXX bits will help factoring code with a comparator for IEEE format (some day)
   isZeroX <= '1' when excX="00" else '0' ;
   isZeroY <= '1' when excY="00" else '0' ;
   isNormalX <= '1' when excX="01" else '0' ;
   isNormalY <= '1' when excY="01" else '0' ;
   isInfX <= '1' when excX="10" else '0' ;
   isInfY <= '1' when excY="10" else '0' ;
   isNaNX <= '1' when excX="11" else '0' ;
   isNaNY <= '1' when excY="11" else '0' ;
   -- Just for readability of the formulae below
   negativeX <= signX ;
   positiveX <= not signX ;
   negativeY <= signY ;
   positiveY <= not signY ;
   -- expfrac comparisons 
   ExpFracCmp: IntComparator_31_010_Freq500_uid4
      port map ( clk  => clk,
                 ce => ce,
                 X => ExpFracX,
                 Y => ExpFracY,
                 XeqY => ExpFracXeqExpFracY);
   -- -- and now the logic
   sameSign <= not (signX xor signY) ;
   XeqYNum <= 
         (isZeroX and isZeroY) -- explicitely stated by IEEE 754
      or (isInfX and isInfY and sameSign)  -- bizarre but also explicitely stated by IEEE 754
      or (isNormalX and isNormalY and sameSign and ExpFracXeqExpFracY)   ;
   unorderedR <=  isNaNX or isNaNY;
   XeqYR <= XeqYNum and not unorderedR;
   unordered <= unorderedR;
   XeqY <= XeqYR;
end architecture;

--------------------------------------------------------------------------------
--                     IntComparator_31_111_Freq500_uid4
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2021)
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: XltY XeqY XgtY

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntComparator_31_111_Freq500_uid4 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(30 downto 0);
          Y : in  std_logic_vector(30 downto 0);
          XltY : out  std_logic;
          XeqY : out  std_logic;
          XgtY : out  std_logic   );
end entity;

architecture arch of IntComparator_31_111_Freq500_uid4 is
signal XltYi :  std_logic;
signal XeqYi :  std_logic;
signal XgtYi :  std_logic;
begin
   XltYi <= '1' when X<Y else '0';
   XeqYi <= '1' when X=Y else '0';
   XgtYi <= not (XeqYi or XltYi);
   XltY <= XltYi;
   XeqY <= XeqYi;
   XgtY <= XgtYi;
end architecture;

--------------------------------------------------------------------------------
--                         FloatingPointComparatorGE
--                      (FPComparator_8_23_Freq500_uid2)
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2021)
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: unordered XgeY

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity FloatingPointComparatorGE is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(8+23+2 downto 0);
          Y : in  std_logic_vector(8+23+2 downto 0);
          unordered : out  std_logic;
          XgeY : out  std_logic   );
end entity;

architecture arch of FloatingPointComparatorGE is
   component IntComparator_31_111_Freq500_uid4 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(30 downto 0);
             Y : in  std_logic_vector(30 downto 0);
             XltY : out  std_logic;
             XeqY : out  std_logic;
             XgtY : out  std_logic   );
   end component;

signal excX :  std_logic_vector(1 downto 0);
signal excY :  std_logic_vector(1 downto 0);
signal signX :  std_logic;
signal signY :  std_logic;
signal ExpFracX :  std_logic_vector(30 downto 0);
signal ExpFracY :  std_logic_vector(30 downto 0);
signal isZeroX :  std_logic;
signal isZeroY :  std_logic;
signal isNormalX :  std_logic;
signal isNormalY :  std_logic;
signal isInfX :  std_logic;
signal isInfY :  std_logic;
signal isNaNX :  std_logic;
signal isNaNY :  std_logic;
signal negativeX :  std_logic;
signal positiveX :  std_logic;
signal negativeY :  std_logic;
signal positiveY :  std_logic;
signal ExpFracXeqExpFracY :  std_logic;
signal ExpFracXltExpFracY :  std_logic;
signal ExpFracXgtExpFracY :  std_logic;
signal sameSign :  std_logic;
signal XeqYNum :  std_logic;
signal XgtYNum :  std_logic;
signal unorderedR :  std_logic;
signal XgeYR :  std_logic;
begin
   excX <= X(33 downto 32);
   excY <= Y(33 downto 32);
   signX <= X(31);
   signY <= Y(31);
   ExpFracX <= X(30 downto 0);
   ExpFracY <= Y(30 downto 0);
   -- Comparing (as integers) excX & ExpFracX with excY & ExpFracY would almost work 
   --  since indeed inf>normal>0	
   -- However we wouldn't capture infinity equality in cases when the infinities have different ExpFracs (who knows)...	 
   -- Besides, expliciting the isXXX bits will help factoring code with a comparator for IEEE format (some day)
   isZeroX <= '1' when excX="00" else '0' ;
   isZeroY <= '1' when excY="00" else '0' ;
   isNormalX <= '1' when excX="01" else '0' ;
   isNormalY <= '1' when excY="01" else '0' ;
   isInfX <= '1' when excX="10" else '0' ;
   isInfY <= '1' when excY="10" else '0' ;
   isNaNX <= '1' when excX="11" else '0' ;
   isNaNY <= '1' when excY="11" else '0' ;
   -- Just for readability of the formulae below
   negativeX <= signX ;
   positiveX <= not signX ;
   negativeY <= signY ;
   positiveY <= not signY ;
   -- expfrac comparisons 
   ExpFracCmp: IntComparator_31_111_Freq500_uid4
      port map ( clk  => clk,
                 ce => ce,
                 X => ExpFracX,
                 Y => ExpFracY,
                 XeqY => ExpFracXeqExpFracY,
                 XgtY => ExpFracXgtExpFracY,
                 XltY => ExpFracXltExpFracY);
   -- -- and now the logic
   sameSign <= not (signX xor signY) ;
   XeqYNum <= 
         (isZeroX and isZeroY) -- explicitely stated by IEEE 754
      or (isInfX and isInfY and sameSign)  -- bizarre but also explicitely stated by IEEE 754
      or (isNormalX and isNormalY and sameSign and ExpFracXeqExpFracY)   ;
   XgtYNum <=     -- case enumeration on X
         ( (not (isInfY and positiveY)) and (isInfX  and positiveX)) 
      or ((negativeY or isZeroY) and (isNormalX and positiveX)) 
      or ((negativeY and not isZeroY) and isZeroX) 
      or (isNormalX and isNormalY and positiveY and positiveX and ExpFracXgtExpFracY)
      or (isNormalX and isNormalY and negativeY and negativeX and ExpFracXltExpFracY)
      or ((isInfY and negativeY) and (not (isInfX and negativeX)))    ;
   unorderedR <=  isNaNX or isNaNY;
   XgeYR <= (XeqYNum or XgtYNum)	 and not unorderedR;
   unordered <= unorderedR;
   XgeY <= XgeYR;
end architecture;

--------------------------------------------------------------------------------
--                     IntComparator_31_101_Freq500_uid4
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2021)
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: XltY XgtY

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntComparator_31_101_Freq500_uid4 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(30 downto 0);
          Y : in  std_logic_vector(30 downto 0);
          XltY : out  std_logic;
          XgtY : out  std_logic   );
end entity;

architecture arch of IntComparator_31_101_Freq500_uid4 is
signal XltYi :  std_logic;
signal XgtYi :  std_logic;
begin
   XltYi <= '1' when X<Y else '0';
   XgtYi <= '1' when X>Y else '0';
   XltY <= XltYi;
   XgtY <= XgtYi;
end architecture;

--------------------------------------------------------------------------------
--                         FloatingPointComparatorGT
--                      (FPComparator_8_23_Freq500_uid2)
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2021)
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: unordered XgtY

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity FloatingPointComparatorGT is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(8+23+2 downto 0);
          Y : in  std_logic_vector(8+23+2 downto 0);
          unordered : out  std_logic;
          XgtY : out  std_logic   );
end entity;

architecture arch of FloatingPointComparatorGT is
   component IntComparator_31_101_Freq500_uid4 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(30 downto 0);
             Y : in  std_logic_vector(30 downto 0);
             XltY : out  std_logic;
             XgtY : out  std_logic   );
   end component;

signal excX :  std_logic_vector(1 downto 0);
signal excY :  std_logic_vector(1 downto 0);
signal signX :  std_logic;
signal signY :  std_logic;
signal ExpFracX :  std_logic_vector(30 downto 0);
signal ExpFracY :  std_logic_vector(30 downto 0);
signal isZeroX :  std_logic;
signal isZeroY :  std_logic;
signal isNormalX :  std_logic;
signal isNormalY :  std_logic;
signal isInfX :  std_logic;
signal isInfY :  std_logic;
signal isNaNX :  std_logic;
signal isNaNY :  std_logic;
signal negativeX :  std_logic;
signal positiveX :  std_logic;
signal negativeY :  std_logic;
signal positiveY :  std_logic;
signal ExpFracXltExpFracY :  std_logic;
signal ExpFracXgtExpFracY :  std_logic;
signal sameSign :  std_logic;
signal XgtYNum :  std_logic;
signal unorderedR :  std_logic;
signal XgtYR :  std_logic;
begin
   excX <= X(33 downto 32);
   excY <= Y(33 downto 32);
   signX <= X(31);
   signY <= Y(31);
   ExpFracX <= X(30 downto 0);
   ExpFracY <= Y(30 downto 0);
   -- Comparing (as integers) excX & ExpFracX with excY & ExpFracY would almost work 
   --  since indeed inf>normal>0	
   -- However we wouldn't capture infinity equality in cases when the infinities have different ExpFracs (who knows)...	 
   -- Besides, expliciting the isXXX bits will help factoring code with a comparator for IEEE format (some day)
   isZeroX <= '1' when excX="00" else '0' ;
   isZeroY <= '1' when excY="00" else '0' ;
   isNormalX <= '1' when excX="01" else '0' ;
   isNormalY <= '1' when excY="01" else '0' ;
   isInfX <= '1' when excX="10" else '0' ;
   isInfY <= '1' when excY="10" else '0' ;
   isNaNX <= '1' when excX="11" else '0' ;
   isNaNY <= '1' when excY="11" else '0' ;
   -- Just for readability of the formulae below
   negativeX <= signX ;
   positiveX <= not signX ;
   negativeY <= signY ;
   positiveY <= not signY ;
   -- expfrac comparisons 
   ExpFracCmp: IntComparator_31_101_Freq500_uid4
      port map ( clk  => clk,
                 ce => ce,
                 X => ExpFracX,
                 Y => ExpFracY,
                 XgtY => ExpFracXgtExpFracY,
                 XltY => ExpFracXltExpFracY);
   -- -- and now the logic
   sameSign <= not (signX xor signY) ;
   XgtYNum <=     -- case enumeration on X
         ( (not (isInfY and positiveY)) and (isInfX  and positiveX)) 
      or ((negativeY or isZeroY) and (isNormalX and positiveX)) 
      or ((negativeY and not isZeroY) and isZeroX) 
      or (isNormalX and isNormalY and positiveY and positiveX and ExpFracXgtExpFracY)
      or (isNormalX and isNormalY and negativeY and negativeX and ExpFracXltExpFracY)
      or ((isInfY and negativeY) and (not (isInfX and negativeX)))    ;
   unorderedR <=  isNaNX or isNaNY;
   XgtYR <= XgtYNum and not unorderedR;
   unordered <= unorderedR;
   XgtY <= XgtYR;
end architecture;

--------------------------------------------------------------------------------
--                     IntComparator_31_111_Freq500_uid4
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2021)
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: XltY XeqY XgtY

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntComparator_31_111_Freq500_uid4 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(30 downto 0);
          Y : in  std_logic_vector(30 downto 0);
          XltY : out  std_logic;
          XeqY : out  std_logic;
          XgtY : out  std_logic   );
end entity;

architecture arch of IntComparator_31_111_Freq500_uid4 is
signal XltYi :  std_logic;
signal XeqYi :  std_logic;
signal XgtYi :  std_logic;
begin
   XltYi <= '1' when X<Y else '0';
   XeqYi <= '1' when X=Y else '0';
   XgtYi <= not (XeqYi or XltYi);
   XltY <= XltYi;
   XeqY <= XeqYi;
   XgtY <= XgtYi;
end architecture;

--------------------------------------------------------------------------------
--                         FloatingPointComparatorLE
--                      (FPComparator_8_23_Freq500_uid2)
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2021)
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: unordered XleY

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity FloatingPointComparatorLE is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(8+23+2 downto 0);
          Y : in  std_logic_vector(8+23+2 downto 0);
          unordered : out  std_logic;
          XleY : out  std_logic   );
end entity;

architecture arch of FloatingPointComparatorLE is
   component IntComparator_31_111_Freq500_uid4 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(30 downto 0);
             Y : in  std_logic_vector(30 downto 0);
             XltY : out  std_logic;
             XeqY : out  std_logic;
             XgtY : out  std_logic   );
   end component;

signal excX :  std_logic_vector(1 downto 0);
signal excY :  std_logic_vector(1 downto 0);
signal signX :  std_logic;
signal signY :  std_logic;
signal ExpFracX :  std_logic_vector(30 downto 0);
signal ExpFracY :  std_logic_vector(30 downto 0);
signal isZeroX :  std_logic;
signal isZeroY :  std_logic;
signal isNormalX :  std_logic;
signal isNormalY :  std_logic;
signal isInfX :  std_logic;
signal isInfY :  std_logic;
signal isNaNX :  std_logic;
signal isNaNY :  std_logic;
signal negativeX :  std_logic;
signal positiveX :  std_logic;
signal negativeY :  std_logic;
signal positiveY :  std_logic;
signal ExpFracXeqExpFracY :  std_logic;
signal ExpFracXltExpFracY :  std_logic;
signal ExpFracXgtExpFracY :  std_logic;
signal sameSign :  std_logic;
signal XeqYNum :  std_logic;
signal XltYNum :  std_logic;
signal unorderedR :  std_logic;
signal XleYR :  std_logic;
begin
   excX <= X(33 downto 32);
   excY <= Y(33 downto 32);
   signX <= X(31);
   signY <= Y(31);
   ExpFracX <= X(30 downto 0);
   ExpFracY <= Y(30 downto 0);
   -- Comparing (as integers) excX & ExpFracX with excY & ExpFracY would almost work 
   --  since indeed inf>normal>0	
   -- However we wouldn't capture infinity equality in cases when the infinities have different ExpFracs (who knows)...	 
   -- Besides, expliciting the isXXX bits will help factoring code with a comparator for IEEE format (some day)
   isZeroX <= '1' when excX="00" else '0' ;
   isZeroY <= '1' when excY="00" else '0' ;
   isNormalX <= '1' when excX="01" else '0' ;
   isNormalY <= '1' when excY="01" else '0' ;
   isInfX <= '1' when excX="10" else '0' ;
   isInfY <= '1' when excY="10" else '0' ;
   isNaNX <= '1' when excX="11" else '0' ;
   isNaNY <= '1' when excY="11" else '0' ;
   -- Just for readability of the formulae below
   negativeX <= signX ;
   positiveX <= not signX ;
   negativeY <= signY ;
   positiveY <= not signY ;
   -- expfrac comparisons 
   ExpFracCmp: IntComparator_31_111_Freq500_uid4
      port map ( clk  => clk,
                 ce => ce,
                 X => ExpFracX,
                 Y => ExpFracY,
                 XeqY => ExpFracXeqExpFracY,
                 XgtY => ExpFracXgtExpFracY,
                 XltY => ExpFracXltExpFracY);
   -- -- and now the logic
   sameSign <= not (signX xor signY) ;
   XeqYNum <= 
         (isZeroX and isZeroY) -- explicitely stated by IEEE 754
      or (isInfX and isInfY and sameSign)  -- bizarre but also explicitely stated by IEEE 754
      or (isNormalX and isNormalY and sameSign and ExpFracXeqExpFracY)   ;
   XltYNum <=     -- case enumeration on Y
         ( (not (isInfX and positiveX)) and (isInfY  and positiveY)) 
      or ((negativeX or isZeroX) and (isNormalY and positiveY)) 
      or ((negativeX and not isZeroX) and isZeroY) 
      or (isNormalX and isNormalY and positiveX and positiveY and ExpFracXltExpFracY)
      or (isNormalX and isNormalY and negativeX and negativeY and ExpFracXgtExpFracY)
      or ((isInfX and negativeX) and (not (isInfY and negativeY)))    ;
   unorderedR <=  isNaNX or isNaNY;
   XleYR <= (XeqYNum or XltYNum)	 and not unorderedR;
   unordered <= unorderedR;
   XleY <= XleYR;
end architecture;

--------------------------------------------------------------------------------
--                     IntComparator_31_101_Freq500_uid4
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2021)
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: XltY XgtY

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntComparator_31_101_Freq500_uid4 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(30 downto 0);
          Y : in  std_logic_vector(30 downto 0);
          XltY : out  std_logic;
          XgtY : out  std_logic   );
end entity;

architecture arch of IntComparator_31_101_Freq500_uid4 is
signal XltYi :  std_logic;
signal XgtYi :  std_logic;
begin
   XltYi <= '1' when X<Y else '0';
   XgtYi <= '1' when X>Y else '0';
   XltY <= XltYi;
   XgtY <= XgtYi;
end architecture;

--------------------------------------------------------------------------------
--                         FloatingPointComparatorLT
--                      (FPComparator_8_23_Freq500_uid2)
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2021)
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: unordered XltY

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity FloatingPointComparatorLT is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(8+23+2 downto 0);
          Y : in  std_logic_vector(8+23+2 downto 0);
          unordered : out  std_logic;
          XltY : out  std_logic   );
end entity;

architecture arch of FloatingPointComparatorLT is
   component IntComparator_31_101_Freq500_uid4 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(30 downto 0);
             Y : in  std_logic_vector(30 downto 0);
             XltY : out  std_logic;
             XgtY : out  std_logic   );
   end component;

signal excX :  std_logic_vector(1 downto 0);
signal excY :  std_logic_vector(1 downto 0);
signal signX :  std_logic;
signal signY :  std_logic;
signal ExpFracX :  std_logic_vector(30 downto 0);
signal ExpFracY :  std_logic_vector(30 downto 0);
signal isZeroX :  std_logic;
signal isZeroY :  std_logic;
signal isNormalX :  std_logic;
signal isNormalY :  std_logic;
signal isInfX :  std_logic;
signal isInfY :  std_logic;
signal isNaNX :  std_logic;
signal isNaNY :  std_logic;
signal negativeX :  std_logic;
signal positiveX :  std_logic;
signal negativeY :  std_logic;
signal positiveY :  std_logic;
signal ExpFracXltExpFracY :  std_logic;
signal ExpFracXgtExpFracY :  std_logic;
signal sameSign :  std_logic;
signal XltYNum :  std_logic;
signal unorderedR :  std_logic;
signal XltYR :  std_logic;
begin
   excX <= X(33 downto 32);
   excY <= Y(33 downto 32);
   signX <= X(31);
   signY <= Y(31);
   ExpFracX <= X(30 downto 0);
   ExpFracY <= Y(30 downto 0);
   -- Comparing (as integers) excX & ExpFracX with excY & ExpFracY would almost work 
   --  since indeed inf>normal>0	
   -- However we wouldn't capture infinity equality in cases when the infinities have different ExpFracs (who knows)...	 
   -- Besides, expliciting the isXXX bits will help factoring code with a comparator for IEEE format (some day)
   isZeroX <= '1' when excX="00" else '0' ;
   isZeroY <= '1' when excY="00" else '0' ;
   isNormalX <= '1' when excX="01" else '0' ;
   isNormalY <= '1' when excY="01" else '0' ;
   isInfX <= '1' when excX="10" else '0' ;
   isInfY <= '1' when excY="10" else '0' ;
   isNaNX <= '1' when excX="11" else '0' ;
   isNaNY <= '1' when excY="11" else '0' ;
   -- Just for readability of the formulae below
   negativeX <= signX ;
   positiveX <= not signX ;
   negativeY <= signY ;
   positiveY <= not signY ;
   -- expfrac comparisons 
   ExpFracCmp: IntComparator_31_101_Freq500_uid4
      port map ( clk  => clk,
                 ce => ce,
                 X => ExpFracX,
                 Y => ExpFracY,
                 XgtY => ExpFracXgtExpFracY,
                 XltY => ExpFracXltExpFracY);
   -- -- and now the logic
   sameSign <= not (signX xor signY) ;
   XltYNum <=     -- case enumeration on Y
         ( (not (isInfX and positiveX)) and (isInfY  and positiveY)) 
      or ((negativeX or isZeroX) and (isNormalY and positiveY)) 
      or ((negativeX and not isZeroX) and isZeroY) 
      or (isNormalX and isNormalY and positiveX and positiveY and ExpFracXltExpFracY)
      or (isNormalX and isNormalY and negativeX and negativeY and ExpFracXgtExpFracY)
      or ((isInfX and negativeX) and (not (isInfY and negativeY)))    ;
   unorderedR <=  isNaNX or isNaNY;
   XltYR <= XltYNum and not unorderedR;
   unordered <= unorderedR;
   XltY <= XltYR;
end architecture;

--------------------------------------------------------------------------------
--                      InputIEEE_8_23_to_8_23_comb_uid2
-- VHDL generated for Kintex7 @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2008)
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: X
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity InputIEEE_32bit is
    port (X : in  std_logic_vector(31 downto 0);
          R : out  std_logic_vector(8+23+2 downto 0)   );
end entity;

architecture arch of InputIEEE_32bit is
signal expX :  std_logic_vector(7 downto 0);
signal fracX :  std_logic_vector(22 downto 0);
signal sX :  std_logic;
signal expZero :  std_logic;
signal expInfty :  std_logic;
signal fracZero :  std_logic;
signal reprSubNormal :  std_logic;
signal sfracX :  std_logic_vector(22 downto 0);
signal fracR :  std_logic_vector(22 downto 0);
signal expR :  std_logic_vector(7 downto 0);
signal infinity :  std_logic;
signal zero :  std_logic;
signal NaN :  std_logic;
signal exnR :  std_logic_vector(1 downto 0);
begin
   expX  <= X(30 downto 23);
   fracX  <= X(22 downto 0);
   sX  <= X(31);
   expZero  <= '1' when expX = (7 downto 0 => '0') else '0';
   expInfty  <= '1' when expX = (7 downto 0 => '1') else '0';
   fracZero <= '1' when fracX = (22 downto 0 => '0') else '0';
   reprSubNormal <= fracX(22);
   -- since we have one more exponent value than IEEE (field 0...0, value emin-1),
   -- we can represent subnormal numbers whose mantissa field begins with a 1
   sfracX <= fracX(21 downto 0) & '0' when (expZero='1' and reprSubNormal='1')    else fracX;
   fracR <= sfracX;
   -- copy exponent. This will be OK even for subnormals, zero and infty since in such cases the exn bits will prevail
   expR <= expX;
   infinity <= expInfty and fracZero;
   zero <= expZero and not reprSubNormal;
   NaN <= expInfty and not fracZero;
   exnR <= 
           "00" when zero='1' 
      else "10" when infinity='1' 
      else "11" when NaN='1' 
      else "01" ;  -- normal number
   R <= exnR & sX & expR & fracR; 
end architecture;

--------------------------------------------------------------------------------
--                     InputIEEE_11_52_to_11_52_comb_uid2
-- VHDL generated for Kintex7 @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2008)
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: X
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity InputIEEE_64bit is
    port (X : in  std_logic_vector(63 downto 0);
          R : out  std_logic_vector(11+52+2 downto 0)   );
end entity;

architecture arch of InputIEEE_64bit is
signal expX :  std_logic_vector(10 downto 0);
signal fracX :  std_logic_vector(51 downto 0);
signal sX :  std_logic;
signal expZero :  std_logic;
signal expInfty :  std_logic;
signal fracZero :  std_logic;
signal reprSubNormal :  std_logic;
signal sfracX :  std_logic_vector(51 downto 0);
signal fracR :  std_logic_vector(51 downto 0);
signal expR :  std_logic_vector(10 downto 0);
signal infinity :  std_logic;
signal zero :  std_logic;
signal NaN :  std_logic;
signal exnR :  std_logic_vector(1 downto 0);
begin
   expX  <= X(62 downto 52);
   fracX  <= X(51 downto 0);
   sX  <= X(63);
   expZero  <= '1' when expX = (10 downto 0 => '0') else '0';
   expInfty  <= '1' when expX = (10 downto 0 => '1') else '0';
   fracZero <= '1' when fracX = (51 downto 0 => '0') else '0';
   reprSubNormal <= fracX(51);
   -- since we have one more exponent value than IEEE (field 0...0, value emin-1),
   -- we can represent subnormal numbers whose mantissa field begins with a 1
   sfracX <= fracX(50 downto 0) & '0' when (expZero='1' and reprSubNormal='1')    else fracX;
   fracR <= sfracX;
   -- copy exponent. This will be OK even for subnormals, zero and infty since in such cases the exn bits will prevail
   expR <= expX;
   infinity <= expInfty and fracZero;
   zero <= expZero and not reprSubNormal;
   NaN <= expInfty and not fracZero;
   exnR <= 
           "00" when zero='1' 
      else "10" when infinity='1' 
      else "11" when NaN='1' 
      else "01" ;  -- normal number
   R <= exnR & sX & expR & fracR; 
end architecture;

--------------------------------------------------------------------------------
--                     OutputIEEE_8_23_to_8_23_comb_uid2
-- VHDL generated for Kintex7 @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: F. Ferrandi  (2009-2012)
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: X
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity OutputIEEE_32bit is
    port (X : in  std_logic_vector(8+23+2 downto 0);
          R : out  std_logic_vector(31 downto 0)   );
end entity;

architecture arch of OutputIEEE_32bit is
signal fracX :  std_logic_vector(22 downto 0);
signal exnX :  std_logic_vector(1 downto 0);
signal expX :  std_logic_vector(7 downto 0);
signal sX :  std_logic;
signal expZero :  std_logic;
signal fracR :  std_logic_vector(22 downto 0);
signal expR :  std_logic_vector(7 downto 0);
begin
   fracX  <= X(22 downto 0);
   exnX  <= X(33 downto 32);
   expX  <= X(30 downto 23);
   sX  <= X(31) when (exnX = "01" or exnX = "10" or exnX = "00") else '0';
   expZero  <= '1' when expX = (7 downto 0 => '0') else '0';
   -- since we have one more exponent value than IEEE (field 0...0, value emin-1),
   -- we can represent subnormal numbers whose mantissa field begins with a 1
   fracR <= 
      "00000000000000000000000" when (exnX = "00") else
      '1' & fracX(22 downto 1) & "" when (expZero = '1' and exnX = "01") else
      fracX  & "" when (exnX = "01") else 
      "0000000000000000000000" & exnX(0);
   expR <=  
      (7 downto 0 => '0') when (exnX = "00") else
      expX when (exnX = "01") else 
      (7 downto 0 => '1');
   R <= sX & expR & fracR; 
end architecture;

--------------------------------------------------------------------------------
--                    OutputIEEE_11_52_to_11_52_comb_uid2
-- VHDL generated for Kintex7 @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: F. Ferrandi  (2009-2012)
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: X
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity OutputIEEE_64bit is
    port (X : in  std_logic_vector(11+52+2 downto 0);
          R : out  std_logic_vector(63 downto 0)   );
end entity;

architecture arch of OutputIEEE_64bit is
signal fracX :  std_logic_vector(51 downto 0);
signal exnX :  std_logic_vector(1 downto 0);
signal expX :  std_logic_vector(10 downto 0);
signal sX :  std_logic;
signal expZero :  std_logic;
signal fracR :  std_logic_vector(51 downto 0);
signal expR :  std_logic_vector(10 downto 0);
begin
   fracX  <= X(51 downto 0);
   exnX  <= X(65 downto 64);
   expX  <= X(62 downto 52);
   sX  <= X(63) when (exnX = "01" or exnX = "10" or exnX = "00") else '0';
   expZero  <= '1' when expX = (10 downto 0 => '0') else '0';
   -- since we have one more exponent value than IEEE (field 0...0, value emin-1),
   -- we can represent subnormal numbers whose mantissa field begins with a 1
   fracR <= 
      "0000000000000000000000000000000000000000000000000000" when (exnX = "00") else
      '1' & fracX(51 downto 1) & "" when (expZero = '1' and exnX = "01") else
      fracX  & "" when (exnX = "01") else 
      "000000000000000000000000000000000000000000000000000" & exnX(0);
   expR <=  
      (10 downto 0 => '0') when (exnX = "00") else
      expX when (exnX = "01") else 
      (10 downto 0 => '1');
   R <= sX & expR & fracR; 
end architecture;



--------------------------------------------------------------------------------
--                       IntComparator_63_111_F500_uid4
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2021)
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: XltY XeqY XgtY

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntComparator_63_111_F500_uid4 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(62 downto 0);
          Y : in  std_logic_vector(62 downto 0);
          XltY : out  std_logic;
          XeqY : out  std_logic;
          XgtY : out  std_logic   );
end entity;

architecture arch of IntComparator_63_111_F500_uid4 is
signal XltYi :  std_logic;
signal XeqYi :  std_logic;
signal XgtYi :  std_logic;
begin
   XltYi <= '1' when X<Y else '0';
   XeqYi <= '1' when X=Y else '0';
   XgtYi <= not (XeqYi or XltYi);
   XltY <= XltYi;
   XeqY <= XeqYi;
   XgtY <= XgtYi;
end architecture;

--------------------------------------------------------------------------------
--                             FPComparator_64bit
--                       (FPComparator_11_52_F500_uid2)
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2021)
--------------------------------------------------------------------------------
-- Pipeline depth: 1 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: unordered XltY XeqY XgtY XleY XgeY

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity FPComparator_64bit is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(11+52+2 downto 0);
          Y : in  std_logic_vector(11+52+2 downto 0);
          unordered : out  std_logic;
          XltY : out  std_logic;
          XeqY : out  std_logic;
          XgtY : out  std_logic;
          XleY : out  std_logic;
          XgeY : out  std_logic   );
end entity;

architecture arch of FPComparator_64bit is
   component IntComparator_63_111_F500_uid4 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(62 downto 0);
             Y : in  std_logic_vector(62 downto 0);
             XltY : out  std_logic;
             XeqY : out  std_logic;
             XgtY : out  std_logic   );
   end component;

signal excX :  std_logic_vector(1 downto 0);
signal excY :  std_logic_vector(1 downto 0);
signal signX :  std_logic;
signal signY :  std_logic;
signal ExpFracX :  std_logic_vector(62 downto 0);
signal ExpFracY :  std_logic_vector(62 downto 0);
signal isZeroX, isZeroX_d1 :  std_logic;
signal isZeroY, isZeroY_d1 :  std_logic;
signal isNormalX, isNormalX_d1 :  std_logic;
signal isNormalY, isNormalY_d1 :  std_logic;
signal isInfX, isInfX_d1 :  std_logic;
signal isInfY, isInfY_d1 :  std_logic;
signal isNaNX :  std_logic;
signal isNaNY :  std_logic;
signal negativeX, negativeX_d1 :  std_logic;
signal positiveX, positiveX_d1 :  std_logic;
signal negativeY, negativeY_d1 :  std_logic;
signal positiveY, positiveY_d1 :  std_logic;
signal ExpFracXeqExpFracY :  std_logic;
signal ExpFracXltExpFracY, ExpFracXltExpFracY_d1 :  std_logic;
signal ExpFracXgtExpFracY, ExpFracXgtExpFracY_d1 :  std_logic;
signal sameSign :  std_logic;
signal XeqYNum, XeqYNum_d1 :  std_logic;
signal XltYNum :  std_logic;
signal XgtYNum :  std_logic;
signal unorderedR, unorderedR_d1 :  std_logic;
signal XltYR :  std_logic;
signal XeqYR :  std_logic;
signal XgtYR :  std_logic;
signal XleYR :  std_logic;
signal XgeYR :  std_logic;
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               isZeroX_d1 <=  isZeroX;
               isZeroY_d1 <=  isZeroY;
               isNormalX_d1 <=  isNormalX;
               isNormalY_d1 <=  isNormalY;
               isInfX_d1 <=  isInfX;
               isInfY_d1 <=  isInfY;
               negativeX_d1 <=  negativeX;
               positiveX_d1 <=  positiveX;
               negativeY_d1 <=  negativeY;
               positiveY_d1 <=  positiveY;
               ExpFracXltExpFracY_d1 <=  ExpFracXltExpFracY;
               ExpFracXgtExpFracY_d1 <=  ExpFracXgtExpFracY;
               XeqYNum_d1 <=  XeqYNum;
               unorderedR_d1 <=  unorderedR;
            end if;
         end if;
      end process;
   excX <= X(65 downto 64);
   excY <= Y(65 downto 64);
   signX <= X(63);
   signY <= Y(63);
   ExpFracX <= X(62 downto 0);
   ExpFracY <= Y(62 downto 0);
   -- Comparing (as integers) excX & ExpFracX with excY & ExpFracY would almost work 
   --  since indeed inf>normal>0	
   -- However we wouldn't capture infinity equality in cases when the infinities have different ExpFracs (who knows)...	 
   -- Besides, expliciting the isXXX bits will help factoring code with a comparator for IEEE format (some day)
   isZeroX <= '1' when excX="00" else '0' ;
   isZeroY <= '1' when excY="00" else '0' ;
   isNormalX <= '1' when excX="01" else '0' ;
   isNormalY <= '1' when excY="01" else '0' ;
   isInfX <= '1' when excX="10" else '0' ;
   isInfY <= '1' when excY="10" else '0' ;
   isNaNX <= '1' when excX="11" else '0' ;
   isNaNY <= '1' when excY="11" else '0' ;
   -- Just for readability of the formulae below
   negativeX <= signX ;
   positiveX <= not signX ;
   negativeY <= signY ;
   positiveY <= not signY ;
   -- expfrac comparisons 
   ExpFracCmp: IntComparator_63_111_F500_uid4
      port map ( clk  => clk,
                 ce => ce,
                 X => ExpFracX,
                 Y => ExpFracY,
                 XeqY => ExpFracXeqExpFracY,
                 XgtY => ExpFracXgtExpFracY,
                 XltY => ExpFracXltExpFracY);
   -- -- and now the logic
   sameSign <= not (signX xor signY) ;
   XeqYNum <= 
         (isZeroX and isZeroY) -- explicitely stated by IEEE 754
      or (isInfX and isInfY and sameSign)  -- bizarre but also explicitely stated by IEEE 754
      or (isNormalX and isNormalY and sameSign and ExpFracXeqExpFracY)   ;
   XltYNum <=     -- case enumeration on Y
         ( (not (isInfX_d1 and positiveX_d1)) and (isInfY_d1  and positiveY_d1)) 
      or ((negativeX_d1 or isZeroX_d1) and (isNormalY_d1 and positiveY_d1)) 
      or ((negativeX_d1 and not isZeroX_d1) and isZeroY_d1) 
      or (isNormalX_d1 and isNormalY_d1 and positiveX_d1 and positiveY_d1 and ExpFracXltExpFracY_d1)
      or (isNormalX_d1 and isNormalY_d1 and negativeX_d1 and negativeY_d1 and ExpFracXgtExpFracY_d1)
      or ((isInfX_d1 and negativeX_d1) and (not (isInfY_d1 and negativeY_d1)))    ;
   XgtYNum <=     -- case enumeration on X
         ( (not (isInfY_d1 and positiveY_d1)) and (isInfX_d1  and positiveX_d1)) 
      or ((negativeY_d1 or isZeroY_d1) and (isNormalX_d1 and positiveX_d1)) 
      or ((negativeY_d1 and not isZeroY_d1) and isZeroX_d1) 
      or (isNormalX_d1 and isNormalY_d1 and positiveY_d1 and positiveX_d1 and ExpFracXgtExpFracY_d1)
      or (isNormalX_d1 and isNormalY_d1 and negativeY_d1 and negativeX_d1 and ExpFracXltExpFracY_d1)
      or ((isInfY_d1 and negativeY_d1) and (not (isInfX_d1 and negativeX_d1)))    ;
   unorderedR <=  isNaNX or isNaNY;
   XltYR <= XltYNum and not unorderedR_d1;
   XeqYR <= XeqYNum and not unorderedR;
   XgtYR <= XgtYNum and not unorderedR_d1;
   XleYR <= (XeqYNum_d1 or XltYNum)	 and not unorderedR_d1;
   XgeYR <= (XeqYNum_d1 or XgtYNum)	 and not unorderedR_d1;
   unordered <= unorderedR;
   XltY <= XltYR;
   XeqY <= XeqYR;
   XgtY <= XgtYR;
   XleY <= XleYR;
   XgeY <= XgeYR;
end architecture;

--------------------------------------------------------------------------------
--                       IntComparator_31_111_F500_uid8
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2021)
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: XltY XeqY XgtY

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntComparator_31_111_F500_uid8 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(30 downto 0);
          Y : in  std_logic_vector(30 downto 0);
          XltY : out  std_logic;
          XeqY : out  std_logic;
          XgtY : out  std_logic   );
end entity;

architecture arch of IntComparator_31_111_F500_uid8 is
signal XltYi :  std_logic;
signal XeqYi :  std_logic;
signal XgtYi :  std_logic;
begin
   XltYi <= '1' when X<Y else '0';
   XeqYi <= '1' when X=Y else '0';
   XgtYi <= not (XeqYi or XltYi);
   XltY <= XltYi;
   XeqY <= XeqYi;
   XgtY <= XgtYi;
end architecture;

--------------------------------------------------------------------------------
--                             FPComparator_32bit
--                       (FPComparator_8_23_F500_uid6)
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2021)
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: unordered XltY XeqY XgtY XleY XgeY

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity FPComparator_32bit is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(8+23+2 downto 0);
          Y : in  std_logic_vector(8+23+2 downto 0);
          unordered : out  std_logic;
          XltY : out  std_logic;
          XeqY : out  std_logic;
          XgtY : out  std_logic;
          XleY : out  std_logic;
          XgeY : out  std_logic   );
end entity;

architecture arch of FPComparator_32bit is
   component IntComparator_31_111_F500_uid8 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(30 downto 0);
             Y : in  std_logic_vector(30 downto 0);
             XltY : out  std_logic;
             XeqY : out  std_logic;
             XgtY : out  std_logic   );
   end component;

signal excX :  std_logic_vector(1 downto 0);
signal excY :  std_logic_vector(1 downto 0);
signal signX :  std_logic;
signal signY :  std_logic;
signal ExpFracX :  std_logic_vector(30 downto 0);
signal ExpFracY :  std_logic_vector(30 downto 0);
signal isZeroX :  std_logic;
signal isZeroY :  std_logic;
signal isNormalX :  std_logic;
signal isNormalY :  std_logic;
signal isInfX :  std_logic;
signal isInfY :  std_logic;
signal isNaNX :  std_logic;
signal isNaNY :  std_logic;
signal negativeX :  std_logic;
signal positiveX :  std_logic;
signal negativeY :  std_logic;
signal positiveY :  std_logic;
signal ExpFracXeqExpFracY :  std_logic;
signal ExpFracXltExpFracY :  std_logic;
signal ExpFracXgtExpFracY :  std_logic;
signal sameSign :  std_logic;
signal XeqYNum :  std_logic;
signal XltYNum :  std_logic;
signal XgtYNum :  std_logic;
signal unorderedR :  std_logic;
signal XltYR :  std_logic;
signal XeqYR :  std_logic;
signal XgtYR :  std_logic;
signal XleYR :  std_logic;
signal XgeYR :  std_logic;
begin
   excX <= X(33 downto 32);
   excY <= Y(33 downto 32);
   signX <= X(31);
   signY <= Y(31);
   ExpFracX <= X(30 downto 0);
   ExpFracY <= Y(30 downto 0);
   -- Comparing (as integers) excX & ExpFracX with excY & ExpFracY would almost work 
   --  since indeed inf>normal>0	
   -- However we wouldn't capture infinity equality in cases when the infinities have different ExpFracs (who knows)...	 
   -- Besides, expliciting the isXXX bits will help factoring code with a comparator for IEEE format (some day)
   isZeroX <= '1' when excX="00" else '0' ;
   isZeroY <= '1' when excY="00" else '0' ;
   isNormalX <= '1' when excX="01" else '0' ;
   isNormalY <= '1' when excY="01" else '0' ;
   isInfX <= '1' when excX="10" else '0' ;
   isInfY <= '1' when excY="10" else '0' ;
   isNaNX <= '1' when excX="11" else '0' ;
   isNaNY <= '1' when excY="11" else '0' ;
   -- Just for readability of the formulae below
   negativeX <= signX ;
   positiveX <= not signX ;
   negativeY <= signY ;
   positiveY <= not signY ;
   -- expfrac comparisons 
   ExpFracCmp: IntComparator_31_111_F500_uid8
      port map ( clk  => clk,
                 ce => ce,
                 X => ExpFracX,
                 Y => ExpFracY,
                 XeqY => ExpFracXeqExpFracY,
                 XgtY => ExpFracXgtExpFracY,
                 XltY => ExpFracXltExpFracY);
   -- -- and now the logic
   sameSign <= not (signX xor signY) ;
   XeqYNum <= 
         (isZeroX and isZeroY) -- explicitely stated by IEEE 754
      or (isInfX and isInfY and sameSign)  -- bizarre but also explicitely stated by IEEE 754
      or (isNormalX and isNormalY and sameSign and ExpFracXeqExpFracY)   ;
   XltYNum <=     -- case enumeration on Y
         ( (not (isInfX and positiveX)) and (isInfY  and positiveY)) 
      or ((negativeX or isZeroX) and (isNormalY and positiveY)) 
      or ((negativeX and not isZeroX) and isZeroY) 
      or (isNormalX and isNormalY and positiveX and positiveY and ExpFracXltExpFracY)
      or (isNormalX and isNormalY and negativeX and negativeY and ExpFracXgtExpFracY)
      or ((isInfX and negativeX) and (not (isInfY and negativeY)))    ;
   XgtYNum <=     -- case enumeration on X
         ( (not (isInfY and positiveY)) and (isInfX  and positiveX)) 
      or ((negativeY or isZeroY) and (isNormalX and positiveX)) 
      or ((negativeY and not isZeroY) and isZeroX) 
      or (isNormalX and isNormalY and positiveY and positiveX and ExpFracXgtExpFracY)
      or (isNormalX and isNormalY and negativeY and negativeX and ExpFracXltExpFracY)
      or ((isInfY and negativeY) and (not (isInfX and negativeX)))    ;
   unorderedR <=  isNaNX or isNaNY;
   XltYR <= XltYNum and not unorderedR;
   XeqYR <= XeqYNum and not unorderedR;
   XgtYR <= XgtYNum and not unorderedR;
   XleYR <= (XeqYNum or XltYNum)	 and not unorderedR;
   XgeYR <= (XeqYNum or XgtYNum)	 and not unorderedR;
   unordered <= unorderedR;
   XltY <= XltYR;
   XeqY <= XeqYR;
   XgtY <= XgtYR;
   XleY <= XleYR;
   XgeY <= XgeYR;
end architecture;

--------------------------------------------------------------------------------
--                 RightShifterSticky53_by_max_55_F500_uid12
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca (2008-2011), Florent de Dinechin (2008-2019)
--------------------------------------------------------------------------------
-- Pipeline depth: 4 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X S
-- Output signals: R Sticky

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity RightShifterSticky53_by_max_55_F500_uid12 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(52 downto 0);
          S : in  std_logic_vector(5 downto 0);
          R : out  std_logic_vector(54 downto 0);
          Sticky : out  std_logic   );
end entity;

architecture arch of RightShifterSticky53_by_max_55_F500_uid12 is
signal ps, ps_d1, ps_d2, ps_d3, ps_d4 :  std_logic_vector(5 downto 0);
signal Xpadded :  std_logic_vector(54 downto 0);
signal level6, level6_d1 :  std_logic_vector(54 downto 0);
signal stk5, stk5_d1 :  std_logic;
signal level5, level5_d1, level5_d2 :  std_logic_vector(54 downto 0);
signal stk4 :  std_logic;
signal level4, level4_d1 :  std_logic_vector(54 downto 0);
signal stk3, stk3_d1 :  std_logic;
signal level3, level3_d1, level3_d2 :  std_logic_vector(54 downto 0);
signal stk2 :  std_logic;
signal level2, level2_d1, level2_d2 :  std_logic_vector(54 downto 0);
signal stk1, stk1_d1 :  std_logic;
signal level1, level1_d1, level1_d2, level1_d3 :  std_logic_vector(54 downto 0);
signal stk0 :  std_logic;
signal level0 :  std_logic_vector(54 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               ps_d1 <=  ps;
               ps_d2 <=  ps_d1;
               ps_d3 <=  ps_d2;
               ps_d4 <=  ps_d3;
               level6_d1 <=  level6;
               stk5_d1 <=  stk5;
               level5_d1 <=  level5;
               level5_d2 <=  level5_d1;
               level4_d1 <=  level4;
               stk3_d1 <=  stk3;
               level3_d1 <=  level3;
               level3_d2 <=  level3_d1;
               level2_d1 <=  level2;
               level2_d2 <=  level2_d1;
               stk1_d1 <=  stk1;
               level1_d1 <=  level1;
               level1_d2 <=  level1_d1;
               level1_d3 <=  level1_d2;
            end if;
         end if;
      end process;
   ps<= S;
   Xpadded <= X&(1 downto 0 => '0');
   level6<= Xpadded;
   stk5 <= '1' when (level6_d1(31 downto 0)/="00000000000000000000000000000000" and ps_d1(5)='1')   else '0';
   level5 <=  level6 when  ps(5)='0'    else (31 downto 0 => '0') & level6(54 downto 32);
   stk4 <= '1' when (level5_d2(15 downto 0)/="0000000000000000" and ps_d2(4)='1') or stk5_d1 ='1'   else '0';
   level4 <=  level5_d1 when  ps_d1(4)='0'    else (15 downto 0 => '0') & level5_d1(54 downto 16);
   stk3 <= '1' when (level4_d1(7 downto 0)/="00000000" and ps_d2(3)='1') or stk4 ='1'   else '0';
   level3 <=  level4 when  ps_d1(3)='0'    else (7 downto 0 => '0') & level4(54 downto 8);
   stk2 <= '1' when (level3_d2(3 downto 0)/="0000" and ps_d3(2)='1') or stk3_d1 ='1'   else '0';
   level2 <=  level3 when  ps_d1(2)='0'    else (3 downto 0 => '0') & level3(54 downto 4);
   stk1 <= '1' when (level2_d2(1 downto 0)/="00" and ps_d3(1)='1') or stk2 ='1'   else '0';
   level1 <=  level2 when  ps_d1(1)='0'    else (1 downto 0 => '0') & level2(54 downto 2);
   stk0 <= '1' when (level1_d3(0 downto 0)/="0" and ps_d4(0)='1') or stk1_d1 ='1'   else '0';
   level0 <=  level1 when  ps_d1(0)='0'    else (0 downto 0 => '0') & level1(54 downto 1);
   R <= level0;
   Sticky <= stk0;
end architecture;

--------------------------------------------------------------------------------
--                           IntAdder_56_F500_uid14
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2016)
--------------------------------------------------------------------------------
-- Pipeline depth: 1 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y Cin
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntAdder_56_F500_uid14 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(55 downto 0);
          Y : in  std_logic_vector(55 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(55 downto 0)   );
end entity;

architecture arch of IntAdder_56_F500_uid14 is
signal Cin_1, Cin_1_d1 :  std_logic;
signal X_1, X_1_d1, X_1_d2, X_1_d3, X_1_d4, X_1_d5 :  std_logic_vector(56 downto 0);
signal Y_1, Y_1_d1, Y_1_d2, Y_1_d3, Y_1_d4 :  std_logic_vector(56 downto 0);
signal S_1 :  std_logic_vector(56 downto 0);
signal R_1 :  std_logic_vector(55 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               Cin_1_d1 <=  Cin_1;
               X_1_d1 <=  X_1;
               X_1_d2 <=  X_1_d1;
               X_1_d3 <=  X_1_d2;
               X_1_d4 <=  X_1_d3;
               X_1_d5 <=  X_1_d4;
               Y_1_d1 <=  Y_1;
               Y_1_d2 <=  Y_1_d1;
               Y_1_d3 <=  Y_1_d2;
               Y_1_d4 <=  Y_1_d3;
            end if;
         end if;
      end process;
   Cin_1 <= Cin;
   X_1 <= '0' & X(55 downto 0);
   Y_1 <= '0' & Y(55 downto 0);
   S_1 <= X_1_d5 + Y_1_d4 + Cin_1_d1;
   R_1 <= S_1(55 downto 0);
   R <= R_1 ;
end architecture;

--------------------------------------------------------------------------------
--                      Normalizer_Z_57_57_57_F500_uid16
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin, (2007-2020)
--------------------------------------------------------------------------------
-- Pipeline depth: 4 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X
-- Output signals: Count R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity Normalizer_Z_57_57_57_F500_uid16 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(56 downto 0);
          Count : out  std_logic_vector(5 downto 0);
          R : out  std_logic_vector(56 downto 0)   );
end entity;

architecture arch of Normalizer_Z_57_57_57_F500_uid16 is
signal level6, level6_d1 :  std_logic_vector(56 downto 0);
signal count5, count5_d1, count5_d2, count5_d3 :  std_logic;
signal level5, level5_d1 :  std_logic_vector(56 downto 0);
signal count4, count4_d1, count4_d2, count4_d3 :  std_logic;
signal level4 :  std_logic_vector(56 downto 0);
signal count3, count3_d1, count3_d2 :  std_logic;
signal level3, level3_d1 :  std_logic_vector(56 downto 0);
signal count2, count2_d1, count2_d2 :  std_logic;
signal level2 :  std_logic_vector(56 downto 0);
signal count1, count1_d1 :  std_logic;
signal level1, level1_d1 :  std_logic_vector(56 downto 0);
signal count0 :  std_logic;
signal level0 :  std_logic_vector(56 downto 0);
signal sCount :  std_logic_vector(5 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               level6_d1 <=  level6;
               count5_d1 <=  count5;
               count5_d2 <=  count5_d1;
               count5_d3 <=  count5_d2;
               level5_d1 <=  level5;
               count4_d1 <=  count4;
               count4_d2 <=  count4_d1;
               count4_d3 <=  count4_d2;
               count3_d1 <=  count3;
               count3_d2 <=  count3_d1;
               level3_d1 <=  level3;
               count2_d1 <=  count2;
               count2_d2 <=  count2_d1;
               count1_d1 <=  count1;
               level1_d1 <=  level1;
            end if;
         end if;
      end process;
   level6 <= X ;
   count5<= '1' when level6_d1(56 downto 25) = (56 downto 25=>'0') else '0';
   level5<= level6_d1(56 downto 0) when count5='0' else level6_d1(24 downto 0) & (31 downto 0 => '0');

   count4<= '1' when level5(56 downto 41) = (56 downto 41=>'0') else '0';
   level4<= level5_d1(56 downto 0) when count4_d1='0' else level5_d1(40 downto 0) & (15 downto 0 => '0');

   count3<= '1' when level4(56 downto 49) = (56 downto 49=>'0') else '0';
   level3<= level4(56 downto 0) when count3='0' else level4(48 downto 0) & (7 downto 0 => '0');

   count2<= '1' when level3(56 downto 53) = (56 downto 53=>'0') else '0';
   level2<= level3_d1(56 downto 0) when count2_d1='0' else level3_d1(52 downto 0) & (3 downto 0 => '0');

   count1<= '1' when level2(56 downto 55) = (56 downto 55=>'0') else '0';
   level1<= level2(56 downto 0) when count1='0' else level2(54 downto 0) & (1 downto 0 => '0');

   count0<= '1' when level1_d1(56 downto 56) = (56 downto 56=>'0') else '0';
   level0<= level1_d1(56 downto 0) when count0='0' else level1_d1(55 downto 0) & (0 downto 0 => '0');

   R <= level0;
   sCount <= count5_d3 & count4_d3 & count3_d2 & count2_d2 & count1_d1 & count0;
   Count <= sCount;
end architecture;

--------------------------------------------------------------------------------
--                           IntAdder_66_F500_uid19
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2016)
--------------------------------------------------------------------------------
-- Pipeline depth: 1 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y Cin
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntAdder_66_F500_uid19 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(65 downto 0);
          Y : in  std_logic_vector(65 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(65 downto 0)   );
end entity;

architecture arch of IntAdder_66_F500_uid19 is
signal Cin_1, Cin_1_d1 :  std_logic;
signal X_1, X_1_d1 :  std_logic_vector(66 downto 0);
signal Y_1, Y_1_d1, Y_1_d2, Y_1_d3, Y_1_d4, Y_1_d5, Y_1_d6, Y_1_d7, Y_1_d8, Y_1_d9, Y_1_d10, Y_1_d11 :  std_logic_vector(66 downto 0);
signal S_1 :  std_logic_vector(66 downto 0);
signal R_1 :  std_logic_vector(65 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               Cin_1_d1 <=  Cin_1;
               X_1_d1 <=  X_1;
               Y_1_d1 <=  Y_1;
               Y_1_d2 <=  Y_1_d1;
               Y_1_d3 <=  Y_1_d2;
               Y_1_d4 <=  Y_1_d3;
               Y_1_d5 <=  Y_1_d4;
               Y_1_d6 <=  Y_1_d5;
               Y_1_d7 <=  Y_1_d6;
               Y_1_d8 <=  Y_1_d7;
               Y_1_d9 <=  Y_1_d8;
               Y_1_d10 <=  Y_1_d9;
               Y_1_d11 <=  Y_1_d10;
            end if;
         end if;
      end process;
   Cin_1 <= Cin;
   X_1 <= '0' & X(65 downto 0);
   Y_1 <= '0' & Y(65 downto 0);
   S_1 <= X_1_d1 + Y_1_d11 + Cin_1_d1;
   R_1 <= S_1(65 downto 0);
   R <= R_1 ;
end architecture;

--------------------------------------------------------------------------------
--                                FPAdd_64bit
--                          (FPAdd_11_52_F500_uid10)
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin, Bogdan Pasca (2010-2017)
--------------------------------------------------------------------------------
-- Pipeline depth: 12 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity FPAdd_64bit is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(11+52+2 downto 0);
          Y : in  std_logic_vector(11+52+2 downto 0);
          R : out  std_logic_vector(11+52+2 downto 0)   );
end entity;

architecture arch of FPAdd_64bit is
   component RightShifterSticky53_by_max_55_F500_uid12 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(52 downto 0);
             S : in  std_logic_vector(5 downto 0);
             R : out  std_logic_vector(54 downto 0);
             Sticky : out  std_logic   );
   end component;

   component IntAdder_56_F500_uid14 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(55 downto 0);
             Y : in  std_logic_vector(55 downto 0);
             Cin : in  std_logic;
             R : out  std_logic_vector(55 downto 0)   );
   end component;

   component Normalizer_Z_57_57_57_F500_uid16 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(56 downto 0);
             Count : out  std_logic_vector(5 downto 0);
             R : out  std_logic_vector(56 downto 0)   );
   end component;

   component IntAdder_66_F500_uid19 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(65 downto 0);
             Y : in  std_logic_vector(65 downto 0);
             Cin : in  std_logic;
             R : out  std_logic_vector(65 downto 0)   );
   end component;

signal excExpFracX :  std_logic_vector(64 downto 0);
signal excExpFracY :  std_logic_vector(64 downto 0);
signal swap, swap_d1 :  std_logic;
signal eXmeY, eXmeY_d1 :  std_logic_vector(10 downto 0);
signal eYmeX, eYmeX_d1 :  std_logic_vector(10 downto 0);
signal expDiff :  std_logic_vector(10 downto 0);
signal newX :  std_logic_vector(65 downto 0);
signal newY :  std_logic_vector(65 downto 0);
signal expX :  std_logic_vector(10 downto 0);
signal excX :  std_logic_vector(1 downto 0);
signal excY :  std_logic_vector(1 downto 0);
signal signX :  std_logic;
signal signY :  std_logic;
signal EffSub, EffSub_d1, EffSub_d2, EffSub_d3, EffSub_d4, EffSub_d5, EffSub_d6, EffSub_d7, EffSub_d8, EffSub_d9, EffSub_d10, EffSub_d11 :  std_logic;
signal sXsYExnXY :  std_logic_vector(5 downto 0);
signal sdExnXY :  std_logic_vector(3 downto 0);
signal fracY :  std_logic_vector(52 downto 0);
signal excRt, excRt_d1, excRt_d2, excRt_d3, excRt_d4, excRt_d5, excRt_d6, excRt_d7, excRt_d8, excRt_d9, excRt_d10, excRt_d11 :  std_logic_vector(1 downto 0);
signal signR, signR_d1, signR_d2, signR_d3, signR_d4, signR_d5, signR_d6, signR_d7, signR_d8, signR_d9 :  std_logic;
signal shiftedOut :  std_logic;
signal shiftVal :  std_logic_vector(5 downto 0);
signal shiftedFracY :  std_logic_vector(54 downto 0);
signal sticky, sticky_d1 :  std_logic;
signal fracYpad :  std_logic_vector(55 downto 0);
signal EffSubVector, EffSubVector_d1 :  std_logic_vector(55 downto 0);
signal fracYpadXorOp :  std_logic_vector(55 downto 0);
signal fracXpad :  std_logic_vector(55 downto 0);
signal cInSigAdd :  std_logic;
signal fracAddResult :  std_logic_vector(55 downto 0);
signal fracSticky :  std_logic_vector(56 downto 0);
signal nZerosNew :  std_logic_vector(5 downto 0);
signal shiftedFrac :  std_logic_vector(56 downto 0);
signal extendedExpInc, extendedExpInc_d1, extendedExpInc_d2, extendedExpInc_d3, extendedExpInc_d4, extendedExpInc_d5, extendedExpInc_d6, extendedExpInc_d7, extendedExpInc_d8, extendedExpInc_d9 :  std_logic_vector(11 downto 0);
signal updatedExp :  std_logic_vector(12 downto 0);
signal eqdiffsign, eqdiffsign_d1, eqdiffsign_d2 :  std_logic;
signal expFrac :  std_logic_vector(65 downto 0);
signal stk :  std_logic;
signal rnd :  std_logic;
signal lsb :  std_logic;
signal needToRound :  std_logic;
signal RoundedExpFrac :  std_logic_vector(65 downto 0);
signal upExc :  std_logic_vector(1 downto 0);
signal fracR, fracR_d1 :  std_logic_vector(51 downto 0);
signal expR, expR_d1 :  std_logic_vector(10 downto 0);
signal exExpExc, exExpExc_d1 :  std_logic_vector(3 downto 0);
signal excRt2 :  std_logic_vector(1 downto 0);
signal excR :  std_logic_vector(1 downto 0);
signal signR2, signR2_d1, signR2_d2 :  std_logic;
signal computedR :  std_logic_vector(65 downto 0);
signal X_d1 :  std_logic_vector(11+52+2 downto 0);
signal Y_d1 :  std_logic_vector(11+52+2 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               swap_d1 <=  swap;
               eXmeY_d1 <=  eXmeY;
               eYmeX_d1 <=  eYmeX;
               EffSub_d1 <=  EffSub;
               EffSub_d2 <=  EffSub_d1;
               EffSub_d3 <=  EffSub_d2;
               EffSub_d4 <=  EffSub_d3;
               EffSub_d5 <=  EffSub_d4;
               EffSub_d6 <=  EffSub_d5;
               EffSub_d7 <=  EffSub_d6;
               EffSub_d8 <=  EffSub_d7;
               EffSub_d9 <=  EffSub_d8;
               EffSub_d10 <=  EffSub_d9;
               EffSub_d11 <=  EffSub_d10;
               excRt_d1 <=  excRt;
               excRt_d2 <=  excRt_d1;
               excRt_d3 <=  excRt_d2;
               excRt_d4 <=  excRt_d3;
               excRt_d5 <=  excRt_d4;
               excRt_d6 <=  excRt_d5;
               excRt_d7 <=  excRt_d6;
               excRt_d8 <=  excRt_d7;
               excRt_d9 <=  excRt_d8;
               excRt_d10 <=  excRt_d9;
               excRt_d11 <=  excRt_d10;
               signR_d1 <=  signR;
               signR_d2 <=  signR_d1;
               signR_d3 <=  signR_d2;
               signR_d4 <=  signR_d3;
               signR_d5 <=  signR_d4;
               signR_d6 <=  signR_d5;
               signR_d7 <=  signR_d6;
               signR_d8 <=  signR_d7;
               signR_d9 <=  signR_d8;
               sticky_d1 <=  sticky;
               EffSubVector_d1 <=  EffSubVector;
               extendedExpInc_d1 <=  extendedExpInc;
               extendedExpInc_d2 <=  extendedExpInc_d1;
               extendedExpInc_d3 <=  extendedExpInc_d2;
               extendedExpInc_d4 <=  extendedExpInc_d3;
               extendedExpInc_d5 <=  extendedExpInc_d4;
               extendedExpInc_d6 <=  extendedExpInc_d5;
               extendedExpInc_d7 <=  extendedExpInc_d6;
               extendedExpInc_d8 <=  extendedExpInc_d7;
               extendedExpInc_d9 <=  extendedExpInc_d8;
               eqdiffsign_d1 <=  eqdiffsign;
               eqdiffsign_d2 <=  eqdiffsign_d1;
               fracR_d1 <=  fracR;
               expR_d1 <=  expR;
               exExpExc_d1 <=  exExpExc;
               signR2_d1 <=  signR2;
               signR2_d2 <=  signR2_d1;
               X_d1 <=  X;
               Y_d1 <=  Y;
            end if;
         end if;
      end process;
   excExpFracX <= X(65 downto 64) & X(62 downto 0);
   excExpFracY <= Y(65 downto 64) & Y(62 downto 0);
   swap <= '1' when excExpFracX < excExpFracY else '0';
   -- exponent difference
   eXmeY <= (X(62 downto 52)) - (Y(62 downto 52));
   eYmeX <= (Y(62 downto 52)) - (X(62 downto 52));
   expDiff <= eXmeY_d1 when swap_d1 = '0' else eYmeX_d1;
   -- input swap so that |X|>|Y|
   newX <= X_d1 when swap_d1 = '0' else Y_d1;
   newY <= Y_d1 when swap_d1 = '0' else X_d1;
   -- now we decompose the inputs into their sign, exponent, fraction
   expX<= newX(62 downto 52);
   excX<= newX(65 downto 64);
   excY<= newY(65 downto 64);
   signX<= newX(63);
   signY<= newY(63);
   EffSub <= signX xor signY;
   sXsYExnXY <= signX & signY & excX & excY;
   sdExnXY <= excX & excY;
   fracY <= "00000000000000000000000000000000000000000000000000000" when excY="00" else ('1' & newY(51 downto 0));
   -- Exception management logic
   with sXsYExnXY  select  
   excRt <= "00" when "000000"|"010000"|"100000"|"110000",
      "01" when "000101"|"010101"|"100101"|"110101"|"000100"|"010100"|"100100"|"110100"|"000001"|"010001"|"100001"|"110001",
      "10" when "111010"|"001010"|"001000"|"011000"|"101000"|"111000"|"000010"|"010010"|"100010"|"110010"|"001001"|"011001"|"101001"|"111001"|"000110"|"010110"|"100110"|"110110", 
      "11" when others;
   signR<= '0' when (sXsYExnXY="100000" or sXsYExnXY="010000") else signX;
   shiftedOut <= '1' when (expDiff > 54) else '0';
   shiftVal <= expDiff(5 downto 0) when shiftedOut='0' else CONV_STD_LOGIC_VECTOR(55,6);
   RightShifterComponent: RightShifterSticky53_by_max_55_F500_uid12
      port map ( clk  => clk,
                 ce => ce,
                 S => shiftVal,
                 X => fracY,
                 R => shiftedFracY,
                 Sticky => sticky);
   fracYpad <= "0" & shiftedFracY;
   EffSubVector <= (55 downto 0 => EffSub);
   fracYpadXorOp <= fracYpad xor EffSubVector_d1;
   fracXpad <= "01" & (newX(51 downto 0)) & "00";
   cInSigAdd <= EffSub_d4 and not sticky; -- if we subtract and the sticky was one, some of the negated sticky bits would have absorbed this carry 
   fracAdder: IntAdder_56_F500_uid14
      port map ( clk  => clk,
                 ce => ce,
                 Cin => cInSigAdd,
                 X => fracXpad,
                 Y => fracYpadXorOp,
                 R => fracAddResult);
   fracSticky<= fracAddResult & sticky_d1; 
   LZCAndShifter: Normalizer_Z_57_57_57_F500_uid16
      port map ( clk  => clk,
                 ce => ce,
                 X => fracSticky,
                 Count => nZerosNew,
                 R => shiftedFrac);
   extendedExpInc<= ("0" & expX) + '1';
   updatedExp <= ("0" &extendedExpInc_d9) - ("0000000" & nZerosNew);
   eqdiffsign <= '1' when nZerosNew="111111" else '0';
   expFrac<= updatedExp & shiftedFrac(55 downto 3);
   stk<= shiftedFrac(2) or shiftedFrac(1) or shiftedFrac(0);
   rnd<= shiftedFrac(3);
   lsb<= shiftedFrac(4);
   needToRound<= '1' when (rnd='1' and stk='1') or (rnd='1' and stk='0' and lsb='1')
  else '0';
   roundingAdder: IntAdder_66_F500_uid19
      port map ( clk  => clk,
                 ce => ce,
                 Cin => needToRound,
                 X => expFrac,
                 Y => "000000000000000000000000000000000000000000000000000000000000000000",
                 R => RoundedExpFrac);
   -- possible update to exception bits
   upExc <= RoundedExpFrac(65 downto 64);
   fracR <= RoundedExpFrac(52 downto 1);
   expR <= RoundedExpFrac(63 downto 53);
   exExpExc <= upExc & excRt_d10;
   with exExpExc_d1  select  
   excRt2<= "00" when "0000"|"0100"|"1000"|"1100"|"1001"|"1101",
      "01" when "0001",
      "10" when "0010"|"0110"|"1010"|"1110"|"0101",
      "11" when others;
   excR <= "00" when (eqdiffsign_d2='1' and EffSub_d11='1'  and not(excRt_d11="11")) else excRt2;
   signR2 <= '0' when (eqdiffsign='1' and EffSub_d9='1') else signR_d9;
   computedR <= excR & signR2_d2 & expR_d1 & fracR_d1;
   R <= computedR;
end architecture;

--------------------------------------------------------------------------------
--                         Compressor_23_3_F500_uid53
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X1 X0
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity Compressor_23_3_F500_uid53 is
    port (X1 : in  std_logic_vector(1 downto 0);
          X0 : in  std_logic_vector(2 downto 0);
          R : out  std_logic_vector(2 downto 0)   );
end entity;

architecture arch of Compressor_23_3_F500_uid53 is
signal X :  std_logic_vector(4 downto 0);
signal R0 :  std_logic_vector(2 downto 0);
begin
   X <= X1 & X0 ;

   with X  select  R0 <= 
      "000" when "00000",
      "001" when "00001" | "00010" | "00100",
      "010" when "00011" | "00101" | "00110" | "01000" | "10000",
      "011" when "00111" | "01001" | "01010" | "01100" | "10001" | "10010" | "10100",
      "100" when "01011" | "01101" | "01110" | "10011" | "10101" | "10110" | "11000",
      "101" when "01111" | "10111" | "11001" | "11010" | "11100",
      "110" when "11011" | "11101" | "11110",
      "111" when "11111",
      "---" when others;
   R <= R0;
end architecture;

--------------------------------------------------------------------------------
--                         Compressor_3_2_F500_uid61
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X0
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity Compressor_3_2_F500_uid61 is
    port (X0 : in  std_logic_vector(2 downto 0);
          R : out  std_logic_vector(1 downto 0)   );
end entity;

architecture arch of Compressor_3_2_F500_uid61 is
signal X :  std_logic_vector(2 downto 0);
signal R0 :  std_logic_vector(1 downto 0);
begin
   X <= X0 ;

   with X  select  R0 <= 
      "00" when "000",
      "01" when "001" | "010" | "100",
      "10" when "011" | "101" | "110",
      "11" when "111",
      "--" when others;
   R <= R0;
end architecture;

--------------------------------------------------------------------------------
--                         Compressor_14_3_F500_uid75
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X1 X0
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity Compressor_14_3_F500_uid75 is
    port (X1 : in  std_logic_vector(0 downto 0);
          X0 : in  std_logic_vector(3 downto 0);
          R : out  std_logic_vector(2 downto 0)   );
end entity;

architecture arch of Compressor_14_3_F500_uid75 is
signal X :  std_logic_vector(4 downto 0);
signal R0 :  std_logic_vector(2 downto 0);
begin
   X <= X1 & X0 ;

   with X  select  R0 <= 
      "000" when "00000",
      "001" when "00001" | "00010" | "00100" | "01000",
      "010" when "00011" | "00101" | "00110" | "01001" | "01010" | "01100" | "10000",
      "011" when "00111" | "01011" | "01101" | "01110" | "10001" | "10010" | "10100" | "11000",
      "100" when "01111" | "10011" | "10101" | "10110" | "11001" | "11010" | "11100",
      "101" when "10111" | "11011" | "11101" | "11110",
      "110" when "11111",
      "---" when others;
   R <= R0;
end architecture;

--------------------------------------------------------------------------------
--                         Compressor_6_3_F500_uid111
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X0
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity Compressor_6_3_F500_uid111 is
    port (X0 : in  std_logic_vector(5 downto 0);
          R : out  std_logic_vector(2 downto 0)   );
end entity;

architecture arch of Compressor_6_3_F500_uid111 is
signal X :  std_logic_vector(5 downto 0);
signal R0 :  std_logic_vector(2 downto 0);
begin
   X <= X0 ;

   with X  select  R0 <= 
      "000" when "000000",
      "001" when "000001" | "000010" | "000100" | "001000" | "010000" | "100000",
      "010" when "000011" | "000101" | "000110" | "001001" | "001010" | "001100" | "010001" | "010010" | "010100" | "011000" | "100001" | "100010" | "100100" | "101000" | "110000",
      "011" when "000111" | "001011" | "001101" | "001110" | "010011" | "010101" | "010110" | "011001" | "011010" | "011100" | "100011" | "100101" | "100110" | "101001" | "101010" | "101100" | "110001" | "110010" | "110100" | "111000",
      "100" when "001111" | "010111" | "011011" | "011101" | "011110" | "100111" | "101011" | "101101" | "101110" | "110011" | "110101" | "110110" | "111001" | "111010" | "111100",
      "101" when "011111" | "101111" | "110111" | "111011" | "111101" | "111110",
      "110" when "111111",
      "---" when others;
   R <= R0;
end architecture;

--------------------------------------------------------------------------------
--                         DSPBlock_17x24_F500_uid28
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity DSPBlock_17x24_F500_uid28 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(16 downto 0);
          Y : in  std_logic_vector(23 downto 0);
          R : out  std_logic_vector(40 downto 0)   );
end entity;

architecture arch of DSPBlock_17x24_F500_uid28 is
signal Mint :  std_logic_vector(40 downto 0);
signal M :  std_logic_vector(40 downto 0);
signal Rtmp :  std_logic_vector(40 downto 0);
begin
   Mint <= std_logic_vector(unsigned(X) * unsigned(Y)); -- multiplier
   M <= Mint(40 downto 0);
   Rtmp <= M;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                         DSPBlock_17x24_F500_uid30
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity DSPBlock_17x24_F500_uid30 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(16 downto 0);
          Y : in  std_logic_vector(23 downto 0);
          R : out  std_logic_vector(40 downto 0)   );
end entity;

architecture arch of DSPBlock_17x24_F500_uid30 is
signal Mint :  std_logic_vector(40 downto 0);
signal M :  std_logic_vector(40 downto 0);
signal Rtmp :  std_logic_vector(40 downto 0);
begin
   Mint <= std_logic_vector(unsigned(X) * unsigned(Y)); -- multiplier
   M <= Mint(40 downto 0);
   Rtmp <= M;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                         DSPBlock_17x24_F500_uid32
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity DSPBlock_17x24_F500_uid32 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(16 downto 0);
          Y : in  std_logic_vector(23 downto 0);
          R : out  std_logic_vector(40 downto 0)   );
end entity;

architecture arch of DSPBlock_17x24_F500_uid32 is
signal Mint :  std_logic_vector(40 downto 0);
signal M :  std_logic_vector(40 downto 0);
signal Rtmp :  std_logic_vector(40 downto 0);
begin
   Mint <= std_logic_vector(unsigned(X) * unsigned(Y)); -- multiplier
   M <= Mint(40 downto 0);
   Rtmp <= M;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                          DSPBlock_2x24_F500_uid34
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity DSPBlock_2x24_F500_uid34 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(1 downto 0);
          Y : in  std_logic_vector(23 downto 0);
          R : out  std_logic_vector(25 downto 0)   );
end entity;

architecture arch of DSPBlock_2x24_F500_uid34 is
signal Mint :  std_logic_vector(25 downto 0);
signal M :  std_logic_vector(25 downto 0);
signal Rtmp :  std_logic_vector(25 downto 0);
begin
   Mint <= std_logic_vector(unsigned(X) * unsigned(Y)); -- multiplier
   M <= Mint(25 downto 0);
   Rtmp <= M;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                         DSPBlock_17x24_F500_uid36
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity DSPBlock_17x24_F500_uid36 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(16 downto 0);
          Y : in  std_logic_vector(23 downto 0);
          R : out  std_logic_vector(40 downto 0)   );
end entity;

architecture arch of DSPBlock_17x24_F500_uid36 is
signal Mint :  std_logic_vector(40 downto 0);
signal M :  std_logic_vector(40 downto 0);
signal Rtmp :  std_logic_vector(40 downto 0);
begin
   Mint <= std_logic_vector(unsigned(X) * unsigned(Y)); -- multiplier
   M <= Mint(40 downto 0);
   Rtmp <= M;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                         DSPBlock_17x24_F500_uid38
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity DSPBlock_17x24_F500_uid38 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(16 downto 0);
          Y : in  std_logic_vector(23 downto 0);
          R : out  std_logic_vector(40 downto 0)   );
end entity;

architecture arch of DSPBlock_17x24_F500_uid38 is
signal Mint :  std_logic_vector(40 downto 0);
signal M :  std_logic_vector(40 downto 0);
signal Rtmp :  std_logic_vector(40 downto 0);
begin
   Mint <= std_logic_vector(unsigned(X) * unsigned(Y)); -- multiplier
   M <= Mint(40 downto 0);
   Rtmp <= M;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                         DSPBlock_17x24_F500_uid40
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity DSPBlock_17x24_F500_uid40 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(16 downto 0);
          Y : in  std_logic_vector(23 downto 0);
          R : out  std_logic_vector(40 downto 0)   );
end entity;

architecture arch of DSPBlock_17x24_F500_uid40 is
signal Mint :  std_logic_vector(40 downto 0);
signal M :  std_logic_vector(40 downto 0);
signal Rtmp :  std_logic_vector(40 downto 0);
begin
   Mint <= std_logic_vector(unsigned(X) * unsigned(Y)); -- multiplier
   M <= Mint(40 downto 0);
   Rtmp <= M;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                          DSPBlock_2x24_F500_uid42
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity DSPBlock_2x24_F500_uid42 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(1 downto 0);
          Y : in  std_logic_vector(23 downto 0);
          R : out  std_logic_vector(25 downto 0)   );
end entity;

architecture arch of DSPBlock_2x24_F500_uid42 is
signal Mint :  std_logic_vector(25 downto 0);
signal M :  std_logic_vector(25 downto 0);
signal Rtmp :  std_logic_vector(25 downto 0);
begin
   Mint <= std_logic_vector(unsigned(X) * unsigned(Y)); -- multiplier
   M <= Mint(25 downto 0);
   Rtmp <= M;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                          DSPBlock_17x5_F500_uid44
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity DSPBlock_17x5_F500_uid44 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(16 downto 0);
          Y : in  std_logic_vector(4 downto 0);
          R : out  std_logic_vector(21 downto 0)   );
end entity;

architecture arch of DSPBlock_17x5_F500_uid44 is
signal Mint :  std_logic_vector(21 downto 0);
signal M :  std_logic_vector(21 downto 0);
signal Rtmp :  std_logic_vector(21 downto 0);
begin
   Mint <= std_logic_vector(unsigned(X) * unsigned(Y)); -- multiplier
   M <= Mint(21 downto 0);
   Rtmp <= M;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                          DSPBlock_17x5_F500_uid46
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity DSPBlock_17x5_F500_uid46 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(16 downto 0);
          Y : in  std_logic_vector(4 downto 0);
          R : out  std_logic_vector(21 downto 0)   );
end entity;

architecture arch of DSPBlock_17x5_F500_uid46 is
signal Mint :  std_logic_vector(21 downto 0);
signal M :  std_logic_vector(21 downto 0);
signal Rtmp :  std_logic_vector(21 downto 0);
begin
   Mint <= std_logic_vector(unsigned(X) * unsigned(Y)); -- multiplier
   M <= Mint(21 downto 0);
   Rtmp <= M;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                          DSPBlock_17x5_F500_uid48
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity DSPBlock_17x5_F500_uid48 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(16 downto 0);
          Y : in  std_logic_vector(4 downto 0);
          R : out  std_logic_vector(21 downto 0)   );
end entity;

architecture arch of DSPBlock_17x5_F500_uid48 is
signal Mint :  std_logic_vector(21 downto 0);
signal M :  std_logic_vector(21 downto 0);
signal Rtmp :  std_logic_vector(21 downto 0);
begin
   Mint <= std_logic_vector(unsigned(X) * unsigned(Y)); -- multiplier
   M <= Mint(21 downto 0);
   Rtmp <= M;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                          DSPBlock_2x5_F500_uid50
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: 
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity DSPBlock_2x5_F500_uid50 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(1 downto 0);
          Y : in  std_logic_vector(4 downto 0);
          R : out  std_logic_vector(6 downto 0)   );
end entity;

architecture arch of DSPBlock_2x5_F500_uid50 is
signal Mint :  std_logic_vector(6 downto 0);
signal M :  std_logic_vector(6 downto 0);
signal Rtmp :  std_logic_vector(6 downto 0);
begin
   Mint <= std_logic_vector(unsigned(X) * unsigned(Y)); -- multiplier
   M <= Mint(6 downto 0);
   Rtmp <= M;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                          IntAdder_84_F500_uid379
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2016)
--------------------------------------------------------------------------------
-- Pipeline depth: 2 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y Cin
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntAdder_84_F500_uid379 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(83 downto 0);
          Y : in  std_logic_vector(83 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(83 downto 0)   );
end entity;

architecture arch of IntAdder_84_F500_uid379 is
signal Cin_0, Cin_0_d1 :  std_logic;
signal X_0, X_0_d1 :  std_logic_vector(67 downto 0);
signal Y_0, Y_0_d1 :  std_logic_vector(67 downto 0);
signal S_0 :  std_logic_vector(67 downto 0);
signal R_0, R_0_d1 :  std_logic_vector(66 downto 0);
signal Cin_1, Cin_1_d1 :  std_logic;
signal X_1, X_1_d1, X_1_d2 :  std_logic_vector(17 downto 0);
signal Y_1, Y_1_d1, Y_1_d2 :  std_logic_vector(17 downto 0);
signal S_1 :  std_logic_vector(17 downto 0);
signal R_1 :  std_logic_vector(16 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               Cin_0_d1 <=  Cin_0;
               X_0_d1 <=  X_0;
               Y_0_d1 <=  Y_0;
               R_0_d1 <=  R_0;
               Cin_1_d1 <=  Cin_1;
               X_1_d1 <=  X_1;
               X_1_d2 <=  X_1_d1;
               Y_1_d1 <=  Y_1;
               Y_1_d2 <=  Y_1_d1;
            end if;
         end if;
      end process;
   Cin_0 <= Cin;
   X_0 <= '0' & X(66 downto 0);
   Y_0 <= '0' & Y(66 downto 0);
   S_0 <= X_0_d1 + Y_0_d1 + Cin_0_d1;
   R_0 <= S_0(66 downto 0);
   Cin_1 <= S_0(67);
   X_1 <= '0' & X(83 downto 67);
   Y_1 <= '0' & Y(83 downto 67);
   S_1 <= X_1_d2 + Y_1_d2 + Cin_1_d1;
   R_1 <= S_1(16 downto 0);
   R <= R_1 & R_0_d1 ;
end architecture;

--------------------------------------------------------------------------------
--                          IntMultiplier_F500_uid24
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Martin Kumm, Florent de Dinechin, Kinga Illyes, Bogdan Popa, Bogdan Pasca, 2012
--------------------------------------------------------------------------------
-- Pipeline depth: 2 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity IntMultiplier_F500_uid24 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(52 downto 0);
          Y : in  std_logic_vector(52 downto 0);
          R : out  std_logic_vector(105 downto 0)   );
end entity;

architecture arch of IntMultiplier_F500_uid24 is
   component DSPBlock_17x24_F500_uid28 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(16 downto 0);
             Y : in  std_logic_vector(23 downto 0);
             R : out  std_logic_vector(40 downto 0)   );
   end component;

   component DSPBlock_17x24_F500_uid30 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(16 downto 0);
             Y : in  std_logic_vector(23 downto 0);
             R : out  std_logic_vector(40 downto 0)   );
   end component;

   component DSPBlock_17x24_F500_uid32 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(16 downto 0);
             Y : in  std_logic_vector(23 downto 0);
             R : out  std_logic_vector(40 downto 0)   );
   end component;

   component DSPBlock_2x24_F500_uid34 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(1 downto 0);
             Y : in  std_logic_vector(23 downto 0);
             R : out  std_logic_vector(25 downto 0)   );
   end component;

   component DSPBlock_17x24_F500_uid36 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(16 downto 0);
             Y : in  std_logic_vector(23 downto 0);
             R : out  std_logic_vector(40 downto 0)   );
   end component;

   component DSPBlock_17x24_F500_uid38 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(16 downto 0);
             Y : in  std_logic_vector(23 downto 0);
             R : out  std_logic_vector(40 downto 0)   );
   end component;

   component DSPBlock_17x24_F500_uid40 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(16 downto 0);
             Y : in  std_logic_vector(23 downto 0);
             R : out  std_logic_vector(40 downto 0)   );
   end component;

   component DSPBlock_2x24_F500_uid42 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(1 downto 0);
             Y : in  std_logic_vector(23 downto 0);
             R : out  std_logic_vector(25 downto 0)   );
   end component;

   component DSPBlock_17x5_F500_uid44 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(16 downto 0);
             Y : in  std_logic_vector(4 downto 0);
             R : out  std_logic_vector(21 downto 0)   );
   end component;

   component DSPBlock_17x5_F500_uid46 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(16 downto 0);
             Y : in  std_logic_vector(4 downto 0);
             R : out  std_logic_vector(21 downto 0)   );
   end component;

   component DSPBlock_17x5_F500_uid48 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(16 downto 0);
             Y : in  std_logic_vector(4 downto 0);
             R : out  std_logic_vector(21 downto 0)   );
   end component;

   component DSPBlock_2x5_F500_uid50 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(1 downto 0);
             Y : in  std_logic_vector(4 downto 0);
             R : out  std_logic_vector(6 downto 0)   );
   end component;

   component Compressor_23_3_F500_uid53 is
      port ( X1 : in  std_logic_vector(1 downto 0);
             X0 : in  std_logic_vector(2 downto 0);
             R : out  std_logic_vector(2 downto 0)   );
   end component;

   component Compressor_3_2_F500_uid61 is
      port ( X0 : in  std_logic_vector(2 downto 0);
             R : out  std_logic_vector(1 downto 0)   );
   end component;

   component Compressor_14_3_F500_uid75 is
      port ( X1 : in  std_logic_vector(0 downto 0);
             X0 : in  std_logic_vector(3 downto 0);
             R : out  std_logic_vector(2 downto 0)   );
   end component;

   component Compressor_6_3_F500_uid111 is
      port ( X0 : in  std_logic_vector(5 downto 0);
             R : out  std_logic_vector(2 downto 0)   );
   end component;

   component IntAdder_84_F500_uid379 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(83 downto 0);
             Y : in  std_logic_vector(83 downto 0);
             Cin : in  std_logic;
             R : out  std_logic_vector(83 downto 0)   );
   end component;

signal XX_m25 :  std_logic_vector(52 downto 0);
signal YY_m25 :  std_logic_vector(52 downto 0);
signal tile_0_X :  std_logic_vector(16 downto 0);
signal tile_0_Y :  std_logic_vector(23 downto 0);
signal tile_0_output :  std_logic_vector(40 downto 0);
signal tile_0_filtered_output :  unsigned(40-0 downto 0);
signal bh26_w0_0 :  std_logic;
signal bh26_w1_0 :  std_logic;
signal bh26_w2_0 :  std_logic;
signal bh26_w3_0 :  std_logic;
signal bh26_w4_0 :  std_logic;
signal bh26_w5_0 :  std_logic;
signal bh26_w6_0 :  std_logic;
signal bh26_w7_0 :  std_logic;
signal bh26_w8_0 :  std_logic;
signal bh26_w9_0 :  std_logic;
signal bh26_w10_0 :  std_logic;
signal bh26_w11_0 :  std_logic;
signal bh26_w12_0 :  std_logic;
signal bh26_w13_0 :  std_logic;
signal bh26_w14_0 :  std_logic;
signal bh26_w15_0 :  std_logic;
signal bh26_w16_0 :  std_logic;
signal bh26_w17_0 :  std_logic;
signal bh26_w18_0 :  std_logic;
signal bh26_w19_0 :  std_logic;
signal bh26_w20_0 :  std_logic;
signal bh26_w21_0 :  std_logic;
signal bh26_w22_0 :  std_logic;
signal bh26_w23_0 :  std_logic;
signal bh26_w24_0 :  std_logic;
signal bh26_w25_0 :  std_logic;
signal bh26_w26_0 :  std_logic;
signal bh26_w27_0 :  std_logic;
signal bh26_w28_0 :  std_logic;
signal bh26_w29_0 :  std_logic;
signal bh26_w30_0 :  std_logic;
signal bh26_w31_0 :  std_logic;
signal bh26_w32_0 :  std_logic;
signal bh26_w33_0 :  std_logic;
signal bh26_w34_0 :  std_logic;
signal bh26_w35_0 :  std_logic;
signal bh26_w36_0 :  std_logic;
signal bh26_w37_0 :  std_logic;
signal bh26_w38_0 :  std_logic;
signal bh26_w39_0 :  std_logic;
signal bh26_w40_0 :  std_logic;
signal tile_1_X :  std_logic_vector(16 downto 0);
signal tile_1_Y :  std_logic_vector(23 downto 0);
signal tile_1_output :  std_logic_vector(40 downto 0);
signal tile_1_filtered_output :  unsigned(40-0 downto 0);
signal bh26_w17_1 :  std_logic;
signal bh26_w18_1 :  std_logic;
signal bh26_w19_1 :  std_logic;
signal bh26_w20_1 :  std_logic;
signal bh26_w21_1 :  std_logic;
signal bh26_w22_1 :  std_logic;
signal bh26_w23_1 :  std_logic;
signal bh26_w24_1 :  std_logic;
signal bh26_w25_1 :  std_logic;
signal bh26_w26_1 :  std_logic;
signal bh26_w27_1 :  std_logic;
signal bh26_w28_1 :  std_logic;
signal bh26_w29_1 :  std_logic;
signal bh26_w30_1 :  std_logic;
signal bh26_w31_1 :  std_logic;
signal bh26_w32_1 :  std_logic;
signal bh26_w33_1 :  std_logic;
signal bh26_w34_1 :  std_logic;
signal bh26_w35_1 :  std_logic;
signal bh26_w36_1 :  std_logic;
signal bh26_w37_1 :  std_logic;
signal bh26_w38_1 :  std_logic;
signal bh26_w39_1 :  std_logic;
signal bh26_w40_1 :  std_logic;
signal bh26_w41_0 :  std_logic;
signal bh26_w42_0 :  std_logic;
signal bh26_w43_0 :  std_logic;
signal bh26_w44_0 :  std_logic;
signal bh26_w45_0 :  std_logic;
signal bh26_w46_0 :  std_logic;
signal bh26_w47_0 :  std_logic;
signal bh26_w48_0 :  std_logic;
signal bh26_w49_0 :  std_logic;
signal bh26_w50_0 :  std_logic;
signal bh26_w51_0 :  std_logic;
signal bh26_w52_0 :  std_logic;
signal bh26_w53_0 :  std_logic;
signal bh26_w54_0 :  std_logic;
signal bh26_w55_0 :  std_logic;
signal bh26_w56_0 :  std_logic;
signal bh26_w57_0 :  std_logic;
signal tile_2_X :  std_logic_vector(16 downto 0);
signal tile_2_Y :  std_logic_vector(23 downto 0);
signal tile_2_output :  std_logic_vector(40 downto 0);
signal tile_2_filtered_output :  unsigned(40-0 downto 0);
signal bh26_w34_2 :  std_logic;
signal bh26_w35_2 :  std_logic;
signal bh26_w36_2 :  std_logic;
signal bh26_w37_2 :  std_logic;
signal bh26_w38_2 :  std_logic;
signal bh26_w39_2 :  std_logic;
signal bh26_w40_2 :  std_logic;
signal bh26_w41_1 :  std_logic;
signal bh26_w42_1 :  std_logic;
signal bh26_w43_1 :  std_logic;
signal bh26_w44_1 :  std_logic;
signal bh26_w45_1 :  std_logic;
signal bh26_w46_1 :  std_logic;
signal bh26_w47_1 :  std_logic;
signal bh26_w48_1 :  std_logic;
signal bh26_w49_1 :  std_logic;
signal bh26_w50_1 :  std_logic;
signal bh26_w51_1 :  std_logic;
signal bh26_w52_1 :  std_logic;
signal bh26_w53_1 :  std_logic;
signal bh26_w54_1 :  std_logic;
signal bh26_w55_1 :  std_logic;
signal bh26_w56_1 :  std_logic;
signal bh26_w57_1 :  std_logic;
signal bh26_w58_0 :  std_logic;
signal bh26_w59_0 :  std_logic;
signal bh26_w60_0 :  std_logic;
signal bh26_w61_0 :  std_logic;
signal bh26_w62_0 :  std_logic;
signal bh26_w63_0 :  std_logic;
signal bh26_w64_0 :  std_logic;
signal bh26_w65_0 :  std_logic;
signal bh26_w66_0 :  std_logic;
signal bh26_w67_0 :  std_logic;
signal bh26_w68_0 :  std_logic;
signal bh26_w69_0 :  std_logic;
signal bh26_w70_0 :  std_logic;
signal bh26_w71_0 :  std_logic;
signal bh26_w72_0 :  std_logic;
signal bh26_w73_0 :  std_logic;
signal bh26_w74_0 :  std_logic;
signal tile_3_X :  std_logic_vector(1 downto 0);
signal tile_3_Y :  std_logic_vector(23 downto 0);
signal tile_3_output :  std_logic_vector(25 downto 0);
signal tile_3_filtered_output :  unsigned(25-0 downto 0);
signal bh26_w51_2 :  std_logic;
signal bh26_w52_2 :  std_logic;
signal bh26_w53_2 :  std_logic;
signal bh26_w54_2 :  std_logic;
signal bh26_w55_2 :  std_logic;
signal bh26_w56_2 :  std_logic;
signal bh26_w57_2 :  std_logic;
signal bh26_w58_1 :  std_logic;
signal bh26_w59_1 :  std_logic;
signal bh26_w60_1 :  std_logic;
signal bh26_w61_1 :  std_logic;
signal bh26_w62_1 :  std_logic;
signal bh26_w63_1 :  std_logic;
signal bh26_w64_1 :  std_logic;
signal bh26_w65_1 :  std_logic;
signal bh26_w66_1 :  std_logic;
signal bh26_w67_1 :  std_logic;
signal bh26_w68_1 :  std_logic;
signal bh26_w69_1 :  std_logic;
signal bh26_w70_1 :  std_logic;
signal bh26_w71_1 :  std_logic;
signal bh26_w72_1 :  std_logic;
signal bh26_w73_1 :  std_logic;
signal bh26_w74_1 :  std_logic;
signal bh26_w75_0 :  std_logic;
signal bh26_w76_0 :  std_logic;
signal tile_4_X :  std_logic_vector(16 downto 0);
signal tile_4_Y :  std_logic_vector(23 downto 0);
signal tile_4_output :  std_logic_vector(40 downto 0);
signal tile_4_filtered_output :  unsigned(40-0 downto 0);
signal bh26_w24_2 :  std_logic;
signal bh26_w25_2 :  std_logic;
signal bh26_w26_2 :  std_logic;
signal bh26_w27_2 :  std_logic;
signal bh26_w28_2 :  std_logic;
signal bh26_w29_2 :  std_logic;
signal bh26_w30_2 :  std_logic;
signal bh26_w31_2 :  std_logic;
signal bh26_w32_2 :  std_logic;
signal bh26_w33_2 :  std_logic;
signal bh26_w34_3 :  std_logic;
signal bh26_w35_3 :  std_logic;
signal bh26_w36_3 :  std_logic;
signal bh26_w37_3 :  std_logic;
signal bh26_w38_3 :  std_logic;
signal bh26_w39_3 :  std_logic;
signal bh26_w40_3 :  std_logic;
signal bh26_w41_2 :  std_logic;
signal bh26_w42_2 :  std_logic;
signal bh26_w43_2 :  std_logic;
signal bh26_w44_2 :  std_logic;
signal bh26_w45_2 :  std_logic;
signal bh26_w46_2 :  std_logic;
signal bh26_w47_2 :  std_logic;
signal bh26_w48_2 :  std_logic;
signal bh26_w49_2 :  std_logic;
signal bh26_w50_2 :  std_logic;
signal bh26_w51_3 :  std_logic;
signal bh26_w52_3 :  std_logic;
signal bh26_w53_3 :  std_logic;
signal bh26_w54_3 :  std_logic;
signal bh26_w55_3 :  std_logic;
signal bh26_w56_3 :  std_logic;
signal bh26_w57_3 :  std_logic;
signal bh26_w58_2 :  std_logic;
signal bh26_w59_2 :  std_logic;
signal bh26_w60_2 :  std_logic;
signal bh26_w61_2 :  std_logic;
signal bh26_w62_2 :  std_logic;
signal bh26_w63_2 :  std_logic;
signal bh26_w64_2 :  std_logic;
signal tile_5_X :  std_logic_vector(16 downto 0);
signal tile_5_Y :  std_logic_vector(23 downto 0);
signal tile_5_output :  std_logic_vector(40 downto 0);
signal tile_5_filtered_output :  unsigned(40-0 downto 0);
signal bh26_w41_3 :  std_logic;
signal bh26_w42_3 :  std_logic;
signal bh26_w43_3 :  std_logic;
signal bh26_w44_3 :  std_logic;
signal bh26_w45_3 :  std_logic;
signal bh26_w46_3 :  std_logic;
signal bh26_w47_3 :  std_logic;
signal bh26_w48_3 :  std_logic;
signal bh26_w49_3 :  std_logic;
signal bh26_w50_3 :  std_logic;
signal bh26_w51_4 :  std_logic;
signal bh26_w52_4 :  std_logic;
signal bh26_w53_4 :  std_logic;
signal bh26_w54_4 :  std_logic;
signal bh26_w55_4 :  std_logic;
signal bh26_w56_4 :  std_logic;
signal bh26_w57_4 :  std_logic;
signal bh26_w58_3 :  std_logic;
signal bh26_w59_3 :  std_logic;
signal bh26_w60_3 :  std_logic;
signal bh26_w61_3 :  std_logic;
signal bh26_w62_3 :  std_logic;
signal bh26_w63_3 :  std_logic;
signal bh26_w64_3 :  std_logic;
signal bh26_w65_2 :  std_logic;
signal bh26_w66_2 :  std_logic;
signal bh26_w67_2 :  std_logic;
signal bh26_w68_2 :  std_logic;
signal bh26_w69_2 :  std_logic;
signal bh26_w70_2 :  std_logic;
signal bh26_w71_2 :  std_logic;
signal bh26_w72_2 :  std_logic;
signal bh26_w73_2 :  std_logic;
signal bh26_w74_2 :  std_logic;
signal bh26_w75_1 :  std_logic;
signal bh26_w76_1 :  std_logic;
signal bh26_w77_0 :  std_logic;
signal bh26_w78_0 :  std_logic;
signal bh26_w79_0 :  std_logic;
signal bh26_w80_0 :  std_logic;
signal bh26_w81_0 :  std_logic;
signal tile_6_X :  std_logic_vector(16 downto 0);
signal tile_6_Y :  std_logic_vector(23 downto 0);
signal tile_6_output :  std_logic_vector(40 downto 0);
signal tile_6_filtered_output :  unsigned(40-0 downto 0);
signal bh26_w58_4 :  std_logic;
signal bh26_w59_4 :  std_logic;
signal bh26_w60_4 :  std_logic;
signal bh26_w61_4 :  std_logic;
signal bh26_w62_4 :  std_logic;
signal bh26_w63_4 :  std_logic;
signal bh26_w64_4 :  std_logic;
signal bh26_w65_3 :  std_logic;
signal bh26_w66_3 :  std_logic;
signal bh26_w67_3 :  std_logic;
signal bh26_w68_3 :  std_logic;
signal bh26_w69_3 :  std_logic;
signal bh26_w70_3 :  std_logic;
signal bh26_w71_3 :  std_logic;
signal bh26_w72_3 :  std_logic;
signal bh26_w73_3 :  std_logic;
signal bh26_w74_3 :  std_logic;
signal bh26_w75_2 :  std_logic;
signal bh26_w76_2 :  std_logic;
signal bh26_w77_1 :  std_logic;
signal bh26_w78_1 :  std_logic;
signal bh26_w79_1 :  std_logic;
signal bh26_w80_1 :  std_logic;
signal bh26_w81_1 :  std_logic;
signal bh26_w82_0 :  std_logic;
signal bh26_w83_0 :  std_logic;
signal bh26_w84_0 :  std_logic;
signal bh26_w85_0 :  std_logic;
signal bh26_w86_0 :  std_logic;
signal bh26_w87_0 :  std_logic;
signal bh26_w88_0 :  std_logic;
signal bh26_w89_0 :  std_logic;
signal bh26_w90_0 :  std_logic;
signal bh26_w91_0 :  std_logic;
signal bh26_w92_0 :  std_logic;
signal bh26_w93_0 :  std_logic;
signal bh26_w94_0 :  std_logic;
signal bh26_w95_0 :  std_logic;
signal bh26_w96_0 :  std_logic;
signal bh26_w97_0 :  std_logic;
signal bh26_w98_0 :  std_logic;
signal tile_7_X :  std_logic_vector(1 downto 0);
signal tile_7_Y :  std_logic_vector(23 downto 0);
signal tile_7_output :  std_logic_vector(25 downto 0);
signal tile_7_filtered_output :  unsigned(25-0 downto 0);
signal bh26_w75_3 :  std_logic;
signal bh26_w76_3 :  std_logic;
signal bh26_w77_2 :  std_logic;
signal bh26_w78_2 :  std_logic;
signal bh26_w79_2 :  std_logic;
signal bh26_w80_2 :  std_logic;
signal bh26_w81_2 :  std_logic;
signal bh26_w82_1 :  std_logic;
signal bh26_w83_1 :  std_logic;
signal bh26_w84_1 :  std_logic;
signal bh26_w85_1 :  std_logic;
signal bh26_w86_1 :  std_logic;
signal bh26_w87_1 :  std_logic;
signal bh26_w88_1 :  std_logic;
signal bh26_w89_1 :  std_logic;
signal bh26_w90_1 :  std_logic;
signal bh26_w91_1 :  std_logic;
signal bh26_w92_1 :  std_logic;
signal bh26_w93_1 :  std_logic;
signal bh26_w94_1 :  std_logic;
signal bh26_w95_1 :  std_logic;
signal bh26_w96_1 :  std_logic;
signal bh26_w97_1 :  std_logic;
signal bh26_w98_1 :  std_logic;
signal bh26_w99_0 :  std_logic;
signal bh26_w100_0 :  std_logic;
signal tile_8_X :  std_logic_vector(16 downto 0);
signal tile_8_Y :  std_logic_vector(4 downto 0);
signal tile_8_output :  std_logic_vector(21 downto 0);
signal tile_8_filtered_output :  unsigned(21-0 downto 0);
signal bh26_w48_4 :  std_logic;
signal bh26_w49_4 :  std_logic;
signal bh26_w50_4 :  std_logic;
signal bh26_w51_5 :  std_logic;
signal bh26_w52_5 :  std_logic;
signal bh26_w53_5 :  std_logic;
signal bh26_w54_5 :  std_logic;
signal bh26_w55_5 :  std_logic;
signal bh26_w56_5 :  std_logic;
signal bh26_w57_5 :  std_logic;
signal bh26_w58_5 :  std_logic;
signal bh26_w59_5 :  std_logic;
signal bh26_w60_5 :  std_logic;
signal bh26_w61_5 :  std_logic;
signal bh26_w62_5 :  std_logic;
signal bh26_w63_5 :  std_logic;
signal bh26_w64_5 :  std_logic;
signal bh26_w65_4 :  std_logic;
signal bh26_w66_4 :  std_logic;
signal bh26_w67_4 :  std_logic;
signal bh26_w68_4 :  std_logic;
signal bh26_w69_4 :  std_logic;
signal tile_9_X :  std_logic_vector(16 downto 0);
signal tile_9_Y :  std_logic_vector(4 downto 0);
signal tile_9_output :  std_logic_vector(21 downto 0);
signal tile_9_filtered_output :  unsigned(21-0 downto 0);
signal bh26_w65_5 :  std_logic;
signal bh26_w66_5 :  std_logic;
signal bh26_w67_5 :  std_logic;
signal bh26_w68_5 :  std_logic;
signal bh26_w69_5 :  std_logic;
signal bh26_w70_4 :  std_logic;
signal bh26_w71_4 :  std_logic;
signal bh26_w72_4 :  std_logic;
signal bh26_w73_4 :  std_logic;
signal bh26_w74_4 :  std_logic;
signal bh26_w75_4 :  std_logic;
signal bh26_w76_4 :  std_logic;
signal bh26_w77_3 :  std_logic;
signal bh26_w78_3 :  std_logic;
signal bh26_w79_3 :  std_logic;
signal bh26_w80_3 :  std_logic;
signal bh26_w81_3 :  std_logic;
signal bh26_w82_2 :  std_logic;
signal bh26_w83_2 :  std_logic;
signal bh26_w84_2 :  std_logic;
signal bh26_w85_2 :  std_logic;
signal bh26_w86_2 :  std_logic;
signal tile_10_X :  std_logic_vector(16 downto 0);
signal tile_10_Y :  std_logic_vector(4 downto 0);
signal tile_10_output :  std_logic_vector(21 downto 0);
signal tile_10_filtered_output :  unsigned(21-0 downto 0);
signal bh26_w82_3 :  std_logic;
signal bh26_w83_3 :  std_logic;
signal bh26_w84_3 :  std_logic;
signal bh26_w85_3 :  std_logic;
signal bh26_w86_3 :  std_logic;
signal bh26_w87_2 :  std_logic;
signal bh26_w88_2 :  std_logic;
signal bh26_w89_2 :  std_logic;
signal bh26_w90_2 :  std_logic;
signal bh26_w91_2 :  std_logic;
signal bh26_w92_2 :  std_logic;
signal bh26_w93_2 :  std_logic;
signal bh26_w94_2 :  std_logic;
signal bh26_w95_2 :  std_logic;
signal bh26_w96_2 :  std_logic;
signal bh26_w97_2 :  std_logic;
signal bh26_w98_2 :  std_logic;
signal bh26_w99_1 :  std_logic;
signal bh26_w100_1 :  std_logic;
signal bh26_w101_0 :  std_logic;
signal bh26_w102_0 :  std_logic;
signal bh26_w103_0 :  std_logic;
signal tile_11_X :  std_logic_vector(1 downto 0);
signal tile_11_Y :  std_logic_vector(4 downto 0);
signal tile_11_output :  std_logic_vector(6 downto 0);
signal tile_11_filtered_output :  unsigned(6-0 downto 0);
signal bh26_w99_2 :  std_logic;
signal bh26_w100_2 :  std_logic;
signal bh26_w101_1 :  std_logic;
signal bh26_w102_1 :  std_logic;
signal bh26_w103_1 :  std_logic;
signal bh26_w104_0 :  std_logic;
signal bh26_w105_0 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid54_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid54_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid54_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid54_Out0_copy55 :  std_logic_vector(2 downto 0);
signal bh26_w17_2 :  std_logic;
signal bh26_w18_2 :  std_logic;
signal bh26_w19_2 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid56_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid56_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid56_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid56_Out0_copy57 :  std_logic_vector(2 downto 0);
signal bh26_w19_3 :  std_logic;
signal bh26_w20_2 :  std_logic;
signal bh26_w21_2 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid58_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid58_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid58_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid58_Out0_copy59 :  std_logic_vector(2 downto 0);
signal bh26_w21_3 :  std_logic;
signal bh26_w22_2 :  std_logic;
signal bh26_w23_2 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid62_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid62_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid62_Out0_copy63 :  std_logic_vector(1 downto 0);
signal bh26_w23_3 :  std_logic;
signal bh26_w24_3 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid64_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid64_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid64_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid64_Out0_copy65 :  std_logic_vector(2 downto 0);
signal bh26_w24_4 :  std_logic;
signal bh26_w25_3 :  std_logic;
signal bh26_w26_3 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid66_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid66_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid66_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid66_Out0_copy67 :  std_logic_vector(2 downto 0);
signal bh26_w26_4 :  std_logic;
signal bh26_w27_3 :  std_logic;
signal bh26_w28_3 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid68_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid68_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid68_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid68_Out0_copy69 :  std_logic_vector(2 downto 0);
signal bh26_w28_4 :  std_logic;
signal bh26_w29_3 :  std_logic;
signal bh26_w30_3 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid70_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid70_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid70_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid70_Out0_copy71 :  std_logic_vector(2 downto 0);
signal bh26_w30_4 :  std_logic;
signal bh26_w31_3 :  std_logic;
signal bh26_w32_3 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid72_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid72_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid72_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid72_Out0_copy73 :  std_logic_vector(2 downto 0);
signal bh26_w32_4 :  std_logic;
signal bh26_w33_3 :  std_logic;
signal bh26_w34_4 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid76_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid76_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid76_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid76_Out0_copy77 :  std_logic_vector(2 downto 0);
signal bh26_w34_5 :  std_logic;
signal bh26_w35_4 :  std_logic;
signal bh26_w36_4 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid78_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid78_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid78_Out0_copy79 :  std_logic_vector(1 downto 0);
signal bh26_w35_5 :  std_logic;
signal bh26_w36_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid80_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid80_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid80_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid80_Out0_copy81 :  std_logic_vector(2 downto 0);
signal bh26_w36_6 :  std_logic;
signal bh26_w37_4 :  std_logic;
signal bh26_w38_4 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid82_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid82_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid82_Out0_copy83 :  std_logic_vector(1 downto 0);
signal bh26_w37_5 :  std_logic;
signal bh26_w38_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid84_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid84_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid84_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid84_Out0_copy85 :  std_logic_vector(2 downto 0);
signal bh26_w38_6 :  std_logic;
signal bh26_w39_4 :  std_logic;
signal bh26_w40_4 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid86_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid86_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid86_Out0_copy87 :  std_logic_vector(1 downto 0);
signal bh26_w39_5 :  std_logic;
signal bh26_w40_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid88_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid88_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid88_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid88_Out0_copy89 :  std_logic_vector(2 downto 0);
signal bh26_w40_6 :  std_logic;
signal bh26_w41_4 :  std_logic;
signal bh26_w42_4 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid90_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid90_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid90_Out0_copy91 :  std_logic_vector(1 downto 0);
signal bh26_w41_5 :  std_logic;
signal bh26_w42_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid92_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid92_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid92_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid92_Out0_copy93 :  std_logic_vector(2 downto 0);
signal bh26_w42_6 :  std_logic;
signal bh26_w43_4 :  std_logic;
signal bh26_w44_4 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid94_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid94_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid94_Out0_copy95 :  std_logic_vector(1 downto 0);
signal bh26_w43_5 :  std_logic;
signal bh26_w44_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid96_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid96_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid96_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid96_Out0_copy97 :  std_logic_vector(2 downto 0);
signal bh26_w44_6 :  std_logic;
signal bh26_w45_4 :  std_logic;
signal bh26_w46_4 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid98_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid98_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid98_Out0_copy99 :  std_logic_vector(1 downto 0);
signal bh26_w45_5 :  std_logic;
signal bh26_w46_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid100_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid100_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid100_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid100_Out0_copy101 :  std_logic_vector(2 downto 0);
signal bh26_w46_6 :  std_logic;
signal bh26_w47_4 :  std_logic;
signal bh26_w48_5 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid102_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid102_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid102_Out0_copy103 :  std_logic_vector(1 downto 0);
signal bh26_w47_5 :  std_logic;
signal bh26_w48_6 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid104_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid104_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid104_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid104_Out0_copy105 :  std_logic_vector(2 downto 0);
signal bh26_w48_7 :  std_logic;
signal bh26_w49_5 :  std_logic;
signal bh26_w50_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid106_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid106_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid106_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid106_Out0_copy107 :  std_logic_vector(2 downto 0);
signal bh26_w49_6 :  std_logic;
signal bh26_w50_6 :  std_logic;
signal bh26_w51_6 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid108_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid108_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid108_Out0_copy109 :  std_logic_vector(1 downto 0);
signal bh26_w50_7 :  std_logic;
signal bh26_w51_7 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid112_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid112_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid112_Out0_copy113 :  std_logic_vector(2 downto 0);
signal bh26_w51_8 :  std_logic;
signal bh26_w52_6 :  std_logic;
signal bh26_w53_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid114_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid114_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid114_Out0_copy115 :  std_logic_vector(2 downto 0);
signal bh26_w52_7 :  std_logic;
signal bh26_w53_7 :  std_logic;
signal bh26_w54_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid116_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid116_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid116_Out0_copy117 :  std_logic_vector(2 downto 0);
signal bh26_w53_8 :  std_logic;
signal bh26_w54_7 :  std_logic;
signal bh26_w55_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid118_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid118_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid118_Out0_copy119 :  std_logic_vector(2 downto 0);
signal bh26_w54_8 :  std_logic;
signal bh26_w55_7 :  std_logic;
signal bh26_w56_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid120_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid120_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid120_Out0_copy121 :  std_logic_vector(2 downto 0);
signal bh26_w55_8 :  std_logic;
signal bh26_w56_7 :  std_logic;
signal bh26_w57_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid122_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid122_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid122_Out0_copy123 :  std_logic_vector(2 downto 0);
signal bh26_w56_8 :  std_logic;
signal bh26_w57_7 :  std_logic;
signal bh26_w58_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid124_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid124_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid124_Out0_copy125 :  std_logic_vector(2 downto 0);
signal bh26_w57_8 :  std_logic;
signal bh26_w58_7 :  std_logic;
signal bh26_w59_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid126_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid126_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid126_Out0_copy127 :  std_logic_vector(2 downto 0);
signal bh26_w58_8 :  std_logic;
signal bh26_w59_7 :  std_logic;
signal bh26_w60_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid128_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid128_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid128_Out0_copy129 :  std_logic_vector(2 downto 0);
signal bh26_w59_8 :  std_logic;
signal bh26_w60_7 :  std_logic;
signal bh26_w61_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid130_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid130_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid130_Out0_copy131 :  std_logic_vector(2 downto 0);
signal bh26_w60_8 :  std_logic;
signal bh26_w61_7 :  std_logic;
signal bh26_w62_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid132_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid132_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid132_Out0_copy133 :  std_logic_vector(2 downto 0);
signal bh26_w61_8 :  std_logic;
signal bh26_w62_7 :  std_logic;
signal bh26_w63_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid134_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid134_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid134_Out0_copy135 :  std_logic_vector(2 downto 0);
signal bh26_w62_8 :  std_logic;
signal bh26_w63_7 :  std_logic;
signal bh26_w64_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid136_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid136_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid136_Out0_copy137 :  std_logic_vector(2 downto 0);
signal bh26_w63_8 :  std_logic;
signal bh26_w64_7 :  std_logic;
signal bh26_w65_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid138_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid138_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid138_Out0_copy139 :  std_logic_vector(2 downto 0);
signal bh26_w64_8 :  std_logic;
signal bh26_w65_7 :  std_logic;
signal bh26_w66_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid140_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid140_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid140_Out0_copy141 :  std_logic_vector(2 downto 0);
signal bh26_w65_8 :  std_logic;
signal bh26_w66_7 :  std_logic;
signal bh26_w67_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid142_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid142_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid142_Out0_copy143 :  std_logic_vector(2 downto 0);
signal bh26_w66_8 :  std_logic;
signal bh26_w67_7 :  std_logic;
signal bh26_w68_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid144_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid144_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid144_Out0_copy145 :  std_logic_vector(2 downto 0);
signal bh26_w67_8 :  std_logic;
signal bh26_w68_7 :  std_logic;
signal bh26_w69_6 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid146_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid146_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid146_Out0_copy147 :  std_logic_vector(2 downto 0);
signal bh26_w68_8 :  std_logic;
signal bh26_w69_7 :  std_logic;
signal bh26_w70_5 :  std_logic;
signal Compressor_6_3_F500_uid111_bh26_uid148_In0 :  std_logic_vector(5 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid148_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_6_3_F500_uid111_bh26_uid148_Out0_copy149 :  std_logic_vector(2 downto 0);
signal bh26_w69_8 :  std_logic;
signal bh26_w70_6 :  std_logic;
signal bh26_w71_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid150_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid150_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid150_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid150_Out0_copy151 :  std_logic_vector(2 downto 0);
signal bh26_w70_7 :  std_logic;
signal bh26_w71_6 :  std_logic;
signal bh26_w72_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid152_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid152_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid152_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid152_Out0_copy153 :  std_logic_vector(2 downto 0);
signal bh26_w71_7 :  std_logic;
signal bh26_w72_6 :  std_logic;
signal bh26_w73_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid154_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid154_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid154_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid154_Out0_copy155 :  std_logic_vector(2 downto 0);
signal bh26_w72_7 :  std_logic;
signal bh26_w73_6 :  std_logic;
signal bh26_w74_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid156_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid156_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid156_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid156_Out0_copy157 :  std_logic_vector(2 downto 0);
signal bh26_w73_7 :  std_logic;
signal bh26_w74_6 :  std_logic;
signal bh26_w75_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid158_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid158_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid158_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid158_Out0_copy159 :  std_logic_vector(2 downto 0);
signal bh26_w74_7 :  std_logic;
signal bh26_w75_6 :  std_logic;
signal bh26_w76_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid160_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid160_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid160_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid160_Out0_copy161 :  std_logic_vector(2 downto 0);
signal bh26_w75_7 :  std_logic;
signal bh26_w76_6 :  std_logic;
signal bh26_w77_4 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid162_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid162_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid162_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid162_Out0_copy163 :  std_logic_vector(2 downto 0);
signal bh26_w76_7 :  std_logic;
signal bh26_w77_5 :  std_logic;
signal bh26_w78_4 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid164_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid164_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid164_Out0_copy165 :  std_logic_vector(1 downto 0);
signal bh26_w77_6 :  std_logic;
signal bh26_w78_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid166_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid166_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid166_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid166_Out0_copy167 :  std_logic_vector(2 downto 0);
signal bh26_w78_6 :  std_logic;
signal bh26_w79_4 :  std_logic;
signal bh26_w80_4 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid168_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid168_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid168_Out0_copy169 :  std_logic_vector(1 downto 0);
signal bh26_w79_5 :  std_logic;
signal bh26_w80_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid170_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid170_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid170_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid170_Out0_copy171 :  std_logic_vector(2 downto 0);
signal bh26_w80_6 :  std_logic;
signal bh26_w81_4 :  std_logic;
signal bh26_w82_4 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid172_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid172_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid172_Out0_copy173 :  std_logic_vector(1 downto 0);
signal bh26_w81_5 :  std_logic;
signal bh26_w82_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid174_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid174_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid174_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid174_Out0_copy175 :  std_logic_vector(2 downto 0);
signal bh26_w82_6 :  std_logic;
signal bh26_w83_4 :  std_logic;
signal bh26_w84_4 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid176_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid176_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid176_Out0_copy177 :  std_logic_vector(1 downto 0);
signal bh26_w83_5 :  std_logic;
signal bh26_w84_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid178_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid178_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid178_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid178_Out0_copy179 :  std_logic_vector(2 downto 0);
signal bh26_w84_6 :  std_logic;
signal bh26_w85_4 :  std_logic;
signal bh26_w86_4 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid180_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid180_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid180_Out0_copy181 :  std_logic_vector(1 downto 0);
signal bh26_w85_5 :  std_logic;
signal bh26_w86_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid182_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid182_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid182_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid182_Out0_copy183 :  std_logic_vector(2 downto 0);
signal bh26_w86_6 :  std_logic;
signal bh26_w87_3 :  std_logic;
signal bh26_w88_3 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid184_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid184_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid184_Out0_copy185 :  std_logic_vector(1 downto 0);
signal bh26_w87_4 :  std_logic;
signal bh26_w88_4 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid186_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid186_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid186_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid186_Out0_copy187 :  std_logic_vector(2 downto 0);
signal bh26_w88_5 :  std_logic;
signal bh26_w89_3 :  std_logic;
signal bh26_w90_3 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid188_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid188_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid188_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid188_Out0_copy189 :  std_logic_vector(2 downto 0);
signal bh26_w90_4 :  std_logic;
signal bh26_w91_3 :  std_logic;
signal bh26_w92_3 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid190_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid190_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid190_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid190_Out0_copy191 :  std_logic_vector(2 downto 0);
signal bh26_w92_4 :  std_logic;
signal bh26_w93_3 :  std_logic;
signal bh26_w94_3 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid192_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid192_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid192_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid192_Out0_copy193 :  std_logic_vector(2 downto 0);
signal bh26_w94_4 :  std_logic;
signal bh26_w95_3 :  std_logic;
signal bh26_w96_3 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid194_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid194_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid194_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid194_Out0_copy195 :  std_logic_vector(2 downto 0);
signal bh26_w96_4 :  std_logic;
signal bh26_w97_3 :  std_logic;
signal bh26_w98_3 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid196_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid196_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid196_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid196_Out0_copy197 :  std_logic_vector(2 downto 0);
signal bh26_w98_4 :  std_logic;
signal bh26_w99_3 :  std_logic;
signal bh26_w100_3 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid198_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid198_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid198_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid198_Out0_copy199 :  std_logic_vector(2 downto 0);
signal bh26_w100_4 :  std_logic;
signal bh26_w101_2 :  std_logic;
signal bh26_w102_2 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid200_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid200_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid200_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid200_Out0_copy201 :  std_logic_vector(2 downto 0);
signal bh26_w102_3 :  std_logic;
signal bh26_w103_2 :  std_logic;
signal bh26_w104_1 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid202_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid202_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid202_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid202_Out0_copy203 :  std_logic_vector(2 downto 0);
signal bh26_w19_4 :  std_logic;
signal bh26_w20_3 :  std_logic;
signal bh26_w21_4 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid204_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid204_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid204_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid204_Out0_copy205 :  std_logic_vector(2 downto 0);
signal bh26_w21_5 :  std_logic;
signal bh26_w22_3 :  std_logic;
signal bh26_w23_4 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid206_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid206_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid206_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid206_Out0_copy207 :  std_logic_vector(2 downto 0);
signal bh26_w23_5 :  std_logic;
signal bh26_w24_5 :  std_logic;
signal bh26_w25_4 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid208_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid208_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid208_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid208_Out0_copy209 :  std_logic_vector(2 downto 0);
signal bh26_w25_5 :  std_logic;
signal bh26_w26_5 :  std_logic;
signal bh26_w27_4 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid210_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid210_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid210_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid210_Out0_copy211 :  std_logic_vector(2 downto 0);
signal bh26_w27_5 :  std_logic;
signal bh26_w28_5 :  std_logic;
signal bh26_w29_4 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid212_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid212_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid212_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid212_Out0_copy213 :  std_logic_vector(2 downto 0);
signal bh26_w29_5 :  std_logic;
signal bh26_w30_5 :  std_logic;
signal bh26_w31_4 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid214_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid214_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid214_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid214_Out0_copy215 :  std_logic_vector(2 downto 0);
signal bh26_w31_5 :  std_logic;
signal bh26_w32_5 :  std_logic;
signal bh26_w33_4 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid216_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid216_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid216_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid216_Out0_copy217 :  std_logic_vector(2 downto 0);
signal bh26_w33_5 :  std_logic;
signal bh26_w34_6 :  std_logic;
signal bh26_w35_6 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid218_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid218_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid218_Out0_copy219 :  std_logic_vector(1 downto 0);
signal bh26_w35_7 :  std_logic;
signal bh26_w36_7 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid220_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid220_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid220_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid220_Out0_copy221 :  std_logic_vector(2 downto 0);
signal bh26_w36_8 :  std_logic;
signal bh26_w37_6 :  std_logic;
signal bh26_w38_7 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid222_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid222_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid222_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid222_Out0_copy223 :  std_logic_vector(2 downto 0);
signal bh26_w38_8 :  std_logic;
signal bh26_w39_6 :  std_logic;
signal bh26_w40_7 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid224_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid224_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid224_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid224_Out0_copy225 :  std_logic_vector(2 downto 0);
signal bh26_w40_8 :  std_logic;
signal bh26_w41_6 :  std_logic;
signal bh26_w42_7 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid226_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid226_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid226_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid226_Out0_copy227 :  std_logic_vector(2 downto 0);
signal bh26_w42_8 :  std_logic;
signal bh26_w43_6 :  std_logic;
signal bh26_w44_7 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid228_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid228_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid228_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid228_Out0_copy229 :  std_logic_vector(2 downto 0);
signal bh26_w44_8 :  std_logic;
signal bh26_w45_6 :  std_logic;
signal bh26_w46_7 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid230_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid230_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid230_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid230_Out0_copy231 :  std_logic_vector(2 downto 0);
signal bh26_w46_8 :  std_logic;
signal bh26_w47_6 :  std_logic;
signal bh26_w48_8 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid232_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid232_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid232_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid232_Out0_copy233 :  std_logic_vector(2 downto 0);
signal bh26_w48_9 :  std_logic;
signal bh26_w49_7 :  std_logic;
signal bh26_w50_8 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid234_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid234_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid234_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid234_Out0_copy235 :  std_logic_vector(2 downto 0);
signal bh26_w50_9 :  std_logic;
signal bh26_w51_9 :  std_logic;
signal bh26_w52_8 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid236_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid236_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid236_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid236_Out0_copy237 :  std_logic_vector(2 downto 0);
signal bh26_w51_10 :  std_logic;
signal bh26_w52_9 :  std_logic;
signal bh26_w53_9 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid238_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid238_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid238_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid238_Out0_copy239 :  std_logic_vector(2 downto 0);
signal bh26_w53_10 :  std_logic;
signal bh26_w54_9 :  std_logic;
signal bh26_w55_9 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid240_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid240_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid240_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid240_Out0_copy241 :  std_logic_vector(2 downto 0);
signal bh26_w55_10 :  std_logic;
signal bh26_w56_9 :  std_logic;
signal bh26_w57_9 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid242_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid242_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid242_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid242_Out0_copy243 :  std_logic_vector(2 downto 0);
signal bh26_w57_10 :  std_logic;
signal bh26_w58_9 :  std_logic;
signal bh26_w59_9 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid244_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid244_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid244_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid244_Out0_copy245 :  std_logic_vector(2 downto 0);
signal bh26_w59_10 :  std_logic;
signal bh26_w60_9 :  std_logic;
signal bh26_w61_9 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid246_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid246_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid246_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid246_Out0_copy247 :  std_logic_vector(2 downto 0);
signal bh26_w61_10 :  std_logic;
signal bh26_w62_9 :  std_logic;
signal bh26_w63_9 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid248_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid248_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid248_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid248_Out0_copy249 :  std_logic_vector(2 downto 0);
signal bh26_w63_10 :  std_logic;
signal bh26_w64_9 :  std_logic;
signal bh26_w65_9 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid250_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid250_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid250_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid250_Out0_copy251 :  std_logic_vector(2 downto 0);
signal bh26_w65_10 :  std_logic;
signal bh26_w66_9 :  std_logic;
signal bh26_w67_9 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid252_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid252_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid252_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid252_Out0_copy253 :  std_logic_vector(2 downto 0);
signal bh26_w67_10 :  std_logic;
signal bh26_w68_9 :  std_logic;
signal bh26_w69_9 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid254_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid254_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid254_Out0_copy255 :  std_logic_vector(1 downto 0);
signal bh26_w69_10 :  std_logic;
signal bh26_w70_8 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid256_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid256_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid256_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid256_Out0_copy257 :  std_logic_vector(2 downto 0);
signal bh26_w70_9 :  std_logic;
signal bh26_w71_8 :  std_logic;
signal bh26_w72_8 :  std_logic;
signal Compressor_3_2_F500_uid61_bh26_uid258_In0 :  std_logic_vector(2 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid258_Out0 :  std_logic_vector(1 downto 0);
signal Compressor_3_2_F500_uid61_bh26_uid258_Out0_copy259 :  std_logic_vector(1 downto 0);
signal bh26_w71_9 :  std_logic;
signal bh26_w72_9 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid260_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid260_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid260_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid260_Out0_copy261 :  std_logic_vector(2 downto 0);
signal bh26_w72_10 :  std_logic;
signal bh26_w73_8 :  std_logic;
signal bh26_w74_8 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid262_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid262_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid262_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid262_Out0_copy263 :  std_logic_vector(2 downto 0);
signal bh26_w74_9 :  std_logic;
signal bh26_w75_8 :  std_logic;
signal bh26_w76_8 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid264_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid264_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid264_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid264_Out0_copy265 :  std_logic_vector(2 downto 0);
signal bh26_w76_9 :  std_logic;
signal bh26_w77_7 :  std_logic;
signal bh26_w78_7 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid266_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid266_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid266_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid266_Out0_copy267 :  std_logic_vector(2 downto 0);
signal bh26_w78_8 :  std_logic;
signal bh26_w79_6 :  std_logic;
signal bh26_w80_7 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid268_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid268_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid268_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid268_Out0_copy269 :  std_logic_vector(2 downto 0);
signal bh26_w80_8 :  std_logic;
signal bh26_w81_6 :  std_logic;
signal bh26_w82_7 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid270_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid270_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid270_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid270_Out0_copy271 :  std_logic_vector(2 downto 0);
signal bh26_w82_8 :  std_logic;
signal bh26_w83_6 :  std_logic;
signal bh26_w84_7 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid272_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid272_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid272_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid272_Out0_copy273 :  std_logic_vector(2 downto 0);
signal bh26_w84_8 :  std_logic;
signal bh26_w85_6 :  std_logic;
signal bh26_w86_7 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid274_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid274_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid274_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid274_Out0_copy275 :  std_logic_vector(2 downto 0);
signal bh26_w86_8 :  std_logic;
signal bh26_w87_5 :  std_logic;
signal bh26_w88_6 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid276_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid276_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid276_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid276_Out0_copy277 :  std_logic_vector(2 downto 0);
signal bh26_w88_7 :  std_logic;
signal bh26_w89_4 :  std_logic;
signal bh26_w90_5 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid278_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid278_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid278_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid278_Out0_copy279 :  std_logic_vector(2 downto 0);
signal bh26_w90_6 :  std_logic;
signal bh26_w91_4 :  std_logic;
signal bh26_w92_5 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid280_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid280_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid280_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid280_Out0_copy281 :  std_logic_vector(2 downto 0);
signal bh26_w92_6 :  std_logic;
signal bh26_w93_4 :  std_logic;
signal bh26_w94_5 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid282_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid282_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid282_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid282_Out0_copy283 :  std_logic_vector(2 downto 0);
signal bh26_w94_6 :  std_logic;
signal bh26_w95_4 :  std_logic;
signal bh26_w96_5 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid284_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid284_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid284_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid284_Out0_copy285 :  std_logic_vector(2 downto 0);
signal bh26_w96_6 :  std_logic;
signal bh26_w97_4 :  std_logic;
signal bh26_w98_5 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid286_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid286_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid286_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid286_Out0_copy287 :  std_logic_vector(2 downto 0);
signal bh26_w98_6 :  std_logic;
signal bh26_w99_4 :  std_logic;
signal bh26_w100_5 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid288_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid288_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid288_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid288_Out0_copy289 :  std_logic_vector(2 downto 0);
signal bh26_w100_6 :  std_logic;
signal bh26_w101_3 :  std_logic;
signal bh26_w102_4 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid290_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid290_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid290_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid290_Out0_copy291 :  std_logic_vector(2 downto 0);
signal bh26_w102_5 :  std_logic;
signal bh26_w103_3 :  std_logic;
signal bh26_w104_2 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid292_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid292_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid292_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid292_Out0_copy293 :  std_logic_vector(2 downto 0);
signal bh26_w104_3 :  std_logic;
signal bh26_w105_1 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid294_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid294_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid294_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid294_Out0_copy295 :  std_logic_vector(2 downto 0);
signal bh26_w21_6 :  std_logic;
signal bh26_w22_4 :  std_logic;
signal bh26_w23_6 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid296_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid296_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid296_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid296_Out0_copy297 :  std_logic_vector(2 downto 0);
signal bh26_w23_7 :  std_logic;
signal bh26_w24_6 :  std_logic;
signal bh26_w25_6 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid298_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid298_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid298_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid298_Out0_copy299 :  std_logic_vector(2 downto 0);
signal bh26_w25_7 :  std_logic;
signal bh26_w26_6 :  std_logic;
signal bh26_w27_6 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid300_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid300_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid300_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid300_Out0_copy301 :  std_logic_vector(2 downto 0);
signal bh26_w27_7 :  std_logic;
signal bh26_w28_6 :  std_logic;
signal bh26_w29_6 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid302_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid302_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid302_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid302_Out0_copy303 :  std_logic_vector(2 downto 0);
signal bh26_w29_7 :  std_logic;
signal bh26_w30_6 :  std_logic;
signal bh26_w31_6 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid304_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid304_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid304_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid304_Out0_copy305 :  std_logic_vector(2 downto 0);
signal bh26_w31_7 :  std_logic;
signal bh26_w32_6 :  std_logic;
signal bh26_w33_6 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid306_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid306_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid306_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid306_Out0_copy307 :  std_logic_vector(2 downto 0);
signal bh26_w33_7 :  std_logic;
signal bh26_w34_7 :  std_logic;
signal bh26_w35_8 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid308_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid308_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid308_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid308_Out0_copy309 :  std_logic_vector(2 downto 0);
signal bh26_w35_9 :  std_logic;
signal bh26_w36_9 :  std_logic;
signal bh26_w37_7 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid310_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid310_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid310_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid310_Out0_copy311 :  std_logic_vector(2 downto 0);
signal bh26_w38_9 :  std_logic;
signal bh26_w39_7 :  std_logic;
signal bh26_w40_9 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid312_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid312_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid312_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid312_Out0_copy313 :  std_logic_vector(2 downto 0);
signal bh26_w40_10 :  std_logic;
signal bh26_w41_7 :  std_logic;
signal bh26_w42_9 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid314_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid314_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid314_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid314_Out0_copy315 :  std_logic_vector(2 downto 0);
signal bh26_w42_10 :  std_logic;
signal bh26_w43_7 :  std_logic;
signal bh26_w44_9 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid316_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid316_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid316_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid316_Out0_copy317 :  std_logic_vector(2 downto 0);
signal bh26_w44_10 :  std_logic;
signal bh26_w45_7 :  std_logic;
signal bh26_w46_9 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid318_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid318_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid318_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid318_Out0_copy319 :  std_logic_vector(2 downto 0);
signal bh26_w46_10 :  std_logic;
signal bh26_w47_7 :  std_logic;
signal bh26_w48_10 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid320_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid320_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid320_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid320_Out0_copy321 :  std_logic_vector(2 downto 0);
signal bh26_w48_11 :  std_logic;
signal bh26_w49_8 :  std_logic;
signal bh26_w50_10 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid322_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid322_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid322_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid322_Out0_copy323 :  std_logic_vector(2 downto 0);
signal bh26_w50_11 :  std_logic;
signal bh26_w51_11 :  std_logic;
signal bh26_w52_10 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid324_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid324_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid324_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid324_Out0_copy325 :  std_logic_vector(2 downto 0);
signal bh26_w52_11 :  std_logic;
signal bh26_w53_11 :  std_logic;
signal bh26_w54_10 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid326_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid326_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid326_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid326_Out0_copy327 :  std_logic_vector(2 downto 0);
signal bh26_w54_11 :  std_logic;
signal bh26_w55_11 :  std_logic;
signal bh26_w56_10 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid328_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid328_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid328_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid328_Out0_copy329 :  std_logic_vector(2 downto 0);
signal bh26_w56_11 :  std_logic;
signal bh26_w57_11 :  std_logic;
signal bh26_w58_10 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid330_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid330_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid330_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid330_Out0_copy331 :  std_logic_vector(2 downto 0);
signal bh26_w58_11 :  std_logic;
signal bh26_w59_11 :  std_logic;
signal bh26_w60_10 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid332_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid332_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid332_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid332_Out0_copy333 :  std_logic_vector(2 downto 0);
signal bh26_w60_11 :  std_logic;
signal bh26_w61_11 :  std_logic;
signal bh26_w62_10 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid334_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid334_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid334_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid334_Out0_copy335 :  std_logic_vector(2 downto 0);
signal bh26_w62_11 :  std_logic;
signal bh26_w63_11 :  std_logic;
signal bh26_w64_10 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid336_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid336_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid336_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid336_Out0_copy337 :  std_logic_vector(2 downto 0);
signal bh26_w64_11 :  std_logic;
signal bh26_w65_11 :  std_logic;
signal bh26_w66_10 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid338_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid338_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid338_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid338_Out0_copy339 :  std_logic_vector(2 downto 0);
signal bh26_w66_11 :  std_logic;
signal bh26_w67_11 :  std_logic;
signal bh26_w68_10 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid340_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid340_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid340_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid340_Out0_copy341 :  std_logic_vector(2 downto 0);
signal bh26_w68_11 :  std_logic;
signal bh26_w69_11 :  std_logic;
signal bh26_w70_10 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid342_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid342_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid342_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid342_Out0_copy343 :  std_logic_vector(2 downto 0);
signal bh26_w70_11 :  std_logic;
signal bh26_w71_10 :  std_logic;
signal bh26_w72_11 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid344_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid344_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid344_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid344_Out0_copy345 :  std_logic_vector(2 downto 0);
signal bh26_w72_12 :  std_logic;
signal bh26_w73_9 :  std_logic;
signal bh26_w74_10 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid346_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid346_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid346_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid346_Out0_copy347 :  std_logic_vector(2 downto 0);
signal bh26_w74_11 :  std_logic;
signal bh26_w75_9 :  std_logic;
signal bh26_w76_10 :  std_logic;
signal Compressor_23_3_F500_uid53_bh26_uid348_In0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid348_In1 :  std_logic_vector(1 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid348_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_23_3_F500_uid53_bh26_uid348_Out0_copy349 :  std_logic_vector(2 downto 0);
signal bh26_w76_11 :  std_logic;
signal bh26_w77_8 :  std_logic;
signal bh26_w78_9 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid350_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid350_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid350_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid350_Out0_copy351 :  std_logic_vector(2 downto 0);
signal bh26_w78_10 :  std_logic;
signal bh26_w79_7 :  std_logic;
signal bh26_w80_9 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid352_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid352_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid352_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid352_Out0_copy353 :  std_logic_vector(2 downto 0);
signal bh26_w80_10 :  std_logic;
signal bh26_w81_7 :  std_logic;
signal bh26_w82_9 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid354_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid354_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid354_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid354_Out0_copy355 :  std_logic_vector(2 downto 0);
signal bh26_w82_10 :  std_logic;
signal bh26_w83_7 :  std_logic;
signal bh26_w84_9 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid356_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid356_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid356_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid356_Out0_copy357 :  std_logic_vector(2 downto 0);
signal bh26_w84_10 :  std_logic;
signal bh26_w85_7 :  std_logic;
signal bh26_w86_9 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid358_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid358_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid358_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid358_Out0_copy359 :  std_logic_vector(2 downto 0);
signal bh26_w86_10 :  std_logic;
signal bh26_w87_6 :  std_logic;
signal bh26_w88_8 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid360_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid360_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid360_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid360_Out0_copy361 :  std_logic_vector(2 downto 0);
signal bh26_w88_9 :  std_logic;
signal bh26_w89_5 :  std_logic;
signal bh26_w90_7 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid362_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid362_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid362_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid362_Out0_copy363 :  std_logic_vector(2 downto 0);
signal bh26_w90_8 :  std_logic;
signal bh26_w91_5 :  std_logic;
signal bh26_w92_7 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid364_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid364_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid364_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid364_Out0_copy365 :  std_logic_vector(2 downto 0);
signal bh26_w92_8 :  std_logic;
signal bh26_w93_5 :  std_logic;
signal bh26_w94_7 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid366_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid366_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid366_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid366_Out0_copy367 :  std_logic_vector(2 downto 0);
signal bh26_w94_8 :  std_logic;
signal bh26_w95_5 :  std_logic;
signal bh26_w96_7 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid368_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid368_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid368_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid368_Out0_copy369 :  std_logic_vector(2 downto 0);
signal bh26_w96_8 :  std_logic;
signal bh26_w97_5 :  std_logic;
signal bh26_w98_7 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid370_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid370_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid370_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid370_Out0_copy371 :  std_logic_vector(2 downto 0);
signal bh26_w98_8 :  std_logic;
signal bh26_w99_5 :  std_logic;
signal bh26_w100_7 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid372_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid372_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid372_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid372_Out0_copy373 :  std_logic_vector(2 downto 0);
signal bh26_w100_8 :  std_logic;
signal bh26_w101_4 :  std_logic;
signal bh26_w102_6 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid374_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid374_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid374_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid374_Out0_copy375 :  std_logic_vector(2 downto 0);
signal bh26_w102_7 :  std_logic;
signal bh26_w103_4 :  std_logic;
signal bh26_w104_4 :  std_logic;
signal Compressor_14_3_F500_uid75_bh26_uid376_In0 :  std_logic_vector(3 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid376_In1 :  std_logic_vector(0 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid376_Out0 :  std_logic_vector(2 downto 0);
signal Compressor_14_3_F500_uid75_bh26_uid376_Out0_copy377 :  std_logic_vector(2 downto 0);
signal bh26_w104_5 :  std_logic;
signal bh26_w105_2 :  std_logic;
signal tmp_bitheapResult_bh26_22, tmp_bitheapResult_bh26_22_d1, tmp_bitheapResult_bh26_22_d2 :  std_logic_vector(22 downto 0);
signal bitheapFinalAdd_bh26_In0 :  std_logic_vector(83 downto 0);
signal bitheapFinalAdd_bh26_In1 :  std_logic_vector(83 downto 0);
signal bitheapFinalAdd_bh26_Cin :  std_logic;
signal bitheapFinalAdd_bh26_Out :  std_logic_vector(83 downto 0);
signal bitheapResult_bh26 :  std_logic_vector(105 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               tmp_bitheapResult_bh26_22_d1 <=  tmp_bitheapResult_bh26_22;
               tmp_bitheapResult_bh26_22_d2 <=  tmp_bitheapResult_bh26_22_d1;
            end if;
         end if;
      end process;
   XX_m25 <= X ;
   YY_m25 <= Y ;
   tile_0_X <= X(16 downto 0);
   tile_0_Y <= Y(23 downto 0);
   tile_0_mult: DSPBlock_17x24_F500_uid28
      port map ( clk  => clk,
                 ce => ce,
                 X => tile_0_X,
                 Y => tile_0_Y,
                 R => tile_0_output);

   tile_0_filtered_output <= unsigned(tile_0_output(40 downto 0));
   bh26_w0_0 <= tile_0_filtered_output(0);
   bh26_w1_0 <= tile_0_filtered_output(1);
   bh26_w2_0 <= tile_0_filtered_output(2);
   bh26_w3_0 <= tile_0_filtered_output(3);
   bh26_w4_0 <= tile_0_filtered_output(4);
   bh26_w5_0 <= tile_0_filtered_output(5);
   bh26_w6_0 <= tile_0_filtered_output(6);
   bh26_w7_0 <= tile_0_filtered_output(7);
   bh26_w8_0 <= tile_0_filtered_output(8);
   bh26_w9_0 <= tile_0_filtered_output(9);
   bh26_w10_0 <= tile_0_filtered_output(10);
   bh26_w11_0 <= tile_0_filtered_output(11);
   bh26_w12_0 <= tile_0_filtered_output(12);
   bh26_w13_0 <= tile_0_filtered_output(13);
   bh26_w14_0 <= tile_0_filtered_output(14);
   bh26_w15_0 <= tile_0_filtered_output(15);
   bh26_w16_0 <= tile_0_filtered_output(16);
   bh26_w17_0 <= tile_0_filtered_output(17);
   bh26_w18_0 <= tile_0_filtered_output(18);
   bh26_w19_0 <= tile_0_filtered_output(19);
   bh26_w20_0 <= tile_0_filtered_output(20);
   bh26_w21_0 <= tile_0_filtered_output(21);
   bh26_w22_0 <= tile_0_filtered_output(22);
   bh26_w23_0 <= tile_0_filtered_output(23);
   bh26_w24_0 <= tile_0_filtered_output(24);
   bh26_w25_0 <= tile_0_filtered_output(25);
   bh26_w26_0 <= tile_0_filtered_output(26);
   bh26_w27_0 <= tile_0_filtered_output(27);
   bh26_w28_0 <= tile_0_filtered_output(28);
   bh26_w29_0 <= tile_0_filtered_output(29);
   bh26_w30_0 <= tile_0_filtered_output(30);
   bh26_w31_0 <= tile_0_filtered_output(31);
   bh26_w32_0 <= tile_0_filtered_output(32);
   bh26_w33_0 <= tile_0_filtered_output(33);
   bh26_w34_0 <= tile_0_filtered_output(34);
   bh26_w35_0 <= tile_0_filtered_output(35);
   bh26_w36_0 <= tile_0_filtered_output(36);
   bh26_w37_0 <= tile_0_filtered_output(37);
   bh26_w38_0 <= tile_0_filtered_output(38);
   bh26_w39_0 <= tile_0_filtered_output(39);
   bh26_w40_0 <= tile_0_filtered_output(40);
   tile_1_X <= X(33 downto 17);
   tile_1_Y <= Y(23 downto 0);
   tile_1_mult: DSPBlock_17x24_F500_uid30
      port map ( clk  => clk,
                 ce => ce,
                 X => tile_1_X,
                 Y => tile_1_Y,
                 R => tile_1_output);

   tile_1_filtered_output <= unsigned(tile_1_output(40 downto 0));
   bh26_w17_1 <= tile_1_filtered_output(0);
   bh26_w18_1 <= tile_1_filtered_output(1);
   bh26_w19_1 <= tile_1_filtered_output(2);
   bh26_w20_1 <= tile_1_filtered_output(3);
   bh26_w21_1 <= tile_1_filtered_output(4);
   bh26_w22_1 <= tile_1_filtered_output(5);
   bh26_w23_1 <= tile_1_filtered_output(6);
   bh26_w24_1 <= tile_1_filtered_output(7);
   bh26_w25_1 <= tile_1_filtered_output(8);
   bh26_w26_1 <= tile_1_filtered_output(9);
   bh26_w27_1 <= tile_1_filtered_output(10);
   bh26_w28_1 <= tile_1_filtered_output(11);
   bh26_w29_1 <= tile_1_filtered_output(12);
   bh26_w30_1 <= tile_1_filtered_output(13);
   bh26_w31_1 <= tile_1_filtered_output(14);
   bh26_w32_1 <= tile_1_filtered_output(15);
   bh26_w33_1 <= tile_1_filtered_output(16);
   bh26_w34_1 <= tile_1_filtered_output(17);
   bh26_w35_1 <= tile_1_filtered_output(18);
   bh26_w36_1 <= tile_1_filtered_output(19);
   bh26_w37_1 <= tile_1_filtered_output(20);
   bh26_w38_1 <= tile_1_filtered_output(21);
   bh26_w39_1 <= tile_1_filtered_output(22);
   bh26_w40_1 <= tile_1_filtered_output(23);
   bh26_w41_0 <= tile_1_filtered_output(24);
   bh26_w42_0 <= tile_1_filtered_output(25);
   bh26_w43_0 <= tile_1_filtered_output(26);
   bh26_w44_0 <= tile_1_filtered_output(27);
   bh26_w45_0 <= tile_1_filtered_output(28);
   bh26_w46_0 <= tile_1_filtered_output(29);
   bh26_w47_0 <= tile_1_filtered_output(30);
   bh26_w48_0 <= tile_1_filtered_output(31);
   bh26_w49_0 <= tile_1_filtered_output(32);
   bh26_w50_0 <= tile_1_filtered_output(33);
   bh26_w51_0 <= tile_1_filtered_output(34);
   bh26_w52_0 <= tile_1_filtered_output(35);
   bh26_w53_0 <= tile_1_filtered_output(36);
   bh26_w54_0 <= tile_1_filtered_output(37);
   bh26_w55_0 <= tile_1_filtered_output(38);
   bh26_w56_0 <= tile_1_filtered_output(39);
   bh26_w57_0 <= tile_1_filtered_output(40);
   tile_2_X <= X(50 downto 34);
   tile_2_Y <= Y(23 downto 0);
   tile_2_mult: DSPBlock_17x24_F500_uid32
      port map ( clk  => clk,
                 ce => ce,
                 X => tile_2_X,
                 Y => tile_2_Y,
                 R => tile_2_output);

   tile_2_filtered_output <= unsigned(tile_2_output(40 downto 0));
   bh26_w34_2 <= tile_2_filtered_output(0);
   bh26_w35_2 <= tile_2_filtered_output(1);
   bh26_w36_2 <= tile_2_filtered_output(2);
   bh26_w37_2 <= tile_2_filtered_output(3);
   bh26_w38_2 <= tile_2_filtered_output(4);
   bh26_w39_2 <= tile_2_filtered_output(5);
   bh26_w40_2 <= tile_2_filtered_output(6);
   bh26_w41_1 <= tile_2_filtered_output(7);
   bh26_w42_1 <= tile_2_filtered_output(8);
   bh26_w43_1 <= tile_2_filtered_output(9);
   bh26_w44_1 <= tile_2_filtered_output(10);
   bh26_w45_1 <= tile_2_filtered_output(11);
   bh26_w46_1 <= tile_2_filtered_output(12);
   bh26_w47_1 <= tile_2_filtered_output(13);
   bh26_w48_1 <= tile_2_filtered_output(14);
   bh26_w49_1 <= tile_2_filtered_output(15);
   bh26_w50_1 <= tile_2_filtered_output(16);
   bh26_w51_1 <= tile_2_filtered_output(17);
   bh26_w52_1 <= tile_2_filtered_output(18);
   bh26_w53_1 <= tile_2_filtered_output(19);
   bh26_w54_1 <= tile_2_filtered_output(20);
   bh26_w55_1 <= tile_2_filtered_output(21);
   bh26_w56_1 <= tile_2_filtered_output(22);
   bh26_w57_1 <= tile_2_filtered_output(23);
   bh26_w58_0 <= tile_2_filtered_output(24);
   bh26_w59_0 <= tile_2_filtered_output(25);
   bh26_w60_0 <= tile_2_filtered_output(26);
   bh26_w61_0 <= tile_2_filtered_output(27);
   bh26_w62_0 <= tile_2_filtered_output(28);
   bh26_w63_0 <= tile_2_filtered_output(29);
   bh26_w64_0 <= tile_2_filtered_output(30);
   bh26_w65_0 <= tile_2_filtered_output(31);
   bh26_w66_0 <= tile_2_filtered_output(32);
   bh26_w67_0 <= tile_2_filtered_output(33);
   bh26_w68_0 <= tile_2_filtered_output(34);
   bh26_w69_0 <= tile_2_filtered_output(35);
   bh26_w70_0 <= tile_2_filtered_output(36);
   bh26_w71_0 <= tile_2_filtered_output(37);
   bh26_w72_0 <= tile_2_filtered_output(38);
   bh26_w73_0 <= tile_2_filtered_output(39);
   bh26_w74_0 <= tile_2_filtered_output(40);
   tile_3_X <= X(52 downto 51);
   tile_3_Y <= Y(23 downto 0);
   tile_3_mult: DSPBlock_2x24_F500_uid34
      port map ( clk  => clk,
                 ce => ce,
                 X => tile_3_X,
                 Y => tile_3_Y,
                 R => tile_3_output);

   tile_3_filtered_output <= unsigned(tile_3_output(25 downto 0));
   bh26_w51_2 <= tile_3_filtered_output(0);
   bh26_w52_2 <= tile_3_filtered_output(1);
   bh26_w53_2 <= tile_3_filtered_output(2);
   bh26_w54_2 <= tile_3_filtered_output(3);
   bh26_w55_2 <= tile_3_filtered_output(4);
   bh26_w56_2 <= tile_3_filtered_output(5);
   bh26_w57_2 <= tile_3_filtered_output(6);
   bh26_w58_1 <= tile_3_filtered_output(7);
   bh26_w59_1 <= tile_3_filtered_output(8);
   bh26_w60_1 <= tile_3_filtered_output(9);
   bh26_w61_1 <= tile_3_filtered_output(10);
   bh26_w62_1 <= tile_3_filtered_output(11);
   bh26_w63_1 <= tile_3_filtered_output(12);
   bh26_w64_1 <= tile_3_filtered_output(13);
   bh26_w65_1 <= tile_3_filtered_output(14);
   bh26_w66_1 <= tile_3_filtered_output(15);
   bh26_w67_1 <= tile_3_filtered_output(16);
   bh26_w68_1 <= tile_3_filtered_output(17);
   bh26_w69_1 <= tile_3_filtered_output(18);
   bh26_w70_1 <= tile_3_filtered_output(19);
   bh26_w71_1 <= tile_3_filtered_output(20);
   bh26_w72_1 <= tile_3_filtered_output(21);
   bh26_w73_1 <= tile_3_filtered_output(22);
   bh26_w74_1 <= tile_3_filtered_output(23);
   bh26_w75_0 <= tile_3_filtered_output(24);
   bh26_w76_0 <= tile_3_filtered_output(25);
   tile_4_X <= X(16 downto 0);
   tile_4_Y <= Y(47 downto 24);
   tile_4_mult: DSPBlock_17x24_F500_uid36
      port map ( clk  => clk,
                 ce => ce,
                 X => tile_4_X,
                 Y => tile_4_Y,
                 R => tile_4_output);

   tile_4_filtered_output <= unsigned(tile_4_output(40 downto 0));
   bh26_w24_2 <= tile_4_filtered_output(0);
   bh26_w25_2 <= tile_4_filtered_output(1);
   bh26_w26_2 <= tile_4_filtered_output(2);
   bh26_w27_2 <= tile_4_filtered_output(3);
   bh26_w28_2 <= tile_4_filtered_output(4);
   bh26_w29_2 <= tile_4_filtered_output(5);
   bh26_w30_2 <= tile_4_filtered_output(6);
   bh26_w31_2 <= tile_4_filtered_output(7);
   bh26_w32_2 <= tile_4_filtered_output(8);
   bh26_w33_2 <= tile_4_filtered_output(9);
   bh26_w34_3 <= tile_4_filtered_output(10);
   bh26_w35_3 <= tile_4_filtered_output(11);
   bh26_w36_3 <= tile_4_filtered_output(12);
   bh26_w37_3 <= tile_4_filtered_output(13);
   bh26_w38_3 <= tile_4_filtered_output(14);
   bh26_w39_3 <= tile_4_filtered_output(15);
   bh26_w40_3 <= tile_4_filtered_output(16);
   bh26_w41_2 <= tile_4_filtered_output(17);
   bh26_w42_2 <= tile_4_filtered_output(18);
   bh26_w43_2 <= tile_4_filtered_output(19);
   bh26_w44_2 <= tile_4_filtered_output(20);
   bh26_w45_2 <= tile_4_filtered_output(21);
   bh26_w46_2 <= tile_4_filtered_output(22);
   bh26_w47_2 <= tile_4_filtered_output(23);
   bh26_w48_2 <= tile_4_filtered_output(24);
   bh26_w49_2 <= tile_4_filtered_output(25);
   bh26_w50_2 <= tile_4_filtered_output(26);
   bh26_w51_3 <= tile_4_filtered_output(27);
   bh26_w52_3 <= tile_4_filtered_output(28);
   bh26_w53_3 <= tile_4_filtered_output(29);
   bh26_w54_3 <= tile_4_filtered_output(30);
   bh26_w55_3 <= tile_4_filtered_output(31);
   bh26_w56_3 <= tile_4_filtered_output(32);
   bh26_w57_3 <= tile_4_filtered_output(33);
   bh26_w58_2 <= tile_4_filtered_output(34);
   bh26_w59_2 <= tile_4_filtered_output(35);
   bh26_w60_2 <= tile_4_filtered_output(36);
   bh26_w61_2 <= tile_4_filtered_output(37);
   bh26_w62_2 <= tile_4_filtered_output(38);
   bh26_w63_2 <= tile_4_filtered_output(39);
   bh26_w64_2 <= tile_4_filtered_output(40);
   tile_5_X <= X(33 downto 17);
   tile_5_Y <= Y(47 downto 24);
   tile_5_mult: DSPBlock_17x24_F500_uid38
      port map ( clk  => clk,
                 ce => ce,
                 X => tile_5_X,
                 Y => tile_5_Y,
                 R => tile_5_output);

   tile_5_filtered_output <= unsigned(tile_5_output(40 downto 0));
   bh26_w41_3 <= tile_5_filtered_output(0);
   bh26_w42_3 <= tile_5_filtered_output(1);
   bh26_w43_3 <= tile_5_filtered_output(2);
   bh26_w44_3 <= tile_5_filtered_output(3);
   bh26_w45_3 <= tile_5_filtered_output(4);
   bh26_w46_3 <= tile_5_filtered_output(5);
   bh26_w47_3 <= tile_5_filtered_output(6);
   bh26_w48_3 <= tile_5_filtered_output(7);
   bh26_w49_3 <= tile_5_filtered_output(8);
   bh26_w50_3 <= tile_5_filtered_output(9);
   bh26_w51_4 <= tile_5_filtered_output(10);
   bh26_w52_4 <= tile_5_filtered_output(11);
   bh26_w53_4 <= tile_5_filtered_output(12);
   bh26_w54_4 <= tile_5_filtered_output(13);
   bh26_w55_4 <= tile_5_filtered_output(14);
   bh26_w56_4 <= tile_5_filtered_output(15);
   bh26_w57_4 <= tile_5_filtered_output(16);
   bh26_w58_3 <= tile_5_filtered_output(17);
   bh26_w59_3 <= tile_5_filtered_output(18);
   bh26_w60_3 <= tile_5_filtered_output(19);
   bh26_w61_3 <= tile_5_filtered_output(20);
   bh26_w62_3 <= tile_5_filtered_output(21);
   bh26_w63_3 <= tile_5_filtered_output(22);
   bh26_w64_3 <= tile_5_filtered_output(23);
   bh26_w65_2 <= tile_5_filtered_output(24);
   bh26_w66_2 <= tile_5_filtered_output(25);
   bh26_w67_2 <= tile_5_filtered_output(26);
   bh26_w68_2 <= tile_5_filtered_output(27);
   bh26_w69_2 <= tile_5_filtered_output(28);
   bh26_w70_2 <= tile_5_filtered_output(29);
   bh26_w71_2 <= tile_5_filtered_output(30);
   bh26_w72_2 <= tile_5_filtered_output(31);
   bh26_w73_2 <= tile_5_filtered_output(32);
   bh26_w74_2 <= tile_5_filtered_output(33);
   bh26_w75_1 <= tile_5_filtered_output(34);
   bh26_w76_1 <= tile_5_filtered_output(35);
   bh26_w77_0 <= tile_5_filtered_output(36);
   bh26_w78_0 <= tile_5_filtered_output(37);
   bh26_w79_0 <= tile_5_filtered_output(38);
   bh26_w80_0 <= tile_5_filtered_output(39);
   bh26_w81_0 <= tile_5_filtered_output(40);
   tile_6_X <= X(50 downto 34);
   tile_6_Y <= Y(47 downto 24);
   tile_6_mult: DSPBlock_17x24_F500_uid40
      port map ( clk  => clk,
                 ce => ce,
                 X => tile_6_X,
                 Y => tile_6_Y,
                 R => tile_6_output);

   tile_6_filtered_output <= unsigned(tile_6_output(40 downto 0));
   bh26_w58_4 <= tile_6_filtered_output(0);
   bh26_w59_4 <= tile_6_filtered_output(1);
   bh26_w60_4 <= tile_6_filtered_output(2);
   bh26_w61_4 <= tile_6_filtered_output(3);
   bh26_w62_4 <= tile_6_filtered_output(4);
   bh26_w63_4 <= tile_6_filtered_output(5);
   bh26_w64_4 <= tile_6_filtered_output(6);
   bh26_w65_3 <= tile_6_filtered_output(7);
   bh26_w66_3 <= tile_6_filtered_output(8);
   bh26_w67_3 <= tile_6_filtered_output(9);
   bh26_w68_3 <= tile_6_filtered_output(10);
   bh26_w69_3 <= tile_6_filtered_output(11);
   bh26_w70_3 <= tile_6_filtered_output(12);
   bh26_w71_3 <= tile_6_filtered_output(13);
   bh26_w72_3 <= tile_6_filtered_output(14);
   bh26_w73_3 <= tile_6_filtered_output(15);
   bh26_w74_3 <= tile_6_filtered_output(16);
   bh26_w75_2 <= tile_6_filtered_output(17);
   bh26_w76_2 <= tile_6_filtered_output(18);
   bh26_w77_1 <= tile_6_filtered_output(19);
   bh26_w78_1 <= tile_6_filtered_output(20);
   bh26_w79_1 <= tile_6_filtered_output(21);
   bh26_w80_1 <= tile_6_filtered_output(22);
   bh26_w81_1 <= tile_6_filtered_output(23);
   bh26_w82_0 <= tile_6_filtered_output(24);
   bh26_w83_0 <= tile_6_filtered_output(25);
   bh26_w84_0 <= tile_6_filtered_output(26);
   bh26_w85_0 <= tile_6_filtered_output(27);
   bh26_w86_0 <= tile_6_filtered_output(28);
   bh26_w87_0 <= tile_6_filtered_output(29);
   bh26_w88_0 <= tile_6_filtered_output(30);
   bh26_w89_0 <= tile_6_filtered_output(31);
   bh26_w90_0 <= tile_6_filtered_output(32);
   bh26_w91_0 <= tile_6_filtered_output(33);
   bh26_w92_0 <= tile_6_filtered_output(34);
   bh26_w93_0 <= tile_6_filtered_output(35);
   bh26_w94_0 <= tile_6_filtered_output(36);
   bh26_w95_0 <= tile_6_filtered_output(37);
   bh26_w96_0 <= tile_6_filtered_output(38);
   bh26_w97_0 <= tile_6_filtered_output(39);
   bh26_w98_0 <= tile_6_filtered_output(40);
   tile_7_X <= X(52 downto 51);
   tile_7_Y <= Y(47 downto 24);
   tile_7_mult: DSPBlock_2x24_F500_uid42
      port map ( clk  => clk,
                 ce => ce,
                 X => tile_7_X,
                 Y => tile_7_Y,
                 R => tile_7_output);

   tile_7_filtered_output <= unsigned(tile_7_output(25 downto 0));
   bh26_w75_3 <= tile_7_filtered_output(0);
   bh26_w76_3 <= tile_7_filtered_output(1);
   bh26_w77_2 <= tile_7_filtered_output(2);
   bh26_w78_2 <= tile_7_filtered_output(3);
   bh26_w79_2 <= tile_7_filtered_output(4);
   bh26_w80_2 <= tile_7_filtered_output(5);
   bh26_w81_2 <= tile_7_filtered_output(6);
   bh26_w82_1 <= tile_7_filtered_output(7);
   bh26_w83_1 <= tile_7_filtered_output(8);
   bh26_w84_1 <= tile_7_filtered_output(9);
   bh26_w85_1 <= tile_7_filtered_output(10);
   bh26_w86_1 <= tile_7_filtered_output(11);
   bh26_w87_1 <= tile_7_filtered_output(12);
   bh26_w88_1 <= tile_7_filtered_output(13);
   bh26_w89_1 <= tile_7_filtered_output(14);
   bh26_w90_1 <= tile_7_filtered_output(15);
   bh26_w91_1 <= tile_7_filtered_output(16);
   bh26_w92_1 <= tile_7_filtered_output(17);
   bh26_w93_1 <= tile_7_filtered_output(18);
   bh26_w94_1 <= tile_7_filtered_output(19);
   bh26_w95_1 <= tile_7_filtered_output(20);
   bh26_w96_1 <= tile_7_filtered_output(21);
   bh26_w97_1 <= tile_7_filtered_output(22);
   bh26_w98_1 <= tile_7_filtered_output(23);
   bh26_w99_0 <= tile_7_filtered_output(24);
   bh26_w100_0 <= tile_7_filtered_output(25);
   tile_8_X <= X(16 downto 0);
   tile_8_Y <= Y(52 downto 48);
   tile_8_mult: DSPBlock_17x5_F500_uid44
      port map ( clk  => clk,
                 ce => ce,
                 X => tile_8_X,
                 Y => tile_8_Y,
                 R => tile_8_output);

   tile_8_filtered_output <= unsigned(tile_8_output(21 downto 0));
   bh26_w48_4 <= tile_8_filtered_output(0);
   bh26_w49_4 <= tile_8_filtered_output(1);
   bh26_w50_4 <= tile_8_filtered_output(2);
   bh26_w51_5 <= tile_8_filtered_output(3);
   bh26_w52_5 <= tile_8_filtered_output(4);
   bh26_w53_5 <= tile_8_filtered_output(5);
   bh26_w54_5 <= tile_8_filtered_output(6);
   bh26_w55_5 <= tile_8_filtered_output(7);
   bh26_w56_5 <= tile_8_filtered_output(8);
   bh26_w57_5 <= tile_8_filtered_output(9);
   bh26_w58_5 <= tile_8_filtered_output(10);
   bh26_w59_5 <= tile_8_filtered_output(11);
   bh26_w60_5 <= tile_8_filtered_output(12);
   bh26_w61_5 <= tile_8_filtered_output(13);
   bh26_w62_5 <= tile_8_filtered_output(14);
   bh26_w63_5 <= tile_8_filtered_output(15);
   bh26_w64_5 <= tile_8_filtered_output(16);
   bh26_w65_4 <= tile_8_filtered_output(17);
   bh26_w66_4 <= tile_8_filtered_output(18);
   bh26_w67_4 <= tile_8_filtered_output(19);
   bh26_w68_4 <= tile_8_filtered_output(20);
   bh26_w69_4 <= tile_8_filtered_output(21);
   tile_9_X <= X(33 downto 17);
   tile_9_Y <= Y(52 downto 48);
   tile_9_mult: DSPBlock_17x5_F500_uid46
      port map ( clk  => clk,
                 ce => ce,
                 X => tile_9_X,
                 Y => tile_9_Y,
                 R => tile_9_output);

   tile_9_filtered_output <= unsigned(tile_9_output(21 downto 0));
   bh26_w65_5 <= tile_9_filtered_output(0);
   bh26_w66_5 <= tile_9_filtered_output(1);
   bh26_w67_5 <= tile_9_filtered_output(2);
   bh26_w68_5 <= tile_9_filtered_output(3);
   bh26_w69_5 <= tile_9_filtered_output(4);
   bh26_w70_4 <= tile_9_filtered_output(5);
   bh26_w71_4 <= tile_9_filtered_output(6);
   bh26_w72_4 <= tile_9_filtered_output(7);
   bh26_w73_4 <= tile_9_filtered_output(8);
   bh26_w74_4 <= tile_9_filtered_output(9);
   bh26_w75_4 <= tile_9_filtered_output(10);
   bh26_w76_4 <= tile_9_filtered_output(11);
   bh26_w77_3 <= tile_9_filtered_output(12);
   bh26_w78_3 <= tile_9_filtered_output(13);
   bh26_w79_3 <= tile_9_filtered_output(14);
   bh26_w80_3 <= tile_9_filtered_output(15);
   bh26_w81_3 <= tile_9_filtered_output(16);
   bh26_w82_2 <= tile_9_filtered_output(17);
   bh26_w83_2 <= tile_9_filtered_output(18);
   bh26_w84_2 <= tile_9_filtered_output(19);
   bh26_w85_2 <= tile_9_filtered_output(20);
   bh26_w86_2 <= tile_9_filtered_output(21);
   tile_10_X <= X(50 downto 34);
   tile_10_Y <= Y(52 downto 48);
   tile_10_mult: DSPBlock_17x5_F500_uid48
      port map ( clk  => clk,
                 ce => ce,
                 X => tile_10_X,
                 Y => tile_10_Y,
                 R => tile_10_output);

   tile_10_filtered_output <= unsigned(tile_10_output(21 downto 0));
   bh26_w82_3 <= tile_10_filtered_output(0);
   bh26_w83_3 <= tile_10_filtered_output(1);
   bh26_w84_3 <= tile_10_filtered_output(2);
   bh26_w85_3 <= tile_10_filtered_output(3);
   bh26_w86_3 <= tile_10_filtered_output(4);
   bh26_w87_2 <= tile_10_filtered_output(5);
   bh26_w88_2 <= tile_10_filtered_output(6);
   bh26_w89_2 <= tile_10_filtered_output(7);
   bh26_w90_2 <= tile_10_filtered_output(8);
   bh26_w91_2 <= tile_10_filtered_output(9);
   bh26_w92_2 <= tile_10_filtered_output(10);
   bh26_w93_2 <= tile_10_filtered_output(11);
   bh26_w94_2 <= tile_10_filtered_output(12);
   bh26_w95_2 <= tile_10_filtered_output(13);
   bh26_w96_2 <= tile_10_filtered_output(14);
   bh26_w97_2 <= tile_10_filtered_output(15);
   bh26_w98_2 <= tile_10_filtered_output(16);
   bh26_w99_1 <= tile_10_filtered_output(17);
   bh26_w100_1 <= tile_10_filtered_output(18);
   bh26_w101_0 <= tile_10_filtered_output(19);
   bh26_w102_0 <= tile_10_filtered_output(20);
   bh26_w103_0 <= tile_10_filtered_output(21);
   tile_11_X <= X(52 downto 51);
   tile_11_Y <= Y(52 downto 48);
   tile_11_mult: DSPBlock_2x5_F500_uid50
      port map ( clk  => clk,
                 ce => ce,
                 X => tile_11_X,
                 Y => tile_11_Y,
                 R => tile_11_output);

   tile_11_filtered_output <= unsigned(tile_11_output(6 downto 0));
   bh26_w99_2 <= tile_11_filtered_output(0);
   bh26_w100_2 <= tile_11_filtered_output(1);
   bh26_w101_1 <= tile_11_filtered_output(2);
   bh26_w102_1 <= tile_11_filtered_output(3);
   bh26_w103_1 <= tile_11_filtered_output(4);
   bh26_w104_0 <= tile_11_filtered_output(5);
   bh26_w105_0 <= tile_11_filtered_output(6);

   -- Adding the constant bits 
      -- All the constant bits are zero, nothing to add


   Compressor_23_3_F500_uid53_bh26_uid54_In0 <= "" & bh26_w17_0 & bh26_w17_1 & "0";
   Compressor_23_3_F500_uid53_bh26_uid54_In1 <= "" & bh26_w18_0 & bh26_w18_1;
   Compressor_23_3_F500_uid53_uid54: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid54_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid54_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid54_Out0_copy55);
   Compressor_23_3_F500_uid53_bh26_uid54_Out0 <= Compressor_23_3_F500_uid53_bh26_uid54_Out0_copy55; -- output copy to hold a pipeline register if needed

   bh26_w17_2 <= Compressor_23_3_F500_uid53_bh26_uid54_Out0(0);
   bh26_w18_2 <= Compressor_23_3_F500_uid53_bh26_uid54_Out0(1);
   bh26_w19_2 <= Compressor_23_3_F500_uid53_bh26_uid54_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid56_In0 <= "" & bh26_w19_0 & bh26_w19_1 & "0";
   Compressor_23_3_F500_uid53_bh26_uid56_In1 <= "" & bh26_w20_0 & bh26_w20_1;
   Compressor_23_3_F500_uid53_uid56: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid56_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid56_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid56_Out0_copy57);
   Compressor_23_3_F500_uid53_bh26_uid56_Out0 <= Compressor_23_3_F500_uid53_bh26_uid56_Out0_copy57; -- output copy to hold a pipeline register if needed

   bh26_w19_3 <= Compressor_23_3_F500_uid53_bh26_uid56_Out0(0);
   bh26_w20_2 <= Compressor_23_3_F500_uid53_bh26_uid56_Out0(1);
   bh26_w21_2 <= Compressor_23_3_F500_uid53_bh26_uid56_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid58_In0 <= "" & bh26_w21_0 & bh26_w21_1 & "0";
   Compressor_23_3_F500_uid53_bh26_uid58_In1 <= "" & bh26_w22_0 & bh26_w22_1;
   Compressor_23_3_F500_uid53_uid58: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid58_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid58_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid58_Out0_copy59);
   Compressor_23_3_F500_uid53_bh26_uid58_Out0 <= Compressor_23_3_F500_uid53_bh26_uid58_Out0_copy59; -- output copy to hold a pipeline register if needed

   bh26_w21_3 <= Compressor_23_3_F500_uid53_bh26_uid58_Out0(0);
   bh26_w22_2 <= Compressor_23_3_F500_uid53_bh26_uid58_Out0(1);
   bh26_w23_2 <= Compressor_23_3_F500_uid53_bh26_uid58_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid62_In0 <= "" & bh26_w23_0 & bh26_w23_1 & "0";
   Compressor_3_2_F500_uid61_uid62: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid62_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid62_Out0_copy63);
   Compressor_3_2_F500_uid61_bh26_uid62_Out0 <= Compressor_3_2_F500_uid61_bh26_uid62_Out0_copy63; -- output copy to hold a pipeline register if needed

   bh26_w23_3 <= Compressor_3_2_F500_uid61_bh26_uid62_Out0(0);
   bh26_w24_3 <= Compressor_3_2_F500_uid61_bh26_uid62_Out0(1);

   Compressor_23_3_F500_uid53_bh26_uid64_In0 <= "" & bh26_w24_0 & bh26_w24_1 & bh26_w24_2;
   Compressor_23_3_F500_uid53_bh26_uid64_In1 <= "" & bh26_w25_0 & bh26_w25_1;
   Compressor_23_3_F500_uid53_uid64: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid64_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid64_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid64_Out0_copy65);
   Compressor_23_3_F500_uid53_bh26_uid64_Out0 <= Compressor_23_3_F500_uid53_bh26_uid64_Out0_copy65; -- output copy to hold a pipeline register if needed

   bh26_w24_4 <= Compressor_23_3_F500_uid53_bh26_uid64_Out0(0);
   bh26_w25_3 <= Compressor_23_3_F500_uid53_bh26_uid64_Out0(1);
   bh26_w26_3 <= Compressor_23_3_F500_uid53_bh26_uid64_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid66_In0 <= "" & bh26_w26_0 & bh26_w26_1 & bh26_w26_2;
   Compressor_23_3_F500_uid53_bh26_uid66_In1 <= "" & bh26_w27_0 & bh26_w27_1;
   Compressor_23_3_F500_uid53_uid66: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid66_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid66_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid66_Out0_copy67);
   Compressor_23_3_F500_uid53_bh26_uid66_Out0 <= Compressor_23_3_F500_uid53_bh26_uid66_Out0_copy67; -- output copy to hold a pipeline register if needed

   bh26_w26_4 <= Compressor_23_3_F500_uid53_bh26_uid66_Out0(0);
   bh26_w27_3 <= Compressor_23_3_F500_uid53_bh26_uid66_Out0(1);
   bh26_w28_3 <= Compressor_23_3_F500_uid53_bh26_uid66_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid68_In0 <= "" & bh26_w28_0 & bh26_w28_1 & bh26_w28_2;
   Compressor_23_3_F500_uid53_bh26_uid68_In1 <= "" & bh26_w29_0 & bh26_w29_1;
   Compressor_23_3_F500_uid53_uid68: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid68_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid68_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid68_Out0_copy69);
   Compressor_23_3_F500_uid53_bh26_uid68_Out0 <= Compressor_23_3_F500_uid53_bh26_uid68_Out0_copy69; -- output copy to hold a pipeline register if needed

   bh26_w28_4 <= Compressor_23_3_F500_uid53_bh26_uid68_Out0(0);
   bh26_w29_3 <= Compressor_23_3_F500_uid53_bh26_uid68_Out0(1);
   bh26_w30_3 <= Compressor_23_3_F500_uid53_bh26_uid68_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid70_In0 <= "" & bh26_w30_0 & bh26_w30_1 & bh26_w30_2;
   Compressor_23_3_F500_uid53_bh26_uid70_In1 <= "" & bh26_w31_0 & bh26_w31_1;
   Compressor_23_3_F500_uid53_uid70: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid70_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid70_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid70_Out0_copy71);
   Compressor_23_3_F500_uid53_bh26_uid70_Out0 <= Compressor_23_3_F500_uid53_bh26_uid70_Out0_copy71; -- output copy to hold a pipeline register if needed

   bh26_w30_4 <= Compressor_23_3_F500_uid53_bh26_uid70_Out0(0);
   bh26_w31_3 <= Compressor_23_3_F500_uid53_bh26_uid70_Out0(1);
   bh26_w32_3 <= Compressor_23_3_F500_uid53_bh26_uid70_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid72_In0 <= "" & bh26_w32_0 & bh26_w32_1 & bh26_w32_2;
   Compressor_23_3_F500_uid53_bh26_uid72_In1 <= "" & bh26_w33_0 & bh26_w33_1;
   Compressor_23_3_F500_uid53_uid72: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid72_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid72_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid72_Out0_copy73);
   Compressor_23_3_F500_uid53_bh26_uid72_Out0 <= Compressor_23_3_F500_uid53_bh26_uid72_Out0_copy73; -- output copy to hold a pipeline register if needed

   bh26_w32_4 <= Compressor_23_3_F500_uid53_bh26_uid72_Out0(0);
   bh26_w33_3 <= Compressor_23_3_F500_uid53_bh26_uid72_Out0(1);
   bh26_w34_4 <= Compressor_23_3_F500_uid53_bh26_uid72_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid76_In0 <= "" & bh26_w34_0 & bh26_w34_1 & bh26_w34_2 & bh26_w34_3;
   Compressor_14_3_F500_uid75_bh26_uid76_In1 <= "" & bh26_w35_0;
   Compressor_14_3_F500_uid75_uid76: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid76_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid76_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid76_Out0_copy77);
   Compressor_14_3_F500_uid75_bh26_uid76_Out0 <= Compressor_14_3_F500_uid75_bh26_uid76_Out0_copy77; -- output copy to hold a pipeline register if needed

   bh26_w34_5 <= Compressor_14_3_F500_uid75_bh26_uid76_Out0(0);
   bh26_w35_4 <= Compressor_14_3_F500_uid75_bh26_uid76_Out0(1);
   bh26_w36_4 <= Compressor_14_3_F500_uid75_bh26_uid76_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid78_In0 <= "" & bh26_w35_1 & bh26_w35_2 & bh26_w35_3;
   Compressor_3_2_F500_uid61_uid78: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid78_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid78_Out0_copy79);
   Compressor_3_2_F500_uid61_bh26_uid78_Out0 <= Compressor_3_2_F500_uid61_bh26_uid78_Out0_copy79; -- output copy to hold a pipeline register if needed

   bh26_w35_5 <= Compressor_3_2_F500_uid61_bh26_uid78_Out0(0);
   bh26_w36_5 <= Compressor_3_2_F500_uid61_bh26_uid78_Out0(1);

   Compressor_14_3_F500_uid75_bh26_uid80_In0 <= "" & bh26_w36_0 & bh26_w36_1 & bh26_w36_2 & bh26_w36_3;
   Compressor_14_3_F500_uid75_bh26_uid80_In1 <= "" & bh26_w37_0;
   Compressor_14_3_F500_uid75_uid80: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid80_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid80_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid80_Out0_copy81);
   Compressor_14_3_F500_uid75_bh26_uid80_Out0 <= Compressor_14_3_F500_uid75_bh26_uid80_Out0_copy81; -- output copy to hold a pipeline register if needed

   bh26_w36_6 <= Compressor_14_3_F500_uid75_bh26_uid80_Out0(0);
   bh26_w37_4 <= Compressor_14_3_F500_uid75_bh26_uid80_Out0(1);
   bh26_w38_4 <= Compressor_14_3_F500_uid75_bh26_uid80_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid82_In0 <= "" & bh26_w37_1 & bh26_w37_2 & bh26_w37_3;
   Compressor_3_2_F500_uid61_uid82: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid82_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid82_Out0_copy83);
   Compressor_3_2_F500_uid61_bh26_uid82_Out0 <= Compressor_3_2_F500_uid61_bh26_uid82_Out0_copy83; -- output copy to hold a pipeline register if needed

   bh26_w37_5 <= Compressor_3_2_F500_uid61_bh26_uid82_Out0(0);
   bh26_w38_5 <= Compressor_3_2_F500_uid61_bh26_uid82_Out0(1);

   Compressor_14_3_F500_uid75_bh26_uid84_In0 <= "" & bh26_w38_0 & bh26_w38_1 & bh26_w38_2 & bh26_w38_3;
   Compressor_14_3_F500_uid75_bh26_uid84_In1 <= "" & bh26_w39_0;
   Compressor_14_3_F500_uid75_uid84: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid84_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid84_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid84_Out0_copy85);
   Compressor_14_3_F500_uid75_bh26_uid84_Out0 <= Compressor_14_3_F500_uid75_bh26_uid84_Out0_copy85; -- output copy to hold a pipeline register if needed

   bh26_w38_6 <= Compressor_14_3_F500_uid75_bh26_uid84_Out0(0);
   bh26_w39_4 <= Compressor_14_3_F500_uid75_bh26_uid84_Out0(1);
   bh26_w40_4 <= Compressor_14_3_F500_uid75_bh26_uid84_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid86_In0 <= "" & bh26_w39_1 & bh26_w39_2 & bh26_w39_3;
   Compressor_3_2_F500_uid61_uid86: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid86_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid86_Out0_copy87);
   Compressor_3_2_F500_uid61_bh26_uid86_Out0 <= Compressor_3_2_F500_uid61_bh26_uid86_Out0_copy87; -- output copy to hold a pipeline register if needed

   bh26_w39_5 <= Compressor_3_2_F500_uid61_bh26_uid86_Out0(0);
   bh26_w40_5 <= Compressor_3_2_F500_uid61_bh26_uid86_Out0(1);

   Compressor_14_3_F500_uid75_bh26_uid88_In0 <= "" & bh26_w40_0 & bh26_w40_1 & bh26_w40_2 & bh26_w40_3;
   Compressor_14_3_F500_uid75_bh26_uid88_In1 <= "" & bh26_w41_0;
   Compressor_14_3_F500_uid75_uid88: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid88_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid88_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid88_Out0_copy89);
   Compressor_14_3_F500_uid75_bh26_uid88_Out0 <= Compressor_14_3_F500_uid75_bh26_uid88_Out0_copy89; -- output copy to hold a pipeline register if needed

   bh26_w40_6 <= Compressor_14_3_F500_uid75_bh26_uid88_Out0(0);
   bh26_w41_4 <= Compressor_14_3_F500_uid75_bh26_uid88_Out0(1);
   bh26_w42_4 <= Compressor_14_3_F500_uid75_bh26_uid88_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid90_In0 <= "" & bh26_w41_1 & bh26_w41_2 & bh26_w41_3;
   Compressor_3_2_F500_uid61_uid90: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid90_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid90_Out0_copy91);
   Compressor_3_2_F500_uid61_bh26_uid90_Out0 <= Compressor_3_2_F500_uid61_bh26_uid90_Out0_copy91; -- output copy to hold a pipeline register if needed

   bh26_w41_5 <= Compressor_3_2_F500_uid61_bh26_uid90_Out0(0);
   bh26_w42_5 <= Compressor_3_2_F500_uid61_bh26_uid90_Out0(1);

   Compressor_14_3_F500_uid75_bh26_uid92_In0 <= "" & bh26_w42_0 & bh26_w42_1 & bh26_w42_2 & bh26_w42_3;
   Compressor_14_3_F500_uid75_bh26_uid92_In1 <= "" & bh26_w43_0;
   Compressor_14_3_F500_uid75_uid92: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid92_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid92_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid92_Out0_copy93);
   Compressor_14_3_F500_uid75_bh26_uid92_Out0 <= Compressor_14_3_F500_uid75_bh26_uid92_Out0_copy93; -- output copy to hold a pipeline register if needed

   bh26_w42_6 <= Compressor_14_3_F500_uid75_bh26_uid92_Out0(0);
   bh26_w43_4 <= Compressor_14_3_F500_uid75_bh26_uid92_Out0(1);
   bh26_w44_4 <= Compressor_14_3_F500_uid75_bh26_uid92_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid94_In0 <= "" & bh26_w43_1 & bh26_w43_2 & bh26_w43_3;
   Compressor_3_2_F500_uid61_uid94: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid94_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid94_Out0_copy95);
   Compressor_3_2_F500_uid61_bh26_uid94_Out0 <= Compressor_3_2_F500_uid61_bh26_uid94_Out0_copy95; -- output copy to hold a pipeline register if needed

   bh26_w43_5 <= Compressor_3_2_F500_uid61_bh26_uid94_Out0(0);
   bh26_w44_5 <= Compressor_3_2_F500_uid61_bh26_uid94_Out0(1);

   Compressor_14_3_F500_uid75_bh26_uid96_In0 <= "" & bh26_w44_0 & bh26_w44_1 & bh26_w44_2 & bh26_w44_3;
   Compressor_14_3_F500_uid75_bh26_uid96_In1 <= "" & bh26_w45_0;
   Compressor_14_3_F500_uid75_uid96: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid96_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid96_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid96_Out0_copy97);
   Compressor_14_3_F500_uid75_bh26_uid96_Out0 <= Compressor_14_3_F500_uid75_bh26_uid96_Out0_copy97; -- output copy to hold a pipeline register if needed

   bh26_w44_6 <= Compressor_14_3_F500_uid75_bh26_uid96_Out0(0);
   bh26_w45_4 <= Compressor_14_3_F500_uid75_bh26_uid96_Out0(1);
   bh26_w46_4 <= Compressor_14_3_F500_uid75_bh26_uid96_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid98_In0 <= "" & bh26_w45_1 & bh26_w45_2 & bh26_w45_3;
   Compressor_3_2_F500_uid61_uid98: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid98_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid98_Out0_copy99);
   Compressor_3_2_F500_uid61_bh26_uid98_Out0 <= Compressor_3_2_F500_uid61_bh26_uid98_Out0_copy99; -- output copy to hold a pipeline register if needed

   bh26_w45_5 <= Compressor_3_2_F500_uid61_bh26_uid98_Out0(0);
   bh26_w46_5 <= Compressor_3_2_F500_uid61_bh26_uid98_Out0(1);

   Compressor_14_3_F500_uid75_bh26_uid100_In0 <= "" & bh26_w46_0 & bh26_w46_1 & bh26_w46_2 & bh26_w46_3;
   Compressor_14_3_F500_uid75_bh26_uid100_In1 <= "" & bh26_w47_0;
   Compressor_14_3_F500_uid75_uid100: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid100_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid100_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid100_Out0_copy101);
   Compressor_14_3_F500_uid75_bh26_uid100_Out0 <= Compressor_14_3_F500_uid75_bh26_uid100_Out0_copy101; -- output copy to hold a pipeline register if needed

   bh26_w46_6 <= Compressor_14_3_F500_uid75_bh26_uid100_Out0(0);
   bh26_w47_4 <= Compressor_14_3_F500_uid75_bh26_uid100_Out0(1);
   bh26_w48_5 <= Compressor_14_3_F500_uid75_bh26_uid100_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid102_In0 <= "" & bh26_w47_1 & bh26_w47_2 & bh26_w47_3;
   Compressor_3_2_F500_uid61_uid102: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid102_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid102_Out0_copy103);
   Compressor_3_2_F500_uid61_bh26_uid102_Out0 <= Compressor_3_2_F500_uid61_bh26_uid102_Out0_copy103; -- output copy to hold a pipeline register if needed

   bh26_w47_5 <= Compressor_3_2_F500_uid61_bh26_uid102_Out0(0);
   bh26_w48_6 <= Compressor_3_2_F500_uid61_bh26_uid102_Out0(1);

   Compressor_14_3_F500_uid75_bh26_uid104_In0 <= "" & bh26_w48_0 & bh26_w48_1 & bh26_w48_2 & bh26_w48_3;
   Compressor_14_3_F500_uid75_bh26_uid104_In1 <= "" & bh26_w49_0;
   Compressor_14_3_F500_uid75_uid104: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid104_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid104_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid104_Out0_copy105);
   Compressor_14_3_F500_uid75_bh26_uid104_Out0 <= Compressor_14_3_F500_uid75_bh26_uid104_Out0_copy105; -- output copy to hold a pipeline register if needed

   bh26_w48_7 <= Compressor_14_3_F500_uid75_bh26_uid104_Out0(0);
   bh26_w49_5 <= Compressor_14_3_F500_uid75_bh26_uid104_Out0(1);
   bh26_w50_5 <= Compressor_14_3_F500_uid75_bh26_uid104_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid106_In0 <= "" & bh26_w49_1 & bh26_w49_2 & bh26_w49_3 & bh26_w49_4;
   Compressor_14_3_F500_uid75_bh26_uid106_In1 <= "" & bh26_w50_0;
   Compressor_14_3_F500_uid75_uid106: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid106_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid106_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid106_Out0_copy107);
   Compressor_14_3_F500_uid75_bh26_uid106_Out0 <= Compressor_14_3_F500_uid75_bh26_uid106_Out0_copy107; -- output copy to hold a pipeline register if needed

   bh26_w49_6 <= Compressor_14_3_F500_uid75_bh26_uid106_Out0(0);
   bh26_w50_6 <= Compressor_14_3_F500_uid75_bh26_uid106_Out0(1);
   bh26_w51_6 <= Compressor_14_3_F500_uid75_bh26_uid106_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid108_In0 <= "" & bh26_w50_1 & bh26_w50_2 & bh26_w50_3;
   Compressor_3_2_F500_uid61_uid108: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid108_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid108_Out0_copy109);
   Compressor_3_2_F500_uid61_bh26_uid108_Out0 <= Compressor_3_2_F500_uid61_bh26_uid108_Out0_copy109; -- output copy to hold a pipeline register if needed

   bh26_w50_7 <= Compressor_3_2_F500_uid61_bh26_uid108_Out0(0);
   bh26_w51_7 <= Compressor_3_2_F500_uid61_bh26_uid108_Out0(1);

   Compressor_6_3_F500_uid111_bh26_uid112_In0 <= "" & bh26_w51_0 & bh26_w51_1 & bh26_w51_2 & bh26_w51_3 & bh26_w51_4 & bh26_w51_5;
   Compressor_6_3_F500_uid111_uid112: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid112_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid112_Out0_copy113);
   Compressor_6_3_F500_uid111_bh26_uid112_Out0 <= Compressor_6_3_F500_uid111_bh26_uid112_Out0_copy113; -- output copy to hold a pipeline register if needed

   bh26_w51_8 <= Compressor_6_3_F500_uid111_bh26_uid112_Out0(0);
   bh26_w52_6 <= Compressor_6_3_F500_uid111_bh26_uid112_Out0(1);
   bh26_w53_6 <= Compressor_6_3_F500_uid111_bh26_uid112_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid114_In0 <= "" & bh26_w52_0 & bh26_w52_1 & bh26_w52_2 & bh26_w52_3 & bh26_w52_4 & bh26_w52_5;
   Compressor_6_3_F500_uid111_uid114: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid114_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid114_Out0_copy115);
   Compressor_6_3_F500_uid111_bh26_uid114_Out0 <= Compressor_6_3_F500_uid111_bh26_uid114_Out0_copy115; -- output copy to hold a pipeline register if needed

   bh26_w52_7 <= Compressor_6_3_F500_uid111_bh26_uid114_Out0(0);
   bh26_w53_7 <= Compressor_6_3_F500_uid111_bh26_uid114_Out0(1);
   bh26_w54_6 <= Compressor_6_3_F500_uid111_bh26_uid114_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid116_In0 <= "" & bh26_w53_0 & bh26_w53_1 & bh26_w53_2 & bh26_w53_3 & bh26_w53_4 & bh26_w53_5;
   Compressor_6_3_F500_uid111_uid116: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid116_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid116_Out0_copy117);
   Compressor_6_3_F500_uid111_bh26_uid116_Out0 <= Compressor_6_3_F500_uid111_bh26_uid116_Out0_copy117; -- output copy to hold a pipeline register if needed

   bh26_w53_8 <= Compressor_6_3_F500_uid111_bh26_uid116_Out0(0);
   bh26_w54_7 <= Compressor_6_3_F500_uid111_bh26_uid116_Out0(1);
   bh26_w55_6 <= Compressor_6_3_F500_uid111_bh26_uid116_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid118_In0 <= "" & bh26_w54_0 & bh26_w54_1 & bh26_w54_2 & bh26_w54_3 & bh26_w54_4 & bh26_w54_5;
   Compressor_6_3_F500_uid111_uid118: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid118_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid118_Out0_copy119);
   Compressor_6_3_F500_uid111_bh26_uid118_Out0 <= Compressor_6_3_F500_uid111_bh26_uid118_Out0_copy119; -- output copy to hold a pipeline register if needed

   bh26_w54_8 <= Compressor_6_3_F500_uid111_bh26_uid118_Out0(0);
   bh26_w55_7 <= Compressor_6_3_F500_uid111_bh26_uid118_Out0(1);
   bh26_w56_6 <= Compressor_6_3_F500_uid111_bh26_uid118_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid120_In0 <= "" & bh26_w55_0 & bh26_w55_1 & bh26_w55_2 & bh26_w55_3 & bh26_w55_4 & bh26_w55_5;
   Compressor_6_3_F500_uid111_uid120: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid120_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid120_Out0_copy121);
   Compressor_6_3_F500_uid111_bh26_uid120_Out0 <= Compressor_6_3_F500_uid111_bh26_uid120_Out0_copy121; -- output copy to hold a pipeline register if needed

   bh26_w55_8 <= Compressor_6_3_F500_uid111_bh26_uid120_Out0(0);
   bh26_w56_7 <= Compressor_6_3_F500_uid111_bh26_uid120_Out0(1);
   bh26_w57_6 <= Compressor_6_3_F500_uid111_bh26_uid120_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid122_In0 <= "" & bh26_w56_0 & bh26_w56_1 & bh26_w56_2 & bh26_w56_3 & bh26_w56_4 & bh26_w56_5;
   Compressor_6_3_F500_uid111_uid122: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid122_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid122_Out0_copy123);
   Compressor_6_3_F500_uid111_bh26_uid122_Out0 <= Compressor_6_3_F500_uid111_bh26_uid122_Out0_copy123; -- output copy to hold a pipeline register if needed

   bh26_w56_8 <= Compressor_6_3_F500_uid111_bh26_uid122_Out0(0);
   bh26_w57_7 <= Compressor_6_3_F500_uid111_bh26_uid122_Out0(1);
   bh26_w58_6 <= Compressor_6_3_F500_uid111_bh26_uid122_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid124_In0 <= "" & bh26_w57_0 & bh26_w57_1 & bh26_w57_2 & bh26_w57_3 & bh26_w57_4 & bh26_w57_5;
   Compressor_6_3_F500_uid111_uid124: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid124_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid124_Out0_copy125);
   Compressor_6_3_F500_uid111_bh26_uid124_Out0 <= Compressor_6_3_F500_uid111_bh26_uid124_Out0_copy125; -- output copy to hold a pipeline register if needed

   bh26_w57_8 <= Compressor_6_3_F500_uid111_bh26_uid124_Out0(0);
   bh26_w58_7 <= Compressor_6_3_F500_uid111_bh26_uid124_Out0(1);
   bh26_w59_6 <= Compressor_6_3_F500_uid111_bh26_uid124_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid126_In0 <= "" & bh26_w58_0 & bh26_w58_1 & bh26_w58_2 & bh26_w58_3 & bh26_w58_4 & bh26_w58_5;
   Compressor_6_3_F500_uid111_uid126: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid126_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid126_Out0_copy127);
   Compressor_6_3_F500_uid111_bh26_uid126_Out0 <= Compressor_6_3_F500_uid111_bh26_uid126_Out0_copy127; -- output copy to hold a pipeline register if needed

   bh26_w58_8 <= Compressor_6_3_F500_uid111_bh26_uid126_Out0(0);
   bh26_w59_7 <= Compressor_6_3_F500_uid111_bh26_uid126_Out0(1);
   bh26_w60_6 <= Compressor_6_3_F500_uid111_bh26_uid126_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid128_In0 <= "" & bh26_w59_0 & bh26_w59_1 & bh26_w59_2 & bh26_w59_3 & bh26_w59_4 & bh26_w59_5;
   Compressor_6_3_F500_uid111_uid128: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid128_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid128_Out0_copy129);
   Compressor_6_3_F500_uid111_bh26_uid128_Out0 <= Compressor_6_3_F500_uid111_bh26_uid128_Out0_copy129; -- output copy to hold a pipeline register if needed

   bh26_w59_8 <= Compressor_6_3_F500_uid111_bh26_uid128_Out0(0);
   bh26_w60_7 <= Compressor_6_3_F500_uid111_bh26_uid128_Out0(1);
   bh26_w61_6 <= Compressor_6_3_F500_uid111_bh26_uid128_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid130_In0 <= "" & bh26_w60_0 & bh26_w60_1 & bh26_w60_2 & bh26_w60_3 & bh26_w60_4 & bh26_w60_5;
   Compressor_6_3_F500_uid111_uid130: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid130_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid130_Out0_copy131);
   Compressor_6_3_F500_uid111_bh26_uid130_Out0 <= Compressor_6_3_F500_uid111_bh26_uid130_Out0_copy131; -- output copy to hold a pipeline register if needed

   bh26_w60_8 <= Compressor_6_3_F500_uid111_bh26_uid130_Out0(0);
   bh26_w61_7 <= Compressor_6_3_F500_uid111_bh26_uid130_Out0(1);
   bh26_w62_6 <= Compressor_6_3_F500_uid111_bh26_uid130_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid132_In0 <= "" & bh26_w61_0 & bh26_w61_1 & bh26_w61_2 & bh26_w61_3 & bh26_w61_4 & bh26_w61_5;
   Compressor_6_3_F500_uid111_uid132: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid132_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid132_Out0_copy133);
   Compressor_6_3_F500_uid111_bh26_uid132_Out0 <= Compressor_6_3_F500_uid111_bh26_uid132_Out0_copy133; -- output copy to hold a pipeline register if needed

   bh26_w61_8 <= Compressor_6_3_F500_uid111_bh26_uid132_Out0(0);
   bh26_w62_7 <= Compressor_6_3_F500_uid111_bh26_uid132_Out0(1);
   bh26_w63_6 <= Compressor_6_3_F500_uid111_bh26_uid132_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid134_In0 <= "" & bh26_w62_0 & bh26_w62_1 & bh26_w62_2 & bh26_w62_3 & bh26_w62_4 & bh26_w62_5;
   Compressor_6_3_F500_uid111_uid134: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid134_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid134_Out0_copy135);
   Compressor_6_3_F500_uid111_bh26_uid134_Out0 <= Compressor_6_3_F500_uid111_bh26_uid134_Out0_copy135; -- output copy to hold a pipeline register if needed

   bh26_w62_8 <= Compressor_6_3_F500_uid111_bh26_uid134_Out0(0);
   bh26_w63_7 <= Compressor_6_3_F500_uid111_bh26_uid134_Out0(1);
   bh26_w64_6 <= Compressor_6_3_F500_uid111_bh26_uid134_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid136_In0 <= "" & bh26_w63_0 & bh26_w63_1 & bh26_w63_2 & bh26_w63_3 & bh26_w63_4 & bh26_w63_5;
   Compressor_6_3_F500_uid111_uid136: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid136_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid136_Out0_copy137);
   Compressor_6_3_F500_uid111_bh26_uid136_Out0 <= Compressor_6_3_F500_uid111_bh26_uid136_Out0_copy137; -- output copy to hold a pipeline register if needed

   bh26_w63_8 <= Compressor_6_3_F500_uid111_bh26_uid136_Out0(0);
   bh26_w64_7 <= Compressor_6_3_F500_uid111_bh26_uid136_Out0(1);
   bh26_w65_6 <= Compressor_6_3_F500_uid111_bh26_uid136_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid138_In0 <= "" & bh26_w64_0 & bh26_w64_1 & bh26_w64_2 & bh26_w64_3 & bh26_w64_4 & bh26_w64_5;
   Compressor_6_3_F500_uid111_uid138: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid138_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid138_Out0_copy139);
   Compressor_6_3_F500_uid111_bh26_uid138_Out0 <= Compressor_6_3_F500_uid111_bh26_uid138_Out0_copy139; -- output copy to hold a pipeline register if needed

   bh26_w64_8 <= Compressor_6_3_F500_uid111_bh26_uid138_Out0(0);
   bh26_w65_7 <= Compressor_6_3_F500_uid111_bh26_uid138_Out0(1);
   bh26_w66_6 <= Compressor_6_3_F500_uid111_bh26_uid138_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid140_In0 <= "" & bh26_w65_0 & bh26_w65_1 & bh26_w65_2 & bh26_w65_3 & bh26_w65_4 & bh26_w65_5;
   Compressor_6_3_F500_uid111_uid140: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid140_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid140_Out0_copy141);
   Compressor_6_3_F500_uid111_bh26_uid140_Out0 <= Compressor_6_3_F500_uid111_bh26_uid140_Out0_copy141; -- output copy to hold a pipeline register if needed

   bh26_w65_8 <= Compressor_6_3_F500_uid111_bh26_uid140_Out0(0);
   bh26_w66_7 <= Compressor_6_3_F500_uid111_bh26_uid140_Out0(1);
   bh26_w67_6 <= Compressor_6_3_F500_uid111_bh26_uid140_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid142_In0 <= "" & bh26_w66_0 & bh26_w66_1 & bh26_w66_2 & bh26_w66_3 & bh26_w66_4 & bh26_w66_5;
   Compressor_6_3_F500_uid111_uid142: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid142_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid142_Out0_copy143);
   Compressor_6_3_F500_uid111_bh26_uid142_Out0 <= Compressor_6_3_F500_uid111_bh26_uid142_Out0_copy143; -- output copy to hold a pipeline register if needed

   bh26_w66_8 <= Compressor_6_3_F500_uid111_bh26_uid142_Out0(0);
   bh26_w67_7 <= Compressor_6_3_F500_uid111_bh26_uid142_Out0(1);
   bh26_w68_6 <= Compressor_6_3_F500_uid111_bh26_uid142_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid144_In0 <= "" & bh26_w67_0 & bh26_w67_1 & bh26_w67_2 & bh26_w67_3 & bh26_w67_4 & bh26_w67_5;
   Compressor_6_3_F500_uid111_uid144: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid144_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid144_Out0_copy145);
   Compressor_6_3_F500_uid111_bh26_uid144_Out0 <= Compressor_6_3_F500_uid111_bh26_uid144_Out0_copy145; -- output copy to hold a pipeline register if needed

   bh26_w67_8 <= Compressor_6_3_F500_uid111_bh26_uid144_Out0(0);
   bh26_w68_7 <= Compressor_6_3_F500_uid111_bh26_uid144_Out0(1);
   bh26_w69_6 <= Compressor_6_3_F500_uid111_bh26_uid144_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid146_In0 <= "" & bh26_w68_0 & bh26_w68_1 & bh26_w68_2 & bh26_w68_3 & bh26_w68_4 & bh26_w68_5;
   Compressor_6_3_F500_uid111_uid146: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid146_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid146_Out0_copy147);
   Compressor_6_3_F500_uid111_bh26_uid146_Out0 <= Compressor_6_3_F500_uid111_bh26_uid146_Out0_copy147; -- output copy to hold a pipeline register if needed

   bh26_w68_8 <= Compressor_6_3_F500_uid111_bh26_uid146_Out0(0);
   bh26_w69_7 <= Compressor_6_3_F500_uid111_bh26_uid146_Out0(1);
   bh26_w70_5 <= Compressor_6_3_F500_uid111_bh26_uid146_Out0(2);

   Compressor_6_3_F500_uid111_bh26_uid148_In0 <= "" & bh26_w69_0 & bh26_w69_1 & bh26_w69_2 & bh26_w69_3 & bh26_w69_4 & bh26_w69_5;
   Compressor_6_3_F500_uid111_uid148: Compressor_6_3_F500_uid111
      port map ( X0 => Compressor_6_3_F500_uid111_bh26_uid148_In0,
                 R => Compressor_6_3_F500_uid111_bh26_uid148_Out0_copy149);
   Compressor_6_3_F500_uid111_bh26_uid148_Out0 <= Compressor_6_3_F500_uid111_bh26_uid148_Out0_copy149; -- output copy to hold a pipeline register if needed

   bh26_w69_8 <= Compressor_6_3_F500_uid111_bh26_uid148_Out0(0);
   bh26_w70_6 <= Compressor_6_3_F500_uid111_bh26_uid148_Out0(1);
   bh26_w71_5 <= Compressor_6_3_F500_uid111_bh26_uid148_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid150_In0 <= "" & bh26_w70_0 & bh26_w70_1 & bh26_w70_2 & bh26_w70_3;
   Compressor_14_3_F500_uid75_bh26_uid150_In1 <= "" & bh26_w71_0;
   Compressor_14_3_F500_uid75_uid150: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid150_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid150_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid150_Out0_copy151);
   Compressor_14_3_F500_uid75_bh26_uid150_Out0 <= Compressor_14_3_F500_uid75_bh26_uid150_Out0_copy151; -- output copy to hold a pipeline register if needed

   bh26_w70_7 <= Compressor_14_3_F500_uid75_bh26_uid150_Out0(0);
   bh26_w71_6 <= Compressor_14_3_F500_uid75_bh26_uid150_Out0(1);
   bh26_w72_5 <= Compressor_14_3_F500_uid75_bh26_uid150_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid152_In0 <= "" & bh26_w71_1 & bh26_w71_2 & bh26_w71_3 & bh26_w71_4;
   Compressor_14_3_F500_uid75_bh26_uid152_In1 <= "" & bh26_w72_0;
   Compressor_14_3_F500_uid75_uid152: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid152_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid152_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid152_Out0_copy153);
   Compressor_14_3_F500_uid75_bh26_uid152_Out0 <= Compressor_14_3_F500_uid75_bh26_uid152_Out0_copy153; -- output copy to hold a pipeline register if needed

   bh26_w71_7 <= Compressor_14_3_F500_uid75_bh26_uid152_Out0(0);
   bh26_w72_6 <= Compressor_14_3_F500_uid75_bh26_uid152_Out0(1);
   bh26_w73_5 <= Compressor_14_3_F500_uid75_bh26_uid152_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid154_In0 <= "" & bh26_w72_1 & bh26_w72_2 & bh26_w72_3 & bh26_w72_4;
   Compressor_14_3_F500_uid75_bh26_uid154_In1 <= "" & bh26_w73_0;
   Compressor_14_3_F500_uid75_uid154: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid154_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid154_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid154_Out0_copy155);
   Compressor_14_3_F500_uid75_bh26_uid154_Out0 <= Compressor_14_3_F500_uid75_bh26_uid154_Out0_copy155; -- output copy to hold a pipeline register if needed

   bh26_w72_7 <= Compressor_14_3_F500_uid75_bh26_uid154_Out0(0);
   bh26_w73_6 <= Compressor_14_3_F500_uid75_bh26_uid154_Out0(1);
   bh26_w74_5 <= Compressor_14_3_F500_uid75_bh26_uid154_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid156_In0 <= "" & bh26_w73_1 & bh26_w73_2 & bh26_w73_3 & bh26_w73_4;
   Compressor_14_3_F500_uid75_bh26_uid156_In1 <= "" & bh26_w74_0;
   Compressor_14_3_F500_uid75_uid156: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid156_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid156_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid156_Out0_copy157);
   Compressor_14_3_F500_uid75_bh26_uid156_Out0 <= Compressor_14_3_F500_uid75_bh26_uid156_Out0_copy157; -- output copy to hold a pipeline register if needed

   bh26_w73_7 <= Compressor_14_3_F500_uid75_bh26_uid156_Out0(0);
   bh26_w74_6 <= Compressor_14_3_F500_uid75_bh26_uid156_Out0(1);
   bh26_w75_5 <= Compressor_14_3_F500_uid75_bh26_uid156_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid158_In0 <= "" & bh26_w74_1 & bh26_w74_2 & bh26_w74_3 & bh26_w74_4;
   Compressor_14_3_F500_uid75_bh26_uid158_In1 <= "" & bh26_w75_0;
   Compressor_14_3_F500_uid75_uid158: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid158_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid158_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid158_Out0_copy159);
   Compressor_14_3_F500_uid75_bh26_uid158_Out0 <= Compressor_14_3_F500_uid75_bh26_uid158_Out0_copy159; -- output copy to hold a pipeline register if needed

   bh26_w74_7 <= Compressor_14_3_F500_uid75_bh26_uid158_Out0(0);
   bh26_w75_6 <= Compressor_14_3_F500_uid75_bh26_uid158_Out0(1);
   bh26_w76_5 <= Compressor_14_3_F500_uid75_bh26_uid158_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid160_In0 <= "" & bh26_w75_1 & bh26_w75_2 & bh26_w75_3 & bh26_w75_4;
   Compressor_14_3_F500_uid75_bh26_uid160_In1 <= "" & bh26_w76_0;
   Compressor_14_3_F500_uid75_uid160: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid160_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid160_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid160_Out0_copy161);
   Compressor_14_3_F500_uid75_bh26_uid160_Out0 <= Compressor_14_3_F500_uid75_bh26_uid160_Out0_copy161; -- output copy to hold a pipeline register if needed

   bh26_w75_7 <= Compressor_14_3_F500_uid75_bh26_uid160_Out0(0);
   bh26_w76_6 <= Compressor_14_3_F500_uid75_bh26_uid160_Out0(1);
   bh26_w77_4 <= Compressor_14_3_F500_uid75_bh26_uid160_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid162_In0 <= "" & bh26_w76_1 & bh26_w76_2 & bh26_w76_3 & bh26_w76_4;
   Compressor_14_3_F500_uid75_bh26_uid162_In1 <= "" & bh26_w77_0;
   Compressor_14_3_F500_uid75_uid162: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid162_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid162_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid162_Out0_copy163);
   Compressor_14_3_F500_uid75_bh26_uid162_Out0 <= Compressor_14_3_F500_uid75_bh26_uid162_Out0_copy163; -- output copy to hold a pipeline register if needed

   bh26_w76_7 <= Compressor_14_3_F500_uid75_bh26_uid162_Out0(0);
   bh26_w77_5 <= Compressor_14_3_F500_uid75_bh26_uid162_Out0(1);
   bh26_w78_4 <= Compressor_14_3_F500_uid75_bh26_uid162_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid164_In0 <= "" & bh26_w77_1 & bh26_w77_2 & bh26_w77_3;
   Compressor_3_2_F500_uid61_uid164: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid164_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid164_Out0_copy165);
   Compressor_3_2_F500_uid61_bh26_uid164_Out0 <= Compressor_3_2_F500_uid61_bh26_uid164_Out0_copy165; -- output copy to hold a pipeline register if needed

   bh26_w77_6 <= Compressor_3_2_F500_uid61_bh26_uid164_Out0(0);
   bh26_w78_5 <= Compressor_3_2_F500_uid61_bh26_uid164_Out0(1);

   Compressor_14_3_F500_uid75_bh26_uid166_In0 <= "" & bh26_w78_0 & bh26_w78_1 & bh26_w78_2 & bh26_w78_3;
   Compressor_14_3_F500_uid75_bh26_uid166_In1 <= "" & bh26_w79_0;
   Compressor_14_3_F500_uid75_uid166: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid166_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid166_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid166_Out0_copy167);
   Compressor_14_3_F500_uid75_bh26_uid166_Out0 <= Compressor_14_3_F500_uid75_bh26_uid166_Out0_copy167; -- output copy to hold a pipeline register if needed

   bh26_w78_6 <= Compressor_14_3_F500_uid75_bh26_uid166_Out0(0);
   bh26_w79_4 <= Compressor_14_3_F500_uid75_bh26_uid166_Out0(1);
   bh26_w80_4 <= Compressor_14_3_F500_uid75_bh26_uid166_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid168_In0 <= "" & bh26_w79_1 & bh26_w79_2 & bh26_w79_3;
   Compressor_3_2_F500_uid61_uid168: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid168_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid168_Out0_copy169);
   Compressor_3_2_F500_uid61_bh26_uid168_Out0 <= Compressor_3_2_F500_uid61_bh26_uid168_Out0_copy169; -- output copy to hold a pipeline register if needed

   bh26_w79_5 <= Compressor_3_2_F500_uid61_bh26_uid168_Out0(0);
   bh26_w80_5 <= Compressor_3_2_F500_uid61_bh26_uid168_Out0(1);

   Compressor_14_3_F500_uid75_bh26_uid170_In0 <= "" & bh26_w80_0 & bh26_w80_1 & bh26_w80_2 & bh26_w80_3;
   Compressor_14_3_F500_uid75_bh26_uid170_In1 <= "" & bh26_w81_0;
   Compressor_14_3_F500_uid75_uid170: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid170_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid170_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid170_Out0_copy171);
   Compressor_14_3_F500_uid75_bh26_uid170_Out0 <= Compressor_14_3_F500_uid75_bh26_uid170_Out0_copy171; -- output copy to hold a pipeline register if needed

   bh26_w80_6 <= Compressor_14_3_F500_uid75_bh26_uid170_Out0(0);
   bh26_w81_4 <= Compressor_14_3_F500_uid75_bh26_uid170_Out0(1);
   bh26_w82_4 <= Compressor_14_3_F500_uid75_bh26_uid170_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid172_In0 <= "" & bh26_w81_1 & bh26_w81_2 & bh26_w81_3;
   Compressor_3_2_F500_uid61_uid172: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid172_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid172_Out0_copy173);
   Compressor_3_2_F500_uid61_bh26_uid172_Out0 <= Compressor_3_2_F500_uid61_bh26_uid172_Out0_copy173; -- output copy to hold a pipeline register if needed

   bh26_w81_5 <= Compressor_3_2_F500_uid61_bh26_uid172_Out0(0);
   bh26_w82_5 <= Compressor_3_2_F500_uid61_bh26_uid172_Out0(1);

   Compressor_14_3_F500_uid75_bh26_uid174_In0 <= "" & bh26_w82_0 & bh26_w82_1 & bh26_w82_2 & bh26_w82_3;
   Compressor_14_3_F500_uid75_bh26_uid174_In1 <= "" & bh26_w83_0;
   Compressor_14_3_F500_uid75_uid174: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid174_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid174_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid174_Out0_copy175);
   Compressor_14_3_F500_uid75_bh26_uid174_Out0 <= Compressor_14_3_F500_uid75_bh26_uid174_Out0_copy175; -- output copy to hold a pipeline register if needed

   bh26_w82_6 <= Compressor_14_3_F500_uid75_bh26_uid174_Out0(0);
   bh26_w83_4 <= Compressor_14_3_F500_uid75_bh26_uid174_Out0(1);
   bh26_w84_4 <= Compressor_14_3_F500_uid75_bh26_uid174_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid176_In0 <= "" & bh26_w83_1 & bh26_w83_2 & bh26_w83_3;
   Compressor_3_2_F500_uid61_uid176: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid176_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid176_Out0_copy177);
   Compressor_3_2_F500_uid61_bh26_uid176_Out0 <= Compressor_3_2_F500_uid61_bh26_uid176_Out0_copy177; -- output copy to hold a pipeline register if needed

   bh26_w83_5 <= Compressor_3_2_F500_uid61_bh26_uid176_Out0(0);
   bh26_w84_5 <= Compressor_3_2_F500_uid61_bh26_uid176_Out0(1);

   Compressor_14_3_F500_uid75_bh26_uid178_In0 <= "" & bh26_w84_0 & bh26_w84_1 & bh26_w84_2 & bh26_w84_3;
   Compressor_14_3_F500_uid75_bh26_uid178_In1 <= "" & bh26_w85_0;
   Compressor_14_3_F500_uid75_uid178: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid178_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid178_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid178_Out0_copy179);
   Compressor_14_3_F500_uid75_bh26_uid178_Out0 <= Compressor_14_3_F500_uid75_bh26_uid178_Out0_copy179; -- output copy to hold a pipeline register if needed

   bh26_w84_6 <= Compressor_14_3_F500_uid75_bh26_uid178_Out0(0);
   bh26_w85_4 <= Compressor_14_3_F500_uid75_bh26_uid178_Out0(1);
   bh26_w86_4 <= Compressor_14_3_F500_uid75_bh26_uid178_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid180_In0 <= "" & bh26_w85_1 & bh26_w85_2 & bh26_w85_3;
   Compressor_3_2_F500_uid61_uid180: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid180_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid180_Out0_copy181);
   Compressor_3_2_F500_uid61_bh26_uid180_Out0 <= Compressor_3_2_F500_uid61_bh26_uid180_Out0_copy181; -- output copy to hold a pipeline register if needed

   bh26_w85_5 <= Compressor_3_2_F500_uid61_bh26_uid180_Out0(0);
   bh26_w86_5 <= Compressor_3_2_F500_uid61_bh26_uid180_Out0(1);

   Compressor_14_3_F500_uid75_bh26_uid182_In0 <= "" & bh26_w86_0 & bh26_w86_1 & bh26_w86_2 & bh26_w86_3;
   Compressor_14_3_F500_uid75_bh26_uid182_In1 <= "" & "0";
   Compressor_14_3_F500_uid75_uid182: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid182_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid182_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid182_Out0_copy183);
   Compressor_14_3_F500_uid75_bh26_uid182_Out0 <= Compressor_14_3_F500_uid75_bh26_uid182_Out0_copy183; -- output copy to hold a pipeline register if needed

   bh26_w86_6 <= Compressor_14_3_F500_uid75_bh26_uid182_Out0(0);
   bh26_w87_3 <= Compressor_14_3_F500_uid75_bh26_uid182_Out0(1);
   bh26_w88_3 <= Compressor_14_3_F500_uid75_bh26_uid182_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid184_In0 <= "" & bh26_w87_0 & bh26_w87_1 & bh26_w87_2;
   Compressor_3_2_F500_uid61_uid184: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid184_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid184_Out0_copy185);
   Compressor_3_2_F500_uid61_bh26_uid184_Out0 <= Compressor_3_2_F500_uid61_bh26_uid184_Out0_copy185; -- output copy to hold a pipeline register if needed

   bh26_w87_4 <= Compressor_3_2_F500_uid61_bh26_uid184_Out0(0);
   bh26_w88_4 <= Compressor_3_2_F500_uid61_bh26_uid184_Out0(1);

   Compressor_23_3_F500_uid53_bh26_uid186_In0 <= "" & bh26_w88_0 & bh26_w88_1 & bh26_w88_2;
   Compressor_23_3_F500_uid53_bh26_uid186_In1 <= "" & bh26_w89_0 & bh26_w89_1;
   Compressor_23_3_F500_uid53_uid186: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid186_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid186_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid186_Out0_copy187);
   Compressor_23_3_F500_uid53_bh26_uid186_Out0 <= Compressor_23_3_F500_uid53_bh26_uid186_Out0_copy187; -- output copy to hold a pipeline register if needed

   bh26_w88_5 <= Compressor_23_3_F500_uid53_bh26_uid186_Out0(0);
   bh26_w89_3 <= Compressor_23_3_F500_uid53_bh26_uid186_Out0(1);
   bh26_w90_3 <= Compressor_23_3_F500_uid53_bh26_uid186_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid188_In0 <= "" & bh26_w90_0 & bh26_w90_1 & bh26_w90_2;
   Compressor_23_3_F500_uid53_bh26_uid188_In1 <= "" & bh26_w91_0 & bh26_w91_1;
   Compressor_23_3_F500_uid53_uid188: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid188_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid188_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid188_Out0_copy189);
   Compressor_23_3_F500_uid53_bh26_uid188_Out0 <= Compressor_23_3_F500_uid53_bh26_uid188_Out0_copy189; -- output copy to hold a pipeline register if needed

   bh26_w90_4 <= Compressor_23_3_F500_uid53_bh26_uid188_Out0(0);
   bh26_w91_3 <= Compressor_23_3_F500_uid53_bh26_uid188_Out0(1);
   bh26_w92_3 <= Compressor_23_3_F500_uid53_bh26_uid188_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid190_In0 <= "" & bh26_w92_0 & bh26_w92_1 & bh26_w92_2;
   Compressor_23_3_F500_uid53_bh26_uid190_In1 <= "" & bh26_w93_0 & bh26_w93_1;
   Compressor_23_3_F500_uid53_uid190: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid190_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid190_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid190_Out0_copy191);
   Compressor_23_3_F500_uid53_bh26_uid190_Out0 <= Compressor_23_3_F500_uid53_bh26_uid190_Out0_copy191; -- output copy to hold a pipeline register if needed

   bh26_w92_4 <= Compressor_23_3_F500_uid53_bh26_uid190_Out0(0);
   bh26_w93_3 <= Compressor_23_3_F500_uid53_bh26_uid190_Out0(1);
   bh26_w94_3 <= Compressor_23_3_F500_uid53_bh26_uid190_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid192_In0 <= "" & bh26_w94_0 & bh26_w94_1 & bh26_w94_2;
   Compressor_23_3_F500_uid53_bh26_uid192_In1 <= "" & bh26_w95_0 & bh26_w95_1;
   Compressor_23_3_F500_uid53_uid192: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid192_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid192_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid192_Out0_copy193);
   Compressor_23_3_F500_uid53_bh26_uid192_Out0 <= Compressor_23_3_F500_uid53_bh26_uid192_Out0_copy193; -- output copy to hold a pipeline register if needed

   bh26_w94_4 <= Compressor_23_3_F500_uid53_bh26_uid192_Out0(0);
   bh26_w95_3 <= Compressor_23_3_F500_uid53_bh26_uid192_Out0(1);
   bh26_w96_3 <= Compressor_23_3_F500_uid53_bh26_uid192_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid194_In0 <= "" & bh26_w96_0 & bh26_w96_1 & bh26_w96_2;
   Compressor_23_3_F500_uid53_bh26_uid194_In1 <= "" & bh26_w97_0 & bh26_w97_1;
   Compressor_23_3_F500_uid53_uid194: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid194_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid194_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid194_Out0_copy195);
   Compressor_23_3_F500_uid53_bh26_uid194_Out0 <= Compressor_23_3_F500_uid53_bh26_uid194_Out0_copy195; -- output copy to hold a pipeline register if needed

   bh26_w96_4 <= Compressor_23_3_F500_uid53_bh26_uid194_Out0(0);
   bh26_w97_3 <= Compressor_23_3_F500_uid53_bh26_uid194_Out0(1);
   bh26_w98_3 <= Compressor_23_3_F500_uid53_bh26_uid194_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid196_In0 <= "" & bh26_w98_0 & bh26_w98_1 & bh26_w98_2;
   Compressor_23_3_F500_uid53_bh26_uid196_In1 <= "" & bh26_w99_0 & bh26_w99_1;
   Compressor_23_3_F500_uid53_uid196: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid196_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid196_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid196_Out0_copy197);
   Compressor_23_3_F500_uid53_bh26_uid196_Out0 <= Compressor_23_3_F500_uid53_bh26_uid196_Out0_copy197; -- output copy to hold a pipeline register if needed

   bh26_w98_4 <= Compressor_23_3_F500_uid53_bh26_uid196_Out0(0);
   bh26_w99_3 <= Compressor_23_3_F500_uid53_bh26_uid196_Out0(1);
   bh26_w100_3 <= Compressor_23_3_F500_uid53_bh26_uid196_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid198_In0 <= "" & bh26_w100_0 & bh26_w100_1 & bh26_w100_2;
   Compressor_23_3_F500_uid53_bh26_uid198_In1 <= "" & bh26_w101_0 & bh26_w101_1;
   Compressor_23_3_F500_uid53_uid198: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid198_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid198_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid198_Out0_copy199);
   Compressor_23_3_F500_uid53_bh26_uid198_Out0 <= Compressor_23_3_F500_uid53_bh26_uid198_Out0_copy199; -- output copy to hold a pipeline register if needed

   bh26_w100_4 <= Compressor_23_3_F500_uid53_bh26_uid198_Out0(0);
   bh26_w101_2 <= Compressor_23_3_F500_uid53_bh26_uid198_Out0(1);
   bh26_w102_2 <= Compressor_23_3_F500_uid53_bh26_uid198_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid200_In0 <= "" & bh26_w102_0 & bh26_w102_1 & "0";
   Compressor_23_3_F500_uid53_bh26_uid200_In1 <= "" & bh26_w103_0 & bh26_w103_1;
   Compressor_23_3_F500_uid53_uid200: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid200_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid200_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid200_Out0_copy201);
   Compressor_23_3_F500_uid53_bh26_uid200_Out0 <= Compressor_23_3_F500_uid53_bh26_uid200_Out0_copy201; -- output copy to hold a pipeline register if needed

   bh26_w102_3 <= Compressor_23_3_F500_uid53_bh26_uid200_Out0(0);
   bh26_w103_2 <= Compressor_23_3_F500_uid53_bh26_uid200_Out0(1);
   bh26_w104_1 <= Compressor_23_3_F500_uid53_bh26_uid200_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid202_In0 <= "" & bh26_w19_3 & bh26_w19_2 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid202_In1 <= "" & bh26_w20_2;
   Compressor_14_3_F500_uid75_uid202: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid202_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid202_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid202_Out0_copy203);
   Compressor_14_3_F500_uid75_bh26_uid202_Out0 <= Compressor_14_3_F500_uid75_bh26_uid202_Out0_copy203; -- output copy to hold a pipeline register if needed

   bh26_w19_4 <= Compressor_14_3_F500_uid75_bh26_uid202_Out0(0);
   bh26_w20_3 <= Compressor_14_3_F500_uid75_bh26_uid202_Out0(1);
   bh26_w21_4 <= Compressor_14_3_F500_uid75_bh26_uid202_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid204_In0 <= "" & bh26_w21_3 & bh26_w21_2 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid204_In1 <= "" & bh26_w22_2;
   Compressor_14_3_F500_uid75_uid204: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid204_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid204_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid204_Out0_copy205);
   Compressor_14_3_F500_uid75_bh26_uid204_Out0 <= Compressor_14_3_F500_uid75_bh26_uid204_Out0_copy205; -- output copy to hold a pipeline register if needed

   bh26_w21_5 <= Compressor_14_3_F500_uid75_bh26_uid204_Out0(0);
   bh26_w22_3 <= Compressor_14_3_F500_uid75_bh26_uid204_Out0(1);
   bh26_w23_4 <= Compressor_14_3_F500_uid75_bh26_uid204_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid206_In0 <= "" & bh26_w23_3 & bh26_w23_2 & "0";
   Compressor_23_3_F500_uid53_bh26_uid206_In1 <= "" & bh26_w24_4 & bh26_w24_3;
   Compressor_23_3_F500_uid53_uid206: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid206_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid206_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid206_Out0_copy207);
   Compressor_23_3_F500_uid53_bh26_uid206_Out0 <= Compressor_23_3_F500_uid53_bh26_uid206_Out0_copy207; -- output copy to hold a pipeline register if needed

   bh26_w23_5 <= Compressor_23_3_F500_uid53_bh26_uid206_Out0(0);
   bh26_w24_5 <= Compressor_23_3_F500_uid53_bh26_uid206_Out0(1);
   bh26_w25_4 <= Compressor_23_3_F500_uid53_bh26_uid206_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid208_In0 <= "" & bh26_w25_2 & bh26_w25_3 & "0";
   Compressor_23_3_F500_uid53_bh26_uid208_In1 <= "" & bh26_w26_4 & bh26_w26_3;
   Compressor_23_3_F500_uid53_uid208: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid208_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid208_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid208_Out0_copy209);
   Compressor_23_3_F500_uid53_bh26_uid208_Out0 <= Compressor_23_3_F500_uid53_bh26_uid208_Out0_copy209; -- output copy to hold a pipeline register if needed

   bh26_w25_5 <= Compressor_23_3_F500_uid53_bh26_uid208_Out0(0);
   bh26_w26_5 <= Compressor_23_3_F500_uid53_bh26_uid208_Out0(1);
   bh26_w27_4 <= Compressor_23_3_F500_uid53_bh26_uid208_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid210_In0 <= "" & bh26_w27_2 & bh26_w27_3 & "0";
   Compressor_23_3_F500_uid53_bh26_uid210_In1 <= "" & bh26_w28_4 & bh26_w28_3;
   Compressor_23_3_F500_uid53_uid210: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid210_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid210_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid210_Out0_copy211);
   Compressor_23_3_F500_uid53_bh26_uid210_Out0 <= Compressor_23_3_F500_uid53_bh26_uid210_Out0_copy211; -- output copy to hold a pipeline register if needed

   bh26_w27_5 <= Compressor_23_3_F500_uid53_bh26_uid210_Out0(0);
   bh26_w28_5 <= Compressor_23_3_F500_uid53_bh26_uid210_Out0(1);
   bh26_w29_4 <= Compressor_23_3_F500_uid53_bh26_uid210_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid212_In0 <= "" & bh26_w29_2 & bh26_w29_3 & "0";
   Compressor_23_3_F500_uid53_bh26_uid212_In1 <= "" & bh26_w30_4 & bh26_w30_3;
   Compressor_23_3_F500_uid53_uid212: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid212_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid212_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid212_Out0_copy213);
   Compressor_23_3_F500_uid53_bh26_uid212_Out0 <= Compressor_23_3_F500_uid53_bh26_uid212_Out0_copy213; -- output copy to hold a pipeline register if needed

   bh26_w29_5 <= Compressor_23_3_F500_uid53_bh26_uid212_Out0(0);
   bh26_w30_5 <= Compressor_23_3_F500_uid53_bh26_uid212_Out0(1);
   bh26_w31_4 <= Compressor_23_3_F500_uid53_bh26_uid212_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid214_In0 <= "" & bh26_w31_2 & bh26_w31_3 & "0";
   Compressor_23_3_F500_uid53_bh26_uid214_In1 <= "" & bh26_w32_4 & bh26_w32_3;
   Compressor_23_3_F500_uid53_uid214: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid214_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid214_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid214_Out0_copy215);
   Compressor_23_3_F500_uid53_bh26_uid214_Out0 <= Compressor_23_3_F500_uid53_bh26_uid214_Out0_copy215; -- output copy to hold a pipeline register if needed

   bh26_w31_5 <= Compressor_23_3_F500_uid53_bh26_uid214_Out0(0);
   bh26_w32_5 <= Compressor_23_3_F500_uid53_bh26_uid214_Out0(1);
   bh26_w33_4 <= Compressor_23_3_F500_uid53_bh26_uid214_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid216_In0 <= "" & bh26_w33_2 & bh26_w33_3 & "0";
   Compressor_23_3_F500_uid53_bh26_uid216_In1 <= "" & bh26_w34_5 & bh26_w34_4;
   Compressor_23_3_F500_uid53_uid216: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid216_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid216_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid216_Out0_copy217);
   Compressor_23_3_F500_uid53_bh26_uid216_Out0 <= Compressor_23_3_F500_uid53_bh26_uid216_Out0_copy217; -- output copy to hold a pipeline register if needed

   bh26_w33_5 <= Compressor_23_3_F500_uid53_bh26_uid216_Out0(0);
   bh26_w34_6 <= Compressor_23_3_F500_uid53_bh26_uid216_Out0(1);
   bh26_w35_6 <= Compressor_23_3_F500_uid53_bh26_uid216_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid218_In0 <= "" & bh26_w35_5 & bh26_w35_4 & "0";
   Compressor_3_2_F500_uid61_uid218: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid218_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid218_Out0_copy219);
   Compressor_3_2_F500_uid61_bh26_uid218_Out0 <= Compressor_3_2_F500_uid61_bh26_uid218_Out0_copy219; -- output copy to hold a pipeline register if needed

   bh26_w35_7 <= Compressor_3_2_F500_uid61_bh26_uid218_Out0(0);
   bh26_w36_7 <= Compressor_3_2_F500_uid61_bh26_uid218_Out0(1);

   Compressor_23_3_F500_uid53_bh26_uid220_In0 <= "" & bh26_w36_6 & bh26_w36_5 & bh26_w36_4;
   Compressor_23_3_F500_uid53_bh26_uid220_In1 <= "" & bh26_w37_5 & bh26_w37_4;
   Compressor_23_3_F500_uid53_uid220: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid220_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid220_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid220_Out0_copy221);
   Compressor_23_3_F500_uid53_bh26_uid220_Out0 <= Compressor_23_3_F500_uid53_bh26_uid220_Out0_copy221; -- output copy to hold a pipeline register if needed

   bh26_w36_8 <= Compressor_23_3_F500_uid53_bh26_uid220_Out0(0);
   bh26_w37_6 <= Compressor_23_3_F500_uid53_bh26_uid220_Out0(1);
   bh26_w38_7 <= Compressor_23_3_F500_uid53_bh26_uid220_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid222_In0 <= "" & bh26_w38_6 & bh26_w38_5 & bh26_w38_4;
   Compressor_23_3_F500_uid53_bh26_uid222_In1 <= "" & bh26_w39_5 & bh26_w39_4;
   Compressor_23_3_F500_uid53_uid222: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid222_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid222_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid222_Out0_copy223);
   Compressor_23_3_F500_uid53_bh26_uid222_Out0 <= Compressor_23_3_F500_uid53_bh26_uid222_Out0_copy223; -- output copy to hold a pipeline register if needed

   bh26_w38_8 <= Compressor_23_3_F500_uid53_bh26_uid222_Out0(0);
   bh26_w39_6 <= Compressor_23_3_F500_uid53_bh26_uid222_Out0(1);
   bh26_w40_7 <= Compressor_23_3_F500_uid53_bh26_uid222_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid224_In0 <= "" & bh26_w40_6 & bh26_w40_5 & bh26_w40_4;
   Compressor_23_3_F500_uid53_bh26_uid224_In1 <= "" & bh26_w41_5 & bh26_w41_4;
   Compressor_23_3_F500_uid53_uid224: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid224_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid224_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid224_Out0_copy225);
   Compressor_23_3_F500_uid53_bh26_uid224_Out0 <= Compressor_23_3_F500_uid53_bh26_uid224_Out0_copy225; -- output copy to hold a pipeline register if needed

   bh26_w40_8 <= Compressor_23_3_F500_uid53_bh26_uid224_Out0(0);
   bh26_w41_6 <= Compressor_23_3_F500_uid53_bh26_uid224_Out0(1);
   bh26_w42_7 <= Compressor_23_3_F500_uid53_bh26_uid224_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid226_In0 <= "" & bh26_w42_6 & bh26_w42_5 & bh26_w42_4;
   Compressor_23_3_F500_uid53_bh26_uid226_In1 <= "" & bh26_w43_5 & bh26_w43_4;
   Compressor_23_3_F500_uid53_uid226: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid226_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid226_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid226_Out0_copy227);
   Compressor_23_3_F500_uid53_bh26_uid226_Out0 <= Compressor_23_3_F500_uid53_bh26_uid226_Out0_copy227; -- output copy to hold a pipeline register if needed

   bh26_w42_8 <= Compressor_23_3_F500_uid53_bh26_uid226_Out0(0);
   bh26_w43_6 <= Compressor_23_3_F500_uid53_bh26_uid226_Out0(1);
   bh26_w44_7 <= Compressor_23_3_F500_uid53_bh26_uid226_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid228_In0 <= "" & bh26_w44_6 & bh26_w44_5 & bh26_w44_4;
   Compressor_23_3_F500_uid53_bh26_uid228_In1 <= "" & bh26_w45_5 & bh26_w45_4;
   Compressor_23_3_F500_uid53_uid228: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid228_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid228_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid228_Out0_copy229);
   Compressor_23_3_F500_uid53_bh26_uid228_Out0 <= Compressor_23_3_F500_uid53_bh26_uid228_Out0_copy229; -- output copy to hold a pipeline register if needed

   bh26_w44_8 <= Compressor_23_3_F500_uid53_bh26_uid228_Out0(0);
   bh26_w45_6 <= Compressor_23_3_F500_uid53_bh26_uid228_Out0(1);
   bh26_w46_7 <= Compressor_23_3_F500_uid53_bh26_uid228_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid230_In0 <= "" & bh26_w46_6 & bh26_w46_5 & bh26_w46_4;
   Compressor_23_3_F500_uid53_bh26_uid230_In1 <= "" & bh26_w47_5 & bh26_w47_4;
   Compressor_23_3_F500_uid53_uid230: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid230_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid230_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid230_Out0_copy231);
   Compressor_23_3_F500_uid53_bh26_uid230_Out0 <= Compressor_23_3_F500_uid53_bh26_uid230_Out0_copy231; -- output copy to hold a pipeline register if needed

   bh26_w46_8 <= Compressor_23_3_F500_uid53_bh26_uid230_Out0(0);
   bh26_w47_6 <= Compressor_23_3_F500_uid53_bh26_uid230_Out0(1);
   bh26_w48_8 <= Compressor_23_3_F500_uid53_bh26_uid230_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid232_In0 <= "" & bh26_w48_4 & bh26_w48_7 & bh26_w48_6 & bh26_w48_5;
   Compressor_14_3_F500_uid75_bh26_uid232_In1 <= "" & bh26_w49_6;
   Compressor_14_3_F500_uid75_uid232: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid232_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid232_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid232_Out0_copy233);
   Compressor_14_3_F500_uid75_bh26_uid232_Out0 <= Compressor_14_3_F500_uid75_bh26_uid232_Out0_copy233; -- output copy to hold a pipeline register if needed

   bh26_w48_9 <= Compressor_14_3_F500_uid75_bh26_uid232_Out0(0);
   bh26_w49_7 <= Compressor_14_3_F500_uid75_bh26_uid232_Out0(1);
   bh26_w50_8 <= Compressor_14_3_F500_uid75_bh26_uid232_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid234_In0 <= "" & bh26_w50_4 & bh26_w50_7 & bh26_w50_6 & bh26_w50_5;
   Compressor_14_3_F500_uid75_bh26_uid234_In1 <= "" & "0";
   Compressor_14_3_F500_uid75_uid234: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid234_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid234_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid234_Out0_copy235);
   Compressor_14_3_F500_uid75_bh26_uid234_Out0 <= Compressor_14_3_F500_uid75_bh26_uid234_Out0_copy235; -- output copy to hold a pipeline register if needed

   bh26_w50_9 <= Compressor_14_3_F500_uid75_bh26_uid234_Out0(0);
   bh26_w51_9 <= Compressor_14_3_F500_uid75_bh26_uid234_Out0(1);
   bh26_w52_8 <= Compressor_14_3_F500_uid75_bh26_uid234_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid236_In0 <= "" & bh26_w51_8 & bh26_w51_7 & bh26_w51_6;
   Compressor_23_3_F500_uid53_bh26_uid236_In1 <= "" & bh26_w52_7 & bh26_w52_6;
   Compressor_23_3_F500_uid53_uid236: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid236_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid236_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid236_Out0_copy237);
   Compressor_23_3_F500_uid53_bh26_uid236_Out0 <= Compressor_23_3_F500_uid53_bh26_uid236_Out0_copy237; -- output copy to hold a pipeline register if needed

   bh26_w51_10 <= Compressor_23_3_F500_uid53_bh26_uid236_Out0(0);
   bh26_w52_9 <= Compressor_23_3_F500_uid53_bh26_uid236_Out0(1);
   bh26_w53_9 <= Compressor_23_3_F500_uid53_bh26_uid236_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid238_In0 <= "" & bh26_w53_8 & bh26_w53_7 & bh26_w53_6;
   Compressor_23_3_F500_uid53_bh26_uid238_In1 <= "" & bh26_w54_8 & bh26_w54_7;
   Compressor_23_3_F500_uid53_uid238: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid238_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid238_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid238_Out0_copy239);
   Compressor_23_3_F500_uid53_bh26_uid238_Out0 <= Compressor_23_3_F500_uid53_bh26_uid238_Out0_copy239; -- output copy to hold a pipeline register if needed

   bh26_w53_10 <= Compressor_23_3_F500_uid53_bh26_uid238_Out0(0);
   bh26_w54_9 <= Compressor_23_3_F500_uid53_bh26_uid238_Out0(1);
   bh26_w55_9 <= Compressor_23_3_F500_uid53_bh26_uid238_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid240_In0 <= "" & bh26_w55_8 & bh26_w55_7 & bh26_w55_6;
   Compressor_23_3_F500_uid53_bh26_uid240_In1 <= "" & bh26_w56_8 & bh26_w56_7;
   Compressor_23_3_F500_uid53_uid240: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid240_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid240_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid240_Out0_copy241);
   Compressor_23_3_F500_uid53_bh26_uid240_Out0 <= Compressor_23_3_F500_uid53_bh26_uid240_Out0_copy241; -- output copy to hold a pipeline register if needed

   bh26_w55_10 <= Compressor_23_3_F500_uid53_bh26_uid240_Out0(0);
   bh26_w56_9 <= Compressor_23_3_F500_uid53_bh26_uid240_Out0(1);
   bh26_w57_9 <= Compressor_23_3_F500_uid53_bh26_uid240_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid242_In0 <= "" & bh26_w57_8 & bh26_w57_7 & bh26_w57_6;
   Compressor_23_3_F500_uid53_bh26_uid242_In1 <= "" & bh26_w58_8 & bh26_w58_7;
   Compressor_23_3_F500_uid53_uid242: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid242_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid242_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid242_Out0_copy243);
   Compressor_23_3_F500_uid53_bh26_uid242_Out0 <= Compressor_23_3_F500_uid53_bh26_uid242_Out0_copy243; -- output copy to hold a pipeline register if needed

   bh26_w57_10 <= Compressor_23_3_F500_uid53_bh26_uid242_Out0(0);
   bh26_w58_9 <= Compressor_23_3_F500_uid53_bh26_uid242_Out0(1);
   bh26_w59_9 <= Compressor_23_3_F500_uid53_bh26_uid242_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid244_In0 <= "" & bh26_w59_8 & bh26_w59_7 & bh26_w59_6;
   Compressor_23_3_F500_uid53_bh26_uid244_In1 <= "" & bh26_w60_8 & bh26_w60_7;
   Compressor_23_3_F500_uid53_uid244: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid244_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid244_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid244_Out0_copy245);
   Compressor_23_3_F500_uid53_bh26_uid244_Out0 <= Compressor_23_3_F500_uid53_bh26_uid244_Out0_copy245; -- output copy to hold a pipeline register if needed

   bh26_w59_10 <= Compressor_23_3_F500_uid53_bh26_uid244_Out0(0);
   bh26_w60_9 <= Compressor_23_3_F500_uid53_bh26_uid244_Out0(1);
   bh26_w61_9 <= Compressor_23_3_F500_uid53_bh26_uid244_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid246_In0 <= "" & bh26_w61_8 & bh26_w61_7 & bh26_w61_6;
   Compressor_23_3_F500_uid53_bh26_uid246_In1 <= "" & bh26_w62_8 & bh26_w62_7;
   Compressor_23_3_F500_uid53_uid246: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid246_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid246_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid246_Out0_copy247);
   Compressor_23_3_F500_uid53_bh26_uid246_Out0 <= Compressor_23_3_F500_uid53_bh26_uid246_Out0_copy247; -- output copy to hold a pipeline register if needed

   bh26_w61_10 <= Compressor_23_3_F500_uid53_bh26_uid246_Out0(0);
   bh26_w62_9 <= Compressor_23_3_F500_uid53_bh26_uid246_Out0(1);
   bh26_w63_9 <= Compressor_23_3_F500_uid53_bh26_uid246_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid248_In0 <= "" & bh26_w63_8 & bh26_w63_7 & bh26_w63_6;
   Compressor_23_3_F500_uid53_bh26_uid248_In1 <= "" & bh26_w64_8 & bh26_w64_7;
   Compressor_23_3_F500_uid53_uid248: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid248_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid248_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid248_Out0_copy249);
   Compressor_23_3_F500_uid53_bh26_uid248_Out0 <= Compressor_23_3_F500_uid53_bh26_uid248_Out0_copy249; -- output copy to hold a pipeline register if needed

   bh26_w63_10 <= Compressor_23_3_F500_uid53_bh26_uid248_Out0(0);
   bh26_w64_9 <= Compressor_23_3_F500_uid53_bh26_uid248_Out0(1);
   bh26_w65_9 <= Compressor_23_3_F500_uid53_bh26_uid248_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid250_In0 <= "" & bh26_w65_8 & bh26_w65_7 & bh26_w65_6;
   Compressor_23_3_F500_uid53_bh26_uid250_In1 <= "" & bh26_w66_8 & bh26_w66_7;
   Compressor_23_3_F500_uid53_uid250: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid250_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid250_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid250_Out0_copy251);
   Compressor_23_3_F500_uid53_bh26_uid250_Out0 <= Compressor_23_3_F500_uid53_bh26_uid250_Out0_copy251; -- output copy to hold a pipeline register if needed

   bh26_w65_10 <= Compressor_23_3_F500_uid53_bh26_uid250_Out0(0);
   bh26_w66_9 <= Compressor_23_3_F500_uid53_bh26_uid250_Out0(1);
   bh26_w67_9 <= Compressor_23_3_F500_uid53_bh26_uid250_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid252_In0 <= "" & bh26_w67_8 & bh26_w67_7 & bh26_w67_6;
   Compressor_23_3_F500_uid53_bh26_uid252_In1 <= "" & bh26_w68_8 & bh26_w68_7;
   Compressor_23_3_F500_uid53_uid252: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid252_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid252_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid252_Out0_copy253);
   Compressor_23_3_F500_uid53_bh26_uid252_Out0 <= Compressor_23_3_F500_uid53_bh26_uid252_Out0_copy253; -- output copy to hold a pipeline register if needed

   bh26_w67_10 <= Compressor_23_3_F500_uid53_bh26_uid252_Out0(0);
   bh26_w68_9 <= Compressor_23_3_F500_uid53_bh26_uid252_Out0(1);
   bh26_w69_9 <= Compressor_23_3_F500_uid53_bh26_uid252_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid254_In0 <= "" & bh26_w69_8 & bh26_w69_7 & bh26_w69_6;
   Compressor_3_2_F500_uid61_uid254: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid254_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid254_Out0_copy255);
   Compressor_3_2_F500_uid61_bh26_uid254_Out0 <= Compressor_3_2_F500_uid61_bh26_uid254_Out0_copy255; -- output copy to hold a pipeline register if needed

   bh26_w69_10 <= Compressor_3_2_F500_uid61_bh26_uid254_Out0(0);
   bh26_w70_8 <= Compressor_3_2_F500_uid61_bh26_uid254_Out0(1);

   Compressor_14_3_F500_uid75_bh26_uid256_In0 <= "" & bh26_w70_4 & bh26_w70_7 & bh26_w70_6 & bh26_w70_5;
   Compressor_14_3_F500_uid75_bh26_uid256_In1 <= "" & "0";
   Compressor_14_3_F500_uid75_uid256: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid256_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid256_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid256_Out0_copy257);
   Compressor_14_3_F500_uid75_bh26_uid256_Out0 <= Compressor_14_3_F500_uid75_bh26_uid256_Out0_copy257; -- output copy to hold a pipeline register if needed

   bh26_w70_9 <= Compressor_14_3_F500_uid75_bh26_uid256_Out0(0);
   bh26_w71_8 <= Compressor_14_3_F500_uid75_bh26_uid256_Out0(1);
   bh26_w72_8 <= Compressor_14_3_F500_uid75_bh26_uid256_Out0(2);

   Compressor_3_2_F500_uid61_bh26_uid258_In0 <= "" & bh26_w71_7 & bh26_w71_6 & bh26_w71_5;
   Compressor_3_2_F500_uid61_uid258: Compressor_3_2_F500_uid61
      port map ( X0 => Compressor_3_2_F500_uid61_bh26_uid258_In0,
                 R => Compressor_3_2_F500_uid61_bh26_uid258_Out0_copy259);
   Compressor_3_2_F500_uid61_bh26_uid258_Out0 <= Compressor_3_2_F500_uid61_bh26_uid258_Out0_copy259; -- output copy to hold a pipeline register if needed

   bh26_w71_9 <= Compressor_3_2_F500_uid61_bh26_uid258_Out0(0);
   bh26_w72_9 <= Compressor_3_2_F500_uid61_bh26_uid258_Out0(1);

   Compressor_23_3_F500_uid53_bh26_uid260_In0 <= "" & bh26_w72_7 & bh26_w72_6 & bh26_w72_5;
   Compressor_23_3_F500_uid53_bh26_uid260_In1 <= "" & bh26_w73_7 & bh26_w73_6;
   Compressor_23_3_F500_uid53_uid260: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid260_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid260_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid260_Out0_copy261);
   Compressor_23_3_F500_uid53_bh26_uid260_Out0 <= Compressor_23_3_F500_uid53_bh26_uid260_Out0_copy261; -- output copy to hold a pipeline register if needed

   bh26_w72_10 <= Compressor_23_3_F500_uid53_bh26_uid260_Out0(0);
   bh26_w73_8 <= Compressor_23_3_F500_uid53_bh26_uid260_Out0(1);
   bh26_w74_8 <= Compressor_23_3_F500_uid53_bh26_uid260_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid262_In0 <= "" & bh26_w74_7 & bh26_w74_6 & bh26_w74_5;
   Compressor_23_3_F500_uid53_bh26_uid262_In1 <= "" & bh26_w75_7 & bh26_w75_6;
   Compressor_23_3_F500_uid53_uid262: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid262_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid262_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid262_Out0_copy263);
   Compressor_23_3_F500_uid53_bh26_uid262_Out0 <= Compressor_23_3_F500_uid53_bh26_uid262_Out0_copy263; -- output copy to hold a pipeline register if needed

   bh26_w74_9 <= Compressor_23_3_F500_uid53_bh26_uid262_Out0(0);
   bh26_w75_8 <= Compressor_23_3_F500_uid53_bh26_uid262_Out0(1);
   bh26_w76_8 <= Compressor_23_3_F500_uid53_bh26_uid262_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid264_In0 <= "" & bh26_w76_7 & bh26_w76_6 & bh26_w76_5;
   Compressor_23_3_F500_uid53_bh26_uid264_In1 <= "" & bh26_w77_6 & bh26_w77_5;
   Compressor_23_3_F500_uid53_uid264: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid264_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid264_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid264_Out0_copy265);
   Compressor_23_3_F500_uid53_bh26_uid264_Out0 <= Compressor_23_3_F500_uid53_bh26_uid264_Out0_copy265; -- output copy to hold a pipeline register if needed

   bh26_w76_9 <= Compressor_23_3_F500_uid53_bh26_uid264_Out0(0);
   bh26_w77_7 <= Compressor_23_3_F500_uid53_bh26_uid264_Out0(1);
   bh26_w78_7 <= Compressor_23_3_F500_uid53_bh26_uid264_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid266_In0 <= "" & bh26_w78_6 & bh26_w78_5 & bh26_w78_4;
   Compressor_23_3_F500_uid53_bh26_uid266_In1 <= "" & bh26_w79_5 & bh26_w79_4;
   Compressor_23_3_F500_uid53_uid266: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid266_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid266_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid266_Out0_copy267);
   Compressor_23_3_F500_uid53_bh26_uid266_Out0 <= Compressor_23_3_F500_uid53_bh26_uid266_Out0_copy267; -- output copy to hold a pipeline register if needed

   bh26_w78_8 <= Compressor_23_3_F500_uid53_bh26_uid266_Out0(0);
   bh26_w79_6 <= Compressor_23_3_F500_uid53_bh26_uid266_Out0(1);
   bh26_w80_7 <= Compressor_23_3_F500_uid53_bh26_uid266_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid268_In0 <= "" & bh26_w80_6 & bh26_w80_5 & bh26_w80_4;
   Compressor_23_3_F500_uid53_bh26_uid268_In1 <= "" & bh26_w81_5 & bh26_w81_4;
   Compressor_23_3_F500_uid53_uid268: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid268_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid268_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid268_Out0_copy269);
   Compressor_23_3_F500_uid53_bh26_uid268_Out0 <= Compressor_23_3_F500_uid53_bh26_uid268_Out0_copy269; -- output copy to hold a pipeline register if needed

   bh26_w80_8 <= Compressor_23_3_F500_uid53_bh26_uid268_Out0(0);
   bh26_w81_6 <= Compressor_23_3_F500_uid53_bh26_uid268_Out0(1);
   bh26_w82_7 <= Compressor_23_3_F500_uid53_bh26_uid268_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid270_In0 <= "" & bh26_w82_6 & bh26_w82_5 & bh26_w82_4;
   Compressor_23_3_F500_uid53_bh26_uid270_In1 <= "" & bh26_w83_5 & bh26_w83_4;
   Compressor_23_3_F500_uid53_uid270: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid270_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid270_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid270_Out0_copy271);
   Compressor_23_3_F500_uid53_bh26_uid270_Out0 <= Compressor_23_3_F500_uid53_bh26_uid270_Out0_copy271; -- output copy to hold a pipeline register if needed

   bh26_w82_8 <= Compressor_23_3_F500_uid53_bh26_uid270_Out0(0);
   bh26_w83_6 <= Compressor_23_3_F500_uid53_bh26_uid270_Out0(1);
   bh26_w84_7 <= Compressor_23_3_F500_uid53_bh26_uid270_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid272_In0 <= "" & bh26_w84_6 & bh26_w84_5 & bh26_w84_4;
   Compressor_23_3_F500_uid53_bh26_uid272_In1 <= "" & bh26_w85_5 & bh26_w85_4;
   Compressor_23_3_F500_uid53_uid272: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid272_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid272_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid272_Out0_copy273);
   Compressor_23_3_F500_uid53_bh26_uid272_Out0 <= Compressor_23_3_F500_uid53_bh26_uid272_Out0_copy273; -- output copy to hold a pipeline register if needed

   bh26_w84_8 <= Compressor_23_3_F500_uid53_bh26_uid272_Out0(0);
   bh26_w85_6 <= Compressor_23_3_F500_uid53_bh26_uid272_Out0(1);
   bh26_w86_7 <= Compressor_23_3_F500_uid53_bh26_uid272_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid274_In0 <= "" & bh26_w86_6 & bh26_w86_5 & bh26_w86_4;
   Compressor_23_3_F500_uid53_bh26_uid274_In1 <= "" & bh26_w87_3 & bh26_w87_4;
   Compressor_23_3_F500_uid53_uid274: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid274_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid274_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid274_Out0_copy275);
   Compressor_23_3_F500_uid53_bh26_uid274_Out0 <= Compressor_23_3_F500_uid53_bh26_uid274_Out0_copy275; -- output copy to hold a pipeline register if needed

   bh26_w86_8 <= Compressor_23_3_F500_uid53_bh26_uid274_Out0(0);
   bh26_w87_5 <= Compressor_23_3_F500_uid53_bh26_uid274_Out0(1);
   bh26_w88_6 <= Compressor_23_3_F500_uid53_bh26_uid274_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid276_In0 <= "" & bh26_w88_3 & bh26_w88_5 & bh26_w88_4;
   Compressor_23_3_F500_uid53_bh26_uid276_In1 <= "" & bh26_w89_2 & bh26_w89_3;
   Compressor_23_3_F500_uid53_uid276: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid276_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid276_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid276_Out0_copy277);
   Compressor_23_3_F500_uid53_bh26_uid276_Out0 <= Compressor_23_3_F500_uid53_bh26_uid276_Out0_copy277; -- output copy to hold a pipeline register if needed

   bh26_w88_7 <= Compressor_23_3_F500_uid53_bh26_uid276_Out0(0);
   bh26_w89_4 <= Compressor_23_3_F500_uid53_bh26_uid276_Out0(1);
   bh26_w90_5 <= Compressor_23_3_F500_uid53_bh26_uid276_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid278_In0 <= "" & bh26_w90_4 & bh26_w90_3 & "0";
   Compressor_23_3_F500_uid53_bh26_uid278_In1 <= "" & bh26_w91_2 & bh26_w91_3;
   Compressor_23_3_F500_uid53_uid278: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid278_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid278_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid278_Out0_copy279);
   Compressor_23_3_F500_uid53_bh26_uid278_Out0 <= Compressor_23_3_F500_uid53_bh26_uid278_Out0_copy279; -- output copy to hold a pipeline register if needed

   bh26_w90_6 <= Compressor_23_3_F500_uid53_bh26_uid278_Out0(0);
   bh26_w91_4 <= Compressor_23_3_F500_uid53_bh26_uid278_Out0(1);
   bh26_w92_5 <= Compressor_23_3_F500_uid53_bh26_uid278_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid280_In0 <= "" & bh26_w92_4 & bh26_w92_3 & "0";
   Compressor_23_3_F500_uid53_bh26_uid280_In1 <= "" & bh26_w93_2 & bh26_w93_3;
   Compressor_23_3_F500_uid53_uid280: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid280_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid280_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid280_Out0_copy281);
   Compressor_23_3_F500_uid53_bh26_uid280_Out0 <= Compressor_23_3_F500_uid53_bh26_uid280_Out0_copy281; -- output copy to hold a pipeline register if needed

   bh26_w92_6 <= Compressor_23_3_F500_uid53_bh26_uid280_Out0(0);
   bh26_w93_4 <= Compressor_23_3_F500_uid53_bh26_uid280_Out0(1);
   bh26_w94_5 <= Compressor_23_3_F500_uid53_bh26_uid280_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid282_In0 <= "" & bh26_w94_4 & bh26_w94_3 & "0";
   Compressor_23_3_F500_uid53_bh26_uid282_In1 <= "" & bh26_w95_2 & bh26_w95_3;
   Compressor_23_3_F500_uid53_uid282: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid282_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid282_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid282_Out0_copy283);
   Compressor_23_3_F500_uid53_bh26_uid282_Out0 <= Compressor_23_3_F500_uid53_bh26_uid282_Out0_copy283; -- output copy to hold a pipeline register if needed

   bh26_w94_6 <= Compressor_23_3_F500_uid53_bh26_uid282_Out0(0);
   bh26_w95_4 <= Compressor_23_3_F500_uid53_bh26_uid282_Out0(1);
   bh26_w96_5 <= Compressor_23_3_F500_uid53_bh26_uid282_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid284_In0 <= "" & bh26_w96_4 & bh26_w96_3 & "0";
   Compressor_23_3_F500_uid53_bh26_uid284_In1 <= "" & bh26_w97_2 & bh26_w97_3;
   Compressor_23_3_F500_uid53_uid284: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid284_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid284_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid284_Out0_copy285);
   Compressor_23_3_F500_uid53_bh26_uid284_Out0 <= Compressor_23_3_F500_uid53_bh26_uid284_Out0_copy285; -- output copy to hold a pipeline register if needed

   bh26_w96_6 <= Compressor_23_3_F500_uid53_bh26_uid284_Out0(0);
   bh26_w97_4 <= Compressor_23_3_F500_uid53_bh26_uid284_Out0(1);
   bh26_w98_5 <= Compressor_23_3_F500_uid53_bh26_uid284_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid286_In0 <= "" & bh26_w98_4 & bh26_w98_3 & "0";
   Compressor_23_3_F500_uid53_bh26_uid286_In1 <= "" & bh26_w99_2 & bh26_w99_3;
   Compressor_23_3_F500_uid53_uid286: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid286_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid286_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid286_Out0_copy287);
   Compressor_23_3_F500_uid53_bh26_uid286_Out0 <= Compressor_23_3_F500_uid53_bh26_uid286_Out0_copy287; -- output copy to hold a pipeline register if needed

   bh26_w98_6 <= Compressor_23_3_F500_uid53_bh26_uid286_Out0(0);
   bh26_w99_4 <= Compressor_23_3_F500_uid53_bh26_uid286_Out0(1);
   bh26_w100_5 <= Compressor_23_3_F500_uid53_bh26_uid286_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid288_In0 <= "" & bh26_w100_4 & bh26_w100_3 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid288_In1 <= "" & bh26_w101_2;
   Compressor_14_3_F500_uid75_uid288: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid288_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid288_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid288_Out0_copy289);
   Compressor_14_3_F500_uid75_bh26_uid288_Out0 <= Compressor_14_3_F500_uid75_bh26_uid288_Out0_copy289; -- output copy to hold a pipeline register if needed

   bh26_w100_6 <= Compressor_14_3_F500_uid75_bh26_uid288_Out0(0);
   bh26_w101_3 <= Compressor_14_3_F500_uid75_bh26_uid288_Out0(1);
   bh26_w102_4 <= Compressor_14_3_F500_uid75_bh26_uid288_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid290_In0 <= "" & bh26_w102_3 & bh26_w102_2 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid290_In1 <= "" & bh26_w103_2;
   Compressor_14_3_F500_uid75_uid290: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid290_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid290_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid290_Out0_copy291);
   Compressor_14_3_F500_uid75_bh26_uid290_Out0 <= Compressor_14_3_F500_uid75_bh26_uid290_Out0_copy291; -- output copy to hold a pipeline register if needed

   bh26_w102_5 <= Compressor_14_3_F500_uid75_bh26_uid290_Out0(0);
   bh26_w103_3 <= Compressor_14_3_F500_uid75_bh26_uid290_Out0(1);
   bh26_w104_2 <= Compressor_14_3_F500_uid75_bh26_uid290_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid292_In0 <= "" & bh26_w104_0 & bh26_w104_1 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid292_In1 <= "" & bh26_w105_0;
   Compressor_14_3_F500_uid75_uid292: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid292_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid292_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid292_Out0_copy293);
   Compressor_14_3_F500_uid75_bh26_uid292_Out0 <= Compressor_14_3_F500_uid75_bh26_uid292_Out0_copy293; -- output copy to hold a pipeline register if needed

   bh26_w104_3 <= Compressor_14_3_F500_uid75_bh26_uid292_Out0(0);
   bh26_w105_1 <= Compressor_14_3_F500_uid75_bh26_uid292_Out0(1);

   Compressor_14_3_F500_uid75_bh26_uid294_In0 <= "" & bh26_w21_5 & bh26_w21_4 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid294_In1 <= "" & bh26_w22_3;
   Compressor_14_3_F500_uid75_uid294: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid294_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid294_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid294_Out0_copy295);
   Compressor_14_3_F500_uid75_bh26_uid294_Out0 <= Compressor_14_3_F500_uid75_bh26_uid294_Out0_copy295; -- output copy to hold a pipeline register if needed

   bh26_w21_6 <= Compressor_14_3_F500_uid75_bh26_uid294_Out0(0);
   bh26_w22_4 <= Compressor_14_3_F500_uid75_bh26_uid294_Out0(1);
   bh26_w23_6 <= Compressor_14_3_F500_uid75_bh26_uid294_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid296_In0 <= "" & bh26_w23_5 & bh26_w23_4 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid296_In1 <= "" & bh26_w24_5;
   Compressor_14_3_F500_uid75_uid296: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid296_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid296_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid296_Out0_copy297);
   Compressor_14_3_F500_uid75_bh26_uid296_Out0 <= Compressor_14_3_F500_uid75_bh26_uid296_Out0_copy297; -- output copy to hold a pipeline register if needed

   bh26_w23_7 <= Compressor_14_3_F500_uid75_bh26_uid296_Out0(0);
   bh26_w24_6 <= Compressor_14_3_F500_uid75_bh26_uid296_Out0(1);
   bh26_w25_6 <= Compressor_14_3_F500_uid75_bh26_uid296_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid298_In0 <= "" & bh26_w25_5 & bh26_w25_4 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid298_In1 <= "" & bh26_w26_5;
   Compressor_14_3_F500_uid75_uid298: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid298_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid298_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid298_Out0_copy299);
   Compressor_14_3_F500_uid75_bh26_uid298_Out0 <= Compressor_14_3_F500_uid75_bh26_uid298_Out0_copy299; -- output copy to hold a pipeline register if needed

   bh26_w25_7 <= Compressor_14_3_F500_uid75_bh26_uid298_Out0(0);
   bh26_w26_6 <= Compressor_14_3_F500_uid75_bh26_uid298_Out0(1);
   bh26_w27_6 <= Compressor_14_3_F500_uid75_bh26_uid298_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid300_In0 <= "" & bh26_w27_5 & bh26_w27_4 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid300_In1 <= "" & bh26_w28_5;
   Compressor_14_3_F500_uid75_uid300: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid300_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid300_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid300_Out0_copy301);
   Compressor_14_3_F500_uid75_bh26_uid300_Out0 <= Compressor_14_3_F500_uid75_bh26_uid300_Out0_copy301; -- output copy to hold a pipeline register if needed

   bh26_w27_7 <= Compressor_14_3_F500_uid75_bh26_uid300_Out0(0);
   bh26_w28_6 <= Compressor_14_3_F500_uid75_bh26_uid300_Out0(1);
   bh26_w29_6 <= Compressor_14_3_F500_uid75_bh26_uid300_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid302_In0 <= "" & bh26_w29_5 & bh26_w29_4 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid302_In1 <= "" & bh26_w30_5;
   Compressor_14_3_F500_uid75_uid302: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid302_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid302_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid302_Out0_copy303);
   Compressor_14_3_F500_uid75_bh26_uid302_Out0 <= Compressor_14_3_F500_uid75_bh26_uid302_Out0_copy303; -- output copy to hold a pipeline register if needed

   bh26_w29_7 <= Compressor_14_3_F500_uid75_bh26_uid302_Out0(0);
   bh26_w30_6 <= Compressor_14_3_F500_uid75_bh26_uid302_Out0(1);
   bh26_w31_6 <= Compressor_14_3_F500_uid75_bh26_uid302_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid304_In0 <= "" & bh26_w31_5 & bh26_w31_4 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid304_In1 <= "" & bh26_w32_5;
   Compressor_14_3_F500_uid75_uid304: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid304_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid304_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid304_Out0_copy305);
   Compressor_14_3_F500_uid75_bh26_uid304_Out0 <= Compressor_14_3_F500_uid75_bh26_uid304_Out0_copy305; -- output copy to hold a pipeline register if needed

   bh26_w31_7 <= Compressor_14_3_F500_uid75_bh26_uid304_Out0(0);
   bh26_w32_6 <= Compressor_14_3_F500_uid75_bh26_uid304_Out0(1);
   bh26_w33_6 <= Compressor_14_3_F500_uid75_bh26_uid304_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid306_In0 <= "" & bh26_w33_5 & bh26_w33_4 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid306_In1 <= "" & bh26_w34_6;
   Compressor_14_3_F500_uid75_uid306: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid306_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid306_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid306_Out0_copy307);
   Compressor_14_3_F500_uid75_bh26_uid306_Out0 <= Compressor_14_3_F500_uid75_bh26_uid306_Out0_copy307; -- output copy to hold a pipeline register if needed

   bh26_w33_7 <= Compressor_14_3_F500_uid75_bh26_uid306_Out0(0);
   bh26_w34_7 <= Compressor_14_3_F500_uid75_bh26_uid306_Out0(1);
   bh26_w35_8 <= Compressor_14_3_F500_uid75_bh26_uid306_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid308_In0 <= "" & bh26_w35_7 & bh26_w35_6 & "0";
   Compressor_23_3_F500_uid53_bh26_uid308_In1 <= "" & bh26_w36_8 & bh26_w36_7;
   Compressor_23_3_F500_uid53_uid308: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid308_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid308_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid308_Out0_copy309);
   Compressor_23_3_F500_uid53_bh26_uid308_Out0 <= Compressor_23_3_F500_uid53_bh26_uid308_Out0_copy309; -- output copy to hold a pipeline register if needed

   bh26_w35_9 <= Compressor_23_3_F500_uid53_bh26_uid308_Out0(0);
   bh26_w36_9 <= Compressor_23_3_F500_uid53_bh26_uid308_Out0(1);
   bh26_w37_7 <= Compressor_23_3_F500_uid53_bh26_uid308_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid310_In0 <= "" & bh26_w38_8 & bh26_w38_7 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid310_In1 <= "" & bh26_w39_6;
   Compressor_14_3_F500_uid75_uid310: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid310_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid310_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid310_Out0_copy311);
   Compressor_14_3_F500_uid75_bh26_uid310_Out0 <= Compressor_14_3_F500_uid75_bh26_uid310_Out0_copy311; -- output copy to hold a pipeline register if needed

   bh26_w38_9 <= Compressor_14_3_F500_uid75_bh26_uid310_Out0(0);
   bh26_w39_7 <= Compressor_14_3_F500_uid75_bh26_uid310_Out0(1);
   bh26_w40_9 <= Compressor_14_3_F500_uid75_bh26_uid310_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid312_In0 <= "" & bh26_w40_8 & bh26_w40_7 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid312_In1 <= "" & bh26_w41_6;
   Compressor_14_3_F500_uid75_uid312: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid312_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid312_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid312_Out0_copy313);
   Compressor_14_3_F500_uid75_bh26_uid312_Out0 <= Compressor_14_3_F500_uid75_bh26_uid312_Out0_copy313; -- output copy to hold a pipeline register if needed

   bh26_w40_10 <= Compressor_14_3_F500_uid75_bh26_uid312_Out0(0);
   bh26_w41_7 <= Compressor_14_3_F500_uid75_bh26_uid312_Out0(1);
   bh26_w42_9 <= Compressor_14_3_F500_uid75_bh26_uid312_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid314_In0 <= "" & bh26_w42_8 & bh26_w42_7 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid314_In1 <= "" & bh26_w43_6;
   Compressor_14_3_F500_uid75_uid314: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid314_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid314_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid314_Out0_copy315);
   Compressor_14_3_F500_uid75_bh26_uid314_Out0 <= Compressor_14_3_F500_uid75_bh26_uid314_Out0_copy315; -- output copy to hold a pipeline register if needed

   bh26_w42_10 <= Compressor_14_3_F500_uid75_bh26_uid314_Out0(0);
   bh26_w43_7 <= Compressor_14_3_F500_uid75_bh26_uid314_Out0(1);
   bh26_w44_9 <= Compressor_14_3_F500_uid75_bh26_uid314_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid316_In0 <= "" & bh26_w44_8 & bh26_w44_7 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid316_In1 <= "" & bh26_w45_6;
   Compressor_14_3_F500_uid75_uid316: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid316_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid316_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid316_Out0_copy317);
   Compressor_14_3_F500_uid75_bh26_uid316_Out0 <= Compressor_14_3_F500_uid75_bh26_uid316_Out0_copy317; -- output copy to hold a pipeline register if needed

   bh26_w44_10 <= Compressor_14_3_F500_uid75_bh26_uid316_Out0(0);
   bh26_w45_7 <= Compressor_14_3_F500_uid75_bh26_uid316_Out0(1);
   bh26_w46_9 <= Compressor_14_3_F500_uid75_bh26_uid316_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid318_In0 <= "" & bh26_w46_8 & bh26_w46_7 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid318_In1 <= "" & bh26_w47_6;
   Compressor_14_3_F500_uid75_uid318: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid318_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid318_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid318_Out0_copy319);
   Compressor_14_3_F500_uid75_bh26_uid318_Out0 <= Compressor_14_3_F500_uid75_bh26_uid318_Out0_copy319; -- output copy to hold a pipeline register if needed

   bh26_w46_10 <= Compressor_14_3_F500_uid75_bh26_uid318_Out0(0);
   bh26_w47_7 <= Compressor_14_3_F500_uid75_bh26_uid318_Out0(1);
   bh26_w48_10 <= Compressor_14_3_F500_uid75_bh26_uid318_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid320_In0 <= "" & bh26_w48_9 & bh26_w48_8 & "0";
   Compressor_23_3_F500_uid53_bh26_uid320_In1 <= "" & bh26_w49_7 & bh26_w49_5;
   Compressor_23_3_F500_uid53_uid320: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid320_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid320_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid320_Out0_copy321);
   Compressor_23_3_F500_uid53_bh26_uid320_Out0 <= Compressor_23_3_F500_uid53_bh26_uid320_Out0_copy321; -- output copy to hold a pipeline register if needed

   bh26_w48_11 <= Compressor_23_3_F500_uid53_bh26_uid320_Out0(0);
   bh26_w49_8 <= Compressor_23_3_F500_uid53_bh26_uid320_Out0(1);
   bh26_w50_10 <= Compressor_23_3_F500_uid53_bh26_uid320_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid322_In0 <= "" & bh26_w50_9 & bh26_w50_8 & "0";
   Compressor_23_3_F500_uid53_bh26_uid322_In1 <= "" & bh26_w51_9 & bh26_w51_10;
   Compressor_23_3_F500_uid53_uid322: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid322_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid322_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid322_Out0_copy323);
   Compressor_23_3_F500_uid53_bh26_uid322_Out0 <= Compressor_23_3_F500_uid53_bh26_uid322_Out0_copy323; -- output copy to hold a pipeline register if needed

   bh26_w50_11 <= Compressor_23_3_F500_uid53_bh26_uid322_Out0(0);
   bh26_w51_11 <= Compressor_23_3_F500_uid53_bh26_uid322_Out0(1);
   bh26_w52_10 <= Compressor_23_3_F500_uid53_bh26_uid322_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid324_In0 <= "" & bh26_w52_8 & bh26_w52_9 & "0";
   Compressor_23_3_F500_uid53_bh26_uid324_In1 <= "" & bh26_w53_10 & bh26_w53_9;
   Compressor_23_3_F500_uid53_uid324: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid324_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid324_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid324_Out0_copy325);
   Compressor_23_3_F500_uid53_bh26_uid324_Out0 <= Compressor_23_3_F500_uid53_bh26_uid324_Out0_copy325; -- output copy to hold a pipeline register if needed

   bh26_w52_11 <= Compressor_23_3_F500_uid53_bh26_uid324_Out0(0);
   bh26_w53_11 <= Compressor_23_3_F500_uid53_bh26_uid324_Out0(1);
   bh26_w54_10 <= Compressor_23_3_F500_uid53_bh26_uid324_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid326_In0 <= "" & bh26_w54_9 & bh26_w54_6 & "0";
   Compressor_23_3_F500_uid53_bh26_uid326_In1 <= "" & bh26_w55_10 & bh26_w55_9;
   Compressor_23_3_F500_uid53_uid326: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid326_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid326_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid326_Out0_copy327);
   Compressor_23_3_F500_uid53_bh26_uid326_Out0 <= Compressor_23_3_F500_uid53_bh26_uid326_Out0_copy327; -- output copy to hold a pipeline register if needed

   bh26_w54_11 <= Compressor_23_3_F500_uid53_bh26_uid326_Out0(0);
   bh26_w55_11 <= Compressor_23_3_F500_uid53_bh26_uid326_Out0(1);
   bh26_w56_10 <= Compressor_23_3_F500_uid53_bh26_uid326_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid328_In0 <= "" & bh26_w56_9 & bh26_w56_6 & "0";
   Compressor_23_3_F500_uid53_bh26_uid328_In1 <= "" & bh26_w57_10 & bh26_w57_9;
   Compressor_23_3_F500_uid53_uid328: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid328_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid328_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid328_Out0_copy329);
   Compressor_23_3_F500_uid53_bh26_uid328_Out0 <= Compressor_23_3_F500_uid53_bh26_uid328_Out0_copy329; -- output copy to hold a pipeline register if needed

   bh26_w56_11 <= Compressor_23_3_F500_uid53_bh26_uid328_Out0(0);
   bh26_w57_11 <= Compressor_23_3_F500_uid53_bh26_uid328_Out0(1);
   bh26_w58_10 <= Compressor_23_3_F500_uid53_bh26_uid328_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid330_In0 <= "" & bh26_w58_9 & bh26_w58_6 & "0";
   Compressor_23_3_F500_uid53_bh26_uid330_In1 <= "" & bh26_w59_10 & bh26_w59_9;
   Compressor_23_3_F500_uid53_uid330: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid330_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid330_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid330_Out0_copy331);
   Compressor_23_3_F500_uid53_bh26_uid330_Out0 <= Compressor_23_3_F500_uid53_bh26_uid330_Out0_copy331; -- output copy to hold a pipeline register if needed

   bh26_w58_11 <= Compressor_23_3_F500_uid53_bh26_uid330_Out0(0);
   bh26_w59_11 <= Compressor_23_3_F500_uid53_bh26_uid330_Out0(1);
   bh26_w60_10 <= Compressor_23_3_F500_uid53_bh26_uid330_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid332_In0 <= "" & bh26_w60_9 & bh26_w60_6 & "0";
   Compressor_23_3_F500_uid53_bh26_uid332_In1 <= "" & bh26_w61_10 & bh26_w61_9;
   Compressor_23_3_F500_uid53_uid332: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid332_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid332_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid332_Out0_copy333);
   Compressor_23_3_F500_uid53_bh26_uid332_Out0 <= Compressor_23_3_F500_uid53_bh26_uid332_Out0_copy333; -- output copy to hold a pipeline register if needed

   bh26_w60_11 <= Compressor_23_3_F500_uid53_bh26_uid332_Out0(0);
   bh26_w61_11 <= Compressor_23_3_F500_uid53_bh26_uid332_Out0(1);
   bh26_w62_10 <= Compressor_23_3_F500_uid53_bh26_uid332_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid334_In0 <= "" & bh26_w62_9 & bh26_w62_6 & "0";
   Compressor_23_3_F500_uid53_bh26_uid334_In1 <= "" & bh26_w63_10 & bh26_w63_9;
   Compressor_23_3_F500_uid53_uid334: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid334_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid334_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid334_Out0_copy335);
   Compressor_23_3_F500_uid53_bh26_uid334_Out0 <= Compressor_23_3_F500_uid53_bh26_uid334_Out0_copy335; -- output copy to hold a pipeline register if needed

   bh26_w62_11 <= Compressor_23_3_F500_uid53_bh26_uid334_Out0(0);
   bh26_w63_11 <= Compressor_23_3_F500_uid53_bh26_uid334_Out0(1);
   bh26_w64_10 <= Compressor_23_3_F500_uid53_bh26_uid334_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid336_In0 <= "" & bh26_w64_9 & bh26_w64_6 & "0";
   Compressor_23_3_F500_uid53_bh26_uid336_In1 <= "" & bh26_w65_10 & bh26_w65_9;
   Compressor_23_3_F500_uid53_uid336: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid336_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid336_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid336_Out0_copy337);
   Compressor_23_3_F500_uid53_bh26_uid336_Out0 <= Compressor_23_3_F500_uid53_bh26_uid336_Out0_copy337; -- output copy to hold a pipeline register if needed

   bh26_w64_11 <= Compressor_23_3_F500_uid53_bh26_uid336_Out0(0);
   bh26_w65_11 <= Compressor_23_3_F500_uid53_bh26_uid336_Out0(1);
   bh26_w66_10 <= Compressor_23_3_F500_uid53_bh26_uid336_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid338_In0 <= "" & bh26_w66_9 & bh26_w66_6 & "0";
   Compressor_23_3_F500_uid53_bh26_uid338_In1 <= "" & bh26_w67_10 & bh26_w67_9;
   Compressor_23_3_F500_uid53_uid338: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid338_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid338_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid338_Out0_copy339);
   Compressor_23_3_F500_uid53_bh26_uid338_Out0 <= Compressor_23_3_F500_uid53_bh26_uid338_Out0_copy339; -- output copy to hold a pipeline register if needed

   bh26_w66_11 <= Compressor_23_3_F500_uid53_bh26_uid338_Out0(0);
   bh26_w67_11 <= Compressor_23_3_F500_uid53_bh26_uid338_Out0(1);
   bh26_w68_10 <= Compressor_23_3_F500_uid53_bh26_uid338_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid340_In0 <= "" & bh26_w68_9 & bh26_w68_6 & "0";
   Compressor_23_3_F500_uid53_bh26_uid340_In1 <= "" & bh26_w69_10 & bh26_w69_9;
   Compressor_23_3_F500_uid53_uid340: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid340_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid340_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid340_Out0_copy341);
   Compressor_23_3_F500_uid53_bh26_uid340_Out0 <= Compressor_23_3_F500_uid53_bh26_uid340_Out0_copy341; -- output copy to hold a pipeline register if needed

   bh26_w68_11 <= Compressor_23_3_F500_uid53_bh26_uid340_Out0(0);
   bh26_w69_11 <= Compressor_23_3_F500_uid53_bh26_uid340_Out0(1);
   bh26_w70_10 <= Compressor_23_3_F500_uid53_bh26_uid340_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid342_In0 <= "" & bh26_w70_9 & bh26_w70_8 & "0";
   Compressor_23_3_F500_uid53_bh26_uid342_In1 <= "" & bh26_w71_8 & bh26_w71_9;
   Compressor_23_3_F500_uid53_uid342: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid342_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid342_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid342_Out0_copy343);
   Compressor_23_3_F500_uid53_bh26_uid342_Out0 <= Compressor_23_3_F500_uid53_bh26_uid342_Out0_copy343; -- output copy to hold a pipeline register if needed

   bh26_w70_11 <= Compressor_23_3_F500_uid53_bh26_uid342_Out0(0);
   bh26_w71_10 <= Compressor_23_3_F500_uid53_bh26_uid342_Out0(1);
   bh26_w72_11 <= Compressor_23_3_F500_uid53_bh26_uid342_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid344_In0 <= "" & bh26_w72_8 & bh26_w72_10 & bh26_w72_9;
   Compressor_23_3_F500_uid53_bh26_uid344_In1 <= "" & bh26_w73_8 & bh26_w73_5;
   Compressor_23_3_F500_uid53_uid344: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid344_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid344_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid344_Out0_copy345);
   Compressor_23_3_F500_uid53_bh26_uid344_Out0 <= Compressor_23_3_F500_uid53_bh26_uid344_Out0_copy345; -- output copy to hold a pipeline register if needed

   bh26_w72_12 <= Compressor_23_3_F500_uid53_bh26_uid344_Out0(0);
   bh26_w73_9 <= Compressor_23_3_F500_uid53_bh26_uid344_Out0(1);
   bh26_w74_10 <= Compressor_23_3_F500_uid53_bh26_uid344_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid346_In0 <= "" & bh26_w74_9 & bh26_w74_8 & "0";
   Compressor_23_3_F500_uid53_bh26_uid346_In1 <= "" & bh26_w75_8 & bh26_w75_5;
   Compressor_23_3_F500_uid53_uid346: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid346_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid346_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid346_Out0_copy347);
   Compressor_23_3_F500_uid53_bh26_uid346_Out0 <= Compressor_23_3_F500_uid53_bh26_uid346_Out0_copy347; -- output copy to hold a pipeline register if needed

   bh26_w74_11 <= Compressor_23_3_F500_uid53_bh26_uid346_Out0(0);
   bh26_w75_9 <= Compressor_23_3_F500_uid53_bh26_uid346_Out0(1);
   bh26_w76_10 <= Compressor_23_3_F500_uid53_bh26_uid346_Out0(2);

   Compressor_23_3_F500_uid53_bh26_uid348_In0 <= "" & bh26_w76_9 & bh26_w76_8 & "0";
   Compressor_23_3_F500_uid53_bh26_uid348_In1 <= "" & bh26_w77_7 & bh26_w77_4;
   Compressor_23_3_F500_uid53_uid348: Compressor_23_3_F500_uid53
      port map ( X0 => Compressor_23_3_F500_uid53_bh26_uid348_In0,
                 X1 => Compressor_23_3_F500_uid53_bh26_uid348_In1,
                 R => Compressor_23_3_F500_uid53_bh26_uid348_Out0_copy349);
   Compressor_23_3_F500_uid53_bh26_uid348_Out0 <= Compressor_23_3_F500_uid53_bh26_uid348_Out0_copy349; -- output copy to hold a pipeline register if needed

   bh26_w76_11 <= Compressor_23_3_F500_uid53_bh26_uid348_Out0(0);
   bh26_w77_8 <= Compressor_23_3_F500_uid53_bh26_uid348_Out0(1);
   bh26_w78_9 <= Compressor_23_3_F500_uid53_bh26_uid348_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid350_In0 <= "" & bh26_w78_8 & bh26_w78_7 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid350_In1 <= "" & bh26_w79_6;
   Compressor_14_3_F500_uid75_uid350: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid350_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid350_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid350_Out0_copy351);
   Compressor_14_3_F500_uid75_bh26_uid350_Out0 <= Compressor_14_3_F500_uid75_bh26_uid350_Out0_copy351; -- output copy to hold a pipeline register if needed

   bh26_w78_10 <= Compressor_14_3_F500_uid75_bh26_uid350_Out0(0);
   bh26_w79_7 <= Compressor_14_3_F500_uid75_bh26_uid350_Out0(1);
   bh26_w80_9 <= Compressor_14_3_F500_uid75_bh26_uid350_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid352_In0 <= "" & bh26_w80_8 & bh26_w80_7 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid352_In1 <= "" & bh26_w81_6;
   Compressor_14_3_F500_uid75_uid352: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid352_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid352_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid352_Out0_copy353);
   Compressor_14_3_F500_uid75_bh26_uid352_Out0 <= Compressor_14_3_F500_uid75_bh26_uid352_Out0_copy353; -- output copy to hold a pipeline register if needed

   bh26_w80_10 <= Compressor_14_3_F500_uid75_bh26_uid352_Out0(0);
   bh26_w81_7 <= Compressor_14_3_F500_uid75_bh26_uid352_Out0(1);
   bh26_w82_9 <= Compressor_14_3_F500_uid75_bh26_uid352_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid354_In0 <= "" & bh26_w82_8 & bh26_w82_7 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid354_In1 <= "" & bh26_w83_6;
   Compressor_14_3_F500_uid75_uid354: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid354_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid354_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid354_Out0_copy355);
   Compressor_14_3_F500_uid75_bh26_uid354_Out0 <= Compressor_14_3_F500_uid75_bh26_uid354_Out0_copy355; -- output copy to hold a pipeline register if needed

   bh26_w82_10 <= Compressor_14_3_F500_uid75_bh26_uid354_Out0(0);
   bh26_w83_7 <= Compressor_14_3_F500_uid75_bh26_uid354_Out0(1);
   bh26_w84_9 <= Compressor_14_3_F500_uid75_bh26_uid354_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid356_In0 <= "" & bh26_w84_8 & bh26_w84_7 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid356_In1 <= "" & bh26_w85_6;
   Compressor_14_3_F500_uid75_uid356: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid356_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid356_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid356_Out0_copy357);
   Compressor_14_3_F500_uid75_bh26_uid356_Out0 <= Compressor_14_3_F500_uid75_bh26_uid356_Out0_copy357; -- output copy to hold a pipeline register if needed

   bh26_w84_10 <= Compressor_14_3_F500_uid75_bh26_uid356_Out0(0);
   bh26_w85_7 <= Compressor_14_3_F500_uid75_bh26_uid356_Out0(1);
   bh26_w86_9 <= Compressor_14_3_F500_uid75_bh26_uid356_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid358_In0 <= "" & bh26_w86_8 & bh26_w86_7 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid358_In1 <= "" & bh26_w87_5;
   Compressor_14_3_F500_uid75_uid358: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid358_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid358_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid358_Out0_copy359);
   Compressor_14_3_F500_uid75_bh26_uid358_Out0 <= Compressor_14_3_F500_uid75_bh26_uid358_Out0_copy359; -- output copy to hold a pipeline register if needed

   bh26_w86_10 <= Compressor_14_3_F500_uid75_bh26_uid358_Out0(0);
   bh26_w87_6 <= Compressor_14_3_F500_uid75_bh26_uid358_Out0(1);
   bh26_w88_8 <= Compressor_14_3_F500_uid75_bh26_uid358_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid360_In0 <= "" & bh26_w88_6 & bh26_w88_7 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid360_In1 <= "" & bh26_w89_4;
   Compressor_14_3_F500_uid75_uid360: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid360_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid360_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid360_Out0_copy361);
   Compressor_14_3_F500_uid75_bh26_uid360_Out0 <= Compressor_14_3_F500_uid75_bh26_uid360_Out0_copy361; -- output copy to hold a pipeline register if needed

   bh26_w88_9 <= Compressor_14_3_F500_uid75_bh26_uid360_Out0(0);
   bh26_w89_5 <= Compressor_14_3_F500_uid75_bh26_uid360_Out0(1);
   bh26_w90_7 <= Compressor_14_3_F500_uid75_bh26_uid360_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid362_In0 <= "" & bh26_w90_5 & bh26_w90_6 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid362_In1 <= "" & bh26_w91_4;
   Compressor_14_3_F500_uid75_uid362: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid362_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid362_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid362_Out0_copy363);
   Compressor_14_3_F500_uid75_bh26_uid362_Out0 <= Compressor_14_3_F500_uid75_bh26_uid362_Out0_copy363; -- output copy to hold a pipeline register if needed

   bh26_w90_8 <= Compressor_14_3_F500_uid75_bh26_uid362_Out0(0);
   bh26_w91_5 <= Compressor_14_3_F500_uid75_bh26_uid362_Out0(1);
   bh26_w92_7 <= Compressor_14_3_F500_uid75_bh26_uid362_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid364_In0 <= "" & bh26_w92_6 & bh26_w92_5 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid364_In1 <= "" & bh26_w93_4;
   Compressor_14_3_F500_uid75_uid364: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid364_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid364_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid364_Out0_copy365);
   Compressor_14_3_F500_uid75_bh26_uid364_Out0 <= Compressor_14_3_F500_uid75_bh26_uid364_Out0_copy365; -- output copy to hold a pipeline register if needed

   bh26_w92_8 <= Compressor_14_3_F500_uid75_bh26_uid364_Out0(0);
   bh26_w93_5 <= Compressor_14_3_F500_uid75_bh26_uid364_Out0(1);
   bh26_w94_7 <= Compressor_14_3_F500_uid75_bh26_uid364_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid366_In0 <= "" & bh26_w94_6 & bh26_w94_5 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid366_In1 <= "" & bh26_w95_4;
   Compressor_14_3_F500_uid75_uid366: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid366_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid366_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid366_Out0_copy367);
   Compressor_14_3_F500_uid75_bh26_uid366_Out0 <= Compressor_14_3_F500_uid75_bh26_uid366_Out0_copy367; -- output copy to hold a pipeline register if needed

   bh26_w94_8 <= Compressor_14_3_F500_uid75_bh26_uid366_Out0(0);
   bh26_w95_5 <= Compressor_14_3_F500_uid75_bh26_uid366_Out0(1);
   bh26_w96_7 <= Compressor_14_3_F500_uid75_bh26_uid366_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid368_In0 <= "" & bh26_w96_6 & bh26_w96_5 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid368_In1 <= "" & bh26_w97_4;
   Compressor_14_3_F500_uid75_uid368: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid368_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid368_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid368_Out0_copy369);
   Compressor_14_3_F500_uid75_bh26_uid368_Out0 <= Compressor_14_3_F500_uid75_bh26_uid368_Out0_copy369; -- output copy to hold a pipeline register if needed

   bh26_w96_8 <= Compressor_14_3_F500_uid75_bh26_uid368_Out0(0);
   bh26_w97_5 <= Compressor_14_3_F500_uid75_bh26_uid368_Out0(1);
   bh26_w98_7 <= Compressor_14_3_F500_uid75_bh26_uid368_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid370_In0 <= "" & bh26_w98_6 & bh26_w98_5 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid370_In1 <= "" & bh26_w99_4;
   Compressor_14_3_F500_uid75_uid370: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid370_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid370_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid370_Out0_copy371);
   Compressor_14_3_F500_uid75_bh26_uid370_Out0 <= Compressor_14_3_F500_uid75_bh26_uid370_Out0_copy371; -- output copy to hold a pipeline register if needed

   bh26_w98_8 <= Compressor_14_3_F500_uid75_bh26_uid370_Out0(0);
   bh26_w99_5 <= Compressor_14_3_F500_uid75_bh26_uid370_Out0(1);
   bh26_w100_7 <= Compressor_14_3_F500_uid75_bh26_uid370_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid372_In0 <= "" & bh26_w100_6 & bh26_w100_5 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid372_In1 <= "" & bh26_w101_3;
   Compressor_14_3_F500_uid75_uid372: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid372_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid372_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid372_Out0_copy373);
   Compressor_14_3_F500_uid75_bh26_uid372_Out0 <= Compressor_14_3_F500_uid75_bh26_uid372_Out0_copy373; -- output copy to hold a pipeline register if needed

   bh26_w100_8 <= Compressor_14_3_F500_uid75_bh26_uid372_Out0(0);
   bh26_w101_4 <= Compressor_14_3_F500_uid75_bh26_uid372_Out0(1);
   bh26_w102_6 <= Compressor_14_3_F500_uid75_bh26_uid372_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid374_In0 <= "" & bh26_w102_5 & bh26_w102_4 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid374_In1 <= "" & bh26_w103_3;
   Compressor_14_3_F500_uid75_uid374: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid374_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid374_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid374_Out0_copy375);
   Compressor_14_3_F500_uid75_bh26_uid374_Out0 <= Compressor_14_3_F500_uid75_bh26_uid374_Out0_copy375; -- output copy to hold a pipeline register if needed

   bh26_w102_7 <= Compressor_14_3_F500_uid75_bh26_uid374_Out0(0);
   bh26_w103_4 <= Compressor_14_3_F500_uid75_bh26_uid374_Out0(1);
   bh26_w104_4 <= Compressor_14_3_F500_uid75_bh26_uid374_Out0(2);

   Compressor_14_3_F500_uid75_bh26_uid376_In0 <= "" & bh26_w104_3 & bh26_w104_2 & "0" & "0";
   Compressor_14_3_F500_uid75_bh26_uid376_In1 <= "" & bh26_w105_1;
   Compressor_14_3_F500_uid75_uid376: Compressor_14_3_F500_uid75
      port map ( X0 => Compressor_14_3_F500_uid75_bh26_uid376_In0,
                 X1 => Compressor_14_3_F500_uid75_bh26_uid376_In1,
                 R => Compressor_14_3_F500_uid75_bh26_uid376_Out0_copy377);
   Compressor_14_3_F500_uid75_bh26_uid376_Out0 <= Compressor_14_3_F500_uid75_bh26_uid376_Out0_copy377; -- output copy to hold a pipeline register if needed

   bh26_w104_5 <= Compressor_14_3_F500_uid75_bh26_uid376_Out0(0);
   bh26_w105_2 <= Compressor_14_3_F500_uid75_bh26_uid376_Out0(1);
   tmp_bitheapResult_bh26_22 <= bh26_w22_4 & bh26_w21_6 & bh26_w20_3 & bh26_w19_4 & bh26_w18_2 & bh26_w17_2 & bh26_w16_0 & bh26_w15_0 & bh26_w14_0 & bh26_w13_0 & bh26_w12_0 & bh26_w11_0 & bh26_w10_0 & bh26_w9_0 & bh26_w8_0 & bh26_w7_0 & bh26_w6_0 & bh26_w5_0 & bh26_w4_0 & bh26_w3_0 & bh26_w2_0 & bh26_w1_0 & bh26_w0_0;

   bitheapFinalAdd_bh26_In0 <= "0" & bh26_w105_2 & bh26_w104_5 & bh26_w103_4 & bh26_w102_7 & bh26_w101_4 & bh26_w100_8 & bh26_w99_5 & bh26_w98_8 & bh26_w97_5 & bh26_w96_8 & bh26_w95_5 & bh26_w94_8 & bh26_w93_5 & bh26_w92_7 & bh26_w91_5 & bh26_w90_7 & bh26_w89_5 & bh26_w88_8 & bh26_w87_6 & bh26_w86_10 & bh26_w85_7 & bh26_w84_10 & bh26_w83_7 & bh26_w82_10 & bh26_w81_7 & bh26_w80_10 & bh26_w79_7 & bh26_w78_10 & bh26_w77_8 & bh26_w76_11 & bh26_w75_9 & bh26_w74_10 & bh26_w73_9 & bh26_w72_11 & bh26_w71_10 & bh26_w70_11 & bh26_w69_11 & bh26_w68_11 & bh26_w67_11 & bh26_w66_11 & bh26_w65_11 & bh26_w64_11 & bh26_w63_11 & bh26_w62_11 & bh26_w61_11 & bh26_w60_11 & bh26_w59_11 & bh26_w58_11 & bh26_w57_11 & bh26_w56_11 & bh26_w55_11 & bh26_w54_10 & bh26_w53_11 & bh26_w52_10 & bh26_w51_11 & bh26_w50_11 & bh26_w49_8 & bh26_w48_11 & bh26_w47_7 & bh26_w46_10 & bh26_w45_7 & bh26_w44_10 & bh26_w43_7 & bh26_w42_10 & bh26_w41_7 & bh26_w40_10 & bh26_w39_7 & bh26_w38_9 & bh26_w37_7 & bh26_w36_9 & bh26_w35_9 & bh26_w34_7 & bh26_w33_7 & bh26_w32_6 & bh26_w31_7 & bh26_w30_6 & bh26_w29_7 & bh26_w28_6 & bh26_w27_7 & bh26_w26_6 & bh26_w25_7 & bh26_w24_6 & bh26_w23_7;
   bitheapFinalAdd_bh26_In1 <= "0" & "0" & bh26_w104_4 & "0" & bh26_w102_6 & "0" & bh26_w100_7 & "0" & bh26_w98_7 & "0" & bh26_w96_7 & "0" & bh26_w94_7 & "0" & bh26_w92_8 & "0" & bh26_w90_8 & "0" & bh26_w88_9 & "0" & bh26_w86_9 & "0" & bh26_w84_9 & "0" & bh26_w82_9 & "0" & bh26_w80_9 & "0" & bh26_w78_9 & "0" & bh26_w76_10 & "0" & bh26_w74_11 & "0" & bh26_w72_12 & "0" & bh26_w70_10 & "0" & bh26_w68_10 & "0" & bh26_w66_10 & "0" & bh26_w64_10 & "0" & bh26_w62_10 & "0" & bh26_w60_10 & "0" & bh26_w58_10 & "0" & bh26_w56_10 & "0" & bh26_w54_11 & "0" & bh26_w52_11 & "0" & bh26_w50_10 & "0" & bh26_w48_10 & "0" & bh26_w46_9 & "0" & bh26_w44_9 & "0" & bh26_w42_9 & "0" & bh26_w40_9 & "0" & "0" & bh26_w37_6 & "0" & bh26_w35_8 & "0" & bh26_w33_6 & "0" & bh26_w31_6 & "0" & bh26_w29_6 & "0" & bh26_w27_6 & "0" & bh26_w25_6 & "0" & bh26_w23_6;
   bitheapFinalAdd_bh26_Cin <= '0';

   bitheapFinalAdd_bh26: IntAdder_84_F500_uid379
      port map ( clk  => clk,
                 ce => ce,
                 Cin => bitheapFinalAdd_bh26_Cin,
                 X => bitheapFinalAdd_bh26_In0,
                 Y => bitheapFinalAdd_bh26_In1,
                 R => bitheapFinalAdd_bh26_Out);
   bitheapResult_bh26 <= bitheapFinalAdd_bh26_Out(82 downto 0) & tmp_bitheapResult_bh26_22_d2;
   R <= bitheapResult_bh26(105 downto 0);
end architecture;

--------------------------------------------------------------------------------
--                          IntAdder_65_F500_uid382
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2016)
--------------------------------------------------------------------------------
-- Pipeline depth: 1 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y Cin
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntAdder_65_F500_uid382 is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(64 downto 0);
          Y : in  std_logic_vector(64 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(64 downto 0)   );
end entity;

architecture arch of IntAdder_65_F500_uid382 is
signal Cin_0, Cin_0_d1 :  std_logic;
signal X_0, X_0_d1, X_0_d2 :  std_logic_vector(55 downto 0);
signal Y_0, Y_0_d1, Y_0_d2, Y_0_d3, Y_0_d4 :  std_logic_vector(55 downto 0);
signal S_0 :  std_logic_vector(55 downto 0);
signal R_0 :  std_logic_vector(54 downto 0);
signal Cin_1 :  std_logic;
signal X_1, X_1_d1, X_1_d2 :  std_logic_vector(10 downto 0);
signal Y_1, Y_1_d1, Y_1_d2, Y_1_d3, Y_1_d4 :  std_logic_vector(10 downto 0);
signal S_1 :  std_logic_vector(10 downto 0);
signal R_1 :  std_logic_vector(9 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               Cin_0_d1 <=  Cin_0;
               X_0_d1 <=  X_0;
               X_0_d2 <=  X_0_d1;
               Y_0_d1 <=  Y_0;
               Y_0_d2 <=  Y_0_d1;
               Y_0_d3 <=  Y_0_d2;
               Y_0_d4 <=  Y_0_d3;
               X_1_d1 <=  X_1;
               X_1_d2 <=  X_1_d1;
               Y_1_d1 <=  Y_1;
               Y_1_d2 <=  Y_1_d1;
               Y_1_d3 <=  Y_1_d2;
               Y_1_d4 <=  Y_1_d3;
            end if;
         end if;
      end process;
   Cin_0 <= Cin;
   X_0 <= '0' & X(54 downto 0);
   Y_0 <= '0' & Y(54 downto 0);
   S_0 <= X_0_d2 + Y_0_d4 + Cin_0_d1;
   R_0 <= S_0(54 downto 0);
   Cin_1 <= S_0(55);
   X_1 <= '0' & X(64 downto 55);
   Y_1 <= '0' & Y(64 downto 55);
   S_1 <= X_1_d2 + Y_1_d4 + Cin_1;
   R_1 <= S_1(9 downto 0);
   R <= R_1 & R_0 ;
end architecture;

--------------------------------------------------------------------------------
--                                FPMult_64bit
--                (FPMult_11_52_11_52_11_52_uid21_F500_uid22)
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin 2008-2021
--------------------------------------------------------------------------------
-- Pipeline depth: 4 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity FPMult_64bit is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(11+52+2 downto 0);
          Y : in  std_logic_vector(11+52+2 downto 0);
          R : out  std_logic_vector(11+52+2 downto 0)   );
end entity;

architecture arch of FPMult_64bit is
   component IntMultiplier_F500_uid24 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(52 downto 0);
             Y : in  std_logic_vector(52 downto 0);
             R : out  std_logic_vector(105 downto 0)   );
   end component;

   component IntAdder_65_F500_uid382 is
      port ( clk, ce : in std_logic;
             X : in  std_logic_vector(64 downto 0);
             Y : in  std_logic_vector(64 downto 0);
             Cin : in  std_logic;
             R : out  std_logic_vector(64 downto 0)   );
   end component;

signal sign, sign_d1, sign_d2, sign_d3, sign_d4 :  std_logic;
signal expX :  std_logic_vector(10 downto 0);
signal expY :  std_logic_vector(10 downto 0);
signal expSumPreSub, expSumPreSub_d1 :  std_logic_vector(12 downto 0);
signal bias, bias_d1 :  std_logic_vector(12 downto 0);
signal expSum, expSum_d1 :  std_logic_vector(12 downto 0);
signal sigX :  std_logic_vector(52 downto 0);
signal sigY :  std_logic_vector(52 downto 0);
signal sigProd :  std_logic_vector(105 downto 0);
signal excSel :  std_logic_vector(3 downto 0);
signal exc, exc_d1, exc_d2, exc_d3, exc_d4 :  std_logic_vector(1 downto 0);
signal norm :  std_logic;
signal expPostNorm :  std_logic_vector(12 downto 0);
signal sigProdExt, sigProdExt_d1 :  std_logic_vector(105 downto 0);
signal expSig :  std_logic_vector(64 downto 0);
signal sticky, sticky_d1 :  std_logic;
signal guard :  std_logic;
signal round :  std_logic;
signal expSigPostRound :  std_logic_vector(64 downto 0);
signal excPostNorm :  std_logic_vector(1 downto 0);
signal finalExc :  std_logic_vector(1 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               sign_d1 <=  sign;
               sign_d2 <=  sign_d1;
               sign_d3 <=  sign_d2;
               sign_d4 <=  sign_d3;
               expSumPreSub_d1 <=  expSumPreSub;
               bias_d1 <=  bias;
               expSum_d1 <=  expSum;
               exc_d1 <=  exc;
               exc_d2 <=  exc_d1;
               exc_d3 <=  exc_d2;
               exc_d4 <=  exc_d3;
               sigProdExt_d1 <=  sigProdExt;
               sticky_d1 <=  sticky;
            end if;
         end if;
      end process;
   sign <= X(63) xor Y(63);
   expX <= X(62 downto 52);
   expY <= Y(62 downto 52);
   expSumPreSub <= ("00" & expX) + ("00" & expY);
   bias <= CONV_STD_LOGIC_VECTOR(1023,13);
   expSum <= expSumPreSub_d1 - bias_d1;
   sigX <= "1" & X(51 downto 0);
   sigY <= "1" & Y(51 downto 0);
   SignificandMultiplication: IntMultiplier_F500_uid24
      port map ( clk  => clk,
                 ce => ce,
                 X => sigX,
                 Y => sigY,
                 R => sigProd);
   excSel <= X(65 downto 64) & Y(65 downto 64);
   with excSel  select  
   exc <= "00" when  "0000" | "0001" | "0100", 
          "01" when "0101",
          "10" when "0110" | "1001" | "1010" ,
          "11" when others;
   norm <= sigProd(105);
   -- exponent update
   expPostNorm <= expSum_d1 + ("000000000000" & norm);
   -- significand normalization shift
   sigProdExt <= sigProd(104 downto 0) & "0" when norm='1' else
                         sigProd(103 downto 0) & "00";
   expSig <= expPostNorm & sigProdExt(105 downto 54);
   sticky <= sigProdExt(53);
   guard <= '0' when sigProdExt_d1(52 downto 0)="00000000000000000000000000000000000000000000000000000" else '1';
   round <= sticky_d1 and ( (guard and not(sigProdExt_d1(54))) or (sigProdExt_d1(54) ))  ;
   RoundingAdder: IntAdder_65_F500_uid382
      port map ( clk  => clk,
                 ce => ce,
                 Cin => round,
                 X => expSig,
                 Y => "00000000000000000000000000000000000000000000000000000000000000000",
                 R => expSigPostRound);
   with expSigPostRound(64 downto 63)  select 
   excPostNorm <=  "01"  when  "00",
                               "10"             when "01", 
                               "00"             when "11"|"10",
                               "11"             when others;
   with exc_d4  select  
   finalExc <= exc_d4 when  "11"|"10"|"00",
                       excPostNorm when others; 
   R <= finalExc & sign_d4 & expSigPostRound(62 downto 0);
end architecture;

--------------------------------------------------------------------------------
--                          selFunction_F500_uid386
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin, Bogdan Pasca (2007-2020)
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X
-- Output signals: Y

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity selFunction_F500_uid386 is
    port (X : in  std_logic_vector(8 downto 0);
          Y : out  std_logic_vector(2 downto 0)   );
end entity;

architecture arch of selFunction_F500_uid386 is
signal Y0 :  std_logic_vector(2 downto 0);
attribute ram_extract: string;
attribute ram_style: string;
attribute ram_extract of Y0: signal is "yes";
attribute ram_style of Y0: signal is "distributed";
signal Y1 :  std_logic_vector(2 downto 0);
begin
   with X  select  Y0 <= 
      "000" when "000000000",
      "000" when "000000001",
      "000" when "000000010",
      "000" when "000000011",
      "000" when "000000100",
      "000" when "000000101",
      "000" when "000000110",
      "000" when "000000111",
      "000" when "000001000",
      "000" when "000001001",
      "000" when "000001010",
      "000" when "000001011",
      "000" when "000001100",
      "000" when "000001101",
      "000" when "000001110",
      "000" when "000001111",
      "001" when "000010000",
      "000" when "000010001",
      "000" when "000010010",
      "000" when "000010011",
      "000" when "000010100",
      "000" when "000010101",
      "000" when "000010110",
      "000" when "000010111",
      "001" when "000011000",
      "001" when "000011001",
      "001" when "000011010",
      "001" when "000011011",
      "000" when "000011100",
      "000" when "000011101",
      "000" when "000011110",
      "000" when "000011111",
      "001" when "000100000",
      "001" when "000100001",
      "001" when "000100010",
      "001" when "000100011",
      "001" when "000100100",
      "001" when "000100101",
      "001" when "000100110",
      "000" when "000100111",
      "001" when "000101000",
      "001" when "000101001",
      "001" when "000101010",
      "001" when "000101011",
      "001" when "000101100",
      "001" when "000101101",
      "001" when "000101110",
      "001" when "000101111",
      "010" when "000110000",
      "001" when "000110001",
      "001" when "000110010",
      "001" when "000110011",
      "001" when "000110100",
      "001" when "000110101",
      "001" when "000110110",
      "001" when "000110111",
      "010" when "000111000",
      "010" when "000111001",
      "001" when "000111010",
      "001" when "000111011",
      "001" when "000111100",
      "001" when "000111101",
      "001" when "000111110",
      "001" when "000111111",
      "010" when "001000000",
      "010" when "001000001",
      "010" when "001000010",
      "001" when "001000011",
      "001" when "001000100",
      "001" when "001000101",
      "001" when "001000110",
      "001" when "001000111",
      "010" when "001001000",
      "010" when "001001001",
      "010" when "001001010",
      "010" when "001001011",
      "001" when "001001100",
      "001" when "001001101",
      "001" when "001001110",
      "001" when "001001111",
      "010" when "001010000",
      "010" when "001010001",
      "010" when "001010010",
      "010" when "001010011",
      "010" when "001010100",
      "010" when "001010101",
      "001" when "001010110",
      "001" when "001010111",
      "010" when "001011000",
      "010" when "001011001",
      "010" when "001011010",
      "010" when "001011011",
      "010" when "001011100",
      "010" when "001011101",
      "010" when "001011110",
      "001" when "001011111",
      "010" when "001100000",
      "010" when "001100001",
      "010" when "001100010",
      "010" when "001100011",
      "010" when "001100100",
      "010" when "001100101",
      "010" when "001100110",
      "010" when "001100111",
      "010" when "001101000",
      "010" when "001101001",
      "010" when "001101010",
      "010" when "001101011",
      "010" when "001101100",
      "010" when "001101101",
      "010" when "001101110",
      "010" when "001101111",
      "010" when "001110000",
      "010" when "001110001",
      "010" when "001110010",
      "010" when "001110011",
      "010" when "001110100",
      "010" when "001110101",
      "010" when "001110110",
      "010" when "001110111",
      "010" when "001111000",
      "010" when "001111001",
      "010" when "001111010",
      "010" when "001111011",
      "010" when "001111100",
      "010" when "001111101",
      "010" when "001111110",
      "010" when "001111111",
      "010" when "010000000",
      "010" when "010000001",
      "010" when "010000010",
      "010" when "010000011",
      "010" when "010000100",
      "010" when "010000101",
      "010" when "010000110",
      "010" when "010000111",
      "010" when "010001000",
      "010" when "010001001",
      "010" when "010001010",
      "010" when "010001011",
      "010" when "010001100",
      "010" when "010001101",
      "010" when "010001110",
      "010" when "010001111",
      "010" when "010010000",
      "010" when "010010001",
      "010" when "010010010",
      "010" when "010010011",
      "010" when "010010100",
      "010" when "010010101",
      "010" when "010010110",
      "010" when "010010111",
      "010" when "010011000",
      "010" when "010011001",
      "010" when "010011010",
      "010" when "010011011",
      "010" when "010011100",
      "010" when "010011101",
      "010" when "010011110",
      "010" when "010011111",
      "010" when "010100000",
      "010" when "010100001",
      "010" when "010100010",
      "010" when "010100011",
      "010" when "010100100",
      "010" when "010100101",
      "010" when "010100110",
      "010" when "010100111",
      "010" when "010101000",
      "010" when "010101001",
      "010" when "010101010",
      "010" when "010101011",
      "010" when "010101100",
      "010" when "010101101",
      "010" when "010101110",
      "010" when "010101111",
      "010" when "010110000",
      "010" when "010110001",
      "010" when "010110010",
      "010" when "010110011",
      "010" when "010110100",
      "010" when "010110101",
      "010" when "010110110",
      "010" when "010110111",
      "010" when "010111000",
      "010" when "010111001",
      "010" when "010111010",
      "010" when "010111011",
      "010" when "010111100",
      "010" when "010111101",
      "010" when "010111110",
      "010" when "010111111",
      "010" when "011000000",
      "010" when "011000001",
      "010" when "011000010",
      "010" when "011000011",
      "010" when "011000100",
      "010" when "011000101",
      "010" when "011000110",
      "010" when "011000111",
      "010" when "011001000",
      "010" when "011001001",
      "010" when "011001010",
      "010" when "011001011",
      "010" when "011001100",
      "010" when "011001101",
      "010" when "011001110",
      "010" when "011001111",
      "010" when "011010000",
      "010" when "011010001",
      "010" when "011010010",
      "010" when "011010011",
      "010" when "011010100",
      "010" when "011010101",
      "010" when "011010110",
      "010" when "011010111",
      "010" when "011011000",
      "010" when "011011001",
      "010" when "011011010",
      "010" when "011011011",
      "010" when "011011100",
      "010" when "011011101",
      "010" when "011011110",
      "010" when "011011111",
      "010" when "011100000",
      "010" when "011100001",
      "010" when "011100010",
      "010" when "011100011",
      "010" when "011100100",
      "010" when "011100101",
      "010" when "011100110",
      "010" when "011100111",
      "010" when "011101000",
      "010" when "011101001",
      "010" when "011101010",
      "010" when "011101011",
      "010" when "011101100",
      "010" when "011101101",
      "010" when "011101110",
      "010" when "011101111",
      "010" when "011110000",
      "010" when "011110001",
      "010" when "011110010",
      "010" when "011110011",
      "010" when "011110100",
      "010" when "011110101",
      "010" when "011110110",
      "010" when "011110111",
      "010" when "011111000",
      "010" when "011111001",
      "010" when "011111010",
      "010" when "011111011",
      "010" when "011111100",
      "010" when "011111101",
      "010" when "011111110",
      "010" when "011111111",
      "110" when "100000000",
      "110" when "100000001",
      "110" when "100000010",
      "110" when "100000011",
      "110" when "100000100",
      "110" when "100000101",
      "110" when "100000110",
      "110" when "100000111",
      "110" when "100001000",
      "110" when "100001001",
      "110" when "100001010",
      "110" when "100001011",
      "110" when "100001100",
      "110" when "100001101",
      "110" when "100001110",
      "110" when "100001111",
      "110" when "100010000",
      "110" when "100010001",
      "110" when "100010010",
      "110" when "100010011",
      "110" when "100010100",
      "110" when "100010101",
      "110" when "100010110",
      "110" when "100010111",
      "110" when "100011000",
      "110" when "100011001",
      "110" when "100011010",
      "110" when "100011011",
      "110" when "100011100",
      "110" when "100011101",
      "110" when "100011110",
      "110" when "100011111",
      "110" when "100100000",
      "110" when "100100001",
      "110" when "100100010",
      "110" when "100100011",
      "110" when "100100100",
      "110" when "100100101",
      "110" when "100100110",
      "110" when "100100111",
      "110" when "100101000",
      "110" when "100101001",
      "110" when "100101010",
      "110" when "100101011",
      "110" when "100101100",
      "110" when "100101101",
      "110" when "100101110",
      "110" when "100101111",
      "110" when "100110000",
      "110" when "100110001",
      "110" when "100110010",
      "110" when "100110011",
      "110" when "100110100",
      "110" when "100110101",
      "110" when "100110110",
      "110" when "100110111",
      "110" when "100111000",
      "110" when "100111001",
      "110" when "100111010",
      "110" when "100111011",
      "110" when "100111100",
      "110" when "100111101",
      "110" when "100111110",
      "110" when "100111111",
      "110" when "101000000",
      "110" when "101000001",
      "110" when "101000010",
      "110" when "101000011",
      "110" when "101000100",
      "110" when "101000101",
      "110" when "101000110",
      "110" when "101000111",
      "110" when "101001000",
      "110" when "101001001",
      "110" when "101001010",
      "110" when "101001011",
      "110" when "101001100",
      "110" when "101001101",
      "110" when "101001110",
      "110" when "101001111",
      "110" when "101010000",
      "110" when "101010001",
      "110" when "101010010",
      "110" when "101010011",
      "110" when "101010100",
      "110" when "101010101",
      "110" when "101010110",
      "110" when "101010111",
      "110" when "101011000",
      "110" when "101011001",
      "110" when "101011010",
      "110" when "101011011",
      "110" when "101011100",
      "110" when "101011101",
      "110" when "101011110",
      "110" when "101011111",
      "110" when "101100000",
      "110" when "101100001",
      "110" when "101100010",
      "110" when "101100011",
      "110" when "101100100",
      "110" when "101100101",
      "110" when "101100110",
      "110" when "101100111",
      "110" when "101101000",
      "110" when "101101001",
      "110" when "101101010",
      "110" when "101101011",
      "110" when "101101100",
      "110" when "101101101",
      "110" when "101101110",
      "110" when "101101111",
      "110" when "101110000",
      "110" when "101110001",
      "110" when "101110010",
      "110" when "101110011",
      "110" when "101110100",
      "110" when "101110101",
      "110" when "101110110",
      "110" when "101110111",
      "110" when "101111000",
      "110" when "101111001",
      "110" when "101111010",
      "110" when "101111011",
      "110" when "101111100",
      "110" when "101111101",
      "110" when "101111110",
      "110" when "101111111",
      "110" when "110000000",
      "110" when "110000001",
      "110" when "110000010",
      "110" when "110000011",
      "110" when "110000100",
      "110" when "110000101",
      "110" when "110000110",
      "110" when "110000111",
      "110" when "110001000",
      "110" when "110001001",
      "110" when "110001010",
      "110" when "110001011",
      "110" when "110001100",
      "110" when "110001101",
      "110" when "110001110",
      "110" when "110001111",
      "110" when "110010000",
      "110" when "110010001",
      "110" when "110010010",
      "110" when "110010011",
      "110" when "110010100",
      "110" when "110010101",
      "110" when "110010110",
      "110" when "110010111",
      "110" when "110011000",
      "110" when "110011001",
      "110" when "110011010",
      "110" when "110011011",
      "110" when "110011100",
      "110" when "110011101",
      "110" when "110011110",
      "110" when "110011111",
      "110" when "110100000",
      "110" when "110100001",
      "110" when "110100010",
      "110" when "110100011",
      "110" when "110100100",
      "110" when "110100101",
      "110" when "110100110",
      "110" when "110100111",
      "110" when "110101000",
      "110" when "110101001",
      "110" when "110101010",
      "110" when "110101011",
      "110" when "110101100",
      "110" when "110101101",
      "110" when "110101110",
      "111" when "110101111",
      "110" when "110110000",
      "110" when "110110001",
      "110" when "110110010",
      "110" when "110110011",
      "110" when "110110100",
      "111" when "110110101",
      "111" when "110110110",
      "111" when "110110111",
      "110" when "110111000",
      "110" when "110111001",
      "110" when "110111010",
      "110" when "110111011",
      "111" when "110111100",
      "111" when "110111101",
      "111" when "110111110",
      "111" when "110111111",
      "110" when "111000000",
      "110" when "111000001",
      "111" when "111000010",
      "111" when "111000011",
      "111" when "111000100",
      "111" when "111000101",
      "111" when "111000110",
      "111" when "111000111",
      "110" when "111001000",
      "111" when "111001001",
      "111" when "111001010",
      "111" when "111001011",
      "111" when "111001100",
      "111" when "111001101",
      "111" when "111001110",
      "111" when "111001111",
      "111" when "111010000",
      "111" when "111010001",
      "111" when "111010010",
      "111" when "111010011",
      "111" when "111010100",
      "111" when "111010101",
      "111" when "111010110",
      "111" when "111010111",
      "111" when "111011000",
      "111" when "111011001",
      "111" when "111011010",
      "111" when "111011011",
      "111" when "111011100",
      "111" when "111011101",
      "111" when "111011110",
      "111" when "111011111",
      "111" when "111100000",
      "111" when "111100001",
      "111" when "111100010",
      "111" when "111100011",
      "111" when "111100100",
      "111" when "111100101",
      "111" when "111100110",
      "111" when "111100111",
      "111" when "111101000",
      "111" when "111101001",
      "111" when "111101010",
      "111" when "111101011",
      "000" when "111101100",
      "000" when "111101101",
      "000" when "111101110",
      "000" when "111101111",
      "000" when "111110000",
      "000" when "111110001",
      "000" when "111110010",
      "000" when "111110011",
      "000" when "111110100",
      "000" when "111110101",
      "000" when "111110110",
      "000" when "111110111",
      "000" when "111111000",
      "000" when "111111001",
      "000" when "111111010",
      "000" when "111111011",
      "000" when "111111100",
      "000" when "111111101",
      "000" when "111111110",
      "000" when "111111111",
      "---" when others;
Y1 <= Y0; -- for the possible blockram register
   Y <= Y1;
end architecture;

--------------------------------------------------------------------------------
--                                FPDiv_64bit
--                         (FPDiv_11_52_F500_uid384)
-- VHDL generated for Kintex7 @ 500MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Maxime Christ, Florent de Dinechin (2015)
--------------------------------------------------------------------------------
-- Pipeline depth: 36 cycles
-- Clock period (ns): 2
-- Target frequency (MHz): 500
-- Input signals: X Y
-- Output signals: R

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity FPDiv_64bit is
    port (clk, ce : in std_logic;
          X : in  std_logic_vector(11+52+2 downto 0);
          Y : in  std_logic_vector(11+52+2 downto 0);
          R : out  std_logic_vector(11+52+2 downto 0)   );
end entity;

architecture arch of FPDiv_64bit is
   component selFunction_F500_uid386 is
      port ( X : in  std_logic_vector(8 downto 0);
             Y : out  std_logic_vector(2 downto 0)   );
   end component;

signal fX :  std_logic_vector(52 downto 0);
signal fY :  std_logic_vector(52 downto 0);
signal expR0, expR0_d1, expR0_d2, expR0_d3, expR0_d4, expR0_d5, expR0_d6, expR0_d7, expR0_d8, expR0_d9, expR0_d10, expR0_d11, expR0_d12, expR0_d13, expR0_d14, expR0_d15, expR0_d16, expR0_d17, expR0_d18, expR0_d19, expR0_d20, expR0_d21, expR0_d22, expR0_d23, expR0_d24, expR0_d25, expR0_d26, expR0_d27, expR0_d28, expR0_d29, expR0_d30, expR0_d31, expR0_d32, expR0_d33, expR0_d34, expR0_d35, expR0_d36 :  std_logic_vector(12 downto 0);
signal sR, sR_d1, sR_d2, sR_d3, sR_d4, sR_d5, sR_d6, sR_d7, sR_d8, sR_d9, sR_d10, sR_d11, sR_d12, sR_d13, sR_d14, sR_d15, sR_d16, sR_d17, sR_d18, sR_d19, sR_d20, sR_d21, sR_d22, sR_d23, sR_d24, sR_d25, sR_d26, sR_d27, sR_d28, sR_d29, sR_d30, sR_d31, sR_d32, sR_d33, sR_d34, sR_d35, sR_d36 :  std_logic;
signal exnXY :  std_logic_vector(3 downto 0);
signal exnR0, exnR0_d1, exnR0_d2, exnR0_d3, exnR0_d4, exnR0_d5, exnR0_d6, exnR0_d7, exnR0_d8, exnR0_d9, exnR0_d10, exnR0_d11, exnR0_d12, exnR0_d13, exnR0_d14, exnR0_d15, exnR0_d16, exnR0_d17, exnR0_d18, exnR0_d19, exnR0_d20, exnR0_d21, exnR0_d22, exnR0_d23, exnR0_d24, exnR0_d25, exnR0_d26, exnR0_d27, exnR0_d28, exnR0_d29, exnR0_d30, exnR0_d31, exnR0_d32, exnR0_d33, exnR0_d34, exnR0_d35, exnR0_d36 :  std_logic_vector(1 downto 0);
signal D, D_d1, D_d2, D_d3, D_d4, D_d5, D_d6, D_d7, D_d8, D_d9, D_d10, D_d11, D_d12, D_d13, D_d14, D_d15, D_d16, D_d17, D_d18, D_d19, D_d20, D_d21, D_d22, D_d23, D_d24, D_d25, D_d26, D_d27, D_d28, D_d29, D_d30, D_d31, D_d32, D_d33 :  std_logic_vector(52 downto 0);
signal psX :  std_logic_vector(53 downto 0);
signal betaw28, betaw28_d1 :  std_logic_vector(55 downto 0);
signal sel28 :  std_logic_vector(8 downto 0);
signal q28, q28_d1 :  std_logic_vector(2 downto 0);
signal q28_copy387 :  std_logic_vector(2 downto 0);
signal absq28D, absq28D_d1 :  std_logic_vector(55 downto 0);
signal w27 :  std_logic_vector(55 downto 0);
signal betaw27, betaw27_d1 :  std_logic_vector(55 downto 0);
signal sel27 :  std_logic_vector(8 downto 0);
signal q27, q27_d1 :  std_logic_vector(2 downto 0);
signal q27_copy388 :  std_logic_vector(2 downto 0);
signal absq27D, absq27D_d1 :  std_logic_vector(55 downto 0);
signal w26 :  std_logic_vector(55 downto 0);
signal betaw26, betaw26_d1 :  std_logic_vector(55 downto 0);
signal sel26 :  std_logic_vector(8 downto 0);
signal q26, q26_d1 :  std_logic_vector(2 downto 0);
signal q26_copy389 :  std_logic_vector(2 downto 0);
signal absq26D, absq26D_d1 :  std_logic_vector(55 downto 0);
signal w25 :  std_logic_vector(55 downto 0);
signal betaw25, betaw25_d1 :  std_logic_vector(55 downto 0);
signal sel25 :  std_logic_vector(8 downto 0);
signal q25 :  std_logic_vector(2 downto 0);
signal q25_copy390, q25_copy390_d1 :  std_logic_vector(2 downto 0);
signal absq25D :  std_logic_vector(55 downto 0);
signal w24 :  std_logic_vector(55 downto 0);
signal betaw24, betaw24_d1, betaw24_d2 :  std_logic_vector(55 downto 0);
signal sel24 :  std_logic_vector(8 downto 0);
signal q24, q24_d1 :  std_logic_vector(2 downto 0);
signal q24_copy391, q24_copy391_d1 :  std_logic_vector(2 downto 0);
signal absq24D, absq24D_d1 :  std_logic_vector(55 downto 0);
signal w23 :  std_logic_vector(55 downto 0);
signal betaw23, betaw23_d1 :  std_logic_vector(55 downto 0);
signal sel23 :  std_logic_vector(8 downto 0);
signal q23, q23_d1 :  std_logic_vector(2 downto 0);
signal q23_copy392 :  std_logic_vector(2 downto 0);
signal absq23D, absq23D_d1 :  std_logic_vector(55 downto 0);
signal w22 :  std_logic_vector(55 downto 0);
signal betaw22, betaw22_d1 :  std_logic_vector(55 downto 0);
signal sel22 :  std_logic_vector(8 downto 0);
signal q22, q22_d1 :  std_logic_vector(2 downto 0);
signal q22_copy393 :  std_logic_vector(2 downto 0);
signal absq22D, absq22D_d1 :  std_logic_vector(55 downto 0);
signal w21 :  std_logic_vector(55 downto 0);
signal betaw21, betaw21_d1 :  std_logic_vector(55 downto 0);
signal sel21 :  std_logic_vector(8 downto 0);
signal q21 :  std_logic_vector(2 downto 0);
signal q21_copy394, q21_copy394_d1 :  std_logic_vector(2 downto 0);
signal absq21D :  std_logic_vector(55 downto 0);
signal w20 :  std_logic_vector(55 downto 0);
signal betaw20, betaw20_d1, betaw20_d2 :  std_logic_vector(55 downto 0);
signal sel20 :  std_logic_vector(8 downto 0);
signal q20, q20_d1 :  std_logic_vector(2 downto 0);
signal q20_copy395, q20_copy395_d1 :  std_logic_vector(2 downto 0);
signal absq20D, absq20D_d1 :  std_logic_vector(55 downto 0);
signal w19 :  std_logic_vector(55 downto 0);
signal betaw19, betaw19_d1 :  std_logic_vector(55 downto 0);
signal sel19 :  std_logic_vector(8 downto 0);
signal q19, q19_d1 :  std_logic_vector(2 downto 0);
signal q19_copy396 :  std_logic_vector(2 downto 0);
signal absq19D, absq19D_d1 :  std_logic_vector(55 downto 0);
signal w18 :  std_logic_vector(55 downto 0);
signal betaw18, betaw18_d1 :  std_logic_vector(55 downto 0);
signal sel18 :  std_logic_vector(8 downto 0);
signal q18, q18_d1 :  std_logic_vector(2 downto 0);
signal q18_copy397 :  std_logic_vector(2 downto 0);
signal absq18D, absq18D_d1 :  std_logic_vector(55 downto 0);
signal w17 :  std_logic_vector(55 downto 0);
signal betaw17, betaw17_d1 :  std_logic_vector(55 downto 0);
signal sel17 :  std_logic_vector(8 downto 0);
signal q17 :  std_logic_vector(2 downto 0);
signal q17_copy398, q17_copy398_d1 :  std_logic_vector(2 downto 0);
signal absq17D :  std_logic_vector(55 downto 0);
signal w16 :  std_logic_vector(55 downto 0);
signal betaw16, betaw16_d1, betaw16_d2 :  std_logic_vector(55 downto 0);
signal sel16 :  std_logic_vector(8 downto 0);
signal q16, q16_d1 :  std_logic_vector(2 downto 0);
signal q16_copy399, q16_copy399_d1 :  std_logic_vector(2 downto 0);
signal absq16D, absq16D_d1 :  std_logic_vector(55 downto 0);
signal w15 :  std_logic_vector(55 downto 0);
signal betaw15, betaw15_d1 :  std_logic_vector(55 downto 0);
signal sel15 :  std_logic_vector(8 downto 0);
signal q15, q15_d1 :  std_logic_vector(2 downto 0);
signal q15_copy400 :  std_logic_vector(2 downto 0);
signal absq15D, absq15D_d1 :  std_logic_vector(55 downto 0);
signal w14 :  std_logic_vector(55 downto 0);
signal betaw14, betaw14_d1 :  std_logic_vector(55 downto 0);
signal sel14 :  std_logic_vector(8 downto 0);
signal q14, q14_d1 :  std_logic_vector(2 downto 0);
signal q14_copy401 :  std_logic_vector(2 downto 0);
signal absq14D, absq14D_d1 :  std_logic_vector(55 downto 0);
signal w13 :  std_logic_vector(55 downto 0);
signal betaw13, betaw13_d1 :  std_logic_vector(55 downto 0);
signal sel13 :  std_logic_vector(8 downto 0);
signal q13, q13_d1 :  std_logic_vector(2 downto 0);
signal q13_copy402 :  std_logic_vector(2 downto 0);
signal absq13D, absq13D_d1 :  std_logic_vector(55 downto 0);
signal w12 :  std_logic_vector(55 downto 0);
signal betaw12, betaw12_d1, betaw12_d2 :  std_logic_vector(55 downto 0);
signal sel12 :  std_logic_vector(8 downto 0);
signal q12, q12_d1 :  std_logic_vector(2 downto 0);
signal q12_copy403, q12_copy403_d1 :  std_logic_vector(2 downto 0);
signal absq12D, absq12D_d1 :  std_logic_vector(55 downto 0);
signal w11 :  std_logic_vector(55 downto 0);
signal betaw11, betaw11_d1 :  std_logic_vector(55 downto 0);
signal sel11 :  std_logic_vector(8 downto 0);
signal q11, q11_d1 :  std_logic_vector(2 downto 0);
signal q11_copy404 :  std_logic_vector(2 downto 0);
signal absq11D, absq11D_d1 :  std_logic_vector(55 downto 0);
signal w10 :  std_logic_vector(55 downto 0);
signal betaw10, betaw10_d1 :  std_logic_vector(55 downto 0);
signal sel10 :  std_logic_vector(8 downto 0);
signal q10, q10_d1 :  std_logic_vector(2 downto 0);
signal q10_copy405 :  std_logic_vector(2 downto 0);
signal absq10D, absq10D_d1 :  std_logic_vector(55 downto 0);
signal w9 :  std_logic_vector(55 downto 0);
signal betaw9, betaw9_d1 :  std_logic_vector(55 downto 0);
signal sel9 :  std_logic_vector(8 downto 0);
signal q9, q9_d1 :  std_logic_vector(2 downto 0);
signal q9_copy406 :  std_logic_vector(2 downto 0);
signal absq9D, absq9D_d1 :  std_logic_vector(55 downto 0);
signal w8 :  std_logic_vector(55 downto 0);
signal betaw8, betaw8_d1, betaw8_d2 :  std_logic_vector(55 downto 0);
signal sel8 :  std_logic_vector(8 downto 0);
signal q8, q8_d1 :  std_logic_vector(2 downto 0);
signal q8_copy407, q8_copy407_d1 :  std_logic_vector(2 downto 0);
signal absq8D, absq8D_d1 :  std_logic_vector(55 downto 0);
signal w7 :  std_logic_vector(55 downto 0);
signal betaw7, betaw7_d1 :  std_logic_vector(55 downto 0);
signal sel7 :  std_logic_vector(8 downto 0);
signal q7, q7_d1 :  std_logic_vector(2 downto 0);
signal q7_copy408 :  std_logic_vector(2 downto 0);
signal absq7D, absq7D_d1 :  std_logic_vector(55 downto 0);
signal w6 :  std_logic_vector(55 downto 0);
signal betaw6, betaw6_d1 :  std_logic_vector(55 downto 0);
signal sel6 :  std_logic_vector(8 downto 0);
signal q6, q6_d1 :  std_logic_vector(2 downto 0);
signal q6_copy409 :  std_logic_vector(2 downto 0);
signal absq6D, absq6D_d1 :  std_logic_vector(55 downto 0);
signal w5 :  std_logic_vector(55 downto 0);
signal betaw5, betaw5_d1 :  std_logic_vector(55 downto 0);
signal sel5 :  std_logic_vector(8 downto 0);
signal q5, q5_d1 :  std_logic_vector(2 downto 0);
signal q5_copy410 :  std_logic_vector(2 downto 0);
signal absq5D, absq5D_d1 :  std_logic_vector(55 downto 0);
signal w4 :  std_logic_vector(55 downto 0);
signal betaw4, betaw4_d1, betaw4_d2 :  std_logic_vector(55 downto 0);
signal sel4 :  std_logic_vector(8 downto 0);
signal q4, q4_d1 :  std_logic_vector(2 downto 0);
signal q4_copy411, q4_copy411_d1 :  std_logic_vector(2 downto 0);
signal absq4D, absq4D_d1 :  std_logic_vector(55 downto 0);
signal w3 :  std_logic_vector(55 downto 0);
signal betaw3, betaw3_d1 :  std_logic_vector(55 downto 0);
signal sel3 :  std_logic_vector(8 downto 0);
signal q3, q3_d1 :  std_logic_vector(2 downto 0);
signal q3_copy412 :  std_logic_vector(2 downto 0);
signal absq3D, absq3D_d1 :  std_logic_vector(55 downto 0);
signal w2 :  std_logic_vector(55 downto 0);
signal betaw2, betaw2_d1 :  std_logic_vector(55 downto 0);
signal sel2 :  std_logic_vector(8 downto 0);
signal q2, q2_d1 :  std_logic_vector(2 downto 0);
signal q2_copy413 :  std_logic_vector(2 downto 0);
signal absq2D, absq2D_d1 :  std_logic_vector(55 downto 0);
signal w1 :  std_logic_vector(55 downto 0);
signal betaw1, betaw1_d1 :  std_logic_vector(55 downto 0);
signal sel1 :  std_logic_vector(8 downto 0);
signal q1, q1_d1 :  std_logic_vector(2 downto 0);
signal q1_copy414 :  std_logic_vector(2 downto 0);
signal absq1D, absq1D_d1 :  std_logic_vector(55 downto 0);
signal w0 :  std_logic_vector(55 downto 0);
signal wfinal :  std_logic_vector(53 downto 0);
signal qM0 :  std_logic;
signal qP28, qP28_d1, qP28_d2, qP28_d3, qP28_d4, qP28_d5, qP28_d6, qP28_d7, qP28_d8, qP28_d9, qP28_d10, qP28_d11, qP28_d12, qP28_d13, qP28_d14, qP28_d15, qP28_d16, qP28_d17, qP28_d18, qP28_d19, qP28_d20, qP28_d21, qP28_d22, qP28_d23, qP28_d24, qP28_d25, qP28_d26, qP28_d27, qP28_d28, qP28_d29, qP28_d30, qP28_d31, qP28_d32, qP28_d33 :  std_logic_vector(1 downto 0);
signal qM28, qM28_d1, qM28_d2, qM28_d3, qM28_d4, qM28_d5, qM28_d6, qM28_d7, qM28_d8, qM28_d9, qM28_d10, qM28_d11, qM28_d12, qM28_d13, qM28_d14, qM28_d15, qM28_d16, qM28_d17, qM28_d18, qM28_d19, qM28_d20, qM28_d21, qM28_d22, qM28_d23, qM28_d24, qM28_d25, qM28_d26, qM28_d27, qM28_d28, qM28_d29, qM28_d30, qM28_d31, qM28_d32, qM28_d33, qM28_d34 :  std_logic_vector(1 downto 0);
signal qP27, qP27_d1, qP27_d2, qP27_d3, qP27_d4, qP27_d5, qP27_d6, qP27_d7, qP27_d8, qP27_d9, qP27_d10, qP27_d11, qP27_d12, qP27_d13, qP27_d14, qP27_d15, qP27_d16, qP27_d17, qP27_d18, qP27_d19, qP27_d20, qP27_d21, qP27_d22, qP27_d23, qP27_d24, qP27_d25, qP27_d26, qP27_d27, qP27_d28, qP27_d29, qP27_d30, qP27_d31, qP27_d32 :  std_logic_vector(1 downto 0);
signal qM27, qM27_d1, qM27_d2, qM27_d3, qM27_d4, qM27_d5, qM27_d6, qM27_d7, qM27_d8, qM27_d9, qM27_d10, qM27_d11, qM27_d12, qM27_d13, qM27_d14, qM27_d15, qM27_d16, qM27_d17, qM27_d18, qM27_d19, qM27_d20, qM27_d21, qM27_d22, qM27_d23, qM27_d24, qM27_d25, qM27_d26, qM27_d27, qM27_d28, qM27_d29, qM27_d30, qM27_d31, qM27_d32, qM27_d33 :  std_logic_vector(1 downto 0);
signal qP26, qP26_d1, qP26_d2, qP26_d3, qP26_d4, qP26_d5, qP26_d6, qP26_d7, qP26_d8, qP26_d9, qP26_d10, qP26_d11, qP26_d12, qP26_d13, qP26_d14, qP26_d15, qP26_d16, qP26_d17, qP26_d18, qP26_d19, qP26_d20, qP26_d21, qP26_d22, qP26_d23, qP26_d24, qP26_d25, qP26_d26, qP26_d27, qP26_d28, qP26_d29, qP26_d30, qP26_d31 :  std_logic_vector(1 downto 0);
signal qM26, qM26_d1, qM26_d2, qM26_d3, qM26_d4, qM26_d5, qM26_d6, qM26_d7, qM26_d8, qM26_d9, qM26_d10, qM26_d11, qM26_d12, qM26_d13, qM26_d14, qM26_d15, qM26_d16, qM26_d17, qM26_d18, qM26_d19, qM26_d20, qM26_d21, qM26_d22, qM26_d23, qM26_d24, qM26_d25, qM26_d26, qM26_d27, qM26_d28, qM26_d29, qM26_d30, qM26_d31, qM26_d32 :  std_logic_vector(1 downto 0);
signal qP25, qP25_d1, qP25_d2, qP25_d3, qP25_d4, qP25_d5, qP25_d6, qP25_d7, qP25_d8, qP25_d9, qP25_d10, qP25_d11, qP25_d12, qP25_d13, qP25_d14, qP25_d15, qP25_d16, qP25_d17, qP25_d18, qP25_d19, qP25_d20, qP25_d21, qP25_d22, qP25_d23, qP25_d24, qP25_d25, qP25_d26, qP25_d27, qP25_d28, qP25_d29 :  std_logic_vector(1 downto 0);
signal qM25, qM25_d1, qM25_d2, qM25_d3, qM25_d4, qM25_d5, qM25_d6, qM25_d7, qM25_d8, qM25_d9, qM25_d10, qM25_d11, qM25_d12, qM25_d13, qM25_d14, qM25_d15, qM25_d16, qM25_d17, qM25_d18, qM25_d19, qM25_d20, qM25_d21, qM25_d22, qM25_d23, qM25_d24, qM25_d25, qM25_d26, qM25_d27, qM25_d28, qM25_d29, qM25_d30 :  std_logic_vector(1 downto 0);
signal qP24, qP24_d1, qP24_d2, qP24_d3, qP24_d4, qP24_d5, qP24_d6, qP24_d7, qP24_d8, qP24_d9, qP24_d10, qP24_d11, qP24_d12, qP24_d13, qP24_d14, qP24_d15, qP24_d16, qP24_d17, qP24_d18, qP24_d19, qP24_d20, qP24_d21, qP24_d22, qP24_d23, qP24_d24, qP24_d25, qP24_d26, qP24_d27, qP24_d28 :  std_logic_vector(1 downto 0);
signal qM24, qM24_d1, qM24_d2, qM24_d3, qM24_d4, qM24_d5, qM24_d6, qM24_d7, qM24_d8, qM24_d9, qM24_d10, qM24_d11, qM24_d12, qM24_d13, qM24_d14, qM24_d15, qM24_d16, qM24_d17, qM24_d18, qM24_d19, qM24_d20, qM24_d21, qM24_d22, qM24_d23, qM24_d24, qM24_d25, qM24_d26, qM24_d27, qM24_d28, qM24_d29 :  std_logic_vector(1 downto 0);
signal qP23, qP23_d1, qP23_d2, qP23_d3, qP23_d4, qP23_d5, qP23_d6, qP23_d7, qP23_d8, qP23_d9, qP23_d10, qP23_d11, qP23_d12, qP23_d13, qP23_d14, qP23_d15, qP23_d16, qP23_d17, qP23_d18, qP23_d19, qP23_d20, qP23_d21, qP23_d22, qP23_d23, qP23_d24, qP23_d25, qP23_d26, qP23_d27 :  std_logic_vector(1 downto 0);
signal qM23, qM23_d1, qM23_d2, qM23_d3, qM23_d4, qM23_d5, qM23_d6, qM23_d7, qM23_d8, qM23_d9, qM23_d10, qM23_d11, qM23_d12, qM23_d13, qM23_d14, qM23_d15, qM23_d16, qM23_d17, qM23_d18, qM23_d19, qM23_d20, qM23_d21, qM23_d22, qM23_d23, qM23_d24, qM23_d25, qM23_d26, qM23_d27, qM23_d28 :  std_logic_vector(1 downto 0);
signal qP22, qP22_d1, qP22_d2, qP22_d3, qP22_d4, qP22_d5, qP22_d6, qP22_d7, qP22_d8, qP22_d9, qP22_d10, qP22_d11, qP22_d12, qP22_d13, qP22_d14, qP22_d15, qP22_d16, qP22_d17, qP22_d18, qP22_d19, qP22_d20, qP22_d21, qP22_d22, qP22_d23, qP22_d24, qP22_d25, qP22_d26 :  std_logic_vector(1 downto 0);
signal qM22, qM22_d1, qM22_d2, qM22_d3, qM22_d4, qM22_d5, qM22_d6, qM22_d7, qM22_d8, qM22_d9, qM22_d10, qM22_d11, qM22_d12, qM22_d13, qM22_d14, qM22_d15, qM22_d16, qM22_d17, qM22_d18, qM22_d19, qM22_d20, qM22_d21, qM22_d22, qM22_d23, qM22_d24, qM22_d25, qM22_d26, qM22_d27 :  std_logic_vector(1 downto 0);
signal qP21, qP21_d1, qP21_d2, qP21_d3, qP21_d4, qP21_d5, qP21_d6, qP21_d7, qP21_d8, qP21_d9, qP21_d10, qP21_d11, qP21_d12, qP21_d13, qP21_d14, qP21_d15, qP21_d16, qP21_d17, qP21_d18, qP21_d19, qP21_d20, qP21_d21, qP21_d22, qP21_d23, qP21_d24 :  std_logic_vector(1 downto 0);
signal qM21, qM21_d1, qM21_d2, qM21_d3, qM21_d4, qM21_d5, qM21_d6, qM21_d7, qM21_d8, qM21_d9, qM21_d10, qM21_d11, qM21_d12, qM21_d13, qM21_d14, qM21_d15, qM21_d16, qM21_d17, qM21_d18, qM21_d19, qM21_d20, qM21_d21, qM21_d22, qM21_d23, qM21_d24, qM21_d25 :  std_logic_vector(1 downto 0);
signal qP20, qP20_d1, qP20_d2, qP20_d3, qP20_d4, qP20_d5, qP20_d6, qP20_d7, qP20_d8, qP20_d9, qP20_d10, qP20_d11, qP20_d12, qP20_d13, qP20_d14, qP20_d15, qP20_d16, qP20_d17, qP20_d18, qP20_d19, qP20_d20, qP20_d21, qP20_d22, qP20_d23 :  std_logic_vector(1 downto 0);
signal qM20, qM20_d1, qM20_d2, qM20_d3, qM20_d4, qM20_d5, qM20_d6, qM20_d7, qM20_d8, qM20_d9, qM20_d10, qM20_d11, qM20_d12, qM20_d13, qM20_d14, qM20_d15, qM20_d16, qM20_d17, qM20_d18, qM20_d19, qM20_d20, qM20_d21, qM20_d22, qM20_d23, qM20_d24 :  std_logic_vector(1 downto 0);
signal qP19, qP19_d1, qP19_d2, qP19_d3, qP19_d4, qP19_d5, qP19_d6, qP19_d7, qP19_d8, qP19_d9, qP19_d10, qP19_d11, qP19_d12, qP19_d13, qP19_d14, qP19_d15, qP19_d16, qP19_d17, qP19_d18, qP19_d19, qP19_d20, qP19_d21, qP19_d22 :  std_logic_vector(1 downto 0);
signal qM19, qM19_d1, qM19_d2, qM19_d3, qM19_d4, qM19_d5, qM19_d6, qM19_d7, qM19_d8, qM19_d9, qM19_d10, qM19_d11, qM19_d12, qM19_d13, qM19_d14, qM19_d15, qM19_d16, qM19_d17, qM19_d18, qM19_d19, qM19_d20, qM19_d21, qM19_d22, qM19_d23 :  std_logic_vector(1 downto 0);
signal qP18, qP18_d1, qP18_d2, qP18_d3, qP18_d4, qP18_d5, qP18_d6, qP18_d7, qP18_d8, qP18_d9, qP18_d10, qP18_d11, qP18_d12, qP18_d13, qP18_d14, qP18_d15, qP18_d16, qP18_d17, qP18_d18, qP18_d19, qP18_d20, qP18_d21 :  std_logic_vector(1 downto 0);
signal qM18, qM18_d1, qM18_d2, qM18_d3, qM18_d4, qM18_d5, qM18_d6, qM18_d7, qM18_d8, qM18_d9, qM18_d10, qM18_d11, qM18_d12, qM18_d13, qM18_d14, qM18_d15, qM18_d16, qM18_d17, qM18_d18, qM18_d19, qM18_d20, qM18_d21, qM18_d22 :  std_logic_vector(1 downto 0);
signal qP17, qP17_d1, qP17_d2, qP17_d3, qP17_d4, qP17_d5, qP17_d6, qP17_d7, qP17_d8, qP17_d9, qP17_d10, qP17_d11, qP17_d12, qP17_d13, qP17_d14, qP17_d15, qP17_d16, qP17_d17, qP17_d18, qP17_d19 :  std_logic_vector(1 downto 0);
signal qM17, qM17_d1, qM17_d2, qM17_d3, qM17_d4, qM17_d5, qM17_d6, qM17_d7, qM17_d8, qM17_d9, qM17_d10, qM17_d11, qM17_d12, qM17_d13, qM17_d14, qM17_d15, qM17_d16, qM17_d17, qM17_d18, qM17_d19, qM17_d20 :  std_logic_vector(1 downto 0);
signal qP16, qP16_d1, qP16_d2, qP16_d3, qP16_d4, qP16_d5, qP16_d6, qP16_d7, qP16_d8, qP16_d9, qP16_d10, qP16_d11, qP16_d12, qP16_d13, qP16_d14, qP16_d15, qP16_d16, qP16_d17, qP16_d18 :  std_logic_vector(1 downto 0);
signal qM16, qM16_d1, qM16_d2, qM16_d3, qM16_d4, qM16_d5, qM16_d6, qM16_d7, qM16_d8, qM16_d9, qM16_d10, qM16_d11, qM16_d12, qM16_d13, qM16_d14, qM16_d15, qM16_d16, qM16_d17, qM16_d18, qM16_d19 :  std_logic_vector(1 downto 0);
signal qP15, qP15_d1, qP15_d2, qP15_d3, qP15_d4, qP15_d5, qP15_d6, qP15_d7, qP15_d8, qP15_d9, qP15_d10, qP15_d11, qP15_d12, qP15_d13, qP15_d14, qP15_d15, qP15_d16, qP15_d17 :  std_logic_vector(1 downto 0);
signal qM15, qM15_d1, qM15_d2, qM15_d3, qM15_d4, qM15_d5, qM15_d6, qM15_d7, qM15_d8, qM15_d9, qM15_d10, qM15_d11, qM15_d12, qM15_d13, qM15_d14, qM15_d15, qM15_d16, qM15_d17, qM15_d18 :  std_logic_vector(1 downto 0);
signal qP14, qP14_d1, qP14_d2, qP14_d3, qP14_d4, qP14_d5, qP14_d6, qP14_d7, qP14_d8, qP14_d9, qP14_d10, qP14_d11, qP14_d12, qP14_d13, qP14_d14, qP14_d15, qP14_d16 :  std_logic_vector(1 downto 0);
signal qM14, qM14_d1, qM14_d2, qM14_d3, qM14_d4, qM14_d5, qM14_d6, qM14_d7, qM14_d8, qM14_d9, qM14_d10, qM14_d11, qM14_d12, qM14_d13, qM14_d14, qM14_d15, qM14_d16, qM14_d17 :  std_logic_vector(1 downto 0);
signal qP13, qP13_d1, qP13_d2, qP13_d3, qP13_d4, qP13_d5, qP13_d6, qP13_d7, qP13_d8, qP13_d9, qP13_d10, qP13_d11, qP13_d12, qP13_d13, qP13_d14, qP13_d15 :  std_logic_vector(1 downto 0);
signal qM13, qM13_d1, qM13_d2, qM13_d3, qM13_d4, qM13_d5, qM13_d6, qM13_d7, qM13_d8, qM13_d9, qM13_d10, qM13_d11, qM13_d12, qM13_d13, qM13_d14, qM13_d15, qM13_d16 :  std_logic_vector(1 downto 0);
signal qP12, qP12_d1, qP12_d2, qP12_d3, qP12_d4, qP12_d5, qP12_d6, qP12_d7, qP12_d8, qP12_d9, qP12_d10, qP12_d11, qP12_d12, qP12_d13 :  std_logic_vector(1 downto 0);
signal qM12, qM12_d1, qM12_d2, qM12_d3, qM12_d4, qM12_d5, qM12_d6, qM12_d7, qM12_d8, qM12_d9, qM12_d10, qM12_d11, qM12_d12, qM12_d13, qM12_d14 :  std_logic_vector(1 downto 0);
signal qP11, qP11_d1, qP11_d2, qP11_d3, qP11_d4, qP11_d5, qP11_d6, qP11_d7, qP11_d8, qP11_d9, qP11_d10, qP11_d11, qP11_d12 :  std_logic_vector(1 downto 0);
signal qM11, qM11_d1, qM11_d2, qM11_d3, qM11_d4, qM11_d5, qM11_d6, qM11_d7, qM11_d8, qM11_d9, qM11_d10, qM11_d11, qM11_d12, qM11_d13 :  std_logic_vector(1 downto 0);
signal qP10, qP10_d1, qP10_d2, qP10_d3, qP10_d4, qP10_d5, qP10_d6, qP10_d7, qP10_d8, qP10_d9, qP10_d10, qP10_d11 :  std_logic_vector(1 downto 0);
signal qM10, qM10_d1, qM10_d2, qM10_d3, qM10_d4, qM10_d5, qM10_d6, qM10_d7, qM10_d8, qM10_d9, qM10_d10, qM10_d11, qM10_d12 :  std_logic_vector(1 downto 0);
signal qP9, qP9_d1, qP9_d2, qP9_d3, qP9_d4, qP9_d5, qP9_d6, qP9_d7, qP9_d8, qP9_d9, qP9_d10 :  std_logic_vector(1 downto 0);
signal qM9, qM9_d1, qM9_d2, qM9_d3, qM9_d4, qM9_d5, qM9_d6, qM9_d7, qM9_d8, qM9_d9, qM9_d10, qM9_d11 :  std_logic_vector(1 downto 0);
signal qP8, qP8_d1, qP8_d2, qP8_d3, qP8_d4, qP8_d5, qP8_d6, qP8_d7, qP8_d8 :  std_logic_vector(1 downto 0);
signal qM8, qM8_d1, qM8_d2, qM8_d3, qM8_d4, qM8_d5, qM8_d6, qM8_d7, qM8_d8, qM8_d9 :  std_logic_vector(1 downto 0);
signal qP7, qP7_d1, qP7_d2, qP7_d3, qP7_d4, qP7_d5, qP7_d6, qP7_d7 :  std_logic_vector(1 downto 0);
signal qM7, qM7_d1, qM7_d2, qM7_d3, qM7_d4, qM7_d5, qM7_d6, qM7_d7, qM7_d8 :  std_logic_vector(1 downto 0);
signal qP6, qP6_d1, qP6_d2, qP6_d3, qP6_d4, qP6_d5, qP6_d6 :  std_logic_vector(1 downto 0);
signal qM6, qM6_d1, qM6_d2, qM6_d3, qM6_d4, qM6_d5, qM6_d6, qM6_d7 :  std_logic_vector(1 downto 0);
signal qP5, qP5_d1, qP5_d2, qP5_d3, qP5_d4, qP5_d5 :  std_logic_vector(1 downto 0);
signal qM5, qM5_d1, qM5_d2, qM5_d3, qM5_d4, qM5_d5, qM5_d6 :  std_logic_vector(1 downto 0);
signal qP4, qP4_d1, qP4_d2, qP4_d3 :  std_logic_vector(1 downto 0);
signal qM4, qM4_d1, qM4_d2, qM4_d3, qM4_d4 :  std_logic_vector(1 downto 0);
signal qP3, qP3_d1, qP3_d2 :  std_logic_vector(1 downto 0);
signal qM3, qM3_d1, qM3_d2, qM3_d3 :  std_logic_vector(1 downto 0);
signal qP2, qP2_d1 :  std_logic_vector(1 downto 0);
signal qM2, qM2_d1, qM2_d2 :  std_logic_vector(1 downto 0);
signal qP1 :  std_logic_vector(1 downto 0);
signal qM1, qM1_d1 :  std_logic_vector(1 downto 0);
signal qP, qP_d1, qP_d2 :  std_logic_vector(55 downto 0);
signal qM, qM_d1 :  std_logic_vector(55 downto 0);
signal quotient :  std_logic_vector(55 downto 0);
signal mR, mR_d1 :  std_logic_vector(54 downto 0);
signal fRnorm, fRnorm_d1 :  std_logic_vector(52 downto 0);
signal round, round_d1 :  std_logic;
signal expR1 :  std_logic_vector(12 downto 0);
signal expfrac :  std_logic_vector(64 downto 0);
signal expfracR :  std_logic_vector(64 downto 0);
signal exnR :  std_logic_vector(1 downto 0);
signal exnRfinal :  std_logic_vector(1 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            if ce = '1' then
               expR0_d1 <=  expR0;
               expR0_d2 <=  expR0_d1;
               expR0_d3 <=  expR0_d2;
               expR0_d4 <=  expR0_d3;
               expR0_d5 <=  expR0_d4;
               expR0_d6 <=  expR0_d5;
               expR0_d7 <=  expR0_d6;
               expR0_d8 <=  expR0_d7;
               expR0_d9 <=  expR0_d8;
               expR0_d10 <=  expR0_d9;
               expR0_d11 <=  expR0_d10;
               expR0_d12 <=  expR0_d11;
               expR0_d13 <=  expR0_d12;
               expR0_d14 <=  expR0_d13;
               expR0_d15 <=  expR0_d14;
               expR0_d16 <=  expR0_d15;
               expR0_d17 <=  expR0_d16;
               expR0_d18 <=  expR0_d17;
               expR0_d19 <=  expR0_d18;
               expR0_d20 <=  expR0_d19;
               expR0_d21 <=  expR0_d20;
               expR0_d22 <=  expR0_d21;
               expR0_d23 <=  expR0_d22;
               expR0_d24 <=  expR0_d23;
               expR0_d25 <=  expR0_d24;
               expR0_d26 <=  expR0_d25;
               expR0_d27 <=  expR0_d26;
               expR0_d28 <=  expR0_d27;
               expR0_d29 <=  expR0_d28;
               expR0_d30 <=  expR0_d29;
               expR0_d31 <=  expR0_d30;
               expR0_d32 <=  expR0_d31;
               expR0_d33 <=  expR0_d32;
               expR0_d34 <=  expR0_d33;
               expR0_d35 <=  expR0_d34;
               expR0_d36 <=  expR0_d35;
               sR_d1 <=  sR;
               sR_d2 <=  sR_d1;
               sR_d3 <=  sR_d2;
               sR_d4 <=  sR_d3;
               sR_d5 <=  sR_d4;
               sR_d6 <=  sR_d5;
               sR_d7 <=  sR_d6;
               sR_d8 <=  sR_d7;
               sR_d9 <=  sR_d8;
               sR_d10 <=  sR_d9;
               sR_d11 <=  sR_d10;
               sR_d12 <=  sR_d11;
               sR_d13 <=  sR_d12;
               sR_d14 <=  sR_d13;
               sR_d15 <=  sR_d14;
               sR_d16 <=  sR_d15;
               sR_d17 <=  sR_d16;
               sR_d18 <=  sR_d17;
               sR_d19 <=  sR_d18;
               sR_d20 <=  sR_d19;
               sR_d21 <=  sR_d20;
               sR_d22 <=  sR_d21;
               sR_d23 <=  sR_d22;
               sR_d24 <=  sR_d23;
               sR_d25 <=  sR_d24;
               sR_d26 <=  sR_d25;
               sR_d27 <=  sR_d26;
               sR_d28 <=  sR_d27;
               sR_d29 <=  sR_d28;
               sR_d30 <=  sR_d29;
               sR_d31 <=  sR_d30;
               sR_d32 <=  sR_d31;
               sR_d33 <=  sR_d32;
               sR_d34 <=  sR_d33;
               sR_d35 <=  sR_d34;
               sR_d36 <=  sR_d35;
               exnR0_d1 <=  exnR0;
               exnR0_d2 <=  exnR0_d1;
               exnR0_d3 <=  exnR0_d2;
               exnR0_d4 <=  exnR0_d3;
               exnR0_d5 <=  exnR0_d4;
               exnR0_d6 <=  exnR0_d5;
               exnR0_d7 <=  exnR0_d6;
               exnR0_d8 <=  exnR0_d7;
               exnR0_d9 <=  exnR0_d8;
               exnR0_d10 <=  exnR0_d9;
               exnR0_d11 <=  exnR0_d10;
               exnR0_d12 <=  exnR0_d11;
               exnR0_d13 <=  exnR0_d12;
               exnR0_d14 <=  exnR0_d13;
               exnR0_d15 <=  exnR0_d14;
               exnR0_d16 <=  exnR0_d15;
               exnR0_d17 <=  exnR0_d16;
               exnR0_d18 <=  exnR0_d17;
               exnR0_d19 <=  exnR0_d18;
               exnR0_d20 <=  exnR0_d19;
               exnR0_d21 <=  exnR0_d20;
               exnR0_d22 <=  exnR0_d21;
               exnR0_d23 <=  exnR0_d22;
               exnR0_d24 <=  exnR0_d23;
               exnR0_d25 <=  exnR0_d24;
               exnR0_d26 <=  exnR0_d25;
               exnR0_d27 <=  exnR0_d26;
               exnR0_d28 <=  exnR0_d27;
               exnR0_d29 <=  exnR0_d28;
               exnR0_d30 <=  exnR0_d29;
               exnR0_d31 <=  exnR0_d30;
               exnR0_d32 <=  exnR0_d31;
               exnR0_d33 <=  exnR0_d32;
               exnR0_d34 <=  exnR0_d33;
               exnR0_d35 <=  exnR0_d34;
               exnR0_d36 <=  exnR0_d35;
               D_d1 <=  D;
               D_d2 <=  D_d1;
               D_d3 <=  D_d2;
               D_d4 <=  D_d3;
               D_d5 <=  D_d4;
               D_d6 <=  D_d5;
               D_d7 <=  D_d6;
               D_d8 <=  D_d7;
               D_d9 <=  D_d8;
               D_d10 <=  D_d9;
               D_d11 <=  D_d10;
               D_d12 <=  D_d11;
               D_d13 <=  D_d12;
               D_d14 <=  D_d13;
               D_d15 <=  D_d14;
               D_d16 <=  D_d15;
               D_d17 <=  D_d16;
               D_d18 <=  D_d17;
               D_d19 <=  D_d18;
               D_d20 <=  D_d19;
               D_d21 <=  D_d20;
               D_d22 <=  D_d21;
               D_d23 <=  D_d22;
               D_d24 <=  D_d23;
               D_d25 <=  D_d24;
               D_d26 <=  D_d25;
               D_d27 <=  D_d26;
               D_d28 <=  D_d27;
               D_d29 <=  D_d28;
               D_d30 <=  D_d29;
               D_d31 <=  D_d30;
               D_d32 <=  D_d31;
               D_d33 <=  D_d32;
               betaw28_d1 <=  betaw28;
               q28_d1 <=  q28;
               absq28D_d1 <=  absq28D;
               betaw27_d1 <=  betaw27;
               q27_d1 <=  q27;
               absq27D_d1 <=  absq27D;
               betaw26_d1 <=  betaw26;
               q26_d1 <=  q26;
               absq26D_d1 <=  absq26D;
               betaw25_d1 <=  betaw25;
               q25_copy390_d1 <=  q25_copy390;
               betaw24_d1 <=  betaw24;
               betaw24_d2 <=  betaw24_d1;
               q24_d1 <=  q24;
               q24_copy391_d1 <=  q24_copy391;
               absq24D_d1 <=  absq24D;
               betaw23_d1 <=  betaw23;
               q23_d1 <=  q23;
               absq23D_d1 <=  absq23D;
               betaw22_d1 <=  betaw22;
               q22_d1 <=  q22;
               absq22D_d1 <=  absq22D;
               betaw21_d1 <=  betaw21;
               q21_copy394_d1 <=  q21_copy394;
               betaw20_d1 <=  betaw20;
               betaw20_d2 <=  betaw20_d1;
               q20_d1 <=  q20;
               q20_copy395_d1 <=  q20_copy395;
               absq20D_d1 <=  absq20D;
               betaw19_d1 <=  betaw19;
               q19_d1 <=  q19;
               absq19D_d1 <=  absq19D;
               betaw18_d1 <=  betaw18;
               q18_d1 <=  q18;
               absq18D_d1 <=  absq18D;
               betaw17_d1 <=  betaw17;
               q17_copy398_d1 <=  q17_copy398;
               betaw16_d1 <=  betaw16;
               betaw16_d2 <=  betaw16_d1;
               q16_d1 <=  q16;
               q16_copy399_d1 <=  q16_copy399;
               absq16D_d1 <=  absq16D;
               betaw15_d1 <=  betaw15;
               q15_d1 <=  q15;
               absq15D_d1 <=  absq15D;
               betaw14_d1 <=  betaw14;
               q14_d1 <=  q14;
               absq14D_d1 <=  absq14D;
               betaw13_d1 <=  betaw13;
               q13_d1 <=  q13;
               absq13D_d1 <=  absq13D;
               betaw12_d1 <=  betaw12;
               betaw12_d2 <=  betaw12_d1;
               q12_d1 <=  q12;
               q12_copy403_d1 <=  q12_copy403;
               absq12D_d1 <=  absq12D;
               betaw11_d1 <=  betaw11;
               q11_d1 <=  q11;
               absq11D_d1 <=  absq11D;
               betaw10_d1 <=  betaw10;
               q10_d1 <=  q10;
               absq10D_d1 <=  absq10D;
               betaw9_d1 <=  betaw9;
               q9_d1 <=  q9;
               absq9D_d1 <=  absq9D;
               betaw8_d1 <=  betaw8;
               betaw8_d2 <=  betaw8_d1;
               q8_d1 <=  q8;
               q8_copy407_d1 <=  q8_copy407;
               absq8D_d1 <=  absq8D;
               betaw7_d1 <=  betaw7;
               q7_d1 <=  q7;
               absq7D_d1 <=  absq7D;
               betaw6_d1 <=  betaw6;
               q6_d1 <=  q6;
               absq6D_d1 <=  absq6D;
               betaw5_d1 <=  betaw5;
               q5_d1 <=  q5;
               absq5D_d1 <=  absq5D;
               betaw4_d1 <=  betaw4;
               betaw4_d2 <=  betaw4_d1;
               q4_d1 <=  q4;
               q4_copy411_d1 <=  q4_copy411;
               absq4D_d1 <=  absq4D;
               betaw3_d1 <=  betaw3;
               q3_d1 <=  q3;
               absq3D_d1 <=  absq3D;
               betaw2_d1 <=  betaw2;
               q2_d1 <=  q2;
               absq2D_d1 <=  absq2D;
               betaw1_d1 <=  betaw1;
               q1_d1 <=  q1;
               absq1D_d1 <=  absq1D;
               qP28_d1 <=  qP28;
               qP28_d2 <=  qP28_d1;
               qP28_d3 <=  qP28_d2;
               qP28_d4 <=  qP28_d3;
               qP28_d5 <=  qP28_d4;
               qP28_d6 <=  qP28_d5;
               qP28_d7 <=  qP28_d6;
               qP28_d8 <=  qP28_d7;
               qP28_d9 <=  qP28_d8;
               qP28_d10 <=  qP28_d9;
               qP28_d11 <=  qP28_d10;
               qP28_d12 <=  qP28_d11;
               qP28_d13 <=  qP28_d12;
               qP28_d14 <=  qP28_d13;
               qP28_d15 <=  qP28_d14;
               qP28_d16 <=  qP28_d15;
               qP28_d17 <=  qP28_d16;
               qP28_d18 <=  qP28_d17;
               qP28_d19 <=  qP28_d18;
               qP28_d20 <=  qP28_d19;
               qP28_d21 <=  qP28_d20;
               qP28_d22 <=  qP28_d21;
               qP28_d23 <=  qP28_d22;
               qP28_d24 <=  qP28_d23;
               qP28_d25 <=  qP28_d24;
               qP28_d26 <=  qP28_d25;
               qP28_d27 <=  qP28_d26;
               qP28_d28 <=  qP28_d27;
               qP28_d29 <=  qP28_d28;
               qP28_d30 <=  qP28_d29;
               qP28_d31 <=  qP28_d30;
               qP28_d32 <=  qP28_d31;
               qP28_d33 <=  qP28_d32;
               qM28_d1 <=  qM28;
               qM28_d2 <=  qM28_d1;
               qM28_d3 <=  qM28_d2;
               qM28_d4 <=  qM28_d3;
               qM28_d5 <=  qM28_d4;
               qM28_d6 <=  qM28_d5;
               qM28_d7 <=  qM28_d6;
               qM28_d8 <=  qM28_d7;
               qM28_d9 <=  qM28_d8;
               qM28_d10 <=  qM28_d9;
               qM28_d11 <=  qM28_d10;
               qM28_d12 <=  qM28_d11;
               qM28_d13 <=  qM28_d12;
               qM28_d14 <=  qM28_d13;
               qM28_d15 <=  qM28_d14;
               qM28_d16 <=  qM28_d15;
               qM28_d17 <=  qM28_d16;
               qM28_d18 <=  qM28_d17;
               qM28_d19 <=  qM28_d18;
               qM28_d20 <=  qM28_d19;
               qM28_d21 <=  qM28_d20;
               qM28_d22 <=  qM28_d21;
               qM28_d23 <=  qM28_d22;
               qM28_d24 <=  qM28_d23;
               qM28_d25 <=  qM28_d24;
               qM28_d26 <=  qM28_d25;
               qM28_d27 <=  qM28_d26;
               qM28_d28 <=  qM28_d27;
               qM28_d29 <=  qM28_d28;
               qM28_d30 <=  qM28_d29;
               qM28_d31 <=  qM28_d30;
               qM28_d32 <=  qM28_d31;
               qM28_d33 <=  qM28_d32;
               qM28_d34 <=  qM28_d33;
               qP27_d1 <=  qP27;
               qP27_d2 <=  qP27_d1;
               qP27_d3 <=  qP27_d2;
               qP27_d4 <=  qP27_d3;
               qP27_d5 <=  qP27_d4;
               qP27_d6 <=  qP27_d5;
               qP27_d7 <=  qP27_d6;
               qP27_d8 <=  qP27_d7;
               qP27_d9 <=  qP27_d8;
               qP27_d10 <=  qP27_d9;
               qP27_d11 <=  qP27_d10;
               qP27_d12 <=  qP27_d11;
               qP27_d13 <=  qP27_d12;
               qP27_d14 <=  qP27_d13;
               qP27_d15 <=  qP27_d14;
               qP27_d16 <=  qP27_d15;
               qP27_d17 <=  qP27_d16;
               qP27_d18 <=  qP27_d17;
               qP27_d19 <=  qP27_d18;
               qP27_d20 <=  qP27_d19;
               qP27_d21 <=  qP27_d20;
               qP27_d22 <=  qP27_d21;
               qP27_d23 <=  qP27_d22;
               qP27_d24 <=  qP27_d23;
               qP27_d25 <=  qP27_d24;
               qP27_d26 <=  qP27_d25;
               qP27_d27 <=  qP27_d26;
               qP27_d28 <=  qP27_d27;
               qP27_d29 <=  qP27_d28;
               qP27_d30 <=  qP27_d29;
               qP27_d31 <=  qP27_d30;
               qP27_d32 <=  qP27_d31;
               qM27_d1 <=  qM27;
               qM27_d2 <=  qM27_d1;
               qM27_d3 <=  qM27_d2;
               qM27_d4 <=  qM27_d3;
               qM27_d5 <=  qM27_d4;
               qM27_d6 <=  qM27_d5;
               qM27_d7 <=  qM27_d6;
               qM27_d8 <=  qM27_d7;
               qM27_d9 <=  qM27_d8;
               qM27_d10 <=  qM27_d9;
               qM27_d11 <=  qM27_d10;
               qM27_d12 <=  qM27_d11;
               qM27_d13 <=  qM27_d12;
               qM27_d14 <=  qM27_d13;
               qM27_d15 <=  qM27_d14;
               qM27_d16 <=  qM27_d15;
               qM27_d17 <=  qM27_d16;
               qM27_d18 <=  qM27_d17;
               qM27_d19 <=  qM27_d18;
               qM27_d20 <=  qM27_d19;
               qM27_d21 <=  qM27_d20;
               qM27_d22 <=  qM27_d21;
               qM27_d23 <=  qM27_d22;
               qM27_d24 <=  qM27_d23;
               qM27_d25 <=  qM27_d24;
               qM27_d26 <=  qM27_d25;
               qM27_d27 <=  qM27_d26;
               qM27_d28 <=  qM27_d27;
               qM27_d29 <=  qM27_d28;
               qM27_d30 <=  qM27_d29;
               qM27_d31 <=  qM27_d30;
               qM27_d32 <=  qM27_d31;
               qM27_d33 <=  qM27_d32;
               qP26_d1 <=  qP26;
               qP26_d2 <=  qP26_d1;
               qP26_d3 <=  qP26_d2;
               qP26_d4 <=  qP26_d3;
               qP26_d5 <=  qP26_d4;
               qP26_d6 <=  qP26_d5;
               qP26_d7 <=  qP26_d6;
               qP26_d8 <=  qP26_d7;
               qP26_d9 <=  qP26_d8;
               qP26_d10 <=  qP26_d9;
               qP26_d11 <=  qP26_d10;
               qP26_d12 <=  qP26_d11;
               qP26_d13 <=  qP26_d12;
               qP26_d14 <=  qP26_d13;
               qP26_d15 <=  qP26_d14;
               qP26_d16 <=  qP26_d15;
               qP26_d17 <=  qP26_d16;
               qP26_d18 <=  qP26_d17;
               qP26_d19 <=  qP26_d18;
               qP26_d20 <=  qP26_d19;
               qP26_d21 <=  qP26_d20;
               qP26_d22 <=  qP26_d21;
               qP26_d23 <=  qP26_d22;
               qP26_d24 <=  qP26_d23;
               qP26_d25 <=  qP26_d24;
               qP26_d26 <=  qP26_d25;
               qP26_d27 <=  qP26_d26;
               qP26_d28 <=  qP26_d27;
               qP26_d29 <=  qP26_d28;
               qP26_d30 <=  qP26_d29;
               qP26_d31 <=  qP26_d30;
               qM26_d1 <=  qM26;
               qM26_d2 <=  qM26_d1;
               qM26_d3 <=  qM26_d2;
               qM26_d4 <=  qM26_d3;
               qM26_d5 <=  qM26_d4;
               qM26_d6 <=  qM26_d5;
               qM26_d7 <=  qM26_d6;
               qM26_d8 <=  qM26_d7;
               qM26_d9 <=  qM26_d8;
               qM26_d10 <=  qM26_d9;
               qM26_d11 <=  qM26_d10;
               qM26_d12 <=  qM26_d11;
               qM26_d13 <=  qM26_d12;
               qM26_d14 <=  qM26_d13;
               qM26_d15 <=  qM26_d14;
               qM26_d16 <=  qM26_d15;
               qM26_d17 <=  qM26_d16;
               qM26_d18 <=  qM26_d17;
               qM26_d19 <=  qM26_d18;
               qM26_d20 <=  qM26_d19;
               qM26_d21 <=  qM26_d20;
               qM26_d22 <=  qM26_d21;
               qM26_d23 <=  qM26_d22;
               qM26_d24 <=  qM26_d23;
               qM26_d25 <=  qM26_d24;
               qM26_d26 <=  qM26_d25;
               qM26_d27 <=  qM26_d26;
               qM26_d28 <=  qM26_d27;
               qM26_d29 <=  qM26_d28;
               qM26_d30 <=  qM26_d29;
               qM26_d31 <=  qM26_d30;
               qM26_d32 <=  qM26_d31;
               qP25_d1 <=  qP25;
               qP25_d2 <=  qP25_d1;
               qP25_d3 <=  qP25_d2;
               qP25_d4 <=  qP25_d3;
               qP25_d5 <=  qP25_d4;
               qP25_d6 <=  qP25_d5;
               qP25_d7 <=  qP25_d6;
               qP25_d8 <=  qP25_d7;
               qP25_d9 <=  qP25_d8;
               qP25_d10 <=  qP25_d9;
               qP25_d11 <=  qP25_d10;
               qP25_d12 <=  qP25_d11;
               qP25_d13 <=  qP25_d12;
               qP25_d14 <=  qP25_d13;
               qP25_d15 <=  qP25_d14;
               qP25_d16 <=  qP25_d15;
               qP25_d17 <=  qP25_d16;
               qP25_d18 <=  qP25_d17;
               qP25_d19 <=  qP25_d18;
               qP25_d20 <=  qP25_d19;
               qP25_d21 <=  qP25_d20;
               qP25_d22 <=  qP25_d21;
               qP25_d23 <=  qP25_d22;
               qP25_d24 <=  qP25_d23;
               qP25_d25 <=  qP25_d24;
               qP25_d26 <=  qP25_d25;
               qP25_d27 <=  qP25_d26;
               qP25_d28 <=  qP25_d27;
               qP25_d29 <=  qP25_d28;
               qM25_d1 <=  qM25;
               qM25_d2 <=  qM25_d1;
               qM25_d3 <=  qM25_d2;
               qM25_d4 <=  qM25_d3;
               qM25_d5 <=  qM25_d4;
               qM25_d6 <=  qM25_d5;
               qM25_d7 <=  qM25_d6;
               qM25_d8 <=  qM25_d7;
               qM25_d9 <=  qM25_d8;
               qM25_d10 <=  qM25_d9;
               qM25_d11 <=  qM25_d10;
               qM25_d12 <=  qM25_d11;
               qM25_d13 <=  qM25_d12;
               qM25_d14 <=  qM25_d13;
               qM25_d15 <=  qM25_d14;
               qM25_d16 <=  qM25_d15;
               qM25_d17 <=  qM25_d16;
               qM25_d18 <=  qM25_d17;
               qM25_d19 <=  qM25_d18;
               qM25_d20 <=  qM25_d19;
               qM25_d21 <=  qM25_d20;
               qM25_d22 <=  qM25_d21;
               qM25_d23 <=  qM25_d22;
               qM25_d24 <=  qM25_d23;
               qM25_d25 <=  qM25_d24;
               qM25_d26 <=  qM25_d25;
               qM25_d27 <=  qM25_d26;
               qM25_d28 <=  qM25_d27;
               qM25_d29 <=  qM25_d28;
               qM25_d30 <=  qM25_d29;
               qP24_d1 <=  qP24;
               qP24_d2 <=  qP24_d1;
               qP24_d3 <=  qP24_d2;
               qP24_d4 <=  qP24_d3;
               qP24_d5 <=  qP24_d4;
               qP24_d6 <=  qP24_d5;
               qP24_d7 <=  qP24_d6;
               qP24_d8 <=  qP24_d7;
               qP24_d9 <=  qP24_d8;
               qP24_d10 <=  qP24_d9;
               qP24_d11 <=  qP24_d10;
               qP24_d12 <=  qP24_d11;
               qP24_d13 <=  qP24_d12;
               qP24_d14 <=  qP24_d13;
               qP24_d15 <=  qP24_d14;
               qP24_d16 <=  qP24_d15;
               qP24_d17 <=  qP24_d16;
               qP24_d18 <=  qP24_d17;
               qP24_d19 <=  qP24_d18;
               qP24_d20 <=  qP24_d19;
               qP24_d21 <=  qP24_d20;
               qP24_d22 <=  qP24_d21;
               qP24_d23 <=  qP24_d22;
               qP24_d24 <=  qP24_d23;
               qP24_d25 <=  qP24_d24;
               qP24_d26 <=  qP24_d25;
               qP24_d27 <=  qP24_d26;
               qP24_d28 <=  qP24_d27;
               qM24_d1 <=  qM24;
               qM24_d2 <=  qM24_d1;
               qM24_d3 <=  qM24_d2;
               qM24_d4 <=  qM24_d3;
               qM24_d5 <=  qM24_d4;
               qM24_d6 <=  qM24_d5;
               qM24_d7 <=  qM24_d6;
               qM24_d8 <=  qM24_d7;
               qM24_d9 <=  qM24_d8;
               qM24_d10 <=  qM24_d9;
               qM24_d11 <=  qM24_d10;
               qM24_d12 <=  qM24_d11;
               qM24_d13 <=  qM24_d12;
               qM24_d14 <=  qM24_d13;
               qM24_d15 <=  qM24_d14;
               qM24_d16 <=  qM24_d15;
               qM24_d17 <=  qM24_d16;
               qM24_d18 <=  qM24_d17;
               qM24_d19 <=  qM24_d18;
               qM24_d20 <=  qM24_d19;
               qM24_d21 <=  qM24_d20;
               qM24_d22 <=  qM24_d21;
               qM24_d23 <=  qM24_d22;
               qM24_d24 <=  qM24_d23;
               qM24_d25 <=  qM24_d24;
               qM24_d26 <=  qM24_d25;
               qM24_d27 <=  qM24_d26;
               qM24_d28 <=  qM24_d27;
               qM24_d29 <=  qM24_d28;
               qP23_d1 <=  qP23;
               qP23_d2 <=  qP23_d1;
               qP23_d3 <=  qP23_d2;
               qP23_d4 <=  qP23_d3;
               qP23_d5 <=  qP23_d4;
               qP23_d6 <=  qP23_d5;
               qP23_d7 <=  qP23_d6;
               qP23_d8 <=  qP23_d7;
               qP23_d9 <=  qP23_d8;
               qP23_d10 <=  qP23_d9;
               qP23_d11 <=  qP23_d10;
               qP23_d12 <=  qP23_d11;
               qP23_d13 <=  qP23_d12;
               qP23_d14 <=  qP23_d13;
               qP23_d15 <=  qP23_d14;
               qP23_d16 <=  qP23_d15;
               qP23_d17 <=  qP23_d16;
               qP23_d18 <=  qP23_d17;
               qP23_d19 <=  qP23_d18;
               qP23_d20 <=  qP23_d19;
               qP23_d21 <=  qP23_d20;
               qP23_d22 <=  qP23_d21;
               qP23_d23 <=  qP23_d22;
               qP23_d24 <=  qP23_d23;
               qP23_d25 <=  qP23_d24;
               qP23_d26 <=  qP23_d25;
               qP23_d27 <=  qP23_d26;
               qM23_d1 <=  qM23;
               qM23_d2 <=  qM23_d1;
               qM23_d3 <=  qM23_d2;
               qM23_d4 <=  qM23_d3;
               qM23_d5 <=  qM23_d4;
               qM23_d6 <=  qM23_d5;
               qM23_d7 <=  qM23_d6;
               qM23_d8 <=  qM23_d7;
               qM23_d9 <=  qM23_d8;
               qM23_d10 <=  qM23_d9;
               qM23_d11 <=  qM23_d10;
               qM23_d12 <=  qM23_d11;
               qM23_d13 <=  qM23_d12;
               qM23_d14 <=  qM23_d13;
               qM23_d15 <=  qM23_d14;
               qM23_d16 <=  qM23_d15;
               qM23_d17 <=  qM23_d16;
               qM23_d18 <=  qM23_d17;
               qM23_d19 <=  qM23_d18;
               qM23_d20 <=  qM23_d19;
               qM23_d21 <=  qM23_d20;
               qM23_d22 <=  qM23_d21;
               qM23_d23 <=  qM23_d22;
               qM23_d24 <=  qM23_d23;
               qM23_d25 <=  qM23_d24;
               qM23_d26 <=  qM23_d25;
               qM23_d27 <=  qM23_d26;
               qM23_d28 <=  qM23_d27;
               qP22_d1 <=  qP22;
               qP22_d2 <=  qP22_d1;
               qP22_d3 <=  qP22_d2;
               qP22_d4 <=  qP22_d3;
               qP22_d5 <=  qP22_d4;
               qP22_d6 <=  qP22_d5;
               qP22_d7 <=  qP22_d6;
               qP22_d8 <=  qP22_d7;
               qP22_d9 <=  qP22_d8;
               qP22_d10 <=  qP22_d9;
               qP22_d11 <=  qP22_d10;
               qP22_d12 <=  qP22_d11;
               qP22_d13 <=  qP22_d12;
               qP22_d14 <=  qP22_d13;
               qP22_d15 <=  qP22_d14;
               qP22_d16 <=  qP22_d15;
               qP22_d17 <=  qP22_d16;
               qP22_d18 <=  qP22_d17;
               qP22_d19 <=  qP22_d18;
               qP22_d20 <=  qP22_d19;
               qP22_d21 <=  qP22_d20;
               qP22_d22 <=  qP22_d21;
               qP22_d23 <=  qP22_d22;
               qP22_d24 <=  qP22_d23;
               qP22_d25 <=  qP22_d24;
               qP22_d26 <=  qP22_d25;
               qM22_d1 <=  qM22;
               qM22_d2 <=  qM22_d1;
               qM22_d3 <=  qM22_d2;
               qM22_d4 <=  qM22_d3;
               qM22_d5 <=  qM22_d4;
               qM22_d6 <=  qM22_d5;
               qM22_d7 <=  qM22_d6;
               qM22_d8 <=  qM22_d7;
               qM22_d9 <=  qM22_d8;
               qM22_d10 <=  qM22_d9;
               qM22_d11 <=  qM22_d10;
               qM22_d12 <=  qM22_d11;
               qM22_d13 <=  qM22_d12;
               qM22_d14 <=  qM22_d13;
               qM22_d15 <=  qM22_d14;
               qM22_d16 <=  qM22_d15;
               qM22_d17 <=  qM22_d16;
               qM22_d18 <=  qM22_d17;
               qM22_d19 <=  qM22_d18;
               qM22_d20 <=  qM22_d19;
               qM22_d21 <=  qM22_d20;
               qM22_d22 <=  qM22_d21;
               qM22_d23 <=  qM22_d22;
               qM22_d24 <=  qM22_d23;
               qM22_d25 <=  qM22_d24;
               qM22_d26 <=  qM22_d25;
               qM22_d27 <=  qM22_d26;
               qP21_d1 <=  qP21;
               qP21_d2 <=  qP21_d1;
               qP21_d3 <=  qP21_d2;
               qP21_d4 <=  qP21_d3;
               qP21_d5 <=  qP21_d4;
               qP21_d6 <=  qP21_d5;
               qP21_d7 <=  qP21_d6;
               qP21_d8 <=  qP21_d7;
               qP21_d9 <=  qP21_d8;
               qP21_d10 <=  qP21_d9;
               qP21_d11 <=  qP21_d10;
               qP21_d12 <=  qP21_d11;
               qP21_d13 <=  qP21_d12;
               qP21_d14 <=  qP21_d13;
               qP21_d15 <=  qP21_d14;
               qP21_d16 <=  qP21_d15;
               qP21_d17 <=  qP21_d16;
               qP21_d18 <=  qP21_d17;
               qP21_d19 <=  qP21_d18;
               qP21_d20 <=  qP21_d19;
               qP21_d21 <=  qP21_d20;
               qP21_d22 <=  qP21_d21;
               qP21_d23 <=  qP21_d22;
               qP21_d24 <=  qP21_d23;
               qM21_d1 <=  qM21;
               qM21_d2 <=  qM21_d1;
               qM21_d3 <=  qM21_d2;
               qM21_d4 <=  qM21_d3;
               qM21_d5 <=  qM21_d4;
               qM21_d6 <=  qM21_d5;
               qM21_d7 <=  qM21_d6;
               qM21_d8 <=  qM21_d7;
               qM21_d9 <=  qM21_d8;
               qM21_d10 <=  qM21_d9;
               qM21_d11 <=  qM21_d10;
               qM21_d12 <=  qM21_d11;
               qM21_d13 <=  qM21_d12;
               qM21_d14 <=  qM21_d13;
               qM21_d15 <=  qM21_d14;
               qM21_d16 <=  qM21_d15;
               qM21_d17 <=  qM21_d16;
               qM21_d18 <=  qM21_d17;
               qM21_d19 <=  qM21_d18;
               qM21_d20 <=  qM21_d19;
               qM21_d21 <=  qM21_d20;
               qM21_d22 <=  qM21_d21;
               qM21_d23 <=  qM21_d22;
               qM21_d24 <=  qM21_d23;
               qM21_d25 <=  qM21_d24;
               qP20_d1 <=  qP20;
               qP20_d2 <=  qP20_d1;
               qP20_d3 <=  qP20_d2;
               qP20_d4 <=  qP20_d3;
               qP20_d5 <=  qP20_d4;
               qP20_d6 <=  qP20_d5;
               qP20_d7 <=  qP20_d6;
               qP20_d8 <=  qP20_d7;
               qP20_d9 <=  qP20_d8;
               qP20_d10 <=  qP20_d9;
               qP20_d11 <=  qP20_d10;
               qP20_d12 <=  qP20_d11;
               qP20_d13 <=  qP20_d12;
               qP20_d14 <=  qP20_d13;
               qP20_d15 <=  qP20_d14;
               qP20_d16 <=  qP20_d15;
               qP20_d17 <=  qP20_d16;
               qP20_d18 <=  qP20_d17;
               qP20_d19 <=  qP20_d18;
               qP20_d20 <=  qP20_d19;
               qP20_d21 <=  qP20_d20;
               qP20_d22 <=  qP20_d21;
               qP20_d23 <=  qP20_d22;
               qM20_d1 <=  qM20;
               qM20_d2 <=  qM20_d1;
               qM20_d3 <=  qM20_d2;
               qM20_d4 <=  qM20_d3;
               qM20_d5 <=  qM20_d4;
               qM20_d6 <=  qM20_d5;
               qM20_d7 <=  qM20_d6;
               qM20_d8 <=  qM20_d7;
               qM20_d9 <=  qM20_d8;
               qM20_d10 <=  qM20_d9;
               qM20_d11 <=  qM20_d10;
               qM20_d12 <=  qM20_d11;
               qM20_d13 <=  qM20_d12;
               qM20_d14 <=  qM20_d13;
               qM20_d15 <=  qM20_d14;
               qM20_d16 <=  qM20_d15;
               qM20_d17 <=  qM20_d16;
               qM20_d18 <=  qM20_d17;
               qM20_d19 <=  qM20_d18;
               qM20_d20 <=  qM20_d19;
               qM20_d21 <=  qM20_d20;
               qM20_d22 <=  qM20_d21;
               qM20_d23 <=  qM20_d22;
               qM20_d24 <=  qM20_d23;
               qP19_d1 <=  qP19;
               qP19_d2 <=  qP19_d1;
               qP19_d3 <=  qP19_d2;
               qP19_d4 <=  qP19_d3;
               qP19_d5 <=  qP19_d4;
               qP19_d6 <=  qP19_d5;
               qP19_d7 <=  qP19_d6;
               qP19_d8 <=  qP19_d7;
               qP19_d9 <=  qP19_d8;
               qP19_d10 <=  qP19_d9;
               qP19_d11 <=  qP19_d10;
               qP19_d12 <=  qP19_d11;
               qP19_d13 <=  qP19_d12;
               qP19_d14 <=  qP19_d13;
               qP19_d15 <=  qP19_d14;
               qP19_d16 <=  qP19_d15;
               qP19_d17 <=  qP19_d16;
               qP19_d18 <=  qP19_d17;
               qP19_d19 <=  qP19_d18;
               qP19_d20 <=  qP19_d19;
               qP19_d21 <=  qP19_d20;
               qP19_d22 <=  qP19_d21;
               qM19_d1 <=  qM19;
               qM19_d2 <=  qM19_d1;
               qM19_d3 <=  qM19_d2;
               qM19_d4 <=  qM19_d3;
               qM19_d5 <=  qM19_d4;
               qM19_d6 <=  qM19_d5;
               qM19_d7 <=  qM19_d6;
               qM19_d8 <=  qM19_d7;
               qM19_d9 <=  qM19_d8;
               qM19_d10 <=  qM19_d9;
               qM19_d11 <=  qM19_d10;
               qM19_d12 <=  qM19_d11;
               qM19_d13 <=  qM19_d12;
               qM19_d14 <=  qM19_d13;
               qM19_d15 <=  qM19_d14;
               qM19_d16 <=  qM19_d15;
               qM19_d17 <=  qM19_d16;
               qM19_d18 <=  qM19_d17;
               qM19_d19 <=  qM19_d18;
               qM19_d20 <=  qM19_d19;
               qM19_d21 <=  qM19_d20;
               qM19_d22 <=  qM19_d21;
               qM19_d23 <=  qM19_d22;
               qP18_d1 <=  qP18;
               qP18_d2 <=  qP18_d1;
               qP18_d3 <=  qP18_d2;
               qP18_d4 <=  qP18_d3;
               qP18_d5 <=  qP18_d4;
               qP18_d6 <=  qP18_d5;
               qP18_d7 <=  qP18_d6;
               qP18_d8 <=  qP18_d7;
               qP18_d9 <=  qP18_d8;
               qP18_d10 <=  qP18_d9;
               qP18_d11 <=  qP18_d10;
               qP18_d12 <=  qP18_d11;
               qP18_d13 <=  qP18_d12;
               qP18_d14 <=  qP18_d13;
               qP18_d15 <=  qP18_d14;
               qP18_d16 <=  qP18_d15;
               qP18_d17 <=  qP18_d16;
               qP18_d18 <=  qP18_d17;
               qP18_d19 <=  qP18_d18;
               qP18_d20 <=  qP18_d19;
               qP18_d21 <=  qP18_d20;
               qM18_d1 <=  qM18;
               qM18_d2 <=  qM18_d1;
               qM18_d3 <=  qM18_d2;
               qM18_d4 <=  qM18_d3;
               qM18_d5 <=  qM18_d4;
               qM18_d6 <=  qM18_d5;
               qM18_d7 <=  qM18_d6;
               qM18_d8 <=  qM18_d7;
               qM18_d9 <=  qM18_d8;
               qM18_d10 <=  qM18_d9;
               qM18_d11 <=  qM18_d10;
               qM18_d12 <=  qM18_d11;
               qM18_d13 <=  qM18_d12;
               qM18_d14 <=  qM18_d13;
               qM18_d15 <=  qM18_d14;
               qM18_d16 <=  qM18_d15;
               qM18_d17 <=  qM18_d16;
               qM18_d18 <=  qM18_d17;
               qM18_d19 <=  qM18_d18;
               qM18_d20 <=  qM18_d19;
               qM18_d21 <=  qM18_d20;
               qM18_d22 <=  qM18_d21;
               qP17_d1 <=  qP17;
               qP17_d2 <=  qP17_d1;
               qP17_d3 <=  qP17_d2;
               qP17_d4 <=  qP17_d3;
               qP17_d5 <=  qP17_d4;
               qP17_d6 <=  qP17_d5;
               qP17_d7 <=  qP17_d6;
               qP17_d8 <=  qP17_d7;
               qP17_d9 <=  qP17_d8;
               qP17_d10 <=  qP17_d9;
               qP17_d11 <=  qP17_d10;
               qP17_d12 <=  qP17_d11;
               qP17_d13 <=  qP17_d12;
               qP17_d14 <=  qP17_d13;
               qP17_d15 <=  qP17_d14;
               qP17_d16 <=  qP17_d15;
               qP17_d17 <=  qP17_d16;
               qP17_d18 <=  qP17_d17;
               qP17_d19 <=  qP17_d18;
               qM17_d1 <=  qM17;
               qM17_d2 <=  qM17_d1;
               qM17_d3 <=  qM17_d2;
               qM17_d4 <=  qM17_d3;
               qM17_d5 <=  qM17_d4;
               qM17_d6 <=  qM17_d5;
               qM17_d7 <=  qM17_d6;
               qM17_d8 <=  qM17_d7;
               qM17_d9 <=  qM17_d8;
               qM17_d10 <=  qM17_d9;
               qM17_d11 <=  qM17_d10;
               qM17_d12 <=  qM17_d11;
               qM17_d13 <=  qM17_d12;
               qM17_d14 <=  qM17_d13;
               qM17_d15 <=  qM17_d14;
               qM17_d16 <=  qM17_d15;
               qM17_d17 <=  qM17_d16;
               qM17_d18 <=  qM17_d17;
               qM17_d19 <=  qM17_d18;
               qM17_d20 <=  qM17_d19;
               qP16_d1 <=  qP16;
               qP16_d2 <=  qP16_d1;
               qP16_d3 <=  qP16_d2;
               qP16_d4 <=  qP16_d3;
               qP16_d5 <=  qP16_d4;
               qP16_d6 <=  qP16_d5;
               qP16_d7 <=  qP16_d6;
               qP16_d8 <=  qP16_d7;
               qP16_d9 <=  qP16_d8;
               qP16_d10 <=  qP16_d9;
               qP16_d11 <=  qP16_d10;
               qP16_d12 <=  qP16_d11;
               qP16_d13 <=  qP16_d12;
               qP16_d14 <=  qP16_d13;
               qP16_d15 <=  qP16_d14;
               qP16_d16 <=  qP16_d15;
               qP16_d17 <=  qP16_d16;
               qP16_d18 <=  qP16_d17;
               qM16_d1 <=  qM16;
               qM16_d2 <=  qM16_d1;
               qM16_d3 <=  qM16_d2;
               qM16_d4 <=  qM16_d3;
               qM16_d5 <=  qM16_d4;
               qM16_d6 <=  qM16_d5;
               qM16_d7 <=  qM16_d6;
               qM16_d8 <=  qM16_d7;
               qM16_d9 <=  qM16_d8;
               qM16_d10 <=  qM16_d9;
               qM16_d11 <=  qM16_d10;
               qM16_d12 <=  qM16_d11;
               qM16_d13 <=  qM16_d12;
               qM16_d14 <=  qM16_d13;
               qM16_d15 <=  qM16_d14;
               qM16_d16 <=  qM16_d15;
               qM16_d17 <=  qM16_d16;
               qM16_d18 <=  qM16_d17;
               qM16_d19 <=  qM16_d18;
               qP15_d1 <=  qP15;
               qP15_d2 <=  qP15_d1;
               qP15_d3 <=  qP15_d2;
               qP15_d4 <=  qP15_d3;
               qP15_d5 <=  qP15_d4;
               qP15_d6 <=  qP15_d5;
               qP15_d7 <=  qP15_d6;
               qP15_d8 <=  qP15_d7;
               qP15_d9 <=  qP15_d8;
               qP15_d10 <=  qP15_d9;
               qP15_d11 <=  qP15_d10;
               qP15_d12 <=  qP15_d11;
               qP15_d13 <=  qP15_d12;
               qP15_d14 <=  qP15_d13;
               qP15_d15 <=  qP15_d14;
               qP15_d16 <=  qP15_d15;
               qP15_d17 <=  qP15_d16;
               qM15_d1 <=  qM15;
               qM15_d2 <=  qM15_d1;
               qM15_d3 <=  qM15_d2;
               qM15_d4 <=  qM15_d3;
               qM15_d5 <=  qM15_d4;
               qM15_d6 <=  qM15_d5;
               qM15_d7 <=  qM15_d6;
               qM15_d8 <=  qM15_d7;
               qM15_d9 <=  qM15_d8;
               qM15_d10 <=  qM15_d9;
               qM15_d11 <=  qM15_d10;
               qM15_d12 <=  qM15_d11;
               qM15_d13 <=  qM15_d12;
               qM15_d14 <=  qM15_d13;
               qM15_d15 <=  qM15_d14;
               qM15_d16 <=  qM15_d15;
               qM15_d17 <=  qM15_d16;
               qM15_d18 <=  qM15_d17;
               qP14_d1 <=  qP14;
               qP14_d2 <=  qP14_d1;
               qP14_d3 <=  qP14_d2;
               qP14_d4 <=  qP14_d3;
               qP14_d5 <=  qP14_d4;
               qP14_d6 <=  qP14_d5;
               qP14_d7 <=  qP14_d6;
               qP14_d8 <=  qP14_d7;
               qP14_d9 <=  qP14_d8;
               qP14_d10 <=  qP14_d9;
               qP14_d11 <=  qP14_d10;
               qP14_d12 <=  qP14_d11;
               qP14_d13 <=  qP14_d12;
               qP14_d14 <=  qP14_d13;
               qP14_d15 <=  qP14_d14;
               qP14_d16 <=  qP14_d15;
               qM14_d1 <=  qM14;
               qM14_d2 <=  qM14_d1;
               qM14_d3 <=  qM14_d2;
               qM14_d4 <=  qM14_d3;
               qM14_d5 <=  qM14_d4;
               qM14_d6 <=  qM14_d5;
               qM14_d7 <=  qM14_d6;
               qM14_d8 <=  qM14_d7;
               qM14_d9 <=  qM14_d8;
               qM14_d10 <=  qM14_d9;
               qM14_d11 <=  qM14_d10;
               qM14_d12 <=  qM14_d11;
               qM14_d13 <=  qM14_d12;
               qM14_d14 <=  qM14_d13;
               qM14_d15 <=  qM14_d14;
               qM14_d16 <=  qM14_d15;
               qM14_d17 <=  qM14_d16;
               qP13_d1 <=  qP13;
               qP13_d2 <=  qP13_d1;
               qP13_d3 <=  qP13_d2;
               qP13_d4 <=  qP13_d3;
               qP13_d5 <=  qP13_d4;
               qP13_d6 <=  qP13_d5;
               qP13_d7 <=  qP13_d6;
               qP13_d8 <=  qP13_d7;
               qP13_d9 <=  qP13_d8;
               qP13_d10 <=  qP13_d9;
               qP13_d11 <=  qP13_d10;
               qP13_d12 <=  qP13_d11;
               qP13_d13 <=  qP13_d12;
               qP13_d14 <=  qP13_d13;
               qP13_d15 <=  qP13_d14;
               qM13_d1 <=  qM13;
               qM13_d2 <=  qM13_d1;
               qM13_d3 <=  qM13_d2;
               qM13_d4 <=  qM13_d3;
               qM13_d5 <=  qM13_d4;
               qM13_d6 <=  qM13_d5;
               qM13_d7 <=  qM13_d6;
               qM13_d8 <=  qM13_d7;
               qM13_d9 <=  qM13_d8;
               qM13_d10 <=  qM13_d9;
               qM13_d11 <=  qM13_d10;
               qM13_d12 <=  qM13_d11;
               qM13_d13 <=  qM13_d12;
               qM13_d14 <=  qM13_d13;
               qM13_d15 <=  qM13_d14;
               qM13_d16 <=  qM13_d15;
               qP12_d1 <=  qP12;
               qP12_d2 <=  qP12_d1;
               qP12_d3 <=  qP12_d2;
               qP12_d4 <=  qP12_d3;
               qP12_d5 <=  qP12_d4;
               qP12_d6 <=  qP12_d5;
               qP12_d7 <=  qP12_d6;
               qP12_d8 <=  qP12_d7;
               qP12_d9 <=  qP12_d8;
               qP12_d10 <=  qP12_d9;
               qP12_d11 <=  qP12_d10;
               qP12_d12 <=  qP12_d11;
               qP12_d13 <=  qP12_d12;
               qM12_d1 <=  qM12;
               qM12_d2 <=  qM12_d1;
               qM12_d3 <=  qM12_d2;
               qM12_d4 <=  qM12_d3;
               qM12_d5 <=  qM12_d4;
               qM12_d6 <=  qM12_d5;
               qM12_d7 <=  qM12_d6;
               qM12_d8 <=  qM12_d7;
               qM12_d9 <=  qM12_d8;
               qM12_d10 <=  qM12_d9;
               qM12_d11 <=  qM12_d10;
               qM12_d12 <=  qM12_d11;
               qM12_d13 <=  qM12_d12;
               qM12_d14 <=  qM12_d13;
               qP11_d1 <=  qP11;
               qP11_d2 <=  qP11_d1;
               qP11_d3 <=  qP11_d2;
               qP11_d4 <=  qP11_d3;
               qP11_d5 <=  qP11_d4;
               qP11_d6 <=  qP11_d5;
               qP11_d7 <=  qP11_d6;
               qP11_d8 <=  qP11_d7;
               qP11_d9 <=  qP11_d8;
               qP11_d10 <=  qP11_d9;
               qP11_d11 <=  qP11_d10;
               qP11_d12 <=  qP11_d11;
               qM11_d1 <=  qM11;
               qM11_d2 <=  qM11_d1;
               qM11_d3 <=  qM11_d2;
               qM11_d4 <=  qM11_d3;
               qM11_d5 <=  qM11_d4;
               qM11_d6 <=  qM11_d5;
               qM11_d7 <=  qM11_d6;
               qM11_d8 <=  qM11_d7;
               qM11_d9 <=  qM11_d8;
               qM11_d10 <=  qM11_d9;
               qM11_d11 <=  qM11_d10;
               qM11_d12 <=  qM11_d11;
               qM11_d13 <=  qM11_d12;
               qP10_d1 <=  qP10;
               qP10_d2 <=  qP10_d1;
               qP10_d3 <=  qP10_d2;
               qP10_d4 <=  qP10_d3;
               qP10_d5 <=  qP10_d4;
               qP10_d6 <=  qP10_d5;
               qP10_d7 <=  qP10_d6;
               qP10_d8 <=  qP10_d7;
               qP10_d9 <=  qP10_d8;
               qP10_d10 <=  qP10_d9;
               qP10_d11 <=  qP10_d10;
               qM10_d1 <=  qM10;
               qM10_d2 <=  qM10_d1;
               qM10_d3 <=  qM10_d2;
               qM10_d4 <=  qM10_d3;
               qM10_d5 <=  qM10_d4;
               qM10_d6 <=  qM10_d5;
               qM10_d7 <=  qM10_d6;
               qM10_d8 <=  qM10_d7;
               qM10_d9 <=  qM10_d8;
               qM10_d10 <=  qM10_d9;
               qM10_d11 <=  qM10_d10;
               qM10_d12 <=  qM10_d11;
               qP9_d1 <=  qP9;
               qP9_d2 <=  qP9_d1;
               qP9_d3 <=  qP9_d2;
               qP9_d4 <=  qP9_d3;
               qP9_d5 <=  qP9_d4;
               qP9_d6 <=  qP9_d5;
               qP9_d7 <=  qP9_d6;
               qP9_d8 <=  qP9_d7;
               qP9_d9 <=  qP9_d8;
               qP9_d10 <=  qP9_d9;
               qM9_d1 <=  qM9;
               qM9_d2 <=  qM9_d1;
               qM9_d3 <=  qM9_d2;
               qM9_d4 <=  qM9_d3;
               qM9_d5 <=  qM9_d4;
               qM9_d6 <=  qM9_d5;
               qM9_d7 <=  qM9_d6;
               qM9_d8 <=  qM9_d7;
               qM9_d9 <=  qM9_d8;
               qM9_d10 <=  qM9_d9;
               qM9_d11 <=  qM9_d10;
               qP8_d1 <=  qP8;
               qP8_d2 <=  qP8_d1;
               qP8_d3 <=  qP8_d2;
               qP8_d4 <=  qP8_d3;
               qP8_d5 <=  qP8_d4;
               qP8_d6 <=  qP8_d5;
               qP8_d7 <=  qP8_d6;
               qP8_d8 <=  qP8_d7;
               qM8_d1 <=  qM8;
               qM8_d2 <=  qM8_d1;
               qM8_d3 <=  qM8_d2;
               qM8_d4 <=  qM8_d3;
               qM8_d5 <=  qM8_d4;
               qM8_d6 <=  qM8_d5;
               qM8_d7 <=  qM8_d6;
               qM8_d8 <=  qM8_d7;
               qM8_d9 <=  qM8_d8;
               qP7_d1 <=  qP7;
               qP7_d2 <=  qP7_d1;
               qP7_d3 <=  qP7_d2;
               qP7_d4 <=  qP7_d3;
               qP7_d5 <=  qP7_d4;
               qP7_d6 <=  qP7_d5;
               qP7_d7 <=  qP7_d6;
               qM7_d1 <=  qM7;
               qM7_d2 <=  qM7_d1;
               qM7_d3 <=  qM7_d2;
               qM7_d4 <=  qM7_d3;
               qM7_d5 <=  qM7_d4;
               qM7_d6 <=  qM7_d5;
               qM7_d7 <=  qM7_d6;
               qM7_d8 <=  qM7_d7;
               qP6_d1 <=  qP6;
               qP6_d2 <=  qP6_d1;
               qP6_d3 <=  qP6_d2;
               qP6_d4 <=  qP6_d3;
               qP6_d5 <=  qP6_d4;
               qP6_d6 <=  qP6_d5;
               qM6_d1 <=  qM6;
               qM6_d2 <=  qM6_d1;
               qM6_d3 <=  qM6_d2;
               qM6_d4 <=  qM6_d3;
               qM6_d5 <=  qM6_d4;
               qM6_d6 <=  qM6_d5;
               qM6_d7 <=  qM6_d6;
               qP5_d1 <=  qP5;
               qP5_d2 <=  qP5_d1;
               qP5_d3 <=  qP5_d2;
               qP5_d4 <=  qP5_d3;
               qP5_d5 <=  qP5_d4;
               qM5_d1 <=  qM5;
               qM5_d2 <=  qM5_d1;
               qM5_d3 <=  qM5_d2;
               qM5_d4 <=  qM5_d3;
               qM5_d5 <=  qM5_d4;
               qM5_d6 <=  qM5_d5;
               qP4_d1 <=  qP4;
               qP4_d2 <=  qP4_d1;
               qP4_d3 <=  qP4_d2;
               qM4_d1 <=  qM4;
               qM4_d2 <=  qM4_d1;
               qM4_d3 <=  qM4_d2;
               qM4_d4 <=  qM4_d3;
               qP3_d1 <=  qP3;
               qP3_d2 <=  qP3_d1;
               qM3_d1 <=  qM3;
               qM3_d2 <=  qM3_d1;
               qM3_d3 <=  qM3_d2;
               qP2_d1 <=  qP2;
               qM2_d1 <=  qM2;
               qM2_d2 <=  qM2_d1;
               qM1_d1 <=  qM1;
               qP_d1 <=  qP;
               qP_d2 <=  qP_d1;
               qM_d1 <=  qM;
               mR_d1 <=  mR;
               fRnorm_d1 <=  fRnorm;
               round_d1 <=  round;
            end if;
         end if;
      end process;
   fX <= "1" & X(51 downto 0);
   fY <= "1" & Y(51 downto 0);
   -- exponent difference, sign and exception combination computed early, to have fewer bits to pipeline
   expR0 <= ("00" & X(62 downto 52)) - ("00" & Y(62 downto 52));
   sR <= X(63) xor Y(63);
   -- early exception handling 
   exnXY <= X(65 downto 64) & Y(65 downto 64);
   with exnXY  select 
      exnR0 <= 
         "01"	 when "0101",										-- normal
         "00"	 when "0001" | "0010" | "0110", -- zero
         "10"	 when "0100" | "1000" | "1001", -- overflow
         "11"	 when others;										-- NaN
   D <= fY ;
   psX <= "0" & fX ;
   betaw28 <=  "00" & psX;
   sel28 <= betaw28(55 downto 50) & D(51 downto 49);
   SelFunctionTable28: selFunction_F500_uid386
      port map ( X => sel28,
                 Y => q28_copy387);
   q28 <= q28_copy387; -- output copy to hold a pipeline register if needed

   with q28  select 
      absq28D <= 
         "000" & D						 when "001" | "111", -- mult by 1
         "00" & D & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q28_d1(2)  select 
   w27<= betaw28_d1 - absq28D_d1 when '0',
         betaw28_d1 + absq28D_d1 when others;

   betaw27 <= w27(53 downto 0) & "00"; -- multiplication by the radix
   sel27 <= betaw27(55 downto 50) & D_d1(51 downto 49);
   SelFunctionTable27: selFunction_F500_uid386
      port map ( X => sel27,
                 Y => q27_copy388);
   q27 <= q27_copy388; -- output copy to hold a pipeline register if needed

   with q27  select 
      absq27D <= 
         "000" & D_d1						 when "001" | "111", -- mult by 1
         "00" & D_d1 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q27_d1(2)  select 
   w26<= betaw27_d1 - absq27D_d1 when '0',
         betaw27_d1 + absq27D_d1 when others;

   betaw26 <= w26(53 downto 0) & "00"; -- multiplication by the radix
   sel26 <= betaw26(55 downto 50) & D_d2(51 downto 49);
   SelFunctionTable26: selFunction_F500_uid386
      port map ( X => sel26,
                 Y => q26_copy389);
   q26 <= q26_copy389; -- output copy to hold a pipeline register if needed

   with q26  select 
      absq26D <= 
         "000" & D_d2						 when "001" | "111", -- mult by 1
         "00" & D_d2 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q26_d1(2)  select 
   w25<= betaw26_d1 - absq26D_d1 when '0',
         betaw26_d1 + absq26D_d1 when others;

   betaw25 <= w25(53 downto 0) & "00"; -- multiplication by the radix
   sel25 <= betaw25(55 downto 50) & D_d3(51 downto 49);
   SelFunctionTable25: selFunction_F500_uid386
      port map ( X => sel25,
                 Y => q25_copy390);
   q25 <= q25_copy390_d1; -- output copy to hold a pipeline register if needed

   with q25  select 
      absq25D <= 
         "000" & D_d4						 when "001" | "111", -- mult by 1
         "00" & D_d4 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q25(2)  select 
   w24<= betaw25_d1 - absq25D when '0',
         betaw25_d1 + absq25D when others;

   betaw24 <= w24(53 downto 0) & "00"; -- multiplication by the radix
   sel24 <= betaw24(55 downto 50) & D_d4(51 downto 49);
   SelFunctionTable24: selFunction_F500_uid386
      port map ( X => sel24,
                 Y => q24_copy391);
   q24 <= q24_copy391_d1; -- output copy to hold a pipeline register if needed

   with q24  select 
      absq24D <= 
         "000" & D_d5						 when "001" | "111", -- mult by 1
         "00" & D_d5 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q24_d1(2)  select 
   w23<= betaw24_d2 - absq24D_d1 when '0',
         betaw24_d2 + absq24D_d1 when others;

   betaw23 <= w23(53 downto 0) & "00"; -- multiplication by the radix
   sel23 <= betaw23(55 downto 50) & D_d6(51 downto 49);
   SelFunctionTable23: selFunction_F500_uid386
      port map ( X => sel23,
                 Y => q23_copy392);
   q23 <= q23_copy392; -- output copy to hold a pipeline register if needed

   with q23  select 
      absq23D <= 
         "000" & D_d6						 when "001" | "111", -- mult by 1
         "00" & D_d6 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q23_d1(2)  select 
   w22<= betaw23_d1 - absq23D_d1 when '0',
         betaw23_d1 + absq23D_d1 when others;

   betaw22 <= w22(53 downto 0) & "00"; -- multiplication by the radix
   sel22 <= betaw22(55 downto 50) & D_d7(51 downto 49);
   SelFunctionTable22: selFunction_F500_uid386
      port map ( X => sel22,
                 Y => q22_copy393);
   q22 <= q22_copy393; -- output copy to hold a pipeline register if needed

   with q22  select 
      absq22D <= 
         "000" & D_d7						 when "001" | "111", -- mult by 1
         "00" & D_d7 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q22_d1(2)  select 
   w21<= betaw22_d1 - absq22D_d1 when '0',
         betaw22_d1 + absq22D_d1 when others;

   betaw21 <= w21(53 downto 0) & "00"; -- multiplication by the radix
   sel21 <= betaw21(55 downto 50) & D_d8(51 downto 49);
   SelFunctionTable21: selFunction_F500_uid386
      port map ( X => sel21,
                 Y => q21_copy394);
   q21 <= q21_copy394_d1; -- output copy to hold a pipeline register if needed

   with q21  select 
      absq21D <= 
         "000" & D_d9						 when "001" | "111", -- mult by 1
         "00" & D_d9 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q21(2)  select 
   w20<= betaw21_d1 - absq21D when '0',
         betaw21_d1 + absq21D when others;

   betaw20 <= w20(53 downto 0) & "00"; -- multiplication by the radix
   sel20 <= betaw20(55 downto 50) & D_d9(51 downto 49);
   SelFunctionTable20: selFunction_F500_uid386
      port map ( X => sel20,
                 Y => q20_copy395);
   q20 <= q20_copy395_d1; -- output copy to hold a pipeline register if needed

   with q20  select 
      absq20D <= 
         "000" & D_d10						 when "001" | "111", -- mult by 1
         "00" & D_d10 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q20_d1(2)  select 
   w19<= betaw20_d2 - absq20D_d1 when '0',
         betaw20_d2 + absq20D_d1 when others;

   betaw19 <= w19(53 downto 0) & "00"; -- multiplication by the radix
   sel19 <= betaw19(55 downto 50) & D_d11(51 downto 49);
   SelFunctionTable19: selFunction_F500_uid386
      port map ( X => sel19,
                 Y => q19_copy396);
   q19 <= q19_copy396; -- output copy to hold a pipeline register if needed

   with q19  select 
      absq19D <= 
         "000" & D_d11						 when "001" | "111", -- mult by 1
         "00" & D_d11 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q19_d1(2)  select 
   w18<= betaw19_d1 - absq19D_d1 when '0',
         betaw19_d1 + absq19D_d1 when others;

   betaw18 <= w18(53 downto 0) & "00"; -- multiplication by the radix
   sel18 <= betaw18(55 downto 50) & D_d12(51 downto 49);
   SelFunctionTable18: selFunction_F500_uid386
      port map ( X => sel18,
                 Y => q18_copy397);
   q18 <= q18_copy397; -- output copy to hold a pipeline register if needed

   with q18  select 
      absq18D <= 
         "000" & D_d12						 when "001" | "111", -- mult by 1
         "00" & D_d12 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q18_d1(2)  select 
   w17<= betaw18_d1 - absq18D_d1 when '0',
         betaw18_d1 + absq18D_d1 when others;

   betaw17 <= w17(53 downto 0) & "00"; -- multiplication by the radix
   sel17 <= betaw17(55 downto 50) & D_d13(51 downto 49);
   SelFunctionTable17: selFunction_F500_uid386
      port map ( X => sel17,
                 Y => q17_copy398);
   q17 <= q17_copy398_d1; -- output copy to hold a pipeline register if needed

   with q17  select 
      absq17D <= 
         "000" & D_d14						 when "001" | "111", -- mult by 1
         "00" & D_d14 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q17(2)  select 
   w16<= betaw17_d1 - absq17D when '0',
         betaw17_d1 + absq17D when others;

   betaw16 <= w16(53 downto 0) & "00"; -- multiplication by the radix
   sel16 <= betaw16(55 downto 50) & D_d14(51 downto 49);
   SelFunctionTable16: selFunction_F500_uid386
      port map ( X => sel16,
                 Y => q16_copy399);
   q16 <= q16_copy399_d1; -- output copy to hold a pipeline register if needed

   with q16  select 
      absq16D <= 
         "000" & D_d15						 when "001" | "111", -- mult by 1
         "00" & D_d15 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q16_d1(2)  select 
   w15<= betaw16_d2 - absq16D_d1 when '0',
         betaw16_d2 + absq16D_d1 when others;

   betaw15 <= w15(53 downto 0) & "00"; -- multiplication by the radix
   sel15 <= betaw15(55 downto 50) & D_d16(51 downto 49);
   SelFunctionTable15: selFunction_F500_uid386
      port map ( X => sel15,
                 Y => q15_copy400);
   q15 <= q15_copy400; -- output copy to hold a pipeline register if needed

   with q15  select 
      absq15D <= 
         "000" & D_d16						 when "001" | "111", -- mult by 1
         "00" & D_d16 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q15_d1(2)  select 
   w14<= betaw15_d1 - absq15D_d1 when '0',
         betaw15_d1 + absq15D_d1 when others;

   betaw14 <= w14(53 downto 0) & "00"; -- multiplication by the radix
   sel14 <= betaw14(55 downto 50) & D_d17(51 downto 49);
   SelFunctionTable14: selFunction_F500_uid386
      port map ( X => sel14,
                 Y => q14_copy401);
   q14 <= q14_copy401; -- output copy to hold a pipeline register if needed

   with q14  select 
      absq14D <= 
         "000" & D_d17						 when "001" | "111", -- mult by 1
         "00" & D_d17 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q14_d1(2)  select 
   w13<= betaw14_d1 - absq14D_d1 when '0',
         betaw14_d1 + absq14D_d1 when others;

   betaw13 <= w13(53 downto 0) & "00"; -- multiplication by the radix
   sel13 <= betaw13(55 downto 50) & D_d18(51 downto 49);
   SelFunctionTable13: selFunction_F500_uid386
      port map ( X => sel13,
                 Y => q13_copy402);
   q13 <= q13_copy402; -- output copy to hold a pipeline register if needed

   with q13  select 
      absq13D <= 
         "000" & D_d18						 when "001" | "111", -- mult by 1
         "00" & D_d18 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q13_d1(2)  select 
   w12<= betaw13_d1 - absq13D_d1 when '0',
         betaw13_d1 + absq13D_d1 when others;

   betaw12 <= w12(53 downto 0) & "00"; -- multiplication by the radix
   sel12 <= betaw12(55 downto 50) & D_d19(51 downto 49);
   SelFunctionTable12: selFunction_F500_uid386
      port map ( X => sel12,
                 Y => q12_copy403);
   q12 <= q12_copy403_d1; -- output copy to hold a pipeline register if needed

   with q12  select 
      absq12D <= 
         "000" & D_d20						 when "001" | "111", -- mult by 1
         "00" & D_d20 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q12_d1(2)  select 
   w11<= betaw12_d2 - absq12D_d1 when '0',
         betaw12_d2 + absq12D_d1 when others;

   betaw11 <= w11(53 downto 0) & "00"; -- multiplication by the radix
   sel11 <= betaw11(55 downto 50) & D_d21(51 downto 49);
   SelFunctionTable11: selFunction_F500_uid386
      port map ( X => sel11,
                 Y => q11_copy404);
   q11 <= q11_copy404; -- output copy to hold a pipeline register if needed

   with q11  select 
      absq11D <= 
         "000" & D_d21						 when "001" | "111", -- mult by 1
         "00" & D_d21 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q11_d1(2)  select 
   w10<= betaw11_d1 - absq11D_d1 when '0',
         betaw11_d1 + absq11D_d1 when others;

   betaw10 <= w10(53 downto 0) & "00"; -- multiplication by the radix
   sel10 <= betaw10(55 downto 50) & D_d22(51 downto 49);
   SelFunctionTable10: selFunction_F500_uid386
      port map ( X => sel10,
                 Y => q10_copy405);
   q10 <= q10_copy405; -- output copy to hold a pipeline register if needed

   with q10  select 
      absq10D <= 
         "000" & D_d22						 when "001" | "111", -- mult by 1
         "00" & D_d22 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q10_d1(2)  select 
   w9<= betaw10_d1 - absq10D_d1 when '0',
         betaw10_d1 + absq10D_d1 when others;

   betaw9 <= w9(53 downto 0) & "00"; -- multiplication by the radix
   sel9 <= betaw9(55 downto 50) & D_d23(51 downto 49);
   SelFunctionTable9: selFunction_F500_uid386
      port map ( X => sel9,
                 Y => q9_copy406);
   q9 <= q9_copy406; -- output copy to hold a pipeline register if needed

   with q9  select 
      absq9D <= 
         "000" & D_d23						 when "001" | "111", -- mult by 1
         "00" & D_d23 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q9_d1(2)  select 
   w8<= betaw9_d1 - absq9D_d1 when '0',
         betaw9_d1 + absq9D_d1 when others;

   betaw8 <= w8(53 downto 0) & "00"; -- multiplication by the radix
   sel8 <= betaw8(55 downto 50) & D_d24(51 downto 49);
   SelFunctionTable8: selFunction_F500_uid386
      port map ( X => sel8,
                 Y => q8_copy407);
   q8 <= q8_copy407_d1; -- output copy to hold a pipeline register if needed

   with q8  select 
      absq8D <= 
         "000" & D_d25						 when "001" | "111", -- mult by 1
         "00" & D_d25 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q8_d1(2)  select 
   w7<= betaw8_d2 - absq8D_d1 when '0',
         betaw8_d2 + absq8D_d1 when others;

   betaw7 <= w7(53 downto 0) & "00"; -- multiplication by the radix
   sel7 <= betaw7(55 downto 50) & D_d26(51 downto 49);
   SelFunctionTable7: selFunction_F500_uid386
      port map ( X => sel7,
                 Y => q7_copy408);
   q7 <= q7_copy408; -- output copy to hold a pipeline register if needed

   with q7  select 
      absq7D <= 
         "000" & D_d26						 when "001" | "111", -- mult by 1
         "00" & D_d26 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q7_d1(2)  select 
   w6<= betaw7_d1 - absq7D_d1 when '0',
         betaw7_d1 + absq7D_d1 when others;

   betaw6 <= w6(53 downto 0) & "00"; -- multiplication by the radix
   sel6 <= betaw6(55 downto 50) & D_d27(51 downto 49);
   SelFunctionTable6: selFunction_F500_uid386
      port map ( X => sel6,
                 Y => q6_copy409);
   q6 <= q6_copy409; -- output copy to hold a pipeline register if needed

   with q6  select 
      absq6D <= 
         "000" & D_d27						 when "001" | "111", -- mult by 1
         "00" & D_d27 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q6_d1(2)  select 
   w5<= betaw6_d1 - absq6D_d1 when '0',
         betaw6_d1 + absq6D_d1 when others;

   betaw5 <= w5(53 downto 0) & "00"; -- multiplication by the radix
   sel5 <= betaw5(55 downto 50) & D_d28(51 downto 49);
   SelFunctionTable5: selFunction_F500_uid386
      port map ( X => sel5,
                 Y => q5_copy410);
   q5 <= q5_copy410; -- output copy to hold a pipeline register if needed

   with q5  select 
      absq5D <= 
         "000" & D_d28						 when "001" | "111", -- mult by 1
         "00" & D_d28 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q5_d1(2)  select 
   w4<= betaw5_d1 - absq5D_d1 when '0',
         betaw5_d1 + absq5D_d1 when others;

   betaw4 <= w4(53 downto 0) & "00"; -- multiplication by the radix
   sel4 <= betaw4(55 downto 50) & D_d29(51 downto 49);
   SelFunctionTable4: selFunction_F500_uid386
      port map ( X => sel4,
                 Y => q4_copy411);
   q4 <= q4_copy411_d1; -- output copy to hold a pipeline register if needed

   with q4  select 
      absq4D <= 
         "000" & D_d30						 when "001" | "111", -- mult by 1
         "00" & D_d30 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q4_d1(2)  select 
   w3<= betaw4_d2 - absq4D_d1 when '0',
         betaw4_d2 + absq4D_d1 when others;

   betaw3 <= w3(53 downto 0) & "00"; -- multiplication by the radix
   sel3 <= betaw3(55 downto 50) & D_d31(51 downto 49);
   SelFunctionTable3: selFunction_F500_uid386
      port map ( X => sel3,
                 Y => q3_copy412);
   q3 <= q3_copy412; -- output copy to hold a pipeline register if needed

   with q3  select 
      absq3D <= 
         "000" & D_d31						 when "001" | "111", -- mult by 1
         "00" & D_d31 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q3_d1(2)  select 
   w2<= betaw3_d1 - absq3D_d1 when '0',
         betaw3_d1 + absq3D_d1 when others;

   betaw2 <= w2(53 downto 0) & "00"; -- multiplication by the radix
   sel2 <= betaw2(55 downto 50) & D_d32(51 downto 49);
   SelFunctionTable2: selFunction_F500_uid386
      port map ( X => sel2,
                 Y => q2_copy413);
   q2 <= q2_copy413; -- output copy to hold a pipeline register if needed

   with q2  select 
      absq2D <= 
         "000" & D_d32						 when "001" | "111", -- mult by 1
         "00" & D_d32 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q2_d1(2)  select 
   w1<= betaw2_d1 - absq2D_d1 when '0',
         betaw2_d1 + absq2D_d1 when others;

   betaw1 <= w1(53 downto 0) & "00"; -- multiplication by the radix
   sel1 <= betaw1(55 downto 50) & D_d33(51 downto 49);
   SelFunctionTable1: selFunction_F500_uid386
      port map ( X => sel1,
                 Y => q1_copy414);
   q1 <= q1_copy414; -- output copy to hold a pipeline register if needed

   with q1  select 
      absq1D <= 
         "000" & D_d33						 when "001" | "111", -- mult by 1
         "00" & D_d33 & "0"			   when "010" | "110", -- mult by 2
         (55 downto 0 => '0')	 when others;        -- mult by 0

   with q1_d1(2)  select 
   w0<= betaw1_d1 - absq1D_d1 when '0',
         betaw1_d1 + absq1D_d1 when others;

   wfinal <= w0(53 downto 0);
   qM0 <= wfinal(53); -- rounding bit is the sign of the remainder
   qP28 <=      q28(1 downto 0);
   qM28 <=      q28(2) & "0";
   qP27 <=      q27(1 downto 0);
   qM27 <=      q27(2) & "0";
   qP26 <=      q26(1 downto 0);
   qM26 <=      q26(2) & "0";
   qP25 <=      q25(1 downto 0);
   qM25 <=      q25(2) & "0";
   qP24 <=      q24(1 downto 0);
   qM24 <=      q24(2) & "0";
   qP23 <=      q23(1 downto 0);
   qM23 <=      q23(2) & "0";
   qP22 <=      q22(1 downto 0);
   qM22 <=      q22(2) & "0";
   qP21 <=      q21(1 downto 0);
   qM21 <=      q21(2) & "0";
   qP20 <=      q20(1 downto 0);
   qM20 <=      q20(2) & "0";
   qP19 <=      q19(1 downto 0);
   qM19 <=      q19(2) & "0";
   qP18 <=      q18(1 downto 0);
   qM18 <=      q18(2) & "0";
   qP17 <=      q17(1 downto 0);
   qM17 <=      q17(2) & "0";
   qP16 <=      q16(1 downto 0);
   qM16 <=      q16(2) & "0";
   qP15 <=      q15(1 downto 0);
   qM15 <=      q15(2) & "0";
   qP14 <=      q14(1 downto 0);
   qM14 <=      q14(2) & "0";
   qP13 <=      q13(1 downto 0);
   qM13 <=      q13(2) & "0";
   qP12 <=      q12(1 downto 0);
   qM12 <=      q12(2) & "0";
   qP11 <=      q11(1 downto 0);
   qM11 <=      q11(2) & "0";
   qP10 <=      q10(1 downto 0);
   qM10 <=      q10(2) & "0";
   qP9 <=      q9(1 downto 0);
   qM9 <=      q9(2) & "0";
   qP8 <=      q8(1 downto 0);
   qM8 <=      q8(2) & "0";
   qP7 <=      q7(1 downto 0);
   qM7 <=      q7(2) & "0";
   qP6 <=      q6(1 downto 0);
   qM6 <=      q6(2) & "0";
   qP5 <=      q5(1 downto 0);
   qM5 <=      q5(2) & "0";
   qP4 <=      q4(1 downto 0);
   qM4 <=      q4(2) & "0";
   qP3 <=      q3(1 downto 0);
   qM3 <=      q3(2) & "0";
   qP2 <=      q2(1 downto 0);
   qM2 <=      q2(2) & "0";
   qP1 <=      q1(1 downto 0);
   qM1 <=      q1(2) & "0";
   qP <= qP28_d33 & qP27_d32 & qP26_d31 & qP25_d29 & qP24_d28 & qP23_d27 & qP22_d26 & qP21_d24 & qP20_d23 & qP19_d22 & qP18_d21 & qP17_d19 & qP16_d18 & qP15_d17 & qP14_d16 & qP13_d15 & qP12_d13 & qP11_d12 & qP10_d11 & qP9_d10 & qP8_d8 & qP7_d7 & qP6_d6 & qP5_d5 & qP4_d3 & qP3_d2 & qP2_d1 & qP1;
   qM <= qM28_d34(0) & qM27_d33 & qM26_d32 & qM25_d30 & qM24_d29 & qM23_d28 & qM22_d27 & qM21_d25 & qM20_d24 & qM19_d23 & qM18_d22 & qM17_d20 & qM16_d19 & qM15_d18 & qM14_d17 & qM13_d16 & qM12_d14 & qM11_d13 & qM10_d12 & qM9_d11 & qM8_d9 & qM7_d8 & qM6_d7 & qM5_d6 & qM4_d4 & qM3_d3 & qM2_d2 & qM1_d1 & qM0;
   quotient <= qP_d2 - qM_d1;
   -- We need a mR in (0, -wf-2) format: 1+wF fraction bits, 1 round bit, and 1 guard bit for the normalisation,
   -- quotient is the truncation of the exact quotient to at least 2^(-wF-2) bits
   -- now discarding its possible known MSB zeroes, and dropping the possible extra LSB bit (due to radix 4) 
   mR <= quotient(54 downto 0); 
   -- normalisation
   fRnorm <=    mR(53 downto 1)  when mR(54)= '1'
           else mR(52 downto 0);  -- now fRnorm is a (-1, -wF-1) fraction
   round <= fRnorm(0); 
   expR1 <= expR0_d36 + ("000" & (9 downto 1 => '1') & mR_d1(54)); -- add back bias
   -- final rounding
   expfrac <= expR1 & fRnorm_d1(52 downto 1) ;
   expfracR <= expfrac + ((64 downto 1 => '0') & round_d1);
   exnR <=      "00"  when expfracR(64) = '1'   -- underflow
           else "10"  when  expfracR(64 downto 63) =  "01" -- overflow
           else "01";      -- 00, normal case
   with exnR0_d36  select 
      exnRfinal <= 
         exnR   when "01", -- normal
         exnR0_d36  when others;
   R <= exnRfinal & sR_d36 & expfracR(62 downto 0);
end architecture;

