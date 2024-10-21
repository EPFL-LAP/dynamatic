library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
use work.types.all;
entity specgenCore is

  Generic (
    DATA_SIZE_IN:integer; 
    DATA_SIZE_OUT:integer
  );
 
  Port ( 
    clk, rst : in std_logic;

    dataInArray : in data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
    specInArray : in std_logic_vector(0 downto 0);
    predictInArray : in data_array(0 downto 0)(DATA_SIZE_IN-1 downto 0);
    fifoInArray : in std_logic_vector(DATA_SIZE_IN-1 downto 0);
    pValidArray : in std_logic_vector(2 downto 0); -- (fifo, predict, data)
    readyArray : out std_logic_vector(2 downto 0); -- (fifo, predict, data)

    dataOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
    specOutArray : out std_logic_vector(0 downto 0);
    fifoOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
    controlOutArray : out std_logic_vector(2 downto 0); -- 000:spec, 001:no cmp, 010:cmp correct, 011:resend, 100:kill, 101:correct-spec
    nReadyArray : in std_logic_vector(1 downto 0); -- (fifo, control)
    validArray : out std_logic_vector(1 downto 0);  -- (fifo, control)
    
    StateInternal : out std_logic_vector(3 downto 0)
    
  );
end specgenCore;
 
architecture arch of specgenCore is

type State_type is (PASS, SPEC, NO_CMP, CMP_CORRECT, CMP_WRONG, KILL1, KILL2, KILL3, KILL_SPEC);
type Control_type is (CONTROL_SPEC, CONTROL_NO_CMP, CONTROL_CMP_CORRECT, CONTROL_RESEND, CONTROL_KILL, CONTROL_CORRECT_SPEC);
signal State : State_type;

signal DatapV : std_logic;
signal PredictpV : std_logic;
signal FifoNotEmpty : std_logic;
signal ControlnR : std_logic;
signal FifoNotFull : std_logic;

signal DataR : std_logic;
signal PredictR : std_logic;
signal FifoR : std_logic;
signal ControlV : std_logic;
signal FifoV : std_logic;

--signal StateInternal : std_logic_vector(3 downto 0);
signal ControlInternal : Control_type;

begin
    DatapV <= pValidArray(0);
    PredictpV <= pValidArray(1);
    FifoNotEmpty <= pValidArray(2);
    ControlnR <= nReadyArray(0);
    FifoNotFull <= nReadyArray(1);

    readyArray(0) <= DataR;
    readyArray(1) <= PredictR;
    readyArray(2) <= FifoR;
    validArray(0) <= ControlV;
    validArray(1) <= FifoV;

process(ControlInternal)
    begin
        case ControlInternal is 
            when CONTROL_SPEC =>
                controlOutArray <= "000";
            when CONTROL_NO_CMP =>
                controlOutArray <= "001";
            when CONTROL_CMP_CORRECT =>
                controlOutArray <= "010";
            when CONTROL_RESEND =>
                controlOutArray <= "011";
            when CONTROL_KILL =>
                controlOutArray <= "100";
            when CONTROL_CORRECT_SPEC =>
                controlOutArray <= "101";
        end case;
    end process;

process(State)
    begin
        case State is 
            when PASS => -- 0
                StateInternal <= "0000";
            when SPEC => -- 1
                StateInternal <= "0001";
            when NO_CMP => -- 2
                StateInternal <= "0010";
            when CMP_CORRECT => -- 3
                StateInternal <= "0011";
            when CMP_WRONG => -- 4
                StateInternal <= "0100";
            when KILL1 => -- 5
                StateInternal <= "0101";
            when KILL2 => -- 6
                StateInternal <= "0110";
            when KILL3 => -- 7
                StateInternal <= "0111";
            when KILL_SPEC => -- 8
                StateInternal <= "1000";

        end case;

    end process;
   
state_proc : process (clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                State <= PASS;
            else
                case State is 
                    when PASS =>
                        if (DatapV = '0' and PredictpV = '1' and FifoNotFull = '1' and ControlnR = '0') then
                            State <= SPEC;
                        elsif (DatapV = '1' and FifoNotEmpty = '0' and ControlnR = '0') then
                            State <= NO_CMP;
                        elsif (DatapV = '1' and FifoNotEmpty = '1' and dataInArray(0) = fifoInArray and ControlnR = '0') then
                            State <= CMP_CORRECT;
                        elsif (DatapV = '1' and FifoNotEmpty = '1' and dataInArray(0) /= fifoInArray and ControlnR = '0') then
                            State <= CMP_WRONG;    
                        elsif (DatapV = '1' and FifoNotEmpty = '1' and dataInArray(0) /= fifoInArray and ControlnR = '1') then
                            State <= KILL1;
                        end if;
                    when SPEC =>
                        if (ControlnR = '1') then
                            State <= PASS;
                        end if;
                    when NO_CMP =>
                        if (ControlnR = '1') then
                            State <= PASS;
                        end if;
                    when CMP_CORRECT =>
                        if (ControlnR = '1') then
                            State <= PASS;
                        end if;
                    when CMP_WRONG =>
                        if (ControlnR = '1') then
                            State <= KILL1;
                        end if;
                    when KILL1 =>
                        if (DatapV = '1' and specInArray(0) = '0') then
                            if (FifoNotEmpty = '0') then
                                State <= PASS;
                            else
                                State <= KILL2;
                            end if;
                        elsif (FifoNotEmpty = '0') then
                            State <= KILL3;
                        end if;
                    
                    when KILL2 =>
                        if (FifoNotEmpty = '0') then
                            State <= PASS;
                        end if;

                    when KILL3 =>
                        if ((DatapV = '0' or (DatapV = '1' and specInArray(0) = '1')) and PredictpV = '1' and FifoNotFull = '1' and ControlnR = '0') then
                            State <= KILL_SPEC;
                        elsif (DatapV = '1' and specInArray(0) = '0' and PredictpV = '1' and FifoNotFull = '1' and ControlnR = '0') then
                            State <= SPEC;
                        elsif (DatapV = '1' and specInArray(0) = '0') then
                            State <= PASS;
                        end if;
                    when KILL_SPEC =>
                        if ((DatapV = '0' or (DatapV = '1' and specInArray(0) = '1')) and ControlnR = '1') then
                            State <= KILL3;
                        elsif (DatapV = '1' and specInArray(0) = '0' and ControlnR = '0') then
                            State <= SPEC;
                        elsif (DatapV = '1' and specInArray(0) = '0' and ControlnR = '1') then
                            State <= PASS;
                        end if;
                end case;
            end if;
        end if;

    end process;

output_proc : process (State, dataInArray, specInArray, fifoInArray, predictInArray, DatapV, PredictpV, FifoNotEmpty, ControlnR, FifoNotFull)
    begin

        dataOutArray <= dataInArray;
        specOutArray(0) <= '0';
        fifoOutArray <= predictInArray;
        ControlInternal <= CONTROL_SPEC;

        case State is 
            when PASS =>    
                DataR <= ControlnR;
                PredictR <= FifoNotFull and ControlnR;

                if (DatapV = '1' and FifoNotEmpty = '1' and dataInArray(0) = fifoInArray) then
                    FifoR <= '1';
                else
                    FifoR <= '0';
                end if;

                --FifoV <= not DatapV and PredictpV;
                FifoV <= '0';

                if (DatapV = '0' and PredictpV = '1' and FifoNotFull = '1') then
                    ControlV <= '1';
                    ControlInternal <= CONTROL_SPEC;
                    dataOutArray <= predictInArray;
                    specOutArray(0) <= '1';
                    FifoV <= '1';
                elsif (DatapV = '1' and FifoNotEmpty = '0') then
                    ControlV <= '1';
                    ControlInternal <= CONTROL_NO_CMP;
                    dataOutArray <= dataInArray;
                    specOutArray(0) <= '0';
                elsif (DatapV = '1' and PredictpV = '1' and FifoNotEmpty = '1' and dataInArray(0) = fifoInArray) then
                    ControlV <= '1';
                    ControlInternal <= CONTROL_CORRECT_SPEC;
                    dataOutArray <= predictInArray;
                    specOutArray(0) <= '1';
                    FifoV <= '1';
                elsif (DatapV = '1' and PredictpV = '0' and FifoNotEmpty = '1' and dataInArray(0) = fifoInArray) then
                    ControlV <= '1';
                    ControlInternal <= CONTROL_CMP_CORRECT;
                elsif (DatapV = '1' and FifoNotEmpty = '1' and dataInArray(0) /= fifoInArray) then
                    ControlV <= '1';
                    ControlInternal <= CONTROL_RESEND;
                    dataOutArray <= dataInArray;
                    specOutArray(0) <= '0';
                else
                    ControlV <= '0';
                end if;
                
            when SPEC => 
                DataR <= '0';
                PredictR <= ControlnR;
                FifoR <= '0';
                ControlV <= '1';
                FifoV <= '0';

                ControlInternal <= CONTROL_SPEC;

                dataOutArray <= predictInArray;
                specOutArray(0) <= '1';
    
            when NO_CMP => 
                DataR <= ControlnR;
                PredictR <= '0';
                FifoR <= '0';
                ControlV <= '1';
                FifoV <= '0';

                ControlInternal <= CONTROL_NO_CMP;

                dataOutArray <= dataInArray;
                specOutArray(0) <= '0';
                
            when CMP_CORRECT => 
                DataR <= ControlnR;
                PredictR <= '0';
                FifoR <= '0';
                ControlV <= '1';
                FifoV <= '0';

                ControlInternal <= CONTROL_CMP_CORRECT;
                
            when CMP_WRONG => 
                DataR <= ControlnR;
                PredictR <= '0';
                FifoR <= '0';
                ControlV <= '1';
                FifoV <= '0';

                ControlInternal <= CONTROL_RESEND;

                dataOutArray <= dataInArray;
                specOutArray(0) <= '0';

            when KILL1 =>
                DataR <= DatapV and specInArray(0);
                FifoR <= ControlnR;
                PredictR<= '0';
                ControlV <= FifoNotEmpty;
                FifoV <= '0';

                ControlInternal <= CONTROL_KILL;

            when KILL2 =>
                DataR <= '0';
                FifoR <= ControlnR;
                PredictR<= '0';
                ControlV <= FifoNotEmpty;
                FifoV <= '0';

                ControlInternal <= CONTROL_KILL;

            when KILL3 =>
                DataR <= DatapV and specInArray(0);
                PredictR <= ControlnR;
                FifoR <= '0';
                ControlV <= PredictpV and FifoNotFull;
                FifoV <= PredictpV;

                ControlInternal <= CONTROL_SPEC;

                dataOutArray <= predictInArray;
                specOutArray(0) <= '1';

            when KILL_SPEC =>
                DataR <= DatapV and specInArray(0);
                PredictR <= ControlnR;
                FifoR <= '0';
                ControlV <= '1';
                FifoV <= '0';

                ControlInternal <= CONTROL_SPEC;

                dataOutArray <= predictInArray;
                specOutArray(0) <= '1';
                
        end case;

    end process;

end architecture;


library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
use work.types.all;
entity decodeSave is
    Port (
        controlInArray : in data_array(0 downto 0)(2 downto 0);
        pValidArray : in std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(0 downto 0);

        controlOutArray : out data_array(0 downto 0)(0 downto 0); -- 0:resend, 1:drop
        validArray : out std_logic_vector(0 downto 0);
        nReadyArray : in std_logic_vector(0 downto 0)

    );
end decodeSave;

architecture arch of decodeSave is

begin
    process (controlInArray, pValidArray, nReadyArray)
    begin
        if (controlInArray(0) = "001" or controlInArray(0) = "010" or controlInArray(0) = "011" or controlInArray(0) = "101") then
            readyArray(0) <= nReadyArray(0);
        else 
            readyArray(0) <= '1';
        end if;

        validArray(0) <= '0';
        controlOutArray(0)(0) <= '0';

        if (pValidArray(0) = '1') then
            if controlInArray(0) = "001" then -- no cmp
                validArray(0) <= '1';
                controlOutArray(0)(0) <= '1';
            elsif controlInArray(0) = "010" then -- cmp correct
                validArray(0) <= '1';
                controlOutArray(0)(0) <= '1';
            elsif controlInArray(0) = "101" then -- correct-spec
                validArray(0) <= '1';
                controlOutArray(0)(0) <= '1';
            elsif controlInArray(0) = "011" then --cmp wrong
                validArray(0) <= '1';
                controlOutArray(0)(0) <= '0';
            end if;
        end if;

    end process;
end architecture;


library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
use work.types.all;
entity decodeCommit is
    Port (
        controlInArray : in data_array(0 downto 0)(2 downto 0);
        pValidArray : in std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(0 downto 0);

        controlOutArray : out data_array(0 downto 0)(0 downto 0); -- 0:pass, 1:discard
        validArray : out std_logic_vector(0 downto 0);
        nReadyArray : in std_logic_vector(0 downto 0)

    );
end decodeCommit;

architecture arch of decodeCommit is

begin
    process (controlInArray, pValidArray, nReadyArray)
    begin
        if (controlInArray(0) = "010" or controlInArray(0) = "100" or controlInArray(0) = "101") then
            readyArray(0) <= nReadyArray(0);
        else 
            readyArray(0) <= '1';
        end if;
        
        validArray(0) <= '0';
        controlOutArray(0)(0) <= '0';

        if (pValidArray(0) = '1') then
            if controlInArray(0) = "010" then -- cmp correct
                validArray(0) <= '1';
                controlOutArray(0)(0) <= '0';
            elsif controlInArray(0) = "101" then -- correct-spec
                validArray(0) <= '1';
                controlOutArray(0)(0) <= '0';
            elsif controlInArray(0) = "100" then -- cmp wrong
                validArray(0) <= '1';
                controlOutArray(0)(0) <= '1';
            end if;
        end if;

    end process;
end architecture;



library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
use work.types.all;
entity decodeBranch is
    Port (
        controlInArray : in data_array(0 downto 0)(2 downto 0);
        pValidArray : in std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(0 downto 0);

        controlOutArray : out data_array(0 downto 0)(0 downto 0); -- 1:pass, 0:discard
        validArray : out std_logic_vector(0 downto 0);
        nReadyArray : in std_logic_vector(0 downto 0)

    );
end decodeBranch;

architecture arch of decodeBranch is

begin
    process (controlInArray, pValidArray, nReadyArray)
    begin
        if (controlInArray(0) = "010" or controlInArray(0) = "100" or controlInArray(0) = "101") then
            readyArray(0) <= nReadyArray(0);
        else 
            readyArray(0) <= '1';
        end if;
        
        validArray(0) <= '0';
        controlOutArray(0)(0) <= '0';

        if (pValidArray(0) = '1') then
            if controlInArray(0) = "010" then -- cmp correct
                validArray(0) <= '1';
                controlOutArray(0)(0) <= '0';
            elsif controlInArray(0) = "101" then -- correct-spec
                validArray(0) <= '1';
                controlOutArray(0)(0) <= '0';
            elsif controlInArray(0) = "100" then --cmp wrong
                validArray(0) <= '1';
                controlOutArray(0)(0) <= '1';
            end if;
        end if;

    end process;
end architecture;



library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
use work.types.all;
entity decodeSC is
    Port (
        controlInArray : in data_array(0 downto 0)(2 downto 0);
        pValidArray : in std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(0 downto 0);

        controlOut0Array : out data_array(0 downto 0)(2 downto 0); -- 000:pass, 001:kill, 010:resend, 011:kill-pass, 100:no_cmp
        controlOut1Array : out data_array(0 downto 0)(2 downto 0); -- 000:pass, 001:kill, 010:resend, 011:kill-pass, 100:no_cmp
        validArray : out std_logic_vector(1 downto 0); -- (control1 control0)
        nReadyArray : in std_logic_vector(1 downto 0) -- (control1 control0)


    );
end decodeSC;

architecture arch of decodeSC is

begin
    process (controlInArray, pValidArray, nReadyArray)
    begin
        if (controlInArray(0) = "000" or controlInArray(0) = "001" or controlInArray(0) = "010" or controlInArray(0) = "011" or controlInArray(0) = "101") then
            readyArray(0) <= nReadyArray(0);
        else
            readyArray(0) <= nReadyArray(1);
        end if;
        
        validArray <= "00";
        controlOut0Array(0) <= "000";
        controlOut1Array(0) <= "000";

        if (pValidArray(0) = '1') then
            if controlInArray(0) = "000" then -- spec
                validArray(0) <= '1';
                controlOut0Array(0) <= "000";
            elsif controlInArray(0) = "001" then -- no cmp
                validArray(0) <= '1';
                controlOut0Array(0) <= "100";
            elsif controlInArray(0) = "010" then -- cmp correct
                validArray(0) <= '1';
                controlOut0Array(0) <= "001";
            elsif controlInArray(0) = "101" then -- correct-spec
                validArray(0) <= '1';
                controlOut0Array(0) <= "011";
            elsif controlInArray(0) = "011" then -- cmp wrong resend
                validArray(0) <= '1';
                controlOut0Array(0) <= "010";
            elsif controlInArray(0) = "100" then -- cmp wrong kill
                validArray(1) <= '1';
                controlOut1Array(0) <= "001";
            end if;
        end if;

    end process;
end architecture;

library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
use work.types.all;
entity decodeOutput is
    Port (
        controlInArray : in data_array(0 downto 0)(2 downto 0);
        pValidArray : in std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(0 downto 0);

        validArray : out std_logic_vector(0 downto 0);
        nReadyArray : in std_logic_vector(0 downto 0)

    );
end decodeOutput;

architecture arch of decodeOutput is

begin
    process (controlInArray, pValidArray, nReadyArray)
    begin
        if (controlInArray(0) = "000" or controlInArray(0) = "001" or controlInArray(0) = "011" or controlInArray(0) = "101") then
            readyArray(0) <= nReadyArray(0);
        else 
            readyArray(0) <= '1';
        end if;
        
        validArray(0) <= '0';

        if (pValidArray(0) = '1') then
            if controlInArray(0) = "000" then -- spec 
                validArray(0) <= '1';
            elsif controlInArray(0) = "101" then -- correct-spec 
                validArray(0) <= '1';
            elsif controlInArray(0) = "001" then -- no cmp
                validArray(0) <= '1';
            elsif controlInArray(0) = "011" then -- cmp wrong resend
                validArray(0) <= '1';
            end if;
        end if;

    end process;
end architecture;


library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
use work.types.all;
entity predictor is
    generic (
        DATA_SIZE_OUT : integer   -- use normal data size, eg- 32
    );
    port (
        clk, rst     : in  std_logic;

        enable       : in std_logic_vector(0 downto 0);

        dataInArray  : in data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        pValidArray  : in  std_logic_vector(0 downto 0);
        readyArray   : out std_logic_vector(1 downto 0); -- (enable, data)

        dataOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        nReadyArray  : in  std_logic_vector(0 downto 0);
        validArray   : out std_logic_vector(0 downto 0)
    );
end predictor;

architecture arch of predictor is
    signal zeros : std_logic_vector(DATA_SIZE_OUT-2 downto 0);
    signal data_reg: std_logic_vector(DATA_SIZE_OUT-1 downto 0);

begin
    
    zeros <= (others => '0');

    --predicted value is 1 by default and updated to the latest real value
    process(clk, rst) is
          begin
           if (rst = '1') then
            
            data_reg <= zeros & '1';
              
            elsif (rising_edge(clk)) then
                if (pValidArray(0) = '1') then
                    data_reg <= dataInArray(0);  
                end if;                  
            end if;
    end process;

     
    
    readyArray(0) <= nReadyArray(0);
    readyArray(1) <= nReadyArray(0);

    -- Predictor output valid if enabled
    dataOutArray(0) <= data_reg;
    validArray(0) <= enable(0);

end arch;



library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
use work.types.all;
entity predFifo is

  Generic (
    DATA_SIZE_IN:integer; 
    DATA_SIZE_OUT:integer; 
    FIFO_DEPTH : integer
  );
 
  Port ( 
    clk, rst : in std_logic;  

    dataInArray : in data_array(0 downto 0)(DATA_SIZE_IN-1 downto 0);
    pValidArray : in std_logic_vector(0 downto 0); 
    readyArray : out std_logic_vector(0 downto 0);

    dataOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT-1 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0)
  );
end predFifo;
 
architecture arch of predFifo is

    signal HeadEn   : std_logic := '0';
    signal TailEn  : std_logic := '0';

    signal Tail : natural range 0 to FIFO_DEPTH - 1;
    signal Head : natural range 0 to FIFO_DEPTH - 1;

    signal Empty    : std_logic;
    signal Full : std_logic;

    type FIFO_Memory is array (0 to FIFO_DEPTH - 1) of STD_LOGIC_VECTOR (DATA_SIZE_IN-1 downto 0);
    signal Memory : FIFO_Memory;


begin
    validArray(0) <= not Empty;
    readyArray(0) <= not Full;
    
    TailEn <= not Full and pValidArray(0);
    HeadEn <= not Empty and nReadyArray(0);
    dataOutArray(0) <= Memory(Head);

----------------------------------------------------------------

-- Sequential Process

----------------------------------------------------------------

-------------------------------------------
-- process for writing data
fifo_proc : process (clk)
   
     begin        
        if rising_edge(clk) then
          if rst = '1' then
           
          else
            
            if (TailEn = '1' ) then
                -- Write Data to Memory
                Memory(Tail) <= dataInArray(0);
                
            end if;
            
          end if;
        end if;
    end process;


 
-------------------------------------------
-- process for updating tail
TailUpdate_proc : process (clk)
   
      begin
        if rising_edge(clk) then
          
            if rst = '1' then
               Tail <= 0;
            else
          
                if (TailEn = '1') then

                    Tail  <= (Tail + 1) mod FIFO_DEPTH;
                              
                end if;
               
            end if;
        end if;
    end process; 

-------------------------------------------
-- process for updating head
HeadUpdate_proc : process (clk)
   
  begin
  if rising_edge(clk) then
  
    if rst = '1' then
       Head <= 0;
    else
  
        if (HeadEn = '1') then

            Head  <= (Head + 1) mod FIFO_DEPTH;
                      
        end if;
       
    end if;
  end if;
end process; 

-------------------------------------------
-- process for updating full
FullUpdate_proc : process (clk)
   
  begin
  if rising_edge(clk) then
  
    if rst = '1' then
       Full <= '0';
    else
  
        -- if only filling but not emptying
        if (TailEn = '1') and (HeadEn = '0') then

            -- if new tail index will reach head index
            if ((Tail +1) mod FIFO_DEPTH = Head) then

                Full  <= '1';

            end if;
        -- if only emptying but not filling
        elsif (TailEn = '0') and (HeadEn = '1') then
                Full <= '0';
        -- otherwise, nothing is happening or simultaneous read and write
                      
        end if;
       
    end if;
  end if;
end process;
  
 -------------------------------------------
-- process for updating empty
EmptyUpdate_proc : process (clk)
   
  begin
  if rising_edge(clk) then
  
    if rst = '1' then
       Empty <= '1';
    else
        -- if only emptying but not filling
        if (TailEn = '0') and (HeadEn = '1') then

            -- if new head index will reach tail index
            if ((Head +1) mod FIFO_DEPTH = Tail) then

                Empty  <= '1';

            end if;
        -- if only filling but not emptying
        elsif (TailEn = '1') and (HeadEn = '0') then
                Empty <= '0';
       -- otherwise, nothing is happening or simultaneous read and write
                      
        end if;
       
    end if;
  end if;
end process;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use work.types.all;
entity speculator is
    generic (
        DATA_SIZE_IN  : integer;  -- use normal data size, eg- 32
        DATA_SIZE_OUT : integer;  -- use normal data size, eg- 32
        FIFO_DEPTH    : integer   -- extra parameter for FIFO
    );
    port (
        clk, rst      : in  std_logic;

        dataInArray   : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        specInArray   : in  data_array(0 downto 0)(0 downto 0);
        enableInArray : in  data_array(0 downto 0)(0 downto 0);
        pValidArray   : in  std_logic_vector(1 downto 0); -- (enable, data)
        readyArray    : out std_logic_vector(1 downto 0); -- (enable, data)
        
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        specOutArray  : out data_array(0 downto 0)(0 downto 0);
        saveOutArray : out data_array(0 downto 0)(0 downto 0);
        commitOutArray : out data_array(0 downto 0)(0 downto 0);
        scOut0Array : out data_array(0 downto 0)(2 downto 0);
        scOut1Array : out data_array(0 downto 0)(2 downto 0);
        scBranchOutArray : out data_array(0 downto 0)(0 downto 0);
        nReadyArray   : in  std_logic_vector(5 downto 0); -- (save, commit, sc1, sc0, sc_branch, data)
        validArray    : out std_logic_vector(5 downto 0)  -- (save, commit, sc1, sc0, sc_branch, data)


        --StateOut : out std_logic_vector(3 downto 0)
    );
end speculator;

architecture arch of speculator is 

signal fork_data_specInArray  : data_array(0 downto 0)(0 downto 0);
signal fork_data_dataInArray  : data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal fork_data_pValidArray  : std_logic_vector(0 downto 0);
signal fork_data_readyArray   : std_logic_vector(0 downto 0);

signal fork_data_dataOutArray : data_array(1 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal fork_data_nReadyArray  : std_logic_vector(1 downto 0);
signal fork_data_validArray   : std_logic_vector(1 downto 0);
signal fork_data_specOutArray : data_array(1 downto 0)(0 downto 0);


signal specgenCore_dataInArray : data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal specgenCore_specInArray : std_logic_vector(0 downto 0);
signal specgenCore_predictInArray : data_array(0 downto 0)(DATA_SIZE_IN-1 downto 0);
signal specgenCore_fifoInArray : std_logic_vector(DATA_SIZE_IN-1 downto 0);
signal specgenCore_pValidArray : std_logic_vector(2 downto 0); -- (fifo, predict, data)
signal specgenCore_readyArray : std_logic_vector(2 downto 0); -- (fifo, predict, data)

signal specgenCore_dataOutArray : data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
signal specgenCore_specOutArray : std_logic_vector(0 downto 0);
signal specgenCore_fifoOutArray : data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
signal specgenCore_controlOutArray : std_logic_vector(2 downto 0); -- 000:spec, 001:no cmp, 010:cmp correct, 011:resend, 100:kill
signal specgenCore_nReadyArray : std_logic_vector(1 downto 0); -- (fifo, control)
signal specgenCore_validArray : std_logic_vector(1 downto 0);  -- (fifo, control)
signal specgenCore_StateInternal : std_logic_vector(3 downto 0);


signal predictor_enable  :  std_logic_vector(0 downto 0);
signal predictor_dataInArray : data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
signal predictor_pValidArray  :  std_logic_vector(0 downto 0);
signal predictor_readyArray   : std_logic_vector(1 downto 0);

signal predictor_dataOutArray : data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
signal predictor_nReadyArray  :  std_logic_vector(0 downto 0);
signal predictor_validArray   : std_logic_vector(0 downto 0);



signal predFifo_dataInArray : data_array(0 downto 0)(DATA_SIZE_IN-1 downto 0);
signal predFifo_pValidArray : std_logic_vector(0 downto 0); 
signal predFifo_readyArray : std_logic_vector(0 downto 0);

signal predFifo_dataOutArray : data_array(0 downto 0)(DATA_SIZE_OUT-1 downto 0);
signal predFifo_validArray : std_logic_vector(0 downto 0);
signal predFifo_nReadyArray : std_logic_vector(0 downto 0);



signal fork_dataInArray  : data_array(0 downto 0)(2 downto 0);
signal fork_pValidArray  : std_logic_vector(0 downto 0);
signal fork_readyArray   : std_logic_vector(0 downto 0);

signal fork_dataOutArray : data_array(4 downto 0)(2 downto 0);
signal fork_nReadyArray  : std_logic_vector(4 downto 0);
signal fork_validArray   : std_logic_vector(4 downto 0);
signal fork_unconnected_spec   : data_array(4 downto 0)(0 downto 0);


signal decodeSave_controlInArray : data_array(0 downto 0)(2 downto 0);
signal decodeSave_pValidArray : std_logic_vector(0 downto 0);
signal decodeSave_readyArray : std_logic_vector(0 downto 0);

signal decodeSave_controlOutArray : data_array(0 downto 0)(0 downto 0);
signal decodeSave_validArray : std_logic_vector(0 downto 0);
signal decodeSave_nReadyArray : std_logic_vector(0 downto 0);



signal decodeCommit_controlInArray : data_array(0 downto 0)(2 downto 0);
signal decodeCommit_pValidArray : std_logic_vector(0 downto 0);
signal decodeCommit_readyArray : std_logic_vector(0 downto 0);

signal decodeCommit_controlOutArray : data_array(0 downto 0)(0 downto 0);
signal decodeCommit_validArray : std_logic_vector(0 downto 0);
signal decodeCommit_nReadyArray : std_logic_vector(0 downto 0);



signal decodeBranch_controlInArray : data_array(0 downto 0)(2 downto 0);
signal decodeBranch_pValidArray : std_logic_vector(0 downto 0);
signal decodeBranch_readyArray : std_logic_vector(0 downto 0);

signal decodeBranch_controlOutArray : data_array(0 downto 0)(0 downto 0);
signal decodeBranch_validArray : std_logic_vector(0 downto 0);
signal decodeBranch_nReadyArray : std_logic_vector(0 downto 0);



signal decodeOutput_controlInArray : data_array(0 downto 0)(2 downto 0);
signal decodeOutput_pValidArray : std_logic_vector(0 downto 0);
signal decodeOutput_readyArray : std_logic_vector(0 downto 0);

signal decodeOutput_validArray : std_logic_vector(0 downto 0);
signal decodeOutput_nReadyArray : std_logic_vector(0 downto 0);



signal decodeSC_controlInArray : data_array(0 downto 0)(2 downto 0);
signal decodeSC_pValidArray : std_logic_vector(0 downto 0);
signal decodeSC_readyArray : std_logic_vector(0 downto 0);

signal decodeSC_controlOut0Array : data_array(0 downto 0)(2 downto 0); -- 00:pass, 01:kill, 10:resend
signal decodeSC_controlOut1Array : data_array(0 downto 0)(2 downto 0); -- 00:pass, 01:kill, 10:resend
signal decodeSC_validArray : std_logic_vector(1 downto 0); -- (control1, control0)
signal decodeSC_nReadyArray : std_logic_vector(1 downto 0); -- (control1, control0)

begin

----------------------------------------------------------------

-- I/O signal connection

----------------------------------------------------------------
fork_data_dataInArray <= dataInArray;
fork_data_specInArray(0) <= specInArray(0);
fork_data_pValidArray(0) <= pValidArray(0);
readyArray(0) <= fork_data_readyArray(0);  -- data

predictor_enable(0) <= pValidArray(1); -- enable
readyArray(1) <= predictor_readyArray(1);-- enable



dataOutArray <=  specgenCore_dataOutArray;
specOutArray(0) <=  specgenCore_specOutArray;
saveOutArray <= decodeSave_controlOutArray;
commitOutArray <= decodeCommit_controlOutArray;
scOut0Array <= decodeSC_controlOut0Array;
scOut1Array <= decodeSC_controlOut1Array;
scBranchOutArray <= decodeBranch_controlOutArray;

decodeSave_nReadyArray(0) <= nReadyArray(5); -- save
decodeCommit_nReadyArray(0) <= nReadyArray(4); -- commit
decodeSC_nReadyArray(1) <= nReadyArray(3); -- sc1
decodeSC_nReadyArray(0) <= nReadyArray(2); -- sc0
decodeBranch_nReadyArray(0) <= nReadyArray(1); -- sc_branch
decodeOutput_nReadyArray(0) <= nReadyArray(0); -- data

  
validArray(5)  <=  decodeSave_validArray(0);-- save
validArray(4)  <=  decodeCommit_validArray(0);-- commit
validArray(3)  <=  decodeSC_validArray(1);-- sc1
validArray(2)  <=  decodeSC_validArray(0);-- sc0
validArray(1)  <=  decodeBranch_validArray(0);-- branch
validArray(0) <=  decodeOutput_validArray(0); -- data 

--StateOut <= specgenCore_StateInternal;

----------------------------------------------------------------

-- Inner signal connection

----------------------------------------------------------------
-- fork_data
specgenCore_dataInArray(0) <= fork_data_dataOutArray(0);
specgenCore_specInArray <= fork_data_specOutArray(0);  
specgenCore_pValidArray(0) <= fork_data_validArray(0);
fork_data_nReadyArray(0) <= specgenCore_readyArray(0); -- data

predictor_dataInArray(0) <= fork_data_dataOutArray(1);
predictor_pValidArray(0) <= fork_data_validArray(1);
fork_data_nReadyArray(1) <= predictor_readyArray(0);


specgenCore_predictInArray <= predictor_dataOutArray;
specgenCore_pValidArray(1) <= predictor_validArray(0);
predictor_nReadyArray(0) <= specgenCore_readyArray(1);

--predFifo
specgenCore_fifoInArray <= predFifo_dataOutArray(0);
specgenCore_pValidArray(2) <=predFifo_validArray(0);
predFifo_nReadyArray(0) <= specgenCore_readyArray(2);

predFifo_dataInArray <= specgenCore_fifoOutArray;
predFifo_pValidArray(0) <= specgenCore_validArray(1);
specgenCore_nReadyArray(1) <= predFifo_readyArray(0);

-- fork
fork_dataInArray(0) <= specgenCore_controlOutArray;
fork_pValidArray(0) <= specgenCore_validArray(0);
specgenCore_nReadyArray(0) <= fork_readyArray(0);

decodeOutput_controlInArray(0) <= fork_dataOutArray(4);
decodeOutput_pValidArray(0) <= fork_validArray(4);
fork_nReadyArray(4) <= decodeOutput_readyArray(0);

decodeSave_controlInArray(0) <= fork_dataOutArray(3);
decodeSave_pValidArray(0) <= fork_validArray(3);
fork_nReadyArray(3) <= decodeSave_readyArray(0);

decodeCommit_controlInArray(0) <= fork_dataOutArray(2);
decodeCommit_pValidArray(0) <= fork_validArray(2);
fork_nReadyArray(2) <= decodeCommit_readyArray(0);

decodeSC_controlInArray(0) <= fork_dataOutArray(1);
decodeSC_pValidArray(0) <= fork_validArray(1);
fork_nReadyArray(1) <= decodeSC_readyArray(0);

decodeBranch_controlInArray(0) <= fork_dataOutArray(0);
decodeBranch_pValidArray(0) <= fork_validArray(0);
fork_nReadyArray(0) <= decodeBranch_readyArray(0);

----------------------------------------------------------------

-- instantiation

----------------------------------------------------------------
data_fork: entity work.fork_old(arch) generic map(1, 2, DATA_SIZE_IN, DATA_SIZE_IN)
    port map (
        clk => clk,
        rst => rst,
        dataInArray         => fork_data_dataInArray,
        specInArray         => fork_data_specInArray,
        dataOutArray        => fork_data_dataOutArray,
        specOutArray        => fork_data_specOutArray,
        pValidArray         => fork_data_pValidArray,
        nReadyArray         => fork_data_nReadyArray,
        validArray          => fork_data_validArray,
        readyArray          => fork_data_readyArray
);

spengenCore0: entity work.specgenCore(arch) generic map(DATA_SIZE_IN, DATA_SIZE_IN)
    port map (
    clk => clk,
    rst => rst,

    dataInArray => specgenCore_dataInArray,
    specInArray => specgenCore_specInArray,
    predictInArray => specgenCore_predictInArray,
    fifoInArray => specgenCore_fifoInArray,
    pValidArray => specgenCore_pValidArray, 
    readyArray => specgenCore_readyArray,  

    dataOutArray => specgenCore_dataOutArray,  
    specOutArray => specgenCore_specOutArray, 
    fifoOutArray => specgenCore_fifoOutArray, 
    controlOutArray => specgenCore_controlOutArray,
    nReadyArray => specgenCore_nReadyArray,
    validArray => specgenCore_validArray,

    StateInternal => specgenCore_StateInternal
);

predictor0: entity work.predictor(arch) generic map(DATA_SIZE_IN)
    port map (
        clk => clk,
        rst => rst,

        enable => predictor_enable,
        dataInArray => predictor_dataInArray,
        pValidArray => predictor_pValidArray,
        readyArray => predictor_readyArray,

        dataOutArray => predictor_dataOutArray,
        validArray => predictor_validArray,
        nReadyArray => predictor_nReadyArray
);

predFifo0: entity work.predFifo(arch) generic map(DATA_SIZE_IN, DATA_SIZE_IN, FIFO_DEPTH)
    port map (
        clk => clk,
        rst => rst,

        dataInArray => predFifo_dataInArray,
        pValidArray => predFifo_pValidArray,
        readyArray => predFifo_readyArray,

        dataOutArray => predFifo_dataOutArray,
        validArray => predFifo_validArray,
        nReadyArray => predFifo_nReadyArray
);

fork0: entity work.fork_old(arch) generic map(1, 5, 3, 3)
    port map (
        clk => clk,
        rst => rst,
        dataInArray         => fork_dataInArray,
        specInArray(0)(0)   => '0',
        dataOutArray        => fork_dataOutArray,
        specOutArray        => fork_unconnected_spec,
        pValidArray         => fork_pValidArray,
        nReadyArray         => fork_nReadyArray,
        validArray          => fork_validArray,
        readyArray          => fork_readyArray
);

decodeSave0: entity work.decodeSave(arch)
    port map (
        controlInArray => decodeSave_controlInArray,
        pValidArray => decodeSave_pValidArray,
        readyArray => decodeSave_readyArray,

        controlOutArray => decodeSave_controlOutArray,
        validArray => decodeSave_validArray,
        nReadyArray => decodeSave_nReadyArray
       
);

decodeCommit0: entity work.decodeCommit(arch)
    port map (

        controlInArray => decodeCommit_controlInArray,
        pValidArray => decodeCommit_pValidArray,
        readyArray => decodeCommit_readyArray,

        controlOutArray => decodeCommit_controlOutArray,
        validArray => decodeCommit_validArray,
        nReadyArray => decodeCommit_nReadyArray
       
);

decodeSC0: entity work.decodeSC(arch)
    port map (

        controlInArray => decodeSC_controlInArray,
        pValidArray => decodeSC_pValidArray,
        readyArray => decodeSC_readyArray,

        controlOut0Array => decodeSC_controlOut0Array,
        controlOut1Array => decodeSC_controlOut1Array,
        validArray => decodeSC_validArray,
        nReadyArray => decodeSC_nReadyArray
       
);

decodeOutput0: entity work.decodeOutput(arch)
    port map (
        controlInArray => decodeOutput_controlInArray,
        pValidArray => decodeOutput_pValidArray,
        readyArray => decodeOutput_readyArray,

        validArray => decodeOutput_validArray,
        nReadyArray => decodeOutput_nReadyArray
       
);

decodeBranch0: entity work.decodeBranch(arch)
    port map (
        controlInArray => decodeBranch_controlInArray,
        pValidArray => decodeBranch_pValidArray,
        readyArray => decodeBranch_readyArray,

        controlOutArray => decodeBranch_controlOutArray,
        validArray => decodeBranch_validArray,
        nReadyArray => decodeBranch_nReadyArray
       
);
end arch;
