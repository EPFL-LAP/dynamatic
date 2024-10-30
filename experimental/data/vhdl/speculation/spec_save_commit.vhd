library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
USE work.types.all;
entity spec_save_commit is

  Generic (
    DATA_SIZE_IN:integer; 
    DATA_SIZE_OUT:integer; 
    FIFO_DEPTH : integer
  );
 
  Port ( 
    clk, rst : in std_logic;  
    dataInArray : in data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
    specInArray   : in  data_array(0 downto 0)(0 downto 0);
    controlInArray : in data_array(0 downto 0)(2 downto 0); -- 000:pass, 001:kill, 010:resend, 011:kill-pass, 100:no_cmp
    pValidArray : in std_logic_vector(1 downto 0); -- (control, data)
    readyArray : out std_logic_vector(1 downto 0); -- (control, data)

    dataOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
    specOutArray  : out data_array(0 downto 0)(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0)

  );
end spec_save_commit;
 
architecture arch of spec_save_commit is

    signal HeadEn   : std_logic := '0';
    signal TailEn  : std_logic := '0';
    signal CurrEn  : std_logic := '0';

    signal PassEn  : std_logic := '0';
    signal KillEn  : std_logic := '0';
    signal ResendEn  : std_logic := '0';
    signal NoCmpEn : std_logic := '0';

    signal Tail : natural range 0 to FIFO_DEPTH - 1;
    signal Head : natural range 0 to FIFO_DEPTH - 1;
    signal Curr : natural range 0 to FIFO_DEPTH - 1;

    signal CurrEmpty    : std_logic;
    signal Empty    : std_logic;
    signal Full : std_logic;

    type FIFO_Memory is array (0 to FIFO_DEPTH - 1) of STD_LOGIC_VECTOR (DATA_SIZE_IN+1 -1 downto 0);
    signal Memory : FIFO_Memory;

    signal specdataInArray  : data_array(0 downto 0)(DATA_SIZE_IN+1 - 1 downto 0);

    signal bypass : std_logic;

begin
    specdataInArray(0) <= specInArray(0) & dataInArray(0);

    readyArray(0) <= not Full;
    validArray(0) <= (PassEn and (not CurrEmpty or pValidArray(0))) or (ResendEn and not Empty);
    --validArray(0) <= (PassEn and not CurrEmpty) or (ResendEn and not Empty);
    
    TailEn <= not Full and pValidArray(0);
    HeadEn <= not Empty and ((nReadyArray(0) and ResendEn) or (readyArray(1) and KillEn));
    CurrEn <= ( (not CurrEmpty or pValidArray(0)) and (nReadyArray(0) and PassEn) ) or (not Empty and (nReadyArray(0) and NoCmpEn) );

    bypass <= pValidArray(0) and CurrEmpty;
-------------------
-- comb process for control en
en_proc : process (pValidArray, controlInArray)
    begin
        PassEn <= '0';
        KillEn <= '0';
        ResendEn <= '0';
        NoCmpEn <= '0';

        if pValidArray(1) = '1' and controlInArray(0) = "000" then
            PassEn <= '1';
        elsif pValidArray(1) = '1' and controlInArray(0) = "001" then
            KillEn <= '1';
        elsif pValidArray(1) = '1' and controlInArray(0) = "010" then
            ResendEn <= '1';
        elsif pValidArray(1) = '1' and controlInArray(0) = "011" then
            PassEn <= '1';
            KillEn <= '1';
        elsif pValidArray(1) = '1' and controlInArray(0) = "100" then
            ResendEn <= '1';
            NoCmpEn <= '1';
        end if;
    end process;

-------------------------------------------
-- comb process for control ready
ready_proc : process (PassEn, KillEn, ResendEn, CurrEmpty, Empty, nReadyArray, pValidArray)
    begin
        -- Note: PassEn and KillEn can be simultaneously '1'
        -- In that case, PassEn is prioritized
        if PassEn = '1' then
            readyArray(1) <= (not CurrEmpty or pValidArray(0)) and nReadyArray(0);
            --readyArray(1) <= not CurrEmpty and nReadyArray(0);
        elsif ResendEn = '1' then
            readyArray(1) <= not Empty and nReadyArray(0);
        elsif KillEn = '1' then
            readyArray(1) <= not Empty;
        else
            readyArray(1) <= '0';
        end if;
    end process;
-------------------------------------------
-- comb process for output data
output_proc : process (PassEn, Memory, Curr, bypass, specdataInArray, Head)
    begin
        if PassEn = '1' then
            if bypass = '1' then
                dataOutArray(0) <=  specdataInArray(0)(DATA_SIZE_OUT - 1 downto 0);
                specOutArray(0)(0) <= specdataInArray(0)(DATA_SIZE_OUT+1 - 1);
            else
                dataOutArray(0) <=  Memory(Curr)(DATA_SIZE_OUT - 1 downto 0);
                specOutArray(0)(0) <= Memory(Curr)(DATA_SIZE_OUT+1 - 1);
            end if;
        else
            dataOutArray(0) <=  Memory(Head)(DATA_SIZE_OUT - 1 downto 0);
            specOutArray(0)(0) <= '0';
        end if;
    end process;


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
                Memory(Tail) <= specdataInArray(0);
                
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
-- process for updating curr
CurrUpdate_proc : process (clk)
   
  begin
  if rising_edge(clk) then
  
    if rst = '1' then
       Curr <= 0;
    else
  
        if (CurrEn = '1') then

            Curr  <= (Curr + 1) mod FIFO_DEPTH;
                      
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

 -------------------------------------------
-- process for updating curr empty
CurrEmptyUpdate_proc : process (clk)
   
  begin
  if rising_edge(clk) then
  
    if rst = '1' then
       CurrEmpty <= '1';
    else
        -- if only emptying but not filling
        if (TailEn = '0') and (CurrEn = '1') then

            -- if new head index will reach tail index
            if ((Curr +1) mod FIFO_DEPTH = Tail) then

                CurrEmpty  <= '1';

            end if;
        -- if only filling but not emptying
        elsif (TailEn = '1') and (CurrEn = '0') then
                CurrEmpty <= '0';
       -- otherwise, nothing is happening or simultaneous read and write
                      
        end if;
       
    end if;
  end if;
end process;

end architecture;
