library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
 use work.customTypes.all;
entity init_elasticFifoInner is

  Generic (
    INPUT_COUNT:integer; OUTPUT_COUNT:integer; DATA_SIZE_IN:integer; DATA_SIZE_OUT:integer; FIFO_DEPTH : integer
  );
 
  Port ( 
    clk, rst : in std_logic;  
    dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0);
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);
    readyArray : out std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    pValidArray : in std_logic_vector(0 downto 0)
  );
end init_elasticFifoInner;
 
architecture arch of init_elasticFifoInner is

    signal ReadEn   : std_logic := '0';
    signal WriteEn  : std_logic := '0';
    signal Tail : natural range 0 to FIFO_DEPTH - 1;
    signal Head : natural range 0 to FIFO_DEPTH - 1;
    signal Empty    : std_logic;
    signal Full : std_logic;
    signal Bypass: std_logic;
    signal fifo_valid: std_logic;
    type FIFO_Memory is array (0 to FIFO_DEPTH - 1) of STD_LOGIC_VECTOR (DATA_SIZE_IN-1 downto 0);
    signal Memory : FIFO_Memory;


begin

    -- ready if there is space in the fifo
    readyArray(0) <= not Full or nReadyArray(0);

    -- read if next can accept and there is sth in fifo to read
    ReadEn <= (nReadyArray(0) and not Empty);

    validArray(0) <= not Empty;
    
    dataOutArray(0) <=  Memory(Head);

    WriteEn <= pValidArray(0) and ( not Full or nReadyArray(0));

    fifo_proc : process (CLK)
   
     begin        
        if rising_edge(CLK) then
          if RST = '1' then
            for I in 0 to FIFO_DEPTH - 1 loop
                -- I set the initialization to the index of each element, but it can be anything else up to us..
                Memory(I) <= std_logic_vector(to_unsigned(I, Memory(I)'length));  
            end loop;
          else
            
            if (WriteEn = '1' ) then
                -- Write Data to Memory
                Memory(Tail) <= dataInArray(0);
                
            end if;
            
          end if;
        end if;
    end process;
 
-------------------------------------------
-- process for updating tail
TailUpdate_proc : process (CLK)
   
      begin
        if rising_edge(CLK) then
          
            if RST = '1' then
               Tail <= FIFO_DEPTH - 1;
            else
          
                if (WriteEn = '1') then

                    Tail  <= (Tail + 1) mod FIFO_DEPTH;
                              
                end if;
               
            end if;
        end if;
    end process; 

-------------------------------------------
-- process for updating head
HeadUpdate_proc : process (CLK)
   
  begin
  if rising_edge(CLK) then
  
    if RST = '1' then
       Head <= 0;
    else
  
        if (ReadEn = '1') then

            Head  <= (Head + 1) mod FIFO_DEPTH;
                      
        end if;
       
    end if;
  end if;
end process; 

-------------------------------------------
-- process for updating full
FullUpdate_proc : process (CLK)
   
  begin
  if rising_edge(CLK) then
  
    if RST = '1' then
       --Full <= '0';
       Full <= '1';  -- initializing to full
    else
  
        -- if only filling but not emptying
        if (WriteEn = '1') and (ReadEn = '0') then

            -- if new tail index will reach head index
            if ((Tail +1) mod FIFO_DEPTH = Head) then

                Full  <= '1';

            end if;
        -- if only emptying but not filling
        elsif (WriteEn = '0') and (ReadEn = '1') then
                Full <= '0';
        -- otherwise, nothing is happening or simultaneous read and write
                      
        end if;
       
    end if;
  end if;
end process;
  
 -------------------------------------------
-- process for updating empty
EmptyUpdate_proc : process (CLK)
   
  begin
  if rising_edge(CLK) then
  
    if RST = '1' then
       Empty <= '0';  -- initializing to full
    else
        -- if only emptying but not filling
        if (WriteEn = '0') and (ReadEn = '1') then

            -- if new head index will reach tail index
            if ((Head +1) mod FIFO_DEPTH = Tail) then

                Empty  <= '1';

            end if;
        -- if only filling but not emptying
        elsif (WriteEn = '1') and (ReadEn = '0') then
                Empty <= '0';
       -- otherwise, nothing is happening or simultaneous read and write
                      
        end if;
       
    end if;
  end if;
end process;
end architecture;
--------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
USE work.customTypes.all;

entity free_tags_fifo is 
    generic(
        INPUTS        : integer;
        OUTPUTS        : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer;
        FIFO_DEPTH : integer
    );
port (
        clk, rst      : in  std_logic;
        dataInArray   : in  data_array(INPUTS - 1 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        pValidArray   : in  std_logic_vector(INPUTS - 1 downto 0);
        nReadyArray   : in  std_logic_vector(0 downto 0);
        validArray    : out std_logic_vector(0 downto 0);
        readyArray    : out std_logic_vector(INPUTS - 1 downto 0));
end free_tags_fifo;

architecture arch of free_tags_fifo is
    signal mux_sel : std_logic;
    signal fifo_valid, fifo_ready : STD_LOGIC;
    signal fifo_pvalid, fifo_nready : STD_LOGIC;
    signal fifo_in, fifo_out: std_logic_vector(DATA_SIZE_IN-1 downto 0);
begin
    

    process (mux_sel, fifo_out, dataInArray) is
        begin
            if (mux_sel = '1') then
                dataOutArray(0) <= fifo_out;
            else
                dataOutArray(0) <= dataInArray(0);
            end if;
    end process;

    validArray(0) <= pValidArray(0) or fifo_valid;    --fifo_valid is 0 only if fifo is empty
    readyArray(0) <= fifo_ready or nReadyArray(0);
    fifo_pvalid <= pValidArray(0) and (not nReadyArray(0) or fifo_valid); --store in FIFO if next is not ready or FIFO is already outputting something
    mux_sel <= fifo_valid;

    fifo_nready <= nReadyArray(0);
    fifo_in <= dataInArray(0);

    fifo: entity work.init_elasticFifoInner(arch) generic map (1, 1, DATA_SIZE_IN, DATA_SIZE_IN, FIFO_DEPTH) 
        port map (
        --inputs
            clk => clk, 
            rst => rst, 
            pValidArray(0)  => fifo_pvalid, 
            nReadyArray(0) => fifo_nready,    
            validArray(0) => fifo_valid, 
        --outputs
            readyArray(0) => fifo_ready,   
            dataInArray(0) =>fifo_in,
            dataOutArray(0) => fifo_out
        );

end arch;