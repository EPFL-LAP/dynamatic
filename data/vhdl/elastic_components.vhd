
library ieee;
use ieee.std_logic_1164.all;
package customTypes is

    type data_array is array(natural range <>) of std_logic_vector;

end package;

-----------------------------------------------------------------  andN
------------------------------------------------------------------------
-- size-generic AND gate used in the size-generic lazy fork and join
------------------------------------------------------------------------
LIBRARY IEEE;
USE ieee.std_logic_1164.all;
use work.customTypes.all;

ENTITY andN IS
GENERIC (n : INTEGER := 4);
PORT (  x : IN std_logic_vector(N-1 downto 0);
        res : OUT STD_LOGIC);
END andN;

ARCHITECTURE vanilla OF andn IS
    SIGNAL dummy : std_logic_vector(n-1 downto 0);
BEGIN
    dummy <= (OTHERS => '1');
    res <= '1' WHEN x = dummy ELSE '0';
END vanilla;

-----------------------------------------------------------------  nandN
------------------------------------------------------------------------
-- size-generic AND gate used in the size-generic lazy fork and join
------------------------------------------------------------------------
LIBRARY IEEE;
USE ieee.std_logic_1164.all;
use work.customTypes.all;

ENTITY nandN IS
GENERIC (n : INTEGER := 4);
PORT (  x : IN std_logic_vector(N-1 downto 0);
        res : OUT STD_LOGIC);
END nandN;

ARCHITECTURE arch OF nandn IS
    SIGNAL dummy : std_logic_vector(n-1 downto 0);
    SIGNAL andRes: STD_LOGIC;
BEGIN
    dummy <= (OTHERS => '1');
    andRes <= '1' WHEN x = dummy ELSE '0';
    res <= not andRes;
END arch;

-----------------------------------------------------------------  orN
------------------------------------------------------------------------
-- size-generic OR gate used in the size-generic eager fork and join
------------------------------------------------------------------------
LIBRARY IEEE;
USE ieee.std_logic_1164.all;
use work.customTypes.all;

ENTITY orN IS
GENERIC (n : INTEGER := 4);
PORT (  x : IN std_logic_vector(N-1 downto 0);
        res : OUT STD_LOGIC);
END orN;

ARCHITECTURE vanilla OF orN IS
    SIGNAL dummy : std_logic_vector(n-1 downto 0);
BEGIN
    dummy <= (OTHERS => '0');
    res <= '0' WHEN x = dummy ELSE '1';
END vanilla;

-----------------------------------------------------------------  norN
------------------------------------------------------------------------
-- size-generic OR gate used in the size-generic eager fork and join
------------------------------------------------------------------------
LIBRARY IEEE;
USE ieee.std_logic_1164.all;
use work.customTypes.all;

ENTITY norN IS
GENERIC (n : INTEGER := 4);
PORT (  x : IN std_logic_vector(N-1 downto 0);
        res : OUT STD_LOGIC);
END norN;

ARCHITECTURE arch OF norN IS
    SIGNAL dummy : std_logic_vector(n-1 downto 0);
    SIGNAL orRes: STD_LOGIC;
BEGIN
    dummy <= (OTHERS => '0');
    orRes <= '0' WHEN x = dummy ELSE '1';
    res <= not orRes;
END arch;

--------------------------------------------------------------  join
---------------------------------------------------------------------
--------------------------------------------------------------  join
---------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;

entity join is generic (SIZE : integer);
port (
    pValidArray     : in std_logic_vector(SIZE-1 downto 0);
    nReady          : in std_logic;
    valid           : out std_logic;
    readyArray      : out std_logic_vector(SIZE-1 downto 0));   
end join;

architecture arch of join is
signal allPValid : std_logic;
    
begin
    
    allPValidAndGate : entity work.andN generic map(SIZE)
            port map(   pValidArray,
                        allPValid);
    
    valid <= allPValid;
    
    process (pValidArray, nReady)
        variable  singlePValid : std_logic_vector(SIZE-1 downto 0);
        begin
        for i in 0 to SIZE-1 loop
            singlePValid(i) := '1';
            for j in 0 to SIZE-1 loop
                if (i /= j) then
                    singlePValid(i) := (singlePValid(i) and pValidArray(j));
                end if;
            end loop;
        end loop;
        for i in 0 to SIZE-1 loop
            readyArray(i) <=  (singlePValid(i) and nReady);
        end loop;
    end process;
          
end arch;


--------------------------------------------------------------  TEHB
---------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
USE work.customTypes.all;

entity TEHB is 
    generic(
        INPUTS        : integer;
        OUTPUTS        : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );
port (
        clk, rst      : in  std_logic;
        dataInArray   : in  data_array(INPUTS - 1 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        pValidArray   : in  std_logic_vector(INPUTS - 1 downto 0);
        nReadyArray   : in  std_logic_vector(0 downto 0);
        validArray    : out std_logic_vector(0 downto 0);
        readyArray    : out std_logic_vector(INPUTS - 1 downto 0));
end TEHB;

architecture arch of TEHB is
    signal full_reg, reg_en, mux_sel : std_logic;
    signal data_reg: std_logic_vector(DATA_SIZE_IN-1 downto 0);
begin
    
    process(clk, rst) is

          begin
           if (rst = '1') then
                full_reg <= '0';
              
            elsif (rising_edge(clk)) then
                full_reg <= validArray(0) and not nReadyArray(0);                    
            
            end if;
    end process; 

    process(clk, rst) is

          begin
           if (rst = '1') then
                data_reg <= (others => '0');
              
            elsif (rising_edge(clk)) then
                if (reg_en) then
                    data_reg<= dataInArray(0);  
                end if;                  
            
            end if;
    end process;

    process (mux_sel, data_reg, dataInArray) is
        begin
            if (mux_sel = '1') then
                dataOutArray(0) <= data_reg;
            else
                dataOutArray(0) <= dataInArray(0);
            end if;


    end process;


    validArray(0) <= pValidArray(0) or full_reg;    
    readyArray(0) <= not full_reg;
    reg_en <= readyArray(0) and pValidArray(0) and not nReadyArray(0);
    mux_sel <= full_reg;


end arch;

--------------------------------------------------------------  OEHB
---------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
USE work.customTypes.all;

entity OEHB is 
    generic(
        INPUTS        : integer;
        OUTPUTS        : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );
port (
        clk, rst      : in  std_logic;
        dataInArray   : in  data_array(INPUTS - 1 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        pValidArray   : in  std_logic_vector(INPUTS - 1 downto 0);
        nReadyArray   : in  std_logic_vector(0 downto 0);
        validArray    : out std_logic_vector(0 downto 0);
        readyArray    : out std_logic_vector(INPUTS - 1 downto 0));
end OEHB;

architecture arch of OEHB is
    signal full_reg, reg_en, mux_sel : std_logic;
    signal data_reg: std_logic_vector(DATA_SIZE_IN-1 downto 0);
begin
    
    process(clk, rst) is

          begin
           if (rst = '1') then
                validArray(0) <= '0';
              
            elsif (rising_edge(clk)) then
                validArray(0) <=  pValidArray(0) or not readyArray(0);                   
            
            end if;
    end process; 

    process(clk, rst) is

          begin
           if (rst = '1') then
                data_reg <= (others => '0');
              
            elsif (rising_edge(clk)) then
                if (reg_en) then
                    data_reg<= dataInArray(0);  
                end if;                  
            
            end if;
    end process;


    readyArray(0) <= not validArray(0) or nReadyArray(0);
    reg_en <= readyArray(0) and pValidArray(0);
    dataOutArray(0) <= data_reg;

end arch;

--------------------------------------------------------------  EB
---------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
USE work.customTypes.all;

entity elasticBuffer is
Generic (
  INPUTS :integer; OUTPUTS :integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer 
);
port(
    clk, rst : in std_logic;  
    dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0);
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);
    ReadyArray : out std_logic_vector(0 downto 0);
    ValidArray : out std_logic_vector(0 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    pValidArray : in std_logic_vector(0 downto 0));
end elasticBuffer;
------------------------------------------------------------------------ 
-- elastic buffer 
------------------------------------------------------------------------ 
architecture arch of elasticBuffer is
    
    signal tehb1_valid, tehb1_ready : STD_LOGIC;
    signal oehb1_valid, oehb1_ready : STD_LOGIC;
    signal tehb1_dataOut, oehb1_dataOut : std_logic_vector(DATA_SIZE_IN-1 downto 0);

    
begin

tehb1: entity work.TEHB(arch) generic map (1, 1, DATA_SIZE_IN, DATA_SIZE_IN)
        port map (
        --inputspValidArray
            clk => clk, 
            rst => rst, 
            pValidArray(0)  => pValidArray(0), -- real or speculatef condition (determined by merge1)
            nReadyArray(0) => oehb1_ready,    
            validArray(0) => tehb1_valid, 
        --outputs
            readyArray(0) => tehb1_ready,   
            dataInArray(0) => dataInArray(0),
            dataOutArray(0) => tehb1_dataOut
        );

oehb1: entity work.OEHB(arch) generic map (1, 1, DATA_SIZE_IN, DATA_SIZE_IN)
        port map (
        --inputspValidArray
            clk => clk, 
            rst => rst, 
            pValidArray(0)  => tehb1_valid, -- real or speculatef condition (determined by merge1)
            nReadyArray(0) => nReadyArray(0),    
            validArray(0) => oehb1_valid, 
        --outputs
            readyArray(0) => oehb1_ready,   
            dataInArray(0) =>tehb1_dataOut,
            dataOutArray(0) => oehb1_dataOut
        );

dataOutArray(0) <= oehb1_dataOut;
ValidArray(0) <= oehb1_valid;
ReadyArray(0) <= tehb1_ready;
    
end arch;


--------------------------------------------------------------  end
---------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity end_node is 
    generic(
        INPUTS        : integer;
        MEM_INPUTS    : integer;
        OUTPUTS        : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );

port (
    clk, rst : in std_logic;  
    dataInArray : in data_array (INPUTS - 1  downto 0)(DATA_SIZE_IN-1 downto 0);
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);
    ReadyArray : out std_logic_vector(INPUTS - 1  downto 0);
    ValidArray : out std_logic_vector(0  downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    pValidArray : in std_logic_vector(INPUTS - 1 downto 0);
    eReadyArray : out std_logic_vector(MEM_INPUTS - 1 downto 0);
    eValidArray : in std_logic_vector(MEM_INPUTS - 1  downto 0) := (others => '1'));
end end_node;

architecture arch of end_node is
    signal allPValid : std_logic;
    signal nReady: STD_LOGIC;
    signal valid: std_logic;
    signal mem_valid: std_logic;
    signal joinValid:std_logic;
    signal joinReady   : std_logic_vector(1 downto 0);

begin
   
    -- process for the return data
    -- there may be multiple return points, check if any is valid and output its data
    process(pValidArray, dataInArray)
        variable tmp_data_out  : unsigned(DATA_SIZE_IN - 1 downto 0);
        variable tmp_valid_out : std_logic;

    begin
        tmp_data_out  := unsigned(dataInArray(0));
        tmp_valid_out := '0';
        for I in INPUTS - 1 downto 0 loop
            if (pValidArray(I) = '1') then
                tmp_data_out  := unsigned(dataInArray(I));
                tmp_valid_out := pValidArray(I);
            end if;
        end loop;
    dataOutArray(0)  <= std_logic_vector(resize(tmp_data_out, DATA_SIZE_OUT));
    valid <= tmp_valid_out;
    end process;

    -- check if all mem controllers are done (and of all valids from memory)
    mem_and: entity work.andN(vanilla) generic map (MEM_INPUTS)
            port map (eValidArray, mem_valid);

    -- join for return data and memory--we exit only in case the first process gets
    -- a single valid and if the AND of all memories is set
    j : entity work.join(arch) generic map(2)
            port map(   (valid, mem_valid),
                        nReadyArray(0),
                        joinValid,
                        joinReady);

   
    -- valid to successor (set by join)
    validArray(0) <= joinValid;

    -- join sends ready to predecessors
    -- not needed for eReady (because memory never reads it)
    process(joinReady)
    begin
        for I in 0 to INPUTS - 1 loop
            readyArray(I) <= joinReady(1);
        end loop;
    end process;

end arch;

-------------------------------------------------------------  branch
---------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;

entity branchSimple is port(
    condition,
    pValid : in std_logic;
    nReadyArray : in std_logic_vector(1 downto 0);  -- (branch1, branch0)
    validArray : out std_logic_vector(1 downto 0);
    ready : out std_logic);
end branchSimple;

---------------------------------------------------------------------
-- simple architecture
---------------------------------------------------------------------
architecture arch of branchSimple is
begin
    
    -- only one branch can announce ready, according to condition
    validArray(1) <= (not condition) and pValid;        
    validArray(0) <= condition and pValid;

    ready <= (nReadyArray(1) and not condition)
             or (nReadyArray(0) and condition);  

end arch;



library ieee;
use ieee.std_logic_1164.all;
USE work.customTypes.all;

entity branch is generic( INPUTS:integer; SIZE : integer; DATA_SIZE_IN : integer;DATA_SIZE_OUT : integer);
port(
    clk, rst : in std_logic;
    pValidArray         : in std_logic_vector(1 downto 0);
    condition: in data_array (0 downto 0)(0 downto 0);
    dataInArray          : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0);
    dataOutArray            : out data_array (SIZE-1 downto 0)(DATA_SIZE_OUT-1 downto 0);
    nReadyArray     : in std_logic_vector(1 downto 0);  -- (branch1, branch0)
    validArray      : out std_logic_vector(1 downto 0); -- (branch1, branch0)
    readyArray      : out std_logic_vector(1 downto 0));    -- (data, condition)
end branch;


architecture arch of branch is 
    signal joinValid, branchReady   : std_logic;
    --signal dataOut0, dataOut1 : std_logic_vector(31 downto 0);
begin

    j : entity work.join(arch) generic map(2)
            port map(   (pValidArray(1), pValidArray(0)),
                        branchReady,
                        joinValid,
                        readyArray);

    br : entity work.branchSimple(arch)
            port map(   condition(0)(0),
                        joinValid,
                        nReadyArray,
                        validArray,
                        branchReady);

    process(dataInArray)
    begin
        for I in 0 to SIZE - 1 loop
            dataOutArray(I) <= dataInArray(0);
        end loop;  
    end process; 

end arch;

-----------------------------------------------  eagerFork_RegisterBLock
------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;

entity eagerFork_RegisterBLock is
port(   clk, reset, 
        p_valid, n_stop, 
        p_valid_and_fork_stop : in std_logic;
        valid,  block_stop : out std_logic);
end eagerFork_RegisterBLock;

architecture arch of eagerFork_RegisterBLock is
    signal reg_value, reg_in, block_stop_internal : std_logic;
begin
    
    block_stop_internal <= n_stop and reg_value;
    
    block_stop <= block_stop_internal;
    
    reg_in <= block_stop_internal or (not p_valid_and_fork_stop);
    
    valid <= reg_value and p_valid; 
    
    reg : process(clk, reset, reg_in)
    begin
        if(reset='1') then
            reg_value <= '1'; --contains a "stop" signal - must be 1 at reset
        else
            if(rising_edge(clk))then
                reg_value <= reg_in;
            end if;
        end if;
    end process reg;
    
end arch;


-------------------------------------------------------------------  fork
------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity fork is generic( INPUTS: integer; SIZE : integer; DATA_SIZE_IN : Integer; DATA_SIZE_OUT : Integer);
port(   clk, rst    : in std_logic; -- the eager implementation uses registers
        dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0);
        pValidArray : in std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(0 downto 0);
        dataOutArray : out data_array (SIZE-1 downto 0)(DATA_SIZE_OUT-1 downto 0); 
        nReadyArray : in std_logic_vector(SIZE-1 downto 0);
        validArray  : out std_logic_vector(SIZE-1 downto 0)
        );
        
end fork;


------------------------------------------------------------------------
-- generic eager implementation
------------------------------------------------------------------------
architecture arch of fork is
-- wrapper signals (internals use "stop" signals instead of "ready" signals)
    signal forkStop : std_logic;
    signal nStopArray : std_logic_vector(SIZE-1 downto 0);
-- internal combinatorial signals
    signal blockStopArray : std_logic_vector(SIZE-1 downto 0);
    signal anyBlockStop : std_logic;
    signal pValidAndForkStop : std_logic;
begin
    
    --can't adapt the signals directly in port map
    wrapper : process(forkStop, nReadyArray)
    begin
        readyArray(0) <= not forkStop;
        for i in 0 to SIZE-1 loop
            nStopArray(i) <= not nReadyArray(i);
        end loop;
    end process;
    
    genericOr : entity work.orN generic map (SIZE)
        port map(blockStopArray, anyBlockStop);
        
    -- internal combinatorial signals
    forkStop <= anyBlockStop; 
    pValidAndForkStop <= pValidArray(0) and forkStop;
    
    --generate blocks
    generateBlocks : for i in SIZE-1 downto 0 generate
        regblock : entity work.eagerFork_RegisterBLock(arch)
                port map(   clk, rst,
                            pValidArray(0), nStopArray(i),
                            pValidAndForkStop,
                            validArray(i), blockStopArray(i));
    end generate;

    process(dataInArray)
    begin
        for I in 0 to SIZE - 1 loop
            dataOutArray(I) <= dataInArray(0);
        end loop;  
    end process;   

end arch;

-------------------------------------------------------------------  fork
------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity lazyfork is generic( INPUTS: integer; SIZE : integer; DATA_SIZE_IN : Integer; DATA_SIZE_OUT : Integer);
port(   clk, rst    : in std_logic; -- the eager implementation uses registers
        dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0);
        pValidArray : in std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(0 downto 0);
        dataOutArray : out data_array (SIZE-1 downto 0)(DATA_SIZE_OUT-1 downto 0); 
        nReadyArray : in std_logic_vector(SIZE-1 downto 0);
        validArray  : out std_logic_vector(SIZE-1 downto 0)
        );
        
end lazyfork;

architecture arch of lazyfork is
    signal allnReady : std_logic;
begin

genericAnd : entity work.andn generic map (SIZE)
    port map(nReadyArray, allnReady);
 
valids : process(pValidArray, nReadyArray, allnReady)
    begin
    for i in 0 to SIZE-1 loop
        validArray(i) <= pValidArray(0) and allnReady;
    end loop;
    end process;
 
readyArray(0) <= allnReady;

process(dataInArray)
    begin
        for I in 0 to SIZE - 1 loop
            dataOutArray(I) <= dataInArray(0);
        end loop;  
    end process;    

end arch;

--------------------------------------------------------------  merge
---------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity merge is

    generic(
        INPUTS        : integer;
        OUTPUTS        : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );
    port(
        clk, rst      : in  std_logic;
        dataInArray   : in  data_array(INPUTS - 1 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        pValidArray   : in  std_logic_vector(INPUTS - 1 downto 0);
        nReadyArray   : in  std_logic_vector(0 downto 0);
        validArray    : out std_logic_vector(0 downto 0);
        readyArray    : out std_logic_vector(INPUTS - 1 downto 0));
end merge;

architecture arch of merge is
signal tehb_data_in  : std_logic_vector(DATA_SIZE_IN - 1 downto 0);
signal tehb_pvalid : std_logic;
signal tehb_ready : std_logic;

begin

    process(pValidArray, dataInArray)
        variable tmp_data_out  : unsigned(DATA_SIZE_IN - 1 downto 0);
        variable tmp_valid_out : std_logic;
    begin
        tmp_data_out  := unsigned(dataInArray(0));
        tmp_valid_out := '0';
        for I in INPUTS - 1 downto 0 loop
            if (pValidArray(I) = '1') then
                tmp_data_out  := unsigned(dataInArray(I));
                tmp_valid_out := pValidArray(I);
            end if;
        end loop;

        tehb_data_in  <= std_logic_vector(resize(tmp_data_out, DATA_SIZE_OUT));
        tehb_pvalid <= tmp_valid_out;

    end process;

    process(tehb_ready)
    begin
        for I in 0 to INPUTS - 1 loop
            readyArray(I) <= tehb_ready;
        end loop;
    end process;

    tehb1: entity work.TEHB(arch) generic map (1, 1, DATA_SIZE_IN, DATA_SIZE_IN)
        port map (
        --inputspValidArray
            clk => clk, 
            rst => rst, 
            pValidArray(0)  => tehb_pvalid, 
            nReadyArray(0) => nReadyArray(0),    
            validArray(0) => validArray(0), 
        --outputs
            readyArray(0) => tehb_ready,   
            dataInArray(0) => tehb_data_in,
            dataOutArray(0) => dataOutArray(0)
        );
end arch;

library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity merge_notehb is

    generic(
        INPUTS        : integer;
        OUTPUTS        : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );
    port(
        clk, rst      : in  std_logic;
        dataInArray   : in  data_array(INPUTS - 1 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        pValidArray   : in  std_logic_vector(INPUTS - 1 downto 0);
        nReadyArray   : in  std_logic_vector(0 downto 0);
        validArray    : out std_logic_vector(0 downto 0);
        readyArray    : out std_logic_vector(INPUTS - 1 downto 0));
end merge_notehb;

architecture arch of merge_notehb is
signal tehb_data_in  : std_logic_vector(DATA_SIZE_IN - 1 downto 0);
signal tehb_pvalid : std_logic;
signal tehb_ready : std_logic;

begin

    process(pValidArray, dataInArray)
        variable tmp_data_out  : unsigned(DATA_SIZE_IN - 1 downto 0);
        variable tmp_valid_out : std_logic;
    begin
        tmp_data_out  := unsigned(dataInArray(0));
        tmp_valid_out := '0';
        for I in INPUTS - 1 downto 0 loop
            if (pValidArray(I) = '1') then
                tmp_data_out  := unsigned(dataInArray(I));
                tmp_valid_out := pValidArray(I);
            end if;
        end loop;

        tehb_data_in  <= std_logic_vector(resize(tmp_data_out, DATA_SIZE_OUT));
        tehb_pvalid <= tmp_valid_out;

    end process;

    process(tehb_ready)
    begin
        for I in 0 to INPUTS - 1 loop
            readyArray(I) <= tehb_ready;
        end loop;
    end process;

    tehb_ready <= nReadyArray(0);
    validArray(0) <= tehb_pvalid;
    dataOutArray(0) <= tehb_data_in;

end arch;

library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
 use work.customTypes.all;
entity start_node_new is

  Generic (
    INPUT_COUNT:integer; OUTPUT_COUNT:integer; DATA_SIZE_IN:integer; DATA_SIZE_OUT:integer
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
end start_node_new;



architecture arch of start_node_new is 

signal set: STD_LOGIC;
signal start_internal:std_logic;
signal startBuff_readyArray : STD_LOGIC_VECTOR(0 downto 0);
signal startBuff_validArray : STD_LOGIC_VECTOR(0 downto 0);
signal startBuff_dataOutArray: data_array(0 downto 0)(DATA_SIZE_IN-1 downto 0);

begin
 
  --process(clk, rst)
  ----  begin

  --      if (rst=  '1')  then
  --      start_internal <= '0';
  --          set <= '0';

   --     elsif rising_edge(clk) then
   --         if (pValidArray(0) = '1' and set = '0') then
   --             start_internal<= '1';
   --             set <= '1';
   --         else 
   --             start_internal <= '0';
   --         end if;
   --     end if;
        
      
   -- end process;

startBuff: entity work.elasticBuffer(arch) generic map (1, 1, DATA_SIZE_IN, DATA_SIZE_IN)
port map (
--inputs
    clk => clk,  --clk
    rst => rst,  --rst
    dataInArray(0) => dataInArray(0),  ----dataInArray
    pValidArray(0) => pValidArray(0),   --pValidArray
    nReadyArray(0) => nReadyArray(0),  --nReadyArray
--outputs
    dataOutArray => startBuff_dataOutArray,    ----dataOutArray
    readyArray => startBuff_readyArray,  --readyArray
    validArray => startBuff_validArray   --validArray
);

validArray(0) <= startBuff_validArray(0);
dataOutArray(0) <= startBuff_dataOutArray(0);
readyArray(0) <= startBuff_readyArray(0);

end arch;
--------------------------------------------------------------  new_start
---------------------------------------------------------------------

library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
 use work.customTypes.all;
entity start_node is

  Generic (
    INPUT_COUNT:integer; OUTPUT_COUNT:integer; DATA_SIZE_IN:integer; DATA_SIZE_OUT:integer
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
end start_node;



architecture arch of start_node is 

signal set: STD_LOGIC;
signal start_internal:std_logic;
signal startBuff_readyArray : STD_LOGIC_VECTOR(0 downto 0);
signal startBuff_validArray : STD_LOGIC_VECTOR(0 downto 0);
signal startBuff_dataOutArray: data_array(0 downto 0)(DATA_SIZE_IN-1 downto 0);

begin
 
  process(clk, rst)
    begin

        if (rst=  '1')  then
        start_internal <= '0';
            set <= '0';

        elsif rising_edge(clk) then
            if (pValidArray(0) = '1' and set = '0') then
                start_internal<= '1';
                set <= '1';
            else 
                start_internal <= '0';
            end if;
        end if;
        
      
    end process;

startBuff: entity work.elasticBuffer(arch) generic map (1, 1, DATA_SIZE_IN, DATA_SIZE_IN)
port map (
--inputs
    clk => clk,  --clk
    rst => rst,  --rst
    dataInArray(0) => dataInArray(0),  ----dataInArray
    pValidArray(0) => start_internal,   --pValidArray
    nReadyArray(0) => nReadyArray(0),  --nReadyArray
--outputs
    dataOutArray => startBuff_dataOutArray,    ----dataOutArray
    readyArray => startBuff_readyArray,  --readyArray
    validArray => startBuff_validArray   --validArray
);

validArray(0) <= startBuff_validArray(0);
dataOutArray(0) <= startBuff_dataOutArray(0);
readyArray(0) <= startBuff_readyArray(0);

end arch;

--------------------------------------------------------------  sink
---------------------------------------------------------------------
library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
 use work.customTypes.all;
entity sink is

  Generic (
    INPUT_COUNT:integer; OUTPUT_COUNT:integer; DATA_SIZE_IN:integer; DATA_SIZE_OUT:integer
  );
 
  Port ( 
    clk, rst : in std_logic;  
    dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0);
    readyArray : out std_logic_vector(0 downto 0);
    pValidArray : in std_logic_vector(0 downto 0)
  );
end sink;



architecture arch of sink is 

begin
 
readyArray(0) <= '1';

end arch;

--------------------------------------------------------------  source
----------------------------------------------------------------------
library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
 use work.customTypes.all;
entity source is

  Generic (
    INPUT_COUNT:integer; OUTPUT_COUNT:integer; DATA_SIZE_IN:integer; DATA_SIZE_OUT:integer
  );
 
  Port ( 
    clk, rst : in std_logic;  
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0)
  );
end source;



architecture arch of source is 

begin
 
validArray(0) <= '1';

end arch;


--------------------------------------------------------------  fifo
--------------------------------------------------------------------
library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
 use work.customTypes.all;
entity elasticFifoInner is

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
end elasticFifoInner;
 
architecture arch of elasticFifoInner is

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

    -- valid 
    process(clk)
        begin
           if (rst = '1' ) then
              fifo_valid <= '0';
            elsif (rising_edge(clk)) then
                if (ReadEn ='1')  then
                    fifo_valid <= '1';
                elsif (nReadyArray(0) = '1') then
                    fifo_valid <= '0';
                end if;
             
            end if;
    end process;

    fifo_proc : process (CLK)
   
     begin        
        if rising_edge(CLK) then
          if RST = '1' then
           
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
               Tail <= 0;
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
       Full <= '0';
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
-- process for updating full
EmptyUpdate_proc : process (CLK)
   
  begin
  if rising_edge(CLK) then
  
    if RST = '1' then
       Empty <= '1';
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



library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
USE work.customTypes.all;

entity nontranspFifo is

  Generic (
    INPUT_COUNT:integer; OUTPUT_COUNT:integer; DATA_SIZE_IN:integer; DATA_SIZE_OUT:integer; FIFO_DEPTH : integer
  );
 
port(
    clk, rst : in std_logic;  
    dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0);
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);
    ReadyArray : out std_logic_vector(0 downto 0);
    ValidArray : out std_logic_vector(0 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    pValidArray : in std_logic_vector(0 downto 0));
end nontranspFifo;

architecture arch of nontranspFifo is
    
    signal tehb_valid, tehb_ready : STD_LOGIC;
    signal fifo_valid, fifo_ready : STD_LOGIC;
    signal tehb_dataOut, fifo_dataOut : std_logic_vector(DATA_SIZE_IN-1 downto 0);
  
begin

tehb: entity work.TEHB(arch) generic map (1, 1, DATA_SIZE_IN, DATA_SIZE_IN)
        port map (
        --inputs
            clk => clk, 
            rst => rst, 
            pValidArray(0)  => pValidArray(0), 
            nReadyArray(0) => fifo_ready,    
            validArray(0) => tehb_valid, 
        --outputs
            readyArray(0) => tehb_ready,   
            dataInArray(0) => dataInArray(0),
            dataOutArray(0) => tehb_dataOut
        );

fifo: entity work.elasticFifoInner(arch) generic map (1, 1, DATA_SIZE_IN, DATA_SIZE_IN, FIFO_DEPTH)
        port map (
        --inputs
            clk => clk, 
            rst => rst, 
            pValidArray(0)  => tehb_valid, 
            nReadyArray(0) => nReadyArray(0),    
            validArray(0) => fifo_valid, 
        --outputs
            readyArray(0) => fifo_ready,   
            dataInArray(0) =>tehb_dataOut,
            dataOutArray(0) => fifo_dataOut
        );

dataOutArray(0) <= fifo_dataOut;
ValidArray(0) <= fifo_valid;
ReadyArray(0) <= tehb_ready;
    
end arch;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
USE work.customTypes.all;

entity transpFIFO is 
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
end transpFIFO;

architecture arch of transpFIFO is
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

    validArray(0) <= pValidArray(0) or fifo_valid;    
    readyArray(0) <= fifo_ready or nReadyArray(0);
    fifo_pvalid <= pValidArray(0) and (not nReadyArray(0) or fifo_valid);
    mux_sel <= fifo_valid;

    fifo_nready <= nReadyArray(0);
    fifo_in <= dataInArray(0);

    fifo: entity work.elasticFifoInner(arch) generic map (1, 1, DATA_SIZE_IN, DATA_SIZE_IN, FIFO_DEPTH)
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
--------------------------------------------------------------  read port
-------------------------------------------------------------------------
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;


entity load_op is generic( INPUTS : Integer;  OUTPUTS : Integer; ADDRESS_SIZE : Integer;  DATA_SIZE : Integer);
port (
    rst: in std_logic;
    clk: in std_logic;

    --- interface to previous
    pValidArray : in std_logic_vector(0 downto 0);--read requests
    readyArray : out std_logic_vector(0 downto 0); -- ready to process read
    dataInArray: in data_array (0 downto 0)(ADDRESS_SIZE -1 downto 0);

    ---interface to next
    nReadyArray : in std_logic_vector(0 downto 0); -- next component can accept data -- error if i put ARBITER_SIZE here??
    validArray : out std_logic_vector(0 downto 0); --sending data to next component
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE-1 downto 0); -- data to next components

    ---interface to memory
    read_enable: out std_logic;
    read_address: out std_logic_vector(ADDRESS_SIZE -1 downto 0);
    data_from_memory: in std_logic_vector(31 downto 0));

end entity;

architecture arch of load_op is 
signal temp, tempen:std_logic;
signal q0,q1 , enable_internal: std_logic;
signal read_address_internal: data_array(0 downto 0)(ADDRESS_SIZE -1 downto 0);
signal valid_temp: std_logic_vector(0 downto 0);
begin

        read_enable <= valid_temp(0) and nReadyArray(0);
        enable_internal <= valid_temp(0) and nReadyArray(0);
        dataOutArray(0) <= data_from_memory;


buff_n0: entity work.elasticBuffer(arch) generic map (1, 1, ADDRESS_SIZE, ADDRESS_SIZE)
        port map (
        --inputs
            clk => clk,  --clk
            rst => rst,    --reset
            dataInArray(0) => dataInArray(0),   --d_in
            pValidArray(0)=> pValidArray(0),  --p_valid
            nReadyArray(0) => nReadyArray(0), --n_ready
        --outputs
            dataOutArray => read_address_internal, --d_out
            readyArray => readyArray, --ready
            validArray  => valid_temp  --valid
        );

read_address <= read_address_internal(0);

    process(clk, rst) is

        begin

            if (rst = '1') then

                validArray(0)  <= '0';
                  
            elsif (rising_edge(clk)) then
                if (enable_internal= '1') then
                    validArray(0)   <= '1';
                            
                else
                    if (nReadyArray(0) = '1') then
                        validArray(0)  <= '0';
                    end if;
                end if;
                               
            end if;
    end process; 

end architecture;



library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
USE work.customTypes.all;

entity Const is
Generic (
  SIZE :integer; INPUTS :integer; DATA_SIZE_IN: integer;DATA_SIZE_OUT: integer 
);
port(
    clk, rst : in std_logic;  
    dataInArray : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0);
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);
    ReadyArray : out std_logic_vector(0 downto 0);
    ValidArray : out std_logic_vector(0 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    pValidArray : in std_logic_vector(0 downto 0));
end Const;

architecture arch of Const is
begin
dataOutArray <= dataInArray;
validArray <= pValidArray;
readyArray <= nReadyArray; 
end architecture;

---------------------------------------------------------------------
---------------------------------------------------------------------
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity write_memory_single_inside is generic( ADDRESS_SIZE : Integer;  DATA_SIZE : Integer);
port (
    clk: in std_logic;

    --- interface to previous
    dataValid: in std_logic; --write requests
    ready: out std_logic; -- ready
    -- need join for address and data! add somewhere
    input_addr: in std_logic_vector(ADDRESS_SIZE -1 downto 0);
    --addrValid: in std_logic;
    data : in std_logic_vector(DATA_SIZE-1 downto 0); -- data from previous that want to write

    ---interface to next
    nReady: in std_logic; -- next component can continue after write
    valid: out std_logic; --sending write confirmation to next component

    ---interface to memory
    write_enable: out std_logic;
    enable: out std_logic;
    write_address: out std_logic_vector(ADDRESS_SIZE -1 downto 0);
    data_to_memory: out std_logic_vector(DATA_SIZE-1 downto 0));

end entity;


architecture arch of write_memory_single_inside is

    begin

       process(clk) is

          begin
              
             if (rising_edge(clk)) then
              
                    write_address <= input_addr;
                              
                    data_to_memory <= data;
                    valid <= dataValid;
                    write_enable <= dataValid and nReady;
                    enable <= dataValid and nReady;
             end if;



    end process; 

ready <= nReady;

 end architecture;



 ---------------------------------------------------------------------
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;


entity store_op is generic(  INPUTS: integer; OUTPUTS: integer; ADDRESS_SIZE : Integer;  DATA_SIZE : Integer);
port (
    clk, rst: in std_logic;


    input_addr: in std_logic_vector(ADDRESS_SIZE -1 downto 0);
    dataInArray : in data_array (0 downto 0)(DATA_SIZE -1 downto 0);

    --- interface to previous
    pValidArray : IN std_logic_vector(1 downto 0);
    readyArray : OUT std_logic_vector(1 downto 0);

    ---interface to next
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE -1 downto 0);
    nReadyArray: in std_logic_vector(0 downto 0);
    validArray: out std_logic_vector(0 downto 0); 

    ---interface to memory
    write_enable: out std_logic;
    enable: out std_logic;
    write_address: out std_logic_vector(ADDRESS_SIZE -1 downto 0);
    data_to_memory: out std_logic_vector(DATA_SIZE -1 downto 0));

end entity;


architecture arch of store_op is
    signal single_ready: std_logic;
    signal join_valid: std_logic;
   
    begin

    join_write:   entity work.join(arch) generic map(2)
            port map(   pValidArray,  --pValidArray
                        single_ready,                  --nready                    
                        join_valid,                    --valid          
                        ReadyArray);   --readyarray 


    Write: entity work.write_memory_single_inside (arch) generic map (ADDRESS_SIZE, DATA_SIZE)
        port map( clk, 
              join_valid,    --pvalid
              single_ready,                  --ready
              input_addr,                       --addr0
              dataInArray(0),             --data0
              nReadyArray(0),  --nready
              validArray(0),                  --valid
              write_enable,                  --write enable
              enable,                  --enable
              write_address,             --write address
              data_to_memory);                  --data to memory


 end architecture;


----------------------------------------------------------------
-----------------------------MUX--------------------------------
library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

entity mux is
    generic(
        INPUTS        : integer;
        OUTPUTS       : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer;
        COND_SIZE     : integer
    );
    port(
        clk, rst      : in  std_logic;
        dataInArray   : in  data_array(INPUTS - 2 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        pValidArray   : in  std_logic_vector(INPUTS -1 downto 0);
        nReadyArray      : in  std_logic_vector(0 downto 0);
        validArray      : out std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(INPUTS -1 downto 0);
        condition     : in  data_array(0 downto 0)(COND_SIZE - 1 downto 0)   ----(integer(ceil(log2(real(INPUTS)))) - 1 downto 0);
        
    );
end mux;

architecture arch of mux is

signal tehb_data_in  : std_logic_vector(DATA_SIZE_IN - 1 downto 0);
signal tehb_pvalid : std_logic;
signal tehb_ready : std_logic;

begin
    process(dataInArray, pValidArray, nReadyArray, condition, tehb_ready)
        variable tmp_data_out  : unsigned(DATA_SIZE_IN - 1 downto 0);
        variable tmp_valid_out : std_logic;

        
    begin
        tmp_data_out  := unsigned(dataInArray(0));
        tmp_valid_out := '0';
        for I in INPUTS - 2 downto 0 loop
            -- if (the condition refers the Ith data input, condition is valid, and the Ith input is valid), assign input data to output and set the output valid high
            if (unsigned(condition(0)) = to_unsigned(I,condition(0)'length) and pValidArray(0) = '1' and pValidArray(I+1) = '1') then
                tmp_data_out  := unsigned(dataInArray(I));
                tmp_valid_out := '1';
            end if;
            -- set the readyOutArray
            if ((unsigned(condition(0)) = to_unsigned(I,condition(0)'length) and pValidArray(0) = '1' and tehb_ready = '1' and pValidArray(I+1) = '1') or pValidArray(I+1) = '0') then
                readyArray(I+1) <= '1';
            else
                readyArray(I+1) <= '0';
            end if;
        end loop;
        -- set the condtionReady
        if (pValidArray(0) = '0' or (tmp_valid_out = '1' and tehb_ready = '1')) then
            readyArray(0) <= '1';
        else
            readyArray(0) <= '0';
        end if;
        --Assign dataout and validout
        tehb_data_in <= std_logic_vector(resize(tmp_data_out,DATA_SIZE_OUT));
        tehb_pvalid     <= tmp_valid_out;
    end process;


    tehb1: entity work.TEHB(arch) generic map (1, 1, DATA_SIZE_IN, DATA_SIZE_IN)
        port map (
        --inputspValidArray
            clk => clk, 
            rst => rst, 
            pValidArray(0)  => tehb_pvalid, 
            nReadyArray(0) => nReadyArray(0),    
            validArray(0) => validArray(0), 
        --outputs
            readyArray(0) => tehb_ready,   
            dataInArray(0) => tehb_data_in,
            dataOutArray(0) => dataOutArray(0)
        );
end arch;


library ieee;
use ieee.std_logic_1164.all;
USE work.customTypes.all;
use ieee.numeric_std.all;

entity cntrlMerge is generic(
INPUTS : integer ; OUTPUTS : integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer; COND_SIZE:integer
);
port(
      clk, rst : in std_logic;    
        pValidArray : in std_logic_vector(INPUTS - 1 downto 0);
        nReadyArray : in std_logic_vector(1 downto 0);
        validArray : out std_logic_vector(1 downto 0);
        readyArray : out std_logic_vector(INPUTS - 1 downto 0);
        dataInArray   : in  data_array(INPUTS - 1 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        condition: out data_array(0 downto 0)(COND_SIZE - 1 downto 0));
end cntrlMerge;
architecture arch of cntrlMerge is

signal phi_C1_readyArray : STD_LOGIC_VECTOR (INPUTS - 1 downto 0);
signal phi_C1_validArray : STD_LOGIC_VECTOR (0 downto 0);
signal phi_C1_dataOutArray : data_array(0 downto 0)(COND_SIZE - 1 downto 0);

signal fork_C1_readyArray : STD_LOGIC_VECTOR (0 downto 0);
signal fork_C1_dataOutArray : data_array(1 downto 0)(0 downto 0);
signal fork_C1_validArray : STD_LOGIC_VECTOR (1 downto 0);

signal all_ones : STD_LOGIC_VECTOR (COND_SIZE - 1 downto 0) := (others => '1') ;
signal index, oehb1_dataOut : STD_LOGIC_VECTOR (COND_SIZE - 1 downto 0);
signal oehb1_valid, oehb1_ready: STD_LOGIC;

begin


readyArray <= phi_C1_readyArray;

phi_C1: entity work.merge_notehb(arch) generic map (INPUTS, 1, COND_SIZE, COND_SIZE)
port map (
--inputs
    clk => clk,  --clk
    rst => rst,  --rst
    pValidArray => pValidArray,    --pValidArray
    dataInArray => (INPUTS - 1 downto 0 => all_ones),
    nReadyArray(0) => oehb1_ready,--outputs
    dataOutArray => phi_C1_dataOutArray,
    readyArray => phi_C1_readyArray,    --readyArray
    validArray => phi_C1_validArray --validArray
);

process(pValidArray)
begin
    index <= (COND_SIZE - 1 downto 0 => '0');
    for i in 0 to (INPUTS - 1) loop
        if (pValidArray(i) = '1') then
            index <= std_logic_vector(to_unsigned(i,COND_SIZE));
            exit;
        end if;
    end loop;
end process;


oehb1: entity work.TEHB(arch) generic map (1, 1, COND_SIZE, COND_SIZE)
        port map (
        --inputspValidArray
            clk => clk, 
            rst => rst, 
            pValidArray(0)  => phi_C1_validArray(0), 
            nReadyArray(0) => fork_C1_readyArray(0),    
            validArray(0) => oehb1_valid, 
        --outputs
            readyArray(0) => oehb1_ready,   
            dataInArray(0) => index,
            dataOutArray(0) => oehb1_dataOut
        );

fork_C1: entity work.fork(arch) generic map (1, 2, 1, 1)
port map (
--inputs
    clk => clk,  --clk
    rst => rst,  --rst
    pValidArray(0) => oehb1_valid, --pValidArray
    dataInArray (0) => "1",
    nReadyArray => nReadyArray, --nReadyArray
--outputs
    dataOutArray => fork_C1_dataOutArray,
    readyArray => fork_C1_readyArray,   --readyArray
    validArray => fork_C1_validArray    --validArray
);


validArray <= fork_C1_validArray;
condition(0) <= oehb1_dataOut;

end arch;

--------------------------------------------------------------  read port
-------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;


entity lsq_load_op is generic( INPUTS : Integer;  OUTPUTS : Integer; ADDRESS_SIZE : Integer;  DATA_SIZE : Integer);
port (
    rst: in std_logic;
    clk: in std_logic;

    --- interface to previous
    pValidArray : in std_logic_vector(INPUTS - 1 downto 0);
    readyArray : out std_logic_vector(INPUTS - 1 downto 0);
    dataInArray: in data_array (0 downto 0)(DATA_SIZE -1 downto 0);
    input_addr: in std_logic_vector(ADDRESS_SIZE -1 downto 0);

    ---interface to next
    nReadyArray : in std_logic_vector(OUTPUTS - 1 downto 0); 
    validArray : out std_logic_vector(OUTPUTS - 1 downto 0);
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE-1 downto 0);
    output_addr: out std_logic_vector(ADDRESS_SIZE -1 downto 0)
    );

end entity;

architecture arch of lsq_load_op is 

begin

    output_addr <= input_addr; -- address request goes to LSQ
    validArray(1) <= pValidArray(1);
    readyArray(1) <= nReadyArray(1);


    dataOutArray(0) <= dataInArray(0); -- data from LSQ to load output
    validArray(0) <= pValidArray(0);
    readyArray(0) <= nReadyArray(0);
        
end architecture;


--------------------------------------------------------------  store port
-------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;


entity lsq_store_op is generic( INPUTS : Integer;  OUTPUTS : Integer; ADDRESS_SIZE : Integer;  DATA_SIZE : Integer);
port (
    rst: in std_logic;
    clk: in std_logic;

    input_addr: in std_logic_vector(ADDRESS_SIZE -1 downto 0);
    dataInArray : in data_array (0 downto 0)(DATA_SIZE -1 downto 0);

    --- interface to previous
    pValidArray : IN std_logic_vector(1 downto 0);
    readyArray : OUT std_logic_vector(1 downto 0);

    ---interface to next
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE -1 downto 0);
    output_addr: out std_logic_vector(ADDRESS_SIZE -1 downto 0);
    nReadyArray: in std_logic_vector(OUTPUTS-1 downto 0);
    validArray: out std_logic_vector(OUTPUTS-1 downto 0)
    );

end entity;

architecture arch of lsq_store_op is 

begin

    dataOutArray(0) <= dataInArray(0); -- data to LSQ
    validArray(0) <= pValidArray(0);
    readyArray(0) <= nReadyArray(0);

    output_addr <= input_addr; -- address to LSQ
    validArray(1) <= pValidArray(1);
    readyArray(1) <= nReadyArray(1);
        
end architecture;



