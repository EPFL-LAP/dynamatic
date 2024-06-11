-- custom types are declared here 
library ieee;
use ieee.std_logic_1164.all;
package customTypes is
    
    type data_array is array(natural range <>) of std_logic_vector;

end package;

-----------------------------------------------------------------------------------------
------------------------------------------------------------------------ Logic components
-----------------------------------------------------------------------------------------

------------------------------------------------------------------------
-- size-generic AND gate used in the size-generic lazy fork and join

library ieee;
use ieee.std_logic_1164.all;
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

------------------------------------------------------------------------
-- size-generic NAND gate used in the size-generic lazy fork and join

library ieee;
use ieee.std_logic_1164.all;
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

------------------------------------------------------------------------
-- size-generic OR gate used in the size-generic eager fork and join

library ieee;
use ieee.std_logic_1164.all;
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

------------------------------------------------------------------------
-- size-generic NOR gate used in the size-generic eager fork and join

library ieee;
use ieee.std_logic_1164.all;
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

-----------------------------------------------------------------------------------------
------------------------------------------------------------------------ Basic components
-----------------------------------------------------------------------------------------

------------------------------------------------------------------------
-- Simple join

library ieee;
use ieee.std_logic_1164.all;

entity join is generic (SIZE : integer);
    port (
        pValidArray     : in  std_logic_vector(SIZE - 1 downto 0);
        nReady          : in  std_logic;
        valid           : out std_logic;
        readyArray      : out std_logic_vector(SIZE - 1 downto 0)
    );   
    end join;

architecture arch of join is
    signal allPValid : std_logic;
begin

    allPValidAndGate : entity work.andN generic map(SIZE)
            port map(   pValidArray,
                        allPValid);
    
    valid <= allPValid;
    
    process (pValidArray, nReady)
        variable  singlePValid : std_logic_vector(SIZE - 1 downto 0);
    begin
        for i in 0 to SIZE - 1 loop
            singlePValid(i) := '1';
            for j in 0 to SIZE - 1 loop
                if (i /= j) then
                    singlePValid(i) := (singlePValid(i) and pValidArray(j));
                end if;
            end loop;
        end loop;
        for i in 0 to SIZE - 1 loop
            readyArray(i) <= (singlePValid(i) and nReady);
        end loop;
    end process;
    
end arch;



-----------------------------------------------------------------------------------------
----------------------------------------------------------------------- Buffer components
-----------------------------------------------------------------------------------------

------------------------------------------------------------------------
-- TEHB ----> SPEC version

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity TEHB is 
    generic(
        INPUTS        : integer;  -- assumed always 1
        OUTPUTS       : integer;  -- assumed always 1
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );
    port (
        clk, rst     : in  std_logic;
        dataInArray  : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        specInArray  : in  data_array(0 downto 0)(0 downto 0);
        specOutArray : out data_array(0 downto 0)(0 downto 0);
        pValidArray  : in  std_logic_vector(0 downto 0);
        nReadyArray  : in  std_logic_vector(0 downto 0);
        validArray   : out std_logic_vector(0 downto 0);
        readyArray   : out std_logic_vector(0 downto 0)
    );
end TEHB;

architecture arch of TEHB is
    signal full_reg, reg_en, mux_sel : std_logic;
    signal data_reg : std_logic_vector(DATA_SIZE_IN+1 -1 downto 0);

    signal specdataInArray  : data_array(0 downto 0)(DATA_SIZE_IN+1 - 1 downto 0);
    signal specdataOutArray : data_array(0 downto 0)(DATA_SIZE_OUT+1 - 1 downto 0);
begin

    -- Speculative bit logic
    specdataInArray(0) <= specInArray(0) & dataInArray(0);
    -----
    
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
                data_reg<= specdataInArray(0);
            end if;
        end if;
    end process;

    process (mux_sel, data_reg, specdataInArray) is
        begin
            if (mux_sel = '1') then
                specdataOutArray(0) <= data_reg;
            else
                specdataOutArray(0) <= specdataInArray(0);
            end if;
    end process;

    validArray(0) <= pValidArray(0) or full_reg;    
    readyArray(0) <= not full_reg;
    reg_en  <= readyArray(0) and pValidArray(0) and not nReadyArray(0);
    mux_sel <= full_reg;

    -- Speculative bit logic
    dataOutArray(0)    <= specdataOutArray(0)(DATA_SIZE_OUT - 1 downto 0);
    specOutArray(0)(0) <= specdataOutArray(0)(DATA_SIZE_OUT+1 - 1);

end arch;

------------------------------------------------------------------------
-- OEHB ----> SPEC version

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity OEHB is 
    generic(
        INPUTS        : integer;  -- assumed always 1
        OUTPUTS       : integer;  -- assumed always 1
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );
    port (
        clk, rst     : in  std_logic;
        dataInArray  : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        specInArray  : in  data_array(0 downto 0)(0 downto 0);
        specOutArray : out data_array(0 downto 0)(0 downto 0);
        pValidArray  : in  std_logic_vector(0 downto 0);
        nReadyArray  : in  std_logic_vector(0 downto 0);
        validArray   : out std_logic_vector(0 downto 0);
        readyArray   : out std_logic_vector(0 downto 0)
    );
end OEHB;

architecture arch of OEHB is
    signal full_reg, reg_en, mux_sel : std_logic;
    signal data_reg : std_logic_vector(DATA_SIZE_IN+1 -1 downto 0);

    signal specdataInArray  : data_array(0 downto 0)(DATA_SIZE_IN+1 - 1 downto 0);
    signal specdataOutArray : data_array(0 downto 0)(DATA_SIZE_OUT+1 - 1 downto 0);
begin

    -- Speculative bit logic
    specdataInArray(0) <= specInArray(0) & dataInArray(0);
    -----

    process(clk, rst) is
    begin
        if (rst = '1') then
            validArray(0) <= '0';
        elsif (rising_edge(clk)) then
            validArray(0) <= pValidArray(0) or not readyArray(0);
        end if;
    end process;

    process(clk, rst) is
    begin
        if (rst = '1') then
            data_reg <= (others => '0');
        elsif (rising_edge(clk)) then
            if (reg_en) then
                data_reg <= specdataInArray(0);
            end if;
        end if;
    end process;

    readyArray(0) <= not validArray(0) or nReadyArray(0);
    reg_en <= readyArray(0) and pValidArray(0);
    specdataOutArray(0) <= data_reg;

    -- Speculative bit logic
    dataOutArray(0)    <= specdataOutArray(0)(DATA_SIZE_OUT - 1 downto 0);
    specOutArray(0)(0) <= specdataOutArray(0)(DATA_SIZE_OUT+1 - 1);

end arch;

------------------------------------------------------------------------
-- elasticBuffer ----> SPEC version

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity elasticBuffer is 
    generic(
        INPUTS        : integer;  -- assumed always 1
        OUTPUTS       : integer;  -- assumed always 1
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );
    port (
        clk, rst     : in  std_logic;
        dataInArray  : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        specInArray  : in  data_array(0 downto 0)(0 downto 0);
        specOutArray : out data_array(0 downto 0)(0 downto 0);
        pValidArray  : in  std_logic_vector(0 downto 0);
        nReadyArray  : in  std_logic_vector(0 downto 0);
        validArray   : out std_logic_vector(0 downto 0);
        readyArray   : out std_logic_vector(0 downto 0)
    );
end elasticBuffer;

architecture arch of elasticBuffer is    
    signal tehb1_valid, tehb1_ready : std_logic;
    signal oehb1_valid, oehb1_ready : std_logic;
    signal tehb1_dataOut, oehb1_dataOut : std_logic_vector(DATA_SIZE_IN+1 -1 downto 0);

    signal specdataInArray  : data_array(0 downto 0)(DATA_SIZE_IN+1 - 1 downto 0);
    signal specdataOutArray : data_array(0 downto 0)(DATA_SIZE_OUT+1 - 1 downto 0);

    signal unconnected_spec : data_array(0 downto 0)(0 downto 0);
begin

    -- Speculative bit logic
    specdataInArray(0) <= specInArray(0) & dataInArray(0);
    -----

    tehb1: entity work.TEHB(arch) generic map (1, 1, DATA_SIZE_IN+1, DATA_SIZE_IN+1)
            port map (
                clk => clk,
                rst => rst,
                dataInArray(0)      => specdataInArray(0),
                specInArray(0)(0)   => '0',
                pValidArray(0)      => pValidArray(0),
                readyArray(0)       => tehb1_ready,
                nReadyArray(0)      => oehb1_ready,
                validArray(0)       => tehb1_valid,
                dataOutArray(0)     => tehb1_dataOut,
                specOutArray        => unconnected_spec
            );

    oehb1: entity work.OEHB(arch) generic map (1, 1, DATA_SIZE_IN+1, DATA_SIZE_IN+1)
            port map (
                clk => clk, 
                rst => rst, 
                dataInArray(0)      => tehb1_dataOut,
                specInArray(0)(0)   => '0',
                pValidArray(0)      => tehb1_valid,
                readyArray(0)       => oehb1_ready,
                nReadyArray(0)      => nReadyArray(0),
                validArray(0)       => oehb1_valid,
                dataOutArray(0)     => oehb1_dataOut,
                specOutArray        => unconnected_spec
            );

    specdataOutArray(0) <= oehb1_dataOut;
    ValidArray(0) <= oehb1_valid;
    ReadyArray(0) <= tehb1_ready;

    -- Speculative bit logic
    dataOutArray(0)    <= specdataOutArray(0)(DATA_SIZE_OUT - 1 downto 0);
    specOutArray(0)(0) <= specdataOutArray(0)(DATA_SIZE_OUT+1 - 1);

end arch;

------------------------------------------------------------------------
-- elasticFifoInner

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity elasticFifoInner is
    generic (
        INPUTS        : integer; 
        OUTPUTS       : integer; 
        DATA_SIZE_IN  : integer; 
        DATA_SIZE_OUT : integer; 
        FIFO_DEPTH    : integer
    );
    port (
        clk, rst      : in  std_logic;
        dataInArray   : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        nReadyArray   : in  std_logic_vector(0 downto 0);
        validArray    : out std_logic_vector(0 downto 0);
        pValidArray   : in  std_logic_vector(0 downto 0);
        readyArray    : out std_logic_vector(0 downto 0)
    );
end elasticFifoInner;

architecture arch of elasticFifoInner is

    signal Tail     : natural range 0 to FIFO_DEPTH - 1;
    signal Head     : natural range 0 to FIFO_DEPTH - 1;
    signal ReadEn   : std_logic := '0';
    signal WriteEn  : std_logic := '0';
    signal Empty    : std_logic;
    signal Full     : std_logic;
    signal fifo_ready : std_logic;
    signal fifo_valid : std_logic;
    type FIFO_Memory is array (0 to FIFO_DEPTH - 1) of std_logic_vector(DATA_SIZE_IN - 1 downto 0);
    signal Memory : FIFO_Memory;

begin

    -- Ready if there is space in the FIFO or bypass
    fifo_ready    <= not Full or nReadyArray(0);
    readyArray(0) <= fifo_ready;

    -- Valid read if FIFO is not empty
    validArray(0) <= not Empty;
    dataOutArray(0) <= Memory(Head);

    -- Read if next can accept and there is sth in FIFO
    ReadEn  <= nReadyArray(0) and not Empty;
    -- Write if there is valid input and FIFO is ready
    WriteEn <= pValidArray(0) and fifo_ready;

    -------------------------------------------
    -- valid process
    val_proc : process (clk) begin
            if rst = '1' then
                fifo_valid <= '0';
            elsif (rising_edge(clk)) then
                if (ReadEn = '1')  then
                    fifo_valid <= '1';
                elsif (nReadyArray(0) = '1') then
                    fifo_valid <= '0';
                end if;
            end if;
    end process;

    -------------------------------------------
    -- write process
    fifo_proc : process (clk) begin
        if rising_edge(clk) then
            if rst = '1' then

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
    TailUpdate_proc : process (clk) begin
        if rising_edge(clk) then
            if rst = '1' then
                Tail <= 0;
            else 
                if (WriteEn = '1') then
                    -- Add writeCount instead of 1
                    Tail <= (Tail + 1) mod FIFO_DEPTH;
                end if;
            end if;
        end if;
    end process;

    -------------------------------------------
    -- process for updating head
    HeadUpdate_proc : process (clk) begin
        if rising_edge(clk) then
            if rst = '1' then
                Head <= 0;
            else
                if (ReadEn = '1') then
                    Head <= (Head + 1) mod FIFO_DEPTH;
                end if;
            end if;
        end if;
    end process;

    -------------------------------------------
    -- process for updating Full
    FullUpdate_proc : process (clk) begin
        if rising_edge(clk) then
            if rst = '1' then
                Full <= '0';
            else
                -- if only filling but not emptying
                if (WriteEn = '1') and (ReadEn = '0') then
                    -- if new tail index will reach head index
                    if ((Tail + 1) mod FIFO_DEPTH = Head) then
                        Full <= '1';
                    end if;
                -- if only emptying but not filling
                elsif (WriteEn = '0') and (ReadEn = '1') then
                    Full <= '0';
                -- otherwise, nothing is happenning or simultaneous read and write
                end if;
            end if;
        end if;
    end process;

    -------------------------------------------
    -- process for updating Empty
    EmptyUpdate_proc : process (clk) begin
        if rising_edge(clk) then
            if rst = '1' then
                Empty <= '1';
            else
                -- if only emptying but not filling
                if (WriteEn = '0') and (ReadEn = '1') then
                    -- if new head index will reach tail index
                    if ((Head + 1) mod FIFO_DEPTH = Tail) then
                        Empty <= '1';
                    end if;
                -- if only filling but not emptying
                elsif (WriteEn = '1') and (ReadEn = '0') then
                    Empty <= '0';

                -- otherwise, nothing is happenning or simultaneous read and write
                end if;       
            end if;
        end if;
    end process;

end architecture;

------------------------------------------------------------------------
-- nontranspFifo ----> SPEC version

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity nontranspFifo is
    generic (
        INPUTS        : integer; 
        OUTPUTS       : integer; 
        DATA_SIZE_IN  : integer; 
        DATA_SIZE_OUT : integer; 
        FIFO_DEPTH    : integer
    );
    port (
        clk, rst      : in  std_logic;
        dataInArray   : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        specInArray   : in  data_array(0 downto 0)(0 downto 0);
        specOutArray  : out data_array(0 downto 0)(0 downto 0);
        nReadyArray   : in  std_logic_vector(0 downto 0);
        validArray    : out std_logic_vector(0 downto 0);
        pValidArray   : in  std_logic_vector(0 downto 0);
        readyArray    : out std_logic_vector(0 downto 0)
    );
end nontranspFifo;

architecture arch of nontranspFifo is
    
    signal tehb_valid, tehb_ready : std_logic;
    signal fifo_valid, fifo_ready : std_logic;
    signal tehb_dataOut, fifo_dataOut : std_logic_vector(DATA_SIZE_IN+1 - 1 downto 0);
  
    signal specdataInArray  : data_array(0 downto 0)(DATA_SIZE_IN+1 - 1 downto 0);
    signal specdataOutArray : data_array(0 downto 0)(DATA_SIZE_OUT+1 - 1 downto 0);
    signal unconnected_spec : data_array(0 downto 0)(0 downto 0);

begin

    -- Speculative bit logic
    specdataInArray(0) <= specInArray(0) & dataInArray(0);
    -----

    tehb: entity work.TEHB(arch) generic map (1, 1, DATA_SIZE_IN+1, DATA_SIZE_IN+1)
        port map (
            clk                 => clk,
            rst                 => rst,
            dataInArray(0)      => specdataInArray(0),
            specInArray(0)(0)   => '0',
            pValidArray(0)      => pValidArray(0),
            readyArray(0)       => tehb_ready,
            nReadyArray(0)      => fifo_ready,
            validArray(0)       => tehb_valid,
            dataOutArray(0)     => tehb_dataOut,
            specOutArray        => unconnected_spec
        );

    fifo: entity work.elasticFifoInner(arch) generic map (1, 1, DATA_SIZE_IN+1, DATA_SIZE_IN+1, FIFO_DEPTH)
        port map (
            clk             => clk,
            rst             => rst,
            dataInArray(0)  => tehb_dataOut,            
            pValidArray(0)  => tehb_valid,
            readyArray(0)   => fifo_ready,
            nReadyArray(0)  => nReadyArray(0),
            validArray(0)   => fifo_valid,
            dataOutArray(0) => fifo_dataOut
        );

    specdataOutArray(0) <= fifo_dataOut;
    ValidArray(0) <= fifo_valid;
    ReadyArray(0) <= tehb_ready;

    -- Speculative bit logic
    dataOutArray(0)    <= specdataOutArray(0)(DATA_SIZE_OUT - 1 downto 0);
    specOutArray(0)(0) <= specdataOutArray(0)(DATA_SIZE_OUT+1 - 1);
    
end arch;

------------------------------------------------------------------------
-- transpFifo ----> SPEC version

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity transpFIFO is 
    generic(
        INPUTS        : integer; -- assumes always 1
        OUTPUTS       : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer;
        FIFO_DEPTH    : integer
    );
    port (
        clk, rst     : in  std_logic;
        dataInArray  : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        specInArray  : in  data_array(0 downto 0)(0 downto 0);
        specOutArray : out data_array(0 downto 0)(0 downto 0);
        pValidArray  : in  std_logic_vector(0 downto 0);
        nReadyArray  : in  std_logic_vector(0 downto 0);
        validArray   : out std_logic_vector(0 downto 0);
        readyArray   : out std_logic_vector(0 downto 0)
    );
end transpFIFO;

architecture arch of transpFIFO is

    signal mux_sel : std_logic;
    signal fifo_valid, fifo_ready : std_logic;
    signal fifo_pvalid, fifo_nready : std_logic;
    signal fifo_in, fifo_out: std_logic_vector(DATA_SIZE_IN+1 - 1 downto 0);

    signal specdataInArray : data_array(0 downto 0)(DATA_SIZE_IN+1 - 1 downto 0);
    signal specdataOutArray : data_array(0 downto 0)(DATA_SIZE_OUT+1 - 1 downto 0);

begin

    -- Speculative bit logic
    specdataInArray(0) <= specInArray(0) & dataInArray(0);
    -----

    process (mux_sel, fifo_out, specdataInArray) is
    begin
        if (mux_sel = '1') then
            specdataOutArray(0) <= fifo_out;
        else
            specdataOutArray(0) <= specdataInArray(0);
        end if;
    end process;

    validArray(0) <= pValidArray(0) or fifo_valid;    
    readyArray(0) <= fifo_ready or nReadyArray(0);
    fifo_pvalid <= pValidArray(0) and (not nReadyArray(0) or fifo_valid);
    mux_sel <= fifo_valid;

    fifo_nready <= nReadyArray(0);
    fifo_in <= specdataInArray(0);

    fifo: entity work.elasticFifoInner(arch) generic map (1, 1, DATA_SIZE_IN+1, DATA_SIZE_IN+1, FIFO_DEPTH)
        port map (
            clk             => clk,
            rst             => rst,
            dataInArray(0)  => fifo_in,
            pValidArray(0)  => fifo_pvalid,
            readyArray(0)   => fifo_ready,
            nReadyArray(0)  => fifo_nready,
            validArray(0)   => fifo_valid,
            dataOutArray(0) => fifo_out
        );

    -- Speculative bit logic
    dataOutArray(0)    <= specdataOutArray(0)(DATA_SIZE_OUT - 1 downto 0);
    specOutArray(0)(0) <= specdataOutArray(0)(DATA_SIZE_OUT+1 - 1);

end arch;


------------------------------------------------------------------------
-- Start node

library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity start_node is
    generic (
        INPUTS        : integer;
        OUTPUTS       : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );
    port ( 
        clk, rst     : in  std_logic;  
        dataInArray  : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        readyArray   : out std_logic_vector(0 downto 0);
        validArray   : out std_logic_vector(0 downto 0);
        nReadyArray  : in  std_logic_vector(0 downto 0);
        pValidArray  : in  std_logic_vector(0 downto 0)
    );
end start_node;

architecture arch of start_node is

    signal set : std_logic;
    signal start_internal : std_logic;
    signal startBuff_readyArray : std_logic_vector(0 downto 0);
    signal startBuff_validArray : std_logic_vector(0 downto 0);
    signal startBuff_dataOutArray : data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
    signal unconnected_spec : data_array(0 downto 0)(0 downto 0);

begin
 
    process(clk, rst) begin
        
        if (rst = '1')  then
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
        clk                 => clk,
        rst                 => rst,
        dataInArray(0)      => dataInArray(0),
        specInArray(0)(0)   => '0',
        pValidArray(0)      => start_internal,
        nReadyArray(0)      => nReadyArray(0),
        --outputs
        dataOutArray        => startBuff_dataOutArray,
        specOutArray        => unconnected_spec,
        readyArray          => startBuff_readyArray,
        validArray          => startBuff_validArray
    );

    validArray(0)   <= startBuff_validArray(0);
    dataOutArray(0) <= startBuff_dataOutArray(0);
    readyArray(0)   <= startBuff_readyArray(0);

end arch;

------------------------------------------------------------------------
-- End node

library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity end_node is 
    generic(
        INPUTS        : integer;
        MEM_INPUTS    : integer;
        OUTPUTS       : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );

port (
        clk, rst     : in  std_logic;  
        dataInArray  : in  data_array(INPUTS - 1 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        ReadyArray   : out std_logic_vector(INPUTS - 1 downto 0);
        ValidArray   : out std_logic_vector(0  downto 0);
        nReadyArray  : in  std_logic_vector(0 downto 0);
        pValidArray  : in  std_logic_vector(INPUTS - 1 downto 0);
        eReadyArray  : out std_logic_vector(MEM_INPUTS - 1 downto 0);
        eValidArray  : in  std_logic_vector(MEM_INPUTS - 1 downto 0) := (others => '1')
    );
end end_node;

architecture arch of end_node is
    signal allPValid : std_logic;
    signal nReady : std_logic;
    signal valid : std_logic;
    signal mem_valid : std_logic;
    signal joinValid : std_logic;
    signal joinReady : std_logic_vector(1 downto 0);
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

------------------------------------------------------------------------
-- Sink

library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity sink is
    generic (
        INPUTS        : integer;
        OUTPUTS       : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );
    port ( 
        clk, rst     : in  std_logic;  
        dataInArray  : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        readyArray   : out std_logic_vector(0 downto 0);
        pValidArray  : in  std_logic_vector(0 downto 0)
    );
end sink;

architecture arch of sink is
begin

    readyArray(0) <= '1';

end arch;

------------------------------------------------------------------------
-- Source

library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity source is
    generic (
        INPUTS        : integer;
        OUTPUTS       : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );
    port ( 
        clk, rst     : in  std_logic;  
        dataOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        validArray   : out std_logic_vector(0 downto 0);
        nReadyArray  : in  std_logic_vector(0 downto 0)
    );
end source;

architecture arch of source is 
begin

    validArray(0) <= '1';

end arch;

------------------------------------------------------------------------
-- Constant ----> SPEC version

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity Const is 
    generic(
        INPUTS        : integer;
        OUTPUTS       : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );

port (
        clk, rst     : in  std_logic;  
        dataInArray  : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        specInArray  : in  data_array(0 downto 0)(0 downto 0);
        specOutArray : out data_array(0 downto 0)(0 downto 0);
        ReadyArray   : out std_logic_vector(0 downto 0);
        ValidArray   : out std_logic_vector(0 downto 0);
        nReadyArray  : in  std_logic_vector(0 downto 0);
        pValidArray  : in  std_logic_vector(0 downto 0)
    );
end Const;

architecture arch of Const is
begin

    dataOutArray <= dataInArray;
    validArray <= pValidArray;
    readyArray <= nReadyArray;

    -- Speculative bit logic
    specOutArray(0) <= specInArray(0);

end architecture;


-----------------------------------------------------------------------------------------
----------------------------------------------------------------- Flow control components
-----------------------------------------------------------------------------------------

------------------------------------------------------------------------
-- Branch ----> SPEC version

-- simple architecture
------------------------
library ieee;
use ieee.std_logic_1164.all;

entity branchSimple is port(
    condition,
    pValid : in std_logic;
    nReadyArray : in std_logic_vector(1 downto 0);  -- (branch1, branch0)
    validArray : out std_logic_vector(1 downto 0);
    ready : out std_logic);
end branchSimple;

architecture arch of branchSimple is
begin   
    -- only one branch can announce ready, according to condition
    validArray(1) <= (not condition) and pValid;        
    validArray(0) <= condition and pValid;

    ready <= (nReadyArray(1) and not condition)
             or (nReadyArray(0) and condition);  

end arch;

-- complete architecture 
------------------------
library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity branch is 
    generic(
        INPUTS        : integer;  -- assumed always 2
        OUTPUTS       : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );
    port (
        clk, rst     : in  std_logic;
        condition    : in  data_array(0 downto 0)(0 downto 0);
        dataInArray  : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        specInArray  : in  data_array(1 downto 0)(0 downto 0);
        dataOutArray : out data_array(OUTPUTS - 1 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        specOutArray : out data_array(OUTPUTS - 1 downto 0)(0 downto 0);
        nReadyArray  : in  std_logic_vector(1 downto 0);  -- (branch1, branch0)
        validArray   : out std_logic_vector(1 downto 0);  -- (branch1, branch0)
        pValidArray  : in  std_logic_vector(1 downto 0);  -- (cond, data)
        readyArray   : out std_logic_vector(1 downto 0)   -- (cond, data)
    );
end branch;

architecture arch of branch is

    signal joinValid, branchReady : std_logic;

    signal spec_bit_0 : std_logic;
    signal spec_bit_1 : std_logic;
    signal spec_bit : std_logic_vector(0 downto 0);
    signal specdataInArray  : data_array(0 downto 0)(DATA_SIZE_IN+1 - 1 downto 0);
    signal specdataOutArray : data_array(OUTPUTS - 1 downto 0)(DATA_SIZE_OUT+1 - 1 downto 0);

begin

    -- Speculative bit logic
    spec_bit_0 <= specInArray(0)(0);
    spec_bit_1 <= specInArray(1)(0);
    spec_bit(0) <= spec_bit_0 or spec_bit_1;
    specdataInArray(0) <= spec_bit & dataInArray(0);
    -----

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

    -- Speculative bit logic
    process(specdataInArray) begin
        for I in 0 to OUTPUTS - 1 loop
            dataOutArray(I)    <= specdataInArray(0)(DATA_SIZE_IN - 1 downto 0);
            specOutArray(I)(0) <= specdataInArray(0)(DATA_SIZE_IN+1 - 1);
        end loop;
    end process;
    -----

end arch;

------------------------------------------------------------------------
-- eagerFork_RegisterBlock

library ieee;
use ieee.std_logic_1164.all;

entity eagerFork_RegisterBlock is
port(   clk, reset, 
        p_valid, n_stop, 
        p_valid_and_fork_stop : in std_logic;
        valid,  block_stop : out std_logic);
end eagerFork_RegisterBlock;

architecture arch of eagerFork_RegisterBlock is
    signal reg_value, reg_in, block_stop_internal : std_logic;
begin
    
    block_stop_internal <= n_stop and reg_value;
    
    block_stop <= block_stop_internal;
    
    reg_in <= block_stop_internal or (not p_valid_and_fork_stop);
    
    valid <= reg_value and p_valid;
    
    reg : process(clk, reset, reg_in) begin
        if (reset = '1') then
            reg_value <= '1'; --contains a "stop" signal - must be 1 at reset
        else
            if (rising_edge(clk)) then
                reg_value <= reg_in;
            end if;
        end if;
    end process;
end arch;

------------------------------------------------------------------------
-- LazyFork ----> SPEC version

-- Do not use. For speculator only

library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity forkNlazy is 
    generic ( 
        INPUTS        : integer;
        OUTPUTS       : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );
    port ( 
        clk, rst     : in  std_logic;
        dataInArray  : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        specInArray  : in  data_array(0 downto 0)(0 downto 0);
        pValidArray  : in  std_logic_vector(0 downto 0);
        readyArray   : out std_logic_vector(0 downto 0);
        dataOutArray : out data_array(OUTPUTS - 1 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        specOutArray : out data_array(OUTPUTS - 1 downto 0)(0 downto 0);
        nReadyArray  : in  std_logic_vector(OUTPUTS - 1 downto 0);
        validArray   : out std_logic_vector(OUTPUTS - 1 downto 0)
    );
end forkNlazy;

-- generic lazy implementation from cortadellas paper
------------------------------------------------------
architecture arch of forkNlazy is
    signal allnReady : std_logic;
begin

    genericAnd : entity work.andn generic map (OUTPUTS)
        port map(nReadyArray, allnReady);

    valid : process(pValidArray, nReadyArray, allnReady)
    begin
        for I in 0 to OUTPUTS-1 loop
            validArray(I) <= pValidArray(0) and allnReady;
        end loop;
    end process;

    readyArray(0) <= allnReady;

    data : process(dataInArray) begin
        for I in 0 to OUTPUTS - 1 loop
            dataOutArray(I) <= dataInArray(0);
        end loop;  
    end process;

    -- Speculative bit logic
    spec : process(specInArray) begin
        for I in OUTPUTS-1 downto 0 loop
            specOutArray(I) <= specInArray(0);
        end loop;
    end process;

end arch;

------------------------------------------------------------------------
-- EagerFork ----> SPEC version

library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity fork is 
    generic ( 
        INPUTS        : integer;
        OUTPUTS       : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );
    port ( 
        clk, rst     : in  std_logic;
        dataInArray  : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        specInArray  : in  data_array(0 downto 0)(0 downto 0);
        pValidArray  : in  std_logic_vector(0 downto 0);
        readyArray   : out std_logic_vector(0 downto 0);
        dataOutArray : out data_array(OUTPUTS - 1 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        specOutArray : out data_array(OUTPUTS - 1 downto 0)(0 downto 0);
        nReadyArray  : in  std_logic_vector(OUTPUTS - 1 downto 0);
        validArray   : out std_logic_vector(OUTPUTS - 1 downto 0)
    );
end fork;

-- generic eager implementation uses registers
------------------------------------------------------
architecture arch of fork is
    
    -- wrapper signals (internals use "stop" signals instead of "ready" signals)
    signal forkStop : std_logic;
    signal nStopArray : std_logic_vector(OUTPUTS - 1 downto 0);
    -- internal combinatorial signals
    signal blockStopArray : std_logic_vector(OUTPUTS - 1 downto 0);
    signal anyBlockStop : std_logic;
    signal pValidAndForkStop : std_logic;

begin

    -- can't adapt the signals directly in port map
    wrapper : process(forkStop, nReadyArray) begin
        readyArray(0) <= not forkStop;
        for i in 0 to OUTPUTS - 1 loop
            nStopArray(i) <= not nReadyArray(i);
        end loop;
    end process;

    genericOr : entity work.orN generic map (OUTPUTS)
        port map(blockStopArray, anyBlockStop);

    -- internal combinatorial signals
    forkStop <= anyBlockStop;
    pValidAndForkStop <= pValidArray(0) and forkStop;

    -- generate blocks
    generateBlocks : for i in OUTPUTS - 1 downto 0 generate
        regblock : entity work.eagerFork_RegisterBlock(arch)
                port map(   clk, rst,
                            pValidArray(0), nStopArray(i),
                            pValidAndForkStop,
                            validArray(i), blockStopArray(i));
    end generate;

    process(dataInArray) begin
        for I in 0 to OUTPUTS - 1 loop
            dataOutArray(I) <= dataInArray(0);
        end loop;  
    end process;

    -- Speculative bit logic
    process(specInArray) begin
        for I in OUTPUTS - 1 downto 0 loop
            specOutArray(I) <= specInArray(0);
        end loop;
    end process;
    -----

end arch;

------------------------------------------------------------------------
-- Merge ----> SPEC version

library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity merge is 
    generic ( 
        INPUTS        : integer;
        OUTPUTS       : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );
    port ( 
        clk, rst     : in  std_logic;
        dataInArray  : in  data_array(INPUTS - 1 downto 0)(DATA_SIZE_IN - 1 downto 0);
        specInArray  : in  data_array(INPUTS - 1 downto 0)(0 downto 0);
        pValidArray  : in  std_logic_vector(INPUTS - 1 downto 0);
        readyArray   : out std_logic_vector(INPUTS - 1 downto 0);
        dataOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        specOutArray : out data_array(0 downto 0)(0 downto 0);
        nReadyArray  : in  std_logic_vector(0 downto 0);
        validArray   : out std_logic_vector(0 downto 0)
    );
end merge;

architecture arch of merge is

    signal tehb_data_in : std_logic_vector(DATA_SIZE_IN+1 - 1 downto 0);
    signal tehb_pvalid : std_logic;
    signal tehb_ready : std_logic;

    signal specdataInArray  : data_array(INPUTS - 1 downto 0)(DATA_SIZE_IN+1 - 1 downto 0);
    signal specdataOutArray : data_array(0 downto 0)(DATA_SIZE_OUT+1 - 1 downto 0);
    signal unconnected_spec : data_array(0 downto 0)(0 downto 0);

begin

    -- Speculative bit logic
    process(specInArray, dataInArray) begin
        for I in INPUTS - 1 downto 0 loop 
            specdataInArray(I) <= specInArray(I) & dataInArray(I);
        end loop;
    end process;
    -----

    process(pValidArray, specdataInArray)
        variable tmp_data_out  : unsigned(DATA_SIZE_IN+1 - 1 downto 0);
        variable tmp_valid_out : std_logic;
    begin
        tmp_data_out  := unsigned(specdataInArray(0));
        tmp_valid_out := '0';
        for I in INPUTS - 1 downto 0 loop
            if (pValidArray(I) = '1') then
                tmp_data_out  := unsigned(specdataInArray(I));
                tmp_valid_out := pValidArray(I);
            end if;
        end loop;

        tehb_data_in <= std_logic_vector(resize(tmp_data_out, DATA_SIZE_OUT+1));
        tehb_pvalid  <= tmp_valid_out;
    end process;

    process(tehb_ready) begin
        for I in 0 to INPUTS - 1 loop
            readyArray(I) <= tehb_ready;
        end loop;
    end process;

    tehb1: entity work.TEHB(arch) generic map (1, 1, DATA_SIZE_IN+1, DATA_SIZE_IN+1)
        port map (
            clk => clk,
            rst => rst,
            dataInArray(0)      => tehb_data_in,
            specInArray(0)(0)   => '0',
            pValidArray(0)      => tehb_pvalid,
            readyArray(0)       => tehb_ready,
            nReadyArray(0)      => nReadyArray(0),
            validArray(0)       => validArray(0),
            dataOutArray(0)     => specdataOutArray(0),
            specOutArray        => unconnected_spec
        );

    -- Speculative bit logic
    dataOutArray(0)    <= specdataOutArray(0)(DATA_SIZE_OUT - 1 downto 0);
    specOutArray(0)(0) <= specdataOutArray(0)(DATA_SIZE_OUT+1 - 1);

end arch;

------------------------------------------------------------------------
-- Merge_notehb ----> SPEC version

library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity merge_notehb is 
    generic ( 
        INPUTS        : integer;
        OUTPUTS       : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer
    );
    port ( 
        clk, rst     : in  std_logic;
        dataInArray  : in  data_array(INPUTS - 1 downto 0)(DATA_SIZE_IN - 1 downto 0);
        specInArray  : in  data_array(INPUTS - 1 downto 0)(0 downto 0);
        pValidArray  : in  std_logic_vector(INPUTS - 1 downto 0);
        readyArray   : out std_logic_vector(INPUTS - 1 downto 0);
        dataOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        specOutArray : out data_array(0 downto 0)(0 downto 0);
        nReadyArray  : in  std_logic_vector(0 downto 0);
        validArray   : out std_logic_vector(0 downto 0)
    );
end merge_notehb;

architecture arch of merge_notehb is

    signal tehb_data_in : std_logic_vector(DATA_SIZE_IN+1 - 1 downto 0);
    signal tehb_pvalid : std_logic;
    signal tehb_ready : std_logic;

    signal specdataInArray  : data_array(INPUTS - 1 downto 0)(DATA_SIZE_IN+1 - 1 downto 0);
    signal specdataOutArray : data_array(0 downto 0)(DATA_SIZE_OUT+1 - 1 downto 0);

begin

    -- Speculative bit logic
    process(specInArray, dataInArray) begin
        for I in INPUTS - 1 downto 0 loop 
            specdataInArray(I) <= specInArray(I) & dataInArray(I);
        end loop;
    end process;
    -----

    process(pValidArray, specdataInArray)
        variable tmp_data_out  : unsigned(DATA_SIZE_IN+1 - 1 downto 0);
        variable tmp_valid_out : std_logic;
    begin
        tmp_data_out  := unsigned(specdataInArray(0));
        tmp_valid_out := '0';
        for I in INPUTS - 1 downto 0 loop
            if (pValidArray(I) = '1') then
                tmp_data_out  := unsigned(specdataInArray(I));
                tmp_valid_out := pValidArray(I);
            end if;
        end loop;

        tehb_data_in <= std_logic_vector(resize(tmp_data_out, DATA_SIZE_OUT+1));
        tehb_pvalid  <= tmp_valid_out;
    end process;

    process(tehb_ready) begin
        for I in 0 to INPUTS - 1 loop
            readyArray(I) <= tehb_ready;
        end loop;
    end process;

    tehb_ready <= nReadyArray(0);
    validArray(0) <= tehb_pvalid;
    specdataOutArray(0) <= tehb_data_in;

    -- Speculative bit logic
    dataOutArray(0)    <= specdataOutArray(0)(DATA_SIZE_OUT - 1 downto 0);
    specOutArray(0)(0) <= specdataOutArray(0)(DATA_SIZE_OUT+1 - 1);

end arch;

------------------------------------------------------------------------
-- Mux ----> SPEC version

library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

entity mux is
    generic(
        INPUTS        : integer;
        OUTPUTS       : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer;
        COND_SIZE     : integer
    );
    port(
        clk, rst     : in  std_logic;
        dataInArray  : in  data_array(INPUTS - 2 downto 0)(DATA_SIZE_IN - 1 downto 0);
        specInArray  : in  data_array(INPUTS - 2 downto 0)(0 downto 0);
        dataOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        specOutArray : out data_array(0 downto 0)(0 downto 0);
        pValidArray  : in  std_logic_vector(INPUTS - 1 downto 0);
        readyArray   : out std_logic_vector(INPUTS - 1 downto 0);
        nReadyArray  : in  std_logic_vector(0 downto 0);
        validArray   : out std_logic_vector(0 downto 0);
        condition    : in  data_array(0 downto 0)(COND_SIZE - 1 downto 0)  --(integer(ceil(log2(real(INPUTS)))) - 1 downto 0);
    );
end mux;

architecture arch of mux is

    signal tehb_data_in : std_logic_vector(DATA_SIZE_IN+1 - 1 downto 0);
    signal tehb_pvalid : std_logic;
    signal tehb_ready : std_logic;

    signal specdataInArray  : data_array(INPUTS - 2 downto 0)(DATA_SIZE_IN+1 - 1 downto 0);
    signal specdataOutArray : data_array(0 downto 0)(DATA_SIZE_OUT+1 - 1 downto 0);
    signal unconnected_spec : data_array(0 downto 0)(0 downto 0);

begin
 
    -- Speculative bit logic
    process(specInArray, dataInArray) begin
        for I in INPUTS - 2 downto 0 loop 
            specdataInArray(I) <= specInArray(I) & dataInArray(I);
        end loop;
    end process;
    -----

    process(specdataInArray, pValidArray, nReadyArray, condition, tehb_ready)
        variable tmp_data_out  : unsigned(DATA_SIZE_IN+1 - 1 downto 0);
        variable tmp_valid_out : std_logic;

    begin
        tmp_data_out  := unsigned(specdataInArray(0));
        tmp_valid_out := '0';
        for I in INPUTS - 2 downto 0 loop
            -- if (the condition refers the Ith data input, condition is valid, and the Ith input is valid), assign input data to output and set the output valid high
            if (unsigned(condition(0)) = to_unsigned(I,condition(0)'length) and pValidArray(0) = '1' and pValidArray(I+1) = '1') then
                tmp_data_out  := unsigned(specdataInArray(I));
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
        tehb_data_in <= std_logic_vector(resize(tmp_data_out,DATA_SIZE_OUT+1));
        tehb_pvalid  <= tmp_valid_out;
    end process;

    tehb1: entity work.TEHB(arch) generic map (1, 1, DATA_SIZE_IN+1, DATA_SIZE_IN+1)
        port map (
            clk => clk,
            rst => rst,
            dataInArray(0)      => tehb_data_in,
            specInArray(0)(0)   => '0',
            pValidArray(0)      => tehb_pvalid,
            readyArray(0)       => tehb_ready,
            nReadyArray(0)      => nReadyArray(0),
            validArray(0)       => validArray(0),
            dataOutArray(0)     => specdataOutArray(0),
            specOutArray        => unconnected_spec
        );

    -- Speculative bit logic
    dataOutArray(0)    <= specdataOutArray(0)(DATA_SIZE_OUT - 1 downto 0);
    specOutArray(0)(0) <= specdataOutArray(0)(DATA_SIZE_OUT+1 - 1);

end arch;

------------------------------------------------------------------------
-- CntrlMerge ----> SPEC version

library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity cntrlMerge is
    generic(
        INPUTS        : integer;
        OUTPUTS       : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer;
        COND_SIZE     : integer
    );
    port(
        clk, rst     : in  std_logic;
        dataInArray  : in  data_array(INPUTS - 1 downto 0)(DATA_SIZE_IN - 1 downto 0);
        specInArray  : in  data_array(INPUTS - 1 downto 0)(0 downto 0);
        dataOutArray : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        specOutArray : out data_array(1 downto 0)(0 downto 0);
        pValidArray  : in  std_logic_vector(1 downto 0);
        readyArray   : out std_logic_vector(1 downto 0);
        nReadyArray  : in  std_logic_vector(1 downto 0);
        validArray   : out std_logic_vector(1 downto 0);
        condition    : out data_array(0 downto 0)(0 downto 0)
    );
end cntrlMerge;

architecture arch of cntrlMerge is

    signal phi_C1_readyArray : std_logic_vector(1 downto 0);
    signal phi_C1_validArray : std_logic_vector(0 downto 0);
    signal phi_C1_dataOutArray : data_array(0 downto 0)(0 downto 0);

    signal fork_C1_readyArray : std_logic_vector(0 downto 0);
    signal fork_C1_validArray : std_logic_vector(1 downto 0);
    signal fork_C1_dataOutArray : data_array(1 downto 0)(0 downto 0);

    signal oehb1_valid, oehb1_ready, index : std_logic;
    signal oehb1_dataOut : std_logic_vector(COND_SIZE - 1 downto 0);

    signal unconnected_spec   : data_array(0 downto 0)(0 downto 0);
    signal unconnected_spec_2 : data_array(1 downto 0)(0 downto 0);

begin

    readyArray <= phi_C1_readyArray;

    phi_C1: entity work.merge_notehb(arch) generic map (2, 1, 1, 1)
        port map (
            clk                 => clk,
            rst                 => rst,
            dataInArray (0)     => "1",
            dataInArray (1)     => "1",
            specInArray(0)(0)   => '0',
            specInArray(1)(0)   => '0',
            pValidArray         => pValidArray,
            readyArray          => phi_C1_readyArray,        
            nReadyArray(0)      => oehb1_ready,
            validArray          => phi_C1_validArray,
            dataOutArray        => phi_C1_dataOutArray,
            specOutArray        => unconnected_spec
        );

    process(pValidArray) begin
        if (pValidArray(0) = '1') then
            index <= '0';
        else
            index <= '1';
        end if;
    end process;

    oehb1: entity work.TEHB(arch) generic map (1, 1, 1, 1)
        port map (
            clk                 => clk,
            rst                 => rst,
            dataInArray(0)(0)   => index,
            specInArray(0)(0)   => '0',
            pValidArray(0)      => phi_C1_validArray(0),
            readyArray(0)       => oehb1_ready,
            nReadyArray(0)      => fork_C1_readyArray(0),
            validArray(0)       => oehb1_valid,
            dataOutArray(0)     => oehb1_dataOut,
            specOutArray        => unconnected_spec
        );

    fork_C1: entity work.fork(arch) generic map (1, 2, 1, 1)
        port map (
            clk => clk,
            rst => rst,
            dataInArray (0) => "1",
            specInArray(0)(0) => '0',
            pValidArray(0) => oehb1_valid,
            readyArray => fork_C1_readyArray,
            nReadyArray => nReadyArray,
            validArray => fork_C1_validArray,
            dataOutArray => fork_C1_dataOutArray,
            specOutArray => unconnected_spec_2
        );

    validArray <= fork_C1_validArray;
    condition(0) <= oehb1_dataOut;

    -- Speculative bit logic
    process(pValidArray, specInArray) begin  -- Assuming always INPUTS = 2
        if (pValidArray(0) = '1') then
            specOutArray(0)(0) <= specInArray(0)(0);
            specOutArray(1)(0) <= specInArray(0)(0);
        elsif (pValidArray(1) = '1') then
            specOutArray(0)(0) <= specInArray(1)(0);
            specOutArray(1)(0) <= specInArray(1)(0);
        else
            specOutArray(0)(0) <= '0';
            specOutArray(1)(0) <= '0';
        end if;
    end process;
    -----

end arch;

-----------------------------------------------------------------------------------------
----------------------------------------------------------------------- Memory operations
-----------------------------------------------------------------------------------------

------------------------------------------------------------------------
-- Miscellaneous store support

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity write_memory_single_inside is generic (ADDRESS_SIZE : Integer; DATA_SIZE : Integer);
    port (

        clk           : in  std_logic;
        -- Input interface
        dataValid     : in  std_logic; -- write request
        --addrValid: in std_logic; -- need join for address and data! add somewhere
        ready         : out std_logic; -- ready
        input_addr    : in  std_logic_vector(ADDRESS_SIZE - 1 downto 0);
        data          : in  std_logic_vector(DATA_SIZE - 1 downto 0); -- data to write
        -- Output interface
        nReady        : in  std_logic; -- next component can continue after write
        valid         : out std_logic; -- sending write confirmation to next component
        -- Memory Interface
        write_enable    : out std_logic;
        enable          : out std_logic;
        write_address   : out std_logic_vector(ADDRESS_SIZE - 1 downto 0);
        data_to_memory  : out std_logic_vector(DATA_SIZE - 1 downto 0)
    );
end entity;

architecture arch of write_memory_single_inside is

begin

    process(clk) is
    begin
        if (rising_edge(clk)) then
            write_address  <= input_addr;
            data_to_memory <= data;
            valid          <= dataValid;
            write_enable   <= dataValid and nReady;
            enable         <= dataValid and nReady;
        end if;
    end process;

    ready <= nReady;

end architecture;

------------------------------------------------------------------------
-- Generic load_op (read port)

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity load_op is
    generic (
        INPUTS        : integer;
        OUTPUTS       : integer;
        ADDRESS_SIZE  : integer;
        DATA_SIZE     : integer
    );
    port (
        clk, rst      : in  std_logic;
        -- Input interface
        dataInArray   : in  data_array(0 downto 0)(ADDRESS_SIZE - 1 downto 0); -- addrIn
        pValidArray   : in  std_logic_vector(0 downto 0); -- read request from prev component
        readyArray    : out std_logic_vector(0 downto 0); -- ready to process read
        -- Output interface
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE - 1 downto 0); -- dataOut
        validArray    : out std_logic_vector(0 downto 0); -- sending data to next component
        nReadyArray   : in  std_logic_vector(0 downto 0); -- next component can accept data
        -- Memory Interface
        read_enable      : out std_logic;
        read_address     : out std_logic_vector(ADDRESS_SIZE - 1 downto 0);
        data_from_memory : in  std_logic_vector(DATA_SIZE - 1 downto 0)
    );
end load_op;

architecture arch of load_op is 

    signal temp, tempen : std_logic;
    signal q0, q1, enable_internal : std_logic;
    signal valid_temp : std_logic_vector(0 downto 0);
    signal read_address_internal : data_array(0 downto 0)(ADDRESS_SIZE - 1 downto 0);
    signal unconnected_spec : data_array(0 downto 0)(0 downto 0);

begin

    read_enable <= valid_temp(0) and nReadyArray(0);
    enable_internal <= valid_temp(0) and nReadyArray(0);
    dataOutArray(0) <= data_from_memory;

    buff_n0: entity work.elasticBuffer(arch) generic map (1, 1, ADDRESS_SIZE, ADDRESS_SIZE)
        port map (
            clk                 => clk,
            rst                 => rst,
            dataInArray(0)      => dataInArray(0),
            specInArray(0)(0)   => '0',
            pValidArray(0)      => pValidArray(0),
            readyArray          => readyArray,
            nReadyArray(0)      => nReadyArray(0),
            validArray          => valid_temp,
            dataOutArray        => read_address_internal,
            specOutArray        => unconnected_spec
        );

    read_address <= read_address_internal(0);

    process(clk, rst) is
    begin
        if (rst = '1') then
            validArray(0) <= '0';
        elsif (rising_edge(clk)) then
            if (enable_internal = '1') then
                validArray(0) <= '1';
            else
                if (nReadyArray(0) = '1') then
                    validArray(0) <= '0';
                end if;
            end if;          
        end if;
    end process;

end architecture;

------------------------------------------------------------------------
-- Generic store_op (write port)

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity store_op is
    generic (
        INPUTS        : integer;
        OUTPUTS       : integer;
        ADDRESS_SIZE  : integer;
        DATA_SIZE     : integer
    );
    port (
        clk, rst      : in  std_logic;
        -- Input interface
        input_addr    : in  std_logic_vector(ADDRESS_SIZE - 1 downto 0);
        dataInArray   : in  data_array(0 downto 0)(DATA_SIZE - 1 downto 0);
        pValidArray   : in  std_logic_vector(1 downto 0);
        readyArray    : out std_logic_vector(1 downto 0);
        -- Output interface
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE - 1 downto 0); -- unused
        validArray    : out std_logic_vector(0 downto 0);
        nReadyArray   : in  std_logic_vector(0 downto 0);
        -- Memory Interface
        write_enable    : out std_logic;
        enable          : out std_logic;
        write_address   : out std_logic_vector(ADDRESS_SIZE - 1 downto 0);
        data_to_memory  : out std_logic_vector(DATA_SIZE - 1 downto 0)
    );
end store_op;

architecture arch of store_op is
    
    signal single_ready, join_valid : std_logic;

begin

    join_write: entity work.join(arch) generic map(2)
        port map(   pValidArray,  --pValidArray
                    single_ready, --nReady                    
                    join_valid,   --valid          
                    ReadyArray);  --readyArray

    Write: entity work.write_memory_single_inside (arch) generic map (ADDRESS_SIZE, DATA_SIZE)
        port map( clk, 
              join_valid,      --pValid
              single_ready,    --ready
              input_addr,      --addr0
              dataInArray(0),  --data0
              nReadyArray(0),  --nReady
              validArray(0),   --valid
              write_enable,    --write enable
              enable,          --enable
              write_address,   --write address
              data_to_memory); --data to memory

 end architecture;


------------------------------------------------------------------------
-- LSQ load_op (read port)

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity lsq_load_op is
    generic (
        INPUTS        : integer;
        OUTPUTS       : integer;
        ADDRESS_SIZE  : integer;
        DATA_SIZE     : integer
    );
    port (
        clk, rst      : in  std_logic;
        -- Input interface
        input_addr    : in  std_logic_vector(ADDRESS_SIZE - 1 downto 0);
        dataInArray   : in  data_array(0 downto 0)(DATA_SIZE - 1 downto 0);
        specInArray   : in  data_array(1 downto 0)(0 downto 0);   -- (addr, data);
        pValidArray   : in  std_logic_vector(INPUTS - 1 downto 0);  -- (addr, data)
        readyArray    : out std_logic_vector(INPUTS - 1 downto 0);  -- (addr, data)
        -- Output interface
        output_addr   : out std_logic_vector(ADDRESS_SIZE - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE - 1 downto 0);
        specOutArray  : out data_array(1 downto 0)(0 downto 0);   -- (addr, data);
        nReadyArray   : in  std_logic_vector(OUTPUTS - 1 downto 0); -- (addr, data)
        validArray    : out std_logic_vector(OUTPUTS - 1 downto 0)  -- (addr, data)
    );
end lsq_load_op;

architecture arch of lsq_load_op is 

signal spec_out : std_logic;

begin

    -- address request goes to LSQ
    output_addr   <= input_addr;
    validArray(1) <= pValidArray(1);
    readyArray(1) <= nReadyArray(1);

    -- data from LSQ to load output
    dataOutArray(0) <= dataInArray(0);
    validArray(0)   <= pValidArray(0);
    readyArray(0)   <= nReadyArray(0);

    -- Speculative bit logic
    spec_out <= specInArray(1)(0) or specInArray(0)(0);
    specOutArray(0)(0) <= spec_out;
    specOutArray(1)(0) <= spec_out;
        
end architecture;

------------------------------------------------------------------------
-- LSQ load_op for speculative load(read port)

--library ieee;
--use ieee.std_logic_1164.all;
--use ieee.numeric_std.all;
--use work.customTypes.all;

--entity lsq_load_op is
--    generic (
--        INPUTS        : integer;
--        OUTPUTS       : integer;
--        ADDRESS_SIZE  : integer;
--        DATA_SIZE     : integer
--    );
--    port (
--        clk, rst      : in  std_logic;
--        -- Input interface
--        input_addr    : in  std_logic_vector(ADDRESS_SIZE - 1 downto 0);
--        dataInArray   : in  data_array(1 downto 0)(DATA_SIZE - 1 downto 0); -- (data_mc, data_lsq)
--        specInArray   : in  data_array(2 downto 0)(0 downto 0);   -- (addr, data_mc, data_lsq);
--        pValidArray   : in  std_logic_vector(2 downto 0);  -- (addr, data_mc, data_lsq)
--        readyArray    : out std_logic_vector(2 downto 0);  -- (addr, data_mc, data_lsq)
--        -- Output interface
--        output_addr   : out std_logic_vector(ADDRESS_SIZE - 1 downto 0);
--        output_mc     : out std_logic_vector(ADDRESS_SIZE - 1 downto 0);
--        dataOutArray  : out data_array(1 downto 0)(DATA_SIZE - 1 downto 0); -- (data_spec, data_normal)
--        specOutArray  : out data_array(3 downto 0)(0 downto 0);   -- (addr_mc, addr_lsq, data_spec, data_normal);
--        nReadyArray   : in  std_logic_vector(3 downto 0); -- (addr_mc, addr_lsq, data_spec, data_normal)
--        validArray    : out std_logic_vector(3 downto 0)  -- (addr_mc, addr_lsq, data_spec, data_normal)
--    );
--end lsq_load_op;

--architecture arch of lsq_load_op is 

--signal spec_bit : std_logic;
-- signal fifo_spec_dataIn  : data_array(0 downto 0)(0 downto 0);
--    signal fifo_spec_dataOut : data_array(0 downto 0)(0 downto 0);
--    signal fifo_spec_ready, fifo_spec_pValid : std_logic_vector(1 downto 0);
--    signal fifo_spec_valid, fifo_spec_nReady : std_logic_vector(0 downto 0);
--    signal fifo_spec_send : std_logic;

--begin

--    -- address request goes to LSQ and MC
--    output_addr   <= input_addr;
--    output_mc <= input_addr;
--    validArray(3) <= pValidArray(2);
--    validArray(2) <= pValidArray(2);
--    readyArray(2) <= nReadyArray(3) and nReadyArray(2);

--    -- data from LSQ to normal output
--    dataOutArray(0) <= dataInArray(0);
--    validArray(0)   <= pValidArray(0);
--    readyArray(0)   <= nReadyArray(0);

--     -- data from MC to spec output
--    dataOutArray(1) <= dataInArray(1);
--    validArray(1)   <= pValidArray(1);
--    readyArray(1)   <= nReadyArray(1);

--    -- Speculative bit logic
--    spec_bit <= specInArray(2)(0);

--    fifo_spec_dataIn(0)(0) <= spec_bit;
--    specOutArray(0) <= fifo_spec_dataOut;

--    fifo_spec_pValid(0) <= pValidArray(2);
--    fifo_spec_pValid(1) <= '1';
--    fifo_spec_nReady(0) <= nReadyArray(0);

--    fifo_spec_send <= pValidArray(0); 
    
--    fifo_spec: entity work.load_spec_FIFO(arch) generic map (2, 1, 1, 1, 16)
--        port map (
--            clk => clk,
--            rst => rst,
--            dataInArray         => fifo_spec_dataIn,
--            dataOutArray        => fifo_spec_dataOut,
--            pValidArray         => fifo_spec_pValid, -- (send, dIn)
--            nReadyArray         => fifo_spec_nReady,
--            validArray          => fifo_spec_valid,
--            readyArray          => fifo_spec_ready,  -- (send, dIn)
--            send                => fifo_spec_send
--    );
        
--end architecture;

------------------------------------------------------------------------
-- LSQ store_op (write port)

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity lsq_store_op is
    generic (
        INPUTS        : integer;
        OUTPUTS       : integer;
        ADDRESS_SIZE  : integer;
        DATA_SIZE     : integer
    );
    port (
        clk, rst      : in  std_logic;
        -- Input interface
        input_addr    : in  std_logic_vector(ADDRESS_SIZE - 1 downto 0);
        dataInArray   : in  data_array(0 downto 0)(DATA_SIZE - 1 downto 0);
        specInArray   : in  data_array(1 downto 0)(0 downto 0);   -- (addr, data);
        pValidArray   : in  std_logic_vector(INPUTS - 1 downto 0);  -- (addr, data)
        readyArray    : out std_logic_vector(INPUTS - 1 downto 0);  -- (addr, data)
        -- Output interface
        output_addr   : out std_logic_vector(ADDRESS_SIZE - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE - 1 downto 0);
        specOutArray  : out data_array(1 downto 0)(0 downto 0);   -- (addr, data);
        nReadyArray   : in  std_logic_vector(OUTPUTS - 1 downto 0); -- (addr, data)
        validArray    : out std_logic_vector(OUTPUTS - 1 downto 0)  -- (addr, data)
    );
end lsq_store_op;

architecture arch of lsq_store_op is 

signal spec_out : std_logic;

begin
    
    -- address goes to LSQ
    output_addr   <= input_addr;
    validArray(1) <= pValidArray(1);
    readyArray(1) <= nReadyArray(1);
    
    -- data goes to LSQ
    dataOutArray(0) <= dataInArray(0);
    validArray(0)   <= pValidArray(0);
    readyArray(0)   <= nReadyArray(0);

    -- Speculative bit logic
    spec_out <= specInArray(1)(0) or specInArray(0)(0);
    specOutArray(0)(0) <= spec_out;
    specOutArray(1)(0) <= spec_out;
        
end architecture;