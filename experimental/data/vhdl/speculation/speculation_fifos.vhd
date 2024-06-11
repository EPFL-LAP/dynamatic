-----------------------------------------------------------------------------------------
--------------------------------------------------------------- FIFO for speculated data
-----------------------------------------------------------------------------------------
-- Same logic as speculating_branch_FIFO
-- Input to discard all, output indicating if not empty

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity speculator_FIFO is
	generic (
		INPUTS        : integer;  -- assuming always 2: 1 input, 1 discardAll signal
        OUTPUTS       : integer;  -- assuming always 2: 1 output, 1 notEmpty signal
        DATA_SIZE_IN  : integer;  -- use normal data size, eg- 32
        DATA_SIZE_OUT : integer;  -- use normal data size, eg- 32
        FIFO_DEPTH    : integer   -- extra parameter for FIFO
	);
	port (
		clk, rst      : in  std_logic;
        dataInArray   : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        nReadyArray   : in  std_logic_vector(1 downto 0);  -- (notEmp, dout)
        validArray    : out std_logic_vector(1 downto 0);  -- (notEmp, dout)
        pValidArray   : in  std_logic_vector(1 downto 0);  -- (discAll, din)
        readyArray    : out std_logic_vector(1 downto 0);  -- (discAll, din)
        ----------
        filledSlots   : out natural range 0 to FIFO_DEPTH + FIFO_DEPTH;
        notEmpty      : out std_logic;
    	discardAll    : in  std_logic
	);
end speculator_FIFO;
 
architecture arch of speculator_FIFO is

    signal ReadEn  : std_logic := '0';
    signal WriteEn : std_logic := '0';
    signal Tail    : natural range 0 to FIFO_DEPTH - 1;
    signal Head    : natural range 0 to FIFO_DEPTH - 1;
    signal Empty   : std_logic;
    signal Full    : std_logic;
    --signal emptySlots : natural range 0 to FIFO_DEPTH + FIFO_DEPTH;
    type FIFO_Memory is array (0 to FIFO_DEPTH - 1) of std_logic_vector(DATA_SIZE_IN - 1 downto 0);
    signal Memory : FIFO_Memory;

begin
    
    -- Ready to write if there is space in the FIFO
    readyArray(0) <= not Full;
    -- FIFO control always ready
    readyArray(1) <= '1';

    -- Valid read if FIFO is not empty
    validArray(0) <= not Empty;
    dataOutArray(0) <= Memory(Head);
    -- FIFO status always valid
    validArray(1) <= '1';
    notEmpty <= not Empty;

    -- Read if there is sth in FIFO, and next is ready to accept (data only) --and status)
    ReadEn <= (not Empty) and (nReadyArray(0)); -- and nReadyArray(1));
    -- Write if there is valid input and there is space in the FIFO
    WriteEn <= pValidArray(0) and not Full;

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
                if (discardAll = '1' and pValidArray(1) = '1') then
                    Tail <= 0;
                elsif (WriteEn = '1') then
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
                if (discardAll = '1' and pValidArray(1) = '1') then
                    Head <= 0;
                elsif (ReadEn = '1') then
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
                if (discardAll = '1' and pValidArray(1) = '1') then
                    Full <= '0';
                -- if only filling but not emptying
                elsif (WriteEn = '1') and (ReadEn = '0') then
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
                if (discardAll = '1' and pValidArray(1) = '1') then
                    Empty <= '1';
                -- if only emptying but not filling
                elsif (WriteEn = '0') and (ReadEn = '1') then
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

    -------------------------------------------
    -- Count filled slots (necessary output)
    CalculateSlots_proc: process (Head, Tail, Empty) begin
        if (Head < Tail or Empty = '1') then 
            --emptySlots <= (Head - Tail + FIFO_DEPTH);
            filledSlots <= (Tail - Head);
        else
            --emptySlots <= (Head - Tail);
            filledSlots <= (FIFO_DEPTH + Tail - Head);
        end if;
    end process;

end arch;

-----------------------------------------------------------------------------------------
---------------------------------------------------------------- FIFO for savecommit unit
-----------------------------------------------------------------------------------------
-- Same logic as SaveResolveFIFO
-- FIFO read on resend = 0, flush all on resend = 1

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity savecommit_FIFO is
    generic (
        INPUTS        : integer;  -- assuming always 2: 1 input, 1 resend signal
        OUTPUTS       : integer;  -- assuming always 1 output
        DATA_SIZE_IN  : integer;  -- use normal data size, eg- 32
        DATA_SIZE_OUT : integer;  -- use normal data size, eg- 32
        FIFO_DEPTH    : integer   -- extra parameter for FIFO
    );
    port (
        clk, rst      : in  std_logic;
        dataInArray   : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        nReadyArray   : in  std_logic_vector(0 downto 0);  -- (dout)
        validArray    : out std_logic_vector(0 downto 0);  -- (dout)
        pValidArray   : in  std_logic_vector(1 downto 0);  -- (resend, din)
        readyArray    : out std_logic_vector(1 downto 0);  -- (resend, din)
        ----------
        --filledSlots   : out natural range 0 to FIFO_DEPTH + FIFO_DEPTH;
        resend        : in  std_logic
    );
end savecommit_FIFO;
 
architecture arch of savecommit_FIFO is

    signal FlushEn : std_logic := '0';
    signal ReadEn  : std_logic := '0';
    signal WriteEn : std_logic := '0';
    signal Tail    : natural range 0 to FIFO_DEPTH - 1;
    signal Head    : natural range 0 to FIFO_DEPTH - 1;
    signal Empty   : std_logic;
    signal Full    : std_logic;
    --signal emptySlots : natural range 0 to FIFO_DEPTH + FIFO_DEPTH;
    type FIFO_Memory is array (0 to FIFO_DEPTH - 1) of std_logic_vector(DATA_SIZE_IN - 1 downto 0);
    signal Memory : FIFO_Memory;

begin
    
    -- Ready to write if there is space in the FIFO
    readyArray(0) <= not Full;
    -- FIFO control always ready
    readyArray(1) <= '1';

    -- Valid read if FIFO is not empty
    validArray(0) <= not Empty;
    dataOutArray(0) <= Memory(Head);

    -- Read if there is a valid resend = 0 (correct spec)
    ReadEn  <= pValidArray(1) and not resend;
    -- Flush if there is a valid resend = 1 (wrong spec)
    -- There is also a simultaneous read with flush
    FlushEn <= pValidArray(1) and resend;

    -- Write if there is valid input and there is space in the FIFO
    WriteEn <= pValidArray(0) and not Full;

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
                if (FlushEn = '1') then
                    Tail <= 0;
                elsif (WriteEn = '1') then
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
                elsif (FlushEn = '1') then
                    Head <= 0;
                end if;
            end if;
        end if;
    end process;

    ------------------------------------------- check ???
    -- process for updating Full
    FullUpdate_proc : process (clk) begin
        if rising_edge(clk) then
            if rst = '1' then
               Full <= '0';
            else
                -- if only filling but not emptying
                --if (WriteEn = '1') and (resend='0' and pValidArray(1)='0') then -- why resend = 0?
                if (WriteEn = '1') and (pValidArray(1) = '0') then
                    -- if new tail index will reach head index
                    if ((Tail + 1) mod FIFO_DEPTH = Head) then
                        Full <= '1';
                    end if;
                -- if only emptying but not filling
                elsif (WriteEn = '0') and (pValidArray(1) = '1') then
                    Full <= '0';
                -- otherwise, nothing is happenning or simultaneous read and write
                end if;
            end if;
        end if;
    end process;

    ------------------------------------------- check ???
    -- process for updating Empty
    EmptyUpdate_proc : process (clk) begin
        if rising_edge(clk) then
            if rst = '1' then
               Empty <= '1';
            else
                -- if flushing and not filling
                if (WriteEn = '0') and (FlushEn = '1') then
                    Empty <= '1';
                -- if only emptying but not filling
                elsif (WriteEn = '0') and (ReadEn = '1') then
                    -- if new head index will reach tail index
                    if ((Head + 1) mod FIFO_DEPTH = Tail) then
                        Empty <= '1';
                    else
                        Empty <= '0';
                    end if;
                -- if only filling but not emptying
                --elsif (WriteEn = '1') and (resend='1' and pValidArray(1)='0') then -- why resend = 1?
                elsif (WriteEn = '1') and (pValidArray(1) = '0') then
                    Empty <= '0';

                -- what about writing and flushing simulateously? also reading.. so Empty <= '1'?

                -- otherwise, nothing is happenning or simultaneous read and write
                end if;       
            end if;
        end if;
    end process;

    -------------------------------------------
    ---- Count filled slots (for debugging only)
    --CalculateSlots_proc: process (Head, Tail, Empty) begin
    --    if (Head < Tail or Empty = '1') then 
    --        --emptySlots <= (Head - Tail + FIFO_DEPTH);
    --        filledSlots <= (Tail - Head);
    --    else
    --        --emptySlots <= (Head - Tail);
    --        filledSlots <= (FIFO_DEPTH + Tail - Head);
    --    end if;
    --end process;

end arch;

-----------------------------------------------------------------------------------------
--------------------------------------------------------------- FIFO for discard signals
-----------------------------------------------------------------------------------------
-- Similar to nonTranspFifo, but with multi-write single-read

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity discard_FIFO is
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
        readyArray    : out std_logic_vector(0 downto 0);
        ----------
        --tmpSlots      : out natural range 0 to FIFO_DEPTH + FIFO_DEPTH;
        writeCount    : in  natural range 0 to FIFO_DEPTH + FIFO_DEPTH
    );
end discard_FIFO;

architecture arch of discard_FIFO is

    signal Tail     : natural range 0 to FIFO_DEPTH - 1;
    signal Head     : natural range 0 to FIFO_DEPTH - 1;
    signal ReadEn   : std_logic := '0';
    signal WriteEn  : std_logic := '0';
    signal Empty    : std_logic;
    signal Full     : std_logic;
    signal Overflow : std_logic;
    signal fifo_ready : std_logic;
    signal emptySlots : natural range 0 to FIFO_DEPTH + FIFO_DEPTH;
    --signal filledSlots : natural range 0 to FIFO_DEPTH + FIFO_DEPTH;
    type FIFO_Memory is array (0 to FIFO_DEPTH - 1) of std_logic_vector(DATA_SIZE_IN - 1 downto 0);
    signal Memory : FIFO_Memory;

begin
    
    -- For debugging only
    --tmpSlots <= filledSlots;

    -- Check for overflow by comparing writecount
    Overflow        <= '1' when (writeCount > emptySlots) else '0';
    -- FIFO ready conventionally only if no overflow
    fifo_ready      <= not Overflow and (not Full or nReadyArray(0));
    readyArray(0)   <= fifo_ready;

    -- Valid read if FIFO is not empty
    validArray(0)   <= not Empty;
    dataOutArray(0) <= Memory(Head);

    -- Read if there is sth in FIFO, and next is ready to accept
    ReadEn          <= nReadyArray(0) and not Empty;
    -- Write if there is valid input and the FIFO is ready
    WriteEn         <= pValidArray(0) and fifo_ready;    

    -------------------------------------------
    -- write process
    fifo_proc : process (clk) begin        
        if rising_edge(clk) then
            if rst = '1' then

            else
                if (WriteEn = '1') then
                    -- Write Data to Memory
                    for I in 0 to (writeCount - 1) loop
                        -- write to Tail + I instead of just Tail
                        -- need to allow wrap-around too
                        Memory((Tail + I) mod FIFO_DEPTH) <= dataInArray(0);
                    end loop;
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
                    Tail <= (Tail + writeCount) mod FIFO_DEPTH;
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

    ------------------------------------------- check ???
    -- process for updating Full
    FullUpdate_proc : process (clk) begin
        if rising_edge(clk) then
            if rst = '1' then
                Full <= '0';
            else
                -- if only filling but not emptying
                if (WriteEn = '1') and (ReadEn = '0') then
                    -- if new tail index will reach head index
                    if ((Tail + writeCount) mod FIFO_DEPTH = Head) then
                        Full <= '1';
                    end if;

                -- if filling and emptying simultaneously
                elsif (WriteEn = '1') and (ReadEn = '1') then
                    -- if new tail index will reach head index
                    if ((Tail + writeCount - 1) mod FIFO_DEPTH = Head) then
                        Full <= '1';
                    end if;

                -- if only emptying but not filling
                elsif (WriteEn = '0') and (ReadEn = '1') then
                    Full <= '0';
                -- otherwise, nothing is happenning
                end if;
            end if;
        end if;
    end process;

    ------------------------------------------- check ???
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

                -- if filling and emptying simultaneously
                elsif (WriteEn = '1') and (ReadEn = '1') then
                    -- if writing more values than reading
                    if (writeCount > 1) then
                        Empty <= '0';
                    end if;

                -- if only filling but not emptying
                elsif (WriteEn = '1') and (ReadEn = '0') then
                    Empty <= '0';

                -- otherwise, nothing is happenning
                end if;       
            end if;
        end if;
    end process;

    -------------------------------------------
    -- Count filled slots (necessary internally)
    CalculateSlots_proc: process (Head, Tail, Empty) begin
        if (Head < Tail or Empty = '1') then 
            emptySlots <= (Head - Tail + FIFO_DEPTH);
            --filledSlots <= (Tail - Head);
        else
            emptySlots <= (Head - Tail);
            --filledSlots <= (FIFO_DEPTH + Tail - Head);
        end if;
    end process;

end arch;

-----------------------------------------------------------------------------------------
-------------------------------------------------------------- FIFO for load_op spec bits
-----------------------------------------------------------------------------------------
-- Same logic as nontranspFIFO, except needs send = 1 to read

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity load_spec_FIFO is
    generic (
        INPUTS        : integer;  -- assuming always 2: 1 input, 1 send signal
        OUTPUTS       : integer;  -- assuming always 1 output
        DATA_SIZE_IN  : integer;  -- use normal data size, eg- 32
        DATA_SIZE_OUT : integer;  -- use normal data size, eg- 32
        FIFO_DEPTH    : integer   -- extra parameter for FIFO
    );
    port (
        clk, rst      : in  std_logic;
        dataInArray   : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        nReadyArray   : in  std_logic_vector(0 downto 0);  -- (dout)
        validArray    : out std_logic_vector(0 downto 0);  -- (dout)
        pValidArray   : in  std_logic_vector(1 downto 0);  -- (send, din)
        readyArray    : out std_logic_vector(1 downto 0);  -- (send, din)
        ----------
        --filledSlots   : out natural range 0 to FIFO_DEPTH + FIFO_DEPTH;
        send          : in  std_logic
    );
end load_spec_FIFO;
 
architecture arch of load_spec_FIFO is

    signal tReadEn : std_logic := '0';
    signal ReadEn  : std_logic := '0';
    signal WriteEn : std_logic := '0';
    signal Tail    : natural range 0 to FIFO_DEPTH - 1;
    signal Head    : natural range 0 to FIFO_DEPTH - 1;
    signal Empty   : std_logic;
    signal Full    : std_logic;
    --signal emptySlots : natural range 0 to FIFO_DEPTH + FIFO_DEPTH;
    type FIFO_Memory is array (0 to FIFO_DEPTH - 1) of std_logic_vector(DATA_SIZE_IN - 1 downto 0);
    signal Memory : FIFO_Memory;

begin
    
    -- Ready to write if there is space in the FIFO
    readyArray(0) <= not Full;
    -- FIFO control always ready
    readyArray(1) <= '1';

    -- Valid read if FIFO is not empty
    validArray(0) <= not Empty;
    dataOutArray(0) <= Memory(Head);

    -- Conventional read if next can accept and there is sth in fifo
    tReadEn <= nReadyArray(0) and not Empty;
    -- Actual read only if there is a valid send = 1
    ReadEn  <= tReadEn and pValidArray(1) and send;
    -- Write if there is valid input and there is space in the FIFO
    WriteEn <= pValidArray(0) and not Full;

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

    ------------------------------------------- check ???
    -- process for updating Full
    FullUpdate_proc : process (clk) begin
        if rising_edge(clk) then
            if rst = '1' then
               Full <= '0';
            else
                -- if only filling but not emptying
                if (WriteEn = '1') and (ReadEn = '0') then
                    -- if new tail index will reach head index
                    if ((Tail +1) mod FIFO_DEPTH = Head) then
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

    ------------------------------------------- check ???
    -- process for updating Empty
    EmptyUpdate_proc : process (clk) begin
        if rising_edge(clk) then
            if rst = '1' then
               Empty <= '1';
            else
                -- if only emptying but not filling
                if (WriteEn = '0') and (ReadEn = '1') then
                    -- if new head index will reach tail index
                    if ((Head +1) mod FIFO_DEPTH = Tail) then
                        Empty <= '1';
                    else
                        Empty <= '0';
                    end if;
                -- if only filling but not emptying
                elsif (WriteEn = '1') and (ReadEn = '0') then
                    Empty <= '0';

                -- otherwise, nothing is happenning or simultaneous read and write
                end if;       
            end if;
        end if;
    end process;

    -------------------------------------------
    -- Count filled slots (for debugging only)
    --CalculateSlots_proc: process (Head, Tail, Empty) begin
    --    if (Head < Tail or Empty = '1') then 
    --        --emptySlots <= (Head - Tail + FIFO_DEPTH);
    --        filledSlots <= (Tail - Head);
    --    else
    --        --emptySlots <= (Head - Tail);
    --        filledSlots <= (FIFO_DEPTH + Tail - Head);
    --    end if;
    --end process;

end arch;