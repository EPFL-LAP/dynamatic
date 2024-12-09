------------------------------------------------------------------------
-- elasticFifoInner

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

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
-- transpFifo ----> SPEC version

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

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
-- TEHB ----> SPEC version

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity TEHB_old is 
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
end TEHB_old;

architecture arch of TEHB_old is
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
use work.types.all;

entity OEHB_old is 
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
end OEHB_old;

architecture arch of OEHB_old is
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
use work.types.all;

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

    tehb1: entity work.TEHB_old(arch) generic map (1, 1, DATA_SIZE_IN+1, DATA_SIZE_IN+1)
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

    oehb1: entity work.OEHB_old(arch) generic map (1, 1, DATA_SIZE_IN+1, DATA_SIZE_IN+1)
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
-- Merge ----> SPEC version

library ieee;
use ieee.std_logic_1164.all;
use work.types.all;
use ieee.numeric_std.all;

entity merge_old is 
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
end merge_old;

architecture arch of merge_old is

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

    tehb1: entity work.TEHB_old(arch) generic map (1, 1, DATA_SIZE_IN+1, DATA_SIZE_IN+1)
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
-- Simple join

library ieee;
use ieee.std_logic_1164.all;

entity join_old is generic (SIZE : integer);
    port (
        pValidArray     : in  std_logic_vector(SIZE - 1 downto 0);
        nReady          : in  std_logic;
        valid           : out std_logic;
        readyArray      : out std_logic_vector(SIZE - 1 downto 0)
    );   
    end join_old;

architecture arch of join_old is
    signal allPValid : std_logic;
begin

    allPValidAndGate : entity work.and_n generic map(SIZE)
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
use work.types.all;

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
-- EagerFork ----> SPEC version

library ieee;
use ieee.std_logic_1164.all;
use work.types.all;

entity fork_old is 
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
end fork_old;

-- generic eager implementation uses registers
------------------------------------------------------
architecture arch of fork_old is
    
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

    genericOr : entity work.or_n generic map (OUTPUTS)
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
