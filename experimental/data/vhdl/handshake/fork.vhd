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

entity fork is generic(OUTPUTS : integer; BITWIDTH : Integer);
port(   clk, rst    : in std_logic; -- the eager implementation uses registers
        dataInArray : in data_array (0 downto 0)(BITWIDTH-1 downto 0);
        pValidArray : in std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(0 downto 0);
        dataOutArray : out data_array (OUTPUTS-1 downto 0)(BITWIDTH-1 downto 0); 
        nReadyArray : in std_logic_vector(OUTPUTS-1 downto 0);
        validArray  : out std_logic_vector(OUTPUTS-1 downto 0)
        );
        
end fork;


------------------------------------------------------------------------
-- generic eager implementation
------------------------------------------------------------------------
architecture arch of fork is
-- wrapper signals (internals use "stop" signals instead of "ready" signals)
    signal forkStop : std_logic;
    signal nStopArray : std_logic_vector(OUTPUTS-1 downto 0);
-- internal combinatorial signals
    signal blockStopArray : std_logic_vector(OUTPUTS-1 downto 0);
    signal anyBlockStop : std_logic;
    signal pValidAndForkStop : std_logic;
begin
    
    --can't adapt the signals directly in port map
    wrapper : process(forkStop, nReadyArray)
    begin
        readyArray(0) <= not forkStop;
        for i in 0 to OUTPUTS-1 loop
            nStopArray(i) <= not nReadyArray(i);
        end loop;
    end process;
    
    genericOr : entity work.orN generic map (OUTPUTS)
        port map(blockStopArray, anyBlockStop);
        
    -- internal combinatorial signals
    forkStop <= anyBlockStop; 
    pValidAndForkStop <= pValidArray(0) and forkStop;
    
    --generate blocks
    generateBlocks : for i in OUTPUTS-1 downto 0 generate
        regblock : entity work.eagerFork_RegisterBLock(arch)
                port map(   clk, rst,
                            pValidArray(0), nStopArray(i),
                            pValidAndForkStop,
                            validArray(i), blockStopArray(i));
    end generate;

    process(dataInArray)
    begin
        for I in 0 to OUTPUTS - 1 loop
            dataOutArray(I) <= dataInArray(0);
        end loop;  
    end process;   

end arch;