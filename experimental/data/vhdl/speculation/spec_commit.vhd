library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity spec_commit is
    generic (
        DATA_SIZE_IN  : integer;  -- use normal data size, eg- 32
        DATA_SIZE_OUT : integer   -- use normal data size, eg- 32
    );
    port (
        clk, rst      : in  std_logic;

        dataInArray   : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        specInArray   : in  data_array(0 downto 0)(0 downto 0);
        ControlInArray  : in  data_array(0 downto 0)(0 downto 0);
        pValidArray   : in  std_logic_vector(1 downto 0); -- (control, data)
        readyArray    : out std_logic_vector(1 downto 0); -- (control, data)

        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        validArray    : out std_logic_vector(0 downto 0);
        nReadyArray   : in  std_logic_vector(0 downto 0)
        ----------
        
    );
end spec_commit;

architecture arch of spec_commit is

signal fifo_disc_dataIn : data_array(0 downto 0)(0 downto 0);
signal fifo_disc_dataOut : data_array(0 downto 0)(0 downto 0);
signal fifo_disc_ready, fifo_disc_pValid : std_logic_vector(0 downto 0);
signal fifo_disc_valid, fifo_disc_nReady : std_logic_vector(0 downto 0);

signal branch_in_dataIn : data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal branch_in_dataOut : data_array(1 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal branch_in_ready, branch_in_pValid : std_logic_vector(1 downto 0);
signal branch_in_valid, branch_in_nReady : std_logic_vector(1 downto 0);
signal branch_in_condn : std_logic;

signal buff_dataIn : data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal buff_dataOut : data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal buff_ready, buff_pValid : std_logic_vector(0 downto 0);
signal buff_valid, buff_nReady : std_logic_vector(0 downto 0);

signal branch_disc_dataIn : data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal branch_disc_dataOut : data_array(1 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal branch_disc_ready, branch_disc_pValid : std_logic_vector(1 downto 0);
signal branch_disc_valid, branch_disc_nReady : std_logic_vector(1 downto 0);
signal branch_disc_condn : std_logic;

signal merge_out_dataIn : data_array(1 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal merge_out_dataOut : data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal merge_out_ready, merge_out_pValid : std_logic_vector(1 downto 0);
signal merge_out_valid, merge_out_nReady : std_logic_vector(0 downto 0);

signal unconnected_spec_2 : data_array(1 downto 0)(0 downto 0);
signal unconnected_spec   : data_array(0 downto 0)(0 downto 0);

begin

-- Design taken directly from the Speculation 2019 paper

--------------------------------------------------
readyArray(0) <= branch_in_ready(0); -- data ready
readyArray(1) <= fifo_disc_ready(0); -- control ready
--------------------------------------------------

fifo_disc_dataIn(0) <= ControlInArray(0);
fifo_disc_pValid(0) <= pValidArray(1);  -- discard pValid
fifo_disc_nReady(0) <= branch_disc_ready(1); -- cond
fifo_disc: entity work.transpFIFO(arch) generic map (1, 1, 1, 1, 16)
    port map (
        clk => clk,
        rst => rst,
        dataInArray         => fifo_disc_dataIn,
        specInArray(0)(0)   => '0',
        dataOutArray        => fifo_disc_dataOut,
        specOutArray        => unconnected_spec,
        pValidArray         => fifo_disc_pValid,
        nReadyArray         => fifo_disc_nReady,
        validArray          => fifo_disc_valid,
        readyArray          => fifo_disc_ready
    );

branch_in_condn <= specInArray(0)(0);  -- actual condition
branch_in_dataIn(0) <= dataInArray(0);
branch_in_pValid <= ('1', pValidArray(0)); -- (cond, data) -- cond always valid
branch_in_nReady <= (buff_ready(0), merge_out_ready(0));
branch_in: entity work.branch(arch) generic map(1, 2, DATA_SIZE_IN, DATA_SIZE_IN)
    port map (
        clk => clk,
        rst => rst,
        condition(0)(0)     => not branch_in_condn, -- condn internally flipped
        dataInArray         => branch_in_dataIn,
        specInArray(1)(0)   => '0',
        specInArray(0)(0)   => '0',
        dataOutArray        => branch_in_dataOut,
        specOutArray        => unconnected_spec_2,
        pValidArray         => branch_in_pValid,
        nReadyArray         => branch_in_nReady,
        validArray          => branch_in_valid,
        readyArray          => branch_in_ready  -- (cond, data)
    );

buff_dataIn(0) <= branch_in_dataOut(1);
buff_pValid(0) <= branch_in_valid(1);
buff_nReady(0) <= branch_disc_ready(0); -- data
buff: entity work.elasticBuffer(arch) generic map (1, 1, DATA_SIZE_IN, DATA_SIZE_IN)
    port map (
        clk => clk,
        rst => rst,
        dataInArray         => buff_dataIn,
        specInArray(0)(0)   => '0',
        dataOutArray        => buff_dataOut,
        specOutArray        => unconnected_spec,
        pValidArray         => buff_pValid,
        nReadyArray         => buff_nReady,
        validArray          => buff_valid,
        readyArray          => buff_ready
    );

branch_disc_condn <= fifo_disc_dataOut(0)(0);  -- actual condition
branch_disc_dataIn(0) <= buff_dataOut(0);
branch_disc_pValid <= (fifo_disc_valid(0), buff_valid(0)); -- (cond, data)
branch_disc_nReady <= ('1', merge_out_ready(1));  -- value (1) sinks
branch_disc: entity work.branch(arch) generic map(1, 2, DATA_SIZE_IN, DATA_SIZE_IN)
    port map (
        clk => clk,
        rst => rst,
        condition(0)(0)     => not branch_disc_condn, -- condn internally flipped
        dataInArray         => branch_disc_dataIn,
        specInArray(1)(0)   => '0',
        specInArray(0)(0)   => '0',
        dataOutArray        => branch_disc_dataOut,
        specOutArray        => unconnected_spec_2,
        pValidArray         => branch_disc_pValid,
        nReadyArray         => branch_disc_nReady,
        validArray          => branch_disc_valid,
        readyArray          => branch_disc_ready  -- (cond, data)
    );

merge_out_dataIn <= (branch_disc_dataOut(0), branch_in_dataOut(0));
merge_out_pValid <= (branch_disc_valid(0), branch_in_valid(0));
merge_out_nReady(0) <= nReadyArray(0);  -- data nReady
merge_out: entity work.merge_old(arch) generic map (2, 1, DATA_SIZE_IN, DATA_SIZE_IN)
    port map (
        clk => clk,
        rst => rst,
        dataInArray         => merge_out_dataIn,
        specInArray(0)(0)   => '0',
        specInArray(1)(0)   => '0',
        dataOutArray        => merge_out_dataOut,
        specOutArray        => unconnected_spec,
        pValidArray         => merge_out_pValid,
        nReadyArray         => merge_out_nReady,
        validArray          => merge_out_valid,
        readyArray          => merge_out_ready
    );

--------------------------------------------------
validArray(0) <= merge_out_valid(0); -- data valid
dataOutArray(0) <= merge_out_dataOut(0);
--------------------------------------------------

end arch;