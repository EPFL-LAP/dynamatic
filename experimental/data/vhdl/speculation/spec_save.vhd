library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity spec_save is
    generic (
        DATA_SIZE_IN  : integer;  -- use normal data size, eg- 32
        DATA_SIZE_OUT : integer   -- use normal data size, eg- 32
    );
    port (
        clk, rst      : in  std_logic;

        dataInArray   : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        controlInArray: in  data_array(0 downto 0)(0 downto 0);
        pValidArray   : in  std_logic_vector(1 downto 0); -- (control, data)
        readyArray    : out std_logic_vector(1 downto 0); -- (control, data)

        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        validArray    : out std_logic_vector(0 downto 0);
        nReadyArray   : in  std_logic_vector(0 downto 0)
    );
end spec_save;

architecture arch of spec_save is

signal fork_in_dataIn : data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal fork_in_dataOut : data_array(1 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal fork_in_ready, fork_in_pValid : std_logic_vector(0 downto 0);
signal fork_in_valid, fork_in_nReady : std_logic_vector(1 downto 0);

signal buff_dataIn : data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal buff_dataOut : data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal buff_ready, buff_pValid : std_logic_vector(0 downto 0);
signal buff_valid, buff_nReady : std_logic_vector(0 downto 0);

signal branch_resend_dataIn : data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal branch_resend_dataOut : data_array(1 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal branch_resend_ready, branch_resend_pValid : std_logic_vector(1 downto 0);
signal branch_resend_valid, branch_resend_nReady : std_logic_vector(1 downto 0);
signal branch_resend_condn : std_logic;

signal merge_out_dataIn : data_array(1 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal merge_out_dataOut : data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
signal merge_out_ready, merge_out_pValid : std_logic_vector(1 downto 0);
signal merge_out_valid, merge_out_nReady : std_logic_vector(0 downto 0);

signal unconnected_spec_2 : data_array(1 downto 0)(0 downto 0);
signal unconnected_spec   : data_array(0 downto 0)(0 downto 0);



begin

--------------------------------------------------------
readyArray(0) <= fork_in_ready(0); -- data ready
readyArray(1) <= branch_resend_ready(1); -- control ready
--------------------------------------------------------

fork_in_dataIn(0) <= dataInArray(0);
fork_in_pValid(0) <= pValidArray(0); -- data pValid
fork_in_nReady <= (buff_ready(0), merge_out_ready(0));
fork_in: entity work.fork(arch) generic map(1, 2, DATA_SIZE_IN, DATA_SIZE_IN)
    port map (
        clk => clk,
        rst => rst,
        dataInArray         => fork_in_dataIn,
        specInArray(0)(0)   => '0',
        dataOutArray        => fork_in_dataOut,
        specOutArray        => unconnected_spec_2,
        pValidArray         => fork_in_pValid,
        nReadyArray         => fork_in_nReady,
        validArray          => fork_in_valid,
        readyArray          => fork_in_ready
    );

buff_dataIn(0) <= fork_in_dataOut(1);
buff_pValid(0) <= fork_in_valid(1);
buff_nReady(0) <= branch_resend_ready(0); -- data
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

branch_resend_condn <= controlInArray(0)(0);  -- actual condition
branch_resend_dataIn(0) <= buff_dataOut(0);
branch_resend_pValid <= (pValidArray(1), buff_valid(0)); -- (cond, data) -- control pValid
branch_resend_nReady <= (merge_out_ready(1), '1'); -- value (0) sinks
branch_resend: entity work.branch(arch) generic map(1, 2, DATA_SIZE_IN, DATA_SIZE_IN)
    port map (
        clk => clk,
        rst => rst,
        condition(0)(0)     => branch_resend_condn, -- condn internally flipped
        dataInArray         => branch_resend_dataIn,
        specInArray(1)(0)   => '0',
        specInArray(0)(0)   => '0',
        dataOutArray        => branch_resend_dataOut,
        specOutArray        => unconnected_spec_2,
        pValidArray         => branch_resend_pValid,
        nReadyArray         => branch_resend_nReady,
        validArray          => branch_resend_valid,
        readyArray          => branch_resend_ready  -- (cond, data)
    );

merge_out_dataIn <= (branch_resend_dataOut(1), fork_in_dataOut(0));
merge_out_pValid <= (branch_resend_valid(1), fork_in_valid(0));
merge_out_nReady(0) <= nReadyArray(0);  -- data nReady
merge_out: entity work.merge(arch) generic map (2, 1, DATA_SIZE_IN, DATA_SIZE_IN)
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
