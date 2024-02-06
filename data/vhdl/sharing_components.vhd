-------------------------------------------------------------  branchGen
---------------------------------------------------------------------
library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
USE work.customTypes.all;

entity branchGen is generic( INPUTS:integer; OUTPUTS : integer; COND_SIZE : integer; DATA_SIZE_IN : integer;DATA_SIZE_OUT : integer);
port(
    clk, rst : in std_logic;
    pValidArray         : in std_logic_vector(1 downto 0);
    condition: in data_array (0 downto 0)(COND_SIZE - 1 downto 0);
    dataInArray          : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0);
    dataOutArray         : out data_array (OUTPUTS-1 downto 0)(DATA_SIZE_OUT-1 downto 0);
    nReadyArray     : in std_logic_vector(OUTPUTS - 1 downto 0); 
    validArray      : out std_logic_vector(OUTPUTS -1 downto 0); 
    readyArray      : out std_logic_vector(1 downto 0));    -- (data, condition)
end branchGen;


architecture arch of branchGen is 
    signal joinValid, branchReady   : std_logic;
    --signal dataOut0, dataOut1 : std_logic_vector(31 downto 0);
begin

    j : entity work.join(arch) generic map(2)
            port map(   (pValidArray(1), pValidArray(0)),
                        branchReady,
                        joinValid,
                        readyArray);

    branchReady <= nReadyArray(to_integer(unsigned(condition(0))));

    gen_validArray: for I in 0 to OUTPUTS - 1 generate
        validArray(I) <= joinValid when condition(0) = std_logic_vector(to_unsigned(I, condition(0)'length)) else '0';
    end generate gen_validArray;

    gen_dataOutArray: for I in 0 to OUTPUTS - 1 generate
        dataOutArray(I) <= dataInArray(0);
    end generate gen_dataOutArray;

end arch;

-------------------------------------------------------------  distributor
---------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity merge_one_hot is

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
end merge_one_hot;

architecture arch of merge_one_hot is
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

    process(tehb_ready, pValidArray)
        variable some_input_valid : std_logic;
    begin
        some_input_valid := '0';
        for I in 0 to INPUTS - 1 loop
            readyArray(I) <= tehb_ready and (not some_input_valid);
            if pValidArray(I) = '1' then
                some_input_valid := '1';
            end if;
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
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
USE work.customTypes.all;

entity distributor is generic( INPUTS:integer; OUTPUTS : integer; COND_SIZE : integer; DATA_SIZE_IN : integer;DATA_SIZE_OUT : integer);
port(
   clk, rst : in std_logic;
   pValidArray         : in std_logic_vector(1 downto 0);
   condition: in data_array (0 downto 0)(COND_SIZE - 1 downto 0);
   dataInArray          : in data_array (0 downto 0)(DATA_SIZE_IN-1 downto 0);
   dataOutArray         : out data_array (OUTPUTS-1 downto 0)(DATA_SIZE_OUT-1 downto 0);
   nReadyArray     : in std_logic_vector(OUTPUTS - 1 downto 0); 
   validArray      : out std_logic_vector(OUTPUTS -1 downto 0); 
   readyArray      : out std_logic_vector(1 downto 0));    -- (data, condition)
end distributor;

architecture arch of distributor is 
   --signal dataOut0, dataOut1 : std_logic_vector(31 downto 0);
   signal branchDataOutArray : data_array (OUTPUTS-1 downto 0)(DATA_SIZE_OUT-1 downto 0);
   signal branchValidArray   : std_logic_vector(OUTPUTS -1 downto 0); 
   signal buffersReadyArray  : std_logic_vector(OUTPUTS - 1 downto 0);

begin

   inner_branch : entity work.branchGen(arch) generic map(INPUTS, OUTPUTS, COND_SIZE, DATA_SIZE_IN, DATA_SIZE_OUT)
           port map(clk => clk,
                    rst => rst,
                    pValidArray => pValidArray,
                    condition => condition,
                    dataInArray => dataInArray,
                    dataOutArray => branchDataOutArray,
                    nReadyArray  => buffersReadyArray,
                    validArray   => branchValidArray,
                    readyArray   => readyArray
           );

   buf_gen : for I in 0 to OUTPUTS - 1 generate
       buff : entity work.TEHB(arch) generic map (1, 1, DATA_SIZE_IN, DATA_SIZE_IN)
           port map (
           --inputs
               clk => clk,  --clk
               rst => rst,  --rst
               dataInArray(0) => branchDataOutArray(I),  ----dataInArray
               pValidArray(0) => branchValidArray(I),   --pValidArray
               nReadyArray(0) => nReadyArray(I),  --nReadyArray
           --outputs
               dataOutArray(0) => dataOutArray(I),    ----dataOutArray
               readyArray(0) => buffersReadyArray(I),  --readyArray
               validArray(0) => validArray(I)   --validArray
           );
   end generate;

end arch;

library ieee;
use ieee.std_logic_1164.all;
package selectorTypes is

    subtype shared_component_id is std_logic_vector;
    type bb_ordering is array(natural range <>) of shared_component_id;
    type ordering_t is array(natural range <>) of bb_ordering;

end package;

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
USE work.customTypes.all;

entity bypassFIFO is 
    generic(
        INPUTS        : integer;
        OUTPUTS        : integer;
        DATA_SIZE_IN  : integer;
        DATA_SIZE_OUT : integer;
        FIFO_DEPTH : integer
    );
port (
        clk, rst      : in  std_logic;
        dataInArray   : in  data_array(0 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        pValidArray   : in  std_logic_vector(INPUTS - 1 downto 0);
        nReadyArray   : in  std_logic_vector(0 downto 0);
        validArray    : out std_logic_vector(0 downto 0);
        readyArray    : out std_logic_vector(INPUTS - 1 downto 0));
end bypassFIFO;

architecture arch of bypassFIFO is
    signal forwarding : boolean;
    signal fifo_valid : std_logic;
    signal fifo_out   : std_logic_vector(DATA_SIZE_IN-1 downto 0);
begin
    
    forwarding <= fifo_valid = '0'and pValidArray(0) = '1';

    validArray(0)   <= pValidArray(0) when forwarding else fifo_valid;
    dataOutArray(0) <= dataInArray(0) when forwarding else fifo_out;

    fifo_comp: entity work.transpFIFO(arch) generic map (1, 1, DATA_SIZE_IN, DATA_SIZE_OUT, FIFO_DEPTH)
        port map (
            clk => clk, 
            rst => rst, 
            pValidArray  => pValidArray, 
            nReadyArray => nReadyArray,    
            dataInArray => dataInArray,
            validArray(0) => fifo_valid, 
            readyArray => readyArray,   
            dataOutArray(0) => fifo_out
        );

end arch;

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
USE work.customTypes.all;
use work.selectorTypes.all;

-- round-robin Merge to control function sharing
entity input_selector is 
Generic (
 AMOUNT_OF_BB_IDS: integer; BB_ID_INFO_SIZE : integer; BB_COUNT_INFO_SIZE : integer; AMOUNT_OF_SHARED_COMPONENT : integer; SHARED_COMPONENT_ID_SIZE : integer
);
port (
    clk, rst : in std_logic;    
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    dataOutArray  : out std_logic_vector(SHARED_COMPONENT_ID_SIZE - 1 downto 0);
    --ordering for each BB, constants
    bbOrderingData   : in ordering_t(AMOUNT_OF_BB_IDS -1 downto 0)(AMOUNT_OF_SHARED_COMPONENT - 1 downto 0)(SHARED_COMPONENT_ID_SIZE - 1 downto 0);
    --info (id and amount) for each B. From MSB to LSB the order is max_amount and then ID
    bbInfoData   : in data_array(AMOUNT_OF_BB_IDS -1 downto 0)(BB_ID_INFO_SIZE  + BB_COUNT_INFO_SIZE - 1 downto 0);
    bbInfoPValid : in std_logic_vector(AMOUNT_OF_BB_IDS - 1 downto 0);
    bbInfoReady : out std_logic_vector(AMOUNT_OF_BB_IDS - 1 downto 0)   
);
end input_selector;

architecture arch of input_selector is

    constant BB_INFO_SIZE : integer := BB_ID_INFO_SIZE + BB_COUNT_INFO_SIZE;

    signal transaction_at_output : boolean;
    signal last_index_reached    : boolean;
    signal ordering_traversed    : boolean;

    signal bb_index       : unsigned(BB_ID_INFO_SIZE    - 1 downto 0);
    signal max_elem_index : unsigned(BB_COUNT_INFO_SIZE - 1 downto 0);

    signal merge_output_valid_array : std_logic_vector(0 downto 0);
    signal merge_output_ready_array : std_logic_vector(0 downto 0);
    signal merge_output_data_array  : data_array(0 downto 0)(BB_INFO_SIZE - 1 downto 0);

    signal fifo_output_valid_array : std_logic_vector(0 downto 0);
    signal fifo_output_ready_array : std_logic_vector(0 downto 0);
    signal fifo_output_data_array  : data_array(0 downto 0)(BB_INFO_SIZE - 1 downto 0);

    signal reg_current_index : unsigned(BB_COUNT_INFO_SIZE - 1 downto 0);

begin

    transaction_at_output <= validArray = "1" and nReadyArray = "1";
    last_index_reached <= reg_current_index = max_elem_index;
    ordering_traversed <= last_index_reached and transaction_at_output;

    bb_index       <= unsigned(fifo_output_data_array(0)(BB_ID_INFO_SIZE - 1 downto 0));
    max_elem_index <= unsigned(fifo_output_data_array(0)(BB_INFO_SIZE - 1 downto BB_INFO_SIZE - BB_COUNT_INFO_SIZE));

    validArray   <= fifo_output_valid_array;
    dataOutArray <= bbOrderingData(to_integer(bb_index))(to_integer(reg_current_index));
    fifo_output_ready_array <= "1" when ordering_traversed else "0";
    
    reg_proc : process(clk, rst)
    begin
        if rst = '1' then
            reg_current_index <= (others => '0');
        elsif rising_edge(clk) then
            if ordering_traversed then
                reg_current_index <= (others => '0');
            elsif transaction_at_output then
                reg_current_index <= reg_current_index + 1;
            end if;
        end if;
    end process ;

    merge_comp: entity work.merge_one_hot(arch) generic map (AMOUNT_OF_BB_IDS, 1, BB_INFO_SIZE, BB_INFO_SIZE)
        port map (
            clk => clk, 
            rst => rst, 
            pValidArray  => bbInfoPValid, 
            nReadyArray => merge_output_ready_array,    
            dataInArray => bbInfoData,
            validArray => merge_output_valid_array, 
            readyArray => bbInfoReady,   
            dataOutArray => merge_output_data_array
        );

    --gigantic fifo depth for now
    fifo_comp: entity work.bypassFIFO(arch) generic map (1, 1, BB_INFO_SIZE, BB_INFO_SIZE, 128)
        port map (
            clk => clk, 
            rst => rst, 
            pValidArray  => merge_output_valid_array, 
            nReadyArray => fifo_output_ready_array,    
            dataInArray => merge_output_data_array,
            validArray => fifo_output_valid_array, 
            readyArray => merge_output_ready_array,   
            dataOutArray => fifo_output_data_array
        );


end arch;

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use work.selectorTypes.all;

-- round-robin Merge to control function sharing
entity selector is 
Generic (
    INPUTS : integer ; OUTPUTS : integer; COND_SIZE : integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer;
    AMOUNT_OF_BB_IDS: integer; AMOUNT_OF_SHARED_COMPONENT : integer; BB_ID_INFO_SIZE : integer; BB_COUNT_INFO_SIZE : integer
);
port (
    clk, rst : in std_logic;    
    pValidArray : in std_logic_vector(INPUTS - 1 downto 0);
    nReadyArray : in std_logic_vector(OUTPUTS - 1 downto 0);
    validArray : out std_logic_vector(OUTPUTS - 1 downto 0);
    readyArray : out std_logic_vector(INPUTS - 1 downto 0);
    dataInArray   : in  data_array(INPUTS - 1 downto 0)(DATA_SIZE_IN - 1 downto 0); 
    dataOutArray  : out data_array(OUTPUTS - 2 downto 0)(DATA_SIZE_OUT - 1 downto 0);
    condition: out data_array(0 downto 0)(COND_SIZE-1 downto 0);

    --ordering for each BB, constants
    bbOrderingData   : in ordering_t(AMOUNT_OF_BB_IDS -1 downto 0)(AMOUNT_OF_SHARED_COMPONENT - 1 downto 0)(COND_SIZE - 1 downto 0);
    --info (id and amount) for each B. From MSB to LSB the order is max_amount and then ID
    bbInfoData   : in data_array(AMOUNT_OF_BB_IDS -1 downto 0)(BB_ID_INFO_SIZE + BB_COUNT_INFO_SIZE - 1 downto 0);
    bbInfoPValid : in std_logic_vector(AMOUNT_OF_BB_IDS - 1 downto 0);
    bbInfoReady : out std_logic_vector(AMOUNT_OF_BB_IDS - 1 downto 0)  
);

end selector;

architecture arch of selector is

    constant BB_INFO_SIZE : integer := BB_ID_INFO_SIZE + BB_COUNT_INFO_SIZE;

    signal nReadyArray_input_selector   : std_logic_vector(0 downto 0);
    signal validArray_input_selector    : std_logic_vector(0 downto 0);
    signal dataOutArray_input_selector  : std_logic_vector(COND_SIZE - 1 downto 0);

    signal nReadyArray_fork   : std_logic_vector(2 downto 0);
    signal validArray_fork    : std_logic_vector(2 downto 0);
    signal dataOutArray_fork  : data_array(2 downto 0)(COND_SIZE - 1 downto 0);

    signal dataInArray_left_mux   : data_array((INPUTS / 2) - 1 downto 0)(DATA_SIZE_IN - 1 downto 0);
    signal condition_left_mux   : data_array(0 downto 0)(COND_SIZE - 1 downto 0);
    signal readyArray_left_mux   : std_logic_vector((INPUTS / 2) downto 0);
    signal pValidArray_left_mux   : std_logic_vector((INPUTS / 2) downto 0);

    signal dataInArray_right_mux   : data_array((INPUTS / 2) - 1 downto 0)(DATA_SIZE_IN - 1 downto 0);
    signal condition_right_mux   : data_array(0 downto 0)(COND_SIZE - 1 downto 0);
    signal readyArray_right_mux   : std_logic_vector((INPUTS / 2) downto 0);
    signal pValidArray_right_mux   : std_logic_vector((INPUTS / 2) downto 0);

begin

    gen_readyArray: for I in 0 to (INPUTS / 2) - 1 generate
        -- index + 1 is because of the fact that pValidArray_left_mux(0) etc is for the condition
        dataInArray_left_mux(I) <= dataInArray(2*I);
        pValidArray_left_mux(I + 1) <= pValidArray(2*I);
        readyArray(2*I)         <= readyArray_left_mux(I + 1);

        dataInArray_right_mux(I) <= dataInArray(2*I + 1);
        pValidArray_right_mux(I + 1) <= pValidArray(2*I + 1);
        readyArray(2*I + 1)          <= readyArray_right_mux(I + 1);
    end generate gen_readyArray;

    pValidArray_left_mux(0) <= validArray_fork(0);
    condition_left_mux(0)   <= dataOutArray_fork(0);
    nReadyArray_fork(0)     <= readyArray_left_mux(0);
    
    pValidArray_right_mux(0) <= validArray_fork(1);
    condition_right_mux(0)   <= dataOutArray_fork(1);
    nReadyArray_fork(1)     <= readyArray_right_mux(0);

    condition(0) <= dataOutArray_fork(2);
    validArray(2) <= validArray_fork(2);
    nReadyArray_fork(2) <= nReadyArray(2);

    input_selector : entity work.input_selector(arch) 
    generic map (
        AMOUNT_OF_BB_IDS         => AMOUNT_OF_BB_IDS,
        BB_ID_INFO_SIZE          => BB_ID_INFO_SIZE,
        BB_COUNT_INFO_SIZE       => BB_COUNT_INFO_SIZE,
        AMOUNT_OF_SHARED_COMPONENT => AMOUNT_OF_SHARED_COMPONENT,
        SHARED_COMPONENT_ID_SIZE => COND_SIZE
    )
    port map (
        clk => clk,
        rst => rst,   
        nReadyArray  => nReadyArray_input_selector,
        validArray   => validArray_input_selector,
        dataOutArray => dataOutArray_input_selector,
        --ordering for each BB, constants
        bbOrderingData  => bbOrderingData,
        --info (id and amount) for each BB
        bbInfoData   => bbInfoData,
        bbInfoPValid => bbInfoPValid,
        bbInfoReady  => bbInfoReady   
    );

    fork : entity work.fork(arch) 
    generic map (INPUTS => 1, SIZE => 3, DATA_SIZE_IN => COND_SIZE, DATA_SIZE_OUT => COND_SIZE)
    port map (
        --inputs
        clk => clk,  --clk
        rst => rst,  --rst
        pValidArray => validArray_input_selector, --pValidArray
        dataInArray(0) => dataOutArray_input_selector,
        nReadyArray => nReadyArray_fork, --nReadyArray
        --outputs
        dataOutArray => dataOutArray_fork,
        readyArray => nReadyArray_input_selector,   --readyArray
        validArray => validArray_fork    --validArray
    );

    left_mux : entity work.mux(arch)
    generic map(
        -- (INPUTS / 2) actual inputs + 1 input for the condition
        INPUTS        => (INPUTS / 2) + 1,
        OUTPUTS       => 1,
        DATA_SIZE_IN  => DATA_SIZE_IN,
        DATA_SIZE_OUT => DATA_SIZE_OUT,
        COND_SIZE     => COND_SIZE
    )
    port map(
        clk             => clk,
        rst             => rst,
        dataInArray     => dataInArray_left_mux,
        dataOutArray(0) => dataOutArray(0),
        pValidArray     => pValidArray_left_mux, 
        nReadyArray(0)  => nReadyArray(0),
        validArray(0)   => validArray(0),
        readyArray      => readyArray_left_mux,
        condition       => condition_left_mux     
    );

    right_mux : entity work.mux(arch)
    generic map(
        -- (INPUTS / 2) actual inputs + 1 input for the condition
        INPUTS        => (INPUTS / 2) + 1,
        OUTPUTS       => 1,
        DATA_SIZE_IN  => DATA_SIZE_IN,
        DATA_SIZE_OUT => DATA_SIZE_OUT,
        COND_SIZE     => COND_SIZE
    )
    port map(
        clk             => clk,
        rst             => rst,
        dataInArray     => dataInArray_right_mux,
        dataOutArray(0) => dataOutArray(1),
        pValidArray     => pValidArray_right_mux, 
        nReadyArray(0)  => nReadyArray(1),
        validArray(0)   => validArray(1),
        readyArray      => readyArray_right_mux,
        condition       => condition_right_mux     
    );

end arch;
