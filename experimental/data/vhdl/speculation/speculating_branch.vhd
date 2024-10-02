library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity speculating_branch is 
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
        pValidArray  : in  std_logic_vector(1 downto 0);  -- (cond, data)
        readyArray   : out std_logic_vector(1 downto 0);  -- (cond, data)
        
        dataOutArray : out data_array(OUTPUTS - 1 downto 0)(DATA_SIZE_OUT - 1 downto 0);
        specOutArray : out data_array(OUTPUTS - 1 downto 0)(0 downto 0);
        validArray   : out std_logic_vector(1 downto 0);  -- (branch1, branch0)
        nReadyArray  : in  std_logic_vector(1 downto 0)   -- (branch1, branch0)
        
        
    );
end speculating_branch;

architecture arch of speculating_branch is

    signal joinValid, branchReady : std_logic;

    signal spec_bit : data_array(0 downto 0)(0 downto 0);
    signal specdataInArray  : data_array(0 downto 0)(DATA_SIZE_IN+1 - 1 downto 0);
    signal specdataOutArray : data_array(OUTPUTS - 1 downto 0)(DATA_SIZE_OUT+1 - 1 downto 0);

begin

    -- Speculative bit logic
    spec_bit(0)(0) <= specInArray(1)(0) or specInArray(0)(0);
    specdataInArray(0) <= spec_bit(0) & dataInArray(0);
    -----

    j : entity work.join(arch) generic map(2)
            port map(   (pValidArray(1), pValidArray(0)),
                        branchReady,
                        joinValid,
                        readyArray);

    br : entity work.branchSimple(arch)
            port map(   specInArray(1)(0),
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
