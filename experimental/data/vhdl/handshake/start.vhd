library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.customTypes.all;
entity start_node is

  generic (
    BITWIDTH : integer
  );

  port (
    clk, rst     : in std_logic;
    dataInArray  : in std_logic_vector(BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(BITWIDTH - 1 downto 0);
    ready        : out std_logic;
    valid        : out std_logic;
    nReady       : in std_logic;
    pValid       : in std_logic
  );
end start_node;

architecture arch of start_node is

  signal set                    : std_logic;
  signal start_internal         : std_logic;
  signal startBuff_readyArray   : std_logic;
  signal startBuff_validArray   : std_logic;
  signal startBuff_dataOutArray : std_logic_vector(BITWIDTH - 1 downto 0);

begin

  process (clk, rst)
  begin

    if (rst = '1') then
      start_internal <= '0';
      set            <= '0';

    elsif rising_edge(clk) then
      if (pValid = '1' and set = '0') then
        start_internal <= '1';
        set            <= '1';
      else
        start_internal <= '0';
      end if;
    end if;
  end process;

  startBuff : entity work.elasticBuffer(arch) generic map (BITWIDTH)
    port map(
      --inputs
      clk         => clk,            --clk
      rst         => rst,            --rst
      dataInArray => dataInArray,    ----dataInArray
      pValid      => start_internal, --pValid
      nReady      => nReady,         --nReady
      --outputs
      dataOutArray => startBuff_dataOutArray, ----dataOutArray
      ready        => startBuff_readyArray,   --readyArray
      valid        => startBuff_validArray    --validArray
    );

  valid        <= startBuff_validArray;
  dataOutArray <= startBuff_dataOutArray;
  ready        <= startBuff_readyArray;

end arch;
