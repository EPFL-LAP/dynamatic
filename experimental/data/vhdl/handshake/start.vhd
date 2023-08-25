library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.customTypes.all;
entity start_node is

  generic (
    BITWIDTH : integer
  );

  port (
    -- inputs
    ins        : in std_logic_vector(BITWIDTH - 1 downto 0);
    ins_valid  : in std_logic;
    clk        : in std_logic;
    rst        : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    ins_ready  : out std_logic;
    outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
    outs_valid : out std_logic);
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

  startBuff : entity work.buffer(arch) generic map (BITWIDTH)
    port map(
      clk        => clk,
      rst        => rst,
      ins        => ins,
      ins_valid  => start_internal,
      outs_ready => outs_ready,
      outs       => startBuff_dataOutArray,
      ins_ready  => startBuff_readyArray,
      outs_valid => startBuff_validArray
    );

  outs_valid <= startBuff_validArray;
  outs       <= startBuff_dataOutArray;
  ins_ready  <= startBuff_readyArray;

end arch;
