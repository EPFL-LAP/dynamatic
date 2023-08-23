library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity merge_notehb is

  generic (
    INPUTS   : integer;
    BITWIDTH : integer;
  );
  port (
    clk, rst     : in std_logic;
    dataInArray  : in data_array(INPUTS - 1 downto 0)(BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(BITWIDTH - 1 downto 0);
    pValidArray  : in std_logic_vector(INPUTS - 1 downto 0);
    nReady       : in std_logic;
    valid        : out std_logic;
    readyArray   : out std_logic_vector(INPUTS - 1 downto 0));
end merge_notehb;

architecture arch of merge_notehb is
  signal tehb_data_in : std_logic_vector(BITWIDTH - 1 downto 0);
  signal tehb_pvalid  : std_logic;
  signal tehb_ready   : std_logic;

begin

  process (pValidArray, dataInArray)
    variable tmp_data_out  : unsigned(BITWIDTH - 1 downto 0);
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

    tehb_data_in <= std_logic_vector(resize(tmp_data_out, BITWIDTH));
    tehb_pvalid  <= tmp_valid_out;

  end process;

  process (tehb_ready)
  begin
    for I in 0 to INPUTS - 1 loop
      readyArray(I) <= tehb_ready;
    end loop;
  end process;

  tehb_ready   <= nReady;
  valid        <= tehb_pvalid;
  dataOutArray <= tehb_data_in;

end arch;
