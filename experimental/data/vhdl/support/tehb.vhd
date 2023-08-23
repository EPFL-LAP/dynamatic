library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity TEHB is
  generic (
    BITWIDTH : integer
  );
  port (
    clk, rst     : in std_logic;
    dataInArray  : in std_logic_vector(BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(BITWIDTH - 1 downto 0);
    pValid       : in std_logic;
    nReady       : in std_logic;
    valid        : out std_logic;
    ready        : out std_logic);
end TEHB;

architecture arch of TEHB is
  signal full_reg, reg_en, mux_sel : std_logic;
  signal data_reg                  : std_logic_vector(BITWIDTH - 1 downto 0);
begin

  process (clk, rst) is

  begin
    if (rst = '1') then
      full_reg <= '0';

    elsif (rising_edge(clk)) then
      full_reg <= valid and not nReady;

    end if;
  end process;

  process (clk, rst) is

  begin
    if (rst = '1') then
      data_reg <= (others => '0');

    elsif (rising_edge(clk)) then
      if (reg_en) then
        data_reg <= dataInArray;
      end if;

    end if;
  end process;

  process (mux_sel, data_reg, dataInArray) is
  begin
    if (mux_sel = '1') then
      dataOutArray <= data_reg;
    else
      dataOutArray <= dataInArray;
    end if;
  end process;
  valid   <= pValid or full_reg;
  ready   <= not full_reg;
  reg_en  <= ready and pValid and not nReady;
  mux_sel <= full_reg;
end arch;
