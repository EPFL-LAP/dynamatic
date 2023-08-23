library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity OEHB is
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
end OEHB;

architecture arch of OEHB is
  signal full_reg, reg_en, mux_sel : std_logic;
  signal data_reg                  : std_logic_vector(BITWIDTH - 1 downto 0);
begin

  process (clk, rst) is

  begin
    if (rst = '1') then
      valid <= '0';

    elsif (rising_edge(clk)) then
      valid <= pValid or not ready;

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
  ready        <= not valid or nReady;
  reg_en       <= ready and pValid;
  dataOutArray <= data_reg;

end arch;
