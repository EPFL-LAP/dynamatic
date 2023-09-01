library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity OEHB is
  generic (
    BITWIDTH : integer
  );
  port (
    clk        : in std_logic;
    rst        : in std_logic;
    ins        : in std_logic_vector(BITWIDTH - 1 downto 0);
    outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    outs_valid : out std_logic;
    ins_ready  : out std_logic);
end OEHB;

architecture arch of OEHB is
  signal full_reg, reg_en, mux_sel : std_logic;
  signal data_reg                  : std_logic_vector(BITWIDTH - 1 downto 0);
begin

  process (clk, rst) is

  begin
    if (rst = '1') then
      outs_valid <= '0';

    elsif (rising_edge(clk)) then
      outs_valid <= ins_valid or not ins_ready;

    end if;
  end process;

  process (clk, rst) is

  begin
    if (rst = '1') then
      data_reg <= (others => '0');

    elsif (rising_edge(clk)) then
      if (reg_en) then
        data_reg <= ins;
      end if;

    end if;
  end process;
  ins_ready <= not outs_valid or outs_ready;
  reg_en    <= ins_ready and ins_valid;
  outs      <= data_reg;

end arch;
