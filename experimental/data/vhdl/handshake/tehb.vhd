library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tehb is
  generic (
    BITWIDTH : integer
  );
  port (
    -- inputs
    clk, rst   : in std_logic;
    ins        : in std_logic_vector(BITWIDTH - 1 downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
end entity;

architecture arch of tehb is
  signal full_reg, empty_reg, reg_en : std_logic;
  signal data_reg                    : std_logic_vector(BITWIDTH - 1 downto 0);

begin
  process (clk, rst) is
  begin
    if (rst = '1') then
      full_reg <= '0';
    elsif (rising_edge(clk)) then
      full_reg <= ins_valid and not outs_ready;
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

  process (full_reg, data_reg, ins) is
  begin
    if (full_reg = '1') then
      outs <= data_reg;
    else
      outs <= ins;
    end if;
  end process;

  empty_reg <= not full_reg;
  reg_en    <= empty_reg and (ins_valid and (not outs_ready));

  outs_valid <= ins_valid or full_reg;
  ins_ready  <= empty_reg;
end architecture;
