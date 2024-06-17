library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity oehb is
  generic (
    BITWIDTH : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(BITWIDTH - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of oehb is
  signal reg_en   : std_logic;
  signal data_reg : std_logic_vector(BITWIDTH - 1 downto 0);
begin

  control : entity work.oehb_dataless
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      ins_ready  => ins_ready,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

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

  reg_en <= ins_ready and ins_valid;
  outs   <= data_reg;
end arch;
