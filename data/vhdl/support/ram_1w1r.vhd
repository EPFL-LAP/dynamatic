library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ram_1w1r is
  generic (
    DATA_WIDTH : integer;
    ADDR_WIDTH : integer;
    SIZE       : integer
  );
  port (
    clk       : in std_logic;
    rst       : in std_logic;
    -- from circuit (mem_controller / LSQ)
    loadEn    : in std_logic;
    loadAddr  : in std_logic_vector(ADDR_WIDTH - 1 downto 0);
    storeEn   : in std_logic;
    storeAddr : in std_logic_vector(ADDR_WIDTH - 1 downto 0);
    storeData : in std_logic_vector(DATA_WIDTH - 1 downto 0);
    -- to circuit (mem_controller / LSQ)
    loadData  : out std_logic_vector(DATA_WIDTH - 1 downto 0)
  );
end entity;

architecture arch of ram_1w1r is
  type ram_type is array (SIZE - 1 downto 0) of std_logic_vector(DATA_WIDTH - 1 downto 0);
  signal ram : ram_type;
begin

  read_proc : process(clk) 
  begin
    if (rising_edge(clk)) then
      if (loadEn = '1') then
        loadData <= ram(to_integer(unsigned(loadAddr)));
      end if;
    end if;
  end process;

  write_proc : process(clk)
  begin
    if (rising_edge(clk)) then
      if (storeEn = '1') then
        ram(to_integer(unsigned(storeAddr))) <= storeData;
      end if;
    end if;
  end process;

end architecture;
