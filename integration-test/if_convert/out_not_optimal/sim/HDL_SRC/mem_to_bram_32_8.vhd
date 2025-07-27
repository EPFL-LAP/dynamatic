-- mem_to_bram_32_8 : mem_to_bram({'addr_bitwidth': 8, 'data_bitwidth': 32})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of mem_to_bram
entity mem_to_bram_32_8 is
  port (
    -- from circuit
    loadEn    : in std_logic;
    loadAddr  : in std_logic_vector(8 - 1 downto 0);
    storeEn   : in std_logic;
    storeAddr : in std_logic_vector(8 - 1 downto 0);
    storeData : in std_logic_vector(32 - 1 downto 0);
    -- to BRAM
    ce0      : out std_logic;
    we0      : out std_logic;
    address0 : out std_logic_vector(8 - 1 downto 0);
    dout0    : out std_logic_vector(32 - 1 downto 0);
    ce1      : out std_logic;
    we1      : out std_logic;
    address1 : out std_logic_vector(8 - 1 downto 0);
    dout1    : out std_logic_vector(32 - 1 downto 0);
    -- back from BRAM
    din0 : in std_logic_vector(32 - 1 downto 0);
    din1 : in std_logic_vector(32 - 1 downto 0);
    -- back to circuit
    loadData : out std_logic_vector(32 - 1 downto 0)
  );
end entity;

-- Architecture of mem_to_bram
architecture arch of mem_to_bram_32_8 is
begin
  -- store request
  ce0      <= storeEn;
  we0      <= storeEn;
  address0 <= storeAddr;
  dout0    <= storeData;
  -- load request
  ce1      <= loadEn;
  we1      <= '0';
  address1 <= loadAddr;
  dout1    <= (others => '0');
  loadData <= din1;
end architecture;

