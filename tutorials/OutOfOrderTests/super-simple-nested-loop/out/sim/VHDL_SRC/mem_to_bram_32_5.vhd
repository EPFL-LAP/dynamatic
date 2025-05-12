-- mem_to_bram_32_5 : mem_to_bram({'port_types': {'loadEn': 'i1', 'loadAddr': 'i5', 'storeEn': 'i1', 'storeAddr': 'i5', 'storeData': 'i32', 'din0': 'i32', 'din1': 'i32', 'ce0': 'i1', 'we0': 'i1', 'address0': 'i5', 'dout0': 'i32', 'ce1': 'i1', 'we1': 'i1', 'address1': 'i5', 'dout1': 'i32', 'loadData': 'i32'}, 'addr_bitwidth': 5, 'data_bitwidth': 32})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of mem_to_bram
entity mem_to_bram_32_5 is
  port (
    -- from circuit
    loadEn    : in std_logic;
    loadAddr  : in std_logic_vector(5 - 1 downto 0);
    storeEn   : in std_logic;
    storeAddr : in std_logic_vector(5 - 1 downto 0);
    storeData : in std_logic_vector(32 - 1 downto 0);
    -- to BRAM
    ce0      : out std_logic;
    we0      : out std_logic;
    address0 : out std_logic_vector(5 - 1 downto 0);
    dout0    : out std_logic_vector(32 - 1 downto 0);
    ce1      : out std_logic;
    we1      : out std_logic;
    address1 : out std_logic_vector(5 - 1 downto 0);
    dout1    : out std_logic_vector(32 - 1 downto 0);
    -- back from BRAM
    din0 : in std_logic_vector(32 - 1 downto 0);
    din1 : in std_logic_vector(32 - 1 downto 0);
    -- back to circuit
    loadData : out std_logic_vector(32 - 1 downto 0)
  );
end entity;

-- Architecture of mem_to_bram
architecture arch of mem_to_bram_32_5 is
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

