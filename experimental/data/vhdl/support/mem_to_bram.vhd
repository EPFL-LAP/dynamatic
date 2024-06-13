library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mem_to_bram is
  generic (
    DATA_WIDTH : integer;
    ADDR_WIDTH : integer
  );
  port (
    -- from circuit
    loadEn    : in std_logic;
    loadAddr  : in std_logic_vector(ADDR_WIDTH - 1 downto 0);
    storeEn   : in std_logic;
    storeAddr : in std_logic_vector(ADDR_WIDTH - 1 downto 0);
    storeData : in std_logic_vector(DATA_WIDTH - 1 downto 0);
    -- to BRAM
    ce0      : out std_logic;
    we0      : out std_logic;
    address0 : out std_logic_vector(ADDR_WIDTH - 1 downto 0);
    din0     : out std_logic_vector(DATA_WIDTH - 1 downto 0);
    ce1      : out std_logic;
    we1      : out std_logic;
    address1 : out std_logic_vector(ADDR_WIDTH - 1 downto 0);
    din1     : out std_logic_vector(DATA_WIDTH - 1 downto 0);
    -- back from BRAM
    dout0 : in std_logic_vector(DATA_WIDTH - 1 downto 0);
    dout1 : in std_logic_vector(DATA_WIDTH - 1 downto 0);
    -- back to circuit
    loadData : out std_logic_vector(DATA_WIDTH - 1 downto 0)
  );
end entity;

architecture arch of mem_to_bram is
begin

  -- store request
  ce0      <= storeEn;
  we0      <= storeEn;
  address0 <= storeAddr;
  din0     <= storeData;

  -- load request
  ce1      <= loadEn;
  address1 <= loadAddr;
  loadData <= dout1;

end architecture;
