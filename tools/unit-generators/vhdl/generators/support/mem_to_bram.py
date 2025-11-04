def generate_mem_to_bram(name, params):
    data_bitwidth = params["data_bitwidth"]
    addr_bitwidth = params["addr_bitwidth"]

    return f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of mem_to_bram
entity {name} is
  port (
    -- from circuit
    loadEn    : in std_logic;
    loadAddr  : in std_logic_vector({addr_bitwidth} - 1 downto 0);
    storeEn   : in std_logic;
    storeAddr : in std_logic_vector({addr_bitwidth} - 1 downto 0);
    storeData : in std_logic_vector({data_bitwidth} - 1 downto 0);
    -- to BRAM
    ce0      : out std_logic;
    we0      : out std_logic;
    address0 : out std_logic_vector({addr_bitwidth} - 1 downto 0);
    dout0    : out std_logic_vector({data_bitwidth} - 1 downto 0);
    ce1      : out std_logic;
    we1      : out std_logic;
    address1 : out std_logic_vector({addr_bitwidth} - 1 downto 0);
    dout1    : out std_logic_vector({data_bitwidth} - 1 downto 0);
    -- back from BRAM
    din0 : in std_logic_vector({data_bitwidth} - 1 downto 0);
    din1 : in std_logic_vector({data_bitwidth} - 1 downto 0);
    -- back to circuit
    loadData : out std_logic_vector({data_bitwidth} - 1 downto 0)
  );
end entity;

-- Architecture of mem_to_bram
architecture arch of {name} is
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
"""
