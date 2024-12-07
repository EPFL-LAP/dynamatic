library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity store is
  generic (
    DATA_TYPE : integer;
    ADDR_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- data from circuit channel
    dataIn       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    dataIn_valid : in  std_logic;
    dataIn_ready : out std_logic;
    -- address from circuit channel
    addrIn       : in  std_logic_vector(ADDR_TYPE - 1 downto 0);
    addrIn_valid : in  std_logic;
    addrIn_ready : out std_logic;
    -- data to interface channel
    dataToMem       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    dataToMem_valid : out std_logic;
    dataToMem_ready : in  std_logic;
    -- address to interface channel
    addrOut       : out std_logic_vector(ADDR_TYPE - 1 downto 0);
    addrOut_valid : out std_logic;
    addrOut_ready : in  std_logic
  );
end entity;

architecture arch of store is
  signal single_ready : std_logic;
  signal join_valid   : std_logic;
begin
  join : entity work.join(arch)
    generic map(
      SIZE => 2
    )
    port map(
      -- input channels
      ins_valid(0) => dataIn_valid,
      ins_valid(1) => addrIn_valid,
      ins_ready(0) => dataIn_ready,
      ins_ready(1) => addrIn_ready,
      -- output channel
      outs_valid => join_valid,
      outs_ready => dataToMem_ready
    );

  -- address
  addrOut       <= addrIn;
  addrOut_valid <= join_valid;
  -- data
  dataToMem       <= dataIn;
  dataToMem_valid <= join_valid;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mc_store_with_tag is
  generic (
    DATA_TYPE : integer;
    ADDR_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- data from circuit channel
    dataIn       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    dataIn_valid : in  std_logic;
    dataIn_spec_tag : in std_logic;
    dataIn_ready : out std_logic;
    -- address from circuit channel
    addrIn       : in  std_logic_vector(ADDR_TYPE - 1 downto 0);
    addrIn_valid : in  std_logic;
    addrIn_spec_tag : in std_logic;
    addrIn_ready : out std_logic;
    -- data to interface channel
    dataToMem       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    dataToMem_valid : out std_logic;
    dataToMem_spec_tag : out std_logic;
    dataToMem_ready : in  std_logic;
    -- address to interface channel
    addrOut       : out std_logic_vector(ADDR_TYPE - 1 downto 0);
    addrOut_valid : out std_logic;
    addrOut_spec_tag : out std_logic;
    addrOut_ready : in  std_logic
  );
end entity;

architecture arch of mc_store_with_tag is
begin
  -- assume that dataIn_spec_tag and addrIn_spec_tag are always '0'
  dataToMem_spec_tag <= '0';
  addrOut_spec_tag   <= '0';
  mc_store : entity work.mc_store(arch)
    generic map(
      DATA_TYPE => DATA_TYPE,
      ADDR_TYPE => ADDR_TYPE
    )
    port map(
      clk => clk,
      rst => rst,
      dataIn => dataIn,
      dataIn_valid => dataIn_valid,
      dataIn_ready => dataIn_ready,
      addrIn => addrIn,
      addrIn_valid => addrIn_valid,
      addrIn_ready => addrIn_ready,
      dataToMem => dataToMem,
      dataToMem_valid => dataToMem_valid,
      dataToMem_ready => dataToMem_ready,
      addrOut => addrOut,
      addrOut_valid => addrOut_valid,
      addrOut_ready => addrOut_ready
    );
end architecture;
