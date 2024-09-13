library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mc_store is
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

architecture arch of mc_store is
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
