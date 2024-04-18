library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mc_store is
  generic (
    DATA_BITWIDTH : integer;
    ADDR_BITWIDTH : integer
  );
  port (
    -- inputs
    clk, rst        : in std_logic;
    addrIn          : in std_logic_vector(ADDR_BITWIDTH - 1 downto 0);
    addrIn_valid    : in std_logic;
    dataIn          : in std_logic_vector(DATA_BITWIDTH - 1 downto 0);
    dataIn_valid    : in std_logic;
    addrOut_ready   : in std_logic;
    dataToMem_ready : in std_logic;
    -- outputs
    addrOut         : out std_logic_vector(ADDR_BITWIDTH - 1 downto 0);
    addrOut_valid   : out std_logic;
    dataToMem       : out std_logic_vector(DATA_BITWIDTH - 1 downto 0);
    dataToMem_valid : out std_logic;
    addrIn_ready    : out std_logic;
    dataIn_ready    : out std_logic
  );
end entity;

architecture arch of mc_store is
  signal single_ready : std_logic;
  signal join_valid   : std_logic;
  signal out_array    : std_logic_vector(1 downto 0);

begin
  addrIn_ready <= out_array(0);
  dataIn_ready <= out_array(1);

  join_write : entity work.join(arch) generic map(2)
    port map(
    (addrIn_valid,
      dataIn_valid),
      addrIn_ready,
      join_valid,
      out_array);

  dataToMem       <= dataIn;
  addrOut_valid   <= join_valid;
  addrOut         <= addrIn;
  dataToMem_valid <= join_valid;
end architecture;
