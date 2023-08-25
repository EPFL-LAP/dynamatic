library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity d_store is generic (
  ADDR_BITWIDTH : integer;
  DATA_BITWIDTH : integer);
port (
  -- inputs
  clk             : in std_logic;
  rst             : in std_logic;
  addrIn          : in std_logic_vector(ADDR_BITWIDTH - 1 downto 0);
  addrIn_valid    : in std_logic;
  dataIn          : in std_logic_vector(DATA_BITWIDTH - 1 downto 0);
  dataIn_valid    : in std_logic;
  addrOut_ready   : in std_logic;
  dataToMem_ready : in std_logic;
  -- outputs
  addrIn_ready    : out std_logic;
  dataIn_ready    : out std_logic;
  addrOut         : out std_logic_vector(ADDR_BITWIDTH - 1 downto 0);
  addrOut_valid   : out std_logic;
  dataToMem       : out std_logic_vector(DATA_BITWIDTH - 1 downto 0);
  dataToMem_valid : out std_logic);

end entity;

architecture arch of d_store is
  signal single_ready : std_logic;
  signal join_valid   : std_logic;

begin

  join_write : entity work.join(arch) generic map(2)
    port map(
    (addrIn_valid,
      dataIn_valid),
      addrIn_ready,
      join_valid,
      (addrOut_ready,
      dataToMem_ready));

  dataToMem       <= dataIn;
  addrOut_valid   <= join_valid;
  addrOut         <= addrIn;
  dataToMem_valid <= join_valid;
end architecture;
