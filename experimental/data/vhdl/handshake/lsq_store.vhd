library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity lsq_store is
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

architecture arch of lsq_store is
begin
  -- data
  dataToMem       <= dataIn;
  dataToMem_valid <= dataIn_valid;
  dataIn_ready    <= dataToMem_ready;
  -- addr
  addrOut       <= addrIn;
  addrOut_valid <= addrIn_valid;
  addrIn_ready  <= addrOut_ready;
end architecture;
