library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity lsq_load is
  generic (
    DATA_TYPE : integer;
    ADDR_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- address from circuit channel
    addrIn       : in  std_logic_vector(ADDR_TYPE - 1 downto 0);
    addrIn_valid : in  std_logic;
    addrIn_ready : out std_logic;
    -- address to interface channel
    addrOut       : out std_logic_vector(ADDR_TYPE - 1 downto 0);
    addrOut_valid : out std_logic;
    addrOut_ready : in  std_logic;
    -- data from interface channel
    dataFromMem       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    dataFromMem_valid : in  std_logic;
    dataFromMem_ready : out std_logic;
    -- data from memory channel
    dataOut       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    dataOut_valid : out std_logic;
    dataOut_ready : in  std_logic
  );
end entity;

architecture arch of lsq_load is
begin
  -- data
  dataOut           <= dataFromMem;
  dataOut_valid     <= dataFromMem_valid;
  dataFromMem_ready <= dataOut_ready;
  -- addr
  addrOut       <= addrIn;
  addrOut_valid <= addrIn_valid;
  addrIn_ready  <= addrOut_ready;
end architecture;
