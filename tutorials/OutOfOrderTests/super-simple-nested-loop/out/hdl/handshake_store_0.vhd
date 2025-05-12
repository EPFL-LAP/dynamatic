-- handshake_store_0 : store({'port_types': {'addrIn': '!handshake.channel<i5>', 'dataIn': '!handshake.channel<i32>', 'addrOut': '!handshake.channel<i5>', 'dataToMem': '!handshake.channel<i32>'}, 'addr_bitwidth': 5, 'data_bitwidth': 32, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of store
entity handshake_store_0 is
  port (
    clk, rst : in std_logic;
    -- data from circuit channel
    dataIn       : in  std_logic_vector(32 - 1 downto 0);
    dataIn_valid : in  std_logic;
    dataIn_ready : out std_logic;
    -- address from circuit channel
    addrIn       : in  std_logic_vector(5 - 1 downto 0);
    addrIn_valid : in  std_logic;
    addrIn_ready : out std_logic;
    -- data to interface channel
    dataToMem       : out std_logic_vector(32 - 1 downto 0);
    dataToMem_valid : out std_logic;
    dataToMem_ready : in  std_logic;
    -- address to interface channel
    addrOut       : out std_logic_vector(5 - 1 downto 0);
    addrOut_valid : out std_logic;
    addrOut_ready : in  std_logic
  );
end entity;

-- Architecture of store
architecture arch of handshake_store_0 is
begin
  -- data
  dataToMem       <= dataIn;
  dataToMem_valid <= dataIn_valid;
  dataIn_ready    <= dataToMem_ready;
  -- addr
  addrOut         <= addrIn;
  addrOut_valid   <= addrIn_valid;
  addrIn_ready    <= addrOut_ready;
end architecture;

