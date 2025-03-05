library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity load is
  generic (
    DATA_TYPE : integer;
    ADDR_TYPE : integer;
    TAG_SIZE  : integer
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
     -- tags
    tagIn       : in  std_logic_vector(TAG_SIZE - 1 downto 0);
    tagOut      : out  std_logic_vector(TAG_SIZE - 1 downto 0);
  );
end entity;

architecture arch of load is

  signal tagged_addrIn : std_logic_vector(ADDR_TYPE + TAG_SIZE - 1 downto 0); 
  signal tagged_addrOut : std_logic_vector(ADDR_TYPE + TAG_SIZE - 1 downto 0); 
  signal tagged_dataFromMem : std_logic_vector(DATA_TYPE + TAG_SIZE - 1 downto 0); 
  signal tagged_dataOut : std_logic_vector(DATA_TYPE + TAG_SIZE - 1 downto 0); 

begin
  tagged_addrIn <= tagIn & addrIn;

  addr_tehb : entity work.tehb(arch)
    generic map(
      DATA_TYPE => ADDR_TYPE + TAG_SIZE
    )
    port map(
      clk => clk,
      rst => rst,
      -- input channel
      ins       => tagged_addrIn,
      ins_valid => addrIn_valid,
      ins_ready => addrIn_ready,
      -- output channel
      outs       => tagged_addrOut,
      outs_valid => addrOut_valid,
      outs_ready => addrOut_ready
    );

  addrOut <= tagged_addrOut(ADDR_TYPE - 1 downto 0);
  tagged_dataFromMem <= tagged_addrOut(TAG_SIZE - 1 downto ADDR_TYPE) & dataFromMem;

  data_tehb : entity work.tehb(arch)
    generic map(
      DATA_TYPE => DATA_TYPE + TAG_SIZE
    )
    port map(
      clk => clk,
      rst => rst,
      -- input channel
      ins       => tagged_dataFromMem,
      ins_valid => dataFromMem_valid,
      ins_ready => dataFromMem_ready,
      -- output channel
      outs       => tagged_dataOut,
      outs_valid => dataOut_valid,
      outs_ready => dataOut_ready
    );

  dataOut <= tagged_dataOut(DATA_TYPE -1 downto 0);
  tagOut <= tagged_dataOut(TAG_SIZE - 1 downto DATA_TYPE);

end architecture;