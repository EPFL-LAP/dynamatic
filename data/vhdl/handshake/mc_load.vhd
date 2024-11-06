library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mc_load is
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

architecture arch of mc_load is
begin
  addr_tehb : entity work.tehb(arch)
    generic map(
      DATA_TYPE => ADDR_TYPE
    )
    port map(
      clk => clk,
      rst => rst,
      -- input channel
      ins       => addrIn,
      ins_valid => addrIn_valid,
      ins_ready => addrIn_ready,
      -- output channel
      outs       => addrOut,
      outs_valid => addrOut_valid,
      outs_ready => addrOut_ready
    );

  data_tehb : entity work.tehb(arch)
    generic map(
      DATA_TYPE => DATA_TYPE
    )
    port map(
      clk => clk,
      rst => rst,
      -- input channel
      ins       => dataFromMem,
      ins_valid => dataFromMem_valid,
      ins_ready => dataFromMem_ready,
      -- output channel
      outs       => dataOut,
      outs_valid => dataOut_valid,
      outs_ready => dataOut_ready
    );
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mc_load_with_tag is
  generic (
    DATA_TYPE : integer;
    ADDR_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- address from circuit channel
    addrIn       : in  std_logic_vector(ADDR_TYPE - 1 downto 0);
    addrIn_valid : in  std_logic;
    addrIn_spec_tag : in std_logic;
    addrIn_ready : out std_logic;
    -- address to interface channel
    addrOut       : out std_logic_vector(ADDR_TYPE - 1 downto 0);
    addrOut_valid : out std_logic;
    addrOut_spec_tag : out std_logic;
    addrOut_ready : in  std_logic;
    -- data from interface channel
    dataFromMem       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    dataFromMem_valid : in  std_logic;
    dataFromMem_spec_tag : in std_logic;
    dataFromMem_ready : out std_logic;
    -- data from memory channel
    dataOut       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    dataOut_valid : out std_logic;
    dataOut_spec_tag : out std_logic;
    dataOut_ready : in  std_logic
  );
end entity;

architecture arch of mc_load_with_tag is
  signal addrIn_ready_inner : std_logic;
  signal spec_tag_tfifo_valid : std_logic; -- not used
  signal spec_tag_tfifo_ready : std_logic;
  signal spec_tag_tfifo_n_ready : std_logic;
  signal spec_tag_tfifo_ins_inner : std_logic_vector(0 downto 0);
  signal spec_tag_tfifo_outs_inner : std_logic_vector(0 downto 0);
begin
  addrIn_ready <= addrIn_ready_inner and spec_tag_tfifo_ready;
  spec_tag_tfifo_n_ready <= dataOut_valid and dataOut_ready;
  spec_tag_tfifo_ins_inner(0) <= addrIn_spec_tag;
  spec_tag_tfifo : entity work.tfifo(arch)
    generic map(
      NUM_SLOTS => 32,
      DATA_TYPE => 1
    )
    port map(
      clk => clk,
      rst => rst,
      ins => spec_tag_tfifo_ins_inner,
      ins_valid => addrIn_valid,
      ins_ready => spec_tag_tfifo_ready,
      outs => spec_tag_tfifo_outs_inner,
      outs_valid => spec_tag_tfifo_valid,
      outs_ready => spec_tag_tfifo_n_ready
    );
  addrOut_spec_tag <= spec_tag_tfifo_outs_inner(0);
  dataOut_spec_tag <= spec_tag_tfifo_outs_inner(0);
  inner : entity work.mc_load(arch)
    generic map(
      DATA_TYPE,
      ADDR_TYPE
    )
    port map(
      clk => clk,
      rst => rst,
      addrIn => addrIn,
      addrIn_valid => addrIn_valid,
      addrIn_ready => addrIn_ready_inner,
      addrOut => addrOut,
      addrOut_valid => addrOut_valid,
      addrOut_ready => addrOut_ready,
      dataFromMem => dataFromMem,
      dataFromMem_valid => dataFromMem_valid,
      dataFromMem_ready => dataFromMem_ready,
      dataOut => dataOut,
      dataOut_valid => dataOut_valid,
      dataOut_ready => dataOut_ready
    );
end architecture;
