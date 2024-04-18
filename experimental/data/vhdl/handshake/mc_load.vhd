library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mc_load is
  generic (
    DATA_BITWIDTH : integer;
    ADDR_BITWIDTH : integer
  );
  port (
    -- inputs
    clk, rst          : in std_logic;
    addrIn            : in std_logic_vector(ADDR_BITWIDTH - 1 downto 0);
    addrIn_valid      : in std_logic;
    dataFromMem       : in std_logic_vector(DATA_BITWIDTH - 1 downto 0);
    dataFromMem_valid : in std_logic;
    addrOut_ready     : in std_logic;
    dataOut_ready     : in std_logic;
    -- outputs
    addrOut           : out std_logic_vector(ADDR_BITWIDTH - 1 downto 0);
    addrOut_valid     : out std_logic;
    dataOut           : out std_logic_vector(DATA_BITWIDTH - 1 downto 0);
    dataOut_valid     : out std_logic;
    addrIn_ready      : out std_logic;
    dataFromMem_ready : out std_logic
  );
end entity;

architecture arch of mc_load is
  signal Buffer_1_readyArray_0   : std_logic;
  signal Buffer_1_validArray_0   : std_logic;
  signal Buffer_1_dataOutArray_0 : std_logic_vector(ADDR_BITWIDTH - 1 downto 0);

  signal Buffer_2_readyArray_0   : std_logic;
  signal Buffer_2_validArray_0   : std_logic;
  signal Buffer_2_dataOutArray_0 : std_logic_vector(DATA_BITWIDTH - 1 downto 0);

  signal addr_from_circuit       : std_logic_vector(ADDR_BITWIDTH - 1 downto 0);
  signal addr_from_circuit_valid : std_logic;
  signal addr_from_circuit_ready : std_logic;

  signal addr_to_lsq       : std_logic_vector(ADDR_BITWIDTH - 1 downto 0);
  signal addr_to_lsq_valid : std_logic;
  signal addr_to_lsq_ready : std_logic;

  signal data_from_lsq       : std_logic_vector(DATA_BITWIDTH - 1 downto 0);
  signal data_from_lsq_valid : std_logic;
  signal data_from_lsq_ready : std_logic;

  signal data_to_circuit       : std_logic_vector(DATA_BITWIDTH - 1 downto 0);
  signal data_to_circuit_valid : std_logic;
  signal data_to_circuit_ready : std_logic;

begin

  addr_from_circuit       <= addrIn;
  addr_from_circuit_valid <= addrIn_valid;
  addrIn_ready            <= Buffer_1_readyArray_0;

  Buffer_1 : entity work.tehb(arch) generic map (ADDR_BITWIDTH)
    port map(
      clk        => clk,
      rst        => rst,
      ins        => addr_from_circuit,
      ins_valid  => addr_from_circuit_valid,
      ins_ready  => Buffer_1_readyArray_0,
      outs_ready => addr_to_lsq_ready,
      outs_valid => Buffer_1_validArray_0,
      outs       => Buffer_1_dataOutArray_0
    );

  addr_to_lsq       <= Buffer_1_dataOutArray_0;
  addr_to_lsq_valid <= Buffer_1_validArray_0;
  addr_to_lsq_ready <= addrOut_ready;

  addrOut       <= addr_to_lsq;
  addrOut_valid <= addr_to_lsq_valid;

  dataFromMem_ready <= data_from_lsq_ready;

  data_from_lsq       <= dataFromMem;
  data_from_lsq_valid <= dataFromMem_valid;
  data_from_lsq_ready <= Buffer_2_readyArray_0;

  dataOut       <= Buffer_2_dataOutArray_0;
  dataOut_valid <= Buffer_2_validArray_0;

  Buffer_2 : entity work.tehb(arch) generic map (DATA_BITWIDTH)
    port map(
      clk        => clk,
      rst        => rst,
      ins        => data_from_lsq,
      ins_valid  => data_from_lsq_valid,
      ins_ready  => Buffer_2_readyArray_0,
      outs_ready => dataOut_ready,
      outs_valid => Buffer_2_validArray_0,
      outs       => Buffer_2_dataOutArray_0
    );

end architecture;
