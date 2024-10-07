library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity spec_commit_wrapper is
  generic (
    DATA_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- inputs
    ins : in std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid : in std_logic;
    ins_spec_tag : in std_logic;
    ctrl : in std_logic;
    ctrl_valid : in std_logic;
    ctrl_spec_tag : in std_logic; -- not used
    result_ready : in std_logic;
    -- outputs
    result : out std_logic_vector(DATA_TYPE - 1 downto 0);
    result_valid : out std_logic;
    result_spec_tag : out std_logic;
    ins_ready : out std_logic;
    ctrl_ready : out std_logic
  );
end entity;

architecture arch of spec_commit_wrapper is
  signal dataInArray   :  data_array(0 downto 0)(DATA_TYPE - 1 downto 0);
  signal specInArray   :  data_array(0 downto 0)(0 downto 0);
  signal ControlInArray  :  data_array(0 downto 0)(0 downto 0);
  signal pValidArray   :  std_logic_vector(1 downto 0); -- (control, data)
  signal readyArray    :  std_logic_vector(1 downto 0); -- (control, data)

  signal dataOutArray  :  data_array(0 downto 0)(DATA_TYPE - 1 downto 0);
  signal validArray    :  std_logic_vector(0 downto 0);
  signal nReadyArray   :  std_logic_vector(0 downto 0);
begin
  dataInArray(0) <= ins;
  specInArray(0)(0) <= ins_spec_tag;
  ControlInArray(0)(0) <= ctrl;
  pValidArray <= ctrl_valid & ins_valid;
  ctrl_ready <= readyArray(1);
  ins_ready <= readyArray(0);
  result <= dataOutArray(0);
  result_valid <= validArray(0);
  nReadyArray(0) <= result_ready;
  result_spec_tag <= '0'; -- always 0
  spec_commit : entity work.spec_commit(arch)
    generic map(
      DATA_SIZE_IN => DATA_TYPE,
      DATA_SIZE_OUT => DATA_TYPE
    )
    port map(
      clk => clk,
      rst => rst,
      dataInArray => dataInArray,
      specInArray => specInArray,
      ControlInArray => ControlInArray,
      pValidArray => pValidArray,
      readyArray => readyArray,
      dataOutArray => dataOutArray,
      validArray => validArray,
      nReadyArray => nReadyArray
    );
end architecture;