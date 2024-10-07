library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity spec_save_commit_wrapper_with_tag is
  generic (
    DATA_TYPE : integer;
    FIFO_DEPTH : integer
  );
  port (
    clk, rst : in std_logic;
    -- inputs
    ins : in std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid : in std_logic;
    ins_spec_tag : in std_logic;
    ctrl : in std_logic_vector(2 downto 0); -- 000:pass, 001:kill, 010:resend, 011:kill-pass, 100:no_cmp
    ctrl_valid : in std_logic;
    ctrl_spec_tag : in std_logic; -- not used
    outs_ready : in std_logic;
    -- outputs
    outs : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_spec_tag : out std_logic;
    ins_ready : out std_logic;
    ctrl_ready : out std_logic
  );
end entity;

architecture arch of spec_save_commit_wrapper_with_tag is
  signal dataInArray   :  data_array(0 downto 0)(DATA_TYPE - 1 downto 0);
  signal specInArray   :  data_array(0 downto 0)(0 downto 0);
  signal controlInArray  :  data_array(0 downto 0)(2 downto 0);
  signal pValidArray : std_logic_vector(1 downto 0);
  signal readyArray  : std_logic_vector(1 downto 0);

  signal dataOutArray : data_array(0 downto 0)(DATA_TYPE - 1 downto 0);
  signal specOutArray  : data_array(0 downto 0)(0 downto 0);
  signal validArray   : std_logic_vector(0 downto 0);
  signal nReadyArray  : std_logic_vector(0 downto 0);
begin
  dataInArray(0) <= ins;
  specInArray(0)(0) <= ins_spec_tag;
  controlInArray(0) <= ctrl;
  pValidArray <= ctrl_valid & ins_valid;
  ctrl_ready <= readyArray(1);
  ins_ready <= readyArray(0);
  outs <= dataOutArray(0);
  outs_spec_tag <= specOutArray(0)(0);
  outs_valid <= validArray(0);
  nReadyArray(0) <= outs_ready;
  spec_save_commit : entity work.spec_save_commit(arch)
    generic map(
      DATA_SIZE_IN => DATA_TYPE,
      DATA_SIZE_OUT => DATA_TYPE,
      FIFO_DEPTH => FIFO_DEPTH
    )
    port map(
      clk => clk,
      rst => rst,
      dataInArray => dataInArray,
      specInArray => specInArray,
      controlInArray => controlInArray,
      pValidArray => pValidArray,
      readyArray => readyArray,
      dataOutArray => dataOutArray,
      specOutArray => specOutArray,
      validArray => validArray,
      nReadyArray => nReadyArray
    );
end architecture;
