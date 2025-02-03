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
  signal ctrl_inner : std_logic_vector(2 downto 0);
  signal ctrl_valid_inner : std_logic;
  signal ctrl_spec_tag_inner : std_logic;
  signal ctrl_ready_inner : std_logic;
begin
  ctrl_buf : entity work.tfifo_with_tag(arch)
    generic map(
      NUM_SLOTS => 32,
      DATA_TYPE => 3
    )
    port map(
      clk => clk,
      rst => rst,
      ins => ctrl,
      ins_valid => ctrl_valid,
      ins_spec_tag => ctrl_spec_tag,
      ins_ready => ctrl_ready,
      outs => ctrl_inner,
      outs_valid => ctrl_valid_inner,
      outs_spec_tag => ctrl_spec_tag_inner,
      outs_ready => ctrl_ready_inner
    );
  spec_save_commit : entity work.spec_save_commit(arch)
    generic map(
      DATA_TYPE => DATA_TYPE,
      FIFO_DEPTH => FIFO_DEPTH
    )
    port map(
      clk => clk,
      rst => rst,
      ins => ins,
      ins_valid => ins_valid,
      ins_spec_tag => ins_spec_tag,
      ctrl => ctrl_inner,
      ctrl_valid => ctrl_valid_inner,
      ctrl_spec_tag => ctrl_spec_tag_inner,
      outs_ready => outs_ready,
      outs => outs,
      outs_valid => outs_valid,
      outs_spec_tag => outs_spec_tag,
      ins_ready => ins_ready,
      ctrl_ready => ctrl_ready_inner
    );
end architecture;
