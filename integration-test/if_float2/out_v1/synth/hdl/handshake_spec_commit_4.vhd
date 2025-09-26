-- handshake_spec_commit_4 : spec_commit({'bitwidth': 32, 'extra_signals': {'spec': 1}})


library ieee;
use ieee.std_logic_1164.all;

-- Entity of and_n
entity handshake_spec_commit_4_cond_br_inner_join_and_n is
  port (
    -- inputs
    ins : in std_logic_vector(2 - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

-- Architecture of and_n
architecture arch of handshake_spec_commit_4_cond_br_inner_join_and_n is
  signal all_ones : std_logic_vector(2 - 1 downto 0) := (others => '1');
begin
  outs <= '1' when ins = all_ones else '0';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of join
entity handshake_spec_commit_4_cond_br_inner_join is
  port (
    -- inputs
    ins_valid  : in std_logic_vector(2 - 1 downto 0);
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic;
    ins_ready  : out std_logic_vector(2 - 1 downto 0)
  );
end entity;

-- Architecture of join
architecture arch of handshake_spec_commit_4_cond_br_inner_join is
  signal allValid : std_logic;
begin
  allValidAndGate : entity work.handshake_spec_commit_4_cond_br_inner_join_and_n port map(ins_valid, allValid);
  outs_valid <= allValid;

  process (ins_valid, outs_ready)
    variable singlePValid : std_logic_vector(2 - 1 downto 0);
  begin
    for i in 0 to 2 - 1 loop
      singlePValid(i) := '1';
      for j in 0 to 2 - 1 loop
        if (i /= j) then
          singlePValid(i) := (singlePValid(i) and ins_valid(j));
        end if;
      end loop;
    end loop;
    for i in 0 to 2 - 1 loop
      ins_ready(i) <= (singlePValid(i) and outs_ready);
    end loop;
  end process;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of cond_br_dataless
entity handshake_spec_commit_4_cond_br_inner is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- data input channel
    data_valid : in  std_logic;
    data_ready : out std_logic;
    -- condition input channel
    condition       : in  std_logic_vector(0 downto 0);
    condition_valid : in  std_logic;
    condition_ready : out std_logic;
    -- true output channel
    trueOut_valid : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut_valid : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;

-- Architecture of cond_br_dataless
architecture arch of handshake_spec_commit_4_cond_br_inner is
  signal branchInputs_valid, branch_ready : std_logic;
begin

  join : entity work.handshake_spec_commit_4_cond_br_inner_join(arch)
    port map(
      -- input channels
      ins_valid(0) => data_valid,
      ins_valid(1) => condition_valid,
      ins_ready(0) => data_ready,
      ins_ready(1) => condition_ready,
      -- output channel
      outs_valid => branchInputs_valid,
      outs_ready => branch_ready
    );

  trueOut_valid  <= condition(0) and branchInputs_valid;
  falseOut_valid <= (not condition(0)) and branchInputs_valid;
  branch_ready   <= (falseOut_ready and not condition(0)) or (trueOut_ready and condition(0));
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of cond_br
entity handshake_spec_commit_4_cond_br is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- data input channel
    data       : in  std_logic_vector(32 - 1 downto 0);
    data_valid : in  std_logic;
    data_ready : out std_logic;
    -- condition input channel
    condition       : in  std_logic_vector(0 downto 0);
    condition_valid : in  std_logic;
    condition_ready : out std_logic;
    -- true output channel
    trueOut       : out std_logic_vector(32 - 1 downto 0);
    trueOut_valid : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut       : out std_logic_vector(32 - 1 downto 0);
    falseOut_valid : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;

-- Architecture of cond_br
architecture arch of handshake_spec_commit_4_cond_br is
begin
  control : entity work.handshake_spec_commit_4_cond_br_inner
    port map(
      clk             => clk,
      rst             => rst,
      data_valid      => data_valid,
      data_ready      => data_ready,
      condition       => condition,
      condition_valid => condition_valid,
      condition_ready => condition_ready,
      trueOut_valid   => trueOut_valid,
      trueOut_ready   => trueOut_ready,
      falseOut_valid  => falseOut_valid,
      falseOut_ready  => falseOut_ready
    );

  trueOut  <= data;
  falseOut <= data;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of merge
entity handshake_spec_commit_4_merge is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channels
    ins       : in  data_array(2 - 1 downto 0)(32 - 1 downto 0);
    ins_valid : in  std_logic_vector(2 - 1 downto 0);
    ins_ready : out std_logic_vector(2 - 1 downto 0);
    -- output channel
    outs       : out std_logic_vector(32 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of merge
architecture arch of handshake_spec_commit_4_merge is
begin
  process (ins_valid, ins, outs_ready)
    variable tmp_data_out  : unsigned(32 - 1 downto 0);
    variable tmp_valid_out : std_logic;
    variable tmp_ready_out : std_logic_vector(2 - 1 downto 0);
  begin
    tmp_data_out  := unsigned(ins(0));
    tmp_valid_out := '0';
    tmp_ready_out := (others => '0');

    for I in 0 to (2 - 1) loop
      if (ins_valid(I) = '1') then
        tmp_data_out  := unsigned(ins(I));
        tmp_valid_out := '1';
        tmp_ready_out(i) := outs_ready;
        exit;
      end if;
    end loop;

    outs <= std_logic_vector(resize(tmp_data_out, 32));
    outs_valid  <= tmp_valid_out;
    ins_ready <= tmp_ready_out;
  end process;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of spec_commit
entity handshake_spec_commit_4 is
  port (
    clk, rst : in  std_logic;
    ins : in std_logic_vector(32 - 1 downto 0);
    ins_valid : in std_logic;
    ins_ready : out std_logic;
    ins_spec : in std_logic_vector(0 downto 0);
    ctrl : in std_logic_vector(0 downto 0);
    ctrl_valid : in std_logic;
    ctrl_ready : out std_logic;
    outs : out std_logic_vector(32 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in std_logic
  );
end entity;

-- Architecture of spec_commit
architecture arch of handshake_spec_commit_4 is
  signal branch_in_condition : std_logic_vector(0 downto 0);
  signal branch_in_trueOut : std_logic_vector(32 - 1 downto 0);
  signal branch_in_trueOut_valid : std_logic;
  signal branch_in_trueOut_ready : std_logic;
  signal branch_in_falseOut : std_logic_vector(32 - 1 downto 0);
  signal branch_in_falseOut_valid : std_logic;
  signal branch_in_falseOut_ready : std_logic;

  signal branch_disc_falseOut : std_logic_vector(32 - 1 downto 0);
  signal branch_disc_falseOut_valid : std_logic;
  signal branch_disc_falseOut_ready : std_logic;

  signal merge_ins : data_array(1 downto 0)(32 - 1 downto 0);
  signal merge_ins_valid : std_logic_vector(1 downto 0);
  signal merge_ins_ready : std_logic_vector(1 downto 0);
begin

  branch_in_condition <= ins_spec;

  branch_in: entity work.handshake_spec_commit_4_cond_br(arch)
    port map (
      clk => clk,
      rst => rst,

      data => ins,
      data_valid => ins_valid,
      data_ready => ins_ready,

      condition => branch_in_condition,
      -- Handshaking is common with `data`, keep valid high and ignore ready
      condition_valid => '1',
      condition_ready => open,

      trueOut => branch_in_trueOut,
      trueOut_valid => branch_in_trueOut_valid,
      trueOut_ready => branch_in_trueOut_ready,

      falseOut => branch_in_falseOut,
      falseOut_valid => branch_in_falseOut_valid,
      falseOut_ready => branch_in_falseOut_ready
    );

  branch_disc: entity work.handshake_spec_commit_4_cond_br(arch)
    port map (
      clk => clk,
      rst => rst,

      data => branch_in_trueOut,
      data_valid => branch_in_trueOut_valid,
      data_ready => branch_in_trueOut_ready,

      condition => ctrl,
      condition_valid => ctrl_valid,
      condition_ready => ctrl_ready,

      -- trueOut sinks
      trueOut => open,
      trueOut_valid => open,
      trueOut_ready => '1',

      falseOut => branch_disc_falseOut,
      falseOut_valid => branch_disc_falseOut_valid,
      falseOut_ready => branch_disc_falseOut_ready
    );

  merge_ins <= (branch_disc_falseOut, branch_in_falseOut);
  merge_ins_valid <= (branch_disc_falseOut_valid, branch_in_falseOut_valid);
  branch_disc_falseOut_ready <= merge_ins_ready(1);
  branch_in_falseOut_ready <= merge_ins_ready(0);

  merge_out: entity work.handshake_spec_commit_4_merge(arch)
    port map (
      clk => clk,
      rst => rst,

      ins => merge_ins,
      ins_valid => merge_ins_valid,
      ins_ready => merge_ins_ready,

      outs => outs,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;

