-- handshake_cond_br_2 : cond_br({'port_types': {'condition': '!handshake.channel<i1>', 'data': '!handshake.control<>', 'trueOut': '!handshake.control<>', 'falseOut': '!handshake.control<>'}, 'bitwidth': 0, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;

-- Entity of and_n
entity handshake_cond_br_2_join_and_n is
  port (
    -- inputs
    ins : in std_logic_vector(2 - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

-- Architecture of and_n
architecture arch of handshake_cond_br_2_join_and_n is
  signal all_ones : std_logic_vector(2 - 1 downto 0) := (others => '1');
begin
  outs <= '1' when ins = all_ones else '0';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of join_dataless
entity handshake_cond_br_2_join is
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    ins_valid  : in std_logic_vector(2 - 1 downto 0);
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic;
    ins_ready  : out std_logic_vector(2 - 1 downto 0)
  );
end entity;

-- Architecture of join_dataless
architecture arch of handshake_cond_br_2_join is
  signal allValid : std_logic;
begin
  allValidAndGate : entity work.handshake_cond_br_2_join_and_n port map(ins_valid, allValid);
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
entity handshake_cond_br_2 is
  port (
    clk, rst : in std_logic;
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
architecture arch of handshake_cond_br_2 is
  signal branchInputs_valid, branch_ready : std_logic;
begin

  join : entity work.handshake_cond_br_2_join(arch)
    port map(
      clk          => clk,
      rst          => rst,
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

