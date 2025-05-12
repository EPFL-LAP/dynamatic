-- handshake_muli_0 : muli({'port_types': {'lhs': '!handshake.channel<i32>', 'rhs': '!handshake.channel<i32>', 'result': '!handshake.channel<i32>'}, 'bitwidth': 32, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;

-- Entity of and_n
entity handshake_muli_0_join_and_n is
  port (
    -- inputs
    ins : in std_logic_vector(2 - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

-- Architecture of and_n
architecture arch of handshake_muli_0_join_and_n is
  signal all_ones : std_logic_vector(2 - 1 downto 0) := (others => '1');
begin
  outs <= '1' when ins = all_ones else '0';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of join_dataless
entity handshake_muli_0_join is
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
architecture arch of handshake_muli_0_join is
  signal allValid : std_logic;
begin
  allValidAndGate : entity work.handshake_muli_0_join_and_n port map(ins_valid, allValid);
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

-- Entity of mul_4_stage
entity handshake_muli_0_mul_4_stage is
  port (
    clk : in  std_logic;
    ce  : in  std_logic;
    a   : in  std_logic_vector(32 - 1 downto 0);
    b   : in  std_logic_vector(32 - 1 downto 0);
    p   : out std_logic_vector(32 - 1 downto 0));
end entity;

-- Architecture of mul_4_stage
architecture behav of handshake_muli_0_mul_4_stage is

  signal a_reg : std_logic_vector(32 - 1 downto 0);
  signal b_reg : std_logic_vector(32 - 1 downto 0);
  signal q0    : std_logic_vector(32 - 1 downto 0);
  signal q1    : std_logic_vector(32 - 1 downto 0);
  signal q2    : std_logic_vector(32 - 1 downto 0);
  signal mul   : std_logic_vector(32 - 1 downto 0);

begin

  mul <= std_logic_vector(resize(unsigned(std_logic_vector(signed(a_reg) * signed(b_reg))), 32));

  process (clk)
  begin
    if (clk'event and clk = '1') then
      if (ce = '1') then
        a_reg <= a;
        b_reg <= b;
        q0    <= mul;
        q1    <= q0;
        q2    <= q1;
      end if;
    end if;
  end process;

  p <= q2;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of delay_buffer
entity handshake_muli_0_buff is
  port (
    clk, rst  : in  std_logic;
    valid_in  : in  std_logic;
    ready_in  : in  std_logic;
    valid_out : out std_logic);
end entity;

-- Architecture of delay_buffer
architecture arch of handshake_muli_0_buff is

  type mem is array (3 - 1 downto 0) of std_logic;
  signal regs : mem;

begin

  gen_assignements : for i in 0 to 3 - 1 generate
    first_assignment : if i = 0 generate
      process (clk) begin
        if rising_edge(clk) then
          if (rst = '1') then
            regs(i) <= '0';
          elsif (ready_in = '1') then
            regs(i) <= valid_in;
          end if;
        end if;
      end process;
    end generate first_assignment;
    other_assignments : if i > 0 generate
      process (clk) begin
        if rising_edge(clk) then
          if (rst = '1') then
            regs(i) <= '0';
          elsif (ready_in = '1') then
            regs(i) <= regs(i - 1);
          end if;
        end if;
      end process;
    end generate other_assignments;
  end generate gen_assignements;

  valid_out <= regs(3 - 1);
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of oehb_dataless
entity handshake_muli_0_oehb_inner is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of oehb_dataless
architecture arch of handshake_muli_0_oehb_inner is
  signal outputValid : std_logic;
begin
  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        outputValid <= '0';
      else
        outputValid <= ins_valid or (outputValid and not outs_ready);
      end if;
    end if;
  end process;

  ins_ready  <= not outputValid or outs_ready;
  outs_valid <= outputValid;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of oehb
entity handshake_muli_0_oehb is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(32 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(32 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of oehb
architecture arch of handshake_muli_0_oehb is
  signal regEn, inputReady : std_logic;
begin

  control : entity work.handshake_muli_0_oehb_inner
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      ins_ready  => inputReady,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        outs <= (others => '0');
      elsif (regEn) then
        outs <= ins;
      end if;
    end if;
  end process;

  ins_ready <= inputReady;
  regEn     <= inputReady and ins_valid;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of muli
entity handshake_muli_0 is
  port (
    -- inputs
    clk, rst     : in std_logic;
    lhs          : in std_logic_vector(32 - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector(32 - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(32 - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;

-- Architecture of muli
architecture arch of handshake_muli_0 is
  signal join_valid                         : std_logic;
  signal buff_valid, oehb_valid, oehb_ready : std_logic;
  signal oehb_dataOut, oehb_datain          : std_logic_vector(32 - 1 downto 0);
begin
  join_inputs : entity work.handshake_muli_0_join(arch)
    port map(
      clk          => clk,
      rst          => rst,
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => oehb_ready,
      -- outputs
      outs_valid   => join_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  multiply_unit : entity work.handshake_muli_0_mul_4_stage(behav)
    port map(
      clk => clk,
      ce  => oehb_ready,
      a   => lhs,
      b   => rhs,
      p   => result
    );

  buff : entity work.handshake_muli_0_buff(arch)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid
    );

  oehb : entity work.handshake_muli_0_oehb(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => buff_valid,
      outs_ready => result_ready,
      outs_valid => result_valid,
      ins_ready  => oehb_ready,
      ins        => oehb_datain,
      outs       => oehb_dataOut
    );
end architecture;

