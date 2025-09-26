-- handshake_addf_1 : addf({'is_double': False, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;

-- Entity of and_n
entity handshake_addf_1_join_and_n is
  port (
    -- inputs
    ins : in std_logic_vector(2 - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

-- Architecture of and_n
architecture arch of handshake_addf_1_join_and_n is
  signal all_ones : std_logic_vector(2 - 1 downto 0) := (others => '1');
begin
  outs <= '1' when ins = all_ones else '0';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of join
entity handshake_addf_1_join is
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
architecture arch of handshake_addf_1_join is
  signal allValid : std_logic;
begin
  allValidAndGate : entity work.handshake_addf_1_join_and_n port map(ins_valid, allValid);
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

-- Entity of one_slot_break_dv_dataless
entity handshake_addf_1_one_slot_break_dv is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of one_slot_break_dv_dataless
architecture arch of handshake_addf_1_one_slot_break_dv is
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

-- Entity of delay_buffer
entity handshake_addf_1_buff is
  port (
    clk, rst  : in  std_logic;
    valid_in  : in  std_logic;
    ready_in  : in  std_logic;
    valid_out : out std_logic);
end entity;

-- Architecture of delay_buffer
architecture arch of handshake_addf_1_buff is

  type mem is array (8 - 1 downto 0) of std_logic;
  signal regs : mem;

begin

  gen_assignements : for i in 0 to 8 - 1 generate
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

  valid_out <= regs(8 - 1);
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of addf_single_precision
entity handshake_addf_1 is
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
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

-- Architecture of addf_single_precision
architecture arch of handshake_addf_1 is
  signal join_valid : std_logic;
  signal buff_valid, one_slot_break_dv_ready : std_logic;

  -- intermediate input signals for IEEE-754 to Flopoco-simple-float conversion
  signal ip_lhs, ip_rhs : std_logic_vector(32 + 1 downto 0);

  -- intermediate output signal for Flopoco-simple-float to IEEE-754 conversion
  signal ip_result : std_logic_vector(32 + 1 downto 0);
begin
  join_inputs : entity work.handshake_addf_1_join(arch)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => one_slot_break_dv_ready,
      -- outputs
      outs_valid   => join_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  one_slot_break_dv : entity work.handshake_addf_1_one_slot_break_dv(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => buff_valid,
      outs_ready => result_ready,
      outs_valid => result_valid,
      ins_ready  => one_slot_break_dv_ready
    );

  buff : entity work.handshake_addf_1_buff(arch)
    port map(
      clk,
      rst,
      join_valid,
      one_slot_break_dv_ready,
      buff_valid
    );

  ieee2nfloat_lhs: entity work.InputIEEE_32bit(arch)
    port map (
        X => lhs,
        R => ip_lhs
    );

  ieee2nfloat_rhs: entity work.InputIEEE_32bit(arch)
    port map (
        X => rhs,
        R => ip_rhs
    );

  nfloat2ieee_result : entity work.OutputIEEE_32bit(arch)
    port map (
        X => ip_result,
        R => result
    );

  ip : entity work.FloatingPointAdder(arch)
    port map (
        clk => clk,
        ce  => one_slot_break_dv_ready,
        X   => ip_lhs,
        Y   => ip_rhs,
        R   => ip_result
    );
end architecture;

