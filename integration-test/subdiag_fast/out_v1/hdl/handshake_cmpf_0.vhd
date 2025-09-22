-- handshake_cmpf_0 : cmpf({'is_double': False, 'extra_signals': {'spec': 1}, 'predicate': 'ugt'})


library ieee;
use ieee.std_logic_1164.all;

-- Entity of and_n
entity handshake_cmpf_0_inner_inner_join_and_n is
  port (
    -- inputs
    ins : in std_logic_vector(2 - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

-- Architecture of and_n
architecture arch of handshake_cmpf_0_inner_inner_join_and_n is
  signal all_ones : std_logic_vector(2 - 1 downto 0) := (others => '1');
begin
  outs <= '1' when ins = all_ones else '0';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of join
entity handshake_cmpf_0_inner_inner_join is
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
architecture arch of handshake_cmpf_0_inner_inner_join is
  signal allValid : std_logic;
begin
  allValidAndGate : entity work.handshake_cmpf_0_inner_inner_join_and_n port map(ins_valid, allValid);
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

-- Entity of cmpf_single_precision
entity handshake_cmpf_0_inner_inner is
  port(
    -- inputs
    clk: in std_logic;
    rst: in std_logic;
    lhs: in std_logic_vector(32 - 1 downto 0);
    lhs_valid: in std_logic;
    rhs: in std_logic_vector(32 - 1 downto 0);
    rhs_valid: in std_logic;
    result_ready: in std_logic;
    -- outputs
    unordered: out std_logic;
    XltY: out std_logic;
    XeqY: out std_logic;
    XgtY: out std_logic;
    XleY: out std_logic;
    XgeY: out std_logic;
    result_valid: out std_logic;
    lhs_ready: out std_logic;
    rhs_ready: out std_logic
  );
end entity;

-- Architecture of cmpf_single_precision
architecture arch of handshake_cmpf_0_inner_inner is
  signal ip_lhs: std_logic_vector(32 + 1 downto 0);
  signal ip_rhs: std_logic_vector(32 + 1 downto 0);
begin
  join_inputs: entity work.handshake_cmpf_0_inner_inner_join(arch)
    port map(
      -- inputs
      ins_valid(0)=> lhs_valid,
      ins_valid(1)=> rhs_valid,
      outs_ready=> result_ready,
      -- outputs
      outs_valid=> result_valid,
      ins_ready(0)=> lhs_ready,
      ins_ready(1)=> rhs_ready
    );

  ieee2nfloat_0: entity work.InputIEEE_32bit(arch)
    port map(
        --input
        X=> lhs,
        --output
        R=> ip_lhs
    );

  ieee2nfloat_1: entity work.InputIEEE_32bit(arch)
    port map(
        --input
        X=> rhs,
        --output
        R=> ip_rhs
    );
  operator: entity work.FPComparator_32bit(arch)
  port map (clk=> clk,
        ce=> '1',
        X=> ip_lhs,
        Y=> ip_rhs,
        unordered=> unordered,
        XltY=> XltY,
        XeqY=> XeqY,
        XgtY=> XgtY,
        XleY=> XleY,
        XgeY=> XgeY);
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of cmpf
entity handshake_cmpf_0_inner is
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
    result       : out std_logic_vector(0 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;

-- Architecture of cmpf
architecture arch of handshake_cmpf_0_inner is
  signal unordered : std_logic;
  signal XltY : std_logic;
  signal XeqY : std_logic;
  signal XgtY : std_logic;
  signal XleY : std_logic;
  signal XgeY : std_logic;
begin
  operator : entity work.handshake_cmpf_0_inner_inner(arch)
    port map(
      clk => clk,
      rst => rst,
      lhs => lhs,
      lhs_valid => lhs_valid,
      rhs => rhs,
      rhs_valid => rhs_valid,
      result_ready => result_ready,
      unordered => unordered,
      XltY => XltY,
      XeqY => XeqY,
      XgtY => XgtY,
      XleY => XleY,
      XgeY => XgeY,
      result_valid => result_valid,
      lhs_ready => lhs_ready,
      rhs_ready => rhs_ready
    );

  result(0) <= unordered or XgtY;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of one_slot_break_dv_dataless
entity handshake_cmpf_0_buff_inner is
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
architecture arch of handshake_cmpf_0_buff_inner is
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

-- Entity of one_slot_break_dv
entity handshake_cmpf_0_buff is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(1 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(1 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of one_slot_break_dv
architecture arch of handshake_cmpf_0_buff is
  signal regEn, inputReady : std_logic;
begin

  control : entity work.handshake_cmpf_0_buff_inner
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
use work.types.all;

-- Entity of signal manager
entity handshake_cmpf_0 is
  port(
    clk : in std_logic;
    rst : in std_logic;
    lhs : in std_logic_vector(32 - 1 downto 0);
    lhs_valid : in std_logic;
    lhs_ready : out std_logic;
    lhs_spec : in std_logic_vector(1 - 1 downto 0);
    rhs : in std_logic_vector(32 - 1 downto 0);
    rhs_valid : in std_logic;
    rhs_ready : out std_logic;
    rhs_spec : in std_logic_vector(1 - 1 downto 0);
    result : out std_logic_vector(1 - 1 downto 0);
    result_valid : out std_logic;
    result_ready : in std_logic;
    result_spec : out std_logic_vector(1 - 1 downto 0)
  );
end entity;

-- Architecture of signal manager (buffered)
architecture arch of handshake_cmpf_0 is
  signal forwarded_spec : std_logic_vector(0 downto 0);
  signal signals_pre_buffer : std_logic_vector(0 downto 0);
  signal signals_post_buffer : std_logic_vector(0 downto 0);
  signal sliced_spec : std_logic_vector(0 downto 0);
  signal transfer_in, transfer_out : std_logic;
begin
  -- Transfer signal assignments
  transfer_in <= lhs_valid and lhs_ready;
  transfer_out <= result_valid and result_ready;

  -- Forward extra signals
  forwarded_spec <= lhs_spec or rhs_spec;

  -- Concat/split extra signals for buffer input/output
  signals_pre_buffer(0 downto 0) <= forwarded_spec;
  sliced_spec <= signals_post_buffer(0 downto 0);

  -- Assign extra signals to output channels
  result_spec <= sliced_spec;

  inner : entity work.handshake_cmpf_0_inner(arch)
    port map(
      clk => clk,
      rst => rst,
      lhs => lhs,
      lhs_valid => lhs_valid,
      lhs_ready => lhs_ready,
      rhs => rhs,
      rhs_valid => rhs_valid,
      rhs_ready => rhs_ready,
      result => result,
      result_valid => result_valid,
      result_ready => result_ready
    );

  -- Generate fifo_break_dv to store extra signals
  -- num_slots = 1, bitwidth = 1
  buff : entity work.handshake_cmpf_0_buff(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => signals_pre_buffer,
      ins_valid => transfer_in,
      ins_ready => open,
      outs => signals_post_buffer,
      outs_valid => open,
      outs_ready => transfer_out
    );
end architecture;

