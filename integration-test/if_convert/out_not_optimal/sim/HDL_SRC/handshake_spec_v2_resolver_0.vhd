-- handshake_spec_v2_resolver_0 : spec_v2_resolver({})


library ieee;
use ieee.std_logic_1164.all;

-- Entity of and_n
entity handshake_spec_v2_resolver_0_join_and_n is
  port (
    -- inputs
    ins : in std_logic_vector(2 - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

-- Architecture of and_n
architecture arch of handshake_spec_v2_resolver_0_join_and_n is
  signal all_ones : std_logic_vector(2 - 1 downto 0) := (others => '1');
begin
  outs <= '1' when ins = all_ones else '0';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of join
entity handshake_spec_v2_resolver_0_join is
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
architecture arch of handshake_spec_v2_resolver_0_join is
  signal allValid : std_logic;
begin
  allValidAndGate : entity work.handshake_spec_v2_resolver_0_join_and_n port map(ins_valid, allValid);
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

-- Entity of resolver
entity handshake_spec_v2_resolver_0 is
  port (
    clk, rst : in std_logic;
    actualCondition : in std_logic_vector(0 downto 0);
    actualCondition_valid : in std_logic;
    actualCondition_ready : out std_logic;
    generatedCondition : in std_logic_vector(0 downto 0);
    generatedCondition_valid : in std_logic;
    generatedCondition_ready : out std_logic;
    confirmSpec : out std_logic_vector(0 downto 0);
    confirmSpec_valid : out std_logic;
    confirmSpec_ready : in std_logic
  );
end entity;

-- Architecture of resolver
architecture arch of handshake_spec_v2_resolver_0 is
  signal idle : std_logic;
  signal transfer : std_logic;
begin
  transfer <= actualCondition_valid and generatedCondition_valid and confirmSpec_ready;
  join_inputs : entity work.handshake_spec_v2_resolver_0_join(arch)
    port map(
      -- inputs
      ins_valid(0) => actualCondition_valid,
      ins_valid(1) => generatedCondition_valid,
      outs_ready   => confirmSpec_ready,
      -- outputs
      outs_valid   => confirmSpec_valid,
      ins_ready(0) => actualCondition_ready,
      ins_ready(1) => generatedCondition_ready
    );

  confirmSpec(0) <= idle;

  idle_proc : process(clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        idle <= '1';
      else
        if transfer then
          if idle then
            if not actualCondition(0) then
              if generatedCondition(0) then
                idle <= '0';
              end if;
            end if;
          else
            if not generatedCondition(0) then
              idle <= '1';
            end if;
          end if;
        end if;
      end if;
    end if;
  end process;
end architecture;

