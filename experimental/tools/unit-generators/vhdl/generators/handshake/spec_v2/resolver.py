from generators.handshake.join import generate_join


def generate_spec_v2_resolver(name, _):
    join_name = f"{name}_join"
    dependencies = generate_join(join_name, {"size": 2})

    return dependencies + f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of resolver
entity {name} is
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
architecture arch of {name} is
  signal idle : std_logic;
  signal transfer : std_logic;
begin
  transfer <= actualCondition_valid and generatedCondition_valid and confirmSpec_ready;
  join_inputs : entity work.{join_name}(arch)
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
"""
