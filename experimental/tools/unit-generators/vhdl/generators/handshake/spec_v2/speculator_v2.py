def generate_speculator_v2(name, params):
  return f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of speculator_v2
entity {name} is
  port (
    clk : in std_logic;
    rst : in std_logic;

    condition : in std_logic_vector(0 downto 0);
    condition_spec : in std_logic_vector(0 downto 0);
    condition_valid : in std_logic;
    condition_ready : out std_logic;

    trigger_valid : in std_logic;
    trigger_ready : out std_logic;

    ctrlOut_valid : out std_logic;
    ctrlOut_ready : in std_logic;

    muxCtrl : out std_logic_vector(1 downto 0);
    muxCtrl_valid : out std_logic;
    muxCtrl_ready : in std_logic;

    commitCtrl : out std_logic_vector(0 downto 0);
    commitCtrl_valid : out std_logic;
    commitCtrl_ready : in std_logic
  );
end entity;

-- Architecture of speculator_v2
architecture arch of {name} is
  type LOOP_STATE is (IDLE, CONTINUE, LOOP_EXIT);
  type SPEC_STATE is (SPEC_DEFAULT, MISSPEC);
  signal loop_mode : LOOP_STATE;
  signal spec_mode : SPEC_STATE;

  signal loop_mode_idle, loop_mode_continue, loop_mode_exit : std_logic;
  signal spec_mode_default, spec_mode_misspec : std_logic;
begin
  loop_mode_idle <= '1' when (loop_mode = IDLE) else '0';
  loop_mode_continue <= '1' when (loop_mode = CONTINUE) else '0';
  loop_mode_exit <= '1' when (loop_mode = LOOP_EXIT) else '0';

  spec_mode_default <= '1' when (spec_mode = SPEC_DEFAULT) else '0';
  spec_mode_misspec <= '1' when (spec_mode = MISSPEC) else '0';

  condition_ready <= (condition_spec(0) and spec_mode_misspec) or
                     (commitCtrl_ready and loop_mode_continue);

  trigger_ready <= muxCtrl_ready and loop_mode_idle;

  ctrlOut_valid <= loop_mode_exit;

  muxCtrl_valid <= (trigger_valid and loop_mode_idle) or
                    (ctrlOut_ready and loop_mode_exit) or
                    loop_mode_continue;
  muxCtrl <= "10" when loop_mode_idle else
             "01" when loop_mode_exit else
             "11";

  commitCtrl_valid <= condition_valid and (spec_mode_default or not condition_spec(0)) and loop_mode_continue;
  commitCtrl <= condition;

  loop_mode_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        loop_mode <= IDLE;
      else
        if (trigger_ready and trigger_valid) then
          loop_mode <= CONTINUE;
        elsif ((not condition(0)) and condition_valid and condition_ready) then
          loop_mode <= LOOP_EXIT;
        elsif (ctrlOut_valid and ctrlOut_ready) then
          loop_mode <= IDLE;
        else
          loop_mode <= loop_mode;
        end if;
      end if;
    end if;
  end process;

  spec_mode_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        spec_mode <= SPEC_DEFAULT;
      else
        if (condition_valid and condition_ready) then
          if (not condition(0)) then
            spec_mode <= MISSPEC;
          else
            spec_mode <= SPEC_DEFAULT;
          end if;
        else
          spec_mode <= spec_mode;
        end if;
      end if;
    end if;
  end process;
end architecture;
"""
