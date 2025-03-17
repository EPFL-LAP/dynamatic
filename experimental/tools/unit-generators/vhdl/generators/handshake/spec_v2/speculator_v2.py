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
  type LOOP_STATE is (IDLE, LOOP, EXIT);
  type SPEC_STATE is (DEFAULT, MISSPEC);
  signal loop_mode : LOOP_STATE;
  signal spec_mode : SPEC_STATE;
begin
  condition_ready <= (condition_spec and (spec_mode = MISSPEC)) or
                     (commitCtrl_ready and (loop_mode = LOOP));

  trigger_ready <= muxCtrl_ready and (loop_mode = IDLE);

  ctrlOut_valid <= (loop_mode = EXIT) and muxCtrl_ready;

  muxCtrl_valid <= (trigger_valid and (loop_mode = IDLE)) or
                    (ctrlOut_ready and (loop_mode = EXIT)) or
                    (loop_mode = LOOP);
  muxCtrl <= "10" when (loop_mode = IDLE) else
             "01" when (loop_mode = EXIT) else
             "11";

  commitCtrl_valid <= condition_valid and ((spec_mode = DEFAULT) or not condition_spec(0)) and (loop_mode = LOOP);
  commitCtrl <= condition;

  loop_mode_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        loop_mode <= IDLE;
      else
        if (trigger_ready and trigger_valid) then
          loop_mode <= LOOP;
        elsif ((not condition) and condition_valid and condition_ready) then
          loop_mode <= EXIT;
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
        spec_mode <= DEFAULT;
      else
        if (condition_valid and condition_ready) then
          if (condition_spec(0)) then
            spec_mode <= MISSPEC;
          else
            spec_mode <= DEFAULT;
          end if;
        else
          spec_mode <= spec_mode;
        end if;
      end if;
    end if;
  end process;
end architecture;
"""
