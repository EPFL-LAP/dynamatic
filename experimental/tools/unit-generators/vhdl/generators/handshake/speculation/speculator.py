from generators.handshake.fork import generate_fork
from generators.handshake.tehb import generate_tehb
from generators.support.signal_manager import generate_spec_units_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth


def generate_speculator(name, params):
    bitwidth = params["bitwidth"]
    fifo_depth = params["fifo_depth"]
    extra_signals = params["extra_signals"]

    # Always contains spec signal
    if len(extra_signals) > 1:
        return _generate_speculator_signal_manager(name, bitwidth, fifo_depth, extra_signals)
    return _generate_speculator(name, bitwidth, fifo_depth)


def _generate_specGen_core(name, bitwidth):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of specgenCore
entity {name} is
  port (
    clk, rst : in std_logic;

    ins : in std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid : in std_logic;
    ins_spec : in std_logic_vector(0 downto 0);
    ins_ready : out std_logic;

    predict_ins : in std_logic_vector({bitwidth} - 1 downto 0);
    predict_ins_valid : in std_logic;
    predict_ins_spec : in std_logic_vector(0 downto 0);
    predict_ins_ready : out std_logic;

    fifo_ins : in std_logic_vector({bitwidth} - 1 downto 0);
    fifo_ins_valid : in std_logic;
    fifo_ins_ready : out std_logic;

    outs : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_spec : out std_logic_vector(0 downto 0);

    fifo_outs : out std_logic_vector({bitwidth} - 1 downto 0);
    fifo_outs_valid : out std_logic;
    fifo_outs_ready : in std_logic;

    control_outs : out std_logic_vector(2 downto 0); -- 000:spec, 001:no cmp, 010:cmp correct, 011:resend, 100:kill, 101:correct-spec
    control_outs_valid : out std_logic;
    control_outs_ready : in std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of specgenCore
architecture arch of {name} is
  type State_type is (IDLE, KILL, KILL_ONLY_DATA);
  type Control_type is (CONTROL_SPEC, CONTROL_NO_CMP, CONTROL_CMP_CORRECT, CONTROL_RESEND, CONTROL_KILL, CONTROL_CORRECT_SPEC);
  signal State : State_type;

  signal DatapV : std_logic;
  signal PredictpV : std_logic;
  signal FifoNotEmpty : std_logic;
  signal ControlnR : std_logic;
  signal FifoNotFull : std_logic;

  signal DataR : std_logic;
  signal PredictR : std_logic;
  signal FifoR : std_logic;
  signal ControlV : std_logic;
  signal FifoV : std_logic;

  signal ControlInternal : Control_type;
begin
  DatapV <= ins_valid;
  PredictpV <= predict_ins_valid;
  FifoNotEmpty <= fifo_ins_valid;
  ControlnR <= control_outs_ready;
  FifoNotFull <= fifo_outs_ready;

  ins_ready <= DataR;
  predict_ins_ready <= PredictR;
  fifo_ins_ready <= FifoR;
  control_outs_valid <= ControlV;
  fifo_outs_valid <= FifoV;

  process (ControlInternal)
  begin
    case ControlInternal is
      when CONTROL_SPEC =>
        control_outs <= "000";
      when CONTROL_NO_CMP =>
        control_outs <= "001";
      when CONTROL_CMP_CORRECT =>
        control_outs <= "010";
      when CONTROL_RESEND =>
        control_outs <= "011";
      when CONTROL_KILL =>
        control_outs <= "100";
      when CONTROL_CORRECT_SPEC =>
        control_outs <= "101";
    end case;
  end process;

  process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        State <= IDLE;
      else
        case State is
          when IDLE =>
            if (DatapV = '1' and FifoNotEmpty = '1' and ins /= fifo_ins and ControlnR = '1') then
              State <= KILL;
            end if;
          when KILL =>
            if (FifoNotEmpty = '0' and -- Killed all data in FIFO
                PredictpV = '1' and predict_ins_spec = "0") then -- Killed incoming spec trigger
                if (DatapV = '1' and ins_spec = "0") then
                  -- Already killed incoming spec data
                  State <= IDLE;
                else
                  -- Wait for all misspec tokens, but accept new speculation
                  State <= KILL_ONLY_DATA;
                end if;
            end if;
          when KILL_ONLY_DATA =>
            if (DatapV = '1' and ins_spec = "0") then
              State <= IDLE;
            end if;
        end case;
      end if;
    end if;
  end process;

  process (State, ins, ins_spec, fifo_ins, predict_ins, predict_ins_spec,
           DatapV, PredictpV, FifoNotEmpty, ControlnR, FifoNotFull)
  begin
    outs <= ins;
    outs_spec <= "0";
    fifo_outs <= predict_ins;
    ControlInternal <= CONTROL_SPEC;

    case State is
      when IDLE =>
        if (DatapV = '0' and PredictpV = '1' and FifoNotFull = '1') then
          DataR <= ControlnR;
          PredictR <= ControlnR;
          FifoV <= ControlnR;
          FifoR <= '0';

          ControlV <= '1';
          ControlInternal <= CONTROL_SPEC;
          outs <= predict_ins;
          outs_spec <= "1";
        elsif (DatapV = '1' and PredictpV = '1' and FifoNotEmpty = '0') then
          DataR <= ControlnR;
          PredictR <= ControlnR;

          FifoV <= '0';
          FifoR <= '0';

          ControlV <= '1';
          ControlInternal <= CONTROL_NO_CMP;
          outs <= ins;
          outs_spec <= "0";
        elsif (DatapV = '1' and PredictpV = '1' and FifoNotEmpty = '1' and FifoNotFull = '1' and ins = fifo_ins) then
          DataR <= ControlnR;
          PredictR <= ControlnR;

          ControlV <= '1';
          ControlInternal <= CONTROL_CORRECT_SPEC;
          outs <= predict_ins;
          outs_spec <= "1";
          FifoV <= ControlnR;
          FifoR <= ControlnR;
        elsif ((DatapV = '1' and PredictpV = '0' and FifoNotEmpty = '1' and ins = fifo_ins) or
               (DatapV = '1' and PredictpV = '1' and FifoNotFull = '0' and ins = fifo_ins)) then
          DataR <= ControlnR;
          PredictR <= '0';
          FifoR <= ControlnR;

          FifoV <= '0';

          ControlV <= '1';
          ControlInternal <= CONTROL_CMP_CORRECT;
        elsif (DatapV = '1' and FifoNotEmpty = '1' and ins /= fifo_ins) then
          DataR <= ControlnR;
          PredictR <= '0';
          FifoV <= '0';
          FifoR <= '0';
          ControlV <= '1';
          ControlInternal <= CONTROL_RESEND;
          outs <= ins;
          outs_spec <= "0";
        else
          DataR <= '0';
          PredictR <= '0';
          ControlV <= '0';
          FifoR <= '0';
          FifoV <= '0';
        end if;
      when KILL =>
        -- Connect FIFO with Control
        FifoR <= ControlnR;
        ControlV <= FifoNotEmpty;

        -- Emit kill signal
        ControlInternal <= CONTROL_KILL;

        -- Accepts spec trigger to kill it
        PredictR <= predict_ins_spec(0);

        -- Accepts spec data to kill it
        DataR <= ins_spec(0);

        -- Never pushes new data to fifo
        FifoV <= '0';
      when KILL_ONLY_DATA =>
        -- Accepts spec data to kill it
        DataR <= ins_spec(0);

        -- Accepts new speculation if no backpressure
        PredictR <= ControlnR and FifoNotFull;

        -- New speculation pushes to FIFO and emits a control if no backpressure
        FifoV <= PredictpV and ControlnR;
        ControlV <= PredictpV and FifoNotFull;

        -- Control signal is always SPEC
        ControlInternal <= CONTROL_SPEC;
        outs <= predict_ins;
        outs_spec <= "1";

        -- Never pops from FIFO
        FifoR <= '0';
    end case;
  end process;
end architecture;
"""

    return entity + architecture


def _generate_decodeSave(name):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of decodeSave
entity {name} is
  port (
    control_in : in std_logic_vector(2 downto 0);
    control_in_valid : in std_logic;
    control_in_ready : out std_logic;

    control_out : out std_logic_vector(0 downto 0); -- 0:resend, 1:drop
    control_out_valid : out std_logic;
    control_out_ready : in std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of decodeSave
architecture arch of {name} is
begin
  process (control_in, control_in_valid, control_out_ready)
  begin
    if (control_in = "001" or control_in = "010" or control_in = "011" or control_in = "101") then
      control_in_ready <= control_out_ready;
    else
      control_in_ready <= '1';
    end if;

    control_out_valid <= '0';
    control_out(0) <= '0';

    if (control_in_valid = '1') then
      if control_in = "001" then -- no cmp
        control_out_valid <= '1';
        control_out(0) <= '1';
      elsif control_in = "010" then -- cmp correct
        control_out_valid <= '1';
        control_out(0) <= '1';
      elsif control_in = "101" then -- correct-spec
        control_out_valid <= '1';
        control_out(0) <= '1';
      elsif control_in = "011" then --cmp wrong
        control_out_valid <= '1';
        control_out(0) <= '0';
      end if;
    end if;
  end process;
end architecture;
"""

    return entity + architecture


def _generate_decodeCommit(name):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of decodeCommit
entity {name} is
  port (
    control_in : in std_logic_vector(2 downto 0);
    control_in_valid : in std_logic;
    control_in_ready : out std_logic;

    control_out : out std_logic_vector(0 downto 0); -- 0:pass, 1:discard
    control_out_valid : out std_logic;
    control_out_ready : in std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of decodeCommit
architecture arch of {name} is
begin
  process (control_in, control_in_valid, control_out_ready)
  begin
    if (control_in = "010" or control_in = "100" or control_in = "101") then
      control_in_ready <= control_out_ready;
    else
      control_in_ready <= '1';
    end if;

    control_out_valid <= '0';
    control_out(0) <= '0';

    if (control_in_valid = '1') then
      if control_in = "010" then -- cmp correct
        control_out_valid <= '1';
        control_out(0) <= '0';
      elsif control_in = "101" then -- correct-spec
        control_out_valid <= '1';
        control_out(0) <= '0';
      elsif control_in = "100" then -- cmp wrong
        control_out_valid <= '1';
        control_out(0) <= '1';
      end if;
    end if;
  end process;
end architecture;
"""

    return entity + architecture


def _generate_decodeBranch(name):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of decodeBranch
entity {name} is
  port (
    control_in : in std_logic_vector(2 downto 0);
    control_in_valid : in std_logic;
    control_in_ready : out std_logic;

    control_out : out std_logic_vector(0 downto 0); -- 1:pass, 0:discard
    control_out_valid : out std_logic;
    control_out_ready : in std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of decodeBranch
architecture arch of {name} is
begin
  process (control_in, control_in_valid, control_out_ready)
  begin
    if (control_in = "010" or control_in = "100" or control_in = "101") then
      control_in_ready <= control_out_ready;
    else
      control_in_ready <= '1';
    end if;

    control_out_valid <= '0';
    control_out(0) <= '0';

    if (control_in_valid = '1') then
      if control_in = "010" then -- cmp correct
        control_out_valid <= '1';
        control_out(0) <= '0';
      elsif control_in = "101" then -- correct-spec
        control_out_valid <= '1';
        control_out(0) <= '0';
      elsif control_in = "100" then --cmp wrong
        control_out_valid <= '1';
        control_out(0) <= '1';
      end if;
    end if;
  end process;
end architecture;
"""

    return entity + architecture


def _generate_decodeSC(name):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of decodeSC
entity {name} is
  port (
    control_in : in std_logic_vector(2 downto 0);
    control_in_valid : in std_logic;
    control_in_ready : out std_logic;

    control_out0 : out std_logic_vector(2 downto 0); -- 000:pass, 001:kill, 010:resend, 011:kill-pass, 100:no_cmp
    control_out0_valid : out std_logic;
    control_out0_ready : in std_logic;

    control_out1 : out std_logic_vector(2 downto 0); -- 000:pass, 001:kill, 010:resend, 011:kill-pass, 100:no_cmp
    control_out1_valid : out std_logic;
    control_out1_ready : in std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of decodeSC
architecture arch of {name} is
begin
  process (control_in, control_in_valid,
           control_out0_ready, control_out1_ready)
  begin
    if (control_in = "000" or control_in = "001" or control_in = "010" or control_in = "011" or control_in = "101") then
      control_in_ready <= control_out0_ready;
    else
      control_in_ready <= control_out1_ready;
    end if;

    control_out0_valid <= '0';
    control_out1_valid <= '0';
    control_out0 <= "000";
    control_out1 <= "000";

    if (control_in_valid = '1') then
      if control_in = "000" then -- spec
        control_out0_valid <= '1';
        control_out0 <= "000";
      elsif control_in = "001" then -- no cmp
        control_out0_valid <= '1';
        control_out0 <= "100";
      elsif control_in = "010" then -- cmp correct
        control_out0_valid <= '1';
        control_out0 <= "001";
      elsif control_in = "101" then -- correct-spec
        control_out0_valid <= '1';
        control_out0 <= "011";
      elsif control_in = "011" then -- cmp wrong resend
        control_out0_valid <= '1';
        control_out0 <= "010";
      elsif control_in = "100" then -- cmp wrong kill
        control_out1_valid <= '1';
        control_out1 <= "001";
      end if;
    end if;
  end process;
end architecture;
"""

    return entity + architecture


def _generate_decodeOutput(name, bitwidth):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of decodeOutput
entity {name} is
  port (
    control_in : in std_logic_vector(2 downto 0);
    control_in_valid : in std_logic;
    control_in_ready : out std_logic;

    tehb_outs : in std_logic_vector({bitwidth} - 1 downto 0);
    tehb_outs_spec : in std_logic_vector(0 downto 0);

    outs : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_spec : out std_logic_vector(0 downto 0);
    outs_valid : out std_logic;
    outs_ready : in std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of decodeOutput
architecture arch of {name} is
begin
  -- Forward outs data and spec bit
  outs <= tehb_outs;
  outs_spec <= tehb_outs_spec;

  process (control_in, control_in_valid, outs_ready)
  begin
    if (control_in = "000" or control_in = "001" or control_in = "011" or control_in = "101") then
      control_in_ready <= outs_ready;
    else
      control_in_ready <= '1';
    end if;

    outs_valid <= '0';

    if (control_in_valid = '1') then
      if control_in = "000" then -- spec
        outs_valid <= '1';
      elsif control_in = "101" then -- correct-spec
        outs_valid <= '1';
      elsif control_in = "001" then -- no cmp
        outs_valid <= '1';
      elsif control_in = "011" then -- cmp wrong resend
        outs_valid <= '1';
      end if;
    end if;
  end process;
end architecture;
"""

    return entity + architecture


def _generate_predictor(name, bitwidth):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of predictor
entity {name} is
  port (
    clk, rst     : in  std_logic;

    trigger_valid : in std_logic;
    trigger_spec : in std_logic_vector(0 downto 0);
    trigger_ready : out std_logic;

    data_in      : in std_logic_vector({bitwidth} - 1 downto 0);
    data_in_valid : in std_logic;
    data_in_ready : out std_logic;

    data_out     : out std_logic_vector({bitwidth} - 1 downto 0);
    data_out_valid : out std_logic;

    -- Whether the data is triggered by spec trigger or not
    data_out_spec : out std_logic_vector(0 downto 0);

    data_out_ready : in std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of predictor
architecture arch of {name} is
  signal zeros : std_logic_vector({bitwidth}-2 downto 0);
  signal data_reg: std_logic_vector({bitwidth}-1 downto 0);
begin
  zeros <= (others => '0');

  -- Predicted value is 1 by default and updated to the latest real value
  process(clk, rst) is
  begin
    if (rst = '1') then
      data_reg <= zeros & '1';
    elsif (rising_edge(clk)) then
      if (data_in_valid = '1') then
        data_reg <= data_in;
      end if;
    end if;
  end process;

  data_in_ready <= '1';

  data_out <= data_reg;
  data_out_valid <= trigger_valid;
  trigger_ready <= data_out_ready;

  data_out_spec <= trigger_spec;
end architecture;
"""

    return entity + architecture


def _generate_predFifo(name, bitwidth, fifo_depth):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of predFifo
entity {name} is
  port (
    clk, rst : in std_logic;

    data_in : in std_logic_vector({bitwidth} - 1 downto 0);
    data_in_valid : in std_logic;
    data_in_ready : out std_logic;

    data_out : out std_logic_vector({bitwidth} - 1 downto 0);
    data_out_valid : out std_logic;
    data_out_ready : in std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of predFifo
architecture arch of {name} is
  signal HeadEn   : std_logic := '0';
  signal TailEn  : std_logic := '0';

  signal Tail : natural range 0 to {fifo_depth} - 1;
  signal Head : natural range 0 to {fifo_depth} - 1;

  signal Empty    : std_logic;
  signal Full : std_logic;

  type FIFO_Memory is array (0 to {fifo_depth} - 1) of STD_LOGIC_VECTOR ({bitwidth}-1 downto 0);
  signal Memory : FIFO_Memory;
begin
  data_out_valid <= not Empty;
  data_in_ready <= not Full;

  TailEn <= not Full and data_in_valid;
  HeadEn <= not Empty and data_out_ready;
  data_out <= Memory(Head);

  ----------------------------------------------------------------
  -- Sequential Process
  ----------------------------------------------------------------

  -------------------------------------------
  -- process for writing data
  -------------------------------------------
  fifo_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        -- TODO: Nothing??
      else
        if (TailEn = '1' ) then
          -- Write Data to Memory
          Memory(Tail) <= data_in;
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating tail
  ----------------------------------------------------------------
  TailUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        Tail <= 0;
      else
        if (TailEn = '1') then
          Tail  <= (Tail + 1) mod {fifo_depth};
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating head
  -------------------------------------------
  HeadUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        Head <= 0;
      else
        if (HeadEn = '1') then
          Head  <= (Head + 1) mod {fifo_depth};
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating full
  -------------------------------------------
  FullUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        Full <= '0';
      else
        -- if only filling but not emptying
        if (TailEn = '1') and (HeadEn = '0') then
          -- if new tail index will reach head index
          if ((Tail +1) mod {fifo_depth} = Head) then
            Full  <= '1';
          end if;
        elsif (TailEn = '0') and (HeadEn = '1') then
          -- if only emptying but not filling
          Full <= '0';
        -- otherwise, nothing is happening or simultaneous read and write
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating empty
  -------------------------------------------
  EmptyUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        Empty <= '1';
      else
        -- if only emptying but not filling
        if (TailEn = '0') and (HeadEn = '1') then
          -- if new head index will reach tail index
          if ((Head +1) mod {fifo_depth} = Tail) then
            Empty  <= '1';
          end if;
        elsif (TailEn = '1') and (HeadEn = '0') then
          -- if only filling but not emptying
            Empty <= '0';
        -- otherwise, nothing is happening or simultaneous read and write
        end if;
      end if;
    end if;
  end process;
end architecture;
"""

    return entity + architecture


def _generate_speculator(name, bitwidth, fifo_depth):
    data_fork_name = f"{name}_data_fork"
    specGen_name = f"{name}_specGen"
    predictor_name = f"{name}_predictor"
    predFifo_name = f"{name}_predFifo"
    control_fork_name = f"{name}_control_fork"
    decodeSave_name = f"{name}_decodeSave"
    decodeCommit_name = f"{name}_decodeCommit"
    decodeSC_name = f"{name}_decodeSC"
    decodeOutput_name = f"{name}_decodeOutput"
    decodeBranch_name = f"{name}_decodeBranch"
    tehb_name = f"{name}_tehb"

    dependencies = \
        generate_fork(data_fork_name, {
            "size": 2,
            "bitwidth": bitwidth,
            "extra_signals": {"spec": 1}
        }) + \
        _generate_specGen_core(specGen_name, bitwidth) + \
        _generate_predictor(predictor_name, bitwidth) + \
        _generate_predFifo(predFifo_name, bitwidth, fifo_depth) + \
        generate_fork(control_fork_name, {
            "size": 5,
            "bitwidth": 3
        }) + \
        _generate_decodeSave(decodeSave_name) + \
        _generate_decodeCommit(decodeCommit_name) + \
        _generate_decodeSC(decodeSC_name) + \
        _generate_decodeOutput(decodeOutput_name, bitwidth) + \
        _generate_decodeBranch(decodeBranch_name) + \
        generate_tehb(tehb_name, {
            "bitwidth": bitwidth,
            "extra_signals": {"internal_ctrl": 3, "spec": 1}
        })

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;

-- Entity of speculator
entity {name} is
  port (
    clk, rst : in  std_logic;
    -- inputs
    ins: in std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid: in std_logic;
    ins_spec: in std_logic_vector(0 downto 0);
    ins_ready: out std_logic;
    -- trigger is dataless (control token)
    trigger_valid: in std_logic;
    trigger_spec: in std_logic_vector(0 downto 0);
    trigger_ready: out std_logic;
    -- outputs
    outs: out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid: out std_logic;
    outs_spec: out std_logic_vector(0 downto 0);
    outs_ready: in std_logic;
    -- control signals
    ctrl_save: out std_logic_vector(0 downto 0);
    ctrl_save_valid: out std_logic;
    ctrl_save_ready: in std_logic;
    ctrl_commit: out std_logic_vector(0 downto 0);
    ctrl_commit_valid: out std_logic;
    ctrl_commit_ready: in std_logic;
    ctrl_sc_save: out std_logic_vector(2 downto 0);
    ctrl_sc_save_valid: out std_logic;
    ctrl_sc_save_ready: in std_logic;
    ctrl_sc_commit: out std_logic_vector(2 downto 0);
    ctrl_sc_commit_valid: out std_logic;
    ctrl_sc_commit_ready: in std_logic;
    ctrl_sc_branch: out std_logic_vector(0 downto 0);
    ctrl_sc_branch_valid: out std_logic;
    ctrl_sc_branch_ready: in std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of speculator
architecture arch of {name} is
  signal fork_data_outs : data_array(1 downto 0)({bitwidth} - 1 downto 0);
  signal fork_data_outs_valid : std_logic_vector(1 downto 0);
  signal fork_data_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork_data_outs_ready : std_logic_vector(1 downto 0);

  signal predictor_data_out : std_logic_vector({bitwidth} - 1 downto 0);
  signal predictor_data_out_valid : std_logic;
  signal predictor_data_out_spec : std_logic_vector(0 downto 0);
  signal predictor_data_out_ready : std_logic;

  signal specgenCore_outs : std_logic_vector({bitwidth} - 1 downto 0);
  signal specgenCore_outs_spec : std_logic_vector(0 downto 0);
  signal specgenCore_fifo_outs : std_logic_vector({bitwidth} - 1 downto 0);
  signal specgenCore_fifo_outs_valid : std_logic;
  signal specgenCore_fifo_outs_ready : std_logic;

  signal specgenCore_control_outs : std_logic_vector(2 downto 0);
  signal specgenCore_control_outs_valid : std_logic;
  signal specgenCore_control_outs_ready : std_logic;

  signal tehb_outs : std_logic_vector({bitwidth} - 1 downto 0);
  signal tehb_outs_spec : std_logic_vector(0 downto 0);
  signal tehb_control_outs : std_logic_vector(2 downto 0);
  signal tehb_control_outs_valid : std_logic;
  signal tehb_control_outs_ready : std_logic;

  signal predFifo_data_out : std_logic_vector({bitwidth} - 1 downto 0);
  signal predFifo_data_out_valid : std_logic;
  signal predFifo_data_out_ready : std_logic;

  signal fork_control_outs : data_array(4 downto 0)(2 downto 0);
  signal fork_control_outs_valid : std_logic_vector(4 downto 0);
  signal fork_control_outs_ready : std_logic_vector(4 downto 0);
begin
  data_fork: entity work.{data_fork_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins,
      ins_valid => ins_valid,
      ins_spec => ins_spec,
      ins_ready => ins_ready,
      outs => fork_data_outs,
      outs_valid => fork_data_outs_valid,
      outs_0_spec => fork_data_outs_0_spec,
      outs_1_spec => open,
      outs_ready => fork_data_outs_ready
    );

  spengenCore0: entity work.{specGen_name}(arch)
    port map (
      clk => clk,
      rst => rst,

      ins => fork_data_outs(0),
      ins_valid => fork_data_outs_valid(0),
      ins_spec => fork_data_outs_0_spec,
      ins_ready => fork_data_outs_ready(0),

      predict_ins => predictor_data_out,
      predict_ins_valid => predictor_data_out_valid,
      predict_ins_spec => predictor_data_out_spec,
      predict_ins_ready => predictor_data_out_ready,

      fifo_ins => predFifo_data_out,
      fifo_ins_valid => predFifo_data_out_valid,
      fifo_ins_ready => predFifo_data_out_ready,

      outs => specgenCore_outs,
      outs_spec => specgenCore_outs_spec,

      fifo_outs => specgenCore_fifo_outs,
      fifo_outs_valid => specgenCore_fifo_outs_valid,
      fifo_outs_ready => specgenCore_fifo_outs_ready,

      control_outs => specgenCore_control_outs,
      control_outs_valid => specgenCore_control_outs_valid,
      control_outs_ready => specgenCore_control_outs_ready
    );

  predictor0: entity work.{predictor_name}(arch)
    port map (
      clk => clk,
      rst => rst,

      trigger_valid => trigger_valid,
      trigger_spec => trigger_spec,
      trigger_ready => trigger_ready,

      data_in => fork_data_outs(1),
      data_in_valid => fork_data_outs_valid(1),
      data_in_ready => fork_data_outs_ready(1),

      data_out => predictor_data_out,
      data_out_valid => predictor_data_out_valid,
      data_out_spec => predictor_data_out_spec,
      data_out_ready => predictor_data_out_ready
    );

  predFifo0: entity work.{predFifo_name}(arch)
    port map (
      clk => clk,
      rst => rst,

      data_in => specgenCore_fifo_outs,
      data_in_valid => specgenCore_fifo_outs_valid,
      data_in_ready => specgenCore_fifo_outs_ready,

      data_out => predFifo_data_out,
      data_out_valid => predFifo_data_out_valid,
      data_out_ready => predFifo_data_out_ready
    );

  tehb: entity work.{tehb_name}(arch)
    port map (
      clk => clk,
      rst => rst,
      ins => specgenCore_outs,
      ins_spec => specgenCore_outs_spec,
      ins_internal_ctrl => specgenCore_control_outs,
      ins_valid => specgenCore_control_outs_valid,
      ins_ready => specgenCore_control_outs_ready,
      outs => tehb_outs,
      outs_spec => tehb_outs_spec,
      outs_internal_ctrl => tehb_control_outs,
      outs_valid => tehb_control_outs_valid,
      outs_ready => tehb_control_outs_ready
    );

  fork0: entity work.{control_fork_name}(arch)
    port map (
      clk => clk,
      rst => rst,
      ins => tehb_control_outs,
      ins_valid => tehb_control_outs_valid,
      ins_ready => tehb_control_outs_ready,
      outs => fork_control_outs,
      outs_valid => fork_control_outs_valid,
      outs_ready => fork_control_outs_ready
    );

  decodeSave0: entity work.{decodeSave_name}(arch)
    port map (
      control_in => fork_control_outs(3),
      control_in_valid => fork_control_outs_valid(3),
      control_in_ready => fork_control_outs_ready(3),
      control_out => ctrl_save,
      control_out_valid => ctrl_save_valid,
      control_out_ready => ctrl_save_ready
    );

  decodeCommit0: entity work.{decodeCommit_name}(arch)
    port map (
      control_in => fork_control_outs(2),
      control_in_valid => fork_control_outs_valid(2),
      control_in_ready => fork_control_outs_ready(2),
      control_out => ctrl_commit,
      control_out_valid => ctrl_commit_valid,
      control_out_ready => ctrl_commit_ready
  );

  decodeSC0: entity work.{decodeSC_name}(arch)
    port map (
      control_in => fork_control_outs(1),
      control_in_valid => fork_control_outs_valid(1),
      control_in_ready => fork_control_outs_ready(1),
      control_out0 => ctrl_sc_save,
      control_out0_valid => ctrl_sc_save_valid,
      control_out0_ready => ctrl_sc_save_ready,
      control_out1 => ctrl_sc_commit,
      control_out1_valid => ctrl_sc_commit_valid,
      control_out1_ready => ctrl_sc_commit_ready
    );

  decodeOutput0: entity work.{decodeOutput_name}(arch)
    port map (
      control_in => fork_control_outs(4),
      control_in_valid => fork_control_outs_valid(4),
      control_in_ready => fork_control_outs_ready(4),
      tehb_outs => tehb_outs,
      tehb_outs_spec => tehb_outs_spec,
      outs => outs,
      outs_spec => outs_spec,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

  decodeBranch0: entity work.{decodeBranch_name}(arch)
    port map (
      control_in => fork_control_outs(0),
      control_in_valid => fork_control_outs_valid(0),
      control_in_ready => fork_control_outs_ready(0),
      control_out => ctrl_sc_branch,
      control_out_valid => ctrl_sc_branch_valid,
      control_out_ready => ctrl_sc_branch_ready
    );
end architecture;
"""

    return dependencies + entity + architecture


def _generate_speculator_signal_manager(name, bitwidth, fifo_depth, extra_signals):
    extra_signals_without_spec = extra_signals.copy()
    extra_signals_without_spec.pop("spec")

    extra_signals_bitwidth = get_concat_extra_signals_bitwidth(
        extra_signals)
    return generate_spec_units_signal_manager(
        name,
        [{
            "name": "ins",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }, {
            "name": "trigger",
            "bitwidth": 0,
            "extra_signals": extra_signals
        }],
        [{
            "name": "outs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }, {
            "name": "ctrl_save",
            "bitwidth": 1,
            "extra_signals": {}
        }, {
            "name": "ctrl_commit",
            "bitwidth": 1,
            "extra_signals": {}
        }, {
            "name": "ctrl_sc_save",
            "bitwidth": 3,
            "extra_signals": {}
        }, {
            "name": "ctrl_sc_commit",
            "bitwidth": 3,
            "extra_signals": {}
        }, {
            "name": "ctrl_sc_branch",
            "bitwidth": 1,
            "extra_signals": {}
        }],
        extra_signals_without_spec,
        ["ctrl_save", "ctrl_commit", "ctrl_sc_save",
         "ctrl_sc_commit", "ctrl_sc_branch"],
        lambda name: _generate_speculator(name, bitwidth + extra_signals_bitwidth - 1, fifo_depth))
