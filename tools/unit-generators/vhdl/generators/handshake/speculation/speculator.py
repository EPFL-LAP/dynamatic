from generators.handshake.fork import generate_fork
from generators.handshake.lazy_fork import generate_lazy_fork
from generators.support.signal_manager import generate_spec_units_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth


def generate_speculator(name, params):
    bitwidth = params["bitwidth"]
    fifo_depth = params["fifo_depth"]
    extra_signals = params["extra_signals"]

    if len(extra_signals) > 1:
        return _generate_speculator_signal_manager(name, bitwidth, fifo_depth, extra_signals)
    return _generate_speculator(name, bitwidth, fifo_depth)


def _generate_specGen_core(name, bitwidth):
    specFSM_name = f"{name}_specFSM"

    specFSM = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.spec_types.all;

-- Entity of specFSM
entity {specFSM_name} is
  port (
    clk : in std_logic;
    rst : in std_logic;

    data_in_valid : in std_logic;
    is_data_in_spec : in std_logic_vector(0 downto 0);

    pred_hist_in_valid : in std_logic;

    new_pred_valid : in std_logic;
    new_pred_spec : in std_logic_vector(0 downto 0);

    misspec_detected : in boolean;

    state : out State_type
  );
end entity;

-- Architecture of specFSM
architecture arch of {specFSM_name} is
  signal next_state : State_type;

  signal no_pred_history : boolean;
  signal nonspec_trigger_arrived : boolean;
  signal nonspec_data_arrived : boolean;
  signal ready_to_re_speculate : boolean;
begin

  -- The prediction history is empty,
  -- which is required to begin a new round of speculation
  no_pred_history <=
    pred_hist_in_valid = '0';

  -- We have a non-speculative input trigger
  -- which is required to begin a new round of speculation
  nonspec_trigger_arrived <=
    new_pred_valid = '1' and new_pred_spec = "0";

  -- We can speculate again once the history is empty
  -- and we have a non-spec trigger
  ready_to_re_speculate <=
    no_pred_history and nonspec_trigger_arrived;

  -- Input data is non-speculative
  nonspec_data_arrived <=
    data_in_valid = '1' and is_data_in_spec = "0";

  -- Register
  process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        state <= IDLE;
      else
        state <= next_state;
      end if;
    end if;
  end process;

  -- Next state logic
  process (all)
  begin
    next_state <= state;

    case state is
      -- while speculating
      when IDLE =>
        -- If real data arrives during speculation,
        -- and does not match the prediction
        if misspec_detected then
          -- transition to mis-prediction killing
          next_state <= KILL;
        end if;
      -- while recovering from mis-prediction
      -- and not yet speculating again
      when KILL =>
        -- We can leave KILL (and therefore speculate again)
        -- when two completion conditions hold:
        --   no_pred_history: prediction history has been emptied
        --                    and is ready to restart
        --   nonspec_trigger_arrived: the basic block containing the speculator
        --                     should execute again
        if ready_to_re_speculate then
          -- If non-speculative data has arrived,
          -- it is also safe to stop killing incoming spec data
          if nonspec_data_arrived then
            next_state <= IDLE;
          -- otherwise we must continue killing
          -- incoming spec data
          else
            next_state <= KILL_ONLY_DATA;
          end if;
        end if;
      -- if we are both speculating
      -- and recovering from mis-prediction
      when KILL_ONLY_DATA =>
        -- If non-speculative data arrives,
        -- we can stop recovering from mis-prediction
        if nonspec_data_arrived then
          next_state <= IDLE;
        end if;
    end case;
  end process;
end architecture;
"""

    specFSMOutputs_name = f"{name}_specFSMOutputs"

    specFSMOutputs = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.spec_types.all;

-- Entity of specFSMOutputs
entity {specFSMOutputs_name} is
  port (
    clk : in std_logic;
    rst : in std_logic;

    state : in State_type;

    data_in : in std_logic_vector({bitwidth} - 1 downto 0);
    data_in_valid : in std_logic;
    is_data_in_spec : in std_logic_vector(0 downto 0);
    data_in_ready : out std_logic;

    trigger_valid : in std_logic;
    trigger_spec : in std_logic_vector(0 downto 0);
    trigger_ready : out std_logic;

    pred_hist_in_valid : in std_logic;
    pred_hist_in_ready : out std_logic;

    data_to_resend : out std_logic_vector({bitwidth} - 1 downto 0);

    pred_hist_out_valid : out std_logic;
    pred_hist_out_ready : in std_logic;

    do_spec_valid : out std_logic;
    do_spec_ready : in std_logic;

    resolve_spec_valid : out std_logic;
    resolve_spec_ready : in std_logic;

    kill_valid : out std_logic;
    kill_ready : in std_logic;

    resend_valid : out std_logic;
    resend_ready : in std_logic;

    no_cmp_valid : out std_logic;
    no_cmp_ready : in std_logic;

    misspec_detected : in boolean;
    data_matches_prediction : in boolean
  );
end entity;

-- Architecture of specFSMOutputs
architecture arch of {specFSMOutputs_name} is
  ----------------------------------------------------------------------
  --      Important Note 1:    TODO: update for new FSM event outputs
  ----------------------------------------------------------------------
  --
  -- The internal FIFO used to store prediction history
  -- has a ready signal.
  -- The existance of this ready signal is considered internal state
  -- not as an input handshaking signal.
  --
  -- Therefore the valid signal to the prediction FIFO is dependant
  -- on its ready signal.
  --
  -- The ready signal of the FSM output is not internal state.
  -- And for critical path reasons, the valid signal to the FSM output
  -- must not be dependant on the FSM output ready signal
  --
  -- Therefore the pred_hist_out_ready signal can influence
  -- all valid signals (simplifying logic) through the booleans
  -- But the fsm_output_ready is used inside the process only

  ----------------------------------------------------------------------
  --      Important Note 2:
  ----------------------------------------------------------------------
  -- The ready signal for data and trigger are both dependant on their
  -- spec bit, creating a looped dependency path from and to the
  -- unit currently holding the token.
  --
  -- Because of this, we also allow the ready signal for data and trigger
  -- to be dependant on their valid signals, since it simplifies the
  -- RTL code.

  signal resend_done : boolean;

  signal data_before_prediction : boolean;
  signal do_spec : boolean;

begin

  -- we have incoming data 
  -- but have not yet predicted
  data_before_prediction <=
    trigger_valid = '1' and data_in_valid = '1' and pred_hist_in_valid = '0';

  -- we have a trigger to speculate with
  -- and space in the history
  do_spec <=
    trigger_valid = '1' and pred_hist_out_ready = '1';


  -- Tracks whether data and RESEND control tokens
  -- have been sent out during the KILL state
  -- Set to true if 
  -- the lazy fork to the decoders is ready
  -- even if was true already since it simplifies logic
  -- Reset to false, and set to false on misspec detected
  resend_done_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        resend_done <= false;
      elsif state = IDLE and misspec_detected then
        resend_done <= false;
      elsif state = KILL and resend_ready = '1' then
        resend_done <= true;
      end if;
    end if;
  end process;

  -- simple register which is written to
  -- when misspec is detected
  -- so doesn't need a reset
  data_to_resend_proc : process (clk)
  begin
    if rising_edge(clk) then
      if state = IDLE and misspec_detected then
        data_to_resend <= data_in;
      end if;
    end if;
  end process;


  process (all)
  begin
    -- defaults
    data_in_ready <= '0';
    trigger_ready <= '0';
    pred_hist_in_ready <= '0';
    pred_hist_out_valid <= '0';

    do_spec_valid <= '0';
    resolve_spec_valid <= '0';
    kill_valid <= '0';
    no_cmp_valid <= '0';
    resend_valid <= '0';

    case state is
      when IDLE =>
        -- Data does not match prediction
        -- need to perform mis-prediction recovery
        if misspec_detected then
          -- We accept data_in by writing it to data_to_resend
          data_in_ready <= '1';

          -- This could be 0 or 1
          -- Any incoming triggers in this cycle are spec
          -- (since the non-spec trigger is not yet re-issued
          -- from whatever save-commit it is in)
          -- and therefore will be killed eventually
          -- Killing them in this cycle is fastest but
          -- unlikely to actually affect performance
          trigger_ready <= '1';

          -- Since misspec is discovered, we will move to KILL
          -- to empty the prediction history.
          -- For now, we leave it alone.
        
        -- Otherwise
        -- we do not need to move to mis-prediction recovery
        else

          -- Independant if statement 1
          -- Confim speculation if we have incoming data
          -- This happens independently of new predictions
          if data_matches_prediction then
            -- Tell the lazy fork to the decoders
            -- that we want to resolve a speculation
            resolve_spec_valid <= '1';

            -- If the lazy fork is ready
            -- We can pop the correct prediction 
            -- from the prediction history
            -- and accept the incoming correct data
            data_in_ready <= resolve_spec_ready;
            pred_hist_in_ready <= resolve_spec_ready;
          end if;

          -- Independent if statement 2
          -- We have incoming data 
          -- before any speculation has occured
          -- so we just use the real data 
          -- instead of speculating
          if data_before_prediction then
            -- Tell the lazy fork to the decoders
            -- to use the real input data directly
            no_cmp_valid <= '1';

            -- If the lazy fork is ready
            -- we can absorb the trigger and data
            data_in_ready <= no_cmp_ready;
            trigger_ready <= no_cmp_ready;

          -- Otherwise, 
          -- we speculate
          -- if we have a trigger 
          -- and space in the pred history
          elsif do_spec then
            -- Tell the lazy fork to the decoders
            -- that we want to speculate again
            do_spec_valid <= '1';

            -- if the lazy fork is ready
            -- we can accept the trigger
            trigger_ready <= do_spec_ready;

            -- we push to prediction fifo when we speculate
            pred_hist_out_valid <= do_spec_ready;
          end if;
        end if;

      when KILL =>
        -- Accepts spec data to kill it
        data_in_ready <= is_data_in_spec(0);

        -- Accepts spec trigger to kill it
        trigger_ready <= trigger_spec(0);

        -- We want to send kills to the lazy fork
        -- if there is still predictions in the 
        -- prediction history
        kill_valid <= pred_hist_in_valid;

        -- If the commits are ready,
        -- we can pop a prediction
        -- from the prediction history
        pred_hist_in_ready <= kill_ready;

        -- We want to send a resend if the
        -- resend done flag is low
        resend_valid <= '1' when not resend_done else '0';

      when KILL_ONLY_DATA =>
        -- Accepts spec data only to kill it
        data_in_ready <= is_data_in_spec(0);

        -- we can speculate again in
        -- KILL_ONLY_DATA
        -- if the conditions are met
        if do_spec then
          do_spec_valid <= '1';

          -- if the lazy fork is ready
          -- we can accept the trigger
          trigger_ready <= do_spec_ready;

          -- push to pred history when we speculate
          pred_hist_out_valid <= do_spec_ready;
        end if;

    end case;
  end process;
end architecture;
"""

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.spec_types.all;

entity {name} is
  port (
    clk : in std_logic;
    rst : in std_logic;

    data_in : in std_logic_vector({bitwidth} - 1 downto 0);
    data_in_valid : in std_logic;
    is_data_in_spec : in std_logic_vector(0 downto 0);
    data_in_ready : out std_logic;

    trigger_valid : in std_logic;
    trigger_spec : in std_logic_vector(0 downto 0);
    trigger_ready : out std_logic;

    pred_hist_in : in std_logic_vector({bitwidth} - 1 downto 0);
    pred_hist_in_valid : in std_logic;
    pred_hist_in_ready : out std_logic;

    data_to_resend : out std_logic_vector({bitwidth} - 1 downto 0);

    pred_hist_out_valid : out std_logic;
    pred_hist_out_ready : in std_logic;

    do_spec_valid : out std_logic;
    do_spec_ready : in std_logic;

    resolve_spec_valid : out std_logic;
    resolve_spec_ready : in std_logic;

    kill_valid : out std_logic;
    kill_ready : in std_logic;

    resend_valid : out std_logic;
    resend_ready : in std_logic;

    no_cmp_valid : out std_logic;
    no_cmp_ready : in std_logic
  );
end entity;
"""

    architecture = f"""
architecture arch of {name} is
  signal state_int : State_type;
  signal misspec_detected : boolean;
  signal data_matches_prediction : boolean;
begin

  misspec_detected <=
    data_in_valid = '1' and pred_hist_in_valid = '1' and
    data_in /= pred_hist_in;

  data_matches_prediction <=
    data_in_valid = '1' and pred_hist_in_valid = '1' and
    data_in = pred_hist_in;

  specFSM: entity work.{specFSM_name}(arch)
    port map (
      clk => clk,
      rst => rst,
      data_in_valid => data_in_valid,
      is_data_in_spec => is_data_in_spec,
      pred_hist_in_valid => pred_hist_in_valid,
      new_pred_valid => trigger_valid,
      new_pred_spec => trigger_spec,
      misspec_detected => misspec_detected,
      state => state_int
    );

  specFSMOutputs: entity work.{specFSMOutputs_name}(arch)
    port map (
      clk => clk,
      rst => rst,
      state => state_int,
      data_in => data_in,
      data_in_valid => data_in_valid,
      is_data_in_spec => is_data_in_spec,
      data_in_ready => data_in_ready,
      trigger_valid => trigger_valid,
      trigger_spec => trigger_spec,
      trigger_ready => trigger_ready,
      pred_hist_in_valid => pred_hist_in_valid,
      pred_hist_in_ready => pred_hist_in_ready,
      data_to_resend => data_to_resend,
      pred_hist_out_valid => pred_hist_out_valid,
      pred_hist_out_ready => pred_hist_out_ready,
      do_spec_valid => do_spec_valid,
      do_spec_ready => do_spec_ready,
      resolve_spec_valid => resolve_spec_valid,
      resolve_spec_ready => resolve_spec_ready,
      kill_valid => kill_valid,
      kill_ready => kill_ready,
      resend_valid => resend_valid,
      resend_ready => resend_ready,
      no_cmp_valid => no_cmp_valid,
      no_cmp_ready => no_cmp_ready,
      misspec_detected => misspec_detected,
      data_matches_prediction => data_matches_prediction
    );
end architecture;
"""

    return specFSM + specFSMOutputs + entity + architecture


def _generate_data_decoder(name, bitwidth):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;

entity {name} is
  port (
    do_spec_in_valid : in std_logic;
    do_spec_in_ready : out std_logic;
    do_spec_data : in std_logic_vector({bitwidth} - 1 downto 0);

    resend_in_valid : in std_logic;
    resend_in_ready : out std_logic;
    resend_data : in std_logic_vector({bitwidth} - 1 downto 0);

    no_cmp_in_valid : in std_logic;
    no_cmp_in_ready : out std_logic;
    no_cmp_data : in std_logic_vector({bitwidth} - 1 downto 0);

    outs : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_spec : out std_logic_vector(0 downto 0);
    outs_valid : out std_logic;
    outs_ready : in std_logic
  );
end entity;
"""

    architecture = f"""
architecture arch of {name} is
begin
  -- Lazy fork means if we receive a valid
  -- we should output
  -- each possible (mutually exclusive) valid
  -- has a corresponding data to mux between
  outs <=
    do_spec_data when do_spec_in_valid = '1' else
    resend_data when resend_in_valid = '1' else
    no_cmp_data;

  -- spec if do spec
  -- nonspec if no_cmp or resend
  outs_spec <= "1" when do_spec_in_valid = '1' else "0";

  -- any of the valids is a valid output
  outs_valid <= do_spec_in_valid or resend_in_valid or no_cmp_in_valid;

  -- ready condition is independent of which valid
  do_spec_in_ready <= outs_ready;
  resend_in_ready <= outs_ready;
  no_cmp_in_ready <= outs_ready;
end architecture;
"""

    return entity + architecture


def _generate_sc_issue_decoder(name):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.spec_types.all;

entity {name} is
  port (
    do_spec_in_valid : in std_logic;
    do_spec_in_ready : out std_logic;

    resend_in_valid : in std_logic;
    resend_in_ready : out std_logic;

    no_cmp_in_valid : in std_logic;
    no_cmp_in_ready : out std_logic;

    outs : out std_logic_vector(1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in std_logic
  );
end entity;
"""

    architecture = f"""
architecture arch of {name} is
begin
  -- resend is sent along both channels for ordering
  -- no_cmp is sent along both channels for ordering
  -- resolve is independant of issue
  outs <=
    to_slv(ISSUE_RESEND)  when resend_in_valid = '1' else
    to_slv(ISSUE_NO_CMP)  when no_cmp_in_valid = '1' else
    to_slv(ISSUE_DO_SPEC);

  -- any (mutually exclusive valid)
  -- means we have a valid output
  outs_valid <= do_spec_in_valid or resend_in_valid or no_cmp_in_valid;

  -- ready condition is independant of which valid
  do_spec_in_ready <= outs_ready;
  resend_in_ready <= outs_ready;
  no_cmp_in_ready <= outs_ready;
end architecture;
"""

    return entity + architecture


def _generate_sc_resolve_decoder(name):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.spec_types.all;

entity {name} is
  port (
    resolve_in_valid : in std_logic;
    resolve_in_ready : out std_logic;

    resend_in_valid : in std_logic;
    resend_in_ready : out std_logic;

    no_cmp_in_valid : in std_logic;
    no_cmp_in_ready : out std_logic;

    outs : out std_logic_vector(1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in std_logic
  );
end entity;
"""

    architecture = f"""
architecture arch of {name} is
begin
  -- resend is sent along both channels for ordering
  -- no_cmp is sent along both channels for ordering
  -- resolve is independant of issue
  outs <=
    to_slv(HISTORY_RESEND)  when resend_in_valid = '1' else
    to_slv(HISTORY_NO_CMP)  when no_cmp_in_valid = '1' else
    to_slv(HISTORY_RESOLVE);

  -- any of the (mutually exclusive) valids
  -- is a valid output
  outs_valid <= resolve_in_valid or resend_in_valid or no_cmp_in_valid;

  -- ready condition is independant of which valid
  resolve_in_ready <= outs_ready;
  resend_in_ready <= outs_ready;
  no_cmp_in_ready <= outs_ready;
end architecture;
"""

    return entity + architecture


def _generate_commit_decoder(name):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;

entity {name} is
  port (
    resolve_in_valid : in std_logic;
    resolve_in_ready : out std_logic;

    kill_in_valid : in std_logic;
    kill_in_ready : out std_logic;

    outs : out std_logic_vector(0 downto 0);
    outs_valid : out std_logic;
    outs_ready : in std_logic
  );
end entity;
"""

    architecture = f"""
architecture arch of {name} is
begin
  -- output is either kill or pass
  outs <= "1" when kill_in_valid = '1' else "0";

  -- valid output if we get a resolve valid
  -- or a kill valid  (mutually exclusive)
  outs_valid <= resolve_in_valid or kill_in_valid;

  -- ready condition is independant of which valid
  resolve_in_ready <= outs_ready;
  kill_in_ready <= outs_ready;
end architecture;
"""

    return entity + architecture


def _generate_predictor(name, bitwidth):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

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
    data_out_spec : out std_logic_vector(0 downto 0);
    data_out_ready : in std_logic
  );
end entity;
"""

    architecture = f"""
architecture arch of {name} is
  signal zeros : std_logic_vector({bitwidth}-2 downto 0);
  signal data_reg: std_logic_vector({bitwidth}-1 downto 0);
begin
  zeros <= (others => '0');

  process(clk, rst) is
  begin
    if (rst = '1') then
      data_reg <= zeros & '0';
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

entity {name} is
  port (
    clk : in std_logic;
    rst : in std_logic;

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

  fifo_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        null;
      else
        if (TailEn = '1' ) then
          Memory(Tail) <= data_in;
        end if;
      end if;
    end if;
  end process;

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

  FullUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        Full <= '0';
      else
        if (TailEn = '1') and (HeadEn = '0') then
          if ((Tail + 2) mod {fifo_depth} = Head) then
            Full  <= '1';
          end if;
        elsif (TailEn = '0') and (HeadEn = '1') then
          Full <= '0';
        end if;
      end if;
    end if;
  end process;

  EmptyUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        Empty <= '1';
      else
        if (TailEn = '0') and (HeadEn = '1') then
          if ((Head +1) mod {fifo_depth} = Tail) then
            Empty  <= '1';
          end if;
        elsif (TailEn = '1') and (HeadEn = '0') then
            Empty <= '0';
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

    do_spec_fork_name = f"{name}_do_spec_fork"
    resolve_fork_name = f"{name}_resolve_fork"
    resend_fork_name = f"{name}_resend_fork"
    no_cmp_fork_name = f"{name}_no_cmp_fork"

    data_dec_name = f"{name}_data_dec"
    sc_issue_dec_name = f"{name}_sc_issue_dec"
    sc_resolve_dec_name = f"{name}_sc_resolve_dec"
    commit_dec_name = f"{name}_commit_dec"

    dependencies = (
        generate_fork(data_fork_name, {
            "size": 2,
            "bitwidth": bitwidth,
            "extra_signals": {"spec": 1}
        }) +
        _generate_specGen_core(specGen_name, bitwidth) +
        _generate_predictor(predictor_name, bitwidth) +
        _generate_predFifo(predFifo_name, bitwidth, fifo_depth) +
        generate_lazy_fork(do_spec_fork_name, {"size": 2, "bitwidth": 0}) +
        generate_lazy_fork(resolve_fork_name, {"size": 2, "bitwidth": 0}) +
        generate_lazy_fork(resend_fork_name, {"size": 3, "bitwidth": 0}) +
        generate_lazy_fork(no_cmp_fork_name, {"size": 3, "bitwidth": 0}) +
        _generate_data_decoder(data_dec_name, bitwidth) +
        _generate_sc_issue_decoder(sc_issue_dec_name) +
        _generate_sc_resolve_decoder(sc_resolve_dec_name) +
        _generate_commit_decoder(commit_dec_name)
    )

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;
use work.spec_types.all;

entity {name} is
  port (
    clk, rst : in  std_logic;
    ins: in std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid: in std_logic;
    ins_spec: in std_logic_vector(0 downto 0);
    ins_ready: out std_logic;
    trigger_valid: in std_logic;
    trigger_spec: in std_logic_vector(0 downto 0);
    trigger_ready: out std_logic;
    outs: out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid: out std_logic;
    outs_spec: out std_logic_vector(0 downto 0);
    outs_ready: in std_logic;
    ctrl_issue: out std_logic_vector(1 downto 0);
    ctrl_issue_valid: out std_logic;
    ctrl_issue_ready: in std_logic;
    ctrl_history: out std_logic_vector(1 downto 0);
    ctrl_history_valid: out std_logic;
    ctrl_history_ready: in std_logic;
    ctrl_commit: out std_logic_vector(0 downto 0);
    ctrl_commit_valid: out std_logic;
    ctrl_commit_ready: in std_logic
  );
end entity;
"""

    architecture = f"""
architecture arch of {name} is
  signal fork_data_outs : data_array(1 downto 0)({bitwidth} - 1 downto 0);
  signal fork_data_outs_valid : std_logic_vector(1 downto 0);
  signal fork_data_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork_data_outs_ready : std_logic_vector(1 downto 0);

  signal predictor_data_out : std_logic_vector({bitwidth} - 1 downto 0);
  signal predictor_data_out_valid : std_logic;
  signal predictor_data_out_spec : std_logic_vector(0 downto 0);
  signal predictor_data_out_ready : std_logic;

  signal core_data_to_resend : std_logic_vector({bitwidth} - 1 downto 0);
  signal core_pred_hist_out_valid : std_logic;
  signal core_pred_hist_out_ready : std_logic;

  signal core_do_spec_valid, core_do_spec_ready : std_logic;
  signal core_resolve_valid, core_resolve_ready : std_logic;
  signal core_kill_valid, core_kill_ready : std_logic;
  signal core_resend_valid, core_resend_ready : std_logic;
  signal core_no_cmp_valid, core_no_cmp_ready : std_logic;

  signal predFifo_data_out : std_logic_vector({bitwidth} - 1 downto 0);
  signal predFifo_data_out_valid : std_logic;
  signal predFifo_data_out_ready : std_logic;

  signal do_spec_fork_outs_valid, do_spec_fork_outs_ready : std_logic_vector(1 downto 0);
  signal resolve_fork_outs_valid, resolve_fork_outs_ready : std_logic_vector(1 downto 0);
  signal resend_fork_outs_valid, resend_fork_outs_ready : std_logic_vector(2 downto 0);
  signal no_cmp_fork_outs_valid, no_cmp_fork_outs_ready : std_logic_vector(2 downto 0);
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

  specgenCore: entity work.{specGen_name}(arch)
    port map (
      clk => clk,
      rst => rst,
      data_in => fork_data_outs(0),
      data_in_valid => fork_data_outs_valid(0),
      is_data_in_spec => fork_data_outs_0_spec,
      data_in_ready => fork_data_outs_ready(0),
      trigger_valid => predictor_data_out_valid,
      trigger_spec => predictor_data_out_spec,
      trigger_ready => predictor_data_out_ready,
      pred_hist_in => predFifo_data_out,
      pred_hist_in_valid => predFifo_data_out_valid,
      pred_hist_in_ready => predFifo_data_out_ready,
      data_to_resend => core_data_to_resend,
      pred_hist_out_valid => core_pred_hist_out_valid,
      pred_hist_out_ready => core_pred_hist_out_ready,
      do_spec_valid => core_do_spec_valid,
      do_spec_ready => core_do_spec_ready,
      resolve_spec_valid => core_resolve_valid,
      resolve_spec_ready => core_resolve_ready,
      kill_valid => core_kill_valid,
      kill_ready => core_kill_ready,
      resend_valid => core_resend_valid,
      resend_ready => core_resend_ready,
      no_cmp_valid => core_no_cmp_valid,
      no_cmp_ready => core_no_cmp_ready
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
      data_in => predictor_data_out,
      data_in_valid => core_pred_hist_out_valid,
      data_in_ready => core_pred_hist_out_ready,
      data_out => predFifo_data_out,
      data_out_valid => predFifo_data_out_valid,
      data_out_ready => predFifo_data_out_ready
    );

  do_spec_fork: entity work.{do_spec_fork_name}(arch)
    port map (
      clk => clk,
      rst => rst,
      ins_valid => core_do_spec_valid,
      ins_ready => core_do_spec_ready,
      outs_valid => do_spec_fork_outs_valid,
      outs_ready => do_spec_fork_outs_ready
    );

  resolve_fork: entity work.{resolve_fork_name}(arch)
    port map (
      clk => clk,
      rst => rst,
      ins_valid => core_resolve_valid,
      ins_ready => core_resolve_ready,
      outs_valid => resolve_fork_outs_valid,
      outs_ready => resolve_fork_outs_ready
    );

  resend_fork: entity work.{resend_fork_name}(arch)
    port map (
      clk => clk,
      rst => rst,
      ins_valid => core_resend_valid,
      ins_ready => core_resend_ready,
      outs_valid => resend_fork_outs_valid,
      outs_ready => resend_fork_outs_ready
    );

  no_cmp_fork: entity work.{no_cmp_fork_name}(arch)
    port map (
      clk => clk,
      rst => rst,
      ins_valid => core_no_cmp_valid,
      ins_ready => core_no_cmp_ready,
      outs_valid => no_cmp_fork_outs_valid,
      outs_ready => no_cmp_fork_outs_ready
    );

  data_dec0: entity work.{data_dec_name}(arch)
    port map (
      do_spec_in_valid => do_spec_fork_outs_valid(0),
      do_spec_in_ready => do_spec_fork_outs_ready(0),
      do_spec_data => predictor_data_out,
      resend_in_valid => resend_fork_outs_valid(0),
      resend_in_ready => resend_fork_outs_ready(0),
      resend_data => core_data_to_resend,
      no_cmp_in_valid => no_cmp_fork_outs_valid(0),
      no_cmp_in_ready => no_cmp_fork_outs_ready(0),
      no_cmp_data => fork_data_outs(0),
      outs => outs,
      outs_spec => outs_spec,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

  sc_issue_dec0: entity work.{sc_issue_dec_name}(arch)
    port map (
      do_spec_in_valid => do_spec_fork_outs_valid(1),
      do_spec_in_ready => do_spec_fork_outs_ready(1),
      resend_in_valid => resend_fork_outs_valid(1),
      resend_in_ready => resend_fork_outs_ready(1),
      no_cmp_in_valid => no_cmp_fork_outs_valid(1),
      no_cmp_in_ready => no_cmp_fork_outs_ready(1),
      outs => ctrl_issue,
      outs_valid => ctrl_issue_valid,
      outs_ready => ctrl_issue_ready
    );

  sc_resolve_dec0: entity work.{sc_resolve_dec_name}(arch)
    port map (
      resolve_in_valid => resolve_fork_outs_valid(0),
      resolve_in_ready => resolve_fork_outs_ready(0),
      resend_in_valid => resend_fork_outs_valid(2),
      resend_in_ready => resend_fork_outs_ready(2),
      no_cmp_in_valid => no_cmp_fork_outs_valid(2),
      no_cmp_in_ready => no_cmp_fork_outs_ready(2),
      outs => ctrl_history,
      outs_valid => ctrl_history_valid,
      outs_ready => ctrl_history_ready
    );

  commit_dec0: entity work.{commit_dec_name}(arch)
    port map (
      resolve_in_valid => resolve_fork_outs_valid(1),
      resolve_in_ready => resolve_fork_outs_ready(1),
      kill_in_valid => core_kill_valid,
      kill_in_ready => core_kill_ready,
      outs => ctrl_commit,
      outs_valid => ctrl_commit_valid,
      outs_ready => ctrl_commit_ready
    );
end architecture;
"""

    return dependencies + entity + architecture


def _generate_speculator_signal_manager(name, bitwidth, fifo_depth, extra_signals):
    extra_signals_without_spec = extra_signals.copy()
    extra_signals_without_spec.pop("spec")

    extra_signals_bitwidth = get_concat_extra_signals_bitwidth(extra_signals)
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
            "name": "ctrl_issue",
            "bitwidth": 2,
            "extra_signals": {}
        }, {
            "name": "ctrl_history",
            "bitwidth": 2,
            "extra_signals": {}
        }, {
            "name": "ctrl_commit",
            "bitwidth": 1,
            "extra_signals": {}
        }],
        extra_signals_without_spec,
        ["ctrl_issue", "ctrl_history", "ctrl_commit"],
        lambda name: _generate_speculator(name, bitwidth + extra_signals_bitwidth - 1, fifo_depth))
