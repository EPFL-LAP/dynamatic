from generators.support.signal_manager import generate_spec_units_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth
from generators.support.utils import data


def generate_spec_save_commit(name, params):
    bitwidth = params["bitwidth"]
    fifo_depth = params["fifo_depth"]
    extra_signals = params["extra_signals"]

    if len(extra_signals) > 1:
        return _generate_spec_save_commit_signal_manager(name, bitwidth, fifo_depth, extra_signals)
    return _generate_spec_save_commit(name, bitwidth, fifo_depth)


def _generate_spec_save_commit(name, bitwidth, fifo_depth):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.spec_types.all;

entity {name} is
  port (
    clk : in std_logic;
    rst : in std_logic;
    {data(f"ins : in std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
    ins_valid : in std_logic;
    ins_spec : in std_logic_vector(0 downto 0);
    ctrl_issue : in std_logic_vector(1 downto 0);
    ctrl_issue_valid : in std_logic;
    ctrl_issue_ready : out std_logic;
    ctrl_history : in std_logic_vector(1 downto 0);
    ctrl_history_valid : in std_logic;
    ctrl_history_ready : out std_logic;
    outs_ready : in std_logic;
    {data(f"outs : out std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
    outs_valid : out std_logic;
    outs_spec : out std_logic_vector(0 downto 0);
    ins_ready : out std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of spec_save_commit
architecture arch of {name} is
  signal HeadEn : std_logic := '0';
  signal TailEn : std_logic := '0';
  signal CurrEn : std_logic := '0';

  signal Tail : natural range 0 to {fifo_depth} - 1;
  signal Head : natural range 0 to {fifo_depth} - 1;
  signal Curr : natural range 0 to {fifo_depth} - 1;

  signal CurrEmpty : std_logic;
  signal Empty     : std_logic;
  signal Full      : std_logic;

  {data(f"type FIFO_Memory is array (0 to {fifo_depth} - 1) of STD_LOGIC_VECTOR ({bitwidth} - 1 downto 0);", bitwidth)}
  {data("signal Memory : FIFO_Memory;", bitwidth)}

  signal misspec_recovery_bit : std_logic;
  signal leaving_recovery : std_logic;
  signal no_unkilled_mispreds : std_logic;

  signal issue_decoded   : IssueCtrl_type;
  signal history_decoded : HistoryCtrl_type;

  signal issue_is_do_spec : std_logic;
  signal issue_is_resend : std_logic;
  signal issue_is_no_cmp : std_logic;

  signal resolve_is_resolve : std_logic;
  signal resolve_is_resend : std_logic;
  signal resolve_is_no_cmp : std_logic;

  signal do_spec : std_logic;
  signal do_cmp_correct : std_logic;
  signal do_resend : std_logic;
  signal do_no_cmp : std_logic;

  signal resend_accept : std_logic;
  signal drop_spec_ins : std_logic;

begin

  ------------------------

  issue_decoded   <= slv_to_issue_ctrl(ctrl_issue);
  history_decoded <= slv_to_history_ctrl(ctrl_history);

  issue_is_do_spec <= ctrl_issue_valid when issue_decoded = ISSUE_DO_SPEC else '0';
  issue_is_resend  <= ctrl_issue_valid when issue_decoded = ISSUE_RESEND  else '0';
  issue_is_no_cmp  <= ctrl_issue_valid when issue_decoded = ISSUE_NO_CMP  else '0';

  resolve_is_resolve <= ctrl_history_valid when history_decoded = HISTORY_RESOLVE else '0';
  resolve_is_resend  <= ctrl_history_valid when history_decoded = HISTORY_RESEND  else '0';
  resolve_is_no_cmp  <= ctrl_history_valid when history_decoded = HISTORY_NO_CMP  else '0';

  -----------------------------

  do_spec <= no_unkilled_mispreds and issue_is_do_spec;
  do_cmp_correct <= no_unkilled_mispreds and resolve_is_resolve;
  do_resend <= issue_is_resend and resolve_is_resend;
  do_no_cmp <= no_unkilled_mispreds and issue_is_no_cmp and resolve_is_no_cmp;

  -----------------------------

  -- the arrival of a non-spec value tells us all misspec values are killed
  leaving_recovery <= misspec_recovery_bit and (not ins_spec(0)) and ins_valid;

  -- we can accept save controls when the next cycle is not misspec recovery
  no_unkilled_mispreds <= (not misspec_recovery_bit) or leaving_recovery;

  ----------------------------

  -- we are able to re-issue the oldest token and so
  -- are accepting the resend
  resend_accept <= do_resend and outs_ready;

  -- a spec input arrived but we are in misspec recovery
  -- or about to be,
  -- so we drop it instead of writing it to memory
  drop_spec_ins <= (do_resend or misspec_recovery_bit) and ins_spec(0);

  -- state affects if we drop a misspec input
  -- otherwise we can always write to the internal FIFO
  ins_ready <= drop_spec_ins or not Full;

  -- write every data which passes through the save-commit
  -- to memory
  -- unless it is a mis-spec data which we just drop
  TailEn <= ins_valid and ins_ready and not drop_spec_ins;

  signal_proc : process (all)
  begin
    -- defaults
    HeadEn <= '0';
    CurrEn <= '0';
    outs_valid <= '0';
    {data("outs <= Memory(Head);", bitwidth)}
    outs_spec <= "0";
    ctrl_issue_ready <= '0';
    ctrl_history_ready <= '0';

    -- if we want to speculate
    if do_spec = '1' then
      -- output is speculative
      outs_spec <= "1";

      -- if we have no unsent data we must
      -- look at input for data
      if CurrEmpty = '1' then
        -- we want to send if we have
        -- valid input data
        -- and space to store the value
        if ins_valid = '1' and Full = '0' then
          -- accept control if consumer is ready
          ctrl_issue_ready <= outs_ready;

          -- we actually record the data as sent
          -- if the consumer is ready
          CurrEn <= outs_ready;

          -- data is valid to issue
          outs_valid <= '1';
          {data("outs <= ins;", bitwidth)}
        end if;
      else
        -- accept control if consumer is ready
        ctrl_issue_ready <= outs_ready;

        -- we actually record the data as sent
        -- if the consumer is ready
        CurrEn <= outs_ready;

        -- data is valid to issue
        outs_valid <= '1';
        {data("outs <= Memory(Curr);", bitwidth)}
      end if;
    end if;

    -- if we want to discard the oldest data
    -- in the save commit
    -- since speculation resolved as correct
    if do_cmp_correct = '1' then
        ctrl_history_ready <= '1';

        HeadEn <= '1';
    end if;

    -- if we want to resend the oldest token
    -- to recover from misspeculation
    if do_resend = '1' then
      -- accept control if consumer is ready
      ctrl_issue_ready <= outs_ready;
      ctrl_history_ready <= outs_ready;

      -- data is valid to issue
      outs_valid <= '1';
      {data("outs <= Memory(Head);", bitwidth)}
    end if;

    -- if we want to use the input data
    -- non-speculatively
    if do_no_cmp = '1' then
      -- if we dont have any data
      if Empty = '1' then
        -- have to look at the input
        -- for valid data
        if ins_valid = '1' then
          -- can accept the control
          -- if the consumer is ready
          ctrl_issue_ready <= outs_ready;
          ctrl_history_ready <= outs_ready;

          -- record the transfer if the consumer is ready
          CurrEn <= outs_ready;
          HeadEn <= outs_ready;

          -- data is valid to issue
          outs_valid <= '1';
          {data("outs <= ins;", bitwidth)}
        end if;
      -- otherwise we can try issue
      -- the oldest data we have
      else
        -- can accept the control
        -- if the consumer is ready
        ctrl_issue_ready <= outs_ready;
        ctrl_history_ready <= outs_ready;

        -- record the transfer if the consumer is ready
        CurrEn <= outs_ready;
        HeadEn <= outs_ready;

        -- data is valid to issue
        outs_valid <= '1';
        {data("outs <= Memory(Head);", bitwidth)}
      end if;
    end if;
  end process;

  misspec_recovery_bit_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        misspec_recovery_bit <= '0';
      else
        if resend_accept = '1' then
          misspec_recovery_bit <= '1';
        elsif leaving_recovery = '1' then
          misspec_recovery_bit <= '0';
        end if;
      end if;
    end if;
  end process;

  {data("""
  fifo_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        null;
      else
        if TailEn = '1' then
          Memory(Tail) <= ins;
        end if;
      end if;
    end if;
  end process;
  """, bitwidth)}

  TailUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' or resend_accept = '1' then
        Tail <= 0;
      else
        if (TailEn = '1') then
          Tail <= (Tail + 1) mod {fifo_depth};
        end if;
      end if;
    end if;
  end process;

  HeadUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' or resend_accept = '1' then
        Head <= 0;
      else
        if (HeadEn = '1') then
          Head  <= (Head + 1) mod {fifo_depth};
        end if;
      end if;
    end if;
  end process;

  CurrUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' or resend_accept = '1' then
        Curr <= 0;
      else
        if (CurrEn = '1') then
          Curr <= (Curr + 1) mod {fifo_depth};
        end if;
      end if;
    end if;
  end process;

  FullUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' or resend_accept = '1' then
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
      if rst = '1' or resend_accept = '1' then
        Empty <= '1';
      else
        if (TailEn = '0') and (HeadEn = '1') then
          if ((Head + 1) mod {fifo_depth} = Tail) then
            Empty  <= '1';
          end if;
        elsif (TailEn = '1') and (HeadEn = '0') then
          Empty <= '0';
        end if;
      end if;
    end if;
  end process;

  CurrEmptyUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' or resend_accept = '1' then
        CurrEmpty <= '1';
      else
        if (TailEn = '0') and (CurrEn = '1') then
          if ((Curr + 1) mod {fifo_depth} = Tail) then
            CurrEmpty  <= '1';
          end if;
        elsif (TailEn = '1') and (CurrEn = '0') then
          CurrEmpty <= '0';
        end if;
      end if;
    end if;
  end process;
end architecture;
"""

    return entity + architecture


def _generate_spec_save_commit_signal_manager(name, bitwidth, fifo_depth, extra_signals):

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
            "name": "ctrl_issue",
            "bitwidth": 2
        }, {
            "name": "ctrl_history",
            "bitwidth": 2
        }],
        [{
            "name": "outs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals_without_spec,
        ["ctrl_issue", "ctrl_history"],
        lambda name: _generate_spec_save_commit(name, bitwidth + extra_signals_bitwidth - 1, fifo_depth))
