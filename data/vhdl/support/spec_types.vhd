library ieee;
use ieee.std_logic_1164.all;

package spec_types is

  -- Speculator FSM state (used internally by the speculator).
  type State_type is (IDLE, KILL, KILL_ONLY_DATA);

  -- Codes carried on the speculator's issueCtrl channel
  -- (produced by the speculator, consumed by save-commits).
  type IssueCtrl_type is (ISSUE_DO_SPEC, ISSUE_RESEND, ISSUE_NO_CMP);

  -- Codes carried on the speculator's historyCtrl channel
  -- (produced by the speculator, consumed by save-commits).
  type HistoryCtrl_type is (HISTORY_RESOLVE, HISTORY_RESEND, HISTORY_NO_CMP);

  function to_slv(ctrl : IssueCtrl_type)   return std_logic_vector;
  function to_slv(ctrl : HistoryCtrl_type) return std_logic_vector;

  function slv_to_issue_ctrl  (slv : std_logic_vector(1 downto 0)) return IssueCtrl_type;
  function slv_to_history_ctrl(slv : std_logic_vector(1 downto 0)) return HistoryCtrl_type;

end package;

package body spec_types is

  function to_slv(ctrl : IssueCtrl_type) return std_logic_vector is
  begin
    case ctrl is
      when ISSUE_DO_SPEC => return "00";
      when ISSUE_RESEND  => return "01";
      when ISSUE_NO_CMP  => return "10";
    end case;
  end function;

  function to_slv(ctrl : HistoryCtrl_type) return std_logic_vector is
  begin
    case ctrl is
      when HISTORY_RESOLVE => return "00";
      when HISTORY_RESEND  => return "01";
      when HISTORY_NO_CMP  => return "10";
    end case;
  end function;

  function slv_to_issue_ctrl(slv : std_logic_vector(1 downto 0)) return IssueCtrl_type is
  begin
    case slv is
      when "00" => return ISSUE_DO_SPEC;
      when "01" => return ISSUE_RESEND;
      when "10" => return ISSUE_NO_CMP;
      when others =>
        assert false
          report "slv_to_issue_ctrl: unexpected std_logic_vector code on issueCtrl"
          severity error;
        return ISSUE_DO_SPEC;
    end case;
  end function;

  function slv_to_history_ctrl(slv : std_logic_vector(1 downto 0)) return HistoryCtrl_type is
  begin
    case slv is
      when "00" => return HISTORY_RESOLVE;
      when "01" => return HISTORY_RESEND;
      when "10" => return HISTORY_NO_CMP;
      when others =>
        assert false
          report "slv_to_history_ctrl: unexpected std_logic_vector code on historyCtrl"
          severity error;
        return HISTORY_RESOLVE;
    end case;
  end function;

end package body;
