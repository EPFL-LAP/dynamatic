#include "dynamatic/Integration.h"
#include "stdlib.h"

#define N 1000

void single_loop(int a[N], int b[N], int c[N]) {
  int i = 0;
  int bound = 1000;
  int sum = 0;

  bool guard = true;

  // v_7 : SCC_guard_i_to_merged_DAG_0_SCC_c_out
  bool v_7_write{};
  bool v_7_data_enable{};
  bool v_7_data_commit_val_0_enable{};
  bool v_7_data_commit_val_0_commit_val_0{};
  bool v_7_data_commit_val_0_commit_val_1{};
  int v_7_data_commit_val_0_commit_val_2{};
  int v_7_data_commit_val_0_commit_val_3{};
  bool v_7_data_commit_val_0_commit_val_4{};
  int arg_10{};
  bool arg_11{};

  // arg_12 : state_guard (flattened)
  unsigned int arg_12_state{};
  // ap_int<3>
  unsigned int arg_12_rewindCpt{};
  bool arg_12_delayed_commit_0{};
  bool arg_12_delayed_commit_1{};
  bool arg_12_delayed_commit_2{};
  bool arg_12_delayed_commit_3{};
  bool arg_12_delayed_commit_4{};
  unsigned int arg_12_array_rollback{};
  unsigned int arg_12_mu_rollback{};
  unsigned int arg_12_rewind{};
  bool arg_12_rbwe{};
  bool arg_12_commit_guard{};
  unsigned int arg_12_selSlowPath_guard{};
  unsigned int arg_12_rollback_guard{};
  bool arg_12_startStall_guard{};
  unsigned int arg_12_rewindDepth{};
  unsigned int arg_12_slowPath_guard{};

  int i_13{};
  bool guard_14{};
  bool delay_15{};
  bool delay_15_buffer[4]{};

  // v_16 : fsm_mispec_in_guard
  unsigned int v_16_guard{};

  state_guard fsm_state_guard_17{};
  fsm_cmd_guard fsm_guard_command_18{};
  state_guard fsm_guard_next_19{};
  commit_type_16 delay_20{};
  commit_type_16 delay_20_buffer[5]{};
  commit_type_18 v_21{};
  int rollback_22{};
  int rollback_22_buffer[6]{};
  int load_23{};
  int load_24{};
  unsigned int v_25{};
  unsigned int v_26{};
  unsigned int mul_27{};
  int v_28{};
  unsigned int v_29{};
  unsigned int add_30{};
  int v_31{};
  bool lt_32{};
  bool rollback_33{};
  bool rollback_33_buffer[6]{};
  bool and_34{};
  bool guard_35{};
  int v_36{};
  bool v_37{};
  commit_type_16 v_38{};
  SCC_guard_i_to_merged_DAG_0_SCC_c_out v_39{};
  commit_type_18 fifo_40{};
  FifoType<commit_type_18> fifo_40_buffer{};
  merged_DAG_0_SCC_c_to_exit_out v_41{};
  int *arg_42{};
  commit_type_18 arg_43{};
  int *c_44{};
  bool xor_45{};
  int *alpha_46{};
  int *v_47{};
  commit_type_17 v_48{};
  merged_DAG_0_SCC_c_to_exit_out v_49{};
  commit_type_17 fifo_50{};
  FifoType<commit_type_17> fifo_50_buffer{};
  commit_type_24 v_51{};
  commit_type_17 arg_52{};
  commit_type_24 v_53{};
  bool exit_ = false;
  while (!exit_) {
    if (!fifo_40_buffer.full) {
      arg_10 = arg_3;
      arg_11 = arg_4;
      arg_12 = arg_5;
      i_13 = i;
      guard_14 = guard;
      delay_15 = delay_pop<bool, 4>(delay_15_buffer);
      v_16 = fsm_mispec_in_guard{static_cast<unsigned int>(delay_15)};
      fsm_state_guard_17 = mu<state_guard, 2>(arg_12, fsm_guard_next_19);
      fsm_guard_command_18 = fsm_guard_command(fsm_state_guard_17);
      fsm_guard_next_19 = fsm_guard_next(v_16, fsm_state_guard_17);
      delay_20 = delay_pop<commit_type_16, 5>(delay_20_buffer);
      v_21 = commit_type_18{static_cast<bool>(fsm_guard_command_18.commit),
                            static_cast<commit_type_16>(delay_20)};
      rollback_22 = rollback<int, 0, 5>(rollback_22_buffer, i_13,
                                        fsm_guard_command_18.muRollBack,
                                        fsm_guard_command_18.rbwe);
      load_23 = a[0 <= rollback_22 && rollback_22 < 32 ? (int)rollback_22 : 0];
      load_24 = b[0 <= rollback_22 && rollback_22 < 32 ? (int)rollback_22 : 0];
      v_25 = (unsigned int)load_23;
      v_26 = (unsigned int)load_24;
      mul_27 = v_25 * v_26;
      v_28 = (int)mul_27;
      v_29 = (unsigned int)rollback_22;
      add_30 = v_29 + 1;
      v_31 = (int)add_30;
      lt_32 = (int)mul_27 < (int)10;
      rollback_33 = rollback<bool, 0, 5>(rollback_33_buffer, guard_14,
                                         fsm_guard_command_18.muRollBack,
                                         fsm_guard_command_18.rbwe);
      and_34 = rollback_33 & lt_32;
      guard_35 =
          gamma<bool>(fsm_guard_command_18.selSlowPath_guard, false, true);
      v_38 = commit_type_16{
          static_cast<bool>(true),        static_cast<bool>(guard_35),
          static_cast<bool>(rollback_33), static_cast<int>(v_28),
          static_cast<int>(rollback_22),  static_cast<bool>(guard_35)};
      v_39 = SCC_guard_i_to_merged_DAG_0_SCC_c_out{
          static_cast<bool>(fsm_guard_command_18.commit),
          static_cast<commit_type_18>(v_21)};
      delay_push<bool, 4>(delay_15_buffer, and_34, true);
      delay_push<commit_type_16, 5>(delay_20_buffer, v_38, true);
      v_7 = v_39;
      fifo_write(fifo_40_buffer, v_7);
    };
    fifo_40 = fifo_40_buffer.data;
    if (!fifo_40_buffer.empty && !fifo_50_buffer.full) {
      arg_42 = arg_6;
      arg_43 = fifo_40;
      c_44 = mu<int *, 3>(arg_42, alpha_46);
      xor_45 = arg_43.commit_val_0.commit_val_0 ^ true;
      alpha_46 = alpha(c_44, arg_43.commit_val_0.commit_val_3,
                       arg_43.commit_val_0.commit_val_2,
                       arg_43.commit_val_0.commit_val_1);
      v_48 = commit_type_17{static_cast<bool>(true), static_cast<bool>(xor_45)};
      v_49 = merged_DAG_0_SCC_c_to_exit_out{static_cast<bool>(true),
                                            static_cast<commit_type_17>(v_48)};
      v_41 = v_49;
      fifo_write(fifo_50_buffer, v_41);
      fifo_read(fifo_40_buffer);
    };
    fifo_50 = fifo_50_buffer.data;
    if (!fifo_50_buffer.empty) {
      arg_52 = fifo_50;
      v_53 = commit_type_24{static_cast<bool>(true),
                            static_cast<bool>(arg_52.commit_val_0)};
      v_51 = v_53;
      fifo_read(fifo_50_buffer);
    };
    exit_ = v_51.commit_val_0;

    i = v_31;
    guard = guard_35;
  }
}

int main(void) {
  int a[N];
  int b[N];
  int c[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = 2;
    b[j] = j;
    c[j] = 0;
  }

  CALL_KERNEL(single_loop, a, b, c);
  return 0;
}
