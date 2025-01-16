module {
  handshake.func @a_miter(%d: !handshake.channel<i32>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["D", "C"], resNames = ["EQ_T", "EQ_F"]} {
    %lhs_d, %rhs_d = fork [2] %d {handshake.bb = 0 : ui32, handshake.name = "input_fork_d"} : <i32>
    %lhs_c, %rhs_c = fork [2] %c {handshake.bb = 0 : ui32, handshake.name = "input_fork_c"} : <i1>

    %lhs_t, %lhs_f = cond_br %lhs_c, %lhs_d {handshake.bb = 1 : ui32, handshake.name = "lhs_branch"} : <i1>, <i32>

    %rhs_d_forked:2 = fork [2] %rhs_d {handshake.bb = 2 : ui32, handshake.name = "rhs_fork_data"} : <i32>
    %rhs_c_forked:2 = fork [2] %rhs_c {handshake.bb = 2 : ui32, handshake.name = "rhs_fork_control"} : <i1>
    %rhs_c_not = not %rhs_c_forked#0 {handshake.bb = 2 : ui32, handshake.name = "rhs_not"} : <i1>
    %rhs_t_0, %rhs_f_0 = cond_br %rhs_c_not, %rhs_d_forked#0 {handshake.bb = 2 : ui32, handshake.name = "rhs_supp_br_0"} : <i1>, <i32>
    sink %rhs_t_0 {handshake.bb = 2 : ui32, handshake.name = "rhs_sink_0"} : <i32>
    %rhs_t_1, %rhs_f_1 = cond_br %rhs_c_forked#1, %rhs_d_forked#1 {handshake.bb = 2 : ui32, handshake.name = "rhs_supp_br_1"} : <i1>, <i32>
    sink %rhs_t_1 {handshake.bb = 2 : ui32, handshake.name = "rhs_sink_1"} : <i32>

    %comp_t = cmpi eq, %lhs_t, %rhs_f_0 {handshake.bb = 3 : ui32, handshake.name = "eq_t"} : <i32>
    %comp_f = cmpi eq, %lhs_f, %rhs_f_1 {handshake.bb = 3 : ui32, handshake.name = "eq_f"} : <i32>

    end {handshake.bb = 4 : ui32, handshake.name = "rhs_end0"} %comp_t, %comp_f : <i1>, <i1>
  }
}