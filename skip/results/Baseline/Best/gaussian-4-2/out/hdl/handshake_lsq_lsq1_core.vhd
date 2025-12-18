

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1_core_ga is
	port(
		rst : in std_logic;
		clk : in std_logic;
		group_init_valid_0_i : in std_logic;
		group_init_ready_0_o : out std_logic;
		ldq_tail_i : in std_logic_vector(1 downto 0);
		ldq_head_i : in std_logic_vector(1 downto 0);
		ldq_empty_i : in std_logic;
		stq_tail_i : in std_logic_vector(0 downto 0);
		stq_head_i : in std_logic_vector(0 downto 0);
		stq_empty_i : in std_logic;
		ldq_wen_0_o : out std_logic;
		ldq_wen_1_o : out std_logic;
		ldq_wen_2_o : out std_logic;
		ldq_wen_3_o : out std_logic;
		num_loads_o : out std_logic_vector(1 downto 0);
		ldq_port_idx_0_o : out std_logic_vector(0 downto 0);
		ldq_port_idx_1_o : out std_logic_vector(0 downto 0);
		ldq_port_idx_2_o : out std_logic_vector(0 downto 0);
		ldq_port_idx_3_o : out std_logic_vector(0 downto 0);
		stq_wen_0_o : out std_logic;
		stq_wen_1_o : out std_logic;
		num_stores_o : out std_logic_vector(0 downto 0);
		ga_ls_order_0_o : out std_logic_vector(1 downto 0);
		ga_ls_order_1_o : out std_logic_vector(1 downto 0);
		ga_ls_order_2_o : out std_logic_vector(1 downto 0);
		ga_ls_order_3_o : out std_logic_vector(1 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_ga is
	signal num_loads : std_logic_vector(1 downto 0);
	signal num_stores : std_logic_vector(0 downto 0);
	signal loads_sub : std_logic_vector(1 downto 0);
	signal stores_sub : std_logic_vector(0 downto 0);
	signal empty_loads : std_logic_vector(2 downto 0);
	signal empty_stores : std_logic_vector(1 downto 0);
	signal group_init_ready_0 : std_logic;
	signal group_init_hs_0 : std_logic;
	signal ldq_port_idx_rom_0 : std_logic_vector(0 downto 0);
	signal ldq_port_idx_rom_1 : std_logic_vector(0 downto 0);
	signal ldq_port_idx_rom_2 : std_logic_vector(0 downto 0);
	signal ldq_port_idx_rom_3 : std_logic_vector(0 downto 0);
	signal ga_ls_order_rom_0 : std_logic_vector(1 downto 0);
	signal ga_ls_order_rom_1 : std_logic_vector(1 downto 0);
	signal ga_ls_order_rom_2 : std_logic_vector(1 downto 0);
	signal ga_ls_order_rom_3 : std_logic_vector(1 downto 0);
	signal ga_ls_order_temp_0 : std_logic_vector(1 downto 0);
	signal ga_ls_order_temp_1 : std_logic_vector(1 downto 0);
	signal ga_ls_order_temp_2 : std_logic_vector(1 downto 0);
	signal ga_ls_order_temp_3 : std_logic_vector(1 downto 0);
	signal TEMP_1_mux_0_0 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_1_0 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_2_0 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_3_0 : std_logic_vector(0 downto 0);
	signal TEMP_2_mux_0_0 : std_logic_vector(1 downto 0);
	signal TEMP_2_mux_1_0 : std_logic_vector(1 downto 0);
	signal TEMP_2_mux_2_0 : std_logic_vector(1 downto 0);
	signal TEMP_2_mux_3_0 : std_logic_vector(1 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(1 downto 0);
	signal TEMP_4_mux_0 : std_logic_vector(0 downto 0);
	signal ldq_wen_unshifted_0 : std_logic;
	signal ldq_wen_unshifted_1 : std_logic;
	signal ldq_wen_unshifted_2 : std_logic;
	signal ldq_wen_unshifted_3 : std_logic;
	signal stq_wen_unshifted_0 : std_logic;
	signal stq_wen_unshifted_1 : std_logic;
	signal TEMP_5_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_5_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_5_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_5_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_0 : std_logic;
	signal TEMP_6_res_1 : std_logic;
	signal TEMP_6_res_2 : std_logic;
	signal TEMP_6_res_3 : std_logic;
	signal TEMP_7_res_0 : std_logic_vector(1 downto 0);
	signal TEMP_7_res_1 : std_logic_vector(1 downto 0);
	signal TEMP_7_res_2 : std_logic_vector(1 downto 0);
	signal TEMP_7_res_3 : std_logic_vector(1 downto 0);
begin
	-- WrapSub Begin
	-- WrapSub(loads_sub, ldq_head, ldq_tail, 4)
	loads_sub <= std_logic_vector(unsigned(ldq_head_i) - unsigned(ldq_tail_i));
	-- WrapAdd End

	-- WrapSub Begin
	-- WrapSub(stores_sub, stq_head, stq_tail, 2)
	stores_sub <= std_logic_vector(unsigned(stq_head_i) - unsigned(stq_tail_i));
	-- WrapAdd End

	empty_loads <= "100" when ldq_empty_i else ( '0' & loads_sub );
	empty_stores <= "10" when stq_empty_i else ( '0' & stores_sub );
	group_init_ready_0 <= '1' when ( empty_loads >= "010" ) and ( empty_stores >= "01" ) else '0';
	group_init_ready_0_o <= group_init_ready_0;
	group_init_hs_0 <= group_init_ready_0 and group_init_valid_0_i;
	-- Mux1H For Rom Begin
	-- Mux1H(ldq_port_idx_rom, group_init_hs)
	-- Loop 0
	TEMP_1_mux_0_0 <= "0";
	ldq_port_idx_rom_0 <= TEMP_1_mux_0_0;
	-- Loop 1
	TEMP_1_mux_1_0 <= "1" when group_init_hs_0 else "0";
	ldq_port_idx_rom_1 <= TEMP_1_mux_1_0;
	-- Loop 2
	TEMP_1_mux_2_0 <= "0";
	ldq_port_idx_rom_2 <= TEMP_1_mux_2_0;
	-- Loop 3
	TEMP_1_mux_3_0 <= "0";
	ldq_port_idx_rom_3 <= TEMP_1_mux_3_0;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(ga_ls_order_rom, group_init_hs)
	-- Loop 0
	TEMP_2_mux_0_0 <= "00";
	ga_ls_order_rom_0 <= TEMP_2_mux_0_0;
	-- Loop 1
	TEMP_2_mux_1_0 <= "00";
	ga_ls_order_rom_1 <= TEMP_2_mux_1_0;
	-- Loop 2
	TEMP_2_mux_2_0 <= "00";
	ga_ls_order_rom_2 <= TEMP_2_mux_2_0;
	-- Loop 3
	TEMP_2_mux_3_0 <= "00";
	ga_ls_order_rom_3 <= TEMP_2_mux_3_0;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(num_loads, group_init_hs)
	TEMP_3_mux_0 <= "10" when group_init_hs_0 else "00";
	num_loads <= TEMP_3_mux_0;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(num_stores, group_init_hs)
	TEMP_4_mux_0 <= "1" when group_init_hs_0 else "0";
	num_stores <= TEMP_4_mux_0;
	-- Mux1H For Rom End

	num_loads_o <= num_loads;
	num_stores_o <= num_stores;
	ldq_wen_unshifted_0 <= '1' when num_loads > "00" else '0';
	ldq_wen_unshifted_1 <= '1' when num_loads > "01" else '0';
	ldq_wen_unshifted_2 <= '1' when num_loads > "10" else '0';
	ldq_wen_unshifted_3 <= '1' when num_loads > "11" else '0';
	stq_wen_unshifted_0 <= '1' when num_stores > "0" else '0';
	stq_wen_unshifted_1 <= '1' when num_stores > "1" else '0';
	-- Shifter Begin
	-- CyclicLeftShift(ldq_port_idx, ldq_port_idx_rom, ldq_tail)
	TEMP_5_res_0 <= ldq_port_idx_rom_2 when ldq_tail_i(1) else ldq_port_idx_rom_0;
	TEMP_5_res_1 <= ldq_port_idx_rom_3 when ldq_tail_i(1) else ldq_port_idx_rom_1;
	TEMP_5_res_2 <= ldq_port_idx_rom_0 when ldq_tail_i(1) else ldq_port_idx_rom_2;
	TEMP_5_res_3 <= ldq_port_idx_rom_1 when ldq_tail_i(1) else ldq_port_idx_rom_3;
	-- Layer End
	ldq_port_idx_0_o <= TEMP_5_res_3 when ldq_tail_i(0) else TEMP_5_res_0;
	ldq_port_idx_1_o <= TEMP_5_res_0 when ldq_tail_i(0) else TEMP_5_res_1;
	ldq_port_idx_2_o <= TEMP_5_res_1 when ldq_tail_i(0) else TEMP_5_res_2;
	ldq_port_idx_3_o <= TEMP_5_res_2 when ldq_tail_i(0) else TEMP_5_res_3;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ldq_wen, ldq_wen_unshifted, ldq_tail)
	TEMP_6_res_0 <= ldq_wen_unshifted_2 when ldq_tail_i(1) else ldq_wen_unshifted_0;
	TEMP_6_res_1 <= ldq_wen_unshifted_3 when ldq_tail_i(1) else ldq_wen_unshifted_1;
	TEMP_6_res_2 <= ldq_wen_unshifted_0 when ldq_tail_i(1) else ldq_wen_unshifted_2;
	TEMP_6_res_3 <= ldq_wen_unshifted_1 when ldq_tail_i(1) else ldq_wen_unshifted_3;
	-- Layer End
	ldq_wen_0_o <= TEMP_6_res_3 when ldq_tail_i(0) else TEMP_6_res_0;
	ldq_wen_1_o <= TEMP_6_res_0 when ldq_tail_i(0) else TEMP_6_res_1;
	ldq_wen_2_o <= TEMP_6_res_1 when ldq_tail_i(0) else TEMP_6_res_2;
	ldq_wen_3_o <= TEMP_6_res_2 when ldq_tail_i(0) else TEMP_6_res_3;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(stq_wen, stq_wen_unshifted, stq_tail)
	stq_wen_0_o <= stq_wen_unshifted_1 when stq_tail_i(0) else stq_wen_unshifted_0;
	stq_wen_1_o <= stq_wen_unshifted_0 when stq_tail_i(0) else stq_wen_unshifted_1;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_0, ga_ls_order_rom_0, stq_tail)
	ga_ls_order_temp_0(0) <= ga_ls_order_rom_0(1) when stq_tail_i(0) else ga_ls_order_rom_0(0);
	ga_ls_order_temp_0(1) <= ga_ls_order_rom_0(0) when stq_tail_i(0) else ga_ls_order_rom_0(1);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_1, ga_ls_order_rom_1, stq_tail)
	ga_ls_order_temp_1(0) <= ga_ls_order_rom_1(1) when stq_tail_i(0) else ga_ls_order_rom_1(0);
	ga_ls_order_temp_1(1) <= ga_ls_order_rom_1(0) when stq_tail_i(0) else ga_ls_order_rom_1(1);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_2, ga_ls_order_rom_2, stq_tail)
	ga_ls_order_temp_2(0) <= ga_ls_order_rom_2(1) when stq_tail_i(0) else ga_ls_order_rom_2(0);
	ga_ls_order_temp_2(1) <= ga_ls_order_rom_2(0) when stq_tail_i(0) else ga_ls_order_rom_2(1);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_3, ga_ls_order_rom_3, stq_tail)
	ga_ls_order_temp_3(0) <= ga_ls_order_rom_3(1) when stq_tail_i(0) else ga_ls_order_rom_3(0);
	ga_ls_order_temp_3(1) <= ga_ls_order_rom_3(0) when stq_tail_i(0) else ga_ls_order_rom_3(1);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order, ga_ls_order_temp, ldq_tail)
	TEMP_7_res_0 <= ga_ls_order_temp_2 when ldq_tail_i(1) else ga_ls_order_temp_0;
	TEMP_7_res_1 <= ga_ls_order_temp_3 when ldq_tail_i(1) else ga_ls_order_temp_1;
	TEMP_7_res_2 <= ga_ls_order_temp_0 when ldq_tail_i(1) else ga_ls_order_temp_2;
	TEMP_7_res_3 <= ga_ls_order_temp_1 when ldq_tail_i(1) else ga_ls_order_temp_3;
	-- Layer End
	ga_ls_order_0_o <= TEMP_7_res_3 when ldq_tail_i(0) else TEMP_7_res_0;
	ga_ls_order_1_o <= TEMP_7_res_0 when ldq_tail_i(0) else TEMP_7_res_1;
	ga_ls_order_2_o <= TEMP_7_res_1 when ldq_tail_i(0) else TEMP_7_res_2;
	ga_ls_order_3_o <= TEMP_7_res_2 when ldq_tail_i(0) else TEMP_7_res_3;
	-- Shifter End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1_core_lda is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_payload_0_i : in std_logic_vector(8 downto 0);
		port_payload_1_i : in std_logic_vector(8 downto 0);
		port_valid_0_i : in std_logic;
		port_valid_1_i : in std_logic;
		port_ready_0_o : out std_logic;
		port_ready_1_o : out std_logic;
		entry_alloc_0_i : in std_logic;
		entry_alloc_1_i : in std_logic;
		entry_alloc_2_i : in std_logic;
		entry_alloc_3_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_valid_2_i : in std_logic;
		entry_payload_valid_3_i : in std_logic;
		entry_port_idx_0_i : in std_logic_vector(0 downto 0);
		entry_port_idx_1_i : in std_logic_vector(0 downto 0);
		entry_port_idx_2_i : in std_logic_vector(0 downto 0);
		entry_port_idx_3_i : in std_logic_vector(0 downto 0);
		entry_payload_0_o : out std_logic_vector(8 downto 0);
		entry_payload_1_o : out std_logic_vector(8 downto 0);
		entry_payload_2_o : out std_logic_vector(8 downto 0);
		entry_payload_3_o : out std_logic_vector(8 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		entry_wen_2_o : out std_logic;
		entry_wen_3_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(3 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_lda is
	signal entry_port_idx_oh_0 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_2 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_3 : std_logic_vector(1 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(8 downto 0);
	signal TEMP_1_mux_1 : std_logic_vector(8 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(8 downto 0);
	signal TEMP_2_mux_1 : std_logic_vector(8 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(8 downto 0);
	signal TEMP_3_mux_1 : std_logic_vector(8 downto 0);
	signal TEMP_4_mux_0 : std_logic_vector(8 downto 0);
	signal TEMP_4_mux_1 : std_logic_vector(8 downto 0);
	signal entry_ptq_ready_0 : std_logic;
	signal entry_ptq_ready_1 : std_logic;
	signal entry_ptq_ready_2 : std_logic;
	signal entry_ptq_ready_3 : std_logic;
	signal entry_waiting_for_port_0 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_1 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_2 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_3 : std_logic_vector(1 downto 0);
	signal port_ready_vec : std_logic_vector(1 downto 0);
	signal TEMP_5_res_0 : std_logic_vector(1 downto 0);
	signal TEMP_5_res_1 : std_logic_vector(1 downto 0);
	signal entry_port_options_0 : std_logic_vector(1 downto 0);
	signal entry_port_options_1 : std_logic_vector(1 downto 0);
	signal entry_port_options_2 : std_logic_vector(1 downto 0);
	signal entry_port_options_3 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_2 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_3 : std_logic_vector(1 downto 0);
	signal TEMP_6_double_in_0 : std_logic_vector(7 downto 0);
	signal TEMP_6_double_out_0 : std_logic_vector(7 downto 0);
	signal TEMP_6_double_in_1 : std_logic_vector(7 downto 0);
	signal TEMP_6_double_out_1 : std_logic_vector(7 downto 0);
begin
	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_0, entry_port_idx_0)
	entry_port_idx_oh_0(0) <= '1' when entry_port_idx_0_i = "0" else '0';
	entry_port_idx_oh_0(1) <= '1' when entry_port_idx_0_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_1, entry_port_idx_1)
	entry_port_idx_oh_1(0) <= '1' when entry_port_idx_1_i = "0" else '0';
	entry_port_idx_oh_1(1) <= '1' when entry_port_idx_1_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_2, entry_port_idx_2)
	entry_port_idx_oh_2(0) <= '1' when entry_port_idx_2_i = "0" else '0';
	entry_port_idx_oh_2(1) <= '1' when entry_port_idx_2_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_3, entry_port_idx_3)
	entry_port_idx_oh_3(0) <= '1' when entry_port_idx_3_i = "0" else '0';
	entry_port_idx_oh_3(1) <= '1' when entry_port_idx_3_i = "1" else '0';
	-- Bits To One-Hot End

	-- Mux1H Begin
	-- Mux1H(entry_payload_0, port_payload, entry_port_idx_oh_0)
	TEMP_1_mux_0 <= port_payload_0_i when entry_port_idx_oh_0(0) = '1' else "000000000";
	TEMP_1_mux_1 <= port_payload_1_i when entry_port_idx_oh_0(1) = '1' else "000000000";
	entry_payload_0_o <= TEMP_1_mux_0 or TEMP_1_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_1, port_payload, entry_port_idx_oh_1)
	TEMP_2_mux_0 <= port_payload_0_i when entry_port_idx_oh_1(0) = '1' else "000000000";
	TEMP_2_mux_1 <= port_payload_1_i when entry_port_idx_oh_1(1) = '1' else "000000000";
	entry_payload_1_o <= TEMP_2_mux_0 or TEMP_2_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_2, port_payload, entry_port_idx_oh_2)
	TEMP_3_mux_0 <= port_payload_0_i when entry_port_idx_oh_2(0) = '1' else "000000000";
	TEMP_3_mux_1 <= port_payload_1_i when entry_port_idx_oh_2(1) = '1' else "000000000";
	entry_payload_2_o <= TEMP_3_mux_0 or TEMP_3_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_3, port_payload, entry_port_idx_oh_3)
	TEMP_4_mux_0 <= port_payload_0_i when entry_port_idx_oh_3(0) = '1' else "000000000";
	TEMP_4_mux_1 <= port_payload_1_i when entry_port_idx_oh_3(1) = '1' else "000000000";
	entry_payload_3_o <= TEMP_4_mux_0 or TEMP_4_mux_1;
	-- Mux1H End

	entry_ptq_ready_0 <= entry_alloc_0_i and not entry_payload_valid_0_i;
	entry_ptq_ready_1 <= entry_alloc_1_i and not entry_payload_valid_1_i;
	entry_ptq_ready_2 <= entry_alloc_2_i and not entry_payload_valid_2_i;
	entry_ptq_ready_3 <= entry_alloc_3_i and not entry_payload_valid_3_i;
	entry_waiting_for_port_0 <= entry_port_idx_oh_0 when entry_ptq_ready_0 else "00";
	entry_waiting_for_port_1 <= entry_port_idx_oh_1 when entry_ptq_ready_1 else "00";
	entry_waiting_for_port_2 <= entry_port_idx_oh_2 when entry_ptq_ready_2 else "00";
	entry_waiting_for_port_3 <= entry_port_idx_oh_3 when entry_ptq_ready_3 else "00";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_waiting_for_port, or)
	TEMP_5_res_0 <= entry_waiting_for_port_0 or entry_waiting_for_port_2;
	TEMP_5_res_1 <= entry_waiting_for_port_1 or entry_waiting_for_port_3;
	-- Layer End
	port_ready_vec <= TEMP_5_res_0 or TEMP_5_res_1;
	-- Reduction End

	port_ready_0_o <= port_ready_vec(0);
	port_ready_1_o <= port_ready_vec(1);
	entry_port_options_0(0) <= entry_waiting_for_port_0(0) and port_valid_0_i;
	entry_port_options_0(1) <= entry_waiting_for_port_0(1) and port_valid_1_i;
	entry_port_options_1(0) <= entry_waiting_for_port_1(0) and port_valid_0_i;
	entry_port_options_1(1) <= entry_waiting_for_port_1(1) and port_valid_1_i;
	entry_port_options_2(0) <= entry_waiting_for_port_2(0) and port_valid_0_i;
	entry_port_options_2(1) <= entry_waiting_for_port_2(1) and port_valid_1_i;
	entry_port_options_3(0) <= entry_waiting_for_port_3(0) and port_valid_0_i;
	entry_port_options_3(1) <= entry_waiting_for_port_3(1) and port_valid_1_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_transfer, entry_port_options, queue_head_oh)
	TEMP_6_double_in_0(0) <= entry_port_options_0(0);
	TEMP_6_double_in_0(4) <= entry_port_options_0(0);
	TEMP_6_double_in_0(1) <= entry_port_options_1(0);
	TEMP_6_double_in_0(5) <= entry_port_options_1(0);
	TEMP_6_double_in_0(2) <= entry_port_options_2(0);
	TEMP_6_double_in_0(6) <= entry_port_options_2(0);
	TEMP_6_double_in_0(3) <= entry_port_options_3(0);
	TEMP_6_double_in_0(7) <= entry_port_options_3(0);
	TEMP_6_double_out_0 <= TEMP_6_double_in_0 and not std_logic_vector( unsigned( TEMP_6_double_in_0 ) - unsigned( "0000" & queue_head_oh_i ) );
	entry_port_transfer_0(0) <= TEMP_6_double_out_0(0) or TEMP_6_double_out_0(4);
	entry_port_transfer_1(0) <= TEMP_6_double_out_0(1) or TEMP_6_double_out_0(5);
	entry_port_transfer_2(0) <= TEMP_6_double_out_0(2) or TEMP_6_double_out_0(6);
	entry_port_transfer_3(0) <= TEMP_6_double_out_0(3) or TEMP_6_double_out_0(7);
	TEMP_6_double_in_1(0) <= entry_port_options_0(1);
	TEMP_6_double_in_1(4) <= entry_port_options_0(1);
	TEMP_6_double_in_1(1) <= entry_port_options_1(1);
	TEMP_6_double_in_1(5) <= entry_port_options_1(1);
	TEMP_6_double_in_1(2) <= entry_port_options_2(1);
	TEMP_6_double_in_1(6) <= entry_port_options_2(1);
	TEMP_6_double_in_1(3) <= entry_port_options_3(1);
	TEMP_6_double_in_1(7) <= entry_port_options_3(1);
	TEMP_6_double_out_1 <= TEMP_6_double_in_1 and not std_logic_vector( unsigned( TEMP_6_double_in_1 ) - unsigned( "0000" & queue_head_oh_i ) );
	entry_port_transfer_0(1) <= TEMP_6_double_out_1(0) or TEMP_6_double_out_1(4);
	entry_port_transfer_1(1) <= TEMP_6_double_out_1(1) or TEMP_6_double_out_1(5);
	entry_port_transfer_2(1) <= TEMP_6_double_out_1(2) or TEMP_6_double_out_1(6);
	entry_port_transfer_3(1) <= TEMP_6_double_out_1(3) or TEMP_6_double_out_1(7);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(entry_wen_0, entry_port_transfer_0, or)
	entry_wen_0_o <= entry_port_transfer_0(0) or entry_port_transfer_0(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_1, entry_port_transfer_1, or)
	entry_wen_1_o <= entry_port_transfer_1(0) or entry_port_transfer_1(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_2, entry_port_transfer_2, or)
	entry_wen_2_o <= entry_port_transfer_2(0) or entry_port_transfer_2(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_3, entry_port_transfer_3, or)
	entry_wen_3_o <= entry_port_transfer_3(0) or entry_port_transfer_3(1);
	-- Reduction End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1_core_ldd is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_payload_0_o : out std_logic_vector(31 downto 0);
		port_payload_1_o : out std_logic_vector(31 downto 0);
		port_valid_0_o : out std_logic;
		port_valid_1_o : out std_logic;
		port_ready_0_i : in std_logic;
		port_ready_1_i : in std_logic;
		entry_alloc_0_i : in std_logic;
		entry_alloc_1_i : in std_logic;
		entry_alloc_2_i : in std_logic;
		entry_alloc_3_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_valid_2_i : in std_logic;
		entry_payload_valid_3_i : in std_logic;
		entry_port_idx_0_i : in std_logic_vector(0 downto 0);
		entry_port_idx_1_i : in std_logic_vector(0 downto 0);
		entry_port_idx_2_i : in std_logic_vector(0 downto 0);
		entry_port_idx_3_i : in std_logic_vector(0 downto 0);
		entry_payload_0_i : in std_logic_vector(31 downto 0);
		entry_payload_1_i : in std_logic_vector(31 downto 0);
		entry_payload_2_i : in std_logic_vector(31 downto 0);
		entry_payload_3_i : in std_logic_vector(31 downto 0);
		entry_reset_0_o : out std_logic;
		entry_reset_1_o : out std_logic;
		entry_reset_2_o : out std_logic;
		entry_reset_3_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(3 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_ldd is
	signal entry_port_idx_oh_0 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_2 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_3 : std_logic_vector(1 downto 0);
	signal entry_allocated_for_port_0 : std_logic_vector(1 downto 0);
	signal entry_allocated_for_port_1 : std_logic_vector(1 downto 0);
	signal entry_allocated_for_port_2 : std_logic_vector(1 downto 0);
	signal entry_allocated_for_port_3 : std_logic_vector(1 downto 0);
	signal oldest_entry_allocated_per_port_0 : std_logic_vector(1 downto 0);
	signal oldest_entry_allocated_per_port_1 : std_logic_vector(1 downto 0);
	signal oldest_entry_allocated_per_port_2 : std_logic_vector(1 downto 0);
	signal oldest_entry_allocated_per_port_3 : std_logic_vector(1 downto 0);
	signal TEMP_1_double_in_0 : std_logic_vector(7 downto 0);
	signal TEMP_1_double_out_0 : std_logic_vector(7 downto 0);
	signal TEMP_1_double_in_1 : std_logic_vector(7 downto 0);
	signal TEMP_1_double_out_1 : std_logic_vector(7 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_4_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_4_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_4_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_4_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_5_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_5_res_1 : std_logic_vector(31 downto 0);
	signal entry_waiting_for_port_valid_0 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_valid_1 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_valid_2 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_valid_3 : std_logic_vector(1 downto 0);
	signal port_valid_vec : std_logic_vector(1 downto 0);
	signal TEMP_6_res_0 : std_logic_vector(1 downto 0);
	signal TEMP_6_res_1 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_2 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_3 : std_logic_vector(1 downto 0);
begin
	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_0, entry_port_idx_0)
	entry_port_idx_oh_0(0) <= '1' when entry_port_idx_0_i = "0" else '0';
	entry_port_idx_oh_0(1) <= '1' when entry_port_idx_0_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_1, entry_port_idx_1)
	entry_port_idx_oh_1(0) <= '1' when entry_port_idx_1_i = "0" else '0';
	entry_port_idx_oh_1(1) <= '1' when entry_port_idx_1_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_2, entry_port_idx_2)
	entry_port_idx_oh_2(0) <= '1' when entry_port_idx_2_i = "0" else '0';
	entry_port_idx_oh_2(1) <= '1' when entry_port_idx_2_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_3, entry_port_idx_3)
	entry_port_idx_oh_3(0) <= '1' when entry_port_idx_3_i = "0" else '0';
	entry_port_idx_oh_3(1) <= '1' when entry_port_idx_3_i = "1" else '0';
	-- Bits To One-Hot End

	entry_allocated_for_port_0 <= entry_port_idx_oh_0 when entry_alloc_0_i else "00";
	entry_allocated_for_port_1 <= entry_port_idx_oh_1 when entry_alloc_1_i else "00";
	entry_allocated_for_port_2 <= entry_port_idx_oh_2 when entry_alloc_2_i else "00";
	entry_allocated_for_port_3 <= entry_port_idx_oh_3 when entry_alloc_3_i else "00";
	-- Priority Masking Begin
	-- CyclicPriorityMask(oldest_entry_allocated_per_port, entry_allocated_for_port, queue_head_oh)
	TEMP_1_double_in_0(0) <= entry_allocated_for_port_0(0);
	TEMP_1_double_in_0(4) <= entry_allocated_for_port_0(0);
	TEMP_1_double_in_0(1) <= entry_allocated_for_port_1(0);
	TEMP_1_double_in_0(5) <= entry_allocated_for_port_1(0);
	TEMP_1_double_in_0(2) <= entry_allocated_for_port_2(0);
	TEMP_1_double_in_0(6) <= entry_allocated_for_port_2(0);
	TEMP_1_double_in_0(3) <= entry_allocated_for_port_3(0);
	TEMP_1_double_in_0(7) <= entry_allocated_for_port_3(0);
	TEMP_1_double_out_0 <= TEMP_1_double_in_0 and not std_logic_vector( unsigned( TEMP_1_double_in_0 ) - unsigned( "0000" & queue_head_oh_i ) );
	oldest_entry_allocated_per_port_0(0) <= TEMP_1_double_out_0(0) or TEMP_1_double_out_0(4);
	oldest_entry_allocated_per_port_1(0) <= TEMP_1_double_out_0(1) or TEMP_1_double_out_0(5);
	oldest_entry_allocated_per_port_2(0) <= TEMP_1_double_out_0(2) or TEMP_1_double_out_0(6);
	oldest_entry_allocated_per_port_3(0) <= TEMP_1_double_out_0(3) or TEMP_1_double_out_0(7);
	TEMP_1_double_in_1(0) <= entry_allocated_for_port_0(1);
	TEMP_1_double_in_1(4) <= entry_allocated_for_port_0(1);
	TEMP_1_double_in_1(1) <= entry_allocated_for_port_1(1);
	TEMP_1_double_in_1(5) <= entry_allocated_for_port_1(1);
	TEMP_1_double_in_1(2) <= entry_allocated_for_port_2(1);
	TEMP_1_double_in_1(6) <= entry_allocated_for_port_2(1);
	TEMP_1_double_in_1(3) <= entry_allocated_for_port_3(1);
	TEMP_1_double_in_1(7) <= entry_allocated_for_port_3(1);
	TEMP_1_double_out_1 <= TEMP_1_double_in_1 and not std_logic_vector( unsigned( TEMP_1_double_in_1 ) - unsigned( "0000" & queue_head_oh_i ) );
	oldest_entry_allocated_per_port_0(1) <= TEMP_1_double_out_1(0) or TEMP_1_double_out_1(4);
	oldest_entry_allocated_per_port_1(1) <= TEMP_1_double_out_1(1) or TEMP_1_double_out_1(5);
	oldest_entry_allocated_per_port_2(1) <= TEMP_1_double_out_1(2) or TEMP_1_double_out_1(6);
	oldest_entry_allocated_per_port_3(1) <= TEMP_1_double_out_1(3) or TEMP_1_double_out_1(7);
	-- Priority Masking End

	-- Mux1H Begin
	-- Mux1H(port_payload_0, entry_payload, oldest_entry_allocated_per_port)
	TEMP_2_mux_0 <= entry_payload_0_i when oldest_entry_allocated_per_port_0(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_1 <= entry_payload_1_i when oldest_entry_allocated_per_port_1(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_2 <= entry_payload_2_i when oldest_entry_allocated_per_port_2(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_3 <= entry_payload_3_i when oldest_entry_allocated_per_port_3(0) = '1' else "00000000000000000000000000000000";
	TEMP_3_res_0 <= TEMP_2_mux_0 or TEMP_2_mux_2;
	TEMP_3_res_1 <= TEMP_2_mux_1 or TEMP_2_mux_3;
	-- Layer End
	port_payload_0_o <= TEMP_3_res_0 or TEMP_3_res_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(port_payload_1, entry_payload, oldest_entry_allocated_per_port)
	TEMP_4_mux_0 <= entry_payload_0_i when oldest_entry_allocated_per_port_0(1) = '1' else "00000000000000000000000000000000";
	TEMP_4_mux_1 <= entry_payload_1_i when oldest_entry_allocated_per_port_1(1) = '1' else "00000000000000000000000000000000";
	TEMP_4_mux_2 <= entry_payload_2_i when oldest_entry_allocated_per_port_2(1) = '1' else "00000000000000000000000000000000";
	TEMP_4_mux_3 <= entry_payload_3_i when oldest_entry_allocated_per_port_3(1) = '1' else "00000000000000000000000000000000";
	TEMP_5_res_0 <= TEMP_4_mux_0 or TEMP_4_mux_2;
	TEMP_5_res_1 <= TEMP_4_mux_1 or TEMP_4_mux_3;
	-- Layer End
	port_payload_1_o <= TEMP_5_res_0 or TEMP_5_res_1;
	-- Mux1H End

	entry_waiting_for_port_valid_0 <= oldest_entry_allocated_per_port_0 when entry_payload_valid_0_i else "00";
	entry_waiting_for_port_valid_1 <= oldest_entry_allocated_per_port_1 when entry_payload_valid_1_i else "00";
	entry_waiting_for_port_valid_2 <= oldest_entry_allocated_per_port_2 when entry_payload_valid_2_i else "00";
	entry_waiting_for_port_valid_3 <= oldest_entry_allocated_per_port_3 when entry_payload_valid_3_i else "00";
	-- Reduction Begin
	-- Reduce(port_valid_vec, entry_waiting_for_port_valid, or)
	TEMP_6_res_0 <= entry_waiting_for_port_valid_0 or entry_waiting_for_port_valid_2;
	TEMP_6_res_1 <= entry_waiting_for_port_valid_1 or entry_waiting_for_port_valid_3;
	-- Layer End
	port_valid_vec <= TEMP_6_res_0 or TEMP_6_res_1;
	-- Reduction End

	port_valid_0_o <= port_valid_vec(0);
	port_valid_1_o <= port_valid_vec(1);
	entry_port_transfer_0(0) <= entry_waiting_for_port_valid_0(0) and port_ready_0_i;
	entry_port_transfer_0(1) <= entry_waiting_for_port_valid_0(1) and port_ready_1_i;
	entry_port_transfer_1(0) <= entry_waiting_for_port_valid_1(0) and port_ready_0_i;
	entry_port_transfer_1(1) <= entry_waiting_for_port_valid_1(1) and port_ready_1_i;
	entry_port_transfer_2(0) <= entry_waiting_for_port_valid_2(0) and port_ready_0_i;
	entry_port_transfer_2(1) <= entry_waiting_for_port_valid_2(1) and port_ready_1_i;
	entry_port_transfer_3(0) <= entry_waiting_for_port_valid_3(0) and port_ready_0_i;
	entry_port_transfer_3(1) <= entry_waiting_for_port_valid_3(1) and port_ready_1_i;
	-- Reduction Begin
	-- Reduce(entry_reset_0, entry_port_transfer_0, or)
	entry_reset_0_o <= entry_port_transfer_0(0) or entry_port_transfer_0(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_1, entry_port_transfer_1, or)
	entry_reset_1_o <= entry_port_transfer_1(0) or entry_port_transfer_1(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_2, entry_port_transfer_2, or)
	entry_reset_2_o <= entry_port_transfer_2(0) or entry_port_transfer_2(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_3, entry_port_transfer_3, or)
	entry_reset_3_o <= entry_port_transfer_3(0) or entry_port_transfer_3(1);
	-- Reduction End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1_core_sta is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_payload_0_i : in std_logic_vector(8 downto 0);
		port_valid_0_i : in std_logic;
		port_ready_0_o : out std_logic;
		entry_alloc_0_i : in std_logic;
		entry_alloc_1_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_0_o : out std_logic_vector(8 downto 0);
		entry_payload_1_o : out std_logic_vector(8 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(1 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_sta is
	signal entry_port_idx_oh_0 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(8 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(8 downto 0);
	signal entry_ptq_ready_0 : std_logic;
	signal entry_ptq_ready_1 : std_logic;
	signal entry_waiting_for_port_0 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_1 : std_logic_vector(0 downto 0);
	signal port_ready_vec : std_logic_vector(0 downto 0);
	signal entry_port_options_0 : std_logic_vector(0 downto 0);
	signal entry_port_options_1 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(0 downto 0);
	signal TEMP_3_double_in_0 : std_logic_vector(3 downto 0);
	signal TEMP_3_double_out_0 : std_logic_vector(3 downto 0);
begin
	entry_port_idx_oh_0 <= "1";
	entry_port_idx_oh_1 <= "1";
	-- Mux1H Begin
	-- Mux1H(entry_payload_0, port_payload, entry_port_idx_oh_0)
	TEMP_1_mux_0 <= port_payload_0_i when entry_port_idx_oh_0(0) = '1' else "000000000";
	entry_payload_0_o <= TEMP_1_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_1, port_payload, entry_port_idx_oh_1)
	TEMP_2_mux_0 <= port_payload_0_i when entry_port_idx_oh_1(0) = '1' else "000000000";
	entry_payload_1_o <= TEMP_2_mux_0;
	-- Mux1H End

	entry_ptq_ready_0 <= entry_alloc_0_i and not entry_payload_valid_0_i;
	entry_ptq_ready_1 <= entry_alloc_1_i and not entry_payload_valid_1_i;
	entry_waiting_for_port_0 <= entry_port_idx_oh_0 when entry_ptq_ready_0 else "0";
	entry_waiting_for_port_1 <= entry_port_idx_oh_1 when entry_ptq_ready_1 else "0";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_waiting_for_port, or)
	port_ready_vec <= entry_waiting_for_port_0 or entry_waiting_for_port_1;
	-- Reduction End

	port_ready_0_o <= port_ready_vec(0);
	entry_port_options_0(0) <= entry_waiting_for_port_0(0) and port_valid_0_i;
	entry_port_options_1(0) <= entry_waiting_for_port_1(0) and port_valid_0_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_transfer, entry_port_options, queue_head_oh)
	TEMP_3_double_in_0(0) <= entry_port_options_0(0);
	TEMP_3_double_in_0(2) <= entry_port_options_0(0);
	TEMP_3_double_in_0(1) <= entry_port_options_1(0);
	TEMP_3_double_in_0(3) <= entry_port_options_1(0);
	TEMP_3_double_out_0 <= TEMP_3_double_in_0 and not std_logic_vector( unsigned( TEMP_3_double_in_0 ) - unsigned( "00" & queue_head_oh_i ) );
	entry_port_transfer_0(0) <= TEMP_3_double_out_0(0) or TEMP_3_double_out_0(2);
	entry_port_transfer_1(0) <= TEMP_3_double_out_0(1) or TEMP_3_double_out_0(3);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(entry_wen_0, entry_port_transfer_0, or)
	entry_wen_0_o <= entry_port_transfer_0(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_1, entry_port_transfer_1, or)
	entry_wen_1_o <= entry_port_transfer_1(0);
	-- Reduction End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1_core_std is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_payload_0_i : in std_logic_vector(31 downto 0);
		port_valid_0_i : in std_logic;
		port_ready_0_o : out std_logic;
		entry_alloc_0_i : in std_logic;
		entry_alloc_1_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_0_o : out std_logic_vector(31 downto 0);
		entry_payload_1_o : out std_logic_vector(31 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(1 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_std is
	signal entry_port_idx_oh_0 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(31 downto 0);
	signal entry_ptq_ready_0 : std_logic;
	signal entry_ptq_ready_1 : std_logic;
	signal entry_waiting_for_port_0 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_1 : std_logic_vector(0 downto 0);
	signal port_ready_vec : std_logic_vector(0 downto 0);
	signal entry_port_options_0 : std_logic_vector(0 downto 0);
	signal entry_port_options_1 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(0 downto 0);
	signal TEMP_3_double_in_0 : std_logic_vector(3 downto 0);
	signal TEMP_3_double_out_0 : std_logic_vector(3 downto 0);
begin
	entry_port_idx_oh_0 <= "1";
	entry_port_idx_oh_1 <= "1";
	-- Mux1H Begin
	-- Mux1H(entry_payload_0, port_payload, entry_port_idx_oh_0)
	TEMP_1_mux_0 <= port_payload_0_i when entry_port_idx_oh_0(0) = '1' else "00000000000000000000000000000000";
	entry_payload_0_o <= TEMP_1_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_1, port_payload, entry_port_idx_oh_1)
	TEMP_2_mux_0 <= port_payload_0_i when entry_port_idx_oh_1(0) = '1' else "00000000000000000000000000000000";
	entry_payload_1_o <= TEMP_2_mux_0;
	-- Mux1H End

	entry_ptq_ready_0 <= entry_alloc_0_i and not entry_payload_valid_0_i;
	entry_ptq_ready_1 <= entry_alloc_1_i and not entry_payload_valid_1_i;
	entry_waiting_for_port_0 <= entry_port_idx_oh_0 when entry_ptq_ready_0 else "0";
	entry_waiting_for_port_1 <= entry_port_idx_oh_1 when entry_ptq_ready_1 else "0";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_waiting_for_port, or)
	port_ready_vec <= entry_waiting_for_port_0 or entry_waiting_for_port_1;
	-- Reduction End

	port_ready_0_o <= port_ready_vec(0);
	entry_port_options_0(0) <= entry_waiting_for_port_0(0) and port_valid_0_i;
	entry_port_options_1(0) <= entry_waiting_for_port_1(0) and port_valid_0_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_transfer, entry_port_options, queue_head_oh)
	TEMP_3_double_in_0(0) <= entry_port_options_0(0);
	TEMP_3_double_in_0(2) <= entry_port_options_0(0);
	TEMP_3_double_in_0(1) <= entry_port_options_1(0);
	TEMP_3_double_in_0(3) <= entry_port_options_1(0);
	TEMP_3_double_out_0 <= TEMP_3_double_in_0 and not std_logic_vector( unsigned( TEMP_3_double_in_0 ) - unsigned( "00" & queue_head_oh_i ) );
	entry_port_transfer_0(0) <= TEMP_3_double_out_0(0) or TEMP_3_double_out_0(2);
	entry_port_transfer_1(0) <= TEMP_3_double_out_0(1) or TEMP_3_double_out_0(3);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(entry_wen_0, entry_port_transfer_0, or)
	entry_wen_0_o <= entry_port_transfer_0(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_1, entry_port_transfer_1, or)
	entry_wen_1_o <= entry_port_transfer_1(0);
	-- Reduction End


end architecture;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1_core is
	port(
		rst : in std_logic;
		clk : in std_logic;
		group_init_valid_0_i : in std_logic;
		group_init_ready_0_o : out std_logic;
		ldp_addr_0_i : in std_logic_vector(8 downto 0);
		ldp_addr_1_i : in std_logic_vector(8 downto 0);
		ldp_addr_valid_0_i : in std_logic;
		ldp_addr_valid_1_i : in std_logic;
		ldp_addr_ready_0_o : out std_logic;
		ldp_addr_ready_1_o : out std_logic;
		ldp_data_0_o : out std_logic_vector(31 downto 0);
		ldp_data_1_o : out std_logic_vector(31 downto 0);
		ldp_data_valid_0_o : out std_logic;
		ldp_data_valid_1_o : out std_logic;
		ldp_data_ready_0_i : in std_logic;
		ldp_data_ready_1_i : in std_logic;
		stp_addr_0_i : in std_logic_vector(8 downto 0);
		stp_addr_valid_0_i : in std_logic;
		stp_addr_ready_0_o : out std_logic;
		stp_data_0_i : in std_logic_vector(31 downto 0);
		stp_data_valid_0_i : in std_logic;
		stp_data_ready_0_o : out std_logic;
		empty_o : out std_logic;
		rreq_valid_0_o : out std_logic;
		rreq_ready_0_i : in std_logic;
		rreq_id_0_o : out std_logic_vector(3 downto 0);
		rreq_addr_0_o : out std_logic_vector(8 downto 0);
		rresp_valid_0_i : in std_logic;
		rresp_ready_0_o : out std_logic;
		rresp_id_0_i : in std_logic_vector(3 downto 0);
		rresp_data_0_i : in std_logic_vector(31 downto 0);
		wreq_valid_0_o : out std_logic;
		wreq_ready_0_i : in std_logic;
		wreq_id_0_o : out std_logic_vector(3 downto 0);
		wreq_addr_0_o : out std_logic_vector(8 downto 0);
		wreq_data_0_o : out std_logic_vector(31 downto 0);
		wresp_valid_0_i : in std_logic;
		wresp_ready_0_o : out std_logic;
		wresp_id_0_i : in std_logic_vector(3 downto 0);
		memStart_ready_o : out std_logic;
		memStart_valid_i : in std_logic;
		ctrlEnd_ready_o : out std_logic;
		ctrlEnd_valid_i : in std_logic;
		memEnd_ready_i : in std_logic;
		memEnd_valid_o : out std_logic
	);
end entity;

architecture arch of handshake_lsq_lsq1_core is
	signal memStartReady : std_logic;
	signal memEndValid : std_logic;
	signal ctrlEndReady : std_logic;
	signal TEMP_GEN_MEM : std_logic;
	signal ldq_alloc_0_d : std_logic;
	signal ldq_alloc_0_q : std_logic;
	signal ldq_alloc_1_d : std_logic;
	signal ldq_alloc_1_q : std_logic;
	signal ldq_alloc_2_d : std_logic;
	signal ldq_alloc_2_q : std_logic;
	signal ldq_alloc_3_d : std_logic;
	signal ldq_alloc_3_q : std_logic;
	signal ldq_issue_0_d : std_logic;
	signal ldq_issue_0_q : std_logic;
	signal ldq_issue_1_d : std_logic;
	signal ldq_issue_1_q : std_logic;
	signal ldq_issue_2_d : std_logic;
	signal ldq_issue_2_q : std_logic;
	signal ldq_issue_3_d : std_logic;
	signal ldq_issue_3_q : std_logic;
	signal ldq_port_idx_0_d : std_logic_vector(0 downto 0);
	signal ldq_port_idx_0_q : std_logic_vector(0 downto 0);
	signal ldq_port_idx_1_d : std_logic_vector(0 downto 0);
	signal ldq_port_idx_1_q : std_logic_vector(0 downto 0);
	signal ldq_port_idx_2_d : std_logic_vector(0 downto 0);
	signal ldq_port_idx_2_q : std_logic_vector(0 downto 0);
	signal ldq_port_idx_3_d : std_logic_vector(0 downto 0);
	signal ldq_port_idx_3_q : std_logic_vector(0 downto 0);
	signal ldq_addr_valid_0_d : std_logic;
	signal ldq_addr_valid_0_q : std_logic;
	signal ldq_addr_valid_1_d : std_logic;
	signal ldq_addr_valid_1_q : std_logic;
	signal ldq_addr_valid_2_d : std_logic;
	signal ldq_addr_valid_2_q : std_logic;
	signal ldq_addr_valid_3_d : std_logic;
	signal ldq_addr_valid_3_q : std_logic;
	signal ldq_addr_0_d : std_logic_vector(8 downto 0);
	signal ldq_addr_0_q : std_logic_vector(8 downto 0);
	signal ldq_addr_1_d : std_logic_vector(8 downto 0);
	signal ldq_addr_1_q : std_logic_vector(8 downto 0);
	signal ldq_addr_2_d : std_logic_vector(8 downto 0);
	signal ldq_addr_2_q : std_logic_vector(8 downto 0);
	signal ldq_addr_3_d : std_logic_vector(8 downto 0);
	signal ldq_addr_3_q : std_logic_vector(8 downto 0);
	signal ldq_data_valid_0_d : std_logic;
	signal ldq_data_valid_0_q : std_logic;
	signal ldq_data_valid_1_d : std_logic;
	signal ldq_data_valid_1_q : std_logic;
	signal ldq_data_valid_2_d : std_logic;
	signal ldq_data_valid_2_q : std_logic;
	signal ldq_data_valid_3_d : std_logic;
	signal ldq_data_valid_3_q : std_logic;
	signal ldq_data_0_d : std_logic_vector(31 downto 0);
	signal ldq_data_0_q : std_logic_vector(31 downto 0);
	signal ldq_data_1_d : std_logic_vector(31 downto 0);
	signal ldq_data_1_q : std_logic_vector(31 downto 0);
	signal ldq_data_2_d : std_logic_vector(31 downto 0);
	signal ldq_data_2_q : std_logic_vector(31 downto 0);
	signal ldq_data_3_d : std_logic_vector(31 downto 0);
	signal ldq_data_3_q : std_logic_vector(31 downto 0);
	signal stq_alloc_0_d : std_logic;
	signal stq_alloc_0_q : std_logic;
	signal stq_alloc_1_d : std_logic;
	signal stq_alloc_1_q : std_logic;
	signal stq_addr_valid_0_d : std_logic;
	signal stq_addr_valid_0_q : std_logic;
	signal stq_addr_valid_1_d : std_logic;
	signal stq_addr_valid_1_q : std_logic;
	signal stq_addr_0_d : std_logic_vector(8 downto 0);
	signal stq_addr_0_q : std_logic_vector(8 downto 0);
	signal stq_addr_1_d : std_logic_vector(8 downto 0);
	signal stq_addr_1_q : std_logic_vector(8 downto 0);
	signal stq_data_valid_0_d : std_logic;
	signal stq_data_valid_0_q : std_logic;
	signal stq_data_valid_1_d : std_logic;
	signal stq_data_valid_1_q : std_logic;
	signal stq_data_0_d : std_logic_vector(31 downto 0);
	signal stq_data_0_q : std_logic_vector(31 downto 0);
	signal stq_data_1_d : std_logic_vector(31 downto 0);
	signal stq_data_1_q : std_logic_vector(31 downto 0);
	signal store_is_older_0_d : std_logic_vector(1 downto 0);
	signal store_is_older_0_q : std_logic_vector(1 downto 0);
	signal store_is_older_1_d : std_logic_vector(1 downto 0);
	signal store_is_older_1_q : std_logic_vector(1 downto 0);
	signal store_is_older_2_d : std_logic_vector(1 downto 0);
	signal store_is_older_2_q : std_logic_vector(1 downto 0);
	signal store_is_older_3_d : std_logic_vector(1 downto 0);
	signal store_is_older_3_q : std_logic_vector(1 downto 0);
	signal ldq_tail_d : std_logic_vector(1 downto 0);
	signal ldq_tail_q : std_logic_vector(1 downto 0);
	signal ldq_head_d : std_logic_vector(1 downto 0);
	signal ldq_head_q : std_logic_vector(1 downto 0);
	signal stq_tail_d : std_logic_vector(0 downto 0);
	signal stq_tail_q : std_logic_vector(0 downto 0);
	signal stq_head_d : std_logic_vector(0 downto 0);
	signal stq_head_q : std_logic_vector(0 downto 0);
	signal stq_issue_d : std_logic_vector(0 downto 0);
	signal stq_issue_q : std_logic_vector(0 downto 0);
	signal stq_resp_d : std_logic_vector(0 downto 0);
	signal stq_resp_q : std_logic_vector(0 downto 0);
	signal ldq_wen_0 : std_logic;
	signal ldq_wen_1 : std_logic;
	signal ldq_wen_2 : std_logic;
	signal ldq_wen_3 : std_logic;
	signal ldq_addr_wen_0 : std_logic;
	signal ldq_addr_wen_1 : std_logic;
	signal ldq_addr_wen_2 : std_logic;
	signal ldq_addr_wen_3 : std_logic;
	signal ldq_reset_0 : std_logic;
	signal ldq_reset_1 : std_logic;
	signal ldq_reset_2 : std_logic;
	signal ldq_reset_3 : std_logic;
	signal stq_wen_0 : std_logic;
	signal stq_wen_1 : std_logic;
	signal stq_addr_wen_0 : std_logic;
	signal stq_addr_wen_1 : std_logic;
	signal stq_data_wen_0 : std_logic;
	signal stq_data_wen_1 : std_logic;
	signal stq_reset_0 : std_logic;
	signal stq_reset_1 : std_logic;
	signal ldq_data_wen_0 : std_logic;
	signal ldq_data_wen_1 : std_logic;
	signal ldq_data_wen_2 : std_logic;
	signal ldq_data_wen_3 : std_logic;
	signal ldq_issue_set_0 : std_logic;
	signal ldq_issue_set_1 : std_logic;
	signal ldq_issue_set_2 : std_logic;
	signal ldq_issue_set_3 : std_logic;
	signal ga_ls_order_0 : std_logic_vector(1 downto 0);
	signal ga_ls_order_1 : std_logic_vector(1 downto 0);
	signal ga_ls_order_2 : std_logic_vector(1 downto 0);
	signal ga_ls_order_3 : std_logic_vector(1 downto 0);
	signal num_loads : std_logic_vector(1 downto 0);
	signal num_stores : std_logic_vector(0 downto 0);
	signal stq_issue_en : std_logic;
	signal stq_resp_en : std_logic;
	signal ldq_empty : std_logic;
	signal stq_empty : std_logic;
	signal ldq_head_oh : std_logic_vector(3 downto 0);
	signal stq_head_oh : std_logic_vector(1 downto 0);
	signal ldq_alloc_next_0 : std_logic;
	signal ldq_alloc_next_1 : std_logic;
	signal ldq_alloc_next_2 : std_logic;
	signal ldq_alloc_next_3 : std_logic;
	signal stq_alloc_next_0 : std_logic;
	signal stq_alloc_next_1 : std_logic;
	signal ldq_not_empty : std_logic;
	signal stq_not_empty : std_logic;
	signal TEMP_1_res_0 : std_logic;
	signal TEMP_1_res_1 : std_logic;
	signal ldq_tail_oh : std_logic_vector(3 downto 0);
	signal ldq_head_next_oh : std_logic_vector(3 downto 0);
	signal ldq_head_next : std_logic_vector(1 downto 0);
	signal ldq_head_sel : std_logic;
	signal TEMP_2_double_in : std_logic_vector(7 downto 0);
	signal TEMP_2_double_out : std_logic_vector(7 downto 0);
	signal TEMP_3_res_0 : std_logic;
	signal TEMP_3_res_1 : std_logic;
	signal TEMP_4_in_0_0 : std_logic;
	signal TEMP_4_in_0_1 : std_logic;
	signal TEMP_4_in_0_2 : std_logic;
	signal TEMP_4_in_0_3 : std_logic;
	signal TEMP_4_out_0 : std_logic;
	signal TEMP_5_res_0 : std_logic;
	signal TEMP_5_res_1 : std_logic;
	signal TEMP_5_in_1_0 : std_logic;
	signal TEMP_5_in_1_1 : std_logic;
	signal TEMP_5_in_1_2 : std_logic;
	signal TEMP_5_in_1_3 : std_logic;
	signal TEMP_5_out_1 : std_logic;
	signal TEMP_6_res_0 : std_logic;
	signal TEMP_6_res_1 : std_logic;
	signal stq_tail_oh : std_logic_vector(1 downto 0);
	signal stq_head_next_oh : std_logic_vector(1 downto 0);
	signal stq_head_next : std_logic_vector(0 downto 0);
	signal stq_head_sel : std_logic;
	signal load_idx_oh_0 : std_logic_vector(3 downto 0);
	signal load_en_0 : std_logic;
	signal store_idx : std_logic_vector(0 downto 0);
	signal store_en : std_logic;
	signal bypass_idx_oh_0 : std_logic_vector(1 downto 0);
	signal bypass_idx_oh_1 : std_logic_vector(1 downto 0);
	signal bypass_idx_oh_2 : std_logic_vector(1 downto 0);
	signal bypass_idx_oh_3 : std_logic_vector(1 downto 0);
	signal bypass_en_0 : std_logic;
	signal bypass_en_1 : std_logic;
	signal bypass_en_2 : std_logic;
	signal bypass_en_3 : std_logic;
	signal ld_st_conflict_0 : std_logic_vector(1 downto 0);
	signal ld_st_conflict_1 : std_logic_vector(1 downto 0);
	signal ld_st_conflict_2 : std_logic_vector(1 downto 0);
	signal ld_st_conflict_3 : std_logic_vector(1 downto 0);
	signal can_bypass_0 : std_logic_vector(1 downto 0);
	signal can_bypass_1 : std_logic_vector(1 downto 0);
	signal can_bypass_2 : std_logic_vector(1 downto 0);
	signal can_bypass_3 : std_logic_vector(1 downto 0);
	signal addr_valid_0 : std_logic_vector(1 downto 0);
	signal addr_valid_1 : std_logic_vector(1 downto 0);
	signal addr_valid_2 : std_logic_vector(1 downto 0);
	signal addr_valid_3 : std_logic_vector(1 downto 0);
	signal addr_same_0 : std_logic_vector(1 downto 0);
	signal addr_same_1 : std_logic_vector(1 downto 0);
	signal addr_same_2 : std_logic_vector(1 downto 0);
	signal addr_same_3 : std_logic_vector(1 downto 0);
	signal load_conflict_0 : std_logic;
	signal load_conflict_1 : std_logic;
	signal load_conflict_2 : std_logic;
	signal load_conflict_3 : std_logic;
	signal load_req_valid_0 : std_logic;
	signal load_req_valid_1 : std_logic;
	signal load_req_valid_2 : std_logic;
	signal load_req_valid_3 : std_logic;
	signal can_load_0 : std_logic;
	signal can_load_1 : std_logic;
	signal can_load_2 : std_logic;
	signal can_load_3 : std_logic;
	signal TEMP_7_double_in : std_logic_vector(7 downto 0);
	signal TEMP_7_double_out : std_logic_vector(7 downto 0);
	signal TEMP_8_res_0 : std_logic;
	signal TEMP_8_res_1 : std_logic;
	signal st_ld_conflict : std_logic_vector(3 downto 0);
	signal store_conflict : std_logic;
	signal store_valid : std_logic;
	signal store_data_valid : std_logic;
	signal store_addr_valid : std_logic;
	signal TEMP_9_res : std_logic_vector(1 downto 0);
	signal stq_last_oh : std_logic_vector(1 downto 0);
	signal bypass_en_vec_0 : std_logic_vector(1 downto 0);
	signal TEMP_10_double_in : std_logic_vector(3 downto 0);
	signal TEMP_10_base_rev : std_logic_vector(1 downto 0);
	signal TEMP_10_double_out : std_logic_vector(3 downto 0);
	signal bypass_en_vec_1 : std_logic_vector(1 downto 0);
	signal TEMP_11_double_in : std_logic_vector(3 downto 0);
	signal TEMP_11_base_rev : std_logic_vector(1 downto 0);
	signal TEMP_11_double_out : std_logic_vector(3 downto 0);
	signal bypass_en_vec_2 : std_logic_vector(1 downto 0);
	signal TEMP_12_double_in : std_logic_vector(3 downto 0);
	signal TEMP_12_base_rev : std_logic_vector(1 downto 0);
	signal TEMP_12_double_out : std_logic_vector(3 downto 0);
	signal bypass_en_vec_3 : std_logic_vector(1 downto 0);
	signal TEMP_13_double_in : std_logic_vector(3 downto 0);
	signal TEMP_13_base_rev : std_logic_vector(1 downto 0);
	signal TEMP_13_double_out : std_logic_vector(3 downto 0);
	signal TEMP_14_in_0_0 : std_logic;
	signal TEMP_14_in_0_1 : std_logic;
	signal TEMP_14_in_0_2 : std_logic;
	signal TEMP_14_in_0_3 : std_logic;
	signal TEMP_14_out_0 : std_logic;
	signal TEMP_15_res_0 : std_logic;
	signal TEMP_15_res_1 : std_logic;
	signal TEMP_15_in_1_0 : std_logic;
	signal TEMP_15_in_1_1 : std_logic;
	signal TEMP_15_in_1_2 : std_logic;
	signal TEMP_15_in_1_3 : std_logic;
	signal TEMP_15_out_1 : std_logic;
	signal TEMP_16_res_0 : std_logic;
	signal TEMP_16_res_1 : std_logic;
	signal TEMP_16_in_2_0 : std_logic;
	signal TEMP_16_in_2_1 : std_logic;
	signal TEMP_16_in_2_2 : std_logic;
	signal TEMP_16_in_2_3 : std_logic;
	signal TEMP_16_out_2 : std_logic;
	signal TEMP_17_res_0 : std_logic;
	signal TEMP_17_res_1 : std_logic;
	signal TEMP_17_in_3_0 : std_logic;
	signal TEMP_17_in_3_1 : std_logic;
	signal TEMP_17_in_3_2 : std_logic;
	signal TEMP_17_in_3_3 : std_logic;
	signal TEMP_17_out_3 : std_logic;
	signal TEMP_18_res_0 : std_logic;
	signal TEMP_18_res_1 : std_logic;
	signal TEMP_19_mux_0 : std_logic_vector(8 downto 0);
	signal TEMP_19_mux_1 : std_logic_vector(8 downto 0);
	signal TEMP_19_mux_2 : std_logic_vector(8 downto 0);
	signal TEMP_19_mux_3 : std_logic_vector(8 downto 0);
	signal TEMP_20_res_0 : std_logic_vector(8 downto 0);
	signal TEMP_20_res_1 : std_logic_vector(8 downto 0);
	signal ldq_issue_set_vec_0 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_1 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_2 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_3 : std_logic_vector(0 downto 0);
	signal read_idx_oh_0_0 : std_logic;
	signal read_valid_0 : std_logic;
	signal read_data_0 : std_logic_vector(31 downto 0);
	signal TEMP_21_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_0 : std_logic_vector(31 downto 0);
	signal TEMP_22_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_22_mux_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_1_0 : std_logic;
	signal read_valid_1 : std_logic;
	signal read_data_1 : std_logic_vector(31 downto 0);
	signal TEMP_23_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_1 : std_logic_vector(31 downto 0);
	signal TEMP_24_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_24_mux_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_2_0 : std_logic;
	signal read_valid_2 : std_logic;
	signal read_data_2 : std_logic_vector(31 downto 0);
	signal TEMP_25_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_2 : std_logic_vector(31 downto 0);
	signal TEMP_26_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_26_mux_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_3_0 : std_logic;
	signal read_valid_3 : std_logic;
	signal read_data_3 : std_logic_vector(31 downto 0);
	signal TEMP_27_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_3 : std_logic_vector(31 downto 0);
	signal TEMP_28_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_28_mux_1 : std_logic_vector(31 downto 0);
begin
	-- Define the intermediate logic
	TEMP_GEN_MEM <= ctrlEnd_valid_i and stq_empty and ldq_empty;
	-- Define logic for the new interfaces needed by dynamatic
	process (clk) is
	begin
		if rising_edge(clk) then
			if rst = '1' then
				memStartReady <= '1';
				memEndValid <= '0';
				ctrlEndReady <= '0';
			else
				memStartReady <= (memEndValid and memEnd_ready_i) or ((not (memStart_valid_i and memStartReady)) and memStartReady);
				memEndValid <= TEMP_GEN_MEM or memEndValid;
				ctrlEndReady <= (not (ctrlEnd_valid_i and ctrlEndReady)) and (TEMP_GEN_MEM or ctrlEndReady);
			end if;
		end if;
	end process;

	-- Update new memory interfaces
	memStart_ready_o <= memStartReady;
	ctrlEnd_ready_o <= ctrlEndReady;
	memEnd_valid_o <= memEndValid;
	-- Bits To One-Hot Begin
	-- BitsToOH(ldq_head_oh, ldq_head)
	ldq_head_oh(0) <= '1' when ldq_head_q = "00" else '0';
	ldq_head_oh(1) <= '1' when ldq_head_q = "01" else '0';
	ldq_head_oh(2) <= '1' when ldq_head_q = "10" else '0';
	ldq_head_oh(3) <= '1' when ldq_head_q = "11" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(stq_head_oh, stq_head)
	stq_head_oh(0) <= '1' when stq_head_q = "0" else '0';
	stq_head_oh(1) <= '1' when stq_head_q = "1" else '0';
	-- Bits To One-Hot End

	ldq_alloc_next_0 <= not ldq_reset_0 and ldq_alloc_0_q;
	ldq_alloc_0_d <= ldq_wen_0 or ldq_alloc_next_0;
	ldq_issue_0_d <= not ldq_wen_0 and ( ldq_issue_set_0 or ldq_issue_0_q );
	ldq_addr_valid_0_d <= not ldq_wen_0 and ( ldq_addr_wen_0 or ldq_addr_valid_0_q );
	ldq_data_valid_0_d <= not ldq_wen_0 and ( ldq_data_wen_0 or ldq_data_valid_0_q );
	ldq_alloc_next_1 <= not ldq_reset_1 and ldq_alloc_1_q;
	ldq_alloc_1_d <= ldq_wen_1 or ldq_alloc_next_1;
	ldq_issue_1_d <= not ldq_wen_1 and ( ldq_issue_set_1 or ldq_issue_1_q );
	ldq_addr_valid_1_d <= not ldq_wen_1 and ( ldq_addr_wen_1 or ldq_addr_valid_1_q );
	ldq_data_valid_1_d <= not ldq_wen_1 and ( ldq_data_wen_1 or ldq_data_valid_1_q );
	ldq_alloc_next_2 <= not ldq_reset_2 and ldq_alloc_2_q;
	ldq_alloc_2_d <= ldq_wen_2 or ldq_alloc_next_2;
	ldq_issue_2_d <= not ldq_wen_2 and ( ldq_issue_set_2 or ldq_issue_2_q );
	ldq_addr_valid_2_d <= not ldq_wen_2 and ( ldq_addr_wen_2 or ldq_addr_valid_2_q );
	ldq_data_valid_2_d <= not ldq_wen_2 and ( ldq_data_wen_2 or ldq_data_valid_2_q );
	ldq_alloc_next_3 <= not ldq_reset_3 and ldq_alloc_3_q;
	ldq_alloc_3_d <= ldq_wen_3 or ldq_alloc_next_3;
	ldq_issue_3_d <= not ldq_wen_3 and ( ldq_issue_set_3 or ldq_issue_3_q );
	ldq_addr_valid_3_d <= not ldq_wen_3 and ( ldq_addr_wen_3 or ldq_addr_valid_3_q );
	ldq_data_valid_3_d <= not ldq_wen_3 and ( ldq_data_wen_3 or ldq_data_valid_3_q );
	stq_alloc_next_0 <= not stq_reset_0 and stq_alloc_0_q;
	stq_alloc_0_d <= stq_wen_0 or stq_alloc_next_0;
	stq_addr_valid_0_d <= not stq_wen_0 and ( stq_addr_wen_0 or stq_addr_valid_0_q );
	stq_data_valid_0_d <= not stq_wen_0 and ( stq_data_wen_0 or stq_data_valid_0_q );
	stq_alloc_next_1 <= not stq_reset_1 and stq_alloc_1_q;
	stq_alloc_1_d <= stq_wen_1 or stq_alloc_next_1;
	stq_addr_valid_1_d <= not stq_wen_1 and ( stq_addr_wen_1 or stq_addr_valid_1_q );
	stq_data_valid_1_d <= not stq_wen_1 and ( stq_data_wen_1 or stq_data_valid_1_q );
	store_is_older_0_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_0(0) ) ) when ldq_wen_0 else not stq_reset_0 and store_is_older_0_q(0);
	store_is_older_0_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_0(1) ) ) when ldq_wen_0 else not stq_reset_1 and store_is_older_0_q(1);
	store_is_older_1_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_1(0) ) ) when ldq_wen_1 else not stq_reset_0 and store_is_older_1_q(0);
	store_is_older_1_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_1(1) ) ) when ldq_wen_1 else not stq_reset_1 and store_is_older_1_q(1);
	store_is_older_2_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_2(0) ) ) when ldq_wen_2 else not stq_reset_0 and store_is_older_2_q(0);
	store_is_older_2_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_2(1) ) ) when ldq_wen_2 else not stq_reset_1 and store_is_older_2_q(1);
	store_is_older_3_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_3(0) ) ) when ldq_wen_3 else not stq_reset_0 and store_is_older_3_q(0);
	store_is_older_3_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_3(1) ) ) when ldq_wen_3 else not stq_reset_1 and store_is_older_3_q(1);
	-- Reduction Begin
	-- Reduce(ldq_not_empty, ldq_alloc, or)
	TEMP_1_res_0 <= ldq_alloc_0_q or ldq_alloc_2_q;
	TEMP_1_res_1 <= ldq_alloc_1_q or ldq_alloc_3_q;
	-- Layer End
	ldq_not_empty <= TEMP_1_res_0 or TEMP_1_res_1;
	-- Reduction End

	ldq_empty <= not ldq_not_empty;
	-- MuxLookUp Begin
	-- MuxLookUp(stq_not_empty, stq_alloc, stq_head)
	stq_not_empty <= 
	stq_alloc_0_q when (stq_head_q = "0") else
	stq_alloc_1_q when (stq_head_q = "1") else
	'0';
	-- MuxLookUp End

	stq_empty <= not stq_not_empty;
	empty_o <= ldq_empty and stq_empty;
	-- WrapAdd Begin
	-- WrapAdd(ldq_tail, ldq_tail, num_loads, 4)
	ldq_tail_d <= std_logic_vector(unsigned(ldq_tail_q) + unsigned(num_loads));
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_tail, stq_tail, num_stores, 2)
	stq_tail_d <= std_logic_vector(unsigned(stq_tail_q) + unsigned(num_stores));
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_issue, stq_issue, 1, 2)
	stq_issue_d <= std_logic_vector(unsigned(stq_issue_q) + 1);
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_resp, stq_resp, 1, 2)
	stq_resp_d <= std_logic_vector(unsigned(stq_resp_q) + 1);
	-- WrapAdd End

	-- Bits To One-Hot Begin
	-- BitsToOH(ldq_tail_oh, ldq_tail)
	ldq_tail_oh(0) <= '1' when ldq_tail_q = "00" else '0';
	ldq_tail_oh(1) <= '1' when ldq_tail_q = "01" else '0';
	ldq_tail_oh(2) <= '1' when ldq_tail_q = "10" else '0';
	ldq_tail_oh(3) <= '1' when ldq_tail_q = "11" else '0';
	-- Bits To One-Hot End

	-- Priority Masking Begin
	-- CyclicPriorityMask(ldq_head_next_oh, ldq_alloc_next, ldq_tail_oh)
	TEMP_2_double_in(0) <= ldq_alloc_next_0;
	TEMP_2_double_in(4) <= ldq_alloc_next_0;
	TEMP_2_double_in(1) <= ldq_alloc_next_1;
	TEMP_2_double_in(5) <= ldq_alloc_next_1;
	TEMP_2_double_in(2) <= ldq_alloc_next_2;
	TEMP_2_double_in(6) <= ldq_alloc_next_2;
	TEMP_2_double_in(3) <= ldq_alloc_next_3;
	TEMP_2_double_in(7) <= ldq_alloc_next_3;
	TEMP_2_double_out <= TEMP_2_double_in and not std_logic_vector( unsigned( TEMP_2_double_in ) - unsigned( "0000" & ldq_tail_oh ) );
	ldq_head_next_oh <= TEMP_2_double_out(3 downto 0) or TEMP_2_double_out(7 downto 4);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(ldq_head_sel, ldq_alloc_next, or)
	TEMP_3_res_0 <= ldq_alloc_next_0 or ldq_alloc_next_2;
	TEMP_3_res_1 <= ldq_alloc_next_1 or ldq_alloc_next_3;
	-- Layer End
	ldq_head_sel <= TEMP_3_res_0 or TEMP_3_res_1;
	-- Reduction End

	-- One-Hot To Bits Begin
	-- OHToBits(ldq_head_next, ldq_head_next_oh)
	TEMP_4_in_0_0 <= '0';
	TEMP_4_in_0_1 <= ldq_head_next_oh(1);
	TEMP_4_in_0_2 <= '0';
	TEMP_4_in_0_3 <= ldq_head_next_oh(3);
	TEMP_5_res_0 <= TEMP_4_in_0_0 or TEMP_4_in_0_2;
	TEMP_5_res_1 <= TEMP_4_in_0_1 or TEMP_4_in_0_3;
	-- Layer End
	TEMP_4_out_0 <= TEMP_5_res_0 or TEMP_5_res_1;
	ldq_head_next(0) <= TEMP_4_out_0;
	TEMP_5_in_1_0 <= '0';
	TEMP_5_in_1_1 <= '0';
	TEMP_5_in_1_2 <= ldq_head_next_oh(2);
	TEMP_5_in_1_3 <= ldq_head_next_oh(3);
	TEMP_6_res_0 <= TEMP_5_in_1_0 or TEMP_5_in_1_2;
	TEMP_6_res_1 <= TEMP_5_in_1_1 or TEMP_5_in_1_3;
	-- Layer End
	TEMP_5_out_1 <= TEMP_6_res_0 or TEMP_6_res_1;
	ldq_head_next(1) <= TEMP_5_out_1;
	-- One-Hot To Bits End

	ldq_head_d <= ldq_head_next when ldq_head_sel else ldq_tail_q;
	-- Bits To One-Hot Begin
	-- BitsToOH(stq_tail_oh, stq_tail)
	stq_tail_oh(0) <= '1' when stq_tail_q = "0" else '0';
	stq_tail_oh(1) <= '1' when stq_tail_q = "1" else '0';
	-- Bits To One-Hot End

	-- WrapAdd Begin
	-- WrapAdd(stq_head_next, stq_head, 1, 2)
	stq_head_next <= std_logic_vector(unsigned(stq_head_q) + 1);
	-- WrapAdd End

	stq_head_sel <= wresp_valid_0_i;
	stq_head_d <= stq_head_next when stq_head_sel else stq_head_q;
	handshake_lsq_lsq1_core_ga : entity work.handshake_lsq_lsq1_core_ga
		port map(
			rst => rst,
			clk => clk,
			group_init_valid_0_i => group_init_valid_0_i,
			group_init_ready_0_o => group_init_ready_0_o,
			ldq_tail_i => ldq_tail_q,
			ldq_head_i => ldq_head_q,
			ldq_empty_i => ldq_empty,
			stq_tail_i => stq_tail_q,
			stq_head_i => stq_head_q,
			stq_empty_i => stq_empty,
			ldq_wen_0_o => ldq_wen_0,
			ldq_wen_1_o => ldq_wen_1,
			ldq_wen_2_o => ldq_wen_2,
			ldq_wen_3_o => ldq_wen_3,
			num_loads_o => num_loads,
			ldq_port_idx_0_o => ldq_port_idx_0_d,
			ldq_port_idx_1_o => ldq_port_idx_1_d,
			ldq_port_idx_2_o => ldq_port_idx_2_d,
			ldq_port_idx_3_o => ldq_port_idx_3_d,
			stq_wen_0_o => stq_wen_0,
			stq_wen_1_o => stq_wen_1,
			ga_ls_order_0_o => ga_ls_order_0,
			ga_ls_order_1_o => ga_ls_order_1,
			ga_ls_order_2_o => ga_ls_order_2,
			ga_ls_order_3_o => ga_ls_order_3,
			num_stores_o => num_stores
		);
	handshake_lsq_lsq1_core_lda_dispatcher : entity work.handshake_lsq_lsq1_core_lda
		port map(
			rst => rst,
			clk => clk,
			port_payload_0_i => ldp_addr_0_i,
			port_payload_1_i => ldp_addr_1_i,
			port_ready_0_o => ldp_addr_ready_0_o,
			port_ready_1_o => ldp_addr_ready_1_o,
			port_valid_0_i => ldp_addr_valid_0_i,
			port_valid_1_i => ldp_addr_valid_1_i,
			entry_alloc_0_i => ldq_alloc_0_q,
			entry_alloc_1_i => ldq_alloc_1_q,
			entry_alloc_2_i => ldq_alloc_2_q,
			entry_alloc_3_i => ldq_alloc_3_q,
			entry_payload_valid_0_i => ldq_addr_valid_0_q,
			entry_payload_valid_1_i => ldq_addr_valid_1_q,
			entry_payload_valid_2_i => ldq_addr_valid_2_q,
			entry_payload_valid_3_i => ldq_addr_valid_3_q,
			entry_port_idx_0_i => ldq_port_idx_0_q,
			entry_port_idx_1_i => ldq_port_idx_1_q,
			entry_port_idx_2_i => ldq_port_idx_2_q,
			entry_port_idx_3_i => ldq_port_idx_3_q,
			entry_payload_0_o => ldq_addr_0_d,
			entry_payload_1_o => ldq_addr_1_d,
			entry_payload_2_o => ldq_addr_2_d,
			entry_payload_3_o => ldq_addr_3_d,
			entry_wen_0_o => ldq_addr_wen_0,
			entry_wen_1_o => ldq_addr_wen_1,
			entry_wen_2_o => ldq_addr_wen_2,
			entry_wen_3_o => ldq_addr_wen_3,
			queue_head_oh_i => ldq_head_oh
		);
	handshake_lsq_lsq1_core_ldd_dispatcher : entity work.handshake_lsq_lsq1_core_ldd
		port map(
			rst => rst,
			clk => clk,
			port_payload_0_o => ldp_data_0_o,
			port_payload_1_o => ldp_data_1_o,
			port_ready_0_i => ldp_data_ready_0_i,
			port_ready_1_i => ldp_data_ready_1_i,
			port_valid_0_o => ldp_data_valid_0_o,
			port_valid_1_o => ldp_data_valid_1_o,
			entry_alloc_0_i => ldq_alloc_0_q,
			entry_alloc_1_i => ldq_alloc_1_q,
			entry_alloc_2_i => ldq_alloc_2_q,
			entry_alloc_3_i => ldq_alloc_3_q,
			entry_payload_valid_0_i => ldq_data_valid_0_q,
			entry_payload_valid_1_i => ldq_data_valid_1_q,
			entry_payload_valid_2_i => ldq_data_valid_2_q,
			entry_payload_valid_3_i => ldq_data_valid_3_q,
			entry_port_idx_0_i => ldq_port_idx_0_q,
			entry_port_idx_1_i => ldq_port_idx_1_q,
			entry_port_idx_2_i => ldq_port_idx_2_q,
			entry_port_idx_3_i => ldq_port_idx_3_q,
			entry_payload_0_i => ldq_data_0_q,
			entry_payload_1_i => ldq_data_1_q,
			entry_payload_2_i => ldq_data_2_q,
			entry_payload_3_i => ldq_data_3_q,
			entry_reset_0_o => ldq_reset_0,
			entry_reset_1_o => ldq_reset_1,
			entry_reset_2_o => ldq_reset_2,
			entry_reset_3_o => ldq_reset_3,
			queue_head_oh_i => ldq_head_oh
		);
	handshake_lsq_lsq1_core_sta_dispatcher : entity work.handshake_lsq_lsq1_core_sta
		port map(
			rst => rst,
			clk => clk,
			port_payload_0_i => stp_addr_0_i,
			port_ready_0_o => stp_addr_ready_0_o,
			port_valid_0_i => stp_addr_valid_0_i,
			entry_alloc_0_i => stq_alloc_0_q,
			entry_alloc_1_i => stq_alloc_1_q,
			entry_payload_valid_0_i => stq_addr_valid_0_q,
			entry_payload_valid_1_i => stq_addr_valid_1_q,
			entry_payload_0_o => stq_addr_0_d,
			entry_payload_1_o => stq_addr_1_d,
			entry_wen_0_o => stq_addr_wen_0,
			entry_wen_1_o => stq_addr_wen_1,
			queue_head_oh_i => stq_head_oh
		);
	handshake_lsq_lsq1_core_std_dispatcher : entity work.handshake_lsq_lsq1_core_std
		port map(
			rst => rst,
			clk => clk,
			port_payload_0_i => stp_data_0_i,
			port_ready_0_o => stp_data_ready_0_o,
			port_valid_0_i => stp_data_valid_0_i,
			entry_alloc_0_i => stq_alloc_0_q,
			entry_alloc_1_i => stq_alloc_1_q,
			entry_payload_valid_0_i => stq_data_valid_0_q,
			entry_payload_valid_1_i => stq_data_valid_1_q,
			entry_payload_0_o => stq_data_0_d,
			entry_payload_1_o => stq_data_1_d,
			entry_wen_0_o => stq_data_wen_0,
			entry_wen_1_o => stq_data_wen_1,
			queue_head_oh_i => stq_head_oh
		);
	addr_valid_0(0) <= ldq_addr_valid_0_q and stq_addr_valid_0_q;
	addr_valid_0(1) <= ldq_addr_valid_0_q and stq_addr_valid_1_q;
	addr_valid_1(0) <= ldq_addr_valid_1_q and stq_addr_valid_0_q;
	addr_valid_1(1) <= ldq_addr_valid_1_q and stq_addr_valid_1_q;
	addr_valid_2(0) <= ldq_addr_valid_2_q and stq_addr_valid_0_q;
	addr_valid_2(1) <= ldq_addr_valid_2_q and stq_addr_valid_1_q;
	addr_valid_3(0) <= ldq_addr_valid_3_q and stq_addr_valid_0_q;
	addr_valid_3(1) <= ldq_addr_valid_3_q and stq_addr_valid_1_q;
	addr_same_0(0) <= '1' when ldq_addr_0_q = stq_addr_0_q else '0';
	addr_same_0(1) <= '1' when ldq_addr_0_q = stq_addr_1_q else '0';
	addr_same_1(0) <= '1' when ldq_addr_1_q = stq_addr_0_q else '0';
	addr_same_1(1) <= '1' when ldq_addr_1_q = stq_addr_1_q else '0';
	addr_same_2(0) <= '1' when ldq_addr_2_q = stq_addr_0_q else '0';
	addr_same_2(1) <= '1' when ldq_addr_2_q = stq_addr_1_q else '0';
	addr_same_3(0) <= '1' when ldq_addr_3_q = stq_addr_0_q else '0';
	addr_same_3(1) <= '1' when ldq_addr_3_q = stq_addr_1_q else '0';
	ld_st_conflict_0(0) <= stq_alloc_0_q and store_is_older_0_q(0) and ( addr_same_0(0) or not stq_addr_valid_0_q );
	ld_st_conflict_0(1) <= stq_alloc_1_q and store_is_older_0_q(1) and ( addr_same_0(1) or not stq_addr_valid_1_q );
	ld_st_conflict_1(0) <= stq_alloc_0_q and store_is_older_1_q(0) and ( addr_same_1(0) or not stq_addr_valid_0_q );
	ld_st_conflict_1(1) <= stq_alloc_1_q and store_is_older_1_q(1) and ( addr_same_1(1) or not stq_addr_valid_1_q );
	ld_st_conflict_2(0) <= stq_alloc_0_q and store_is_older_2_q(0) and ( addr_same_2(0) or not stq_addr_valid_0_q );
	ld_st_conflict_2(1) <= stq_alloc_1_q and store_is_older_2_q(1) and ( addr_same_2(1) or not stq_addr_valid_1_q );
	ld_st_conflict_3(0) <= stq_alloc_0_q and store_is_older_3_q(0) and ( addr_same_3(0) or not stq_addr_valid_0_q );
	ld_st_conflict_3(1) <= stq_alloc_1_q and store_is_older_3_q(1) and ( addr_same_3(1) or not stq_addr_valid_1_q );
	can_bypass_0(0) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_0_q and addr_same_0(0) and addr_valid_0(0);
	can_bypass_0(1) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_1_q and addr_same_0(1) and addr_valid_0(1);
	can_bypass_1(0) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_0_q and addr_same_1(0) and addr_valid_1(0);
	can_bypass_1(1) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_1_q and addr_same_1(1) and addr_valid_1(1);
	can_bypass_2(0) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_0_q and addr_same_2(0) and addr_valid_2(0);
	can_bypass_2(1) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_1_q and addr_same_2(1) and addr_valid_2(1);
	can_bypass_3(0) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_0_q and addr_same_3(0) and addr_valid_3(0);
	can_bypass_3(1) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_1_q and addr_same_3(1) and addr_valid_3(1);
	-- Reduction Begin
	-- Reduce(load_conflict_0, ld_st_conflict_0, or)
	load_conflict_0 <= ld_st_conflict_0(0) or ld_st_conflict_0(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_1, ld_st_conflict_1, or)
	load_conflict_1 <= ld_st_conflict_1(0) or ld_st_conflict_1(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_2, ld_st_conflict_2, or)
	load_conflict_2 <= ld_st_conflict_2(0) or ld_st_conflict_2(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_3, ld_st_conflict_3, or)
	load_conflict_3 <= ld_st_conflict_3(0) or ld_st_conflict_3(1);
	-- Reduction End

	load_req_valid_0 <= ldq_alloc_0_q and not ldq_issue_0_q and ldq_addr_valid_0_q;
	load_req_valid_1 <= ldq_alloc_1_q and not ldq_issue_1_q and ldq_addr_valid_1_q;
	load_req_valid_2 <= ldq_alloc_2_q and not ldq_issue_2_q and ldq_addr_valid_2_q;
	load_req_valid_3 <= ldq_alloc_3_q and not ldq_issue_3_q and ldq_addr_valid_3_q;
	can_load_0 <= not load_conflict_0 and load_req_valid_0;
	can_load_1 <= not load_conflict_1 and load_req_valid_1;
	can_load_2 <= not load_conflict_2 and load_req_valid_2;
	can_load_3 <= not load_conflict_3 and load_req_valid_3;
	-- Priority Masking Begin
	-- CyclicPriorityMask(load_idx_oh_0, can_load, ldq_head_oh)
	TEMP_7_double_in(0) <= can_load_0;
	TEMP_7_double_in(4) <= can_load_0;
	TEMP_7_double_in(1) <= can_load_1;
	TEMP_7_double_in(5) <= can_load_1;
	TEMP_7_double_in(2) <= can_load_2;
	TEMP_7_double_in(6) <= can_load_2;
	TEMP_7_double_in(3) <= can_load_3;
	TEMP_7_double_in(7) <= can_load_3;
	TEMP_7_double_out <= TEMP_7_double_in and not std_logic_vector( unsigned( TEMP_7_double_in ) - unsigned( "0000" & ldq_head_oh ) );
	load_idx_oh_0 <= TEMP_7_double_out(3 downto 0) or TEMP_7_double_out(7 downto 4);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(load_en_0, can_load, or)
	TEMP_8_res_0 <= can_load_0 or can_load_2;
	TEMP_8_res_1 <= can_load_1 or can_load_3;
	-- Layer End
	load_en_0 <= TEMP_8_res_0 or TEMP_8_res_1;
	-- Reduction End

	st_ld_conflict(0) <= ldq_alloc_0_q and not store_is_older_0_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_0(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_0_q );
	st_ld_conflict(1) <= ldq_alloc_1_q and not store_is_older_1_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_1(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_1_q );
	st_ld_conflict(2) <= ldq_alloc_2_q and not store_is_older_2_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_2(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_2_q );
	st_ld_conflict(3) <= ldq_alloc_3_q and not store_is_older_3_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_3(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_3_q );
	-- Reduction Begin
	-- Reduce(store_conflict, st_ld_conflict, or)
	TEMP_9_res(0) <= st_ld_conflict(0) or st_ld_conflict(2);
	TEMP_9_res(1) <= st_ld_conflict(1) or st_ld_conflict(3);
	-- Layer End
	store_conflict <= TEMP_9_res(0) or TEMP_9_res(1);
	-- Reduction End

	-- MuxLookUp Begin
	-- MuxLookUp(store_valid, stq_alloc, stq_issue)
	store_valid <= 
	stq_alloc_0_q when (stq_issue_q = "0") else
	stq_alloc_1_q when (stq_issue_q = "1") else
	'0';
	-- MuxLookUp End

	-- MuxLookUp Begin
	-- MuxLookUp(store_data_valid, stq_data_valid, stq_issue)
	store_data_valid <= 
	stq_data_valid_0_q when (stq_issue_q = "0") else
	stq_data_valid_1_q when (stq_issue_q = "1") else
	'0';
	-- MuxLookUp End

	-- MuxLookUp Begin
	-- MuxLookUp(store_addr_valid, stq_addr_valid, stq_issue)
	store_addr_valid <= 
	stq_addr_valid_0_q when (stq_issue_q = "0") else
	stq_addr_valid_1_q when (stq_issue_q = "1") else
	'0';
	-- MuxLookUp End

	store_en <= not store_conflict and store_valid and store_data_valid and store_addr_valid;
	store_idx <= stq_issue_q;
	-- Bits To One-Hot Begin
	-- BitsToOHSub1(stq_last_oh, stq_tail)
	stq_last_oh(0) <= '1' when stq_tail_q = "1" else '0';
	stq_last_oh(1) <= '1' when stq_tail_q = "0" else '0';
	-- Bits To One-Hot End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_0, ld_st_conflict_0, stq_last_oh)
	TEMP_10_double_in(0) <= ld_st_conflict_0(1);
	TEMP_10_double_in(2) <= ld_st_conflict_0(1);
	TEMP_10_double_in(1) <= ld_st_conflict_0(0);
	TEMP_10_double_in(3) <= ld_st_conflict_0(0);
	TEMP_10_base_rev(0) <= stq_last_oh(1);
	TEMP_10_base_rev(1) <= stq_last_oh(0);
	TEMP_10_double_out <= TEMP_10_double_in and not std_logic_vector( unsigned( TEMP_10_double_in ) - unsigned( "00" & TEMP_10_base_rev ) );
	bypass_idx_oh_0(1) <= TEMP_10_double_out(0) or TEMP_10_double_out(2);
	bypass_idx_oh_0(0) <= TEMP_10_double_out(1) or TEMP_10_double_out(3);
	-- Priority Masking End

	bypass_en_vec_0 <= bypass_idx_oh_0 and can_bypass_0;
	-- Reduction Begin
	-- Reduce(bypass_en_0, bypass_en_vec_0, or)
	bypass_en_0 <= bypass_en_vec_0(0) or bypass_en_vec_0(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_1, ld_st_conflict_1, stq_last_oh)
	TEMP_11_double_in(0) <= ld_st_conflict_1(1);
	TEMP_11_double_in(2) <= ld_st_conflict_1(1);
	TEMP_11_double_in(1) <= ld_st_conflict_1(0);
	TEMP_11_double_in(3) <= ld_st_conflict_1(0);
	TEMP_11_base_rev(0) <= stq_last_oh(1);
	TEMP_11_base_rev(1) <= stq_last_oh(0);
	TEMP_11_double_out <= TEMP_11_double_in and not std_logic_vector( unsigned( TEMP_11_double_in ) - unsigned( "00" & TEMP_11_base_rev ) );
	bypass_idx_oh_1(1) <= TEMP_11_double_out(0) or TEMP_11_double_out(2);
	bypass_idx_oh_1(0) <= TEMP_11_double_out(1) or TEMP_11_double_out(3);
	-- Priority Masking End

	bypass_en_vec_1 <= bypass_idx_oh_1 and can_bypass_1;
	-- Reduction Begin
	-- Reduce(bypass_en_1, bypass_en_vec_1, or)
	bypass_en_1 <= bypass_en_vec_1(0) or bypass_en_vec_1(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_2, ld_st_conflict_2, stq_last_oh)
	TEMP_12_double_in(0) <= ld_st_conflict_2(1);
	TEMP_12_double_in(2) <= ld_st_conflict_2(1);
	TEMP_12_double_in(1) <= ld_st_conflict_2(0);
	TEMP_12_double_in(3) <= ld_st_conflict_2(0);
	TEMP_12_base_rev(0) <= stq_last_oh(1);
	TEMP_12_base_rev(1) <= stq_last_oh(0);
	TEMP_12_double_out <= TEMP_12_double_in and not std_logic_vector( unsigned( TEMP_12_double_in ) - unsigned( "00" & TEMP_12_base_rev ) );
	bypass_idx_oh_2(1) <= TEMP_12_double_out(0) or TEMP_12_double_out(2);
	bypass_idx_oh_2(0) <= TEMP_12_double_out(1) or TEMP_12_double_out(3);
	-- Priority Masking End

	bypass_en_vec_2 <= bypass_idx_oh_2 and can_bypass_2;
	-- Reduction Begin
	-- Reduce(bypass_en_2, bypass_en_vec_2, or)
	bypass_en_2 <= bypass_en_vec_2(0) or bypass_en_vec_2(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_3, ld_st_conflict_3, stq_last_oh)
	TEMP_13_double_in(0) <= ld_st_conflict_3(1);
	TEMP_13_double_in(2) <= ld_st_conflict_3(1);
	TEMP_13_double_in(1) <= ld_st_conflict_3(0);
	TEMP_13_double_in(3) <= ld_st_conflict_3(0);
	TEMP_13_base_rev(0) <= stq_last_oh(1);
	TEMP_13_base_rev(1) <= stq_last_oh(0);
	TEMP_13_double_out <= TEMP_13_double_in and not std_logic_vector( unsigned( TEMP_13_double_in ) - unsigned( "00" & TEMP_13_base_rev ) );
	bypass_idx_oh_3(1) <= TEMP_13_double_out(0) or TEMP_13_double_out(2);
	bypass_idx_oh_3(0) <= TEMP_13_double_out(1) or TEMP_13_double_out(3);
	-- Priority Masking End

	bypass_en_vec_3 <= bypass_idx_oh_3 and can_bypass_3;
	-- Reduction Begin
	-- Reduce(bypass_en_3, bypass_en_vec_3, or)
	bypass_en_3 <= bypass_en_vec_3(0) or bypass_en_vec_3(1);
	-- Reduction End

	rreq_valid_0_o <= load_en_0;
	-- One-Hot To Bits Begin
	-- OHToBits(rreq_id_0, load_idx_oh_0)
	TEMP_14_in_0_0 <= '0';
	TEMP_14_in_0_1 <= load_idx_oh_0(1);
	TEMP_14_in_0_2 <= '0';
	TEMP_14_in_0_3 <= load_idx_oh_0(3);
	TEMP_15_res_0 <= TEMP_14_in_0_0 or TEMP_14_in_0_2;
	TEMP_15_res_1 <= TEMP_14_in_0_1 or TEMP_14_in_0_3;
	-- Layer End
	TEMP_14_out_0 <= TEMP_15_res_0 or TEMP_15_res_1;
	rreq_id_0_o(0) <= TEMP_14_out_0;
	TEMP_15_in_1_0 <= '0';
	TEMP_15_in_1_1 <= '0';
	TEMP_15_in_1_2 <= load_idx_oh_0(2);
	TEMP_15_in_1_3 <= load_idx_oh_0(3);
	TEMP_16_res_0 <= TEMP_15_in_1_0 or TEMP_15_in_1_2;
	TEMP_16_res_1 <= TEMP_15_in_1_1 or TEMP_15_in_1_3;
	-- Layer End
	TEMP_15_out_1 <= TEMP_16_res_0 or TEMP_16_res_1;
	rreq_id_0_o(1) <= TEMP_15_out_1;
	TEMP_16_in_2_0 <= '0';
	TEMP_16_in_2_1 <= '0';
	TEMP_16_in_2_2 <= '0';
	TEMP_16_in_2_3 <= '0';
	TEMP_17_res_0 <= TEMP_16_in_2_0 or TEMP_16_in_2_2;
	TEMP_17_res_1 <= TEMP_16_in_2_1 or TEMP_16_in_2_3;
	-- Layer End
	TEMP_16_out_2 <= TEMP_17_res_0 or TEMP_17_res_1;
	rreq_id_0_o(2) <= TEMP_16_out_2;
	TEMP_17_in_3_0 <= '0';
	TEMP_17_in_3_1 <= '0';
	TEMP_17_in_3_2 <= '0';
	TEMP_17_in_3_3 <= '0';
	TEMP_18_res_0 <= TEMP_17_in_3_0 or TEMP_17_in_3_2;
	TEMP_18_res_1 <= TEMP_17_in_3_1 or TEMP_17_in_3_3;
	-- Layer End
	TEMP_17_out_3 <= TEMP_18_res_0 or TEMP_18_res_1;
	rreq_id_0_o(3) <= TEMP_17_out_3;
	-- One-Hot To Bits End

	-- Mux1H Begin
	-- Mux1H(rreq_addr_0, ldq_addr, load_idx_oh_0)
	TEMP_19_mux_0 <= ldq_addr_0_q when load_idx_oh_0(0) = '1' else "000000000";
	TEMP_19_mux_1 <= ldq_addr_1_q when load_idx_oh_0(1) = '1' else "000000000";
	TEMP_19_mux_2 <= ldq_addr_2_q when load_idx_oh_0(2) = '1' else "000000000";
	TEMP_19_mux_3 <= ldq_addr_3_q when load_idx_oh_0(3) = '1' else "000000000";
	TEMP_20_res_0 <= TEMP_19_mux_0 or TEMP_19_mux_2;
	TEMP_20_res_1 <= TEMP_19_mux_1 or TEMP_19_mux_3;
	-- Layer End
	rreq_addr_0_o <= TEMP_20_res_0 or TEMP_20_res_1;
	-- Mux1H End

	ldq_issue_set_vec_0(0) <= ( load_idx_oh_0(0) and rreq_ready_0_i and load_en_0 ) or bypass_en_0;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_0, ldq_issue_set_vec_0, or)
	ldq_issue_set_0 <= ldq_issue_set_vec_0(0);
	-- Reduction End

	ldq_issue_set_vec_1(0) <= ( load_idx_oh_0(1) and rreq_ready_0_i and load_en_0 ) or bypass_en_1;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_1, ldq_issue_set_vec_1, or)
	ldq_issue_set_1 <= ldq_issue_set_vec_1(0);
	-- Reduction End

	ldq_issue_set_vec_2(0) <= ( load_idx_oh_0(2) and rreq_ready_0_i and load_en_0 ) or bypass_en_2;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_2, ldq_issue_set_vec_2, or)
	ldq_issue_set_2 <= ldq_issue_set_vec_2(0);
	-- Reduction End

	ldq_issue_set_vec_3(0) <= ( load_idx_oh_0(3) and rreq_ready_0_i and load_en_0 ) or bypass_en_3;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_3, ldq_issue_set_vec_3, or)
	ldq_issue_set_3 <= ldq_issue_set_vec_3(0);
	-- Reduction End

	wreq_valid_0_o <= store_en;
	wreq_id_0_o <= "0000";
	-- MuxLookUp Begin
	-- MuxLookUp(wreq_addr_0, stq_addr, store_idx)
	wreq_addr_0_o <= 
	stq_addr_0_q when (store_idx = "0") else
	stq_addr_1_q when (store_idx = "1") else
	"000000000";
	-- MuxLookUp End

	-- MuxLookUp Begin
	-- MuxLookUp(wreq_data_0, stq_data, store_idx)
	wreq_data_0_o <= 
	stq_data_0_q when (store_idx = "0") else
	stq_data_1_q when (store_idx = "1") else
	"00000000000000000000000000000000";
	-- MuxLookUp End

	stq_issue_en <= store_en and wreq_ready_0_i;
	read_idx_oh_0_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0000" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_0, rresp_data, read_idx_oh_0)
	TEMP_21_mux_0 <= rresp_data_0_i when read_idx_oh_0_0 = '1' else "00000000000000000000000000000000";
	read_data_0 <= TEMP_21_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_0, read_idx_oh_0, or)
	read_valid_0 <= read_idx_oh_0_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_0, stq_data, bypass_idx_oh_0)
	TEMP_22_mux_0 <= stq_data_0_q when bypass_idx_oh_0(0) = '1' else "00000000000000000000000000000000";
	TEMP_22_mux_1 <= stq_data_1_q when bypass_idx_oh_0(1) = '1' else "00000000000000000000000000000000";
	bypass_data_0 <= TEMP_22_mux_0 or TEMP_22_mux_1;
	-- Mux1H End

	ldq_data_0_d <= read_data_0 or bypass_data_0;
	ldq_data_wen_0 <= bypass_en_0 or read_valid_0;
	read_idx_oh_1_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0001" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_1, rresp_data, read_idx_oh_1)
	TEMP_23_mux_0 <= rresp_data_0_i when read_idx_oh_1_0 = '1' else "00000000000000000000000000000000";
	read_data_1 <= TEMP_23_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_1, read_idx_oh_1, or)
	read_valid_1 <= read_idx_oh_1_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_1, stq_data, bypass_idx_oh_1)
	TEMP_24_mux_0 <= stq_data_0_q when bypass_idx_oh_1(0) = '1' else "00000000000000000000000000000000";
	TEMP_24_mux_1 <= stq_data_1_q when bypass_idx_oh_1(1) = '1' else "00000000000000000000000000000000";
	bypass_data_1 <= TEMP_24_mux_0 or TEMP_24_mux_1;
	-- Mux1H End

	ldq_data_1_d <= read_data_1 or bypass_data_1;
	ldq_data_wen_1 <= bypass_en_1 or read_valid_1;
	read_idx_oh_2_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0010" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_2, rresp_data, read_idx_oh_2)
	TEMP_25_mux_0 <= rresp_data_0_i when read_idx_oh_2_0 = '1' else "00000000000000000000000000000000";
	read_data_2 <= TEMP_25_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_2, read_idx_oh_2, or)
	read_valid_2 <= read_idx_oh_2_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_2, stq_data, bypass_idx_oh_2)
	TEMP_26_mux_0 <= stq_data_0_q when bypass_idx_oh_2(0) = '1' else "00000000000000000000000000000000";
	TEMP_26_mux_1 <= stq_data_1_q when bypass_idx_oh_2(1) = '1' else "00000000000000000000000000000000";
	bypass_data_2 <= TEMP_26_mux_0 or TEMP_26_mux_1;
	-- Mux1H End

	ldq_data_2_d <= read_data_2 or bypass_data_2;
	ldq_data_wen_2 <= bypass_en_2 or read_valid_2;
	read_idx_oh_3_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0011" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_3, rresp_data, read_idx_oh_3)
	TEMP_27_mux_0 <= rresp_data_0_i when read_idx_oh_3_0 = '1' else "00000000000000000000000000000000";
	read_data_3 <= TEMP_27_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_3, read_idx_oh_3, or)
	read_valid_3 <= read_idx_oh_3_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_3, stq_data, bypass_idx_oh_3)
	TEMP_28_mux_0 <= stq_data_0_q when bypass_idx_oh_3(0) = '1' else "00000000000000000000000000000000";
	TEMP_28_mux_1 <= stq_data_1_q when bypass_idx_oh_3(1) = '1' else "00000000000000000000000000000000";
	bypass_data_3 <= TEMP_28_mux_0 or TEMP_28_mux_1;
	-- Mux1H End

	ldq_data_3_d <= read_data_3 or bypass_data_3;
	ldq_data_wen_3 <= bypass_en_3 or read_valid_3;
	rresp_ready_0_o <= '1';
	stq_reset_0 <= wresp_valid_0_i when ( stq_resp_q = "0" ) else '0';
	stq_reset_1 <= wresp_valid_0_i when ( stq_resp_q = "1" ) else '0';
	stq_resp_en <= wresp_valid_0_i;
	wresp_ready_0_o <= '1';

	process (clk, rst) is
	begin
		if (rst = '1') then
			ldq_alloc_0_q <= '0';
			ldq_alloc_1_q <= '0';
			ldq_alloc_2_q <= '0';
			ldq_alloc_3_q <= '0';
		elsif (rising_edge(clk)) then
			ldq_alloc_0_q <= ldq_alloc_0_d;
			ldq_alloc_1_q <= ldq_alloc_1_d;
			ldq_alloc_2_q <= ldq_alloc_2_d;
			ldq_alloc_3_q <= ldq_alloc_3_d;
		end if;
		if (rising_edge(clk)) then
			ldq_issue_0_q <= ldq_issue_0_d;
			ldq_issue_1_q <= ldq_issue_1_d;
			ldq_issue_2_q <= ldq_issue_2_d;
			ldq_issue_3_q <= ldq_issue_3_d;
		end if;
		if (rising_edge(clk)) then
			if (ldq_wen_0 = '1') then
				ldq_port_idx_0_q <= ldq_port_idx_0_d;
			end if;
			if (ldq_wen_1 = '1') then
				ldq_port_idx_1_q <= ldq_port_idx_1_d;
			end if;
			if (ldq_wen_2 = '1') then
				ldq_port_idx_2_q <= ldq_port_idx_2_d;
			end if;
			if (ldq_wen_3 = '1') then
				ldq_port_idx_3_q <= ldq_port_idx_3_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			ldq_addr_valid_0_q <= ldq_addr_valid_0_d;
			ldq_addr_valid_1_q <= ldq_addr_valid_1_d;
			ldq_addr_valid_2_q <= ldq_addr_valid_2_d;
			ldq_addr_valid_3_q <= ldq_addr_valid_3_d;
		end if;
		if (rising_edge(clk)) then
			if (ldq_addr_wen_0 = '1') then
				ldq_addr_0_q <= ldq_addr_0_d;
			end if;
			if (ldq_addr_wen_1 = '1') then
				ldq_addr_1_q <= ldq_addr_1_d;
			end if;
			if (ldq_addr_wen_2 = '1') then
				ldq_addr_2_q <= ldq_addr_2_d;
			end if;
			if (ldq_addr_wen_3 = '1') then
				ldq_addr_3_q <= ldq_addr_3_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			ldq_data_valid_0_q <= ldq_data_valid_0_d;
			ldq_data_valid_1_q <= ldq_data_valid_1_d;
			ldq_data_valid_2_q <= ldq_data_valid_2_d;
			ldq_data_valid_3_q <= ldq_data_valid_3_d;
		end if;
		if (rising_edge(clk)) then
			if (ldq_data_wen_0 = '1') then
				ldq_data_0_q <= ldq_data_0_d;
			end if;
			if (ldq_data_wen_1 = '1') then
				ldq_data_1_q <= ldq_data_1_d;
			end if;
			if (ldq_data_wen_2 = '1') then
				ldq_data_2_q <= ldq_data_2_d;
			end if;
			if (ldq_data_wen_3 = '1') then
				ldq_data_3_q <= ldq_data_3_d;
			end if;
		end if;
		if (rst = '1') then
			stq_alloc_0_q <= '0';
			stq_alloc_1_q <= '0';
		elsif (rising_edge(clk)) then
			stq_alloc_0_q <= stq_alloc_0_d;
			stq_alloc_1_q <= stq_alloc_1_d;
		end if;
		if (rising_edge(clk)) then
			stq_addr_valid_0_q <= stq_addr_valid_0_d;
			stq_addr_valid_1_q <= stq_addr_valid_1_d;
		end if;
		if (rising_edge(clk)) then
			if (stq_addr_wen_0 = '1') then
				stq_addr_0_q <= stq_addr_0_d;
			end if;
			if (stq_addr_wen_1 = '1') then
				stq_addr_1_q <= stq_addr_1_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			stq_data_valid_0_q <= stq_data_valid_0_d;
			stq_data_valid_1_q <= stq_data_valid_1_d;
		end if;
		if (rising_edge(clk)) then
			if (stq_data_wen_0 = '1') then
				stq_data_0_q <= stq_data_0_d;
			end if;
			if (stq_data_wen_1 = '1') then
				stq_data_1_q <= stq_data_1_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			store_is_older_0_q <= store_is_older_0_d;
			store_is_older_1_q <= store_is_older_1_d;
			store_is_older_2_q <= store_is_older_2_d;
			store_is_older_3_q <= store_is_older_3_d;
		end if;
		if (rst = '1') then
			ldq_tail_q <= "00";
		elsif (rising_edge(clk)) then
			ldq_tail_q <= ldq_tail_d;
		end if;
		if (rst = '1') then
			ldq_head_q <= "00";
		elsif (rising_edge(clk)) then
			ldq_head_q <= ldq_head_d;
		end if;
		if (rst = '1') then
			stq_tail_q <= "0";
		elsif (rising_edge(clk)) then
			stq_tail_q <= stq_tail_d;
		end if;
		if (rst = '1') then
			stq_head_q <= "0";
		elsif (rising_edge(clk)) then
			stq_head_q <= stq_head_d;
		end if;
		if (rst = '1') then
			stq_issue_q <= "0";
		elsif (rising_edge(clk)) then
			if (stq_issue_en = '1') then
				stq_issue_q <= stq_issue_d;
			end if;
		end if;
		if (rst = '1') then
			stq_resp_q <= "0";
		elsif (rising_edge(clk)) then
			if (stq_resp_en = '1') then
				stq_resp_q <= stq_resp_d;
			end if;
		end if;
	end process;
end architecture;
