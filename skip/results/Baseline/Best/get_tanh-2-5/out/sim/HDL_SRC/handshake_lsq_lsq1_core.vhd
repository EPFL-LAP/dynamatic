

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1_core_ga is
	port(
		rst : in std_logic;
		clk : in std_logic;
		group_init_valid_0_i : in std_logic;
		group_init_valid_1_i : in std_logic;
		group_init_ready_0_o : out std_logic;
		group_init_ready_1_o : out std_logic;
		ldq_tail_i : in std_logic_vector(0 downto 0);
		ldq_head_i : in std_logic_vector(0 downto 0);
		ldq_empty_i : in std_logic;
		stq_tail_i : in std_logic_vector(2 downto 0);
		stq_head_i : in std_logic_vector(2 downto 0);
		stq_empty_i : in std_logic;
		ldq_wen_0_o : out std_logic;
		ldq_wen_1_o : out std_logic;
		num_loads_o : out std_logic_vector(0 downto 0);
		stq_wen_0_o : out std_logic;
		stq_wen_1_o : out std_logic;
		stq_wen_2_o : out std_logic;
		stq_wen_3_o : out std_logic;
		stq_wen_4_o : out std_logic;
		num_stores_o : out std_logic_vector(2 downto 0);
		ga_ls_order_0_o : out std_logic_vector(4 downto 0);
		ga_ls_order_1_o : out std_logic_vector(4 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_ga is
	signal num_loads : std_logic_vector(0 downto 0);
	signal num_stores : std_logic_vector(2 downto 0);
	signal loads_sub : std_logic_vector(0 downto 0);
	signal stores_sub : std_logic_vector(2 downto 0);
	signal empty_loads : std_logic_vector(1 downto 0);
	signal empty_stores : std_logic_vector(2 downto 0);
	signal group_init_ready_0 : std_logic;
	signal group_init_ready_1 : std_logic;
	signal group_init_hs_0 : std_logic;
	signal group_init_hs_1 : std_logic;
	signal ga_ls_order_rom_0 : std_logic_vector(4 downto 0);
	signal ga_ls_order_rom_1 : std_logic_vector(4 downto 0);
	signal ga_ls_order_temp_0 : std_logic_vector(4 downto 0);
	signal ga_ls_order_temp_1 : std_logic_vector(4 downto 0);
	signal TEMP_1_mux_0_0 : std_logic_vector(4 downto 0);
	signal TEMP_1_mux_0_1 : std_logic_vector(4 downto 0);
	signal TEMP_1_mux_1_0 : std_logic_vector(4 downto 0);
	signal TEMP_1_mux_1_1 : std_logic_vector(4 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(0 downto 0);
	signal TEMP_2_mux_1 : std_logic_vector(0 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(2 downto 0);
	signal TEMP_3_mux_1 : std_logic_vector(2 downto 0);
	signal ldq_wen_unshifted_0 : std_logic;
	signal ldq_wen_unshifted_1 : std_logic;
	signal stq_wen_unshifted_0 : std_logic;
	signal stq_wen_unshifted_1 : std_logic;
	signal stq_wen_unshifted_2 : std_logic;
	signal stq_wen_unshifted_3 : std_logic;
	signal stq_wen_unshifted_4 : std_logic;
	signal TEMP_4_res_0 : std_logic;
	signal TEMP_4_res_1 : std_logic;
	signal TEMP_4_res_2 : std_logic;
	signal TEMP_4_res_3 : std_logic;
	signal TEMP_4_res_4 : std_logic;
	signal TEMP_5_res_0 : std_logic;
	signal TEMP_5_res_1 : std_logic;
	signal TEMP_5_res_2 : std_logic;
	signal TEMP_5_res_3 : std_logic;
	signal TEMP_5_res_4 : std_logic;
	signal TEMP_6_res : std_logic_vector(4 downto 0);
	signal TEMP_7_res : std_logic_vector(4 downto 0);
	signal TEMP_8_res : std_logic_vector(4 downto 0);
	signal TEMP_9_res : std_logic_vector(4 downto 0);
begin
	-- WrapSub Begin
	-- WrapSub(loads_sub, ldq_head, ldq_tail, 2)
	loads_sub <= std_logic_vector(unsigned(ldq_head_i) - unsigned(ldq_tail_i));
	-- WrapAdd End

	-- WrapSub Begin
	-- WrapSub(stores_sub, stq_head, stq_tail, 5)
	stores_sub <= std_logic_vector(unsigned(stq_head_i) - unsigned(stq_tail_i)) when stq_head_i >= stq_tail_i else
		std_logic_vector(5 - unsigned(stq_tail_i) + unsigned(stq_head_i));
	-- WrapAdd End

	empty_loads <= "10" when ldq_empty_i else ( '0' & loads_sub );
	empty_stores <= "101" when stq_empty_i else stores_sub;
	group_init_ready_0 <= '1' when ( empty_loads >= "01" ) and ( empty_stores >= "000" ) else '0';
	group_init_ready_1 <= '1' when ( empty_loads >= "00" ) and ( empty_stores >= "001" ) else '0';
	group_init_ready_0_o <= group_init_ready_0;
	group_init_hs_0 <= group_init_ready_0 and group_init_valid_0_i;
	group_init_ready_1_o <= group_init_ready_1;
	group_init_hs_1 <= group_init_ready_1 and group_init_valid_1_i;
	-- Mux1H For Rom Begin
	-- Mux1H(ga_ls_order_rom, group_init_hs)
	-- Loop 0
	TEMP_1_mux_0_0 <= "00000";
	TEMP_1_mux_0_1 <= "00000";
	ga_ls_order_rom_0 <= TEMP_1_mux_0_0 or TEMP_1_mux_0_1;
	-- Loop 1
	TEMP_1_mux_1_0 <= "00000";
	TEMP_1_mux_1_1 <= "00000";
	ga_ls_order_rom_1 <= TEMP_1_mux_1_0 or TEMP_1_mux_1_1;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(num_loads, group_init_hs)
	TEMP_2_mux_0 <= "1" when group_init_hs_0 else "0";
	TEMP_2_mux_1 <= "0";
	num_loads <= TEMP_2_mux_0 or TEMP_2_mux_1;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(num_stores, group_init_hs)
	TEMP_3_mux_0 <= "000";
	TEMP_3_mux_1 <= "001" when group_init_hs_1 else "000";
	num_stores <= TEMP_3_mux_0 or TEMP_3_mux_1;
	-- Mux1H For Rom End

	num_loads_o <= num_loads;
	num_stores_o <= num_stores;
	ldq_wen_unshifted_0 <= '1' when num_loads > "0" else '0';
	ldq_wen_unshifted_1 <= '1' when num_loads > "1" else '0';
	stq_wen_unshifted_0 <= '1' when num_stores > "000" else '0';
	stq_wen_unshifted_1 <= '1' when num_stores > "001" else '0';
	stq_wen_unshifted_2 <= '1' when num_stores > "010" else '0';
	stq_wen_unshifted_3 <= '1' when num_stores > "011" else '0';
	stq_wen_unshifted_4 <= '1' when num_stores > "100" else '0';
	-- Shifter Begin
	-- CyclicLeftShift(ldq_wen, ldq_wen_unshifted, ldq_tail)
	ldq_wen_0_o <= ldq_wen_unshifted_1 when ldq_tail_i(0) else ldq_wen_unshifted_0;
	ldq_wen_1_o <= ldq_wen_unshifted_0 when ldq_tail_i(0) else ldq_wen_unshifted_1;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(stq_wen, stq_wen_unshifted, stq_tail)
	TEMP_4_res_0 <= stq_wen_unshifted_1 when stq_tail_i(2) else stq_wen_unshifted_0;
	TEMP_4_res_1 <= stq_wen_unshifted_2 when stq_tail_i(2) else stq_wen_unshifted_1;
	TEMP_4_res_2 <= stq_wen_unshifted_3 when stq_tail_i(2) else stq_wen_unshifted_2;
	TEMP_4_res_3 <= stq_wen_unshifted_4 when stq_tail_i(2) else stq_wen_unshifted_3;
	TEMP_4_res_4 <= stq_wen_unshifted_0 when stq_tail_i(2) else stq_wen_unshifted_4;
	-- Layer End
	TEMP_5_res_0 <= TEMP_4_res_3 when stq_tail_i(1) else TEMP_4_res_0;
	TEMP_5_res_1 <= TEMP_4_res_4 when stq_tail_i(1) else TEMP_4_res_1;
	TEMP_5_res_2 <= TEMP_4_res_0 when stq_tail_i(1) else TEMP_4_res_2;
	TEMP_5_res_3 <= TEMP_4_res_1 when stq_tail_i(1) else TEMP_4_res_3;
	TEMP_5_res_4 <= TEMP_4_res_2 when stq_tail_i(1) else TEMP_4_res_4;
	-- Layer End
	stq_wen_0_o <= TEMP_5_res_4 when stq_tail_i(0) else TEMP_5_res_0;
	stq_wen_1_o <= TEMP_5_res_0 when stq_tail_i(0) else TEMP_5_res_1;
	stq_wen_2_o <= TEMP_5_res_1 when stq_tail_i(0) else TEMP_5_res_2;
	stq_wen_3_o <= TEMP_5_res_2 when stq_tail_i(0) else TEMP_5_res_3;
	stq_wen_4_o <= TEMP_5_res_3 when stq_tail_i(0) else TEMP_5_res_4;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_0, ga_ls_order_rom_0, stq_tail)
	TEMP_6_res(0) <= ga_ls_order_rom_0(1) when stq_tail_i(2) else ga_ls_order_rom_0(0);
	TEMP_6_res(1) <= ga_ls_order_rom_0(2) when stq_tail_i(2) else ga_ls_order_rom_0(1);
	TEMP_6_res(2) <= ga_ls_order_rom_0(3) when stq_tail_i(2) else ga_ls_order_rom_0(2);
	TEMP_6_res(3) <= ga_ls_order_rom_0(4) when stq_tail_i(2) else ga_ls_order_rom_0(3);
	TEMP_6_res(4) <= ga_ls_order_rom_0(0) when stq_tail_i(2) else ga_ls_order_rom_0(4);
	-- Layer End
	TEMP_7_res(0) <= TEMP_6_res(3) when stq_tail_i(1) else TEMP_6_res(0);
	TEMP_7_res(1) <= TEMP_6_res(4) when stq_tail_i(1) else TEMP_6_res(1);
	TEMP_7_res(2) <= TEMP_6_res(0) when stq_tail_i(1) else TEMP_6_res(2);
	TEMP_7_res(3) <= TEMP_6_res(1) when stq_tail_i(1) else TEMP_6_res(3);
	TEMP_7_res(4) <= TEMP_6_res(2) when stq_tail_i(1) else TEMP_6_res(4);
	-- Layer End
	ga_ls_order_temp_0(0) <= TEMP_7_res(4) when stq_tail_i(0) else TEMP_7_res(0);
	ga_ls_order_temp_0(1) <= TEMP_7_res(0) when stq_tail_i(0) else TEMP_7_res(1);
	ga_ls_order_temp_0(2) <= TEMP_7_res(1) when stq_tail_i(0) else TEMP_7_res(2);
	ga_ls_order_temp_0(3) <= TEMP_7_res(2) when stq_tail_i(0) else TEMP_7_res(3);
	ga_ls_order_temp_0(4) <= TEMP_7_res(3) when stq_tail_i(0) else TEMP_7_res(4);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_1, ga_ls_order_rom_1, stq_tail)
	TEMP_8_res(0) <= ga_ls_order_rom_1(1) when stq_tail_i(2) else ga_ls_order_rom_1(0);
	TEMP_8_res(1) <= ga_ls_order_rom_1(2) when stq_tail_i(2) else ga_ls_order_rom_1(1);
	TEMP_8_res(2) <= ga_ls_order_rom_1(3) when stq_tail_i(2) else ga_ls_order_rom_1(2);
	TEMP_8_res(3) <= ga_ls_order_rom_1(4) when stq_tail_i(2) else ga_ls_order_rom_1(3);
	TEMP_8_res(4) <= ga_ls_order_rom_1(0) when stq_tail_i(2) else ga_ls_order_rom_1(4);
	-- Layer End
	TEMP_9_res(0) <= TEMP_8_res(3) when stq_tail_i(1) else TEMP_8_res(0);
	TEMP_9_res(1) <= TEMP_8_res(4) when stq_tail_i(1) else TEMP_8_res(1);
	TEMP_9_res(2) <= TEMP_8_res(0) when stq_tail_i(1) else TEMP_8_res(2);
	TEMP_9_res(3) <= TEMP_8_res(1) when stq_tail_i(1) else TEMP_8_res(3);
	TEMP_9_res(4) <= TEMP_8_res(2) when stq_tail_i(1) else TEMP_8_res(4);
	-- Layer End
	ga_ls_order_temp_1(0) <= TEMP_9_res(4) when stq_tail_i(0) else TEMP_9_res(0);
	ga_ls_order_temp_1(1) <= TEMP_9_res(0) when stq_tail_i(0) else TEMP_9_res(1);
	ga_ls_order_temp_1(2) <= TEMP_9_res(1) when stq_tail_i(0) else TEMP_9_res(2);
	ga_ls_order_temp_1(3) <= TEMP_9_res(2) when stq_tail_i(0) else TEMP_9_res(3);
	ga_ls_order_temp_1(4) <= TEMP_9_res(3) when stq_tail_i(0) else TEMP_9_res(4);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order, ga_ls_order_temp, ldq_tail)
	ga_ls_order_0_o <= ga_ls_order_temp_1 when ldq_tail_i(0) else ga_ls_order_temp_0;
	ga_ls_order_1_o <= ga_ls_order_temp_0 when ldq_tail_i(0) else ga_ls_order_temp_1;
	-- Shifter End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1_core_lda is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_payload_0_i : in std_logic_vector(9 downto 0);
		port_valid_0_i : in std_logic;
		port_ready_0_o : out std_logic;
		entry_alloc_0_i : in std_logic;
		entry_alloc_1_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_0_o : out std_logic_vector(9 downto 0);
		entry_payload_1_o : out std_logic_vector(9 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(1 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_lda is
	signal entry_port_idx_oh_0 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(9 downto 0);
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
	TEMP_1_mux_0 <= port_payload_0_i when entry_port_idx_oh_0(0) = '1' else "0000000000";
	entry_payload_0_o <= TEMP_1_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_1, port_payload, entry_port_idx_oh_1)
	TEMP_2_mux_0 <= port_payload_0_i when entry_port_idx_oh_1(0) = '1' else "0000000000";
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

entity handshake_lsq_lsq1_core_ldd is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_payload_0_o : out std_logic_vector(31 downto 0);
		port_valid_0_o : out std_logic;
		port_ready_0_i : in std_logic;
		entry_alloc_0_i : in std_logic;
		entry_alloc_1_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_0_i : in std_logic_vector(31 downto 0);
		entry_payload_1_i : in std_logic_vector(31 downto 0);
		entry_reset_0_o : out std_logic;
		entry_reset_1_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(1 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_ldd is
	signal entry_port_idx_oh_0 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_0 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_1 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_0 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_1 : std_logic_vector(0 downto 0);
	signal TEMP_1_double_in_0 : std_logic_vector(3 downto 0);
	signal TEMP_1_double_out_0 : std_logic_vector(3 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_1 : std_logic_vector(31 downto 0);
	signal entry_waiting_for_port_valid_0 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_1 : std_logic_vector(0 downto 0);
	signal port_valid_vec : std_logic_vector(0 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(0 downto 0);
begin
	entry_port_idx_oh_0 <= "1";
	entry_port_idx_oh_1 <= "1";
	entry_allocated_for_port_0 <= entry_port_idx_oh_0 when entry_alloc_0_i else "0";
	entry_allocated_for_port_1 <= entry_port_idx_oh_1 when entry_alloc_1_i else "0";
	-- Priority Masking Begin
	-- CyclicPriorityMask(oldest_entry_allocated_per_port, entry_allocated_for_port, queue_head_oh)
	TEMP_1_double_in_0(0) <= entry_allocated_for_port_0(0);
	TEMP_1_double_in_0(2) <= entry_allocated_for_port_0(0);
	TEMP_1_double_in_0(1) <= entry_allocated_for_port_1(0);
	TEMP_1_double_in_0(3) <= entry_allocated_for_port_1(0);
	TEMP_1_double_out_0 <= TEMP_1_double_in_0 and not std_logic_vector( unsigned( TEMP_1_double_in_0 ) - unsigned( "00" & queue_head_oh_i ) );
	oldest_entry_allocated_per_port_0(0) <= TEMP_1_double_out_0(0) or TEMP_1_double_out_0(2);
	oldest_entry_allocated_per_port_1(0) <= TEMP_1_double_out_0(1) or TEMP_1_double_out_0(3);
	-- Priority Masking End

	-- Mux1H Begin
	-- Mux1H(port_payload_0, entry_payload, oldest_entry_allocated_per_port)
	TEMP_2_mux_0 <= entry_payload_0_i when oldest_entry_allocated_per_port_0(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_1 <= entry_payload_1_i when oldest_entry_allocated_per_port_1(0) = '1' else "00000000000000000000000000000000";
	port_payload_0_o <= TEMP_2_mux_0 or TEMP_2_mux_1;
	-- Mux1H End

	entry_waiting_for_port_valid_0 <= oldest_entry_allocated_per_port_0 when entry_payload_valid_0_i else "0";
	entry_waiting_for_port_valid_1 <= oldest_entry_allocated_per_port_1 when entry_payload_valid_1_i else "0";
	-- Reduction Begin
	-- Reduce(port_valid_vec, entry_waiting_for_port_valid, or)
	port_valid_vec <= entry_waiting_for_port_valid_0 or entry_waiting_for_port_valid_1;
	-- Reduction End

	port_valid_0_o <= port_valid_vec(0);
	entry_port_transfer_0(0) <= entry_waiting_for_port_valid_0(0) and port_ready_0_i;
	entry_port_transfer_1(0) <= entry_waiting_for_port_valid_1(0) and port_ready_0_i;
	-- Reduction Begin
	-- Reduce(entry_reset_0, entry_port_transfer_0, or)
	entry_reset_0_o <= entry_port_transfer_0(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_1, entry_port_transfer_1, or)
	entry_reset_1_o <= entry_port_transfer_1(0);
	-- Reduction End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1_core_sta is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_payload_0_i : in std_logic_vector(9 downto 0);
		port_valid_0_i : in std_logic;
		port_ready_0_o : out std_logic;
		entry_alloc_0_i : in std_logic;
		entry_alloc_1_i : in std_logic;
		entry_alloc_2_i : in std_logic;
		entry_alloc_3_i : in std_logic;
		entry_alloc_4_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_valid_2_i : in std_logic;
		entry_payload_valid_3_i : in std_logic;
		entry_payload_valid_4_i : in std_logic;
		entry_payload_0_o : out std_logic_vector(9 downto 0);
		entry_payload_1_o : out std_logic_vector(9 downto 0);
		entry_payload_2_o : out std_logic_vector(9 downto 0);
		entry_payload_3_o : out std_logic_vector(9 downto 0);
		entry_payload_4_o : out std_logic_vector(9 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		entry_wen_2_o : out std_logic;
		entry_wen_3_o : out std_logic;
		entry_wen_4_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(4 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_sta is
	signal entry_port_idx_oh_0 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_2 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_3 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_4 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_4_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_5_mux_0 : std_logic_vector(9 downto 0);
	signal entry_ptq_ready_0 : std_logic;
	signal entry_ptq_ready_1 : std_logic;
	signal entry_ptq_ready_2 : std_logic;
	signal entry_ptq_ready_3 : std_logic;
	signal entry_ptq_ready_4 : std_logic;
	signal entry_waiting_for_port_0 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_1 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_2 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_3 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_4 : std_logic_vector(0 downto 0);
	signal port_ready_vec : std_logic_vector(0 downto 0);
	signal TEMP_6_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_7_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_7_res_1 : std_logic_vector(0 downto 0);
	signal entry_port_options_0 : std_logic_vector(0 downto 0);
	signal entry_port_options_1 : std_logic_vector(0 downto 0);
	signal entry_port_options_2 : std_logic_vector(0 downto 0);
	signal entry_port_options_3 : std_logic_vector(0 downto 0);
	signal entry_port_options_4 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_2 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_3 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_4 : std_logic_vector(0 downto 0);
	signal TEMP_8_double_in_0 : std_logic_vector(9 downto 0);
	signal TEMP_8_double_out_0 : std_logic_vector(9 downto 0);
begin
	entry_port_idx_oh_0 <= "1";
	entry_port_idx_oh_1 <= "1";
	entry_port_idx_oh_2 <= "1";
	entry_port_idx_oh_3 <= "1";
	entry_port_idx_oh_4 <= "1";
	-- Mux1H Begin
	-- Mux1H(entry_payload_0, port_payload, entry_port_idx_oh_0)
	TEMP_1_mux_0 <= port_payload_0_i when entry_port_idx_oh_0(0) = '1' else "0000000000";
	entry_payload_0_o <= TEMP_1_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_1, port_payload, entry_port_idx_oh_1)
	TEMP_2_mux_0 <= port_payload_0_i when entry_port_idx_oh_1(0) = '1' else "0000000000";
	entry_payload_1_o <= TEMP_2_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_2, port_payload, entry_port_idx_oh_2)
	TEMP_3_mux_0 <= port_payload_0_i when entry_port_idx_oh_2(0) = '1' else "0000000000";
	entry_payload_2_o <= TEMP_3_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_3, port_payload, entry_port_idx_oh_3)
	TEMP_4_mux_0 <= port_payload_0_i when entry_port_idx_oh_3(0) = '1' else "0000000000";
	entry_payload_3_o <= TEMP_4_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_4, port_payload, entry_port_idx_oh_4)
	TEMP_5_mux_0 <= port_payload_0_i when entry_port_idx_oh_4(0) = '1' else "0000000000";
	entry_payload_4_o <= TEMP_5_mux_0;
	-- Mux1H End

	entry_ptq_ready_0 <= entry_alloc_0_i and not entry_payload_valid_0_i;
	entry_ptq_ready_1 <= entry_alloc_1_i and not entry_payload_valid_1_i;
	entry_ptq_ready_2 <= entry_alloc_2_i and not entry_payload_valid_2_i;
	entry_ptq_ready_3 <= entry_alloc_3_i and not entry_payload_valid_3_i;
	entry_ptq_ready_4 <= entry_alloc_4_i and not entry_payload_valid_4_i;
	entry_waiting_for_port_0 <= entry_port_idx_oh_0 when entry_ptq_ready_0 else "0";
	entry_waiting_for_port_1 <= entry_port_idx_oh_1 when entry_ptq_ready_1 else "0";
	entry_waiting_for_port_2 <= entry_port_idx_oh_2 when entry_ptq_ready_2 else "0";
	entry_waiting_for_port_3 <= entry_port_idx_oh_3 when entry_ptq_ready_3 else "0";
	entry_waiting_for_port_4 <= entry_port_idx_oh_4 when entry_ptq_ready_4 else "0";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_waiting_for_port, or)
	TEMP_6_res_0 <= entry_waiting_for_port_0 or entry_waiting_for_port_4;
	TEMP_6_res_1 <= entry_waiting_for_port_1;
	TEMP_6_res_2 <= entry_waiting_for_port_2;
	TEMP_6_res_3 <= entry_waiting_for_port_3;
	-- Layer End
	TEMP_7_res_0 <= TEMP_6_res_0 or TEMP_6_res_2;
	TEMP_7_res_1 <= TEMP_6_res_1 or TEMP_6_res_3;
	-- Layer End
	port_ready_vec <= TEMP_7_res_0 or TEMP_7_res_1;
	-- Reduction End

	port_ready_0_o <= port_ready_vec(0);
	entry_port_options_0(0) <= entry_waiting_for_port_0(0) and port_valid_0_i;
	entry_port_options_1(0) <= entry_waiting_for_port_1(0) and port_valid_0_i;
	entry_port_options_2(0) <= entry_waiting_for_port_2(0) and port_valid_0_i;
	entry_port_options_3(0) <= entry_waiting_for_port_3(0) and port_valid_0_i;
	entry_port_options_4(0) <= entry_waiting_for_port_4(0) and port_valid_0_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_transfer, entry_port_options, queue_head_oh)
	TEMP_8_double_in_0(0) <= entry_port_options_0(0);
	TEMP_8_double_in_0(5) <= entry_port_options_0(0);
	TEMP_8_double_in_0(1) <= entry_port_options_1(0);
	TEMP_8_double_in_0(6) <= entry_port_options_1(0);
	TEMP_8_double_in_0(2) <= entry_port_options_2(0);
	TEMP_8_double_in_0(7) <= entry_port_options_2(0);
	TEMP_8_double_in_0(3) <= entry_port_options_3(0);
	TEMP_8_double_in_0(8) <= entry_port_options_3(0);
	TEMP_8_double_in_0(4) <= entry_port_options_4(0);
	TEMP_8_double_in_0(9) <= entry_port_options_4(0);
	TEMP_8_double_out_0 <= TEMP_8_double_in_0 and not std_logic_vector( unsigned( TEMP_8_double_in_0 ) - unsigned( "00000" & queue_head_oh_i ) );
	entry_port_transfer_0(0) <= TEMP_8_double_out_0(0) or TEMP_8_double_out_0(5);
	entry_port_transfer_1(0) <= TEMP_8_double_out_0(1) or TEMP_8_double_out_0(6);
	entry_port_transfer_2(0) <= TEMP_8_double_out_0(2) or TEMP_8_double_out_0(7);
	entry_port_transfer_3(0) <= TEMP_8_double_out_0(3) or TEMP_8_double_out_0(8);
	entry_port_transfer_4(0) <= TEMP_8_double_out_0(4) or TEMP_8_double_out_0(9);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(entry_wen_0, entry_port_transfer_0, or)
	entry_wen_0_o <= entry_port_transfer_0(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_1, entry_port_transfer_1, or)
	entry_wen_1_o <= entry_port_transfer_1(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_2, entry_port_transfer_2, or)
	entry_wen_2_o <= entry_port_transfer_2(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_3, entry_port_transfer_3, or)
	entry_wen_3_o <= entry_port_transfer_3(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_4, entry_port_transfer_4, or)
	entry_wen_4_o <= entry_port_transfer_4(0);
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
		entry_alloc_2_i : in std_logic;
		entry_alloc_3_i : in std_logic;
		entry_alloc_4_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_valid_2_i : in std_logic;
		entry_payload_valid_3_i : in std_logic;
		entry_payload_valid_4_i : in std_logic;
		entry_payload_0_o : out std_logic_vector(31 downto 0);
		entry_payload_1_o : out std_logic_vector(31 downto 0);
		entry_payload_2_o : out std_logic_vector(31 downto 0);
		entry_payload_3_o : out std_logic_vector(31 downto 0);
		entry_payload_4_o : out std_logic_vector(31 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		entry_wen_2_o : out std_logic;
		entry_wen_3_o : out std_logic;
		entry_wen_4_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(4 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_std is
	signal entry_port_idx_oh_0 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_2 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_3 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_4 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_4_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_5_mux_0 : std_logic_vector(31 downto 0);
	signal entry_ptq_ready_0 : std_logic;
	signal entry_ptq_ready_1 : std_logic;
	signal entry_ptq_ready_2 : std_logic;
	signal entry_ptq_ready_3 : std_logic;
	signal entry_ptq_ready_4 : std_logic;
	signal entry_waiting_for_port_0 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_1 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_2 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_3 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_4 : std_logic_vector(0 downto 0);
	signal port_ready_vec : std_logic_vector(0 downto 0);
	signal TEMP_6_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_7_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_7_res_1 : std_logic_vector(0 downto 0);
	signal entry_port_options_0 : std_logic_vector(0 downto 0);
	signal entry_port_options_1 : std_logic_vector(0 downto 0);
	signal entry_port_options_2 : std_logic_vector(0 downto 0);
	signal entry_port_options_3 : std_logic_vector(0 downto 0);
	signal entry_port_options_4 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_2 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_3 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_4 : std_logic_vector(0 downto 0);
	signal TEMP_8_double_in_0 : std_logic_vector(9 downto 0);
	signal TEMP_8_double_out_0 : std_logic_vector(9 downto 0);
begin
	entry_port_idx_oh_0 <= "1";
	entry_port_idx_oh_1 <= "1";
	entry_port_idx_oh_2 <= "1";
	entry_port_idx_oh_3 <= "1";
	entry_port_idx_oh_4 <= "1";
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

	-- Mux1H Begin
	-- Mux1H(entry_payload_2, port_payload, entry_port_idx_oh_2)
	TEMP_3_mux_0 <= port_payload_0_i when entry_port_idx_oh_2(0) = '1' else "00000000000000000000000000000000";
	entry_payload_2_o <= TEMP_3_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_3, port_payload, entry_port_idx_oh_3)
	TEMP_4_mux_0 <= port_payload_0_i when entry_port_idx_oh_3(0) = '1' else "00000000000000000000000000000000";
	entry_payload_3_o <= TEMP_4_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_4, port_payload, entry_port_idx_oh_4)
	TEMP_5_mux_0 <= port_payload_0_i when entry_port_idx_oh_4(0) = '1' else "00000000000000000000000000000000";
	entry_payload_4_o <= TEMP_5_mux_0;
	-- Mux1H End

	entry_ptq_ready_0 <= entry_alloc_0_i and not entry_payload_valid_0_i;
	entry_ptq_ready_1 <= entry_alloc_1_i and not entry_payload_valid_1_i;
	entry_ptq_ready_2 <= entry_alloc_2_i and not entry_payload_valid_2_i;
	entry_ptq_ready_3 <= entry_alloc_3_i and not entry_payload_valid_3_i;
	entry_ptq_ready_4 <= entry_alloc_4_i and not entry_payload_valid_4_i;
	entry_waiting_for_port_0 <= entry_port_idx_oh_0 when entry_ptq_ready_0 else "0";
	entry_waiting_for_port_1 <= entry_port_idx_oh_1 when entry_ptq_ready_1 else "0";
	entry_waiting_for_port_2 <= entry_port_idx_oh_2 when entry_ptq_ready_2 else "0";
	entry_waiting_for_port_3 <= entry_port_idx_oh_3 when entry_ptq_ready_3 else "0";
	entry_waiting_for_port_4 <= entry_port_idx_oh_4 when entry_ptq_ready_4 else "0";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_waiting_for_port, or)
	TEMP_6_res_0 <= entry_waiting_for_port_0 or entry_waiting_for_port_4;
	TEMP_6_res_1 <= entry_waiting_for_port_1;
	TEMP_6_res_2 <= entry_waiting_for_port_2;
	TEMP_6_res_3 <= entry_waiting_for_port_3;
	-- Layer End
	TEMP_7_res_0 <= TEMP_6_res_0 or TEMP_6_res_2;
	TEMP_7_res_1 <= TEMP_6_res_1 or TEMP_6_res_3;
	-- Layer End
	port_ready_vec <= TEMP_7_res_0 or TEMP_7_res_1;
	-- Reduction End

	port_ready_0_o <= port_ready_vec(0);
	entry_port_options_0(0) <= entry_waiting_for_port_0(0) and port_valid_0_i;
	entry_port_options_1(0) <= entry_waiting_for_port_1(0) and port_valid_0_i;
	entry_port_options_2(0) <= entry_waiting_for_port_2(0) and port_valid_0_i;
	entry_port_options_3(0) <= entry_waiting_for_port_3(0) and port_valid_0_i;
	entry_port_options_4(0) <= entry_waiting_for_port_4(0) and port_valid_0_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_transfer, entry_port_options, queue_head_oh)
	TEMP_8_double_in_0(0) <= entry_port_options_0(0);
	TEMP_8_double_in_0(5) <= entry_port_options_0(0);
	TEMP_8_double_in_0(1) <= entry_port_options_1(0);
	TEMP_8_double_in_0(6) <= entry_port_options_1(0);
	TEMP_8_double_in_0(2) <= entry_port_options_2(0);
	TEMP_8_double_in_0(7) <= entry_port_options_2(0);
	TEMP_8_double_in_0(3) <= entry_port_options_3(0);
	TEMP_8_double_in_0(8) <= entry_port_options_3(0);
	TEMP_8_double_in_0(4) <= entry_port_options_4(0);
	TEMP_8_double_in_0(9) <= entry_port_options_4(0);
	TEMP_8_double_out_0 <= TEMP_8_double_in_0 and not std_logic_vector( unsigned( TEMP_8_double_in_0 ) - unsigned( "00000" & queue_head_oh_i ) );
	entry_port_transfer_0(0) <= TEMP_8_double_out_0(0) or TEMP_8_double_out_0(5);
	entry_port_transfer_1(0) <= TEMP_8_double_out_0(1) or TEMP_8_double_out_0(6);
	entry_port_transfer_2(0) <= TEMP_8_double_out_0(2) or TEMP_8_double_out_0(7);
	entry_port_transfer_3(0) <= TEMP_8_double_out_0(3) or TEMP_8_double_out_0(8);
	entry_port_transfer_4(0) <= TEMP_8_double_out_0(4) or TEMP_8_double_out_0(9);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(entry_wen_0, entry_port_transfer_0, or)
	entry_wen_0_o <= entry_port_transfer_0(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_1, entry_port_transfer_1, or)
	entry_wen_1_o <= entry_port_transfer_1(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_2, entry_port_transfer_2, or)
	entry_wen_2_o <= entry_port_transfer_2(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_3, entry_port_transfer_3, or)
	entry_wen_3_o <= entry_port_transfer_3(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_4, entry_port_transfer_4, or)
	entry_wen_4_o <= entry_port_transfer_4(0);
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
		group_init_valid_1_i : in std_logic;
		group_init_ready_0_o : out std_logic;
		group_init_ready_1_o : out std_logic;
		ldp_addr_0_i : in std_logic_vector(9 downto 0);
		ldp_addr_valid_0_i : in std_logic;
		ldp_addr_ready_0_o : out std_logic;
		ldp_data_0_o : out std_logic_vector(31 downto 0);
		ldp_data_valid_0_o : out std_logic;
		ldp_data_ready_0_i : in std_logic;
		stp_addr_0_i : in std_logic_vector(9 downto 0);
		stp_addr_valid_0_i : in std_logic;
		stp_addr_ready_0_o : out std_logic;
		stp_data_0_i : in std_logic_vector(31 downto 0);
		stp_data_valid_0_i : in std_logic;
		stp_data_ready_0_o : out std_logic;
		empty_o : out std_logic;
		rreq_valid_0_o : out std_logic;
		rreq_ready_0_i : in std_logic;
		rreq_id_0_o : out std_logic_vector(3 downto 0);
		rreq_addr_0_o : out std_logic_vector(9 downto 0);
		rresp_valid_0_i : in std_logic;
		rresp_ready_0_o : out std_logic;
		rresp_id_0_i : in std_logic_vector(3 downto 0);
		rresp_data_0_i : in std_logic_vector(31 downto 0);
		wreq_valid_0_o : out std_logic;
		wreq_ready_0_i : in std_logic;
		wreq_id_0_o : out std_logic_vector(3 downto 0);
		wreq_addr_0_o : out std_logic_vector(9 downto 0);
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
	signal ldq_issue_0_d : std_logic;
	signal ldq_issue_0_q : std_logic;
	signal ldq_issue_1_d : std_logic;
	signal ldq_issue_1_q : std_logic;
	signal ldq_addr_valid_0_d : std_logic;
	signal ldq_addr_valid_0_q : std_logic;
	signal ldq_addr_valid_1_d : std_logic;
	signal ldq_addr_valid_1_q : std_logic;
	signal ldq_addr_0_d : std_logic_vector(9 downto 0);
	signal ldq_addr_0_q : std_logic_vector(9 downto 0);
	signal ldq_addr_1_d : std_logic_vector(9 downto 0);
	signal ldq_addr_1_q : std_logic_vector(9 downto 0);
	signal ldq_data_valid_0_d : std_logic;
	signal ldq_data_valid_0_q : std_logic;
	signal ldq_data_valid_1_d : std_logic;
	signal ldq_data_valid_1_q : std_logic;
	signal ldq_data_0_d : std_logic_vector(31 downto 0);
	signal ldq_data_0_q : std_logic_vector(31 downto 0);
	signal ldq_data_1_d : std_logic_vector(31 downto 0);
	signal ldq_data_1_q : std_logic_vector(31 downto 0);
	signal stq_alloc_0_d : std_logic;
	signal stq_alloc_0_q : std_logic;
	signal stq_alloc_1_d : std_logic;
	signal stq_alloc_1_q : std_logic;
	signal stq_alloc_2_d : std_logic;
	signal stq_alloc_2_q : std_logic;
	signal stq_alloc_3_d : std_logic;
	signal stq_alloc_3_q : std_logic;
	signal stq_alloc_4_d : std_logic;
	signal stq_alloc_4_q : std_logic;
	signal stq_addr_valid_0_d : std_logic;
	signal stq_addr_valid_0_q : std_logic;
	signal stq_addr_valid_1_d : std_logic;
	signal stq_addr_valid_1_q : std_logic;
	signal stq_addr_valid_2_d : std_logic;
	signal stq_addr_valid_2_q : std_logic;
	signal stq_addr_valid_3_d : std_logic;
	signal stq_addr_valid_3_q : std_logic;
	signal stq_addr_valid_4_d : std_logic;
	signal stq_addr_valid_4_q : std_logic;
	signal stq_addr_0_d : std_logic_vector(9 downto 0);
	signal stq_addr_0_q : std_logic_vector(9 downto 0);
	signal stq_addr_1_d : std_logic_vector(9 downto 0);
	signal stq_addr_1_q : std_logic_vector(9 downto 0);
	signal stq_addr_2_d : std_logic_vector(9 downto 0);
	signal stq_addr_2_q : std_logic_vector(9 downto 0);
	signal stq_addr_3_d : std_logic_vector(9 downto 0);
	signal stq_addr_3_q : std_logic_vector(9 downto 0);
	signal stq_addr_4_d : std_logic_vector(9 downto 0);
	signal stq_addr_4_q : std_logic_vector(9 downto 0);
	signal stq_data_valid_0_d : std_logic;
	signal stq_data_valid_0_q : std_logic;
	signal stq_data_valid_1_d : std_logic;
	signal stq_data_valid_1_q : std_logic;
	signal stq_data_valid_2_d : std_logic;
	signal stq_data_valid_2_q : std_logic;
	signal stq_data_valid_3_d : std_logic;
	signal stq_data_valid_3_q : std_logic;
	signal stq_data_valid_4_d : std_logic;
	signal stq_data_valid_4_q : std_logic;
	signal stq_data_0_d : std_logic_vector(31 downto 0);
	signal stq_data_0_q : std_logic_vector(31 downto 0);
	signal stq_data_1_d : std_logic_vector(31 downto 0);
	signal stq_data_1_q : std_logic_vector(31 downto 0);
	signal stq_data_2_d : std_logic_vector(31 downto 0);
	signal stq_data_2_q : std_logic_vector(31 downto 0);
	signal stq_data_3_d : std_logic_vector(31 downto 0);
	signal stq_data_3_q : std_logic_vector(31 downto 0);
	signal stq_data_4_d : std_logic_vector(31 downto 0);
	signal stq_data_4_q : std_logic_vector(31 downto 0);
	signal store_is_older_0_d : std_logic_vector(4 downto 0);
	signal store_is_older_0_q : std_logic_vector(4 downto 0);
	signal store_is_older_1_d : std_logic_vector(4 downto 0);
	signal store_is_older_1_q : std_logic_vector(4 downto 0);
	signal ldq_tail_d : std_logic_vector(0 downto 0);
	signal ldq_tail_q : std_logic_vector(0 downto 0);
	signal ldq_head_d : std_logic_vector(0 downto 0);
	signal ldq_head_q : std_logic_vector(0 downto 0);
	signal stq_tail_d : std_logic_vector(2 downto 0);
	signal stq_tail_q : std_logic_vector(2 downto 0);
	signal stq_head_d : std_logic_vector(2 downto 0);
	signal stq_head_q : std_logic_vector(2 downto 0);
	signal stq_issue_d : std_logic_vector(2 downto 0);
	signal stq_issue_q : std_logic_vector(2 downto 0);
	signal stq_resp_d : std_logic_vector(2 downto 0);
	signal stq_resp_q : std_logic_vector(2 downto 0);
	signal ldq_wen_0 : std_logic;
	signal ldq_wen_1 : std_logic;
	signal ldq_addr_wen_0 : std_logic;
	signal ldq_addr_wen_1 : std_logic;
	signal ldq_reset_0 : std_logic;
	signal ldq_reset_1 : std_logic;
	signal stq_wen_0 : std_logic;
	signal stq_wen_1 : std_logic;
	signal stq_wen_2 : std_logic;
	signal stq_wen_3 : std_logic;
	signal stq_wen_4 : std_logic;
	signal stq_addr_wen_0 : std_logic;
	signal stq_addr_wen_1 : std_logic;
	signal stq_addr_wen_2 : std_logic;
	signal stq_addr_wen_3 : std_logic;
	signal stq_addr_wen_4 : std_logic;
	signal stq_data_wen_0 : std_logic;
	signal stq_data_wen_1 : std_logic;
	signal stq_data_wen_2 : std_logic;
	signal stq_data_wen_3 : std_logic;
	signal stq_data_wen_4 : std_logic;
	signal stq_reset_0 : std_logic;
	signal stq_reset_1 : std_logic;
	signal stq_reset_2 : std_logic;
	signal stq_reset_3 : std_logic;
	signal stq_reset_4 : std_logic;
	signal ldq_data_wen_0 : std_logic;
	signal ldq_data_wen_1 : std_logic;
	signal ldq_issue_set_0 : std_logic;
	signal ldq_issue_set_1 : std_logic;
	signal ga_ls_order_0 : std_logic_vector(4 downto 0);
	signal ga_ls_order_1 : std_logic_vector(4 downto 0);
	signal num_loads : std_logic_vector(0 downto 0);
	signal num_stores : std_logic_vector(2 downto 0);
	signal stq_issue_en : std_logic;
	signal stq_resp_en : std_logic;
	signal ldq_empty : std_logic;
	signal stq_empty : std_logic;
	signal ldq_head_oh : std_logic_vector(1 downto 0);
	signal stq_head_oh : std_logic_vector(4 downto 0);
	signal ldq_alloc_next_0 : std_logic;
	signal ldq_alloc_next_1 : std_logic;
	signal stq_alloc_next_0 : std_logic;
	signal stq_alloc_next_1 : std_logic;
	signal stq_alloc_next_2 : std_logic;
	signal stq_alloc_next_3 : std_logic;
	signal stq_alloc_next_4 : std_logic;
	signal ldq_not_empty : std_logic;
	signal stq_not_empty : std_logic;
	signal TEMP_1_sum : std_logic_vector(3 downto 0);
	signal TEMP_1_res : std_logic_vector(3 downto 0);
	signal ldq_tail_oh : std_logic_vector(1 downto 0);
	signal ldq_head_next_oh : std_logic_vector(1 downto 0);
	signal ldq_head_next : std_logic_vector(0 downto 0);
	signal ldq_head_sel : std_logic;
	signal TEMP_2_double_in : std_logic_vector(3 downto 0);
	signal TEMP_2_double_out : std_logic_vector(3 downto 0);
	signal TEMP_3_in_0_0 : std_logic;
	signal TEMP_3_in_0_1 : std_logic;
	signal TEMP_3_out_0 : std_logic;
	signal stq_tail_oh : std_logic_vector(4 downto 0);
	signal stq_head_next_oh : std_logic_vector(4 downto 0);
	signal stq_head_next : std_logic_vector(2 downto 0);
	signal stq_head_sel : std_logic;
	signal load_idx_oh_0 : std_logic_vector(1 downto 0);
	signal load_en_0 : std_logic;
	signal store_idx : std_logic_vector(2 downto 0);
	signal store_en : std_logic;
	signal bypass_idx_oh_0 : std_logic_vector(4 downto 0);
	signal bypass_idx_oh_1 : std_logic_vector(4 downto 0);
	signal bypass_en_0 : std_logic;
	signal bypass_en_1 : std_logic;
	signal ld_st_conflict_0 : std_logic_vector(4 downto 0);
	signal ld_st_conflict_1 : std_logic_vector(4 downto 0);
	signal can_bypass_0 : std_logic_vector(4 downto 0);
	signal can_bypass_1 : std_logic_vector(4 downto 0);
	signal addr_valid_0 : std_logic_vector(4 downto 0);
	signal addr_valid_1 : std_logic_vector(4 downto 0);
	signal addr_same_0 : std_logic_vector(4 downto 0);
	signal addr_same_1 : std_logic_vector(4 downto 0);
	signal load_conflict_0 : std_logic;
	signal load_conflict_1 : std_logic;
	signal load_req_valid_0 : std_logic;
	signal load_req_valid_1 : std_logic;
	signal can_load_0 : std_logic;
	signal can_load_1 : std_logic;
	signal TEMP_4_res : std_logic_vector(3 downto 0);
	signal TEMP_5_res : std_logic_vector(1 downto 0);
	signal TEMP_6_res : std_logic_vector(3 downto 0);
	signal TEMP_7_res : std_logic_vector(1 downto 0);
	signal TEMP_8_double_in : std_logic_vector(3 downto 0);
	signal TEMP_8_double_out : std_logic_vector(3 downto 0);
	signal st_ld_conflict : std_logic_vector(1 downto 0);
	signal store_conflict : std_logic;
	signal store_valid : std_logic;
	signal store_data_valid : std_logic;
	signal store_addr_valid : std_logic;
	signal stq_last_oh : std_logic_vector(4 downto 0);
	signal bypass_en_vec_0 : std_logic_vector(4 downto 0);
	signal TEMP_9_double_in : std_logic_vector(9 downto 0);
	signal TEMP_9_base_rev : std_logic_vector(4 downto 0);
	signal TEMP_9_double_out : std_logic_vector(9 downto 0);
	signal TEMP_10_res : std_logic_vector(3 downto 0);
	signal TEMP_11_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_1 : std_logic_vector(4 downto 0);
	signal TEMP_12_double_in : std_logic_vector(9 downto 0);
	signal TEMP_12_base_rev : std_logic_vector(4 downto 0);
	signal TEMP_12_double_out : std_logic_vector(9 downto 0);
	signal TEMP_13_res : std_logic_vector(3 downto 0);
	signal TEMP_14_res : std_logic_vector(1 downto 0);
	signal TEMP_15_in_0_0 : std_logic;
	signal TEMP_15_in_0_1 : std_logic;
	signal TEMP_15_out_0 : std_logic;
	signal TEMP_15_in_1_0 : std_logic;
	signal TEMP_15_in_1_1 : std_logic;
	signal TEMP_15_out_1 : std_logic;
	signal TEMP_15_in_2_0 : std_logic;
	signal TEMP_15_in_2_1 : std_logic;
	signal TEMP_15_out_2 : std_logic;
	signal TEMP_15_in_3_0 : std_logic;
	signal TEMP_15_in_3_1 : std_logic;
	signal TEMP_15_out_3 : std_logic;
	signal TEMP_16_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_16_mux_1 : std_logic_vector(9 downto 0);
	signal ldq_issue_set_vec_0 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_1 : std_logic_vector(0 downto 0);
	signal read_idx_oh_0_0 : std_logic;
	signal read_valid_0 : std_logic;
	signal read_data_0 : std_logic_vector(31 downto 0);
	signal TEMP_17_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_0 : std_logic_vector(31 downto 0);
	signal TEMP_18_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_18_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_18_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_18_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_18_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_19_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_19_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_19_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_19_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_20_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_20_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_1_0 : std_logic;
	signal read_valid_1 : std_logic;
	signal read_data_1 : std_logic_vector(31 downto 0);
	signal TEMP_21_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_1 : std_logic_vector(31 downto 0);
	signal TEMP_22_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_22_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_22_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_22_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_22_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_23_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_23_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_23_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_23_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_24_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_24_res_1 : std_logic_vector(31 downto 0);
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
	ldq_head_oh(0) <= '1' when ldq_head_q = "0" else '0';
	ldq_head_oh(1) <= '1' when ldq_head_q = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(stq_head_oh, stq_head)
	stq_head_oh(0) <= '1' when stq_head_q = "000" else '0';
	stq_head_oh(1) <= '1' when stq_head_q = "001" else '0';
	stq_head_oh(2) <= '1' when stq_head_q = "010" else '0';
	stq_head_oh(3) <= '1' when stq_head_q = "011" else '0';
	stq_head_oh(4) <= '1' when stq_head_q = "100" else '0';
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
	stq_alloc_next_0 <= not stq_reset_0 and stq_alloc_0_q;
	stq_alloc_0_d <= stq_wen_0 or stq_alloc_next_0;
	stq_addr_valid_0_d <= not stq_wen_0 and ( stq_addr_wen_0 or stq_addr_valid_0_q );
	stq_data_valid_0_d <= not stq_wen_0 and ( stq_data_wen_0 or stq_data_valid_0_q );
	stq_alloc_next_1 <= not stq_reset_1 and stq_alloc_1_q;
	stq_alloc_1_d <= stq_wen_1 or stq_alloc_next_1;
	stq_addr_valid_1_d <= not stq_wen_1 and ( stq_addr_wen_1 or stq_addr_valid_1_q );
	stq_data_valid_1_d <= not stq_wen_1 and ( stq_data_wen_1 or stq_data_valid_1_q );
	stq_alloc_next_2 <= not stq_reset_2 and stq_alloc_2_q;
	stq_alloc_2_d <= stq_wen_2 or stq_alloc_next_2;
	stq_addr_valid_2_d <= not stq_wen_2 and ( stq_addr_wen_2 or stq_addr_valid_2_q );
	stq_data_valid_2_d <= not stq_wen_2 and ( stq_data_wen_2 or stq_data_valid_2_q );
	stq_alloc_next_3 <= not stq_reset_3 and stq_alloc_3_q;
	stq_alloc_3_d <= stq_wen_3 or stq_alloc_next_3;
	stq_addr_valid_3_d <= not stq_wen_3 and ( stq_addr_wen_3 or stq_addr_valid_3_q );
	stq_data_valid_3_d <= not stq_wen_3 and ( stq_data_wen_3 or stq_data_valid_3_q );
	stq_alloc_next_4 <= not stq_reset_4 and stq_alloc_4_q;
	stq_alloc_4_d <= stq_wen_4 or stq_alloc_next_4;
	stq_addr_valid_4_d <= not stq_wen_4 and ( stq_addr_wen_4 or stq_addr_valid_4_q );
	stq_data_valid_4_d <= not stq_wen_4 and ( stq_data_wen_4 or stq_data_valid_4_q );
	store_is_older_0_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_0(0) ) ) when ldq_wen_0 else not stq_reset_0 and store_is_older_0_q(0);
	store_is_older_0_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_0(1) ) ) when ldq_wen_0 else not stq_reset_1 and store_is_older_0_q(1);
	store_is_older_0_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_0(2) ) ) when ldq_wen_0 else not stq_reset_2 and store_is_older_0_q(2);
	store_is_older_0_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_0(3) ) ) when ldq_wen_0 else not stq_reset_3 and store_is_older_0_q(3);
	store_is_older_0_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_0(4) ) ) when ldq_wen_0 else not stq_reset_4 and store_is_older_0_q(4);
	store_is_older_1_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_1(0) ) ) when ldq_wen_1 else not stq_reset_0 and store_is_older_1_q(0);
	store_is_older_1_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_1(1) ) ) when ldq_wen_1 else not stq_reset_1 and store_is_older_1_q(1);
	store_is_older_1_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_1(2) ) ) when ldq_wen_1 else not stq_reset_2 and store_is_older_1_q(2);
	store_is_older_1_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_1(3) ) ) when ldq_wen_1 else not stq_reset_3 and store_is_older_1_q(3);
	store_is_older_1_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_1(4) ) ) when ldq_wen_1 else not stq_reset_4 and store_is_older_1_q(4);
	-- Reduction Begin
	-- Reduce(ldq_not_empty, ldq_alloc, or)
	ldq_not_empty <= ldq_alloc_0_q or ldq_alloc_1_q;
	-- Reduction End

	ldq_empty <= not ldq_not_empty;
	-- MuxLookUp Begin
	-- MuxLookUp(stq_not_empty, stq_alloc, stq_head)
	stq_not_empty <= 
	stq_alloc_0_q when (stq_head_q = "000") else
	stq_alloc_1_q when (stq_head_q = "001") else
	stq_alloc_2_q when (stq_head_q = "010") else
	stq_alloc_3_q when (stq_head_q = "011") else
	stq_alloc_4_q when (stq_head_q = "100") else
	'0';
	-- MuxLookUp End

	stq_empty <= not stq_not_empty;
	empty_o <= ldq_empty and stq_empty;
	-- WrapAdd Begin
	-- WrapAdd(ldq_tail, ldq_tail, num_loads, 2)
	ldq_tail_d <= std_logic_vector(unsigned(ldq_tail_q) + unsigned(num_loads));
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_tail, stq_tail, num_stores, 5)
	TEMP_1_sum <= std_logic_vector(unsigned('0' & stq_tail_q) + unsigned('0' & num_stores));
	TEMP_1_res <= std_logic_vector(unsigned(TEMP_1_sum) - 5) when unsigned(TEMP_1_sum) >= 5 else TEMP_1_sum;
	stq_tail_d <= TEMP_1_res(2 downto 0);
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_issue, stq_issue, 1, 5)
	stq_issue_d <= std_logic_vector(unsigned(stq_issue_q) - 4) when unsigned(stq_issue_q) >= 4 else std_logic_vector(unsigned(stq_issue_q) + 1);
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_resp, stq_resp, 1, 5)
	stq_resp_d <= std_logic_vector(unsigned(stq_resp_q) - 4) when unsigned(stq_resp_q) >= 4 else std_logic_vector(unsigned(stq_resp_q) + 1);
	-- WrapAdd End

	-- Bits To One-Hot Begin
	-- BitsToOH(ldq_tail_oh, ldq_tail)
	ldq_tail_oh(0) <= '1' when ldq_tail_q = "0" else '0';
	ldq_tail_oh(1) <= '1' when ldq_tail_q = "1" else '0';
	-- Bits To One-Hot End

	-- Priority Masking Begin
	-- CyclicPriorityMask(ldq_head_next_oh, ldq_alloc_next, ldq_tail_oh)
	TEMP_2_double_in(0) <= ldq_alloc_next_0;
	TEMP_2_double_in(2) <= ldq_alloc_next_0;
	TEMP_2_double_in(1) <= ldq_alloc_next_1;
	TEMP_2_double_in(3) <= ldq_alloc_next_1;
	TEMP_2_double_out <= TEMP_2_double_in and not std_logic_vector( unsigned( TEMP_2_double_in ) - unsigned( "00" & ldq_tail_oh ) );
	ldq_head_next_oh <= TEMP_2_double_out(1 downto 0) or TEMP_2_double_out(3 downto 2);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(ldq_head_sel, ldq_alloc_next, or)
	ldq_head_sel <= ldq_alloc_next_0 or ldq_alloc_next_1;
	-- Reduction End

	-- One-Hot To Bits Begin
	-- OHToBits(ldq_head_next, ldq_head_next_oh)
	TEMP_3_in_0_0 <= '0';
	TEMP_3_in_0_1 <= ldq_head_next_oh(1);
	TEMP_3_out_0 <= TEMP_3_in_0_0 or TEMP_3_in_0_1;
	ldq_head_next(0) <= TEMP_3_out_0;
	-- One-Hot To Bits End

	ldq_head_d <= ldq_head_next when ldq_head_sel else ldq_tail_q;
	-- Bits To One-Hot Begin
	-- BitsToOH(stq_tail_oh, stq_tail)
	stq_tail_oh(0) <= '1' when stq_tail_q = "000" else '0';
	stq_tail_oh(1) <= '1' when stq_tail_q = "001" else '0';
	stq_tail_oh(2) <= '1' when stq_tail_q = "010" else '0';
	stq_tail_oh(3) <= '1' when stq_tail_q = "011" else '0';
	stq_tail_oh(4) <= '1' when stq_tail_q = "100" else '0';
	-- Bits To One-Hot End

	-- WrapAdd Begin
	-- WrapAdd(stq_head_next, stq_head, 1, 5)
	stq_head_next <= std_logic_vector(unsigned(stq_head_q) - 4) when unsigned(stq_head_q) >= 4 else std_logic_vector(unsigned(stq_head_q) + 1);
	-- WrapAdd End

	stq_head_sel <= wresp_valid_0_i;
	stq_head_d <= stq_head_next when stq_head_sel else stq_head_q;
	handshake_lsq_lsq1_core_ga : entity work.handshake_lsq_lsq1_core_ga
		port map(
			rst => rst,
			clk => clk,
			group_init_valid_0_i => group_init_valid_0_i,
			group_init_valid_1_i => group_init_valid_1_i,
			group_init_ready_0_o => group_init_ready_0_o,
			group_init_ready_1_o => group_init_ready_1_o,
			ldq_tail_i => ldq_tail_q,
			ldq_head_i => ldq_head_q,
			ldq_empty_i => ldq_empty,
			stq_tail_i => stq_tail_q,
			stq_head_i => stq_head_q,
			stq_empty_i => stq_empty,
			ldq_wen_0_o => ldq_wen_0,
			ldq_wen_1_o => ldq_wen_1,
			num_loads_o => num_loads,
			stq_wen_0_o => stq_wen_0,
			stq_wen_1_o => stq_wen_1,
			stq_wen_2_o => stq_wen_2,
			stq_wen_3_o => stq_wen_3,
			stq_wen_4_o => stq_wen_4,
			ga_ls_order_0_o => ga_ls_order_0,
			ga_ls_order_1_o => ga_ls_order_1,
			num_stores_o => num_stores
		);
	handshake_lsq_lsq1_core_lda_dispatcher : entity work.handshake_lsq_lsq1_core_lda
		port map(
			rst => rst,
			clk => clk,
			port_payload_0_i => ldp_addr_0_i,
			port_ready_0_o => ldp_addr_ready_0_o,
			port_valid_0_i => ldp_addr_valid_0_i,
			entry_alloc_0_i => ldq_alloc_0_q,
			entry_alloc_1_i => ldq_alloc_1_q,
			entry_payload_valid_0_i => ldq_addr_valid_0_q,
			entry_payload_valid_1_i => ldq_addr_valid_1_q,
			entry_payload_0_o => ldq_addr_0_d,
			entry_payload_1_o => ldq_addr_1_d,
			entry_wen_0_o => ldq_addr_wen_0,
			entry_wen_1_o => ldq_addr_wen_1,
			queue_head_oh_i => ldq_head_oh
		);
	handshake_lsq_lsq1_core_ldd_dispatcher : entity work.handshake_lsq_lsq1_core_ldd
		port map(
			rst => rst,
			clk => clk,
			port_payload_0_o => ldp_data_0_o,
			port_ready_0_i => ldp_data_ready_0_i,
			port_valid_0_o => ldp_data_valid_0_o,
			entry_alloc_0_i => ldq_alloc_0_q,
			entry_alloc_1_i => ldq_alloc_1_q,
			entry_payload_valid_0_i => ldq_data_valid_0_q,
			entry_payload_valid_1_i => ldq_data_valid_1_q,
			entry_payload_0_i => ldq_data_0_q,
			entry_payload_1_i => ldq_data_1_q,
			entry_reset_0_o => ldq_reset_0,
			entry_reset_1_o => ldq_reset_1,
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
			entry_alloc_2_i => stq_alloc_2_q,
			entry_alloc_3_i => stq_alloc_3_q,
			entry_alloc_4_i => stq_alloc_4_q,
			entry_payload_valid_0_i => stq_addr_valid_0_q,
			entry_payload_valid_1_i => stq_addr_valid_1_q,
			entry_payload_valid_2_i => stq_addr_valid_2_q,
			entry_payload_valid_3_i => stq_addr_valid_3_q,
			entry_payload_valid_4_i => stq_addr_valid_4_q,
			entry_payload_0_o => stq_addr_0_d,
			entry_payload_1_o => stq_addr_1_d,
			entry_payload_2_o => stq_addr_2_d,
			entry_payload_3_o => stq_addr_3_d,
			entry_payload_4_o => stq_addr_4_d,
			entry_wen_0_o => stq_addr_wen_0,
			entry_wen_1_o => stq_addr_wen_1,
			entry_wen_2_o => stq_addr_wen_2,
			entry_wen_3_o => stq_addr_wen_3,
			entry_wen_4_o => stq_addr_wen_4,
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
			entry_alloc_2_i => stq_alloc_2_q,
			entry_alloc_3_i => stq_alloc_3_q,
			entry_alloc_4_i => stq_alloc_4_q,
			entry_payload_valid_0_i => stq_data_valid_0_q,
			entry_payload_valid_1_i => stq_data_valid_1_q,
			entry_payload_valid_2_i => stq_data_valid_2_q,
			entry_payload_valid_3_i => stq_data_valid_3_q,
			entry_payload_valid_4_i => stq_data_valid_4_q,
			entry_payload_0_o => stq_data_0_d,
			entry_payload_1_o => stq_data_1_d,
			entry_payload_2_o => stq_data_2_d,
			entry_payload_3_o => stq_data_3_d,
			entry_payload_4_o => stq_data_4_d,
			entry_wen_0_o => stq_data_wen_0,
			entry_wen_1_o => stq_data_wen_1,
			entry_wen_2_o => stq_data_wen_2,
			entry_wen_3_o => stq_data_wen_3,
			entry_wen_4_o => stq_data_wen_4,
			queue_head_oh_i => stq_head_oh
		);
	addr_valid_0(0) <= ldq_addr_valid_0_q and stq_addr_valid_0_q;
	addr_valid_0(1) <= ldq_addr_valid_0_q and stq_addr_valid_1_q;
	addr_valid_0(2) <= ldq_addr_valid_0_q and stq_addr_valid_2_q;
	addr_valid_0(3) <= ldq_addr_valid_0_q and stq_addr_valid_3_q;
	addr_valid_0(4) <= ldq_addr_valid_0_q and stq_addr_valid_4_q;
	addr_valid_1(0) <= ldq_addr_valid_1_q and stq_addr_valid_0_q;
	addr_valid_1(1) <= ldq_addr_valid_1_q and stq_addr_valid_1_q;
	addr_valid_1(2) <= ldq_addr_valid_1_q and stq_addr_valid_2_q;
	addr_valid_1(3) <= ldq_addr_valid_1_q and stq_addr_valid_3_q;
	addr_valid_1(4) <= ldq_addr_valid_1_q and stq_addr_valid_4_q;
	addr_same_0(0) <= '1' when ldq_addr_0_q = stq_addr_0_q else '0';
	addr_same_0(1) <= '1' when ldq_addr_0_q = stq_addr_1_q else '0';
	addr_same_0(2) <= '1' when ldq_addr_0_q = stq_addr_2_q else '0';
	addr_same_0(3) <= '1' when ldq_addr_0_q = stq_addr_3_q else '0';
	addr_same_0(4) <= '1' when ldq_addr_0_q = stq_addr_4_q else '0';
	addr_same_1(0) <= '1' when ldq_addr_1_q = stq_addr_0_q else '0';
	addr_same_1(1) <= '1' when ldq_addr_1_q = stq_addr_1_q else '0';
	addr_same_1(2) <= '1' when ldq_addr_1_q = stq_addr_2_q else '0';
	addr_same_1(3) <= '1' when ldq_addr_1_q = stq_addr_3_q else '0';
	addr_same_1(4) <= '1' when ldq_addr_1_q = stq_addr_4_q else '0';
	ld_st_conflict_0(0) <= stq_alloc_0_q and store_is_older_0_q(0) and ( addr_same_0(0) or not stq_addr_valid_0_q );
	ld_st_conflict_0(1) <= stq_alloc_1_q and store_is_older_0_q(1) and ( addr_same_0(1) or not stq_addr_valid_1_q );
	ld_st_conflict_0(2) <= stq_alloc_2_q and store_is_older_0_q(2) and ( addr_same_0(2) or not stq_addr_valid_2_q );
	ld_st_conflict_0(3) <= stq_alloc_3_q and store_is_older_0_q(3) and ( addr_same_0(3) or not stq_addr_valid_3_q );
	ld_st_conflict_0(4) <= stq_alloc_4_q and store_is_older_0_q(4) and ( addr_same_0(4) or not stq_addr_valid_4_q );
	ld_st_conflict_1(0) <= stq_alloc_0_q and store_is_older_1_q(0) and ( addr_same_1(0) or not stq_addr_valid_0_q );
	ld_st_conflict_1(1) <= stq_alloc_1_q and store_is_older_1_q(1) and ( addr_same_1(1) or not stq_addr_valid_1_q );
	ld_st_conflict_1(2) <= stq_alloc_2_q and store_is_older_1_q(2) and ( addr_same_1(2) or not stq_addr_valid_2_q );
	ld_st_conflict_1(3) <= stq_alloc_3_q and store_is_older_1_q(3) and ( addr_same_1(3) or not stq_addr_valid_3_q );
	ld_st_conflict_1(4) <= stq_alloc_4_q and store_is_older_1_q(4) and ( addr_same_1(4) or not stq_addr_valid_4_q );
	can_bypass_0(0) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_0_q and addr_same_0(0) and addr_valid_0(0);
	can_bypass_0(1) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_1_q and addr_same_0(1) and addr_valid_0(1);
	can_bypass_0(2) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_2_q and addr_same_0(2) and addr_valid_0(2);
	can_bypass_0(3) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_3_q and addr_same_0(3) and addr_valid_0(3);
	can_bypass_0(4) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_4_q and addr_same_0(4) and addr_valid_0(4);
	can_bypass_1(0) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_0_q and addr_same_1(0) and addr_valid_1(0);
	can_bypass_1(1) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_1_q and addr_same_1(1) and addr_valid_1(1);
	can_bypass_1(2) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_2_q and addr_same_1(2) and addr_valid_1(2);
	can_bypass_1(3) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_3_q and addr_same_1(3) and addr_valid_1(3);
	can_bypass_1(4) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_4_q and addr_same_1(4) and addr_valid_1(4);
	-- Reduction Begin
	-- Reduce(load_conflict_0, ld_st_conflict_0, or)
	TEMP_4_res(0) <= ld_st_conflict_0(0) or ld_st_conflict_0(4);
	TEMP_4_res(1) <= ld_st_conflict_0(1);
	TEMP_4_res(2) <= ld_st_conflict_0(2);
	TEMP_4_res(3) <= ld_st_conflict_0(3);
	-- Layer End
	TEMP_5_res(0) <= TEMP_4_res(0) or TEMP_4_res(2);
	TEMP_5_res(1) <= TEMP_4_res(1) or TEMP_4_res(3);
	-- Layer End
	load_conflict_0 <= TEMP_5_res(0) or TEMP_5_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_1, ld_st_conflict_1, or)
	TEMP_6_res(0) <= ld_st_conflict_1(0) or ld_st_conflict_1(4);
	TEMP_6_res(1) <= ld_st_conflict_1(1);
	TEMP_6_res(2) <= ld_st_conflict_1(2);
	TEMP_6_res(3) <= ld_st_conflict_1(3);
	-- Layer End
	TEMP_7_res(0) <= TEMP_6_res(0) or TEMP_6_res(2);
	TEMP_7_res(1) <= TEMP_6_res(1) or TEMP_6_res(3);
	-- Layer End
	load_conflict_1 <= TEMP_7_res(0) or TEMP_7_res(1);
	-- Reduction End

	load_req_valid_0 <= ldq_alloc_0_q and not ldq_issue_0_q and ldq_addr_valid_0_q;
	load_req_valid_1 <= ldq_alloc_1_q and not ldq_issue_1_q and ldq_addr_valid_1_q;
	can_load_0 <= not load_conflict_0 and load_req_valid_0;
	can_load_1 <= not load_conflict_1 and load_req_valid_1;
	-- Priority Masking Begin
	-- CyclicPriorityMask(load_idx_oh_0, can_load, ldq_head_oh)
	TEMP_8_double_in(0) <= can_load_0;
	TEMP_8_double_in(2) <= can_load_0;
	TEMP_8_double_in(1) <= can_load_1;
	TEMP_8_double_in(3) <= can_load_1;
	TEMP_8_double_out <= TEMP_8_double_in and not std_logic_vector( unsigned( TEMP_8_double_in ) - unsigned( "00" & ldq_head_oh ) );
	load_idx_oh_0 <= TEMP_8_double_out(1 downto 0) or TEMP_8_double_out(3 downto 2);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(load_en_0, can_load, or)
	load_en_0 <= can_load_0 or can_load_1;
	-- Reduction End

	st_ld_conflict(0) <= ldq_alloc_0_q and not store_is_older_0_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_0(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_0_q );
	st_ld_conflict(1) <= ldq_alloc_1_q and not store_is_older_1_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_1(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_1_q );
	-- Reduction Begin
	-- Reduce(store_conflict, st_ld_conflict, or)
	store_conflict <= st_ld_conflict(0) or st_ld_conflict(1);
	-- Reduction End

	-- MuxLookUp Begin
	-- MuxLookUp(store_valid, stq_alloc, stq_issue)
	store_valid <= 
	stq_alloc_0_q when (stq_issue_q = "000") else
	stq_alloc_1_q when (stq_issue_q = "001") else
	stq_alloc_2_q when (stq_issue_q = "010") else
	stq_alloc_3_q when (stq_issue_q = "011") else
	stq_alloc_4_q when (stq_issue_q = "100") else
	'0';
	-- MuxLookUp End

	-- MuxLookUp Begin
	-- MuxLookUp(store_data_valid, stq_data_valid, stq_issue)
	store_data_valid <= 
	stq_data_valid_0_q when (stq_issue_q = "000") else
	stq_data_valid_1_q when (stq_issue_q = "001") else
	stq_data_valid_2_q when (stq_issue_q = "010") else
	stq_data_valid_3_q when (stq_issue_q = "011") else
	stq_data_valid_4_q when (stq_issue_q = "100") else
	'0';
	-- MuxLookUp End

	-- MuxLookUp Begin
	-- MuxLookUp(store_addr_valid, stq_addr_valid, stq_issue)
	store_addr_valid <= 
	stq_addr_valid_0_q when (stq_issue_q = "000") else
	stq_addr_valid_1_q when (stq_issue_q = "001") else
	stq_addr_valid_2_q when (stq_issue_q = "010") else
	stq_addr_valid_3_q when (stq_issue_q = "011") else
	stq_addr_valid_4_q when (stq_issue_q = "100") else
	'0';
	-- MuxLookUp End

	store_en <= not store_conflict and store_valid and store_data_valid and store_addr_valid;
	store_idx <= stq_issue_q;
	-- Bits To One-Hot Begin
	-- BitsToOHSub1(stq_last_oh, stq_tail)
	stq_last_oh(0) <= '1' when stq_tail_q = "001" else '0';
	stq_last_oh(1) <= '1' when stq_tail_q = "010" else '0';
	stq_last_oh(2) <= '1' when stq_tail_q = "011" else '0';
	stq_last_oh(3) <= '1' when stq_tail_q = "100" else '0';
	stq_last_oh(4) <= '1' when stq_tail_q = "000" else '0';
	-- Bits To One-Hot End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_0, ld_st_conflict_0, stq_last_oh)
	TEMP_9_double_in(0) <= ld_st_conflict_0(4);
	TEMP_9_double_in(5) <= ld_st_conflict_0(4);
	TEMP_9_double_in(1) <= ld_st_conflict_0(3);
	TEMP_9_double_in(6) <= ld_st_conflict_0(3);
	TEMP_9_double_in(2) <= ld_st_conflict_0(2);
	TEMP_9_double_in(7) <= ld_st_conflict_0(2);
	TEMP_9_double_in(3) <= ld_st_conflict_0(1);
	TEMP_9_double_in(8) <= ld_st_conflict_0(1);
	TEMP_9_double_in(4) <= ld_st_conflict_0(0);
	TEMP_9_double_in(9) <= ld_st_conflict_0(0);
	TEMP_9_base_rev(0) <= stq_last_oh(4);
	TEMP_9_base_rev(1) <= stq_last_oh(3);
	TEMP_9_base_rev(2) <= stq_last_oh(2);
	TEMP_9_base_rev(3) <= stq_last_oh(1);
	TEMP_9_base_rev(4) <= stq_last_oh(0);
	TEMP_9_double_out <= TEMP_9_double_in and not std_logic_vector( unsigned( TEMP_9_double_in ) - unsigned( "00000" & TEMP_9_base_rev ) );
	bypass_idx_oh_0(4) <= TEMP_9_double_out(0) or TEMP_9_double_out(5);
	bypass_idx_oh_0(3) <= TEMP_9_double_out(1) or TEMP_9_double_out(6);
	bypass_idx_oh_0(2) <= TEMP_9_double_out(2) or TEMP_9_double_out(7);
	bypass_idx_oh_0(1) <= TEMP_9_double_out(3) or TEMP_9_double_out(8);
	bypass_idx_oh_0(0) <= TEMP_9_double_out(4) or TEMP_9_double_out(9);
	-- Priority Masking End

	bypass_en_vec_0 <= bypass_idx_oh_0 and can_bypass_0;
	-- Reduction Begin
	-- Reduce(bypass_en_0, bypass_en_vec_0, or)
	TEMP_10_res(0) <= bypass_en_vec_0(0) or bypass_en_vec_0(4);
	TEMP_10_res(1) <= bypass_en_vec_0(1);
	TEMP_10_res(2) <= bypass_en_vec_0(2);
	TEMP_10_res(3) <= bypass_en_vec_0(3);
	-- Layer End
	TEMP_11_res(0) <= TEMP_10_res(0) or TEMP_10_res(2);
	TEMP_11_res(1) <= TEMP_10_res(1) or TEMP_10_res(3);
	-- Layer End
	bypass_en_0 <= TEMP_11_res(0) or TEMP_11_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_1, ld_st_conflict_1, stq_last_oh)
	TEMP_12_double_in(0) <= ld_st_conflict_1(4);
	TEMP_12_double_in(5) <= ld_st_conflict_1(4);
	TEMP_12_double_in(1) <= ld_st_conflict_1(3);
	TEMP_12_double_in(6) <= ld_st_conflict_1(3);
	TEMP_12_double_in(2) <= ld_st_conflict_1(2);
	TEMP_12_double_in(7) <= ld_st_conflict_1(2);
	TEMP_12_double_in(3) <= ld_st_conflict_1(1);
	TEMP_12_double_in(8) <= ld_st_conflict_1(1);
	TEMP_12_double_in(4) <= ld_st_conflict_1(0);
	TEMP_12_double_in(9) <= ld_st_conflict_1(0);
	TEMP_12_base_rev(0) <= stq_last_oh(4);
	TEMP_12_base_rev(1) <= stq_last_oh(3);
	TEMP_12_base_rev(2) <= stq_last_oh(2);
	TEMP_12_base_rev(3) <= stq_last_oh(1);
	TEMP_12_base_rev(4) <= stq_last_oh(0);
	TEMP_12_double_out <= TEMP_12_double_in and not std_logic_vector( unsigned( TEMP_12_double_in ) - unsigned( "00000" & TEMP_12_base_rev ) );
	bypass_idx_oh_1(4) <= TEMP_12_double_out(0) or TEMP_12_double_out(5);
	bypass_idx_oh_1(3) <= TEMP_12_double_out(1) or TEMP_12_double_out(6);
	bypass_idx_oh_1(2) <= TEMP_12_double_out(2) or TEMP_12_double_out(7);
	bypass_idx_oh_1(1) <= TEMP_12_double_out(3) or TEMP_12_double_out(8);
	bypass_idx_oh_1(0) <= TEMP_12_double_out(4) or TEMP_12_double_out(9);
	-- Priority Masking End

	bypass_en_vec_1 <= bypass_idx_oh_1 and can_bypass_1;
	-- Reduction Begin
	-- Reduce(bypass_en_1, bypass_en_vec_1, or)
	TEMP_13_res(0) <= bypass_en_vec_1(0) or bypass_en_vec_1(4);
	TEMP_13_res(1) <= bypass_en_vec_1(1);
	TEMP_13_res(2) <= bypass_en_vec_1(2);
	TEMP_13_res(3) <= bypass_en_vec_1(3);
	-- Layer End
	TEMP_14_res(0) <= TEMP_13_res(0) or TEMP_13_res(2);
	TEMP_14_res(1) <= TEMP_13_res(1) or TEMP_13_res(3);
	-- Layer End
	bypass_en_1 <= TEMP_14_res(0) or TEMP_14_res(1);
	-- Reduction End

	rreq_valid_0_o <= load_en_0;
	-- One-Hot To Bits Begin
	-- OHToBits(rreq_id_0, load_idx_oh_0)
	TEMP_15_in_0_0 <= '0';
	TEMP_15_in_0_1 <= load_idx_oh_0(1);
	TEMP_15_out_0 <= TEMP_15_in_0_0 or TEMP_15_in_0_1;
	rreq_id_0_o(0) <= TEMP_15_out_0;
	TEMP_15_in_1_0 <= '0';
	TEMP_15_in_1_1 <= '0';
	TEMP_15_out_1 <= TEMP_15_in_1_0 or TEMP_15_in_1_1;
	rreq_id_0_o(1) <= TEMP_15_out_1;
	TEMP_15_in_2_0 <= '0';
	TEMP_15_in_2_1 <= '0';
	TEMP_15_out_2 <= TEMP_15_in_2_0 or TEMP_15_in_2_1;
	rreq_id_0_o(2) <= TEMP_15_out_2;
	TEMP_15_in_3_0 <= '0';
	TEMP_15_in_3_1 <= '0';
	TEMP_15_out_3 <= TEMP_15_in_3_0 or TEMP_15_in_3_1;
	rreq_id_0_o(3) <= TEMP_15_out_3;
	-- One-Hot To Bits End

	-- Mux1H Begin
	-- Mux1H(rreq_addr_0, ldq_addr, load_idx_oh_0)
	TEMP_16_mux_0 <= ldq_addr_0_q when load_idx_oh_0(0) = '1' else "0000000000";
	TEMP_16_mux_1 <= ldq_addr_1_q when load_idx_oh_0(1) = '1' else "0000000000";
	rreq_addr_0_o <= TEMP_16_mux_0 or TEMP_16_mux_1;
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

	wreq_valid_0_o <= store_en;
	wreq_id_0_o <= "0000";
	-- MuxLookUp Begin
	-- MuxLookUp(wreq_addr_0, stq_addr, store_idx)
	wreq_addr_0_o <= 
	stq_addr_0_q when (store_idx = "000") else
	stq_addr_1_q when (store_idx = "001") else
	stq_addr_2_q when (store_idx = "010") else
	stq_addr_3_q when (store_idx = "011") else
	stq_addr_4_q when (store_idx = "100") else
	"0000000000";
	-- MuxLookUp End

	-- MuxLookUp Begin
	-- MuxLookUp(wreq_data_0, stq_data, store_idx)
	wreq_data_0_o <= 
	stq_data_0_q when (store_idx = "000") else
	stq_data_1_q when (store_idx = "001") else
	stq_data_2_q when (store_idx = "010") else
	stq_data_3_q when (store_idx = "011") else
	stq_data_4_q when (store_idx = "100") else
	"00000000000000000000000000000000";
	-- MuxLookUp End

	stq_issue_en <= store_en and wreq_ready_0_i;
	read_idx_oh_0_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0000" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_0, rresp_data, read_idx_oh_0)
	TEMP_17_mux_0 <= rresp_data_0_i when read_idx_oh_0_0 = '1' else "00000000000000000000000000000000";
	read_data_0 <= TEMP_17_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_0, read_idx_oh_0, or)
	read_valid_0 <= read_idx_oh_0_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_0, stq_data, bypass_idx_oh_0)
	TEMP_18_mux_0 <= stq_data_0_q when bypass_idx_oh_0(0) = '1' else "00000000000000000000000000000000";
	TEMP_18_mux_1 <= stq_data_1_q when bypass_idx_oh_0(1) = '1' else "00000000000000000000000000000000";
	TEMP_18_mux_2 <= stq_data_2_q when bypass_idx_oh_0(2) = '1' else "00000000000000000000000000000000";
	TEMP_18_mux_3 <= stq_data_3_q when bypass_idx_oh_0(3) = '1' else "00000000000000000000000000000000";
	TEMP_18_mux_4 <= stq_data_4_q when bypass_idx_oh_0(4) = '1' else "00000000000000000000000000000000";
	TEMP_19_res_0 <= TEMP_18_mux_0 or TEMP_18_mux_4;
	TEMP_19_res_1 <= TEMP_18_mux_1;
	TEMP_19_res_2 <= TEMP_18_mux_2;
	TEMP_19_res_3 <= TEMP_18_mux_3;
	-- Layer End
	TEMP_20_res_0 <= TEMP_19_res_0 or TEMP_19_res_2;
	TEMP_20_res_1 <= TEMP_19_res_1 or TEMP_19_res_3;
	-- Layer End
	bypass_data_0 <= TEMP_20_res_0 or TEMP_20_res_1;
	-- Mux1H End

	ldq_data_0_d <= read_data_0 or bypass_data_0;
	ldq_data_wen_0 <= bypass_en_0 or read_valid_0;
	read_idx_oh_1_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0001" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_1, rresp_data, read_idx_oh_1)
	TEMP_21_mux_0 <= rresp_data_0_i when read_idx_oh_1_0 = '1' else "00000000000000000000000000000000";
	read_data_1 <= TEMP_21_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_1, read_idx_oh_1, or)
	read_valid_1 <= read_idx_oh_1_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_1, stq_data, bypass_idx_oh_1)
	TEMP_22_mux_0 <= stq_data_0_q when bypass_idx_oh_1(0) = '1' else "00000000000000000000000000000000";
	TEMP_22_mux_1 <= stq_data_1_q when bypass_idx_oh_1(1) = '1' else "00000000000000000000000000000000";
	TEMP_22_mux_2 <= stq_data_2_q when bypass_idx_oh_1(2) = '1' else "00000000000000000000000000000000";
	TEMP_22_mux_3 <= stq_data_3_q when bypass_idx_oh_1(3) = '1' else "00000000000000000000000000000000";
	TEMP_22_mux_4 <= stq_data_4_q when bypass_idx_oh_1(4) = '1' else "00000000000000000000000000000000";
	TEMP_23_res_0 <= TEMP_22_mux_0 or TEMP_22_mux_4;
	TEMP_23_res_1 <= TEMP_22_mux_1;
	TEMP_23_res_2 <= TEMP_22_mux_2;
	TEMP_23_res_3 <= TEMP_22_mux_3;
	-- Layer End
	TEMP_24_res_0 <= TEMP_23_res_0 or TEMP_23_res_2;
	TEMP_24_res_1 <= TEMP_23_res_1 or TEMP_23_res_3;
	-- Layer End
	bypass_data_1 <= TEMP_24_res_0 or TEMP_24_res_1;
	-- Mux1H End

	ldq_data_1_d <= read_data_1 or bypass_data_1;
	ldq_data_wen_1 <= bypass_en_1 or read_valid_1;
	rresp_ready_0_o <= '1';
	stq_reset_0 <= wresp_valid_0_i when ( stq_resp_q = "000" ) else '0';
	stq_reset_1 <= wresp_valid_0_i when ( stq_resp_q = "001" ) else '0';
	stq_reset_2 <= wresp_valid_0_i when ( stq_resp_q = "010" ) else '0';
	stq_reset_3 <= wresp_valid_0_i when ( stq_resp_q = "011" ) else '0';
	stq_reset_4 <= wresp_valid_0_i when ( stq_resp_q = "100" ) else '0';
	stq_resp_en <= wresp_valid_0_i;
	wresp_ready_0_o <= '1';

	process (clk, rst) is
	begin
		if (rst = '1') then
			ldq_alloc_0_q <= '0';
			ldq_alloc_1_q <= '0';
		elsif (rising_edge(clk)) then
			ldq_alloc_0_q <= ldq_alloc_0_d;
			ldq_alloc_1_q <= ldq_alloc_1_d;
		end if;
		if (rising_edge(clk)) then
			ldq_issue_0_q <= ldq_issue_0_d;
			ldq_issue_1_q <= ldq_issue_1_d;
		end if;
		if (rising_edge(clk)) then
			ldq_addr_valid_0_q <= ldq_addr_valid_0_d;
			ldq_addr_valid_1_q <= ldq_addr_valid_1_d;
		end if;
		if (rising_edge(clk)) then
			if (ldq_addr_wen_0 = '1') then
				ldq_addr_0_q <= ldq_addr_0_d;
			end if;
			if (ldq_addr_wen_1 = '1') then
				ldq_addr_1_q <= ldq_addr_1_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			ldq_data_valid_0_q <= ldq_data_valid_0_d;
			ldq_data_valid_1_q <= ldq_data_valid_1_d;
		end if;
		if (rising_edge(clk)) then
			if (ldq_data_wen_0 = '1') then
				ldq_data_0_q <= ldq_data_0_d;
			end if;
			if (ldq_data_wen_1 = '1') then
				ldq_data_1_q <= ldq_data_1_d;
			end if;
		end if;
		if (rst = '1') then
			stq_alloc_0_q <= '0';
			stq_alloc_1_q <= '0';
			stq_alloc_2_q <= '0';
			stq_alloc_3_q <= '0';
			stq_alloc_4_q <= '0';
		elsif (rising_edge(clk)) then
			stq_alloc_0_q <= stq_alloc_0_d;
			stq_alloc_1_q <= stq_alloc_1_d;
			stq_alloc_2_q <= stq_alloc_2_d;
			stq_alloc_3_q <= stq_alloc_3_d;
			stq_alloc_4_q <= stq_alloc_4_d;
		end if;
		if (rising_edge(clk)) then
			stq_addr_valid_0_q <= stq_addr_valid_0_d;
			stq_addr_valid_1_q <= stq_addr_valid_1_d;
			stq_addr_valid_2_q <= stq_addr_valid_2_d;
			stq_addr_valid_3_q <= stq_addr_valid_3_d;
			stq_addr_valid_4_q <= stq_addr_valid_4_d;
		end if;
		if (rising_edge(clk)) then
			if (stq_addr_wen_0 = '1') then
				stq_addr_0_q <= stq_addr_0_d;
			end if;
			if (stq_addr_wen_1 = '1') then
				stq_addr_1_q <= stq_addr_1_d;
			end if;
			if (stq_addr_wen_2 = '1') then
				stq_addr_2_q <= stq_addr_2_d;
			end if;
			if (stq_addr_wen_3 = '1') then
				stq_addr_3_q <= stq_addr_3_d;
			end if;
			if (stq_addr_wen_4 = '1') then
				stq_addr_4_q <= stq_addr_4_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			stq_data_valid_0_q <= stq_data_valid_0_d;
			stq_data_valid_1_q <= stq_data_valid_1_d;
			stq_data_valid_2_q <= stq_data_valid_2_d;
			stq_data_valid_3_q <= stq_data_valid_3_d;
			stq_data_valid_4_q <= stq_data_valid_4_d;
		end if;
		if (rising_edge(clk)) then
			if (stq_data_wen_0 = '1') then
				stq_data_0_q <= stq_data_0_d;
			end if;
			if (stq_data_wen_1 = '1') then
				stq_data_1_q <= stq_data_1_d;
			end if;
			if (stq_data_wen_2 = '1') then
				stq_data_2_q <= stq_data_2_d;
			end if;
			if (stq_data_wen_3 = '1') then
				stq_data_3_q <= stq_data_3_d;
			end if;
			if (stq_data_wen_4 = '1') then
				stq_data_4_q <= stq_data_4_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			store_is_older_0_q <= store_is_older_0_d;
			store_is_older_1_q <= store_is_older_1_d;
		end if;
		if (rst = '1') then
			ldq_tail_q <= "0";
		elsif (rising_edge(clk)) then
			ldq_tail_q <= ldq_tail_d;
		end if;
		if (rst = '1') then
			ldq_head_q <= "0";
		elsif (rising_edge(clk)) then
			ldq_head_q <= ldq_head_d;
		end if;
		if (rst = '1') then
			stq_tail_q <= "000";
		elsif (rising_edge(clk)) then
			stq_tail_q <= stq_tail_d;
		end if;
		if (rst = '1') then
			stq_head_q <= "000";
		elsif (rising_edge(clk)) then
			stq_head_q <= stq_head_d;
		end if;
		if (rst = '1') then
			stq_issue_q <= "000";
		elsif (rising_edge(clk)) then
			if (stq_issue_en = '1') then
				stq_issue_q <= stq_issue_d;
			end if;
		end if;
		if (rst = '1') then
			stq_resp_q <= "000";
		elsif (rising_edge(clk)) then
			if (stq_resp_en = '1') then
				stq_resp_q <= stq_resp_d;
			end if;
		end if;
	end process;
end architecture;
