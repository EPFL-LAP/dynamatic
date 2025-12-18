

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1_core_ga is
	port(
		rst : in std_logic;
		clk : in std_logic;
		group_init_valid_0_i : in std_logic;
		group_init_ready_0_o : out std_logic;
		ldq_tail_i : in std_logic_vector(3 downto 0);
		ldq_head_i : in std_logic_vector(3 downto 0);
		ldq_empty_i : in std_logic;
		stq_tail_i : in std_logic_vector(3 downto 0);
		stq_head_i : in std_logic_vector(3 downto 0);
		stq_empty_i : in std_logic;
		ldq_wen_0_o : out std_logic;
		ldq_wen_1_o : out std_logic;
		ldq_wen_2_o : out std_logic;
		ldq_wen_3_o : out std_logic;
		ldq_wen_4_o : out std_logic;
		ldq_wen_5_o : out std_logic;
		ldq_wen_6_o : out std_logic;
		ldq_wen_7_o : out std_logic;
		ldq_wen_8_o : out std_logic;
		ldq_wen_9_o : out std_logic;
		ldq_wen_10_o : out std_logic;
		ldq_wen_11_o : out std_logic;
		ldq_wen_12_o : out std_logic;
		ldq_wen_13_o : out std_logic;
		num_loads_o : out std_logic_vector(3 downto 0);
		stq_wen_0_o : out std_logic;
		stq_wen_1_o : out std_logic;
		stq_wen_2_o : out std_logic;
		stq_wen_3_o : out std_logic;
		stq_wen_4_o : out std_logic;
		stq_wen_5_o : out std_logic;
		stq_wen_6_o : out std_logic;
		stq_wen_7_o : out std_logic;
		stq_wen_8_o : out std_logic;
		stq_wen_9_o : out std_logic;
		stq_wen_10_o : out std_logic;
		stq_wen_11_o : out std_logic;
		stq_wen_12_o : out std_logic;
		stq_wen_13_o : out std_logic;
		num_stores_o : out std_logic_vector(3 downto 0);
		ga_ls_order_0_o : out std_logic_vector(13 downto 0);
		ga_ls_order_1_o : out std_logic_vector(13 downto 0);
		ga_ls_order_2_o : out std_logic_vector(13 downto 0);
		ga_ls_order_3_o : out std_logic_vector(13 downto 0);
		ga_ls_order_4_o : out std_logic_vector(13 downto 0);
		ga_ls_order_5_o : out std_logic_vector(13 downto 0);
		ga_ls_order_6_o : out std_logic_vector(13 downto 0);
		ga_ls_order_7_o : out std_logic_vector(13 downto 0);
		ga_ls_order_8_o : out std_logic_vector(13 downto 0);
		ga_ls_order_9_o : out std_logic_vector(13 downto 0);
		ga_ls_order_10_o : out std_logic_vector(13 downto 0);
		ga_ls_order_11_o : out std_logic_vector(13 downto 0);
		ga_ls_order_12_o : out std_logic_vector(13 downto 0);
		ga_ls_order_13_o : out std_logic_vector(13 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_ga is
	signal num_loads : std_logic_vector(3 downto 0);
	signal num_stores : std_logic_vector(3 downto 0);
	signal loads_sub : std_logic_vector(3 downto 0);
	signal stores_sub : std_logic_vector(3 downto 0);
	signal empty_loads : std_logic_vector(3 downto 0);
	signal empty_stores : std_logic_vector(3 downto 0);
	signal group_init_ready_0 : std_logic;
	signal group_init_hs_0 : std_logic;
	signal ga_ls_order_rom_0 : std_logic_vector(13 downto 0);
	signal ga_ls_order_rom_1 : std_logic_vector(13 downto 0);
	signal ga_ls_order_rom_2 : std_logic_vector(13 downto 0);
	signal ga_ls_order_rom_3 : std_logic_vector(13 downto 0);
	signal ga_ls_order_rom_4 : std_logic_vector(13 downto 0);
	signal ga_ls_order_rom_5 : std_logic_vector(13 downto 0);
	signal ga_ls_order_rom_6 : std_logic_vector(13 downto 0);
	signal ga_ls_order_rom_7 : std_logic_vector(13 downto 0);
	signal ga_ls_order_rom_8 : std_logic_vector(13 downto 0);
	signal ga_ls_order_rom_9 : std_logic_vector(13 downto 0);
	signal ga_ls_order_rom_10 : std_logic_vector(13 downto 0);
	signal ga_ls_order_rom_11 : std_logic_vector(13 downto 0);
	signal ga_ls_order_rom_12 : std_logic_vector(13 downto 0);
	signal ga_ls_order_rom_13 : std_logic_vector(13 downto 0);
	signal ga_ls_order_temp_0 : std_logic_vector(13 downto 0);
	signal ga_ls_order_temp_1 : std_logic_vector(13 downto 0);
	signal ga_ls_order_temp_2 : std_logic_vector(13 downto 0);
	signal ga_ls_order_temp_3 : std_logic_vector(13 downto 0);
	signal ga_ls_order_temp_4 : std_logic_vector(13 downto 0);
	signal ga_ls_order_temp_5 : std_logic_vector(13 downto 0);
	signal ga_ls_order_temp_6 : std_logic_vector(13 downto 0);
	signal ga_ls_order_temp_7 : std_logic_vector(13 downto 0);
	signal ga_ls_order_temp_8 : std_logic_vector(13 downto 0);
	signal ga_ls_order_temp_9 : std_logic_vector(13 downto 0);
	signal ga_ls_order_temp_10 : std_logic_vector(13 downto 0);
	signal ga_ls_order_temp_11 : std_logic_vector(13 downto 0);
	signal ga_ls_order_temp_12 : std_logic_vector(13 downto 0);
	signal ga_ls_order_temp_13 : std_logic_vector(13 downto 0);
	signal TEMP_1_mux_0_0 : std_logic_vector(13 downto 0);
	signal TEMP_1_mux_1_0 : std_logic_vector(13 downto 0);
	signal TEMP_1_mux_2_0 : std_logic_vector(13 downto 0);
	signal TEMP_1_mux_3_0 : std_logic_vector(13 downto 0);
	signal TEMP_1_mux_4_0 : std_logic_vector(13 downto 0);
	signal TEMP_1_mux_5_0 : std_logic_vector(13 downto 0);
	signal TEMP_1_mux_6_0 : std_logic_vector(13 downto 0);
	signal TEMP_1_mux_7_0 : std_logic_vector(13 downto 0);
	signal TEMP_1_mux_8_0 : std_logic_vector(13 downto 0);
	signal TEMP_1_mux_9_0 : std_logic_vector(13 downto 0);
	signal TEMP_1_mux_10_0 : std_logic_vector(13 downto 0);
	signal TEMP_1_mux_11_0 : std_logic_vector(13 downto 0);
	signal TEMP_1_mux_12_0 : std_logic_vector(13 downto 0);
	signal TEMP_1_mux_13_0 : std_logic_vector(13 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(3 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(3 downto 0);
	signal ldq_wen_unshifted_0 : std_logic;
	signal ldq_wen_unshifted_1 : std_logic;
	signal ldq_wen_unshifted_2 : std_logic;
	signal ldq_wen_unshifted_3 : std_logic;
	signal ldq_wen_unshifted_4 : std_logic;
	signal ldq_wen_unshifted_5 : std_logic;
	signal ldq_wen_unshifted_6 : std_logic;
	signal ldq_wen_unshifted_7 : std_logic;
	signal ldq_wen_unshifted_8 : std_logic;
	signal ldq_wen_unshifted_9 : std_logic;
	signal ldq_wen_unshifted_10 : std_logic;
	signal ldq_wen_unshifted_11 : std_logic;
	signal ldq_wen_unshifted_12 : std_logic;
	signal ldq_wen_unshifted_13 : std_logic;
	signal stq_wen_unshifted_0 : std_logic;
	signal stq_wen_unshifted_1 : std_logic;
	signal stq_wen_unshifted_2 : std_logic;
	signal stq_wen_unshifted_3 : std_logic;
	signal stq_wen_unshifted_4 : std_logic;
	signal stq_wen_unshifted_5 : std_logic;
	signal stq_wen_unshifted_6 : std_logic;
	signal stq_wen_unshifted_7 : std_logic;
	signal stq_wen_unshifted_8 : std_logic;
	signal stq_wen_unshifted_9 : std_logic;
	signal stq_wen_unshifted_10 : std_logic;
	signal stq_wen_unshifted_11 : std_logic;
	signal stq_wen_unshifted_12 : std_logic;
	signal stq_wen_unshifted_13 : std_logic;
	signal TEMP_4_res_0 : std_logic;
	signal TEMP_4_res_1 : std_logic;
	signal TEMP_4_res_2 : std_logic;
	signal TEMP_4_res_3 : std_logic;
	signal TEMP_4_res_4 : std_logic;
	signal TEMP_4_res_5 : std_logic;
	signal TEMP_4_res_6 : std_logic;
	signal TEMP_4_res_7 : std_logic;
	signal TEMP_4_res_8 : std_logic;
	signal TEMP_4_res_9 : std_logic;
	signal TEMP_4_res_10 : std_logic;
	signal TEMP_4_res_11 : std_logic;
	signal TEMP_4_res_12 : std_logic;
	signal TEMP_4_res_13 : std_logic;
	signal TEMP_5_res_0 : std_logic;
	signal TEMP_5_res_1 : std_logic;
	signal TEMP_5_res_2 : std_logic;
	signal TEMP_5_res_3 : std_logic;
	signal TEMP_5_res_4 : std_logic;
	signal TEMP_5_res_5 : std_logic;
	signal TEMP_5_res_6 : std_logic;
	signal TEMP_5_res_7 : std_logic;
	signal TEMP_5_res_8 : std_logic;
	signal TEMP_5_res_9 : std_logic;
	signal TEMP_5_res_10 : std_logic;
	signal TEMP_5_res_11 : std_logic;
	signal TEMP_5_res_12 : std_logic;
	signal TEMP_5_res_13 : std_logic;
	signal TEMP_6_res_0 : std_logic;
	signal TEMP_6_res_1 : std_logic;
	signal TEMP_6_res_2 : std_logic;
	signal TEMP_6_res_3 : std_logic;
	signal TEMP_6_res_4 : std_logic;
	signal TEMP_6_res_5 : std_logic;
	signal TEMP_6_res_6 : std_logic;
	signal TEMP_6_res_7 : std_logic;
	signal TEMP_6_res_8 : std_logic;
	signal TEMP_6_res_9 : std_logic;
	signal TEMP_6_res_10 : std_logic;
	signal TEMP_6_res_11 : std_logic;
	signal TEMP_6_res_12 : std_logic;
	signal TEMP_6_res_13 : std_logic;
	signal TEMP_7_res_0 : std_logic;
	signal TEMP_7_res_1 : std_logic;
	signal TEMP_7_res_2 : std_logic;
	signal TEMP_7_res_3 : std_logic;
	signal TEMP_7_res_4 : std_logic;
	signal TEMP_7_res_5 : std_logic;
	signal TEMP_7_res_6 : std_logic;
	signal TEMP_7_res_7 : std_logic;
	signal TEMP_7_res_8 : std_logic;
	signal TEMP_7_res_9 : std_logic;
	signal TEMP_7_res_10 : std_logic;
	signal TEMP_7_res_11 : std_logic;
	signal TEMP_7_res_12 : std_logic;
	signal TEMP_7_res_13 : std_logic;
	signal TEMP_8_res_0 : std_logic;
	signal TEMP_8_res_1 : std_logic;
	signal TEMP_8_res_2 : std_logic;
	signal TEMP_8_res_3 : std_logic;
	signal TEMP_8_res_4 : std_logic;
	signal TEMP_8_res_5 : std_logic;
	signal TEMP_8_res_6 : std_logic;
	signal TEMP_8_res_7 : std_logic;
	signal TEMP_8_res_8 : std_logic;
	signal TEMP_8_res_9 : std_logic;
	signal TEMP_8_res_10 : std_logic;
	signal TEMP_8_res_11 : std_logic;
	signal TEMP_8_res_12 : std_logic;
	signal TEMP_8_res_13 : std_logic;
	signal TEMP_9_res_0 : std_logic;
	signal TEMP_9_res_1 : std_logic;
	signal TEMP_9_res_2 : std_logic;
	signal TEMP_9_res_3 : std_logic;
	signal TEMP_9_res_4 : std_logic;
	signal TEMP_9_res_5 : std_logic;
	signal TEMP_9_res_6 : std_logic;
	signal TEMP_9_res_7 : std_logic;
	signal TEMP_9_res_8 : std_logic;
	signal TEMP_9_res_9 : std_logic;
	signal TEMP_9_res_10 : std_logic;
	signal TEMP_9_res_11 : std_logic;
	signal TEMP_9_res_12 : std_logic;
	signal TEMP_9_res_13 : std_logic;
	signal TEMP_10_res : std_logic_vector(13 downto 0);
	signal TEMP_11_res : std_logic_vector(13 downto 0);
	signal TEMP_12_res : std_logic_vector(13 downto 0);
	signal TEMP_13_res : std_logic_vector(13 downto 0);
	signal TEMP_14_res : std_logic_vector(13 downto 0);
	signal TEMP_15_res : std_logic_vector(13 downto 0);
	signal TEMP_16_res : std_logic_vector(13 downto 0);
	signal TEMP_17_res : std_logic_vector(13 downto 0);
	signal TEMP_18_res : std_logic_vector(13 downto 0);
	signal TEMP_19_res : std_logic_vector(13 downto 0);
	signal TEMP_20_res : std_logic_vector(13 downto 0);
	signal TEMP_21_res : std_logic_vector(13 downto 0);
	signal TEMP_22_res : std_logic_vector(13 downto 0);
	signal TEMP_23_res : std_logic_vector(13 downto 0);
	signal TEMP_24_res : std_logic_vector(13 downto 0);
	signal TEMP_25_res : std_logic_vector(13 downto 0);
	signal TEMP_26_res : std_logic_vector(13 downto 0);
	signal TEMP_27_res : std_logic_vector(13 downto 0);
	signal TEMP_28_res : std_logic_vector(13 downto 0);
	signal TEMP_29_res : std_logic_vector(13 downto 0);
	signal TEMP_30_res : std_logic_vector(13 downto 0);
	signal TEMP_31_res : std_logic_vector(13 downto 0);
	signal TEMP_32_res : std_logic_vector(13 downto 0);
	signal TEMP_33_res : std_logic_vector(13 downto 0);
	signal TEMP_34_res : std_logic_vector(13 downto 0);
	signal TEMP_35_res : std_logic_vector(13 downto 0);
	signal TEMP_36_res : std_logic_vector(13 downto 0);
	signal TEMP_37_res : std_logic_vector(13 downto 0);
	signal TEMP_38_res : std_logic_vector(13 downto 0);
	signal TEMP_39_res : std_logic_vector(13 downto 0);
	signal TEMP_40_res : std_logic_vector(13 downto 0);
	signal TEMP_41_res : std_logic_vector(13 downto 0);
	signal TEMP_42_res : std_logic_vector(13 downto 0);
	signal TEMP_43_res : std_logic_vector(13 downto 0);
	signal TEMP_44_res : std_logic_vector(13 downto 0);
	signal TEMP_45_res : std_logic_vector(13 downto 0);
	signal TEMP_46_res : std_logic_vector(13 downto 0);
	signal TEMP_47_res : std_logic_vector(13 downto 0);
	signal TEMP_48_res : std_logic_vector(13 downto 0);
	signal TEMP_49_res : std_logic_vector(13 downto 0);
	signal TEMP_50_res : std_logic_vector(13 downto 0);
	signal TEMP_51_res : std_logic_vector(13 downto 0);
	signal TEMP_52_res_0 : std_logic_vector(13 downto 0);
	signal TEMP_52_res_1 : std_logic_vector(13 downto 0);
	signal TEMP_52_res_2 : std_logic_vector(13 downto 0);
	signal TEMP_52_res_3 : std_logic_vector(13 downto 0);
	signal TEMP_52_res_4 : std_logic_vector(13 downto 0);
	signal TEMP_52_res_5 : std_logic_vector(13 downto 0);
	signal TEMP_52_res_6 : std_logic_vector(13 downto 0);
	signal TEMP_52_res_7 : std_logic_vector(13 downto 0);
	signal TEMP_52_res_8 : std_logic_vector(13 downto 0);
	signal TEMP_52_res_9 : std_logic_vector(13 downto 0);
	signal TEMP_52_res_10 : std_logic_vector(13 downto 0);
	signal TEMP_52_res_11 : std_logic_vector(13 downto 0);
	signal TEMP_52_res_12 : std_logic_vector(13 downto 0);
	signal TEMP_52_res_13 : std_logic_vector(13 downto 0);
	signal TEMP_53_res_0 : std_logic_vector(13 downto 0);
	signal TEMP_53_res_1 : std_logic_vector(13 downto 0);
	signal TEMP_53_res_2 : std_logic_vector(13 downto 0);
	signal TEMP_53_res_3 : std_logic_vector(13 downto 0);
	signal TEMP_53_res_4 : std_logic_vector(13 downto 0);
	signal TEMP_53_res_5 : std_logic_vector(13 downto 0);
	signal TEMP_53_res_6 : std_logic_vector(13 downto 0);
	signal TEMP_53_res_7 : std_logic_vector(13 downto 0);
	signal TEMP_53_res_8 : std_logic_vector(13 downto 0);
	signal TEMP_53_res_9 : std_logic_vector(13 downto 0);
	signal TEMP_53_res_10 : std_logic_vector(13 downto 0);
	signal TEMP_53_res_11 : std_logic_vector(13 downto 0);
	signal TEMP_53_res_12 : std_logic_vector(13 downto 0);
	signal TEMP_53_res_13 : std_logic_vector(13 downto 0);
	signal TEMP_54_res_0 : std_logic_vector(13 downto 0);
	signal TEMP_54_res_1 : std_logic_vector(13 downto 0);
	signal TEMP_54_res_2 : std_logic_vector(13 downto 0);
	signal TEMP_54_res_3 : std_logic_vector(13 downto 0);
	signal TEMP_54_res_4 : std_logic_vector(13 downto 0);
	signal TEMP_54_res_5 : std_logic_vector(13 downto 0);
	signal TEMP_54_res_6 : std_logic_vector(13 downto 0);
	signal TEMP_54_res_7 : std_logic_vector(13 downto 0);
	signal TEMP_54_res_8 : std_logic_vector(13 downto 0);
	signal TEMP_54_res_9 : std_logic_vector(13 downto 0);
	signal TEMP_54_res_10 : std_logic_vector(13 downto 0);
	signal TEMP_54_res_11 : std_logic_vector(13 downto 0);
	signal TEMP_54_res_12 : std_logic_vector(13 downto 0);
	signal TEMP_54_res_13 : std_logic_vector(13 downto 0);
begin
	-- WrapSub Begin
	-- WrapSub(loads_sub, ldq_head, ldq_tail, 14)
	loads_sub <= std_logic_vector(unsigned(ldq_head_i) - unsigned(ldq_tail_i)) when ldq_head_i >= ldq_tail_i else
		std_logic_vector(14 - unsigned(ldq_tail_i) + unsigned(ldq_head_i));
	-- WrapAdd End

	-- WrapSub Begin
	-- WrapSub(stores_sub, stq_head, stq_tail, 14)
	stores_sub <= std_logic_vector(unsigned(stq_head_i) - unsigned(stq_tail_i)) when stq_head_i >= stq_tail_i else
		std_logic_vector(14 - unsigned(stq_tail_i) + unsigned(stq_head_i));
	-- WrapAdd End

	empty_loads <= "1110" when ldq_empty_i else loads_sub;
	empty_stores <= "1110" when stq_empty_i else stores_sub;
	group_init_ready_0 <= '1' when ( empty_loads >= "0001" ) and ( empty_stores >= "0001" ) else '0';
	group_init_ready_0_o <= group_init_ready_0;
	group_init_hs_0 <= group_init_ready_0 and group_init_valid_0_i;
	-- Mux1H For Rom Begin
	-- Mux1H(ga_ls_order_rom, group_init_hs)
	-- Loop 0
	TEMP_1_mux_0_0 <= "00000000000000";
	ga_ls_order_rom_0 <= TEMP_1_mux_0_0;
	-- Loop 1
	TEMP_1_mux_1_0 <= "00000000000000";
	ga_ls_order_rom_1 <= TEMP_1_mux_1_0;
	-- Loop 2
	TEMP_1_mux_2_0 <= "00000000000000";
	ga_ls_order_rom_2 <= TEMP_1_mux_2_0;
	-- Loop 3
	TEMP_1_mux_3_0 <= "00000000000000";
	ga_ls_order_rom_3 <= TEMP_1_mux_3_0;
	-- Loop 4
	TEMP_1_mux_4_0 <= "00000000000000";
	ga_ls_order_rom_4 <= TEMP_1_mux_4_0;
	-- Loop 5
	TEMP_1_mux_5_0 <= "00000000000000";
	ga_ls_order_rom_5 <= TEMP_1_mux_5_0;
	-- Loop 6
	TEMP_1_mux_6_0 <= "00000000000000";
	ga_ls_order_rom_6 <= TEMP_1_mux_6_0;
	-- Loop 7
	TEMP_1_mux_7_0 <= "00000000000000";
	ga_ls_order_rom_7 <= TEMP_1_mux_7_0;
	-- Loop 8
	TEMP_1_mux_8_0 <= "00000000000000";
	ga_ls_order_rom_8 <= TEMP_1_mux_8_0;
	-- Loop 9
	TEMP_1_mux_9_0 <= "00000000000000";
	ga_ls_order_rom_9 <= TEMP_1_mux_9_0;
	-- Loop 10
	TEMP_1_mux_10_0 <= "00000000000000";
	ga_ls_order_rom_10 <= TEMP_1_mux_10_0;
	-- Loop 11
	TEMP_1_mux_11_0 <= "00000000000000";
	ga_ls_order_rom_11 <= TEMP_1_mux_11_0;
	-- Loop 12
	TEMP_1_mux_12_0 <= "00000000000000";
	ga_ls_order_rom_12 <= TEMP_1_mux_12_0;
	-- Loop 13
	TEMP_1_mux_13_0 <= "00000000000000";
	ga_ls_order_rom_13 <= TEMP_1_mux_13_0;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(num_loads, group_init_hs)
	TEMP_2_mux_0 <= "0001" when group_init_hs_0 else "0000";
	num_loads <= TEMP_2_mux_0;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(num_stores, group_init_hs)
	TEMP_3_mux_0 <= "0001" when group_init_hs_0 else "0000";
	num_stores <= TEMP_3_mux_0;
	-- Mux1H For Rom End

	num_loads_o <= num_loads;
	num_stores_o <= num_stores;
	ldq_wen_unshifted_0 <= '1' when num_loads > "0000" else '0';
	ldq_wen_unshifted_1 <= '1' when num_loads > "0001" else '0';
	ldq_wen_unshifted_2 <= '1' when num_loads > "0010" else '0';
	ldq_wen_unshifted_3 <= '1' when num_loads > "0011" else '0';
	ldq_wen_unshifted_4 <= '1' when num_loads > "0100" else '0';
	ldq_wen_unshifted_5 <= '1' when num_loads > "0101" else '0';
	ldq_wen_unshifted_6 <= '1' when num_loads > "0110" else '0';
	ldq_wen_unshifted_7 <= '1' when num_loads > "0111" else '0';
	ldq_wen_unshifted_8 <= '1' when num_loads > "1000" else '0';
	ldq_wen_unshifted_9 <= '1' when num_loads > "1001" else '0';
	ldq_wen_unshifted_10 <= '1' when num_loads > "1010" else '0';
	ldq_wen_unshifted_11 <= '1' when num_loads > "1011" else '0';
	ldq_wen_unshifted_12 <= '1' when num_loads > "1100" else '0';
	ldq_wen_unshifted_13 <= '1' when num_loads > "1101" else '0';
	stq_wen_unshifted_0 <= '1' when num_stores > "0000" else '0';
	stq_wen_unshifted_1 <= '1' when num_stores > "0001" else '0';
	stq_wen_unshifted_2 <= '1' when num_stores > "0010" else '0';
	stq_wen_unshifted_3 <= '1' when num_stores > "0011" else '0';
	stq_wen_unshifted_4 <= '1' when num_stores > "0100" else '0';
	stq_wen_unshifted_5 <= '1' when num_stores > "0101" else '0';
	stq_wen_unshifted_6 <= '1' when num_stores > "0110" else '0';
	stq_wen_unshifted_7 <= '1' when num_stores > "0111" else '0';
	stq_wen_unshifted_8 <= '1' when num_stores > "1000" else '0';
	stq_wen_unshifted_9 <= '1' when num_stores > "1001" else '0';
	stq_wen_unshifted_10 <= '1' when num_stores > "1010" else '0';
	stq_wen_unshifted_11 <= '1' when num_stores > "1011" else '0';
	stq_wen_unshifted_12 <= '1' when num_stores > "1100" else '0';
	stq_wen_unshifted_13 <= '1' when num_stores > "1101" else '0';
	-- Shifter Begin
	-- CyclicLeftShift(ldq_wen, ldq_wen_unshifted, ldq_tail)
	TEMP_4_res_0 <= ldq_wen_unshifted_6 when ldq_tail_i(3) else ldq_wen_unshifted_0;
	TEMP_4_res_1 <= ldq_wen_unshifted_7 when ldq_tail_i(3) else ldq_wen_unshifted_1;
	TEMP_4_res_2 <= ldq_wen_unshifted_8 when ldq_tail_i(3) else ldq_wen_unshifted_2;
	TEMP_4_res_3 <= ldq_wen_unshifted_9 when ldq_tail_i(3) else ldq_wen_unshifted_3;
	TEMP_4_res_4 <= ldq_wen_unshifted_10 when ldq_tail_i(3) else ldq_wen_unshifted_4;
	TEMP_4_res_5 <= ldq_wen_unshifted_11 when ldq_tail_i(3) else ldq_wen_unshifted_5;
	TEMP_4_res_6 <= ldq_wen_unshifted_12 when ldq_tail_i(3) else ldq_wen_unshifted_6;
	TEMP_4_res_7 <= ldq_wen_unshifted_13 when ldq_tail_i(3) else ldq_wen_unshifted_7;
	TEMP_4_res_8 <= ldq_wen_unshifted_0 when ldq_tail_i(3) else ldq_wen_unshifted_8;
	TEMP_4_res_9 <= ldq_wen_unshifted_1 when ldq_tail_i(3) else ldq_wen_unshifted_9;
	TEMP_4_res_10 <= ldq_wen_unshifted_2 when ldq_tail_i(3) else ldq_wen_unshifted_10;
	TEMP_4_res_11 <= ldq_wen_unshifted_3 when ldq_tail_i(3) else ldq_wen_unshifted_11;
	TEMP_4_res_12 <= ldq_wen_unshifted_4 when ldq_tail_i(3) else ldq_wen_unshifted_12;
	TEMP_4_res_13 <= ldq_wen_unshifted_5 when ldq_tail_i(3) else ldq_wen_unshifted_13;
	-- Layer End
	TEMP_5_res_0 <= TEMP_4_res_10 when ldq_tail_i(2) else TEMP_4_res_0;
	TEMP_5_res_1 <= TEMP_4_res_11 when ldq_tail_i(2) else TEMP_4_res_1;
	TEMP_5_res_2 <= TEMP_4_res_12 when ldq_tail_i(2) else TEMP_4_res_2;
	TEMP_5_res_3 <= TEMP_4_res_13 when ldq_tail_i(2) else TEMP_4_res_3;
	TEMP_5_res_4 <= TEMP_4_res_0 when ldq_tail_i(2) else TEMP_4_res_4;
	TEMP_5_res_5 <= TEMP_4_res_1 when ldq_tail_i(2) else TEMP_4_res_5;
	TEMP_5_res_6 <= TEMP_4_res_2 when ldq_tail_i(2) else TEMP_4_res_6;
	TEMP_5_res_7 <= TEMP_4_res_3 when ldq_tail_i(2) else TEMP_4_res_7;
	TEMP_5_res_8 <= TEMP_4_res_4 when ldq_tail_i(2) else TEMP_4_res_8;
	TEMP_5_res_9 <= TEMP_4_res_5 when ldq_tail_i(2) else TEMP_4_res_9;
	TEMP_5_res_10 <= TEMP_4_res_6 when ldq_tail_i(2) else TEMP_4_res_10;
	TEMP_5_res_11 <= TEMP_4_res_7 when ldq_tail_i(2) else TEMP_4_res_11;
	TEMP_5_res_12 <= TEMP_4_res_8 when ldq_tail_i(2) else TEMP_4_res_12;
	TEMP_5_res_13 <= TEMP_4_res_9 when ldq_tail_i(2) else TEMP_4_res_13;
	-- Layer End
	TEMP_6_res_0 <= TEMP_5_res_12 when ldq_tail_i(1) else TEMP_5_res_0;
	TEMP_6_res_1 <= TEMP_5_res_13 when ldq_tail_i(1) else TEMP_5_res_1;
	TEMP_6_res_2 <= TEMP_5_res_0 when ldq_tail_i(1) else TEMP_5_res_2;
	TEMP_6_res_3 <= TEMP_5_res_1 when ldq_tail_i(1) else TEMP_5_res_3;
	TEMP_6_res_4 <= TEMP_5_res_2 when ldq_tail_i(1) else TEMP_5_res_4;
	TEMP_6_res_5 <= TEMP_5_res_3 when ldq_tail_i(1) else TEMP_5_res_5;
	TEMP_6_res_6 <= TEMP_5_res_4 when ldq_tail_i(1) else TEMP_5_res_6;
	TEMP_6_res_7 <= TEMP_5_res_5 when ldq_tail_i(1) else TEMP_5_res_7;
	TEMP_6_res_8 <= TEMP_5_res_6 when ldq_tail_i(1) else TEMP_5_res_8;
	TEMP_6_res_9 <= TEMP_5_res_7 when ldq_tail_i(1) else TEMP_5_res_9;
	TEMP_6_res_10 <= TEMP_5_res_8 when ldq_tail_i(1) else TEMP_5_res_10;
	TEMP_6_res_11 <= TEMP_5_res_9 when ldq_tail_i(1) else TEMP_5_res_11;
	TEMP_6_res_12 <= TEMP_5_res_10 when ldq_tail_i(1) else TEMP_5_res_12;
	TEMP_6_res_13 <= TEMP_5_res_11 when ldq_tail_i(1) else TEMP_5_res_13;
	-- Layer End
	ldq_wen_0_o <= TEMP_6_res_13 when ldq_tail_i(0) else TEMP_6_res_0;
	ldq_wen_1_o <= TEMP_6_res_0 when ldq_tail_i(0) else TEMP_6_res_1;
	ldq_wen_2_o <= TEMP_6_res_1 when ldq_tail_i(0) else TEMP_6_res_2;
	ldq_wen_3_o <= TEMP_6_res_2 when ldq_tail_i(0) else TEMP_6_res_3;
	ldq_wen_4_o <= TEMP_6_res_3 when ldq_tail_i(0) else TEMP_6_res_4;
	ldq_wen_5_o <= TEMP_6_res_4 when ldq_tail_i(0) else TEMP_6_res_5;
	ldq_wen_6_o <= TEMP_6_res_5 when ldq_tail_i(0) else TEMP_6_res_6;
	ldq_wen_7_o <= TEMP_6_res_6 when ldq_tail_i(0) else TEMP_6_res_7;
	ldq_wen_8_o <= TEMP_6_res_7 when ldq_tail_i(0) else TEMP_6_res_8;
	ldq_wen_9_o <= TEMP_6_res_8 when ldq_tail_i(0) else TEMP_6_res_9;
	ldq_wen_10_o <= TEMP_6_res_9 when ldq_tail_i(0) else TEMP_6_res_10;
	ldq_wen_11_o <= TEMP_6_res_10 when ldq_tail_i(0) else TEMP_6_res_11;
	ldq_wen_12_o <= TEMP_6_res_11 when ldq_tail_i(0) else TEMP_6_res_12;
	ldq_wen_13_o <= TEMP_6_res_12 when ldq_tail_i(0) else TEMP_6_res_13;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(stq_wen, stq_wen_unshifted, stq_tail)
	TEMP_7_res_0 <= stq_wen_unshifted_6 when stq_tail_i(3) else stq_wen_unshifted_0;
	TEMP_7_res_1 <= stq_wen_unshifted_7 when stq_tail_i(3) else stq_wen_unshifted_1;
	TEMP_7_res_2 <= stq_wen_unshifted_8 when stq_tail_i(3) else stq_wen_unshifted_2;
	TEMP_7_res_3 <= stq_wen_unshifted_9 when stq_tail_i(3) else stq_wen_unshifted_3;
	TEMP_7_res_4 <= stq_wen_unshifted_10 when stq_tail_i(3) else stq_wen_unshifted_4;
	TEMP_7_res_5 <= stq_wen_unshifted_11 when stq_tail_i(3) else stq_wen_unshifted_5;
	TEMP_7_res_6 <= stq_wen_unshifted_12 when stq_tail_i(3) else stq_wen_unshifted_6;
	TEMP_7_res_7 <= stq_wen_unshifted_13 when stq_tail_i(3) else stq_wen_unshifted_7;
	TEMP_7_res_8 <= stq_wen_unshifted_0 when stq_tail_i(3) else stq_wen_unshifted_8;
	TEMP_7_res_9 <= stq_wen_unshifted_1 when stq_tail_i(3) else stq_wen_unshifted_9;
	TEMP_7_res_10 <= stq_wen_unshifted_2 when stq_tail_i(3) else stq_wen_unshifted_10;
	TEMP_7_res_11 <= stq_wen_unshifted_3 when stq_tail_i(3) else stq_wen_unshifted_11;
	TEMP_7_res_12 <= stq_wen_unshifted_4 when stq_tail_i(3) else stq_wen_unshifted_12;
	TEMP_7_res_13 <= stq_wen_unshifted_5 when stq_tail_i(3) else stq_wen_unshifted_13;
	-- Layer End
	TEMP_8_res_0 <= TEMP_7_res_10 when stq_tail_i(2) else TEMP_7_res_0;
	TEMP_8_res_1 <= TEMP_7_res_11 when stq_tail_i(2) else TEMP_7_res_1;
	TEMP_8_res_2 <= TEMP_7_res_12 when stq_tail_i(2) else TEMP_7_res_2;
	TEMP_8_res_3 <= TEMP_7_res_13 when stq_tail_i(2) else TEMP_7_res_3;
	TEMP_8_res_4 <= TEMP_7_res_0 when stq_tail_i(2) else TEMP_7_res_4;
	TEMP_8_res_5 <= TEMP_7_res_1 when stq_tail_i(2) else TEMP_7_res_5;
	TEMP_8_res_6 <= TEMP_7_res_2 when stq_tail_i(2) else TEMP_7_res_6;
	TEMP_8_res_7 <= TEMP_7_res_3 when stq_tail_i(2) else TEMP_7_res_7;
	TEMP_8_res_8 <= TEMP_7_res_4 when stq_tail_i(2) else TEMP_7_res_8;
	TEMP_8_res_9 <= TEMP_7_res_5 when stq_tail_i(2) else TEMP_7_res_9;
	TEMP_8_res_10 <= TEMP_7_res_6 when stq_tail_i(2) else TEMP_7_res_10;
	TEMP_8_res_11 <= TEMP_7_res_7 when stq_tail_i(2) else TEMP_7_res_11;
	TEMP_8_res_12 <= TEMP_7_res_8 when stq_tail_i(2) else TEMP_7_res_12;
	TEMP_8_res_13 <= TEMP_7_res_9 when stq_tail_i(2) else TEMP_7_res_13;
	-- Layer End
	TEMP_9_res_0 <= TEMP_8_res_12 when stq_tail_i(1) else TEMP_8_res_0;
	TEMP_9_res_1 <= TEMP_8_res_13 when stq_tail_i(1) else TEMP_8_res_1;
	TEMP_9_res_2 <= TEMP_8_res_0 when stq_tail_i(1) else TEMP_8_res_2;
	TEMP_9_res_3 <= TEMP_8_res_1 when stq_tail_i(1) else TEMP_8_res_3;
	TEMP_9_res_4 <= TEMP_8_res_2 when stq_tail_i(1) else TEMP_8_res_4;
	TEMP_9_res_5 <= TEMP_8_res_3 when stq_tail_i(1) else TEMP_8_res_5;
	TEMP_9_res_6 <= TEMP_8_res_4 when stq_tail_i(1) else TEMP_8_res_6;
	TEMP_9_res_7 <= TEMP_8_res_5 when stq_tail_i(1) else TEMP_8_res_7;
	TEMP_9_res_8 <= TEMP_8_res_6 when stq_tail_i(1) else TEMP_8_res_8;
	TEMP_9_res_9 <= TEMP_8_res_7 when stq_tail_i(1) else TEMP_8_res_9;
	TEMP_9_res_10 <= TEMP_8_res_8 when stq_tail_i(1) else TEMP_8_res_10;
	TEMP_9_res_11 <= TEMP_8_res_9 when stq_tail_i(1) else TEMP_8_res_11;
	TEMP_9_res_12 <= TEMP_8_res_10 when stq_tail_i(1) else TEMP_8_res_12;
	TEMP_9_res_13 <= TEMP_8_res_11 when stq_tail_i(1) else TEMP_8_res_13;
	-- Layer End
	stq_wen_0_o <= TEMP_9_res_13 when stq_tail_i(0) else TEMP_9_res_0;
	stq_wen_1_o <= TEMP_9_res_0 when stq_tail_i(0) else TEMP_9_res_1;
	stq_wen_2_o <= TEMP_9_res_1 when stq_tail_i(0) else TEMP_9_res_2;
	stq_wen_3_o <= TEMP_9_res_2 when stq_tail_i(0) else TEMP_9_res_3;
	stq_wen_4_o <= TEMP_9_res_3 when stq_tail_i(0) else TEMP_9_res_4;
	stq_wen_5_o <= TEMP_9_res_4 when stq_tail_i(0) else TEMP_9_res_5;
	stq_wen_6_o <= TEMP_9_res_5 when stq_tail_i(0) else TEMP_9_res_6;
	stq_wen_7_o <= TEMP_9_res_6 when stq_tail_i(0) else TEMP_9_res_7;
	stq_wen_8_o <= TEMP_9_res_7 when stq_tail_i(0) else TEMP_9_res_8;
	stq_wen_9_o <= TEMP_9_res_8 when stq_tail_i(0) else TEMP_9_res_9;
	stq_wen_10_o <= TEMP_9_res_9 when stq_tail_i(0) else TEMP_9_res_10;
	stq_wen_11_o <= TEMP_9_res_10 when stq_tail_i(0) else TEMP_9_res_11;
	stq_wen_12_o <= TEMP_9_res_11 when stq_tail_i(0) else TEMP_9_res_12;
	stq_wen_13_o <= TEMP_9_res_12 when stq_tail_i(0) else TEMP_9_res_13;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_0, ga_ls_order_rom_0, stq_tail)
	TEMP_10_res(0) <= ga_ls_order_rom_0(6) when stq_tail_i(3) else ga_ls_order_rom_0(0);
	TEMP_10_res(1) <= ga_ls_order_rom_0(7) when stq_tail_i(3) else ga_ls_order_rom_0(1);
	TEMP_10_res(2) <= ga_ls_order_rom_0(8) when stq_tail_i(3) else ga_ls_order_rom_0(2);
	TEMP_10_res(3) <= ga_ls_order_rom_0(9) when stq_tail_i(3) else ga_ls_order_rom_0(3);
	TEMP_10_res(4) <= ga_ls_order_rom_0(10) when stq_tail_i(3) else ga_ls_order_rom_0(4);
	TEMP_10_res(5) <= ga_ls_order_rom_0(11) when stq_tail_i(3) else ga_ls_order_rom_0(5);
	TEMP_10_res(6) <= ga_ls_order_rom_0(12) when stq_tail_i(3) else ga_ls_order_rom_0(6);
	TEMP_10_res(7) <= ga_ls_order_rom_0(13) when stq_tail_i(3) else ga_ls_order_rom_0(7);
	TEMP_10_res(8) <= ga_ls_order_rom_0(0) when stq_tail_i(3) else ga_ls_order_rom_0(8);
	TEMP_10_res(9) <= ga_ls_order_rom_0(1) when stq_tail_i(3) else ga_ls_order_rom_0(9);
	TEMP_10_res(10) <= ga_ls_order_rom_0(2) when stq_tail_i(3) else ga_ls_order_rom_0(10);
	TEMP_10_res(11) <= ga_ls_order_rom_0(3) when stq_tail_i(3) else ga_ls_order_rom_0(11);
	TEMP_10_res(12) <= ga_ls_order_rom_0(4) when stq_tail_i(3) else ga_ls_order_rom_0(12);
	TEMP_10_res(13) <= ga_ls_order_rom_0(5) when stq_tail_i(3) else ga_ls_order_rom_0(13);
	-- Layer End
	TEMP_11_res(0) <= TEMP_10_res(10) when stq_tail_i(2) else TEMP_10_res(0);
	TEMP_11_res(1) <= TEMP_10_res(11) when stq_tail_i(2) else TEMP_10_res(1);
	TEMP_11_res(2) <= TEMP_10_res(12) when stq_tail_i(2) else TEMP_10_res(2);
	TEMP_11_res(3) <= TEMP_10_res(13) when stq_tail_i(2) else TEMP_10_res(3);
	TEMP_11_res(4) <= TEMP_10_res(0) when stq_tail_i(2) else TEMP_10_res(4);
	TEMP_11_res(5) <= TEMP_10_res(1) when stq_tail_i(2) else TEMP_10_res(5);
	TEMP_11_res(6) <= TEMP_10_res(2) when stq_tail_i(2) else TEMP_10_res(6);
	TEMP_11_res(7) <= TEMP_10_res(3) when stq_tail_i(2) else TEMP_10_res(7);
	TEMP_11_res(8) <= TEMP_10_res(4) when stq_tail_i(2) else TEMP_10_res(8);
	TEMP_11_res(9) <= TEMP_10_res(5) when stq_tail_i(2) else TEMP_10_res(9);
	TEMP_11_res(10) <= TEMP_10_res(6) when stq_tail_i(2) else TEMP_10_res(10);
	TEMP_11_res(11) <= TEMP_10_res(7) when stq_tail_i(2) else TEMP_10_res(11);
	TEMP_11_res(12) <= TEMP_10_res(8) when stq_tail_i(2) else TEMP_10_res(12);
	TEMP_11_res(13) <= TEMP_10_res(9) when stq_tail_i(2) else TEMP_10_res(13);
	-- Layer End
	TEMP_12_res(0) <= TEMP_11_res(12) when stq_tail_i(1) else TEMP_11_res(0);
	TEMP_12_res(1) <= TEMP_11_res(13) when stq_tail_i(1) else TEMP_11_res(1);
	TEMP_12_res(2) <= TEMP_11_res(0) when stq_tail_i(1) else TEMP_11_res(2);
	TEMP_12_res(3) <= TEMP_11_res(1) when stq_tail_i(1) else TEMP_11_res(3);
	TEMP_12_res(4) <= TEMP_11_res(2) when stq_tail_i(1) else TEMP_11_res(4);
	TEMP_12_res(5) <= TEMP_11_res(3) when stq_tail_i(1) else TEMP_11_res(5);
	TEMP_12_res(6) <= TEMP_11_res(4) when stq_tail_i(1) else TEMP_11_res(6);
	TEMP_12_res(7) <= TEMP_11_res(5) when stq_tail_i(1) else TEMP_11_res(7);
	TEMP_12_res(8) <= TEMP_11_res(6) when stq_tail_i(1) else TEMP_11_res(8);
	TEMP_12_res(9) <= TEMP_11_res(7) when stq_tail_i(1) else TEMP_11_res(9);
	TEMP_12_res(10) <= TEMP_11_res(8) when stq_tail_i(1) else TEMP_11_res(10);
	TEMP_12_res(11) <= TEMP_11_res(9) when stq_tail_i(1) else TEMP_11_res(11);
	TEMP_12_res(12) <= TEMP_11_res(10) when stq_tail_i(1) else TEMP_11_res(12);
	TEMP_12_res(13) <= TEMP_11_res(11) when stq_tail_i(1) else TEMP_11_res(13);
	-- Layer End
	ga_ls_order_temp_0(0) <= TEMP_12_res(13) when stq_tail_i(0) else TEMP_12_res(0);
	ga_ls_order_temp_0(1) <= TEMP_12_res(0) when stq_tail_i(0) else TEMP_12_res(1);
	ga_ls_order_temp_0(2) <= TEMP_12_res(1) when stq_tail_i(0) else TEMP_12_res(2);
	ga_ls_order_temp_0(3) <= TEMP_12_res(2) when stq_tail_i(0) else TEMP_12_res(3);
	ga_ls_order_temp_0(4) <= TEMP_12_res(3) when stq_tail_i(0) else TEMP_12_res(4);
	ga_ls_order_temp_0(5) <= TEMP_12_res(4) when stq_tail_i(0) else TEMP_12_res(5);
	ga_ls_order_temp_0(6) <= TEMP_12_res(5) when stq_tail_i(0) else TEMP_12_res(6);
	ga_ls_order_temp_0(7) <= TEMP_12_res(6) when stq_tail_i(0) else TEMP_12_res(7);
	ga_ls_order_temp_0(8) <= TEMP_12_res(7) when stq_tail_i(0) else TEMP_12_res(8);
	ga_ls_order_temp_0(9) <= TEMP_12_res(8) when stq_tail_i(0) else TEMP_12_res(9);
	ga_ls_order_temp_0(10) <= TEMP_12_res(9) when stq_tail_i(0) else TEMP_12_res(10);
	ga_ls_order_temp_0(11) <= TEMP_12_res(10) when stq_tail_i(0) else TEMP_12_res(11);
	ga_ls_order_temp_0(12) <= TEMP_12_res(11) when stq_tail_i(0) else TEMP_12_res(12);
	ga_ls_order_temp_0(13) <= TEMP_12_res(12) when stq_tail_i(0) else TEMP_12_res(13);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_1, ga_ls_order_rom_1, stq_tail)
	TEMP_13_res(0) <= ga_ls_order_rom_1(6) when stq_tail_i(3) else ga_ls_order_rom_1(0);
	TEMP_13_res(1) <= ga_ls_order_rom_1(7) when stq_tail_i(3) else ga_ls_order_rom_1(1);
	TEMP_13_res(2) <= ga_ls_order_rom_1(8) when stq_tail_i(3) else ga_ls_order_rom_1(2);
	TEMP_13_res(3) <= ga_ls_order_rom_1(9) when stq_tail_i(3) else ga_ls_order_rom_1(3);
	TEMP_13_res(4) <= ga_ls_order_rom_1(10) when stq_tail_i(3) else ga_ls_order_rom_1(4);
	TEMP_13_res(5) <= ga_ls_order_rom_1(11) when stq_tail_i(3) else ga_ls_order_rom_1(5);
	TEMP_13_res(6) <= ga_ls_order_rom_1(12) when stq_tail_i(3) else ga_ls_order_rom_1(6);
	TEMP_13_res(7) <= ga_ls_order_rom_1(13) when stq_tail_i(3) else ga_ls_order_rom_1(7);
	TEMP_13_res(8) <= ga_ls_order_rom_1(0) when stq_tail_i(3) else ga_ls_order_rom_1(8);
	TEMP_13_res(9) <= ga_ls_order_rom_1(1) when stq_tail_i(3) else ga_ls_order_rom_1(9);
	TEMP_13_res(10) <= ga_ls_order_rom_1(2) when stq_tail_i(3) else ga_ls_order_rom_1(10);
	TEMP_13_res(11) <= ga_ls_order_rom_1(3) when stq_tail_i(3) else ga_ls_order_rom_1(11);
	TEMP_13_res(12) <= ga_ls_order_rom_1(4) when stq_tail_i(3) else ga_ls_order_rom_1(12);
	TEMP_13_res(13) <= ga_ls_order_rom_1(5) when stq_tail_i(3) else ga_ls_order_rom_1(13);
	-- Layer End
	TEMP_14_res(0) <= TEMP_13_res(10) when stq_tail_i(2) else TEMP_13_res(0);
	TEMP_14_res(1) <= TEMP_13_res(11) when stq_tail_i(2) else TEMP_13_res(1);
	TEMP_14_res(2) <= TEMP_13_res(12) when stq_tail_i(2) else TEMP_13_res(2);
	TEMP_14_res(3) <= TEMP_13_res(13) when stq_tail_i(2) else TEMP_13_res(3);
	TEMP_14_res(4) <= TEMP_13_res(0) when stq_tail_i(2) else TEMP_13_res(4);
	TEMP_14_res(5) <= TEMP_13_res(1) when stq_tail_i(2) else TEMP_13_res(5);
	TEMP_14_res(6) <= TEMP_13_res(2) when stq_tail_i(2) else TEMP_13_res(6);
	TEMP_14_res(7) <= TEMP_13_res(3) when stq_tail_i(2) else TEMP_13_res(7);
	TEMP_14_res(8) <= TEMP_13_res(4) when stq_tail_i(2) else TEMP_13_res(8);
	TEMP_14_res(9) <= TEMP_13_res(5) when stq_tail_i(2) else TEMP_13_res(9);
	TEMP_14_res(10) <= TEMP_13_res(6) when stq_tail_i(2) else TEMP_13_res(10);
	TEMP_14_res(11) <= TEMP_13_res(7) when stq_tail_i(2) else TEMP_13_res(11);
	TEMP_14_res(12) <= TEMP_13_res(8) when stq_tail_i(2) else TEMP_13_res(12);
	TEMP_14_res(13) <= TEMP_13_res(9) when stq_tail_i(2) else TEMP_13_res(13);
	-- Layer End
	TEMP_15_res(0) <= TEMP_14_res(12) when stq_tail_i(1) else TEMP_14_res(0);
	TEMP_15_res(1) <= TEMP_14_res(13) when stq_tail_i(1) else TEMP_14_res(1);
	TEMP_15_res(2) <= TEMP_14_res(0) when stq_tail_i(1) else TEMP_14_res(2);
	TEMP_15_res(3) <= TEMP_14_res(1) when stq_tail_i(1) else TEMP_14_res(3);
	TEMP_15_res(4) <= TEMP_14_res(2) when stq_tail_i(1) else TEMP_14_res(4);
	TEMP_15_res(5) <= TEMP_14_res(3) when stq_tail_i(1) else TEMP_14_res(5);
	TEMP_15_res(6) <= TEMP_14_res(4) when stq_tail_i(1) else TEMP_14_res(6);
	TEMP_15_res(7) <= TEMP_14_res(5) when stq_tail_i(1) else TEMP_14_res(7);
	TEMP_15_res(8) <= TEMP_14_res(6) when stq_tail_i(1) else TEMP_14_res(8);
	TEMP_15_res(9) <= TEMP_14_res(7) when stq_tail_i(1) else TEMP_14_res(9);
	TEMP_15_res(10) <= TEMP_14_res(8) when stq_tail_i(1) else TEMP_14_res(10);
	TEMP_15_res(11) <= TEMP_14_res(9) when stq_tail_i(1) else TEMP_14_res(11);
	TEMP_15_res(12) <= TEMP_14_res(10) when stq_tail_i(1) else TEMP_14_res(12);
	TEMP_15_res(13) <= TEMP_14_res(11) when stq_tail_i(1) else TEMP_14_res(13);
	-- Layer End
	ga_ls_order_temp_1(0) <= TEMP_15_res(13) when stq_tail_i(0) else TEMP_15_res(0);
	ga_ls_order_temp_1(1) <= TEMP_15_res(0) when stq_tail_i(0) else TEMP_15_res(1);
	ga_ls_order_temp_1(2) <= TEMP_15_res(1) when stq_tail_i(0) else TEMP_15_res(2);
	ga_ls_order_temp_1(3) <= TEMP_15_res(2) when stq_tail_i(0) else TEMP_15_res(3);
	ga_ls_order_temp_1(4) <= TEMP_15_res(3) when stq_tail_i(0) else TEMP_15_res(4);
	ga_ls_order_temp_1(5) <= TEMP_15_res(4) when stq_tail_i(0) else TEMP_15_res(5);
	ga_ls_order_temp_1(6) <= TEMP_15_res(5) when stq_tail_i(0) else TEMP_15_res(6);
	ga_ls_order_temp_1(7) <= TEMP_15_res(6) when stq_tail_i(0) else TEMP_15_res(7);
	ga_ls_order_temp_1(8) <= TEMP_15_res(7) when stq_tail_i(0) else TEMP_15_res(8);
	ga_ls_order_temp_1(9) <= TEMP_15_res(8) when stq_tail_i(0) else TEMP_15_res(9);
	ga_ls_order_temp_1(10) <= TEMP_15_res(9) when stq_tail_i(0) else TEMP_15_res(10);
	ga_ls_order_temp_1(11) <= TEMP_15_res(10) when stq_tail_i(0) else TEMP_15_res(11);
	ga_ls_order_temp_1(12) <= TEMP_15_res(11) when stq_tail_i(0) else TEMP_15_res(12);
	ga_ls_order_temp_1(13) <= TEMP_15_res(12) when stq_tail_i(0) else TEMP_15_res(13);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_2, ga_ls_order_rom_2, stq_tail)
	TEMP_16_res(0) <= ga_ls_order_rom_2(6) when stq_tail_i(3) else ga_ls_order_rom_2(0);
	TEMP_16_res(1) <= ga_ls_order_rom_2(7) when stq_tail_i(3) else ga_ls_order_rom_2(1);
	TEMP_16_res(2) <= ga_ls_order_rom_2(8) when stq_tail_i(3) else ga_ls_order_rom_2(2);
	TEMP_16_res(3) <= ga_ls_order_rom_2(9) when stq_tail_i(3) else ga_ls_order_rom_2(3);
	TEMP_16_res(4) <= ga_ls_order_rom_2(10) when stq_tail_i(3) else ga_ls_order_rom_2(4);
	TEMP_16_res(5) <= ga_ls_order_rom_2(11) when stq_tail_i(3) else ga_ls_order_rom_2(5);
	TEMP_16_res(6) <= ga_ls_order_rom_2(12) when stq_tail_i(3) else ga_ls_order_rom_2(6);
	TEMP_16_res(7) <= ga_ls_order_rom_2(13) when stq_tail_i(3) else ga_ls_order_rom_2(7);
	TEMP_16_res(8) <= ga_ls_order_rom_2(0) when stq_tail_i(3) else ga_ls_order_rom_2(8);
	TEMP_16_res(9) <= ga_ls_order_rom_2(1) when stq_tail_i(3) else ga_ls_order_rom_2(9);
	TEMP_16_res(10) <= ga_ls_order_rom_2(2) when stq_tail_i(3) else ga_ls_order_rom_2(10);
	TEMP_16_res(11) <= ga_ls_order_rom_2(3) when stq_tail_i(3) else ga_ls_order_rom_2(11);
	TEMP_16_res(12) <= ga_ls_order_rom_2(4) when stq_tail_i(3) else ga_ls_order_rom_2(12);
	TEMP_16_res(13) <= ga_ls_order_rom_2(5) when stq_tail_i(3) else ga_ls_order_rom_2(13);
	-- Layer End
	TEMP_17_res(0) <= TEMP_16_res(10) when stq_tail_i(2) else TEMP_16_res(0);
	TEMP_17_res(1) <= TEMP_16_res(11) when stq_tail_i(2) else TEMP_16_res(1);
	TEMP_17_res(2) <= TEMP_16_res(12) when stq_tail_i(2) else TEMP_16_res(2);
	TEMP_17_res(3) <= TEMP_16_res(13) when stq_tail_i(2) else TEMP_16_res(3);
	TEMP_17_res(4) <= TEMP_16_res(0) when stq_tail_i(2) else TEMP_16_res(4);
	TEMP_17_res(5) <= TEMP_16_res(1) when stq_tail_i(2) else TEMP_16_res(5);
	TEMP_17_res(6) <= TEMP_16_res(2) when stq_tail_i(2) else TEMP_16_res(6);
	TEMP_17_res(7) <= TEMP_16_res(3) when stq_tail_i(2) else TEMP_16_res(7);
	TEMP_17_res(8) <= TEMP_16_res(4) when stq_tail_i(2) else TEMP_16_res(8);
	TEMP_17_res(9) <= TEMP_16_res(5) when stq_tail_i(2) else TEMP_16_res(9);
	TEMP_17_res(10) <= TEMP_16_res(6) when stq_tail_i(2) else TEMP_16_res(10);
	TEMP_17_res(11) <= TEMP_16_res(7) when stq_tail_i(2) else TEMP_16_res(11);
	TEMP_17_res(12) <= TEMP_16_res(8) when stq_tail_i(2) else TEMP_16_res(12);
	TEMP_17_res(13) <= TEMP_16_res(9) when stq_tail_i(2) else TEMP_16_res(13);
	-- Layer End
	TEMP_18_res(0) <= TEMP_17_res(12) when stq_tail_i(1) else TEMP_17_res(0);
	TEMP_18_res(1) <= TEMP_17_res(13) when stq_tail_i(1) else TEMP_17_res(1);
	TEMP_18_res(2) <= TEMP_17_res(0) when stq_tail_i(1) else TEMP_17_res(2);
	TEMP_18_res(3) <= TEMP_17_res(1) when stq_tail_i(1) else TEMP_17_res(3);
	TEMP_18_res(4) <= TEMP_17_res(2) when stq_tail_i(1) else TEMP_17_res(4);
	TEMP_18_res(5) <= TEMP_17_res(3) when stq_tail_i(1) else TEMP_17_res(5);
	TEMP_18_res(6) <= TEMP_17_res(4) when stq_tail_i(1) else TEMP_17_res(6);
	TEMP_18_res(7) <= TEMP_17_res(5) when stq_tail_i(1) else TEMP_17_res(7);
	TEMP_18_res(8) <= TEMP_17_res(6) when stq_tail_i(1) else TEMP_17_res(8);
	TEMP_18_res(9) <= TEMP_17_res(7) when stq_tail_i(1) else TEMP_17_res(9);
	TEMP_18_res(10) <= TEMP_17_res(8) when stq_tail_i(1) else TEMP_17_res(10);
	TEMP_18_res(11) <= TEMP_17_res(9) when stq_tail_i(1) else TEMP_17_res(11);
	TEMP_18_res(12) <= TEMP_17_res(10) when stq_tail_i(1) else TEMP_17_res(12);
	TEMP_18_res(13) <= TEMP_17_res(11) when stq_tail_i(1) else TEMP_17_res(13);
	-- Layer End
	ga_ls_order_temp_2(0) <= TEMP_18_res(13) when stq_tail_i(0) else TEMP_18_res(0);
	ga_ls_order_temp_2(1) <= TEMP_18_res(0) when stq_tail_i(0) else TEMP_18_res(1);
	ga_ls_order_temp_2(2) <= TEMP_18_res(1) when stq_tail_i(0) else TEMP_18_res(2);
	ga_ls_order_temp_2(3) <= TEMP_18_res(2) when stq_tail_i(0) else TEMP_18_res(3);
	ga_ls_order_temp_2(4) <= TEMP_18_res(3) when stq_tail_i(0) else TEMP_18_res(4);
	ga_ls_order_temp_2(5) <= TEMP_18_res(4) when stq_tail_i(0) else TEMP_18_res(5);
	ga_ls_order_temp_2(6) <= TEMP_18_res(5) when stq_tail_i(0) else TEMP_18_res(6);
	ga_ls_order_temp_2(7) <= TEMP_18_res(6) when stq_tail_i(0) else TEMP_18_res(7);
	ga_ls_order_temp_2(8) <= TEMP_18_res(7) when stq_tail_i(0) else TEMP_18_res(8);
	ga_ls_order_temp_2(9) <= TEMP_18_res(8) when stq_tail_i(0) else TEMP_18_res(9);
	ga_ls_order_temp_2(10) <= TEMP_18_res(9) when stq_tail_i(0) else TEMP_18_res(10);
	ga_ls_order_temp_2(11) <= TEMP_18_res(10) when stq_tail_i(0) else TEMP_18_res(11);
	ga_ls_order_temp_2(12) <= TEMP_18_res(11) when stq_tail_i(0) else TEMP_18_res(12);
	ga_ls_order_temp_2(13) <= TEMP_18_res(12) when stq_tail_i(0) else TEMP_18_res(13);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_3, ga_ls_order_rom_3, stq_tail)
	TEMP_19_res(0) <= ga_ls_order_rom_3(6) when stq_tail_i(3) else ga_ls_order_rom_3(0);
	TEMP_19_res(1) <= ga_ls_order_rom_3(7) when stq_tail_i(3) else ga_ls_order_rom_3(1);
	TEMP_19_res(2) <= ga_ls_order_rom_3(8) when stq_tail_i(3) else ga_ls_order_rom_3(2);
	TEMP_19_res(3) <= ga_ls_order_rom_3(9) when stq_tail_i(3) else ga_ls_order_rom_3(3);
	TEMP_19_res(4) <= ga_ls_order_rom_3(10) when stq_tail_i(3) else ga_ls_order_rom_3(4);
	TEMP_19_res(5) <= ga_ls_order_rom_3(11) when stq_tail_i(3) else ga_ls_order_rom_3(5);
	TEMP_19_res(6) <= ga_ls_order_rom_3(12) when stq_tail_i(3) else ga_ls_order_rom_3(6);
	TEMP_19_res(7) <= ga_ls_order_rom_3(13) when stq_tail_i(3) else ga_ls_order_rom_3(7);
	TEMP_19_res(8) <= ga_ls_order_rom_3(0) when stq_tail_i(3) else ga_ls_order_rom_3(8);
	TEMP_19_res(9) <= ga_ls_order_rom_3(1) when stq_tail_i(3) else ga_ls_order_rom_3(9);
	TEMP_19_res(10) <= ga_ls_order_rom_3(2) when stq_tail_i(3) else ga_ls_order_rom_3(10);
	TEMP_19_res(11) <= ga_ls_order_rom_3(3) when stq_tail_i(3) else ga_ls_order_rom_3(11);
	TEMP_19_res(12) <= ga_ls_order_rom_3(4) when stq_tail_i(3) else ga_ls_order_rom_3(12);
	TEMP_19_res(13) <= ga_ls_order_rom_3(5) when stq_tail_i(3) else ga_ls_order_rom_3(13);
	-- Layer End
	TEMP_20_res(0) <= TEMP_19_res(10) when stq_tail_i(2) else TEMP_19_res(0);
	TEMP_20_res(1) <= TEMP_19_res(11) when stq_tail_i(2) else TEMP_19_res(1);
	TEMP_20_res(2) <= TEMP_19_res(12) when stq_tail_i(2) else TEMP_19_res(2);
	TEMP_20_res(3) <= TEMP_19_res(13) when stq_tail_i(2) else TEMP_19_res(3);
	TEMP_20_res(4) <= TEMP_19_res(0) when stq_tail_i(2) else TEMP_19_res(4);
	TEMP_20_res(5) <= TEMP_19_res(1) when stq_tail_i(2) else TEMP_19_res(5);
	TEMP_20_res(6) <= TEMP_19_res(2) when stq_tail_i(2) else TEMP_19_res(6);
	TEMP_20_res(7) <= TEMP_19_res(3) when stq_tail_i(2) else TEMP_19_res(7);
	TEMP_20_res(8) <= TEMP_19_res(4) when stq_tail_i(2) else TEMP_19_res(8);
	TEMP_20_res(9) <= TEMP_19_res(5) when stq_tail_i(2) else TEMP_19_res(9);
	TEMP_20_res(10) <= TEMP_19_res(6) when stq_tail_i(2) else TEMP_19_res(10);
	TEMP_20_res(11) <= TEMP_19_res(7) when stq_tail_i(2) else TEMP_19_res(11);
	TEMP_20_res(12) <= TEMP_19_res(8) when stq_tail_i(2) else TEMP_19_res(12);
	TEMP_20_res(13) <= TEMP_19_res(9) when stq_tail_i(2) else TEMP_19_res(13);
	-- Layer End
	TEMP_21_res(0) <= TEMP_20_res(12) when stq_tail_i(1) else TEMP_20_res(0);
	TEMP_21_res(1) <= TEMP_20_res(13) when stq_tail_i(1) else TEMP_20_res(1);
	TEMP_21_res(2) <= TEMP_20_res(0) when stq_tail_i(1) else TEMP_20_res(2);
	TEMP_21_res(3) <= TEMP_20_res(1) when stq_tail_i(1) else TEMP_20_res(3);
	TEMP_21_res(4) <= TEMP_20_res(2) when stq_tail_i(1) else TEMP_20_res(4);
	TEMP_21_res(5) <= TEMP_20_res(3) when stq_tail_i(1) else TEMP_20_res(5);
	TEMP_21_res(6) <= TEMP_20_res(4) when stq_tail_i(1) else TEMP_20_res(6);
	TEMP_21_res(7) <= TEMP_20_res(5) when stq_tail_i(1) else TEMP_20_res(7);
	TEMP_21_res(8) <= TEMP_20_res(6) when stq_tail_i(1) else TEMP_20_res(8);
	TEMP_21_res(9) <= TEMP_20_res(7) when stq_tail_i(1) else TEMP_20_res(9);
	TEMP_21_res(10) <= TEMP_20_res(8) when stq_tail_i(1) else TEMP_20_res(10);
	TEMP_21_res(11) <= TEMP_20_res(9) when stq_tail_i(1) else TEMP_20_res(11);
	TEMP_21_res(12) <= TEMP_20_res(10) when stq_tail_i(1) else TEMP_20_res(12);
	TEMP_21_res(13) <= TEMP_20_res(11) when stq_tail_i(1) else TEMP_20_res(13);
	-- Layer End
	ga_ls_order_temp_3(0) <= TEMP_21_res(13) when stq_tail_i(0) else TEMP_21_res(0);
	ga_ls_order_temp_3(1) <= TEMP_21_res(0) when stq_tail_i(0) else TEMP_21_res(1);
	ga_ls_order_temp_3(2) <= TEMP_21_res(1) when stq_tail_i(0) else TEMP_21_res(2);
	ga_ls_order_temp_3(3) <= TEMP_21_res(2) when stq_tail_i(0) else TEMP_21_res(3);
	ga_ls_order_temp_3(4) <= TEMP_21_res(3) when stq_tail_i(0) else TEMP_21_res(4);
	ga_ls_order_temp_3(5) <= TEMP_21_res(4) when stq_tail_i(0) else TEMP_21_res(5);
	ga_ls_order_temp_3(6) <= TEMP_21_res(5) when stq_tail_i(0) else TEMP_21_res(6);
	ga_ls_order_temp_3(7) <= TEMP_21_res(6) when stq_tail_i(0) else TEMP_21_res(7);
	ga_ls_order_temp_3(8) <= TEMP_21_res(7) when stq_tail_i(0) else TEMP_21_res(8);
	ga_ls_order_temp_3(9) <= TEMP_21_res(8) when stq_tail_i(0) else TEMP_21_res(9);
	ga_ls_order_temp_3(10) <= TEMP_21_res(9) when stq_tail_i(0) else TEMP_21_res(10);
	ga_ls_order_temp_3(11) <= TEMP_21_res(10) when stq_tail_i(0) else TEMP_21_res(11);
	ga_ls_order_temp_3(12) <= TEMP_21_res(11) when stq_tail_i(0) else TEMP_21_res(12);
	ga_ls_order_temp_3(13) <= TEMP_21_res(12) when stq_tail_i(0) else TEMP_21_res(13);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_4, ga_ls_order_rom_4, stq_tail)
	TEMP_22_res(0) <= ga_ls_order_rom_4(6) when stq_tail_i(3) else ga_ls_order_rom_4(0);
	TEMP_22_res(1) <= ga_ls_order_rom_4(7) when stq_tail_i(3) else ga_ls_order_rom_4(1);
	TEMP_22_res(2) <= ga_ls_order_rom_4(8) when stq_tail_i(3) else ga_ls_order_rom_4(2);
	TEMP_22_res(3) <= ga_ls_order_rom_4(9) when stq_tail_i(3) else ga_ls_order_rom_4(3);
	TEMP_22_res(4) <= ga_ls_order_rom_4(10) when stq_tail_i(3) else ga_ls_order_rom_4(4);
	TEMP_22_res(5) <= ga_ls_order_rom_4(11) when stq_tail_i(3) else ga_ls_order_rom_4(5);
	TEMP_22_res(6) <= ga_ls_order_rom_4(12) when stq_tail_i(3) else ga_ls_order_rom_4(6);
	TEMP_22_res(7) <= ga_ls_order_rom_4(13) when stq_tail_i(3) else ga_ls_order_rom_4(7);
	TEMP_22_res(8) <= ga_ls_order_rom_4(0) when stq_tail_i(3) else ga_ls_order_rom_4(8);
	TEMP_22_res(9) <= ga_ls_order_rom_4(1) when stq_tail_i(3) else ga_ls_order_rom_4(9);
	TEMP_22_res(10) <= ga_ls_order_rom_4(2) when stq_tail_i(3) else ga_ls_order_rom_4(10);
	TEMP_22_res(11) <= ga_ls_order_rom_4(3) when stq_tail_i(3) else ga_ls_order_rom_4(11);
	TEMP_22_res(12) <= ga_ls_order_rom_4(4) when stq_tail_i(3) else ga_ls_order_rom_4(12);
	TEMP_22_res(13) <= ga_ls_order_rom_4(5) when stq_tail_i(3) else ga_ls_order_rom_4(13);
	-- Layer End
	TEMP_23_res(0) <= TEMP_22_res(10) when stq_tail_i(2) else TEMP_22_res(0);
	TEMP_23_res(1) <= TEMP_22_res(11) when stq_tail_i(2) else TEMP_22_res(1);
	TEMP_23_res(2) <= TEMP_22_res(12) when stq_tail_i(2) else TEMP_22_res(2);
	TEMP_23_res(3) <= TEMP_22_res(13) when stq_tail_i(2) else TEMP_22_res(3);
	TEMP_23_res(4) <= TEMP_22_res(0) when stq_tail_i(2) else TEMP_22_res(4);
	TEMP_23_res(5) <= TEMP_22_res(1) when stq_tail_i(2) else TEMP_22_res(5);
	TEMP_23_res(6) <= TEMP_22_res(2) when stq_tail_i(2) else TEMP_22_res(6);
	TEMP_23_res(7) <= TEMP_22_res(3) when stq_tail_i(2) else TEMP_22_res(7);
	TEMP_23_res(8) <= TEMP_22_res(4) when stq_tail_i(2) else TEMP_22_res(8);
	TEMP_23_res(9) <= TEMP_22_res(5) when stq_tail_i(2) else TEMP_22_res(9);
	TEMP_23_res(10) <= TEMP_22_res(6) when stq_tail_i(2) else TEMP_22_res(10);
	TEMP_23_res(11) <= TEMP_22_res(7) when stq_tail_i(2) else TEMP_22_res(11);
	TEMP_23_res(12) <= TEMP_22_res(8) when stq_tail_i(2) else TEMP_22_res(12);
	TEMP_23_res(13) <= TEMP_22_res(9) when stq_tail_i(2) else TEMP_22_res(13);
	-- Layer End
	TEMP_24_res(0) <= TEMP_23_res(12) when stq_tail_i(1) else TEMP_23_res(0);
	TEMP_24_res(1) <= TEMP_23_res(13) when stq_tail_i(1) else TEMP_23_res(1);
	TEMP_24_res(2) <= TEMP_23_res(0) when stq_tail_i(1) else TEMP_23_res(2);
	TEMP_24_res(3) <= TEMP_23_res(1) when stq_tail_i(1) else TEMP_23_res(3);
	TEMP_24_res(4) <= TEMP_23_res(2) when stq_tail_i(1) else TEMP_23_res(4);
	TEMP_24_res(5) <= TEMP_23_res(3) when stq_tail_i(1) else TEMP_23_res(5);
	TEMP_24_res(6) <= TEMP_23_res(4) when stq_tail_i(1) else TEMP_23_res(6);
	TEMP_24_res(7) <= TEMP_23_res(5) when stq_tail_i(1) else TEMP_23_res(7);
	TEMP_24_res(8) <= TEMP_23_res(6) when stq_tail_i(1) else TEMP_23_res(8);
	TEMP_24_res(9) <= TEMP_23_res(7) when stq_tail_i(1) else TEMP_23_res(9);
	TEMP_24_res(10) <= TEMP_23_res(8) when stq_tail_i(1) else TEMP_23_res(10);
	TEMP_24_res(11) <= TEMP_23_res(9) when stq_tail_i(1) else TEMP_23_res(11);
	TEMP_24_res(12) <= TEMP_23_res(10) when stq_tail_i(1) else TEMP_23_res(12);
	TEMP_24_res(13) <= TEMP_23_res(11) when stq_tail_i(1) else TEMP_23_res(13);
	-- Layer End
	ga_ls_order_temp_4(0) <= TEMP_24_res(13) when stq_tail_i(0) else TEMP_24_res(0);
	ga_ls_order_temp_4(1) <= TEMP_24_res(0) when stq_tail_i(0) else TEMP_24_res(1);
	ga_ls_order_temp_4(2) <= TEMP_24_res(1) when stq_tail_i(0) else TEMP_24_res(2);
	ga_ls_order_temp_4(3) <= TEMP_24_res(2) when stq_tail_i(0) else TEMP_24_res(3);
	ga_ls_order_temp_4(4) <= TEMP_24_res(3) when stq_tail_i(0) else TEMP_24_res(4);
	ga_ls_order_temp_4(5) <= TEMP_24_res(4) when stq_tail_i(0) else TEMP_24_res(5);
	ga_ls_order_temp_4(6) <= TEMP_24_res(5) when stq_tail_i(0) else TEMP_24_res(6);
	ga_ls_order_temp_4(7) <= TEMP_24_res(6) when stq_tail_i(0) else TEMP_24_res(7);
	ga_ls_order_temp_4(8) <= TEMP_24_res(7) when stq_tail_i(0) else TEMP_24_res(8);
	ga_ls_order_temp_4(9) <= TEMP_24_res(8) when stq_tail_i(0) else TEMP_24_res(9);
	ga_ls_order_temp_4(10) <= TEMP_24_res(9) when stq_tail_i(0) else TEMP_24_res(10);
	ga_ls_order_temp_4(11) <= TEMP_24_res(10) when stq_tail_i(0) else TEMP_24_res(11);
	ga_ls_order_temp_4(12) <= TEMP_24_res(11) when stq_tail_i(0) else TEMP_24_res(12);
	ga_ls_order_temp_4(13) <= TEMP_24_res(12) when stq_tail_i(0) else TEMP_24_res(13);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_5, ga_ls_order_rom_5, stq_tail)
	TEMP_25_res(0) <= ga_ls_order_rom_5(6) when stq_tail_i(3) else ga_ls_order_rom_5(0);
	TEMP_25_res(1) <= ga_ls_order_rom_5(7) when stq_tail_i(3) else ga_ls_order_rom_5(1);
	TEMP_25_res(2) <= ga_ls_order_rom_5(8) when stq_tail_i(3) else ga_ls_order_rom_5(2);
	TEMP_25_res(3) <= ga_ls_order_rom_5(9) when stq_tail_i(3) else ga_ls_order_rom_5(3);
	TEMP_25_res(4) <= ga_ls_order_rom_5(10) when stq_tail_i(3) else ga_ls_order_rom_5(4);
	TEMP_25_res(5) <= ga_ls_order_rom_5(11) when stq_tail_i(3) else ga_ls_order_rom_5(5);
	TEMP_25_res(6) <= ga_ls_order_rom_5(12) when stq_tail_i(3) else ga_ls_order_rom_5(6);
	TEMP_25_res(7) <= ga_ls_order_rom_5(13) when stq_tail_i(3) else ga_ls_order_rom_5(7);
	TEMP_25_res(8) <= ga_ls_order_rom_5(0) when stq_tail_i(3) else ga_ls_order_rom_5(8);
	TEMP_25_res(9) <= ga_ls_order_rom_5(1) when stq_tail_i(3) else ga_ls_order_rom_5(9);
	TEMP_25_res(10) <= ga_ls_order_rom_5(2) when stq_tail_i(3) else ga_ls_order_rom_5(10);
	TEMP_25_res(11) <= ga_ls_order_rom_5(3) when stq_tail_i(3) else ga_ls_order_rom_5(11);
	TEMP_25_res(12) <= ga_ls_order_rom_5(4) when stq_tail_i(3) else ga_ls_order_rom_5(12);
	TEMP_25_res(13) <= ga_ls_order_rom_5(5) when stq_tail_i(3) else ga_ls_order_rom_5(13);
	-- Layer End
	TEMP_26_res(0) <= TEMP_25_res(10) when stq_tail_i(2) else TEMP_25_res(0);
	TEMP_26_res(1) <= TEMP_25_res(11) when stq_tail_i(2) else TEMP_25_res(1);
	TEMP_26_res(2) <= TEMP_25_res(12) when stq_tail_i(2) else TEMP_25_res(2);
	TEMP_26_res(3) <= TEMP_25_res(13) when stq_tail_i(2) else TEMP_25_res(3);
	TEMP_26_res(4) <= TEMP_25_res(0) when stq_tail_i(2) else TEMP_25_res(4);
	TEMP_26_res(5) <= TEMP_25_res(1) when stq_tail_i(2) else TEMP_25_res(5);
	TEMP_26_res(6) <= TEMP_25_res(2) when stq_tail_i(2) else TEMP_25_res(6);
	TEMP_26_res(7) <= TEMP_25_res(3) when stq_tail_i(2) else TEMP_25_res(7);
	TEMP_26_res(8) <= TEMP_25_res(4) when stq_tail_i(2) else TEMP_25_res(8);
	TEMP_26_res(9) <= TEMP_25_res(5) when stq_tail_i(2) else TEMP_25_res(9);
	TEMP_26_res(10) <= TEMP_25_res(6) when stq_tail_i(2) else TEMP_25_res(10);
	TEMP_26_res(11) <= TEMP_25_res(7) when stq_tail_i(2) else TEMP_25_res(11);
	TEMP_26_res(12) <= TEMP_25_res(8) when stq_tail_i(2) else TEMP_25_res(12);
	TEMP_26_res(13) <= TEMP_25_res(9) when stq_tail_i(2) else TEMP_25_res(13);
	-- Layer End
	TEMP_27_res(0) <= TEMP_26_res(12) when stq_tail_i(1) else TEMP_26_res(0);
	TEMP_27_res(1) <= TEMP_26_res(13) when stq_tail_i(1) else TEMP_26_res(1);
	TEMP_27_res(2) <= TEMP_26_res(0) when stq_tail_i(1) else TEMP_26_res(2);
	TEMP_27_res(3) <= TEMP_26_res(1) when stq_tail_i(1) else TEMP_26_res(3);
	TEMP_27_res(4) <= TEMP_26_res(2) when stq_tail_i(1) else TEMP_26_res(4);
	TEMP_27_res(5) <= TEMP_26_res(3) when stq_tail_i(1) else TEMP_26_res(5);
	TEMP_27_res(6) <= TEMP_26_res(4) when stq_tail_i(1) else TEMP_26_res(6);
	TEMP_27_res(7) <= TEMP_26_res(5) when stq_tail_i(1) else TEMP_26_res(7);
	TEMP_27_res(8) <= TEMP_26_res(6) when stq_tail_i(1) else TEMP_26_res(8);
	TEMP_27_res(9) <= TEMP_26_res(7) when stq_tail_i(1) else TEMP_26_res(9);
	TEMP_27_res(10) <= TEMP_26_res(8) when stq_tail_i(1) else TEMP_26_res(10);
	TEMP_27_res(11) <= TEMP_26_res(9) when stq_tail_i(1) else TEMP_26_res(11);
	TEMP_27_res(12) <= TEMP_26_res(10) when stq_tail_i(1) else TEMP_26_res(12);
	TEMP_27_res(13) <= TEMP_26_res(11) when stq_tail_i(1) else TEMP_26_res(13);
	-- Layer End
	ga_ls_order_temp_5(0) <= TEMP_27_res(13) when stq_tail_i(0) else TEMP_27_res(0);
	ga_ls_order_temp_5(1) <= TEMP_27_res(0) when stq_tail_i(0) else TEMP_27_res(1);
	ga_ls_order_temp_5(2) <= TEMP_27_res(1) when stq_tail_i(0) else TEMP_27_res(2);
	ga_ls_order_temp_5(3) <= TEMP_27_res(2) when stq_tail_i(0) else TEMP_27_res(3);
	ga_ls_order_temp_5(4) <= TEMP_27_res(3) when stq_tail_i(0) else TEMP_27_res(4);
	ga_ls_order_temp_5(5) <= TEMP_27_res(4) when stq_tail_i(0) else TEMP_27_res(5);
	ga_ls_order_temp_5(6) <= TEMP_27_res(5) when stq_tail_i(0) else TEMP_27_res(6);
	ga_ls_order_temp_5(7) <= TEMP_27_res(6) when stq_tail_i(0) else TEMP_27_res(7);
	ga_ls_order_temp_5(8) <= TEMP_27_res(7) when stq_tail_i(0) else TEMP_27_res(8);
	ga_ls_order_temp_5(9) <= TEMP_27_res(8) when stq_tail_i(0) else TEMP_27_res(9);
	ga_ls_order_temp_5(10) <= TEMP_27_res(9) when stq_tail_i(0) else TEMP_27_res(10);
	ga_ls_order_temp_5(11) <= TEMP_27_res(10) when stq_tail_i(0) else TEMP_27_res(11);
	ga_ls_order_temp_5(12) <= TEMP_27_res(11) when stq_tail_i(0) else TEMP_27_res(12);
	ga_ls_order_temp_5(13) <= TEMP_27_res(12) when stq_tail_i(0) else TEMP_27_res(13);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_6, ga_ls_order_rom_6, stq_tail)
	TEMP_28_res(0) <= ga_ls_order_rom_6(6) when stq_tail_i(3) else ga_ls_order_rom_6(0);
	TEMP_28_res(1) <= ga_ls_order_rom_6(7) when stq_tail_i(3) else ga_ls_order_rom_6(1);
	TEMP_28_res(2) <= ga_ls_order_rom_6(8) when stq_tail_i(3) else ga_ls_order_rom_6(2);
	TEMP_28_res(3) <= ga_ls_order_rom_6(9) when stq_tail_i(3) else ga_ls_order_rom_6(3);
	TEMP_28_res(4) <= ga_ls_order_rom_6(10) when stq_tail_i(3) else ga_ls_order_rom_6(4);
	TEMP_28_res(5) <= ga_ls_order_rom_6(11) when stq_tail_i(3) else ga_ls_order_rom_6(5);
	TEMP_28_res(6) <= ga_ls_order_rom_6(12) when stq_tail_i(3) else ga_ls_order_rom_6(6);
	TEMP_28_res(7) <= ga_ls_order_rom_6(13) when stq_tail_i(3) else ga_ls_order_rom_6(7);
	TEMP_28_res(8) <= ga_ls_order_rom_6(0) when stq_tail_i(3) else ga_ls_order_rom_6(8);
	TEMP_28_res(9) <= ga_ls_order_rom_6(1) when stq_tail_i(3) else ga_ls_order_rom_6(9);
	TEMP_28_res(10) <= ga_ls_order_rom_6(2) when stq_tail_i(3) else ga_ls_order_rom_6(10);
	TEMP_28_res(11) <= ga_ls_order_rom_6(3) when stq_tail_i(3) else ga_ls_order_rom_6(11);
	TEMP_28_res(12) <= ga_ls_order_rom_6(4) when stq_tail_i(3) else ga_ls_order_rom_6(12);
	TEMP_28_res(13) <= ga_ls_order_rom_6(5) when stq_tail_i(3) else ga_ls_order_rom_6(13);
	-- Layer End
	TEMP_29_res(0) <= TEMP_28_res(10) when stq_tail_i(2) else TEMP_28_res(0);
	TEMP_29_res(1) <= TEMP_28_res(11) when stq_tail_i(2) else TEMP_28_res(1);
	TEMP_29_res(2) <= TEMP_28_res(12) when stq_tail_i(2) else TEMP_28_res(2);
	TEMP_29_res(3) <= TEMP_28_res(13) when stq_tail_i(2) else TEMP_28_res(3);
	TEMP_29_res(4) <= TEMP_28_res(0) when stq_tail_i(2) else TEMP_28_res(4);
	TEMP_29_res(5) <= TEMP_28_res(1) when stq_tail_i(2) else TEMP_28_res(5);
	TEMP_29_res(6) <= TEMP_28_res(2) when stq_tail_i(2) else TEMP_28_res(6);
	TEMP_29_res(7) <= TEMP_28_res(3) when stq_tail_i(2) else TEMP_28_res(7);
	TEMP_29_res(8) <= TEMP_28_res(4) when stq_tail_i(2) else TEMP_28_res(8);
	TEMP_29_res(9) <= TEMP_28_res(5) when stq_tail_i(2) else TEMP_28_res(9);
	TEMP_29_res(10) <= TEMP_28_res(6) when stq_tail_i(2) else TEMP_28_res(10);
	TEMP_29_res(11) <= TEMP_28_res(7) when stq_tail_i(2) else TEMP_28_res(11);
	TEMP_29_res(12) <= TEMP_28_res(8) when stq_tail_i(2) else TEMP_28_res(12);
	TEMP_29_res(13) <= TEMP_28_res(9) when stq_tail_i(2) else TEMP_28_res(13);
	-- Layer End
	TEMP_30_res(0) <= TEMP_29_res(12) when stq_tail_i(1) else TEMP_29_res(0);
	TEMP_30_res(1) <= TEMP_29_res(13) when stq_tail_i(1) else TEMP_29_res(1);
	TEMP_30_res(2) <= TEMP_29_res(0) when stq_tail_i(1) else TEMP_29_res(2);
	TEMP_30_res(3) <= TEMP_29_res(1) when stq_tail_i(1) else TEMP_29_res(3);
	TEMP_30_res(4) <= TEMP_29_res(2) when stq_tail_i(1) else TEMP_29_res(4);
	TEMP_30_res(5) <= TEMP_29_res(3) when stq_tail_i(1) else TEMP_29_res(5);
	TEMP_30_res(6) <= TEMP_29_res(4) when stq_tail_i(1) else TEMP_29_res(6);
	TEMP_30_res(7) <= TEMP_29_res(5) when stq_tail_i(1) else TEMP_29_res(7);
	TEMP_30_res(8) <= TEMP_29_res(6) when stq_tail_i(1) else TEMP_29_res(8);
	TEMP_30_res(9) <= TEMP_29_res(7) when stq_tail_i(1) else TEMP_29_res(9);
	TEMP_30_res(10) <= TEMP_29_res(8) when stq_tail_i(1) else TEMP_29_res(10);
	TEMP_30_res(11) <= TEMP_29_res(9) when stq_tail_i(1) else TEMP_29_res(11);
	TEMP_30_res(12) <= TEMP_29_res(10) when stq_tail_i(1) else TEMP_29_res(12);
	TEMP_30_res(13) <= TEMP_29_res(11) when stq_tail_i(1) else TEMP_29_res(13);
	-- Layer End
	ga_ls_order_temp_6(0) <= TEMP_30_res(13) when stq_tail_i(0) else TEMP_30_res(0);
	ga_ls_order_temp_6(1) <= TEMP_30_res(0) when stq_tail_i(0) else TEMP_30_res(1);
	ga_ls_order_temp_6(2) <= TEMP_30_res(1) when stq_tail_i(0) else TEMP_30_res(2);
	ga_ls_order_temp_6(3) <= TEMP_30_res(2) when stq_tail_i(0) else TEMP_30_res(3);
	ga_ls_order_temp_6(4) <= TEMP_30_res(3) when stq_tail_i(0) else TEMP_30_res(4);
	ga_ls_order_temp_6(5) <= TEMP_30_res(4) when stq_tail_i(0) else TEMP_30_res(5);
	ga_ls_order_temp_6(6) <= TEMP_30_res(5) when stq_tail_i(0) else TEMP_30_res(6);
	ga_ls_order_temp_6(7) <= TEMP_30_res(6) when stq_tail_i(0) else TEMP_30_res(7);
	ga_ls_order_temp_6(8) <= TEMP_30_res(7) when stq_tail_i(0) else TEMP_30_res(8);
	ga_ls_order_temp_6(9) <= TEMP_30_res(8) when stq_tail_i(0) else TEMP_30_res(9);
	ga_ls_order_temp_6(10) <= TEMP_30_res(9) when stq_tail_i(0) else TEMP_30_res(10);
	ga_ls_order_temp_6(11) <= TEMP_30_res(10) when stq_tail_i(0) else TEMP_30_res(11);
	ga_ls_order_temp_6(12) <= TEMP_30_res(11) when stq_tail_i(0) else TEMP_30_res(12);
	ga_ls_order_temp_6(13) <= TEMP_30_res(12) when stq_tail_i(0) else TEMP_30_res(13);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_7, ga_ls_order_rom_7, stq_tail)
	TEMP_31_res(0) <= ga_ls_order_rom_7(6) when stq_tail_i(3) else ga_ls_order_rom_7(0);
	TEMP_31_res(1) <= ga_ls_order_rom_7(7) when stq_tail_i(3) else ga_ls_order_rom_7(1);
	TEMP_31_res(2) <= ga_ls_order_rom_7(8) when stq_tail_i(3) else ga_ls_order_rom_7(2);
	TEMP_31_res(3) <= ga_ls_order_rom_7(9) when stq_tail_i(3) else ga_ls_order_rom_7(3);
	TEMP_31_res(4) <= ga_ls_order_rom_7(10) when stq_tail_i(3) else ga_ls_order_rom_7(4);
	TEMP_31_res(5) <= ga_ls_order_rom_7(11) when stq_tail_i(3) else ga_ls_order_rom_7(5);
	TEMP_31_res(6) <= ga_ls_order_rom_7(12) when stq_tail_i(3) else ga_ls_order_rom_7(6);
	TEMP_31_res(7) <= ga_ls_order_rom_7(13) when stq_tail_i(3) else ga_ls_order_rom_7(7);
	TEMP_31_res(8) <= ga_ls_order_rom_7(0) when stq_tail_i(3) else ga_ls_order_rom_7(8);
	TEMP_31_res(9) <= ga_ls_order_rom_7(1) when stq_tail_i(3) else ga_ls_order_rom_7(9);
	TEMP_31_res(10) <= ga_ls_order_rom_7(2) when stq_tail_i(3) else ga_ls_order_rom_7(10);
	TEMP_31_res(11) <= ga_ls_order_rom_7(3) when stq_tail_i(3) else ga_ls_order_rom_7(11);
	TEMP_31_res(12) <= ga_ls_order_rom_7(4) when stq_tail_i(3) else ga_ls_order_rom_7(12);
	TEMP_31_res(13) <= ga_ls_order_rom_7(5) when stq_tail_i(3) else ga_ls_order_rom_7(13);
	-- Layer End
	TEMP_32_res(0) <= TEMP_31_res(10) when stq_tail_i(2) else TEMP_31_res(0);
	TEMP_32_res(1) <= TEMP_31_res(11) when stq_tail_i(2) else TEMP_31_res(1);
	TEMP_32_res(2) <= TEMP_31_res(12) when stq_tail_i(2) else TEMP_31_res(2);
	TEMP_32_res(3) <= TEMP_31_res(13) when stq_tail_i(2) else TEMP_31_res(3);
	TEMP_32_res(4) <= TEMP_31_res(0) when stq_tail_i(2) else TEMP_31_res(4);
	TEMP_32_res(5) <= TEMP_31_res(1) when stq_tail_i(2) else TEMP_31_res(5);
	TEMP_32_res(6) <= TEMP_31_res(2) when stq_tail_i(2) else TEMP_31_res(6);
	TEMP_32_res(7) <= TEMP_31_res(3) when stq_tail_i(2) else TEMP_31_res(7);
	TEMP_32_res(8) <= TEMP_31_res(4) when stq_tail_i(2) else TEMP_31_res(8);
	TEMP_32_res(9) <= TEMP_31_res(5) when stq_tail_i(2) else TEMP_31_res(9);
	TEMP_32_res(10) <= TEMP_31_res(6) when stq_tail_i(2) else TEMP_31_res(10);
	TEMP_32_res(11) <= TEMP_31_res(7) when stq_tail_i(2) else TEMP_31_res(11);
	TEMP_32_res(12) <= TEMP_31_res(8) when stq_tail_i(2) else TEMP_31_res(12);
	TEMP_32_res(13) <= TEMP_31_res(9) when stq_tail_i(2) else TEMP_31_res(13);
	-- Layer End
	TEMP_33_res(0) <= TEMP_32_res(12) when stq_tail_i(1) else TEMP_32_res(0);
	TEMP_33_res(1) <= TEMP_32_res(13) when stq_tail_i(1) else TEMP_32_res(1);
	TEMP_33_res(2) <= TEMP_32_res(0) when stq_tail_i(1) else TEMP_32_res(2);
	TEMP_33_res(3) <= TEMP_32_res(1) when stq_tail_i(1) else TEMP_32_res(3);
	TEMP_33_res(4) <= TEMP_32_res(2) when stq_tail_i(1) else TEMP_32_res(4);
	TEMP_33_res(5) <= TEMP_32_res(3) when stq_tail_i(1) else TEMP_32_res(5);
	TEMP_33_res(6) <= TEMP_32_res(4) when stq_tail_i(1) else TEMP_32_res(6);
	TEMP_33_res(7) <= TEMP_32_res(5) when stq_tail_i(1) else TEMP_32_res(7);
	TEMP_33_res(8) <= TEMP_32_res(6) when stq_tail_i(1) else TEMP_32_res(8);
	TEMP_33_res(9) <= TEMP_32_res(7) when stq_tail_i(1) else TEMP_32_res(9);
	TEMP_33_res(10) <= TEMP_32_res(8) when stq_tail_i(1) else TEMP_32_res(10);
	TEMP_33_res(11) <= TEMP_32_res(9) when stq_tail_i(1) else TEMP_32_res(11);
	TEMP_33_res(12) <= TEMP_32_res(10) when stq_tail_i(1) else TEMP_32_res(12);
	TEMP_33_res(13) <= TEMP_32_res(11) when stq_tail_i(1) else TEMP_32_res(13);
	-- Layer End
	ga_ls_order_temp_7(0) <= TEMP_33_res(13) when stq_tail_i(0) else TEMP_33_res(0);
	ga_ls_order_temp_7(1) <= TEMP_33_res(0) when stq_tail_i(0) else TEMP_33_res(1);
	ga_ls_order_temp_7(2) <= TEMP_33_res(1) when stq_tail_i(0) else TEMP_33_res(2);
	ga_ls_order_temp_7(3) <= TEMP_33_res(2) when stq_tail_i(0) else TEMP_33_res(3);
	ga_ls_order_temp_7(4) <= TEMP_33_res(3) when stq_tail_i(0) else TEMP_33_res(4);
	ga_ls_order_temp_7(5) <= TEMP_33_res(4) when stq_tail_i(0) else TEMP_33_res(5);
	ga_ls_order_temp_7(6) <= TEMP_33_res(5) when stq_tail_i(0) else TEMP_33_res(6);
	ga_ls_order_temp_7(7) <= TEMP_33_res(6) when stq_tail_i(0) else TEMP_33_res(7);
	ga_ls_order_temp_7(8) <= TEMP_33_res(7) when stq_tail_i(0) else TEMP_33_res(8);
	ga_ls_order_temp_7(9) <= TEMP_33_res(8) when stq_tail_i(0) else TEMP_33_res(9);
	ga_ls_order_temp_7(10) <= TEMP_33_res(9) when stq_tail_i(0) else TEMP_33_res(10);
	ga_ls_order_temp_7(11) <= TEMP_33_res(10) when stq_tail_i(0) else TEMP_33_res(11);
	ga_ls_order_temp_7(12) <= TEMP_33_res(11) when stq_tail_i(0) else TEMP_33_res(12);
	ga_ls_order_temp_7(13) <= TEMP_33_res(12) when stq_tail_i(0) else TEMP_33_res(13);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_8, ga_ls_order_rom_8, stq_tail)
	TEMP_34_res(0) <= ga_ls_order_rom_8(6) when stq_tail_i(3) else ga_ls_order_rom_8(0);
	TEMP_34_res(1) <= ga_ls_order_rom_8(7) when stq_tail_i(3) else ga_ls_order_rom_8(1);
	TEMP_34_res(2) <= ga_ls_order_rom_8(8) when stq_tail_i(3) else ga_ls_order_rom_8(2);
	TEMP_34_res(3) <= ga_ls_order_rom_8(9) when stq_tail_i(3) else ga_ls_order_rom_8(3);
	TEMP_34_res(4) <= ga_ls_order_rom_8(10) when stq_tail_i(3) else ga_ls_order_rom_8(4);
	TEMP_34_res(5) <= ga_ls_order_rom_8(11) when stq_tail_i(3) else ga_ls_order_rom_8(5);
	TEMP_34_res(6) <= ga_ls_order_rom_8(12) when stq_tail_i(3) else ga_ls_order_rom_8(6);
	TEMP_34_res(7) <= ga_ls_order_rom_8(13) when stq_tail_i(3) else ga_ls_order_rom_8(7);
	TEMP_34_res(8) <= ga_ls_order_rom_8(0) when stq_tail_i(3) else ga_ls_order_rom_8(8);
	TEMP_34_res(9) <= ga_ls_order_rom_8(1) when stq_tail_i(3) else ga_ls_order_rom_8(9);
	TEMP_34_res(10) <= ga_ls_order_rom_8(2) when stq_tail_i(3) else ga_ls_order_rom_8(10);
	TEMP_34_res(11) <= ga_ls_order_rom_8(3) when stq_tail_i(3) else ga_ls_order_rom_8(11);
	TEMP_34_res(12) <= ga_ls_order_rom_8(4) when stq_tail_i(3) else ga_ls_order_rom_8(12);
	TEMP_34_res(13) <= ga_ls_order_rom_8(5) when stq_tail_i(3) else ga_ls_order_rom_8(13);
	-- Layer End
	TEMP_35_res(0) <= TEMP_34_res(10) when stq_tail_i(2) else TEMP_34_res(0);
	TEMP_35_res(1) <= TEMP_34_res(11) when stq_tail_i(2) else TEMP_34_res(1);
	TEMP_35_res(2) <= TEMP_34_res(12) when stq_tail_i(2) else TEMP_34_res(2);
	TEMP_35_res(3) <= TEMP_34_res(13) when stq_tail_i(2) else TEMP_34_res(3);
	TEMP_35_res(4) <= TEMP_34_res(0) when stq_tail_i(2) else TEMP_34_res(4);
	TEMP_35_res(5) <= TEMP_34_res(1) when stq_tail_i(2) else TEMP_34_res(5);
	TEMP_35_res(6) <= TEMP_34_res(2) when stq_tail_i(2) else TEMP_34_res(6);
	TEMP_35_res(7) <= TEMP_34_res(3) when stq_tail_i(2) else TEMP_34_res(7);
	TEMP_35_res(8) <= TEMP_34_res(4) when stq_tail_i(2) else TEMP_34_res(8);
	TEMP_35_res(9) <= TEMP_34_res(5) when stq_tail_i(2) else TEMP_34_res(9);
	TEMP_35_res(10) <= TEMP_34_res(6) when stq_tail_i(2) else TEMP_34_res(10);
	TEMP_35_res(11) <= TEMP_34_res(7) when stq_tail_i(2) else TEMP_34_res(11);
	TEMP_35_res(12) <= TEMP_34_res(8) when stq_tail_i(2) else TEMP_34_res(12);
	TEMP_35_res(13) <= TEMP_34_res(9) when stq_tail_i(2) else TEMP_34_res(13);
	-- Layer End
	TEMP_36_res(0) <= TEMP_35_res(12) when stq_tail_i(1) else TEMP_35_res(0);
	TEMP_36_res(1) <= TEMP_35_res(13) when stq_tail_i(1) else TEMP_35_res(1);
	TEMP_36_res(2) <= TEMP_35_res(0) when stq_tail_i(1) else TEMP_35_res(2);
	TEMP_36_res(3) <= TEMP_35_res(1) when stq_tail_i(1) else TEMP_35_res(3);
	TEMP_36_res(4) <= TEMP_35_res(2) when stq_tail_i(1) else TEMP_35_res(4);
	TEMP_36_res(5) <= TEMP_35_res(3) when stq_tail_i(1) else TEMP_35_res(5);
	TEMP_36_res(6) <= TEMP_35_res(4) when stq_tail_i(1) else TEMP_35_res(6);
	TEMP_36_res(7) <= TEMP_35_res(5) when stq_tail_i(1) else TEMP_35_res(7);
	TEMP_36_res(8) <= TEMP_35_res(6) when stq_tail_i(1) else TEMP_35_res(8);
	TEMP_36_res(9) <= TEMP_35_res(7) when stq_tail_i(1) else TEMP_35_res(9);
	TEMP_36_res(10) <= TEMP_35_res(8) when stq_tail_i(1) else TEMP_35_res(10);
	TEMP_36_res(11) <= TEMP_35_res(9) when stq_tail_i(1) else TEMP_35_res(11);
	TEMP_36_res(12) <= TEMP_35_res(10) when stq_tail_i(1) else TEMP_35_res(12);
	TEMP_36_res(13) <= TEMP_35_res(11) when stq_tail_i(1) else TEMP_35_res(13);
	-- Layer End
	ga_ls_order_temp_8(0) <= TEMP_36_res(13) when stq_tail_i(0) else TEMP_36_res(0);
	ga_ls_order_temp_8(1) <= TEMP_36_res(0) when stq_tail_i(0) else TEMP_36_res(1);
	ga_ls_order_temp_8(2) <= TEMP_36_res(1) when stq_tail_i(0) else TEMP_36_res(2);
	ga_ls_order_temp_8(3) <= TEMP_36_res(2) when stq_tail_i(0) else TEMP_36_res(3);
	ga_ls_order_temp_8(4) <= TEMP_36_res(3) when stq_tail_i(0) else TEMP_36_res(4);
	ga_ls_order_temp_8(5) <= TEMP_36_res(4) when stq_tail_i(0) else TEMP_36_res(5);
	ga_ls_order_temp_8(6) <= TEMP_36_res(5) when stq_tail_i(0) else TEMP_36_res(6);
	ga_ls_order_temp_8(7) <= TEMP_36_res(6) when stq_tail_i(0) else TEMP_36_res(7);
	ga_ls_order_temp_8(8) <= TEMP_36_res(7) when stq_tail_i(0) else TEMP_36_res(8);
	ga_ls_order_temp_8(9) <= TEMP_36_res(8) when stq_tail_i(0) else TEMP_36_res(9);
	ga_ls_order_temp_8(10) <= TEMP_36_res(9) when stq_tail_i(0) else TEMP_36_res(10);
	ga_ls_order_temp_8(11) <= TEMP_36_res(10) when stq_tail_i(0) else TEMP_36_res(11);
	ga_ls_order_temp_8(12) <= TEMP_36_res(11) when stq_tail_i(0) else TEMP_36_res(12);
	ga_ls_order_temp_8(13) <= TEMP_36_res(12) when stq_tail_i(0) else TEMP_36_res(13);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_9, ga_ls_order_rom_9, stq_tail)
	TEMP_37_res(0) <= ga_ls_order_rom_9(6) when stq_tail_i(3) else ga_ls_order_rom_9(0);
	TEMP_37_res(1) <= ga_ls_order_rom_9(7) when stq_tail_i(3) else ga_ls_order_rom_9(1);
	TEMP_37_res(2) <= ga_ls_order_rom_9(8) when stq_tail_i(3) else ga_ls_order_rom_9(2);
	TEMP_37_res(3) <= ga_ls_order_rom_9(9) when stq_tail_i(3) else ga_ls_order_rom_9(3);
	TEMP_37_res(4) <= ga_ls_order_rom_9(10) when stq_tail_i(3) else ga_ls_order_rom_9(4);
	TEMP_37_res(5) <= ga_ls_order_rom_9(11) when stq_tail_i(3) else ga_ls_order_rom_9(5);
	TEMP_37_res(6) <= ga_ls_order_rom_9(12) when stq_tail_i(3) else ga_ls_order_rom_9(6);
	TEMP_37_res(7) <= ga_ls_order_rom_9(13) when stq_tail_i(3) else ga_ls_order_rom_9(7);
	TEMP_37_res(8) <= ga_ls_order_rom_9(0) when stq_tail_i(3) else ga_ls_order_rom_9(8);
	TEMP_37_res(9) <= ga_ls_order_rom_9(1) when stq_tail_i(3) else ga_ls_order_rom_9(9);
	TEMP_37_res(10) <= ga_ls_order_rom_9(2) when stq_tail_i(3) else ga_ls_order_rom_9(10);
	TEMP_37_res(11) <= ga_ls_order_rom_9(3) when stq_tail_i(3) else ga_ls_order_rom_9(11);
	TEMP_37_res(12) <= ga_ls_order_rom_9(4) when stq_tail_i(3) else ga_ls_order_rom_9(12);
	TEMP_37_res(13) <= ga_ls_order_rom_9(5) when stq_tail_i(3) else ga_ls_order_rom_9(13);
	-- Layer End
	TEMP_38_res(0) <= TEMP_37_res(10) when stq_tail_i(2) else TEMP_37_res(0);
	TEMP_38_res(1) <= TEMP_37_res(11) when stq_tail_i(2) else TEMP_37_res(1);
	TEMP_38_res(2) <= TEMP_37_res(12) when stq_tail_i(2) else TEMP_37_res(2);
	TEMP_38_res(3) <= TEMP_37_res(13) when stq_tail_i(2) else TEMP_37_res(3);
	TEMP_38_res(4) <= TEMP_37_res(0) when stq_tail_i(2) else TEMP_37_res(4);
	TEMP_38_res(5) <= TEMP_37_res(1) when stq_tail_i(2) else TEMP_37_res(5);
	TEMP_38_res(6) <= TEMP_37_res(2) when stq_tail_i(2) else TEMP_37_res(6);
	TEMP_38_res(7) <= TEMP_37_res(3) when stq_tail_i(2) else TEMP_37_res(7);
	TEMP_38_res(8) <= TEMP_37_res(4) when stq_tail_i(2) else TEMP_37_res(8);
	TEMP_38_res(9) <= TEMP_37_res(5) when stq_tail_i(2) else TEMP_37_res(9);
	TEMP_38_res(10) <= TEMP_37_res(6) when stq_tail_i(2) else TEMP_37_res(10);
	TEMP_38_res(11) <= TEMP_37_res(7) when stq_tail_i(2) else TEMP_37_res(11);
	TEMP_38_res(12) <= TEMP_37_res(8) when stq_tail_i(2) else TEMP_37_res(12);
	TEMP_38_res(13) <= TEMP_37_res(9) when stq_tail_i(2) else TEMP_37_res(13);
	-- Layer End
	TEMP_39_res(0) <= TEMP_38_res(12) when stq_tail_i(1) else TEMP_38_res(0);
	TEMP_39_res(1) <= TEMP_38_res(13) when stq_tail_i(1) else TEMP_38_res(1);
	TEMP_39_res(2) <= TEMP_38_res(0) when stq_tail_i(1) else TEMP_38_res(2);
	TEMP_39_res(3) <= TEMP_38_res(1) when stq_tail_i(1) else TEMP_38_res(3);
	TEMP_39_res(4) <= TEMP_38_res(2) when stq_tail_i(1) else TEMP_38_res(4);
	TEMP_39_res(5) <= TEMP_38_res(3) when stq_tail_i(1) else TEMP_38_res(5);
	TEMP_39_res(6) <= TEMP_38_res(4) when stq_tail_i(1) else TEMP_38_res(6);
	TEMP_39_res(7) <= TEMP_38_res(5) when stq_tail_i(1) else TEMP_38_res(7);
	TEMP_39_res(8) <= TEMP_38_res(6) when stq_tail_i(1) else TEMP_38_res(8);
	TEMP_39_res(9) <= TEMP_38_res(7) when stq_tail_i(1) else TEMP_38_res(9);
	TEMP_39_res(10) <= TEMP_38_res(8) when stq_tail_i(1) else TEMP_38_res(10);
	TEMP_39_res(11) <= TEMP_38_res(9) when stq_tail_i(1) else TEMP_38_res(11);
	TEMP_39_res(12) <= TEMP_38_res(10) when stq_tail_i(1) else TEMP_38_res(12);
	TEMP_39_res(13) <= TEMP_38_res(11) when stq_tail_i(1) else TEMP_38_res(13);
	-- Layer End
	ga_ls_order_temp_9(0) <= TEMP_39_res(13) when stq_tail_i(0) else TEMP_39_res(0);
	ga_ls_order_temp_9(1) <= TEMP_39_res(0) when stq_tail_i(0) else TEMP_39_res(1);
	ga_ls_order_temp_9(2) <= TEMP_39_res(1) when stq_tail_i(0) else TEMP_39_res(2);
	ga_ls_order_temp_9(3) <= TEMP_39_res(2) when stq_tail_i(0) else TEMP_39_res(3);
	ga_ls_order_temp_9(4) <= TEMP_39_res(3) when stq_tail_i(0) else TEMP_39_res(4);
	ga_ls_order_temp_9(5) <= TEMP_39_res(4) when stq_tail_i(0) else TEMP_39_res(5);
	ga_ls_order_temp_9(6) <= TEMP_39_res(5) when stq_tail_i(0) else TEMP_39_res(6);
	ga_ls_order_temp_9(7) <= TEMP_39_res(6) when stq_tail_i(0) else TEMP_39_res(7);
	ga_ls_order_temp_9(8) <= TEMP_39_res(7) when stq_tail_i(0) else TEMP_39_res(8);
	ga_ls_order_temp_9(9) <= TEMP_39_res(8) when stq_tail_i(0) else TEMP_39_res(9);
	ga_ls_order_temp_9(10) <= TEMP_39_res(9) when stq_tail_i(0) else TEMP_39_res(10);
	ga_ls_order_temp_9(11) <= TEMP_39_res(10) when stq_tail_i(0) else TEMP_39_res(11);
	ga_ls_order_temp_9(12) <= TEMP_39_res(11) when stq_tail_i(0) else TEMP_39_res(12);
	ga_ls_order_temp_9(13) <= TEMP_39_res(12) when stq_tail_i(0) else TEMP_39_res(13);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_10, ga_ls_order_rom_10, stq_tail)
	TEMP_40_res(0) <= ga_ls_order_rom_10(6) when stq_tail_i(3) else ga_ls_order_rom_10(0);
	TEMP_40_res(1) <= ga_ls_order_rom_10(7) when stq_tail_i(3) else ga_ls_order_rom_10(1);
	TEMP_40_res(2) <= ga_ls_order_rom_10(8) when stq_tail_i(3) else ga_ls_order_rom_10(2);
	TEMP_40_res(3) <= ga_ls_order_rom_10(9) when stq_tail_i(3) else ga_ls_order_rom_10(3);
	TEMP_40_res(4) <= ga_ls_order_rom_10(10) when stq_tail_i(3) else ga_ls_order_rom_10(4);
	TEMP_40_res(5) <= ga_ls_order_rom_10(11) when stq_tail_i(3) else ga_ls_order_rom_10(5);
	TEMP_40_res(6) <= ga_ls_order_rom_10(12) when stq_tail_i(3) else ga_ls_order_rom_10(6);
	TEMP_40_res(7) <= ga_ls_order_rom_10(13) when stq_tail_i(3) else ga_ls_order_rom_10(7);
	TEMP_40_res(8) <= ga_ls_order_rom_10(0) when stq_tail_i(3) else ga_ls_order_rom_10(8);
	TEMP_40_res(9) <= ga_ls_order_rom_10(1) when stq_tail_i(3) else ga_ls_order_rom_10(9);
	TEMP_40_res(10) <= ga_ls_order_rom_10(2) when stq_tail_i(3) else ga_ls_order_rom_10(10);
	TEMP_40_res(11) <= ga_ls_order_rom_10(3) when stq_tail_i(3) else ga_ls_order_rom_10(11);
	TEMP_40_res(12) <= ga_ls_order_rom_10(4) when stq_tail_i(3) else ga_ls_order_rom_10(12);
	TEMP_40_res(13) <= ga_ls_order_rom_10(5) when stq_tail_i(3) else ga_ls_order_rom_10(13);
	-- Layer End
	TEMP_41_res(0) <= TEMP_40_res(10) when stq_tail_i(2) else TEMP_40_res(0);
	TEMP_41_res(1) <= TEMP_40_res(11) when stq_tail_i(2) else TEMP_40_res(1);
	TEMP_41_res(2) <= TEMP_40_res(12) when stq_tail_i(2) else TEMP_40_res(2);
	TEMP_41_res(3) <= TEMP_40_res(13) when stq_tail_i(2) else TEMP_40_res(3);
	TEMP_41_res(4) <= TEMP_40_res(0) when stq_tail_i(2) else TEMP_40_res(4);
	TEMP_41_res(5) <= TEMP_40_res(1) when stq_tail_i(2) else TEMP_40_res(5);
	TEMP_41_res(6) <= TEMP_40_res(2) when stq_tail_i(2) else TEMP_40_res(6);
	TEMP_41_res(7) <= TEMP_40_res(3) when stq_tail_i(2) else TEMP_40_res(7);
	TEMP_41_res(8) <= TEMP_40_res(4) when stq_tail_i(2) else TEMP_40_res(8);
	TEMP_41_res(9) <= TEMP_40_res(5) when stq_tail_i(2) else TEMP_40_res(9);
	TEMP_41_res(10) <= TEMP_40_res(6) when stq_tail_i(2) else TEMP_40_res(10);
	TEMP_41_res(11) <= TEMP_40_res(7) when stq_tail_i(2) else TEMP_40_res(11);
	TEMP_41_res(12) <= TEMP_40_res(8) when stq_tail_i(2) else TEMP_40_res(12);
	TEMP_41_res(13) <= TEMP_40_res(9) when stq_tail_i(2) else TEMP_40_res(13);
	-- Layer End
	TEMP_42_res(0) <= TEMP_41_res(12) when stq_tail_i(1) else TEMP_41_res(0);
	TEMP_42_res(1) <= TEMP_41_res(13) when stq_tail_i(1) else TEMP_41_res(1);
	TEMP_42_res(2) <= TEMP_41_res(0) when stq_tail_i(1) else TEMP_41_res(2);
	TEMP_42_res(3) <= TEMP_41_res(1) when stq_tail_i(1) else TEMP_41_res(3);
	TEMP_42_res(4) <= TEMP_41_res(2) when stq_tail_i(1) else TEMP_41_res(4);
	TEMP_42_res(5) <= TEMP_41_res(3) when stq_tail_i(1) else TEMP_41_res(5);
	TEMP_42_res(6) <= TEMP_41_res(4) when stq_tail_i(1) else TEMP_41_res(6);
	TEMP_42_res(7) <= TEMP_41_res(5) when stq_tail_i(1) else TEMP_41_res(7);
	TEMP_42_res(8) <= TEMP_41_res(6) when stq_tail_i(1) else TEMP_41_res(8);
	TEMP_42_res(9) <= TEMP_41_res(7) when stq_tail_i(1) else TEMP_41_res(9);
	TEMP_42_res(10) <= TEMP_41_res(8) when stq_tail_i(1) else TEMP_41_res(10);
	TEMP_42_res(11) <= TEMP_41_res(9) when stq_tail_i(1) else TEMP_41_res(11);
	TEMP_42_res(12) <= TEMP_41_res(10) when stq_tail_i(1) else TEMP_41_res(12);
	TEMP_42_res(13) <= TEMP_41_res(11) when stq_tail_i(1) else TEMP_41_res(13);
	-- Layer End
	ga_ls_order_temp_10(0) <= TEMP_42_res(13) when stq_tail_i(0) else TEMP_42_res(0);
	ga_ls_order_temp_10(1) <= TEMP_42_res(0) when stq_tail_i(0) else TEMP_42_res(1);
	ga_ls_order_temp_10(2) <= TEMP_42_res(1) when stq_tail_i(0) else TEMP_42_res(2);
	ga_ls_order_temp_10(3) <= TEMP_42_res(2) when stq_tail_i(0) else TEMP_42_res(3);
	ga_ls_order_temp_10(4) <= TEMP_42_res(3) when stq_tail_i(0) else TEMP_42_res(4);
	ga_ls_order_temp_10(5) <= TEMP_42_res(4) when stq_tail_i(0) else TEMP_42_res(5);
	ga_ls_order_temp_10(6) <= TEMP_42_res(5) when stq_tail_i(0) else TEMP_42_res(6);
	ga_ls_order_temp_10(7) <= TEMP_42_res(6) when stq_tail_i(0) else TEMP_42_res(7);
	ga_ls_order_temp_10(8) <= TEMP_42_res(7) when stq_tail_i(0) else TEMP_42_res(8);
	ga_ls_order_temp_10(9) <= TEMP_42_res(8) when stq_tail_i(0) else TEMP_42_res(9);
	ga_ls_order_temp_10(10) <= TEMP_42_res(9) when stq_tail_i(0) else TEMP_42_res(10);
	ga_ls_order_temp_10(11) <= TEMP_42_res(10) when stq_tail_i(0) else TEMP_42_res(11);
	ga_ls_order_temp_10(12) <= TEMP_42_res(11) when stq_tail_i(0) else TEMP_42_res(12);
	ga_ls_order_temp_10(13) <= TEMP_42_res(12) when stq_tail_i(0) else TEMP_42_res(13);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_11, ga_ls_order_rom_11, stq_tail)
	TEMP_43_res(0) <= ga_ls_order_rom_11(6) when stq_tail_i(3) else ga_ls_order_rom_11(0);
	TEMP_43_res(1) <= ga_ls_order_rom_11(7) when stq_tail_i(3) else ga_ls_order_rom_11(1);
	TEMP_43_res(2) <= ga_ls_order_rom_11(8) when stq_tail_i(3) else ga_ls_order_rom_11(2);
	TEMP_43_res(3) <= ga_ls_order_rom_11(9) when stq_tail_i(3) else ga_ls_order_rom_11(3);
	TEMP_43_res(4) <= ga_ls_order_rom_11(10) when stq_tail_i(3) else ga_ls_order_rom_11(4);
	TEMP_43_res(5) <= ga_ls_order_rom_11(11) when stq_tail_i(3) else ga_ls_order_rom_11(5);
	TEMP_43_res(6) <= ga_ls_order_rom_11(12) when stq_tail_i(3) else ga_ls_order_rom_11(6);
	TEMP_43_res(7) <= ga_ls_order_rom_11(13) when stq_tail_i(3) else ga_ls_order_rom_11(7);
	TEMP_43_res(8) <= ga_ls_order_rom_11(0) when stq_tail_i(3) else ga_ls_order_rom_11(8);
	TEMP_43_res(9) <= ga_ls_order_rom_11(1) when stq_tail_i(3) else ga_ls_order_rom_11(9);
	TEMP_43_res(10) <= ga_ls_order_rom_11(2) when stq_tail_i(3) else ga_ls_order_rom_11(10);
	TEMP_43_res(11) <= ga_ls_order_rom_11(3) when stq_tail_i(3) else ga_ls_order_rom_11(11);
	TEMP_43_res(12) <= ga_ls_order_rom_11(4) when stq_tail_i(3) else ga_ls_order_rom_11(12);
	TEMP_43_res(13) <= ga_ls_order_rom_11(5) when stq_tail_i(3) else ga_ls_order_rom_11(13);
	-- Layer End
	TEMP_44_res(0) <= TEMP_43_res(10) when stq_tail_i(2) else TEMP_43_res(0);
	TEMP_44_res(1) <= TEMP_43_res(11) when stq_tail_i(2) else TEMP_43_res(1);
	TEMP_44_res(2) <= TEMP_43_res(12) when stq_tail_i(2) else TEMP_43_res(2);
	TEMP_44_res(3) <= TEMP_43_res(13) when stq_tail_i(2) else TEMP_43_res(3);
	TEMP_44_res(4) <= TEMP_43_res(0) when stq_tail_i(2) else TEMP_43_res(4);
	TEMP_44_res(5) <= TEMP_43_res(1) when stq_tail_i(2) else TEMP_43_res(5);
	TEMP_44_res(6) <= TEMP_43_res(2) when stq_tail_i(2) else TEMP_43_res(6);
	TEMP_44_res(7) <= TEMP_43_res(3) when stq_tail_i(2) else TEMP_43_res(7);
	TEMP_44_res(8) <= TEMP_43_res(4) when stq_tail_i(2) else TEMP_43_res(8);
	TEMP_44_res(9) <= TEMP_43_res(5) when stq_tail_i(2) else TEMP_43_res(9);
	TEMP_44_res(10) <= TEMP_43_res(6) when stq_tail_i(2) else TEMP_43_res(10);
	TEMP_44_res(11) <= TEMP_43_res(7) when stq_tail_i(2) else TEMP_43_res(11);
	TEMP_44_res(12) <= TEMP_43_res(8) when stq_tail_i(2) else TEMP_43_res(12);
	TEMP_44_res(13) <= TEMP_43_res(9) when stq_tail_i(2) else TEMP_43_res(13);
	-- Layer End
	TEMP_45_res(0) <= TEMP_44_res(12) when stq_tail_i(1) else TEMP_44_res(0);
	TEMP_45_res(1) <= TEMP_44_res(13) when stq_tail_i(1) else TEMP_44_res(1);
	TEMP_45_res(2) <= TEMP_44_res(0) when stq_tail_i(1) else TEMP_44_res(2);
	TEMP_45_res(3) <= TEMP_44_res(1) when stq_tail_i(1) else TEMP_44_res(3);
	TEMP_45_res(4) <= TEMP_44_res(2) when stq_tail_i(1) else TEMP_44_res(4);
	TEMP_45_res(5) <= TEMP_44_res(3) when stq_tail_i(1) else TEMP_44_res(5);
	TEMP_45_res(6) <= TEMP_44_res(4) when stq_tail_i(1) else TEMP_44_res(6);
	TEMP_45_res(7) <= TEMP_44_res(5) when stq_tail_i(1) else TEMP_44_res(7);
	TEMP_45_res(8) <= TEMP_44_res(6) when stq_tail_i(1) else TEMP_44_res(8);
	TEMP_45_res(9) <= TEMP_44_res(7) when stq_tail_i(1) else TEMP_44_res(9);
	TEMP_45_res(10) <= TEMP_44_res(8) when stq_tail_i(1) else TEMP_44_res(10);
	TEMP_45_res(11) <= TEMP_44_res(9) when stq_tail_i(1) else TEMP_44_res(11);
	TEMP_45_res(12) <= TEMP_44_res(10) when stq_tail_i(1) else TEMP_44_res(12);
	TEMP_45_res(13) <= TEMP_44_res(11) when stq_tail_i(1) else TEMP_44_res(13);
	-- Layer End
	ga_ls_order_temp_11(0) <= TEMP_45_res(13) when stq_tail_i(0) else TEMP_45_res(0);
	ga_ls_order_temp_11(1) <= TEMP_45_res(0) when stq_tail_i(0) else TEMP_45_res(1);
	ga_ls_order_temp_11(2) <= TEMP_45_res(1) when stq_tail_i(0) else TEMP_45_res(2);
	ga_ls_order_temp_11(3) <= TEMP_45_res(2) when stq_tail_i(0) else TEMP_45_res(3);
	ga_ls_order_temp_11(4) <= TEMP_45_res(3) when stq_tail_i(0) else TEMP_45_res(4);
	ga_ls_order_temp_11(5) <= TEMP_45_res(4) when stq_tail_i(0) else TEMP_45_res(5);
	ga_ls_order_temp_11(6) <= TEMP_45_res(5) when stq_tail_i(0) else TEMP_45_res(6);
	ga_ls_order_temp_11(7) <= TEMP_45_res(6) when stq_tail_i(0) else TEMP_45_res(7);
	ga_ls_order_temp_11(8) <= TEMP_45_res(7) when stq_tail_i(0) else TEMP_45_res(8);
	ga_ls_order_temp_11(9) <= TEMP_45_res(8) when stq_tail_i(0) else TEMP_45_res(9);
	ga_ls_order_temp_11(10) <= TEMP_45_res(9) when stq_tail_i(0) else TEMP_45_res(10);
	ga_ls_order_temp_11(11) <= TEMP_45_res(10) when stq_tail_i(0) else TEMP_45_res(11);
	ga_ls_order_temp_11(12) <= TEMP_45_res(11) when stq_tail_i(0) else TEMP_45_res(12);
	ga_ls_order_temp_11(13) <= TEMP_45_res(12) when stq_tail_i(0) else TEMP_45_res(13);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_12, ga_ls_order_rom_12, stq_tail)
	TEMP_46_res(0) <= ga_ls_order_rom_12(6) when stq_tail_i(3) else ga_ls_order_rom_12(0);
	TEMP_46_res(1) <= ga_ls_order_rom_12(7) when stq_tail_i(3) else ga_ls_order_rom_12(1);
	TEMP_46_res(2) <= ga_ls_order_rom_12(8) when stq_tail_i(3) else ga_ls_order_rom_12(2);
	TEMP_46_res(3) <= ga_ls_order_rom_12(9) when stq_tail_i(3) else ga_ls_order_rom_12(3);
	TEMP_46_res(4) <= ga_ls_order_rom_12(10) when stq_tail_i(3) else ga_ls_order_rom_12(4);
	TEMP_46_res(5) <= ga_ls_order_rom_12(11) when stq_tail_i(3) else ga_ls_order_rom_12(5);
	TEMP_46_res(6) <= ga_ls_order_rom_12(12) when stq_tail_i(3) else ga_ls_order_rom_12(6);
	TEMP_46_res(7) <= ga_ls_order_rom_12(13) when stq_tail_i(3) else ga_ls_order_rom_12(7);
	TEMP_46_res(8) <= ga_ls_order_rom_12(0) when stq_tail_i(3) else ga_ls_order_rom_12(8);
	TEMP_46_res(9) <= ga_ls_order_rom_12(1) when stq_tail_i(3) else ga_ls_order_rom_12(9);
	TEMP_46_res(10) <= ga_ls_order_rom_12(2) when stq_tail_i(3) else ga_ls_order_rom_12(10);
	TEMP_46_res(11) <= ga_ls_order_rom_12(3) when stq_tail_i(3) else ga_ls_order_rom_12(11);
	TEMP_46_res(12) <= ga_ls_order_rom_12(4) when stq_tail_i(3) else ga_ls_order_rom_12(12);
	TEMP_46_res(13) <= ga_ls_order_rom_12(5) when stq_tail_i(3) else ga_ls_order_rom_12(13);
	-- Layer End
	TEMP_47_res(0) <= TEMP_46_res(10) when stq_tail_i(2) else TEMP_46_res(0);
	TEMP_47_res(1) <= TEMP_46_res(11) when stq_tail_i(2) else TEMP_46_res(1);
	TEMP_47_res(2) <= TEMP_46_res(12) when stq_tail_i(2) else TEMP_46_res(2);
	TEMP_47_res(3) <= TEMP_46_res(13) when stq_tail_i(2) else TEMP_46_res(3);
	TEMP_47_res(4) <= TEMP_46_res(0) when stq_tail_i(2) else TEMP_46_res(4);
	TEMP_47_res(5) <= TEMP_46_res(1) when stq_tail_i(2) else TEMP_46_res(5);
	TEMP_47_res(6) <= TEMP_46_res(2) when stq_tail_i(2) else TEMP_46_res(6);
	TEMP_47_res(7) <= TEMP_46_res(3) when stq_tail_i(2) else TEMP_46_res(7);
	TEMP_47_res(8) <= TEMP_46_res(4) when stq_tail_i(2) else TEMP_46_res(8);
	TEMP_47_res(9) <= TEMP_46_res(5) when stq_tail_i(2) else TEMP_46_res(9);
	TEMP_47_res(10) <= TEMP_46_res(6) when stq_tail_i(2) else TEMP_46_res(10);
	TEMP_47_res(11) <= TEMP_46_res(7) when stq_tail_i(2) else TEMP_46_res(11);
	TEMP_47_res(12) <= TEMP_46_res(8) when stq_tail_i(2) else TEMP_46_res(12);
	TEMP_47_res(13) <= TEMP_46_res(9) when stq_tail_i(2) else TEMP_46_res(13);
	-- Layer End
	TEMP_48_res(0) <= TEMP_47_res(12) when stq_tail_i(1) else TEMP_47_res(0);
	TEMP_48_res(1) <= TEMP_47_res(13) when stq_tail_i(1) else TEMP_47_res(1);
	TEMP_48_res(2) <= TEMP_47_res(0) when stq_tail_i(1) else TEMP_47_res(2);
	TEMP_48_res(3) <= TEMP_47_res(1) when stq_tail_i(1) else TEMP_47_res(3);
	TEMP_48_res(4) <= TEMP_47_res(2) when stq_tail_i(1) else TEMP_47_res(4);
	TEMP_48_res(5) <= TEMP_47_res(3) when stq_tail_i(1) else TEMP_47_res(5);
	TEMP_48_res(6) <= TEMP_47_res(4) when stq_tail_i(1) else TEMP_47_res(6);
	TEMP_48_res(7) <= TEMP_47_res(5) when stq_tail_i(1) else TEMP_47_res(7);
	TEMP_48_res(8) <= TEMP_47_res(6) when stq_tail_i(1) else TEMP_47_res(8);
	TEMP_48_res(9) <= TEMP_47_res(7) when stq_tail_i(1) else TEMP_47_res(9);
	TEMP_48_res(10) <= TEMP_47_res(8) when stq_tail_i(1) else TEMP_47_res(10);
	TEMP_48_res(11) <= TEMP_47_res(9) when stq_tail_i(1) else TEMP_47_res(11);
	TEMP_48_res(12) <= TEMP_47_res(10) when stq_tail_i(1) else TEMP_47_res(12);
	TEMP_48_res(13) <= TEMP_47_res(11) when stq_tail_i(1) else TEMP_47_res(13);
	-- Layer End
	ga_ls_order_temp_12(0) <= TEMP_48_res(13) when stq_tail_i(0) else TEMP_48_res(0);
	ga_ls_order_temp_12(1) <= TEMP_48_res(0) when stq_tail_i(0) else TEMP_48_res(1);
	ga_ls_order_temp_12(2) <= TEMP_48_res(1) when stq_tail_i(0) else TEMP_48_res(2);
	ga_ls_order_temp_12(3) <= TEMP_48_res(2) when stq_tail_i(0) else TEMP_48_res(3);
	ga_ls_order_temp_12(4) <= TEMP_48_res(3) when stq_tail_i(0) else TEMP_48_res(4);
	ga_ls_order_temp_12(5) <= TEMP_48_res(4) when stq_tail_i(0) else TEMP_48_res(5);
	ga_ls_order_temp_12(6) <= TEMP_48_res(5) when stq_tail_i(0) else TEMP_48_res(6);
	ga_ls_order_temp_12(7) <= TEMP_48_res(6) when stq_tail_i(0) else TEMP_48_res(7);
	ga_ls_order_temp_12(8) <= TEMP_48_res(7) when stq_tail_i(0) else TEMP_48_res(8);
	ga_ls_order_temp_12(9) <= TEMP_48_res(8) when stq_tail_i(0) else TEMP_48_res(9);
	ga_ls_order_temp_12(10) <= TEMP_48_res(9) when stq_tail_i(0) else TEMP_48_res(10);
	ga_ls_order_temp_12(11) <= TEMP_48_res(10) when stq_tail_i(0) else TEMP_48_res(11);
	ga_ls_order_temp_12(12) <= TEMP_48_res(11) when stq_tail_i(0) else TEMP_48_res(12);
	ga_ls_order_temp_12(13) <= TEMP_48_res(12) when stq_tail_i(0) else TEMP_48_res(13);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_13, ga_ls_order_rom_13, stq_tail)
	TEMP_49_res(0) <= ga_ls_order_rom_13(6) when stq_tail_i(3) else ga_ls_order_rom_13(0);
	TEMP_49_res(1) <= ga_ls_order_rom_13(7) when stq_tail_i(3) else ga_ls_order_rom_13(1);
	TEMP_49_res(2) <= ga_ls_order_rom_13(8) when stq_tail_i(3) else ga_ls_order_rom_13(2);
	TEMP_49_res(3) <= ga_ls_order_rom_13(9) when stq_tail_i(3) else ga_ls_order_rom_13(3);
	TEMP_49_res(4) <= ga_ls_order_rom_13(10) when stq_tail_i(3) else ga_ls_order_rom_13(4);
	TEMP_49_res(5) <= ga_ls_order_rom_13(11) when stq_tail_i(3) else ga_ls_order_rom_13(5);
	TEMP_49_res(6) <= ga_ls_order_rom_13(12) when stq_tail_i(3) else ga_ls_order_rom_13(6);
	TEMP_49_res(7) <= ga_ls_order_rom_13(13) when stq_tail_i(3) else ga_ls_order_rom_13(7);
	TEMP_49_res(8) <= ga_ls_order_rom_13(0) when stq_tail_i(3) else ga_ls_order_rom_13(8);
	TEMP_49_res(9) <= ga_ls_order_rom_13(1) when stq_tail_i(3) else ga_ls_order_rom_13(9);
	TEMP_49_res(10) <= ga_ls_order_rom_13(2) when stq_tail_i(3) else ga_ls_order_rom_13(10);
	TEMP_49_res(11) <= ga_ls_order_rom_13(3) when stq_tail_i(3) else ga_ls_order_rom_13(11);
	TEMP_49_res(12) <= ga_ls_order_rom_13(4) when stq_tail_i(3) else ga_ls_order_rom_13(12);
	TEMP_49_res(13) <= ga_ls_order_rom_13(5) when stq_tail_i(3) else ga_ls_order_rom_13(13);
	-- Layer End
	TEMP_50_res(0) <= TEMP_49_res(10) when stq_tail_i(2) else TEMP_49_res(0);
	TEMP_50_res(1) <= TEMP_49_res(11) when stq_tail_i(2) else TEMP_49_res(1);
	TEMP_50_res(2) <= TEMP_49_res(12) when stq_tail_i(2) else TEMP_49_res(2);
	TEMP_50_res(3) <= TEMP_49_res(13) when stq_tail_i(2) else TEMP_49_res(3);
	TEMP_50_res(4) <= TEMP_49_res(0) when stq_tail_i(2) else TEMP_49_res(4);
	TEMP_50_res(5) <= TEMP_49_res(1) when stq_tail_i(2) else TEMP_49_res(5);
	TEMP_50_res(6) <= TEMP_49_res(2) when stq_tail_i(2) else TEMP_49_res(6);
	TEMP_50_res(7) <= TEMP_49_res(3) when stq_tail_i(2) else TEMP_49_res(7);
	TEMP_50_res(8) <= TEMP_49_res(4) when stq_tail_i(2) else TEMP_49_res(8);
	TEMP_50_res(9) <= TEMP_49_res(5) when stq_tail_i(2) else TEMP_49_res(9);
	TEMP_50_res(10) <= TEMP_49_res(6) when stq_tail_i(2) else TEMP_49_res(10);
	TEMP_50_res(11) <= TEMP_49_res(7) when stq_tail_i(2) else TEMP_49_res(11);
	TEMP_50_res(12) <= TEMP_49_res(8) when stq_tail_i(2) else TEMP_49_res(12);
	TEMP_50_res(13) <= TEMP_49_res(9) when stq_tail_i(2) else TEMP_49_res(13);
	-- Layer End
	TEMP_51_res(0) <= TEMP_50_res(12) when stq_tail_i(1) else TEMP_50_res(0);
	TEMP_51_res(1) <= TEMP_50_res(13) when stq_tail_i(1) else TEMP_50_res(1);
	TEMP_51_res(2) <= TEMP_50_res(0) when stq_tail_i(1) else TEMP_50_res(2);
	TEMP_51_res(3) <= TEMP_50_res(1) when stq_tail_i(1) else TEMP_50_res(3);
	TEMP_51_res(4) <= TEMP_50_res(2) when stq_tail_i(1) else TEMP_50_res(4);
	TEMP_51_res(5) <= TEMP_50_res(3) when stq_tail_i(1) else TEMP_50_res(5);
	TEMP_51_res(6) <= TEMP_50_res(4) when stq_tail_i(1) else TEMP_50_res(6);
	TEMP_51_res(7) <= TEMP_50_res(5) when stq_tail_i(1) else TEMP_50_res(7);
	TEMP_51_res(8) <= TEMP_50_res(6) when stq_tail_i(1) else TEMP_50_res(8);
	TEMP_51_res(9) <= TEMP_50_res(7) when stq_tail_i(1) else TEMP_50_res(9);
	TEMP_51_res(10) <= TEMP_50_res(8) when stq_tail_i(1) else TEMP_50_res(10);
	TEMP_51_res(11) <= TEMP_50_res(9) when stq_tail_i(1) else TEMP_50_res(11);
	TEMP_51_res(12) <= TEMP_50_res(10) when stq_tail_i(1) else TEMP_50_res(12);
	TEMP_51_res(13) <= TEMP_50_res(11) when stq_tail_i(1) else TEMP_50_res(13);
	-- Layer End
	ga_ls_order_temp_13(0) <= TEMP_51_res(13) when stq_tail_i(0) else TEMP_51_res(0);
	ga_ls_order_temp_13(1) <= TEMP_51_res(0) when stq_tail_i(0) else TEMP_51_res(1);
	ga_ls_order_temp_13(2) <= TEMP_51_res(1) when stq_tail_i(0) else TEMP_51_res(2);
	ga_ls_order_temp_13(3) <= TEMP_51_res(2) when stq_tail_i(0) else TEMP_51_res(3);
	ga_ls_order_temp_13(4) <= TEMP_51_res(3) when stq_tail_i(0) else TEMP_51_res(4);
	ga_ls_order_temp_13(5) <= TEMP_51_res(4) when stq_tail_i(0) else TEMP_51_res(5);
	ga_ls_order_temp_13(6) <= TEMP_51_res(5) when stq_tail_i(0) else TEMP_51_res(6);
	ga_ls_order_temp_13(7) <= TEMP_51_res(6) when stq_tail_i(0) else TEMP_51_res(7);
	ga_ls_order_temp_13(8) <= TEMP_51_res(7) when stq_tail_i(0) else TEMP_51_res(8);
	ga_ls_order_temp_13(9) <= TEMP_51_res(8) when stq_tail_i(0) else TEMP_51_res(9);
	ga_ls_order_temp_13(10) <= TEMP_51_res(9) when stq_tail_i(0) else TEMP_51_res(10);
	ga_ls_order_temp_13(11) <= TEMP_51_res(10) when stq_tail_i(0) else TEMP_51_res(11);
	ga_ls_order_temp_13(12) <= TEMP_51_res(11) when stq_tail_i(0) else TEMP_51_res(12);
	ga_ls_order_temp_13(13) <= TEMP_51_res(12) when stq_tail_i(0) else TEMP_51_res(13);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order, ga_ls_order_temp, ldq_tail)
	TEMP_52_res_0 <= ga_ls_order_temp_6 when ldq_tail_i(3) else ga_ls_order_temp_0;
	TEMP_52_res_1 <= ga_ls_order_temp_7 when ldq_tail_i(3) else ga_ls_order_temp_1;
	TEMP_52_res_2 <= ga_ls_order_temp_8 when ldq_tail_i(3) else ga_ls_order_temp_2;
	TEMP_52_res_3 <= ga_ls_order_temp_9 when ldq_tail_i(3) else ga_ls_order_temp_3;
	TEMP_52_res_4 <= ga_ls_order_temp_10 when ldq_tail_i(3) else ga_ls_order_temp_4;
	TEMP_52_res_5 <= ga_ls_order_temp_11 when ldq_tail_i(3) else ga_ls_order_temp_5;
	TEMP_52_res_6 <= ga_ls_order_temp_12 when ldq_tail_i(3) else ga_ls_order_temp_6;
	TEMP_52_res_7 <= ga_ls_order_temp_13 when ldq_tail_i(3) else ga_ls_order_temp_7;
	TEMP_52_res_8 <= ga_ls_order_temp_0 when ldq_tail_i(3) else ga_ls_order_temp_8;
	TEMP_52_res_9 <= ga_ls_order_temp_1 when ldq_tail_i(3) else ga_ls_order_temp_9;
	TEMP_52_res_10 <= ga_ls_order_temp_2 when ldq_tail_i(3) else ga_ls_order_temp_10;
	TEMP_52_res_11 <= ga_ls_order_temp_3 when ldq_tail_i(3) else ga_ls_order_temp_11;
	TEMP_52_res_12 <= ga_ls_order_temp_4 when ldq_tail_i(3) else ga_ls_order_temp_12;
	TEMP_52_res_13 <= ga_ls_order_temp_5 when ldq_tail_i(3) else ga_ls_order_temp_13;
	-- Layer End
	TEMP_53_res_0 <= TEMP_52_res_10 when ldq_tail_i(2) else TEMP_52_res_0;
	TEMP_53_res_1 <= TEMP_52_res_11 when ldq_tail_i(2) else TEMP_52_res_1;
	TEMP_53_res_2 <= TEMP_52_res_12 when ldq_tail_i(2) else TEMP_52_res_2;
	TEMP_53_res_3 <= TEMP_52_res_13 when ldq_tail_i(2) else TEMP_52_res_3;
	TEMP_53_res_4 <= TEMP_52_res_0 when ldq_tail_i(2) else TEMP_52_res_4;
	TEMP_53_res_5 <= TEMP_52_res_1 when ldq_tail_i(2) else TEMP_52_res_5;
	TEMP_53_res_6 <= TEMP_52_res_2 when ldq_tail_i(2) else TEMP_52_res_6;
	TEMP_53_res_7 <= TEMP_52_res_3 when ldq_tail_i(2) else TEMP_52_res_7;
	TEMP_53_res_8 <= TEMP_52_res_4 when ldq_tail_i(2) else TEMP_52_res_8;
	TEMP_53_res_9 <= TEMP_52_res_5 when ldq_tail_i(2) else TEMP_52_res_9;
	TEMP_53_res_10 <= TEMP_52_res_6 when ldq_tail_i(2) else TEMP_52_res_10;
	TEMP_53_res_11 <= TEMP_52_res_7 when ldq_tail_i(2) else TEMP_52_res_11;
	TEMP_53_res_12 <= TEMP_52_res_8 when ldq_tail_i(2) else TEMP_52_res_12;
	TEMP_53_res_13 <= TEMP_52_res_9 when ldq_tail_i(2) else TEMP_52_res_13;
	-- Layer End
	TEMP_54_res_0 <= TEMP_53_res_12 when ldq_tail_i(1) else TEMP_53_res_0;
	TEMP_54_res_1 <= TEMP_53_res_13 when ldq_tail_i(1) else TEMP_53_res_1;
	TEMP_54_res_2 <= TEMP_53_res_0 when ldq_tail_i(1) else TEMP_53_res_2;
	TEMP_54_res_3 <= TEMP_53_res_1 when ldq_tail_i(1) else TEMP_53_res_3;
	TEMP_54_res_4 <= TEMP_53_res_2 when ldq_tail_i(1) else TEMP_53_res_4;
	TEMP_54_res_5 <= TEMP_53_res_3 when ldq_tail_i(1) else TEMP_53_res_5;
	TEMP_54_res_6 <= TEMP_53_res_4 when ldq_tail_i(1) else TEMP_53_res_6;
	TEMP_54_res_7 <= TEMP_53_res_5 when ldq_tail_i(1) else TEMP_53_res_7;
	TEMP_54_res_8 <= TEMP_53_res_6 when ldq_tail_i(1) else TEMP_53_res_8;
	TEMP_54_res_9 <= TEMP_53_res_7 when ldq_tail_i(1) else TEMP_53_res_9;
	TEMP_54_res_10 <= TEMP_53_res_8 when ldq_tail_i(1) else TEMP_53_res_10;
	TEMP_54_res_11 <= TEMP_53_res_9 when ldq_tail_i(1) else TEMP_53_res_11;
	TEMP_54_res_12 <= TEMP_53_res_10 when ldq_tail_i(1) else TEMP_53_res_12;
	TEMP_54_res_13 <= TEMP_53_res_11 when ldq_tail_i(1) else TEMP_53_res_13;
	-- Layer End
	ga_ls_order_0_o <= TEMP_54_res_13 when ldq_tail_i(0) else TEMP_54_res_0;
	ga_ls_order_1_o <= TEMP_54_res_0 when ldq_tail_i(0) else TEMP_54_res_1;
	ga_ls_order_2_o <= TEMP_54_res_1 when ldq_tail_i(0) else TEMP_54_res_2;
	ga_ls_order_3_o <= TEMP_54_res_2 when ldq_tail_i(0) else TEMP_54_res_3;
	ga_ls_order_4_o <= TEMP_54_res_3 when ldq_tail_i(0) else TEMP_54_res_4;
	ga_ls_order_5_o <= TEMP_54_res_4 when ldq_tail_i(0) else TEMP_54_res_5;
	ga_ls_order_6_o <= TEMP_54_res_5 when ldq_tail_i(0) else TEMP_54_res_6;
	ga_ls_order_7_o <= TEMP_54_res_6 when ldq_tail_i(0) else TEMP_54_res_7;
	ga_ls_order_8_o <= TEMP_54_res_7 when ldq_tail_i(0) else TEMP_54_res_8;
	ga_ls_order_9_o <= TEMP_54_res_8 when ldq_tail_i(0) else TEMP_54_res_9;
	ga_ls_order_10_o <= TEMP_54_res_9 when ldq_tail_i(0) else TEMP_54_res_10;
	ga_ls_order_11_o <= TEMP_54_res_10 when ldq_tail_i(0) else TEMP_54_res_11;
	ga_ls_order_12_o <= TEMP_54_res_11 when ldq_tail_i(0) else TEMP_54_res_12;
	ga_ls_order_13_o <= TEMP_54_res_12 when ldq_tail_i(0) else TEMP_54_res_13;
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
		entry_alloc_2_i : in std_logic;
		entry_alloc_3_i : in std_logic;
		entry_alloc_4_i : in std_logic;
		entry_alloc_5_i : in std_logic;
		entry_alloc_6_i : in std_logic;
		entry_alloc_7_i : in std_logic;
		entry_alloc_8_i : in std_logic;
		entry_alloc_9_i : in std_logic;
		entry_alloc_10_i : in std_logic;
		entry_alloc_11_i : in std_logic;
		entry_alloc_12_i : in std_logic;
		entry_alloc_13_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_valid_2_i : in std_logic;
		entry_payload_valid_3_i : in std_logic;
		entry_payload_valid_4_i : in std_logic;
		entry_payload_valid_5_i : in std_logic;
		entry_payload_valid_6_i : in std_logic;
		entry_payload_valid_7_i : in std_logic;
		entry_payload_valid_8_i : in std_logic;
		entry_payload_valid_9_i : in std_logic;
		entry_payload_valid_10_i : in std_logic;
		entry_payload_valid_11_i : in std_logic;
		entry_payload_valid_12_i : in std_logic;
		entry_payload_valid_13_i : in std_logic;
		entry_payload_0_o : out std_logic_vector(9 downto 0);
		entry_payload_1_o : out std_logic_vector(9 downto 0);
		entry_payload_2_o : out std_logic_vector(9 downto 0);
		entry_payload_3_o : out std_logic_vector(9 downto 0);
		entry_payload_4_o : out std_logic_vector(9 downto 0);
		entry_payload_5_o : out std_logic_vector(9 downto 0);
		entry_payload_6_o : out std_logic_vector(9 downto 0);
		entry_payload_7_o : out std_logic_vector(9 downto 0);
		entry_payload_8_o : out std_logic_vector(9 downto 0);
		entry_payload_9_o : out std_logic_vector(9 downto 0);
		entry_payload_10_o : out std_logic_vector(9 downto 0);
		entry_payload_11_o : out std_logic_vector(9 downto 0);
		entry_payload_12_o : out std_logic_vector(9 downto 0);
		entry_payload_13_o : out std_logic_vector(9 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		entry_wen_2_o : out std_logic;
		entry_wen_3_o : out std_logic;
		entry_wen_4_o : out std_logic;
		entry_wen_5_o : out std_logic;
		entry_wen_6_o : out std_logic;
		entry_wen_7_o : out std_logic;
		entry_wen_8_o : out std_logic;
		entry_wen_9_o : out std_logic;
		entry_wen_10_o : out std_logic;
		entry_wen_11_o : out std_logic;
		entry_wen_12_o : out std_logic;
		entry_wen_13_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(13 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_lda is
	signal entry_port_idx_oh_0 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_2 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_3 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_4 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_5 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_6 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_7 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_8 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_9 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_10 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_11 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_12 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_13 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_4_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_5_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_6_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_7_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_8_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_9_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_10_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_11_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_12_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_13_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_14_mux_0 : std_logic_vector(9 downto 0);
	signal entry_ptq_ready_0 : std_logic;
	signal entry_ptq_ready_1 : std_logic;
	signal entry_ptq_ready_2 : std_logic;
	signal entry_ptq_ready_3 : std_logic;
	signal entry_ptq_ready_4 : std_logic;
	signal entry_ptq_ready_5 : std_logic;
	signal entry_ptq_ready_6 : std_logic;
	signal entry_ptq_ready_7 : std_logic;
	signal entry_ptq_ready_8 : std_logic;
	signal entry_ptq_ready_9 : std_logic;
	signal entry_ptq_ready_10 : std_logic;
	signal entry_ptq_ready_11 : std_logic;
	signal entry_ptq_ready_12 : std_logic;
	signal entry_ptq_ready_13 : std_logic;
	signal entry_waiting_for_port_0 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_1 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_2 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_3 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_4 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_5 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_6 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_7 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_8 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_9 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_10 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_11 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_12 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_13 : std_logic_vector(0 downto 0);
	signal port_ready_vec : std_logic_vector(0 downto 0);
	signal TEMP_15_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_4 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_5 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_6 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_7 : std_logic_vector(0 downto 0);
	signal TEMP_16_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_16_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_16_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_16_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_17_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_17_res_1 : std_logic_vector(0 downto 0);
	signal entry_port_options_0 : std_logic_vector(0 downto 0);
	signal entry_port_options_1 : std_logic_vector(0 downto 0);
	signal entry_port_options_2 : std_logic_vector(0 downto 0);
	signal entry_port_options_3 : std_logic_vector(0 downto 0);
	signal entry_port_options_4 : std_logic_vector(0 downto 0);
	signal entry_port_options_5 : std_logic_vector(0 downto 0);
	signal entry_port_options_6 : std_logic_vector(0 downto 0);
	signal entry_port_options_7 : std_logic_vector(0 downto 0);
	signal entry_port_options_8 : std_logic_vector(0 downto 0);
	signal entry_port_options_9 : std_logic_vector(0 downto 0);
	signal entry_port_options_10 : std_logic_vector(0 downto 0);
	signal entry_port_options_11 : std_logic_vector(0 downto 0);
	signal entry_port_options_12 : std_logic_vector(0 downto 0);
	signal entry_port_options_13 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_2 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_3 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_4 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_5 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_6 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_7 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_8 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_9 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_10 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_11 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_12 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_13 : std_logic_vector(0 downto 0);
	signal TEMP_18_double_in_0 : std_logic_vector(27 downto 0);
	signal TEMP_18_double_out_0 : std_logic_vector(27 downto 0);
begin
	entry_port_idx_oh_0 <= "1";
	entry_port_idx_oh_1 <= "1";
	entry_port_idx_oh_2 <= "1";
	entry_port_idx_oh_3 <= "1";
	entry_port_idx_oh_4 <= "1";
	entry_port_idx_oh_5 <= "1";
	entry_port_idx_oh_6 <= "1";
	entry_port_idx_oh_7 <= "1";
	entry_port_idx_oh_8 <= "1";
	entry_port_idx_oh_9 <= "1";
	entry_port_idx_oh_10 <= "1";
	entry_port_idx_oh_11 <= "1";
	entry_port_idx_oh_12 <= "1";
	entry_port_idx_oh_13 <= "1";
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

	-- Mux1H Begin
	-- Mux1H(entry_payload_5, port_payload, entry_port_idx_oh_5)
	TEMP_6_mux_0 <= port_payload_0_i when entry_port_idx_oh_5(0) = '1' else "0000000000";
	entry_payload_5_o <= TEMP_6_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_6, port_payload, entry_port_idx_oh_6)
	TEMP_7_mux_0 <= port_payload_0_i when entry_port_idx_oh_6(0) = '1' else "0000000000";
	entry_payload_6_o <= TEMP_7_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_7, port_payload, entry_port_idx_oh_7)
	TEMP_8_mux_0 <= port_payload_0_i when entry_port_idx_oh_7(0) = '1' else "0000000000";
	entry_payload_7_o <= TEMP_8_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_8, port_payload, entry_port_idx_oh_8)
	TEMP_9_mux_0 <= port_payload_0_i when entry_port_idx_oh_8(0) = '1' else "0000000000";
	entry_payload_8_o <= TEMP_9_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_9, port_payload, entry_port_idx_oh_9)
	TEMP_10_mux_0 <= port_payload_0_i when entry_port_idx_oh_9(0) = '1' else "0000000000";
	entry_payload_9_o <= TEMP_10_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_10, port_payload, entry_port_idx_oh_10)
	TEMP_11_mux_0 <= port_payload_0_i when entry_port_idx_oh_10(0) = '1' else "0000000000";
	entry_payload_10_o <= TEMP_11_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_11, port_payload, entry_port_idx_oh_11)
	TEMP_12_mux_0 <= port_payload_0_i when entry_port_idx_oh_11(0) = '1' else "0000000000";
	entry_payload_11_o <= TEMP_12_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_12, port_payload, entry_port_idx_oh_12)
	TEMP_13_mux_0 <= port_payload_0_i when entry_port_idx_oh_12(0) = '1' else "0000000000";
	entry_payload_12_o <= TEMP_13_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_13, port_payload, entry_port_idx_oh_13)
	TEMP_14_mux_0 <= port_payload_0_i when entry_port_idx_oh_13(0) = '1' else "0000000000";
	entry_payload_13_o <= TEMP_14_mux_0;
	-- Mux1H End

	entry_ptq_ready_0 <= entry_alloc_0_i and not entry_payload_valid_0_i;
	entry_ptq_ready_1 <= entry_alloc_1_i and not entry_payload_valid_1_i;
	entry_ptq_ready_2 <= entry_alloc_2_i and not entry_payload_valid_2_i;
	entry_ptq_ready_3 <= entry_alloc_3_i and not entry_payload_valid_3_i;
	entry_ptq_ready_4 <= entry_alloc_4_i and not entry_payload_valid_4_i;
	entry_ptq_ready_5 <= entry_alloc_5_i and not entry_payload_valid_5_i;
	entry_ptq_ready_6 <= entry_alloc_6_i and not entry_payload_valid_6_i;
	entry_ptq_ready_7 <= entry_alloc_7_i and not entry_payload_valid_7_i;
	entry_ptq_ready_8 <= entry_alloc_8_i and not entry_payload_valid_8_i;
	entry_ptq_ready_9 <= entry_alloc_9_i and not entry_payload_valid_9_i;
	entry_ptq_ready_10 <= entry_alloc_10_i and not entry_payload_valid_10_i;
	entry_ptq_ready_11 <= entry_alloc_11_i and not entry_payload_valid_11_i;
	entry_ptq_ready_12 <= entry_alloc_12_i and not entry_payload_valid_12_i;
	entry_ptq_ready_13 <= entry_alloc_13_i and not entry_payload_valid_13_i;
	entry_waiting_for_port_0 <= entry_port_idx_oh_0 when entry_ptq_ready_0 else "0";
	entry_waiting_for_port_1 <= entry_port_idx_oh_1 when entry_ptq_ready_1 else "0";
	entry_waiting_for_port_2 <= entry_port_idx_oh_2 when entry_ptq_ready_2 else "0";
	entry_waiting_for_port_3 <= entry_port_idx_oh_3 when entry_ptq_ready_3 else "0";
	entry_waiting_for_port_4 <= entry_port_idx_oh_4 when entry_ptq_ready_4 else "0";
	entry_waiting_for_port_5 <= entry_port_idx_oh_5 when entry_ptq_ready_5 else "0";
	entry_waiting_for_port_6 <= entry_port_idx_oh_6 when entry_ptq_ready_6 else "0";
	entry_waiting_for_port_7 <= entry_port_idx_oh_7 when entry_ptq_ready_7 else "0";
	entry_waiting_for_port_8 <= entry_port_idx_oh_8 when entry_ptq_ready_8 else "0";
	entry_waiting_for_port_9 <= entry_port_idx_oh_9 when entry_ptq_ready_9 else "0";
	entry_waiting_for_port_10 <= entry_port_idx_oh_10 when entry_ptq_ready_10 else "0";
	entry_waiting_for_port_11 <= entry_port_idx_oh_11 when entry_ptq_ready_11 else "0";
	entry_waiting_for_port_12 <= entry_port_idx_oh_12 when entry_ptq_ready_12 else "0";
	entry_waiting_for_port_13 <= entry_port_idx_oh_13 when entry_ptq_ready_13 else "0";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_waiting_for_port, or)
	TEMP_15_res_0 <= entry_waiting_for_port_0 or entry_waiting_for_port_8;
	TEMP_15_res_1 <= entry_waiting_for_port_1 or entry_waiting_for_port_9;
	TEMP_15_res_2 <= entry_waiting_for_port_2 or entry_waiting_for_port_10;
	TEMP_15_res_3 <= entry_waiting_for_port_3 or entry_waiting_for_port_11;
	TEMP_15_res_4 <= entry_waiting_for_port_4 or entry_waiting_for_port_12;
	TEMP_15_res_5 <= entry_waiting_for_port_5 or entry_waiting_for_port_13;
	TEMP_15_res_6 <= entry_waiting_for_port_6;
	TEMP_15_res_7 <= entry_waiting_for_port_7;
	-- Layer End
	TEMP_16_res_0 <= TEMP_15_res_0 or TEMP_15_res_4;
	TEMP_16_res_1 <= TEMP_15_res_1 or TEMP_15_res_5;
	TEMP_16_res_2 <= TEMP_15_res_2 or TEMP_15_res_6;
	TEMP_16_res_3 <= TEMP_15_res_3 or TEMP_15_res_7;
	-- Layer End
	TEMP_17_res_0 <= TEMP_16_res_0 or TEMP_16_res_2;
	TEMP_17_res_1 <= TEMP_16_res_1 or TEMP_16_res_3;
	-- Layer End
	port_ready_vec <= TEMP_17_res_0 or TEMP_17_res_1;
	-- Reduction End

	port_ready_0_o <= port_ready_vec(0);
	entry_port_options_0(0) <= entry_waiting_for_port_0(0) and port_valid_0_i;
	entry_port_options_1(0) <= entry_waiting_for_port_1(0) and port_valid_0_i;
	entry_port_options_2(0) <= entry_waiting_for_port_2(0) and port_valid_0_i;
	entry_port_options_3(0) <= entry_waiting_for_port_3(0) and port_valid_0_i;
	entry_port_options_4(0) <= entry_waiting_for_port_4(0) and port_valid_0_i;
	entry_port_options_5(0) <= entry_waiting_for_port_5(0) and port_valid_0_i;
	entry_port_options_6(0) <= entry_waiting_for_port_6(0) and port_valid_0_i;
	entry_port_options_7(0) <= entry_waiting_for_port_7(0) and port_valid_0_i;
	entry_port_options_8(0) <= entry_waiting_for_port_8(0) and port_valid_0_i;
	entry_port_options_9(0) <= entry_waiting_for_port_9(0) and port_valid_0_i;
	entry_port_options_10(0) <= entry_waiting_for_port_10(0) and port_valid_0_i;
	entry_port_options_11(0) <= entry_waiting_for_port_11(0) and port_valid_0_i;
	entry_port_options_12(0) <= entry_waiting_for_port_12(0) and port_valid_0_i;
	entry_port_options_13(0) <= entry_waiting_for_port_13(0) and port_valid_0_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_transfer, entry_port_options, queue_head_oh)
	TEMP_18_double_in_0(0) <= entry_port_options_0(0);
	TEMP_18_double_in_0(14) <= entry_port_options_0(0);
	TEMP_18_double_in_0(1) <= entry_port_options_1(0);
	TEMP_18_double_in_0(15) <= entry_port_options_1(0);
	TEMP_18_double_in_0(2) <= entry_port_options_2(0);
	TEMP_18_double_in_0(16) <= entry_port_options_2(0);
	TEMP_18_double_in_0(3) <= entry_port_options_3(0);
	TEMP_18_double_in_0(17) <= entry_port_options_3(0);
	TEMP_18_double_in_0(4) <= entry_port_options_4(0);
	TEMP_18_double_in_0(18) <= entry_port_options_4(0);
	TEMP_18_double_in_0(5) <= entry_port_options_5(0);
	TEMP_18_double_in_0(19) <= entry_port_options_5(0);
	TEMP_18_double_in_0(6) <= entry_port_options_6(0);
	TEMP_18_double_in_0(20) <= entry_port_options_6(0);
	TEMP_18_double_in_0(7) <= entry_port_options_7(0);
	TEMP_18_double_in_0(21) <= entry_port_options_7(0);
	TEMP_18_double_in_0(8) <= entry_port_options_8(0);
	TEMP_18_double_in_0(22) <= entry_port_options_8(0);
	TEMP_18_double_in_0(9) <= entry_port_options_9(0);
	TEMP_18_double_in_0(23) <= entry_port_options_9(0);
	TEMP_18_double_in_0(10) <= entry_port_options_10(0);
	TEMP_18_double_in_0(24) <= entry_port_options_10(0);
	TEMP_18_double_in_0(11) <= entry_port_options_11(0);
	TEMP_18_double_in_0(25) <= entry_port_options_11(0);
	TEMP_18_double_in_0(12) <= entry_port_options_12(0);
	TEMP_18_double_in_0(26) <= entry_port_options_12(0);
	TEMP_18_double_in_0(13) <= entry_port_options_13(0);
	TEMP_18_double_in_0(27) <= entry_port_options_13(0);
	TEMP_18_double_out_0 <= TEMP_18_double_in_0 and not std_logic_vector( unsigned( TEMP_18_double_in_0 ) - unsigned( "00000000000000" & queue_head_oh_i ) );
	entry_port_transfer_0(0) <= TEMP_18_double_out_0(0) or TEMP_18_double_out_0(14);
	entry_port_transfer_1(0) <= TEMP_18_double_out_0(1) or TEMP_18_double_out_0(15);
	entry_port_transfer_2(0) <= TEMP_18_double_out_0(2) or TEMP_18_double_out_0(16);
	entry_port_transfer_3(0) <= TEMP_18_double_out_0(3) or TEMP_18_double_out_0(17);
	entry_port_transfer_4(0) <= TEMP_18_double_out_0(4) or TEMP_18_double_out_0(18);
	entry_port_transfer_5(0) <= TEMP_18_double_out_0(5) or TEMP_18_double_out_0(19);
	entry_port_transfer_6(0) <= TEMP_18_double_out_0(6) or TEMP_18_double_out_0(20);
	entry_port_transfer_7(0) <= TEMP_18_double_out_0(7) or TEMP_18_double_out_0(21);
	entry_port_transfer_8(0) <= TEMP_18_double_out_0(8) or TEMP_18_double_out_0(22);
	entry_port_transfer_9(0) <= TEMP_18_double_out_0(9) or TEMP_18_double_out_0(23);
	entry_port_transfer_10(0) <= TEMP_18_double_out_0(10) or TEMP_18_double_out_0(24);
	entry_port_transfer_11(0) <= TEMP_18_double_out_0(11) or TEMP_18_double_out_0(25);
	entry_port_transfer_12(0) <= TEMP_18_double_out_0(12) or TEMP_18_double_out_0(26);
	entry_port_transfer_13(0) <= TEMP_18_double_out_0(13) or TEMP_18_double_out_0(27);
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

	-- Reduction Begin
	-- Reduce(entry_wen_5, entry_port_transfer_5, or)
	entry_wen_5_o <= entry_port_transfer_5(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_6, entry_port_transfer_6, or)
	entry_wen_6_o <= entry_port_transfer_6(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_7, entry_port_transfer_7, or)
	entry_wen_7_o <= entry_port_transfer_7(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_8, entry_port_transfer_8, or)
	entry_wen_8_o <= entry_port_transfer_8(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_9, entry_port_transfer_9, or)
	entry_wen_9_o <= entry_port_transfer_9(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_10, entry_port_transfer_10, or)
	entry_wen_10_o <= entry_port_transfer_10(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_11, entry_port_transfer_11, or)
	entry_wen_11_o <= entry_port_transfer_11(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_12, entry_port_transfer_12, or)
	entry_wen_12_o <= entry_port_transfer_12(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_13, entry_port_transfer_13, or)
	entry_wen_13_o <= entry_port_transfer_13(0);
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
		entry_alloc_2_i : in std_logic;
		entry_alloc_3_i : in std_logic;
		entry_alloc_4_i : in std_logic;
		entry_alloc_5_i : in std_logic;
		entry_alloc_6_i : in std_logic;
		entry_alloc_7_i : in std_logic;
		entry_alloc_8_i : in std_logic;
		entry_alloc_9_i : in std_logic;
		entry_alloc_10_i : in std_logic;
		entry_alloc_11_i : in std_logic;
		entry_alloc_12_i : in std_logic;
		entry_alloc_13_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_valid_2_i : in std_logic;
		entry_payload_valid_3_i : in std_logic;
		entry_payload_valid_4_i : in std_logic;
		entry_payload_valid_5_i : in std_logic;
		entry_payload_valid_6_i : in std_logic;
		entry_payload_valid_7_i : in std_logic;
		entry_payload_valid_8_i : in std_logic;
		entry_payload_valid_9_i : in std_logic;
		entry_payload_valid_10_i : in std_logic;
		entry_payload_valid_11_i : in std_logic;
		entry_payload_valid_12_i : in std_logic;
		entry_payload_valid_13_i : in std_logic;
		entry_payload_0_i : in std_logic_vector(31 downto 0);
		entry_payload_1_i : in std_logic_vector(31 downto 0);
		entry_payload_2_i : in std_logic_vector(31 downto 0);
		entry_payload_3_i : in std_logic_vector(31 downto 0);
		entry_payload_4_i : in std_logic_vector(31 downto 0);
		entry_payload_5_i : in std_logic_vector(31 downto 0);
		entry_payload_6_i : in std_logic_vector(31 downto 0);
		entry_payload_7_i : in std_logic_vector(31 downto 0);
		entry_payload_8_i : in std_logic_vector(31 downto 0);
		entry_payload_9_i : in std_logic_vector(31 downto 0);
		entry_payload_10_i : in std_logic_vector(31 downto 0);
		entry_payload_11_i : in std_logic_vector(31 downto 0);
		entry_payload_12_i : in std_logic_vector(31 downto 0);
		entry_payload_13_i : in std_logic_vector(31 downto 0);
		entry_reset_0_o : out std_logic;
		entry_reset_1_o : out std_logic;
		entry_reset_2_o : out std_logic;
		entry_reset_3_o : out std_logic;
		entry_reset_4_o : out std_logic;
		entry_reset_5_o : out std_logic;
		entry_reset_6_o : out std_logic;
		entry_reset_7_o : out std_logic;
		entry_reset_8_o : out std_logic;
		entry_reset_9_o : out std_logic;
		entry_reset_10_o : out std_logic;
		entry_reset_11_o : out std_logic;
		entry_reset_12_o : out std_logic;
		entry_reset_13_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(13 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_ldd is
	signal entry_port_idx_oh_0 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_2 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_3 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_4 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_5 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_6 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_7 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_8 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_9 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_10 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_11 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_12 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_13 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_0 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_1 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_2 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_3 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_4 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_5 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_6 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_7 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_8 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_9 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_10 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_11 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_12 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_13 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_0 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_1 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_2 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_3 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_4 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_5 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_6 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_7 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_8 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_9 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_10 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_11 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_12 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_13 : std_logic_vector(0 downto 0);
	signal TEMP_1_double_in_0 : std_logic_vector(27 downto 0);
	signal TEMP_1_double_out_0 : std_logic_vector(27 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_10 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_11 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_12 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_13 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_4_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_4_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_4_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_4_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_5_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_5_res_1 : std_logic_vector(31 downto 0);
	signal entry_waiting_for_port_valid_0 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_1 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_2 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_3 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_4 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_5 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_6 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_7 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_8 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_9 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_10 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_11 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_12 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_13 : std_logic_vector(0 downto 0);
	signal port_valid_vec : std_logic_vector(0 downto 0);
	signal TEMP_6_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_4 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_5 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_6 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_7 : std_logic_vector(0 downto 0);
	signal TEMP_7_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_7_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_7_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_7_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_8_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_8_res_1 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_2 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_3 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_4 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_5 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_6 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_7 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_8 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_9 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_10 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_11 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_12 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_13 : std_logic_vector(0 downto 0);
begin
	entry_port_idx_oh_0 <= "1";
	entry_port_idx_oh_1 <= "1";
	entry_port_idx_oh_2 <= "1";
	entry_port_idx_oh_3 <= "1";
	entry_port_idx_oh_4 <= "1";
	entry_port_idx_oh_5 <= "1";
	entry_port_idx_oh_6 <= "1";
	entry_port_idx_oh_7 <= "1";
	entry_port_idx_oh_8 <= "1";
	entry_port_idx_oh_9 <= "1";
	entry_port_idx_oh_10 <= "1";
	entry_port_idx_oh_11 <= "1";
	entry_port_idx_oh_12 <= "1";
	entry_port_idx_oh_13 <= "1";
	entry_allocated_for_port_0 <= entry_port_idx_oh_0 when entry_alloc_0_i else "0";
	entry_allocated_for_port_1 <= entry_port_idx_oh_1 when entry_alloc_1_i else "0";
	entry_allocated_for_port_2 <= entry_port_idx_oh_2 when entry_alloc_2_i else "0";
	entry_allocated_for_port_3 <= entry_port_idx_oh_3 when entry_alloc_3_i else "0";
	entry_allocated_for_port_4 <= entry_port_idx_oh_4 when entry_alloc_4_i else "0";
	entry_allocated_for_port_5 <= entry_port_idx_oh_5 when entry_alloc_5_i else "0";
	entry_allocated_for_port_6 <= entry_port_idx_oh_6 when entry_alloc_6_i else "0";
	entry_allocated_for_port_7 <= entry_port_idx_oh_7 when entry_alloc_7_i else "0";
	entry_allocated_for_port_8 <= entry_port_idx_oh_8 when entry_alloc_8_i else "0";
	entry_allocated_for_port_9 <= entry_port_idx_oh_9 when entry_alloc_9_i else "0";
	entry_allocated_for_port_10 <= entry_port_idx_oh_10 when entry_alloc_10_i else "0";
	entry_allocated_for_port_11 <= entry_port_idx_oh_11 when entry_alloc_11_i else "0";
	entry_allocated_for_port_12 <= entry_port_idx_oh_12 when entry_alloc_12_i else "0";
	entry_allocated_for_port_13 <= entry_port_idx_oh_13 when entry_alloc_13_i else "0";
	-- Priority Masking Begin
	-- CyclicPriorityMask(oldest_entry_allocated_per_port, entry_allocated_for_port, queue_head_oh)
	TEMP_1_double_in_0(0) <= entry_allocated_for_port_0(0);
	TEMP_1_double_in_0(14) <= entry_allocated_for_port_0(0);
	TEMP_1_double_in_0(1) <= entry_allocated_for_port_1(0);
	TEMP_1_double_in_0(15) <= entry_allocated_for_port_1(0);
	TEMP_1_double_in_0(2) <= entry_allocated_for_port_2(0);
	TEMP_1_double_in_0(16) <= entry_allocated_for_port_2(0);
	TEMP_1_double_in_0(3) <= entry_allocated_for_port_3(0);
	TEMP_1_double_in_0(17) <= entry_allocated_for_port_3(0);
	TEMP_1_double_in_0(4) <= entry_allocated_for_port_4(0);
	TEMP_1_double_in_0(18) <= entry_allocated_for_port_4(0);
	TEMP_1_double_in_0(5) <= entry_allocated_for_port_5(0);
	TEMP_1_double_in_0(19) <= entry_allocated_for_port_5(0);
	TEMP_1_double_in_0(6) <= entry_allocated_for_port_6(0);
	TEMP_1_double_in_0(20) <= entry_allocated_for_port_6(0);
	TEMP_1_double_in_0(7) <= entry_allocated_for_port_7(0);
	TEMP_1_double_in_0(21) <= entry_allocated_for_port_7(0);
	TEMP_1_double_in_0(8) <= entry_allocated_for_port_8(0);
	TEMP_1_double_in_0(22) <= entry_allocated_for_port_8(0);
	TEMP_1_double_in_0(9) <= entry_allocated_for_port_9(0);
	TEMP_1_double_in_0(23) <= entry_allocated_for_port_9(0);
	TEMP_1_double_in_0(10) <= entry_allocated_for_port_10(0);
	TEMP_1_double_in_0(24) <= entry_allocated_for_port_10(0);
	TEMP_1_double_in_0(11) <= entry_allocated_for_port_11(0);
	TEMP_1_double_in_0(25) <= entry_allocated_for_port_11(0);
	TEMP_1_double_in_0(12) <= entry_allocated_for_port_12(0);
	TEMP_1_double_in_0(26) <= entry_allocated_for_port_12(0);
	TEMP_1_double_in_0(13) <= entry_allocated_for_port_13(0);
	TEMP_1_double_in_0(27) <= entry_allocated_for_port_13(0);
	TEMP_1_double_out_0 <= TEMP_1_double_in_0 and not std_logic_vector( unsigned( TEMP_1_double_in_0 ) - unsigned( "00000000000000" & queue_head_oh_i ) );
	oldest_entry_allocated_per_port_0(0) <= TEMP_1_double_out_0(0) or TEMP_1_double_out_0(14);
	oldest_entry_allocated_per_port_1(0) <= TEMP_1_double_out_0(1) or TEMP_1_double_out_0(15);
	oldest_entry_allocated_per_port_2(0) <= TEMP_1_double_out_0(2) or TEMP_1_double_out_0(16);
	oldest_entry_allocated_per_port_3(0) <= TEMP_1_double_out_0(3) or TEMP_1_double_out_0(17);
	oldest_entry_allocated_per_port_4(0) <= TEMP_1_double_out_0(4) or TEMP_1_double_out_0(18);
	oldest_entry_allocated_per_port_5(0) <= TEMP_1_double_out_0(5) or TEMP_1_double_out_0(19);
	oldest_entry_allocated_per_port_6(0) <= TEMP_1_double_out_0(6) or TEMP_1_double_out_0(20);
	oldest_entry_allocated_per_port_7(0) <= TEMP_1_double_out_0(7) or TEMP_1_double_out_0(21);
	oldest_entry_allocated_per_port_8(0) <= TEMP_1_double_out_0(8) or TEMP_1_double_out_0(22);
	oldest_entry_allocated_per_port_9(0) <= TEMP_1_double_out_0(9) or TEMP_1_double_out_0(23);
	oldest_entry_allocated_per_port_10(0) <= TEMP_1_double_out_0(10) or TEMP_1_double_out_0(24);
	oldest_entry_allocated_per_port_11(0) <= TEMP_1_double_out_0(11) or TEMP_1_double_out_0(25);
	oldest_entry_allocated_per_port_12(0) <= TEMP_1_double_out_0(12) or TEMP_1_double_out_0(26);
	oldest_entry_allocated_per_port_13(0) <= TEMP_1_double_out_0(13) or TEMP_1_double_out_0(27);
	-- Priority Masking End

	-- Mux1H Begin
	-- Mux1H(port_payload_0, entry_payload, oldest_entry_allocated_per_port)
	TEMP_2_mux_0 <= entry_payload_0_i when oldest_entry_allocated_per_port_0(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_1 <= entry_payload_1_i when oldest_entry_allocated_per_port_1(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_2 <= entry_payload_2_i when oldest_entry_allocated_per_port_2(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_3 <= entry_payload_3_i when oldest_entry_allocated_per_port_3(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_4 <= entry_payload_4_i when oldest_entry_allocated_per_port_4(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_5 <= entry_payload_5_i when oldest_entry_allocated_per_port_5(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_6 <= entry_payload_6_i when oldest_entry_allocated_per_port_6(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_7 <= entry_payload_7_i when oldest_entry_allocated_per_port_7(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_8 <= entry_payload_8_i when oldest_entry_allocated_per_port_8(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_9 <= entry_payload_9_i when oldest_entry_allocated_per_port_9(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_10 <= entry_payload_10_i when oldest_entry_allocated_per_port_10(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_11 <= entry_payload_11_i when oldest_entry_allocated_per_port_11(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_12 <= entry_payload_12_i when oldest_entry_allocated_per_port_12(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_13 <= entry_payload_13_i when oldest_entry_allocated_per_port_13(0) = '1' else "00000000000000000000000000000000";
	TEMP_3_res_0 <= TEMP_2_mux_0 or TEMP_2_mux_8;
	TEMP_3_res_1 <= TEMP_2_mux_1 or TEMP_2_mux_9;
	TEMP_3_res_2 <= TEMP_2_mux_2 or TEMP_2_mux_10;
	TEMP_3_res_3 <= TEMP_2_mux_3 or TEMP_2_mux_11;
	TEMP_3_res_4 <= TEMP_2_mux_4 or TEMP_2_mux_12;
	TEMP_3_res_5 <= TEMP_2_mux_5 or TEMP_2_mux_13;
	TEMP_3_res_6 <= TEMP_2_mux_6;
	TEMP_3_res_7 <= TEMP_2_mux_7;
	-- Layer End
	TEMP_4_res_0 <= TEMP_3_res_0 or TEMP_3_res_4;
	TEMP_4_res_1 <= TEMP_3_res_1 or TEMP_3_res_5;
	TEMP_4_res_2 <= TEMP_3_res_2 or TEMP_3_res_6;
	TEMP_4_res_3 <= TEMP_3_res_3 or TEMP_3_res_7;
	-- Layer End
	TEMP_5_res_0 <= TEMP_4_res_0 or TEMP_4_res_2;
	TEMP_5_res_1 <= TEMP_4_res_1 or TEMP_4_res_3;
	-- Layer End
	port_payload_0_o <= TEMP_5_res_0 or TEMP_5_res_1;
	-- Mux1H End

	entry_waiting_for_port_valid_0 <= oldest_entry_allocated_per_port_0 when entry_payload_valid_0_i else "0";
	entry_waiting_for_port_valid_1 <= oldest_entry_allocated_per_port_1 when entry_payload_valid_1_i else "0";
	entry_waiting_for_port_valid_2 <= oldest_entry_allocated_per_port_2 when entry_payload_valid_2_i else "0";
	entry_waiting_for_port_valid_3 <= oldest_entry_allocated_per_port_3 when entry_payload_valid_3_i else "0";
	entry_waiting_for_port_valid_4 <= oldest_entry_allocated_per_port_4 when entry_payload_valid_4_i else "0";
	entry_waiting_for_port_valid_5 <= oldest_entry_allocated_per_port_5 when entry_payload_valid_5_i else "0";
	entry_waiting_for_port_valid_6 <= oldest_entry_allocated_per_port_6 when entry_payload_valid_6_i else "0";
	entry_waiting_for_port_valid_7 <= oldest_entry_allocated_per_port_7 when entry_payload_valid_7_i else "0";
	entry_waiting_for_port_valid_8 <= oldest_entry_allocated_per_port_8 when entry_payload_valid_8_i else "0";
	entry_waiting_for_port_valid_9 <= oldest_entry_allocated_per_port_9 when entry_payload_valid_9_i else "0";
	entry_waiting_for_port_valid_10 <= oldest_entry_allocated_per_port_10 when entry_payload_valid_10_i else "0";
	entry_waiting_for_port_valid_11 <= oldest_entry_allocated_per_port_11 when entry_payload_valid_11_i else "0";
	entry_waiting_for_port_valid_12 <= oldest_entry_allocated_per_port_12 when entry_payload_valid_12_i else "0";
	entry_waiting_for_port_valid_13 <= oldest_entry_allocated_per_port_13 when entry_payload_valid_13_i else "0";
	-- Reduction Begin
	-- Reduce(port_valid_vec, entry_waiting_for_port_valid, or)
	TEMP_6_res_0 <= entry_waiting_for_port_valid_0 or entry_waiting_for_port_valid_8;
	TEMP_6_res_1 <= entry_waiting_for_port_valid_1 or entry_waiting_for_port_valid_9;
	TEMP_6_res_2 <= entry_waiting_for_port_valid_2 or entry_waiting_for_port_valid_10;
	TEMP_6_res_3 <= entry_waiting_for_port_valid_3 or entry_waiting_for_port_valid_11;
	TEMP_6_res_4 <= entry_waiting_for_port_valid_4 or entry_waiting_for_port_valid_12;
	TEMP_6_res_5 <= entry_waiting_for_port_valid_5 or entry_waiting_for_port_valid_13;
	TEMP_6_res_6 <= entry_waiting_for_port_valid_6;
	TEMP_6_res_7 <= entry_waiting_for_port_valid_7;
	-- Layer End
	TEMP_7_res_0 <= TEMP_6_res_0 or TEMP_6_res_4;
	TEMP_7_res_1 <= TEMP_6_res_1 or TEMP_6_res_5;
	TEMP_7_res_2 <= TEMP_6_res_2 or TEMP_6_res_6;
	TEMP_7_res_3 <= TEMP_6_res_3 or TEMP_6_res_7;
	-- Layer End
	TEMP_8_res_0 <= TEMP_7_res_0 or TEMP_7_res_2;
	TEMP_8_res_1 <= TEMP_7_res_1 or TEMP_7_res_3;
	-- Layer End
	port_valid_vec <= TEMP_8_res_0 or TEMP_8_res_1;
	-- Reduction End

	port_valid_0_o <= port_valid_vec(0);
	entry_port_transfer_0(0) <= entry_waiting_for_port_valid_0(0) and port_ready_0_i;
	entry_port_transfer_1(0) <= entry_waiting_for_port_valid_1(0) and port_ready_0_i;
	entry_port_transfer_2(0) <= entry_waiting_for_port_valid_2(0) and port_ready_0_i;
	entry_port_transfer_3(0) <= entry_waiting_for_port_valid_3(0) and port_ready_0_i;
	entry_port_transfer_4(0) <= entry_waiting_for_port_valid_4(0) and port_ready_0_i;
	entry_port_transfer_5(0) <= entry_waiting_for_port_valid_5(0) and port_ready_0_i;
	entry_port_transfer_6(0) <= entry_waiting_for_port_valid_6(0) and port_ready_0_i;
	entry_port_transfer_7(0) <= entry_waiting_for_port_valid_7(0) and port_ready_0_i;
	entry_port_transfer_8(0) <= entry_waiting_for_port_valid_8(0) and port_ready_0_i;
	entry_port_transfer_9(0) <= entry_waiting_for_port_valid_9(0) and port_ready_0_i;
	entry_port_transfer_10(0) <= entry_waiting_for_port_valid_10(0) and port_ready_0_i;
	entry_port_transfer_11(0) <= entry_waiting_for_port_valid_11(0) and port_ready_0_i;
	entry_port_transfer_12(0) <= entry_waiting_for_port_valid_12(0) and port_ready_0_i;
	entry_port_transfer_13(0) <= entry_waiting_for_port_valid_13(0) and port_ready_0_i;
	-- Reduction Begin
	-- Reduce(entry_reset_0, entry_port_transfer_0, or)
	entry_reset_0_o <= entry_port_transfer_0(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_1, entry_port_transfer_1, or)
	entry_reset_1_o <= entry_port_transfer_1(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_2, entry_port_transfer_2, or)
	entry_reset_2_o <= entry_port_transfer_2(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_3, entry_port_transfer_3, or)
	entry_reset_3_o <= entry_port_transfer_3(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_4, entry_port_transfer_4, or)
	entry_reset_4_o <= entry_port_transfer_4(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_5, entry_port_transfer_5, or)
	entry_reset_5_o <= entry_port_transfer_5(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_6, entry_port_transfer_6, or)
	entry_reset_6_o <= entry_port_transfer_6(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_7, entry_port_transfer_7, or)
	entry_reset_7_o <= entry_port_transfer_7(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_8, entry_port_transfer_8, or)
	entry_reset_8_o <= entry_port_transfer_8(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_9, entry_port_transfer_9, or)
	entry_reset_9_o <= entry_port_transfer_9(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_10, entry_port_transfer_10, or)
	entry_reset_10_o <= entry_port_transfer_10(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_11, entry_port_transfer_11, or)
	entry_reset_11_o <= entry_port_transfer_11(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_12, entry_port_transfer_12, or)
	entry_reset_12_o <= entry_port_transfer_12(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_13, entry_port_transfer_13, or)
	entry_reset_13_o <= entry_port_transfer_13(0);
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
		entry_alloc_5_i : in std_logic;
		entry_alloc_6_i : in std_logic;
		entry_alloc_7_i : in std_logic;
		entry_alloc_8_i : in std_logic;
		entry_alloc_9_i : in std_logic;
		entry_alloc_10_i : in std_logic;
		entry_alloc_11_i : in std_logic;
		entry_alloc_12_i : in std_logic;
		entry_alloc_13_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_valid_2_i : in std_logic;
		entry_payload_valid_3_i : in std_logic;
		entry_payload_valid_4_i : in std_logic;
		entry_payload_valid_5_i : in std_logic;
		entry_payload_valid_6_i : in std_logic;
		entry_payload_valid_7_i : in std_logic;
		entry_payload_valid_8_i : in std_logic;
		entry_payload_valid_9_i : in std_logic;
		entry_payload_valid_10_i : in std_logic;
		entry_payload_valid_11_i : in std_logic;
		entry_payload_valid_12_i : in std_logic;
		entry_payload_valid_13_i : in std_logic;
		entry_payload_0_o : out std_logic_vector(9 downto 0);
		entry_payload_1_o : out std_logic_vector(9 downto 0);
		entry_payload_2_o : out std_logic_vector(9 downto 0);
		entry_payload_3_o : out std_logic_vector(9 downto 0);
		entry_payload_4_o : out std_logic_vector(9 downto 0);
		entry_payload_5_o : out std_logic_vector(9 downto 0);
		entry_payload_6_o : out std_logic_vector(9 downto 0);
		entry_payload_7_o : out std_logic_vector(9 downto 0);
		entry_payload_8_o : out std_logic_vector(9 downto 0);
		entry_payload_9_o : out std_logic_vector(9 downto 0);
		entry_payload_10_o : out std_logic_vector(9 downto 0);
		entry_payload_11_o : out std_logic_vector(9 downto 0);
		entry_payload_12_o : out std_logic_vector(9 downto 0);
		entry_payload_13_o : out std_logic_vector(9 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		entry_wen_2_o : out std_logic;
		entry_wen_3_o : out std_logic;
		entry_wen_4_o : out std_logic;
		entry_wen_5_o : out std_logic;
		entry_wen_6_o : out std_logic;
		entry_wen_7_o : out std_logic;
		entry_wen_8_o : out std_logic;
		entry_wen_9_o : out std_logic;
		entry_wen_10_o : out std_logic;
		entry_wen_11_o : out std_logic;
		entry_wen_12_o : out std_logic;
		entry_wen_13_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(13 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_sta is
	signal entry_port_idx_oh_0 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_2 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_3 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_4 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_5 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_6 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_7 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_8 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_9 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_10 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_11 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_12 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_13 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_4_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_5_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_6_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_7_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_8_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_9_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_10_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_11_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_12_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_13_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_14_mux_0 : std_logic_vector(9 downto 0);
	signal entry_ptq_ready_0 : std_logic;
	signal entry_ptq_ready_1 : std_logic;
	signal entry_ptq_ready_2 : std_logic;
	signal entry_ptq_ready_3 : std_logic;
	signal entry_ptq_ready_4 : std_logic;
	signal entry_ptq_ready_5 : std_logic;
	signal entry_ptq_ready_6 : std_logic;
	signal entry_ptq_ready_7 : std_logic;
	signal entry_ptq_ready_8 : std_logic;
	signal entry_ptq_ready_9 : std_logic;
	signal entry_ptq_ready_10 : std_logic;
	signal entry_ptq_ready_11 : std_logic;
	signal entry_ptq_ready_12 : std_logic;
	signal entry_ptq_ready_13 : std_logic;
	signal entry_waiting_for_port_0 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_1 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_2 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_3 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_4 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_5 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_6 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_7 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_8 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_9 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_10 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_11 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_12 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_13 : std_logic_vector(0 downto 0);
	signal port_ready_vec : std_logic_vector(0 downto 0);
	signal TEMP_15_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_4 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_5 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_6 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_7 : std_logic_vector(0 downto 0);
	signal TEMP_16_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_16_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_16_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_16_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_17_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_17_res_1 : std_logic_vector(0 downto 0);
	signal entry_port_options_0 : std_logic_vector(0 downto 0);
	signal entry_port_options_1 : std_logic_vector(0 downto 0);
	signal entry_port_options_2 : std_logic_vector(0 downto 0);
	signal entry_port_options_3 : std_logic_vector(0 downto 0);
	signal entry_port_options_4 : std_logic_vector(0 downto 0);
	signal entry_port_options_5 : std_logic_vector(0 downto 0);
	signal entry_port_options_6 : std_logic_vector(0 downto 0);
	signal entry_port_options_7 : std_logic_vector(0 downto 0);
	signal entry_port_options_8 : std_logic_vector(0 downto 0);
	signal entry_port_options_9 : std_logic_vector(0 downto 0);
	signal entry_port_options_10 : std_logic_vector(0 downto 0);
	signal entry_port_options_11 : std_logic_vector(0 downto 0);
	signal entry_port_options_12 : std_logic_vector(0 downto 0);
	signal entry_port_options_13 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_2 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_3 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_4 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_5 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_6 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_7 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_8 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_9 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_10 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_11 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_12 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_13 : std_logic_vector(0 downto 0);
	signal TEMP_18_double_in_0 : std_logic_vector(27 downto 0);
	signal TEMP_18_double_out_0 : std_logic_vector(27 downto 0);
begin
	entry_port_idx_oh_0 <= "1";
	entry_port_idx_oh_1 <= "1";
	entry_port_idx_oh_2 <= "1";
	entry_port_idx_oh_3 <= "1";
	entry_port_idx_oh_4 <= "1";
	entry_port_idx_oh_5 <= "1";
	entry_port_idx_oh_6 <= "1";
	entry_port_idx_oh_7 <= "1";
	entry_port_idx_oh_8 <= "1";
	entry_port_idx_oh_9 <= "1";
	entry_port_idx_oh_10 <= "1";
	entry_port_idx_oh_11 <= "1";
	entry_port_idx_oh_12 <= "1";
	entry_port_idx_oh_13 <= "1";
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

	-- Mux1H Begin
	-- Mux1H(entry_payload_5, port_payload, entry_port_idx_oh_5)
	TEMP_6_mux_0 <= port_payload_0_i when entry_port_idx_oh_5(0) = '1' else "0000000000";
	entry_payload_5_o <= TEMP_6_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_6, port_payload, entry_port_idx_oh_6)
	TEMP_7_mux_0 <= port_payload_0_i when entry_port_idx_oh_6(0) = '1' else "0000000000";
	entry_payload_6_o <= TEMP_7_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_7, port_payload, entry_port_idx_oh_7)
	TEMP_8_mux_0 <= port_payload_0_i when entry_port_idx_oh_7(0) = '1' else "0000000000";
	entry_payload_7_o <= TEMP_8_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_8, port_payload, entry_port_idx_oh_8)
	TEMP_9_mux_0 <= port_payload_0_i when entry_port_idx_oh_8(0) = '1' else "0000000000";
	entry_payload_8_o <= TEMP_9_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_9, port_payload, entry_port_idx_oh_9)
	TEMP_10_mux_0 <= port_payload_0_i when entry_port_idx_oh_9(0) = '1' else "0000000000";
	entry_payload_9_o <= TEMP_10_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_10, port_payload, entry_port_idx_oh_10)
	TEMP_11_mux_0 <= port_payload_0_i when entry_port_idx_oh_10(0) = '1' else "0000000000";
	entry_payload_10_o <= TEMP_11_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_11, port_payload, entry_port_idx_oh_11)
	TEMP_12_mux_0 <= port_payload_0_i when entry_port_idx_oh_11(0) = '1' else "0000000000";
	entry_payload_11_o <= TEMP_12_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_12, port_payload, entry_port_idx_oh_12)
	TEMP_13_mux_0 <= port_payload_0_i when entry_port_idx_oh_12(0) = '1' else "0000000000";
	entry_payload_12_o <= TEMP_13_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_13, port_payload, entry_port_idx_oh_13)
	TEMP_14_mux_0 <= port_payload_0_i when entry_port_idx_oh_13(0) = '1' else "0000000000";
	entry_payload_13_o <= TEMP_14_mux_0;
	-- Mux1H End

	entry_ptq_ready_0 <= entry_alloc_0_i and not entry_payload_valid_0_i;
	entry_ptq_ready_1 <= entry_alloc_1_i and not entry_payload_valid_1_i;
	entry_ptq_ready_2 <= entry_alloc_2_i and not entry_payload_valid_2_i;
	entry_ptq_ready_3 <= entry_alloc_3_i and not entry_payload_valid_3_i;
	entry_ptq_ready_4 <= entry_alloc_4_i and not entry_payload_valid_4_i;
	entry_ptq_ready_5 <= entry_alloc_5_i and not entry_payload_valid_5_i;
	entry_ptq_ready_6 <= entry_alloc_6_i and not entry_payload_valid_6_i;
	entry_ptq_ready_7 <= entry_alloc_7_i and not entry_payload_valid_7_i;
	entry_ptq_ready_8 <= entry_alloc_8_i and not entry_payload_valid_8_i;
	entry_ptq_ready_9 <= entry_alloc_9_i and not entry_payload_valid_9_i;
	entry_ptq_ready_10 <= entry_alloc_10_i and not entry_payload_valid_10_i;
	entry_ptq_ready_11 <= entry_alloc_11_i and not entry_payload_valid_11_i;
	entry_ptq_ready_12 <= entry_alloc_12_i and not entry_payload_valid_12_i;
	entry_ptq_ready_13 <= entry_alloc_13_i and not entry_payload_valid_13_i;
	entry_waiting_for_port_0 <= entry_port_idx_oh_0 when entry_ptq_ready_0 else "0";
	entry_waiting_for_port_1 <= entry_port_idx_oh_1 when entry_ptq_ready_1 else "0";
	entry_waiting_for_port_2 <= entry_port_idx_oh_2 when entry_ptq_ready_2 else "0";
	entry_waiting_for_port_3 <= entry_port_idx_oh_3 when entry_ptq_ready_3 else "0";
	entry_waiting_for_port_4 <= entry_port_idx_oh_4 when entry_ptq_ready_4 else "0";
	entry_waiting_for_port_5 <= entry_port_idx_oh_5 when entry_ptq_ready_5 else "0";
	entry_waiting_for_port_6 <= entry_port_idx_oh_6 when entry_ptq_ready_6 else "0";
	entry_waiting_for_port_7 <= entry_port_idx_oh_7 when entry_ptq_ready_7 else "0";
	entry_waiting_for_port_8 <= entry_port_idx_oh_8 when entry_ptq_ready_8 else "0";
	entry_waiting_for_port_9 <= entry_port_idx_oh_9 when entry_ptq_ready_9 else "0";
	entry_waiting_for_port_10 <= entry_port_idx_oh_10 when entry_ptq_ready_10 else "0";
	entry_waiting_for_port_11 <= entry_port_idx_oh_11 when entry_ptq_ready_11 else "0";
	entry_waiting_for_port_12 <= entry_port_idx_oh_12 when entry_ptq_ready_12 else "0";
	entry_waiting_for_port_13 <= entry_port_idx_oh_13 when entry_ptq_ready_13 else "0";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_waiting_for_port, or)
	TEMP_15_res_0 <= entry_waiting_for_port_0 or entry_waiting_for_port_8;
	TEMP_15_res_1 <= entry_waiting_for_port_1 or entry_waiting_for_port_9;
	TEMP_15_res_2 <= entry_waiting_for_port_2 or entry_waiting_for_port_10;
	TEMP_15_res_3 <= entry_waiting_for_port_3 or entry_waiting_for_port_11;
	TEMP_15_res_4 <= entry_waiting_for_port_4 or entry_waiting_for_port_12;
	TEMP_15_res_5 <= entry_waiting_for_port_5 or entry_waiting_for_port_13;
	TEMP_15_res_6 <= entry_waiting_for_port_6;
	TEMP_15_res_7 <= entry_waiting_for_port_7;
	-- Layer End
	TEMP_16_res_0 <= TEMP_15_res_0 or TEMP_15_res_4;
	TEMP_16_res_1 <= TEMP_15_res_1 or TEMP_15_res_5;
	TEMP_16_res_2 <= TEMP_15_res_2 or TEMP_15_res_6;
	TEMP_16_res_3 <= TEMP_15_res_3 or TEMP_15_res_7;
	-- Layer End
	TEMP_17_res_0 <= TEMP_16_res_0 or TEMP_16_res_2;
	TEMP_17_res_1 <= TEMP_16_res_1 or TEMP_16_res_3;
	-- Layer End
	port_ready_vec <= TEMP_17_res_0 or TEMP_17_res_1;
	-- Reduction End

	port_ready_0_o <= port_ready_vec(0);
	entry_port_options_0(0) <= entry_waiting_for_port_0(0) and port_valid_0_i;
	entry_port_options_1(0) <= entry_waiting_for_port_1(0) and port_valid_0_i;
	entry_port_options_2(0) <= entry_waiting_for_port_2(0) and port_valid_0_i;
	entry_port_options_3(0) <= entry_waiting_for_port_3(0) and port_valid_0_i;
	entry_port_options_4(0) <= entry_waiting_for_port_4(0) and port_valid_0_i;
	entry_port_options_5(0) <= entry_waiting_for_port_5(0) and port_valid_0_i;
	entry_port_options_6(0) <= entry_waiting_for_port_6(0) and port_valid_0_i;
	entry_port_options_7(0) <= entry_waiting_for_port_7(0) and port_valid_0_i;
	entry_port_options_8(0) <= entry_waiting_for_port_8(0) and port_valid_0_i;
	entry_port_options_9(0) <= entry_waiting_for_port_9(0) and port_valid_0_i;
	entry_port_options_10(0) <= entry_waiting_for_port_10(0) and port_valid_0_i;
	entry_port_options_11(0) <= entry_waiting_for_port_11(0) and port_valid_0_i;
	entry_port_options_12(0) <= entry_waiting_for_port_12(0) and port_valid_0_i;
	entry_port_options_13(0) <= entry_waiting_for_port_13(0) and port_valid_0_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_transfer, entry_port_options, queue_head_oh)
	TEMP_18_double_in_0(0) <= entry_port_options_0(0);
	TEMP_18_double_in_0(14) <= entry_port_options_0(0);
	TEMP_18_double_in_0(1) <= entry_port_options_1(0);
	TEMP_18_double_in_0(15) <= entry_port_options_1(0);
	TEMP_18_double_in_0(2) <= entry_port_options_2(0);
	TEMP_18_double_in_0(16) <= entry_port_options_2(0);
	TEMP_18_double_in_0(3) <= entry_port_options_3(0);
	TEMP_18_double_in_0(17) <= entry_port_options_3(0);
	TEMP_18_double_in_0(4) <= entry_port_options_4(0);
	TEMP_18_double_in_0(18) <= entry_port_options_4(0);
	TEMP_18_double_in_0(5) <= entry_port_options_5(0);
	TEMP_18_double_in_0(19) <= entry_port_options_5(0);
	TEMP_18_double_in_0(6) <= entry_port_options_6(0);
	TEMP_18_double_in_0(20) <= entry_port_options_6(0);
	TEMP_18_double_in_0(7) <= entry_port_options_7(0);
	TEMP_18_double_in_0(21) <= entry_port_options_7(0);
	TEMP_18_double_in_0(8) <= entry_port_options_8(0);
	TEMP_18_double_in_0(22) <= entry_port_options_8(0);
	TEMP_18_double_in_0(9) <= entry_port_options_9(0);
	TEMP_18_double_in_0(23) <= entry_port_options_9(0);
	TEMP_18_double_in_0(10) <= entry_port_options_10(0);
	TEMP_18_double_in_0(24) <= entry_port_options_10(0);
	TEMP_18_double_in_0(11) <= entry_port_options_11(0);
	TEMP_18_double_in_0(25) <= entry_port_options_11(0);
	TEMP_18_double_in_0(12) <= entry_port_options_12(0);
	TEMP_18_double_in_0(26) <= entry_port_options_12(0);
	TEMP_18_double_in_0(13) <= entry_port_options_13(0);
	TEMP_18_double_in_0(27) <= entry_port_options_13(0);
	TEMP_18_double_out_0 <= TEMP_18_double_in_0 and not std_logic_vector( unsigned( TEMP_18_double_in_0 ) - unsigned( "00000000000000" & queue_head_oh_i ) );
	entry_port_transfer_0(0) <= TEMP_18_double_out_0(0) or TEMP_18_double_out_0(14);
	entry_port_transfer_1(0) <= TEMP_18_double_out_0(1) or TEMP_18_double_out_0(15);
	entry_port_transfer_2(0) <= TEMP_18_double_out_0(2) or TEMP_18_double_out_0(16);
	entry_port_transfer_3(0) <= TEMP_18_double_out_0(3) or TEMP_18_double_out_0(17);
	entry_port_transfer_4(0) <= TEMP_18_double_out_0(4) or TEMP_18_double_out_0(18);
	entry_port_transfer_5(0) <= TEMP_18_double_out_0(5) or TEMP_18_double_out_0(19);
	entry_port_transfer_6(0) <= TEMP_18_double_out_0(6) or TEMP_18_double_out_0(20);
	entry_port_transfer_7(0) <= TEMP_18_double_out_0(7) or TEMP_18_double_out_0(21);
	entry_port_transfer_8(0) <= TEMP_18_double_out_0(8) or TEMP_18_double_out_0(22);
	entry_port_transfer_9(0) <= TEMP_18_double_out_0(9) or TEMP_18_double_out_0(23);
	entry_port_transfer_10(0) <= TEMP_18_double_out_0(10) or TEMP_18_double_out_0(24);
	entry_port_transfer_11(0) <= TEMP_18_double_out_0(11) or TEMP_18_double_out_0(25);
	entry_port_transfer_12(0) <= TEMP_18_double_out_0(12) or TEMP_18_double_out_0(26);
	entry_port_transfer_13(0) <= TEMP_18_double_out_0(13) or TEMP_18_double_out_0(27);
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

	-- Reduction Begin
	-- Reduce(entry_wen_5, entry_port_transfer_5, or)
	entry_wen_5_o <= entry_port_transfer_5(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_6, entry_port_transfer_6, or)
	entry_wen_6_o <= entry_port_transfer_6(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_7, entry_port_transfer_7, or)
	entry_wen_7_o <= entry_port_transfer_7(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_8, entry_port_transfer_8, or)
	entry_wen_8_o <= entry_port_transfer_8(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_9, entry_port_transfer_9, or)
	entry_wen_9_o <= entry_port_transfer_9(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_10, entry_port_transfer_10, or)
	entry_wen_10_o <= entry_port_transfer_10(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_11, entry_port_transfer_11, or)
	entry_wen_11_o <= entry_port_transfer_11(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_12, entry_port_transfer_12, or)
	entry_wen_12_o <= entry_port_transfer_12(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_13, entry_port_transfer_13, or)
	entry_wen_13_o <= entry_port_transfer_13(0);
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
		entry_alloc_5_i : in std_logic;
		entry_alloc_6_i : in std_logic;
		entry_alloc_7_i : in std_logic;
		entry_alloc_8_i : in std_logic;
		entry_alloc_9_i : in std_logic;
		entry_alloc_10_i : in std_logic;
		entry_alloc_11_i : in std_logic;
		entry_alloc_12_i : in std_logic;
		entry_alloc_13_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_valid_2_i : in std_logic;
		entry_payload_valid_3_i : in std_logic;
		entry_payload_valid_4_i : in std_logic;
		entry_payload_valid_5_i : in std_logic;
		entry_payload_valid_6_i : in std_logic;
		entry_payload_valid_7_i : in std_logic;
		entry_payload_valid_8_i : in std_logic;
		entry_payload_valid_9_i : in std_logic;
		entry_payload_valid_10_i : in std_logic;
		entry_payload_valid_11_i : in std_logic;
		entry_payload_valid_12_i : in std_logic;
		entry_payload_valid_13_i : in std_logic;
		entry_payload_0_o : out std_logic_vector(31 downto 0);
		entry_payload_1_o : out std_logic_vector(31 downto 0);
		entry_payload_2_o : out std_logic_vector(31 downto 0);
		entry_payload_3_o : out std_logic_vector(31 downto 0);
		entry_payload_4_o : out std_logic_vector(31 downto 0);
		entry_payload_5_o : out std_logic_vector(31 downto 0);
		entry_payload_6_o : out std_logic_vector(31 downto 0);
		entry_payload_7_o : out std_logic_vector(31 downto 0);
		entry_payload_8_o : out std_logic_vector(31 downto 0);
		entry_payload_9_o : out std_logic_vector(31 downto 0);
		entry_payload_10_o : out std_logic_vector(31 downto 0);
		entry_payload_11_o : out std_logic_vector(31 downto 0);
		entry_payload_12_o : out std_logic_vector(31 downto 0);
		entry_payload_13_o : out std_logic_vector(31 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		entry_wen_2_o : out std_logic;
		entry_wen_3_o : out std_logic;
		entry_wen_4_o : out std_logic;
		entry_wen_5_o : out std_logic;
		entry_wen_6_o : out std_logic;
		entry_wen_7_o : out std_logic;
		entry_wen_8_o : out std_logic;
		entry_wen_9_o : out std_logic;
		entry_wen_10_o : out std_logic;
		entry_wen_11_o : out std_logic;
		entry_wen_12_o : out std_logic;
		entry_wen_13_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(13 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_std is
	signal entry_port_idx_oh_0 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_2 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_3 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_4 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_5 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_6 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_7 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_8 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_9 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_10 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_11 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_12 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_13 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_4_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_5_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_6_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_7_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_8_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_9_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_10_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_11_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_12_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_13_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_14_mux_0 : std_logic_vector(31 downto 0);
	signal entry_ptq_ready_0 : std_logic;
	signal entry_ptq_ready_1 : std_logic;
	signal entry_ptq_ready_2 : std_logic;
	signal entry_ptq_ready_3 : std_logic;
	signal entry_ptq_ready_4 : std_logic;
	signal entry_ptq_ready_5 : std_logic;
	signal entry_ptq_ready_6 : std_logic;
	signal entry_ptq_ready_7 : std_logic;
	signal entry_ptq_ready_8 : std_logic;
	signal entry_ptq_ready_9 : std_logic;
	signal entry_ptq_ready_10 : std_logic;
	signal entry_ptq_ready_11 : std_logic;
	signal entry_ptq_ready_12 : std_logic;
	signal entry_ptq_ready_13 : std_logic;
	signal entry_waiting_for_port_0 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_1 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_2 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_3 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_4 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_5 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_6 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_7 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_8 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_9 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_10 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_11 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_12 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_13 : std_logic_vector(0 downto 0);
	signal port_ready_vec : std_logic_vector(0 downto 0);
	signal TEMP_15_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_4 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_5 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_6 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_7 : std_logic_vector(0 downto 0);
	signal TEMP_16_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_16_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_16_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_16_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_17_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_17_res_1 : std_logic_vector(0 downto 0);
	signal entry_port_options_0 : std_logic_vector(0 downto 0);
	signal entry_port_options_1 : std_logic_vector(0 downto 0);
	signal entry_port_options_2 : std_logic_vector(0 downto 0);
	signal entry_port_options_3 : std_logic_vector(0 downto 0);
	signal entry_port_options_4 : std_logic_vector(0 downto 0);
	signal entry_port_options_5 : std_logic_vector(0 downto 0);
	signal entry_port_options_6 : std_logic_vector(0 downto 0);
	signal entry_port_options_7 : std_logic_vector(0 downto 0);
	signal entry_port_options_8 : std_logic_vector(0 downto 0);
	signal entry_port_options_9 : std_logic_vector(0 downto 0);
	signal entry_port_options_10 : std_logic_vector(0 downto 0);
	signal entry_port_options_11 : std_logic_vector(0 downto 0);
	signal entry_port_options_12 : std_logic_vector(0 downto 0);
	signal entry_port_options_13 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_2 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_3 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_4 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_5 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_6 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_7 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_8 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_9 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_10 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_11 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_12 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_13 : std_logic_vector(0 downto 0);
	signal TEMP_18_double_in_0 : std_logic_vector(27 downto 0);
	signal TEMP_18_double_out_0 : std_logic_vector(27 downto 0);
begin
	entry_port_idx_oh_0 <= "1";
	entry_port_idx_oh_1 <= "1";
	entry_port_idx_oh_2 <= "1";
	entry_port_idx_oh_3 <= "1";
	entry_port_idx_oh_4 <= "1";
	entry_port_idx_oh_5 <= "1";
	entry_port_idx_oh_6 <= "1";
	entry_port_idx_oh_7 <= "1";
	entry_port_idx_oh_8 <= "1";
	entry_port_idx_oh_9 <= "1";
	entry_port_idx_oh_10 <= "1";
	entry_port_idx_oh_11 <= "1";
	entry_port_idx_oh_12 <= "1";
	entry_port_idx_oh_13 <= "1";
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

	-- Mux1H Begin
	-- Mux1H(entry_payload_5, port_payload, entry_port_idx_oh_5)
	TEMP_6_mux_0 <= port_payload_0_i when entry_port_idx_oh_5(0) = '1' else "00000000000000000000000000000000";
	entry_payload_5_o <= TEMP_6_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_6, port_payload, entry_port_idx_oh_6)
	TEMP_7_mux_0 <= port_payload_0_i when entry_port_idx_oh_6(0) = '1' else "00000000000000000000000000000000";
	entry_payload_6_o <= TEMP_7_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_7, port_payload, entry_port_idx_oh_7)
	TEMP_8_mux_0 <= port_payload_0_i when entry_port_idx_oh_7(0) = '1' else "00000000000000000000000000000000";
	entry_payload_7_o <= TEMP_8_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_8, port_payload, entry_port_idx_oh_8)
	TEMP_9_mux_0 <= port_payload_0_i when entry_port_idx_oh_8(0) = '1' else "00000000000000000000000000000000";
	entry_payload_8_o <= TEMP_9_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_9, port_payload, entry_port_idx_oh_9)
	TEMP_10_mux_0 <= port_payload_0_i when entry_port_idx_oh_9(0) = '1' else "00000000000000000000000000000000";
	entry_payload_9_o <= TEMP_10_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_10, port_payload, entry_port_idx_oh_10)
	TEMP_11_mux_0 <= port_payload_0_i when entry_port_idx_oh_10(0) = '1' else "00000000000000000000000000000000";
	entry_payload_10_o <= TEMP_11_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_11, port_payload, entry_port_idx_oh_11)
	TEMP_12_mux_0 <= port_payload_0_i when entry_port_idx_oh_11(0) = '1' else "00000000000000000000000000000000";
	entry_payload_11_o <= TEMP_12_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_12, port_payload, entry_port_idx_oh_12)
	TEMP_13_mux_0 <= port_payload_0_i when entry_port_idx_oh_12(0) = '1' else "00000000000000000000000000000000";
	entry_payload_12_o <= TEMP_13_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_13, port_payload, entry_port_idx_oh_13)
	TEMP_14_mux_0 <= port_payload_0_i when entry_port_idx_oh_13(0) = '1' else "00000000000000000000000000000000";
	entry_payload_13_o <= TEMP_14_mux_0;
	-- Mux1H End

	entry_ptq_ready_0 <= entry_alloc_0_i and not entry_payload_valid_0_i;
	entry_ptq_ready_1 <= entry_alloc_1_i and not entry_payload_valid_1_i;
	entry_ptq_ready_2 <= entry_alloc_2_i and not entry_payload_valid_2_i;
	entry_ptq_ready_3 <= entry_alloc_3_i and not entry_payload_valid_3_i;
	entry_ptq_ready_4 <= entry_alloc_4_i and not entry_payload_valid_4_i;
	entry_ptq_ready_5 <= entry_alloc_5_i and not entry_payload_valid_5_i;
	entry_ptq_ready_6 <= entry_alloc_6_i and not entry_payload_valid_6_i;
	entry_ptq_ready_7 <= entry_alloc_7_i and not entry_payload_valid_7_i;
	entry_ptq_ready_8 <= entry_alloc_8_i and not entry_payload_valid_8_i;
	entry_ptq_ready_9 <= entry_alloc_9_i and not entry_payload_valid_9_i;
	entry_ptq_ready_10 <= entry_alloc_10_i and not entry_payload_valid_10_i;
	entry_ptq_ready_11 <= entry_alloc_11_i and not entry_payload_valid_11_i;
	entry_ptq_ready_12 <= entry_alloc_12_i and not entry_payload_valid_12_i;
	entry_ptq_ready_13 <= entry_alloc_13_i and not entry_payload_valid_13_i;
	entry_waiting_for_port_0 <= entry_port_idx_oh_0 when entry_ptq_ready_0 else "0";
	entry_waiting_for_port_1 <= entry_port_idx_oh_1 when entry_ptq_ready_1 else "0";
	entry_waiting_for_port_2 <= entry_port_idx_oh_2 when entry_ptq_ready_2 else "0";
	entry_waiting_for_port_3 <= entry_port_idx_oh_3 when entry_ptq_ready_3 else "0";
	entry_waiting_for_port_4 <= entry_port_idx_oh_4 when entry_ptq_ready_4 else "0";
	entry_waiting_for_port_5 <= entry_port_idx_oh_5 when entry_ptq_ready_5 else "0";
	entry_waiting_for_port_6 <= entry_port_idx_oh_6 when entry_ptq_ready_6 else "0";
	entry_waiting_for_port_7 <= entry_port_idx_oh_7 when entry_ptq_ready_7 else "0";
	entry_waiting_for_port_8 <= entry_port_idx_oh_8 when entry_ptq_ready_8 else "0";
	entry_waiting_for_port_9 <= entry_port_idx_oh_9 when entry_ptq_ready_9 else "0";
	entry_waiting_for_port_10 <= entry_port_idx_oh_10 when entry_ptq_ready_10 else "0";
	entry_waiting_for_port_11 <= entry_port_idx_oh_11 when entry_ptq_ready_11 else "0";
	entry_waiting_for_port_12 <= entry_port_idx_oh_12 when entry_ptq_ready_12 else "0";
	entry_waiting_for_port_13 <= entry_port_idx_oh_13 when entry_ptq_ready_13 else "0";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_waiting_for_port, or)
	TEMP_15_res_0 <= entry_waiting_for_port_0 or entry_waiting_for_port_8;
	TEMP_15_res_1 <= entry_waiting_for_port_1 or entry_waiting_for_port_9;
	TEMP_15_res_2 <= entry_waiting_for_port_2 or entry_waiting_for_port_10;
	TEMP_15_res_3 <= entry_waiting_for_port_3 or entry_waiting_for_port_11;
	TEMP_15_res_4 <= entry_waiting_for_port_4 or entry_waiting_for_port_12;
	TEMP_15_res_5 <= entry_waiting_for_port_5 or entry_waiting_for_port_13;
	TEMP_15_res_6 <= entry_waiting_for_port_6;
	TEMP_15_res_7 <= entry_waiting_for_port_7;
	-- Layer End
	TEMP_16_res_0 <= TEMP_15_res_0 or TEMP_15_res_4;
	TEMP_16_res_1 <= TEMP_15_res_1 or TEMP_15_res_5;
	TEMP_16_res_2 <= TEMP_15_res_2 or TEMP_15_res_6;
	TEMP_16_res_3 <= TEMP_15_res_3 or TEMP_15_res_7;
	-- Layer End
	TEMP_17_res_0 <= TEMP_16_res_0 or TEMP_16_res_2;
	TEMP_17_res_1 <= TEMP_16_res_1 or TEMP_16_res_3;
	-- Layer End
	port_ready_vec <= TEMP_17_res_0 or TEMP_17_res_1;
	-- Reduction End

	port_ready_0_o <= port_ready_vec(0);
	entry_port_options_0(0) <= entry_waiting_for_port_0(0) and port_valid_0_i;
	entry_port_options_1(0) <= entry_waiting_for_port_1(0) and port_valid_0_i;
	entry_port_options_2(0) <= entry_waiting_for_port_2(0) and port_valid_0_i;
	entry_port_options_3(0) <= entry_waiting_for_port_3(0) and port_valid_0_i;
	entry_port_options_4(0) <= entry_waiting_for_port_4(0) and port_valid_0_i;
	entry_port_options_5(0) <= entry_waiting_for_port_5(0) and port_valid_0_i;
	entry_port_options_6(0) <= entry_waiting_for_port_6(0) and port_valid_0_i;
	entry_port_options_7(0) <= entry_waiting_for_port_7(0) and port_valid_0_i;
	entry_port_options_8(0) <= entry_waiting_for_port_8(0) and port_valid_0_i;
	entry_port_options_9(0) <= entry_waiting_for_port_9(0) and port_valid_0_i;
	entry_port_options_10(0) <= entry_waiting_for_port_10(0) and port_valid_0_i;
	entry_port_options_11(0) <= entry_waiting_for_port_11(0) and port_valid_0_i;
	entry_port_options_12(0) <= entry_waiting_for_port_12(0) and port_valid_0_i;
	entry_port_options_13(0) <= entry_waiting_for_port_13(0) and port_valid_0_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_transfer, entry_port_options, queue_head_oh)
	TEMP_18_double_in_0(0) <= entry_port_options_0(0);
	TEMP_18_double_in_0(14) <= entry_port_options_0(0);
	TEMP_18_double_in_0(1) <= entry_port_options_1(0);
	TEMP_18_double_in_0(15) <= entry_port_options_1(0);
	TEMP_18_double_in_0(2) <= entry_port_options_2(0);
	TEMP_18_double_in_0(16) <= entry_port_options_2(0);
	TEMP_18_double_in_0(3) <= entry_port_options_3(0);
	TEMP_18_double_in_0(17) <= entry_port_options_3(0);
	TEMP_18_double_in_0(4) <= entry_port_options_4(0);
	TEMP_18_double_in_0(18) <= entry_port_options_4(0);
	TEMP_18_double_in_0(5) <= entry_port_options_5(0);
	TEMP_18_double_in_0(19) <= entry_port_options_5(0);
	TEMP_18_double_in_0(6) <= entry_port_options_6(0);
	TEMP_18_double_in_0(20) <= entry_port_options_6(0);
	TEMP_18_double_in_0(7) <= entry_port_options_7(0);
	TEMP_18_double_in_0(21) <= entry_port_options_7(0);
	TEMP_18_double_in_0(8) <= entry_port_options_8(0);
	TEMP_18_double_in_0(22) <= entry_port_options_8(0);
	TEMP_18_double_in_0(9) <= entry_port_options_9(0);
	TEMP_18_double_in_0(23) <= entry_port_options_9(0);
	TEMP_18_double_in_0(10) <= entry_port_options_10(0);
	TEMP_18_double_in_0(24) <= entry_port_options_10(0);
	TEMP_18_double_in_0(11) <= entry_port_options_11(0);
	TEMP_18_double_in_0(25) <= entry_port_options_11(0);
	TEMP_18_double_in_0(12) <= entry_port_options_12(0);
	TEMP_18_double_in_0(26) <= entry_port_options_12(0);
	TEMP_18_double_in_0(13) <= entry_port_options_13(0);
	TEMP_18_double_in_0(27) <= entry_port_options_13(0);
	TEMP_18_double_out_0 <= TEMP_18_double_in_0 and not std_logic_vector( unsigned( TEMP_18_double_in_0 ) - unsigned( "00000000000000" & queue_head_oh_i ) );
	entry_port_transfer_0(0) <= TEMP_18_double_out_0(0) or TEMP_18_double_out_0(14);
	entry_port_transfer_1(0) <= TEMP_18_double_out_0(1) or TEMP_18_double_out_0(15);
	entry_port_transfer_2(0) <= TEMP_18_double_out_0(2) or TEMP_18_double_out_0(16);
	entry_port_transfer_3(0) <= TEMP_18_double_out_0(3) or TEMP_18_double_out_0(17);
	entry_port_transfer_4(0) <= TEMP_18_double_out_0(4) or TEMP_18_double_out_0(18);
	entry_port_transfer_5(0) <= TEMP_18_double_out_0(5) or TEMP_18_double_out_0(19);
	entry_port_transfer_6(0) <= TEMP_18_double_out_0(6) or TEMP_18_double_out_0(20);
	entry_port_transfer_7(0) <= TEMP_18_double_out_0(7) or TEMP_18_double_out_0(21);
	entry_port_transfer_8(0) <= TEMP_18_double_out_0(8) or TEMP_18_double_out_0(22);
	entry_port_transfer_9(0) <= TEMP_18_double_out_0(9) or TEMP_18_double_out_0(23);
	entry_port_transfer_10(0) <= TEMP_18_double_out_0(10) or TEMP_18_double_out_0(24);
	entry_port_transfer_11(0) <= TEMP_18_double_out_0(11) or TEMP_18_double_out_0(25);
	entry_port_transfer_12(0) <= TEMP_18_double_out_0(12) or TEMP_18_double_out_0(26);
	entry_port_transfer_13(0) <= TEMP_18_double_out_0(13) or TEMP_18_double_out_0(27);
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

	-- Reduction Begin
	-- Reduce(entry_wen_5, entry_port_transfer_5, or)
	entry_wen_5_o <= entry_port_transfer_5(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_6, entry_port_transfer_6, or)
	entry_wen_6_o <= entry_port_transfer_6(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_7, entry_port_transfer_7, or)
	entry_wen_7_o <= entry_port_transfer_7(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_8, entry_port_transfer_8, or)
	entry_wen_8_o <= entry_port_transfer_8(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_9, entry_port_transfer_9, or)
	entry_wen_9_o <= entry_port_transfer_9(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_10, entry_port_transfer_10, or)
	entry_wen_10_o <= entry_port_transfer_10(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_11, entry_port_transfer_11, or)
	entry_wen_11_o <= entry_port_transfer_11(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_12, entry_port_transfer_12, or)
	entry_wen_12_o <= entry_port_transfer_12(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_13, entry_port_transfer_13, or)
	entry_wen_13_o <= entry_port_transfer_13(0);
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
	signal ldq_alloc_2_d : std_logic;
	signal ldq_alloc_2_q : std_logic;
	signal ldq_alloc_3_d : std_logic;
	signal ldq_alloc_3_q : std_logic;
	signal ldq_alloc_4_d : std_logic;
	signal ldq_alloc_4_q : std_logic;
	signal ldq_alloc_5_d : std_logic;
	signal ldq_alloc_5_q : std_logic;
	signal ldq_alloc_6_d : std_logic;
	signal ldq_alloc_6_q : std_logic;
	signal ldq_alloc_7_d : std_logic;
	signal ldq_alloc_7_q : std_logic;
	signal ldq_alloc_8_d : std_logic;
	signal ldq_alloc_8_q : std_logic;
	signal ldq_alloc_9_d : std_logic;
	signal ldq_alloc_9_q : std_logic;
	signal ldq_alloc_10_d : std_logic;
	signal ldq_alloc_10_q : std_logic;
	signal ldq_alloc_11_d : std_logic;
	signal ldq_alloc_11_q : std_logic;
	signal ldq_alloc_12_d : std_logic;
	signal ldq_alloc_12_q : std_logic;
	signal ldq_alloc_13_d : std_logic;
	signal ldq_alloc_13_q : std_logic;
	signal ldq_issue_0_d : std_logic;
	signal ldq_issue_0_q : std_logic;
	signal ldq_issue_1_d : std_logic;
	signal ldq_issue_1_q : std_logic;
	signal ldq_issue_2_d : std_logic;
	signal ldq_issue_2_q : std_logic;
	signal ldq_issue_3_d : std_logic;
	signal ldq_issue_3_q : std_logic;
	signal ldq_issue_4_d : std_logic;
	signal ldq_issue_4_q : std_logic;
	signal ldq_issue_5_d : std_logic;
	signal ldq_issue_5_q : std_logic;
	signal ldq_issue_6_d : std_logic;
	signal ldq_issue_6_q : std_logic;
	signal ldq_issue_7_d : std_logic;
	signal ldq_issue_7_q : std_logic;
	signal ldq_issue_8_d : std_logic;
	signal ldq_issue_8_q : std_logic;
	signal ldq_issue_9_d : std_logic;
	signal ldq_issue_9_q : std_logic;
	signal ldq_issue_10_d : std_logic;
	signal ldq_issue_10_q : std_logic;
	signal ldq_issue_11_d : std_logic;
	signal ldq_issue_11_q : std_logic;
	signal ldq_issue_12_d : std_logic;
	signal ldq_issue_12_q : std_logic;
	signal ldq_issue_13_d : std_logic;
	signal ldq_issue_13_q : std_logic;
	signal ldq_addr_valid_0_d : std_logic;
	signal ldq_addr_valid_0_q : std_logic;
	signal ldq_addr_valid_1_d : std_logic;
	signal ldq_addr_valid_1_q : std_logic;
	signal ldq_addr_valid_2_d : std_logic;
	signal ldq_addr_valid_2_q : std_logic;
	signal ldq_addr_valid_3_d : std_logic;
	signal ldq_addr_valid_3_q : std_logic;
	signal ldq_addr_valid_4_d : std_logic;
	signal ldq_addr_valid_4_q : std_logic;
	signal ldq_addr_valid_5_d : std_logic;
	signal ldq_addr_valid_5_q : std_logic;
	signal ldq_addr_valid_6_d : std_logic;
	signal ldq_addr_valid_6_q : std_logic;
	signal ldq_addr_valid_7_d : std_logic;
	signal ldq_addr_valid_7_q : std_logic;
	signal ldq_addr_valid_8_d : std_logic;
	signal ldq_addr_valid_8_q : std_logic;
	signal ldq_addr_valid_9_d : std_logic;
	signal ldq_addr_valid_9_q : std_logic;
	signal ldq_addr_valid_10_d : std_logic;
	signal ldq_addr_valid_10_q : std_logic;
	signal ldq_addr_valid_11_d : std_logic;
	signal ldq_addr_valid_11_q : std_logic;
	signal ldq_addr_valid_12_d : std_logic;
	signal ldq_addr_valid_12_q : std_logic;
	signal ldq_addr_valid_13_d : std_logic;
	signal ldq_addr_valid_13_q : std_logic;
	signal ldq_addr_0_d : std_logic_vector(9 downto 0);
	signal ldq_addr_0_q : std_logic_vector(9 downto 0);
	signal ldq_addr_1_d : std_logic_vector(9 downto 0);
	signal ldq_addr_1_q : std_logic_vector(9 downto 0);
	signal ldq_addr_2_d : std_logic_vector(9 downto 0);
	signal ldq_addr_2_q : std_logic_vector(9 downto 0);
	signal ldq_addr_3_d : std_logic_vector(9 downto 0);
	signal ldq_addr_3_q : std_logic_vector(9 downto 0);
	signal ldq_addr_4_d : std_logic_vector(9 downto 0);
	signal ldq_addr_4_q : std_logic_vector(9 downto 0);
	signal ldq_addr_5_d : std_logic_vector(9 downto 0);
	signal ldq_addr_5_q : std_logic_vector(9 downto 0);
	signal ldq_addr_6_d : std_logic_vector(9 downto 0);
	signal ldq_addr_6_q : std_logic_vector(9 downto 0);
	signal ldq_addr_7_d : std_logic_vector(9 downto 0);
	signal ldq_addr_7_q : std_logic_vector(9 downto 0);
	signal ldq_addr_8_d : std_logic_vector(9 downto 0);
	signal ldq_addr_8_q : std_logic_vector(9 downto 0);
	signal ldq_addr_9_d : std_logic_vector(9 downto 0);
	signal ldq_addr_9_q : std_logic_vector(9 downto 0);
	signal ldq_addr_10_d : std_logic_vector(9 downto 0);
	signal ldq_addr_10_q : std_logic_vector(9 downto 0);
	signal ldq_addr_11_d : std_logic_vector(9 downto 0);
	signal ldq_addr_11_q : std_logic_vector(9 downto 0);
	signal ldq_addr_12_d : std_logic_vector(9 downto 0);
	signal ldq_addr_12_q : std_logic_vector(9 downto 0);
	signal ldq_addr_13_d : std_logic_vector(9 downto 0);
	signal ldq_addr_13_q : std_logic_vector(9 downto 0);
	signal ldq_data_valid_0_d : std_logic;
	signal ldq_data_valid_0_q : std_logic;
	signal ldq_data_valid_1_d : std_logic;
	signal ldq_data_valid_1_q : std_logic;
	signal ldq_data_valid_2_d : std_logic;
	signal ldq_data_valid_2_q : std_logic;
	signal ldq_data_valid_3_d : std_logic;
	signal ldq_data_valid_3_q : std_logic;
	signal ldq_data_valid_4_d : std_logic;
	signal ldq_data_valid_4_q : std_logic;
	signal ldq_data_valid_5_d : std_logic;
	signal ldq_data_valid_5_q : std_logic;
	signal ldq_data_valid_6_d : std_logic;
	signal ldq_data_valid_6_q : std_logic;
	signal ldq_data_valid_7_d : std_logic;
	signal ldq_data_valid_7_q : std_logic;
	signal ldq_data_valid_8_d : std_logic;
	signal ldq_data_valid_8_q : std_logic;
	signal ldq_data_valid_9_d : std_logic;
	signal ldq_data_valid_9_q : std_logic;
	signal ldq_data_valid_10_d : std_logic;
	signal ldq_data_valid_10_q : std_logic;
	signal ldq_data_valid_11_d : std_logic;
	signal ldq_data_valid_11_q : std_logic;
	signal ldq_data_valid_12_d : std_logic;
	signal ldq_data_valid_12_q : std_logic;
	signal ldq_data_valid_13_d : std_logic;
	signal ldq_data_valid_13_q : std_logic;
	signal ldq_data_0_d : std_logic_vector(31 downto 0);
	signal ldq_data_0_q : std_logic_vector(31 downto 0);
	signal ldq_data_1_d : std_logic_vector(31 downto 0);
	signal ldq_data_1_q : std_logic_vector(31 downto 0);
	signal ldq_data_2_d : std_logic_vector(31 downto 0);
	signal ldq_data_2_q : std_logic_vector(31 downto 0);
	signal ldq_data_3_d : std_logic_vector(31 downto 0);
	signal ldq_data_3_q : std_logic_vector(31 downto 0);
	signal ldq_data_4_d : std_logic_vector(31 downto 0);
	signal ldq_data_4_q : std_logic_vector(31 downto 0);
	signal ldq_data_5_d : std_logic_vector(31 downto 0);
	signal ldq_data_5_q : std_logic_vector(31 downto 0);
	signal ldq_data_6_d : std_logic_vector(31 downto 0);
	signal ldq_data_6_q : std_logic_vector(31 downto 0);
	signal ldq_data_7_d : std_logic_vector(31 downto 0);
	signal ldq_data_7_q : std_logic_vector(31 downto 0);
	signal ldq_data_8_d : std_logic_vector(31 downto 0);
	signal ldq_data_8_q : std_logic_vector(31 downto 0);
	signal ldq_data_9_d : std_logic_vector(31 downto 0);
	signal ldq_data_9_q : std_logic_vector(31 downto 0);
	signal ldq_data_10_d : std_logic_vector(31 downto 0);
	signal ldq_data_10_q : std_logic_vector(31 downto 0);
	signal ldq_data_11_d : std_logic_vector(31 downto 0);
	signal ldq_data_11_q : std_logic_vector(31 downto 0);
	signal ldq_data_12_d : std_logic_vector(31 downto 0);
	signal ldq_data_12_q : std_logic_vector(31 downto 0);
	signal ldq_data_13_d : std_logic_vector(31 downto 0);
	signal ldq_data_13_q : std_logic_vector(31 downto 0);
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
	signal stq_alloc_5_d : std_logic;
	signal stq_alloc_5_q : std_logic;
	signal stq_alloc_6_d : std_logic;
	signal stq_alloc_6_q : std_logic;
	signal stq_alloc_7_d : std_logic;
	signal stq_alloc_7_q : std_logic;
	signal stq_alloc_8_d : std_logic;
	signal stq_alloc_8_q : std_logic;
	signal stq_alloc_9_d : std_logic;
	signal stq_alloc_9_q : std_logic;
	signal stq_alloc_10_d : std_logic;
	signal stq_alloc_10_q : std_logic;
	signal stq_alloc_11_d : std_logic;
	signal stq_alloc_11_q : std_logic;
	signal stq_alloc_12_d : std_logic;
	signal stq_alloc_12_q : std_logic;
	signal stq_alloc_13_d : std_logic;
	signal stq_alloc_13_q : std_logic;
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
	signal stq_addr_valid_5_d : std_logic;
	signal stq_addr_valid_5_q : std_logic;
	signal stq_addr_valid_6_d : std_logic;
	signal stq_addr_valid_6_q : std_logic;
	signal stq_addr_valid_7_d : std_logic;
	signal stq_addr_valid_7_q : std_logic;
	signal stq_addr_valid_8_d : std_logic;
	signal stq_addr_valid_8_q : std_logic;
	signal stq_addr_valid_9_d : std_logic;
	signal stq_addr_valid_9_q : std_logic;
	signal stq_addr_valid_10_d : std_logic;
	signal stq_addr_valid_10_q : std_logic;
	signal stq_addr_valid_11_d : std_logic;
	signal stq_addr_valid_11_q : std_logic;
	signal stq_addr_valid_12_d : std_logic;
	signal stq_addr_valid_12_q : std_logic;
	signal stq_addr_valid_13_d : std_logic;
	signal stq_addr_valid_13_q : std_logic;
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
	signal stq_addr_5_d : std_logic_vector(9 downto 0);
	signal stq_addr_5_q : std_logic_vector(9 downto 0);
	signal stq_addr_6_d : std_logic_vector(9 downto 0);
	signal stq_addr_6_q : std_logic_vector(9 downto 0);
	signal stq_addr_7_d : std_logic_vector(9 downto 0);
	signal stq_addr_7_q : std_logic_vector(9 downto 0);
	signal stq_addr_8_d : std_logic_vector(9 downto 0);
	signal stq_addr_8_q : std_logic_vector(9 downto 0);
	signal stq_addr_9_d : std_logic_vector(9 downto 0);
	signal stq_addr_9_q : std_logic_vector(9 downto 0);
	signal stq_addr_10_d : std_logic_vector(9 downto 0);
	signal stq_addr_10_q : std_logic_vector(9 downto 0);
	signal stq_addr_11_d : std_logic_vector(9 downto 0);
	signal stq_addr_11_q : std_logic_vector(9 downto 0);
	signal stq_addr_12_d : std_logic_vector(9 downto 0);
	signal stq_addr_12_q : std_logic_vector(9 downto 0);
	signal stq_addr_13_d : std_logic_vector(9 downto 0);
	signal stq_addr_13_q : std_logic_vector(9 downto 0);
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
	signal stq_data_valid_5_d : std_logic;
	signal stq_data_valid_5_q : std_logic;
	signal stq_data_valid_6_d : std_logic;
	signal stq_data_valid_6_q : std_logic;
	signal stq_data_valid_7_d : std_logic;
	signal stq_data_valid_7_q : std_logic;
	signal stq_data_valid_8_d : std_logic;
	signal stq_data_valid_8_q : std_logic;
	signal stq_data_valid_9_d : std_logic;
	signal stq_data_valid_9_q : std_logic;
	signal stq_data_valid_10_d : std_logic;
	signal stq_data_valid_10_q : std_logic;
	signal stq_data_valid_11_d : std_logic;
	signal stq_data_valid_11_q : std_logic;
	signal stq_data_valid_12_d : std_logic;
	signal stq_data_valid_12_q : std_logic;
	signal stq_data_valid_13_d : std_logic;
	signal stq_data_valid_13_q : std_logic;
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
	signal stq_data_5_d : std_logic_vector(31 downto 0);
	signal stq_data_5_q : std_logic_vector(31 downto 0);
	signal stq_data_6_d : std_logic_vector(31 downto 0);
	signal stq_data_6_q : std_logic_vector(31 downto 0);
	signal stq_data_7_d : std_logic_vector(31 downto 0);
	signal stq_data_7_q : std_logic_vector(31 downto 0);
	signal stq_data_8_d : std_logic_vector(31 downto 0);
	signal stq_data_8_q : std_logic_vector(31 downto 0);
	signal stq_data_9_d : std_logic_vector(31 downto 0);
	signal stq_data_9_q : std_logic_vector(31 downto 0);
	signal stq_data_10_d : std_logic_vector(31 downto 0);
	signal stq_data_10_q : std_logic_vector(31 downto 0);
	signal stq_data_11_d : std_logic_vector(31 downto 0);
	signal stq_data_11_q : std_logic_vector(31 downto 0);
	signal stq_data_12_d : std_logic_vector(31 downto 0);
	signal stq_data_12_q : std_logic_vector(31 downto 0);
	signal stq_data_13_d : std_logic_vector(31 downto 0);
	signal stq_data_13_q : std_logic_vector(31 downto 0);
	signal store_is_older_0_d : std_logic_vector(13 downto 0);
	signal store_is_older_0_q : std_logic_vector(13 downto 0);
	signal store_is_older_1_d : std_logic_vector(13 downto 0);
	signal store_is_older_1_q : std_logic_vector(13 downto 0);
	signal store_is_older_2_d : std_logic_vector(13 downto 0);
	signal store_is_older_2_q : std_logic_vector(13 downto 0);
	signal store_is_older_3_d : std_logic_vector(13 downto 0);
	signal store_is_older_3_q : std_logic_vector(13 downto 0);
	signal store_is_older_4_d : std_logic_vector(13 downto 0);
	signal store_is_older_4_q : std_logic_vector(13 downto 0);
	signal store_is_older_5_d : std_logic_vector(13 downto 0);
	signal store_is_older_5_q : std_logic_vector(13 downto 0);
	signal store_is_older_6_d : std_logic_vector(13 downto 0);
	signal store_is_older_6_q : std_logic_vector(13 downto 0);
	signal store_is_older_7_d : std_logic_vector(13 downto 0);
	signal store_is_older_7_q : std_logic_vector(13 downto 0);
	signal store_is_older_8_d : std_logic_vector(13 downto 0);
	signal store_is_older_8_q : std_logic_vector(13 downto 0);
	signal store_is_older_9_d : std_logic_vector(13 downto 0);
	signal store_is_older_9_q : std_logic_vector(13 downto 0);
	signal store_is_older_10_d : std_logic_vector(13 downto 0);
	signal store_is_older_10_q : std_logic_vector(13 downto 0);
	signal store_is_older_11_d : std_logic_vector(13 downto 0);
	signal store_is_older_11_q : std_logic_vector(13 downto 0);
	signal store_is_older_12_d : std_logic_vector(13 downto 0);
	signal store_is_older_12_q : std_logic_vector(13 downto 0);
	signal store_is_older_13_d : std_logic_vector(13 downto 0);
	signal store_is_older_13_q : std_logic_vector(13 downto 0);
	signal ldq_tail_d : std_logic_vector(3 downto 0);
	signal ldq_tail_q : std_logic_vector(3 downto 0);
	signal ldq_head_d : std_logic_vector(3 downto 0);
	signal ldq_head_q : std_logic_vector(3 downto 0);
	signal stq_tail_d : std_logic_vector(3 downto 0);
	signal stq_tail_q : std_logic_vector(3 downto 0);
	signal stq_head_d : std_logic_vector(3 downto 0);
	signal stq_head_q : std_logic_vector(3 downto 0);
	signal stq_issue_d : std_logic_vector(3 downto 0);
	signal stq_issue_q : std_logic_vector(3 downto 0);
	signal stq_resp_d : std_logic_vector(3 downto 0);
	signal stq_resp_q : std_logic_vector(3 downto 0);
	signal ldq_wen_0 : std_logic;
	signal ldq_wen_1 : std_logic;
	signal ldq_wen_2 : std_logic;
	signal ldq_wen_3 : std_logic;
	signal ldq_wen_4 : std_logic;
	signal ldq_wen_5 : std_logic;
	signal ldq_wen_6 : std_logic;
	signal ldq_wen_7 : std_logic;
	signal ldq_wen_8 : std_logic;
	signal ldq_wen_9 : std_logic;
	signal ldq_wen_10 : std_logic;
	signal ldq_wen_11 : std_logic;
	signal ldq_wen_12 : std_logic;
	signal ldq_wen_13 : std_logic;
	signal ldq_addr_wen_0 : std_logic;
	signal ldq_addr_wen_1 : std_logic;
	signal ldq_addr_wen_2 : std_logic;
	signal ldq_addr_wen_3 : std_logic;
	signal ldq_addr_wen_4 : std_logic;
	signal ldq_addr_wen_5 : std_logic;
	signal ldq_addr_wen_6 : std_logic;
	signal ldq_addr_wen_7 : std_logic;
	signal ldq_addr_wen_8 : std_logic;
	signal ldq_addr_wen_9 : std_logic;
	signal ldq_addr_wen_10 : std_logic;
	signal ldq_addr_wen_11 : std_logic;
	signal ldq_addr_wen_12 : std_logic;
	signal ldq_addr_wen_13 : std_logic;
	signal ldq_reset_0 : std_logic;
	signal ldq_reset_1 : std_logic;
	signal ldq_reset_2 : std_logic;
	signal ldq_reset_3 : std_logic;
	signal ldq_reset_4 : std_logic;
	signal ldq_reset_5 : std_logic;
	signal ldq_reset_6 : std_logic;
	signal ldq_reset_7 : std_logic;
	signal ldq_reset_8 : std_logic;
	signal ldq_reset_9 : std_logic;
	signal ldq_reset_10 : std_logic;
	signal ldq_reset_11 : std_logic;
	signal ldq_reset_12 : std_logic;
	signal ldq_reset_13 : std_logic;
	signal stq_wen_0 : std_logic;
	signal stq_wen_1 : std_logic;
	signal stq_wen_2 : std_logic;
	signal stq_wen_3 : std_logic;
	signal stq_wen_4 : std_logic;
	signal stq_wen_5 : std_logic;
	signal stq_wen_6 : std_logic;
	signal stq_wen_7 : std_logic;
	signal stq_wen_8 : std_logic;
	signal stq_wen_9 : std_logic;
	signal stq_wen_10 : std_logic;
	signal stq_wen_11 : std_logic;
	signal stq_wen_12 : std_logic;
	signal stq_wen_13 : std_logic;
	signal stq_addr_wen_0 : std_logic;
	signal stq_addr_wen_1 : std_logic;
	signal stq_addr_wen_2 : std_logic;
	signal stq_addr_wen_3 : std_logic;
	signal stq_addr_wen_4 : std_logic;
	signal stq_addr_wen_5 : std_logic;
	signal stq_addr_wen_6 : std_logic;
	signal stq_addr_wen_7 : std_logic;
	signal stq_addr_wen_8 : std_logic;
	signal stq_addr_wen_9 : std_logic;
	signal stq_addr_wen_10 : std_logic;
	signal stq_addr_wen_11 : std_logic;
	signal stq_addr_wen_12 : std_logic;
	signal stq_addr_wen_13 : std_logic;
	signal stq_data_wen_0 : std_logic;
	signal stq_data_wen_1 : std_logic;
	signal stq_data_wen_2 : std_logic;
	signal stq_data_wen_3 : std_logic;
	signal stq_data_wen_4 : std_logic;
	signal stq_data_wen_5 : std_logic;
	signal stq_data_wen_6 : std_logic;
	signal stq_data_wen_7 : std_logic;
	signal stq_data_wen_8 : std_logic;
	signal stq_data_wen_9 : std_logic;
	signal stq_data_wen_10 : std_logic;
	signal stq_data_wen_11 : std_logic;
	signal stq_data_wen_12 : std_logic;
	signal stq_data_wen_13 : std_logic;
	signal stq_reset_0 : std_logic;
	signal stq_reset_1 : std_logic;
	signal stq_reset_2 : std_logic;
	signal stq_reset_3 : std_logic;
	signal stq_reset_4 : std_logic;
	signal stq_reset_5 : std_logic;
	signal stq_reset_6 : std_logic;
	signal stq_reset_7 : std_logic;
	signal stq_reset_8 : std_logic;
	signal stq_reset_9 : std_logic;
	signal stq_reset_10 : std_logic;
	signal stq_reset_11 : std_logic;
	signal stq_reset_12 : std_logic;
	signal stq_reset_13 : std_logic;
	signal ldq_data_wen_0 : std_logic;
	signal ldq_data_wen_1 : std_logic;
	signal ldq_data_wen_2 : std_logic;
	signal ldq_data_wen_3 : std_logic;
	signal ldq_data_wen_4 : std_logic;
	signal ldq_data_wen_5 : std_logic;
	signal ldq_data_wen_6 : std_logic;
	signal ldq_data_wen_7 : std_logic;
	signal ldq_data_wen_8 : std_logic;
	signal ldq_data_wen_9 : std_logic;
	signal ldq_data_wen_10 : std_logic;
	signal ldq_data_wen_11 : std_logic;
	signal ldq_data_wen_12 : std_logic;
	signal ldq_data_wen_13 : std_logic;
	signal ldq_issue_set_0 : std_logic;
	signal ldq_issue_set_1 : std_logic;
	signal ldq_issue_set_2 : std_logic;
	signal ldq_issue_set_3 : std_logic;
	signal ldq_issue_set_4 : std_logic;
	signal ldq_issue_set_5 : std_logic;
	signal ldq_issue_set_6 : std_logic;
	signal ldq_issue_set_7 : std_logic;
	signal ldq_issue_set_8 : std_logic;
	signal ldq_issue_set_9 : std_logic;
	signal ldq_issue_set_10 : std_logic;
	signal ldq_issue_set_11 : std_logic;
	signal ldq_issue_set_12 : std_logic;
	signal ldq_issue_set_13 : std_logic;
	signal ga_ls_order_0 : std_logic_vector(13 downto 0);
	signal ga_ls_order_1 : std_logic_vector(13 downto 0);
	signal ga_ls_order_2 : std_logic_vector(13 downto 0);
	signal ga_ls_order_3 : std_logic_vector(13 downto 0);
	signal ga_ls_order_4 : std_logic_vector(13 downto 0);
	signal ga_ls_order_5 : std_logic_vector(13 downto 0);
	signal ga_ls_order_6 : std_logic_vector(13 downto 0);
	signal ga_ls_order_7 : std_logic_vector(13 downto 0);
	signal ga_ls_order_8 : std_logic_vector(13 downto 0);
	signal ga_ls_order_9 : std_logic_vector(13 downto 0);
	signal ga_ls_order_10 : std_logic_vector(13 downto 0);
	signal ga_ls_order_11 : std_logic_vector(13 downto 0);
	signal ga_ls_order_12 : std_logic_vector(13 downto 0);
	signal ga_ls_order_13 : std_logic_vector(13 downto 0);
	signal num_loads : std_logic_vector(3 downto 0);
	signal num_stores : std_logic_vector(3 downto 0);
	signal stq_issue_en : std_logic;
	signal stq_resp_en : std_logic;
	signal ldq_empty : std_logic;
	signal stq_empty : std_logic;
	signal ldq_head_oh : std_logic_vector(13 downto 0);
	signal stq_head_oh : std_logic_vector(13 downto 0);
	signal ldq_alloc_next_0 : std_logic;
	signal ldq_alloc_next_1 : std_logic;
	signal ldq_alloc_next_2 : std_logic;
	signal ldq_alloc_next_3 : std_logic;
	signal ldq_alloc_next_4 : std_logic;
	signal ldq_alloc_next_5 : std_logic;
	signal ldq_alloc_next_6 : std_logic;
	signal ldq_alloc_next_7 : std_logic;
	signal ldq_alloc_next_8 : std_logic;
	signal ldq_alloc_next_9 : std_logic;
	signal ldq_alloc_next_10 : std_logic;
	signal ldq_alloc_next_11 : std_logic;
	signal ldq_alloc_next_12 : std_logic;
	signal ldq_alloc_next_13 : std_logic;
	signal stq_alloc_next_0 : std_logic;
	signal stq_alloc_next_1 : std_logic;
	signal stq_alloc_next_2 : std_logic;
	signal stq_alloc_next_3 : std_logic;
	signal stq_alloc_next_4 : std_logic;
	signal stq_alloc_next_5 : std_logic;
	signal stq_alloc_next_6 : std_logic;
	signal stq_alloc_next_7 : std_logic;
	signal stq_alloc_next_8 : std_logic;
	signal stq_alloc_next_9 : std_logic;
	signal stq_alloc_next_10 : std_logic;
	signal stq_alloc_next_11 : std_logic;
	signal stq_alloc_next_12 : std_logic;
	signal stq_alloc_next_13 : std_logic;
	signal ldq_not_empty : std_logic;
	signal stq_not_empty : std_logic;
	signal TEMP_1_res_0 : std_logic;
	signal TEMP_1_res_1 : std_logic;
	signal TEMP_1_res_2 : std_logic;
	signal TEMP_1_res_3 : std_logic;
	signal TEMP_1_res_4 : std_logic;
	signal TEMP_1_res_5 : std_logic;
	signal TEMP_1_res_6 : std_logic;
	signal TEMP_1_res_7 : std_logic;
	signal TEMP_2_res_0 : std_logic;
	signal TEMP_2_res_1 : std_logic;
	signal TEMP_2_res_2 : std_logic;
	signal TEMP_2_res_3 : std_logic;
	signal TEMP_3_res_0 : std_logic;
	signal TEMP_3_res_1 : std_logic;
	signal TEMP_4_sum : std_logic_vector(4 downto 0);
	signal TEMP_4_res : std_logic_vector(4 downto 0);
	signal TEMP_5_sum : std_logic_vector(4 downto 0);
	signal TEMP_5_res : std_logic_vector(4 downto 0);
	signal ldq_tail_oh : std_logic_vector(13 downto 0);
	signal ldq_head_next_oh : std_logic_vector(13 downto 0);
	signal ldq_head_next : std_logic_vector(3 downto 0);
	signal ldq_head_sel : std_logic;
	signal TEMP_6_double_in : std_logic_vector(27 downto 0);
	signal TEMP_6_double_out : std_logic_vector(27 downto 0);
	signal TEMP_7_res_0 : std_logic;
	signal TEMP_7_res_1 : std_logic;
	signal TEMP_7_res_2 : std_logic;
	signal TEMP_7_res_3 : std_logic;
	signal TEMP_7_res_4 : std_logic;
	signal TEMP_7_res_5 : std_logic;
	signal TEMP_7_res_6 : std_logic;
	signal TEMP_7_res_7 : std_logic;
	signal TEMP_8_res_0 : std_logic;
	signal TEMP_8_res_1 : std_logic;
	signal TEMP_8_res_2 : std_logic;
	signal TEMP_8_res_3 : std_logic;
	signal TEMP_9_res_0 : std_logic;
	signal TEMP_9_res_1 : std_logic;
	signal TEMP_10_in_0_0 : std_logic;
	signal TEMP_10_in_0_1 : std_logic;
	signal TEMP_10_in_0_2 : std_logic;
	signal TEMP_10_in_0_3 : std_logic;
	signal TEMP_10_in_0_4 : std_logic;
	signal TEMP_10_in_0_5 : std_logic;
	signal TEMP_10_in_0_6 : std_logic;
	signal TEMP_10_in_0_7 : std_logic;
	signal TEMP_10_in_0_8 : std_logic;
	signal TEMP_10_in_0_9 : std_logic;
	signal TEMP_10_in_0_10 : std_logic;
	signal TEMP_10_in_0_11 : std_logic;
	signal TEMP_10_in_0_12 : std_logic;
	signal TEMP_10_in_0_13 : std_logic;
	signal TEMP_10_out_0 : std_logic;
	signal TEMP_11_res_0 : std_logic;
	signal TEMP_11_res_1 : std_logic;
	signal TEMP_11_res_2 : std_logic;
	signal TEMP_11_res_3 : std_logic;
	signal TEMP_11_res_4 : std_logic;
	signal TEMP_11_res_5 : std_logic;
	signal TEMP_11_res_6 : std_logic;
	signal TEMP_11_res_7 : std_logic;
	signal TEMP_12_res_0 : std_logic;
	signal TEMP_12_res_1 : std_logic;
	signal TEMP_12_res_2 : std_logic;
	signal TEMP_12_res_3 : std_logic;
	signal TEMP_13_res_0 : std_logic;
	signal TEMP_13_res_1 : std_logic;
	signal TEMP_13_in_1_0 : std_logic;
	signal TEMP_13_in_1_1 : std_logic;
	signal TEMP_13_in_1_2 : std_logic;
	signal TEMP_13_in_1_3 : std_logic;
	signal TEMP_13_in_1_4 : std_logic;
	signal TEMP_13_in_1_5 : std_logic;
	signal TEMP_13_in_1_6 : std_logic;
	signal TEMP_13_in_1_7 : std_logic;
	signal TEMP_13_in_1_8 : std_logic;
	signal TEMP_13_in_1_9 : std_logic;
	signal TEMP_13_in_1_10 : std_logic;
	signal TEMP_13_in_1_11 : std_logic;
	signal TEMP_13_in_1_12 : std_logic;
	signal TEMP_13_in_1_13 : std_logic;
	signal TEMP_13_out_1 : std_logic;
	signal TEMP_14_res_0 : std_logic;
	signal TEMP_14_res_1 : std_logic;
	signal TEMP_14_res_2 : std_logic;
	signal TEMP_14_res_3 : std_logic;
	signal TEMP_14_res_4 : std_logic;
	signal TEMP_14_res_5 : std_logic;
	signal TEMP_14_res_6 : std_logic;
	signal TEMP_14_res_7 : std_logic;
	signal TEMP_15_res_0 : std_logic;
	signal TEMP_15_res_1 : std_logic;
	signal TEMP_15_res_2 : std_logic;
	signal TEMP_15_res_3 : std_logic;
	signal TEMP_16_res_0 : std_logic;
	signal TEMP_16_res_1 : std_logic;
	signal TEMP_16_in_2_0 : std_logic;
	signal TEMP_16_in_2_1 : std_logic;
	signal TEMP_16_in_2_2 : std_logic;
	signal TEMP_16_in_2_3 : std_logic;
	signal TEMP_16_in_2_4 : std_logic;
	signal TEMP_16_in_2_5 : std_logic;
	signal TEMP_16_in_2_6 : std_logic;
	signal TEMP_16_in_2_7 : std_logic;
	signal TEMP_16_in_2_8 : std_logic;
	signal TEMP_16_in_2_9 : std_logic;
	signal TEMP_16_in_2_10 : std_logic;
	signal TEMP_16_in_2_11 : std_logic;
	signal TEMP_16_in_2_12 : std_logic;
	signal TEMP_16_in_2_13 : std_logic;
	signal TEMP_16_out_2 : std_logic;
	signal TEMP_17_res_0 : std_logic;
	signal TEMP_17_res_1 : std_logic;
	signal TEMP_17_res_2 : std_logic;
	signal TEMP_17_res_3 : std_logic;
	signal TEMP_17_res_4 : std_logic;
	signal TEMP_17_res_5 : std_logic;
	signal TEMP_17_res_6 : std_logic;
	signal TEMP_17_res_7 : std_logic;
	signal TEMP_18_res_0 : std_logic;
	signal TEMP_18_res_1 : std_logic;
	signal TEMP_18_res_2 : std_logic;
	signal TEMP_18_res_3 : std_logic;
	signal TEMP_19_res_0 : std_logic;
	signal TEMP_19_res_1 : std_logic;
	signal TEMP_19_in_3_0 : std_logic;
	signal TEMP_19_in_3_1 : std_logic;
	signal TEMP_19_in_3_2 : std_logic;
	signal TEMP_19_in_3_3 : std_logic;
	signal TEMP_19_in_3_4 : std_logic;
	signal TEMP_19_in_3_5 : std_logic;
	signal TEMP_19_in_3_6 : std_logic;
	signal TEMP_19_in_3_7 : std_logic;
	signal TEMP_19_in_3_8 : std_logic;
	signal TEMP_19_in_3_9 : std_logic;
	signal TEMP_19_in_3_10 : std_logic;
	signal TEMP_19_in_3_11 : std_logic;
	signal TEMP_19_in_3_12 : std_logic;
	signal TEMP_19_in_3_13 : std_logic;
	signal TEMP_19_out_3 : std_logic;
	signal TEMP_20_res_0 : std_logic;
	signal TEMP_20_res_1 : std_logic;
	signal TEMP_20_res_2 : std_logic;
	signal TEMP_20_res_3 : std_logic;
	signal TEMP_20_res_4 : std_logic;
	signal TEMP_20_res_5 : std_logic;
	signal TEMP_20_res_6 : std_logic;
	signal TEMP_20_res_7 : std_logic;
	signal TEMP_21_res_0 : std_logic;
	signal TEMP_21_res_1 : std_logic;
	signal TEMP_21_res_2 : std_logic;
	signal TEMP_21_res_3 : std_logic;
	signal TEMP_22_res_0 : std_logic;
	signal TEMP_22_res_1 : std_logic;
	signal stq_tail_oh : std_logic_vector(13 downto 0);
	signal stq_head_next_oh : std_logic_vector(13 downto 0);
	signal stq_head_next : std_logic_vector(3 downto 0);
	signal stq_head_sel : std_logic;
	signal load_idx_oh_0 : std_logic_vector(13 downto 0);
	signal load_en_0 : std_logic;
	signal store_idx : std_logic_vector(3 downto 0);
	signal store_en : std_logic;
	signal bypass_idx_oh_0 : std_logic_vector(13 downto 0);
	signal bypass_idx_oh_1 : std_logic_vector(13 downto 0);
	signal bypass_idx_oh_2 : std_logic_vector(13 downto 0);
	signal bypass_idx_oh_3 : std_logic_vector(13 downto 0);
	signal bypass_idx_oh_4 : std_logic_vector(13 downto 0);
	signal bypass_idx_oh_5 : std_logic_vector(13 downto 0);
	signal bypass_idx_oh_6 : std_logic_vector(13 downto 0);
	signal bypass_idx_oh_7 : std_logic_vector(13 downto 0);
	signal bypass_idx_oh_8 : std_logic_vector(13 downto 0);
	signal bypass_idx_oh_9 : std_logic_vector(13 downto 0);
	signal bypass_idx_oh_10 : std_logic_vector(13 downto 0);
	signal bypass_idx_oh_11 : std_logic_vector(13 downto 0);
	signal bypass_idx_oh_12 : std_logic_vector(13 downto 0);
	signal bypass_idx_oh_13 : std_logic_vector(13 downto 0);
	signal bypass_en_0 : std_logic;
	signal bypass_en_1 : std_logic;
	signal bypass_en_2 : std_logic;
	signal bypass_en_3 : std_logic;
	signal bypass_en_4 : std_logic;
	signal bypass_en_5 : std_logic;
	signal bypass_en_6 : std_logic;
	signal bypass_en_7 : std_logic;
	signal bypass_en_8 : std_logic;
	signal bypass_en_9 : std_logic;
	signal bypass_en_10 : std_logic;
	signal bypass_en_11 : std_logic;
	signal bypass_en_12 : std_logic;
	signal bypass_en_13 : std_logic;
	signal ld_st_conflict_0 : std_logic_vector(13 downto 0);
	signal ld_st_conflict_1 : std_logic_vector(13 downto 0);
	signal ld_st_conflict_2 : std_logic_vector(13 downto 0);
	signal ld_st_conflict_3 : std_logic_vector(13 downto 0);
	signal ld_st_conflict_4 : std_logic_vector(13 downto 0);
	signal ld_st_conflict_5 : std_logic_vector(13 downto 0);
	signal ld_st_conflict_6 : std_logic_vector(13 downto 0);
	signal ld_st_conflict_7 : std_logic_vector(13 downto 0);
	signal ld_st_conflict_8 : std_logic_vector(13 downto 0);
	signal ld_st_conflict_9 : std_logic_vector(13 downto 0);
	signal ld_st_conflict_10 : std_logic_vector(13 downto 0);
	signal ld_st_conflict_11 : std_logic_vector(13 downto 0);
	signal ld_st_conflict_12 : std_logic_vector(13 downto 0);
	signal ld_st_conflict_13 : std_logic_vector(13 downto 0);
	signal can_bypass_0 : std_logic_vector(13 downto 0);
	signal can_bypass_1 : std_logic_vector(13 downto 0);
	signal can_bypass_2 : std_logic_vector(13 downto 0);
	signal can_bypass_3 : std_logic_vector(13 downto 0);
	signal can_bypass_4 : std_logic_vector(13 downto 0);
	signal can_bypass_5 : std_logic_vector(13 downto 0);
	signal can_bypass_6 : std_logic_vector(13 downto 0);
	signal can_bypass_7 : std_logic_vector(13 downto 0);
	signal can_bypass_8 : std_logic_vector(13 downto 0);
	signal can_bypass_9 : std_logic_vector(13 downto 0);
	signal can_bypass_10 : std_logic_vector(13 downto 0);
	signal can_bypass_11 : std_logic_vector(13 downto 0);
	signal can_bypass_12 : std_logic_vector(13 downto 0);
	signal can_bypass_13 : std_logic_vector(13 downto 0);
	signal addr_valid_0 : std_logic_vector(13 downto 0);
	signal addr_valid_1 : std_logic_vector(13 downto 0);
	signal addr_valid_2 : std_logic_vector(13 downto 0);
	signal addr_valid_3 : std_logic_vector(13 downto 0);
	signal addr_valid_4 : std_logic_vector(13 downto 0);
	signal addr_valid_5 : std_logic_vector(13 downto 0);
	signal addr_valid_6 : std_logic_vector(13 downto 0);
	signal addr_valid_7 : std_logic_vector(13 downto 0);
	signal addr_valid_8 : std_logic_vector(13 downto 0);
	signal addr_valid_9 : std_logic_vector(13 downto 0);
	signal addr_valid_10 : std_logic_vector(13 downto 0);
	signal addr_valid_11 : std_logic_vector(13 downto 0);
	signal addr_valid_12 : std_logic_vector(13 downto 0);
	signal addr_valid_13 : std_logic_vector(13 downto 0);
	signal addr_same_0 : std_logic_vector(13 downto 0);
	signal addr_same_1 : std_logic_vector(13 downto 0);
	signal addr_same_2 : std_logic_vector(13 downto 0);
	signal addr_same_3 : std_logic_vector(13 downto 0);
	signal addr_same_4 : std_logic_vector(13 downto 0);
	signal addr_same_5 : std_logic_vector(13 downto 0);
	signal addr_same_6 : std_logic_vector(13 downto 0);
	signal addr_same_7 : std_logic_vector(13 downto 0);
	signal addr_same_8 : std_logic_vector(13 downto 0);
	signal addr_same_9 : std_logic_vector(13 downto 0);
	signal addr_same_10 : std_logic_vector(13 downto 0);
	signal addr_same_11 : std_logic_vector(13 downto 0);
	signal addr_same_12 : std_logic_vector(13 downto 0);
	signal addr_same_13 : std_logic_vector(13 downto 0);
	signal load_conflict_0 : std_logic;
	signal load_conflict_1 : std_logic;
	signal load_conflict_2 : std_logic;
	signal load_conflict_3 : std_logic;
	signal load_conflict_4 : std_logic;
	signal load_conflict_5 : std_logic;
	signal load_conflict_6 : std_logic;
	signal load_conflict_7 : std_logic;
	signal load_conflict_8 : std_logic;
	signal load_conflict_9 : std_logic;
	signal load_conflict_10 : std_logic;
	signal load_conflict_11 : std_logic;
	signal load_conflict_12 : std_logic;
	signal load_conflict_13 : std_logic;
	signal load_req_valid_0 : std_logic;
	signal load_req_valid_1 : std_logic;
	signal load_req_valid_2 : std_logic;
	signal load_req_valid_3 : std_logic;
	signal load_req_valid_4 : std_logic;
	signal load_req_valid_5 : std_logic;
	signal load_req_valid_6 : std_logic;
	signal load_req_valid_7 : std_logic;
	signal load_req_valid_8 : std_logic;
	signal load_req_valid_9 : std_logic;
	signal load_req_valid_10 : std_logic;
	signal load_req_valid_11 : std_logic;
	signal load_req_valid_12 : std_logic;
	signal load_req_valid_13 : std_logic;
	signal can_load_0 : std_logic;
	signal can_load_1 : std_logic;
	signal can_load_2 : std_logic;
	signal can_load_3 : std_logic;
	signal can_load_4 : std_logic;
	signal can_load_5 : std_logic;
	signal can_load_6 : std_logic;
	signal can_load_7 : std_logic;
	signal can_load_8 : std_logic;
	signal can_load_9 : std_logic;
	signal can_load_10 : std_logic;
	signal can_load_11 : std_logic;
	signal can_load_12 : std_logic;
	signal can_load_13 : std_logic;
	signal TEMP_23_res : std_logic_vector(7 downto 0);
	signal TEMP_24_res : std_logic_vector(3 downto 0);
	signal TEMP_25_res : std_logic_vector(1 downto 0);
	signal TEMP_26_res : std_logic_vector(7 downto 0);
	signal TEMP_27_res : std_logic_vector(3 downto 0);
	signal TEMP_28_res : std_logic_vector(1 downto 0);
	signal TEMP_29_res : std_logic_vector(7 downto 0);
	signal TEMP_30_res : std_logic_vector(3 downto 0);
	signal TEMP_31_res : std_logic_vector(1 downto 0);
	signal TEMP_32_res : std_logic_vector(7 downto 0);
	signal TEMP_33_res : std_logic_vector(3 downto 0);
	signal TEMP_34_res : std_logic_vector(1 downto 0);
	signal TEMP_35_res : std_logic_vector(7 downto 0);
	signal TEMP_36_res : std_logic_vector(3 downto 0);
	signal TEMP_37_res : std_logic_vector(1 downto 0);
	signal TEMP_38_res : std_logic_vector(7 downto 0);
	signal TEMP_39_res : std_logic_vector(3 downto 0);
	signal TEMP_40_res : std_logic_vector(1 downto 0);
	signal TEMP_41_res : std_logic_vector(7 downto 0);
	signal TEMP_42_res : std_logic_vector(3 downto 0);
	signal TEMP_43_res : std_logic_vector(1 downto 0);
	signal TEMP_44_res : std_logic_vector(7 downto 0);
	signal TEMP_45_res : std_logic_vector(3 downto 0);
	signal TEMP_46_res : std_logic_vector(1 downto 0);
	signal TEMP_47_res : std_logic_vector(7 downto 0);
	signal TEMP_48_res : std_logic_vector(3 downto 0);
	signal TEMP_49_res : std_logic_vector(1 downto 0);
	signal TEMP_50_res : std_logic_vector(7 downto 0);
	signal TEMP_51_res : std_logic_vector(3 downto 0);
	signal TEMP_52_res : std_logic_vector(1 downto 0);
	signal TEMP_53_res : std_logic_vector(7 downto 0);
	signal TEMP_54_res : std_logic_vector(3 downto 0);
	signal TEMP_55_res : std_logic_vector(1 downto 0);
	signal TEMP_56_res : std_logic_vector(7 downto 0);
	signal TEMP_57_res : std_logic_vector(3 downto 0);
	signal TEMP_58_res : std_logic_vector(1 downto 0);
	signal TEMP_59_res : std_logic_vector(7 downto 0);
	signal TEMP_60_res : std_logic_vector(3 downto 0);
	signal TEMP_61_res : std_logic_vector(1 downto 0);
	signal TEMP_62_res : std_logic_vector(7 downto 0);
	signal TEMP_63_res : std_logic_vector(3 downto 0);
	signal TEMP_64_res : std_logic_vector(1 downto 0);
	signal TEMP_65_double_in : std_logic_vector(27 downto 0);
	signal TEMP_65_double_out : std_logic_vector(27 downto 0);
	signal TEMP_66_res_0 : std_logic;
	signal TEMP_66_res_1 : std_logic;
	signal TEMP_66_res_2 : std_logic;
	signal TEMP_66_res_3 : std_logic;
	signal TEMP_66_res_4 : std_logic;
	signal TEMP_66_res_5 : std_logic;
	signal TEMP_66_res_6 : std_logic;
	signal TEMP_66_res_7 : std_logic;
	signal TEMP_67_res_0 : std_logic;
	signal TEMP_67_res_1 : std_logic;
	signal TEMP_67_res_2 : std_logic;
	signal TEMP_67_res_3 : std_logic;
	signal TEMP_68_res_0 : std_logic;
	signal TEMP_68_res_1 : std_logic;
	signal st_ld_conflict : std_logic_vector(13 downto 0);
	signal store_conflict : std_logic;
	signal store_valid : std_logic;
	signal store_data_valid : std_logic;
	signal store_addr_valid : std_logic;
	signal TEMP_69_res : std_logic_vector(7 downto 0);
	signal TEMP_70_res : std_logic_vector(3 downto 0);
	signal TEMP_71_res : std_logic_vector(1 downto 0);
	signal stq_last_oh : std_logic_vector(13 downto 0);
	signal bypass_en_vec_0 : std_logic_vector(13 downto 0);
	signal TEMP_72_double_in : std_logic_vector(27 downto 0);
	signal TEMP_72_base_rev : std_logic_vector(13 downto 0);
	signal TEMP_72_double_out : std_logic_vector(27 downto 0);
	signal TEMP_73_res : std_logic_vector(7 downto 0);
	signal TEMP_74_res : std_logic_vector(3 downto 0);
	signal TEMP_75_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_1 : std_logic_vector(13 downto 0);
	signal TEMP_76_double_in : std_logic_vector(27 downto 0);
	signal TEMP_76_base_rev : std_logic_vector(13 downto 0);
	signal TEMP_76_double_out : std_logic_vector(27 downto 0);
	signal TEMP_77_res : std_logic_vector(7 downto 0);
	signal TEMP_78_res : std_logic_vector(3 downto 0);
	signal TEMP_79_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_2 : std_logic_vector(13 downto 0);
	signal TEMP_80_double_in : std_logic_vector(27 downto 0);
	signal TEMP_80_base_rev : std_logic_vector(13 downto 0);
	signal TEMP_80_double_out : std_logic_vector(27 downto 0);
	signal TEMP_81_res : std_logic_vector(7 downto 0);
	signal TEMP_82_res : std_logic_vector(3 downto 0);
	signal TEMP_83_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_3 : std_logic_vector(13 downto 0);
	signal TEMP_84_double_in : std_logic_vector(27 downto 0);
	signal TEMP_84_base_rev : std_logic_vector(13 downto 0);
	signal TEMP_84_double_out : std_logic_vector(27 downto 0);
	signal TEMP_85_res : std_logic_vector(7 downto 0);
	signal TEMP_86_res : std_logic_vector(3 downto 0);
	signal TEMP_87_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_4 : std_logic_vector(13 downto 0);
	signal TEMP_88_double_in : std_logic_vector(27 downto 0);
	signal TEMP_88_base_rev : std_logic_vector(13 downto 0);
	signal TEMP_88_double_out : std_logic_vector(27 downto 0);
	signal TEMP_89_res : std_logic_vector(7 downto 0);
	signal TEMP_90_res : std_logic_vector(3 downto 0);
	signal TEMP_91_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_5 : std_logic_vector(13 downto 0);
	signal TEMP_92_double_in : std_logic_vector(27 downto 0);
	signal TEMP_92_base_rev : std_logic_vector(13 downto 0);
	signal TEMP_92_double_out : std_logic_vector(27 downto 0);
	signal TEMP_93_res : std_logic_vector(7 downto 0);
	signal TEMP_94_res : std_logic_vector(3 downto 0);
	signal TEMP_95_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_6 : std_logic_vector(13 downto 0);
	signal TEMP_96_double_in : std_logic_vector(27 downto 0);
	signal TEMP_96_base_rev : std_logic_vector(13 downto 0);
	signal TEMP_96_double_out : std_logic_vector(27 downto 0);
	signal TEMP_97_res : std_logic_vector(7 downto 0);
	signal TEMP_98_res : std_logic_vector(3 downto 0);
	signal TEMP_99_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_7 : std_logic_vector(13 downto 0);
	signal TEMP_100_double_in : std_logic_vector(27 downto 0);
	signal TEMP_100_base_rev : std_logic_vector(13 downto 0);
	signal TEMP_100_double_out : std_logic_vector(27 downto 0);
	signal TEMP_101_res : std_logic_vector(7 downto 0);
	signal TEMP_102_res : std_logic_vector(3 downto 0);
	signal TEMP_103_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_8 : std_logic_vector(13 downto 0);
	signal TEMP_104_double_in : std_logic_vector(27 downto 0);
	signal TEMP_104_base_rev : std_logic_vector(13 downto 0);
	signal TEMP_104_double_out : std_logic_vector(27 downto 0);
	signal TEMP_105_res : std_logic_vector(7 downto 0);
	signal TEMP_106_res : std_logic_vector(3 downto 0);
	signal TEMP_107_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_9 : std_logic_vector(13 downto 0);
	signal TEMP_108_double_in : std_logic_vector(27 downto 0);
	signal TEMP_108_base_rev : std_logic_vector(13 downto 0);
	signal TEMP_108_double_out : std_logic_vector(27 downto 0);
	signal TEMP_109_res : std_logic_vector(7 downto 0);
	signal TEMP_110_res : std_logic_vector(3 downto 0);
	signal TEMP_111_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_10 : std_logic_vector(13 downto 0);
	signal TEMP_112_double_in : std_logic_vector(27 downto 0);
	signal TEMP_112_base_rev : std_logic_vector(13 downto 0);
	signal TEMP_112_double_out : std_logic_vector(27 downto 0);
	signal TEMP_113_res : std_logic_vector(7 downto 0);
	signal TEMP_114_res : std_logic_vector(3 downto 0);
	signal TEMP_115_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_11 : std_logic_vector(13 downto 0);
	signal TEMP_116_double_in : std_logic_vector(27 downto 0);
	signal TEMP_116_base_rev : std_logic_vector(13 downto 0);
	signal TEMP_116_double_out : std_logic_vector(27 downto 0);
	signal TEMP_117_res : std_logic_vector(7 downto 0);
	signal TEMP_118_res : std_logic_vector(3 downto 0);
	signal TEMP_119_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_12 : std_logic_vector(13 downto 0);
	signal TEMP_120_double_in : std_logic_vector(27 downto 0);
	signal TEMP_120_base_rev : std_logic_vector(13 downto 0);
	signal TEMP_120_double_out : std_logic_vector(27 downto 0);
	signal TEMP_121_res : std_logic_vector(7 downto 0);
	signal TEMP_122_res : std_logic_vector(3 downto 0);
	signal TEMP_123_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_13 : std_logic_vector(13 downto 0);
	signal TEMP_124_double_in : std_logic_vector(27 downto 0);
	signal TEMP_124_base_rev : std_logic_vector(13 downto 0);
	signal TEMP_124_double_out : std_logic_vector(27 downto 0);
	signal TEMP_125_res : std_logic_vector(7 downto 0);
	signal TEMP_126_res : std_logic_vector(3 downto 0);
	signal TEMP_127_res : std_logic_vector(1 downto 0);
	signal TEMP_128_in_0_0 : std_logic;
	signal TEMP_128_in_0_1 : std_logic;
	signal TEMP_128_in_0_2 : std_logic;
	signal TEMP_128_in_0_3 : std_logic;
	signal TEMP_128_in_0_4 : std_logic;
	signal TEMP_128_in_0_5 : std_logic;
	signal TEMP_128_in_0_6 : std_logic;
	signal TEMP_128_in_0_7 : std_logic;
	signal TEMP_128_in_0_8 : std_logic;
	signal TEMP_128_in_0_9 : std_logic;
	signal TEMP_128_in_0_10 : std_logic;
	signal TEMP_128_in_0_11 : std_logic;
	signal TEMP_128_in_0_12 : std_logic;
	signal TEMP_128_in_0_13 : std_logic;
	signal TEMP_128_out_0 : std_logic;
	signal TEMP_129_res_0 : std_logic;
	signal TEMP_129_res_1 : std_logic;
	signal TEMP_129_res_2 : std_logic;
	signal TEMP_129_res_3 : std_logic;
	signal TEMP_129_res_4 : std_logic;
	signal TEMP_129_res_5 : std_logic;
	signal TEMP_129_res_6 : std_logic;
	signal TEMP_129_res_7 : std_logic;
	signal TEMP_130_res_0 : std_logic;
	signal TEMP_130_res_1 : std_logic;
	signal TEMP_130_res_2 : std_logic;
	signal TEMP_130_res_3 : std_logic;
	signal TEMP_131_res_0 : std_logic;
	signal TEMP_131_res_1 : std_logic;
	signal TEMP_131_in_1_0 : std_logic;
	signal TEMP_131_in_1_1 : std_logic;
	signal TEMP_131_in_1_2 : std_logic;
	signal TEMP_131_in_1_3 : std_logic;
	signal TEMP_131_in_1_4 : std_logic;
	signal TEMP_131_in_1_5 : std_logic;
	signal TEMP_131_in_1_6 : std_logic;
	signal TEMP_131_in_1_7 : std_logic;
	signal TEMP_131_in_1_8 : std_logic;
	signal TEMP_131_in_1_9 : std_logic;
	signal TEMP_131_in_1_10 : std_logic;
	signal TEMP_131_in_1_11 : std_logic;
	signal TEMP_131_in_1_12 : std_logic;
	signal TEMP_131_in_1_13 : std_logic;
	signal TEMP_131_out_1 : std_logic;
	signal TEMP_132_res_0 : std_logic;
	signal TEMP_132_res_1 : std_logic;
	signal TEMP_132_res_2 : std_logic;
	signal TEMP_132_res_3 : std_logic;
	signal TEMP_132_res_4 : std_logic;
	signal TEMP_132_res_5 : std_logic;
	signal TEMP_132_res_6 : std_logic;
	signal TEMP_132_res_7 : std_logic;
	signal TEMP_133_res_0 : std_logic;
	signal TEMP_133_res_1 : std_logic;
	signal TEMP_133_res_2 : std_logic;
	signal TEMP_133_res_3 : std_logic;
	signal TEMP_134_res_0 : std_logic;
	signal TEMP_134_res_1 : std_logic;
	signal TEMP_134_in_2_0 : std_logic;
	signal TEMP_134_in_2_1 : std_logic;
	signal TEMP_134_in_2_2 : std_logic;
	signal TEMP_134_in_2_3 : std_logic;
	signal TEMP_134_in_2_4 : std_logic;
	signal TEMP_134_in_2_5 : std_logic;
	signal TEMP_134_in_2_6 : std_logic;
	signal TEMP_134_in_2_7 : std_logic;
	signal TEMP_134_in_2_8 : std_logic;
	signal TEMP_134_in_2_9 : std_logic;
	signal TEMP_134_in_2_10 : std_logic;
	signal TEMP_134_in_2_11 : std_logic;
	signal TEMP_134_in_2_12 : std_logic;
	signal TEMP_134_in_2_13 : std_logic;
	signal TEMP_134_out_2 : std_logic;
	signal TEMP_135_res_0 : std_logic;
	signal TEMP_135_res_1 : std_logic;
	signal TEMP_135_res_2 : std_logic;
	signal TEMP_135_res_3 : std_logic;
	signal TEMP_135_res_4 : std_logic;
	signal TEMP_135_res_5 : std_logic;
	signal TEMP_135_res_6 : std_logic;
	signal TEMP_135_res_7 : std_logic;
	signal TEMP_136_res_0 : std_logic;
	signal TEMP_136_res_1 : std_logic;
	signal TEMP_136_res_2 : std_logic;
	signal TEMP_136_res_3 : std_logic;
	signal TEMP_137_res_0 : std_logic;
	signal TEMP_137_res_1 : std_logic;
	signal TEMP_137_in_3_0 : std_logic;
	signal TEMP_137_in_3_1 : std_logic;
	signal TEMP_137_in_3_2 : std_logic;
	signal TEMP_137_in_3_3 : std_logic;
	signal TEMP_137_in_3_4 : std_logic;
	signal TEMP_137_in_3_5 : std_logic;
	signal TEMP_137_in_3_6 : std_logic;
	signal TEMP_137_in_3_7 : std_logic;
	signal TEMP_137_in_3_8 : std_logic;
	signal TEMP_137_in_3_9 : std_logic;
	signal TEMP_137_in_3_10 : std_logic;
	signal TEMP_137_in_3_11 : std_logic;
	signal TEMP_137_in_3_12 : std_logic;
	signal TEMP_137_in_3_13 : std_logic;
	signal TEMP_137_out_3 : std_logic;
	signal TEMP_138_res_0 : std_logic;
	signal TEMP_138_res_1 : std_logic;
	signal TEMP_138_res_2 : std_logic;
	signal TEMP_138_res_3 : std_logic;
	signal TEMP_138_res_4 : std_logic;
	signal TEMP_138_res_5 : std_logic;
	signal TEMP_138_res_6 : std_logic;
	signal TEMP_138_res_7 : std_logic;
	signal TEMP_139_res_0 : std_logic;
	signal TEMP_139_res_1 : std_logic;
	signal TEMP_139_res_2 : std_logic;
	signal TEMP_139_res_3 : std_logic;
	signal TEMP_140_res_0 : std_logic;
	signal TEMP_140_res_1 : std_logic;
	signal TEMP_141_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_141_mux_1 : std_logic_vector(9 downto 0);
	signal TEMP_141_mux_2 : std_logic_vector(9 downto 0);
	signal TEMP_141_mux_3 : std_logic_vector(9 downto 0);
	signal TEMP_141_mux_4 : std_logic_vector(9 downto 0);
	signal TEMP_141_mux_5 : std_logic_vector(9 downto 0);
	signal TEMP_141_mux_6 : std_logic_vector(9 downto 0);
	signal TEMP_141_mux_7 : std_logic_vector(9 downto 0);
	signal TEMP_141_mux_8 : std_logic_vector(9 downto 0);
	signal TEMP_141_mux_9 : std_logic_vector(9 downto 0);
	signal TEMP_141_mux_10 : std_logic_vector(9 downto 0);
	signal TEMP_141_mux_11 : std_logic_vector(9 downto 0);
	signal TEMP_141_mux_12 : std_logic_vector(9 downto 0);
	signal TEMP_141_mux_13 : std_logic_vector(9 downto 0);
	signal TEMP_142_res_0 : std_logic_vector(9 downto 0);
	signal TEMP_142_res_1 : std_logic_vector(9 downto 0);
	signal TEMP_142_res_2 : std_logic_vector(9 downto 0);
	signal TEMP_142_res_3 : std_logic_vector(9 downto 0);
	signal TEMP_142_res_4 : std_logic_vector(9 downto 0);
	signal TEMP_142_res_5 : std_logic_vector(9 downto 0);
	signal TEMP_142_res_6 : std_logic_vector(9 downto 0);
	signal TEMP_142_res_7 : std_logic_vector(9 downto 0);
	signal TEMP_143_res_0 : std_logic_vector(9 downto 0);
	signal TEMP_143_res_1 : std_logic_vector(9 downto 0);
	signal TEMP_143_res_2 : std_logic_vector(9 downto 0);
	signal TEMP_143_res_3 : std_logic_vector(9 downto 0);
	signal TEMP_144_res_0 : std_logic_vector(9 downto 0);
	signal TEMP_144_res_1 : std_logic_vector(9 downto 0);
	signal ldq_issue_set_vec_0 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_1 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_2 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_3 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_4 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_5 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_6 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_7 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_8 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_9 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_10 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_11 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_12 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_13 : std_logic_vector(0 downto 0);
	signal read_idx_oh_0_0 : std_logic;
	signal read_valid_0 : std_logic;
	signal read_data_0 : std_logic_vector(31 downto 0);
	signal TEMP_145_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_0 : std_logic_vector(31 downto 0);
	signal TEMP_146_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_146_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_146_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_146_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_146_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_146_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_146_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_146_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_146_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_146_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_146_mux_10 : std_logic_vector(31 downto 0);
	signal TEMP_146_mux_11 : std_logic_vector(31 downto 0);
	signal TEMP_146_mux_12 : std_logic_vector(31 downto 0);
	signal TEMP_146_mux_13 : std_logic_vector(31 downto 0);
	signal TEMP_147_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_147_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_147_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_147_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_147_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_147_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_147_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_147_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_148_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_148_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_148_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_148_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_149_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_149_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_1_0 : std_logic;
	signal read_valid_1 : std_logic;
	signal read_data_1 : std_logic_vector(31 downto 0);
	signal TEMP_150_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_1 : std_logic_vector(31 downto 0);
	signal TEMP_151_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_151_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_151_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_151_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_151_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_151_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_151_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_151_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_151_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_151_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_151_mux_10 : std_logic_vector(31 downto 0);
	signal TEMP_151_mux_11 : std_logic_vector(31 downto 0);
	signal TEMP_151_mux_12 : std_logic_vector(31 downto 0);
	signal TEMP_151_mux_13 : std_logic_vector(31 downto 0);
	signal TEMP_152_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_152_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_152_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_152_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_152_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_152_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_152_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_152_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_153_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_153_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_153_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_153_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_154_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_154_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_2_0 : std_logic;
	signal read_valid_2 : std_logic;
	signal read_data_2 : std_logic_vector(31 downto 0);
	signal TEMP_155_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_2 : std_logic_vector(31 downto 0);
	signal TEMP_156_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_156_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_156_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_156_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_156_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_156_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_156_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_156_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_156_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_156_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_156_mux_10 : std_logic_vector(31 downto 0);
	signal TEMP_156_mux_11 : std_logic_vector(31 downto 0);
	signal TEMP_156_mux_12 : std_logic_vector(31 downto 0);
	signal TEMP_156_mux_13 : std_logic_vector(31 downto 0);
	signal TEMP_157_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_157_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_157_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_157_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_157_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_157_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_157_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_157_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_158_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_158_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_158_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_158_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_159_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_159_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_3_0 : std_logic;
	signal read_valid_3 : std_logic;
	signal read_data_3 : std_logic_vector(31 downto 0);
	signal TEMP_160_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_3 : std_logic_vector(31 downto 0);
	signal TEMP_161_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_161_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_161_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_161_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_161_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_161_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_161_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_161_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_161_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_161_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_161_mux_10 : std_logic_vector(31 downto 0);
	signal TEMP_161_mux_11 : std_logic_vector(31 downto 0);
	signal TEMP_161_mux_12 : std_logic_vector(31 downto 0);
	signal TEMP_161_mux_13 : std_logic_vector(31 downto 0);
	signal TEMP_162_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_162_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_162_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_162_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_162_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_162_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_162_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_162_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_163_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_163_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_163_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_163_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_164_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_164_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_4_0 : std_logic;
	signal read_valid_4 : std_logic;
	signal read_data_4 : std_logic_vector(31 downto 0);
	signal TEMP_165_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_4 : std_logic_vector(31 downto 0);
	signal TEMP_166_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_166_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_166_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_166_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_166_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_166_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_166_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_166_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_166_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_166_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_166_mux_10 : std_logic_vector(31 downto 0);
	signal TEMP_166_mux_11 : std_logic_vector(31 downto 0);
	signal TEMP_166_mux_12 : std_logic_vector(31 downto 0);
	signal TEMP_166_mux_13 : std_logic_vector(31 downto 0);
	signal TEMP_167_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_167_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_167_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_167_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_167_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_167_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_167_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_167_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_168_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_168_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_168_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_168_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_169_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_169_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_5_0 : std_logic;
	signal read_valid_5 : std_logic;
	signal read_data_5 : std_logic_vector(31 downto 0);
	signal TEMP_170_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_5 : std_logic_vector(31 downto 0);
	signal TEMP_171_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_171_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_171_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_171_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_171_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_171_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_171_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_171_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_171_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_171_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_171_mux_10 : std_logic_vector(31 downto 0);
	signal TEMP_171_mux_11 : std_logic_vector(31 downto 0);
	signal TEMP_171_mux_12 : std_logic_vector(31 downto 0);
	signal TEMP_171_mux_13 : std_logic_vector(31 downto 0);
	signal TEMP_172_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_172_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_172_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_172_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_172_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_172_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_172_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_172_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_173_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_173_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_173_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_173_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_174_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_174_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_6_0 : std_logic;
	signal read_valid_6 : std_logic;
	signal read_data_6 : std_logic_vector(31 downto 0);
	signal TEMP_175_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_6 : std_logic_vector(31 downto 0);
	signal TEMP_176_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_176_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_176_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_176_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_176_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_176_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_176_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_176_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_176_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_176_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_176_mux_10 : std_logic_vector(31 downto 0);
	signal TEMP_176_mux_11 : std_logic_vector(31 downto 0);
	signal TEMP_176_mux_12 : std_logic_vector(31 downto 0);
	signal TEMP_176_mux_13 : std_logic_vector(31 downto 0);
	signal TEMP_177_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_177_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_177_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_177_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_177_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_177_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_177_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_177_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_178_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_178_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_178_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_178_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_179_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_179_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_7_0 : std_logic;
	signal read_valid_7 : std_logic;
	signal read_data_7 : std_logic_vector(31 downto 0);
	signal TEMP_180_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_7 : std_logic_vector(31 downto 0);
	signal TEMP_181_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_181_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_181_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_181_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_181_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_181_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_181_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_181_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_181_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_181_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_181_mux_10 : std_logic_vector(31 downto 0);
	signal TEMP_181_mux_11 : std_logic_vector(31 downto 0);
	signal TEMP_181_mux_12 : std_logic_vector(31 downto 0);
	signal TEMP_181_mux_13 : std_logic_vector(31 downto 0);
	signal TEMP_182_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_182_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_182_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_182_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_182_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_182_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_182_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_182_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_183_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_183_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_183_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_183_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_184_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_184_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_8_0 : std_logic;
	signal read_valid_8 : std_logic;
	signal read_data_8 : std_logic_vector(31 downto 0);
	signal TEMP_185_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_8 : std_logic_vector(31 downto 0);
	signal TEMP_186_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_186_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_186_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_186_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_186_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_186_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_186_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_186_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_186_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_186_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_186_mux_10 : std_logic_vector(31 downto 0);
	signal TEMP_186_mux_11 : std_logic_vector(31 downto 0);
	signal TEMP_186_mux_12 : std_logic_vector(31 downto 0);
	signal TEMP_186_mux_13 : std_logic_vector(31 downto 0);
	signal TEMP_187_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_187_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_187_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_187_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_187_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_187_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_187_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_187_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_188_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_188_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_188_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_188_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_189_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_189_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_9_0 : std_logic;
	signal read_valid_9 : std_logic;
	signal read_data_9 : std_logic_vector(31 downto 0);
	signal TEMP_190_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_9 : std_logic_vector(31 downto 0);
	signal TEMP_191_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_191_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_191_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_191_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_191_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_191_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_191_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_191_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_191_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_191_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_191_mux_10 : std_logic_vector(31 downto 0);
	signal TEMP_191_mux_11 : std_logic_vector(31 downto 0);
	signal TEMP_191_mux_12 : std_logic_vector(31 downto 0);
	signal TEMP_191_mux_13 : std_logic_vector(31 downto 0);
	signal TEMP_192_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_192_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_192_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_192_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_192_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_192_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_192_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_192_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_193_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_193_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_193_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_193_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_194_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_194_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_10_0 : std_logic;
	signal read_valid_10 : std_logic;
	signal read_data_10 : std_logic_vector(31 downto 0);
	signal TEMP_195_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_10 : std_logic_vector(31 downto 0);
	signal TEMP_196_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_196_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_196_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_196_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_196_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_196_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_196_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_196_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_196_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_196_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_196_mux_10 : std_logic_vector(31 downto 0);
	signal TEMP_196_mux_11 : std_logic_vector(31 downto 0);
	signal TEMP_196_mux_12 : std_logic_vector(31 downto 0);
	signal TEMP_196_mux_13 : std_logic_vector(31 downto 0);
	signal TEMP_197_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_197_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_197_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_197_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_197_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_197_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_197_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_197_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_198_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_198_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_198_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_198_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_199_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_199_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_11_0 : std_logic;
	signal read_valid_11 : std_logic;
	signal read_data_11 : std_logic_vector(31 downto 0);
	signal TEMP_200_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_11 : std_logic_vector(31 downto 0);
	signal TEMP_201_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_201_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_201_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_201_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_201_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_201_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_201_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_201_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_201_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_201_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_201_mux_10 : std_logic_vector(31 downto 0);
	signal TEMP_201_mux_11 : std_logic_vector(31 downto 0);
	signal TEMP_201_mux_12 : std_logic_vector(31 downto 0);
	signal TEMP_201_mux_13 : std_logic_vector(31 downto 0);
	signal TEMP_202_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_202_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_202_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_202_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_202_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_202_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_202_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_202_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_203_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_203_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_203_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_203_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_204_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_204_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_12_0 : std_logic;
	signal read_valid_12 : std_logic;
	signal read_data_12 : std_logic_vector(31 downto 0);
	signal TEMP_205_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_12 : std_logic_vector(31 downto 0);
	signal TEMP_206_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_206_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_206_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_206_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_206_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_206_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_206_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_206_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_206_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_206_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_206_mux_10 : std_logic_vector(31 downto 0);
	signal TEMP_206_mux_11 : std_logic_vector(31 downto 0);
	signal TEMP_206_mux_12 : std_logic_vector(31 downto 0);
	signal TEMP_206_mux_13 : std_logic_vector(31 downto 0);
	signal TEMP_207_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_207_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_207_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_207_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_207_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_207_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_207_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_207_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_208_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_208_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_208_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_208_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_209_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_209_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_13_0 : std_logic;
	signal read_valid_13 : std_logic;
	signal read_data_13 : std_logic_vector(31 downto 0);
	signal TEMP_210_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_13 : std_logic_vector(31 downto 0);
	signal TEMP_211_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_211_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_211_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_211_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_211_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_211_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_211_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_211_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_211_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_211_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_211_mux_10 : std_logic_vector(31 downto 0);
	signal TEMP_211_mux_11 : std_logic_vector(31 downto 0);
	signal TEMP_211_mux_12 : std_logic_vector(31 downto 0);
	signal TEMP_211_mux_13 : std_logic_vector(31 downto 0);
	signal TEMP_212_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_212_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_212_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_212_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_212_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_212_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_212_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_212_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_213_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_213_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_213_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_213_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_214_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_214_res_1 : std_logic_vector(31 downto 0);
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
	ldq_head_oh(0) <= '1' when ldq_head_q = "0000" else '0';
	ldq_head_oh(1) <= '1' when ldq_head_q = "0001" else '0';
	ldq_head_oh(2) <= '1' when ldq_head_q = "0010" else '0';
	ldq_head_oh(3) <= '1' when ldq_head_q = "0011" else '0';
	ldq_head_oh(4) <= '1' when ldq_head_q = "0100" else '0';
	ldq_head_oh(5) <= '1' when ldq_head_q = "0101" else '0';
	ldq_head_oh(6) <= '1' when ldq_head_q = "0110" else '0';
	ldq_head_oh(7) <= '1' when ldq_head_q = "0111" else '0';
	ldq_head_oh(8) <= '1' when ldq_head_q = "1000" else '0';
	ldq_head_oh(9) <= '1' when ldq_head_q = "1001" else '0';
	ldq_head_oh(10) <= '1' when ldq_head_q = "1010" else '0';
	ldq_head_oh(11) <= '1' when ldq_head_q = "1011" else '0';
	ldq_head_oh(12) <= '1' when ldq_head_q = "1100" else '0';
	ldq_head_oh(13) <= '1' when ldq_head_q = "1101" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(stq_head_oh, stq_head)
	stq_head_oh(0) <= '1' when stq_head_q = "0000" else '0';
	stq_head_oh(1) <= '1' when stq_head_q = "0001" else '0';
	stq_head_oh(2) <= '1' when stq_head_q = "0010" else '0';
	stq_head_oh(3) <= '1' when stq_head_q = "0011" else '0';
	stq_head_oh(4) <= '1' when stq_head_q = "0100" else '0';
	stq_head_oh(5) <= '1' when stq_head_q = "0101" else '0';
	stq_head_oh(6) <= '1' when stq_head_q = "0110" else '0';
	stq_head_oh(7) <= '1' when stq_head_q = "0111" else '0';
	stq_head_oh(8) <= '1' when stq_head_q = "1000" else '0';
	stq_head_oh(9) <= '1' when stq_head_q = "1001" else '0';
	stq_head_oh(10) <= '1' when stq_head_q = "1010" else '0';
	stq_head_oh(11) <= '1' when stq_head_q = "1011" else '0';
	stq_head_oh(12) <= '1' when stq_head_q = "1100" else '0';
	stq_head_oh(13) <= '1' when stq_head_q = "1101" else '0';
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
	ldq_alloc_next_4 <= not ldq_reset_4 and ldq_alloc_4_q;
	ldq_alloc_4_d <= ldq_wen_4 or ldq_alloc_next_4;
	ldq_issue_4_d <= not ldq_wen_4 and ( ldq_issue_set_4 or ldq_issue_4_q );
	ldq_addr_valid_4_d <= not ldq_wen_4 and ( ldq_addr_wen_4 or ldq_addr_valid_4_q );
	ldq_data_valid_4_d <= not ldq_wen_4 and ( ldq_data_wen_4 or ldq_data_valid_4_q );
	ldq_alloc_next_5 <= not ldq_reset_5 and ldq_alloc_5_q;
	ldq_alloc_5_d <= ldq_wen_5 or ldq_alloc_next_5;
	ldq_issue_5_d <= not ldq_wen_5 and ( ldq_issue_set_5 or ldq_issue_5_q );
	ldq_addr_valid_5_d <= not ldq_wen_5 and ( ldq_addr_wen_5 or ldq_addr_valid_5_q );
	ldq_data_valid_5_d <= not ldq_wen_5 and ( ldq_data_wen_5 or ldq_data_valid_5_q );
	ldq_alloc_next_6 <= not ldq_reset_6 and ldq_alloc_6_q;
	ldq_alloc_6_d <= ldq_wen_6 or ldq_alloc_next_6;
	ldq_issue_6_d <= not ldq_wen_6 and ( ldq_issue_set_6 or ldq_issue_6_q );
	ldq_addr_valid_6_d <= not ldq_wen_6 and ( ldq_addr_wen_6 or ldq_addr_valid_6_q );
	ldq_data_valid_6_d <= not ldq_wen_6 and ( ldq_data_wen_6 or ldq_data_valid_6_q );
	ldq_alloc_next_7 <= not ldq_reset_7 and ldq_alloc_7_q;
	ldq_alloc_7_d <= ldq_wen_7 or ldq_alloc_next_7;
	ldq_issue_7_d <= not ldq_wen_7 and ( ldq_issue_set_7 or ldq_issue_7_q );
	ldq_addr_valid_7_d <= not ldq_wen_7 and ( ldq_addr_wen_7 or ldq_addr_valid_7_q );
	ldq_data_valid_7_d <= not ldq_wen_7 and ( ldq_data_wen_7 or ldq_data_valid_7_q );
	ldq_alloc_next_8 <= not ldq_reset_8 and ldq_alloc_8_q;
	ldq_alloc_8_d <= ldq_wen_8 or ldq_alloc_next_8;
	ldq_issue_8_d <= not ldq_wen_8 and ( ldq_issue_set_8 or ldq_issue_8_q );
	ldq_addr_valid_8_d <= not ldq_wen_8 and ( ldq_addr_wen_8 or ldq_addr_valid_8_q );
	ldq_data_valid_8_d <= not ldq_wen_8 and ( ldq_data_wen_8 or ldq_data_valid_8_q );
	ldq_alloc_next_9 <= not ldq_reset_9 and ldq_alloc_9_q;
	ldq_alloc_9_d <= ldq_wen_9 or ldq_alloc_next_9;
	ldq_issue_9_d <= not ldq_wen_9 and ( ldq_issue_set_9 or ldq_issue_9_q );
	ldq_addr_valid_9_d <= not ldq_wen_9 and ( ldq_addr_wen_9 or ldq_addr_valid_9_q );
	ldq_data_valid_9_d <= not ldq_wen_9 and ( ldq_data_wen_9 or ldq_data_valid_9_q );
	ldq_alloc_next_10 <= not ldq_reset_10 and ldq_alloc_10_q;
	ldq_alloc_10_d <= ldq_wen_10 or ldq_alloc_next_10;
	ldq_issue_10_d <= not ldq_wen_10 and ( ldq_issue_set_10 or ldq_issue_10_q );
	ldq_addr_valid_10_d <= not ldq_wen_10 and ( ldq_addr_wen_10 or ldq_addr_valid_10_q );
	ldq_data_valid_10_d <= not ldq_wen_10 and ( ldq_data_wen_10 or ldq_data_valid_10_q );
	ldq_alloc_next_11 <= not ldq_reset_11 and ldq_alloc_11_q;
	ldq_alloc_11_d <= ldq_wen_11 or ldq_alloc_next_11;
	ldq_issue_11_d <= not ldq_wen_11 and ( ldq_issue_set_11 or ldq_issue_11_q );
	ldq_addr_valid_11_d <= not ldq_wen_11 and ( ldq_addr_wen_11 or ldq_addr_valid_11_q );
	ldq_data_valid_11_d <= not ldq_wen_11 and ( ldq_data_wen_11 or ldq_data_valid_11_q );
	ldq_alloc_next_12 <= not ldq_reset_12 and ldq_alloc_12_q;
	ldq_alloc_12_d <= ldq_wen_12 or ldq_alloc_next_12;
	ldq_issue_12_d <= not ldq_wen_12 and ( ldq_issue_set_12 or ldq_issue_12_q );
	ldq_addr_valid_12_d <= not ldq_wen_12 and ( ldq_addr_wen_12 or ldq_addr_valid_12_q );
	ldq_data_valid_12_d <= not ldq_wen_12 and ( ldq_data_wen_12 or ldq_data_valid_12_q );
	ldq_alloc_next_13 <= not ldq_reset_13 and ldq_alloc_13_q;
	ldq_alloc_13_d <= ldq_wen_13 or ldq_alloc_next_13;
	ldq_issue_13_d <= not ldq_wen_13 and ( ldq_issue_set_13 or ldq_issue_13_q );
	ldq_addr_valid_13_d <= not ldq_wen_13 and ( ldq_addr_wen_13 or ldq_addr_valid_13_q );
	ldq_data_valid_13_d <= not ldq_wen_13 and ( ldq_data_wen_13 or ldq_data_valid_13_q );
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
	stq_alloc_next_5 <= not stq_reset_5 and stq_alloc_5_q;
	stq_alloc_5_d <= stq_wen_5 or stq_alloc_next_5;
	stq_addr_valid_5_d <= not stq_wen_5 and ( stq_addr_wen_5 or stq_addr_valid_5_q );
	stq_data_valid_5_d <= not stq_wen_5 and ( stq_data_wen_5 or stq_data_valid_5_q );
	stq_alloc_next_6 <= not stq_reset_6 and stq_alloc_6_q;
	stq_alloc_6_d <= stq_wen_6 or stq_alloc_next_6;
	stq_addr_valid_6_d <= not stq_wen_6 and ( stq_addr_wen_6 or stq_addr_valid_6_q );
	stq_data_valid_6_d <= not stq_wen_6 and ( stq_data_wen_6 or stq_data_valid_6_q );
	stq_alloc_next_7 <= not stq_reset_7 and stq_alloc_7_q;
	stq_alloc_7_d <= stq_wen_7 or stq_alloc_next_7;
	stq_addr_valid_7_d <= not stq_wen_7 and ( stq_addr_wen_7 or stq_addr_valid_7_q );
	stq_data_valid_7_d <= not stq_wen_7 and ( stq_data_wen_7 or stq_data_valid_7_q );
	stq_alloc_next_8 <= not stq_reset_8 and stq_alloc_8_q;
	stq_alloc_8_d <= stq_wen_8 or stq_alloc_next_8;
	stq_addr_valid_8_d <= not stq_wen_8 and ( stq_addr_wen_8 or stq_addr_valid_8_q );
	stq_data_valid_8_d <= not stq_wen_8 and ( stq_data_wen_8 or stq_data_valid_8_q );
	stq_alloc_next_9 <= not stq_reset_9 and stq_alloc_9_q;
	stq_alloc_9_d <= stq_wen_9 or stq_alloc_next_9;
	stq_addr_valid_9_d <= not stq_wen_9 and ( stq_addr_wen_9 or stq_addr_valid_9_q );
	stq_data_valid_9_d <= not stq_wen_9 and ( stq_data_wen_9 or stq_data_valid_9_q );
	stq_alloc_next_10 <= not stq_reset_10 and stq_alloc_10_q;
	stq_alloc_10_d <= stq_wen_10 or stq_alloc_next_10;
	stq_addr_valid_10_d <= not stq_wen_10 and ( stq_addr_wen_10 or stq_addr_valid_10_q );
	stq_data_valid_10_d <= not stq_wen_10 and ( stq_data_wen_10 or stq_data_valid_10_q );
	stq_alloc_next_11 <= not stq_reset_11 and stq_alloc_11_q;
	stq_alloc_11_d <= stq_wen_11 or stq_alloc_next_11;
	stq_addr_valid_11_d <= not stq_wen_11 and ( stq_addr_wen_11 or stq_addr_valid_11_q );
	stq_data_valid_11_d <= not stq_wen_11 and ( stq_data_wen_11 or stq_data_valid_11_q );
	stq_alloc_next_12 <= not stq_reset_12 and stq_alloc_12_q;
	stq_alloc_12_d <= stq_wen_12 or stq_alloc_next_12;
	stq_addr_valid_12_d <= not stq_wen_12 and ( stq_addr_wen_12 or stq_addr_valid_12_q );
	stq_data_valid_12_d <= not stq_wen_12 and ( stq_data_wen_12 or stq_data_valid_12_q );
	stq_alloc_next_13 <= not stq_reset_13 and stq_alloc_13_q;
	stq_alloc_13_d <= stq_wen_13 or stq_alloc_next_13;
	stq_addr_valid_13_d <= not stq_wen_13 and ( stq_addr_wen_13 or stq_addr_valid_13_q );
	stq_data_valid_13_d <= not stq_wen_13 and ( stq_data_wen_13 or stq_data_valid_13_q );
	store_is_older_0_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_0(0) ) ) when ldq_wen_0 else not stq_reset_0 and store_is_older_0_q(0);
	store_is_older_0_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_0(1) ) ) when ldq_wen_0 else not stq_reset_1 and store_is_older_0_q(1);
	store_is_older_0_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_0(2) ) ) when ldq_wen_0 else not stq_reset_2 and store_is_older_0_q(2);
	store_is_older_0_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_0(3) ) ) when ldq_wen_0 else not stq_reset_3 and store_is_older_0_q(3);
	store_is_older_0_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_0(4) ) ) when ldq_wen_0 else not stq_reset_4 and store_is_older_0_q(4);
	store_is_older_0_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_0(5) ) ) when ldq_wen_0 else not stq_reset_5 and store_is_older_0_q(5);
	store_is_older_0_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_0(6) ) ) when ldq_wen_0 else not stq_reset_6 and store_is_older_0_q(6);
	store_is_older_0_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_0(7) ) ) when ldq_wen_0 else not stq_reset_7 and store_is_older_0_q(7);
	store_is_older_0_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_0(8) ) ) when ldq_wen_0 else not stq_reset_8 and store_is_older_0_q(8);
	store_is_older_0_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_0(9) ) ) when ldq_wen_0 else not stq_reset_9 and store_is_older_0_q(9);
	store_is_older_0_d(10) <= ( not stq_reset_10 and ( stq_alloc_10_q or ga_ls_order_0(10) ) ) when ldq_wen_0 else not stq_reset_10 and store_is_older_0_q(10);
	store_is_older_0_d(11) <= ( not stq_reset_11 and ( stq_alloc_11_q or ga_ls_order_0(11) ) ) when ldq_wen_0 else not stq_reset_11 and store_is_older_0_q(11);
	store_is_older_0_d(12) <= ( not stq_reset_12 and ( stq_alloc_12_q or ga_ls_order_0(12) ) ) when ldq_wen_0 else not stq_reset_12 and store_is_older_0_q(12);
	store_is_older_0_d(13) <= ( not stq_reset_13 and ( stq_alloc_13_q or ga_ls_order_0(13) ) ) when ldq_wen_0 else not stq_reset_13 and store_is_older_0_q(13);
	store_is_older_1_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_1(0) ) ) when ldq_wen_1 else not stq_reset_0 and store_is_older_1_q(0);
	store_is_older_1_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_1(1) ) ) when ldq_wen_1 else not stq_reset_1 and store_is_older_1_q(1);
	store_is_older_1_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_1(2) ) ) when ldq_wen_1 else not stq_reset_2 and store_is_older_1_q(2);
	store_is_older_1_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_1(3) ) ) when ldq_wen_1 else not stq_reset_3 and store_is_older_1_q(3);
	store_is_older_1_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_1(4) ) ) when ldq_wen_1 else not stq_reset_4 and store_is_older_1_q(4);
	store_is_older_1_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_1(5) ) ) when ldq_wen_1 else not stq_reset_5 and store_is_older_1_q(5);
	store_is_older_1_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_1(6) ) ) when ldq_wen_1 else not stq_reset_6 and store_is_older_1_q(6);
	store_is_older_1_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_1(7) ) ) when ldq_wen_1 else not stq_reset_7 and store_is_older_1_q(7);
	store_is_older_1_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_1(8) ) ) when ldq_wen_1 else not stq_reset_8 and store_is_older_1_q(8);
	store_is_older_1_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_1(9) ) ) when ldq_wen_1 else not stq_reset_9 and store_is_older_1_q(9);
	store_is_older_1_d(10) <= ( not stq_reset_10 and ( stq_alloc_10_q or ga_ls_order_1(10) ) ) when ldq_wen_1 else not stq_reset_10 and store_is_older_1_q(10);
	store_is_older_1_d(11) <= ( not stq_reset_11 and ( stq_alloc_11_q or ga_ls_order_1(11) ) ) when ldq_wen_1 else not stq_reset_11 and store_is_older_1_q(11);
	store_is_older_1_d(12) <= ( not stq_reset_12 and ( stq_alloc_12_q or ga_ls_order_1(12) ) ) when ldq_wen_1 else not stq_reset_12 and store_is_older_1_q(12);
	store_is_older_1_d(13) <= ( not stq_reset_13 and ( stq_alloc_13_q or ga_ls_order_1(13) ) ) when ldq_wen_1 else not stq_reset_13 and store_is_older_1_q(13);
	store_is_older_2_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_2(0) ) ) when ldq_wen_2 else not stq_reset_0 and store_is_older_2_q(0);
	store_is_older_2_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_2(1) ) ) when ldq_wen_2 else not stq_reset_1 and store_is_older_2_q(1);
	store_is_older_2_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_2(2) ) ) when ldq_wen_2 else not stq_reset_2 and store_is_older_2_q(2);
	store_is_older_2_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_2(3) ) ) when ldq_wen_2 else not stq_reset_3 and store_is_older_2_q(3);
	store_is_older_2_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_2(4) ) ) when ldq_wen_2 else not stq_reset_4 and store_is_older_2_q(4);
	store_is_older_2_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_2(5) ) ) when ldq_wen_2 else not stq_reset_5 and store_is_older_2_q(5);
	store_is_older_2_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_2(6) ) ) when ldq_wen_2 else not stq_reset_6 and store_is_older_2_q(6);
	store_is_older_2_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_2(7) ) ) when ldq_wen_2 else not stq_reset_7 and store_is_older_2_q(7);
	store_is_older_2_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_2(8) ) ) when ldq_wen_2 else not stq_reset_8 and store_is_older_2_q(8);
	store_is_older_2_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_2(9) ) ) when ldq_wen_2 else not stq_reset_9 and store_is_older_2_q(9);
	store_is_older_2_d(10) <= ( not stq_reset_10 and ( stq_alloc_10_q or ga_ls_order_2(10) ) ) when ldq_wen_2 else not stq_reset_10 and store_is_older_2_q(10);
	store_is_older_2_d(11) <= ( not stq_reset_11 and ( stq_alloc_11_q or ga_ls_order_2(11) ) ) when ldq_wen_2 else not stq_reset_11 and store_is_older_2_q(11);
	store_is_older_2_d(12) <= ( not stq_reset_12 and ( stq_alloc_12_q or ga_ls_order_2(12) ) ) when ldq_wen_2 else not stq_reset_12 and store_is_older_2_q(12);
	store_is_older_2_d(13) <= ( not stq_reset_13 and ( stq_alloc_13_q or ga_ls_order_2(13) ) ) when ldq_wen_2 else not stq_reset_13 and store_is_older_2_q(13);
	store_is_older_3_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_3(0) ) ) when ldq_wen_3 else not stq_reset_0 and store_is_older_3_q(0);
	store_is_older_3_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_3(1) ) ) when ldq_wen_3 else not stq_reset_1 and store_is_older_3_q(1);
	store_is_older_3_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_3(2) ) ) when ldq_wen_3 else not stq_reset_2 and store_is_older_3_q(2);
	store_is_older_3_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_3(3) ) ) when ldq_wen_3 else not stq_reset_3 and store_is_older_3_q(3);
	store_is_older_3_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_3(4) ) ) when ldq_wen_3 else not stq_reset_4 and store_is_older_3_q(4);
	store_is_older_3_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_3(5) ) ) when ldq_wen_3 else not stq_reset_5 and store_is_older_3_q(5);
	store_is_older_3_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_3(6) ) ) when ldq_wen_3 else not stq_reset_6 and store_is_older_3_q(6);
	store_is_older_3_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_3(7) ) ) when ldq_wen_3 else not stq_reset_7 and store_is_older_3_q(7);
	store_is_older_3_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_3(8) ) ) when ldq_wen_3 else not stq_reset_8 and store_is_older_3_q(8);
	store_is_older_3_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_3(9) ) ) when ldq_wen_3 else not stq_reset_9 and store_is_older_3_q(9);
	store_is_older_3_d(10) <= ( not stq_reset_10 and ( stq_alloc_10_q or ga_ls_order_3(10) ) ) when ldq_wen_3 else not stq_reset_10 and store_is_older_3_q(10);
	store_is_older_3_d(11) <= ( not stq_reset_11 and ( stq_alloc_11_q or ga_ls_order_3(11) ) ) when ldq_wen_3 else not stq_reset_11 and store_is_older_3_q(11);
	store_is_older_3_d(12) <= ( not stq_reset_12 and ( stq_alloc_12_q or ga_ls_order_3(12) ) ) when ldq_wen_3 else not stq_reset_12 and store_is_older_3_q(12);
	store_is_older_3_d(13) <= ( not stq_reset_13 and ( stq_alloc_13_q or ga_ls_order_3(13) ) ) when ldq_wen_3 else not stq_reset_13 and store_is_older_3_q(13);
	store_is_older_4_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_4(0) ) ) when ldq_wen_4 else not stq_reset_0 and store_is_older_4_q(0);
	store_is_older_4_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_4(1) ) ) when ldq_wen_4 else not stq_reset_1 and store_is_older_4_q(1);
	store_is_older_4_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_4(2) ) ) when ldq_wen_4 else not stq_reset_2 and store_is_older_4_q(2);
	store_is_older_4_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_4(3) ) ) when ldq_wen_4 else not stq_reset_3 and store_is_older_4_q(3);
	store_is_older_4_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_4(4) ) ) when ldq_wen_4 else not stq_reset_4 and store_is_older_4_q(4);
	store_is_older_4_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_4(5) ) ) when ldq_wen_4 else not stq_reset_5 and store_is_older_4_q(5);
	store_is_older_4_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_4(6) ) ) when ldq_wen_4 else not stq_reset_6 and store_is_older_4_q(6);
	store_is_older_4_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_4(7) ) ) when ldq_wen_4 else not stq_reset_7 and store_is_older_4_q(7);
	store_is_older_4_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_4(8) ) ) when ldq_wen_4 else not stq_reset_8 and store_is_older_4_q(8);
	store_is_older_4_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_4(9) ) ) when ldq_wen_4 else not stq_reset_9 and store_is_older_4_q(9);
	store_is_older_4_d(10) <= ( not stq_reset_10 and ( stq_alloc_10_q or ga_ls_order_4(10) ) ) when ldq_wen_4 else not stq_reset_10 and store_is_older_4_q(10);
	store_is_older_4_d(11) <= ( not stq_reset_11 and ( stq_alloc_11_q or ga_ls_order_4(11) ) ) when ldq_wen_4 else not stq_reset_11 and store_is_older_4_q(11);
	store_is_older_4_d(12) <= ( not stq_reset_12 and ( stq_alloc_12_q or ga_ls_order_4(12) ) ) when ldq_wen_4 else not stq_reset_12 and store_is_older_4_q(12);
	store_is_older_4_d(13) <= ( not stq_reset_13 and ( stq_alloc_13_q or ga_ls_order_4(13) ) ) when ldq_wen_4 else not stq_reset_13 and store_is_older_4_q(13);
	store_is_older_5_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_5(0) ) ) when ldq_wen_5 else not stq_reset_0 and store_is_older_5_q(0);
	store_is_older_5_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_5(1) ) ) when ldq_wen_5 else not stq_reset_1 and store_is_older_5_q(1);
	store_is_older_5_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_5(2) ) ) when ldq_wen_5 else not stq_reset_2 and store_is_older_5_q(2);
	store_is_older_5_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_5(3) ) ) when ldq_wen_5 else not stq_reset_3 and store_is_older_5_q(3);
	store_is_older_5_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_5(4) ) ) when ldq_wen_5 else not stq_reset_4 and store_is_older_5_q(4);
	store_is_older_5_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_5(5) ) ) when ldq_wen_5 else not stq_reset_5 and store_is_older_5_q(5);
	store_is_older_5_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_5(6) ) ) when ldq_wen_5 else not stq_reset_6 and store_is_older_5_q(6);
	store_is_older_5_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_5(7) ) ) when ldq_wen_5 else not stq_reset_7 and store_is_older_5_q(7);
	store_is_older_5_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_5(8) ) ) when ldq_wen_5 else not stq_reset_8 and store_is_older_5_q(8);
	store_is_older_5_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_5(9) ) ) when ldq_wen_5 else not stq_reset_9 and store_is_older_5_q(9);
	store_is_older_5_d(10) <= ( not stq_reset_10 and ( stq_alloc_10_q or ga_ls_order_5(10) ) ) when ldq_wen_5 else not stq_reset_10 and store_is_older_5_q(10);
	store_is_older_5_d(11) <= ( not stq_reset_11 and ( stq_alloc_11_q or ga_ls_order_5(11) ) ) when ldq_wen_5 else not stq_reset_11 and store_is_older_5_q(11);
	store_is_older_5_d(12) <= ( not stq_reset_12 and ( stq_alloc_12_q or ga_ls_order_5(12) ) ) when ldq_wen_5 else not stq_reset_12 and store_is_older_5_q(12);
	store_is_older_5_d(13) <= ( not stq_reset_13 and ( stq_alloc_13_q or ga_ls_order_5(13) ) ) when ldq_wen_5 else not stq_reset_13 and store_is_older_5_q(13);
	store_is_older_6_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_6(0) ) ) when ldq_wen_6 else not stq_reset_0 and store_is_older_6_q(0);
	store_is_older_6_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_6(1) ) ) when ldq_wen_6 else not stq_reset_1 and store_is_older_6_q(1);
	store_is_older_6_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_6(2) ) ) when ldq_wen_6 else not stq_reset_2 and store_is_older_6_q(2);
	store_is_older_6_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_6(3) ) ) when ldq_wen_6 else not stq_reset_3 and store_is_older_6_q(3);
	store_is_older_6_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_6(4) ) ) when ldq_wen_6 else not stq_reset_4 and store_is_older_6_q(4);
	store_is_older_6_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_6(5) ) ) when ldq_wen_6 else not stq_reset_5 and store_is_older_6_q(5);
	store_is_older_6_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_6(6) ) ) when ldq_wen_6 else not stq_reset_6 and store_is_older_6_q(6);
	store_is_older_6_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_6(7) ) ) when ldq_wen_6 else not stq_reset_7 and store_is_older_6_q(7);
	store_is_older_6_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_6(8) ) ) when ldq_wen_6 else not stq_reset_8 and store_is_older_6_q(8);
	store_is_older_6_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_6(9) ) ) when ldq_wen_6 else not stq_reset_9 and store_is_older_6_q(9);
	store_is_older_6_d(10) <= ( not stq_reset_10 and ( stq_alloc_10_q or ga_ls_order_6(10) ) ) when ldq_wen_6 else not stq_reset_10 and store_is_older_6_q(10);
	store_is_older_6_d(11) <= ( not stq_reset_11 and ( stq_alloc_11_q or ga_ls_order_6(11) ) ) when ldq_wen_6 else not stq_reset_11 and store_is_older_6_q(11);
	store_is_older_6_d(12) <= ( not stq_reset_12 and ( stq_alloc_12_q or ga_ls_order_6(12) ) ) when ldq_wen_6 else not stq_reset_12 and store_is_older_6_q(12);
	store_is_older_6_d(13) <= ( not stq_reset_13 and ( stq_alloc_13_q or ga_ls_order_6(13) ) ) when ldq_wen_6 else not stq_reset_13 and store_is_older_6_q(13);
	store_is_older_7_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_7(0) ) ) when ldq_wen_7 else not stq_reset_0 and store_is_older_7_q(0);
	store_is_older_7_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_7(1) ) ) when ldq_wen_7 else not stq_reset_1 and store_is_older_7_q(1);
	store_is_older_7_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_7(2) ) ) when ldq_wen_7 else not stq_reset_2 and store_is_older_7_q(2);
	store_is_older_7_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_7(3) ) ) when ldq_wen_7 else not stq_reset_3 and store_is_older_7_q(3);
	store_is_older_7_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_7(4) ) ) when ldq_wen_7 else not stq_reset_4 and store_is_older_7_q(4);
	store_is_older_7_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_7(5) ) ) when ldq_wen_7 else not stq_reset_5 and store_is_older_7_q(5);
	store_is_older_7_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_7(6) ) ) when ldq_wen_7 else not stq_reset_6 and store_is_older_7_q(6);
	store_is_older_7_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_7(7) ) ) when ldq_wen_7 else not stq_reset_7 and store_is_older_7_q(7);
	store_is_older_7_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_7(8) ) ) when ldq_wen_7 else not stq_reset_8 and store_is_older_7_q(8);
	store_is_older_7_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_7(9) ) ) when ldq_wen_7 else not stq_reset_9 and store_is_older_7_q(9);
	store_is_older_7_d(10) <= ( not stq_reset_10 and ( stq_alloc_10_q or ga_ls_order_7(10) ) ) when ldq_wen_7 else not stq_reset_10 and store_is_older_7_q(10);
	store_is_older_7_d(11) <= ( not stq_reset_11 and ( stq_alloc_11_q or ga_ls_order_7(11) ) ) when ldq_wen_7 else not stq_reset_11 and store_is_older_7_q(11);
	store_is_older_7_d(12) <= ( not stq_reset_12 and ( stq_alloc_12_q or ga_ls_order_7(12) ) ) when ldq_wen_7 else not stq_reset_12 and store_is_older_7_q(12);
	store_is_older_7_d(13) <= ( not stq_reset_13 and ( stq_alloc_13_q or ga_ls_order_7(13) ) ) when ldq_wen_7 else not stq_reset_13 and store_is_older_7_q(13);
	store_is_older_8_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_8(0) ) ) when ldq_wen_8 else not stq_reset_0 and store_is_older_8_q(0);
	store_is_older_8_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_8(1) ) ) when ldq_wen_8 else not stq_reset_1 and store_is_older_8_q(1);
	store_is_older_8_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_8(2) ) ) when ldq_wen_8 else not stq_reset_2 and store_is_older_8_q(2);
	store_is_older_8_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_8(3) ) ) when ldq_wen_8 else not stq_reset_3 and store_is_older_8_q(3);
	store_is_older_8_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_8(4) ) ) when ldq_wen_8 else not stq_reset_4 and store_is_older_8_q(4);
	store_is_older_8_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_8(5) ) ) when ldq_wen_8 else not stq_reset_5 and store_is_older_8_q(5);
	store_is_older_8_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_8(6) ) ) when ldq_wen_8 else not stq_reset_6 and store_is_older_8_q(6);
	store_is_older_8_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_8(7) ) ) when ldq_wen_8 else not stq_reset_7 and store_is_older_8_q(7);
	store_is_older_8_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_8(8) ) ) when ldq_wen_8 else not stq_reset_8 and store_is_older_8_q(8);
	store_is_older_8_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_8(9) ) ) when ldq_wen_8 else not stq_reset_9 and store_is_older_8_q(9);
	store_is_older_8_d(10) <= ( not stq_reset_10 and ( stq_alloc_10_q or ga_ls_order_8(10) ) ) when ldq_wen_8 else not stq_reset_10 and store_is_older_8_q(10);
	store_is_older_8_d(11) <= ( not stq_reset_11 and ( stq_alloc_11_q or ga_ls_order_8(11) ) ) when ldq_wen_8 else not stq_reset_11 and store_is_older_8_q(11);
	store_is_older_8_d(12) <= ( not stq_reset_12 and ( stq_alloc_12_q or ga_ls_order_8(12) ) ) when ldq_wen_8 else not stq_reset_12 and store_is_older_8_q(12);
	store_is_older_8_d(13) <= ( not stq_reset_13 and ( stq_alloc_13_q or ga_ls_order_8(13) ) ) when ldq_wen_8 else not stq_reset_13 and store_is_older_8_q(13);
	store_is_older_9_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_9(0) ) ) when ldq_wen_9 else not stq_reset_0 and store_is_older_9_q(0);
	store_is_older_9_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_9(1) ) ) when ldq_wen_9 else not stq_reset_1 and store_is_older_9_q(1);
	store_is_older_9_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_9(2) ) ) when ldq_wen_9 else not stq_reset_2 and store_is_older_9_q(2);
	store_is_older_9_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_9(3) ) ) when ldq_wen_9 else not stq_reset_3 and store_is_older_9_q(3);
	store_is_older_9_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_9(4) ) ) when ldq_wen_9 else not stq_reset_4 and store_is_older_9_q(4);
	store_is_older_9_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_9(5) ) ) when ldq_wen_9 else not stq_reset_5 and store_is_older_9_q(5);
	store_is_older_9_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_9(6) ) ) when ldq_wen_9 else not stq_reset_6 and store_is_older_9_q(6);
	store_is_older_9_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_9(7) ) ) when ldq_wen_9 else not stq_reset_7 and store_is_older_9_q(7);
	store_is_older_9_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_9(8) ) ) when ldq_wen_9 else not stq_reset_8 and store_is_older_9_q(8);
	store_is_older_9_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_9(9) ) ) when ldq_wen_9 else not stq_reset_9 and store_is_older_9_q(9);
	store_is_older_9_d(10) <= ( not stq_reset_10 and ( stq_alloc_10_q or ga_ls_order_9(10) ) ) when ldq_wen_9 else not stq_reset_10 and store_is_older_9_q(10);
	store_is_older_9_d(11) <= ( not stq_reset_11 and ( stq_alloc_11_q or ga_ls_order_9(11) ) ) when ldq_wen_9 else not stq_reset_11 and store_is_older_9_q(11);
	store_is_older_9_d(12) <= ( not stq_reset_12 and ( stq_alloc_12_q or ga_ls_order_9(12) ) ) when ldq_wen_9 else not stq_reset_12 and store_is_older_9_q(12);
	store_is_older_9_d(13) <= ( not stq_reset_13 and ( stq_alloc_13_q or ga_ls_order_9(13) ) ) when ldq_wen_9 else not stq_reset_13 and store_is_older_9_q(13);
	store_is_older_10_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_10(0) ) ) when ldq_wen_10 else not stq_reset_0 and store_is_older_10_q(0);
	store_is_older_10_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_10(1) ) ) when ldq_wen_10 else not stq_reset_1 and store_is_older_10_q(1);
	store_is_older_10_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_10(2) ) ) when ldq_wen_10 else not stq_reset_2 and store_is_older_10_q(2);
	store_is_older_10_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_10(3) ) ) when ldq_wen_10 else not stq_reset_3 and store_is_older_10_q(3);
	store_is_older_10_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_10(4) ) ) when ldq_wen_10 else not stq_reset_4 and store_is_older_10_q(4);
	store_is_older_10_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_10(5) ) ) when ldq_wen_10 else not stq_reset_5 and store_is_older_10_q(5);
	store_is_older_10_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_10(6) ) ) when ldq_wen_10 else not stq_reset_6 and store_is_older_10_q(6);
	store_is_older_10_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_10(7) ) ) when ldq_wen_10 else not stq_reset_7 and store_is_older_10_q(7);
	store_is_older_10_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_10(8) ) ) when ldq_wen_10 else not stq_reset_8 and store_is_older_10_q(8);
	store_is_older_10_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_10(9) ) ) when ldq_wen_10 else not stq_reset_9 and store_is_older_10_q(9);
	store_is_older_10_d(10) <= ( not stq_reset_10 and ( stq_alloc_10_q or ga_ls_order_10(10) ) ) when ldq_wen_10 else not stq_reset_10 and store_is_older_10_q(10);
	store_is_older_10_d(11) <= ( not stq_reset_11 and ( stq_alloc_11_q or ga_ls_order_10(11) ) ) when ldq_wen_10 else not stq_reset_11 and store_is_older_10_q(11);
	store_is_older_10_d(12) <= ( not stq_reset_12 and ( stq_alloc_12_q or ga_ls_order_10(12) ) ) when ldq_wen_10 else not stq_reset_12 and store_is_older_10_q(12);
	store_is_older_10_d(13) <= ( not stq_reset_13 and ( stq_alloc_13_q or ga_ls_order_10(13) ) ) when ldq_wen_10 else not stq_reset_13 and store_is_older_10_q(13);
	store_is_older_11_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_11(0) ) ) when ldq_wen_11 else not stq_reset_0 and store_is_older_11_q(0);
	store_is_older_11_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_11(1) ) ) when ldq_wen_11 else not stq_reset_1 and store_is_older_11_q(1);
	store_is_older_11_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_11(2) ) ) when ldq_wen_11 else not stq_reset_2 and store_is_older_11_q(2);
	store_is_older_11_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_11(3) ) ) when ldq_wen_11 else not stq_reset_3 and store_is_older_11_q(3);
	store_is_older_11_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_11(4) ) ) when ldq_wen_11 else not stq_reset_4 and store_is_older_11_q(4);
	store_is_older_11_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_11(5) ) ) when ldq_wen_11 else not stq_reset_5 and store_is_older_11_q(5);
	store_is_older_11_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_11(6) ) ) when ldq_wen_11 else not stq_reset_6 and store_is_older_11_q(6);
	store_is_older_11_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_11(7) ) ) when ldq_wen_11 else not stq_reset_7 and store_is_older_11_q(7);
	store_is_older_11_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_11(8) ) ) when ldq_wen_11 else not stq_reset_8 and store_is_older_11_q(8);
	store_is_older_11_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_11(9) ) ) when ldq_wen_11 else not stq_reset_9 and store_is_older_11_q(9);
	store_is_older_11_d(10) <= ( not stq_reset_10 and ( stq_alloc_10_q or ga_ls_order_11(10) ) ) when ldq_wen_11 else not stq_reset_10 and store_is_older_11_q(10);
	store_is_older_11_d(11) <= ( not stq_reset_11 and ( stq_alloc_11_q or ga_ls_order_11(11) ) ) when ldq_wen_11 else not stq_reset_11 and store_is_older_11_q(11);
	store_is_older_11_d(12) <= ( not stq_reset_12 and ( stq_alloc_12_q or ga_ls_order_11(12) ) ) when ldq_wen_11 else not stq_reset_12 and store_is_older_11_q(12);
	store_is_older_11_d(13) <= ( not stq_reset_13 and ( stq_alloc_13_q or ga_ls_order_11(13) ) ) when ldq_wen_11 else not stq_reset_13 and store_is_older_11_q(13);
	store_is_older_12_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_12(0) ) ) when ldq_wen_12 else not stq_reset_0 and store_is_older_12_q(0);
	store_is_older_12_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_12(1) ) ) when ldq_wen_12 else not stq_reset_1 and store_is_older_12_q(1);
	store_is_older_12_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_12(2) ) ) when ldq_wen_12 else not stq_reset_2 and store_is_older_12_q(2);
	store_is_older_12_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_12(3) ) ) when ldq_wen_12 else not stq_reset_3 and store_is_older_12_q(3);
	store_is_older_12_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_12(4) ) ) when ldq_wen_12 else not stq_reset_4 and store_is_older_12_q(4);
	store_is_older_12_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_12(5) ) ) when ldq_wen_12 else not stq_reset_5 and store_is_older_12_q(5);
	store_is_older_12_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_12(6) ) ) when ldq_wen_12 else not stq_reset_6 and store_is_older_12_q(6);
	store_is_older_12_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_12(7) ) ) when ldq_wen_12 else not stq_reset_7 and store_is_older_12_q(7);
	store_is_older_12_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_12(8) ) ) when ldq_wen_12 else not stq_reset_8 and store_is_older_12_q(8);
	store_is_older_12_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_12(9) ) ) when ldq_wen_12 else not stq_reset_9 and store_is_older_12_q(9);
	store_is_older_12_d(10) <= ( not stq_reset_10 and ( stq_alloc_10_q or ga_ls_order_12(10) ) ) when ldq_wen_12 else not stq_reset_10 and store_is_older_12_q(10);
	store_is_older_12_d(11) <= ( not stq_reset_11 and ( stq_alloc_11_q or ga_ls_order_12(11) ) ) when ldq_wen_12 else not stq_reset_11 and store_is_older_12_q(11);
	store_is_older_12_d(12) <= ( not stq_reset_12 and ( stq_alloc_12_q or ga_ls_order_12(12) ) ) when ldq_wen_12 else not stq_reset_12 and store_is_older_12_q(12);
	store_is_older_12_d(13) <= ( not stq_reset_13 and ( stq_alloc_13_q or ga_ls_order_12(13) ) ) when ldq_wen_12 else not stq_reset_13 and store_is_older_12_q(13);
	store_is_older_13_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_13(0) ) ) when ldq_wen_13 else not stq_reset_0 and store_is_older_13_q(0);
	store_is_older_13_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_13(1) ) ) when ldq_wen_13 else not stq_reset_1 and store_is_older_13_q(1);
	store_is_older_13_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_13(2) ) ) when ldq_wen_13 else not stq_reset_2 and store_is_older_13_q(2);
	store_is_older_13_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_13(3) ) ) when ldq_wen_13 else not stq_reset_3 and store_is_older_13_q(3);
	store_is_older_13_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_13(4) ) ) when ldq_wen_13 else not stq_reset_4 and store_is_older_13_q(4);
	store_is_older_13_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_13(5) ) ) when ldq_wen_13 else not stq_reset_5 and store_is_older_13_q(5);
	store_is_older_13_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_13(6) ) ) when ldq_wen_13 else not stq_reset_6 and store_is_older_13_q(6);
	store_is_older_13_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_13(7) ) ) when ldq_wen_13 else not stq_reset_7 and store_is_older_13_q(7);
	store_is_older_13_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_13(8) ) ) when ldq_wen_13 else not stq_reset_8 and store_is_older_13_q(8);
	store_is_older_13_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_13(9) ) ) when ldq_wen_13 else not stq_reset_9 and store_is_older_13_q(9);
	store_is_older_13_d(10) <= ( not stq_reset_10 and ( stq_alloc_10_q or ga_ls_order_13(10) ) ) when ldq_wen_13 else not stq_reset_10 and store_is_older_13_q(10);
	store_is_older_13_d(11) <= ( not stq_reset_11 and ( stq_alloc_11_q or ga_ls_order_13(11) ) ) when ldq_wen_13 else not stq_reset_11 and store_is_older_13_q(11);
	store_is_older_13_d(12) <= ( not stq_reset_12 and ( stq_alloc_12_q or ga_ls_order_13(12) ) ) when ldq_wen_13 else not stq_reset_12 and store_is_older_13_q(12);
	store_is_older_13_d(13) <= ( not stq_reset_13 and ( stq_alloc_13_q or ga_ls_order_13(13) ) ) when ldq_wen_13 else not stq_reset_13 and store_is_older_13_q(13);
	-- Reduction Begin
	-- Reduce(ldq_not_empty, ldq_alloc, or)
	TEMP_1_res_0 <= ldq_alloc_0_q or ldq_alloc_8_q;
	TEMP_1_res_1 <= ldq_alloc_1_q or ldq_alloc_9_q;
	TEMP_1_res_2 <= ldq_alloc_2_q or ldq_alloc_10_q;
	TEMP_1_res_3 <= ldq_alloc_3_q or ldq_alloc_11_q;
	TEMP_1_res_4 <= ldq_alloc_4_q or ldq_alloc_12_q;
	TEMP_1_res_5 <= ldq_alloc_5_q or ldq_alloc_13_q;
	TEMP_1_res_6 <= ldq_alloc_6_q;
	TEMP_1_res_7 <= ldq_alloc_7_q;
	-- Layer End
	TEMP_2_res_0 <= TEMP_1_res_0 or TEMP_1_res_4;
	TEMP_2_res_1 <= TEMP_1_res_1 or TEMP_1_res_5;
	TEMP_2_res_2 <= TEMP_1_res_2 or TEMP_1_res_6;
	TEMP_2_res_3 <= TEMP_1_res_3 or TEMP_1_res_7;
	-- Layer End
	TEMP_3_res_0 <= TEMP_2_res_0 or TEMP_2_res_2;
	TEMP_3_res_1 <= TEMP_2_res_1 or TEMP_2_res_3;
	-- Layer End
	ldq_not_empty <= TEMP_3_res_0 or TEMP_3_res_1;
	-- Reduction End

	ldq_empty <= not ldq_not_empty;
	-- MuxLookUp Begin
	-- MuxLookUp(stq_not_empty, stq_alloc, stq_head)
	stq_not_empty <= 
	stq_alloc_0_q when (stq_head_q = "0000") else
	stq_alloc_1_q when (stq_head_q = "0001") else
	stq_alloc_2_q when (stq_head_q = "0010") else
	stq_alloc_3_q when (stq_head_q = "0011") else
	stq_alloc_4_q when (stq_head_q = "0100") else
	stq_alloc_5_q when (stq_head_q = "0101") else
	stq_alloc_6_q when (stq_head_q = "0110") else
	stq_alloc_7_q when (stq_head_q = "0111") else
	stq_alloc_8_q when (stq_head_q = "1000") else
	stq_alloc_9_q when (stq_head_q = "1001") else
	stq_alloc_10_q when (stq_head_q = "1010") else
	stq_alloc_11_q when (stq_head_q = "1011") else
	stq_alloc_12_q when (stq_head_q = "1100") else
	stq_alloc_13_q when (stq_head_q = "1101") else
	'0';
	-- MuxLookUp End

	stq_empty <= not stq_not_empty;
	empty_o <= ldq_empty and stq_empty;
	-- WrapAdd Begin
	-- WrapAdd(ldq_tail, ldq_tail, num_loads, 14)
	TEMP_4_sum <= std_logic_vector(unsigned('0' & ldq_tail_q) + unsigned('0' & num_loads));
	TEMP_4_res <= std_logic_vector(unsigned(TEMP_4_sum) - 14) when unsigned(TEMP_4_sum) >= 14 else TEMP_4_sum;
	ldq_tail_d <= TEMP_4_res(3 downto 0);
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_tail, stq_tail, num_stores, 14)
	TEMP_5_sum <= std_logic_vector(unsigned('0' & stq_tail_q) + unsigned('0' & num_stores));
	TEMP_5_res <= std_logic_vector(unsigned(TEMP_5_sum) - 14) when unsigned(TEMP_5_sum) >= 14 else TEMP_5_sum;
	stq_tail_d <= TEMP_5_res(3 downto 0);
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_issue, stq_issue, 1, 14)
	stq_issue_d <= std_logic_vector(unsigned(stq_issue_q) - 13) when unsigned(stq_issue_q) >= 13 else std_logic_vector(unsigned(stq_issue_q) + 1);
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_resp, stq_resp, 1, 14)
	stq_resp_d <= std_logic_vector(unsigned(stq_resp_q) - 13) when unsigned(stq_resp_q) >= 13 else std_logic_vector(unsigned(stq_resp_q) + 1);
	-- WrapAdd End

	-- Bits To One-Hot Begin
	-- BitsToOH(ldq_tail_oh, ldq_tail)
	ldq_tail_oh(0) <= '1' when ldq_tail_q = "0000" else '0';
	ldq_tail_oh(1) <= '1' when ldq_tail_q = "0001" else '0';
	ldq_tail_oh(2) <= '1' when ldq_tail_q = "0010" else '0';
	ldq_tail_oh(3) <= '1' when ldq_tail_q = "0011" else '0';
	ldq_tail_oh(4) <= '1' when ldq_tail_q = "0100" else '0';
	ldq_tail_oh(5) <= '1' when ldq_tail_q = "0101" else '0';
	ldq_tail_oh(6) <= '1' when ldq_tail_q = "0110" else '0';
	ldq_tail_oh(7) <= '1' when ldq_tail_q = "0111" else '0';
	ldq_tail_oh(8) <= '1' when ldq_tail_q = "1000" else '0';
	ldq_tail_oh(9) <= '1' when ldq_tail_q = "1001" else '0';
	ldq_tail_oh(10) <= '1' when ldq_tail_q = "1010" else '0';
	ldq_tail_oh(11) <= '1' when ldq_tail_q = "1011" else '0';
	ldq_tail_oh(12) <= '1' when ldq_tail_q = "1100" else '0';
	ldq_tail_oh(13) <= '1' when ldq_tail_q = "1101" else '0';
	-- Bits To One-Hot End

	-- Priority Masking Begin
	-- CyclicPriorityMask(ldq_head_next_oh, ldq_alloc_next, ldq_tail_oh)
	TEMP_6_double_in(0) <= ldq_alloc_next_0;
	TEMP_6_double_in(14) <= ldq_alloc_next_0;
	TEMP_6_double_in(1) <= ldq_alloc_next_1;
	TEMP_6_double_in(15) <= ldq_alloc_next_1;
	TEMP_6_double_in(2) <= ldq_alloc_next_2;
	TEMP_6_double_in(16) <= ldq_alloc_next_2;
	TEMP_6_double_in(3) <= ldq_alloc_next_3;
	TEMP_6_double_in(17) <= ldq_alloc_next_3;
	TEMP_6_double_in(4) <= ldq_alloc_next_4;
	TEMP_6_double_in(18) <= ldq_alloc_next_4;
	TEMP_6_double_in(5) <= ldq_alloc_next_5;
	TEMP_6_double_in(19) <= ldq_alloc_next_5;
	TEMP_6_double_in(6) <= ldq_alloc_next_6;
	TEMP_6_double_in(20) <= ldq_alloc_next_6;
	TEMP_6_double_in(7) <= ldq_alloc_next_7;
	TEMP_6_double_in(21) <= ldq_alloc_next_7;
	TEMP_6_double_in(8) <= ldq_alloc_next_8;
	TEMP_6_double_in(22) <= ldq_alloc_next_8;
	TEMP_6_double_in(9) <= ldq_alloc_next_9;
	TEMP_6_double_in(23) <= ldq_alloc_next_9;
	TEMP_6_double_in(10) <= ldq_alloc_next_10;
	TEMP_6_double_in(24) <= ldq_alloc_next_10;
	TEMP_6_double_in(11) <= ldq_alloc_next_11;
	TEMP_6_double_in(25) <= ldq_alloc_next_11;
	TEMP_6_double_in(12) <= ldq_alloc_next_12;
	TEMP_6_double_in(26) <= ldq_alloc_next_12;
	TEMP_6_double_in(13) <= ldq_alloc_next_13;
	TEMP_6_double_in(27) <= ldq_alloc_next_13;
	TEMP_6_double_out <= TEMP_6_double_in and not std_logic_vector( unsigned( TEMP_6_double_in ) - unsigned( "00000000000000" & ldq_tail_oh ) );
	ldq_head_next_oh <= TEMP_6_double_out(13 downto 0) or TEMP_6_double_out(27 downto 14);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(ldq_head_sel, ldq_alloc_next, or)
	TEMP_7_res_0 <= ldq_alloc_next_0 or ldq_alloc_next_8;
	TEMP_7_res_1 <= ldq_alloc_next_1 or ldq_alloc_next_9;
	TEMP_7_res_2 <= ldq_alloc_next_2 or ldq_alloc_next_10;
	TEMP_7_res_3 <= ldq_alloc_next_3 or ldq_alloc_next_11;
	TEMP_7_res_4 <= ldq_alloc_next_4 or ldq_alloc_next_12;
	TEMP_7_res_5 <= ldq_alloc_next_5 or ldq_alloc_next_13;
	TEMP_7_res_6 <= ldq_alloc_next_6;
	TEMP_7_res_7 <= ldq_alloc_next_7;
	-- Layer End
	TEMP_8_res_0 <= TEMP_7_res_0 or TEMP_7_res_4;
	TEMP_8_res_1 <= TEMP_7_res_1 or TEMP_7_res_5;
	TEMP_8_res_2 <= TEMP_7_res_2 or TEMP_7_res_6;
	TEMP_8_res_3 <= TEMP_7_res_3 or TEMP_7_res_7;
	-- Layer End
	TEMP_9_res_0 <= TEMP_8_res_0 or TEMP_8_res_2;
	TEMP_9_res_1 <= TEMP_8_res_1 or TEMP_8_res_3;
	-- Layer End
	ldq_head_sel <= TEMP_9_res_0 or TEMP_9_res_1;
	-- Reduction End

	-- One-Hot To Bits Begin
	-- OHToBits(ldq_head_next, ldq_head_next_oh)
	TEMP_10_in_0_0 <= '0';
	TEMP_10_in_0_1 <= ldq_head_next_oh(1);
	TEMP_10_in_0_2 <= '0';
	TEMP_10_in_0_3 <= ldq_head_next_oh(3);
	TEMP_10_in_0_4 <= '0';
	TEMP_10_in_0_5 <= ldq_head_next_oh(5);
	TEMP_10_in_0_6 <= '0';
	TEMP_10_in_0_7 <= ldq_head_next_oh(7);
	TEMP_10_in_0_8 <= '0';
	TEMP_10_in_0_9 <= ldq_head_next_oh(9);
	TEMP_10_in_0_10 <= '0';
	TEMP_10_in_0_11 <= ldq_head_next_oh(11);
	TEMP_10_in_0_12 <= '0';
	TEMP_10_in_0_13 <= ldq_head_next_oh(13);
	TEMP_11_res_0 <= TEMP_10_in_0_0 or TEMP_10_in_0_8;
	TEMP_11_res_1 <= TEMP_10_in_0_1 or TEMP_10_in_0_9;
	TEMP_11_res_2 <= TEMP_10_in_0_2 or TEMP_10_in_0_10;
	TEMP_11_res_3 <= TEMP_10_in_0_3 or TEMP_10_in_0_11;
	TEMP_11_res_4 <= TEMP_10_in_0_4 or TEMP_10_in_0_12;
	TEMP_11_res_5 <= TEMP_10_in_0_5 or TEMP_10_in_0_13;
	TEMP_11_res_6 <= TEMP_10_in_0_6;
	TEMP_11_res_7 <= TEMP_10_in_0_7;
	-- Layer End
	TEMP_12_res_0 <= TEMP_11_res_0 or TEMP_11_res_4;
	TEMP_12_res_1 <= TEMP_11_res_1 or TEMP_11_res_5;
	TEMP_12_res_2 <= TEMP_11_res_2 or TEMP_11_res_6;
	TEMP_12_res_3 <= TEMP_11_res_3 or TEMP_11_res_7;
	-- Layer End
	TEMP_13_res_0 <= TEMP_12_res_0 or TEMP_12_res_2;
	TEMP_13_res_1 <= TEMP_12_res_1 or TEMP_12_res_3;
	-- Layer End
	TEMP_10_out_0 <= TEMP_13_res_0 or TEMP_13_res_1;
	ldq_head_next(0) <= TEMP_10_out_0;
	TEMP_13_in_1_0 <= '0';
	TEMP_13_in_1_1 <= '0';
	TEMP_13_in_1_2 <= ldq_head_next_oh(2);
	TEMP_13_in_1_3 <= ldq_head_next_oh(3);
	TEMP_13_in_1_4 <= '0';
	TEMP_13_in_1_5 <= '0';
	TEMP_13_in_1_6 <= ldq_head_next_oh(6);
	TEMP_13_in_1_7 <= ldq_head_next_oh(7);
	TEMP_13_in_1_8 <= '0';
	TEMP_13_in_1_9 <= '0';
	TEMP_13_in_1_10 <= ldq_head_next_oh(10);
	TEMP_13_in_1_11 <= ldq_head_next_oh(11);
	TEMP_13_in_1_12 <= '0';
	TEMP_13_in_1_13 <= '0';
	TEMP_14_res_0 <= TEMP_13_in_1_0 or TEMP_13_in_1_8;
	TEMP_14_res_1 <= TEMP_13_in_1_1 or TEMP_13_in_1_9;
	TEMP_14_res_2 <= TEMP_13_in_1_2 or TEMP_13_in_1_10;
	TEMP_14_res_3 <= TEMP_13_in_1_3 or TEMP_13_in_1_11;
	TEMP_14_res_4 <= TEMP_13_in_1_4 or TEMP_13_in_1_12;
	TEMP_14_res_5 <= TEMP_13_in_1_5 or TEMP_13_in_1_13;
	TEMP_14_res_6 <= TEMP_13_in_1_6;
	TEMP_14_res_7 <= TEMP_13_in_1_7;
	-- Layer End
	TEMP_15_res_0 <= TEMP_14_res_0 or TEMP_14_res_4;
	TEMP_15_res_1 <= TEMP_14_res_1 or TEMP_14_res_5;
	TEMP_15_res_2 <= TEMP_14_res_2 or TEMP_14_res_6;
	TEMP_15_res_3 <= TEMP_14_res_3 or TEMP_14_res_7;
	-- Layer End
	TEMP_16_res_0 <= TEMP_15_res_0 or TEMP_15_res_2;
	TEMP_16_res_1 <= TEMP_15_res_1 or TEMP_15_res_3;
	-- Layer End
	TEMP_13_out_1 <= TEMP_16_res_0 or TEMP_16_res_1;
	ldq_head_next(1) <= TEMP_13_out_1;
	TEMP_16_in_2_0 <= '0';
	TEMP_16_in_2_1 <= '0';
	TEMP_16_in_2_2 <= '0';
	TEMP_16_in_2_3 <= '0';
	TEMP_16_in_2_4 <= ldq_head_next_oh(4);
	TEMP_16_in_2_5 <= ldq_head_next_oh(5);
	TEMP_16_in_2_6 <= ldq_head_next_oh(6);
	TEMP_16_in_2_7 <= ldq_head_next_oh(7);
	TEMP_16_in_2_8 <= '0';
	TEMP_16_in_2_9 <= '0';
	TEMP_16_in_2_10 <= '0';
	TEMP_16_in_2_11 <= '0';
	TEMP_16_in_2_12 <= ldq_head_next_oh(12);
	TEMP_16_in_2_13 <= ldq_head_next_oh(13);
	TEMP_17_res_0 <= TEMP_16_in_2_0 or TEMP_16_in_2_8;
	TEMP_17_res_1 <= TEMP_16_in_2_1 or TEMP_16_in_2_9;
	TEMP_17_res_2 <= TEMP_16_in_2_2 or TEMP_16_in_2_10;
	TEMP_17_res_3 <= TEMP_16_in_2_3 or TEMP_16_in_2_11;
	TEMP_17_res_4 <= TEMP_16_in_2_4 or TEMP_16_in_2_12;
	TEMP_17_res_5 <= TEMP_16_in_2_5 or TEMP_16_in_2_13;
	TEMP_17_res_6 <= TEMP_16_in_2_6;
	TEMP_17_res_7 <= TEMP_16_in_2_7;
	-- Layer End
	TEMP_18_res_0 <= TEMP_17_res_0 or TEMP_17_res_4;
	TEMP_18_res_1 <= TEMP_17_res_1 or TEMP_17_res_5;
	TEMP_18_res_2 <= TEMP_17_res_2 or TEMP_17_res_6;
	TEMP_18_res_3 <= TEMP_17_res_3 or TEMP_17_res_7;
	-- Layer End
	TEMP_19_res_0 <= TEMP_18_res_0 or TEMP_18_res_2;
	TEMP_19_res_1 <= TEMP_18_res_1 or TEMP_18_res_3;
	-- Layer End
	TEMP_16_out_2 <= TEMP_19_res_0 or TEMP_19_res_1;
	ldq_head_next(2) <= TEMP_16_out_2;
	TEMP_19_in_3_0 <= '0';
	TEMP_19_in_3_1 <= '0';
	TEMP_19_in_3_2 <= '0';
	TEMP_19_in_3_3 <= '0';
	TEMP_19_in_3_4 <= '0';
	TEMP_19_in_3_5 <= '0';
	TEMP_19_in_3_6 <= '0';
	TEMP_19_in_3_7 <= '0';
	TEMP_19_in_3_8 <= ldq_head_next_oh(8);
	TEMP_19_in_3_9 <= ldq_head_next_oh(9);
	TEMP_19_in_3_10 <= ldq_head_next_oh(10);
	TEMP_19_in_3_11 <= ldq_head_next_oh(11);
	TEMP_19_in_3_12 <= ldq_head_next_oh(12);
	TEMP_19_in_3_13 <= ldq_head_next_oh(13);
	TEMP_20_res_0 <= TEMP_19_in_3_0 or TEMP_19_in_3_8;
	TEMP_20_res_1 <= TEMP_19_in_3_1 or TEMP_19_in_3_9;
	TEMP_20_res_2 <= TEMP_19_in_3_2 or TEMP_19_in_3_10;
	TEMP_20_res_3 <= TEMP_19_in_3_3 or TEMP_19_in_3_11;
	TEMP_20_res_4 <= TEMP_19_in_3_4 or TEMP_19_in_3_12;
	TEMP_20_res_5 <= TEMP_19_in_3_5 or TEMP_19_in_3_13;
	TEMP_20_res_6 <= TEMP_19_in_3_6;
	TEMP_20_res_7 <= TEMP_19_in_3_7;
	-- Layer End
	TEMP_21_res_0 <= TEMP_20_res_0 or TEMP_20_res_4;
	TEMP_21_res_1 <= TEMP_20_res_1 or TEMP_20_res_5;
	TEMP_21_res_2 <= TEMP_20_res_2 or TEMP_20_res_6;
	TEMP_21_res_3 <= TEMP_20_res_3 or TEMP_20_res_7;
	-- Layer End
	TEMP_22_res_0 <= TEMP_21_res_0 or TEMP_21_res_2;
	TEMP_22_res_1 <= TEMP_21_res_1 or TEMP_21_res_3;
	-- Layer End
	TEMP_19_out_3 <= TEMP_22_res_0 or TEMP_22_res_1;
	ldq_head_next(3) <= TEMP_19_out_3;
	-- One-Hot To Bits End

	ldq_head_d <= ldq_head_next when ldq_head_sel else ldq_tail_q;
	-- Bits To One-Hot Begin
	-- BitsToOH(stq_tail_oh, stq_tail)
	stq_tail_oh(0) <= '1' when stq_tail_q = "0000" else '0';
	stq_tail_oh(1) <= '1' when stq_tail_q = "0001" else '0';
	stq_tail_oh(2) <= '1' when stq_tail_q = "0010" else '0';
	stq_tail_oh(3) <= '1' when stq_tail_q = "0011" else '0';
	stq_tail_oh(4) <= '1' when stq_tail_q = "0100" else '0';
	stq_tail_oh(5) <= '1' when stq_tail_q = "0101" else '0';
	stq_tail_oh(6) <= '1' when stq_tail_q = "0110" else '0';
	stq_tail_oh(7) <= '1' when stq_tail_q = "0111" else '0';
	stq_tail_oh(8) <= '1' when stq_tail_q = "1000" else '0';
	stq_tail_oh(9) <= '1' when stq_tail_q = "1001" else '0';
	stq_tail_oh(10) <= '1' when stq_tail_q = "1010" else '0';
	stq_tail_oh(11) <= '1' when stq_tail_q = "1011" else '0';
	stq_tail_oh(12) <= '1' when stq_tail_q = "1100" else '0';
	stq_tail_oh(13) <= '1' when stq_tail_q = "1101" else '0';
	-- Bits To One-Hot End

	-- WrapAdd Begin
	-- WrapAdd(stq_head_next, stq_head, 1, 14)
	stq_head_next <= std_logic_vector(unsigned(stq_head_q) - 13) when unsigned(stq_head_q) >= 13 else std_logic_vector(unsigned(stq_head_q) + 1);
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
			ldq_wen_4_o => ldq_wen_4,
			ldq_wen_5_o => ldq_wen_5,
			ldq_wen_6_o => ldq_wen_6,
			ldq_wen_7_o => ldq_wen_7,
			ldq_wen_8_o => ldq_wen_8,
			ldq_wen_9_o => ldq_wen_9,
			ldq_wen_10_o => ldq_wen_10,
			ldq_wen_11_o => ldq_wen_11,
			ldq_wen_12_o => ldq_wen_12,
			ldq_wen_13_o => ldq_wen_13,
			num_loads_o => num_loads,
			stq_wen_0_o => stq_wen_0,
			stq_wen_1_o => stq_wen_1,
			stq_wen_2_o => stq_wen_2,
			stq_wen_3_o => stq_wen_3,
			stq_wen_4_o => stq_wen_4,
			stq_wen_5_o => stq_wen_5,
			stq_wen_6_o => stq_wen_6,
			stq_wen_7_o => stq_wen_7,
			stq_wen_8_o => stq_wen_8,
			stq_wen_9_o => stq_wen_9,
			stq_wen_10_o => stq_wen_10,
			stq_wen_11_o => stq_wen_11,
			stq_wen_12_o => stq_wen_12,
			stq_wen_13_o => stq_wen_13,
			ga_ls_order_0_o => ga_ls_order_0,
			ga_ls_order_1_o => ga_ls_order_1,
			ga_ls_order_2_o => ga_ls_order_2,
			ga_ls_order_3_o => ga_ls_order_3,
			ga_ls_order_4_o => ga_ls_order_4,
			ga_ls_order_5_o => ga_ls_order_5,
			ga_ls_order_6_o => ga_ls_order_6,
			ga_ls_order_7_o => ga_ls_order_7,
			ga_ls_order_8_o => ga_ls_order_8,
			ga_ls_order_9_o => ga_ls_order_9,
			ga_ls_order_10_o => ga_ls_order_10,
			ga_ls_order_11_o => ga_ls_order_11,
			ga_ls_order_12_o => ga_ls_order_12,
			ga_ls_order_13_o => ga_ls_order_13,
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
			entry_alloc_2_i => ldq_alloc_2_q,
			entry_alloc_3_i => ldq_alloc_3_q,
			entry_alloc_4_i => ldq_alloc_4_q,
			entry_alloc_5_i => ldq_alloc_5_q,
			entry_alloc_6_i => ldq_alloc_6_q,
			entry_alloc_7_i => ldq_alloc_7_q,
			entry_alloc_8_i => ldq_alloc_8_q,
			entry_alloc_9_i => ldq_alloc_9_q,
			entry_alloc_10_i => ldq_alloc_10_q,
			entry_alloc_11_i => ldq_alloc_11_q,
			entry_alloc_12_i => ldq_alloc_12_q,
			entry_alloc_13_i => ldq_alloc_13_q,
			entry_payload_valid_0_i => ldq_addr_valid_0_q,
			entry_payload_valid_1_i => ldq_addr_valid_1_q,
			entry_payload_valid_2_i => ldq_addr_valid_2_q,
			entry_payload_valid_3_i => ldq_addr_valid_3_q,
			entry_payload_valid_4_i => ldq_addr_valid_4_q,
			entry_payload_valid_5_i => ldq_addr_valid_5_q,
			entry_payload_valid_6_i => ldq_addr_valid_6_q,
			entry_payload_valid_7_i => ldq_addr_valid_7_q,
			entry_payload_valid_8_i => ldq_addr_valid_8_q,
			entry_payload_valid_9_i => ldq_addr_valid_9_q,
			entry_payload_valid_10_i => ldq_addr_valid_10_q,
			entry_payload_valid_11_i => ldq_addr_valid_11_q,
			entry_payload_valid_12_i => ldq_addr_valid_12_q,
			entry_payload_valid_13_i => ldq_addr_valid_13_q,
			entry_payload_0_o => ldq_addr_0_d,
			entry_payload_1_o => ldq_addr_1_d,
			entry_payload_2_o => ldq_addr_2_d,
			entry_payload_3_o => ldq_addr_3_d,
			entry_payload_4_o => ldq_addr_4_d,
			entry_payload_5_o => ldq_addr_5_d,
			entry_payload_6_o => ldq_addr_6_d,
			entry_payload_7_o => ldq_addr_7_d,
			entry_payload_8_o => ldq_addr_8_d,
			entry_payload_9_o => ldq_addr_9_d,
			entry_payload_10_o => ldq_addr_10_d,
			entry_payload_11_o => ldq_addr_11_d,
			entry_payload_12_o => ldq_addr_12_d,
			entry_payload_13_o => ldq_addr_13_d,
			entry_wen_0_o => ldq_addr_wen_0,
			entry_wen_1_o => ldq_addr_wen_1,
			entry_wen_2_o => ldq_addr_wen_2,
			entry_wen_3_o => ldq_addr_wen_3,
			entry_wen_4_o => ldq_addr_wen_4,
			entry_wen_5_o => ldq_addr_wen_5,
			entry_wen_6_o => ldq_addr_wen_6,
			entry_wen_7_o => ldq_addr_wen_7,
			entry_wen_8_o => ldq_addr_wen_8,
			entry_wen_9_o => ldq_addr_wen_9,
			entry_wen_10_o => ldq_addr_wen_10,
			entry_wen_11_o => ldq_addr_wen_11,
			entry_wen_12_o => ldq_addr_wen_12,
			entry_wen_13_o => ldq_addr_wen_13,
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
			entry_alloc_2_i => ldq_alloc_2_q,
			entry_alloc_3_i => ldq_alloc_3_q,
			entry_alloc_4_i => ldq_alloc_4_q,
			entry_alloc_5_i => ldq_alloc_5_q,
			entry_alloc_6_i => ldq_alloc_6_q,
			entry_alloc_7_i => ldq_alloc_7_q,
			entry_alloc_8_i => ldq_alloc_8_q,
			entry_alloc_9_i => ldq_alloc_9_q,
			entry_alloc_10_i => ldq_alloc_10_q,
			entry_alloc_11_i => ldq_alloc_11_q,
			entry_alloc_12_i => ldq_alloc_12_q,
			entry_alloc_13_i => ldq_alloc_13_q,
			entry_payload_valid_0_i => ldq_data_valid_0_q,
			entry_payload_valid_1_i => ldq_data_valid_1_q,
			entry_payload_valid_2_i => ldq_data_valid_2_q,
			entry_payload_valid_3_i => ldq_data_valid_3_q,
			entry_payload_valid_4_i => ldq_data_valid_4_q,
			entry_payload_valid_5_i => ldq_data_valid_5_q,
			entry_payload_valid_6_i => ldq_data_valid_6_q,
			entry_payload_valid_7_i => ldq_data_valid_7_q,
			entry_payload_valid_8_i => ldq_data_valid_8_q,
			entry_payload_valid_9_i => ldq_data_valid_9_q,
			entry_payload_valid_10_i => ldq_data_valid_10_q,
			entry_payload_valid_11_i => ldq_data_valid_11_q,
			entry_payload_valid_12_i => ldq_data_valid_12_q,
			entry_payload_valid_13_i => ldq_data_valid_13_q,
			entry_payload_0_i => ldq_data_0_q,
			entry_payload_1_i => ldq_data_1_q,
			entry_payload_2_i => ldq_data_2_q,
			entry_payload_3_i => ldq_data_3_q,
			entry_payload_4_i => ldq_data_4_q,
			entry_payload_5_i => ldq_data_5_q,
			entry_payload_6_i => ldq_data_6_q,
			entry_payload_7_i => ldq_data_7_q,
			entry_payload_8_i => ldq_data_8_q,
			entry_payload_9_i => ldq_data_9_q,
			entry_payload_10_i => ldq_data_10_q,
			entry_payload_11_i => ldq_data_11_q,
			entry_payload_12_i => ldq_data_12_q,
			entry_payload_13_i => ldq_data_13_q,
			entry_reset_0_o => ldq_reset_0,
			entry_reset_1_o => ldq_reset_1,
			entry_reset_2_o => ldq_reset_2,
			entry_reset_3_o => ldq_reset_3,
			entry_reset_4_o => ldq_reset_4,
			entry_reset_5_o => ldq_reset_5,
			entry_reset_6_o => ldq_reset_6,
			entry_reset_7_o => ldq_reset_7,
			entry_reset_8_o => ldq_reset_8,
			entry_reset_9_o => ldq_reset_9,
			entry_reset_10_o => ldq_reset_10,
			entry_reset_11_o => ldq_reset_11,
			entry_reset_12_o => ldq_reset_12,
			entry_reset_13_o => ldq_reset_13,
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
			entry_alloc_5_i => stq_alloc_5_q,
			entry_alloc_6_i => stq_alloc_6_q,
			entry_alloc_7_i => stq_alloc_7_q,
			entry_alloc_8_i => stq_alloc_8_q,
			entry_alloc_9_i => stq_alloc_9_q,
			entry_alloc_10_i => stq_alloc_10_q,
			entry_alloc_11_i => stq_alloc_11_q,
			entry_alloc_12_i => stq_alloc_12_q,
			entry_alloc_13_i => stq_alloc_13_q,
			entry_payload_valid_0_i => stq_addr_valid_0_q,
			entry_payload_valid_1_i => stq_addr_valid_1_q,
			entry_payload_valid_2_i => stq_addr_valid_2_q,
			entry_payload_valid_3_i => stq_addr_valid_3_q,
			entry_payload_valid_4_i => stq_addr_valid_4_q,
			entry_payload_valid_5_i => stq_addr_valid_5_q,
			entry_payload_valid_6_i => stq_addr_valid_6_q,
			entry_payload_valid_7_i => stq_addr_valid_7_q,
			entry_payload_valid_8_i => stq_addr_valid_8_q,
			entry_payload_valid_9_i => stq_addr_valid_9_q,
			entry_payload_valid_10_i => stq_addr_valid_10_q,
			entry_payload_valid_11_i => stq_addr_valid_11_q,
			entry_payload_valid_12_i => stq_addr_valid_12_q,
			entry_payload_valid_13_i => stq_addr_valid_13_q,
			entry_payload_0_o => stq_addr_0_d,
			entry_payload_1_o => stq_addr_1_d,
			entry_payload_2_o => stq_addr_2_d,
			entry_payload_3_o => stq_addr_3_d,
			entry_payload_4_o => stq_addr_4_d,
			entry_payload_5_o => stq_addr_5_d,
			entry_payload_6_o => stq_addr_6_d,
			entry_payload_7_o => stq_addr_7_d,
			entry_payload_8_o => stq_addr_8_d,
			entry_payload_9_o => stq_addr_9_d,
			entry_payload_10_o => stq_addr_10_d,
			entry_payload_11_o => stq_addr_11_d,
			entry_payload_12_o => stq_addr_12_d,
			entry_payload_13_o => stq_addr_13_d,
			entry_wen_0_o => stq_addr_wen_0,
			entry_wen_1_o => stq_addr_wen_1,
			entry_wen_2_o => stq_addr_wen_2,
			entry_wen_3_o => stq_addr_wen_3,
			entry_wen_4_o => stq_addr_wen_4,
			entry_wen_5_o => stq_addr_wen_5,
			entry_wen_6_o => stq_addr_wen_6,
			entry_wen_7_o => stq_addr_wen_7,
			entry_wen_8_o => stq_addr_wen_8,
			entry_wen_9_o => stq_addr_wen_9,
			entry_wen_10_o => stq_addr_wen_10,
			entry_wen_11_o => stq_addr_wen_11,
			entry_wen_12_o => stq_addr_wen_12,
			entry_wen_13_o => stq_addr_wen_13,
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
			entry_alloc_5_i => stq_alloc_5_q,
			entry_alloc_6_i => stq_alloc_6_q,
			entry_alloc_7_i => stq_alloc_7_q,
			entry_alloc_8_i => stq_alloc_8_q,
			entry_alloc_9_i => stq_alloc_9_q,
			entry_alloc_10_i => stq_alloc_10_q,
			entry_alloc_11_i => stq_alloc_11_q,
			entry_alloc_12_i => stq_alloc_12_q,
			entry_alloc_13_i => stq_alloc_13_q,
			entry_payload_valid_0_i => stq_data_valid_0_q,
			entry_payload_valid_1_i => stq_data_valid_1_q,
			entry_payload_valid_2_i => stq_data_valid_2_q,
			entry_payload_valid_3_i => stq_data_valid_3_q,
			entry_payload_valid_4_i => stq_data_valid_4_q,
			entry_payload_valid_5_i => stq_data_valid_5_q,
			entry_payload_valid_6_i => stq_data_valid_6_q,
			entry_payload_valid_7_i => stq_data_valid_7_q,
			entry_payload_valid_8_i => stq_data_valid_8_q,
			entry_payload_valid_9_i => stq_data_valid_9_q,
			entry_payload_valid_10_i => stq_data_valid_10_q,
			entry_payload_valid_11_i => stq_data_valid_11_q,
			entry_payload_valid_12_i => stq_data_valid_12_q,
			entry_payload_valid_13_i => stq_data_valid_13_q,
			entry_payload_0_o => stq_data_0_d,
			entry_payload_1_o => stq_data_1_d,
			entry_payload_2_o => stq_data_2_d,
			entry_payload_3_o => stq_data_3_d,
			entry_payload_4_o => stq_data_4_d,
			entry_payload_5_o => stq_data_5_d,
			entry_payload_6_o => stq_data_6_d,
			entry_payload_7_o => stq_data_7_d,
			entry_payload_8_o => stq_data_8_d,
			entry_payload_9_o => stq_data_9_d,
			entry_payload_10_o => stq_data_10_d,
			entry_payload_11_o => stq_data_11_d,
			entry_payload_12_o => stq_data_12_d,
			entry_payload_13_o => stq_data_13_d,
			entry_wen_0_o => stq_data_wen_0,
			entry_wen_1_o => stq_data_wen_1,
			entry_wen_2_o => stq_data_wen_2,
			entry_wen_3_o => stq_data_wen_3,
			entry_wen_4_o => stq_data_wen_4,
			entry_wen_5_o => stq_data_wen_5,
			entry_wen_6_o => stq_data_wen_6,
			entry_wen_7_o => stq_data_wen_7,
			entry_wen_8_o => stq_data_wen_8,
			entry_wen_9_o => stq_data_wen_9,
			entry_wen_10_o => stq_data_wen_10,
			entry_wen_11_o => stq_data_wen_11,
			entry_wen_12_o => stq_data_wen_12,
			entry_wen_13_o => stq_data_wen_13,
			queue_head_oh_i => stq_head_oh
		);
	addr_valid_0(0) <= ldq_addr_valid_0_q and stq_addr_valid_0_q;
	addr_valid_0(1) <= ldq_addr_valid_0_q and stq_addr_valid_1_q;
	addr_valid_0(2) <= ldq_addr_valid_0_q and stq_addr_valid_2_q;
	addr_valid_0(3) <= ldq_addr_valid_0_q and stq_addr_valid_3_q;
	addr_valid_0(4) <= ldq_addr_valid_0_q and stq_addr_valid_4_q;
	addr_valid_0(5) <= ldq_addr_valid_0_q and stq_addr_valid_5_q;
	addr_valid_0(6) <= ldq_addr_valid_0_q and stq_addr_valid_6_q;
	addr_valid_0(7) <= ldq_addr_valid_0_q and stq_addr_valid_7_q;
	addr_valid_0(8) <= ldq_addr_valid_0_q and stq_addr_valid_8_q;
	addr_valid_0(9) <= ldq_addr_valid_0_q and stq_addr_valid_9_q;
	addr_valid_0(10) <= ldq_addr_valid_0_q and stq_addr_valid_10_q;
	addr_valid_0(11) <= ldq_addr_valid_0_q and stq_addr_valid_11_q;
	addr_valid_0(12) <= ldq_addr_valid_0_q and stq_addr_valid_12_q;
	addr_valid_0(13) <= ldq_addr_valid_0_q and stq_addr_valid_13_q;
	addr_valid_1(0) <= ldq_addr_valid_1_q and stq_addr_valid_0_q;
	addr_valid_1(1) <= ldq_addr_valid_1_q and stq_addr_valid_1_q;
	addr_valid_1(2) <= ldq_addr_valid_1_q and stq_addr_valid_2_q;
	addr_valid_1(3) <= ldq_addr_valid_1_q and stq_addr_valid_3_q;
	addr_valid_1(4) <= ldq_addr_valid_1_q and stq_addr_valid_4_q;
	addr_valid_1(5) <= ldq_addr_valid_1_q and stq_addr_valid_5_q;
	addr_valid_1(6) <= ldq_addr_valid_1_q and stq_addr_valid_6_q;
	addr_valid_1(7) <= ldq_addr_valid_1_q and stq_addr_valid_7_q;
	addr_valid_1(8) <= ldq_addr_valid_1_q and stq_addr_valid_8_q;
	addr_valid_1(9) <= ldq_addr_valid_1_q and stq_addr_valid_9_q;
	addr_valid_1(10) <= ldq_addr_valid_1_q and stq_addr_valid_10_q;
	addr_valid_1(11) <= ldq_addr_valid_1_q and stq_addr_valid_11_q;
	addr_valid_1(12) <= ldq_addr_valid_1_q and stq_addr_valid_12_q;
	addr_valid_1(13) <= ldq_addr_valid_1_q and stq_addr_valid_13_q;
	addr_valid_2(0) <= ldq_addr_valid_2_q and stq_addr_valid_0_q;
	addr_valid_2(1) <= ldq_addr_valid_2_q and stq_addr_valid_1_q;
	addr_valid_2(2) <= ldq_addr_valid_2_q and stq_addr_valid_2_q;
	addr_valid_2(3) <= ldq_addr_valid_2_q and stq_addr_valid_3_q;
	addr_valid_2(4) <= ldq_addr_valid_2_q and stq_addr_valid_4_q;
	addr_valid_2(5) <= ldq_addr_valid_2_q and stq_addr_valid_5_q;
	addr_valid_2(6) <= ldq_addr_valid_2_q and stq_addr_valid_6_q;
	addr_valid_2(7) <= ldq_addr_valid_2_q and stq_addr_valid_7_q;
	addr_valid_2(8) <= ldq_addr_valid_2_q and stq_addr_valid_8_q;
	addr_valid_2(9) <= ldq_addr_valid_2_q and stq_addr_valid_9_q;
	addr_valid_2(10) <= ldq_addr_valid_2_q and stq_addr_valid_10_q;
	addr_valid_2(11) <= ldq_addr_valid_2_q and stq_addr_valid_11_q;
	addr_valid_2(12) <= ldq_addr_valid_2_q and stq_addr_valid_12_q;
	addr_valid_2(13) <= ldq_addr_valid_2_q and stq_addr_valid_13_q;
	addr_valid_3(0) <= ldq_addr_valid_3_q and stq_addr_valid_0_q;
	addr_valid_3(1) <= ldq_addr_valid_3_q and stq_addr_valid_1_q;
	addr_valid_3(2) <= ldq_addr_valid_3_q and stq_addr_valid_2_q;
	addr_valid_3(3) <= ldq_addr_valid_3_q and stq_addr_valid_3_q;
	addr_valid_3(4) <= ldq_addr_valid_3_q and stq_addr_valid_4_q;
	addr_valid_3(5) <= ldq_addr_valid_3_q and stq_addr_valid_5_q;
	addr_valid_3(6) <= ldq_addr_valid_3_q and stq_addr_valid_6_q;
	addr_valid_3(7) <= ldq_addr_valid_3_q and stq_addr_valid_7_q;
	addr_valid_3(8) <= ldq_addr_valid_3_q and stq_addr_valid_8_q;
	addr_valid_3(9) <= ldq_addr_valid_3_q and stq_addr_valid_9_q;
	addr_valid_3(10) <= ldq_addr_valid_3_q and stq_addr_valid_10_q;
	addr_valid_3(11) <= ldq_addr_valid_3_q and stq_addr_valid_11_q;
	addr_valid_3(12) <= ldq_addr_valid_3_q and stq_addr_valid_12_q;
	addr_valid_3(13) <= ldq_addr_valid_3_q and stq_addr_valid_13_q;
	addr_valid_4(0) <= ldq_addr_valid_4_q and stq_addr_valid_0_q;
	addr_valid_4(1) <= ldq_addr_valid_4_q and stq_addr_valid_1_q;
	addr_valid_4(2) <= ldq_addr_valid_4_q and stq_addr_valid_2_q;
	addr_valid_4(3) <= ldq_addr_valid_4_q and stq_addr_valid_3_q;
	addr_valid_4(4) <= ldq_addr_valid_4_q and stq_addr_valid_4_q;
	addr_valid_4(5) <= ldq_addr_valid_4_q and stq_addr_valid_5_q;
	addr_valid_4(6) <= ldq_addr_valid_4_q and stq_addr_valid_6_q;
	addr_valid_4(7) <= ldq_addr_valid_4_q and stq_addr_valid_7_q;
	addr_valid_4(8) <= ldq_addr_valid_4_q and stq_addr_valid_8_q;
	addr_valid_4(9) <= ldq_addr_valid_4_q and stq_addr_valid_9_q;
	addr_valid_4(10) <= ldq_addr_valid_4_q and stq_addr_valid_10_q;
	addr_valid_4(11) <= ldq_addr_valid_4_q and stq_addr_valid_11_q;
	addr_valid_4(12) <= ldq_addr_valid_4_q and stq_addr_valid_12_q;
	addr_valid_4(13) <= ldq_addr_valid_4_q and stq_addr_valid_13_q;
	addr_valid_5(0) <= ldq_addr_valid_5_q and stq_addr_valid_0_q;
	addr_valid_5(1) <= ldq_addr_valid_5_q and stq_addr_valid_1_q;
	addr_valid_5(2) <= ldq_addr_valid_5_q and stq_addr_valid_2_q;
	addr_valid_5(3) <= ldq_addr_valid_5_q and stq_addr_valid_3_q;
	addr_valid_5(4) <= ldq_addr_valid_5_q and stq_addr_valid_4_q;
	addr_valid_5(5) <= ldq_addr_valid_5_q and stq_addr_valid_5_q;
	addr_valid_5(6) <= ldq_addr_valid_5_q and stq_addr_valid_6_q;
	addr_valid_5(7) <= ldq_addr_valid_5_q and stq_addr_valid_7_q;
	addr_valid_5(8) <= ldq_addr_valid_5_q and stq_addr_valid_8_q;
	addr_valid_5(9) <= ldq_addr_valid_5_q and stq_addr_valid_9_q;
	addr_valid_5(10) <= ldq_addr_valid_5_q and stq_addr_valid_10_q;
	addr_valid_5(11) <= ldq_addr_valid_5_q and stq_addr_valid_11_q;
	addr_valid_5(12) <= ldq_addr_valid_5_q and stq_addr_valid_12_q;
	addr_valid_5(13) <= ldq_addr_valid_5_q and stq_addr_valid_13_q;
	addr_valid_6(0) <= ldq_addr_valid_6_q and stq_addr_valid_0_q;
	addr_valid_6(1) <= ldq_addr_valid_6_q and stq_addr_valid_1_q;
	addr_valid_6(2) <= ldq_addr_valid_6_q and stq_addr_valid_2_q;
	addr_valid_6(3) <= ldq_addr_valid_6_q and stq_addr_valid_3_q;
	addr_valid_6(4) <= ldq_addr_valid_6_q and stq_addr_valid_4_q;
	addr_valid_6(5) <= ldq_addr_valid_6_q and stq_addr_valid_5_q;
	addr_valid_6(6) <= ldq_addr_valid_6_q and stq_addr_valid_6_q;
	addr_valid_6(7) <= ldq_addr_valid_6_q and stq_addr_valid_7_q;
	addr_valid_6(8) <= ldq_addr_valid_6_q and stq_addr_valid_8_q;
	addr_valid_6(9) <= ldq_addr_valid_6_q and stq_addr_valid_9_q;
	addr_valid_6(10) <= ldq_addr_valid_6_q and stq_addr_valid_10_q;
	addr_valid_6(11) <= ldq_addr_valid_6_q and stq_addr_valid_11_q;
	addr_valid_6(12) <= ldq_addr_valid_6_q and stq_addr_valid_12_q;
	addr_valid_6(13) <= ldq_addr_valid_6_q and stq_addr_valid_13_q;
	addr_valid_7(0) <= ldq_addr_valid_7_q and stq_addr_valid_0_q;
	addr_valid_7(1) <= ldq_addr_valid_7_q and stq_addr_valid_1_q;
	addr_valid_7(2) <= ldq_addr_valid_7_q and stq_addr_valid_2_q;
	addr_valid_7(3) <= ldq_addr_valid_7_q and stq_addr_valid_3_q;
	addr_valid_7(4) <= ldq_addr_valid_7_q and stq_addr_valid_4_q;
	addr_valid_7(5) <= ldq_addr_valid_7_q and stq_addr_valid_5_q;
	addr_valid_7(6) <= ldq_addr_valid_7_q and stq_addr_valid_6_q;
	addr_valid_7(7) <= ldq_addr_valid_7_q and stq_addr_valid_7_q;
	addr_valid_7(8) <= ldq_addr_valid_7_q and stq_addr_valid_8_q;
	addr_valid_7(9) <= ldq_addr_valid_7_q and stq_addr_valid_9_q;
	addr_valid_7(10) <= ldq_addr_valid_7_q and stq_addr_valid_10_q;
	addr_valid_7(11) <= ldq_addr_valid_7_q and stq_addr_valid_11_q;
	addr_valid_7(12) <= ldq_addr_valid_7_q and stq_addr_valid_12_q;
	addr_valid_7(13) <= ldq_addr_valid_7_q and stq_addr_valid_13_q;
	addr_valid_8(0) <= ldq_addr_valid_8_q and stq_addr_valid_0_q;
	addr_valid_8(1) <= ldq_addr_valid_8_q and stq_addr_valid_1_q;
	addr_valid_8(2) <= ldq_addr_valid_8_q and stq_addr_valid_2_q;
	addr_valid_8(3) <= ldq_addr_valid_8_q and stq_addr_valid_3_q;
	addr_valid_8(4) <= ldq_addr_valid_8_q and stq_addr_valid_4_q;
	addr_valid_8(5) <= ldq_addr_valid_8_q and stq_addr_valid_5_q;
	addr_valid_8(6) <= ldq_addr_valid_8_q and stq_addr_valid_6_q;
	addr_valid_8(7) <= ldq_addr_valid_8_q and stq_addr_valid_7_q;
	addr_valid_8(8) <= ldq_addr_valid_8_q and stq_addr_valid_8_q;
	addr_valid_8(9) <= ldq_addr_valid_8_q and stq_addr_valid_9_q;
	addr_valid_8(10) <= ldq_addr_valid_8_q and stq_addr_valid_10_q;
	addr_valid_8(11) <= ldq_addr_valid_8_q and stq_addr_valid_11_q;
	addr_valid_8(12) <= ldq_addr_valid_8_q and stq_addr_valid_12_q;
	addr_valid_8(13) <= ldq_addr_valid_8_q and stq_addr_valid_13_q;
	addr_valid_9(0) <= ldq_addr_valid_9_q and stq_addr_valid_0_q;
	addr_valid_9(1) <= ldq_addr_valid_9_q and stq_addr_valid_1_q;
	addr_valid_9(2) <= ldq_addr_valid_9_q and stq_addr_valid_2_q;
	addr_valid_9(3) <= ldq_addr_valid_9_q and stq_addr_valid_3_q;
	addr_valid_9(4) <= ldq_addr_valid_9_q and stq_addr_valid_4_q;
	addr_valid_9(5) <= ldq_addr_valid_9_q and stq_addr_valid_5_q;
	addr_valid_9(6) <= ldq_addr_valid_9_q and stq_addr_valid_6_q;
	addr_valid_9(7) <= ldq_addr_valid_9_q and stq_addr_valid_7_q;
	addr_valid_9(8) <= ldq_addr_valid_9_q and stq_addr_valid_8_q;
	addr_valid_9(9) <= ldq_addr_valid_9_q and stq_addr_valid_9_q;
	addr_valid_9(10) <= ldq_addr_valid_9_q and stq_addr_valid_10_q;
	addr_valid_9(11) <= ldq_addr_valid_9_q and stq_addr_valid_11_q;
	addr_valid_9(12) <= ldq_addr_valid_9_q and stq_addr_valid_12_q;
	addr_valid_9(13) <= ldq_addr_valid_9_q and stq_addr_valid_13_q;
	addr_valid_10(0) <= ldq_addr_valid_10_q and stq_addr_valid_0_q;
	addr_valid_10(1) <= ldq_addr_valid_10_q and stq_addr_valid_1_q;
	addr_valid_10(2) <= ldq_addr_valid_10_q and stq_addr_valid_2_q;
	addr_valid_10(3) <= ldq_addr_valid_10_q and stq_addr_valid_3_q;
	addr_valid_10(4) <= ldq_addr_valid_10_q and stq_addr_valid_4_q;
	addr_valid_10(5) <= ldq_addr_valid_10_q and stq_addr_valid_5_q;
	addr_valid_10(6) <= ldq_addr_valid_10_q and stq_addr_valid_6_q;
	addr_valid_10(7) <= ldq_addr_valid_10_q and stq_addr_valid_7_q;
	addr_valid_10(8) <= ldq_addr_valid_10_q and stq_addr_valid_8_q;
	addr_valid_10(9) <= ldq_addr_valid_10_q and stq_addr_valid_9_q;
	addr_valid_10(10) <= ldq_addr_valid_10_q and stq_addr_valid_10_q;
	addr_valid_10(11) <= ldq_addr_valid_10_q and stq_addr_valid_11_q;
	addr_valid_10(12) <= ldq_addr_valid_10_q and stq_addr_valid_12_q;
	addr_valid_10(13) <= ldq_addr_valid_10_q and stq_addr_valid_13_q;
	addr_valid_11(0) <= ldq_addr_valid_11_q and stq_addr_valid_0_q;
	addr_valid_11(1) <= ldq_addr_valid_11_q and stq_addr_valid_1_q;
	addr_valid_11(2) <= ldq_addr_valid_11_q and stq_addr_valid_2_q;
	addr_valid_11(3) <= ldq_addr_valid_11_q and stq_addr_valid_3_q;
	addr_valid_11(4) <= ldq_addr_valid_11_q and stq_addr_valid_4_q;
	addr_valid_11(5) <= ldq_addr_valid_11_q and stq_addr_valid_5_q;
	addr_valid_11(6) <= ldq_addr_valid_11_q and stq_addr_valid_6_q;
	addr_valid_11(7) <= ldq_addr_valid_11_q and stq_addr_valid_7_q;
	addr_valid_11(8) <= ldq_addr_valid_11_q and stq_addr_valid_8_q;
	addr_valid_11(9) <= ldq_addr_valid_11_q and stq_addr_valid_9_q;
	addr_valid_11(10) <= ldq_addr_valid_11_q and stq_addr_valid_10_q;
	addr_valid_11(11) <= ldq_addr_valid_11_q and stq_addr_valid_11_q;
	addr_valid_11(12) <= ldq_addr_valid_11_q and stq_addr_valid_12_q;
	addr_valid_11(13) <= ldq_addr_valid_11_q and stq_addr_valid_13_q;
	addr_valid_12(0) <= ldq_addr_valid_12_q and stq_addr_valid_0_q;
	addr_valid_12(1) <= ldq_addr_valid_12_q and stq_addr_valid_1_q;
	addr_valid_12(2) <= ldq_addr_valid_12_q and stq_addr_valid_2_q;
	addr_valid_12(3) <= ldq_addr_valid_12_q and stq_addr_valid_3_q;
	addr_valid_12(4) <= ldq_addr_valid_12_q and stq_addr_valid_4_q;
	addr_valid_12(5) <= ldq_addr_valid_12_q and stq_addr_valid_5_q;
	addr_valid_12(6) <= ldq_addr_valid_12_q and stq_addr_valid_6_q;
	addr_valid_12(7) <= ldq_addr_valid_12_q and stq_addr_valid_7_q;
	addr_valid_12(8) <= ldq_addr_valid_12_q and stq_addr_valid_8_q;
	addr_valid_12(9) <= ldq_addr_valid_12_q and stq_addr_valid_9_q;
	addr_valid_12(10) <= ldq_addr_valid_12_q and stq_addr_valid_10_q;
	addr_valid_12(11) <= ldq_addr_valid_12_q and stq_addr_valid_11_q;
	addr_valid_12(12) <= ldq_addr_valid_12_q and stq_addr_valid_12_q;
	addr_valid_12(13) <= ldq_addr_valid_12_q and stq_addr_valid_13_q;
	addr_valid_13(0) <= ldq_addr_valid_13_q and stq_addr_valid_0_q;
	addr_valid_13(1) <= ldq_addr_valid_13_q and stq_addr_valid_1_q;
	addr_valid_13(2) <= ldq_addr_valid_13_q and stq_addr_valid_2_q;
	addr_valid_13(3) <= ldq_addr_valid_13_q and stq_addr_valid_3_q;
	addr_valid_13(4) <= ldq_addr_valid_13_q and stq_addr_valid_4_q;
	addr_valid_13(5) <= ldq_addr_valid_13_q and stq_addr_valid_5_q;
	addr_valid_13(6) <= ldq_addr_valid_13_q and stq_addr_valid_6_q;
	addr_valid_13(7) <= ldq_addr_valid_13_q and stq_addr_valid_7_q;
	addr_valid_13(8) <= ldq_addr_valid_13_q and stq_addr_valid_8_q;
	addr_valid_13(9) <= ldq_addr_valid_13_q and stq_addr_valid_9_q;
	addr_valid_13(10) <= ldq_addr_valid_13_q and stq_addr_valid_10_q;
	addr_valid_13(11) <= ldq_addr_valid_13_q and stq_addr_valid_11_q;
	addr_valid_13(12) <= ldq_addr_valid_13_q and stq_addr_valid_12_q;
	addr_valid_13(13) <= ldq_addr_valid_13_q and stq_addr_valid_13_q;
	addr_same_0(0) <= '1' when ldq_addr_0_q = stq_addr_0_q else '0';
	addr_same_0(1) <= '1' when ldq_addr_0_q = stq_addr_1_q else '0';
	addr_same_0(2) <= '1' when ldq_addr_0_q = stq_addr_2_q else '0';
	addr_same_0(3) <= '1' when ldq_addr_0_q = stq_addr_3_q else '0';
	addr_same_0(4) <= '1' when ldq_addr_0_q = stq_addr_4_q else '0';
	addr_same_0(5) <= '1' when ldq_addr_0_q = stq_addr_5_q else '0';
	addr_same_0(6) <= '1' when ldq_addr_0_q = stq_addr_6_q else '0';
	addr_same_0(7) <= '1' when ldq_addr_0_q = stq_addr_7_q else '0';
	addr_same_0(8) <= '1' when ldq_addr_0_q = stq_addr_8_q else '0';
	addr_same_0(9) <= '1' when ldq_addr_0_q = stq_addr_9_q else '0';
	addr_same_0(10) <= '1' when ldq_addr_0_q = stq_addr_10_q else '0';
	addr_same_0(11) <= '1' when ldq_addr_0_q = stq_addr_11_q else '0';
	addr_same_0(12) <= '1' when ldq_addr_0_q = stq_addr_12_q else '0';
	addr_same_0(13) <= '1' when ldq_addr_0_q = stq_addr_13_q else '0';
	addr_same_1(0) <= '1' when ldq_addr_1_q = stq_addr_0_q else '0';
	addr_same_1(1) <= '1' when ldq_addr_1_q = stq_addr_1_q else '0';
	addr_same_1(2) <= '1' when ldq_addr_1_q = stq_addr_2_q else '0';
	addr_same_1(3) <= '1' when ldq_addr_1_q = stq_addr_3_q else '0';
	addr_same_1(4) <= '1' when ldq_addr_1_q = stq_addr_4_q else '0';
	addr_same_1(5) <= '1' when ldq_addr_1_q = stq_addr_5_q else '0';
	addr_same_1(6) <= '1' when ldq_addr_1_q = stq_addr_6_q else '0';
	addr_same_1(7) <= '1' when ldq_addr_1_q = stq_addr_7_q else '0';
	addr_same_1(8) <= '1' when ldq_addr_1_q = stq_addr_8_q else '0';
	addr_same_1(9) <= '1' when ldq_addr_1_q = stq_addr_9_q else '0';
	addr_same_1(10) <= '1' when ldq_addr_1_q = stq_addr_10_q else '0';
	addr_same_1(11) <= '1' when ldq_addr_1_q = stq_addr_11_q else '0';
	addr_same_1(12) <= '1' when ldq_addr_1_q = stq_addr_12_q else '0';
	addr_same_1(13) <= '1' when ldq_addr_1_q = stq_addr_13_q else '0';
	addr_same_2(0) <= '1' when ldq_addr_2_q = stq_addr_0_q else '0';
	addr_same_2(1) <= '1' when ldq_addr_2_q = stq_addr_1_q else '0';
	addr_same_2(2) <= '1' when ldq_addr_2_q = stq_addr_2_q else '0';
	addr_same_2(3) <= '1' when ldq_addr_2_q = stq_addr_3_q else '0';
	addr_same_2(4) <= '1' when ldq_addr_2_q = stq_addr_4_q else '0';
	addr_same_2(5) <= '1' when ldq_addr_2_q = stq_addr_5_q else '0';
	addr_same_2(6) <= '1' when ldq_addr_2_q = stq_addr_6_q else '0';
	addr_same_2(7) <= '1' when ldq_addr_2_q = stq_addr_7_q else '0';
	addr_same_2(8) <= '1' when ldq_addr_2_q = stq_addr_8_q else '0';
	addr_same_2(9) <= '1' when ldq_addr_2_q = stq_addr_9_q else '0';
	addr_same_2(10) <= '1' when ldq_addr_2_q = stq_addr_10_q else '0';
	addr_same_2(11) <= '1' when ldq_addr_2_q = stq_addr_11_q else '0';
	addr_same_2(12) <= '1' when ldq_addr_2_q = stq_addr_12_q else '0';
	addr_same_2(13) <= '1' when ldq_addr_2_q = stq_addr_13_q else '0';
	addr_same_3(0) <= '1' when ldq_addr_3_q = stq_addr_0_q else '0';
	addr_same_3(1) <= '1' when ldq_addr_3_q = stq_addr_1_q else '0';
	addr_same_3(2) <= '1' when ldq_addr_3_q = stq_addr_2_q else '0';
	addr_same_3(3) <= '1' when ldq_addr_3_q = stq_addr_3_q else '0';
	addr_same_3(4) <= '1' when ldq_addr_3_q = stq_addr_4_q else '0';
	addr_same_3(5) <= '1' when ldq_addr_3_q = stq_addr_5_q else '0';
	addr_same_3(6) <= '1' when ldq_addr_3_q = stq_addr_6_q else '0';
	addr_same_3(7) <= '1' when ldq_addr_3_q = stq_addr_7_q else '0';
	addr_same_3(8) <= '1' when ldq_addr_3_q = stq_addr_8_q else '0';
	addr_same_3(9) <= '1' when ldq_addr_3_q = stq_addr_9_q else '0';
	addr_same_3(10) <= '1' when ldq_addr_3_q = stq_addr_10_q else '0';
	addr_same_3(11) <= '1' when ldq_addr_3_q = stq_addr_11_q else '0';
	addr_same_3(12) <= '1' when ldq_addr_3_q = stq_addr_12_q else '0';
	addr_same_3(13) <= '1' when ldq_addr_3_q = stq_addr_13_q else '0';
	addr_same_4(0) <= '1' when ldq_addr_4_q = stq_addr_0_q else '0';
	addr_same_4(1) <= '1' when ldq_addr_4_q = stq_addr_1_q else '0';
	addr_same_4(2) <= '1' when ldq_addr_4_q = stq_addr_2_q else '0';
	addr_same_4(3) <= '1' when ldq_addr_4_q = stq_addr_3_q else '0';
	addr_same_4(4) <= '1' when ldq_addr_4_q = stq_addr_4_q else '0';
	addr_same_4(5) <= '1' when ldq_addr_4_q = stq_addr_5_q else '0';
	addr_same_4(6) <= '1' when ldq_addr_4_q = stq_addr_6_q else '0';
	addr_same_4(7) <= '1' when ldq_addr_4_q = stq_addr_7_q else '0';
	addr_same_4(8) <= '1' when ldq_addr_4_q = stq_addr_8_q else '0';
	addr_same_4(9) <= '1' when ldq_addr_4_q = stq_addr_9_q else '0';
	addr_same_4(10) <= '1' when ldq_addr_4_q = stq_addr_10_q else '0';
	addr_same_4(11) <= '1' when ldq_addr_4_q = stq_addr_11_q else '0';
	addr_same_4(12) <= '1' when ldq_addr_4_q = stq_addr_12_q else '0';
	addr_same_4(13) <= '1' when ldq_addr_4_q = stq_addr_13_q else '0';
	addr_same_5(0) <= '1' when ldq_addr_5_q = stq_addr_0_q else '0';
	addr_same_5(1) <= '1' when ldq_addr_5_q = stq_addr_1_q else '0';
	addr_same_5(2) <= '1' when ldq_addr_5_q = stq_addr_2_q else '0';
	addr_same_5(3) <= '1' when ldq_addr_5_q = stq_addr_3_q else '0';
	addr_same_5(4) <= '1' when ldq_addr_5_q = stq_addr_4_q else '0';
	addr_same_5(5) <= '1' when ldq_addr_5_q = stq_addr_5_q else '0';
	addr_same_5(6) <= '1' when ldq_addr_5_q = stq_addr_6_q else '0';
	addr_same_5(7) <= '1' when ldq_addr_5_q = stq_addr_7_q else '0';
	addr_same_5(8) <= '1' when ldq_addr_5_q = stq_addr_8_q else '0';
	addr_same_5(9) <= '1' when ldq_addr_5_q = stq_addr_9_q else '0';
	addr_same_5(10) <= '1' when ldq_addr_5_q = stq_addr_10_q else '0';
	addr_same_5(11) <= '1' when ldq_addr_5_q = stq_addr_11_q else '0';
	addr_same_5(12) <= '1' when ldq_addr_5_q = stq_addr_12_q else '0';
	addr_same_5(13) <= '1' when ldq_addr_5_q = stq_addr_13_q else '0';
	addr_same_6(0) <= '1' when ldq_addr_6_q = stq_addr_0_q else '0';
	addr_same_6(1) <= '1' when ldq_addr_6_q = stq_addr_1_q else '0';
	addr_same_6(2) <= '1' when ldq_addr_6_q = stq_addr_2_q else '0';
	addr_same_6(3) <= '1' when ldq_addr_6_q = stq_addr_3_q else '0';
	addr_same_6(4) <= '1' when ldq_addr_6_q = stq_addr_4_q else '0';
	addr_same_6(5) <= '1' when ldq_addr_6_q = stq_addr_5_q else '0';
	addr_same_6(6) <= '1' when ldq_addr_6_q = stq_addr_6_q else '0';
	addr_same_6(7) <= '1' when ldq_addr_6_q = stq_addr_7_q else '0';
	addr_same_6(8) <= '1' when ldq_addr_6_q = stq_addr_8_q else '0';
	addr_same_6(9) <= '1' when ldq_addr_6_q = stq_addr_9_q else '0';
	addr_same_6(10) <= '1' when ldq_addr_6_q = stq_addr_10_q else '0';
	addr_same_6(11) <= '1' when ldq_addr_6_q = stq_addr_11_q else '0';
	addr_same_6(12) <= '1' when ldq_addr_6_q = stq_addr_12_q else '0';
	addr_same_6(13) <= '1' when ldq_addr_6_q = stq_addr_13_q else '0';
	addr_same_7(0) <= '1' when ldq_addr_7_q = stq_addr_0_q else '0';
	addr_same_7(1) <= '1' when ldq_addr_7_q = stq_addr_1_q else '0';
	addr_same_7(2) <= '1' when ldq_addr_7_q = stq_addr_2_q else '0';
	addr_same_7(3) <= '1' when ldq_addr_7_q = stq_addr_3_q else '0';
	addr_same_7(4) <= '1' when ldq_addr_7_q = stq_addr_4_q else '0';
	addr_same_7(5) <= '1' when ldq_addr_7_q = stq_addr_5_q else '0';
	addr_same_7(6) <= '1' when ldq_addr_7_q = stq_addr_6_q else '0';
	addr_same_7(7) <= '1' when ldq_addr_7_q = stq_addr_7_q else '0';
	addr_same_7(8) <= '1' when ldq_addr_7_q = stq_addr_8_q else '0';
	addr_same_7(9) <= '1' when ldq_addr_7_q = stq_addr_9_q else '0';
	addr_same_7(10) <= '1' when ldq_addr_7_q = stq_addr_10_q else '0';
	addr_same_7(11) <= '1' when ldq_addr_7_q = stq_addr_11_q else '0';
	addr_same_7(12) <= '1' when ldq_addr_7_q = stq_addr_12_q else '0';
	addr_same_7(13) <= '1' when ldq_addr_7_q = stq_addr_13_q else '0';
	addr_same_8(0) <= '1' when ldq_addr_8_q = stq_addr_0_q else '0';
	addr_same_8(1) <= '1' when ldq_addr_8_q = stq_addr_1_q else '0';
	addr_same_8(2) <= '1' when ldq_addr_8_q = stq_addr_2_q else '0';
	addr_same_8(3) <= '1' when ldq_addr_8_q = stq_addr_3_q else '0';
	addr_same_8(4) <= '1' when ldq_addr_8_q = stq_addr_4_q else '0';
	addr_same_8(5) <= '1' when ldq_addr_8_q = stq_addr_5_q else '0';
	addr_same_8(6) <= '1' when ldq_addr_8_q = stq_addr_6_q else '0';
	addr_same_8(7) <= '1' when ldq_addr_8_q = stq_addr_7_q else '0';
	addr_same_8(8) <= '1' when ldq_addr_8_q = stq_addr_8_q else '0';
	addr_same_8(9) <= '1' when ldq_addr_8_q = stq_addr_9_q else '0';
	addr_same_8(10) <= '1' when ldq_addr_8_q = stq_addr_10_q else '0';
	addr_same_8(11) <= '1' when ldq_addr_8_q = stq_addr_11_q else '0';
	addr_same_8(12) <= '1' when ldq_addr_8_q = stq_addr_12_q else '0';
	addr_same_8(13) <= '1' when ldq_addr_8_q = stq_addr_13_q else '0';
	addr_same_9(0) <= '1' when ldq_addr_9_q = stq_addr_0_q else '0';
	addr_same_9(1) <= '1' when ldq_addr_9_q = stq_addr_1_q else '0';
	addr_same_9(2) <= '1' when ldq_addr_9_q = stq_addr_2_q else '0';
	addr_same_9(3) <= '1' when ldq_addr_9_q = stq_addr_3_q else '0';
	addr_same_9(4) <= '1' when ldq_addr_9_q = stq_addr_4_q else '0';
	addr_same_9(5) <= '1' when ldq_addr_9_q = stq_addr_5_q else '0';
	addr_same_9(6) <= '1' when ldq_addr_9_q = stq_addr_6_q else '0';
	addr_same_9(7) <= '1' when ldq_addr_9_q = stq_addr_7_q else '0';
	addr_same_9(8) <= '1' when ldq_addr_9_q = stq_addr_8_q else '0';
	addr_same_9(9) <= '1' when ldq_addr_9_q = stq_addr_9_q else '0';
	addr_same_9(10) <= '1' when ldq_addr_9_q = stq_addr_10_q else '0';
	addr_same_9(11) <= '1' when ldq_addr_9_q = stq_addr_11_q else '0';
	addr_same_9(12) <= '1' when ldq_addr_9_q = stq_addr_12_q else '0';
	addr_same_9(13) <= '1' when ldq_addr_9_q = stq_addr_13_q else '0';
	addr_same_10(0) <= '1' when ldq_addr_10_q = stq_addr_0_q else '0';
	addr_same_10(1) <= '1' when ldq_addr_10_q = stq_addr_1_q else '0';
	addr_same_10(2) <= '1' when ldq_addr_10_q = stq_addr_2_q else '0';
	addr_same_10(3) <= '1' when ldq_addr_10_q = stq_addr_3_q else '0';
	addr_same_10(4) <= '1' when ldq_addr_10_q = stq_addr_4_q else '0';
	addr_same_10(5) <= '1' when ldq_addr_10_q = stq_addr_5_q else '0';
	addr_same_10(6) <= '1' when ldq_addr_10_q = stq_addr_6_q else '0';
	addr_same_10(7) <= '1' when ldq_addr_10_q = stq_addr_7_q else '0';
	addr_same_10(8) <= '1' when ldq_addr_10_q = stq_addr_8_q else '0';
	addr_same_10(9) <= '1' when ldq_addr_10_q = stq_addr_9_q else '0';
	addr_same_10(10) <= '1' when ldq_addr_10_q = stq_addr_10_q else '0';
	addr_same_10(11) <= '1' when ldq_addr_10_q = stq_addr_11_q else '0';
	addr_same_10(12) <= '1' when ldq_addr_10_q = stq_addr_12_q else '0';
	addr_same_10(13) <= '1' when ldq_addr_10_q = stq_addr_13_q else '0';
	addr_same_11(0) <= '1' when ldq_addr_11_q = stq_addr_0_q else '0';
	addr_same_11(1) <= '1' when ldq_addr_11_q = stq_addr_1_q else '0';
	addr_same_11(2) <= '1' when ldq_addr_11_q = stq_addr_2_q else '0';
	addr_same_11(3) <= '1' when ldq_addr_11_q = stq_addr_3_q else '0';
	addr_same_11(4) <= '1' when ldq_addr_11_q = stq_addr_4_q else '0';
	addr_same_11(5) <= '1' when ldq_addr_11_q = stq_addr_5_q else '0';
	addr_same_11(6) <= '1' when ldq_addr_11_q = stq_addr_6_q else '0';
	addr_same_11(7) <= '1' when ldq_addr_11_q = stq_addr_7_q else '0';
	addr_same_11(8) <= '1' when ldq_addr_11_q = stq_addr_8_q else '0';
	addr_same_11(9) <= '1' when ldq_addr_11_q = stq_addr_9_q else '0';
	addr_same_11(10) <= '1' when ldq_addr_11_q = stq_addr_10_q else '0';
	addr_same_11(11) <= '1' when ldq_addr_11_q = stq_addr_11_q else '0';
	addr_same_11(12) <= '1' when ldq_addr_11_q = stq_addr_12_q else '0';
	addr_same_11(13) <= '1' when ldq_addr_11_q = stq_addr_13_q else '0';
	addr_same_12(0) <= '1' when ldq_addr_12_q = stq_addr_0_q else '0';
	addr_same_12(1) <= '1' when ldq_addr_12_q = stq_addr_1_q else '0';
	addr_same_12(2) <= '1' when ldq_addr_12_q = stq_addr_2_q else '0';
	addr_same_12(3) <= '1' when ldq_addr_12_q = stq_addr_3_q else '0';
	addr_same_12(4) <= '1' when ldq_addr_12_q = stq_addr_4_q else '0';
	addr_same_12(5) <= '1' when ldq_addr_12_q = stq_addr_5_q else '0';
	addr_same_12(6) <= '1' when ldq_addr_12_q = stq_addr_6_q else '0';
	addr_same_12(7) <= '1' when ldq_addr_12_q = stq_addr_7_q else '0';
	addr_same_12(8) <= '1' when ldq_addr_12_q = stq_addr_8_q else '0';
	addr_same_12(9) <= '1' when ldq_addr_12_q = stq_addr_9_q else '0';
	addr_same_12(10) <= '1' when ldq_addr_12_q = stq_addr_10_q else '0';
	addr_same_12(11) <= '1' when ldq_addr_12_q = stq_addr_11_q else '0';
	addr_same_12(12) <= '1' when ldq_addr_12_q = stq_addr_12_q else '0';
	addr_same_12(13) <= '1' when ldq_addr_12_q = stq_addr_13_q else '0';
	addr_same_13(0) <= '1' when ldq_addr_13_q = stq_addr_0_q else '0';
	addr_same_13(1) <= '1' when ldq_addr_13_q = stq_addr_1_q else '0';
	addr_same_13(2) <= '1' when ldq_addr_13_q = stq_addr_2_q else '0';
	addr_same_13(3) <= '1' when ldq_addr_13_q = stq_addr_3_q else '0';
	addr_same_13(4) <= '1' when ldq_addr_13_q = stq_addr_4_q else '0';
	addr_same_13(5) <= '1' when ldq_addr_13_q = stq_addr_5_q else '0';
	addr_same_13(6) <= '1' when ldq_addr_13_q = stq_addr_6_q else '0';
	addr_same_13(7) <= '1' when ldq_addr_13_q = stq_addr_7_q else '0';
	addr_same_13(8) <= '1' when ldq_addr_13_q = stq_addr_8_q else '0';
	addr_same_13(9) <= '1' when ldq_addr_13_q = stq_addr_9_q else '0';
	addr_same_13(10) <= '1' when ldq_addr_13_q = stq_addr_10_q else '0';
	addr_same_13(11) <= '1' when ldq_addr_13_q = stq_addr_11_q else '0';
	addr_same_13(12) <= '1' when ldq_addr_13_q = stq_addr_12_q else '0';
	addr_same_13(13) <= '1' when ldq_addr_13_q = stq_addr_13_q else '0';
	ld_st_conflict_0(0) <= stq_alloc_0_q and store_is_older_0_q(0) and ( addr_same_0(0) or not stq_addr_valid_0_q );
	ld_st_conflict_0(1) <= stq_alloc_1_q and store_is_older_0_q(1) and ( addr_same_0(1) or not stq_addr_valid_1_q );
	ld_st_conflict_0(2) <= stq_alloc_2_q and store_is_older_0_q(2) and ( addr_same_0(2) or not stq_addr_valid_2_q );
	ld_st_conflict_0(3) <= stq_alloc_3_q and store_is_older_0_q(3) and ( addr_same_0(3) or not stq_addr_valid_3_q );
	ld_st_conflict_0(4) <= stq_alloc_4_q and store_is_older_0_q(4) and ( addr_same_0(4) or not stq_addr_valid_4_q );
	ld_st_conflict_0(5) <= stq_alloc_5_q and store_is_older_0_q(5) and ( addr_same_0(5) or not stq_addr_valid_5_q );
	ld_st_conflict_0(6) <= stq_alloc_6_q and store_is_older_0_q(6) and ( addr_same_0(6) or not stq_addr_valid_6_q );
	ld_st_conflict_0(7) <= stq_alloc_7_q and store_is_older_0_q(7) and ( addr_same_0(7) or not stq_addr_valid_7_q );
	ld_st_conflict_0(8) <= stq_alloc_8_q and store_is_older_0_q(8) and ( addr_same_0(8) or not stq_addr_valid_8_q );
	ld_st_conflict_0(9) <= stq_alloc_9_q and store_is_older_0_q(9) and ( addr_same_0(9) or not stq_addr_valid_9_q );
	ld_st_conflict_0(10) <= stq_alloc_10_q and store_is_older_0_q(10) and ( addr_same_0(10) or not stq_addr_valid_10_q );
	ld_st_conflict_0(11) <= stq_alloc_11_q and store_is_older_0_q(11) and ( addr_same_0(11) or not stq_addr_valid_11_q );
	ld_st_conflict_0(12) <= stq_alloc_12_q and store_is_older_0_q(12) and ( addr_same_0(12) or not stq_addr_valid_12_q );
	ld_st_conflict_0(13) <= stq_alloc_13_q and store_is_older_0_q(13) and ( addr_same_0(13) or not stq_addr_valid_13_q );
	ld_st_conflict_1(0) <= stq_alloc_0_q and store_is_older_1_q(0) and ( addr_same_1(0) or not stq_addr_valid_0_q );
	ld_st_conflict_1(1) <= stq_alloc_1_q and store_is_older_1_q(1) and ( addr_same_1(1) or not stq_addr_valid_1_q );
	ld_st_conflict_1(2) <= stq_alloc_2_q and store_is_older_1_q(2) and ( addr_same_1(2) or not stq_addr_valid_2_q );
	ld_st_conflict_1(3) <= stq_alloc_3_q and store_is_older_1_q(3) and ( addr_same_1(3) or not stq_addr_valid_3_q );
	ld_st_conflict_1(4) <= stq_alloc_4_q and store_is_older_1_q(4) and ( addr_same_1(4) or not stq_addr_valid_4_q );
	ld_st_conflict_1(5) <= stq_alloc_5_q and store_is_older_1_q(5) and ( addr_same_1(5) or not stq_addr_valid_5_q );
	ld_st_conflict_1(6) <= stq_alloc_6_q and store_is_older_1_q(6) and ( addr_same_1(6) or not stq_addr_valid_6_q );
	ld_st_conflict_1(7) <= stq_alloc_7_q and store_is_older_1_q(7) and ( addr_same_1(7) or not stq_addr_valid_7_q );
	ld_st_conflict_1(8) <= stq_alloc_8_q and store_is_older_1_q(8) and ( addr_same_1(8) or not stq_addr_valid_8_q );
	ld_st_conflict_1(9) <= stq_alloc_9_q and store_is_older_1_q(9) and ( addr_same_1(9) or not stq_addr_valid_9_q );
	ld_st_conflict_1(10) <= stq_alloc_10_q and store_is_older_1_q(10) and ( addr_same_1(10) or not stq_addr_valid_10_q );
	ld_st_conflict_1(11) <= stq_alloc_11_q and store_is_older_1_q(11) and ( addr_same_1(11) or not stq_addr_valid_11_q );
	ld_st_conflict_1(12) <= stq_alloc_12_q and store_is_older_1_q(12) and ( addr_same_1(12) or not stq_addr_valid_12_q );
	ld_st_conflict_1(13) <= stq_alloc_13_q and store_is_older_1_q(13) and ( addr_same_1(13) or not stq_addr_valid_13_q );
	ld_st_conflict_2(0) <= stq_alloc_0_q and store_is_older_2_q(0) and ( addr_same_2(0) or not stq_addr_valid_0_q );
	ld_st_conflict_2(1) <= stq_alloc_1_q and store_is_older_2_q(1) and ( addr_same_2(1) or not stq_addr_valid_1_q );
	ld_st_conflict_2(2) <= stq_alloc_2_q and store_is_older_2_q(2) and ( addr_same_2(2) or not stq_addr_valid_2_q );
	ld_st_conflict_2(3) <= stq_alloc_3_q and store_is_older_2_q(3) and ( addr_same_2(3) or not stq_addr_valid_3_q );
	ld_st_conflict_2(4) <= stq_alloc_4_q and store_is_older_2_q(4) and ( addr_same_2(4) or not stq_addr_valid_4_q );
	ld_st_conflict_2(5) <= stq_alloc_5_q and store_is_older_2_q(5) and ( addr_same_2(5) or not stq_addr_valid_5_q );
	ld_st_conflict_2(6) <= stq_alloc_6_q and store_is_older_2_q(6) and ( addr_same_2(6) or not stq_addr_valid_6_q );
	ld_st_conflict_2(7) <= stq_alloc_7_q and store_is_older_2_q(7) and ( addr_same_2(7) or not stq_addr_valid_7_q );
	ld_st_conflict_2(8) <= stq_alloc_8_q and store_is_older_2_q(8) and ( addr_same_2(8) or not stq_addr_valid_8_q );
	ld_st_conflict_2(9) <= stq_alloc_9_q and store_is_older_2_q(9) and ( addr_same_2(9) or not stq_addr_valid_9_q );
	ld_st_conflict_2(10) <= stq_alloc_10_q and store_is_older_2_q(10) and ( addr_same_2(10) or not stq_addr_valid_10_q );
	ld_st_conflict_2(11) <= stq_alloc_11_q and store_is_older_2_q(11) and ( addr_same_2(11) or not stq_addr_valid_11_q );
	ld_st_conflict_2(12) <= stq_alloc_12_q and store_is_older_2_q(12) and ( addr_same_2(12) or not stq_addr_valid_12_q );
	ld_st_conflict_2(13) <= stq_alloc_13_q and store_is_older_2_q(13) and ( addr_same_2(13) or not stq_addr_valid_13_q );
	ld_st_conflict_3(0) <= stq_alloc_0_q and store_is_older_3_q(0) and ( addr_same_3(0) or not stq_addr_valid_0_q );
	ld_st_conflict_3(1) <= stq_alloc_1_q and store_is_older_3_q(1) and ( addr_same_3(1) or not stq_addr_valid_1_q );
	ld_st_conflict_3(2) <= stq_alloc_2_q and store_is_older_3_q(2) and ( addr_same_3(2) or not stq_addr_valid_2_q );
	ld_st_conflict_3(3) <= stq_alloc_3_q and store_is_older_3_q(3) and ( addr_same_3(3) or not stq_addr_valid_3_q );
	ld_st_conflict_3(4) <= stq_alloc_4_q and store_is_older_3_q(4) and ( addr_same_3(4) or not stq_addr_valid_4_q );
	ld_st_conflict_3(5) <= stq_alloc_5_q and store_is_older_3_q(5) and ( addr_same_3(5) or not stq_addr_valid_5_q );
	ld_st_conflict_3(6) <= stq_alloc_6_q and store_is_older_3_q(6) and ( addr_same_3(6) or not stq_addr_valid_6_q );
	ld_st_conflict_3(7) <= stq_alloc_7_q and store_is_older_3_q(7) and ( addr_same_3(7) or not stq_addr_valid_7_q );
	ld_st_conflict_3(8) <= stq_alloc_8_q and store_is_older_3_q(8) and ( addr_same_3(8) or not stq_addr_valid_8_q );
	ld_st_conflict_3(9) <= stq_alloc_9_q and store_is_older_3_q(9) and ( addr_same_3(9) or not stq_addr_valid_9_q );
	ld_st_conflict_3(10) <= stq_alloc_10_q and store_is_older_3_q(10) and ( addr_same_3(10) or not stq_addr_valid_10_q );
	ld_st_conflict_3(11) <= stq_alloc_11_q and store_is_older_3_q(11) and ( addr_same_3(11) or not stq_addr_valid_11_q );
	ld_st_conflict_3(12) <= stq_alloc_12_q and store_is_older_3_q(12) and ( addr_same_3(12) or not stq_addr_valid_12_q );
	ld_st_conflict_3(13) <= stq_alloc_13_q and store_is_older_3_q(13) and ( addr_same_3(13) or not stq_addr_valid_13_q );
	ld_st_conflict_4(0) <= stq_alloc_0_q and store_is_older_4_q(0) and ( addr_same_4(0) or not stq_addr_valid_0_q );
	ld_st_conflict_4(1) <= stq_alloc_1_q and store_is_older_4_q(1) and ( addr_same_4(1) or not stq_addr_valid_1_q );
	ld_st_conflict_4(2) <= stq_alloc_2_q and store_is_older_4_q(2) and ( addr_same_4(2) or not stq_addr_valid_2_q );
	ld_st_conflict_4(3) <= stq_alloc_3_q and store_is_older_4_q(3) and ( addr_same_4(3) or not stq_addr_valid_3_q );
	ld_st_conflict_4(4) <= stq_alloc_4_q and store_is_older_4_q(4) and ( addr_same_4(4) or not stq_addr_valid_4_q );
	ld_st_conflict_4(5) <= stq_alloc_5_q and store_is_older_4_q(5) and ( addr_same_4(5) or not stq_addr_valid_5_q );
	ld_st_conflict_4(6) <= stq_alloc_6_q and store_is_older_4_q(6) and ( addr_same_4(6) or not stq_addr_valid_6_q );
	ld_st_conflict_4(7) <= stq_alloc_7_q and store_is_older_4_q(7) and ( addr_same_4(7) or not stq_addr_valid_7_q );
	ld_st_conflict_4(8) <= stq_alloc_8_q and store_is_older_4_q(8) and ( addr_same_4(8) or not stq_addr_valid_8_q );
	ld_st_conflict_4(9) <= stq_alloc_9_q and store_is_older_4_q(9) and ( addr_same_4(9) or not stq_addr_valid_9_q );
	ld_st_conflict_4(10) <= stq_alloc_10_q and store_is_older_4_q(10) and ( addr_same_4(10) or not stq_addr_valid_10_q );
	ld_st_conflict_4(11) <= stq_alloc_11_q and store_is_older_4_q(11) and ( addr_same_4(11) or not stq_addr_valid_11_q );
	ld_st_conflict_4(12) <= stq_alloc_12_q and store_is_older_4_q(12) and ( addr_same_4(12) or not stq_addr_valid_12_q );
	ld_st_conflict_4(13) <= stq_alloc_13_q and store_is_older_4_q(13) and ( addr_same_4(13) or not stq_addr_valid_13_q );
	ld_st_conflict_5(0) <= stq_alloc_0_q and store_is_older_5_q(0) and ( addr_same_5(0) or not stq_addr_valid_0_q );
	ld_st_conflict_5(1) <= stq_alloc_1_q and store_is_older_5_q(1) and ( addr_same_5(1) or not stq_addr_valid_1_q );
	ld_st_conflict_5(2) <= stq_alloc_2_q and store_is_older_5_q(2) and ( addr_same_5(2) or not stq_addr_valid_2_q );
	ld_st_conflict_5(3) <= stq_alloc_3_q and store_is_older_5_q(3) and ( addr_same_5(3) or not stq_addr_valid_3_q );
	ld_st_conflict_5(4) <= stq_alloc_4_q and store_is_older_5_q(4) and ( addr_same_5(4) or not stq_addr_valid_4_q );
	ld_st_conflict_5(5) <= stq_alloc_5_q and store_is_older_5_q(5) and ( addr_same_5(5) or not stq_addr_valid_5_q );
	ld_st_conflict_5(6) <= stq_alloc_6_q and store_is_older_5_q(6) and ( addr_same_5(6) or not stq_addr_valid_6_q );
	ld_st_conflict_5(7) <= stq_alloc_7_q and store_is_older_5_q(7) and ( addr_same_5(7) or not stq_addr_valid_7_q );
	ld_st_conflict_5(8) <= stq_alloc_8_q and store_is_older_5_q(8) and ( addr_same_5(8) or not stq_addr_valid_8_q );
	ld_st_conflict_5(9) <= stq_alloc_9_q and store_is_older_5_q(9) and ( addr_same_5(9) or not stq_addr_valid_9_q );
	ld_st_conflict_5(10) <= stq_alloc_10_q and store_is_older_5_q(10) and ( addr_same_5(10) or not stq_addr_valid_10_q );
	ld_st_conflict_5(11) <= stq_alloc_11_q and store_is_older_5_q(11) and ( addr_same_5(11) or not stq_addr_valid_11_q );
	ld_st_conflict_5(12) <= stq_alloc_12_q and store_is_older_5_q(12) and ( addr_same_5(12) or not stq_addr_valid_12_q );
	ld_st_conflict_5(13) <= stq_alloc_13_q and store_is_older_5_q(13) and ( addr_same_5(13) or not stq_addr_valid_13_q );
	ld_st_conflict_6(0) <= stq_alloc_0_q and store_is_older_6_q(0) and ( addr_same_6(0) or not stq_addr_valid_0_q );
	ld_st_conflict_6(1) <= stq_alloc_1_q and store_is_older_6_q(1) and ( addr_same_6(1) or not stq_addr_valid_1_q );
	ld_st_conflict_6(2) <= stq_alloc_2_q and store_is_older_6_q(2) and ( addr_same_6(2) or not stq_addr_valid_2_q );
	ld_st_conflict_6(3) <= stq_alloc_3_q and store_is_older_6_q(3) and ( addr_same_6(3) or not stq_addr_valid_3_q );
	ld_st_conflict_6(4) <= stq_alloc_4_q and store_is_older_6_q(4) and ( addr_same_6(4) or not stq_addr_valid_4_q );
	ld_st_conflict_6(5) <= stq_alloc_5_q and store_is_older_6_q(5) and ( addr_same_6(5) or not stq_addr_valid_5_q );
	ld_st_conflict_6(6) <= stq_alloc_6_q and store_is_older_6_q(6) and ( addr_same_6(6) or not stq_addr_valid_6_q );
	ld_st_conflict_6(7) <= stq_alloc_7_q and store_is_older_6_q(7) and ( addr_same_6(7) or not stq_addr_valid_7_q );
	ld_st_conflict_6(8) <= stq_alloc_8_q and store_is_older_6_q(8) and ( addr_same_6(8) or not stq_addr_valid_8_q );
	ld_st_conflict_6(9) <= stq_alloc_9_q and store_is_older_6_q(9) and ( addr_same_6(9) or not stq_addr_valid_9_q );
	ld_st_conflict_6(10) <= stq_alloc_10_q and store_is_older_6_q(10) and ( addr_same_6(10) or not stq_addr_valid_10_q );
	ld_st_conflict_6(11) <= stq_alloc_11_q and store_is_older_6_q(11) and ( addr_same_6(11) or not stq_addr_valid_11_q );
	ld_st_conflict_6(12) <= stq_alloc_12_q and store_is_older_6_q(12) and ( addr_same_6(12) or not stq_addr_valid_12_q );
	ld_st_conflict_6(13) <= stq_alloc_13_q and store_is_older_6_q(13) and ( addr_same_6(13) or not stq_addr_valid_13_q );
	ld_st_conflict_7(0) <= stq_alloc_0_q and store_is_older_7_q(0) and ( addr_same_7(0) or not stq_addr_valid_0_q );
	ld_st_conflict_7(1) <= stq_alloc_1_q and store_is_older_7_q(1) and ( addr_same_7(1) or not stq_addr_valid_1_q );
	ld_st_conflict_7(2) <= stq_alloc_2_q and store_is_older_7_q(2) and ( addr_same_7(2) or not stq_addr_valid_2_q );
	ld_st_conflict_7(3) <= stq_alloc_3_q and store_is_older_7_q(3) and ( addr_same_7(3) or not stq_addr_valid_3_q );
	ld_st_conflict_7(4) <= stq_alloc_4_q and store_is_older_7_q(4) and ( addr_same_7(4) or not stq_addr_valid_4_q );
	ld_st_conflict_7(5) <= stq_alloc_5_q and store_is_older_7_q(5) and ( addr_same_7(5) or not stq_addr_valid_5_q );
	ld_st_conflict_7(6) <= stq_alloc_6_q and store_is_older_7_q(6) and ( addr_same_7(6) or not stq_addr_valid_6_q );
	ld_st_conflict_7(7) <= stq_alloc_7_q and store_is_older_7_q(7) and ( addr_same_7(7) or not stq_addr_valid_7_q );
	ld_st_conflict_7(8) <= stq_alloc_8_q and store_is_older_7_q(8) and ( addr_same_7(8) or not stq_addr_valid_8_q );
	ld_st_conflict_7(9) <= stq_alloc_9_q and store_is_older_7_q(9) and ( addr_same_7(9) or not stq_addr_valid_9_q );
	ld_st_conflict_7(10) <= stq_alloc_10_q and store_is_older_7_q(10) and ( addr_same_7(10) or not stq_addr_valid_10_q );
	ld_st_conflict_7(11) <= stq_alloc_11_q and store_is_older_7_q(11) and ( addr_same_7(11) or not stq_addr_valid_11_q );
	ld_st_conflict_7(12) <= stq_alloc_12_q and store_is_older_7_q(12) and ( addr_same_7(12) or not stq_addr_valid_12_q );
	ld_st_conflict_7(13) <= stq_alloc_13_q and store_is_older_7_q(13) and ( addr_same_7(13) or not stq_addr_valid_13_q );
	ld_st_conflict_8(0) <= stq_alloc_0_q and store_is_older_8_q(0) and ( addr_same_8(0) or not stq_addr_valid_0_q );
	ld_st_conflict_8(1) <= stq_alloc_1_q and store_is_older_8_q(1) and ( addr_same_8(1) or not stq_addr_valid_1_q );
	ld_st_conflict_8(2) <= stq_alloc_2_q and store_is_older_8_q(2) and ( addr_same_8(2) or not stq_addr_valid_2_q );
	ld_st_conflict_8(3) <= stq_alloc_3_q and store_is_older_8_q(3) and ( addr_same_8(3) or not stq_addr_valid_3_q );
	ld_st_conflict_8(4) <= stq_alloc_4_q and store_is_older_8_q(4) and ( addr_same_8(4) or not stq_addr_valid_4_q );
	ld_st_conflict_8(5) <= stq_alloc_5_q and store_is_older_8_q(5) and ( addr_same_8(5) or not stq_addr_valid_5_q );
	ld_st_conflict_8(6) <= stq_alloc_6_q and store_is_older_8_q(6) and ( addr_same_8(6) or not stq_addr_valid_6_q );
	ld_st_conflict_8(7) <= stq_alloc_7_q and store_is_older_8_q(7) and ( addr_same_8(7) or not stq_addr_valid_7_q );
	ld_st_conflict_8(8) <= stq_alloc_8_q and store_is_older_8_q(8) and ( addr_same_8(8) or not stq_addr_valid_8_q );
	ld_st_conflict_8(9) <= stq_alloc_9_q and store_is_older_8_q(9) and ( addr_same_8(9) or not stq_addr_valid_9_q );
	ld_st_conflict_8(10) <= stq_alloc_10_q and store_is_older_8_q(10) and ( addr_same_8(10) or not stq_addr_valid_10_q );
	ld_st_conflict_8(11) <= stq_alloc_11_q and store_is_older_8_q(11) and ( addr_same_8(11) or not stq_addr_valid_11_q );
	ld_st_conflict_8(12) <= stq_alloc_12_q and store_is_older_8_q(12) and ( addr_same_8(12) or not stq_addr_valid_12_q );
	ld_st_conflict_8(13) <= stq_alloc_13_q and store_is_older_8_q(13) and ( addr_same_8(13) or not stq_addr_valid_13_q );
	ld_st_conflict_9(0) <= stq_alloc_0_q and store_is_older_9_q(0) and ( addr_same_9(0) or not stq_addr_valid_0_q );
	ld_st_conflict_9(1) <= stq_alloc_1_q and store_is_older_9_q(1) and ( addr_same_9(1) or not stq_addr_valid_1_q );
	ld_st_conflict_9(2) <= stq_alloc_2_q and store_is_older_9_q(2) and ( addr_same_9(2) or not stq_addr_valid_2_q );
	ld_st_conflict_9(3) <= stq_alloc_3_q and store_is_older_9_q(3) and ( addr_same_9(3) or not stq_addr_valid_3_q );
	ld_st_conflict_9(4) <= stq_alloc_4_q and store_is_older_9_q(4) and ( addr_same_9(4) or not stq_addr_valid_4_q );
	ld_st_conflict_9(5) <= stq_alloc_5_q and store_is_older_9_q(5) and ( addr_same_9(5) or not stq_addr_valid_5_q );
	ld_st_conflict_9(6) <= stq_alloc_6_q and store_is_older_9_q(6) and ( addr_same_9(6) or not stq_addr_valid_6_q );
	ld_st_conflict_9(7) <= stq_alloc_7_q and store_is_older_9_q(7) and ( addr_same_9(7) or not stq_addr_valid_7_q );
	ld_st_conflict_9(8) <= stq_alloc_8_q and store_is_older_9_q(8) and ( addr_same_9(8) or not stq_addr_valid_8_q );
	ld_st_conflict_9(9) <= stq_alloc_9_q and store_is_older_9_q(9) and ( addr_same_9(9) or not stq_addr_valid_9_q );
	ld_st_conflict_9(10) <= stq_alloc_10_q and store_is_older_9_q(10) and ( addr_same_9(10) or not stq_addr_valid_10_q );
	ld_st_conflict_9(11) <= stq_alloc_11_q and store_is_older_9_q(11) and ( addr_same_9(11) or not stq_addr_valid_11_q );
	ld_st_conflict_9(12) <= stq_alloc_12_q and store_is_older_9_q(12) and ( addr_same_9(12) or not stq_addr_valid_12_q );
	ld_st_conflict_9(13) <= stq_alloc_13_q and store_is_older_9_q(13) and ( addr_same_9(13) or not stq_addr_valid_13_q );
	ld_st_conflict_10(0) <= stq_alloc_0_q and store_is_older_10_q(0) and ( addr_same_10(0) or not stq_addr_valid_0_q );
	ld_st_conflict_10(1) <= stq_alloc_1_q and store_is_older_10_q(1) and ( addr_same_10(1) or not stq_addr_valid_1_q );
	ld_st_conflict_10(2) <= stq_alloc_2_q and store_is_older_10_q(2) and ( addr_same_10(2) or not stq_addr_valid_2_q );
	ld_st_conflict_10(3) <= stq_alloc_3_q and store_is_older_10_q(3) and ( addr_same_10(3) or not stq_addr_valid_3_q );
	ld_st_conflict_10(4) <= stq_alloc_4_q and store_is_older_10_q(4) and ( addr_same_10(4) or not stq_addr_valid_4_q );
	ld_st_conflict_10(5) <= stq_alloc_5_q and store_is_older_10_q(5) and ( addr_same_10(5) or not stq_addr_valid_5_q );
	ld_st_conflict_10(6) <= stq_alloc_6_q and store_is_older_10_q(6) and ( addr_same_10(6) or not stq_addr_valid_6_q );
	ld_st_conflict_10(7) <= stq_alloc_7_q and store_is_older_10_q(7) and ( addr_same_10(7) or not stq_addr_valid_7_q );
	ld_st_conflict_10(8) <= stq_alloc_8_q and store_is_older_10_q(8) and ( addr_same_10(8) or not stq_addr_valid_8_q );
	ld_st_conflict_10(9) <= stq_alloc_9_q and store_is_older_10_q(9) and ( addr_same_10(9) or not stq_addr_valid_9_q );
	ld_st_conflict_10(10) <= stq_alloc_10_q and store_is_older_10_q(10) and ( addr_same_10(10) or not stq_addr_valid_10_q );
	ld_st_conflict_10(11) <= stq_alloc_11_q and store_is_older_10_q(11) and ( addr_same_10(11) or not stq_addr_valid_11_q );
	ld_st_conflict_10(12) <= stq_alloc_12_q and store_is_older_10_q(12) and ( addr_same_10(12) or not stq_addr_valid_12_q );
	ld_st_conflict_10(13) <= stq_alloc_13_q and store_is_older_10_q(13) and ( addr_same_10(13) or not stq_addr_valid_13_q );
	ld_st_conflict_11(0) <= stq_alloc_0_q and store_is_older_11_q(0) and ( addr_same_11(0) or not stq_addr_valid_0_q );
	ld_st_conflict_11(1) <= stq_alloc_1_q and store_is_older_11_q(1) and ( addr_same_11(1) or not stq_addr_valid_1_q );
	ld_st_conflict_11(2) <= stq_alloc_2_q and store_is_older_11_q(2) and ( addr_same_11(2) or not stq_addr_valid_2_q );
	ld_st_conflict_11(3) <= stq_alloc_3_q and store_is_older_11_q(3) and ( addr_same_11(3) or not stq_addr_valid_3_q );
	ld_st_conflict_11(4) <= stq_alloc_4_q and store_is_older_11_q(4) and ( addr_same_11(4) or not stq_addr_valid_4_q );
	ld_st_conflict_11(5) <= stq_alloc_5_q and store_is_older_11_q(5) and ( addr_same_11(5) or not stq_addr_valid_5_q );
	ld_st_conflict_11(6) <= stq_alloc_6_q and store_is_older_11_q(6) and ( addr_same_11(6) or not stq_addr_valid_6_q );
	ld_st_conflict_11(7) <= stq_alloc_7_q and store_is_older_11_q(7) and ( addr_same_11(7) or not stq_addr_valid_7_q );
	ld_st_conflict_11(8) <= stq_alloc_8_q and store_is_older_11_q(8) and ( addr_same_11(8) or not stq_addr_valid_8_q );
	ld_st_conflict_11(9) <= stq_alloc_9_q and store_is_older_11_q(9) and ( addr_same_11(9) or not stq_addr_valid_9_q );
	ld_st_conflict_11(10) <= stq_alloc_10_q and store_is_older_11_q(10) and ( addr_same_11(10) or not stq_addr_valid_10_q );
	ld_st_conflict_11(11) <= stq_alloc_11_q and store_is_older_11_q(11) and ( addr_same_11(11) or not stq_addr_valid_11_q );
	ld_st_conflict_11(12) <= stq_alloc_12_q and store_is_older_11_q(12) and ( addr_same_11(12) or not stq_addr_valid_12_q );
	ld_st_conflict_11(13) <= stq_alloc_13_q and store_is_older_11_q(13) and ( addr_same_11(13) or not stq_addr_valid_13_q );
	ld_st_conflict_12(0) <= stq_alloc_0_q and store_is_older_12_q(0) and ( addr_same_12(0) or not stq_addr_valid_0_q );
	ld_st_conflict_12(1) <= stq_alloc_1_q and store_is_older_12_q(1) and ( addr_same_12(1) or not stq_addr_valid_1_q );
	ld_st_conflict_12(2) <= stq_alloc_2_q and store_is_older_12_q(2) and ( addr_same_12(2) or not stq_addr_valid_2_q );
	ld_st_conflict_12(3) <= stq_alloc_3_q and store_is_older_12_q(3) and ( addr_same_12(3) or not stq_addr_valid_3_q );
	ld_st_conflict_12(4) <= stq_alloc_4_q and store_is_older_12_q(4) and ( addr_same_12(4) or not stq_addr_valid_4_q );
	ld_st_conflict_12(5) <= stq_alloc_5_q and store_is_older_12_q(5) and ( addr_same_12(5) or not stq_addr_valid_5_q );
	ld_st_conflict_12(6) <= stq_alloc_6_q and store_is_older_12_q(6) and ( addr_same_12(6) or not stq_addr_valid_6_q );
	ld_st_conflict_12(7) <= stq_alloc_7_q and store_is_older_12_q(7) and ( addr_same_12(7) or not stq_addr_valid_7_q );
	ld_st_conflict_12(8) <= stq_alloc_8_q and store_is_older_12_q(8) and ( addr_same_12(8) or not stq_addr_valid_8_q );
	ld_st_conflict_12(9) <= stq_alloc_9_q and store_is_older_12_q(9) and ( addr_same_12(9) or not stq_addr_valid_9_q );
	ld_st_conflict_12(10) <= stq_alloc_10_q and store_is_older_12_q(10) and ( addr_same_12(10) or not stq_addr_valid_10_q );
	ld_st_conflict_12(11) <= stq_alloc_11_q and store_is_older_12_q(11) and ( addr_same_12(11) or not stq_addr_valid_11_q );
	ld_st_conflict_12(12) <= stq_alloc_12_q and store_is_older_12_q(12) and ( addr_same_12(12) or not stq_addr_valid_12_q );
	ld_st_conflict_12(13) <= stq_alloc_13_q and store_is_older_12_q(13) and ( addr_same_12(13) or not stq_addr_valid_13_q );
	ld_st_conflict_13(0) <= stq_alloc_0_q and store_is_older_13_q(0) and ( addr_same_13(0) or not stq_addr_valid_0_q );
	ld_st_conflict_13(1) <= stq_alloc_1_q and store_is_older_13_q(1) and ( addr_same_13(1) or not stq_addr_valid_1_q );
	ld_st_conflict_13(2) <= stq_alloc_2_q and store_is_older_13_q(2) and ( addr_same_13(2) or not stq_addr_valid_2_q );
	ld_st_conflict_13(3) <= stq_alloc_3_q and store_is_older_13_q(3) and ( addr_same_13(3) or not stq_addr_valid_3_q );
	ld_st_conflict_13(4) <= stq_alloc_4_q and store_is_older_13_q(4) and ( addr_same_13(4) or not stq_addr_valid_4_q );
	ld_st_conflict_13(5) <= stq_alloc_5_q and store_is_older_13_q(5) and ( addr_same_13(5) or not stq_addr_valid_5_q );
	ld_st_conflict_13(6) <= stq_alloc_6_q and store_is_older_13_q(6) and ( addr_same_13(6) or not stq_addr_valid_6_q );
	ld_st_conflict_13(7) <= stq_alloc_7_q and store_is_older_13_q(7) and ( addr_same_13(7) or not stq_addr_valid_7_q );
	ld_st_conflict_13(8) <= stq_alloc_8_q and store_is_older_13_q(8) and ( addr_same_13(8) or not stq_addr_valid_8_q );
	ld_st_conflict_13(9) <= stq_alloc_9_q and store_is_older_13_q(9) and ( addr_same_13(9) or not stq_addr_valid_9_q );
	ld_st_conflict_13(10) <= stq_alloc_10_q and store_is_older_13_q(10) and ( addr_same_13(10) or not stq_addr_valid_10_q );
	ld_st_conflict_13(11) <= stq_alloc_11_q and store_is_older_13_q(11) and ( addr_same_13(11) or not stq_addr_valid_11_q );
	ld_st_conflict_13(12) <= stq_alloc_12_q and store_is_older_13_q(12) and ( addr_same_13(12) or not stq_addr_valid_12_q );
	ld_st_conflict_13(13) <= stq_alloc_13_q and store_is_older_13_q(13) and ( addr_same_13(13) or not stq_addr_valid_13_q );
	can_bypass_0(0) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_0_q and addr_same_0(0) and addr_valid_0(0);
	can_bypass_0(1) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_1_q and addr_same_0(1) and addr_valid_0(1);
	can_bypass_0(2) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_2_q and addr_same_0(2) and addr_valid_0(2);
	can_bypass_0(3) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_3_q and addr_same_0(3) and addr_valid_0(3);
	can_bypass_0(4) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_4_q and addr_same_0(4) and addr_valid_0(4);
	can_bypass_0(5) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_5_q and addr_same_0(5) and addr_valid_0(5);
	can_bypass_0(6) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_6_q and addr_same_0(6) and addr_valid_0(6);
	can_bypass_0(7) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_7_q and addr_same_0(7) and addr_valid_0(7);
	can_bypass_0(8) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_8_q and addr_same_0(8) and addr_valid_0(8);
	can_bypass_0(9) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_9_q and addr_same_0(9) and addr_valid_0(9);
	can_bypass_0(10) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_10_q and addr_same_0(10) and addr_valid_0(10);
	can_bypass_0(11) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_11_q and addr_same_0(11) and addr_valid_0(11);
	can_bypass_0(12) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_12_q and addr_same_0(12) and addr_valid_0(12);
	can_bypass_0(13) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_13_q and addr_same_0(13) and addr_valid_0(13);
	can_bypass_1(0) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_0_q and addr_same_1(0) and addr_valid_1(0);
	can_bypass_1(1) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_1_q and addr_same_1(1) and addr_valid_1(1);
	can_bypass_1(2) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_2_q and addr_same_1(2) and addr_valid_1(2);
	can_bypass_1(3) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_3_q and addr_same_1(3) and addr_valid_1(3);
	can_bypass_1(4) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_4_q and addr_same_1(4) and addr_valid_1(4);
	can_bypass_1(5) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_5_q and addr_same_1(5) and addr_valid_1(5);
	can_bypass_1(6) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_6_q and addr_same_1(6) and addr_valid_1(6);
	can_bypass_1(7) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_7_q and addr_same_1(7) and addr_valid_1(7);
	can_bypass_1(8) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_8_q and addr_same_1(8) and addr_valid_1(8);
	can_bypass_1(9) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_9_q and addr_same_1(9) and addr_valid_1(9);
	can_bypass_1(10) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_10_q and addr_same_1(10) and addr_valid_1(10);
	can_bypass_1(11) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_11_q and addr_same_1(11) and addr_valid_1(11);
	can_bypass_1(12) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_12_q and addr_same_1(12) and addr_valid_1(12);
	can_bypass_1(13) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_13_q and addr_same_1(13) and addr_valid_1(13);
	can_bypass_2(0) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_0_q and addr_same_2(0) and addr_valid_2(0);
	can_bypass_2(1) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_1_q and addr_same_2(1) and addr_valid_2(1);
	can_bypass_2(2) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_2_q and addr_same_2(2) and addr_valid_2(2);
	can_bypass_2(3) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_3_q and addr_same_2(3) and addr_valid_2(3);
	can_bypass_2(4) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_4_q and addr_same_2(4) and addr_valid_2(4);
	can_bypass_2(5) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_5_q and addr_same_2(5) and addr_valid_2(5);
	can_bypass_2(6) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_6_q and addr_same_2(6) and addr_valid_2(6);
	can_bypass_2(7) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_7_q and addr_same_2(7) and addr_valid_2(7);
	can_bypass_2(8) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_8_q and addr_same_2(8) and addr_valid_2(8);
	can_bypass_2(9) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_9_q and addr_same_2(9) and addr_valid_2(9);
	can_bypass_2(10) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_10_q and addr_same_2(10) and addr_valid_2(10);
	can_bypass_2(11) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_11_q and addr_same_2(11) and addr_valid_2(11);
	can_bypass_2(12) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_12_q and addr_same_2(12) and addr_valid_2(12);
	can_bypass_2(13) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_13_q and addr_same_2(13) and addr_valid_2(13);
	can_bypass_3(0) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_0_q and addr_same_3(0) and addr_valid_3(0);
	can_bypass_3(1) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_1_q and addr_same_3(1) and addr_valid_3(1);
	can_bypass_3(2) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_2_q and addr_same_3(2) and addr_valid_3(2);
	can_bypass_3(3) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_3_q and addr_same_3(3) and addr_valid_3(3);
	can_bypass_3(4) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_4_q and addr_same_3(4) and addr_valid_3(4);
	can_bypass_3(5) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_5_q and addr_same_3(5) and addr_valid_3(5);
	can_bypass_3(6) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_6_q and addr_same_3(6) and addr_valid_3(6);
	can_bypass_3(7) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_7_q and addr_same_3(7) and addr_valid_3(7);
	can_bypass_3(8) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_8_q and addr_same_3(8) and addr_valid_3(8);
	can_bypass_3(9) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_9_q and addr_same_3(9) and addr_valid_3(9);
	can_bypass_3(10) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_10_q and addr_same_3(10) and addr_valid_3(10);
	can_bypass_3(11) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_11_q and addr_same_3(11) and addr_valid_3(11);
	can_bypass_3(12) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_12_q and addr_same_3(12) and addr_valid_3(12);
	can_bypass_3(13) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_13_q and addr_same_3(13) and addr_valid_3(13);
	can_bypass_4(0) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_0_q and addr_same_4(0) and addr_valid_4(0);
	can_bypass_4(1) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_1_q and addr_same_4(1) and addr_valid_4(1);
	can_bypass_4(2) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_2_q and addr_same_4(2) and addr_valid_4(2);
	can_bypass_4(3) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_3_q and addr_same_4(3) and addr_valid_4(3);
	can_bypass_4(4) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_4_q and addr_same_4(4) and addr_valid_4(4);
	can_bypass_4(5) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_5_q and addr_same_4(5) and addr_valid_4(5);
	can_bypass_4(6) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_6_q and addr_same_4(6) and addr_valid_4(6);
	can_bypass_4(7) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_7_q and addr_same_4(7) and addr_valid_4(7);
	can_bypass_4(8) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_8_q and addr_same_4(8) and addr_valid_4(8);
	can_bypass_4(9) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_9_q and addr_same_4(9) and addr_valid_4(9);
	can_bypass_4(10) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_10_q and addr_same_4(10) and addr_valid_4(10);
	can_bypass_4(11) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_11_q and addr_same_4(11) and addr_valid_4(11);
	can_bypass_4(12) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_12_q and addr_same_4(12) and addr_valid_4(12);
	can_bypass_4(13) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_13_q and addr_same_4(13) and addr_valid_4(13);
	can_bypass_5(0) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_0_q and addr_same_5(0) and addr_valid_5(0);
	can_bypass_5(1) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_1_q and addr_same_5(1) and addr_valid_5(1);
	can_bypass_5(2) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_2_q and addr_same_5(2) and addr_valid_5(2);
	can_bypass_5(3) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_3_q and addr_same_5(3) and addr_valid_5(3);
	can_bypass_5(4) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_4_q and addr_same_5(4) and addr_valid_5(4);
	can_bypass_5(5) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_5_q and addr_same_5(5) and addr_valid_5(5);
	can_bypass_5(6) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_6_q and addr_same_5(6) and addr_valid_5(6);
	can_bypass_5(7) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_7_q and addr_same_5(7) and addr_valid_5(7);
	can_bypass_5(8) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_8_q and addr_same_5(8) and addr_valid_5(8);
	can_bypass_5(9) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_9_q and addr_same_5(9) and addr_valid_5(9);
	can_bypass_5(10) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_10_q and addr_same_5(10) and addr_valid_5(10);
	can_bypass_5(11) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_11_q and addr_same_5(11) and addr_valid_5(11);
	can_bypass_5(12) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_12_q and addr_same_5(12) and addr_valid_5(12);
	can_bypass_5(13) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_13_q and addr_same_5(13) and addr_valid_5(13);
	can_bypass_6(0) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_0_q and addr_same_6(0) and addr_valid_6(0);
	can_bypass_6(1) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_1_q and addr_same_6(1) and addr_valid_6(1);
	can_bypass_6(2) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_2_q and addr_same_6(2) and addr_valid_6(2);
	can_bypass_6(3) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_3_q and addr_same_6(3) and addr_valid_6(3);
	can_bypass_6(4) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_4_q and addr_same_6(4) and addr_valid_6(4);
	can_bypass_6(5) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_5_q and addr_same_6(5) and addr_valid_6(5);
	can_bypass_6(6) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_6_q and addr_same_6(6) and addr_valid_6(6);
	can_bypass_6(7) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_7_q and addr_same_6(7) and addr_valid_6(7);
	can_bypass_6(8) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_8_q and addr_same_6(8) and addr_valid_6(8);
	can_bypass_6(9) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_9_q and addr_same_6(9) and addr_valid_6(9);
	can_bypass_6(10) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_10_q and addr_same_6(10) and addr_valid_6(10);
	can_bypass_6(11) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_11_q and addr_same_6(11) and addr_valid_6(11);
	can_bypass_6(12) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_12_q and addr_same_6(12) and addr_valid_6(12);
	can_bypass_6(13) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_13_q and addr_same_6(13) and addr_valid_6(13);
	can_bypass_7(0) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_0_q and addr_same_7(0) and addr_valid_7(0);
	can_bypass_7(1) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_1_q and addr_same_7(1) and addr_valid_7(1);
	can_bypass_7(2) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_2_q and addr_same_7(2) and addr_valid_7(2);
	can_bypass_7(3) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_3_q and addr_same_7(3) and addr_valid_7(3);
	can_bypass_7(4) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_4_q and addr_same_7(4) and addr_valid_7(4);
	can_bypass_7(5) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_5_q and addr_same_7(5) and addr_valid_7(5);
	can_bypass_7(6) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_6_q and addr_same_7(6) and addr_valid_7(6);
	can_bypass_7(7) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_7_q and addr_same_7(7) and addr_valid_7(7);
	can_bypass_7(8) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_8_q and addr_same_7(8) and addr_valid_7(8);
	can_bypass_7(9) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_9_q and addr_same_7(9) and addr_valid_7(9);
	can_bypass_7(10) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_10_q and addr_same_7(10) and addr_valid_7(10);
	can_bypass_7(11) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_11_q and addr_same_7(11) and addr_valid_7(11);
	can_bypass_7(12) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_12_q and addr_same_7(12) and addr_valid_7(12);
	can_bypass_7(13) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_13_q and addr_same_7(13) and addr_valid_7(13);
	can_bypass_8(0) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_0_q and addr_same_8(0) and addr_valid_8(0);
	can_bypass_8(1) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_1_q and addr_same_8(1) and addr_valid_8(1);
	can_bypass_8(2) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_2_q and addr_same_8(2) and addr_valid_8(2);
	can_bypass_8(3) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_3_q and addr_same_8(3) and addr_valid_8(3);
	can_bypass_8(4) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_4_q and addr_same_8(4) and addr_valid_8(4);
	can_bypass_8(5) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_5_q and addr_same_8(5) and addr_valid_8(5);
	can_bypass_8(6) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_6_q and addr_same_8(6) and addr_valid_8(6);
	can_bypass_8(7) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_7_q and addr_same_8(7) and addr_valid_8(7);
	can_bypass_8(8) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_8_q and addr_same_8(8) and addr_valid_8(8);
	can_bypass_8(9) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_9_q and addr_same_8(9) and addr_valid_8(9);
	can_bypass_8(10) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_10_q and addr_same_8(10) and addr_valid_8(10);
	can_bypass_8(11) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_11_q and addr_same_8(11) and addr_valid_8(11);
	can_bypass_8(12) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_12_q and addr_same_8(12) and addr_valid_8(12);
	can_bypass_8(13) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_13_q and addr_same_8(13) and addr_valid_8(13);
	can_bypass_9(0) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_0_q and addr_same_9(0) and addr_valid_9(0);
	can_bypass_9(1) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_1_q and addr_same_9(1) and addr_valid_9(1);
	can_bypass_9(2) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_2_q and addr_same_9(2) and addr_valid_9(2);
	can_bypass_9(3) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_3_q and addr_same_9(3) and addr_valid_9(3);
	can_bypass_9(4) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_4_q and addr_same_9(4) and addr_valid_9(4);
	can_bypass_9(5) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_5_q and addr_same_9(5) and addr_valid_9(5);
	can_bypass_9(6) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_6_q and addr_same_9(6) and addr_valid_9(6);
	can_bypass_9(7) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_7_q and addr_same_9(7) and addr_valid_9(7);
	can_bypass_9(8) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_8_q and addr_same_9(8) and addr_valid_9(8);
	can_bypass_9(9) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_9_q and addr_same_9(9) and addr_valid_9(9);
	can_bypass_9(10) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_10_q and addr_same_9(10) and addr_valid_9(10);
	can_bypass_9(11) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_11_q and addr_same_9(11) and addr_valid_9(11);
	can_bypass_9(12) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_12_q and addr_same_9(12) and addr_valid_9(12);
	can_bypass_9(13) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_13_q and addr_same_9(13) and addr_valid_9(13);
	can_bypass_10(0) <= ldq_alloc_10_q and not ldq_issue_10_q and stq_data_valid_0_q and addr_same_10(0) and addr_valid_10(0);
	can_bypass_10(1) <= ldq_alloc_10_q and not ldq_issue_10_q and stq_data_valid_1_q and addr_same_10(1) and addr_valid_10(1);
	can_bypass_10(2) <= ldq_alloc_10_q and not ldq_issue_10_q and stq_data_valid_2_q and addr_same_10(2) and addr_valid_10(2);
	can_bypass_10(3) <= ldq_alloc_10_q and not ldq_issue_10_q and stq_data_valid_3_q and addr_same_10(3) and addr_valid_10(3);
	can_bypass_10(4) <= ldq_alloc_10_q and not ldq_issue_10_q and stq_data_valid_4_q and addr_same_10(4) and addr_valid_10(4);
	can_bypass_10(5) <= ldq_alloc_10_q and not ldq_issue_10_q and stq_data_valid_5_q and addr_same_10(5) and addr_valid_10(5);
	can_bypass_10(6) <= ldq_alloc_10_q and not ldq_issue_10_q and stq_data_valid_6_q and addr_same_10(6) and addr_valid_10(6);
	can_bypass_10(7) <= ldq_alloc_10_q and not ldq_issue_10_q and stq_data_valid_7_q and addr_same_10(7) and addr_valid_10(7);
	can_bypass_10(8) <= ldq_alloc_10_q and not ldq_issue_10_q and stq_data_valid_8_q and addr_same_10(8) and addr_valid_10(8);
	can_bypass_10(9) <= ldq_alloc_10_q and not ldq_issue_10_q and stq_data_valid_9_q and addr_same_10(9) and addr_valid_10(9);
	can_bypass_10(10) <= ldq_alloc_10_q and not ldq_issue_10_q and stq_data_valid_10_q and addr_same_10(10) and addr_valid_10(10);
	can_bypass_10(11) <= ldq_alloc_10_q and not ldq_issue_10_q and stq_data_valid_11_q and addr_same_10(11) and addr_valid_10(11);
	can_bypass_10(12) <= ldq_alloc_10_q and not ldq_issue_10_q and stq_data_valid_12_q and addr_same_10(12) and addr_valid_10(12);
	can_bypass_10(13) <= ldq_alloc_10_q and not ldq_issue_10_q and stq_data_valid_13_q and addr_same_10(13) and addr_valid_10(13);
	can_bypass_11(0) <= ldq_alloc_11_q and not ldq_issue_11_q and stq_data_valid_0_q and addr_same_11(0) and addr_valid_11(0);
	can_bypass_11(1) <= ldq_alloc_11_q and not ldq_issue_11_q and stq_data_valid_1_q and addr_same_11(1) and addr_valid_11(1);
	can_bypass_11(2) <= ldq_alloc_11_q and not ldq_issue_11_q and stq_data_valid_2_q and addr_same_11(2) and addr_valid_11(2);
	can_bypass_11(3) <= ldq_alloc_11_q and not ldq_issue_11_q and stq_data_valid_3_q and addr_same_11(3) and addr_valid_11(3);
	can_bypass_11(4) <= ldq_alloc_11_q and not ldq_issue_11_q and stq_data_valid_4_q and addr_same_11(4) and addr_valid_11(4);
	can_bypass_11(5) <= ldq_alloc_11_q and not ldq_issue_11_q and stq_data_valid_5_q and addr_same_11(5) and addr_valid_11(5);
	can_bypass_11(6) <= ldq_alloc_11_q and not ldq_issue_11_q and stq_data_valid_6_q and addr_same_11(6) and addr_valid_11(6);
	can_bypass_11(7) <= ldq_alloc_11_q and not ldq_issue_11_q and stq_data_valid_7_q and addr_same_11(7) and addr_valid_11(7);
	can_bypass_11(8) <= ldq_alloc_11_q and not ldq_issue_11_q and stq_data_valid_8_q and addr_same_11(8) and addr_valid_11(8);
	can_bypass_11(9) <= ldq_alloc_11_q and not ldq_issue_11_q and stq_data_valid_9_q and addr_same_11(9) and addr_valid_11(9);
	can_bypass_11(10) <= ldq_alloc_11_q and not ldq_issue_11_q and stq_data_valid_10_q and addr_same_11(10) and addr_valid_11(10);
	can_bypass_11(11) <= ldq_alloc_11_q and not ldq_issue_11_q and stq_data_valid_11_q and addr_same_11(11) and addr_valid_11(11);
	can_bypass_11(12) <= ldq_alloc_11_q and not ldq_issue_11_q and stq_data_valid_12_q and addr_same_11(12) and addr_valid_11(12);
	can_bypass_11(13) <= ldq_alloc_11_q and not ldq_issue_11_q and stq_data_valid_13_q and addr_same_11(13) and addr_valid_11(13);
	can_bypass_12(0) <= ldq_alloc_12_q and not ldq_issue_12_q and stq_data_valid_0_q and addr_same_12(0) and addr_valid_12(0);
	can_bypass_12(1) <= ldq_alloc_12_q and not ldq_issue_12_q and stq_data_valid_1_q and addr_same_12(1) and addr_valid_12(1);
	can_bypass_12(2) <= ldq_alloc_12_q and not ldq_issue_12_q and stq_data_valid_2_q and addr_same_12(2) and addr_valid_12(2);
	can_bypass_12(3) <= ldq_alloc_12_q and not ldq_issue_12_q and stq_data_valid_3_q and addr_same_12(3) and addr_valid_12(3);
	can_bypass_12(4) <= ldq_alloc_12_q and not ldq_issue_12_q and stq_data_valid_4_q and addr_same_12(4) and addr_valid_12(4);
	can_bypass_12(5) <= ldq_alloc_12_q and not ldq_issue_12_q and stq_data_valid_5_q and addr_same_12(5) and addr_valid_12(5);
	can_bypass_12(6) <= ldq_alloc_12_q and not ldq_issue_12_q and stq_data_valid_6_q and addr_same_12(6) and addr_valid_12(6);
	can_bypass_12(7) <= ldq_alloc_12_q and not ldq_issue_12_q and stq_data_valid_7_q and addr_same_12(7) and addr_valid_12(7);
	can_bypass_12(8) <= ldq_alloc_12_q and not ldq_issue_12_q and stq_data_valid_8_q and addr_same_12(8) and addr_valid_12(8);
	can_bypass_12(9) <= ldq_alloc_12_q and not ldq_issue_12_q and stq_data_valid_9_q and addr_same_12(9) and addr_valid_12(9);
	can_bypass_12(10) <= ldq_alloc_12_q and not ldq_issue_12_q and stq_data_valid_10_q and addr_same_12(10) and addr_valid_12(10);
	can_bypass_12(11) <= ldq_alloc_12_q and not ldq_issue_12_q and stq_data_valid_11_q and addr_same_12(11) and addr_valid_12(11);
	can_bypass_12(12) <= ldq_alloc_12_q and not ldq_issue_12_q and stq_data_valid_12_q and addr_same_12(12) and addr_valid_12(12);
	can_bypass_12(13) <= ldq_alloc_12_q and not ldq_issue_12_q and stq_data_valid_13_q and addr_same_12(13) and addr_valid_12(13);
	can_bypass_13(0) <= ldq_alloc_13_q and not ldq_issue_13_q and stq_data_valid_0_q and addr_same_13(0) and addr_valid_13(0);
	can_bypass_13(1) <= ldq_alloc_13_q and not ldq_issue_13_q and stq_data_valid_1_q and addr_same_13(1) and addr_valid_13(1);
	can_bypass_13(2) <= ldq_alloc_13_q and not ldq_issue_13_q and stq_data_valid_2_q and addr_same_13(2) and addr_valid_13(2);
	can_bypass_13(3) <= ldq_alloc_13_q and not ldq_issue_13_q and stq_data_valid_3_q and addr_same_13(3) and addr_valid_13(3);
	can_bypass_13(4) <= ldq_alloc_13_q and not ldq_issue_13_q and stq_data_valid_4_q and addr_same_13(4) and addr_valid_13(4);
	can_bypass_13(5) <= ldq_alloc_13_q and not ldq_issue_13_q and stq_data_valid_5_q and addr_same_13(5) and addr_valid_13(5);
	can_bypass_13(6) <= ldq_alloc_13_q and not ldq_issue_13_q and stq_data_valid_6_q and addr_same_13(6) and addr_valid_13(6);
	can_bypass_13(7) <= ldq_alloc_13_q and not ldq_issue_13_q and stq_data_valid_7_q and addr_same_13(7) and addr_valid_13(7);
	can_bypass_13(8) <= ldq_alloc_13_q and not ldq_issue_13_q and stq_data_valid_8_q and addr_same_13(8) and addr_valid_13(8);
	can_bypass_13(9) <= ldq_alloc_13_q and not ldq_issue_13_q and stq_data_valid_9_q and addr_same_13(9) and addr_valid_13(9);
	can_bypass_13(10) <= ldq_alloc_13_q and not ldq_issue_13_q and stq_data_valid_10_q and addr_same_13(10) and addr_valid_13(10);
	can_bypass_13(11) <= ldq_alloc_13_q and not ldq_issue_13_q and stq_data_valid_11_q and addr_same_13(11) and addr_valid_13(11);
	can_bypass_13(12) <= ldq_alloc_13_q and not ldq_issue_13_q and stq_data_valid_12_q and addr_same_13(12) and addr_valid_13(12);
	can_bypass_13(13) <= ldq_alloc_13_q and not ldq_issue_13_q and stq_data_valid_13_q and addr_same_13(13) and addr_valid_13(13);
	-- Reduction Begin
	-- Reduce(load_conflict_0, ld_st_conflict_0, or)
	TEMP_23_res(0) <= ld_st_conflict_0(0) or ld_st_conflict_0(8);
	TEMP_23_res(1) <= ld_st_conflict_0(1) or ld_st_conflict_0(9);
	TEMP_23_res(2) <= ld_st_conflict_0(2) or ld_st_conflict_0(10);
	TEMP_23_res(3) <= ld_st_conflict_0(3) or ld_st_conflict_0(11);
	TEMP_23_res(4) <= ld_st_conflict_0(4) or ld_st_conflict_0(12);
	TEMP_23_res(5) <= ld_st_conflict_0(5) or ld_st_conflict_0(13);
	TEMP_23_res(6) <= ld_st_conflict_0(6);
	TEMP_23_res(7) <= ld_st_conflict_0(7);
	-- Layer End
	TEMP_24_res(0) <= TEMP_23_res(0) or TEMP_23_res(4);
	TEMP_24_res(1) <= TEMP_23_res(1) or TEMP_23_res(5);
	TEMP_24_res(2) <= TEMP_23_res(2) or TEMP_23_res(6);
	TEMP_24_res(3) <= TEMP_23_res(3) or TEMP_23_res(7);
	-- Layer End
	TEMP_25_res(0) <= TEMP_24_res(0) or TEMP_24_res(2);
	TEMP_25_res(1) <= TEMP_24_res(1) or TEMP_24_res(3);
	-- Layer End
	load_conflict_0 <= TEMP_25_res(0) or TEMP_25_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_1, ld_st_conflict_1, or)
	TEMP_26_res(0) <= ld_st_conflict_1(0) or ld_st_conflict_1(8);
	TEMP_26_res(1) <= ld_st_conflict_1(1) or ld_st_conflict_1(9);
	TEMP_26_res(2) <= ld_st_conflict_1(2) or ld_st_conflict_1(10);
	TEMP_26_res(3) <= ld_st_conflict_1(3) or ld_st_conflict_1(11);
	TEMP_26_res(4) <= ld_st_conflict_1(4) or ld_st_conflict_1(12);
	TEMP_26_res(5) <= ld_st_conflict_1(5) or ld_st_conflict_1(13);
	TEMP_26_res(6) <= ld_st_conflict_1(6);
	TEMP_26_res(7) <= ld_st_conflict_1(7);
	-- Layer End
	TEMP_27_res(0) <= TEMP_26_res(0) or TEMP_26_res(4);
	TEMP_27_res(1) <= TEMP_26_res(1) or TEMP_26_res(5);
	TEMP_27_res(2) <= TEMP_26_res(2) or TEMP_26_res(6);
	TEMP_27_res(3) <= TEMP_26_res(3) or TEMP_26_res(7);
	-- Layer End
	TEMP_28_res(0) <= TEMP_27_res(0) or TEMP_27_res(2);
	TEMP_28_res(1) <= TEMP_27_res(1) or TEMP_27_res(3);
	-- Layer End
	load_conflict_1 <= TEMP_28_res(0) or TEMP_28_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_2, ld_st_conflict_2, or)
	TEMP_29_res(0) <= ld_st_conflict_2(0) or ld_st_conflict_2(8);
	TEMP_29_res(1) <= ld_st_conflict_2(1) or ld_st_conflict_2(9);
	TEMP_29_res(2) <= ld_st_conflict_2(2) or ld_st_conflict_2(10);
	TEMP_29_res(3) <= ld_st_conflict_2(3) or ld_st_conflict_2(11);
	TEMP_29_res(4) <= ld_st_conflict_2(4) or ld_st_conflict_2(12);
	TEMP_29_res(5) <= ld_st_conflict_2(5) or ld_st_conflict_2(13);
	TEMP_29_res(6) <= ld_st_conflict_2(6);
	TEMP_29_res(7) <= ld_st_conflict_2(7);
	-- Layer End
	TEMP_30_res(0) <= TEMP_29_res(0) or TEMP_29_res(4);
	TEMP_30_res(1) <= TEMP_29_res(1) or TEMP_29_res(5);
	TEMP_30_res(2) <= TEMP_29_res(2) or TEMP_29_res(6);
	TEMP_30_res(3) <= TEMP_29_res(3) or TEMP_29_res(7);
	-- Layer End
	TEMP_31_res(0) <= TEMP_30_res(0) or TEMP_30_res(2);
	TEMP_31_res(1) <= TEMP_30_res(1) or TEMP_30_res(3);
	-- Layer End
	load_conflict_2 <= TEMP_31_res(0) or TEMP_31_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_3, ld_st_conflict_3, or)
	TEMP_32_res(0) <= ld_st_conflict_3(0) or ld_st_conflict_3(8);
	TEMP_32_res(1) <= ld_st_conflict_3(1) or ld_st_conflict_3(9);
	TEMP_32_res(2) <= ld_st_conflict_3(2) or ld_st_conflict_3(10);
	TEMP_32_res(3) <= ld_st_conflict_3(3) or ld_st_conflict_3(11);
	TEMP_32_res(4) <= ld_st_conflict_3(4) or ld_st_conflict_3(12);
	TEMP_32_res(5) <= ld_st_conflict_3(5) or ld_st_conflict_3(13);
	TEMP_32_res(6) <= ld_st_conflict_3(6);
	TEMP_32_res(7) <= ld_st_conflict_3(7);
	-- Layer End
	TEMP_33_res(0) <= TEMP_32_res(0) or TEMP_32_res(4);
	TEMP_33_res(1) <= TEMP_32_res(1) or TEMP_32_res(5);
	TEMP_33_res(2) <= TEMP_32_res(2) or TEMP_32_res(6);
	TEMP_33_res(3) <= TEMP_32_res(3) or TEMP_32_res(7);
	-- Layer End
	TEMP_34_res(0) <= TEMP_33_res(0) or TEMP_33_res(2);
	TEMP_34_res(1) <= TEMP_33_res(1) or TEMP_33_res(3);
	-- Layer End
	load_conflict_3 <= TEMP_34_res(0) or TEMP_34_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_4, ld_st_conflict_4, or)
	TEMP_35_res(0) <= ld_st_conflict_4(0) or ld_st_conflict_4(8);
	TEMP_35_res(1) <= ld_st_conflict_4(1) or ld_st_conflict_4(9);
	TEMP_35_res(2) <= ld_st_conflict_4(2) or ld_st_conflict_4(10);
	TEMP_35_res(3) <= ld_st_conflict_4(3) or ld_st_conflict_4(11);
	TEMP_35_res(4) <= ld_st_conflict_4(4) or ld_st_conflict_4(12);
	TEMP_35_res(5) <= ld_st_conflict_4(5) or ld_st_conflict_4(13);
	TEMP_35_res(6) <= ld_st_conflict_4(6);
	TEMP_35_res(7) <= ld_st_conflict_4(7);
	-- Layer End
	TEMP_36_res(0) <= TEMP_35_res(0) or TEMP_35_res(4);
	TEMP_36_res(1) <= TEMP_35_res(1) or TEMP_35_res(5);
	TEMP_36_res(2) <= TEMP_35_res(2) or TEMP_35_res(6);
	TEMP_36_res(3) <= TEMP_35_res(3) or TEMP_35_res(7);
	-- Layer End
	TEMP_37_res(0) <= TEMP_36_res(0) or TEMP_36_res(2);
	TEMP_37_res(1) <= TEMP_36_res(1) or TEMP_36_res(3);
	-- Layer End
	load_conflict_4 <= TEMP_37_res(0) or TEMP_37_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_5, ld_st_conflict_5, or)
	TEMP_38_res(0) <= ld_st_conflict_5(0) or ld_st_conflict_5(8);
	TEMP_38_res(1) <= ld_st_conflict_5(1) or ld_st_conflict_5(9);
	TEMP_38_res(2) <= ld_st_conflict_5(2) or ld_st_conflict_5(10);
	TEMP_38_res(3) <= ld_st_conflict_5(3) or ld_st_conflict_5(11);
	TEMP_38_res(4) <= ld_st_conflict_5(4) or ld_st_conflict_5(12);
	TEMP_38_res(5) <= ld_st_conflict_5(5) or ld_st_conflict_5(13);
	TEMP_38_res(6) <= ld_st_conflict_5(6);
	TEMP_38_res(7) <= ld_st_conflict_5(7);
	-- Layer End
	TEMP_39_res(0) <= TEMP_38_res(0) or TEMP_38_res(4);
	TEMP_39_res(1) <= TEMP_38_res(1) or TEMP_38_res(5);
	TEMP_39_res(2) <= TEMP_38_res(2) or TEMP_38_res(6);
	TEMP_39_res(3) <= TEMP_38_res(3) or TEMP_38_res(7);
	-- Layer End
	TEMP_40_res(0) <= TEMP_39_res(0) or TEMP_39_res(2);
	TEMP_40_res(1) <= TEMP_39_res(1) or TEMP_39_res(3);
	-- Layer End
	load_conflict_5 <= TEMP_40_res(0) or TEMP_40_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_6, ld_st_conflict_6, or)
	TEMP_41_res(0) <= ld_st_conflict_6(0) or ld_st_conflict_6(8);
	TEMP_41_res(1) <= ld_st_conflict_6(1) or ld_st_conflict_6(9);
	TEMP_41_res(2) <= ld_st_conflict_6(2) or ld_st_conflict_6(10);
	TEMP_41_res(3) <= ld_st_conflict_6(3) or ld_st_conflict_6(11);
	TEMP_41_res(4) <= ld_st_conflict_6(4) or ld_st_conflict_6(12);
	TEMP_41_res(5) <= ld_st_conflict_6(5) or ld_st_conflict_6(13);
	TEMP_41_res(6) <= ld_st_conflict_6(6);
	TEMP_41_res(7) <= ld_st_conflict_6(7);
	-- Layer End
	TEMP_42_res(0) <= TEMP_41_res(0) or TEMP_41_res(4);
	TEMP_42_res(1) <= TEMP_41_res(1) or TEMP_41_res(5);
	TEMP_42_res(2) <= TEMP_41_res(2) or TEMP_41_res(6);
	TEMP_42_res(3) <= TEMP_41_res(3) or TEMP_41_res(7);
	-- Layer End
	TEMP_43_res(0) <= TEMP_42_res(0) or TEMP_42_res(2);
	TEMP_43_res(1) <= TEMP_42_res(1) or TEMP_42_res(3);
	-- Layer End
	load_conflict_6 <= TEMP_43_res(0) or TEMP_43_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_7, ld_st_conflict_7, or)
	TEMP_44_res(0) <= ld_st_conflict_7(0) or ld_st_conflict_7(8);
	TEMP_44_res(1) <= ld_st_conflict_7(1) or ld_st_conflict_7(9);
	TEMP_44_res(2) <= ld_st_conflict_7(2) or ld_st_conflict_7(10);
	TEMP_44_res(3) <= ld_st_conflict_7(3) or ld_st_conflict_7(11);
	TEMP_44_res(4) <= ld_st_conflict_7(4) or ld_st_conflict_7(12);
	TEMP_44_res(5) <= ld_st_conflict_7(5) or ld_st_conflict_7(13);
	TEMP_44_res(6) <= ld_st_conflict_7(6);
	TEMP_44_res(7) <= ld_st_conflict_7(7);
	-- Layer End
	TEMP_45_res(0) <= TEMP_44_res(0) or TEMP_44_res(4);
	TEMP_45_res(1) <= TEMP_44_res(1) or TEMP_44_res(5);
	TEMP_45_res(2) <= TEMP_44_res(2) or TEMP_44_res(6);
	TEMP_45_res(3) <= TEMP_44_res(3) or TEMP_44_res(7);
	-- Layer End
	TEMP_46_res(0) <= TEMP_45_res(0) or TEMP_45_res(2);
	TEMP_46_res(1) <= TEMP_45_res(1) or TEMP_45_res(3);
	-- Layer End
	load_conflict_7 <= TEMP_46_res(0) or TEMP_46_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_8, ld_st_conflict_8, or)
	TEMP_47_res(0) <= ld_st_conflict_8(0) or ld_st_conflict_8(8);
	TEMP_47_res(1) <= ld_st_conflict_8(1) or ld_st_conflict_8(9);
	TEMP_47_res(2) <= ld_st_conflict_8(2) or ld_st_conflict_8(10);
	TEMP_47_res(3) <= ld_st_conflict_8(3) or ld_st_conflict_8(11);
	TEMP_47_res(4) <= ld_st_conflict_8(4) or ld_st_conflict_8(12);
	TEMP_47_res(5) <= ld_st_conflict_8(5) or ld_st_conflict_8(13);
	TEMP_47_res(6) <= ld_st_conflict_8(6);
	TEMP_47_res(7) <= ld_st_conflict_8(7);
	-- Layer End
	TEMP_48_res(0) <= TEMP_47_res(0) or TEMP_47_res(4);
	TEMP_48_res(1) <= TEMP_47_res(1) or TEMP_47_res(5);
	TEMP_48_res(2) <= TEMP_47_res(2) or TEMP_47_res(6);
	TEMP_48_res(3) <= TEMP_47_res(3) or TEMP_47_res(7);
	-- Layer End
	TEMP_49_res(0) <= TEMP_48_res(0) or TEMP_48_res(2);
	TEMP_49_res(1) <= TEMP_48_res(1) or TEMP_48_res(3);
	-- Layer End
	load_conflict_8 <= TEMP_49_res(0) or TEMP_49_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_9, ld_st_conflict_9, or)
	TEMP_50_res(0) <= ld_st_conflict_9(0) or ld_st_conflict_9(8);
	TEMP_50_res(1) <= ld_st_conflict_9(1) or ld_st_conflict_9(9);
	TEMP_50_res(2) <= ld_st_conflict_9(2) or ld_st_conflict_9(10);
	TEMP_50_res(3) <= ld_st_conflict_9(3) or ld_st_conflict_9(11);
	TEMP_50_res(4) <= ld_st_conflict_9(4) or ld_st_conflict_9(12);
	TEMP_50_res(5) <= ld_st_conflict_9(5) or ld_st_conflict_9(13);
	TEMP_50_res(6) <= ld_st_conflict_9(6);
	TEMP_50_res(7) <= ld_st_conflict_9(7);
	-- Layer End
	TEMP_51_res(0) <= TEMP_50_res(0) or TEMP_50_res(4);
	TEMP_51_res(1) <= TEMP_50_res(1) or TEMP_50_res(5);
	TEMP_51_res(2) <= TEMP_50_res(2) or TEMP_50_res(6);
	TEMP_51_res(3) <= TEMP_50_res(3) or TEMP_50_res(7);
	-- Layer End
	TEMP_52_res(0) <= TEMP_51_res(0) or TEMP_51_res(2);
	TEMP_52_res(1) <= TEMP_51_res(1) or TEMP_51_res(3);
	-- Layer End
	load_conflict_9 <= TEMP_52_res(0) or TEMP_52_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_10, ld_st_conflict_10, or)
	TEMP_53_res(0) <= ld_st_conflict_10(0) or ld_st_conflict_10(8);
	TEMP_53_res(1) <= ld_st_conflict_10(1) or ld_st_conflict_10(9);
	TEMP_53_res(2) <= ld_st_conflict_10(2) or ld_st_conflict_10(10);
	TEMP_53_res(3) <= ld_st_conflict_10(3) or ld_st_conflict_10(11);
	TEMP_53_res(4) <= ld_st_conflict_10(4) or ld_st_conflict_10(12);
	TEMP_53_res(5) <= ld_st_conflict_10(5) or ld_st_conflict_10(13);
	TEMP_53_res(6) <= ld_st_conflict_10(6);
	TEMP_53_res(7) <= ld_st_conflict_10(7);
	-- Layer End
	TEMP_54_res(0) <= TEMP_53_res(0) or TEMP_53_res(4);
	TEMP_54_res(1) <= TEMP_53_res(1) or TEMP_53_res(5);
	TEMP_54_res(2) <= TEMP_53_res(2) or TEMP_53_res(6);
	TEMP_54_res(3) <= TEMP_53_res(3) or TEMP_53_res(7);
	-- Layer End
	TEMP_55_res(0) <= TEMP_54_res(0) or TEMP_54_res(2);
	TEMP_55_res(1) <= TEMP_54_res(1) or TEMP_54_res(3);
	-- Layer End
	load_conflict_10 <= TEMP_55_res(0) or TEMP_55_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_11, ld_st_conflict_11, or)
	TEMP_56_res(0) <= ld_st_conflict_11(0) or ld_st_conflict_11(8);
	TEMP_56_res(1) <= ld_st_conflict_11(1) or ld_st_conflict_11(9);
	TEMP_56_res(2) <= ld_st_conflict_11(2) or ld_st_conflict_11(10);
	TEMP_56_res(3) <= ld_st_conflict_11(3) or ld_st_conflict_11(11);
	TEMP_56_res(4) <= ld_st_conflict_11(4) or ld_st_conflict_11(12);
	TEMP_56_res(5) <= ld_st_conflict_11(5) or ld_st_conflict_11(13);
	TEMP_56_res(6) <= ld_st_conflict_11(6);
	TEMP_56_res(7) <= ld_st_conflict_11(7);
	-- Layer End
	TEMP_57_res(0) <= TEMP_56_res(0) or TEMP_56_res(4);
	TEMP_57_res(1) <= TEMP_56_res(1) or TEMP_56_res(5);
	TEMP_57_res(2) <= TEMP_56_res(2) or TEMP_56_res(6);
	TEMP_57_res(3) <= TEMP_56_res(3) or TEMP_56_res(7);
	-- Layer End
	TEMP_58_res(0) <= TEMP_57_res(0) or TEMP_57_res(2);
	TEMP_58_res(1) <= TEMP_57_res(1) or TEMP_57_res(3);
	-- Layer End
	load_conflict_11 <= TEMP_58_res(0) or TEMP_58_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_12, ld_st_conflict_12, or)
	TEMP_59_res(0) <= ld_st_conflict_12(0) or ld_st_conflict_12(8);
	TEMP_59_res(1) <= ld_st_conflict_12(1) or ld_st_conflict_12(9);
	TEMP_59_res(2) <= ld_st_conflict_12(2) or ld_st_conflict_12(10);
	TEMP_59_res(3) <= ld_st_conflict_12(3) or ld_st_conflict_12(11);
	TEMP_59_res(4) <= ld_st_conflict_12(4) or ld_st_conflict_12(12);
	TEMP_59_res(5) <= ld_st_conflict_12(5) or ld_st_conflict_12(13);
	TEMP_59_res(6) <= ld_st_conflict_12(6);
	TEMP_59_res(7) <= ld_st_conflict_12(7);
	-- Layer End
	TEMP_60_res(0) <= TEMP_59_res(0) or TEMP_59_res(4);
	TEMP_60_res(1) <= TEMP_59_res(1) or TEMP_59_res(5);
	TEMP_60_res(2) <= TEMP_59_res(2) or TEMP_59_res(6);
	TEMP_60_res(3) <= TEMP_59_res(3) or TEMP_59_res(7);
	-- Layer End
	TEMP_61_res(0) <= TEMP_60_res(0) or TEMP_60_res(2);
	TEMP_61_res(1) <= TEMP_60_res(1) or TEMP_60_res(3);
	-- Layer End
	load_conflict_12 <= TEMP_61_res(0) or TEMP_61_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_13, ld_st_conflict_13, or)
	TEMP_62_res(0) <= ld_st_conflict_13(0) or ld_st_conflict_13(8);
	TEMP_62_res(1) <= ld_st_conflict_13(1) or ld_st_conflict_13(9);
	TEMP_62_res(2) <= ld_st_conflict_13(2) or ld_st_conflict_13(10);
	TEMP_62_res(3) <= ld_st_conflict_13(3) or ld_st_conflict_13(11);
	TEMP_62_res(4) <= ld_st_conflict_13(4) or ld_st_conflict_13(12);
	TEMP_62_res(5) <= ld_st_conflict_13(5) or ld_st_conflict_13(13);
	TEMP_62_res(6) <= ld_st_conflict_13(6);
	TEMP_62_res(7) <= ld_st_conflict_13(7);
	-- Layer End
	TEMP_63_res(0) <= TEMP_62_res(0) or TEMP_62_res(4);
	TEMP_63_res(1) <= TEMP_62_res(1) or TEMP_62_res(5);
	TEMP_63_res(2) <= TEMP_62_res(2) or TEMP_62_res(6);
	TEMP_63_res(3) <= TEMP_62_res(3) or TEMP_62_res(7);
	-- Layer End
	TEMP_64_res(0) <= TEMP_63_res(0) or TEMP_63_res(2);
	TEMP_64_res(1) <= TEMP_63_res(1) or TEMP_63_res(3);
	-- Layer End
	load_conflict_13 <= TEMP_64_res(0) or TEMP_64_res(1);
	-- Reduction End

	load_req_valid_0 <= ldq_alloc_0_q and not ldq_issue_0_q and ldq_addr_valid_0_q;
	load_req_valid_1 <= ldq_alloc_1_q and not ldq_issue_1_q and ldq_addr_valid_1_q;
	load_req_valid_2 <= ldq_alloc_2_q and not ldq_issue_2_q and ldq_addr_valid_2_q;
	load_req_valid_3 <= ldq_alloc_3_q and not ldq_issue_3_q and ldq_addr_valid_3_q;
	load_req_valid_4 <= ldq_alloc_4_q and not ldq_issue_4_q and ldq_addr_valid_4_q;
	load_req_valid_5 <= ldq_alloc_5_q and not ldq_issue_5_q and ldq_addr_valid_5_q;
	load_req_valid_6 <= ldq_alloc_6_q and not ldq_issue_6_q and ldq_addr_valid_6_q;
	load_req_valid_7 <= ldq_alloc_7_q and not ldq_issue_7_q and ldq_addr_valid_7_q;
	load_req_valid_8 <= ldq_alloc_8_q and not ldq_issue_8_q and ldq_addr_valid_8_q;
	load_req_valid_9 <= ldq_alloc_9_q and not ldq_issue_9_q and ldq_addr_valid_9_q;
	load_req_valid_10 <= ldq_alloc_10_q and not ldq_issue_10_q and ldq_addr_valid_10_q;
	load_req_valid_11 <= ldq_alloc_11_q and not ldq_issue_11_q and ldq_addr_valid_11_q;
	load_req_valid_12 <= ldq_alloc_12_q and not ldq_issue_12_q and ldq_addr_valid_12_q;
	load_req_valid_13 <= ldq_alloc_13_q and not ldq_issue_13_q and ldq_addr_valid_13_q;
	can_load_0 <= not load_conflict_0 and load_req_valid_0;
	can_load_1 <= not load_conflict_1 and load_req_valid_1;
	can_load_2 <= not load_conflict_2 and load_req_valid_2;
	can_load_3 <= not load_conflict_3 and load_req_valid_3;
	can_load_4 <= not load_conflict_4 and load_req_valid_4;
	can_load_5 <= not load_conflict_5 and load_req_valid_5;
	can_load_6 <= not load_conflict_6 and load_req_valid_6;
	can_load_7 <= not load_conflict_7 and load_req_valid_7;
	can_load_8 <= not load_conflict_8 and load_req_valid_8;
	can_load_9 <= not load_conflict_9 and load_req_valid_9;
	can_load_10 <= not load_conflict_10 and load_req_valid_10;
	can_load_11 <= not load_conflict_11 and load_req_valid_11;
	can_load_12 <= not load_conflict_12 and load_req_valid_12;
	can_load_13 <= not load_conflict_13 and load_req_valid_13;
	-- Priority Masking Begin
	-- CyclicPriorityMask(load_idx_oh_0, can_load, ldq_head_oh)
	TEMP_65_double_in(0) <= can_load_0;
	TEMP_65_double_in(14) <= can_load_0;
	TEMP_65_double_in(1) <= can_load_1;
	TEMP_65_double_in(15) <= can_load_1;
	TEMP_65_double_in(2) <= can_load_2;
	TEMP_65_double_in(16) <= can_load_2;
	TEMP_65_double_in(3) <= can_load_3;
	TEMP_65_double_in(17) <= can_load_3;
	TEMP_65_double_in(4) <= can_load_4;
	TEMP_65_double_in(18) <= can_load_4;
	TEMP_65_double_in(5) <= can_load_5;
	TEMP_65_double_in(19) <= can_load_5;
	TEMP_65_double_in(6) <= can_load_6;
	TEMP_65_double_in(20) <= can_load_6;
	TEMP_65_double_in(7) <= can_load_7;
	TEMP_65_double_in(21) <= can_load_7;
	TEMP_65_double_in(8) <= can_load_8;
	TEMP_65_double_in(22) <= can_load_8;
	TEMP_65_double_in(9) <= can_load_9;
	TEMP_65_double_in(23) <= can_load_9;
	TEMP_65_double_in(10) <= can_load_10;
	TEMP_65_double_in(24) <= can_load_10;
	TEMP_65_double_in(11) <= can_load_11;
	TEMP_65_double_in(25) <= can_load_11;
	TEMP_65_double_in(12) <= can_load_12;
	TEMP_65_double_in(26) <= can_load_12;
	TEMP_65_double_in(13) <= can_load_13;
	TEMP_65_double_in(27) <= can_load_13;
	TEMP_65_double_out <= TEMP_65_double_in and not std_logic_vector( unsigned( TEMP_65_double_in ) - unsigned( "00000000000000" & ldq_head_oh ) );
	load_idx_oh_0 <= TEMP_65_double_out(13 downto 0) or TEMP_65_double_out(27 downto 14);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(load_en_0, can_load, or)
	TEMP_66_res_0 <= can_load_0 or can_load_8;
	TEMP_66_res_1 <= can_load_1 or can_load_9;
	TEMP_66_res_2 <= can_load_2 or can_load_10;
	TEMP_66_res_3 <= can_load_3 or can_load_11;
	TEMP_66_res_4 <= can_load_4 or can_load_12;
	TEMP_66_res_5 <= can_load_5 or can_load_13;
	TEMP_66_res_6 <= can_load_6;
	TEMP_66_res_7 <= can_load_7;
	-- Layer End
	TEMP_67_res_0 <= TEMP_66_res_0 or TEMP_66_res_4;
	TEMP_67_res_1 <= TEMP_66_res_1 or TEMP_66_res_5;
	TEMP_67_res_2 <= TEMP_66_res_2 or TEMP_66_res_6;
	TEMP_67_res_3 <= TEMP_66_res_3 or TEMP_66_res_7;
	-- Layer End
	TEMP_68_res_0 <= TEMP_67_res_0 or TEMP_67_res_2;
	TEMP_68_res_1 <= TEMP_67_res_1 or TEMP_67_res_3;
	-- Layer End
	load_en_0 <= TEMP_68_res_0 or TEMP_68_res_1;
	-- Reduction End

	st_ld_conflict(0) <= ldq_alloc_0_q and not store_is_older_0_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_0(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_0_q );
	st_ld_conflict(1) <= ldq_alloc_1_q and not store_is_older_1_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_1(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_1_q );
	st_ld_conflict(2) <= ldq_alloc_2_q and not store_is_older_2_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_2(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_2_q );
	st_ld_conflict(3) <= ldq_alloc_3_q and not store_is_older_3_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_3(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_3_q );
	st_ld_conflict(4) <= ldq_alloc_4_q and not store_is_older_4_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_4(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_4_q );
	st_ld_conflict(5) <= ldq_alloc_5_q and not store_is_older_5_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_5(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_5_q );
	st_ld_conflict(6) <= ldq_alloc_6_q and not store_is_older_6_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_6(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_6_q );
	st_ld_conflict(7) <= ldq_alloc_7_q and not store_is_older_7_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_7(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_7_q );
	st_ld_conflict(8) <= ldq_alloc_8_q and not store_is_older_8_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_8(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_8_q );
	st_ld_conflict(9) <= ldq_alloc_9_q and not store_is_older_9_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_9(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_9_q );
	st_ld_conflict(10) <= ldq_alloc_10_q and not store_is_older_10_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_10(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_10_q );
	st_ld_conflict(11) <= ldq_alloc_11_q and not store_is_older_11_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_11(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_11_q );
	st_ld_conflict(12) <= ldq_alloc_12_q and not store_is_older_12_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_12(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_12_q );
	st_ld_conflict(13) <= ldq_alloc_13_q and not store_is_older_13_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_13(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_13_q );
	-- Reduction Begin
	-- Reduce(store_conflict, st_ld_conflict, or)
	TEMP_69_res(0) <= st_ld_conflict(0) or st_ld_conflict(8);
	TEMP_69_res(1) <= st_ld_conflict(1) or st_ld_conflict(9);
	TEMP_69_res(2) <= st_ld_conflict(2) or st_ld_conflict(10);
	TEMP_69_res(3) <= st_ld_conflict(3) or st_ld_conflict(11);
	TEMP_69_res(4) <= st_ld_conflict(4) or st_ld_conflict(12);
	TEMP_69_res(5) <= st_ld_conflict(5) or st_ld_conflict(13);
	TEMP_69_res(6) <= st_ld_conflict(6);
	TEMP_69_res(7) <= st_ld_conflict(7);
	-- Layer End
	TEMP_70_res(0) <= TEMP_69_res(0) or TEMP_69_res(4);
	TEMP_70_res(1) <= TEMP_69_res(1) or TEMP_69_res(5);
	TEMP_70_res(2) <= TEMP_69_res(2) or TEMP_69_res(6);
	TEMP_70_res(3) <= TEMP_69_res(3) or TEMP_69_res(7);
	-- Layer End
	TEMP_71_res(0) <= TEMP_70_res(0) or TEMP_70_res(2);
	TEMP_71_res(1) <= TEMP_70_res(1) or TEMP_70_res(3);
	-- Layer End
	store_conflict <= TEMP_71_res(0) or TEMP_71_res(1);
	-- Reduction End

	-- MuxLookUp Begin
	-- MuxLookUp(store_valid, stq_alloc, stq_issue)
	store_valid <= 
	stq_alloc_0_q when (stq_issue_q = "0000") else
	stq_alloc_1_q when (stq_issue_q = "0001") else
	stq_alloc_2_q when (stq_issue_q = "0010") else
	stq_alloc_3_q when (stq_issue_q = "0011") else
	stq_alloc_4_q when (stq_issue_q = "0100") else
	stq_alloc_5_q when (stq_issue_q = "0101") else
	stq_alloc_6_q when (stq_issue_q = "0110") else
	stq_alloc_7_q when (stq_issue_q = "0111") else
	stq_alloc_8_q when (stq_issue_q = "1000") else
	stq_alloc_9_q when (stq_issue_q = "1001") else
	stq_alloc_10_q when (stq_issue_q = "1010") else
	stq_alloc_11_q when (stq_issue_q = "1011") else
	stq_alloc_12_q when (stq_issue_q = "1100") else
	stq_alloc_13_q when (stq_issue_q = "1101") else
	'0';
	-- MuxLookUp End

	-- MuxLookUp Begin
	-- MuxLookUp(store_data_valid, stq_data_valid, stq_issue)
	store_data_valid <= 
	stq_data_valid_0_q when (stq_issue_q = "0000") else
	stq_data_valid_1_q when (stq_issue_q = "0001") else
	stq_data_valid_2_q when (stq_issue_q = "0010") else
	stq_data_valid_3_q when (stq_issue_q = "0011") else
	stq_data_valid_4_q when (stq_issue_q = "0100") else
	stq_data_valid_5_q when (stq_issue_q = "0101") else
	stq_data_valid_6_q when (stq_issue_q = "0110") else
	stq_data_valid_7_q when (stq_issue_q = "0111") else
	stq_data_valid_8_q when (stq_issue_q = "1000") else
	stq_data_valid_9_q when (stq_issue_q = "1001") else
	stq_data_valid_10_q when (stq_issue_q = "1010") else
	stq_data_valid_11_q when (stq_issue_q = "1011") else
	stq_data_valid_12_q when (stq_issue_q = "1100") else
	stq_data_valid_13_q when (stq_issue_q = "1101") else
	'0';
	-- MuxLookUp End

	-- MuxLookUp Begin
	-- MuxLookUp(store_addr_valid, stq_addr_valid, stq_issue)
	store_addr_valid <= 
	stq_addr_valid_0_q when (stq_issue_q = "0000") else
	stq_addr_valid_1_q when (stq_issue_q = "0001") else
	stq_addr_valid_2_q when (stq_issue_q = "0010") else
	stq_addr_valid_3_q when (stq_issue_q = "0011") else
	stq_addr_valid_4_q when (stq_issue_q = "0100") else
	stq_addr_valid_5_q when (stq_issue_q = "0101") else
	stq_addr_valid_6_q when (stq_issue_q = "0110") else
	stq_addr_valid_7_q when (stq_issue_q = "0111") else
	stq_addr_valid_8_q when (stq_issue_q = "1000") else
	stq_addr_valid_9_q when (stq_issue_q = "1001") else
	stq_addr_valid_10_q when (stq_issue_q = "1010") else
	stq_addr_valid_11_q when (stq_issue_q = "1011") else
	stq_addr_valid_12_q when (stq_issue_q = "1100") else
	stq_addr_valid_13_q when (stq_issue_q = "1101") else
	'0';
	-- MuxLookUp End

	store_en <= not store_conflict and store_valid and store_data_valid and store_addr_valid;
	store_idx <= stq_issue_q;
	-- Bits To One-Hot Begin
	-- BitsToOHSub1(stq_last_oh, stq_tail)
	stq_last_oh(0) <= '1' when stq_tail_q = "0001" else '0';
	stq_last_oh(1) <= '1' when stq_tail_q = "0010" else '0';
	stq_last_oh(2) <= '1' when stq_tail_q = "0011" else '0';
	stq_last_oh(3) <= '1' when stq_tail_q = "0100" else '0';
	stq_last_oh(4) <= '1' when stq_tail_q = "0101" else '0';
	stq_last_oh(5) <= '1' when stq_tail_q = "0110" else '0';
	stq_last_oh(6) <= '1' when stq_tail_q = "0111" else '0';
	stq_last_oh(7) <= '1' when stq_tail_q = "1000" else '0';
	stq_last_oh(8) <= '1' when stq_tail_q = "1001" else '0';
	stq_last_oh(9) <= '1' when stq_tail_q = "1010" else '0';
	stq_last_oh(10) <= '1' when stq_tail_q = "1011" else '0';
	stq_last_oh(11) <= '1' when stq_tail_q = "1100" else '0';
	stq_last_oh(12) <= '1' when stq_tail_q = "1101" else '0';
	stq_last_oh(13) <= '1' when stq_tail_q = "0000" else '0';
	-- Bits To One-Hot End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_0, ld_st_conflict_0, stq_last_oh)
	TEMP_72_double_in(0) <= ld_st_conflict_0(13);
	TEMP_72_double_in(14) <= ld_st_conflict_0(13);
	TEMP_72_double_in(1) <= ld_st_conflict_0(12);
	TEMP_72_double_in(15) <= ld_st_conflict_0(12);
	TEMP_72_double_in(2) <= ld_st_conflict_0(11);
	TEMP_72_double_in(16) <= ld_st_conflict_0(11);
	TEMP_72_double_in(3) <= ld_st_conflict_0(10);
	TEMP_72_double_in(17) <= ld_st_conflict_0(10);
	TEMP_72_double_in(4) <= ld_st_conflict_0(9);
	TEMP_72_double_in(18) <= ld_st_conflict_0(9);
	TEMP_72_double_in(5) <= ld_st_conflict_0(8);
	TEMP_72_double_in(19) <= ld_st_conflict_0(8);
	TEMP_72_double_in(6) <= ld_st_conflict_0(7);
	TEMP_72_double_in(20) <= ld_st_conflict_0(7);
	TEMP_72_double_in(7) <= ld_st_conflict_0(6);
	TEMP_72_double_in(21) <= ld_st_conflict_0(6);
	TEMP_72_double_in(8) <= ld_st_conflict_0(5);
	TEMP_72_double_in(22) <= ld_st_conflict_0(5);
	TEMP_72_double_in(9) <= ld_st_conflict_0(4);
	TEMP_72_double_in(23) <= ld_st_conflict_0(4);
	TEMP_72_double_in(10) <= ld_st_conflict_0(3);
	TEMP_72_double_in(24) <= ld_st_conflict_0(3);
	TEMP_72_double_in(11) <= ld_st_conflict_0(2);
	TEMP_72_double_in(25) <= ld_st_conflict_0(2);
	TEMP_72_double_in(12) <= ld_st_conflict_0(1);
	TEMP_72_double_in(26) <= ld_st_conflict_0(1);
	TEMP_72_double_in(13) <= ld_st_conflict_0(0);
	TEMP_72_double_in(27) <= ld_st_conflict_0(0);
	TEMP_72_base_rev(0) <= stq_last_oh(13);
	TEMP_72_base_rev(1) <= stq_last_oh(12);
	TEMP_72_base_rev(2) <= stq_last_oh(11);
	TEMP_72_base_rev(3) <= stq_last_oh(10);
	TEMP_72_base_rev(4) <= stq_last_oh(9);
	TEMP_72_base_rev(5) <= stq_last_oh(8);
	TEMP_72_base_rev(6) <= stq_last_oh(7);
	TEMP_72_base_rev(7) <= stq_last_oh(6);
	TEMP_72_base_rev(8) <= stq_last_oh(5);
	TEMP_72_base_rev(9) <= stq_last_oh(4);
	TEMP_72_base_rev(10) <= stq_last_oh(3);
	TEMP_72_base_rev(11) <= stq_last_oh(2);
	TEMP_72_base_rev(12) <= stq_last_oh(1);
	TEMP_72_base_rev(13) <= stq_last_oh(0);
	TEMP_72_double_out <= TEMP_72_double_in and not std_logic_vector( unsigned( TEMP_72_double_in ) - unsigned( "00000000000000" & TEMP_72_base_rev ) );
	bypass_idx_oh_0(13) <= TEMP_72_double_out(0) or TEMP_72_double_out(14);
	bypass_idx_oh_0(12) <= TEMP_72_double_out(1) or TEMP_72_double_out(15);
	bypass_idx_oh_0(11) <= TEMP_72_double_out(2) or TEMP_72_double_out(16);
	bypass_idx_oh_0(10) <= TEMP_72_double_out(3) or TEMP_72_double_out(17);
	bypass_idx_oh_0(9) <= TEMP_72_double_out(4) or TEMP_72_double_out(18);
	bypass_idx_oh_0(8) <= TEMP_72_double_out(5) or TEMP_72_double_out(19);
	bypass_idx_oh_0(7) <= TEMP_72_double_out(6) or TEMP_72_double_out(20);
	bypass_idx_oh_0(6) <= TEMP_72_double_out(7) or TEMP_72_double_out(21);
	bypass_idx_oh_0(5) <= TEMP_72_double_out(8) or TEMP_72_double_out(22);
	bypass_idx_oh_0(4) <= TEMP_72_double_out(9) or TEMP_72_double_out(23);
	bypass_idx_oh_0(3) <= TEMP_72_double_out(10) or TEMP_72_double_out(24);
	bypass_idx_oh_0(2) <= TEMP_72_double_out(11) or TEMP_72_double_out(25);
	bypass_idx_oh_0(1) <= TEMP_72_double_out(12) or TEMP_72_double_out(26);
	bypass_idx_oh_0(0) <= TEMP_72_double_out(13) or TEMP_72_double_out(27);
	-- Priority Masking End

	bypass_en_vec_0 <= bypass_idx_oh_0 and can_bypass_0;
	-- Reduction Begin
	-- Reduce(bypass_en_0, bypass_en_vec_0, or)
	TEMP_73_res(0) <= bypass_en_vec_0(0) or bypass_en_vec_0(8);
	TEMP_73_res(1) <= bypass_en_vec_0(1) or bypass_en_vec_0(9);
	TEMP_73_res(2) <= bypass_en_vec_0(2) or bypass_en_vec_0(10);
	TEMP_73_res(3) <= bypass_en_vec_0(3) or bypass_en_vec_0(11);
	TEMP_73_res(4) <= bypass_en_vec_0(4) or bypass_en_vec_0(12);
	TEMP_73_res(5) <= bypass_en_vec_0(5) or bypass_en_vec_0(13);
	TEMP_73_res(6) <= bypass_en_vec_0(6);
	TEMP_73_res(7) <= bypass_en_vec_0(7);
	-- Layer End
	TEMP_74_res(0) <= TEMP_73_res(0) or TEMP_73_res(4);
	TEMP_74_res(1) <= TEMP_73_res(1) or TEMP_73_res(5);
	TEMP_74_res(2) <= TEMP_73_res(2) or TEMP_73_res(6);
	TEMP_74_res(3) <= TEMP_73_res(3) or TEMP_73_res(7);
	-- Layer End
	TEMP_75_res(0) <= TEMP_74_res(0) or TEMP_74_res(2);
	TEMP_75_res(1) <= TEMP_74_res(1) or TEMP_74_res(3);
	-- Layer End
	bypass_en_0 <= TEMP_75_res(0) or TEMP_75_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_1, ld_st_conflict_1, stq_last_oh)
	TEMP_76_double_in(0) <= ld_st_conflict_1(13);
	TEMP_76_double_in(14) <= ld_st_conflict_1(13);
	TEMP_76_double_in(1) <= ld_st_conflict_1(12);
	TEMP_76_double_in(15) <= ld_st_conflict_1(12);
	TEMP_76_double_in(2) <= ld_st_conflict_1(11);
	TEMP_76_double_in(16) <= ld_st_conflict_1(11);
	TEMP_76_double_in(3) <= ld_st_conflict_1(10);
	TEMP_76_double_in(17) <= ld_st_conflict_1(10);
	TEMP_76_double_in(4) <= ld_st_conflict_1(9);
	TEMP_76_double_in(18) <= ld_st_conflict_1(9);
	TEMP_76_double_in(5) <= ld_st_conflict_1(8);
	TEMP_76_double_in(19) <= ld_st_conflict_1(8);
	TEMP_76_double_in(6) <= ld_st_conflict_1(7);
	TEMP_76_double_in(20) <= ld_st_conflict_1(7);
	TEMP_76_double_in(7) <= ld_st_conflict_1(6);
	TEMP_76_double_in(21) <= ld_st_conflict_1(6);
	TEMP_76_double_in(8) <= ld_st_conflict_1(5);
	TEMP_76_double_in(22) <= ld_st_conflict_1(5);
	TEMP_76_double_in(9) <= ld_st_conflict_1(4);
	TEMP_76_double_in(23) <= ld_st_conflict_1(4);
	TEMP_76_double_in(10) <= ld_st_conflict_1(3);
	TEMP_76_double_in(24) <= ld_st_conflict_1(3);
	TEMP_76_double_in(11) <= ld_st_conflict_1(2);
	TEMP_76_double_in(25) <= ld_st_conflict_1(2);
	TEMP_76_double_in(12) <= ld_st_conflict_1(1);
	TEMP_76_double_in(26) <= ld_st_conflict_1(1);
	TEMP_76_double_in(13) <= ld_st_conflict_1(0);
	TEMP_76_double_in(27) <= ld_st_conflict_1(0);
	TEMP_76_base_rev(0) <= stq_last_oh(13);
	TEMP_76_base_rev(1) <= stq_last_oh(12);
	TEMP_76_base_rev(2) <= stq_last_oh(11);
	TEMP_76_base_rev(3) <= stq_last_oh(10);
	TEMP_76_base_rev(4) <= stq_last_oh(9);
	TEMP_76_base_rev(5) <= stq_last_oh(8);
	TEMP_76_base_rev(6) <= stq_last_oh(7);
	TEMP_76_base_rev(7) <= stq_last_oh(6);
	TEMP_76_base_rev(8) <= stq_last_oh(5);
	TEMP_76_base_rev(9) <= stq_last_oh(4);
	TEMP_76_base_rev(10) <= stq_last_oh(3);
	TEMP_76_base_rev(11) <= stq_last_oh(2);
	TEMP_76_base_rev(12) <= stq_last_oh(1);
	TEMP_76_base_rev(13) <= stq_last_oh(0);
	TEMP_76_double_out <= TEMP_76_double_in and not std_logic_vector( unsigned( TEMP_76_double_in ) - unsigned( "00000000000000" & TEMP_76_base_rev ) );
	bypass_idx_oh_1(13) <= TEMP_76_double_out(0) or TEMP_76_double_out(14);
	bypass_idx_oh_1(12) <= TEMP_76_double_out(1) or TEMP_76_double_out(15);
	bypass_idx_oh_1(11) <= TEMP_76_double_out(2) or TEMP_76_double_out(16);
	bypass_idx_oh_1(10) <= TEMP_76_double_out(3) or TEMP_76_double_out(17);
	bypass_idx_oh_1(9) <= TEMP_76_double_out(4) or TEMP_76_double_out(18);
	bypass_idx_oh_1(8) <= TEMP_76_double_out(5) or TEMP_76_double_out(19);
	bypass_idx_oh_1(7) <= TEMP_76_double_out(6) or TEMP_76_double_out(20);
	bypass_idx_oh_1(6) <= TEMP_76_double_out(7) or TEMP_76_double_out(21);
	bypass_idx_oh_1(5) <= TEMP_76_double_out(8) or TEMP_76_double_out(22);
	bypass_idx_oh_1(4) <= TEMP_76_double_out(9) or TEMP_76_double_out(23);
	bypass_idx_oh_1(3) <= TEMP_76_double_out(10) or TEMP_76_double_out(24);
	bypass_idx_oh_1(2) <= TEMP_76_double_out(11) or TEMP_76_double_out(25);
	bypass_idx_oh_1(1) <= TEMP_76_double_out(12) or TEMP_76_double_out(26);
	bypass_idx_oh_1(0) <= TEMP_76_double_out(13) or TEMP_76_double_out(27);
	-- Priority Masking End

	bypass_en_vec_1 <= bypass_idx_oh_1 and can_bypass_1;
	-- Reduction Begin
	-- Reduce(bypass_en_1, bypass_en_vec_1, or)
	TEMP_77_res(0) <= bypass_en_vec_1(0) or bypass_en_vec_1(8);
	TEMP_77_res(1) <= bypass_en_vec_1(1) or bypass_en_vec_1(9);
	TEMP_77_res(2) <= bypass_en_vec_1(2) or bypass_en_vec_1(10);
	TEMP_77_res(3) <= bypass_en_vec_1(3) or bypass_en_vec_1(11);
	TEMP_77_res(4) <= bypass_en_vec_1(4) or bypass_en_vec_1(12);
	TEMP_77_res(5) <= bypass_en_vec_1(5) or bypass_en_vec_1(13);
	TEMP_77_res(6) <= bypass_en_vec_1(6);
	TEMP_77_res(7) <= bypass_en_vec_1(7);
	-- Layer End
	TEMP_78_res(0) <= TEMP_77_res(0) or TEMP_77_res(4);
	TEMP_78_res(1) <= TEMP_77_res(1) or TEMP_77_res(5);
	TEMP_78_res(2) <= TEMP_77_res(2) or TEMP_77_res(6);
	TEMP_78_res(3) <= TEMP_77_res(3) or TEMP_77_res(7);
	-- Layer End
	TEMP_79_res(0) <= TEMP_78_res(0) or TEMP_78_res(2);
	TEMP_79_res(1) <= TEMP_78_res(1) or TEMP_78_res(3);
	-- Layer End
	bypass_en_1 <= TEMP_79_res(0) or TEMP_79_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_2, ld_st_conflict_2, stq_last_oh)
	TEMP_80_double_in(0) <= ld_st_conflict_2(13);
	TEMP_80_double_in(14) <= ld_st_conflict_2(13);
	TEMP_80_double_in(1) <= ld_st_conflict_2(12);
	TEMP_80_double_in(15) <= ld_st_conflict_2(12);
	TEMP_80_double_in(2) <= ld_st_conflict_2(11);
	TEMP_80_double_in(16) <= ld_st_conflict_2(11);
	TEMP_80_double_in(3) <= ld_st_conflict_2(10);
	TEMP_80_double_in(17) <= ld_st_conflict_2(10);
	TEMP_80_double_in(4) <= ld_st_conflict_2(9);
	TEMP_80_double_in(18) <= ld_st_conflict_2(9);
	TEMP_80_double_in(5) <= ld_st_conflict_2(8);
	TEMP_80_double_in(19) <= ld_st_conflict_2(8);
	TEMP_80_double_in(6) <= ld_st_conflict_2(7);
	TEMP_80_double_in(20) <= ld_st_conflict_2(7);
	TEMP_80_double_in(7) <= ld_st_conflict_2(6);
	TEMP_80_double_in(21) <= ld_st_conflict_2(6);
	TEMP_80_double_in(8) <= ld_st_conflict_2(5);
	TEMP_80_double_in(22) <= ld_st_conflict_2(5);
	TEMP_80_double_in(9) <= ld_st_conflict_2(4);
	TEMP_80_double_in(23) <= ld_st_conflict_2(4);
	TEMP_80_double_in(10) <= ld_st_conflict_2(3);
	TEMP_80_double_in(24) <= ld_st_conflict_2(3);
	TEMP_80_double_in(11) <= ld_st_conflict_2(2);
	TEMP_80_double_in(25) <= ld_st_conflict_2(2);
	TEMP_80_double_in(12) <= ld_st_conflict_2(1);
	TEMP_80_double_in(26) <= ld_st_conflict_2(1);
	TEMP_80_double_in(13) <= ld_st_conflict_2(0);
	TEMP_80_double_in(27) <= ld_st_conflict_2(0);
	TEMP_80_base_rev(0) <= stq_last_oh(13);
	TEMP_80_base_rev(1) <= stq_last_oh(12);
	TEMP_80_base_rev(2) <= stq_last_oh(11);
	TEMP_80_base_rev(3) <= stq_last_oh(10);
	TEMP_80_base_rev(4) <= stq_last_oh(9);
	TEMP_80_base_rev(5) <= stq_last_oh(8);
	TEMP_80_base_rev(6) <= stq_last_oh(7);
	TEMP_80_base_rev(7) <= stq_last_oh(6);
	TEMP_80_base_rev(8) <= stq_last_oh(5);
	TEMP_80_base_rev(9) <= stq_last_oh(4);
	TEMP_80_base_rev(10) <= stq_last_oh(3);
	TEMP_80_base_rev(11) <= stq_last_oh(2);
	TEMP_80_base_rev(12) <= stq_last_oh(1);
	TEMP_80_base_rev(13) <= stq_last_oh(0);
	TEMP_80_double_out <= TEMP_80_double_in and not std_logic_vector( unsigned( TEMP_80_double_in ) - unsigned( "00000000000000" & TEMP_80_base_rev ) );
	bypass_idx_oh_2(13) <= TEMP_80_double_out(0) or TEMP_80_double_out(14);
	bypass_idx_oh_2(12) <= TEMP_80_double_out(1) or TEMP_80_double_out(15);
	bypass_idx_oh_2(11) <= TEMP_80_double_out(2) or TEMP_80_double_out(16);
	bypass_idx_oh_2(10) <= TEMP_80_double_out(3) or TEMP_80_double_out(17);
	bypass_idx_oh_2(9) <= TEMP_80_double_out(4) or TEMP_80_double_out(18);
	bypass_idx_oh_2(8) <= TEMP_80_double_out(5) or TEMP_80_double_out(19);
	bypass_idx_oh_2(7) <= TEMP_80_double_out(6) or TEMP_80_double_out(20);
	bypass_idx_oh_2(6) <= TEMP_80_double_out(7) or TEMP_80_double_out(21);
	bypass_idx_oh_2(5) <= TEMP_80_double_out(8) or TEMP_80_double_out(22);
	bypass_idx_oh_2(4) <= TEMP_80_double_out(9) or TEMP_80_double_out(23);
	bypass_idx_oh_2(3) <= TEMP_80_double_out(10) or TEMP_80_double_out(24);
	bypass_idx_oh_2(2) <= TEMP_80_double_out(11) or TEMP_80_double_out(25);
	bypass_idx_oh_2(1) <= TEMP_80_double_out(12) or TEMP_80_double_out(26);
	bypass_idx_oh_2(0) <= TEMP_80_double_out(13) or TEMP_80_double_out(27);
	-- Priority Masking End

	bypass_en_vec_2 <= bypass_idx_oh_2 and can_bypass_2;
	-- Reduction Begin
	-- Reduce(bypass_en_2, bypass_en_vec_2, or)
	TEMP_81_res(0) <= bypass_en_vec_2(0) or bypass_en_vec_2(8);
	TEMP_81_res(1) <= bypass_en_vec_2(1) or bypass_en_vec_2(9);
	TEMP_81_res(2) <= bypass_en_vec_2(2) or bypass_en_vec_2(10);
	TEMP_81_res(3) <= bypass_en_vec_2(3) or bypass_en_vec_2(11);
	TEMP_81_res(4) <= bypass_en_vec_2(4) or bypass_en_vec_2(12);
	TEMP_81_res(5) <= bypass_en_vec_2(5) or bypass_en_vec_2(13);
	TEMP_81_res(6) <= bypass_en_vec_2(6);
	TEMP_81_res(7) <= bypass_en_vec_2(7);
	-- Layer End
	TEMP_82_res(0) <= TEMP_81_res(0) or TEMP_81_res(4);
	TEMP_82_res(1) <= TEMP_81_res(1) or TEMP_81_res(5);
	TEMP_82_res(2) <= TEMP_81_res(2) or TEMP_81_res(6);
	TEMP_82_res(3) <= TEMP_81_res(3) or TEMP_81_res(7);
	-- Layer End
	TEMP_83_res(0) <= TEMP_82_res(0) or TEMP_82_res(2);
	TEMP_83_res(1) <= TEMP_82_res(1) or TEMP_82_res(3);
	-- Layer End
	bypass_en_2 <= TEMP_83_res(0) or TEMP_83_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_3, ld_st_conflict_3, stq_last_oh)
	TEMP_84_double_in(0) <= ld_st_conflict_3(13);
	TEMP_84_double_in(14) <= ld_st_conflict_3(13);
	TEMP_84_double_in(1) <= ld_st_conflict_3(12);
	TEMP_84_double_in(15) <= ld_st_conflict_3(12);
	TEMP_84_double_in(2) <= ld_st_conflict_3(11);
	TEMP_84_double_in(16) <= ld_st_conflict_3(11);
	TEMP_84_double_in(3) <= ld_st_conflict_3(10);
	TEMP_84_double_in(17) <= ld_st_conflict_3(10);
	TEMP_84_double_in(4) <= ld_st_conflict_3(9);
	TEMP_84_double_in(18) <= ld_st_conflict_3(9);
	TEMP_84_double_in(5) <= ld_st_conflict_3(8);
	TEMP_84_double_in(19) <= ld_st_conflict_3(8);
	TEMP_84_double_in(6) <= ld_st_conflict_3(7);
	TEMP_84_double_in(20) <= ld_st_conflict_3(7);
	TEMP_84_double_in(7) <= ld_st_conflict_3(6);
	TEMP_84_double_in(21) <= ld_st_conflict_3(6);
	TEMP_84_double_in(8) <= ld_st_conflict_3(5);
	TEMP_84_double_in(22) <= ld_st_conflict_3(5);
	TEMP_84_double_in(9) <= ld_st_conflict_3(4);
	TEMP_84_double_in(23) <= ld_st_conflict_3(4);
	TEMP_84_double_in(10) <= ld_st_conflict_3(3);
	TEMP_84_double_in(24) <= ld_st_conflict_3(3);
	TEMP_84_double_in(11) <= ld_st_conflict_3(2);
	TEMP_84_double_in(25) <= ld_st_conflict_3(2);
	TEMP_84_double_in(12) <= ld_st_conflict_3(1);
	TEMP_84_double_in(26) <= ld_st_conflict_3(1);
	TEMP_84_double_in(13) <= ld_st_conflict_3(0);
	TEMP_84_double_in(27) <= ld_st_conflict_3(0);
	TEMP_84_base_rev(0) <= stq_last_oh(13);
	TEMP_84_base_rev(1) <= stq_last_oh(12);
	TEMP_84_base_rev(2) <= stq_last_oh(11);
	TEMP_84_base_rev(3) <= stq_last_oh(10);
	TEMP_84_base_rev(4) <= stq_last_oh(9);
	TEMP_84_base_rev(5) <= stq_last_oh(8);
	TEMP_84_base_rev(6) <= stq_last_oh(7);
	TEMP_84_base_rev(7) <= stq_last_oh(6);
	TEMP_84_base_rev(8) <= stq_last_oh(5);
	TEMP_84_base_rev(9) <= stq_last_oh(4);
	TEMP_84_base_rev(10) <= stq_last_oh(3);
	TEMP_84_base_rev(11) <= stq_last_oh(2);
	TEMP_84_base_rev(12) <= stq_last_oh(1);
	TEMP_84_base_rev(13) <= stq_last_oh(0);
	TEMP_84_double_out <= TEMP_84_double_in and not std_logic_vector( unsigned( TEMP_84_double_in ) - unsigned( "00000000000000" & TEMP_84_base_rev ) );
	bypass_idx_oh_3(13) <= TEMP_84_double_out(0) or TEMP_84_double_out(14);
	bypass_idx_oh_3(12) <= TEMP_84_double_out(1) or TEMP_84_double_out(15);
	bypass_idx_oh_3(11) <= TEMP_84_double_out(2) or TEMP_84_double_out(16);
	bypass_idx_oh_3(10) <= TEMP_84_double_out(3) or TEMP_84_double_out(17);
	bypass_idx_oh_3(9) <= TEMP_84_double_out(4) or TEMP_84_double_out(18);
	bypass_idx_oh_3(8) <= TEMP_84_double_out(5) or TEMP_84_double_out(19);
	bypass_idx_oh_3(7) <= TEMP_84_double_out(6) or TEMP_84_double_out(20);
	bypass_idx_oh_3(6) <= TEMP_84_double_out(7) or TEMP_84_double_out(21);
	bypass_idx_oh_3(5) <= TEMP_84_double_out(8) or TEMP_84_double_out(22);
	bypass_idx_oh_3(4) <= TEMP_84_double_out(9) or TEMP_84_double_out(23);
	bypass_idx_oh_3(3) <= TEMP_84_double_out(10) or TEMP_84_double_out(24);
	bypass_idx_oh_3(2) <= TEMP_84_double_out(11) or TEMP_84_double_out(25);
	bypass_idx_oh_3(1) <= TEMP_84_double_out(12) or TEMP_84_double_out(26);
	bypass_idx_oh_3(0) <= TEMP_84_double_out(13) or TEMP_84_double_out(27);
	-- Priority Masking End

	bypass_en_vec_3 <= bypass_idx_oh_3 and can_bypass_3;
	-- Reduction Begin
	-- Reduce(bypass_en_3, bypass_en_vec_3, or)
	TEMP_85_res(0) <= bypass_en_vec_3(0) or bypass_en_vec_3(8);
	TEMP_85_res(1) <= bypass_en_vec_3(1) or bypass_en_vec_3(9);
	TEMP_85_res(2) <= bypass_en_vec_3(2) or bypass_en_vec_3(10);
	TEMP_85_res(3) <= bypass_en_vec_3(3) or bypass_en_vec_3(11);
	TEMP_85_res(4) <= bypass_en_vec_3(4) or bypass_en_vec_3(12);
	TEMP_85_res(5) <= bypass_en_vec_3(5) or bypass_en_vec_3(13);
	TEMP_85_res(6) <= bypass_en_vec_3(6);
	TEMP_85_res(7) <= bypass_en_vec_3(7);
	-- Layer End
	TEMP_86_res(0) <= TEMP_85_res(0) or TEMP_85_res(4);
	TEMP_86_res(1) <= TEMP_85_res(1) or TEMP_85_res(5);
	TEMP_86_res(2) <= TEMP_85_res(2) or TEMP_85_res(6);
	TEMP_86_res(3) <= TEMP_85_res(3) or TEMP_85_res(7);
	-- Layer End
	TEMP_87_res(0) <= TEMP_86_res(0) or TEMP_86_res(2);
	TEMP_87_res(1) <= TEMP_86_res(1) or TEMP_86_res(3);
	-- Layer End
	bypass_en_3 <= TEMP_87_res(0) or TEMP_87_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_4, ld_st_conflict_4, stq_last_oh)
	TEMP_88_double_in(0) <= ld_st_conflict_4(13);
	TEMP_88_double_in(14) <= ld_st_conflict_4(13);
	TEMP_88_double_in(1) <= ld_st_conflict_4(12);
	TEMP_88_double_in(15) <= ld_st_conflict_4(12);
	TEMP_88_double_in(2) <= ld_st_conflict_4(11);
	TEMP_88_double_in(16) <= ld_st_conflict_4(11);
	TEMP_88_double_in(3) <= ld_st_conflict_4(10);
	TEMP_88_double_in(17) <= ld_st_conflict_4(10);
	TEMP_88_double_in(4) <= ld_st_conflict_4(9);
	TEMP_88_double_in(18) <= ld_st_conflict_4(9);
	TEMP_88_double_in(5) <= ld_st_conflict_4(8);
	TEMP_88_double_in(19) <= ld_st_conflict_4(8);
	TEMP_88_double_in(6) <= ld_st_conflict_4(7);
	TEMP_88_double_in(20) <= ld_st_conflict_4(7);
	TEMP_88_double_in(7) <= ld_st_conflict_4(6);
	TEMP_88_double_in(21) <= ld_st_conflict_4(6);
	TEMP_88_double_in(8) <= ld_st_conflict_4(5);
	TEMP_88_double_in(22) <= ld_st_conflict_4(5);
	TEMP_88_double_in(9) <= ld_st_conflict_4(4);
	TEMP_88_double_in(23) <= ld_st_conflict_4(4);
	TEMP_88_double_in(10) <= ld_st_conflict_4(3);
	TEMP_88_double_in(24) <= ld_st_conflict_4(3);
	TEMP_88_double_in(11) <= ld_st_conflict_4(2);
	TEMP_88_double_in(25) <= ld_st_conflict_4(2);
	TEMP_88_double_in(12) <= ld_st_conflict_4(1);
	TEMP_88_double_in(26) <= ld_st_conflict_4(1);
	TEMP_88_double_in(13) <= ld_st_conflict_4(0);
	TEMP_88_double_in(27) <= ld_st_conflict_4(0);
	TEMP_88_base_rev(0) <= stq_last_oh(13);
	TEMP_88_base_rev(1) <= stq_last_oh(12);
	TEMP_88_base_rev(2) <= stq_last_oh(11);
	TEMP_88_base_rev(3) <= stq_last_oh(10);
	TEMP_88_base_rev(4) <= stq_last_oh(9);
	TEMP_88_base_rev(5) <= stq_last_oh(8);
	TEMP_88_base_rev(6) <= stq_last_oh(7);
	TEMP_88_base_rev(7) <= stq_last_oh(6);
	TEMP_88_base_rev(8) <= stq_last_oh(5);
	TEMP_88_base_rev(9) <= stq_last_oh(4);
	TEMP_88_base_rev(10) <= stq_last_oh(3);
	TEMP_88_base_rev(11) <= stq_last_oh(2);
	TEMP_88_base_rev(12) <= stq_last_oh(1);
	TEMP_88_base_rev(13) <= stq_last_oh(0);
	TEMP_88_double_out <= TEMP_88_double_in and not std_logic_vector( unsigned( TEMP_88_double_in ) - unsigned( "00000000000000" & TEMP_88_base_rev ) );
	bypass_idx_oh_4(13) <= TEMP_88_double_out(0) or TEMP_88_double_out(14);
	bypass_idx_oh_4(12) <= TEMP_88_double_out(1) or TEMP_88_double_out(15);
	bypass_idx_oh_4(11) <= TEMP_88_double_out(2) or TEMP_88_double_out(16);
	bypass_idx_oh_4(10) <= TEMP_88_double_out(3) or TEMP_88_double_out(17);
	bypass_idx_oh_4(9) <= TEMP_88_double_out(4) or TEMP_88_double_out(18);
	bypass_idx_oh_4(8) <= TEMP_88_double_out(5) or TEMP_88_double_out(19);
	bypass_idx_oh_4(7) <= TEMP_88_double_out(6) or TEMP_88_double_out(20);
	bypass_idx_oh_4(6) <= TEMP_88_double_out(7) or TEMP_88_double_out(21);
	bypass_idx_oh_4(5) <= TEMP_88_double_out(8) or TEMP_88_double_out(22);
	bypass_idx_oh_4(4) <= TEMP_88_double_out(9) or TEMP_88_double_out(23);
	bypass_idx_oh_4(3) <= TEMP_88_double_out(10) or TEMP_88_double_out(24);
	bypass_idx_oh_4(2) <= TEMP_88_double_out(11) or TEMP_88_double_out(25);
	bypass_idx_oh_4(1) <= TEMP_88_double_out(12) or TEMP_88_double_out(26);
	bypass_idx_oh_4(0) <= TEMP_88_double_out(13) or TEMP_88_double_out(27);
	-- Priority Masking End

	bypass_en_vec_4 <= bypass_idx_oh_4 and can_bypass_4;
	-- Reduction Begin
	-- Reduce(bypass_en_4, bypass_en_vec_4, or)
	TEMP_89_res(0) <= bypass_en_vec_4(0) or bypass_en_vec_4(8);
	TEMP_89_res(1) <= bypass_en_vec_4(1) or bypass_en_vec_4(9);
	TEMP_89_res(2) <= bypass_en_vec_4(2) or bypass_en_vec_4(10);
	TEMP_89_res(3) <= bypass_en_vec_4(3) or bypass_en_vec_4(11);
	TEMP_89_res(4) <= bypass_en_vec_4(4) or bypass_en_vec_4(12);
	TEMP_89_res(5) <= bypass_en_vec_4(5) or bypass_en_vec_4(13);
	TEMP_89_res(6) <= bypass_en_vec_4(6);
	TEMP_89_res(7) <= bypass_en_vec_4(7);
	-- Layer End
	TEMP_90_res(0) <= TEMP_89_res(0) or TEMP_89_res(4);
	TEMP_90_res(1) <= TEMP_89_res(1) or TEMP_89_res(5);
	TEMP_90_res(2) <= TEMP_89_res(2) or TEMP_89_res(6);
	TEMP_90_res(3) <= TEMP_89_res(3) or TEMP_89_res(7);
	-- Layer End
	TEMP_91_res(0) <= TEMP_90_res(0) or TEMP_90_res(2);
	TEMP_91_res(1) <= TEMP_90_res(1) or TEMP_90_res(3);
	-- Layer End
	bypass_en_4 <= TEMP_91_res(0) or TEMP_91_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_5, ld_st_conflict_5, stq_last_oh)
	TEMP_92_double_in(0) <= ld_st_conflict_5(13);
	TEMP_92_double_in(14) <= ld_st_conflict_5(13);
	TEMP_92_double_in(1) <= ld_st_conflict_5(12);
	TEMP_92_double_in(15) <= ld_st_conflict_5(12);
	TEMP_92_double_in(2) <= ld_st_conflict_5(11);
	TEMP_92_double_in(16) <= ld_st_conflict_5(11);
	TEMP_92_double_in(3) <= ld_st_conflict_5(10);
	TEMP_92_double_in(17) <= ld_st_conflict_5(10);
	TEMP_92_double_in(4) <= ld_st_conflict_5(9);
	TEMP_92_double_in(18) <= ld_st_conflict_5(9);
	TEMP_92_double_in(5) <= ld_st_conflict_5(8);
	TEMP_92_double_in(19) <= ld_st_conflict_5(8);
	TEMP_92_double_in(6) <= ld_st_conflict_5(7);
	TEMP_92_double_in(20) <= ld_st_conflict_5(7);
	TEMP_92_double_in(7) <= ld_st_conflict_5(6);
	TEMP_92_double_in(21) <= ld_st_conflict_5(6);
	TEMP_92_double_in(8) <= ld_st_conflict_5(5);
	TEMP_92_double_in(22) <= ld_st_conflict_5(5);
	TEMP_92_double_in(9) <= ld_st_conflict_5(4);
	TEMP_92_double_in(23) <= ld_st_conflict_5(4);
	TEMP_92_double_in(10) <= ld_st_conflict_5(3);
	TEMP_92_double_in(24) <= ld_st_conflict_5(3);
	TEMP_92_double_in(11) <= ld_st_conflict_5(2);
	TEMP_92_double_in(25) <= ld_st_conflict_5(2);
	TEMP_92_double_in(12) <= ld_st_conflict_5(1);
	TEMP_92_double_in(26) <= ld_st_conflict_5(1);
	TEMP_92_double_in(13) <= ld_st_conflict_5(0);
	TEMP_92_double_in(27) <= ld_st_conflict_5(0);
	TEMP_92_base_rev(0) <= stq_last_oh(13);
	TEMP_92_base_rev(1) <= stq_last_oh(12);
	TEMP_92_base_rev(2) <= stq_last_oh(11);
	TEMP_92_base_rev(3) <= stq_last_oh(10);
	TEMP_92_base_rev(4) <= stq_last_oh(9);
	TEMP_92_base_rev(5) <= stq_last_oh(8);
	TEMP_92_base_rev(6) <= stq_last_oh(7);
	TEMP_92_base_rev(7) <= stq_last_oh(6);
	TEMP_92_base_rev(8) <= stq_last_oh(5);
	TEMP_92_base_rev(9) <= stq_last_oh(4);
	TEMP_92_base_rev(10) <= stq_last_oh(3);
	TEMP_92_base_rev(11) <= stq_last_oh(2);
	TEMP_92_base_rev(12) <= stq_last_oh(1);
	TEMP_92_base_rev(13) <= stq_last_oh(0);
	TEMP_92_double_out <= TEMP_92_double_in and not std_logic_vector( unsigned( TEMP_92_double_in ) - unsigned( "00000000000000" & TEMP_92_base_rev ) );
	bypass_idx_oh_5(13) <= TEMP_92_double_out(0) or TEMP_92_double_out(14);
	bypass_idx_oh_5(12) <= TEMP_92_double_out(1) or TEMP_92_double_out(15);
	bypass_idx_oh_5(11) <= TEMP_92_double_out(2) or TEMP_92_double_out(16);
	bypass_idx_oh_5(10) <= TEMP_92_double_out(3) or TEMP_92_double_out(17);
	bypass_idx_oh_5(9) <= TEMP_92_double_out(4) or TEMP_92_double_out(18);
	bypass_idx_oh_5(8) <= TEMP_92_double_out(5) or TEMP_92_double_out(19);
	bypass_idx_oh_5(7) <= TEMP_92_double_out(6) or TEMP_92_double_out(20);
	bypass_idx_oh_5(6) <= TEMP_92_double_out(7) or TEMP_92_double_out(21);
	bypass_idx_oh_5(5) <= TEMP_92_double_out(8) or TEMP_92_double_out(22);
	bypass_idx_oh_5(4) <= TEMP_92_double_out(9) or TEMP_92_double_out(23);
	bypass_idx_oh_5(3) <= TEMP_92_double_out(10) or TEMP_92_double_out(24);
	bypass_idx_oh_5(2) <= TEMP_92_double_out(11) or TEMP_92_double_out(25);
	bypass_idx_oh_5(1) <= TEMP_92_double_out(12) or TEMP_92_double_out(26);
	bypass_idx_oh_5(0) <= TEMP_92_double_out(13) or TEMP_92_double_out(27);
	-- Priority Masking End

	bypass_en_vec_5 <= bypass_idx_oh_5 and can_bypass_5;
	-- Reduction Begin
	-- Reduce(bypass_en_5, bypass_en_vec_5, or)
	TEMP_93_res(0) <= bypass_en_vec_5(0) or bypass_en_vec_5(8);
	TEMP_93_res(1) <= bypass_en_vec_5(1) or bypass_en_vec_5(9);
	TEMP_93_res(2) <= bypass_en_vec_5(2) or bypass_en_vec_5(10);
	TEMP_93_res(3) <= bypass_en_vec_5(3) or bypass_en_vec_5(11);
	TEMP_93_res(4) <= bypass_en_vec_5(4) or bypass_en_vec_5(12);
	TEMP_93_res(5) <= bypass_en_vec_5(5) or bypass_en_vec_5(13);
	TEMP_93_res(6) <= bypass_en_vec_5(6);
	TEMP_93_res(7) <= bypass_en_vec_5(7);
	-- Layer End
	TEMP_94_res(0) <= TEMP_93_res(0) or TEMP_93_res(4);
	TEMP_94_res(1) <= TEMP_93_res(1) or TEMP_93_res(5);
	TEMP_94_res(2) <= TEMP_93_res(2) or TEMP_93_res(6);
	TEMP_94_res(3) <= TEMP_93_res(3) or TEMP_93_res(7);
	-- Layer End
	TEMP_95_res(0) <= TEMP_94_res(0) or TEMP_94_res(2);
	TEMP_95_res(1) <= TEMP_94_res(1) or TEMP_94_res(3);
	-- Layer End
	bypass_en_5 <= TEMP_95_res(0) or TEMP_95_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_6, ld_st_conflict_6, stq_last_oh)
	TEMP_96_double_in(0) <= ld_st_conflict_6(13);
	TEMP_96_double_in(14) <= ld_st_conflict_6(13);
	TEMP_96_double_in(1) <= ld_st_conflict_6(12);
	TEMP_96_double_in(15) <= ld_st_conflict_6(12);
	TEMP_96_double_in(2) <= ld_st_conflict_6(11);
	TEMP_96_double_in(16) <= ld_st_conflict_6(11);
	TEMP_96_double_in(3) <= ld_st_conflict_6(10);
	TEMP_96_double_in(17) <= ld_st_conflict_6(10);
	TEMP_96_double_in(4) <= ld_st_conflict_6(9);
	TEMP_96_double_in(18) <= ld_st_conflict_6(9);
	TEMP_96_double_in(5) <= ld_st_conflict_6(8);
	TEMP_96_double_in(19) <= ld_st_conflict_6(8);
	TEMP_96_double_in(6) <= ld_st_conflict_6(7);
	TEMP_96_double_in(20) <= ld_st_conflict_6(7);
	TEMP_96_double_in(7) <= ld_st_conflict_6(6);
	TEMP_96_double_in(21) <= ld_st_conflict_6(6);
	TEMP_96_double_in(8) <= ld_st_conflict_6(5);
	TEMP_96_double_in(22) <= ld_st_conflict_6(5);
	TEMP_96_double_in(9) <= ld_st_conflict_6(4);
	TEMP_96_double_in(23) <= ld_st_conflict_6(4);
	TEMP_96_double_in(10) <= ld_st_conflict_6(3);
	TEMP_96_double_in(24) <= ld_st_conflict_6(3);
	TEMP_96_double_in(11) <= ld_st_conflict_6(2);
	TEMP_96_double_in(25) <= ld_st_conflict_6(2);
	TEMP_96_double_in(12) <= ld_st_conflict_6(1);
	TEMP_96_double_in(26) <= ld_st_conflict_6(1);
	TEMP_96_double_in(13) <= ld_st_conflict_6(0);
	TEMP_96_double_in(27) <= ld_st_conflict_6(0);
	TEMP_96_base_rev(0) <= stq_last_oh(13);
	TEMP_96_base_rev(1) <= stq_last_oh(12);
	TEMP_96_base_rev(2) <= stq_last_oh(11);
	TEMP_96_base_rev(3) <= stq_last_oh(10);
	TEMP_96_base_rev(4) <= stq_last_oh(9);
	TEMP_96_base_rev(5) <= stq_last_oh(8);
	TEMP_96_base_rev(6) <= stq_last_oh(7);
	TEMP_96_base_rev(7) <= stq_last_oh(6);
	TEMP_96_base_rev(8) <= stq_last_oh(5);
	TEMP_96_base_rev(9) <= stq_last_oh(4);
	TEMP_96_base_rev(10) <= stq_last_oh(3);
	TEMP_96_base_rev(11) <= stq_last_oh(2);
	TEMP_96_base_rev(12) <= stq_last_oh(1);
	TEMP_96_base_rev(13) <= stq_last_oh(0);
	TEMP_96_double_out <= TEMP_96_double_in and not std_logic_vector( unsigned( TEMP_96_double_in ) - unsigned( "00000000000000" & TEMP_96_base_rev ) );
	bypass_idx_oh_6(13) <= TEMP_96_double_out(0) or TEMP_96_double_out(14);
	bypass_idx_oh_6(12) <= TEMP_96_double_out(1) or TEMP_96_double_out(15);
	bypass_idx_oh_6(11) <= TEMP_96_double_out(2) or TEMP_96_double_out(16);
	bypass_idx_oh_6(10) <= TEMP_96_double_out(3) or TEMP_96_double_out(17);
	bypass_idx_oh_6(9) <= TEMP_96_double_out(4) or TEMP_96_double_out(18);
	bypass_idx_oh_6(8) <= TEMP_96_double_out(5) or TEMP_96_double_out(19);
	bypass_idx_oh_6(7) <= TEMP_96_double_out(6) or TEMP_96_double_out(20);
	bypass_idx_oh_6(6) <= TEMP_96_double_out(7) or TEMP_96_double_out(21);
	bypass_idx_oh_6(5) <= TEMP_96_double_out(8) or TEMP_96_double_out(22);
	bypass_idx_oh_6(4) <= TEMP_96_double_out(9) or TEMP_96_double_out(23);
	bypass_idx_oh_6(3) <= TEMP_96_double_out(10) or TEMP_96_double_out(24);
	bypass_idx_oh_6(2) <= TEMP_96_double_out(11) or TEMP_96_double_out(25);
	bypass_idx_oh_6(1) <= TEMP_96_double_out(12) or TEMP_96_double_out(26);
	bypass_idx_oh_6(0) <= TEMP_96_double_out(13) or TEMP_96_double_out(27);
	-- Priority Masking End

	bypass_en_vec_6 <= bypass_idx_oh_6 and can_bypass_6;
	-- Reduction Begin
	-- Reduce(bypass_en_6, bypass_en_vec_6, or)
	TEMP_97_res(0) <= bypass_en_vec_6(0) or bypass_en_vec_6(8);
	TEMP_97_res(1) <= bypass_en_vec_6(1) or bypass_en_vec_6(9);
	TEMP_97_res(2) <= bypass_en_vec_6(2) or bypass_en_vec_6(10);
	TEMP_97_res(3) <= bypass_en_vec_6(3) or bypass_en_vec_6(11);
	TEMP_97_res(4) <= bypass_en_vec_6(4) or bypass_en_vec_6(12);
	TEMP_97_res(5) <= bypass_en_vec_6(5) or bypass_en_vec_6(13);
	TEMP_97_res(6) <= bypass_en_vec_6(6);
	TEMP_97_res(7) <= bypass_en_vec_6(7);
	-- Layer End
	TEMP_98_res(0) <= TEMP_97_res(0) or TEMP_97_res(4);
	TEMP_98_res(1) <= TEMP_97_res(1) or TEMP_97_res(5);
	TEMP_98_res(2) <= TEMP_97_res(2) or TEMP_97_res(6);
	TEMP_98_res(3) <= TEMP_97_res(3) or TEMP_97_res(7);
	-- Layer End
	TEMP_99_res(0) <= TEMP_98_res(0) or TEMP_98_res(2);
	TEMP_99_res(1) <= TEMP_98_res(1) or TEMP_98_res(3);
	-- Layer End
	bypass_en_6 <= TEMP_99_res(0) or TEMP_99_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_7, ld_st_conflict_7, stq_last_oh)
	TEMP_100_double_in(0) <= ld_st_conflict_7(13);
	TEMP_100_double_in(14) <= ld_st_conflict_7(13);
	TEMP_100_double_in(1) <= ld_st_conflict_7(12);
	TEMP_100_double_in(15) <= ld_st_conflict_7(12);
	TEMP_100_double_in(2) <= ld_st_conflict_7(11);
	TEMP_100_double_in(16) <= ld_st_conflict_7(11);
	TEMP_100_double_in(3) <= ld_st_conflict_7(10);
	TEMP_100_double_in(17) <= ld_st_conflict_7(10);
	TEMP_100_double_in(4) <= ld_st_conflict_7(9);
	TEMP_100_double_in(18) <= ld_st_conflict_7(9);
	TEMP_100_double_in(5) <= ld_st_conflict_7(8);
	TEMP_100_double_in(19) <= ld_st_conflict_7(8);
	TEMP_100_double_in(6) <= ld_st_conflict_7(7);
	TEMP_100_double_in(20) <= ld_st_conflict_7(7);
	TEMP_100_double_in(7) <= ld_st_conflict_7(6);
	TEMP_100_double_in(21) <= ld_st_conflict_7(6);
	TEMP_100_double_in(8) <= ld_st_conflict_7(5);
	TEMP_100_double_in(22) <= ld_st_conflict_7(5);
	TEMP_100_double_in(9) <= ld_st_conflict_7(4);
	TEMP_100_double_in(23) <= ld_st_conflict_7(4);
	TEMP_100_double_in(10) <= ld_st_conflict_7(3);
	TEMP_100_double_in(24) <= ld_st_conflict_7(3);
	TEMP_100_double_in(11) <= ld_st_conflict_7(2);
	TEMP_100_double_in(25) <= ld_st_conflict_7(2);
	TEMP_100_double_in(12) <= ld_st_conflict_7(1);
	TEMP_100_double_in(26) <= ld_st_conflict_7(1);
	TEMP_100_double_in(13) <= ld_st_conflict_7(0);
	TEMP_100_double_in(27) <= ld_st_conflict_7(0);
	TEMP_100_base_rev(0) <= stq_last_oh(13);
	TEMP_100_base_rev(1) <= stq_last_oh(12);
	TEMP_100_base_rev(2) <= stq_last_oh(11);
	TEMP_100_base_rev(3) <= stq_last_oh(10);
	TEMP_100_base_rev(4) <= stq_last_oh(9);
	TEMP_100_base_rev(5) <= stq_last_oh(8);
	TEMP_100_base_rev(6) <= stq_last_oh(7);
	TEMP_100_base_rev(7) <= stq_last_oh(6);
	TEMP_100_base_rev(8) <= stq_last_oh(5);
	TEMP_100_base_rev(9) <= stq_last_oh(4);
	TEMP_100_base_rev(10) <= stq_last_oh(3);
	TEMP_100_base_rev(11) <= stq_last_oh(2);
	TEMP_100_base_rev(12) <= stq_last_oh(1);
	TEMP_100_base_rev(13) <= stq_last_oh(0);
	TEMP_100_double_out <= TEMP_100_double_in and not std_logic_vector( unsigned( TEMP_100_double_in ) - unsigned( "00000000000000" & TEMP_100_base_rev ) );
	bypass_idx_oh_7(13) <= TEMP_100_double_out(0) or TEMP_100_double_out(14);
	bypass_idx_oh_7(12) <= TEMP_100_double_out(1) or TEMP_100_double_out(15);
	bypass_idx_oh_7(11) <= TEMP_100_double_out(2) or TEMP_100_double_out(16);
	bypass_idx_oh_7(10) <= TEMP_100_double_out(3) or TEMP_100_double_out(17);
	bypass_idx_oh_7(9) <= TEMP_100_double_out(4) or TEMP_100_double_out(18);
	bypass_idx_oh_7(8) <= TEMP_100_double_out(5) or TEMP_100_double_out(19);
	bypass_idx_oh_7(7) <= TEMP_100_double_out(6) or TEMP_100_double_out(20);
	bypass_idx_oh_7(6) <= TEMP_100_double_out(7) or TEMP_100_double_out(21);
	bypass_idx_oh_7(5) <= TEMP_100_double_out(8) or TEMP_100_double_out(22);
	bypass_idx_oh_7(4) <= TEMP_100_double_out(9) or TEMP_100_double_out(23);
	bypass_idx_oh_7(3) <= TEMP_100_double_out(10) or TEMP_100_double_out(24);
	bypass_idx_oh_7(2) <= TEMP_100_double_out(11) or TEMP_100_double_out(25);
	bypass_idx_oh_7(1) <= TEMP_100_double_out(12) or TEMP_100_double_out(26);
	bypass_idx_oh_7(0) <= TEMP_100_double_out(13) or TEMP_100_double_out(27);
	-- Priority Masking End

	bypass_en_vec_7 <= bypass_idx_oh_7 and can_bypass_7;
	-- Reduction Begin
	-- Reduce(bypass_en_7, bypass_en_vec_7, or)
	TEMP_101_res(0) <= bypass_en_vec_7(0) or bypass_en_vec_7(8);
	TEMP_101_res(1) <= bypass_en_vec_7(1) or bypass_en_vec_7(9);
	TEMP_101_res(2) <= bypass_en_vec_7(2) or bypass_en_vec_7(10);
	TEMP_101_res(3) <= bypass_en_vec_7(3) or bypass_en_vec_7(11);
	TEMP_101_res(4) <= bypass_en_vec_7(4) or bypass_en_vec_7(12);
	TEMP_101_res(5) <= bypass_en_vec_7(5) or bypass_en_vec_7(13);
	TEMP_101_res(6) <= bypass_en_vec_7(6);
	TEMP_101_res(7) <= bypass_en_vec_7(7);
	-- Layer End
	TEMP_102_res(0) <= TEMP_101_res(0) or TEMP_101_res(4);
	TEMP_102_res(1) <= TEMP_101_res(1) or TEMP_101_res(5);
	TEMP_102_res(2) <= TEMP_101_res(2) or TEMP_101_res(6);
	TEMP_102_res(3) <= TEMP_101_res(3) or TEMP_101_res(7);
	-- Layer End
	TEMP_103_res(0) <= TEMP_102_res(0) or TEMP_102_res(2);
	TEMP_103_res(1) <= TEMP_102_res(1) or TEMP_102_res(3);
	-- Layer End
	bypass_en_7 <= TEMP_103_res(0) or TEMP_103_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_8, ld_st_conflict_8, stq_last_oh)
	TEMP_104_double_in(0) <= ld_st_conflict_8(13);
	TEMP_104_double_in(14) <= ld_st_conflict_8(13);
	TEMP_104_double_in(1) <= ld_st_conflict_8(12);
	TEMP_104_double_in(15) <= ld_st_conflict_8(12);
	TEMP_104_double_in(2) <= ld_st_conflict_8(11);
	TEMP_104_double_in(16) <= ld_st_conflict_8(11);
	TEMP_104_double_in(3) <= ld_st_conflict_8(10);
	TEMP_104_double_in(17) <= ld_st_conflict_8(10);
	TEMP_104_double_in(4) <= ld_st_conflict_8(9);
	TEMP_104_double_in(18) <= ld_st_conflict_8(9);
	TEMP_104_double_in(5) <= ld_st_conflict_8(8);
	TEMP_104_double_in(19) <= ld_st_conflict_8(8);
	TEMP_104_double_in(6) <= ld_st_conflict_8(7);
	TEMP_104_double_in(20) <= ld_st_conflict_8(7);
	TEMP_104_double_in(7) <= ld_st_conflict_8(6);
	TEMP_104_double_in(21) <= ld_st_conflict_8(6);
	TEMP_104_double_in(8) <= ld_st_conflict_8(5);
	TEMP_104_double_in(22) <= ld_st_conflict_8(5);
	TEMP_104_double_in(9) <= ld_st_conflict_8(4);
	TEMP_104_double_in(23) <= ld_st_conflict_8(4);
	TEMP_104_double_in(10) <= ld_st_conflict_8(3);
	TEMP_104_double_in(24) <= ld_st_conflict_8(3);
	TEMP_104_double_in(11) <= ld_st_conflict_8(2);
	TEMP_104_double_in(25) <= ld_st_conflict_8(2);
	TEMP_104_double_in(12) <= ld_st_conflict_8(1);
	TEMP_104_double_in(26) <= ld_st_conflict_8(1);
	TEMP_104_double_in(13) <= ld_st_conflict_8(0);
	TEMP_104_double_in(27) <= ld_st_conflict_8(0);
	TEMP_104_base_rev(0) <= stq_last_oh(13);
	TEMP_104_base_rev(1) <= stq_last_oh(12);
	TEMP_104_base_rev(2) <= stq_last_oh(11);
	TEMP_104_base_rev(3) <= stq_last_oh(10);
	TEMP_104_base_rev(4) <= stq_last_oh(9);
	TEMP_104_base_rev(5) <= stq_last_oh(8);
	TEMP_104_base_rev(6) <= stq_last_oh(7);
	TEMP_104_base_rev(7) <= stq_last_oh(6);
	TEMP_104_base_rev(8) <= stq_last_oh(5);
	TEMP_104_base_rev(9) <= stq_last_oh(4);
	TEMP_104_base_rev(10) <= stq_last_oh(3);
	TEMP_104_base_rev(11) <= stq_last_oh(2);
	TEMP_104_base_rev(12) <= stq_last_oh(1);
	TEMP_104_base_rev(13) <= stq_last_oh(0);
	TEMP_104_double_out <= TEMP_104_double_in and not std_logic_vector( unsigned( TEMP_104_double_in ) - unsigned( "00000000000000" & TEMP_104_base_rev ) );
	bypass_idx_oh_8(13) <= TEMP_104_double_out(0) or TEMP_104_double_out(14);
	bypass_idx_oh_8(12) <= TEMP_104_double_out(1) or TEMP_104_double_out(15);
	bypass_idx_oh_8(11) <= TEMP_104_double_out(2) or TEMP_104_double_out(16);
	bypass_idx_oh_8(10) <= TEMP_104_double_out(3) or TEMP_104_double_out(17);
	bypass_idx_oh_8(9) <= TEMP_104_double_out(4) or TEMP_104_double_out(18);
	bypass_idx_oh_8(8) <= TEMP_104_double_out(5) or TEMP_104_double_out(19);
	bypass_idx_oh_8(7) <= TEMP_104_double_out(6) or TEMP_104_double_out(20);
	bypass_idx_oh_8(6) <= TEMP_104_double_out(7) or TEMP_104_double_out(21);
	bypass_idx_oh_8(5) <= TEMP_104_double_out(8) or TEMP_104_double_out(22);
	bypass_idx_oh_8(4) <= TEMP_104_double_out(9) or TEMP_104_double_out(23);
	bypass_idx_oh_8(3) <= TEMP_104_double_out(10) or TEMP_104_double_out(24);
	bypass_idx_oh_8(2) <= TEMP_104_double_out(11) or TEMP_104_double_out(25);
	bypass_idx_oh_8(1) <= TEMP_104_double_out(12) or TEMP_104_double_out(26);
	bypass_idx_oh_8(0) <= TEMP_104_double_out(13) or TEMP_104_double_out(27);
	-- Priority Masking End

	bypass_en_vec_8 <= bypass_idx_oh_8 and can_bypass_8;
	-- Reduction Begin
	-- Reduce(bypass_en_8, bypass_en_vec_8, or)
	TEMP_105_res(0) <= bypass_en_vec_8(0) or bypass_en_vec_8(8);
	TEMP_105_res(1) <= bypass_en_vec_8(1) or bypass_en_vec_8(9);
	TEMP_105_res(2) <= bypass_en_vec_8(2) or bypass_en_vec_8(10);
	TEMP_105_res(3) <= bypass_en_vec_8(3) or bypass_en_vec_8(11);
	TEMP_105_res(4) <= bypass_en_vec_8(4) or bypass_en_vec_8(12);
	TEMP_105_res(5) <= bypass_en_vec_8(5) or bypass_en_vec_8(13);
	TEMP_105_res(6) <= bypass_en_vec_8(6);
	TEMP_105_res(7) <= bypass_en_vec_8(7);
	-- Layer End
	TEMP_106_res(0) <= TEMP_105_res(0) or TEMP_105_res(4);
	TEMP_106_res(1) <= TEMP_105_res(1) or TEMP_105_res(5);
	TEMP_106_res(2) <= TEMP_105_res(2) or TEMP_105_res(6);
	TEMP_106_res(3) <= TEMP_105_res(3) or TEMP_105_res(7);
	-- Layer End
	TEMP_107_res(0) <= TEMP_106_res(0) or TEMP_106_res(2);
	TEMP_107_res(1) <= TEMP_106_res(1) or TEMP_106_res(3);
	-- Layer End
	bypass_en_8 <= TEMP_107_res(0) or TEMP_107_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_9, ld_st_conflict_9, stq_last_oh)
	TEMP_108_double_in(0) <= ld_st_conflict_9(13);
	TEMP_108_double_in(14) <= ld_st_conflict_9(13);
	TEMP_108_double_in(1) <= ld_st_conflict_9(12);
	TEMP_108_double_in(15) <= ld_st_conflict_9(12);
	TEMP_108_double_in(2) <= ld_st_conflict_9(11);
	TEMP_108_double_in(16) <= ld_st_conflict_9(11);
	TEMP_108_double_in(3) <= ld_st_conflict_9(10);
	TEMP_108_double_in(17) <= ld_st_conflict_9(10);
	TEMP_108_double_in(4) <= ld_st_conflict_9(9);
	TEMP_108_double_in(18) <= ld_st_conflict_9(9);
	TEMP_108_double_in(5) <= ld_st_conflict_9(8);
	TEMP_108_double_in(19) <= ld_st_conflict_9(8);
	TEMP_108_double_in(6) <= ld_st_conflict_9(7);
	TEMP_108_double_in(20) <= ld_st_conflict_9(7);
	TEMP_108_double_in(7) <= ld_st_conflict_9(6);
	TEMP_108_double_in(21) <= ld_st_conflict_9(6);
	TEMP_108_double_in(8) <= ld_st_conflict_9(5);
	TEMP_108_double_in(22) <= ld_st_conflict_9(5);
	TEMP_108_double_in(9) <= ld_st_conflict_9(4);
	TEMP_108_double_in(23) <= ld_st_conflict_9(4);
	TEMP_108_double_in(10) <= ld_st_conflict_9(3);
	TEMP_108_double_in(24) <= ld_st_conflict_9(3);
	TEMP_108_double_in(11) <= ld_st_conflict_9(2);
	TEMP_108_double_in(25) <= ld_st_conflict_9(2);
	TEMP_108_double_in(12) <= ld_st_conflict_9(1);
	TEMP_108_double_in(26) <= ld_st_conflict_9(1);
	TEMP_108_double_in(13) <= ld_st_conflict_9(0);
	TEMP_108_double_in(27) <= ld_st_conflict_9(0);
	TEMP_108_base_rev(0) <= stq_last_oh(13);
	TEMP_108_base_rev(1) <= stq_last_oh(12);
	TEMP_108_base_rev(2) <= stq_last_oh(11);
	TEMP_108_base_rev(3) <= stq_last_oh(10);
	TEMP_108_base_rev(4) <= stq_last_oh(9);
	TEMP_108_base_rev(5) <= stq_last_oh(8);
	TEMP_108_base_rev(6) <= stq_last_oh(7);
	TEMP_108_base_rev(7) <= stq_last_oh(6);
	TEMP_108_base_rev(8) <= stq_last_oh(5);
	TEMP_108_base_rev(9) <= stq_last_oh(4);
	TEMP_108_base_rev(10) <= stq_last_oh(3);
	TEMP_108_base_rev(11) <= stq_last_oh(2);
	TEMP_108_base_rev(12) <= stq_last_oh(1);
	TEMP_108_base_rev(13) <= stq_last_oh(0);
	TEMP_108_double_out <= TEMP_108_double_in and not std_logic_vector( unsigned( TEMP_108_double_in ) - unsigned( "00000000000000" & TEMP_108_base_rev ) );
	bypass_idx_oh_9(13) <= TEMP_108_double_out(0) or TEMP_108_double_out(14);
	bypass_idx_oh_9(12) <= TEMP_108_double_out(1) or TEMP_108_double_out(15);
	bypass_idx_oh_9(11) <= TEMP_108_double_out(2) or TEMP_108_double_out(16);
	bypass_idx_oh_9(10) <= TEMP_108_double_out(3) or TEMP_108_double_out(17);
	bypass_idx_oh_9(9) <= TEMP_108_double_out(4) or TEMP_108_double_out(18);
	bypass_idx_oh_9(8) <= TEMP_108_double_out(5) or TEMP_108_double_out(19);
	bypass_idx_oh_9(7) <= TEMP_108_double_out(6) or TEMP_108_double_out(20);
	bypass_idx_oh_9(6) <= TEMP_108_double_out(7) or TEMP_108_double_out(21);
	bypass_idx_oh_9(5) <= TEMP_108_double_out(8) or TEMP_108_double_out(22);
	bypass_idx_oh_9(4) <= TEMP_108_double_out(9) or TEMP_108_double_out(23);
	bypass_idx_oh_9(3) <= TEMP_108_double_out(10) or TEMP_108_double_out(24);
	bypass_idx_oh_9(2) <= TEMP_108_double_out(11) or TEMP_108_double_out(25);
	bypass_idx_oh_9(1) <= TEMP_108_double_out(12) or TEMP_108_double_out(26);
	bypass_idx_oh_9(0) <= TEMP_108_double_out(13) or TEMP_108_double_out(27);
	-- Priority Masking End

	bypass_en_vec_9 <= bypass_idx_oh_9 and can_bypass_9;
	-- Reduction Begin
	-- Reduce(bypass_en_9, bypass_en_vec_9, or)
	TEMP_109_res(0) <= bypass_en_vec_9(0) or bypass_en_vec_9(8);
	TEMP_109_res(1) <= bypass_en_vec_9(1) or bypass_en_vec_9(9);
	TEMP_109_res(2) <= bypass_en_vec_9(2) or bypass_en_vec_9(10);
	TEMP_109_res(3) <= bypass_en_vec_9(3) or bypass_en_vec_9(11);
	TEMP_109_res(4) <= bypass_en_vec_9(4) or bypass_en_vec_9(12);
	TEMP_109_res(5) <= bypass_en_vec_9(5) or bypass_en_vec_9(13);
	TEMP_109_res(6) <= bypass_en_vec_9(6);
	TEMP_109_res(7) <= bypass_en_vec_9(7);
	-- Layer End
	TEMP_110_res(0) <= TEMP_109_res(0) or TEMP_109_res(4);
	TEMP_110_res(1) <= TEMP_109_res(1) or TEMP_109_res(5);
	TEMP_110_res(2) <= TEMP_109_res(2) or TEMP_109_res(6);
	TEMP_110_res(3) <= TEMP_109_res(3) or TEMP_109_res(7);
	-- Layer End
	TEMP_111_res(0) <= TEMP_110_res(0) or TEMP_110_res(2);
	TEMP_111_res(1) <= TEMP_110_res(1) or TEMP_110_res(3);
	-- Layer End
	bypass_en_9 <= TEMP_111_res(0) or TEMP_111_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_10, ld_st_conflict_10, stq_last_oh)
	TEMP_112_double_in(0) <= ld_st_conflict_10(13);
	TEMP_112_double_in(14) <= ld_st_conflict_10(13);
	TEMP_112_double_in(1) <= ld_st_conflict_10(12);
	TEMP_112_double_in(15) <= ld_st_conflict_10(12);
	TEMP_112_double_in(2) <= ld_st_conflict_10(11);
	TEMP_112_double_in(16) <= ld_st_conflict_10(11);
	TEMP_112_double_in(3) <= ld_st_conflict_10(10);
	TEMP_112_double_in(17) <= ld_st_conflict_10(10);
	TEMP_112_double_in(4) <= ld_st_conflict_10(9);
	TEMP_112_double_in(18) <= ld_st_conflict_10(9);
	TEMP_112_double_in(5) <= ld_st_conflict_10(8);
	TEMP_112_double_in(19) <= ld_st_conflict_10(8);
	TEMP_112_double_in(6) <= ld_st_conflict_10(7);
	TEMP_112_double_in(20) <= ld_st_conflict_10(7);
	TEMP_112_double_in(7) <= ld_st_conflict_10(6);
	TEMP_112_double_in(21) <= ld_st_conflict_10(6);
	TEMP_112_double_in(8) <= ld_st_conflict_10(5);
	TEMP_112_double_in(22) <= ld_st_conflict_10(5);
	TEMP_112_double_in(9) <= ld_st_conflict_10(4);
	TEMP_112_double_in(23) <= ld_st_conflict_10(4);
	TEMP_112_double_in(10) <= ld_st_conflict_10(3);
	TEMP_112_double_in(24) <= ld_st_conflict_10(3);
	TEMP_112_double_in(11) <= ld_st_conflict_10(2);
	TEMP_112_double_in(25) <= ld_st_conflict_10(2);
	TEMP_112_double_in(12) <= ld_st_conflict_10(1);
	TEMP_112_double_in(26) <= ld_st_conflict_10(1);
	TEMP_112_double_in(13) <= ld_st_conflict_10(0);
	TEMP_112_double_in(27) <= ld_st_conflict_10(0);
	TEMP_112_base_rev(0) <= stq_last_oh(13);
	TEMP_112_base_rev(1) <= stq_last_oh(12);
	TEMP_112_base_rev(2) <= stq_last_oh(11);
	TEMP_112_base_rev(3) <= stq_last_oh(10);
	TEMP_112_base_rev(4) <= stq_last_oh(9);
	TEMP_112_base_rev(5) <= stq_last_oh(8);
	TEMP_112_base_rev(6) <= stq_last_oh(7);
	TEMP_112_base_rev(7) <= stq_last_oh(6);
	TEMP_112_base_rev(8) <= stq_last_oh(5);
	TEMP_112_base_rev(9) <= stq_last_oh(4);
	TEMP_112_base_rev(10) <= stq_last_oh(3);
	TEMP_112_base_rev(11) <= stq_last_oh(2);
	TEMP_112_base_rev(12) <= stq_last_oh(1);
	TEMP_112_base_rev(13) <= stq_last_oh(0);
	TEMP_112_double_out <= TEMP_112_double_in and not std_logic_vector( unsigned( TEMP_112_double_in ) - unsigned( "00000000000000" & TEMP_112_base_rev ) );
	bypass_idx_oh_10(13) <= TEMP_112_double_out(0) or TEMP_112_double_out(14);
	bypass_idx_oh_10(12) <= TEMP_112_double_out(1) or TEMP_112_double_out(15);
	bypass_idx_oh_10(11) <= TEMP_112_double_out(2) or TEMP_112_double_out(16);
	bypass_idx_oh_10(10) <= TEMP_112_double_out(3) or TEMP_112_double_out(17);
	bypass_idx_oh_10(9) <= TEMP_112_double_out(4) or TEMP_112_double_out(18);
	bypass_idx_oh_10(8) <= TEMP_112_double_out(5) or TEMP_112_double_out(19);
	bypass_idx_oh_10(7) <= TEMP_112_double_out(6) or TEMP_112_double_out(20);
	bypass_idx_oh_10(6) <= TEMP_112_double_out(7) or TEMP_112_double_out(21);
	bypass_idx_oh_10(5) <= TEMP_112_double_out(8) or TEMP_112_double_out(22);
	bypass_idx_oh_10(4) <= TEMP_112_double_out(9) or TEMP_112_double_out(23);
	bypass_idx_oh_10(3) <= TEMP_112_double_out(10) or TEMP_112_double_out(24);
	bypass_idx_oh_10(2) <= TEMP_112_double_out(11) or TEMP_112_double_out(25);
	bypass_idx_oh_10(1) <= TEMP_112_double_out(12) or TEMP_112_double_out(26);
	bypass_idx_oh_10(0) <= TEMP_112_double_out(13) or TEMP_112_double_out(27);
	-- Priority Masking End

	bypass_en_vec_10 <= bypass_idx_oh_10 and can_bypass_10;
	-- Reduction Begin
	-- Reduce(bypass_en_10, bypass_en_vec_10, or)
	TEMP_113_res(0) <= bypass_en_vec_10(0) or bypass_en_vec_10(8);
	TEMP_113_res(1) <= bypass_en_vec_10(1) or bypass_en_vec_10(9);
	TEMP_113_res(2) <= bypass_en_vec_10(2) or bypass_en_vec_10(10);
	TEMP_113_res(3) <= bypass_en_vec_10(3) or bypass_en_vec_10(11);
	TEMP_113_res(4) <= bypass_en_vec_10(4) or bypass_en_vec_10(12);
	TEMP_113_res(5) <= bypass_en_vec_10(5) or bypass_en_vec_10(13);
	TEMP_113_res(6) <= bypass_en_vec_10(6);
	TEMP_113_res(7) <= bypass_en_vec_10(7);
	-- Layer End
	TEMP_114_res(0) <= TEMP_113_res(0) or TEMP_113_res(4);
	TEMP_114_res(1) <= TEMP_113_res(1) or TEMP_113_res(5);
	TEMP_114_res(2) <= TEMP_113_res(2) or TEMP_113_res(6);
	TEMP_114_res(3) <= TEMP_113_res(3) or TEMP_113_res(7);
	-- Layer End
	TEMP_115_res(0) <= TEMP_114_res(0) or TEMP_114_res(2);
	TEMP_115_res(1) <= TEMP_114_res(1) or TEMP_114_res(3);
	-- Layer End
	bypass_en_10 <= TEMP_115_res(0) or TEMP_115_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_11, ld_st_conflict_11, stq_last_oh)
	TEMP_116_double_in(0) <= ld_st_conflict_11(13);
	TEMP_116_double_in(14) <= ld_st_conflict_11(13);
	TEMP_116_double_in(1) <= ld_st_conflict_11(12);
	TEMP_116_double_in(15) <= ld_st_conflict_11(12);
	TEMP_116_double_in(2) <= ld_st_conflict_11(11);
	TEMP_116_double_in(16) <= ld_st_conflict_11(11);
	TEMP_116_double_in(3) <= ld_st_conflict_11(10);
	TEMP_116_double_in(17) <= ld_st_conflict_11(10);
	TEMP_116_double_in(4) <= ld_st_conflict_11(9);
	TEMP_116_double_in(18) <= ld_st_conflict_11(9);
	TEMP_116_double_in(5) <= ld_st_conflict_11(8);
	TEMP_116_double_in(19) <= ld_st_conflict_11(8);
	TEMP_116_double_in(6) <= ld_st_conflict_11(7);
	TEMP_116_double_in(20) <= ld_st_conflict_11(7);
	TEMP_116_double_in(7) <= ld_st_conflict_11(6);
	TEMP_116_double_in(21) <= ld_st_conflict_11(6);
	TEMP_116_double_in(8) <= ld_st_conflict_11(5);
	TEMP_116_double_in(22) <= ld_st_conflict_11(5);
	TEMP_116_double_in(9) <= ld_st_conflict_11(4);
	TEMP_116_double_in(23) <= ld_st_conflict_11(4);
	TEMP_116_double_in(10) <= ld_st_conflict_11(3);
	TEMP_116_double_in(24) <= ld_st_conflict_11(3);
	TEMP_116_double_in(11) <= ld_st_conflict_11(2);
	TEMP_116_double_in(25) <= ld_st_conflict_11(2);
	TEMP_116_double_in(12) <= ld_st_conflict_11(1);
	TEMP_116_double_in(26) <= ld_st_conflict_11(1);
	TEMP_116_double_in(13) <= ld_st_conflict_11(0);
	TEMP_116_double_in(27) <= ld_st_conflict_11(0);
	TEMP_116_base_rev(0) <= stq_last_oh(13);
	TEMP_116_base_rev(1) <= stq_last_oh(12);
	TEMP_116_base_rev(2) <= stq_last_oh(11);
	TEMP_116_base_rev(3) <= stq_last_oh(10);
	TEMP_116_base_rev(4) <= stq_last_oh(9);
	TEMP_116_base_rev(5) <= stq_last_oh(8);
	TEMP_116_base_rev(6) <= stq_last_oh(7);
	TEMP_116_base_rev(7) <= stq_last_oh(6);
	TEMP_116_base_rev(8) <= stq_last_oh(5);
	TEMP_116_base_rev(9) <= stq_last_oh(4);
	TEMP_116_base_rev(10) <= stq_last_oh(3);
	TEMP_116_base_rev(11) <= stq_last_oh(2);
	TEMP_116_base_rev(12) <= stq_last_oh(1);
	TEMP_116_base_rev(13) <= stq_last_oh(0);
	TEMP_116_double_out <= TEMP_116_double_in and not std_logic_vector( unsigned( TEMP_116_double_in ) - unsigned( "00000000000000" & TEMP_116_base_rev ) );
	bypass_idx_oh_11(13) <= TEMP_116_double_out(0) or TEMP_116_double_out(14);
	bypass_idx_oh_11(12) <= TEMP_116_double_out(1) or TEMP_116_double_out(15);
	bypass_idx_oh_11(11) <= TEMP_116_double_out(2) or TEMP_116_double_out(16);
	bypass_idx_oh_11(10) <= TEMP_116_double_out(3) or TEMP_116_double_out(17);
	bypass_idx_oh_11(9) <= TEMP_116_double_out(4) or TEMP_116_double_out(18);
	bypass_idx_oh_11(8) <= TEMP_116_double_out(5) or TEMP_116_double_out(19);
	bypass_idx_oh_11(7) <= TEMP_116_double_out(6) or TEMP_116_double_out(20);
	bypass_idx_oh_11(6) <= TEMP_116_double_out(7) or TEMP_116_double_out(21);
	bypass_idx_oh_11(5) <= TEMP_116_double_out(8) or TEMP_116_double_out(22);
	bypass_idx_oh_11(4) <= TEMP_116_double_out(9) or TEMP_116_double_out(23);
	bypass_idx_oh_11(3) <= TEMP_116_double_out(10) or TEMP_116_double_out(24);
	bypass_idx_oh_11(2) <= TEMP_116_double_out(11) or TEMP_116_double_out(25);
	bypass_idx_oh_11(1) <= TEMP_116_double_out(12) or TEMP_116_double_out(26);
	bypass_idx_oh_11(0) <= TEMP_116_double_out(13) or TEMP_116_double_out(27);
	-- Priority Masking End

	bypass_en_vec_11 <= bypass_idx_oh_11 and can_bypass_11;
	-- Reduction Begin
	-- Reduce(bypass_en_11, bypass_en_vec_11, or)
	TEMP_117_res(0) <= bypass_en_vec_11(0) or bypass_en_vec_11(8);
	TEMP_117_res(1) <= bypass_en_vec_11(1) or bypass_en_vec_11(9);
	TEMP_117_res(2) <= bypass_en_vec_11(2) or bypass_en_vec_11(10);
	TEMP_117_res(3) <= bypass_en_vec_11(3) or bypass_en_vec_11(11);
	TEMP_117_res(4) <= bypass_en_vec_11(4) or bypass_en_vec_11(12);
	TEMP_117_res(5) <= bypass_en_vec_11(5) or bypass_en_vec_11(13);
	TEMP_117_res(6) <= bypass_en_vec_11(6);
	TEMP_117_res(7) <= bypass_en_vec_11(7);
	-- Layer End
	TEMP_118_res(0) <= TEMP_117_res(0) or TEMP_117_res(4);
	TEMP_118_res(1) <= TEMP_117_res(1) or TEMP_117_res(5);
	TEMP_118_res(2) <= TEMP_117_res(2) or TEMP_117_res(6);
	TEMP_118_res(3) <= TEMP_117_res(3) or TEMP_117_res(7);
	-- Layer End
	TEMP_119_res(0) <= TEMP_118_res(0) or TEMP_118_res(2);
	TEMP_119_res(1) <= TEMP_118_res(1) or TEMP_118_res(3);
	-- Layer End
	bypass_en_11 <= TEMP_119_res(0) or TEMP_119_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_12, ld_st_conflict_12, stq_last_oh)
	TEMP_120_double_in(0) <= ld_st_conflict_12(13);
	TEMP_120_double_in(14) <= ld_st_conflict_12(13);
	TEMP_120_double_in(1) <= ld_st_conflict_12(12);
	TEMP_120_double_in(15) <= ld_st_conflict_12(12);
	TEMP_120_double_in(2) <= ld_st_conflict_12(11);
	TEMP_120_double_in(16) <= ld_st_conflict_12(11);
	TEMP_120_double_in(3) <= ld_st_conflict_12(10);
	TEMP_120_double_in(17) <= ld_st_conflict_12(10);
	TEMP_120_double_in(4) <= ld_st_conflict_12(9);
	TEMP_120_double_in(18) <= ld_st_conflict_12(9);
	TEMP_120_double_in(5) <= ld_st_conflict_12(8);
	TEMP_120_double_in(19) <= ld_st_conflict_12(8);
	TEMP_120_double_in(6) <= ld_st_conflict_12(7);
	TEMP_120_double_in(20) <= ld_st_conflict_12(7);
	TEMP_120_double_in(7) <= ld_st_conflict_12(6);
	TEMP_120_double_in(21) <= ld_st_conflict_12(6);
	TEMP_120_double_in(8) <= ld_st_conflict_12(5);
	TEMP_120_double_in(22) <= ld_st_conflict_12(5);
	TEMP_120_double_in(9) <= ld_st_conflict_12(4);
	TEMP_120_double_in(23) <= ld_st_conflict_12(4);
	TEMP_120_double_in(10) <= ld_st_conflict_12(3);
	TEMP_120_double_in(24) <= ld_st_conflict_12(3);
	TEMP_120_double_in(11) <= ld_st_conflict_12(2);
	TEMP_120_double_in(25) <= ld_st_conflict_12(2);
	TEMP_120_double_in(12) <= ld_st_conflict_12(1);
	TEMP_120_double_in(26) <= ld_st_conflict_12(1);
	TEMP_120_double_in(13) <= ld_st_conflict_12(0);
	TEMP_120_double_in(27) <= ld_st_conflict_12(0);
	TEMP_120_base_rev(0) <= stq_last_oh(13);
	TEMP_120_base_rev(1) <= stq_last_oh(12);
	TEMP_120_base_rev(2) <= stq_last_oh(11);
	TEMP_120_base_rev(3) <= stq_last_oh(10);
	TEMP_120_base_rev(4) <= stq_last_oh(9);
	TEMP_120_base_rev(5) <= stq_last_oh(8);
	TEMP_120_base_rev(6) <= stq_last_oh(7);
	TEMP_120_base_rev(7) <= stq_last_oh(6);
	TEMP_120_base_rev(8) <= stq_last_oh(5);
	TEMP_120_base_rev(9) <= stq_last_oh(4);
	TEMP_120_base_rev(10) <= stq_last_oh(3);
	TEMP_120_base_rev(11) <= stq_last_oh(2);
	TEMP_120_base_rev(12) <= stq_last_oh(1);
	TEMP_120_base_rev(13) <= stq_last_oh(0);
	TEMP_120_double_out <= TEMP_120_double_in and not std_logic_vector( unsigned( TEMP_120_double_in ) - unsigned( "00000000000000" & TEMP_120_base_rev ) );
	bypass_idx_oh_12(13) <= TEMP_120_double_out(0) or TEMP_120_double_out(14);
	bypass_idx_oh_12(12) <= TEMP_120_double_out(1) or TEMP_120_double_out(15);
	bypass_idx_oh_12(11) <= TEMP_120_double_out(2) or TEMP_120_double_out(16);
	bypass_idx_oh_12(10) <= TEMP_120_double_out(3) or TEMP_120_double_out(17);
	bypass_idx_oh_12(9) <= TEMP_120_double_out(4) or TEMP_120_double_out(18);
	bypass_idx_oh_12(8) <= TEMP_120_double_out(5) or TEMP_120_double_out(19);
	bypass_idx_oh_12(7) <= TEMP_120_double_out(6) or TEMP_120_double_out(20);
	bypass_idx_oh_12(6) <= TEMP_120_double_out(7) or TEMP_120_double_out(21);
	bypass_idx_oh_12(5) <= TEMP_120_double_out(8) or TEMP_120_double_out(22);
	bypass_idx_oh_12(4) <= TEMP_120_double_out(9) or TEMP_120_double_out(23);
	bypass_idx_oh_12(3) <= TEMP_120_double_out(10) or TEMP_120_double_out(24);
	bypass_idx_oh_12(2) <= TEMP_120_double_out(11) or TEMP_120_double_out(25);
	bypass_idx_oh_12(1) <= TEMP_120_double_out(12) or TEMP_120_double_out(26);
	bypass_idx_oh_12(0) <= TEMP_120_double_out(13) or TEMP_120_double_out(27);
	-- Priority Masking End

	bypass_en_vec_12 <= bypass_idx_oh_12 and can_bypass_12;
	-- Reduction Begin
	-- Reduce(bypass_en_12, bypass_en_vec_12, or)
	TEMP_121_res(0) <= bypass_en_vec_12(0) or bypass_en_vec_12(8);
	TEMP_121_res(1) <= bypass_en_vec_12(1) or bypass_en_vec_12(9);
	TEMP_121_res(2) <= bypass_en_vec_12(2) or bypass_en_vec_12(10);
	TEMP_121_res(3) <= bypass_en_vec_12(3) or bypass_en_vec_12(11);
	TEMP_121_res(4) <= bypass_en_vec_12(4) or bypass_en_vec_12(12);
	TEMP_121_res(5) <= bypass_en_vec_12(5) or bypass_en_vec_12(13);
	TEMP_121_res(6) <= bypass_en_vec_12(6);
	TEMP_121_res(7) <= bypass_en_vec_12(7);
	-- Layer End
	TEMP_122_res(0) <= TEMP_121_res(0) or TEMP_121_res(4);
	TEMP_122_res(1) <= TEMP_121_res(1) or TEMP_121_res(5);
	TEMP_122_res(2) <= TEMP_121_res(2) or TEMP_121_res(6);
	TEMP_122_res(3) <= TEMP_121_res(3) or TEMP_121_res(7);
	-- Layer End
	TEMP_123_res(0) <= TEMP_122_res(0) or TEMP_122_res(2);
	TEMP_123_res(1) <= TEMP_122_res(1) or TEMP_122_res(3);
	-- Layer End
	bypass_en_12 <= TEMP_123_res(0) or TEMP_123_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_13, ld_st_conflict_13, stq_last_oh)
	TEMP_124_double_in(0) <= ld_st_conflict_13(13);
	TEMP_124_double_in(14) <= ld_st_conflict_13(13);
	TEMP_124_double_in(1) <= ld_st_conflict_13(12);
	TEMP_124_double_in(15) <= ld_st_conflict_13(12);
	TEMP_124_double_in(2) <= ld_st_conflict_13(11);
	TEMP_124_double_in(16) <= ld_st_conflict_13(11);
	TEMP_124_double_in(3) <= ld_st_conflict_13(10);
	TEMP_124_double_in(17) <= ld_st_conflict_13(10);
	TEMP_124_double_in(4) <= ld_st_conflict_13(9);
	TEMP_124_double_in(18) <= ld_st_conflict_13(9);
	TEMP_124_double_in(5) <= ld_st_conflict_13(8);
	TEMP_124_double_in(19) <= ld_st_conflict_13(8);
	TEMP_124_double_in(6) <= ld_st_conflict_13(7);
	TEMP_124_double_in(20) <= ld_st_conflict_13(7);
	TEMP_124_double_in(7) <= ld_st_conflict_13(6);
	TEMP_124_double_in(21) <= ld_st_conflict_13(6);
	TEMP_124_double_in(8) <= ld_st_conflict_13(5);
	TEMP_124_double_in(22) <= ld_st_conflict_13(5);
	TEMP_124_double_in(9) <= ld_st_conflict_13(4);
	TEMP_124_double_in(23) <= ld_st_conflict_13(4);
	TEMP_124_double_in(10) <= ld_st_conflict_13(3);
	TEMP_124_double_in(24) <= ld_st_conflict_13(3);
	TEMP_124_double_in(11) <= ld_st_conflict_13(2);
	TEMP_124_double_in(25) <= ld_st_conflict_13(2);
	TEMP_124_double_in(12) <= ld_st_conflict_13(1);
	TEMP_124_double_in(26) <= ld_st_conflict_13(1);
	TEMP_124_double_in(13) <= ld_st_conflict_13(0);
	TEMP_124_double_in(27) <= ld_st_conflict_13(0);
	TEMP_124_base_rev(0) <= stq_last_oh(13);
	TEMP_124_base_rev(1) <= stq_last_oh(12);
	TEMP_124_base_rev(2) <= stq_last_oh(11);
	TEMP_124_base_rev(3) <= stq_last_oh(10);
	TEMP_124_base_rev(4) <= stq_last_oh(9);
	TEMP_124_base_rev(5) <= stq_last_oh(8);
	TEMP_124_base_rev(6) <= stq_last_oh(7);
	TEMP_124_base_rev(7) <= stq_last_oh(6);
	TEMP_124_base_rev(8) <= stq_last_oh(5);
	TEMP_124_base_rev(9) <= stq_last_oh(4);
	TEMP_124_base_rev(10) <= stq_last_oh(3);
	TEMP_124_base_rev(11) <= stq_last_oh(2);
	TEMP_124_base_rev(12) <= stq_last_oh(1);
	TEMP_124_base_rev(13) <= stq_last_oh(0);
	TEMP_124_double_out <= TEMP_124_double_in and not std_logic_vector( unsigned( TEMP_124_double_in ) - unsigned( "00000000000000" & TEMP_124_base_rev ) );
	bypass_idx_oh_13(13) <= TEMP_124_double_out(0) or TEMP_124_double_out(14);
	bypass_idx_oh_13(12) <= TEMP_124_double_out(1) or TEMP_124_double_out(15);
	bypass_idx_oh_13(11) <= TEMP_124_double_out(2) or TEMP_124_double_out(16);
	bypass_idx_oh_13(10) <= TEMP_124_double_out(3) or TEMP_124_double_out(17);
	bypass_idx_oh_13(9) <= TEMP_124_double_out(4) or TEMP_124_double_out(18);
	bypass_idx_oh_13(8) <= TEMP_124_double_out(5) or TEMP_124_double_out(19);
	bypass_idx_oh_13(7) <= TEMP_124_double_out(6) or TEMP_124_double_out(20);
	bypass_idx_oh_13(6) <= TEMP_124_double_out(7) or TEMP_124_double_out(21);
	bypass_idx_oh_13(5) <= TEMP_124_double_out(8) or TEMP_124_double_out(22);
	bypass_idx_oh_13(4) <= TEMP_124_double_out(9) or TEMP_124_double_out(23);
	bypass_idx_oh_13(3) <= TEMP_124_double_out(10) or TEMP_124_double_out(24);
	bypass_idx_oh_13(2) <= TEMP_124_double_out(11) or TEMP_124_double_out(25);
	bypass_idx_oh_13(1) <= TEMP_124_double_out(12) or TEMP_124_double_out(26);
	bypass_idx_oh_13(0) <= TEMP_124_double_out(13) or TEMP_124_double_out(27);
	-- Priority Masking End

	bypass_en_vec_13 <= bypass_idx_oh_13 and can_bypass_13;
	-- Reduction Begin
	-- Reduce(bypass_en_13, bypass_en_vec_13, or)
	TEMP_125_res(0) <= bypass_en_vec_13(0) or bypass_en_vec_13(8);
	TEMP_125_res(1) <= bypass_en_vec_13(1) or bypass_en_vec_13(9);
	TEMP_125_res(2) <= bypass_en_vec_13(2) or bypass_en_vec_13(10);
	TEMP_125_res(3) <= bypass_en_vec_13(3) or bypass_en_vec_13(11);
	TEMP_125_res(4) <= bypass_en_vec_13(4) or bypass_en_vec_13(12);
	TEMP_125_res(5) <= bypass_en_vec_13(5) or bypass_en_vec_13(13);
	TEMP_125_res(6) <= bypass_en_vec_13(6);
	TEMP_125_res(7) <= bypass_en_vec_13(7);
	-- Layer End
	TEMP_126_res(0) <= TEMP_125_res(0) or TEMP_125_res(4);
	TEMP_126_res(1) <= TEMP_125_res(1) or TEMP_125_res(5);
	TEMP_126_res(2) <= TEMP_125_res(2) or TEMP_125_res(6);
	TEMP_126_res(3) <= TEMP_125_res(3) or TEMP_125_res(7);
	-- Layer End
	TEMP_127_res(0) <= TEMP_126_res(0) or TEMP_126_res(2);
	TEMP_127_res(1) <= TEMP_126_res(1) or TEMP_126_res(3);
	-- Layer End
	bypass_en_13 <= TEMP_127_res(0) or TEMP_127_res(1);
	-- Reduction End

	rreq_valid_0_o <= load_en_0;
	-- One-Hot To Bits Begin
	-- OHToBits(rreq_id_0, load_idx_oh_0)
	TEMP_128_in_0_0 <= '0';
	TEMP_128_in_0_1 <= load_idx_oh_0(1);
	TEMP_128_in_0_2 <= '0';
	TEMP_128_in_0_3 <= load_idx_oh_0(3);
	TEMP_128_in_0_4 <= '0';
	TEMP_128_in_0_5 <= load_idx_oh_0(5);
	TEMP_128_in_0_6 <= '0';
	TEMP_128_in_0_7 <= load_idx_oh_0(7);
	TEMP_128_in_0_8 <= '0';
	TEMP_128_in_0_9 <= load_idx_oh_0(9);
	TEMP_128_in_0_10 <= '0';
	TEMP_128_in_0_11 <= load_idx_oh_0(11);
	TEMP_128_in_0_12 <= '0';
	TEMP_128_in_0_13 <= load_idx_oh_0(13);
	TEMP_129_res_0 <= TEMP_128_in_0_0 or TEMP_128_in_0_8;
	TEMP_129_res_1 <= TEMP_128_in_0_1 or TEMP_128_in_0_9;
	TEMP_129_res_2 <= TEMP_128_in_0_2 or TEMP_128_in_0_10;
	TEMP_129_res_3 <= TEMP_128_in_0_3 or TEMP_128_in_0_11;
	TEMP_129_res_4 <= TEMP_128_in_0_4 or TEMP_128_in_0_12;
	TEMP_129_res_5 <= TEMP_128_in_0_5 or TEMP_128_in_0_13;
	TEMP_129_res_6 <= TEMP_128_in_0_6;
	TEMP_129_res_7 <= TEMP_128_in_0_7;
	-- Layer End
	TEMP_130_res_0 <= TEMP_129_res_0 or TEMP_129_res_4;
	TEMP_130_res_1 <= TEMP_129_res_1 or TEMP_129_res_5;
	TEMP_130_res_2 <= TEMP_129_res_2 or TEMP_129_res_6;
	TEMP_130_res_3 <= TEMP_129_res_3 or TEMP_129_res_7;
	-- Layer End
	TEMP_131_res_0 <= TEMP_130_res_0 or TEMP_130_res_2;
	TEMP_131_res_1 <= TEMP_130_res_1 or TEMP_130_res_3;
	-- Layer End
	TEMP_128_out_0 <= TEMP_131_res_0 or TEMP_131_res_1;
	rreq_id_0_o(0) <= TEMP_128_out_0;
	TEMP_131_in_1_0 <= '0';
	TEMP_131_in_1_1 <= '0';
	TEMP_131_in_1_2 <= load_idx_oh_0(2);
	TEMP_131_in_1_3 <= load_idx_oh_0(3);
	TEMP_131_in_1_4 <= '0';
	TEMP_131_in_1_5 <= '0';
	TEMP_131_in_1_6 <= load_idx_oh_0(6);
	TEMP_131_in_1_7 <= load_idx_oh_0(7);
	TEMP_131_in_1_8 <= '0';
	TEMP_131_in_1_9 <= '0';
	TEMP_131_in_1_10 <= load_idx_oh_0(10);
	TEMP_131_in_1_11 <= load_idx_oh_0(11);
	TEMP_131_in_1_12 <= '0';
	TEMP_131_in_1_13 <= '0';
	TEMP_132_res_0 <= TEMP_131_in_1_0 or TEMP_131_in_1_8;
	TEMP_132_res_1 <= TEMP_131_in_1_1 or TEMP_131_in_1_9;
	TEMP_132_res_2 <= TEMP_131_in_1_2 or TEMP_131_in_1_10;
	TEMP_132_res_3 <= TEMP_131_in_1_3 or TEMP_131_in_1_11;
	TEMP_132_res_4 <= TEMP_131_in_1_4 or TEMP_131_in_1_12;
	TEMP_132_res_5 <= TEMP_131_in_1_5 or TEMP_131_in_1_13;
	TEMP_132_res_6 <= TEMP_131_in_1_6;
	TEMP_132_res_7 <= TEMP_131_in_1_7;
	-- Layer End
	TEMP_133_res_0 <= TEMP_132_res_0 or TEMP_132_res_4;
	TEMP_133_res_1 <= TEMP_132_res_1 or TEMP_132_res_5;
	TEMP_133_res_2 <= TEMP_132_res_2 or TEMP_132_res_6;
	TEMP_133_res_3 <= TEMP_132_res_3 or TEMP_132_res_7;
	-- Layer End
	TEMP_134_res_0 <= TEMP_133_res_0 or TEMP_133_res_2;
	TEMP_134_res_1 <= TEMP_133_res_1 or TEMP_133_res_3;
	-- Layer End
	TEMP_131_out_1 <= TEMP_134_res_0 or TEMP_134_res_1;
	rreq_id_0_o(1) <= TEMP_131_out_1;
	TEMP_134_in_2_0 <= '0';
	TEMP_134_in_2_1 <= '0';
	TEMP_134_in_2_2 <= '0';
	TEMP_134_in_2_3 <= '0';
	TEMP_134_in_2_4 <= load_idx_oh_0(4);
	TEMP_134_in_2_5 <= load_idx_oh_0(5);
	TEMP_134_in_2_6 <= load_idx_oh_0(6);
	TEMP_134_in_2_7 <= load_idx_oh_0(7);
	TEMP_134_in_2_8 <= '0';
	TEMP_134_in_2_9 <= '0';
	TEMP_134_in_2_10 <= '0';
	TEMP_134_in_2_11 <= '0';
	TEMP_134_in_2_12 <= load_idx_oh_0(12);
	TEMP_134_in_2_13 <= load_idx_oh_0(13);
	TEMP_135_res_0 <= TEMP_134_in_2_0 or TEMP_134_in_2_8;
	TEMP_135_res_1 <= TEMP_134_in_2_1 or TEMP_134_in_2_9;
	TEMP_135_res_2 <= TEMP_134_in_2_2 or TEMP_134_in_2_10;
	TEMP_135_res_3 <= TEMP_134_in_2_3 or TEMP_134_in_2_11;
	TEMP_135_res_4 <= TEMP_134_in_2_4 or TEMP_134_in_2_12;
	TEMP_135_res_5 <= TEMP_134_in_2_5 or TEMP_134_in_2_13;
	TEMP_135_res_6 <= TEMP_134_in_2_6;
	TEMP_135_res_7 <= TEMP_134_in_2_7;
	-- Layer End
	TEMP_136_res_0 <= TEMP_135_res_0 or TEMP_135_res_4;
	TEMP_136_res_1 <= TEMP_135_res_1 or TEMP_135_res_5;
	TEMP_136_res_2 <= TEMP_135_res_2 or TEMP_135_res_6;
	TEMP_136_res_3 <= TEMP_135_res_3 or TEMP_135_res_7;
	-- Layer End
	TEMP_137_res_0 <= TEMP_136_res_0 or TEMP_136_res_2;
	TEMP_137_res_1 <= TEMP_136_res_1 or TEMP_136_res_3;
	-- Layer End
	TEMP_134_out_2 <= TEMP_137_res_0 or TEMP_137_res_1;
	rreq_id_0_o(2) <= TEMP_134_out_2;
	TEMP_137_in_3_0 <= '0';
	TEMP_137_in_3_1 <= '0';
	TEMP_137_in_3_2 <= '0';
	TEMP_137_in_3_3 <= '0';
	TEMP_137_in_3_4 <= '0';
	TEMP_137_in_3_5 <= '0';
	TEMP_137_in_3_6 <= '0';
	TEMP_137_in_3_7 <= '0';
	TEMP_137_in_3_8 <= load_idx_oh_0(8);
	TEMP_137_in_3_9 <= load_idx_oh_0(9);
	TEMP_137_in_3_10 <= load_idx_oh_0(10);
	TEMP_137_in_3_11 <= load_idx_oh_0(11);
	TEMP_137_in_3_12 <= load_idx_oh_0(12);
	TEMP_137_in_3_13 <= load_idx_oh_0(13);
	TEMP_138_res_0 <= TEMP_137_in_3_0 or TEMP_137_in_3_8;
	TEMP_138_res_1 <= TEMP_137_in_3_1 or TEMP_137_in_3_9;
	TEMP_138_res_2 <= TEMP_137_in_3_2 or TEMP_137_in_3_10;
	TEMP_138_res_3 <= TEMP_137_in_3_3 or TEMP_137_in_3_11;
	TEMP_138_res_4 <= TEMP_137_in_3_4 or TEMP_137_in_3_12;
	TEMP_138_res_5 <= TEMP_137_in_3_5 or TEMP_137_in_3_13;
	TEMP_138_res_6 <= TEMP_137_in_3_6;
	TEMP_138_res_7 <= TEMP_137_in_3_7;
	-- Layer End
	TEMP_139_res_0 <= TEMP_138_res_0 or TEMP_138_res_4;
	TEMP_139_res_1 <= TEMP_138_res_1 or TEMP_138_res_5;
	TEMP_139_res_2 <= TEMP_138_res_2 or TEMP_138_res_6;
	TEMP_139_res_3 <= TEMP_138_res_3 or TEMP_138_res_7;
	-- Layer End
	TEMP_140_res_0 <= TEMP_139_res_0 or TEMP_139_res_2;
	TEMP_140_res_1 <= TEMP_139_res_1 or TEMP_139_res_3;
	-- Layer End
	TEMP_137_out_3 <= TEMP_140_res_0 or TEMP_140_res_1;
	rreq_id_0_o(3) <= TEMP_137_out_3;
	-- One-Hot To Bits End

	-- Mux1H Begin
	-- Mux1H(rreq_addr_0, ldq_addr, load_idx_oh_0)
	TEMP_141_mux_0 <= ldq_addr_0_q when load_idx_oh_0(0) = '1' else "0000000000";
	TEMP_141_mux_1 <= ldq_addr_1_q when load_idx_oh_0(1) = '1' else "0000000000";
	TEMP_141_mux_2 <= ldq_addr_2_q when load_idx_oh_0(2) = '1' else "0000000000";
	TEMP_141_mux_3 <= ldq_addr_3_q when load_idx_oh_0(3) = '1' else "0000000000";
	TEMP_141_mux_4 <= ldq_addr_4_q when load_idx_oh_0(4) = '1' else "0000000000";
	TEMP_141_mux_5 <= ldq_addr_5_q when load_idx_oh_0(5) = '1' else "0000000000";
	TEMP_141_mux_6 <= ldq_addr_6_q when load_idx_oh_0(6) = '1' else "0000000000";
	TEMP_141_mux_7 <= ldq_addr_7_q when load_idx_oh_0(7) = '1' else "0000000000";
	TEMP_141_mux_8 <= ldq_addr_8_q when load_idx_oh_0(8) = '1' else "0000000000";
	TEMP_141_mux_9 <= ldq_addr_9_q when load_idx_oh_0(9) = '1' else "0000000000";
	TEMP_141_mux_10 <= ldq_addr_10_q when load_idx_oh_0(10) = '1' else "0000000000";
	TEMP_141_mux_11 <= ldq_addr_11_q when load_idx_oh_0(11) = '1' else "0000000000";
	TEMP_141_mux_12 <= ldq_addr_12_q when load_idx_oh_0(12) = '1' else "0000000000";
	TEMP_141_mux_13 <= ldq_addr_13_q when load_idx_oh_0(13) = '1' else "0000000000";
	TEMP_142_res_0 <= TEMP_141_mux_0 or TEMP_141_mux_8;
	TEMP_142_res_1 <= TEMP_141_mux_1 or TEMP_141_mux_9;
	TEMP_142_res_2 <= TEMP_141_mux_2 or TEMP_141_mux_10;
	TEMP_142_res_3 <= TEMP_141_mux_3 or TEMP_141_mux_11;
	TEMP_142_res_4 <= TEMP_141_mux_4 or TEMP_141_mux_12;
	TEMP_142_res_5 <= TEMP_141_mux_5 or TEMP_141_mux_13;
	TEMP_142_res_6 <= TEMP_141_mux_6;
	TEMP_142_res_7 <= TEMP_141_mux_7;
	-- Layer End
	TEMP_143_res_0 <= TEMP_142_res_0 or TEMP_142_res_4;
	TEMP_143_res_1 <= TEMP_142_res_1 or TEMP_142_res_5;
	TEMP_143_res_2 <= TEMP_142_res_2 or TEMP_142_res_6;
	TEMP_143_res_3 <= TEMP_142_res_3 or TEMP_142_res_7;
	-- Layer End
	TEMP_144_res_0 <= TEMP_143_res_0 or TEMP_143_res_2;
	TEMP_144_res_1 <= TEMP_143_res_1 or TEMP_143_res_3;
	-- Layer End
	rreq_addr_0_o <= TEMP_144_res_0 or TEMP_144_res_1;
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

	ldq_issue_set_vec_4(0) <= ( load_idx_oh_0(4) and rreq_ready_0_i and load_en_0 ) or bypass_en_4;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_4, ldq_issue_set_vec_4, or)
	ldq_issue_set_4 <= ldq_issue_set_vec_4(0);
	-- Reduction End

	ldq_issue_set_vec_5(0) <= ( load_idx_oh_0(5) and rreq_ready_0_i and load_en_0 ) or bypass_en_5;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_5, ldq_issue_set_vec_5, or)
	ldq_issue_set_5 <= ldq_issue_set_vec_5(0);
	-- Reduction End

	ldq_issue_set_vec_6(0) <= ( load_idx_oh_0(6) and rreq_ready_0_i and load_en_0 ) or bypass_en_6;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_6, ldq_issue_set_vec_6, or)
	ldq_issue_set_6 <= ldq_issue_set_vec_6(0);
	-- Reduction End

	ldq_issue_set_vec_7(0) <= ( load_idx_oh_0(7) and rreq_ready_0_i and load_en_0 ) or bypass_en_7;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_7, ldq_issue_set_vec_7, or)
	ldq_issue_set_7 <= ldq_issue_set_vec_7(0);
	-- Reduction End

	ldq_issue_set_vec_8(0) <= ( load_idx_oh_0(8) and rreq_ready_0_i and load_en_0 ) or bypass_en_8;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_8, ldq_issue_set_vec_8, or)
	ldq_issue_set_8 <= ldq_issue_set_vec_8(0);
	-- Reduction End

	ldq_issue_set_vec_9(0) <= ( load_idx_oh_0(9) and rreq_ready_0_i and load_en_0 ) or bypass_en_9;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_9, ldq_issue_set_vec_9, or)
	ldq_issue_set_9 <= ldq_issue_set_vec_9(0);
	-- Reduction End

	ldq_issue_set_vec_10(0) <= ( load_idx_oh_0(10) and rreq_ready_0_i and load_en_0 ) or bypass_en_10;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_10, ldq_issue_set_vec_10, or)
	ldq_issue_set_10 <= ldq_issue_set_vec_10(0);
	-- Reduction End

	ldq_issue_set_vec_11(0) <= ( load_idx_oh_0(11) and rreq_ready_0_i and load_en_0 ) or bypass_en_11;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_11, ldq_issue_set_vec_11, or)
	ldq_issue_set_11 <= ldq_issue_set_vec_11(0);
	-- Reduction End

	ldq_issue_set_vec_12(0) <= ( load_idx_oh_0(12) and rreq_ready_0_i and load_en_0 ) or bypass_en_12;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_12, ldq_issue_set_vec_12, or)
	ldq_issue_set_12 <= ldq_issue_set_vec_12(0);
	-- Reduction End

	ldq_issue_set_vec_13(0) <= ( load_idx_oh_0(13) and rreq_ready_0_i and load_en_0 ) or bypass_en_13;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_13, ldq_issue_set_vec_13, or)
	ldq_issue_set_13 <= ldq_issue_set_vec_13(0);
	-- Reduction End

	wreq_valid_0_o <= store_en;
	wreq_id_0_o <= "0000";
	-- MuxLookUp Begin
	-- MuxLookUp(wreq_addr_0, stq_addr, store_idx)
	wreq_addr_0_o <= 
	stq_addr_0_q when (store_idx = "0000") else
	stq_addr_1_q when (store_idx = "0001") else
	stq_addr_2_q when (store_idx = "0010") else
	stq_addr_3_q when (store_idx = "0011") else
	stq_addr_4_q when (store_idx = "0100") else
	stq_addr_5_q when (store_idx = "0101") else
	stq_addr_6_q when (store_idx = "0110") else
	stq_addr_7_q when (store_idx = "0111") else
	stq_addr_8_q when (store_idx = "1000") else
	stq_addr_9_q when (store_idx = "1001") else
	stq_addr_10_q when (store_idx = "1010") else
	stq_addr_11_q when (store_idx = "1011") else
	stq_addr_12_q when (store_idx = "1100") else
	stq_addr_13_q when (store_idx = "1101") else
	"0000000000";
	-- MuxLookUp End

	-- MuxLookUp Begin
	-- MuxLookUp(wreq_data_0, stq_data, store_idx)
	wreq_data_0_o <= 
	stq_data_0_q when (store_idx = "0000") else
	stq_data_1_q when (store_idx = "0001") else
	stq_data_2_q when (store_idx = "0010") else
	stq_data_3_q when (store_idx = "0011") else
	stq_data_4_q when (store_idx = "0100") else
	stq_data_5_q when (store_idx = "0101") else
	stq_data_6_q when (store_idx = "0110") else
	stq_data_7_q when (store_idx = "0111") else
	stq_data_8_q when (store_idx = "1000") else
	stq_data_9_q when (store_idx = "1001") else
	stq_data_10_q when (store_idx = "1010") else
	stq_data_11_q when (store_idx = "1011") else
	stq_data_12_q when (store_idx = "1100") else
	stq_data_13_q when (store_idx = "1101") else
	"00000000000000000000000000000000";
	-- MuxLookUp End

	stq_issue_en <= store_en and wreq_ready_0_i;
	read_idx_oh_0_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0000" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_0, rresp_data, read_idx_oh_0)
	TEMP_145_mux_0 <= rresp_data_0_i when read_idx_oh_0_0 = '1' else "00000000000000000000000000000000";
	read_data_0 <= TEMP_145_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_0, read_idx_oh_0, or)
	read_valid_0 <= read_idx_oh_0_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_0, stq_data, bypass_idx_oh_0)
	TEMP_146_mux_0 <= stq_data_0_q when bypass_idx_oh_0(0) = '1' else "00000000000000000000000000000000";
	TEMP_146_mux_1 <= stq_data_1_q when bypass_idx_oh_0(1) = '1' else "00000000000000000000000000000000";
	TEMP_146_mux_2 <= stq_data_2_q when bypass_idx_oh_0(2) = '1' else "00000000000000000000000000000000";
	TEMP_146_mux_3 <= stq_data_3_q when bypass_idx_oh_0(3) = '1' else "00000000000000000000000000000000";
	TEMP_146_mux_4 <= stq_data_4_q when bypass_idx_oh_0(4) = '1' else "00000000000000000000000000000000";
	TEMP_146_mux_5 <= stq_data_5_q when bypass_idx_oh_0(5) = '1' else "00000000000000000000000000000000";
	TEMP_146_mux_6 <= stq_data_6_q when bypass_idx_oh_0(6) = '1' else "00000000000000000000000000000000";
	TEMP_146_mux_7 <= stq_data_7_q when bypass_idx_oh_0(7) = '1' else "00000000000000000000000000000000";
	TEMP_146_mux_8 <= stq_data_8_q when bypass_idx_oh_0(8) = '1' else "00000000000000000000000000000000";
	TEMP_146_mux_9 <= stq_data_9_q when bypass_idx_oh_0(9) = '1' else "00000000000000000000000000000000";
	TEMP_146_mux_10 <= stq_data_10_q when bypass_idx_oh_0(10) = '1' else "00000000000000000000000000000000";
	TEMP_146_mux_11 <= stq_data_11_q when bypass_idx_oh_0(11) = '1' else "00000000000000000000000000000000";
	TEMP_146_mux_12 <= stq_data_12_q when bypass_idx_oh_0(12) = '1' else "00000000000000000000000000000000";
	TEMP_146_mux_13 <= stq_data_13_q when bypass_idx_oh_0(13) = '1' else "00000000000000000000000000000000";
	TEMP_147_res_0 <= TEMP_146_mux_0 or TEMP_146_mux_8;
	TEMP_147_res_1 <= TEMP_146_mux_1 or TEMP_146_mux_9;
	TEMP_147_res_2 <= TEMP_146_mux_2 or TEMP_146_mux_10;
	TEMP_147_res_3 <= TEMP_146_mux_3 or TEMP_146_mux_11;
	TEMP_147_res_4 <= TEMP_146_mux_4 or TEMP_146_mux_12;
	TEMP_147_res_5 <= TEMP_146_mux_5 or TEMP_146_mux_13;
	TEMP_147_res_6 <= TEMP_146_mux_6;
	TEMP_147_res_7 <= TEMP_146_mux_7;
	-- Layer End
	TEMP_148_res_0 <= TEMP_147_res_0 or TEMP_147_res_4;
	TEMP_148_res_1 <= TEMP_147_res_1 or TEMP_147_res_5;
	TEMP_148_res_2 <= TEMP_147_res_2 or TEMP_147_res_6;
	TEMP_148_res_3 <= TEMP_147_res_3 or TEMP_147_res_7;
	-- Layer End
	TEMP_149_res_0 <= TEMP_148_res_0 or TEMP_148_res_2;
	TEMP_149_res_1 <= TEMP_148_res_1 or TEMP_148_res_3;
	-- Layer End
	bypass_data_0 <= TEMP_149_res_0 or TEMP_149_res_1;
	-- Mux1H End

	ldq_data_0_d <= read_data_0 or bypass_data_0;
	ldq_data_wen_0 <= bypass_en_0 or read_valid_0;
	read_idx_oh_1_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0001" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_1, rresp_data, read_idx_oh_1)
	TEMP_150_mux_0 <= rresp_data_0_i when read_idx_oh_1_0 = '1' else "00000000000000000000000000000000";
	read_data_1 <= TEMP_150_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_1, read_idx_oh_1, or)
	read_valid_1 <= read_idx_oh_1_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_1, stq_data, bypass_idx_oh_1)
	TEMP_151_mux_0 <= stq_data_0_q when bypass_idx_oh_1(0) = '1' else "00000000000000000000000000000000";
	TEMP_151_mux_1 <= stq_data_1_q when bypass_idx_oh_1(1) = '1' else "00000000000000000000000000000000";
	TEMP_151_mux_2 <= stq_data_2_q when bypass_idx_oh_1(2) = '1' else "00000000000000000000000000000000";
	TEMP_151_mux_3 <= stq_data_3_q when bypass_idx_oh_1(3) = '1' else "00000000000000000000000000000000";
	TEMP_151_mux_4 <= stq_data_4_q when bypass_idx_oh_1(4) = '1' else "00000000000000000000000000000000";
	TEMP_151_mux_5 <= stq_data_5_q when bypass_idx_oh_1(5) = '1' else "00000000000000000000000000000000";
	TEMP_151_mux_6 <= stq_data_6_q when bypass_idx_oh_1(6) = '1' else "00000000000000000000000000000000";
	TEMP_151_mux_7 <= stq_data_7_q when bypass_idx_oh_1(7) = '1' else "00000000000000000000000000000000";
	TEMP_151_mux_8 <= stq_data_8_q when bypass_idx_oh_1(8) = '1' else "00000000000000000000000000000000";
	TEMP_151_mux_9 <= stq_data_9_q when bypass_idx_oh_1(9) = '1' else "00000000000000000000000000000000";
	TEMP_151_mux_10 <= stq_data_10_q when bypass_idx_oh_1(10) = '1' else "00000000000000000000000000000000";
	TEMP_151_mux_11 <= stq_data_11_q when bypass_idx_oh_1(11) = '1' else "00000000000000000000000000000000";
	TEMP_151_mux_12 <= stq_data_12_q when bypass_idx_oh_1(12) = '1' else "00000000000000000000000000000000";
	TEMP_151_mux_13 <= stq_data_13_q when bypass_idx_oh_1(13) = '1' else "00000000000000000000000000000000";
	TEMP_152_res_0 <= TEMP_151_mux_0 or TEMP_151_mux_8;
	TEMP_152_res_1 <= TEMP_151_mux_1 or TEMP_151_mux_9;
	TEMP_152_res_2 <= TEMP_151_mux_2 or TEMP_151_mux_10;
	TEMP_152_res_3 <= TEMP_151_mux_3 or TEMP_151_mux_11;
	TEMP_152_res_4 <= TEMP_151_mux_4 or TEMP_151_mux_12;
	TEMP_152_res_5 <= TEMP_151_mux_5 or TEMP_151_mux_13;
	TEMP_152_res_6 <= TEMP_151_mux_6;
	TEMP_152_res_7 <= TEMP_151_mux_7;
	-- Layer End
	TEMP_153_res_0 <= TEMP_152_res_0 or TEMP_152_res_4;
	TEMP_153_res_1 <= TEMP_152_res_1 or TEMP_152_res_5;
	TEMP_153_res_2 <= TEMP_152_res_2 or TEMP_152_res_6;
	TEMP_153_res_3 <= TEMP_152_res_3 or TEMP_152_res_7;
	-- Layer End
	TEMP_154_res_0 <= TEMP_153_res_0 or TEMP_153_res_2;
	TEMP_154_res_1 <= TEMP_153_res_1 or TEMP_153_res_3;
	-- Layer End
	bypass_data_1 <= TEMP_154_res_0 or TEMP_154_res_1;
	-- Mux1H End

	ldq_data_1_d <= read_data_1 or bypass_data_1;
	ldq_data_wen_1 <= bypass_en_1 or read_valid_1;
	read_idx_oh_2_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0010" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_2, rresp_data, read_idx_oh_2)
	TEMP_155_mux_0 <= rresp_data_0_i when read_idx_oh_2_0 = '1' else "00000000000000000000000000000000";
	read_data_2 <= TEMP_155_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_2, read_idx_oh_2, or)
	read_valid_2 <= read_idx_oh_2_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_2, stq_data, bypass_idx_oh_2)
	TEMP_156_mux_0 <= stq_data_0_q when bypass_idx_oh_2(0) = '1' else "00000000000000000000000000000000";
	TEMP_156_mux_1 <= stq_data_1_q when bypass_idx_oh_2(1) = '1' else "00000000000000000000000000000000";
	TEMP_156_mux_2 <= stq_data_2_q when bypass_idx_oh_2(2) = '1' else "00000000000000000000000000000000";
	TEMP_156_mux_3 <= stq_data_3_q when bypass_idx_oh_2(3) = '1' else "00000000000000000000000000000000";
	TEMP_156_mux_4 <= stq_data_4_q when bypass_idx_oh_2(4) = '1' else "00000000000000000000000000000000";
	TEMP_156_mux_5 <= stq_data_5_q when bypass_idx_oh_2(5) = '1' else "00000000000000000000000000000000";
	TEMP_156_mux_6 <= stq_data_6_q when bypass_idx_oh_2(6) = '1' else "00000000000000000000000000000000";
	TEMP_156_mux_7 <= stq_data_7_q when bypass_idx_oh_2(7) = '1' else "00000000000000000000000000000000";
	TEMP_156_mux_8 <= stq_data_8_q when bypass_idx_oh_2(8) = '1' else "00000000000000000000000000000000";
	TEMP_156_mux_9 <= stq_data_9_q when bypass_idx_oh_2(9) = '1' else "00000000000000000000000000000000";
	TEMP_156_mux_10 <= stq_data_10_q when bypass_idx_oh_2(10) = '1' else "00000000000000000000000000000000";
	TEMP_156_mux_11 <= stq_data_11_q when bypass_idx_oh_2(11) = '1' else "00000000000000000000000000000000";
	TEMP_156_mux_12 <= stq_data_12_q when bypass_idx_oh_2(12) = '1' else "00000000000000000000000000000000";
	TEMP_156_mux_13 <= stq_data_13_q when bypass_idx_oh_2(13) = '1' else "00000000000000000000000000000000";
	TEMP_157_res_0 <= TEMP_156_mux_0 or TEMP_156_mux_8;
	TEMP_157_res_1 <= TEMP_156_mux_1 or TEMP_156_mux_9;
	TEMP_157_res_2 <= TEMP_156_mux_2 or TEMP_156_mux_10;
	TEMP_157_res_3 <= TEMP_156_mux_3 or TEMP_156_mux_11;
	TEMP_157_res_4 <= TEMP_156_mux_4 or TEMP_156_mux_12;
	TEMP_157_res_5 <= TEMP_156_mux_5 or TEMP_156_mux_13;
	TEMP_157_res_6 <= TEMP_156_mux_6;
	TEMP_157_res_7 <= TEMP_156_mux_7;
	-- Layer End
	TEMP_158_res_0 <= TEMP_157_res_0 or TEMP_157_res_4;
	TEMP_158_res_1 <= TEMP_157_res_1 or TEMP_157_res_5;
	TEMP_158_res_2 <= TEMP_157_res_2 or TEMP_157_res_6;
	TEMP_158_res_3 <= TEMP_157_res_3 or TEMP_157_res_7;
	-- Layer End
	TEMP_159_res_0 <= TEMP_158_res_0 or TEMP_158_res_2;
	TEMP_159_res_1 <= TEMP_158_res_1 or TEMP_158_res_3;
	-- Layer End
	bypass_data_2 <= TEMP_159_res_0 or TEMP_159_res_1;
	-- Mux1H End

	ldq_data_2_d <= read_data_2 or bypass_data_2;
	ldq_data_wen_2 <= bypass_en_2 or read_valid_2;
	read_idx_oh_3_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0011" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_3, rresp_data, read_idx_oh_3)
	TEMP_160_mux_0 <= rresp_data_0_i when read_idx_oh_3_0 = '1' else "00000000000000000000000000000000";
	read_data_3 <= TEMP_160_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_3, read_idx_oh_3, or)
	read_valid_3 <= read_idx_oh_3_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_3, stq_data, bypass_idx_oh_3)
	TEMP_161_mux_0 <= stq_data_0_q when bypass_idx_oh_3(0) = '1' else "00000000000000000000000000000000";
	TEMP_161_mux_1 <= stq_data_1_q when bypass_idx_oh_3(1) = '1' else "00000000000000000000000000000000";
	TEMP_161_mux_2 <= stq_data_2_q when bypass_idx_oh_3(2) = '1' else "00000000000000000000000000000000";
	TEMP_161_mux_3 <= stq_data_3_q when bypass_idx_oh_3(3) = '1' else "00000000000000000000000000000000";
	TEMP_161_mux_4 <= stq_data_4_q when bypass_idx_oh_3(4) = '1' else "00000000000000000000000000000000";
	TEMP_161_mux_5 <= stq_data_5_q when bypass_idx_oh_3(5) = '1' else "00000000000000000000000000000000";
	TEMP_161_mux_6 <= stq_data_6_q when bypass_idx_oh_3(6) = '1' else "00000000000000000000000000000000";
	TEMP_161_mux_7 <= stq_data_7_q when bypass_idx_oh_3(7) = '1' else "00000000000000000000000000000000";
	TEMP_161_mux_8 <= stq_data_8_q when bypass_idx_oh_3(8) = '1' else "00000000000000000000000000000000";
	TEMP_161_mux_9 <= stq_data_9_q when bypass_idx_oh_3(9) = '1' else "00000000000000000000000000000000";
	TEMP_161_mux_10 <= stq_data_10_q when bypass_idx_oh_3(10) = '1' else "00000000000000000000000000000000";
	TEMP_161_mux_11 <= stq_data_11_q when bypass_idx_oh_3(11) = '1' else "00000000000000000000000000000000";
	TEMP_161_mux_12 <= stq_data_12_q when bypass_idx_oh_3(12) = '1' else "00000000000000000000000000000000";
	TEMP_161_mux_13 <= stq_data_13_q when bypass_idx_oh_3(13) = '1' else "00000000000000000000000000000000";
	TEMP_162_res_0 <= TEMP_161_mux_0 or TEMP_161_mux_8;
	TEMP_162_res_1 <= TEMP_161_mux_1 or TEMP_161_mux_9;
	TEMP_162_res_2 <= TEMP_161_mux_2 or TEMP_161_mux_10;
	TEMP_162_res_3 <= TEMP_161_mux_3 or TEMP_161_mux_11;
	TEMP_162_res_4 <= TEMP_161_mux_4 or TEMP_161_mux_12;
	TEMP_162_res_5 <= TEMP_161_mux_5 or TEMP_161_mux_13;
	TEMP_162_res_6 <= TEMP_161_mux_6;
	TEMP_162_res_7 <= TEMP_161_mux_7;
	-- Layer End
	TEMP_163_res_0 <= TEMP_162_res_0 or TEMP_162_res_4;
	TEMP_163_res_1 <= TEMP_162_res_1 or TEMP_162_res_5;
	TEMP_163_res_2 <= TEMP_162_res_2 or TEMP_162_res_6;
	TEMP_163_res_3 <= TEMP_162_res_3 or TEMP_162_res_7;
	-- Layer End
	TEMP_164_res_0 <= TEMP_163_res_0 or TEMP_163_res_2;
	TEMP_164_res_1 <= TEMP_163_res_1 or TEMP_163_res_3;
	-- Layer End
	bypass_data_3 <= TEMP_164_res_0 or TEMP_164_res_1;
	-- Mux1H End

	ldq_data_3_d <= read_data_3 or bypass_data_3;
	ldq_data_wen_3 <= bypass_en_3 or read_valid_3;
	read_idx_oh_4_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0100" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_4, rresp_data, read_idx_oh_4)
	TEMP_165_mux_0 <= rresp_data_0_i when read_idx_oh_4_0 = '1' else "00000000000000000000000000000000";
	read_data_4 <= TEMP_165_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_4, read_idx_oh_4, or)
	read_valid_4 <= read_idx_oh_4_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_4, stq_data, bypass_idx_oh_4)
	TEMP_166_mux_0 <= stq_data_0_q when bypass_idx_oh_4(0) = '1' else "00000000000000000000000000000000";
	TEMP_166_mux_1 <= stq_data_1_q when bypass_idx_oh_4(1) = '1' else "00000000000000000000000000000000";
	TEMP_166_mux_2 <= stq_data_2_q when bypass_idx_oh_4(2) = '1' else "00000000000000000000000000000000";
	TEMP_166_mux_3 <= stq_data_3_q when bypass_idx_oh_4(3) = '1' else "00000000000000000000000000000000";
	TEMP_166_mux_4 <= stq_data_4_q when bypass_idx_oh_4(4) = '1' else "00000000000000000000000000000000";
	TEMP_166_mux_5 <= stq_data_5_q when bypass_idx_oh_4(5) = '1' else "00000000000000000000000000000000";
	TEMP_166_mux_6 <= stq_data_6_q when bypass_idx_oh_4(6) = '1' else "00000000000000000000000000000000";
	TEMP_166_mux_7 <= stq_data_7_q when bypass_idx_oh_4(7) = '1' else "00000000000000000000000000000000";
	TEMP_166_mux_8 <= stq_data_8_q when bypass_idx_oh_4(8) = '1' else "00000000000000000000000000000000";
	TEMP_166_mux_9 <= stq_data_9_q when bypass_idx_oh_4(9) = '1' else "00000000000000000000000000000000";
	TEMP_166_mux_10 <= stq_data_10_q when bypass_idx_oh_4(10) = '1' else "00000000000000000000000000000000";
	TEMP_166_mux_11 <= stq_data_11_q when bypass_idx_oh_4(11) = '1' else "00000000000000000000000000000000";
	TEMP_166_mux_12 <= stq_data_12_q when bypass_idx_oh_4(12) = '1' else "00000000000000000000000000000000";
	TEMP_166_mux_13 <= stq_data_13_q when bypass_idx_oh_4(13) = '1' else "00000000000000000000000000000000";
	TEMP_167_res_0 <= TEMP_166_mux_0 or TEMP_166_mux_8;
	TEMP_167_res_1 <= TEMP_166_mux_1 or TEMP_166_mux_9;
	TEMP_167_res_2 <= TEMP_166_mux_2 or TEMP_166_mux_10;
	TEMP_167_res_3 <= TEMP_166_mux_3 or TEMP_166_mux_11;
	TEMP_167_res_4 <= TEMP_166_mux_4 or TEMP_166_mux_12;
	TEMP_167_res_5 <= TEMP_166_mux_5 or TEMP_166_mux_13;
	TEMP_167_res_6 <= TEMP_166_mux_6;
	TEMP_167_res_7 <= TEMP_166_mux_7;
	-- Layer End
	TEMP_168_res_0 <= TEMP_167_res_0 or TEMP_167_res_4;
	TEMP_168_res_1 <= TEMP_167_res_1 or TEMP_167_res_5;
	TEMP_168_res_2 <= TEMP_167_res_2 or TEMP_167_res_6;
	TEMP_168_res_3 <= TEMP_167_res_3 or TEMP_167_res_7;
	-- Layer End
	TEMP_169_res_0 <= TEMP_168_res_0 or TEMP_168_res_2;
	TEMP_169_res_1 <= TEMP_168_res_1 or TEMP_168_res_3;
	-- Layer End
	bypass_data_4 <= TEMP_169_res_0 or TEMP_169_res_1;
	-- Mux1H End

	ldq_data_4_d <= read_data_4 or bypass_data_4;
	ldq_data_wen_4 <= bypass_en_4 or read_valid_4;
	read_idx_oh_5_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0101" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_5, rresp_data, read_idx_oh_5)
	TEMP_170_mux_0 <= rresp_data_0_i when read_idx_oh_5_0 = '1' else "00000000000000000000000000000000";
	read_data_5 <= TEMP_170_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_5, read_idx_oh_5, or)
	read_valid_5 <= read_idx_oh_5_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_5, stq_data, bypass_idx_oh_5)
	TEMP_171_mux_0 <= stq_data_0_q when bypass_idx_oh_5(0) = '1' else "00000000000000000000000000000000";
	TEMP_171_mux_1 <= stq_data_1_q when bypass_idx_oh_5(1) = '1' else "00000000000000000000000000000000";
	TEMP_171_mux_2 <= stq_data_2_q when bypass_idx_oh_5(2) = '1' else "00000000000000000000000000000000";
	TEMP_171_mux_3 <= stq_data_3_q when bypass_idx_oh_5(3) = '1' else "00000000000000000000000000000000";
	TEMP_171_mux_4 <= stq_data_4_q when bypass_idx_oh_5(4) = '1' else "00000000000000000000000000000000";
	TEMP_171_mux_5 <= stq_data_5_q when bypass_idx_oh_5(5) = '1' else "00000000000000000000000000000000";
	TEMP_171_mux_6 <= stq_data_6_q when bypass_idx_oh_5(6) = '1' else "00000000000000000000000000000000";
	TEMP_171_mux_7 <= stq_data_7_q when bypass_idx_oh_5(7) = '1' else "00000000000000000000000000000000";
	TEMP_171_mux_8 <= stq_data_8_q when bypass_idx_oh_5(8) = '1' else "00000000000000000000000000000000";
	TEMP_171_mux_9 <= stq_data_9_q when bypass_idx_oh_5(9) = '1' else "00000000000000000000000000000000";
	TEMP_171_mux_10 <= stq_data_10_q when bypass_idx_oh_5(10) = '1' else "00000000000000000000000000000000";
	TEMP_171_mux_11 <= stq_data_11_q when bypass_idx_oh_5(11) = '1' else "00000000000000000000000000000000";
	TEMP_171_mux_12 <= stq_data_12_q when bypass_idx_oh_5(12) = '1' else "00000000000000000000000000000000";
	TEMP_171_mux_13 <= stq_data_13_q when bypass_idx_oh_5(13) = '1' else "00000000000000000000000000000000";
	TEMP_172_res_0 <= TEMP_171_mux_0 or TEMP_171_mux_8;
	TEMP_172_res_1 <= TEMP_171_mux_1 or TEMP_171_mux_9;
	TEMP_172_res_2 <= TEMP_171_mux_2 or TEMP_171_mux_10;
	TEMP_172_res_3 <= TEMP_171_mux_3 or TEMP_171_mux_11;
	TEMP_172_res_4 <= TEMP_171_mux_4 or TEMP_171_mux_12;
	TEMP_172_res_5 <= TEMP_171_mux_5 or TEMP_171_mux_13;
	TEMP_172_res_6 <= TEMP_171_mux_6;
	TEMP_172_res_7 <= TEMP_171_mux_7;
	-- Layer End
	TEMP_173_res_0 <= TEMP_172_res_0 or TEMP_172_res_4;
	TEMP_173_res_1 <= TEMP_172_res_1 or TEMP_172_res_5;
	TEMP_173_res_2 <= TEMP_172_res_2 or TEMP_172_res_6;
	TEMP_173_res_3 <= TEMP_172_res_3 or TEMP_172_res_7;
	-- Layer End
	TEMP_174_res_0 <= TEMP_173_res_0 or TEMP_173_res_2;
	TEMP_174_res_1 <= TEMP_173_res_1 or TEMP_173_res_3;
	-- Layer End
	bypass_data_5 <= TEMP_174_res_0 or TEMP_174_res_1;
	-- Mux1H End

	ldq_data_5_d <= read_data_5 or bypass_data_5;
	ldq_data_wen_5 <= bypass_en_5 or read_valid_5;
	read_idx_oh_6_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0110" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_6, rresp_data, read_idx_oh_6)
	TEMP_175_mux_0 <= rresp_data_0_i when read_idx_oh_6_0 = '1' else "00000000000000000000000000000000";
	read_data_6 <= TEMP_175_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_6, read_idx_oh_6, or)
	read_valid_6 <= read_idx_oh_6_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_6, stq_data, bypass_idx_oh_6)
	TEMP_176_mux_0 <= stq_data_0_q when bypass_idx_oh_6(0) = '1' else "00000000000000000000000000000000";
	TEMP_176_mux_1 <= stq_data_1_q when bypass_idx_oh_6(1) = '1' else "00000000000000000000000000000000";
	TEMP_176_mux_2 <= stq_data_2_q when bypass_idx_oh_6(2) = '1' else "00000000000000000000000000000000";
	TEMP_176_mux_3 <= stq_data_3_q when bypass_idx_oh_6(3) = '1' else "00000000000000000000000000000000";
	TEMP_176_mux_4 <= stq_data_4_q when bypass_idx_oh_6(4) = '1' else "00000000000000000000000000000000";
	TEMP_176_mux_5 <= stq_data_5_q when bypass_idx_oh_6(5) = '1' else "00000000000000000000000000000000";
	TEMP_176_mux_6 <= stq_data_6_q when bypass_idx_oh_6(6) = '1' else "00000000000000000000000000000000";
	TEMP_176_mux_7 <= stq_data_7_q when bypass_idx_oh_6(7) = '1' else "00000000000000000000000000000000";
	TEMP_176_mux_8 <= stq_data_8_q when bypass_idx_oh_6(8) = '1' else "00000000000000000000000000000000";
	TEMP_176_mux_9 <= stq_data_9_q when bypass_idx_oh_6(9) = '1' else "00000000000000000000000000000000";
	TEMP_176_mux_10 <= stq_data_10_q when bypass_idx_oh_6(10) = '1' else "00000000000000000000000000000000";
	TEMP_176_mux_11 <= stq_data_11_q when bypass_idx_oh_6(11) = '1' else "00000000000000000000000000000000";
	TEMP_176_mux_12 <= stq_data_12_q when bypass_idx_oh_6(12) = '1' else "00000000000000000000000000000000";
	TEMP_176_mux_13 <= stq_data_13_q when bypass_idx_oh_6(13) = '1' else "00000000000000000000000000000000";
	TEMP_177_res_0 <= TEMP_176_mux_0 or TEMP_176_mux_8;
	TEMP_177_res_1 <= TEMP_176_mux_1 or TEMP_176_mux_9;
	TEMP_177_res_2 <= TEMP_176_mux_2 or TEMP_176_mux_10;
	TEMP_177_res_3 <= TEMP_176_mux_3 or TEMP_176_mux_11;
	TEMP_177_res_4 <= TEMP_176_mux_4 or TEMP_176_mux_12;
	TEMP_177_res_5 <= TEMP_176_mux_5 or TEMP_176_mux_13;
	TEMP_177_res_6 <= TEMP_176_mux_6;
	TEMP_177_res_7 <= TEMP_176_mux_7;
	-- Layer End
	TEMP_178_res_0 <= TEMP_177_res_0 or TEMP_177_res_4;
	TEMP_178_res_1 <= TEMP_177_res_1 or TEMP_177_res_5;
	TEMP_178_res_2 <= TEMP_177_res_2 or TEMP_177_res_6;
	TEMP_178_res_3 <= TEMP_177_res_3 or TEMP_177_res_7;
	-- Layer End
	TEMP_179_res_0 <= TEMP_178_res_0 or TEMP_178_res_2;
	TEMP_179_res_1 <= TEMP_178_res_1 or TEMP_178_res_3;
	-- Layer End
	bypass_data_6 <= TEMP_179_res_0 or TEMP_179_res_1;
	-- Mux1H End

	ldq_data_6_d <= read_data_6 or bypass_data_6;
	ldq_data_wen_6 <= bypass_en_6 or read_valid_6;
	read_idx_oh_7_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0111" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_7, rresp_data, read_idx_oh_7)
	TEMP_180_mux_0 <= rresp_data_0_i when read_idx_oh_7_0 = '1' else "00000000000000000000000000000000";
	read_data_7 <= TEMP_180_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_7, read_idx_oh_7, or)
	read_valid_7 <= read_idx_oh_7_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_7, stq_data, bypass_idx_oh_7)
	TEMP_181_mux_0 <= stq_data_0_q when bypass_idx_oh_7(0) = '1' else "00000000000000000000000000000000";
	TEMP_181_mux_1 <= stq_data_1_q when bypass_idx_oh_7(1) = '1' else "00000000000000000000000000000000";
	TEMP_181_mux_2 <= stq_data_2_q when bypass_idx_oh_7(2) = '1' else "00000000000000000000000000000000";
	TEMP_181_mux_3 <= stq_data_3_q when bypass_idx_oh_7(3) = '1' else "00000000000000000000000000000000";
	TEMP_181_mux_4 <= stq_data_4_q when bypass_idx_oh_7(4) = '1' else "00000000000000000000000000000000";
	TEMP_181_mux_5 <= stq_data_5_q when bypass_idx_oh_7(5) = '1' else "00000000000000000000000000000000";
	TEMP_181_mux_6 <= stq_data_6_q when bypass_idx_oh_7(6) = '1' else "00000000000000000000000000000000";
	TEMP_181_mux_7 <= stq_data_7_q when bypass_idx_oh_7(7) = '1' else "00000000000000000000000000000000";
	TEMP_181_mux_8 <= stq_data_8_q when bypass_idx_oh_7(8) = '1' else "00000000000000000000000000000000";
	TEMP_181_mux_9 <= stq_data_9_q when bypass_idx_oh_7(9) = '1' else "00000000000000000000000000000000";
	TEMP_181_mux_10 <= stq_data_10_q when bypass_idx_oh_7(10) = '1' else "00000000000000000000000000000000";
	TEMP_181_mux_11 <= stq_data_11_q when bypass_idx_oh_7(11) = '1' else "00000000000000000000000000000000";
	TEMP_181_mux_12 <= stq_data_12_q when bypass_idx_oh_7(12) = '1' else "00000000000000000000000000000000";
	TEMP_181_mux_13 <= stq_data_13_q when bypass_idx_oh_7(13) = '1' else "00000000000000000000000000000000";
	TEMP_182_res_0 <= TEMP_181_mux_0 or TEMP_181_mux_8;
	TEMP_182_res_1 <= TEMP_181_mux_1 or TEMP_181_mux_9;
	TEMP_182_res_2 <= TEMP_181_mux_2 or TEMP_181_mux_10;
	TEMP_182_res_3 <= TEMP_181_mux_3 or TEMP_181_mux_11;
	TEMP_182_res_4 <= TEMP_181_mux_4 or TEMP_181_mux_12;
	TEMP_182_res_5 <= TEMP_181_mux_5 or TEMP_181_mux_13;
	TEMP_182_res_6 <= TEMP_181_mux_6;
	TEMP_182_res_7 <= TEMP_181_mux_7;
	-- Layer End
	TEMP_183_res_0 <= TEMP_182_res_0 or TEMP_182_res_4;
	TEMP_183_res_1 <= TEMP_182_res_1 or TEMP_182_res_5;
	TEMP_183_res_2 <= TEMP_182_res_2 or TEMP_182_res_6;
	TEMP_183_res_3 <= TEMP_182_res_3 or TEMP_182_res_7;
	-- Layer End
	TEMP_184_res_0 <= TEMP_183_res_0 or TEMP_183_res_2;
	TEMP_184_res_1 <= TEMP_183_res_1 or TEMP_183_res_3;
	-- Layer End
	bypass_data_7 <= TEMP_184_res_0 or TEMP_184_res_1;
	-- Mux1H End

	ldq_data_7_d <= read_data_7 or bypass_data_7;
	ldq_data_wen_7 <= bypass_en_7 or read_valid_7;
	read_idx_oh_8_0 <= rresp_valid_0_i when ( rresp_id_0_i = "1000" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_8, rresp_data, read_idx_oh_8)
	TEMP_185_mux_0 <= rresp_data_0_i when read_idx_oh_8_0 = '1' else "00000000000000000000000000000000";
	read_data_8 <= TEMP_185_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_8, read_idx_oh_8, or)
	read_valid_8 <= read_idx_oh_8_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_8, stq_data, bypass_idx_oh_8)
	TEMP_186_mux_0 <= stq_data_0_q when bypass_idx_oh_8(0) = '1' else "00000000000000000000000000000000";
	TEMP_186_mux_1 <= stq_data_1_q when bypass_idx_oh_8(1) = '1' else "00000000000000000000000000000000";
	TEMP_186_mux_2 <= stq_data_2_q when bypass_idx_oh_8(2) = '1' else "00000000000000000000000000000000";
	TEMP_186_mux_3 <= stq_data_3_q when bypass_idx_oh_8(3) = '1' else "00000000000000000000000000000000";
	TEMP_186_mux_4 <= stq_data_4_q when bypass_idx_oh_8(4) = '1' else "00000000000000000000000000000000";
	TEMP_186_mux_5 <= stq_data_5_q when bypass_idx_oh_8(5) = '1' else "00000000000000000000000000000000";
	TEMP_186_mux_6 <= stq_data_6_q when bypass_idx_oh_8(6) = '1' else "00000000000000000000000000000000";
	TEMP_186_mux_7 <= stq_data_7_q when bypass_idx_oh_8(7) = '1' else "00000000000000000000000000000000";
	TEMP_186_mux_8 <= stq_data_8_q when bypass_idx_oh_8(8) = '1' else "00000000000000000000000000000000";
	TEMP_186_mux_9 <= stq_data_9_q when bypass_idx_oh_8(9) = '1' else "00000000000000000000000000000000";
	TEMP_186_mux_10 <= stq_data_10_q when bypass_idx_oh_8(10) = '1' else "00000000000000000000000000000000";
	TEMP_186_mux_11 <= stq_data_11_q when bypass_idx_oh_8(11) = '1' else "00000000000000000000000000000000";
	TEMP_186_mux_12 <= stq_data_12_q when bypass_idx_oh_8(12) = '1' else "00000000000000000000000000000000";
	TEMP_186_mux_13 <= stq_data_13_q when bypass_idx_oh_8(13) = '1' else "00000000000000000000000000000000";
	TEMP_187_res_0 <= TEMP_186_mux_0 or TEMP_186_mux_8;
	TEMP_187_res_1 <= TEMP_186_mux_1 or TEMP_186_mux_9;
	TEMP_187_res_2 <= TEMP_186_mux_2 or TEMP_186_mux_10;
	TEMP_187_res_3 <= TEMP_186_mux_3 or TEMP_186_mux_11;
	TEMP_187_res_4 <= TEMP_186_mux_4 or TEMP_186_mux_12;
	TEMP_187_res_5 <= TEMP_186_mux_5 or TEMP_186_mux_13;
	TEMP_187_res_6 <= TEMP_186_mux_6;
	TEMP_187_res_7 <= TEMP_186_mux_7;
	-- Layer End
	TEMP_188_res_0 <= TEMP_187_res_0 or TEMP_187_res_4;
	TEMP_188_res_1 <= TEMP_187_res_1 or TEMP_187_res_5;
	TEMP_188_res_2 <= TEMP_187_res_2 or TEMP_187_res_6;
	TEMP_188_res_3 <= TEMP_187_res_3 or TEMP_187_res_7;
	-- Layer End
	TEMP_189_res_0 <= TEMP_188_res_0 or TEMP_188_res_2;
	TEMP_189_res_1 <= TEMP_188_res_1 or TEMP_188_res_3;
	-- Layer End
	bypass_data_8 <= TEMP_189_res_0 or TEMP_189_res_1;
	-- Mux1H End

	ldq_data_8_d <= read_data_8 or bypass_data_8;
	ldq_data_wen_8 <= bypass_en_8 or read_valid_8;
	read_idx_oh_9_0 <= rresp_valid_0_i when ( rresp_id_0_i = "1001" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_9, rresp_data, read_idx_oh_9)
	TEMP_190_mux_0 <= rresp_data_0_i when read_idx_oh_9_0 = '1' else "00000000000000000000000000000000";
	read_data_9 <= TEMP_190_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_9, read_idx_oh_9, or)
	read_valid_9 <= read_idx_oh_9_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_9, stq_data, bypass_idx_oh_9)
	TEMP_191_mux_0 <= stq_data_0_q when bypass_idx_oh_9(0) = '1' else "00000000000000000000000000000000";
	TEMP_191_mux_1 <= stq_data_1_q when bypass_idx_oh_9(1) = '1' else "00000000000000000000000000000000";
	TEMP_191_mux_2 <= stq_data_2_q when bypass_idx_oh_9(2) = '1' else "00000000000000000000000000000000";
	TEMP_191_mux_3 <= stq_data_3_q when bypass_idx_oh_9(3) = '1' else "00000000000000000000000000000000";
	TEMP_191_mux_4 <= stq_data_4_q when bypass_idx_oh_9(4) = '1' else "00000000000000000000000000000000";
	TEMP_191_mux_5 <= stq_data_5_q when bypass_idx_oh_9(5) = '1' else "00000000000000000000000000000000";
	TEMP_191_mux_6 <= stq_data_6_q when bypass_idx_oh_9(6) = '1' else "00000000000000000000000000000000";
	TEMP_191_mux_7 <= stq_data_7_q when bypass_idx_oh_9(7) = '1' else "00000000000000000000000000000000";
	TEMP_191_mux_8 <= stq_data_8_q when bypass_idx_oh_9(8) = '1' else "00000000000000000000000000000000";
	TEMP_191_mux_9 <= stq_data_9_q when bypass_idx_oh_9(9) = '1' else "00000000000000000000000000000000";
	TEMP_191_mux_10 <= stq_data_10_q when bypass_idx_oh_9(10) = '1' else "00000000000000000000000000000000";
	TEMP_191_mux_11 <= stq_data_11_q when bypass_idx_oh_9(11) = '1' else "00000000000000000000000000000000";
	TEMP_191_mux_12 <= stq_data_12_q when bypass_idx_oh_9(12) = '1' else "00000000000000000000000000000000";
	TEMP_191_mux_13 <= stq_data_13_q when bypass_idx_oh_9(13) = '1' else "00000000000000000000000000000000";
	TEMP_192_res_0 <= TEMP_191_mux_0 or TEMP_191_mux_8;
	TEMP_192_res_1 <= TEMP_191_mux_1 or TEMP_191_mux_9;
	TEMP_192_res_2 <= TEMP_191_mux_2 or TEMP_191_mux_10;
	TEMP_192_res_3 <= TEMP_191_mux_3 or TEMP_191_mux_11;
	TEMP_192_res_4 <= TEMP_191_mux_4 or TEMP_191_mux_12;
	TEMP_192_res_5 <= TEMP_191_mux_5 or TEMP_191_mux_13;
	TEMP_192_res_6 <= TEMP_191_mux_6;
	TEMP_192_res_7 <= TEMP_191_mux_7;
	-- Layer End
	TEMP_193_res_0 <= TEMP_192_res_0 or TEMP_192_res_4;
	TEMP_193_res_1 <= TEMP_192_res_1 or TEMP_192_res_5;
	TEMP_193_res_2 <= TEMP_192_res_2 or TEMP_192_res_6;
	TEMP_193_res_3 <= TEMP_192_res_3 or TEMP_192_res_7;
	-- Layer End
	TEMP_194_res_0 <= TEMP_193_res_0 or TEMP_193_res_2;
	TEMP_194_res_1 <= TEMP_193_res_1 or TEMP_193_res_3;
	-- Layer End
	bypass_data_9 <= TEMP_194_res_0 or TEMP_194_res_1;
	-- Mux1H End

	ldq_data_9_d <= read_data_9 or bypass_data_9;
	ldq_data_wen_9 <= bypass_en_9 or read_valid_9;
	read_idx_oh_10_0 <= rresp_valid_0_i when ( rresp_id_0_i = "1010" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_10, rresp_data, read_idx_oh_10)
	TEMP_195_mux_0 <= rresp_data_0_i when read_idx_oh_10_0 = '1' else "00000000000000000000000000000000";
	read_data_10 <= TEMP_195_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_10, read_idx_oh_10, or)
	read_valid_10 <= read_idx_oh_10_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_10, stq_data, bypass_idx_oh_10)
	TEMP_196_mux_0 <= stq_data_0_q when bypass_idx_oh_10(0) = '1' else "00000000000000000000000000000000";
	TEMP_196_mux_1 <= stq_data_1_q when bypass_idx_oh_10(1) = '1' else "00000000000000000000000000000000";
	TEMP_196_mux_2 <= stq_data_2_q when bypass_idx_oh_10(2) = '1' else "00000000000000000000000000000000";
	TEMP_196_mux_3 <= stq_data_3_q when bypass_idx_oh_10(3) = '1' else "00000000000000000000000000000000";
	TEMP_196_mux_4 <= stq_data_4_q when bypass_idx_oh_10(4) = '1' else "00000000000000000000000000000000";
	TEMP_196_mux_5 <= stq_data_5_q when bypass_idx_oh_10(5) = '1' else "00000000000000000000000000000000";
	TEMP_196_mux_6 <= stq_data_6_q when bypass_idx_oh_10(6) = '1' else "00000000000000000000000000000000";
	TEMP_196_mux_7 <= stq_data_7_q when bypass_idx_oh_10(7) = '1' else "00000000000000000000000000000000";
	TEMP_196_mux_8 <= stq_data_8_q when bypass_idx_oh_10(8) = '1' else "00000000000000000000000000000000";
	TEMP_196_mux_9 <= stq_data_9_q when bypass_idx_oh_10(9) = '1' else "00000000000000000000000000000000";
	TEMP_196_mux_10 <= stq_data_10_q when bypass_idx_oh_10(10) = '1' else "00000000000000000000000000000000";
	TEMP_196_mux_11 <= stq_data_11_q when bypass_idx_oh_10(11) = '1' else "00000000000000000000000000000000";
	TEMP_196_mux_12 <= stq_data_12_q when bypass_idx_oh_10(12) = '1' else "00000000000000000000000000000000";
	TEMP_196_mux_13 <= stq_data_13_q when bypass_idx_oh_10(13) = '1' else "00000000000000000000000000000000";
	TEMP_197_res_0 <= TEMP_196_mux_0 or TEMP_196_mux_8;
	TEMP_197_res_1 <= TEMP_196_mux_1 or TEMP_196_mux_9;
	TEMP_197_res_2 <= TEMP_196_mux_2 or TEMP_196_mux_10;
	TEMP_197_res_3 <= TEMP_196_mux_3 or TEMP_196_mux_11;
	TEMP_197_res_4 <= TEMP_196_mux_4 or TEMP_196_mux_12;
	TEMP_197_res_5 <= TEMP_196_mux_5 or TEMP_196_mux_13;
	TEMP_197_res_6 <= TEMP_196_mux_6;
	TEMP_197_res_7 <= TEMP_196_mux_7;
	-- Layer End
	TEMP_198_res_0 <= TEMP_197_res_0 or TEMP_197_res_4;
	TEMP_198_res_1 <= TEMP_197_res_1 or TEMP_197_res_5;
	TEMP_198_res_2 <= TEMP_197_res_2 or TEMP_197_res_6;
	TEMP_198_res_3 <= TEMP_197_res_3 or TEMP_197_res_7;
	-- Layer End
	TEMP_199_res_0 <= TEMP_198_res_0 or TEMP_198_res_2;
	TEMP_199_res_1 <= TEMP_198_res_1 or TEMP_198_res_3;
	-- Layer End
	bypass_data_10 <= TEMP_199_res_0 or TEMP_199_res_1;
	-- Mux1H End

	ldq_data_10_d <= read_data_10 or bypass_data_10;
	ldq_data_wen_10 <= bypass_en_10 or read_valid_10;
	read_idx_oh_11_0 <= rresp_valid_0_i when ( rresp_id_0_i = "1011" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_11, rresp_data, read_idx_oh_11)
	TEMP_200_mux_0 <= rresp_data_0_i when read_idx_oh_11_0 = '1' else "00000000000000000000000000000000";
	read_data_11 <= TEMP_200_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_11, read_idx_oh_11, or)
	read_valid_11 <= read_idx_oh_11_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_11, stq_data, bypass_idx_oh_11)
	TEMP_201_mux_0 <= stq_data_0_q when bypass_idx_oh_11(0) = '1' else "00000000000000000000000000000000";
	TEMP_201_mux_1 <= stq_data_1_q when bypass_idx_oh_11(1) = '1' else "00000000000000000000000000000000";
	TEMP_201_mux_2 <= stq_data_2_q when bypass_idx_oh_11(2) = '1' else "00000000000000000000000000000000";
	TEMP_201_mux_3 <= stq_data_3_q when bypass_idx_oh_11(3) = '1' else "00000000000000000000000000000000";
	TEMP_201_mux_4 <= stq_data_4_q when bypass_idx_oh_11(4) = '1' else "00000000000000000000000000000000";
	TEMP_201_mux_5 <= stq_data_5_q when bypass_idx_oh_11(5) = '1' else "00000000000000000000000000000000";
	TEMP_201_mux_6 <= stq_data_6_q when bypass_idx_oh_11(6) = '1' else "00000000000000000000000000000000";
	TEMP_201_mux_7 <= stq_data_7_q when bypass_idx_oh_11(7) = '1' else "00000000000000000000000000000000";
	TEMP_201_mux_8 <= stq_data_8_q when bypass_idx_oh_11(8) = '1' else "00000000000000000000000000000000";
	TEMP_201_mux_9 <= stq_data_9_q when bypass_idx_oh_11(9) = '1' else "00000000000000000000000000000000";
	TEMP_201_mux_10 <= stq_data_10_q when bypass_idx_oh_11(10) = '1' else "00000000000000000000000000000000";
	TEMP_201_mux_11 <= stq_data_11_q when bypass_idx_oh_11(11) = '1' else "00000000000000000000000000000000";
	TEMP_201_mux_12 <= stq_data_12_q when bypass_idx_oh_11(12) = '1' else "00000000000000000000000000000000";
	TEMP_201_mux_13 <= stq_data_13_q when bypass_idx_oh_11(13) = '1' else "00000000000000000000000000000000";
	TEMP_202_res_0 <= TEMP_201_mux_0 or TEMP_201_mux_8;
	TEMP_202_res_1 <= TEMP_201_mux_1 or TEMP_201_mux_9;
	TEMP_202_res_2 <= TEMP_201_mux_2 or TEMP_201_mux_10;
	TEMP_202_res_3 <= TEMP_201_mux_3 or TEMP_201_mux_11;
	TEMP_202_res_4 <= TEMP_201_mux_4 or TEMP_201_mux_12;
	TEMP_202_res_5 <= TEMP_201_mux_5 or TEMP_201_mux_13;
	TEMP_202_res_6 <= TEMP_201_mux_6;
	TEMP_202_res_7 <= TEMP_201_mux_7;
	-- Layer End
	TEMP_203_res_0 <= TEMP_202_res_0 or TEMP_202_res_4;
	TEMP_203_res_1 <= TEMP_202_res_1 or TEMP_202_res_5;
	TEMP_203_res_2 <= TEMP_202_res_2 or TEMP_202_res_6;
	TEMP_203_res_3 <= TEMP_202_res_3 or TEMP_202_res_7;
	-- Layer End
	TEMP_204_res_0 <= TEMP_203_res_0 or TEMP_203_res_2;
	TEMP_204_res_1 <= TEMP_203_res_1 or TEMP_203_res_3;
	-- Layer End
	bypass_data_11 <= TEMP_204_res_0 or TEMP_204_res_1;
	-- Mux1H End

	ldq_data_11_d <= read_data_11 or bypass_data_11;
	ldq_data_wen_11 <= bypass_en_11 or read_valid_11;
	read_idx_oh_12_0 <= rresp_valid_0_i when ( rresp_id_0_i = "1100" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_12, rresp_data, read_idx_oh_12)
	TEMP_205_mux_0 <= rresp_data_0_i when read_idx_oh_12_0 = '1' else "00000000000000000000000000000000";
	read_data_12 <= TEMP_205_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_12, read_idx_oh_12, or)
	read_valid_12 <= read_idx_oh_12_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_12, stq_data, bypass_idx_oh_12)
	TEMP_206_mux_0 <= stq_data_0_q when bypass_idx_oh_12(0) = '1' else "00000000000000000000000000000000";
	TEMP_206_mux_1 <= stq_data_1_q when bypass_idx_oh_12(1) = '1' else "00000000000000000000000000000000";
	TEMP_206_mux_2 <= stq_data_2_q when bypass_idx_oh_12(2) = '1' else "00000000000000000000000000000000";
	TEMP_206_mux_3 <= stq_data_3_q when bypass_idx_oh_12(3) = '1' else "00000000000000000000000000000000";
	TEMP_206_mux_4 <= stq_data_4_q when bypass_idx_oh_12(4) = '1' else "00000000000000000000000000000000";
	TEMP_206_mux_5 <= stq_data_5_q when bypass_idx_oh_12(5) = '1' else "00000000000000000000000000000000";
	TEMP_206_mux_6 <= stq_data_6_q when bypass_idx_oh_12(6) = '1' else "00000000000000000000000000000000";
	TEMP_206_mux_7 <= stq_data_7_q when bypass_idx_oh_12(7) = '1' else "00000000000000000000000000000000";
	TEMP_206_mux_8 <= stq_data_8_q when bypass_idx_oh_12(8) = '1' else "00000000000000000000000000000000";
	TEMP_206_mux_9 <= stq_data_9_q when bypass_idx_oh_12(9) = '1' else "00000000000000000000000000000000";
	TEMP_206_mux_10 <= stq_data_10_q when bypass_idx_oh_12(10) = '1' else "00000000000000000000000000000000";
	TEMP_206_mux_11 <= stq_data_11_q when bypass_idx_oh_12(11) = '1' else "00000000000000000000000000000000";
	TEMP_206_mux_12 <= stq_data_12_q when bypass_idx_oh_12(12) = '1' else "00000000000000000000000000000000";
	TEMP_206_mux_13 <= stq_data_13_q when bypass_idx_oh_12(13) = '1' else "00000000000000000000000000000000";
	TEMP_207_res_0 <= TEMP_206_mux_0 or TEMP_206_mux_8;
	TEMP_207_res_1 <= TEMP_206_mux_1 or TEMP_206_mux_9;
	TEMP_207_res_2 <= TEMP_206_mux_2 or TEMP_206_mux_10;
	TEMP_207_res_3 <= TEMP_206_mux_3 or TEMP_206_mux_11;
	TEMP_207_res_4 <= TEMP_206_mux_4 or TEMP_206_mux_12;
	TEMP_207_res_5 <= TEMP_206_mux_5 or TEMP_206_mux_13;
	TEMP_207_res_6 <= TEMP_206_mux_6;
	TEMP_207_res_7 <= TEMP_206_mux_7;
	-- Layer End
	TEMP_208_res_0 <= TEMP_207_res_0 or TEMP_207_res_4;
	TEMP_208_res_1 <= TEMP_207_res_1 or TEMP_207_res_5;
	TEMP_208_res_2 <= TEMP_207_res_2 or TEMP_207_res_6;
	TEMP_208_res_3 <= TEMP_207_res_3 or TEMP_207_res_7;
	-- Layer End
	TEMP_209_res_0 <= TEMP_208_res_0 or TEMP_208_res_2;
	TEMP_209_res_1 <= TEMP_208_res_1 or TEMP_208_res_3;
	-- Layer End
	bypass_data_12 <= TEMP_209_res_0 or TEMP_209_res_1;
	-- Mux1H End

	ldq_data_12_d <= read_data_12 or bypass_data_12;
	ldq_data_wen_12 <= bypass_en_12 or read_valid_12;
	read_idx_oh_13_0 <= rresp_valid_0_i when ( rresp_id_0_i = "1101" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_13, rresp_data, read_idx_oh_13)
	TEMP_210_mux_0 <= rresp_data_0_i when read_idx_oh_13_0 = '1' else "00000000000000000000000000000000";
	read_data_13 <= TEMP_210_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_13, read_idx_oh_13, or)
	read_valid_13 <= read_idx_oh_13_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_13, stq_data, bypass_idx_oh_13)
	TEMP_211_mux_0 <= stq_data_0_q when bypass_idx_oh_13(0) = '1' else "00000000000000000000000000000000";
	TEMP_211_mux_1 <= stq_data_1_q when bypass_idx_oh_13(1) = '1' else "00000000000000000000000000000000";
	TEMP_211_mux_2 <= stq_data_2_q when bypass_idx_oh_13(2) = '1' else "00000000000000000000000000000000";
	TEMP_211_mux_3 <= stq_data_3_q when bypass_idx_oh_13(3) = '1' else "00000000000000000000000000000000";
	TEMP_211_mux_4 <= stq_data_4_q when bypass_idx_oh_13(4) = '1' else "00000000000000000000000000000000";
	TEMP_211_mux_5 <= stq_data_5_q when bypass_idx_oh_13(5) = '1' else "00000000000000000000000000000000";
	TEMP_211_mux_6 <= stq_data_6_q when bypass_idx_oh_13(6) = '1' else "00000000000000000000000000000000";
	TEMP_211_mux_7 <= stq_data_7_q when bypass_idx_oh_13(7) = '1' else "00000000000000000000000000000000";
	TEMP_211_mux_8 <= stq_data_8_q when bypass_idx_oh_13(8) = '1' else "00000000000000000000000000000000";
	TEMP_211_mux_9 <= stq_data_9_q when bypass_idx_oh_13(9) = '1' else "00000000000000000000000000000000";
	TEMP_211_mux_10 <= stq_data_10_q when bypass_idx_oh_13(10) = '1' else "00000000000000000000000000000000";
	TEMP_211_mux_11 <= stq_data_11_q when bypass_idx_oh_13(11) = '1' else "00000000000000000000000000000000";
	TEMP_211_mux_12 <= stq_data_12_q when bypass_idx_oh_13(12) = '1' else "00000000000000000000000000000000";
	TEMP_211_mux_13 <= stq_data_13_q when bypass_idx_oh_13(13) = '1' else "00000000000000000000000000000000";
	TEMP_212_res_0 <= TEMP_211_mux_0 or TEMP_211_mux_8;
	TEMP_212_res_1 <= TEMP_211_mux_1 or TEMP_211_mux_9;
	TEMP_212_res_2 <= TEMP_211_mux_2 or TEMP_211_mux_10;
	TEMP_212_res_3 <= TEMP_211_mux_3 or TEMP_211_mux_11;
	TEMP_212_res_4 <= TEMP_211_mux_4 or TEMP_211_mux_12;
	TEMP_212_res_5 <= TEMP_211_mux_5 or TEMP_211_mux_13;
	TEMP_212_res_6 <= TEMP_211_mux_6;
	TEMP_212_res_7 <= TEMP_211_mux_7;
	-- Layer End
	TEMP_213_res_0 <= TEMP_212_res_0 or TEMP_212_res_4;
	TEMP_213_res_1 <= TEMP_212_res_1 or TEMP_212_res_5;
	TEMP_213_res_2 <= TEMP_212_res_2 or TEMP_212_res_6;
	TEMP_213_res_3 <= TEMP_212_res_3 or TEMP_212_res_7;
	-- Layer End
	TEMP_214_res_0 <= TEMP_213_res_0 or TEMP_213_res_2;
	TEMP_214_res_1 <= TEMP_213_res_1 or TEMP_213_res_3;
	-- Layer End
	bypass_data_13 <= TEMP_214_res_0 or TEMP_214_res_1;
	-- Mux1H End

	ldq_data_13_d <= read_data_13 or bypass_data_13;
	ldq_data_wen_13 <= bypass_en_13 or read_valid_13;
	rresp_ready_0_o <= '1';
	stq_reset_0 <= wresp_valid_0_i when ( stq_resp_q = "0000" ) else '0';
	stq_reset_1 <= wresp_valid_0_i when ( stq_resp_q = "0001" ) else '0';
	stq_reset_2 <= wresp_valid_0_i when ( stq_resp_q = "0010" ) else '0';
	stq_reset_3 <= wresp_valid_0_i when ( stq_resp_q = "0011" ) else '0';
	stq_reset_4 <= wresp_valid_0_i when ( stq_resp_q = "0100" ) else '0';
	stq_reset_5 <= wresp_valid_0_i when ( stq_resp_q = "0101" ) else '0';
	stq_reset_6 <= wresp_valid_0_i when ( stq_resp_q = "0110" ) else '0';
	stq_reset_7 <= wresp_valid_0_i when ( stq_resp_q = "0111" ) else '0';
	stq_reset_8 <= wresp_valid_0_i when ( stq_resp_q = "1000" ) else '0';
	stq_reset_9 <= wresp_valid_0_i when ( stq_resp_q = "1001" ) else '0';
	stq_reset_10 <= wresp_valid_0_i when ( stq_resp_q = "1010" ) else '0';
	stq_reset_11 <= wresp_valid_0_i when ( stq_resp_q = "1011" ) else '0';
	stq_reset_12 <= wresp_valid_0_i when ( stq_resp_q = "1100" ) else '0';
	stq_reset_13 <= wresp_valid_0_i when ( stq_resp_q = "1101" ) else '0';
	stq_resp_en <= wresp_valid_0_i;
	wresp_ready_0_o <= '1';

	process (clk, rst) is
	begin
		if (rst = '1') then
			ldq_alloc_0_q <= '0';
			ldq_alloc_1_q <= '0';
			ldq_alloc_2_q <= '0';
			ldq_alloc_3_q <= '0';
			ldq_alloc_4_q <= '0';
			ldq_alloc_5_q <= '0';
			ldq_alloc_6_q <= '0';
			ldq_alloc_7_q <= '0';
			ldq_alloc_8_q <= '0';
			ldq_alloc_9_q <= '0';
			ldq_alloc_10_q <= '0';
			ldq_alloc_11_q <= '0';
			ldq_alloc_12_q <= '0';
			ldq_alloc_13_q <= '0';
		elsif (rising_edge(clk)) then
			ldq_alloc_0_q <= ldq_alloc_0_d;
			ldq_alloc_1_q <= ldq_alloc_1_d;
			ldq_alloc_2_q <= ldq_alloc_2_d;
			ldq_alloc_3_q <= ldq_alloc_3_d;
			ldq_alloc_4_q <= ldq_alloc_4_d;
			ldq_alloc_5_q <= ldq_alloc_5_d;
			ldq_alloc_6_q <= ldq_alloc_6_d;
			ldq_alloc_7_q <= ldq_alloc_7_d;
			ldq_alloc_8_q <= ldq_alloc_8_d;
			ldq_alloc_9_q <= ldq_alloc_9_d;
			ldq_alloc_10_q <= ldq_alloc_10_d;
			ldq_alloc_11_q <= ldq_alloc_11_d;
			ldq_alloc_12_q <= ldq_alloc_12_d;
			ldq_alloc_13_q <= ldq_alloc_13_d;
		end if;
		if (rising_edge(clk)) then
			ldq_issue_0_q <= ldq_issue_0_d;
			ldq_issue_1_q <= ldq_issue_1_d;
			ldq_issue_2_q <= ldq_issue_2_d;
			ldq_issue_3_q <= ldq_issue_3_d;
			ldq_issue_4_q <= ldq_issue_4_d;
			ldq_issue_5_q <= ldq_issue_5_d;
			ldq_issue_6_q <= ldq_issue_6_d;
			ldq_issue_7_q <= ldq_issue_7_d;
			ldq_issue_8_q <= ldq_issue_8_d;
			ldq_issue_9_q <= ldq_issue_9_d;
			ldq_issue_10_q <= ldq_issue_10_d;
			ldq_issue_11_q <= ldq_issue_11_d;
			ldq_issue_12_q <= ldq_issue_12_d;
			ldq_issue_13_q <= ldq_issue_13_d;
		end if;
		if (rising_edge(clk)) then
			ldq_addr_valid_0_q <= ldq_addr_valid_0_d;
			ldq_addr_valid_1_q <= ldq_addr_valid_1_d;
			ldq_addr_valid_2_q <= ldq_addr_valid_2_d;
			ldq_addr_valid_3_q <= ldq_addr_valid_3_d;
			ldq_addr_valid_4_q <= ldq_addr_valid_4_d;
			ldq_addr_valid_5_q <= ldq_addr_valid_5_d;
			ldq_addr_valid_6_q <= ldq_addr_valid_6_d;
			ldq_addr_valid_7_q <= ldq_addr_valid_7_d;
			ldq_addr_valid_8_q <= ldq_addr_valid_8_d;
			ldq_addr_valid_9_q <= ldq_addr_valid_9_d;
			ldq_addr_valid_10_q <= ldq_addr_valid_10_d;
			ldq_addr_valid_11_q <= ldq_addr_valid_11_d;
			ldq_addr_valid_12_q <= ldq_addr_valid_12_d;
			ldq_addr_valid_13_q <= ldq_addr_valid_13_d;
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
			if (ldq_addr_wen_4 = '1') then
				ldq_addr_4_q <= ldq_addr_4_d;
			end if;
			if (ldq_addr_wen_5 = '1') then
				ldq_addr_5_q <= ldq_addr_5_d;
			end if;
			if (ldq_addr_wen_6 = '1') then
				ldq_addr_6_q <= ldq_addr_6_d;
			end if;
			if (ldq_addr_wen_7 = '1') then
				ldq_addr_7_q <= ldq_addr_7_d;
			end if;
			if (ldq_addr_wen_8 = '1') then
				ldq_addr_8_q <= ldq_addr_8_d;
			end if;
			if (ldq_addr_wen_9 = '1') then
				ldq_addr_9_q <= ldq_addr_9_d;
			end if;
			if (ldq_addr_wen_10 = '1') then
				ldq_addr_10_q <= ldq_addr_10_d;
			end if;
			if (ldq_addr_wen_11 = '1') then
				ldq_addr_11_q <= ldq_addr_11_d;
			end if;
			if (ldq_addr_wen_12 = '1') then
				ldq_addr_12_q <= ldq_addr_12_d;
			end if;
			if (ldq_addr_wen_13 = '1') then
				ldq_addr_13_q <= ldq_addr_13_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			ldq_data_valid_0_q <= ldq_data_valid_0_d;
			ldq_data_valid_1_q <= ldq_data_valid_1_d;
			ldq_data_valid_2_q <= ldq_data_valid_2_d;
			ldq_data_valid_3_q <= ldq_data_valid_3_d;
			ldq_data_valid_4_q <= ldq_data_valid_4_d;
			ldq_data_valid_5_q <= ldq_data_valid_5_d;
			ldq_data_valid_6_q <= ldq_data_valid_6_d;
			ldq_data_valid_7_q <= ldq_data_valid_7_d;
			ldq_data_valid_8_q <= ldq_data_valid_8_d;
			ldq_data_valid_9_q <= ldq_data_valid_9_d;
			ldq_data_valid_10_q <= ldq_data_valid_10_d;
			ldq_data_valid_11_q <= ldq_data_valid_11_d;
			ldq_data_valid_12_q <= ldq_data_valid_12_d;
			ldq_data_valid_13_q <= ldq_data_valid_13_d;
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
			if (ldq_data_wen_4 = '1') then
				ldq_data_4_q <= ldq_data_4_d;
			end if;
			if (ldq_data_wen_5 = '1') then
				ldq_data_5_q <= ldq_data_5_d;
			end if;
			if (ldq_data_wen_6 = '1') then
				ldq_data_6_q <= ldq_data_6_d;
			end if;
			if (ldq_data_wen_7 = '1') then
				ldq_data_7_q <= ldq_data_7_d;
			end if;
			if (ldq_data_wen_8 = '1') then
				ldq_data_8_q <= ldq_data_8_d;
			end if;
			if (ldq_data_wen_9 = '1') then
				ldq_data_9_q <= ldq_data_9_d;
			end if;
			if (ldq_data_wen_10 = '1') then
				ldq_data_10_q <= ldq_data_10_d;
			end if;
			if (ldq_data_wen_11 = '1') then
				ldq_data_11_q <= ldq_data_11_d;
			end if;
			if (ldq_data_wen_12 = '1') then
				ldq_data_12_q <= ldq_data_12_d;
			end if;
			if (ldq_data_wen_13 = '1') then
				ldq_data_13_q <= ldq_data_13_d;
			end if;
		end if;
		if (rst = '1') then
			stq_alloc_0_q <= '0';
			stq_alloc_1_q <= '0';
			stq_alloc_2_q <= '0';
			stq_alloc_3_q <= '0';
			stq_alloc_4_q <= '0';
			stq_alloc_5_q <= '0';
			stq_alloc_6_q <= '0';
			stq_alloc_7_q <= '0';
			stq_alloc_8_q <= '0';
			stq_alloc_9_q <= '0';
			stq_alloc_10_q <= '0';
			stq_alloc_11_q <= '0';
			stq_alloc_12_q <= '0';
			stq_alloc_13_q <= '0';
		elsif (rising_edge(clk)) then
			stq_alloc_0_q <= stq_alloc_0_d;
			stq_alloc_1_q <= stq_alloc_1_d;
			stq_alloc_2_q <= stq_alloc_2_d;
			stq_alloc_3_q <= stq_alloc_3_d;
			stq_alloc_4_q <= stq_alloc_4_d;
			stq_alloc_5_q <= stq_alloc_5_d;
			stq_alloc_6_q <= stq_alloc_6_d;
			stq_alloc_7_q <= stq_alloc_7_d;
			stq_alloc_8_q <= stq_alloc_8_d;
			stq_alloc_9_q <= stq_alloc_9_d;
			stq_alloc_10_q <= stq_alloc_10_d;
			stq_alloc_11_q <= stq_alloc_11_d;
			stq_alloc_12_q <= stq_alloc_12_d;
			stq_alloc_13_q <= stq_alloc_13_d;
		end if;
		if (rising_edge(clk)) then
			stq_addr_valid_0_q <= stq_addr_valid_0_d;
			stq_addr_valid_1_q <= stq_addr_valid_1_d;
			stq_addr_valid_2_q <= stq_addr_valid_2_d;
			stq_addr_valid_3_q <= stq_addr_valid_3_d;
			stq_addr_valid_4_q <= stq_addr_valid_4_d;
			stq_addr_valid_5_q <= stq_addr_valid_5_d;
			stq_addr_valid_6_q <= stq_addr_valid_6_d;
			stq_addr_valid_7_q <= stq_addr_valid_7_d;
			stq_addr_valid_8_q <= stq_addr_valid_8_d;
			stq_addr_valid_9_q <= stq_addr_valid_9_d;
			stq_addr_valid_10_q <= stq_addr_valid_10_d;
			stq_addr_valid_11_q <= stq_addr_valid_11_d;
			stq_addr_valid_12_q <= stq_addr_valid_12_d;
			stq_addr_valid_13_q <= stq_addr_valid_13_d;
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
			if (stq_addr_wen_5 = '1') then
				stq_addr_5_q <= stq_addr_5_d;
			end if;
			if (stq_addr_wen_6 = '1') then
				stq_addr_6_q <= stq_addr_6_d;
			end if;
			if (stq_addr_wen_7 = '1') then
				stq_addr_7_q <= stq_addr_7_d;
			end if;
			if (stq_addr_wen_8 = '1') then
				stq_addr_8_q <= stq_addr_8_d;
			end if;
			if (stq_addr_wen_9 = '1') then
				stq_addr_9_q <= stq_addr_9_d;
			end if;
			if (stq_addr_wen_10 = '1') then
				stq_addr_10_q <= stq_addr_10_d;
			end if;
			if (stq_addr_wen_11 = '1') then
				stq_addr_11_q <= stq_addr_11_d;
			end if;
			if (stq_addr_wen_12 = '1') then
				stq_addr_12_q <= stq_addr_12_d;
			end if;
			if (stq_addr_wen_13 = '1') then
				stq_addr_13_q <= stq_addr_13_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			stq_data_valid_0_q <= stq_data_valid_0_d;
			stq_data_valid_1_q <= stq_data_valid_1_d;
			stq_data_valid_2_q <= stq_data_valid_2_d;
			stq_data_valid_3_q <= stq_data_valid_3_d;
			stq_data_valid_4_q <= stq_data_valid_4_d;
			stq_data_valid_5_q <= stq_data_valid_5_d;
			stq_data_valid_6_q <= stq_data_valid_6_d;
			stq_data_valid_7_q <= stq_data_valid_7_d;
			stq_data_valid_8_q <= stq_data_valid_8_d;
			stq_data_valid_9_q <= stq_data_valid_9_d;
			stq_data_valid_10_q <= stq_data_valid_10_d;
			stq_data_valid_11_q <= stq_data_valid_11_d;
			stq_data_valid_12_q <= stq_data_valid_12_d;
			stq_data_valid_13_q <= stq_data_valid_13_d;
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
			if (stq_data_wen_5 = '1') then
				stq_data_5_q <= stq_data_5_d;
			end if;
			if (stq_data_wen_6 = '1') then
				stq_data_6_q <= stq_data_6_d;
			end if;
			if (stq_data_wen_7 = '1') then
				stq_data_7_q <= stq_data_7_d;
			end if;
			if (stq_data_wen_8 = '1') then
				stq_data_8_q <= stq_data_8_d;
			end if;
			if (stq_data_wen_9 = '1') then
				stq_data_9_q <= stq_data_9_d;
			end if;
			if (stq_data_wen_10 = '1') then
				stq_data_10_q <= stq_data_10_d;
			end if;
			if (stq_data_wen_11 = '1') then
				stq_data_11_q <= stq_data_11_d;
			end if;
			if (stq_data_wen_12 = '1') then
				stq_data_12_q <= stq_data_12_d;
			end if;
			if (stq_data_wen_13 = '1') then
				stq_data_13_q <= stq_data_13_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			store_is_older_0_q <= store_is_older_0_d;
			store_is_older_1_q <= store_is_older_1_d;
			store_is_older_2_q <= store_is_older_2_d;
			store_is_older_3_q <= store_is_older_3_d;
			store_is_older_4_q <= store_is_older_4_d;
			store_is_older_5_q <= store_is_older_5_d;
			store_is_older_6_q <= store_is_older_6_d;
			store_is_older_7_q <= store_is_older_7_d;
			store_is_older_8_q <= store_is_older_8_d;
			store_is_older_9_q <= store_is_older_9_d;
			store_is_older_10_q <= store_is_older_10_d;
			store_is_older_11_q <= store_is_older_11_d;
			store_is_older_12_q <= store_is_older_12_d;
			store_is_older_13_q <= store_is_older_13_d;
		end if;
		if (rst = '1') then
			ldq_tail_q <= "0000";
		elsif (rising_edge(clk)) then
			ldq_tail_q <= ldq_tail_d;
		end if;
		if (rst = '1') then
			ldq_head_q <= "0000";
		elsif (rising_edge(clk)) then
			ldq_head_q <= ldq_head_d;
		end if;
		if (rst = '1') then
			stq_tail_q <= "0000";
		elsif (rising_edge(clk)) then
			stq_tail_q <= stq_tail_d;
		end if;
		if (rst = '1') then
			stq_head_q <= "0000";
		elsif (rising_edge(clk)) then
			stq_head_q <= stq_head_d;
		end if;
		if (rst = '1') then
			stq_issue_q <= "0000";
		elsif (rising_edge(clk)) then
			if (stq_issue_en = '1') then
				stq_issue_q <= stq_issue_d;
			end if;
		end if;
		if (rst = '1') then
			stq_resp_q <= "0000";
		elsif (rising_edge(clk)) then
			if (stq_resp_en = '1') then
				stq_resp_q <= stq_resp_d;
			end if;
		end if;
	end process;
end architecture;
