

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq4_core_ga is
	port(
		rst : in std_logic;
		clk : in std_logic;
		group_init_valid_0_i : in std_logic;
		group_init_valid_1_i : in std_logic;
		group_init_valid_2_i : in std_logic;
		group_init_ready_0_o : out std_logic;
		group_init_ready_1_o : out std_logic;
		group_init_ready_2_o : out std_logic;
		ldq_tail_i : in std_logic_vector(2 downto 0);
		ldq_head_i : in std_logic_vector(2 downto 0);
		ldq_empty_i : in std_logic;
		stq_tail_i : in std_logic_vector(2 downto 0);
		stq_head_i : in std_logic_vector(2 downto 0);
		stq_empty_i : in std_logic;
		ldq_wen_0_o : out std_logic;
		ldq_wen_1_o : out std_logic;
		ldq_wen_2_o : out std_logic;
		ldq_wen_3_o : out std_logic;
		ldq_wen_4_o : out std_logic;
		ldq_wen_5_o : out std_logic;
		ldq_wen_6_o : out std_logic;
		ldq_wen_7_o : out std_logic;
		num_loads_o : out std_logic_vector(2 downto 0);
		ldq_port_idx_0_o : out std_logic_vector(0 downto 0);
		ldq_port_idx_1_o : out std_logic_vector(0 downto 0);
		ldq_port_idx_2_o : out std_logic_vector(0 downto 0);
		ldq_port_idx_3_o : out std_logic_vector(0 downto 0);
		ldq_port_idx_4_o : out std_logic_vector(0 downto 0);
		ldq_port_idx_5_o : out std_logic_vector(0 downto 0);
		ldq_port_idx_6_o : out std_logic_vector(0 downto 0);
		ldq_port_idx_7_o : out std_logic_vector(0 downto 0);
		stq_wen_0_o : out std_logic;
		stq_wen_1_o : out std_logic;
		stq_wen_2_o : out std_logic;
		stq_wen_3_o : out std_logic;
		stq_wen_4_o : out std_logic;
		stq_wen_5_o : out std_logic;
		stq_wen_6_o : out std_logic;
		stq_wen_7_o : out std_logic;
		num_stores_o : out std_logic_vector(2 downto 0);
		stq_port_idx_0_o : out std_logic_vector(0 downto 0);
		stq_port_idx_1_o : out std_logic_vector(0 downto 0);
		stq_port_idx_2_o : out std_logic_vector(0 downto 0);
		stq_port_idx_3_o : out std_logic_vector(0 downto 0);
		stq_port_idx_4_o : out std_logic_vector(0 downto 0);
		stq_port_idx_5_o : out std_logic_vector(0 downto 0);
		stq_port_idx_6_o : out std_logic_vector(0 downto 0);
		stq_port_idx_7_o : out std_logic_vector(0 downto 0);
		ga_ls_order_0_o : out std_logic_vector(7 downto 0);
		ga_ls_order_1_o : out std_logic_vector(7 downto 0);
		ga_ls_order_2_o : out std_logic_vector(7 downto 0);
		ga_ls_order_3_o : out std_logic_vector(7 downto 0);
		ga_ls_order_4_o : out std_logic_vector(7 downto 0);
		ga_ls_order_5_o : out std_logic_vector(7 downto 0);
		ga_ls_order_6_o : out std_logic_vector(7 downto 0);
		ga_ls_order_7_o : out std_logic_vector(7 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq4_core_ga is
	signal num_loads : std_logic_vector(2 downto 0);
	signal num_stores : std_logic_vector(2 downto 0);
	signal loads_sub : std_logic_vector(2 downto 0);
	signal stores_sub : std_logic_vector(2 downto 0);
	signal empty_loads : std_logic_vector(3 downto 0);
	signal empty_stores : std_logic_vector(3 downto 0);
	signal group_init_ready_0 : std_logic;
	signal group_init_ready_1 : std_logic;
	signal group_init_ready_2 : std_logic;
	signal group_init_hs_0 : std_logic;
	signal group_init_hs_1 : std_logic;
	signal group_init_hs_2 : std_logic;
	signal ldq_port_idx_rom_0 : std_logic_vector(0 downto 0);
	signal ldq_port_idx_rom_1 : std_logic_vector(0 downto 0);
	signal ldq_port_idx_rom_2 : std_logic_vector(0 downto 0);
	signal ldq_port_idx_rom_3 : std_logic_vector(0 downto 0);
	signal ldq_port_idx_rom_4 : std_logic_vector(0 downto 0);
	signal ldq_port_idx_rom_5 : std_logic_vector(0 downto 0);
	signal ldq_port_idx_rom_6 : std_logic_vector(0 downto 0);
	signal ldq_port_idx_rom_7 : std_logic_vector(0 downto 0);
	signal stq_port_idx_rom_0 : std_logic_vector(0 downto 0);
	signal stq_port_idx_rom_1 : std_logic_vector(0 downto 0);
	signal stq_port_idx_rom_2 : std_logic_vector(0 downto 0);
	signal stq_port_idx_rom_3 : std_logic_vector(0 downto 0);
	signal stq_port_idx_rom_4 : std_logic_vector(0 downto 0);
	signal stq_port_idx_rom_5 : std_logic_vector(0 downto 0);
	signal stq_port_idx_rom_6 : std_logic_vector(0 downto 0);
	signal stq_port_idx_rom_7 : std_logic_vector(0 downto 0);
	signal ga_ls_order_rom_0 : std_logic_vector(7 downto 0);
	signal ga_ls_order_rom_1 : std_logic_vector(7 downto 0);
	signal ga_ls_order_rom_2 : std_logic_vector(7 downto 0);
	signal ga_ls_order_rom_3 : std_logic_vector(7 downto 0);
	signal ga_ls_order_rom_4 : std_logic_vector(7 downto 0);
	signal ga_ls_order_rom_5 : std_logic_vector(7 downto 0);
	signal ga_ls_order_rom_6 : std_logic_vector(7 downto 0);
	signal ga_ls_order_rom_7 : std_logic_vector(7 downto 0);
	signal ga_ls_order_temp_0 : std_logic_vector(7 downto 0);
	signal ga_ls_order_temp_1 : std_logic_vector(7 downto 0);
	signal ga_ls_order_temp_2 : std_logic_vector(7 downto 0);
	signal ga_ls_order_temp_3 : std_logic_vector(7 downto 0);
	signal ga_ls_order_temp_4 : std_logic_vector(7 downto 0);
	signal ga_ls_order_temp_5 : std_logic_vector(7 downto 0);
	signal ga_ls_order_temp_6 : std_logic_vector(7 downto 0);
	signal ga_ls_order_temp_7 : std_logic_vector(7 downto 0);
	signal TEMP_1_mux_0_0 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_0_1 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_0_2 : std_logic_vector(0 downto 0);
	signal TEMP_2_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_2_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_2_mux_1_0 : std_logic_vector(0 downto 0);
	signal TEMP_2_mux_1_1 : std_logic_vector(0 downto 0);
	signal TEMP_2_mux_1_2 : std_logic_vector(0 downto 0);
	signal TEMP_3_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_3_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_3_mux_2_0 : std_logic_vector(0 downto 0);
	signal TEMP_3_mux_2_1 : std_logic_vector(0 downto 0);
	signal TEMP_3_mux_2_2 : std_logic_vector(0 downto 0);
	signal TEMP_4_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_4_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_4_mux_3_0 : std_logic_vector(0 downto 0);
	signal TEMP_4_mux_3_1 : std_logic_vector(0 downto 0);
	signal TEMP_4_mux_3_2 : std_logic_vector(0 downto 0);
	signal TEMP_5_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_5_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_5_mux_4_0 : std_logic_vector(0 downto 0);
	signal TEMP_5_mux_4_1 : std_logic_vector(0 downto 0);
	signal TEMP_5_mux_4_2 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_6_mux_5_0 : std_logic_vector(0 downto 0);
	signal TEMP_6_mux_5_1 : std_logic_vector(0 downto 0);
	signal TEMP_6_mux_5_2 : std_logic_vector(0 downto 0);
	signal TEMP_7_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_7_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_7_mux_6_0 : std_logic_vector(0 downto 0);
	signal TEMP_7_mux_6_1 : std_logic_vector(0 downto 0);
	signal TEMP_7_mux_6_2 : std_logic_vector(0 downto 0);
	signal TEMP_8_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_8_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_8_mux_7_0 : std_logic_vector(0 downto 0);
	signal TEMP_8_mux_7_1 : std_logic_vector(0 downto 0);
	signal TEMP_8_mux_7_2 : std_logic_vector(0 downto 0);
	signal TEMP_9_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_9_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_10_mux_0_0 : std_logic_vector(0 downto 0);
	signal TEMP_10_mux_0_1 : std_logic_vector(0 downto 0);
	signal TEMP_10_mux_0_2 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_11_mux_1_0 : std_logic_vector(0 downto 0);
	signal TEMP_11_mux_1_1 : std_logic_vector(0 downto 0);
	signal TEMP_11_mux_1_2 : std_logic_vector(0 downto 0);
	signal TEMP_12_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_12_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_12_mux_2_0 : std_logic_vector(0 downto 0);
	signal TEMP_12_mux_2_1 : std_logic_vector(0 downto 0);
	signal TEMP_12_mux_2_2 : std_logic_vector(0 downto 0);
	signal TEMP_13_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_13_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_13_mux_3_0 : std_logic_vector(0 downto 0);
	signal TEMP_13_mux_3_1 : std_logic_vector(0 downto 0);
	signal TEMP_13_mux_3_2 : std_logic_vector(0 downto 0);
	signal TEMP_14_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_14_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_14_mux_4_0 : std_logic_vector(0 downto 0);
	signal TEMP_14_mux_4_1 : std_logic_vector(0 downto 0);
	signal TEMP_14_mux_4_2 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_15_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_15_mux_5_0 : std_logic_vector(0 downto 0);
	signal TEMP_15_mux_5_1 : std_logic_vector(0 downto 0);
	signal TEMP_15_mux_5_2 : std_logic_vector(0 downto 0);
	signal TEMP_16_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_16_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_16_mux_6_0 : std_logic_vector(0 downto 0);
	signal TEMP_16_mux_6_1 : std_logic_vector(0 downto 0);
	signal TEMP_16_mux_6_2 : std_logic_vector(0 downto 0);
	signal TEMP_17_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_17_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_17_mux_7_0 : std_logic_vector(0 downto 0);
	signal TEMP_17_mux_7_1 : std_logic_vector(0 downto 0);
	signal TEMP_17_mux_7_2 : std_logic_vector(0 downto 0);
	signal TEMP_18_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_18_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_19_mux_0_0 : std_logic_vector(7 downto 0);
	signal TEMP_19_mux_0_1 : std_logic_vector(7 downto 0);
	signal TEMP_19_mux_0_2 : std_logic_vector(7 downto 0);
	signal TEMP_20_res_0 : std_logic_vector(7 downto 0);
	signal TEMP_20_res_1 : std_logic_vector(7 downto 0);
	signal TEMP_20_mux_1_0 : std_logic_vector(7 downto 0);
	signal TEMP_20_mux_1_1 : std_logic_vector(7 downto 0);
	signal TEMP_20_mux_1_2 : std_logic_vector(7 downto 0);
	signal TEMP_21_res_0 : std_logic_vector(7 downto 0);
	signal TEMP_21_res_1 : std_logic_vector(7 downto 0);
	signal TEMP_21_mux_2_0 : std_logic_vector(7 downto 0);
	signal TEMP_21_mux_2_1 : std_logic_vector(7 downto 0);
	signal TEMP_21_mux_2_2 : std_logic_vector(7 downto 0);
	signal TEMP_22_res_0 : std_logic_vector(7 downto 0);
	signal TEMP_22_res_1 : std_logic_vector(7 downto 0);
	signal TEMP_22_mux_3_0 : std_logic_vector(7 downto 0);
	signal TEMP_22_mux_3_1 : std_logic_vector(7 downto 0);
	signal TEMP_22_mux_3_2 : std_logic_vector(7 downto 0);
	signal TEMP_23_res_0 : std_logic_vector(7 downto 0);
	signal TEMP_23_res_1 : std_logic_vector(7 downto 0);
	signal TEMP_23_mux_4_0 : std_logic_vector(7 downto 0);
	signal TEMP_23_mux_4_1 : std_logic_vector(7 downto 0);
	signal TEMP_23_mux_4_2 : std_logic_vector(7 downto 0);
	signal TEMP_24_res_0 : std_logic_vector(7 downto 0);
	signal TEMP_24_res_1 : std_logic_vector(7 downto 0);
	signal TEMP_24_mux_5_0 : std_logic_vector(7 downto 0);
	signal TEMP_24_mux_5_1 : std_logic_vector(7 downto 0);
	signal TEMP_24_mux_5_2 : std_logic_vector(7 downto 0);
	signal TEMP_25_res_0 : std_logic_vector(7 downto 0);
	signal TEMP_25_res_1 : std_logic_vector(7 downto 0);
	signal TEMP_25_mux_6_0 : std_logic_vector(7 downto 0);
	signal TEMP_25_mux_6_1 : std_logic_vector(7 downto 0);
	signal TEMP_25_mux_6_2 : std_logic_vector(7 downto 0);
	signal TEMP_26_res_0 : std_logic_vector(7 downto 0);
	signal TEMP_26_res_1 : std_logic_vector(7 downto 0);
	signal TEMP_26_mux_7_0 : std_logic_vector(7 downto 0);
	signal TEMP_26_mux_7_1 : std_logic_vector(7 downto 0);
	signal TEMP_26_mux_7_2 : std_logic_vector(7 downto 0);
	signal TEMP_27_res_0 : std_logic_vector(7 downto 0);
	signal TEMP_27_res_1 : std_logic_vector(7 downto 0);
	signal TEMP_28_mux_0 : std_logic_vector(2 downto 0);
	signal TEMP_28_mux_1 : std_logic_vector(2 downto 0);
	signal TEMP_28_mux_2 : std_logic_vector(2 downto 0);
	signal TEMP_29_res_0 : std_logic_vector(2 downto 0);
	signal TEMP_29_res_1 : std_logic_vector(2 downto 0);
	signal TEMP_30_mux_0 : std_logic_vector(2 downto 0);
	signal TEMP_30_mux_1 : std_logic_vector(2 downto 0);
	signal TEMP_30_mux_2 : std_logic_vector(2 downto 0);
	signal TEMP_31_res_0 : std_logic_vector(2 downto 0);
	signal TEMP_31_res_1 : std_logic_vector(2 downto 0);
	signal ldq_wen_unshifted_0 : std_logic;
	signal ldq_wen_unshifted_1 : std_logic;
	signal ldq_wen_unshifted_2 : std_logic;
	signal ldq_wen_unshifted_3 : std_logic;
	signal ldq_wen_unshifted_4 : std_logic;
	signal ldq_wen_unshifted_5 : std_logic;
	signal ldq_wen_unshifted_6 : std_logic;
	signal ldq_wen_unshifted_7 : std_logic;
	signal stq_wen_unshifted_0 : std_logic;
	signal stq_wen_unshifted_1 : std_logic;
	signal stq_wen_unshifted_2 : std_logic;
	signal stq_wen_unshifted_3 : std_logic;
	signal stq_wen_unshifted_4 : std_logic;
	signal stq_wen_unshifted_5 : std_logic;
	signal stq_wen_unshifted_6 : std_logic;
	signal stq_wen_unshifted_7 : std_logic;
	signal TEMP_32_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_32_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_32_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_32_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_32_res_4 : std_logic_vector(0 downto 0);
	signal TEMP_32_res_5 : std_logic_vector(0 downto 0);
	signal TEMP_32_res_6 : std_logic_vector(0 downto 0);
	signal TEMP_32_res_7 : std_logic_vector(0 downto 0);
	signal TEMP_33_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_33_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_33_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_33_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_33_res_4 : std_logic_vector(0 downto 0);
	signal TEMP_33_res_5 : std_logic_vector(0 downto 0);
	signal TEMP_33_res_6 : std_logic_vector(0 downto 0);
	signal TEMP_33_res_7 : std_logic_vector(0 downto 0);
	signal TEMP_34_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_34_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_34_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_34_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_34_res_4 : std_logic_vector(0 downto 0);
	signal TEMP_34_res_5 : std_logic_vector(0 downto 0);
	signal TEMP_34_res_6 : std_logic_vector(0 downto 0);
	signal TEMP_34_res_7 : std_logic_vector(0 downto 0);
	signal TEMP_35_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_35_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_35_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_35_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_35_res_4 : std_logic_vector(0 downto 0);
	signal TEMP_35_res_5 : std_logic_vector(0 downto 0);
	signal TEMP_35_res_6 : std_logic_vector(0 downto 0);
	signal TEMP_35_res_7 : std_logic_vector(0 downto 0);
	signal TEMP_36_res_0 : std_logic;
	signal TEMP_36_res_1 : std_logic;
	signal TEMP_36_res_2 : std_logic;
	signal TEMP_36_res_3 : std_logic;
	signal TEMP_36_res_4 : std_logic;
	signal TEMP_36_res_5 : std_logic;
	signal TEMP_36_res_6 : std_logic;
	signal TEMP_36_res_7 : std_logic;
	signal TEMP_37_res_0 : std_logic;
	signal TEMP_37_res_1 : std_logic;
	signal TEMP_37_res_2 : std_logic;
	signal TEMP_37_res_3 : std_logic;
	signal TEMP_37_res_4 : std_logic;
	signal TEMP_37_res_5 : std_logic;
	signal TEMP_37_res_6 : std_logic;
	signal TEMP_37_res_7 : std_logic;
	signal TEMP_38_res_0 : std_logic;
	signal TEMP_38_res_1 : std_logic;
	signal TEMP_38_res_2 : std_logic;
	signal TEMP_38_res_3 : std_logic;
	signal TEMP_38_res_4 : std_logic;
	signal TEMP_38_res_5 : std_logic;
	signal TEMP_38_res_6 : std_logic;
	signal TEMP_38_res_7 : std_logic;
	signal TEMP_39_res_0 : std_logic;
	signal TEMP_39_res_1 : std_logic;
	signal TEMP_39_res_2 : std_logic;
	signal TEMP_39_res_3 : std_logic;
	signal TEMP_39_res_4 : std_logic;
	signal TEMP_39_res_5 : std_logic;
	signal TEMP_39_res_6 : std_logic;
	signal TEMP_39_res_7 : std_logic;
	signal TEMP_40_res : std_logic_vector(7 downto 0);
	signal TEMP_41_res : std_logic_vector(7 downto 0);
	signal TEMP_42_res : std_logic_vector(7 downto 0);
	signal TEMP_43_res : std_logic_vector(7 downto 0);
	signal TEMP_44_res : std_logic_vector(7 downto 0);
	signal TEMP_45_res : std_logic_vector(7 downto 0);
	signal TEMP_46_res : std_logic_vector(7 downto 0);
	signal TEMP_47_res : std_logic_vector(7 downto 0);
	signal TEMP_48_res : std_logic_vector(7 downto 0);
	signal TEMP_49_res : std_logic_vector(7 downto 0);
	signal TEMP_50_res : std_logic_vector(7 downto 0);
	signal TEMP_51_res : std_logic_vector(7 downto 0);
	signal TEMP_52_res : std_logic_vector(7 downto 0);
	signal TEMP_53_res : std_logic_vector(7 downto 0);
	signal TEMP_54_res : std_logic_vector(7 downto 0);
	signal TEMP_55_res : std_logic_vector(7 downto 0);
	signal TEMP_56_res_0 : std_logic_vector(7 downto 0);
	signal TEMP_56_res_1 : std_logic_vector(7 downto 0);
	signal TEMP_56_res_2 : std_logic_vector(7 downto 0);
	signal TEMP_56_res_3 : std_logic_vector(7 downto 0);
	signal TEMP_56_res_4 : std_logic_vector(7 downto 0);
	signal TEMP_56_res_5 : std_logic_vector(7 downto 0);
	signal TEMP_56_res_6 : std_logic_vector(7 downto 0);
	signal TEMP_56_res_7 : std_logic_vector(7 downto 0);
	signal TEMP_57_res_0 : std_logic_vector(7 downto 0);
	signal TEMP_57_res_1 : std_logic_vector(7 downto 0);
	signal TEMP_57_res_2 : std_logic_vector(7 downto 0);
	signal TEMP_57_res_3 : std_logic_vector(7 downto 0);
	signal TEMP_57_res_4 : std_logic_vector(7 downto 0);
	signal TEMP_57_res_5 : std_logic_vector(7 downto 0);
	signal TEMP_57_res_6 : std_logic_vector(7 downto 0);
	signal TEMP_57_res_7 : std_logic_vector(7 downto 0);
begin
	-- WrapSub Begin
	-- WrapSub(loads_sub, ldq_head, ldq_tail, 8)
	loads_sub <= std_logic_vector(unsigned(ldq_head_i) - unsigned(ldq_tail_i));
	-- WrapAdd End

	-- WrapSub Begin
	-- WrapSub(stores_sub, stq_head, stq_tail, 8)
	stores_sub <= std_logic_vector(unsigned(stq_head_i) - unsigned(stq_tail_i));
	-- WrapAdd End

	empty_loads <= "1000" when ldq_empty_i else ( '0' & loads_sub );
	empty_stores <= "1000" when stq_empty_i else ( '0' & stores_sub );
	group_init_ready_0 <= '1' when ( empty_loads >= "0000" ) and ( empty_stores >= "0001" ) else '0';
	group_init_ready_1 <= '1' when ( empty_loads >= "0001" ) and ( empty_stores >= "0001" ) else '0';
	group_init_ready_2 <= '1' when ( empty_loads >= "0001" ) and ( empty_stores >= "0000" ) else '0';
	group_init_ready_0_o <= group_init_ready_0;
	group_init_hs_0 <= group_init_ready_0 and group_init_valid_0_i;
	group_init_ready_1_o <= group_init_ready_1;
	group_init_hs_1 <= group_init_ready_1 and group_init_valid_1_i;
	group_init_ready_2_o <= group_init_ready_2;
	group_init_hs_2 <= group_init_ready_2 and group_init_valid_2_i;
	-- Mux1H For Rom Begin
	-- Mux1H(ldq_port_idx_rom, group_init_hs)
	-- Loop 0
	TEMP_1_mux_0_0 <= "0";
	TEMP_1_mux_0_1 <= "0";
	TEMP_1_mux_0_2 <= "1" when group_init_hs_2 else "0";
	TEMP_2_res_0 <= TEMP_1_mux_0_0 or TEMP_1_mux_0_2;
	TEMP_2_res_1 <= TEMP_1_mux_0_1;
	-- Layer End
	ldq_port_idx_rom_0 <= TEMP_2_res_0 or TEMP_2_res_1;
	-- Loop 1
	TEMP_2_mux_1_0 <= "0";
	TEMP_2_mux_1_1 <= "0";
	TEMP_2_mux_1_2 <= "0";
	TEMP_3_res_0 <= TEMP_2_mux_1_0 or TEMP_2_mux_1_2;
	TEMP_3_res_1 <= TEMP_2_mux_1_1;
	-- Layer End
	ldq_port_idx_rom_1 <= TEMP_3_res_0 or TEMP_3_res_1;
	-- Loop 2
	TEMP_3_mux_2_0 <= "0";
	TEMP_3_mux_2_1 <= "0";
	TEMP_3_mux_2_2 <= "0";
	TEMP_4_res_0 <= TEMP_3_mux_2_0 or TEMP_3_mux_2_2;
	TEMP_4_res_1 <= TEMP_3_mux_2_1;
	-- Layer End
	ldq_port_idx_rom_2 <= TEMP_4_res_0 or TEMP_4_res_1;
	-- Loop 3
	TEMP_4_mux_3_0 <= "0";
	TEMP_4_mux_3_1 <= "0";
	TEMP_4_mux_3_2 <= "0";
	TEMP_5_res_0 <= TEMP_4_mux_3_0 or TEMP_4_mux_3_2;
	TEMP_5_res_1 <= TEMP_4_mux_3_1;
	-- Layer End
	ldq_port_idx_rom_3 <= TEMP_5_res_0 or TEMP_5_res_1;
	-- Loop 4
	TEMP_5_mux_4_0 <= "0";
	TEMP_5_mux_4_1 <= "0";
	TEMP_5_mux_4_2 <= "0";
	TEMP_6_res_0 <= TEMP_5_mux_4_0 or TEMP_5_mux_4_2;
	TEMP_6_res_1 <= TEMP_5_mux_4_1;
	-- Layer End
	ldq_port_idx_rom_4 <= TEMP_6_res_0 or TEMP_6_res_1;
	-- Loop 5
	TEMP_6_mux_5_0 <= "0";
	TEMP_6_mux_5_1 <= "0";
	TEMP_6_mux_5_2 <= "0";
	TEMP_7_res_0 <= TEMP_6_mux_5_0 or TEMP_6_mux_5_2;
	TEMP_7_res_1 <= TEMP_6_mux_5_1;
	-- Layer End
	ldq_port_idx_rom_5 <= TEMP_7_res_0 or TEMP_7_res_1;
	-- Loop 6
	TEMP_7_mux_6_0 <= "0";
	TEMP_7_mux_6_1 <= "0";
	TEMP_7_mux_6_2 <= "0";
	TEMP_8_res_0 <= TEMP_7_mux_6_0 or TEMP_7_mux_6_2;
	TEMP_8_res_1 <= TEMP_7_mux_6_1;
	-- Layer End
	ldq_port_idx_rom_6 <= TEMP_8_res_0 or TEMP_8_res_1;
	-- Loop 7
	TEMP_8_mux_7_0 <= "0";
	TEMP_8_mux_7_1 <= "0";
	TEMP_8_mux_7_2 <= "0";
	TEMP_9_res_0 <= TEMP_8_mux_7_0 or TEMP_8_mux_7_2;
	TEMP_9_res_1 <= TEMP_8_mux_7_1;
	-- Layer End
	ldq_port_idx_rom_7 <= TEMP_9_res_0 or TEMP_9_res_1;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(stq_port_idx_rom, group_init_hs)
	-- Loop 0
	TEMP_10_mux_0_0 <= "0";
	TEMP_10_mux_0_1 <= "1" when group_init_hs_1 else "0";
	TEMP_10_mux_0_2 <= "0";
	TEMP_11_res_0 <= TEMP_10_mux_0_0 or TEMP_10_mux_0_2;
	TEMP_11_res_1 <= TEMP_10_mux_0_1;
	-- Layer End
	stq_port_idx_rom_0 <= TEMP_11_res_0 or TEMP_11_res_1;
	-- Loop 1
	TEMP_11_mux_1_0 <= "0";
	TEMP_11_mux_1_1 <= "0";
	TEMP_11_mux_1_2 <= "0";
	TEMP_12_res_0 <= TEMP_11_mux_1_0 or TEMP_11_mux_1_2;
	TEMP_12_res_1 <= TEMP_11_mux_1_1;
	-- Layer End
	stq_port_idx_rom_1 <= TEMP_12_res_0 or TEMP_12_res_1;
	-- Loop 2
	TEMP_12_mux_2_0 <= "0";
	TEMP_12_mux_2_1 <= "0";
	TEMP_12_mux_2_2 <= "0";
	TEMP_13_res_0 <= TEMP_12_mux_2_0 or TEMP_12_mux_2_2;
	TEMP_13_res_1 <= TEMP_12_mux_2_1;
	-- Layer End
	stq_port_idx_rom_2 <= TEMP_13_res_0 or TEMP_13_res_1;
	-- Loop 3
	TEMP_13_mux_3_0 <= "0";
	TEMP_13_mux_3_1 <= "0";
	TEMP_13_mux_3_2 <= "0";
	TEMP_14_res_0 <= TEMP_13_mux_3_0 or TEMP_13_mux_3_2;
	TEMP_14_res_1 <= TEMP_13_mux_3_1;
	-- Layer End
	stq_port_idx_rom_3 <= TEMP_14_res_0 or TEMP_14_res_1;
	-- Loop 4
	TEMP_14_mux_4_0 <= "0";
	TEMP_14_mux_4_1 <= "0";
	TEMP_14_mux_4_2 <= "0";
	TEMP_15_res_0 <= TEMP_14_mux_4_0 or TEMP_14_mux_4_2;
	TEMP_15_res_1 <= TEMP_14_mux_4_1;
	-- Layer End
	stq_port_idx_rom_4 <= TEMP_15_res_0 or TEMP_15_res_1;
	-- Loop 5
	TEMP_15_mux_5_0 <= "0";
	TEMP_15_mux_5_1 <= "0";
	TEMP_15_mux_5_2 <= "0";
	TEMP_16_res_0 <= TEMP_15_mux_5_0 or TEMP_15_mux_5_2;
	TEMP_16_res_1 <= TEMP_15_mux_5_1;
	-- Layer End
	stq_port_idx_rom_5 <= TEMP_16_res_0 or TEMP_16_res_1;
	-- Loop 6
	TEMP_16_mux_6_0 <= "0";
	TEMP_16_mux_6_1 <= "0";
	TEMP_16_mux_6_2 <= "0";
	TEMP_17_res_0 <= TEMP_16_mux_6_0 or TEMP_16_mux_6_2;
	TEMP_17_res_1 <= TEMP_16_mux_6_1;
	-- Layer End
	stq_port_idx_rom_6 <= TEMP_17_res_0 or TEMP_17_res_1;
	-- Loop 7
	TEMP_17_mux_7_0 <= "0";
	TEMP_17_mux_7_1 <= "0";
	TEMP_17_mux_7_2 <= "0";
	TEMP_18_res_0 <= TEMP_17_mux_7_0 or TEMP_17_mux_7_2;
	TEMP_18_res_1 <= TEMP_17_mux_7_1;
	-- Layer End
	stq_port_idx_rom_7 <= TEMP_18_res_0 or TEMP_18_res_1;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(ga_ls_order_rom, group_init_hs)
	-- Loop 0
	TEMP_19_mux_0_0 <= "00000000";
	TEMP_19_mux_0_1 <= "00000000";
	TEMP_19_mux_0_2 <= "00000000";
	TEMP_20_res_0 <= TEMP_19_mux_0_0 or TEMP_19_mux_0_2;
	TEMP_20_res_1 <= TEMP_19_mux_0_1;
	-- Layer End
	ga_ls_order_rom_0 <= TEMP_20_res_0 or TEMP_20_res_1;
	-- Loop 1
	TEMP_20_mux_1_0 <= "00000000";
	TEMP_20_mux_1_1 <= "00000000";
	TEMP_20_mux_1_2 <= "00000000";
	TEMP_21_res_0 <= TEMP_20_mux_1_0 or TEMP_20_mux_1_2;
	TEMP_21_res_1 <= TEMP_20_mux_1_1;
	-- Layer End
	ga_ls_order_rom_1 <= TEMP_21_res_0 or TEMP_21_res_1;
	-- Loop 2
	TEMP_21_mux_2_0 <= "00000000";
	TEMP_21_mux_2_1 <= "00000000";
	TEMP_21_mux_2_2 <= "00000000";
	TEMP_22_res_0 <= TEMP_21_mux_2_0 or TEMP_21_mux_2_2;
	TEMP_22_res_1 <= TEMP_21_mux_2_1;
	-- Layer End
	ga_ls_order_rom_2 <= TEMP_22_res_0 or TEMP_22_res_1;
	-- Loop 3
	TEMP_22_mux_3_0 <= "00000000";
	TEMP_22_mux_3_1 <= "00000000";
	TEMP_22_mux_3_2 <= "00000000";
	TEMP_23_res_0 <= TEMP_22_mux_3_0 or TEMP_22_mux_3_2;
	TEMP_23_res_1 <= TEMP_22_mux_3_1;
	-- Layer End
	ga_ls_order_rom_3 <= TEMP_23_res_0 or TEMP_23_res_1;
	-- Loop 4
	TEMP_23_mux_4_0 <= "00000000";
	TEMP_23_mux_4_1 <= "00000000";
	TEMP_23_mux_4_2 <= "00000000";
	TEMP_24_res_0 <= TEMP_23_mux_4_0 or TEMP_23_mux_4_2;
	TEMP_24_res_1 <= TEMP_23_mux_4_1;
	-- Layer End
	ga_ls_order_rom_4 <= TEMP_24_res_0 or TEMP_24_res_1;
	-- Loop 5
	TEMP_24_mux_5_0 <= "00000000";
	TEMP_24_mux_5_1 <= "00000000";
	TEMP_24_mux_5_2 <= "00000000";
	TEMP_25_res_0 <= TEMP_24_mux_5_0 or TEMP_24_mux_5_2;
	TEMP_25_res_1 <= TEMP_24_mux_5_1;
	-- Layer End
	ga_ls_order_rom_5 <= TEMP_25_res_0 or TEMP_25_res_1;
	-- Loop 6
	TEMP_25_mux_6_0 <= "00000000";
	TEMP_25_mux_6_1 <= "00000000";
	TEMP_25_mux_6_2 <= "00000000";
	TEMP_26_res_0 <= TEMP_25_mux_6_0 or TEMP_25_mux_6_2;
	TEMP_26_res_1 <= TEMP_25_mux_6_1;
	-- Layer End
	ga_ls_order_rom_6 <= TEMP_26_res_0 or TEMP_26_res_1;
	-- Loop 7
	TEMP_26_mux_7_0 <= "00000000";
	TEMP_26_mux_7_1 <= "00000000";
	TEMP_26_mux_7_2 <= "00000000";
	TEMP_27_res_0 <= TEMP_26_mux_7_0 or TEMP_26_mux_7_2;
	TEMP_27_res_1 <= TEMP_26_mux_7_1;
	-- Layer End
	ga_ls_order_rom_7 <= TEMP_27_res_0 or TEMP_27_res_1;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(num_loads, group_init_hs)
	TEMP_28_mux_0 <= "000";
	TEMP_28_mux_1 <= "001" when group_init_hs_1 else "000";
	TEMP_28_mux_2 <= "001" when group_init_hs_2 else "000";
	TEMP_29_res_0 <= TEMP_28_mux_0 or TEMP_28_mux_2;
	TEMP_29_res_1 <= TEMP_28_mux_1;
	-- Layer End
	num_loads <= TEMP_29_res_0 or TEMP_29_res_1;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(num_stores, group_init_hs)
	TEMP_30_mux_0 <= "001" when group_init_hs_0 else "000";
	TEMP_30_mux_1 <= "001" when group_init_hs_1 else "000";
	TEMP_30_mux_2 <= "000";
	TEMP_31_res_0 <= TEMP_30_mux_0 or TEMP_30_mux_2;
	TEMP_31_res_1 <= TEMP_30_mux_1;
	-- Layer End
	num_stores <= TEMP_31_res_0 or TEMP_31_res_1;
	-- Mux1H For Rom End

	num_loads_o <= num_loads;
	num_stores_o <= num_stores;
	ldq_wen_unshifted_0 <= '1' when num_loads > "000" else '0';
	ldq_wen_unshifted_1 <= '1' when num_loads > "001" else '0';
	ldq_wen_unshifted_2 <= '1' when num_loads > "010" else '0';
	ldq_wen_unshifted_3 <= '1' when num_loads > "011" else '0';
	ldq_wen_unshifted_4 <= '1' when num_loads > "100" else '0';
	ldq_wen_unshifted_5 <= '1' when num_loads > "101" else '0';
	ldq_wen_unshifted_6 <= '1' when num_loads > "110" else '0';
	ldq_wen_unshifted_7 <= '1' when num_loads > "111" else '0';
	stq_wen_unshifted_0 <= '1' when num_stores > "000" else '0';
	stq_wen_unshifted_1 <= '1' when num_stores > "001" else '0';
	stq_wen_unshifted_2 <= '1' when num_stores > "010" else '0';
	stq_wen_unshifted_3 <= '1' when num_stores > "011" else '0';
	stq_wen_unshifted_4 <= '1' when num_stores > "100" else '0';
	stq_wen_unshifted_5 <= '1' when num_stores > "101" else '0';
	stq_wen_unshifted_6 <= '1' when num_stores > "110" else '0';
	stq_wen_unshifted_7 <= '1' when num_stores > "111" else '0';
	-- Shifter Begin
	-- CyclicLeftShift(ldq_port_idx, ldq_port_idx_rom, ldq_tail)
	TEMP_32_res_0 <= ldq_port_idx_rom_4 when ldq_tail_i(2) else ldq_port_idx_rom_0;
	TEMP_32_res_1 <= ldq_port_idx_rom_5 when ldq_tail_i(2) else ldq_port_idx_rom_1;
	TEMP_32_res_2 <= ldq_port_idx_rom_6 when ldq_tail_i(2) else ldq_port_idx_rom_2;
	TEMP_32_res_3 <= ldq_port_idx_rom_7 when ldq_tail_i(2) else ldq_port_idx_rom_3;
	TEMP_32_res_4 <= ldq_port_idx_rom_0 when ldq_tail_i(2) else ldq_port_idx_rom_4;
	TEMP_32_res_5 <= ldq_port_idx_rom_1 when ldq_tail_i(2) else ldq_port_idx_rom_5;
	TEMP_32_res_6 <= ldq_port_idx_rom_2 when ldq_tail_i(2) else ldq_port_idx_rom_6;
	TEMP_32_res_7 <= ldq_port_idx_rom_3 when ldq_tail_i(2) else ldq_port_idx_rom_7;
	-- Layer End
	TEMP_33_res_0 <= TEMP_32_res_6 when ldq_tail_i(1) else TEMP_32_res_0;
	TEMP_33_res_1 <= TEMP_32_res_7 when ldq_tail_i(1) else TEMP_32_res_1;
	TEMP_33_res_2 <= TEMP_32_res_0 when ldq_tail_i(1) else TEMP_32_res_2;
	TEMP_33_res_3 <= TEMP_32_res_1 when ldq_tail_i(1) else TEMP_32_res_3;
	TEMP_33_res_4 <= TEMP_32_res_2 when ldq_tail_i(1) else TEMP_32_res_4;
	TEMP_33_res_5 <= TEMP_32_res_3 when ldq_tail_i(1) else TEMP_32_res_5;
	TEMP_33_res_6 <= TEMP_32_res_4 when ldq_tail_i(1) else TEMP_32_res_6;
	TEMP_33_res_7 <= TEMP_32_res_5 when ldq_tail_i(1) else TEMP_32_res_7;
	-- Layer End
	ldq_port_idx_0_o <= TEMP_33_res_7 when ldq_tail_i(0) else TEMP_33_res_0;
	ldq_port_idx_1_o <= TEMP_33_res_0 when ldq_tail_i(0) else TEMP_33_res_1;
	ldq_port_idx_2_o <= TEMP_33_res_1 when ldq_tail_i(0) else TEMP_33_res_2;
	ldq_port_idx_3_o <= TEMP_33_res_2 when ldq_tail_i(0) else TEMP_33_res_3;
	ldq_port_idx_4_o <= TEMP_33_res_3 when ldq_tail_i(0) else TEMP_33_res_4;
	ldq_port_idx_5_o <= TEMP_33_res_4 when ldq_tail_i(0) else TEMP_33_res_5;
	ldq_port_idx_6_o <= TEMP_33_res_5 when ldq_tail_i(0) else TEMP_33_res_6;
	ldq_port_idx_7_o <= TEMP_33_res_6 when ldq_tail_i(0) else TEMP_33_res_7;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(stq_port_idx, stq_port_idx_rom, stq_tail)
	TEMP_34_res_0 <= stq_port_idx_rom_4 when stq_tail_i(2) else stq_port_idx_rom_0;
	TEMP_34_res_1 <= stq_port_idx_rom_5 when stq_tail_i(2) else stq_port_idx_rom_1;
	TEMP_34_res_2 <= stq_port_idx_rom_6 when stq_tail_i(2) else stq_port_idx_rom_2;
	TEMP_34_res_3 <= stq_port_idx_rom_7 when stq_tail_i(2) else stq_port_idx_rom_3;
	TEMP_34_res_4 <= stq_port_idx_rom_0 when stq_tail_i(2) else stq_port_idx_rom_4;
	TEMP_34_res_5 <= stq_port_idx_rom_1 when stq_tail_i(2) else stq_port_idx_rom_5;
	TEMP_34_res_6 <= stq_port_idx_rom_2 when stq_tail_i(2) else stq_port_idx_rom_6;
	TEMP_34_res_7 <= stq_port_idx_rom_3 when stq_tail_i(2) else stq_port_idx_rom_7;
	-- Layer End
	TEMP_35_res_0 <= TEMP_34_res_6 when stq_tail_i(1) else TEMP_34_res_0;
	TEMP_35_res_1 <= TEMP_34_res_7 when stq_tail_i(1) else TEMP_34_res_1;
	TEMP_35_res_2 <= TEMP_34_res_0 when stq_tail_i(1) else TEMP_34_res_2;
	TEMP_35_res_3 <= TEMP_34_res_1 when stq_tail_i(1) else TEMP_34_res_3;
	TEMP_35_res_4 <= TEMP_34_res_2 when stq_tail_i(1) else TEMP_34_res_4;
	TEMP_35_res_5 <= TEMP_34_res_3 when stq_tail_i(1) else TEMP_34_res_5;
	TEMP_35_res_6 <= TEMP_34_res_4 when stq_tail_i(1) else TEMP_34_res_6;
	TEMP_35_res_7 <= TEMP_34_res_5 when stq_tail_i(1) else TEMP_34_res_7;
	-- Layer End
	stq_port_idx_0_o <= TEMP_35_res_7 when stq_tail_i(0) else TEMP_35_res_0;
	stq_port_idx_1_o <= TEMP_35_res_0 when stq_tail_i(0) else TEMP_35_res_1;
	stq_port_idx_2_o <= TEMP_35_res_1 when stq_tail_i(0) else TEMP_35_res_2;
	stq_port_idx_3_o <= TEMP_35_res_2 when stq_tail_i(0) else TEMP_35_res_3;
	stq_port_idx_4_o <= TEMP_35_res_3 when stq_tail_i(0) else TEMP_35_res_4;
	stq_port_idx_5_o <= TEMP_35_res_4 when stq_tail_i(0) else TEMP_35_res_5;
	stq_port_idx_6_o <= TEMP_35_res_5 when stq_tail_i(0) else TEMP_35_res_6;
	stq_port_idx_7_o <= TEMP_35_res_6 when stq_tail_i(0) else TEMP_35_res_7;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ldq_wen, ldq_wen_unshifted, ldq_tail)
	TEMP_36_res_0 <= ldq_wen_unshifted_4 when ldq_tail_i(2) else ldq_wen_unshifted_0;
	TEMP_36_res_1 <= ldq_wen_unshifted_5 when ldq_tail_i(2) else ldq_wen_unshifted_1;
	TEMP_36_res_2 <= ldq_wen_unshifted_6 when ldq_tail_i(2) else ldq_wen_unshifted_2;
	TEMP_36_res_3 <= ldq_wen_unshifted_7 when ldq_tail_i(2) else ldq_wen_unshifted_3;
	TEMP_36_res_4 <= ldq_wen_unshifted_0 when ldq_tail_i(2) else ldq_wen_unshifted_4;
	TEMP_36_res_5 <= ldq_wen_unshifted_1 when ldq_tail_i(2) else ldq_wen_unshifted_5;
	TEMP_36_res_6 <= ldq_wen_unshifted_2 when ldq_tail_i(2) else ldq_wen_unshifted_6;
	TEMP_36_res_7 <= ldq_wen_unshifted_3 when ldq_tail_i(2) else ldq_wen_unshifted_7;
	-- Layer End
	TEMP_37_res_0 <= TEMP_36_res_6 when ldq_tail_i(1) else TEMP_36_res_0;
	TEMP_37_res_1 <= TEMP_36_res_7 when ldq_tail_i(1) else TEMP_36_res_1;
	TEMP_37_res_2 <= TEMP_36_res_0 when ldq_tail_i(1) else TEMP_36_res_2;
	TEMP_37_res_3 <= TEMP_36_res_1 when ldq_tail_i(1) else TEMP_36_res_3;
	TEMP_37_res_4 <= TEMP_36_res_2 when ldq_tail_i(1) else TEMP_36_res_4;
	TEMP_37_res_5 <= TEMP_36_res_3 when ldq_tail_i(1) else TEMP_36_res_5;
	TEMP_37_res_6 <= TEMP_36_res_4 when ldq_tail_i(1) else TEMP_36_res_6;
	TEMP_37_res_7 <= TEMP_36_res_5 when ldq_tail_i(1) else TEMP_36_res_7;
	-- Layer End
	ldq_wen_0_o <= TEMP_37_res_7 when ldq_tail_i(0) else TEMP_37_res_0;
	ldq_wen_1_o <= TEMP_37_res_0 when ldq_tail_i(0) else TEMP_37_res_1;
	ldq_wen_2_o <= TEMP_37_res_1 when ldq_tail_i(0) else TEMP_37_res_2;
	ldq_wen_3_o <= TEMP_37_res_2 when ldq_tail_i(0) else TEMP_37_res_3;
	ldq_wen_4_o <= TEMP_37_res_3 when ldq_tail_i(0) else TEMP_37_res_4;
	ldq_wen_5_o <= TEMP_37_res_4 when ldq_tail_i(0) else TEMP_37_res_5;
	ldq_wen_6_o <= TEMP_37_res_5 when ldq_tail_i(0) else TEMP_37_res_6;
	ldq_wen_7_o <= TEMP_37_res_6 when ldq_tail_i(0) else TEMP_37_res_7;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(stq_wen, stq_wen_unshifted, stq_tail)
	TEMP_38_res_0 <= stq_wen_unshifted_4 when stq_tail_i(2) else stq_wen_unshifted_0;
	TEMP_38_res_1 <= stq_wen_unshifted_5 when stq_tail_i(2) else stq_wen_unshifted_1;
	TEMP_38_res_2 <= stq_wen_unshifted_6 when stq_tail_i(2) else stq_wen_unshifted_2;
	TEMP_38_res_3 <= stq_wen_unshifted_7 when stq_tail_i(2) else stq_wen_unshifted_3;
	TEMP_38_res_4 <= stq_wen_unshifted_0 when stq_tail_i(2) else stq_wen_unshifted_4;
	TEMP_38_res_5 <= stq_wen_unshifted_1 when stq_tail_i(2) else stq_wen_unshifted_5;
	TEMP_38_res_6 <= stq_wen_unshifted_2 when stq_tail_i(2) else stq_wen_unshifted_6;
	TEMP_38_res_7 <= stq_wen_unshifted_3 when stq_tail_i(2) else stq_wen_unshifted_7;
	-- Layer End
	TEMP_39_res_0 <= TEMP_38_res_6 when stq_tail_i(1) else TEMP_38_res_0;
	TEMP_39_res_1 <= TEMP_38_res_7 when stq_tail_i(1) else TEMP_38_res_1;
	TEMP_39_res_2 <= TEMP_38_res_0 when stq_tail_i(1) else TEMP_38_res_2;
	TEMP_39_res_3 <= TEMP_38_res_1 when stq_tail_i(1) else TEMP_38_res_3;
	TEMP_39_res_4 <= TEMP_38_res_2 when stq_tail_i(1) else TEMP_38_res_4;
	TEMP_39_res_5 <= TEMP_38_res_3 when stq_tail_i(1) else TEMP_38_res_5;
	TEMP_39_res_6 <= TEMP_38_res_4 when stq_tail_i(1) else TEMP_38_res_6;
	TEMP_39_res_7 <= TEMP_38_res_5 when stq_tail_i(1) else TEMP_38_res_7;
	-- Layer End
	stq_wen_0_o <= TEMP_39_res_7 when stq_tail_i(0) else TEMP_39_res_0;
	stq_wen_1_o <= TEMP_39_res_0 when stq_tail_i(0) else TEMP_39_res_1;
	stq_wen_2_o <= TEMP_39_res_1 when stq_tail_i(0) else TEMP_39_res_2;
	stq_wen_3_o <= TEMP_39_res_2 when stq_tail_i(0) else TEMP_39_res_3;
	stq_wen_4_o <= TEMP_39_res_3 when stq_tail_i(0) else TEMP_39_res_4;
	stq_wen_5_o <= TEMP_39_res_4 when stq_tail_i(0) else TEMP_39_res_5;
	stq_wen_6_o <= TEMP_39_res_5 when stq_tail_i(0) else TEMP_39_res_6;
	stq_wen_7_o <= TEMP_39_res_6 when stq_tail_i(0) else TEMP_39_res_7;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_0, ga_ls_order_rom_0, stq_tail)
	TEMP_40_res(0) <= ga_ls_order_rom_0(4) when stq_tail_i(2) else ga_ls_order_rom_0(0);
	TEMP_40_res(1) <= ga_ls_order_rom_0(5) when stq_tail_i(2) else ga_ls_order_rom_0(1);
	TEMP_40_res(2) <= ga_ls_order_rom_0(6) when stq_tail_i(2) else ga_ls_order_rom_0(2);
	TEMP_40_res(3) <= ga_ls_order_rom_0(7) when stq_tail_i(2) else ga_ls_order_rom_0(3);
	TEMP_40_res(4) <= ga_ls_order_rom_0(0) when stq_tail_i(2) else ga_ls_order_rom_0(4);
	TEMP_40_res(5) <= ga_ls_order_rom_0(1) when stq_tail_i(2) else ga_ls_order_rom_0(5);
	TEMP_40_res(6) <= ga_ls_order_rom_0(2) when stq_tail_i(2) else ga_ls_order_rom_0(6);
	TEMP_40_res(7) <= ga_ls_order_rom_0(3) when stq_tail_i(2) else ga_ls_order_rom_0(7);
	-- Layer End
	TEMP_41_res(0) <= TEMP_40_res(6) when stq_tail_i(1) else TEMP_40_res(0);
	TEMP_41_res(1) <= TEMP_40_res(7) when stq_tail_i(1) else TEMP_40_res(1);
	TEMP_41_res(2) <= TEMP_40_res(0) when stq_tail_i(1) else TEMP_40_res(2);
	TEMP_41_res(3) <= TEMP_40_res(1) when stq_tail_i(1) else TEMP_40_res(3);
	TEMP_41_res(4) <= TEMP_40_res(2) when stq_tail_i(1) else TEMP_40_res(4);
	TEMP_41_res(5) <= TEMP_40_res(3) when stq_tail_i(1) else TEMP_40_res(5);
	TEMP_41_res(6) <= TEMP_40_res(4) when stq_tail_i(1) else TEMP_40_res(6);
	TEMP_41_res(7) <= TEMP_40_res(5) when stq_tail_i(1) else TEMP_40_res(7);
	-- Layer End
	ga_ls_order_temp_0(0) <= TEMP_41_res(7) when stq_tail_i(0) else TEMP_41_res(0);
	ga_ls_order_temp_0(1) <= TEMP_41_res(0) when stq_tail_i(0) else TEMP_41_res(1);
	ga_ls_order_temp_0(2) <= TEMP_41_res(1) when stq_tail_i(0) else TEMP_41_res(2);
	ga_ls_order_temp_0(3) <= TEMP_41_res(2) when stq_tail_i(0) else TEMP_41_res(3);
	ga_ls_order_temp_0(4) <= TEMP_41_res(3) when stq_tail_i(0) else TEMP_41_res(4);
	ga_ls_order_temp_0(5) <= TEMP_41_res(4) when stq_tail_i(0) else TEMP_41_res(5);
	ga_ls_order_temp_0(6) <= TEMP_41_res(5) when stq_tail_i(0) else TEMP_41_res(6);
	ga_ls_order_temp_0(7) <= TEMP_41_res(6) when stq_tail_i(0) else TEMP_41_res(7);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_1, ga_ls_order_rom_1, stq_tail)
	TEMP_42_res(0) <= ga_ls_order_rom_1(4) when stq_tail_i(2) else ga_ls_order_rom_1(0);
	TEMP_42_res(1) <= ga_ls_order_rom_1(5) when stq_tail_i(2) else ga_ls_order_rom_1(1);
	TEMP_42_res(2) <= ga_ls_order_rom_1(6) when stq_tail_i(2) else ga_ls_order_rom_1(2);
	TEMP_42_res(3) <= ga_ls_order_rom_1(7) when stq_tail_i(2) else ga_ls_order_rom_1(3);
	TEMP_42_res(4) <= ga_ls_order_rom_1(0) when stq_tail_i(2) else ga_ls_order_rom_1(4);
	TEMP_42_res(5) <= ga_ls_order_rom_1(1) when stq_tail_i(2) else ga_ls_order_rom_1(5);
	TEMP_42_res(6) <= ga_ls_order_rom_1(2) when stq_tail_i(2) else ga_ls_order_rom_1(6);
	TEMP_42_res(7) <= ga_ls_order_rom_1(3) when stq_tail_i(2) else ga_ls_order_rom_1(7);
	-- Layer End
	TEMP_43_res(0) <= TEMP_42_res(6) when stq_tail_i(1) else TEMP_42_res(0);
	TEMP_43_res(1) <= TEMP_42_res(7) when stq_tail_i(1) else TEMP_42_res(1);
	TEMP_43_res(2) <= TEMP_42_res(0) when stq_tail_i(1) else TEMP_42_res(2);
	TEMP_43_res(3) <= TEMP_42_res(1) when stq_tail_i(1) else TEMP_42_res(3);
	TEMP_43_res(4) <= TEMP_42_res(2) when stq_tail_i(1) else TEMP_42_res(4);
	TEMP_43_res(5) <= TEMP_42_res(3) when stq_tail_i(1) else TEMP_42_res(5);
	TEMP_43_res(6) <= TEMP_42_res(4) when stq_tail_i(1) else TEMP_42_res(6);
	TEMP_43_res(7) <= TEMP_42_res(5) when stq_tail_i(1) else TEMP_42_res(7);
	-- Layer End
	ga_ls_order_temp_1(0) <= TEMP_43_res(7) when stq_tail_i(0) else TEMP_43_res(0);
	ga_ls_order_temp_1(1) <= TEMP_43_res(0) when stq_tail_i(0) else TEMP_43_res(1);
	ga_ls_order_temp_1(2) <= TEMP_43_res(1) when stq_tail_i(0) else TEMP_43_res(2);
	ga_ls_order_temp_1(3) <= TEMP_43_res(2) when stq_tail_i(0) else TEMP_43_res(3);
	ga_ls_order_temp_1(4) <= TEMP_43_res(3) when stq_tail_i(0) else TEMP_43_res(4);
	ga_ls_order_temp_1(5) <= TEMP_43_res(4) when stq_tail_i(0) else TEMP_43_res(5);
	ga_ls_order_temp_1(6) <= TEMP_43_res(5) when stq_tail_i(0) else TEMP_43_res(6);
	ga_ls_order_temp_1(7) <= TEMP_43_res(6) when stq_tail_i(0) else TEMP_43_res(7);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_2, ga_ls_order_rom_2, stq_tail)
	TEMP_44_res(0) <= ga_ls_order_rom_2(4) when stq_tail_i(2) else ga_ls_order_rom_2(0);
	TEMP_44_res(1) <= ga_ls_order_rom_2(5) when stq_tail_i(2) else ga_ls_order_rom_2(1);
	TEMP_44_res(2) <= ga_ls_order_rom_2(6) when stq_tail_i(2) else ga_ls_order_rom_2(2);
	TEMP_44_res(3) <= ga_ls_order_rom_2(7) when stq_tail_i(2) else ga_ls_order_rom_2(3);
	TEMP_44_res(4) <= ga_ls_order_rom_2(0) when stq_tail_i(2) else ga_ls_order_rom_2(4);
	TEMP_44_res(5) <= ga_ls_order_rom_2(1) when stq_tail_i(2) else ga_ls_order_rom_2(5);
	TEMP_44_res(6) <= ga_ls_order_rom_2(2) when stq_tail_i(2) else ga_ls_order_rom_2(6);
	TEMP_44_res(7) <= ga_ls_order_rom_2(3) when stq_tail_i(2) else ga_ls_order_rom_2(7);
	-- Layer End
	TEMP_45_res(0) <= TEMP_44_res(6) when stq_tail_i(1) else TEMP_44_res(0);
	TEMP_45_res(1) <= TEMP_44_res(7) when stq_tail_i(1) else TEMP_44_res(1);
	TEMP_45_res(2) <= TEMP_44_res(0) when stq_tail_i(1) else TEMP_44_res(2);
	TEMP_45_res(3) <= TEMP_44_res(1) when stq_tail_i(1) else TEMP_44_res(3);
	TEMP_45_res(4) <= TEMP_44_res(2) when stq_tail_i(1) else TEMP_44_res(4);
	TEMP_45_res(5) <= TEMP_44_res(3) when stq_tail_i(1) else TEMP_44_res(5);
	TEMP_45_res(6) <= TEMP_44_res(4) when stq_tail_i(1) else TEMP_44_res(6);
	TEMP_45_res(7) <= TEMP_44_res(5) when stq_tail_i(1) else TEMP_44_res(7);
	-- Layer End
	ga_ls_order_temp_2(0) <= TEMP_45_res(7) when stq_tail_i(0) else TEMP_45_res(0);
	ga_ls_order_temp_2(1) <= TEMP_45_res(0) when stq_tail_i(0) else TEMP_45_res(1);
	ga_ls_order_temp_2(2) <= TEMP_45_res(1) when stq_tail_i(0) else TEMP_45_res(2);
	ga_ls_order_temp_2(3) <= TEMP_45_res(2) when stq_tail_i(0) else TEMP_45_res(3);
	ga_ls_order_temp_2(4) <= TEMP_45_res(3) when stq_tail_i(0) else TEMP_45_res(4);
	ga_ls_order_temp_2(5) <= TEMP_45_res(4) when stq_tail_i(0) else TEMP_45_res(5);
	ga_ls_order_temp_2(6) <= TEMP_45_res(5) when stq_tail_i(0) else TEMP_45_res(6);
	ga_ls_order_temp_2(7) <= TEMP_45_res(6) when stq_tail_i(0) else TEMP_45_res(7);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_3, ga_ls_order_rom_3, stq_tail)
	TEMP_46_res(0) <= ga_ls_order_rom_3(4) when stq_tail_i(2) else ga_ls_order_rom_3(0);
	TEMP_46_res(1) <= ga_ls_order_rom_3(5) when stq_tail_i(2) else ga_ls_order_rom_3(1);
	TEMP_46_res(2) <= ga_ls_order_rom_3(6) when stq_tail_i(2) else ga_ls_order_rom_3(2);
	TEMP_46_res(3) <= ga_ls_order_rom_3(7) when stq_tail_i(2) else ga_ls_order_rom_3(3);
	TEMP_46_res(4) <= ga_ls_order_rom_3(0) when stq_tail_i(2) else ga_ls_order_rom_3(4);
	TEMP_46_res(5) <= ga_ls_order_rom_3(1) when stq_tail_i(2) else ga_ls_order_rom_3(5);
	TEMP_46_res(6) <= ga_ls_order_rom_3(2) when stq_tail_i(2) else ga_ls_order_rom_3(6);
	TEMP_46_res(7) <= ga_ls_order_rom_3(3) when stq_tail_i(2) else ga_ls_order_rom_3(7);
	-- Layer End
	TEMP_47_res(0) <= TEMP_46_res(6) when stq_tail_i(1) else TEMP_46_res(0);
	TEMP_47_res(1) <= TEMP_46_res(7) when stq_tail_i(1) else TEMP_46_res(1);
	TEMP_47_res(2) <= TEMP_46_res(0) when stq_tail_i(1) else TEMP_46_res(2);
	TEMP_47_res(3) <= TEMP_46_res(1) when stq_tail_i(1) else TEMP_46_res(3);
	TEMP_47_res(4) <= TEMP_46_res(2) when stq_tail_i(1) else TEMP_46_res(4);
	TEMP_47_res(5) <= TEMP_46_res(3) when stq_tail_i(1) else TEMP_46_res(5);
	TEMP_47_res(6) <= TEMP_46_res(4) when stq_tail_i(1) else TEMP_46_res(6);
	TEMP_47_res(7) <= TEMP_46_res(5) when stq_tail_i(1) else TEMP_46_res(7);
	-- Layer End
	ga_ls_order_temp_3(0) <= TEMP_47_res(7) when stq_tail_i(0) else TEMP_47_res(0);
	ga_ls_order_temp_3(1) <= TEMP_47_res(0) when stq_tail_i(0) else TEMP_47_res(1);
	ga_ls_order_temp_3(2) <= TEMP_47_res(1) when stq_tail_i(0) else TEMP_47_res(2);
	ga_ls_order_temp_3(3) <= TEMP_47_res(2) when stq_tail_i(0) else TEMP_47_res(3);
	ga_ls_order_temp_3(4) <= TEMP_47_res(3) when stq_tail_i(0) else TEMP_47_res(4);
	ga_ls_order_temp_3(5) <= TEMP_47_res(4) when stq_tail_i(0) else TEMP_47_res(5);
	ga_ls_order_temp_3(6) <= TEMP_47_res(5) when stq_tail_i(0) else TEMP_47_res(6);
	ga_ls_order_temp_3(7) <= TEMP_47_res(6) when stq_tail_i(0) else TEMP_47_res(7);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_4, ga_ls_order_rom_4, stq_tail)
	TEMP_48_res(0) <= ga_ls_order_rom_4(4) when stq_tail_i(2) else ga_ls_order_rom_4(0);
	TEMP_48_res(1) <= ga_ls_order_rom_4(5) when stq_tail_i(2) else ga_ls_order_rom_4(1);
	TEMP_48_res(2) <= ga_ls_order_rom_4(6) when stq_tail_i(2) else ga_ls_order_rom_4(2);
	TEMP_48_res(3) <= ga_ls_order_rom_4(7) when stq_tail_i(2) else ga_ls_order_rom_4(3);
	TEMP_48_res(4) <= ga_ls_order_rom_4(0) when stq_tail_i(2) else ga_ls_order_rom_4(4);
	TEMP_48_res(5) <= ga_ls_order_rom_4(1) when stq_tail_i(2) else ga_ls_order_rom_4(5);
	TEMP_48_res(6) <= ga_ls_order_rom_4(2) when stq_tail_i(2) else ga_ls_order_rom_4(6);
	TEMP_48_res(7) <= ga_ls_order_rom_4(3) when stq_tail_i(2) else ga_ls_order_rom_4(7);
	-- Layer End
	TEMP_49_res(0) <= TEMP_48_res(6) when stq_tail_i(1) else TEMP_48_res(0);
	TEMP_49_res(1) <= TEMP_48_res(7) when stq_tail_i(1) else TEMP_48_res(1);
	TEMP_49_res(2) <= TEMP_48_res(0) when stq_tail_i(1) else TEMP_48_res(2);
	TEMP_49_res(3) <= TEMP_48_res(1) when stq_tail_i(1) else TEMP_48_res(3);
	TEMP_49_res(4) <= TEMP_48_res(2) when stq_tail_i(1) else TEMP_48_res(4);
	TEMP_49_res(5) <= TEMP_48_res(3) when stq_tail_i(1) else TEMP_48_res(5);
	TEMP_49_res(6) <= TEMP_48_res(4) when stq_tail_i(1) else TEMP_48_res(6);
	TEMP_49_res(7) <= TEMP_48_res(5) when stq_tail_i(1) else TEMP_48_res(7);
	-- Layer End
	ga_ls_order_temp_4(0) <= TEMP_49_res(7) when stq_tail_i(0) else TEMP_49_res(0);
	ga_ls_order_temp_4(1) <= TEMP_49_res(0) when stq_tail_i(0) else TEMP_49_res(1);
	ga_ls_order_temp_4(2) <= TEMP_49_res(1) when stq_tail_i(0) else TEMP_49_res(2);
	ga_ls_order_temp_4(3) <= TEMP_49_res(2) when stq_tail_i(0) else TEMP_49_res(3);
	ga_ls_order_temp_4(4) <= TEMP_49_res(3) when stq_tail_i(0) else TEMP_49_res(4);
	ga_ls_order_temp_4(5) <= TEMP_49_res(4) when stq_tail_i(0) else TEMP_49_res(5);
	ga_ls_order_temp_4(6) <= TEMP_49_res(5) when stq_tail_i(0) else TEMP_49_res(6);
	ga_ls_order_temp_4(7) <= TEMP_49_res(6) when stq_tail_i(0) else TEMP_49_res(7);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_5, ga_ls_order_rom_5, stq_tail)
	TEMP_50_res(0) <= ga_ls_order_rom_5(4) when stq_tail_i(2) else ga_ls_order_rom_5(0);
	TEMP_50_res(1) <= ga_ls_order_rom_5(5) when stq_tail_i(2) else ga_ls_order_rom_5(1);
	TEMP_50_res(2) <= ga_ls_order_rom_5(6) when stq_tail_i(2) else ga_ls_order_rom_5(2);
	TEMP_50_res(3) <= ga_ls_order_rom_5(7) when stq_tail_i(2) else ga_ls_order_rom_5(3);
	TEMP_50_res(4) <= ga_ls_order_rom_5(0) when stq_tail_i(2) else ga_ls_order_rom_5(4);
	TEMP_50_res(5) <= ga_ls_order_rom_5(1) when stq_tail_i(2) else ga_ls_order_rom_5(5);
	TEMP_50_res(6) <= ga_ls_order_rom_5(2) when stq_tail_i(2) else ga_ls_order_rom_5(6);
	TEMP_50_res(7) <= ga_ls_order_rom_5(3) when stq_tail_i(2) else ga_ls_order_rom_5(7);
	-- Layer End
	TEMP_51_res(0) <= TEMP_50_res(6) when stq_tail_i(1) else TEMP_50_res(0);
	TEMP_51_res(1) <= TEMP_50_res(7) when stq_tail_i(1) else TEMP_50_res(1);
	TEMP_51_res(2) <= TEMP_50_res(0) when stq_tail_i(1) else TEMP_50_res(2);
	TEMP_51_res(3) <= TEMP_50_res(1) when stq_tail_i(1) else TEMP_50_res(3);
	TEMP_51_res(4) <= TEMP_50_res(2) when stq_tail_i(1) else TEMP_50_res(4);
	TEMP_51_res(5) <= TEMP_50_res(3) when stq_tail_i(1) else TEMP_50_res(5);
	TEMP_51_res(6) <= TEMP_50_res(4) when stq_tail_i(1) else TEMP_50_res(6);
	TEMP_51_res(7) <= TEMP_50_res(5) when stq_tail_i(1) else TEMP_50_res(7);
	-- Layer End
	ga_ls_order_temp_5(0) <= TEMP_51_res(7) when stq_tail_i(0) else TEMP_51_res(0);
	ga_ls_order_temp_5(1) <= TEMP_51_res(0) when stq_tail_i(0) else TEMP_51_res(1);
	ga_ls_order_temp_5(2) <= TEMP_51_res(1) when stq_tail_i(0) else TEMP_51_res(2);
	ga_ls_order_temp_5(3) <= TEMP_51_res(2) when stq_tail_i(0) else TEMP_51_res(3);
	ga_ls_order_temp_5(4) <= TEMP_51_res(3) when stq_tail_i(0) else TEMP_51_res(4);
	ga_ls_order_temp_5(5) <= TEMP_51_res(4) when stq_tail_i(0) else TEMP_51_res(5);
	ga_ls_order_temp_5(6) <= TEMP_51_res(5) when stq_tail_i(0) else TEMP_51_res(6);
	ga_ls_order_temp_5(7) <= TEMP_51_res(6) when stq_tail_i(0) else TEMP_51_res(7);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_6, ga_ls_order_rom_6, stq_tail)
	TEMP_52_res(0) <= ga_ls_order_rom_6(4) when stq_tail_i(2) else ga_ls_order_rom_6(0);
	TEMP_52_res(1) <= ga_ls_order_rom_6(5) when stq_tail_i(2) else ga_ls_order_rom_6(1);
	TEMP_52_res(2) <= ga_ls_order_rom_6(6) when stq_tail_i(2) else ga_ls_order_rom_6(2);
	TEMP_52_res(3) <= ga_ls_order_rom_6(7) when stq_tail_i(2) else ga_ls_order_rom_6(3);
	TEMP_52_res(4) <= ga_ls_order_rom_6(0) when stq_tail_i(2) else ga_ls_order_rom_6(4);
	TEMP_52_res(5) <= ga_ls_order_rom_6(1) when stq_tail_i(2) else ga_ls_order_rom_6(5);
	TEMP_52_res(6) <= ga_ls_order_rom_6(2) when stq_tail_i(2) else ga_ls_order_rom_6(6);
	TEMP_52_res(7) <= ga_ls_order_rom_6(3) when stq_tail_i(2) else ga_ls_order_rom_6(7);
	-- Layer End
	TEMP_53_res(0) <= TEMP_52_res(6) when stq_tail_i(1) else TEMP_52_res(0);
	TEMP_53_res(1) <= TEMP_52_res(7) when stq_tail_i(1) else TEMP_52_res(1);
	TEMP_53_res(2) <= TEMP_52_res(0) when stq_tail_i(1) else TEMP_52_res(2);
	TEMP_53_res(3) <= TEMP_52_res(1) when stq_tail_i(1) else TEMP_52_res(3);
	TEMP_53_res(4) <= TEMP_52_res(2) when stq_tail_i(1) else TEMP_52_res(4);
	TEMP_53_res(5) <= TEMP_52_res(3) when stq_tail_i(1) else TEMP_52_res(5);
	TEMP_53_res(6) <= TEMP_52_res(4) when stq_tail_i(1) else TEMP_52_res(6);
	TEMP_53_res(7) <= TEMP_52_res(5) when stq_tail_i(1) else TEMP_52_res(7);
	-- Layer End
	ga_ls_order_temp_6(0) <= TEMP_53_res(7) when stq_tail_i(0) else TEMP_53_res(0);
	ga_ls_order_temp_6(1) <= TEMP_53_res(0) when stq_tail_i(0) else TEMP_53_res(1);
	ga_ls_order_temp_6(2) <= TEMP_53_res(1) when stq_tail_i(0) else TEMP_53_res(2);
	ga_ls_order_temp_6(3) <= TEMP_53_res(2) when stq_tail_i(0) else TEMP_53_res(3);
	ga_ls_order_temp_6(4) <= TEMP_53_res(3) when stq_tail_i(0) else TEMP_53_res(4);
	ga_ls_order_temp_6(5) <= TEMP_53_res(4) when stq_tail_i(0) else TEMP_53_res(5);
	ga_ls_order_temp_6(6) <= TEMP_53_res(5) when stq_tail_i(0) else TEMP_53_res(6);
	ga_ls_order_temp_6(7) <= TEMP_53_res(6) when stq_tail_i(0) else TEMP_53_res(7);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_7, ga_ls_order_rom_7, stq_tail)
	TEMP_54_res(0) <= ga_ls_order_rom_7(4) when stq_tail_i(2) else ga_ls_order_rom_7(0);
	TEMP_54_res(1) <= ga_ls_order_rom_7(5) when stq_tail_i(2) else ga_ls_order_rom_7(1);
	TEMP_54_res(2) <= ga_ls_order_rom_7(6) when stq_tail_i(2) else ga_ls_order_rom_7(2);
	TEMP_54_res(3) <= ga_ls_order_rom_7(7) when stq_tail_i(2) else ga_ls_order_rom_7(3);
	TEMP_54_res(4) <= ga_ls_order_rom_7(0) when stq_tail_i(2) else ga_ls_order_rom_7(4);
	TEMP_54_res(5) <= ga_ls_order_rom_7(1) when stq_tail_i(2) else ga_ls_order_rom_7(5);
	TEMP_54_res(6) <= ga_ls_order_rom_7(2) when stq_tail_i(2) else ga_ls_order_rom_7(6);
	TEMP_54_res(7) <= ga_ls_order_rom_7(3) when stq_tail_i(2) else ga_ls_order_rom_7(7);
	-- Layer End
	TEMP_55_res(0) <= TEMP_54_res(6) when stq_tail_i(1) else TEMP_54_res(0);
	TEMP_55_res(1) <= TEMP_54_res(7) when stq_tail_i(1) else TEMP_54_res(1);
	TEMP_55_res(2) <= TEMP_54_res(0) when stq_tail_i(1) else TEMP_54_res(2);
	TEMP_55_res(3) <= TEMP_54_res(1) when stq_tail_i(1) else TEMP_54_res(3);
	TEMP_55_res(4) <= TEMP_54_res(2) when stq_tail_i(1) else TEMP_54_res(4);
	TEMP_55_res(5) <= TEMP_54_res(3) when stq_tail_i(1) else TEMP_54_res(5);
	TEMP_55_res(6) <= TEMP_54_res(4) when stq_tail_i(1) else TEMP_54_res(6);
	TEMP_55_res(7) <= TEMP_54_res(5) when stq_tail_i(1) else TEMP_54_res(7);
	-- Layer End
	ga_ls_order_temp_7(0) <= TEMP_55_res(7) when stq_tail_i(0) else TEMP_55_res(0);
	ga_ls_order_temp_7(1) <= TEMP_55_res(0) when stq_tail_i(0) else TEMP_55_res(1);
	ga_ls_order_temp_7(2) <= TEMP_55_res(1) when stq_tail_i(0) else TEMP_55_res(2);
	ga_ls_order_temp_7(3) <= TEMP_55_res(2) when stq_tail_i(0) else TEMP_55_res(3);
	ga_ls_order_temp_7(4) <= TEMP_55_res(3) when stq_tail_i(0) else TEMP_55_res(4);
	ga_ls_order_temp_7(5) <= TEMP_55_res(4) when stq_tail_i(0) else TEMP_55_res(5);
	ga_ls_order_temp_7(6) <= TEMP_55_res(5) when stq_tail_i(0) else TEMP_55_res(6);
	ga_ls_order_temp_7(7) <= TEMP_55_res(6) when stq_tail_i(0) else TEMP_55_res(7);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order, ga_ls_order_temp, ldq_tail)
	TEMP_56_res_0 <= ga_ls_order_temp_4 when ldq_tail_i(2) else ga_ls_order_temp_0;
	TEMP_56_res_1 <= ga_ls_order_temp_5 when ldq_tail_i(2) else ga_ls_order_temp_1;
	TEMP_56_res_2 <= ga_ls_order_temp_6 when ldq_tail_i(2) else ga_ls_order_temp_2;
	TEMP_56_res_3 <= ga_ls_order_temp_7 when ldq_tail_i(2) else ga_ls_order_temp_3;
	TEMP_56_res_4 <= ga_ls_order_temp_0 when ldq_tail_i(2) else ga_ls_order_temp_4;
	TEMP_56_res_5 <= ga_ls_order_temp_1 when ldq_tail_i(2) else ga_ls_order_temp_5;
	TEMP_56_res_6 <= ga_ls_order_temp_2 when ldq_tail_i(2) else ga_ls_order_temp_6;
	TEMP_56_res_7 <= ga_ls_order_temp_3 when ldq_tail_i(2) else ga_ls_order_temp_7;
	-- Layer End
	TEMP_57_res_0 <= TEMP_56_res_6 when ldq_tail_i(1) else TEMP_56_res_0;
	TEMP_57_res_1 <= TEMP_56_res_7 when ldq_tail_i(1) else TEMP_56_res_1;
	TEMP_57_res_2 <= TEMP_56_res_0 when ldq_tail_i(1) else TEMP_56_res_2;
	TEMP_57_res_3 <= TEMP_56_res_1 when ldq_tail_i(1) else TEMP_56_res_3;
	TEMP_57_res_4 <= TEMP_56_res_2 when ldq_tail_i(1) else TEMP_56_res_4;
	TEMP_57_res_5 <= TEMP_56_res_3 when ldq_tail_i(1) else TEMP_56_res_5;
	TEMP_57_res_6 <= TEMP_56_res_4 when ldq_tail_i(1) else TEMP_56_res_6;
	TEMP_57_res_7 <= TEMP_56_res_5 when ldq_tail_i(1) else TEMP_56_res_7;
	-- Layer End
	ga_ls_order_0_o <= TEMP_57_res_7 when ldq_tail_i(0) else TEMP_57_res_0;
	ga_ls_order_1_o <= TEMP_57_res_0 when ldq_tail_i(0) else TEMP_57_res_1;
	ga_ls_order_2_o <= TEMP_57_res_1 when ldq_tail_i(0) else TEMP_57_res_2;
	ga_ls_order_3_o <= TEMP_57_res_2 when ldq_tail_i(0) else TEMP_57_res_3;
	ga_ls_order_4_o <= TEMP_57_res_3 when ldq_tail_i(0) else TEMP_57_res_4;
	ga_ls_order_5_o <= TEMP_57_res_4 when ldq_tail_i(0) else TEMP_57_res_5;
	ga_ls_order_6_o <= TEMP_57_res_5 when ldq_tail_i(0) else TEMP_57_res_6;
	ga_ls_order_7_o <= TEMP_57_res_6 when ldq_tail_i(0) else TEMP_57_res_7;
	-- Shifter End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq4_core_lda is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_payload_0_i : in std_logic_vector(6 downto 0);
		port_payload_1_i : in std_logic_vector(6 downto 0);
		port_valid_0_i : in std_logic;
		port_valid_1_i : in std_logic;
		port_ready_0_o : out std_logic;
		port_ready_1_o : out std_logic;
		entry_alloc_0_i : in std_logic;
		entry_alloc_1_i : in std_logic;
		entry_alloc_2_i : in std_logic;
		entry_alloc_3_i : in std_logic;
		entry_alloc_4_i : in std_logic;
		entry_alloc_5_i : in std_logic;
		entry_alloc_6_i : in std_logic;
		entry_alloc_7_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_valid_2_i : in std_logic;
		entry_payload_valid_3_i : in std_logic;
		entry_payload_valid_4_i : in std_logic;
		entry_payload_valid_5_i : in std_logic;
		entry_payload_valid_6_i : in std_logic;
		entry_payload_valid_7_i : in std_logic;
		entry_port_idx_0_i : in std_logic_vector(0 downto 0);
		entry_port_idx_1_i : in std_logic_vector(0 downto 0);
		entry_port_idx_2_i : in std_logic_vector(0 downto 0);
		entry_port_idx_3_i : in std_logic_vector(0 downto 0);
		entry_port_idx_4_i : in std_logic_vector(0 downto 0);
		entry_port_idx_5_i : in std_logic_vector(0 downto 0);
		entry_port_idx_6_i : in std_logic_vector(0 downto 0);
		entry_port_idx_7_i : in std_logic_vector(0 downto 0);
		entry_payload_0_o : out std_logic_vector(6 downto 0);
		entry_payload_1_o : out std_logic_vector(6 downto 0);
		entry_payload_2_o : out std_logic_vector(6 downto 0);
		entry_payload_3_o : out std_logic_vector(6 downto 0);
		entry_payload_4_o : out std_logic_vector(6 downto 0);
		entry_payload_5_o : out std_logic_vector(6 downto 0);
		entry_payload_6_o : out std_logic_vector(6 downto 0);
		entry_payload_7_o : out std_logic_vector(6 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		entry_wen_2_o : out std_logic;
		entry_wen_3_o : out std_logic;
		entry_wen_4_o : out std_logic;
		entry_wen_5_o : out std_logic;
		entry_wen_6_o : out std_logic;
		entry_wen_7_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(7 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq4_core_lda is
	signal entry_port_idx_oh_0 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_2 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_3 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_4 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_5 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_6 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_7 : std_logic_vector(1 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_1_mux_1 : std_logic_vector(6 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_2_mux_1 : std_logic_vector(6 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_3_mux_1 : std_logic_vector(6 downto 0);
	signal TEMP_4_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_4_mux_1 : std_logic_vector(6 downto 0);
	signal TEMP_5_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_5_mux_1 : std_logic_vector(6 downto 0);
	signal TEMP_6_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_6_mux_1 : std_logic_vector(6 downto 0);
	signal TEMP_7_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_7_mux_1 : std_logic_vector(6 downto 0);
	signal TEMP_8_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_8_mux_1 : std_logic_vector(6 downto 0);
	signal entry_ptq_ready_0 : std_logic;
	signal entry_ptq_ready_1 : std_logic;
	signal entry_ptq_ready_2 : std_logic;
	signal entry_ptq_ready_3 : std_logic;
	signal entry_ptq_ready_4 : std_logic;
	signal entry_ptq_ready_5 : std_logic;
	signal entry_ptq_ready_6 : std_logic;
	signal entry_ptq_ready_7 : std_logic;
	signal entry_waiting_for_port_0 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_1 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_2 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_3 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_4 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_5 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_6 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_7 : std_logic_vector(1 downto 0);
	signal port_ready_vec : std_logic_vector(1 downto 0);
	signal TEMP_9_res_0 : std_logic_vector(1 downto 0);
	signal TEMP_9_res_1 : std_logic_vector(1 downto 0);
	signal TEMP_9_res_2 : std_logic_vector(1 downto 0);
	signal TEMP_9_res_3 : std_logic_vector(1 downto 0);
	signal TEMP_10_res_0 : std_logic_vector(1 downto 0);
	signal TEMP_10_res_1 : std_logic_vector(1 downto 0);
	signal entry_port_options_0 : std_logic_vector(1 downto 0);
	signal entry_port_options_1 : std_logic_vector(1 downto 0);
	signal entry_port_options_2 : std_logic_vector(1 downto 0);
	signal entry_port_options_3 : std_logic_vector(1 downto 0);
	signal entry_port_options_4 : std_logic_vector(1 downto 0);
	signal entry_port_options_5 : std_logic_vector(1 downto 0);
	signal entry_port_options_6 : std_logic_vector(1 downto 0);
	signal entry_port_options_7 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_2 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_3 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_4 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_5 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_6 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_7 : std_logic_vector(1 downto 0);
	signal TEMP_11_double_in_0 : std_logic_vector(15 downto 0);
	signal TEMP_11_double_out_0 : std_logic_vector(15 downto 0);
	signal TEMP_11_double_in_1 : std_logic_vector(15 downto 0);
	signal TEMP_11_double_out_1 : std_logic_vector(15 downto 0);
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

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_4, entry_port_idx_4)
	entry_port_idx_oh_4(0) <= '1' when entry_port_idx_4_i = "0" else '0';
	entry_port_idx_oh_4(1) <= '1' when entry_port_idx_4_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_5, entry_port_idx_5)
	entry_port_idx_oh_5(0) <= '1' when entry_port_idx_5_i = "0" else '0';
	entry_port_idx_oh_5(1) <= '1' when entry_port_idx_5_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_6, entry_port_idx_6)
	entry_port_idx_oh_6(0) <= '1' when entry_port_idx_6_i = "0" else '0';
	entry_port_idx_oh_6(1) <= '1' when entry_port_idx_6_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_7, entry_port_idx_7)
	entry_port_idx_oh_7(0) <= '1' when entry_port_idx_7_i = "0" else '0';
	entry_port_idx_oh_7(1) <= '1' when entry_port_idx_7_i = "1" else '0';
	-- Bits To One-Hot End

	-- Mux1H Begin
	-- Mux1H(entry_payload_0, port_payload, entry_port_idx_oh_0)
	TEMP_1_mux_0 <= port_payload_0_i when entry_port_idx_oh_0(0) = '1' else "0000000";
	TEMP_1_mux_1 <= port_payload_1_i when entry_port_idx_oh_0(1) = '1' else "0000000";
	entry_payload_0_o <= TEMP_1_mux_0 or TEMP_1_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_1, port_payload, entry_port_idx_oh_1)
	TEMP_2_mux_0 <= port_payload_0_i when entry_port_idx_oh_1(0) = '1' else "0000000";
	TEMP_2_mux_1 <= port_payload_1_i when entry_port_idx_oh_1(1) = '1' else "0000000";
	entry_payload_1_o <= TEMP_2_mux_0 or TEMP_2_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_2, port_payload, entry_port_idx_oh_2)
	TEMP_3_mux_0 <= port_payload_0_i when entry_port_idx_oh_2(0) = '1' else "0000000";
	TEMP_3_mux_1 <= port_payload_1_i when entry_port_idx_oh_2(1) = '1' else "0000000";
	entry_payload_2_o <= TEMP_3_mux_0 or TEMP_3_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_3, port_payload, entry_port_idx_oh_3)
	TEMP_4_mux_0 <= port_payload_0_i when entry_port_idx_oh_3(0) = '1' else "0000000";
	TEMP_4_mux_1 <= port_payload_1_i when entry_port_idx_oh_3(1) = '1' else "0000000";
	entry_payload_3_o <= TEMP_4_mux_0 or TEMP_4_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_4, port_payload, entry_port_idx_oh_4)
	TEMP_5_mux_0 <= port_payload_0_i when entry_port_idx_oh_4(0) = '1' else "0000000";
	TEMP_5_mux_1 <= port_payload_1_i when entry_port_idx_oh_4(1) = '1' else "0000000";
	entry_payload_4_o <= TEMP_5_mux_0 or TEMP_5_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_5, port_payload, entry_port_idx_oh_5)
	TEMP_6_mux_0 <= port_payload_0_i when entry_port_idx_oh_5(0) = '1' else "0000000";
	TEMP_6_mux_1 <= port_payload_1_i when entry_port_idx_oh_5(1) = '1' else "0000000";
	entry_payload_5_o <= TEMP_6_mux_0 or TEMP_6_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_6, port_payload, entry_port_idx_oh_6)
	TEMP_7_mux_0 <= port_payload_0_i when entry_port_idx_oh_6(0) = '1' else "0000000";
	TEMP_7_mux_1 <= port_payload_1_i when entry_port_idx_oh_6(1) = '1' else "0000000";
	entry_payload_6_o <= TEMP_7_mux_0 or TEMP_7_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_7, port_payload, entry_port_idx_oh_7)
	TEMP_8_mux_0 <= port_payload_0_i when entry_port_idx_oh_7(0) = '1' else "0000000";
	TEMP_8_mux_1 <= port_payload_1_i when entry_port_idx_oh_7(1) = '1' else "0000000";
	entry_payload_7_o <= TEMP_8_mux_0 or TEMP_8_mux_1;
	-- Mux1H End

	entry_ptq_ready_0 <= entry_alloc_0_i and not entry_payload_valid_0_i;
	entry_ptq_ready_1 <= entry_alloc_1_i and not entry_payload_valid_1_i;
	entry_ptq_ready_2 <= entry_alloc_2_i and not entry_payload_valid_2_i;
	entry_ptq_ready_3 <= entry_alloc_3_i and not entry_payload_valid_3_i;
	entry_ptq_ready_4 <= entry_alloc_4_i and not entry_payload_valid_4_i;
	entry_ptq_ready_5 <= entry_alloc_5_i and not entry_payload_valid_5_i;
	entry_ptq_ready_6 <= entry_alloc_6_i and not entry_payload_valid_6_i;
	entry_ptq_ready_7 <= entry_alloc_7_i and not entry_payload_valid_7_i;
	entry_waiting_for_port_0 <= entry_port_idx_oh_0 when entry_ptq_ready_0 else "00";
	entry_waiting_for_port_1 <= entry_port_idx_oh_1 when entry_ptq_ready_1 else "00";
	entry_waiting_for_port_2 <= entry_port_idx_oh_2 when entry_ptq_ready_2 else "00";
	entry_waiting_for_port_3 <= entry_port_idx_oh_3 when entry_ptq_ready_3 else "00";
	entry_waiting_for_port_4 <= entry_port_idx_oh_4 when entry_ptq_ready_4 else "00";
	entry_waiting_for_port_5 <= entry_port_idx_oh_5 when entry_ptq_ready_5 else "00";
	entry_waiting_for_port_6 <= entry_port_idx_oh_6 when entry_ptq_ready_6 else "00";
	entry_waiting_for_port_7 <= entry_port_idx_oh_7 when entry_ptq_ready_7 else "00";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_waiting_for_port, or)
	TEMP_9_res_0 <= entry_waiting_for_port_0 or entry_waiting_for_port_4;
	TEMP_9_res_1 <= entry_waiting_for_port_1 or entry_waiting_for_port_5;
	TEMP_9_res_2 <= entry_waiting_for_port_2 or entry_waiting_for_port_6;
	TEMP_9_res_3 <= entry_waiting_for_port_3 or entry_waiting_for_port_7;
	-- Layer End
	TEMP_10_res_0 <= TEMP_9_res_0 or TEMP_9_res_2;
	TEMP_10_res_1 <= TEMP_9_res_1 or TEMP_9_res_3;
	-- Layer End
	port_ready_vec <= TEMP_10_res_0 or TEMP_10_res_1;
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
	entry_port_options_4(0) <= entry_waiting_for_port_4(0) and port_valid_0_i;
	entry_port_options_4(1) <= entry_waiting_for_port_4(1) and port_valid_1_i;
	entry_port_options_5(0) <= entry_waiting_for_port_5(0) and port_valid_0_i;
	entry_port_options_5(1) <= entry_waiting_for_port_5(1) and port_valid_1_i;
	entry_port_options_6(0) <= entry_waiting_for_port_6(0) and port_valid_0_i;
	entry_port_options_6(1) <= entry_waiting_for_port_6(1) and port_valid_1_i;
	entry_port_options_7(0) <= entry_waiting_for_port_7(0) and port_valid_0_i;
	entry_port_options_7(1) <= entry_waiting_for_port_7(1) and port_valid_1_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_transfer, entry_port_options, queue_head_oh)
	TEMP_11_double_in_0(0) <= entry_port_options_0(0);
	TEMP_11_double_in_0(8) <= entry_port_options_0(0);
	TEMP_11_double_in_0(1) <= entry_port_options_1(0);
	TEMP_11_double_in_0(9) <= entry_port_options_1(0);
	TEMP_11_double_in_0(2) <= entry_port_options_2(0);
	TEMP_11_double_in_0(10) <= entry_port_options_2(0);
	TEMP_11_double_in_0(3) <= entry_port_options_3(0);
	TEMP_11_double_in_0(11) <= entry_port_options_3(0);
	TEMP_11_double_in_0(4) <= entry_port_options_4(0);
	TEMP_11_double_in_0(12) <= entry_port_options_4(0);
	TEMP_11_double_in_0(5) <= entry_port_options_5(0);
	TEMP_11_double_in_0(13) <= entry_port_options_5(0);
	TEMP_11_double_in_0(6) <= entry_port_options_6(0);
	TEMP_11_double_in_0(14) <= entry_port_options_6(0);
	TEMP_11_double_in_0(7) <= entry_port_options_7(0);
	TEMP_11_double_in_0(15) <= entry_port_options_7(0);
	TEMP_11_double_out_0 <= TEMP_11_double_in_0 and not std_logic_vector( unsigned( TEMP_11_double_in_0 ) - unsigned( "00000000" & queue_head_oh_i ) );
	entry_port_transfer_0(0) <= TEMP_11_double_out_0(0) or TEMP_11_double_out_0(8);
	entry_port_transfer_1(0) <= TEMP_11_double_out_0(1) or TEMP_11_double_out_0(9);
	entry_port_transfer_2(0) <= TEMP_11_double_out_0(2) or TEMP_11_double_out_0(10);
	entry_port_transfer_3(0) <= TEMP_11_double_out_0(3) or TEMP_11_double_out_0(11);
	entry_port_transfer_4(0) <= TEMP_11_double_out_0(4) or TEMP_11_double_out_0(12);
	entry_port_transfer_5(0) <= TEMP_11_double_out_0(5) or TEMP_11_double_out_0(13);
	entry_port_transfer_6(0) <= TEMP_11_double_out_0(6) or TEMP_11_double_out_0(14);
	entry_port_transfer_7(0) <= TEMP_11_double_out_0(7) or TEMP_11_double_out_0(15);
	TEMP_11_double_in_1(0) <= entry_port_options_0(1);
	TEMP_11_double_in_1(8) <= entry_port_options_0(1);
	TEMP_11_double_in_1(1) <= entry_port_options_1(1);
	TEMP_11_double_in_1(9) <= entry_port_options_1(1);
	TEMP_11_double_in_1(2) <= entry_port_options_2(1);
	TEMP_11_double_in_1(10) <= entry_port_options_2(1);
	TEMP_11_double_in_1(3) <= entry_port_options_3(1);
	TEMP_11_double_in_1(11) <= entry_port_options_3(1);
	TEMP_11_double_in_1(4) <= entry_port_options_4(1);
	TEMP_11_double_in_1(12) <= entry_port_options_4(1);
	TEMP_11_double_in_1(5) <= entry_port_options_5(1);
	TEMP_11_double_in_1(13) <= entry_port_options_5(1);
	TEMP_11_double_in_1(6) <= entry_port_options_6(1);
	TEMP_11_double_in_1(14) <= entry_port_options_6(1);
	TEMP_11_double_in_1(7) <= entry_port_options_7(1);
	TEMP_11_double_in_1(15) <= entry_port_options_7(1);
	TEMP_11_double_out_1 <= TEMP_11_double_in_1 and not std_logic_vector( unsigned( TEMP_11_double_in_1 ) - unsigned( "00000000" & queue_head_oh_i ) );
	entry_port_transfer_0(1) <= TEMP_11_double_out_1(0) or TEMP_11_double_out_1(8);
	entry_port_transfer_1(1) <= TEMP_11_double_out_1(1) or TEMP_11_double_out_1(9);
	entry_port_transfer_2(1) <= TEMP_11_double_out_1(2) or TEMP_11_double_out_1(10);
	entry_port_transfer_3(1) <= TEMP_11_double_out_1(3) or TEMP_11_double_out_1(11);
	entry_port_transfer_4(1) <= TEMP_11_double_out_1(4) or TEMP_11_double_out_1(12);
	entry_port_transfer_5(1) <= TEMP_11_double_out_1(5) or TEMP_11_double_out_1(13);
	entry_port_transfer_6(1) <= TEMP_11_double_out_1(6) or TEMP_11_double_out_1(14);
	entry_port_transfer_7(1) <= TEMP_11_double_out_1(7) or TEMP_11_double_out_1(15);
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

	-- Reduction Begin
	-- Reduce(entry_wen_4, entry_port_transfer_4, or)
	entry_wen_4_o <= entry_port_transfer_4(0) or entry_port_transfer_4(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_5, entry_port_transfer_5, or)
	entry_wen_5_o <= entry_port_transfer_5(0) or entry_port_transfer_5(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_6, entry_port_transfer_6, or)
	entry_wen_6_o <= entry_port_transfer_6(0) or entry_port_transfer_6(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_7, entry_port_transfer_7, or)
	entry_wen_7_o <= entry_port_transfer_7(0) or entry_port_transfer_7(1);
	-- Reduction End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq4_core_ldd is
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
		entry_alloc_4_i : in std_logic;
		entry_alloc_5_i : in std_logic;
		entry_alloc_6_i : in std_logic;
		entry_alloc_7_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_valid_2_i : in std_logic;
		entry_payload_valid_3_i : in std_logic;
		entry_payload_valid_4_i : in std_logic;
		entry_payload_valid_5_i : in std_logic;
		entry_payload_valid_6_i : in std_logic;
		entry_payload_valid_7_i : in std_logic;
		entry_port_idx_0_i : in std_logic_vector(0 downto 0);
		entry_port_idx_1_i : in std_logic_vector(0 downto 0);
		entry_port_idx_2_i : in std_logic_vector(0 downto 0);
		entry_port_idx_3_i : in std_logic_vector(0 downto 0);
		entry_port_idx_4_i : in std_logic_vector(0 downto 0);
		entry_port_idx_5_i : in std_logic_vector(0 downto 0);
		entry_port_idx_6_i : in std_logic_vector(0 downto 0);
		entry_port_idx_7_i : in std_logic_vector(0 downto 0);
		entry_payload_0_i : in std_logic_vector(31 downto 0);
		entry_payload_1_i : in std_logic_vector(31 downto 0);
		entry_payload_2_i : in std_logic_vector(31 downto 0);
		entry_payload_3_i : in std_logic_vector(31 downto 0);
		entry_payload_4_i : in std_logic_vector(31 downto 0);
		entry_payload_5_i : in std_logic_vector(31 downto 0);
		entry_payload_6_i : in std_logic_vector(31 downto 0);
		entry_payload_7_i : in std_logic_vector(31 downto 0);
		entry_reset_0_o : out std_logic;
		entry_reset_1_o : out std_logic;
		entry_reset_2_o : out std_logic;
		entry_reset_3_o : out std_logic;
		entry_reset_4_o : out std_logic;
		entry_reset_5_o : out std_logic;
		entry_reset_6_o : out std_logic;
		entry_reset_7_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(7 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq4_core_ldd is
	signal entry_port_idx_oh_0 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_2 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_3 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_4 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_5 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_6 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_7 : std_logic_vector(1 downto 0);
	signal entry_allocated_for_port_0 : std_logic_vector(1 downto 0);
	signal entry_allocated_for_port_1 : std_logic_vector(1 downto 0);
	signal entry_allocated_for_port_2 : std_logic_vector(1 downto 0);
	signal entry_allocated_for_port_3 : std_logic_vector(1 downto 0);
	signal entry_allocated_for_port_4 : std_logic_vector(1 downto 0);
	signal entry_allocated_for_port_5 : std_logic_vector(1 downto 0);
	signal entry_allocated_for_port_6 : std_logic_vector(1 downto 0);
	signal entry_allocated_for_port_7 : std_logic_vector(1 downto 0);
	signal oldest_entry_allocated_per_port_0 : std_logic_vector(1 downto 0);
	signal oldest_entry_allocated_per_port_1 : std_logic_vector(1 downto 0);
	signal oldest_entry_allocated_per_port_2 : std_logic_vector(1 downto 0);
	signal oldest_entry_allocated_per_port_3 : std_logic_vector(1 downto 0);
	signal oldest_entry_allocated_per_port_4 : std_logic_vector(1 downto 0);
	signal oldest_entry_allocated_per_port_5 : std_logic_vector(1 downto 0);
	signal oldest_entry_allocated_per_port_6 : std_logic_vector(1 downto 0);
	signal oldest_entry_allocated_per_port_7 : std_logic_vector(1 downto 0);
	signal TEMP_1_double_in_0 : std_logic_vector(15 downto 0);
	signal TEMP_1_double_out_0 : std_logic_vector(15 downto 0);
	signal TEMP_1_double_in_1 : std_logic_vector(15 downto 0);
	signal TEMP_1_double_out_1 : std_logic_vector(15 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_4_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_4_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_5_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_5_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_5_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_5_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_5_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_5_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_5_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_5_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_6_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_6_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_6_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_6_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_7_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_7_res_1 : std_logic_vector(31 downto 0);
	signal entry_waiting_for_port_valid_0 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_valid_1 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_valid_2 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_valid_3 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_valid_4 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_valid_5 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_valid_6 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_valid_7 : std_logic_vector(1 downto 0);
	signal port_valid_vec : std_logic_vector(1 downto 0);
	signal TEMP_8_res_0 : std_logic_vector(1 downto 0);
	signal TEMP_8_res_1 : std_logic_vector(1 downto 0);
	signal TEMP_8_res_2 : std_logic_vector(1 downto 0);
	signal TEMP_8_res_3 : std_logic_vector(1 downto 0);
	signal TEMP_9_res_0 : std_logic_vector(1 downto 0);
	signal TEMP_9_res_1 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_2 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_3 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_4 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_5 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_6 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_7 : std_logic_vector(1 downto 0);
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

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_4, entry_port_idx_4)
	entry_port_idx_oh_4(0) <= '1' when entry_port_idx_4_i = "0" else '0';
	entry_port_idx_oh_4(1) <= '1' when entry_port_idx_4_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_5, entry_port_idx_5)
	entry_port_idx_oh_5(0) <= '1' when entry_port_idx_5_i = "0" else '0';
	entry_port_idx_oh_5(1) <= '1' when entry_port_idx_5_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_6, entry_port_idx_6)
	entry_port_idx_oh_6(0) <= '1' when entry_port_idx_6_i = "0" else '0';
	entry_port_idx_oh_6(1) <= '1' when entry_port_idx_6_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_7, entry_port_idx_7)
	entry_port_idx_oh_7(0) <= '1' when entry_port_idx_7_i = "0" else '0';
	entry_port_idx_oh_7(1) <= '1' when entry_port_idx_7_i = "1" else '0';
	-- Bits To One-Hot End

	entry_allocated_for_port_0 <= entry_port_idx_oh_0 when entry_alloc_0_i else "00";
	entry_allocated_for_port_1 <= entry_port_idx_oh_1 when entry_alloc_1_i else "00";
	entry_allocated_for_port_2 <= entry_port_idx_oh_2 when entry_alloc_2_i else "00";
	entry_allocated_for_port_3 <= entry_port_idx_oh_3 when entry_alloc_3_i else "00";
	entry_allocated_for_port_4 <= entry_port_idx_oh_4 when entry_alloc_4_i else "00";
	entry_allocated_for_port_5 <= entry_port_idx_oh_5 when entry_alloc_5_i else "00";
	entry_allocated_for_port_6 <= entry_port_idx_oh_6 when entry_alloc_6_i else "00";
	entry_allocated_for_port_7 <= entry_port_idx_oh_7 when entry_alloc_7_i else "00";
	-- Priority Masking Begin
	-- CyclicPriorityMask(oldest_entry_allocated_per_port, entry_allocated_for_port, queue_head_oh)
	TEMP_1_double_in_0(0) <= entry_allocated_for_port_0(0);
	TEMP_1_double_in_0(8) <= entry_allocated_for_port_0(0);
	TEMP_1_double_in_0(1) <= entry_allocated_for_port_1(0);
	TEMP_1_double_in_0(9) <= entry_allocated_for_port_1(0);
	TEMP_1_double_in_0(2) <= entry_allocated_for_port_2(0);
	TEMP_1_double_in_0(10) <= entry_allocated_for_port_2(0);
	TEMP_1_double_in_0(3) <= entry_allocated_for_port_3(0);
	TEMP_1_double_in_0(11) <= entry_allocated_for_port_3(0);
	TEMP_1_double_in_0(4) <= entry_allocated_for_port_4(0);
	TEMP_1_double_in_0(12) <= entry_allocated_for_port_4(0);
	TEMP_1_double_in_0(5) <= entry_allocated_for_port_5(0);
	TEMP_1_double_in_0(13) <= entry_allocated_for_port_5(0);
	TEMP_1_double_in_0(6) <= entry_allocated_for_port_6(0);
	TEMP_1_double_in_0(14) <= entry_allocated_for_port_6(0);
	TEMP_1_double_in_0(7) <= entry_allocated_for_port_7(0);
	TEMP_1_double_in_0(15) <= entry_allocated_for_port_7(0);
	TEMP_1_double_out_0 <= TEMP_1_double_in_0 and not std_logic_vector( unsigned( TEMP_1_double_in_0 ) - unsigned( "00000000" & queue_head_oh_i ) );
	oldest_entry_allocated_per_port_0(0) <= TEMP_1_double_out_0(0) or TEMP_1_double_out_0(8);
	oldest_entry_allocated_per_port_1(0) <= TEMP_1_double_out_0(1) or TEMP_1_double_out_0(9);
	oldest_entry_allocated_per_port_2(0) <= TEMP_1_double_out_0(2) or TEMP_1_double_out_0(10);
	oldest_entry_allocated_per_port_3(0) <= TEMP_1_double_out_0(3) or TEMP_1_double_out_0(11);
	oldest_entry_allocated_per_port_4(0) <= TEMP_1_double_out_0(4) or TEMP_1_double_out_0(12);
	oldest_entry_allocated_per_port_5(0) <= TEMP_1_double_out_0(5) or TEMP_1_double_out_0(13);
	oldest_entry_allocated_per_port_6(0) <= TEMP_1_double_out_0(6) or TEMP_1_double_out_0(14);
	oldest_entry_allocated_per_port_7(0) <= TEMP_1_double_out_0(7) or TEMP_1_double_out_0(15);
	TEMP_1_double_in_1(0) <= entry_allocated_for_port_0(1);
	TEMP_1_double_in_1(8) <= entry_allocated_for_port_0(1);
	TEMP_1_double_in_1(1) <= entry_allocated_for_port_1(1);
	TEMP_1_double_in_1(9) <= entry_allocated_for_port_1(1);
	TEMP_1_double_in_1(2) <= entry_allocated_for_port_2(1);
	TEMP_1_double_in_1(10) <= entry_allocated_for_port_2(1);
	TEMP_1_double_in_1(3) <= entry_allocated_for_port_3(1);
	TEMP_1_double_in_1(11) <= entry_allocated_for_port_3(1);
	TEMP_1_double_in_1(4) <= entry_allocated_for_port_4(1);
	TEMP_1_double_in_1(12) <= entry_allocated_for_port_4(1);
	TEMP_1_double_in_1(5) <= entry_allocated_for_port_5(1);
	TEMP_1_double_in_1(13) <= entry_allocated_for_port_5(1);
	TEMP_1_double_in_1(6) <= entry_allocated_for_port_6(1);
	TEMP_1_double_in_1(14) <= entry_allocated_for_port_6(1);
	TEMP_1_double_in_1(7) <= entry_allocated_for_port_7(1);
	TEMP_1_double_in_1(15) <= entry_allocated_for_port_7(1);
	TEMP_1_double_out_1 <= TEMP_1_double_in_1 and not std_logic_vector( unsigned( TEMP_1_double_in_1 ) - unsigned( "00000000" & queue_head_oh_i ) );
	oldest_entry_allocated_per_port_0(1) <= TEMP_1_double_out_1(0) or TEMP_1_double_out_1(8);
	oldest_entry_allocated_per_port_1(1) <= TEMP_1_double_out_1(1) or TEMP_1_double_out_1(9);
	oldest_entry_allocated_per_port_2(1) <= TEMP_1_double_out_1(2) or TEMP_1_double_out_1(10);
	oldest_entry_allocated_per_port_3(1) <= TEMP_1_double_out_1(3) or TEMP_1_double_out_1(11);
	oldest_entry_allocated_per_port_4(1) <= TEMP_1_double_out_1(4) or TEMP_1_double_out_1(12);
	oldest_entry_allocated_per_port_5(1) <= TEMP_1_double_out_1(5) or TEMP_1_double_out_1(13);
	oldest_entry_allocated_per_port_6(1) <= TEMP_1_double_out_1(6) or TEMP_1_double_out_1(14);
	oldest_entry_allocated_per_port_7(1) <= TEMP_1_double_out_1(7) or TEMP_1_double_out_1(15);
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
	TEMP_3_res_0 <= TEMP_2_mux_0 or TEMP_2_mux_4;
	TEMP_3_res_1 <= TEMP_2_mux_1 or TEMP_2_mux_5;
	TEMP_3_res_2 <= TEMP_2_mux_2 or TEMP_2_mux_6;
	TEMP_3_res_3 <= TEMP_2_mux_3 or TEMP_2_mux_7;
	-- Layer End
	TEMP_4_res_0 <= TEMP_3_res_0 or TEMP_3_res_2;
	TEMP_4_res_1 <= TEMP_3_res_1 or TEMP_3_res_3;
	-- Layer End
	port_payload_0_o <= TEMP_4_res_0 or TEMP_4_res_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(port_payload_1, entry_payload, oldest_entry_allocated_per_port)
	TEMP_5_mux_0 <= entry_payload_0_i when oldest_entry_allocated_per_port_0(1) = '1' else "00000000000000000000000000000000";
	TEMP_5_mux_1 <= entry_payload_1_i when oldest_entry_allocated_per_port_1(1) = '1' else "00000000000000000000000000000000";
	TEMP_5_mux_2 <= entry_payload_2_i when oldest_entry_allocated_per_port_2(1) = '1' else "00000000000000000000000000000000";
	TEMP_5_mux_3 <= entry_payload_3_i when oldest_entry_allocated_per_port_3(1) = '1' else "00000000000000000000000000000000";
	TEMP_5_mux_4 <= entry_payload_4_i when oldest_entry_allocated_per_port_4(1) = '1' else "00000000000000000000000000000000";
	TEMP_5_mux_5 <= entry_payload_5_i when oldest_entry_allocated_per_port_5(1) = '1' else "00000000000000000000000000000000";
	TEMP_5_mux_6 <= entry_payload_6_i when oldest_entry_allocated_per_port_6(1) = '1' else "00000000000000000000000000000000";
	TEMP_5_mux_7 <= entry_payload_7_i when oldest_entry_allocated_per_port_7(1) = '1' else "00000000000000000000000000000000";
	TEMP_6_res_0 <= TEMP_5_mux_0 or TEMP_5_mux_4;
	TEMP_6_res_1 <= TEMP_5_mux_1 or TEMP_5_mux_5;
	TEMP_6_res_2 <= TEMP_5_mux_2 or TEMP_5_mux_6;
	TEMP_6_res_3 <= TEMP_5_mux_3 or TEMP_5_mux_7;
	-- Layer End
	TEMP_7_res_0 <= TEMP_6_res_0 or TEMP_6_res_2;
	TEMP_7_res_1 <= TEMP_6_res_1 or TEMP_6_res_3;
	-- Layer End
	port_payload_1_o <= TEMP_7_res_0 or TEMP_7_res_1;
	-- Mux1H End

	entry_waiting_for_port_valid_0 <= oldest_entry_allocated_per_port_0 when entry_payload_valid_0_i else "00";
	entry_waiting_for_port_valid_1 <= oldest_entry_allocated_per_port_1 when entry_payload_valid_1_i else "00";
	entry_waiting_for_port_valid_2 <= oldest_entry_allocated_per_port_2 when entry_payload_valid_2_i else "00";
	entry_waiting_for_port_valid_3 <= oldest_entry_allocated_per_port_3 when entry_payload_valid_3_i else "00";
	entry_waiting_for_port_valid_4 <= oldest_entry_allocated_per_port_4 when entry_payload_valid_4_i else "00";
	entry_waiting_for_port_valid_5 <= oldest_entry_allocated_per_port_5 when entry_payload_valid_5_i else "00";
	entry_waiting_for_port_valid_6 <= oldest_entry_allocated_per_port_6 when entry_payload_valid_6_i else "00";
	entry_waiting_for_port_valid_7 <= oldest_entry_allocated_per_port_7 when entry_payload_valid_7_i else "00";
	-- Reduction Begin
	-- Reduce(port_valid_vec, entry_waiting_for_port_valid, or)
	TEMP_8_res_0 <= entry_waiting_for_port_valid_0 or entry_waiting_for_port_valid_4;
	TEMP_8_res_1 <= entry_waiting_for_port_valid_1 or entry_waiting_for_port_valid_5;
	TEMP_8_res_2 <= entry_waiting_for_port_valid_2 or entry_waiting_for_port_valid_6;
	TEMP_8_res_3 <= entry_waiting_for_port_valid_3 or entry_waiting_for_port_valid_7;
	-- Layer End
	TEMP_9_res_0 <= TEMP_8_res_0 or TEMP_8_res_2;
	TEMP_9_res_1 <= TEMP_8_res_1 or TEMP_8_res_3;
	-- Layer End
	port_valid_vec <= TEMP_9_res_0 or TEMP_9_res_1;
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
	entry_port_transfer_4(0) <= entry_waiting_for_port_valid_4(0) and port_ready_0_i;
	entry_port_transfer_4(1) <= entry_waiting_for_port_valid_4(1) and port_ready_1_i;
	entry_port_transfer_5(0) <= entry_waiting_for_port_valid_5(0) and port_ready_0_i;
	entry_port_transfer_5(1) <= entry_waiting_for_port_valid_5(1) and port_ready_1_i;
	entry_port_transfer_6(0) <= entry_waiting_for_port_valid_6(0) and port_ready_0_i;
	entry_port_transfer_6(1) <= entry_waiting_for_port_valid_6(1) and port_ready_1_i;
	entry_port_transfer_7(0) <= entry_waiting_for_port_valid_7(0) and port_ready_0_i;
	entry_port_transfer_7(1) <= entry_waiting_for_port_valid_7(1) and port_ready_1_i;
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

	-- Reduction Begin
	-- Reduce(entry_reset_4, entry_port_transfer_4, or)
	entry_reset_4_o <= entry_port_transfer_4(0) or entry_port_transfer_4(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_5, entry_port_transfer_5, or)
	entry_reset_5_o <= entry_port_transfer_5(0) or entry_port_transfer_5(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_6, entry_port_transfer_6, or)
	entry_reset_6_o <= entry_port_transfer_6(0) or entry_port_transfer_6(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_7, entry_port_transfer_7, or)
	entry_reset_7_o <= entry_port_transfer_7(0) or entry_port_transfer_7(1);
	-- Reduction End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq4_core_sta is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_payload_0_i : in std_logic_vector(6 downto 0);
		port_payload_1_i : in std_logic_vector(6 downto 0);
		port_valid_0_i : in std_logic;
		port_valid_1_i : in std_logic;
		port_ready_0_o : out std_logic;
		port_ready_1_o : out std_logic;
		entry_alloc_0_i : in std_logic;
		entry_alloc_1_i : in std_logic;
		entry_alloc_2_i : in std_logic;
		entry_alloc_3_i : in std_logic;
		entry_alloc_4_i : in std_logic;
		entry_alloc_5_i : in std_logic;
		entry_alloc_6_i : in std_logic;
		entry_alloc_7_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_valid_2_i : in std_logic;
		entry_payload_valid_3_i : in std_logic;
		entry_payload_valid_4_i : in std_logic;
		entry_payload_valid_5_i : in std_logic;
		entry_payload_valid_6_i : in std_logic;
		entry_payload_valid_7_i : in std_logic;
		entry_port_idx_0_i : in std_logic_vector(0 downto 0);
		entry_port_idx_1_i : in std_logic_vector(0 downto 0);
		entry_port_idx_2_i : in std_logic_vector(0 downto 0);
		entry_port_idx_3_i : in std_logic_vector(0 downto 0);
		entry_port_idx_4_i : in std_logic_vector(0 downto 0);
		entry_port_idx_5_i : in std_logic_vector(0 downto 0);
		entry_port_idx_6_i : in std_logic_vector(0 downto 0);
		entry_port_idx_7_i : in std_logic_vector(0 downto 0);
		entry_payload_0_o : out std_logic_vector(6 downto 0);
		entry_payload_1_o : out std_logic_vector(6 downto 0);
		entry_payload_2_o : out std_logic_vector(6 downto 0);
		entry_payload_3_o : out std_logic_vector(6 downto 0);
		entry_payload_4_o : out std_logic_vector(6 downto 0);
		entry_payload_5_o : out std_logic_vector(6 downto 0);
		entry_payload_6_o : out std_logic_vector(6 downto 0);
		entry_payload_7_o : out std_logic_vector(6 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		entry_wen_2_o : out std_logic;
		entry_wen_3_o : out std_logic;
		entry_wen_4_o : out std_logic;
		entry_wen_5_o : out std_logic;
		entry_wen_6_o : out std_logic;
		entry_wen_7_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(7 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq4_core_sta is
	signal entry_port_idx_oh_0 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_2 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_3 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_4 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_5 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_6 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_7 : std_logic_vector(1 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_1_mux_1 : std_logic_vector(6 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_2_mux_1 : std_logic_vector(6 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_3_mux_1 : std_logic_vector(6 downto 0);
	signal TEMP_4_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_4_mux_1 : std_logic_vector(6 downto 0);
	signal TEMP_5_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_5_mux_1 : std_logic_vector(6 downto 0);
	signal TEMP_6_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_6_mux_1 : std_logic_vector(6 downto 0);
	signal TEMP_7_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_7_mux_1 : std_logic_vector(6 downto 0);
	signal TEMP_8_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_8_mux_1 : std_logic_vector(6 downto 0);
	signal entry_ptq_ready_0 : std_logic;
	signal entry_ptq_ready_1 : std_logic;
	signal entry_ptq_ready_2 : std_logic;
	signal entry_ptq_ready_3 : std_logic;
	signal entry_ptq_ready_4 : std_logic;
	signal entry_ptq_ready_5 : std_logic;
	signal entry_ptq_ready_6 : std_logic;
	signal entry_ptq_ready_7 : std_logic;
	signal entry_waiting_for_port_0 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_1 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_2 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_3 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_4 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_5 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_6 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_7 : std_logic_vector(1 downto 0);
	signal port_ready_vec : std_logic_vector(1 downto 0);
	signal TEMP_9_res_0 : std_logic_vector(1 downto 0);
	signal TEMP_9_res_1 : std_logic_vector(1 downto 0);
	signal TEMP_9_res_2 : std_logic_vector(1 downto 0);
	signal TEMP_9_res_3 : std_logic_vector(1 downto 0);
	signal TEMP_10_res_0 : std_logic_vector(1 downto 0);
	signal TEMP_10_res_1 : std_logic_vector(1 downto 0);
	signal entry_port_options_0 : std_logic_vector(1 downto 0);
	signal entry_port_options_1 : std_logic_vector(1 downto 0);
	signal entry_port_options_2 : std_logic_vector(1 downto 0);
	signal entry_port_options_3 : std_logic_vector(1 downto 0);
	signal entry_port_options_4 : std_logic_vector(1 downto 0);
	signal entry_port_options_5 : std_logic_vector(1 downto 0);
	signal entry_port_options_6 : std_logic_vector(1 downto 0);
	signal entry_port_options_7 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_2 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_3 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_4 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_5 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_6 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_7 : std_logic_vector(1 downto 0);
	signal TEMP_11_double_in_0 : std_logic_vector(15 downto 0);
	signal TEMP_11_double_out_0 : std_logic_vector(15 downto 0);
	signal TEMP_11_double_in_1 : std_logic_vector(15 downto 0);
	signal TEMP_11_double_out_1 : std_logic_vector(15 downto 0);
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

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_4, entry_port_idx_4)
	entry_port_idx_oh_4(0) <= '1' when entry_port_idx_4_i = "0" else '0';
	entry_port_idx_oh_4(1) <= '1' when entry_port_idx_4_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_5, entry_port_idx_5)
	entry_port_idx_oh_5(0) <= '1' when entry_port_idx_5_i = "0" else '0';
	entry_port_idx_oh_5(1) <= '1' when entry_port_idx_5_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_6, entry_port_idx_6)
	entry_port_idx_oh_6(0) <= '1' when entry_port_idx_6_i = "0" else '0';
	entry_port_idx_oh_6(1) <= '1' when entry_port_idx_6_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_7, entry_port_idx_7)
	entry_port_idx_oh_7(0) <= '1' when entry_port_idx_7_i = "0" else '0';
	entry_port_idx_oh_7(1) <= '1' when entry_port_idx_7_i = "1" else '0';
	-- Bits To One-Hot End

	-- Mux1H Begin
	-- Mux1H(entry_payload_0, port_payload, entry_port_idx_oh_0)
	TEMP_1_mux_0 <= port_payload_0_i when entry_port_idx_oh_0(0) = '1' else "0000000";
	TEMP_1_mux_1 <= port_payload_1_i when entry_port_idx_oh_0(1) = '1' else "0000000";
	entry_payload_0_o <= TEMP_1_mux_0 or TEMP_1_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_1, port_payload, entry_port_idx_oh_1)
	TEMP_2_mux_0 <= port_payload_0_i when entry_port_idx_oh_1(0) = '1' else "0000000";
	TEMP_2_mux_1 <= port_payload_1_i when entry_port_idx_oh_1(1) = '1' else "0000000";
	entry_payload_1_o <= TEMP_2_mux_0 or TEMP_2_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_2, port_payload, entry_port_idx_oh_2)
	TEMP_3_mux_0 <= port_payload_0_i when entry_port_idx_oh_2(0) = '1' else "0000000";
	TEMP_3_mux_1 <= port_payload_1_i when entry_port_idx_oh_2(1) = '1' else "0000000";
	entry_payload_2_o <= TEMP_3_mux_0 or TEMP_3_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_3, port_payload, entry_port_idx_oh_3)
	TEMP_4_mux_0 <= port_payload_0_i when entry_port_idx_oh_3(0) = '1' else "0000000";
	TEMP_4_mux_1 <= port_payload_1_i when entry_port_idx_oh_3(1) = '1' else "0000000";
	entry_payload_3_o <= TEMP_4_mux_0 or TEMP_4_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_4, port_payload, entry_port_idx_oh_4)
	TEMP_5_mux_0 <= port_payload_0_i when entry_port_idx_oh_4(0) = '1' else "0000000";
	TEMP_5_mux_1 <= port_payload_1_i when entry_port_idx_oh_4(1) = '1' else "0000000";
	entry_payload_4_o <= TEMP_5_mux_0 or TEMP_5_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_5, port_payload, entry_port_idx_oh_5)
	TEMP_6_mux_0 <= port_payload_0_i when entry_port_idx_oh_5(0) = '1' else "0000000";
	TEMP_6_mux_1 <= port_payload_1_i when entry_port_idx_oh_5(1) = '1' else "0000000";
	entry_payload_5_o <= TEMP_6_mux_0 or TEMP_6_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_6, port_payload, entry_port_idx_oh_6)
	TEMP_7_mux_0 <= port_payload_0_i when entry_port_idx_oh_6(0) = '1' else "0000000";
	TEMP_7_mux_1 <= port_payload_1_i when entry_port_idx_oh_6(1) = '1' else "0000000";
	entry_payload_6_o <= TEMP_7_mux_0 or TEMP_7_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_7, port_payload, entry_port_idx_oh_7)
	TEMP_8_mux_0 <= port_payload_0_i when entry_port_idx_oh_7(0) = '1' else "0000000";
	TEMP_8_mux_1 <= port_payload_1_i when entry_port_idx_oh_7(1) = '1' else "0000000";
	entry_payload_7_o <= TEMP_8_mux_0 or TEMP_8_mux_1;
	-- Mux1H End

	entry_ptq_ready_0 <= entry_alloc_0_i and not entry_payload_valid_0_i;
	entry_ptq_ready_1 <= entry_alloc_1_i and not entry_payload_valid_1_i;
	entry_ptq_ready_2 <= entry_alloc_2_i and not entry_payload_valid_2_i;
	entry_ptq_ready_3 <= entry_alloc_3_i and not entry_payload_valid_3_i;
	entry_ptq_ready_4 <= entry_alloc_4_i and not entry_payload_valid_4_i;
	entry_ptq_ready_5 <= entry_alloc_5_i and not entry_payload_valid_5_i;
	entry_ptq_ready_6 <= entry_alloc_6_i and not entry_payload_valid_6_i;
	entry_ptq_ready_7 <= entry_alloc_7_i and not entry_payload_valid_7_i;
	entry_waiting_for_port_0 <= entry_port_idx_oh_0 when entry_ptq_ready_0 else "00";
	entry_waiting_for_port_1 <= entry_port_idx_oh_1 when entry_ptq_ready_1 else "00";
	entry_waiting_for_port_2 <= entry_port_idx_oh_2 when entry_ptq_ready_2 else "00";
	entry_waiting_for_port_3 <= entry_port_idx_oh_3 when entry_ptq_ready_3 else "00";
	entry_waiting_for_port_4 <= entry_port_idx_oh_4 when entry_ptq_ready_4 else "00";
	entry_waiting_for_port_5 <= entry_port_idx_oh_5 when entry_ptq_ready_5 else "00";
	entry_waiting_for_port_6 <= entry_port_idx_oh_6 when entry_ptq_ready_6 else "00";
	entry_waiting_for_port_7 <= entry_port_idx_oh_7 when entry_ptq_ready_7 else "00";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_waiting_for_port, or)
	TEMP_9_res_0 <= entry_waiting_for_port_0 or entry_waiting_for_port_4;
	TEMP_9_res_1 <= entry_waiting_for_port_1 or entry_waiting_for_port_5;
	TEMP_9_res_2 <= entry_waiting_for_port_2 or entry_waiting_for_port_6;
	TEMP_9_res_3 <= entry_waiting_for_port_3 or entry_waiting_for_port_7;
	-- Layer End
	TEMP_10_res_0 <= TEMP_9_res_0 or TEMP_9_res_2;
	TEMP_10_res_1 <= TEMP_9_res_1 or TEMP_9_res_3;
	-- Layer End
	port_ready_vec <= TEMP_10_res_0 or TEMP_10_res_1;
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
	entry_port_options_4(0) <= entry_waiting_for_port_4(0) and port_valid_0_i;
	entry_port_options_4(1) <= entry_waiting_for_port_4(1) and port_valid_1_i;
	entry_port_options_5(0) <= entry_waiting_for_port_5(0) and port_valid_0_i;
	entry_port_options_5(1) <= entry_waiting_for_port_5(1) and port_valid_1_i;
	entry_port_options_6(0) <= entry_waiting_for_port_6(0) and port_valid_0_i;
	entry_port_options_6(1) <= entry_waiting_for_port_6(1) and port_valid_1_i;
	entry_port_options_7(0) <= entry_waiting_for_port_7(0) and port_valid_0_i;
	entry_port_options_7(1) <= entry_waiting_for_port_7(1) and port_valid_1_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_transfer, entry_port_options, queue_head_oh)
	TEMP_11_double_in_0(0) <= entry_port_options_0(0);
	TEMP_11_double_in_0(8) <= entry_port_options_0(0);
	TEMP_11_double_in_0(1) <= entry_port_options_1(0);
	TEMP_11_double_in_0(9) <= entry_port_options_1(0);
	TEMP_11_double_in_0(2) <= entry_port_options_2(0);
	TEMP_11_double_in_0(10) <= entry_port_options_2(0);
	TEMP_11_double_in_0(3) <= entry_port_options_3(0);
	TEMP_11_double_in_0(11) <= entry_port_options_3(0);
	TEMP_11_double_in_0(4) <= entry_port_options_4(0);
	TEMP_11_double_in_0(12) <= entry_port_options_4(0);
	TEMP_11_double_in_0(5) <= entry_port_options_5(0);
	TEMP_11_double_in_0(13) <= entry_port_options_5(0);
	TEMP_11_double_in_0(6) <= entry_port_options_6(0);
	TEMP_11_double_in_0(14) <= entry_port_options_6(0);
	TEMP_11_double_in_0(7) <= entry_port_options_7(0);
	TEMP_11_double_in_0(15) <= entry_port_options_7(0);
	TEMP_11_double_out_0 <= TEMP_11_double_in_0 and not std_logic_vector( unsigned( TEMP_11_double_in_0 ) - unsigned( "00000000" & queue_head_oh_i ) );
	entry_port_transfer_0(0) <= TEMP_11_double_out_0(0) or TEMP_11_double_out_0(8);
	entry_port_transfer_1(0) <= TEMP_11_double_out_0(1) or TEMP_11_double_out_0(9);
	entry_port_transfer_2(0) <= TEMP_11_double_out_0(2) or TEMP_11_double_out_0(10);
	entry_port_transfer_3(0) <= TEMP_11_double_out_0(3) or TEMP_11_double_out_0(11);
	entry_port_transfer_4(0) <= TEMP_11_double_out_0(4) or TEMP_11_double_out_0(12);
	entry_port_transfer_5(0) <= TEMP_11_double_out_0(5) or TEMP_11_double_out_0(13);
	entry_port_transfer_6(0) <= TEMP_11_double_out_0(6) or TEMP_11_double_out_0(14);
	entry_port_transfer_7(0) <= TEMP_11_double_out_0(7) or TEMP_11_double_out_0(15);
	TEMP_11_double_in_1(0) <= entry_port_options_0(1);
	TEMP_11_double_in_1(8) <= entry_port_options_0(1);
	TEMP_11_double_in_1(1) <= entry_port_options_1(1);
	TEMP_11_double_in_1(9) <= entry_port_options_1(1);
	TEMP_11_double_in_1(2) <= entry_port_options_2(1);
	TEMP_11_double_in_1(10) <= entry_port_options_2(1);
	TEMP_11_double_in_1(3) <= entry_port_options_3(1);
	TEMP_11_double_in_1(11) <= entry_port_options_3(1);
	TEMP_11_double_in_1(4) <= entry_port_options_4(1);
	TEMP_11_double_in_1(12) <= entry_port_options_4(1);
	TEMP_11_double_in_1(5) <= entry_port_options_5(1);
	TEMP_11_double_in_1(13) <= entry_port_options_5(1);
	TEMP_11_double_in_1(6) <= entry_port_options_6(1);
	TEMP_11_double_in_1(14) <= entry_port_options_6(1);
	TEMP_11_double_in_1(7) <= entry_port_options_7(1);
	TEMP_11_double_in_1(15) <= entry_port_options_7(1);
	TEMP_11_double_out_1 <= TEMP_11_double_in_1 and not std_logic_vector( unsigned( TEMP_11_double_in_1 ) - unsigned( "00000000" & queue_head_oh_i ) );
	entry_port_transfer_0(1) <= TEMP_11_double_out_1(0) or TEMP_11_double_out_1(8);
	entry_port_transfer_1(1) <= TEMP_11_double_out_1(1) or TEMP_11_double_out_1(9);
	entry_port_transfer_2(1) <= TEMP_11_double_out_1(2) or TEMP_11_double_out_1(10);
	entry_port_transfer_3(1) <= TEMP_11_double_out_1(3) or TEMP_11_double_out_1(11);
	entry_port_transfer_4(1) <= TEMP_11_double_out_1(4) or TEMP_11_double_out_1(12);
	entry_port_transfer_5(1) <= TEMP_11_double_out_1(5) or TEMP_11_double_out_1(13);
	entry_port_transfer_6(1) <= TEMP_11_double_out_1(6) or TEMP_11_double_out_1(14);
	entry_port_transfer_7(1) <= TEMP_11_double_out_1(7) or TEMP_11_double_out_1(15);
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

	-- Reduction Begin
	-- Reduce(entry_wen_4, entry_port_transfer_4, or)
	entry_wen_4_o <= entry_port_transfer_4(0) or entry_port_transfer_4(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_5, entry_port_transfer_5, or)
	entry_wen_5_o <= entry_port_transfer_5(0) or entry_port_transfer_5(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_6, entry_port_transfer_6, or)
	entry_wen_6_o <= entry_port_transfer_6(0) or entry_port_transfer_6(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_7, entry_port_transfer_7, or)
	entry_wen_7_o <= entry_port_transfer_7(0) or entry_port_transfer_7(1);
	-- Reduction End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq4_core_std is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_payload_0_i : in std_logic_vector(31 downto 0);
		port_payload_1_i : in std_logic_vector(31 downto 0);
		port_valid_0_i : in std_logic;
		port_valid_1_i : in std_logic;
		port_ready_0_o : out std_logic;
		port_ready_1_o : out std_logic;
		entry_alloc_0_i : in std_logic;
		entry_alloc_1_i : in std_logic;
		entry_alloc_2_i : in std_logic;
		entry_alloc_3_i : in std_logic;
		entry_alloc_4_i : in std_logic;
		entry_alloc_5_i : in std_logic;
		entry_alloc_6_i : in std_logic;
		entry_alloc_7_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_valid_2_i : in std_logic;
		entry_payload_valid_3_i : in std_logic;
		entry_payload_valid_4_i : in std_logic;
		entry_payload_valid_5_i : in std_logic;
		entry_payload_valid_6_i : in std_logic;
		entry_payload_valid_7_i : in std_logic;
		entry_port_idx_0_i : in std_logic_vector(0 downto 0);
		entry_port_idx_1_i : in std_logic_vector(0 downto 0);
		entry_port_idx_2_i : in std_logic_vector(0 downto 0);
		entry_port_idx_3_i : in std_logic_vector(0 downto 0);
		entry_port_idx_4_i : in std_logic_vector(0 downto 0);
		entry_port_idx_5_i : in std_logic_vector(0 downto 0);
		entry_port_idx_6_i : in std_logic_vector(0 downto 0);
		entry_port_idx_7_i : in std_logic_vector(0 downto 0);
		entry_payload_0_o : out std_logic_vector(31 downto 0);
		entry_payload_1_o : out std_logic_vector(31 downto 0);
		entry_payload_2_o : out std_logic_vector(31 downto 0);
		entry_payload_3_o : out std_logic_vector(31 downto 0);
		entry_payload_4_o : out std_logic_vector(31 downto 0);
		entry_payload_5_o : out std_logic_vector(31 downto 0);
		entry_payload_6_o : out std_logic_vector(31 downto 0);
		entry_payload_7_o : out std_logic_vector(31 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		entry_wen_2_o : out std_logic;
		entry_wen_3_o : out std_logic;
		entry_wen_4_o : out std_logic;
		entry_wen_5_o : out std_logic;
		entry_wen_6_o : out std_logic;
		entry_wen_7_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(7 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq4_core_std is
	signal entry_port_idx_oh_0 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_2 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_3 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_4 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_5 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_6 : std_logic_vector(1 downto 0);
	signal entry_port_idx_oh_7 : std_logic_vector(1 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_1_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_3_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_4_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_4_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_5_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_5_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_6_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_6_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_7_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_7_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_8_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_8_mux_1 : std_logic_vector(31 downto 0);
	signal entry_ptq_ready_0 : std_logic;
	signal entry_ptq_ready_1 : std_logic;
	signal entry_ptq_ready_2 : std_logic;
	signal entry_ptq_ready_3 : std_logic;
	signal entry_ptq_ready_4 : std_logic;
	signal entry_ptq_ready_5 : std_logic;
	signal entry_ptq_ready_6 : std_logic;
	signal entry_ptq_ready_7 : std_logic;
	signal entry_waiting_for_port_0 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_1 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_2 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_3 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_4 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_5 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_6 : std_logic_vector(1 downto 0);
	signal entry_waiting_for_port_7 : std_logic_vector(1 downto 0);
	signal port_ready_vec : std_logic_vector(1 downto 0);
	signal TEMP_9_res_0 : std_logic_vector(1 downto 0);
	signal TEMP_9_res_1 : std_logic_vector(1 downto 0);
	signal TEMP_9_res_2 : std_logic_vector(1 downto 0);
	signal TEMP_9_res_3 : std_logic_vector(1 downto 0);
	signal TEMP_10_res_0 : std_logic_vector(1 downto 0);
	signal TEMP_10_res_1 : std_logic_vector(1 downto 0);
	signal entry_port_options_0 : std_logic_vector(1 downto 0);
	signal entry_port_options_1 : std_logic_vector(1 downto 0);
	signal entry_port_options_2 : std_logic_vector(1 downto 0);
	signal entry_port_options_3 : std_logic_vector(1 downto 0);
	signal entry_port_options_4 : std_logic_vector(1 downto 0);
	signal entry_port_options_5 : std_logic_vector(1 downto 0);
	signal entry_port_options_6 : std_logic_vector(1 downto 0);
	signal entry_port_options_7 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_2 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_3 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_4 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_5 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_6 : std_logic_vector(1 downto 0);
	signal entry_port_transfer_7 : std_logic_vector(1 downto 0);
	signal TEMP_11_double_in_0 : std_logic_vector(15 downto 0);
	signal TEMP_11_double_out_0 : std_logic_vector(15 downto 0);
	signal TEMP_11_double_in_1 : std_logic_vector(15 downto 0);
	signal TEMP_11_double_out_1 : std_logic_vector(15 downto 0);
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

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_4, entry_port_idx_4)
	entry_port_idx_oh_4(0) <= '1' when entry_port_idx_4_i = "0" else '0';
	entry_port_idx_oh_4(1) <= '1' when entry_port_idx_4_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_5, entry_port_idx_5)
	entry_port_idx_oh_5(0) <= '1' when entry_port_idx_5_i = "0" else '0';
	entry_port_idx_oh_5(1) <= '1' when entry_port_idx_5_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_6, entry_port_idx_6)
	entry_port_idx_oh_6(0) <= '1' when entry_port_idx_6_i = "0" else '0';
	entry_port_idx_oh_6(1) <= '1' when entry_port_idx_6_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_idx_oh_7, entry_port_idx_7)
	entry_port_idx_oh_7(0) <= '1' when entry_port_idx_7_i = "0" else '0';
	entry_port_idx_oh_7(1) <= '1' when entry_port_idx_7_i = "1" else '0';
	-- Bits To One-Hot End

	-- Mux1H Begin
	-- Mux1H(entry_payload_0, port_payload, entry_port_idx_oh_0)
	TEMP_1_mux_0 <= port_payload_0_i when entry_port_idx_oh_0(0) = '1' else "00000000000000000000000000000000";
	TEMP_1_mux_1 <= port_payload_1_i when entry_port_idx_oh_0(1) = '1' else "00000000000000000000000000000000";
	entry_payload_0_o <= TEMP_1_mux_0 or TEMP_1_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_1, port_payload, entry_port_idx_oh_1)
	TEMP_2_mux_0 <= port_payload_0_i when entry_port_idx_oh_1(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_1 <= port_payload_1_i when entry_port_idx_oh_1(1) = '1' else "00000000000000000000000000000000";
	entry_payload_1_o <= TEMP_2_mux_0 or TEMP_2_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_2, port_payload, entry_port_idx_oh_2)
	TEMP_3_mux_0 <= port_payload_0_i when entry_port_idx_oh_2(0) = '1' else "00000000000000000000000000000000";
	TEMP_3_mux_1 <= port_payload_1_i when entry_port_idx_oh_2(1) = '1' else "00000000000000000000000000000000";
	entry_payload_2_o <= TEMP_3_mux_0 or TEMP_3_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_3, port_payload, entry_port_idx_oh_3)
	TEMP_4_mux_0 <= port_payload_0_i when entry_port_idx_oh_3(0) = '1' else "00000000000000000000000000000000";
	TEMP_4_mux_1 <= port_payload_1_i when entry_port_idx_oh_3(1) = '1' else "00000000000000000000000000000000";
	entry_payload_3_o <= TEMP_4_mux_0 or TEMP_4_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_4, port_payload, entry_port_idx_oh_4)
	TEMP_5_mux_0 <= port_payload_0_i when entry_port_idx_oh_4(0) = '1' else "00000000000000000000000000000000";
	TEMP_5_mux_1 <= port_payload_1_i when entry_port_idx_oh_4(1) = '1' else "00000000000000000000000000000000";
	entry_payload_4_o <= TEMP_5_mux_0 or TEMP_5_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_5, port_payload, entry_port_idx_oh_5)
	TEMP_6_mux_0 <= port_payload_0_i when entry_port_idx_oh_5(0) = '1' else "00000000000000000000000000000000";
	TEMP_6_mux_1 <= port_payload_1_i when entry_port_idx_oh_5(1) = '1' else "00000000000000000000000000000000";
	entry_payload_5_o <= TEMP_6_mux_0 or TEMP_6_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_6, port_payload, entry_port_idx_oh_6)
	TEMP_7_mux_0 <= port_payload_0_i when entry_port_idx_oh_6(0) = '1' else "00000000000000000000000000000000";
	TEMP_7_mux_1 <= port_payload_1_i when entry_port_idx_oh_6(1) = '1' else "00000000000000000000000000000000";
	entry_payload_6_o <= TEMP_7_mux_0 or TEMP_7_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_7, port_payload, entry_port_idx_oh_7)
	TEMP_8_mux_0 <= port_payload_0_i when entry_port_idx_oh_7(0) = '1' else "00000000000000000000000000000000";
	TEMP_8_mux_1 <= port_payload_1_i when entry_port_idx_oh_7(1) = '1' else "00000000000000000000000000000000";
	entry_payload_7_o <= TEMP_8_mux_0 or TEMP_8_mux_1;
	-- Mux1H End

	entry_ptq_ready_0 <= entry_alloc_0_i and not entry_payload_valid_0_i;
	entry_ptq_ready_1 <= entry_alloc_1_i and not entry_payload_valid_1_i;
	entry_ptq_ready_2 <= entry_alloc_2_i and not entry_payload_valid_2_i;
	entry_ptq_ready_3 <= entry_alloc_3_i and not entry_payload_valid_3_i;
	entry_ptq_ready_4 <= entry_alloc_4_i and not entry_payload_valid_4_i;
	entry_ptq_ready_5 <= entry_alloc_5_i and not entry_payload_valid_5_i;
	entry_ptq_ready_6 <= entry_alloc_6_i and not entry_payload_valid_6_i;
	entry_ptq_ready_7 <= entry_alloc_7_i and not entry_payload_valid_7_i;
	entry_waiting_for_port_0 <= entry_port_idx_oh_0 when entry_ptq_ready_0 else "00";
	entry_waiting_for_port_1 <= entry_port_idx_oh_1 when entry_ptq_ready_1 else "00";
	entry_waiting_for_port_2 <= entry_port_idx_oh_2 when entry_ptq_ready_2 else "00";
	entry_waiting_for_port_3 <= entry_port_idx_oh_3 when entry_ptq_ready_3 else "00";
	entry_waiting_for_port_4 <= entry_port_idx_oh_4 when entry_ptq_ready_4 else "00";
	entry_waiting_for_port_5 <= entry_port_idx_oh_5 when entry_ptq_ready_5 else "00";
	entry_waiting_for_port_6 <= entry_port_idx_oh_6 when entry_ptq_ready_6 else "00";
	entry_waiting_for_port_7 <= entry_port_idx_oh_7 when entry_ptq_ready_7 else "00";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_waiting_for_port, or)
	TEMP_9_res_0 <= entry_waiting_for_port_0 or entry_waiting_for_port_4;
	TEMP_9_res_1 <= entry_waiting_for_port_1 or entry_waiting_for_port_5;
	TEMP_9_res_2 <= entry_waiting_for_port_2 or entry_waiting_for_port_6;
	TEMP_9_res_3 <= entry_waiting_for_port_3 or entry_waiting_for_port_7;
	-- Layer End
	TEMP_10_res_0 <= TEMP_9_res_0 or TEMP_9_res_2;
	TEMP_10_res_1 <= TEMP_9_res_1 or TEMP_9_res_3;
	-- Layer End
	port_ready_vec <= TEMP_10_res_0 or TEMP_10_res_1;
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
	entry_port_options_4(0) <= entry_waiting_for_port_4(0) and port_valid_0_i;
	entry_port_options_4(1) <= entry_waiting_for_port_4(1) and port_valid_1_i;
	entry_port_options_5(0) <= entry_waiting_for_port_5(0) and port_valid_0_i;
	entry_port_options_5(1) <= entry_waiting_for_port_5(1) and port_valid_1_i;
	entry_port_options_6(0) <= entry_waiting_for_port_6(0) and port_valid_0_i;
	entry_port_options_6(1) <= entry_waiting_for_port_6(1) and port_valid_1_i;
	entry_port_options_7(0) <= entry_waiting_for_port_7(0) and port_valid_0_i;
	entry_port_options_7(1) <= entry_waiting_for_port_7(1) and port_valid_1_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_transfer, entry_port_options, queue_head_oh)
	TEMP_11_double_in_0(0) <= entry_port_options_0(0);
	TEMP_11_double_in_0(8) <= entry_port_options_0(0);
	TEMP_11_double_in_0(1) <= entry_port_options_1(0);
	TEMP_11_double_in_0(9) <= entry_port_options_1(0);
	TEMP_11_double_in_0(2) <= entry_port_options_2(0);
	TEMP_11_double_in_0(10) <= entry_port_options_2(0);
	TEMP_11_double_in_0(3) <= entry_port_options_3(0);
	TEMP_11_double_in_0(11) <= entry_port_options_3(0);
	TEMP_11_double_in_0(4) <= entry_port_options_4(0);
	TEMP_11_double_in_0(12) <= entry_port_options_4(0);
	TEMP_11_double_in_0(5) <= entry_port_options_5(0);
	TEMP_11_double_in_0(13) <= entry_port_options_5(0);
	TEMP_11_double_in_0(6) <= entry_port_options_6(0);
	TEMP_11_double_in_0(14) <= entry_port_options_6(0);
	TEMP_11_double_in_0(7) <= entry_port_options_7(0);
	TEMP_11_double_in_0(15) <= entry_port_options_7(0);
	TEMP_11_double_out_0 <= TEMP_11_double_in_0 and not std_logic_vector( unsigned( TEMP_11_double_in_0 ) - unsigned( "00000000" & queue_head_oh_i ) );
	entry_port_transfer_0(0) <= TEMP_11_double_out_0(0) or TEMP_11_double_out_0(8);
	entry_port_transfer_1(0) <= TEMP_11_double_out_0(1) or TEMP_11_double_out_0(9);
	entry_port_transfer_2(0) <= TEMP_11_double_out_0(2) or TEMP_11_double_out_0(10);
	entry_port_transfer_3(0) <= TEMP_11_double_out_0(3) or TEMP_11_double_out_0(11);
	entry_port_transfer_4(0) <= TEMP_11_double_out_0(4) or TEMP_11_double_out_0(12);
	entry_port_transfer_5(0) <= TEMP_11_double_out_0(5) or TEMP_11_double_out_0(13);
	entry_port_transfer_6(0) <= TEMP_11_double_out_0(6) or TEMP_11_double_out_0(14);
	entry_port_transfer_7(0) <= TEMP_11_double_out_0(7) or TEMP_11_double_out_0(15);
	TEMP_11_double_in_1(0) <= entry_port_options_0(1);
	TEMP_11_double_in_1(8) <= entry_port_options_0(1);
	TEMP_11_double_in_1(1) <= entry_port_options_1(1);
	TEMP_11_double_in_1(9) <= entry_port_options_1(1);
	TEMP_11_double_in_1(2) <= entry_port_options_2(1);
	TEMP_11_double_in_1(10) <= entry_port_options_2(1);
	TEMP_11_double_in_1(3) <= entry_port_options_3(1);
	TEMP_11_double_in_1(11) <= entry_port_options_3(1);
	TEMP_11_double_in_1(4) <= entry_port_options_4(1);
	TEMP_11_double_in_1(12) <= entry_port_options_4(1);
	TEMP_11_double_in_1(5) <= entry_port_options_5(1);
	TEMP_11_double_in_1(13) <= entry_port_options_5(1);
	TEMP_11_double_in_1(6) <= entry_port_options_6(1);
	TEMP_11_double_in_1(14) <= entry_port_options_6(1);
	TEMP_11_double_in_1(7) <= entry_port_options_7(1);
	TEMP_11_double_in_1(15) <= entry_port_options_7(1);
	TEMP_11_double_out_1 <= TEMP_11_double_in_1 and not std_logic_vector( unsigned( TEMP_11_double_in_1 ) - unsigned( "00000000" & queue_head_oh_i ) );
	entry_port_transfer_0(1) <= TEMP_11_double_out_1(0) or TEMP_11_double_out_1(8);
	entry_port_transfer_1(1) <= TEMP_11_double_out_1(1) or TEMP_11_double_out_1(9);
	entry_port_transfer_2(1) <= TEMP_11_double_out_1(2) or TEMP_11_double_out_1(10);
	entry_port_transfer_3(1) <= TEMP_11_double_out_1(3) or TEMP_11_double_out_1(11);
	entry_port_transfer_4(1) <= TEMP_11_double_out_1(4) or TEMP_11_double_out_1(12);
	entry_port_transfer_5(1) <= TEMP_11_double_out_1(5) or TEMP_11_double_out_1(13);
	entry_port_transfer_6(1) <= TEMP_11_double_out_1(6) or TEMP_11_double_out_1(14);
	entry_port_transfer_7(1) <= TEMP_11_double_out_1(7) or TEMP_11_double_out_1(15);
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

	-- Reduction Begin
	-- Reduce(entry_wen_4, entry_port_transfer_4, or)
	entry_wen_4_o <= entry_port_transfer_4(0) or entry_port_transfer_4(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_5, entry_port_transfer_5, or)
	entry_wen_5_o <= entry_port_transfer_5(0) or entry_port_transfer_5(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_6, entry_port_transfer_6, or)
	entry_wen_6_o <= entry_port_transfer_6(0) or entry_port_transfer_6(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_7, entry_port_transfer_7, or)
	entry_wen_7_o <= entry_port_transfer_7(0) or entry_port_transfer_7(1);
	-- Reduction End


end architecture;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq4_core is
	port(
		rst : in std_logic;
		clk : in std_logic;
		group_init_valid_0_i : in std_logic;
		group_init_valid_1_i : in std_logic;
		group_init_valid_2_i : in std_logic;
		group_init_ready_0_o : out std_logic;
		group_init_ready_1_o : out std_logic;
		group_init_ready_2_o : out std_logic;
		ldp_addr_0_i : in std_logic_vector(6 downto 0);
		ldp_addr_1_i : in std_logic_vector(6 downto 0);
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
		stp_addr_0_i : in std_logic_vector(6 downto 0);
		stp_addr_1_i : in std_logic_vector(6 downto 0);
		stp_addr_valid_0_i : in std_logic;
		stp_addr_valid_1_i : in std_logic;
		stp_addr_ready_0_o : out std_logic;
		stp_addr_ready_1_o : out std_logic;
		stp_data_0_i : in std_logic_vector(31 downto 0);
		stp_data_1_i : in std_logic_vector(31 downto 0);
		stp_data_valid_0_i : in std_logic;
		stp_data_valid_1_i : in std_logic;
		stp_data_ready_0_o : out std_logic;
		stp_data_ready_1_o : out std_logic;
		empty_o : out std_logic;
		rreq_valid_0_o : out std_logic;
		rreq_ready_0_i : in std_logic;
		rreq_id_0_o : out std_logic_vector(3 downto 0);
		rreq_addr_0_o : out std_logic_vector(6 downto 0);
		rresp_valid_0_i : in std_logic;
		rresp_ready_0_o : out std_logic;
		rresp_id_0_i : in std_logic_vector(3 downto 0);
		rresp_data_0_i : in std_logic_vector(31 downto 0);
		wreq_valid_0_o : out std_logic;
		wreq_ready_0_i : in std_logic;
		wreq_id_0_o : out std_logic_vector(3 downto 0);
		wreq_addr_0_o : out std_logic_vector(6 downto 0);
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

architecture arch of handshake_lsq_lsq4_core is
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
	signal ldq_port_idx_0_d : std_logic_vector(0 downto 0);
	signal ldq_port_idx_0_q : std_logic_vector(0 downto 0);
	signal ldq_port_idx_1_d : std_logic_vector(0 downto 0);
	signal ldq_port_idx_1_q : std_logic_vector(0 downto 0);
	signal ldq_port_idx_2_d : std_logic_vector(0 downto 0);
	signal ldq_port_idx_2_q : std_logic_vector(0 downto 0);
	signal ldq_port_idx_3_d : std_logic_vector(0 downto 0);
	signal ldq_port_idx_3_q : std_logic_vector(0 downto 0);
	signal ldq_port_idx_4_d : std_logic_vector(0 downto 0);
	signal ldq_port_idx_4_q : std_logic_vector(0 downto 0);
	signal ldq_port_idx_5_d : std_logic_vector(0 downto 0);
	signal ldq_port_idx_5_q : std_logic_vector(0 downto 0);
	signal ldq_port_idx_6_d : std_logic_vector(0 downto 0);
	signal ldq_port_idx_6_q : std_logic_vector(0 downto 0);
	signal ldq_port_idx_7_d : std_logic_vector(0 downto 0);
	signal ldq_port_idx_7_q : std_logic_vector(0 downto 0);
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
	signal ldq_addr_0_d : std_logic_vector(6 downto 0);
	signal ldq_addr_0_q : std_logic_vector(6 downto 0);
	signal ldq_addr_1_d : std_logic_vector(6 downto 0);
	signal ldq_addr_1_q : std_logic_vector(6 downto 0);
	signal ldq_addr_2_d : std_logic_vector(6 downto 0);
	signal ldq_addr_2_q : std_logic_vector(6 downto 0);
	signal ldq_addr_3_d : std_logic_vector(6 downto 0);
	signal ldq_addr_3_q : std_logic_vector(6 downto 0);
	signal ldq_addr_4_d : std_logic_vector(6 downto 0);
	signal ldq_addr_4_q : std_logic_vector(6 downto 0);
	signal ldq_addr_5_d : std_logic_vector(6 downto 0);
	signal ldq_addr_5_q : std_logic_vector(6 downto 0);
	signal ldq_addr_6_d : std_logic_vector(6 downto 0);
	signal ldq_addr_6_q : std_logic_vector(6 downto 0);
	signal ldq_addr_7_d : std_logic_vector(6 downto 0);
	signal ldq_addr_7_q : std_logic_vector(6 downto 0);
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
	signal stq_port_idx_0_d : std_logic_vector(0 downto 0);
	signal stq_port_idx_0_q : std_logic_vector(0 downto 0);
	signal stq_port_idx_1_d : std_logic_vector(0 downto 0);
	signal stq_port_idx_1_q : std_logic_vector(0 downto 0);
	signal stq_port_idx_2_d : std_logic_vector(0 downto 0);
	signal stq_port_idx_2_q : std_logic_vector(0 downto 0);
	signal stq_port_idx_3_d : std_logic_vector(0 downto 0);
	signal stq_port_idx_3_q : std_logic_vector(0 downto 0);
	signal stq_port_idx_4_d : std_logic_vector(0 downto 0);
	signal stq_port_idx_4_q : std_logic_vector(0 downto 0);
	signal stq_port_idx_5_d : std_logic_vector(0 downto 0);
	signal stq_port_idx_5_q : std_logic_vector(0 downto 0);
	signal stq_port_idx_6_d : std_logic_vector(0 downto 0);
	signal stq_port_idx_6_q : std_logic_vector(0 downto 0);
	signal stq_port_idx_7_d : std_logic_vector(0 downto 0);
	signal stq_port_idx_7_q : std_logic_vector(0 downto 0);
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
	signal stq_addr_0_d : std_logic_vector(6 downto 0);
	signal stq_addr_0_q : std_logic_vector(6 downto 0);
	signal stq_addr_1_d : std_logic_vector(6 downto 0);
	signal stq_addr_1_q : std_logic_vector(6 downto 0);
	signal stq_addr_2_d : std_logic_vector(6 downto 0);
	signal stq_addr_2_q : std_logic_vector(6 downto 0);
	signal stq_addr_3_d : std_logic_vector(6 downto 0);
	signal stq_addr_3_q : std_logic_vector(6 downto 0);
	signal stq_addr_4_d : std_logic_vector(6 downto 0);
	signal stq_addr_4_q : std_logic_vector(6 downto 0);
	signal stq_addr_5_d : std_logic_vector(6 downto 0);
	signal stq_addr_5_q : std_logic_vector(6 downto 0);
	signal stq_addr_6_d : std_logic_vector(6 downto 0);
	signal stq_addr_6_q : std_logic_vector(6 downto 0);
	signal stq_addr_7_d : std_logic_vector(6 downto 0);
	signal stq_addr_7_q : std_logic_vector(6 downto 0);
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
	signal store_is_older_0_d : std_logic_vector(7 downto 0);
	signal store_is_older_0_q : std_logic_vector(7 downto 0);
	signal store_is_older_1_d : std_logic_vector(7 downto 0);
	signal store_is_older_1_q : std_logic_vector(7 downto 0);
	signal store_is_older_2_d : std_logic_vector(7 downto 0);
	signal store_is_older_2_q : std_logic_vector(7 downto 0);
	signal store_is_older_3_d : std_logic_vector(7 downto 0);
	signal store_is_older_3_q : std_logic_vector(7 downto 0);
	signal store_is_older_4_d : std_logic_vector(7 downto 0);
	signal store_is_older_4_q : std_logic_vector(7 downto 0);
	signal store_is_older_5_d : std_logic_vector(7 downto 0);
	signal store_is_older_5_q : std_logic_vector(7 downto 0);
	signal store_is_older_6_d : std_logic_vector(7 downto 0);
	signal store_is_older_6_q : std_logic_vector(7 downto 0);
	signal store_is_older_7_d : std_logic_vector(7 downto 0);
	signal store_is_older_7_q : std_logic_vector(7 downto 0);
	signal ldq_tail_d : std_logic_vector(2 downto 0);
	signal ldq_tail_q : std_logic_vector(2 downto 0);
	signal ldq_head_d : std_logic_vector(2 downto 0);
	signal ldq_head_q : std_logic_vector(2 downto 0);
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
	signal ldq_wen_2 : std_logic;
	signal ldq_wen_3 : std_logic;
	signal ldq_wen_4 : std_logic;
	signal ldq_wen_5 : std_logic;
	signal ldq_wen_6 : std_logic;
	signal ldq_wen_7 : std_logic;
	signal ldq_addr_wen_0 : std_logic;
	signal ldq_addr_wen_1 : std_logic;
	signal ldq_addr_wen_2 : std_logic;
	signal ldq_addr_wen_3 : std_logic;
	signal ldq_addr_wen_4 : std_logic;
	signal ldq_addr_wen_5 : std_logic;
	signal ldq_addr_wen_6 : std_logic;
	signal ldq_addr_wen_7 : std_logic;
	signal ldq_reset_0 : std_logic;
	signal ldq_reset_1 : std_logic;
	signal ldq_reset_2 : std_logic;
	signal ldq_reset_3 : std_logic;
	signal ldq_reset_4 : std_logic;
	signal ldq_reset_5 : std_logic;
	signal ldq_reset_6 : std_logic;
	signal ldq_reset_7 : std_logic;
	signal stq_wen_0 : std_logic;
	signal stq_wen_1 : std_logic;
	signal stq_wen_2 : std_logic;
	signal stq_wen_3 : std_logic;
	signal stq_wen_4 : std_logic;
	signal stq_wen_5 : std_logic;
	signal stq_wen_6 : std_logic;
	signal stq_wen_7 : std_logic;
	signal stq_addr_wen_0 : std_logic;
	signal stq_addr_wen_1 : std_logic;
	signal stq_addr_wen_2 : std_logic;
	signal stq_addr_wen_3 : std_logic;
	signal stq_addr_wen_4 : std_logic;
	signal stq_addr_wen_5 : std_logic;
	signal stq_addr_wen_6 : std_logic;
	signal stq_addr_wen_7 : std_logic;
	signal stq_data_wen_0 : std_logic;
	signal stq_data_wen_1 : std_logic;
	signal stq_data_wen_2 : std_logic;
	signal stq_data_wen_3 : std_logic;
	signal stq_data_wen_4 : std_logic;
	signal stq_data_wen_5 : std_logic;
	signal stq_data_wen_6 : std_logic;
	signal stq_data_wen_7 : std_logic;
	signal stq_reset_0 : std_logic;
	signal stq_reset_1 : std_logic;
	signal stq_reset_2 : std_logic;
	signal stq_reset_3 : std_logic;
	signal stq_reset_4 : std_logic;
	signal stq_reset_5 : std_logic;
	signal stq_reset_6 : std_logic;
	signal stq_reset_7 : std_logic;
	signal ldq_data_wen_0 : std_logic;
	signal ldq_data_wen_1 : std_logic;
	signal ldq_data_wen_2 : std_logic;
	signal ldq_data_wen_3 : std_logic;
	signal ldq_data_wen_4 : std_logic;
	signal ldq_data_wen_5 : std_logic;
	signal ldq_data_wen_6 : std_logic;
	signal ldq_data_wen_7 : std_logic;
	signal ldq_issue_set_0 : std_logic;
	signal ldq_issue_set_1 : std_logic;
	signal ldq_issue_set_2 : std_logic;
	signal ldq_issue_set_3 : std_logic;
	signal ldq_issue_set_4 : std_logic;
	signal ldq_issue_set_5 : std_logic;
	signal ldq_issue_set_6 : std_logic;
	signal ldq_issue_set_7 : std_logic;
	signal ga_ls_order_0 : std_logic_vector(7 downto 0);
	signal ga_ls_order_1 : std_logic_vector(7 downto 0);
	signal ga_ls_order_2 : std_logic_vector(7 downto 0);
	signal ga_ls_order_3 : std_logic_vector(7 downto 0);
	signal ga_ls_order_4 : std_logic_vector(7 downto 0);
	signal ga_ls_order_5 : std_logic_vector(7 downto 0);
	signal ga_ls_order_6 : std_logic_vector(7 downto 0);
	signal ga_ls_order_7 : std_logic_vector(7 downto 0);
	signal num_loads : std_logic_vector(2 downto 0);
	signal num_stores : std_logic_vector(2 downto 0);
	signal stq_issue_en : std_logic;
	signal stq_resp_en : std_logic;
	signal ldq_empty : std_logic;
	signal stq_empty : std_logic;
	signal ldq_head_oh : std_logic_vector(7 downto 0);
	signal stq_head_oh : std_logic_vector(7 downto 0);
	signal ldq_alloc_next_0 : std_logic;
	signal ldq_alloc_next_1 : std_logic;
	signal ldq_alloc_next_2 : std_logic;
	signal ldq_alloc_next_3 : std_logic;
	signal ldq_alloc_next_4 : std_logic;
	signal ldq_alloc_next_5 : std_logic;
	signal ldq_alloc_next_6 : std_logic;
	signal ldq_alloc_next_7 : std_logic;
	signal stq_alloc_next_0 : std_logic;
	signal stq_alloc_next_1 : std_logic;
	signal stq_alloc_next_2 : std_logic;
	signal stq_alloc_next_3 : std_logic;
	signal stq_alloc_next_4 : std_logic;
	signal stq_alloc_next_5 : std_logic;
	signal stq_alloc_next_6 : std_logic;
	signal stq_alloc_next_7 : std_logic;
	signal ldq_not_empty : std_logic;
	signal stq_not_empty : std_logic;
	signal TEMP_1_res_0 : std_logic;
	signal TEMP_1_res_1 : std_logic;
	signal TEMP_1_res_2 : std_logic;
	signal TEMP_1_res_3 : std_logic;
	signal TEMP_2_res_0 : std_logic;
	signal TEMP_2_res_1 : std_logic;
	signal ldq_tail_oh : std_logic_vector(7 downto 0);
	signal ldq_head_next_oh : std_logic_vector(7 downto 0);
	signal ldq_head_next : std_logic_vector(2 downto 0);
	signal ldq_head_sel : std_logic;
	signal TEMP_3_double_in : std_logic_vector(15 downto 0);
	signal TEMP_3_double_out : std_logic_vector(15 downto 0);
	signal TEMP_4_res_0 : std_logic;
	signal TEMP_4_res_1 : std_logic;
	signal TEMP_4_res_2 : std_logic;
	signal TEMP_4_res_3 : std_logic;
	signal TEMP_5_res_0 : std_logic;
	signal TEMP_5_res_1 : std_logic;
	signal TEMP_6_in_0_0 : std_logic;
	signal TEMP_6_in_0_1 : std_logic;
	signal TEMP_6_in_0_2 : std_logic;
	signal TEMP_6_in_0_3 : std_logic;
	signal TEMP_6_in_0_4 : std_logic;
	signal TEMP_6_in_0_5 : std_logic;
	signal TEMP_6_in_0_6 : std_logic;
	signal TEMP_6_in_0_7 : std_logic;
	signal TEMP_6_out_0 : std_logic;
	signal TEMP_7_res_0 : std_logic;
	signal TEMP_7_res_1 : std_logic;
	signal TEMP_7_res_2 : std_logic;
	signal TEMP_7_res_3 : std_logic;
	signal TEMP_8_res_0 : std_logic;
	signal TEMP_8_res_1 : std_logic;
	signal TEMP_8_in_1_0 : std_logic;
	signal TEMP_8_in_1_1 : std_logic;
	signal TEMP_8_in_1_2 : std_logic;
	signal TEMP_8_in_1_3 : std_logic;
	signal TEMP_8_in_1_4 : std_logic;
	signal TEMP_8_in_1_5 : std_logic;
	signal TEMP_8_in_1_6 : std_logic;
	signal TEMP_8_in_1_7 : std_logic;
	signal TEMP_8_out_1 : std_logic;
	signal TEMP_9_res_0 : std_logic;
	signal TEMP_9_res_1 : std_logic;
	signal TEMP_9_res_2 : std_logic;
	signal TEMP_9_res_3 : std_logic;
	signal TEMP_10_res_0 : std_logic;
	signal TEMP_10_res_1 : std_logic;
	signal TEMP_10_in_2_0 : std_logic;
	signal TEMP_10_in_2_1 : std_logic;
	signal TEMP_10_in_2_2 : std_logic;
	signal TEMP_10_in_2_3 : std_logic;
	signal TEMP_10_in_2_4 : std_logic;
	signal TEMP_10_in_2_5 : std_logic;
	signal TEMP_10_in_2_6 : std_logic;
	signal TEMP_10_in_2_7 : std_logic;
	signal TEMP_10_out_2 : std_logic;
	signal TEMP_11_res_0 : std_logic;
	signal TEMP_11_res_1 : std_logic;
	signal TEMP_11_res_2 : std_logic;
	signal TEMP_11_res_3 : std_logic;
	signal TEMP_12_res_0 : std_logic;
	signal TEMP_12_res_1 : std_logic;
	signal stq_tail_oh : std_logic_vector(7 downto 0);
	signal stq_head_next_oh : std_logic_vector(7 downto 0);
	signal stq_head_next : std_logic_vector(2 downto 0);
	signal stq_head_sel : std_logic;
	signal load_idx_oh_0 : std_logic_vector(7 downto 0);
	signal load_en_0 : std_logic;
	signal store_idx : std_logic_vector(2 downto 0);
	signal store_en : std_logic;
	signal bypass_idx_oh_0 : std_logic_vector(7 downto 0);
	signal bypass_idx_oh_1 : std_logic_vector(7 downto 0);
	signal bypass_idx_oh_2 : std_logic_vector(7 downto 0);
	signal bypass_idx_oh_3 : std_logic_vector(7 downto 0);
	signal bypass_idx_oh_4 : std_logic_vector(7 downto 0);
	signal bypass_idx_oh_5 : std_logic_vector(7 downto 0);
	signal bypass_idx_oh_6 : std_logic_vector(7 downto 0);
	signal bypass_idx_oh_7 : std_logic_vector(7 downto 0);
	signal bypass_en_0 : std_logic;
	signal bypass_en_1 : std_logic;
	signal bypass_en_2 : std_logic;
	signal bypass_en_3 : std_logic;
	signal bypass_en_4 : std_logic;
	signal bypass_en_5 : std_logic;
	signal bypass_en_6 : std_logic;
	signal bypass_en_7 : std_logic;
	signal ld_st_conflict_0 : std_logic_vector(7 downto 0);
	signal ld_st_conflict_1 : std_logic_vector(7 downto 0);
	signal ld_st_conflict_2 : std_logic_vector(7 downto 0);
	signal ld_st_conflict_3 : std_logic_vector(7 downto 0);
	signal ld_st_conflict_4 : std_logic_vector(7 downto 0);
	signal ld_st_conflict_5 : std_logic_vector(7 downto 0);
	signal ld_st_conflict_6 : std_logic_vector(7 downto 0);
	signal ld_st_conflict_7 : std_logic_vector(7 downto 0);
	signal can_bypass_0 : std_logic_vector(7 downto 0);
	signal can_bypass_1 : std_logic_vector(7 downto 0);
	signal can_bypass_2 : std_logic_vector(7 downto 0);
	signal can_bypass_3 : std_logic_vector(7 downto 0);
	signal can_bypass_4 : std_logic_vector(7 downto 0);
	signal can_bypass_5 : std_logic_vector(7 downto 0);
	signal can_bypass_6 : std_logic_vector(7 downto 0);
	signal can_bypass_7 : std_logic_vector(7 downto 0);
	signal addr_valid_0 : std_logic_vector(7 downto 0);
	signal addr_valid_1 : std_logic_vector(7 downto 0);
	signal addr_valid_2 : std_logic_vector(7 downto 0);
	signal addr_valid_3 : std_logic_vector(7 downto 0);
	signal addr_valid_4 : std_logic_vector(7 downto 0);
	signal addr_valid_5 : std_logic_vector(7 downto 0);
	signal addr_valid_6 : std_logic_vector(7 downto 0);
	signal addr_valid_7 : std_logic_vector(7 downto 0);
	signal addr_same_0 : std_logic_vector(7 downto 0);
	signal addr_same_1 : std_logic_vector(7 downto 0);
	signal addr_same_2 : std_logic_vector(7 downto 0);
	signal addr_same_3 : std_logic_vector(7 downto 0);
	signal addr_same_4 : std_logic_vector(7 downto 0);
	signal addr_same_5 : std_logic_vector(7 downto 0);
	signal addr_same_6 : std_logic_vector(7 downto 0);
	signal addr_same_7 : std_logic_vector(7 downto 0);
	signal load_conflict_0 : std_logic;
	signal load_conflict_1 : std_logic;
	signal load_conflict_2 : std_logic;
	signal load_conflict_3 : std_logic;
	signal load_conflict_4 : std_logic;
	signal load_conflict_5 : std_logic;
	signal load_conflict_6 : std_logic;
	signal load_conflict_7 : std_logic;
	signal load_req_valid_0 : std_logic;
	signal load_req_valid_1 : std_logic;
	signal load_req_valid_2 : std_logic;
	signal load_req_valid_3 : std_logic;
	signal load_req_valid_4 : std_logic;
	signal load_req_valid_5 : std_logic;
	signal load_req_valid_6 : std_logic;
	signal load_req_valid_7 : std_logic;
	signal can_load_0 : std_logic;
	signal can_load_1 : std_logic;
	signal can_load_2 : std_logic;
	signal can_load_3 : std_logic;
	signal can_load_4 : std_logic;
	signal can_load_5 : std_logic;
	signal can_load_6 : std_logic;
	signal can_load_7 : std_logic;
	signal TEMP_13_res : std_logic_vector(3 downto 0);
	signal TEMP_14_res : std_logic_vector(1 downto 0);
	signal TEMP_15_res : std_logic_vector(3 downto 0);
	signal TEMP_16_res : std_logic_vector(1 downto 0);
	signal TEMP_17_res : std_logic_vector(3 downto 0);
	signal TEMP_18_res : std_logic_vector(1 downto 0);
	signal TEMP_19_res : std_logic_vector(3 downto 0);
	signal TEMP_20_res : std_logic_vector(1 downto 0);
	signal TEMP_21_res : std_logic_vector(3 downto 0);
	signal TEMP_22_res : std_logic_vector(1 downto 0);
	signal TEMP_23_res : std_logic_vector(3 downto 0);
	signal TEMP_24_res : std_logic_vector(1 downto 0);
	signal TEMP_25_res : std_logic_vector(3 downto 0);
	signal TEMP_26_res : std_logic_vector(1 downto 0);
	signal TEMP_27_res : std_logic_vector(3 downto 0);
	signal TEMP_28_res : std_logic_vector(1 downto 0);
	signal TEMP_29_double_in : std_logic_vector(15 downto 0);
	signal TEMP_29_double_out : std_logic_vector(15 downto 0);
	signal TEMP_30_res_0 : std_logic;
	signal TEMP_30_res_1 : std_logic;
	signal TEMP_30_res_2 : std_logic;
	signal TEMP_30_res_3 : std_logic;
	signal TEMP_31_res_0 : std_logic;
	signal TEMP_31_res_1 : std_logic;
	signal st_ld_conflict : std_logic_vector(7 downto 0);
	signal store_conflict : std_logic;
	signal store_valid : std_logic;
	signal store_data_valid : std_logic;
	signal store_addr_valid : std_logic;
	signal TEMP_32_res : std_logic_vector(3 downto 0);
	signal TEMP_33_res : std_logic_vector(1 downto 0);
	signal stq_last_oh : std_logic_vector(7 downto 0);
	signal bypass_en_vec_0 : std_logic_vector(7 downto 0);
	signal TEMP_34_double_in : std_logic_vector(15 downto 0);
	signal TEMP_34_base_rev : std_logic_vector(7 downto 0);
	signal TEMP_34_double_out : std_logic_vector(15 downto 0);
	signal TEMP_35_res : std_logic_vector(3 downto 0);
	signal TEMP_36_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_1 : std_logic_vector(7 downto 0);
	signal TEMP_37_double_in : std_logic_vector(15 downto 0);
	signal TEMP_37_base_rev : std_logic_vector(7 downto 0);
	signal TEMP_37_double_out : std_logic_vector(15 downto 0);
	signal TEMP_38_res : std_logic_vector(3 downto 0);
	signal TEMP_39_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_2 : std_logic_vector(7 downto 0);
	signal TEMP_40_double_in : std_logic_vector(15 downto 0);
	signal TEMP_40_base_rev : std_logic_vector(7 downto 0);
	signal TEMP_40_double_out : std_logic_vector(15 downto 0);
	signal TEMP_41_res : std_logic_vector(3 downto 0);
	signal TEMP_42_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_3 : std_logic_vector(7 downto 0);
	signal TEMP_43_double_in : std_logic_vector(15 downto 0);
	signal TEMP_43_base_rev : std_logic_vector(7 downto 0);
	signal TEMP_43_double_out : std_logic_vector(15 downto 0);
	signal TEMP_44_res : std_logic_vector(3 downto 0);
	signal TEMP_45_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_4 : std_logic_vector(7 downto 0);
	signal TEMP_46_double_in : std_logic_vector(15 downto 0);
	signal TEMP_46_base_rev : std_logic_vector(7 downto 0);
	signal TEMP_46_double_out : std_logic_vector(15 downto 0);
	signal TEMP_47_res : std_logic_vector(3 downto 0);
	signal TEMP_48_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_5 : std_logic_vector(7 downto 0);
	signal TEMP_49_double_in : std_logic_vector(15 downto 0);
	signal TEMP_49_base_rev : std_logic_vector(7 downto 0);
	signal TEMP_49_double_out : std_logic_vector(15 downto 0);
	signal TEMP_50_res : std_logic_vector(3 downto 0);
	signal TEMP_51_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_6 : std_logic_vector(7 downto 0);
	signal TEMP_52_double_in : std_logic_vector(15 downto 0);
	signal TEMP_52_base_rev : std_logic_vector(7 downto 0);
	signal TEMP_52_double_out : std_logic_vector(15 downto 0);
	signal TEMP_53_res : std_logic_vector(3 downto 0);
	signal TEMP_54_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_7 : std_logic_vector(7 downto 0);
	signal TEMP_55_double_in : std_logic_vector(15 downto 0);
	signal TEMP_55_base_rev : std_logic_vector(7 downto 0);
	signal TEMP_55_double_out : std_logic_vector(15 downto 0);
	signal TEMP_56_res : std_logic_vector(3 downto 0);
	signal TEMP_57_res : std_logic_vector(1 downto 0);
	signal TEMP_58_in_0_0 : std_logic;
	signal TEMP_58_in_0_1 : std_logic;
	signal TEMP_58_in_0_2 : std_logic;
	signal TEMP_58_in_0_3 : std_logic;
	signal TEMP_58_in_0_4 : std_logic;
	signal TEMP_58_in_0_5 : std_logic;
	signal TEMP_58_in_0_6 : std_logic;
	signal TEMP_58_in_0_7 : std_logic;
	signal TEMP_58_out_0 : std_logic;
	signal TEMP_59_res_0 : std_logic;
	signal TEMP_59_res_1 : std_logic;
	signal TEMP_59_res_2 : std_logic;
	signal TEMP_59_res_3 : std_logic;
	signal TEMP_60_res_0 : std_logic;
	signal TEMP_60_res_1 : std_logic;
	signal TEMP_60_in_1_0 : std_logic;
	signal TEMP_60_in_1_1 : std_logic;
	signal TEMP_60_in_1_2 : std_logic;
	signal TEMP_60_in_1_3 : std_logic;
	signal TEMP_60_in_1_4 : std_logic;
	signal TEMP_60_in_1_5 : std_logic;
	signal TEMP_60_in_1_6 : std_logic;
	signal TEMP_60_in_1_7 : std_logic;
	signal TEMP_60_out_1 : std_logic;
	signal TEMP_61_res_0 : std_logic;
	signal TEMP_61_res_1 : std_logic;
	signal TEMP_61_res_2 : std_logic;
	signal TEMP_61_res_3 : std_logic;
	signal TEMP_62_res_0 : std_logic;
	signal TEMP_62_res_1 : std_logic;
	signal TEMP_62_in_2_0 : std_logic;
	signal TEMP_62_in_2_1 : std_logic;
	signal TEMP_62_in_2_2 : std_logic;
	signal TEMP_62_in_2_3 : std_logic;
	signal TEMP_62_in_2_4 : std_logic;
	signal TEMP_62_in_2_5 : std_logic;
	signal TEMP_62_in_2_6 : std_logic;
	signal TEMP_62_in_2_7 : std_logic;
	signal TEMP_62_out_2 : std_logic;
	signal TEMP_63_res_0 : std_logic;
	signal TEMP_63_res_1 : std_logic;
	signal TEMP_63_res_2 : std_logic;
	signal TEMP_63_res_3 : std_logic;
	signal TEMP_64_res_0 : std_logic;
	signal TEMP_64_res_1 : std_logic;
	signal TEMP_64_in_3_0 : std_logic;
	signal TEMP_64_in_3_1 : std_logic;
	signal TEMP_64_in_3_2 : std_logic;
	signal TEMP_64_in_3_3 : std_logic;
	signal TEMP_64_in_3_4 : std_logic;
	signal TEMP_64_in_3_5 : std_logic;
	signal TEMP_64_in_3_6 : std_logic;
	signal TEMP_64_in_3_7 : std_logic;
	signal TEMP_64_out_3 : std_logic;
	signal TEMP_65_res_0 : std_logic;
	signal TEMP_65_res_1 : std_logic;
	signal TEMP_65_res_2 : std_logic;
	signal TEMP_65_res_3 : std_logic;
	signal TEMP_66_res_0 : std_logic;
	signal TEMP_66_res_1 : std_logic;
	signal TEMP_67_mux_0 : std_logic_vector(6 downto 0);
	signal TEMP_67_mux_1 : std_logic_vector(6 downto 0);
	signal TEMP_67_mux_2 : std_logic_vector(6 downto 0);
	signal TEMP_67_mux_3 : std_logic_vector(6 downto 0);
	signal TEMP_67_mux_4 : std_logic_vector(6 downto 0);
	signal TEMP_67_mux_5 : std_logic_vector(6 downto 0);
	signal TEMP_67_mux_6 : std_logic_vector(6 downto 0);
	signal TEMP_67_mux_7 : std_logic_vector(6 downto 0);
	signal TEMP_68_res_0 : std_logic_vector(6 downto 0);
	signal TEMP_68_res_1 : std_logic_vector(6 downto 0);
	signal TEMP_68_res_2 : std_logic_vector(6 downto 0);
	signal TEMP_68_res_3 : std_logic_vector(6 downto 0);
	signal TEMP_69_res_0 : std_logic_vector(6 downto 0);
	signal TEMP_69_res_1 : std_logic_vector(6 downto 0);
	signal ldq_issue_set_vec_0 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_1 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_2 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_3 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_4 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_5 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_6 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_7 : std_logic_vector(0 downto 0);
	signal read_idx_oh_0_0 : std_logic;
	signal read_valid_0 : std_logic;
	signal read_data_0 : std_logic_vector(31 downto 0);
	signal TEMP_70_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_0 : std_logic_vector(31 downto 0);
	signal TEMP_71_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_71_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_71_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_71_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_71_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_71_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_71_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_71_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_72_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_72_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_72_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_72_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_73_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_73_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_1_0 : std_logic;
	signal read_valid_1 : std_logic;
	signal read_data_1 : std_logic_vector(31 downto 0);
	signal TEMP_74_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_1 : std_logic_vector(31 downto 0);
	signal TEMP_75_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_75_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_75_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_75_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_75_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_75_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_75_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_75_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_76_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_76_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_76_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_76_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_77_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_77_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_2_0 : std_logic;
	signal read_valid_2 : std_logic;
	signal read_data_2 : std_logic_vector(31 downto 0);
	signal TEMP_78_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_2 : std_logic_vector(31 downto 0);
	signal TEMP_79_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_79_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_79_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_79_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_79_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_79_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_79_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_79_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_80_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_80_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_80_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_80_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_81_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_81_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_3_0 : std_logic;
	signal read_valid_3 : std_logic;
	signal read_data_3 : std_logic_vector(31 downto 0);
	signal TEMP_82_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_3 : std_logic_vector(31 downto 0);
	signal TEMP_83_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_83_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_83_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_83_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_83_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_83_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_83_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_83_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_84_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_84_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_84_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_84_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_85_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_85_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_4_0 : std_logic;
	signal read_valid_4 : std_logic;
	signal read_data_4 : std_logic_vector(31 downto 0);
	signal TEMP_86_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_4 : std_logic_vector(31 downto 0);
	signal TEMP_87_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_87_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_87_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_87_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_87_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_87_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_87_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_87_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_88_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_88_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_88_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_88_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_89_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_89_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_5_0 : std_logic;
	signal read_valid_5 : std_logic;
	signal read_data_5 : std_logic_vector(31 downto 0);
	signal TEMP_90_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_5 : std_logic_vector(31 downto 0);
	signal TEMP_91_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_91_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_91_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_91_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_91_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_91_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_91_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_91_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_92_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_92_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_92_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_92_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_93_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_93_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_6_0 : std_logic;
	signal read_valid_6 : std_logic;
	signal read_data_6 : std_logic_vector(31 downto 0);
	signal TEMP_94_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_6 : std_logic_vector(31 downto 0);
	signal TEMP_95_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_95_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_95_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_95_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_95_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_95_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_95_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_95_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_96_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_96_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_96_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_96_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_97_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_97_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_7_0 : std_logic;
	signal read_valid_7 : std_logic;
	signal read_data_7 : std_logic_vector(31 downto 0);
	signal TEMP_98_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_7 : std_logic_vector(31 downto 0);
	signal TEMP_99_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_99_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_99_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_99_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_99_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_99_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_99_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_99_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_100_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_100_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_100_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_100_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_101_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_101_res_1 : std_logic_vector(31 downto 0);
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
	ldq_head_oh(0) <= '1' when ldq_head_q = "000" else '0';
	ldq_head_oh(1) <= '1' when ldq_head_q = "001" else '0';
	ldq_head_oh(2) <= '1' when ldq_head_q = "010" else '0';
	ldq_head_oh(3) <= '1' when ldq_head_q = "011" else '0';
	ldq_head_oh(4) <= '1' when ldq_head_q = "100" else '0';
	ldq_head_oh(5) <= '1' when ldq_head_q = "101" else '0';
	ldq_head_oh(6) <= '1' when ldq_head_q = "110" else '0';
	ldq_head_oh(7) <= '1' when ldq_head_q = "111" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(stq_head_oh, stq_head)
	stq_head_oh(0) <= '1' when stq_head_q = "000" else '0';
	stq_head_oh(1) <= '1' when stq_head_q = "001" else '0';
	stq_head_oh(2) <= '1' when stq_head_q = "010" else '0';
	stq_head_oh(3) <= '1' when stq_head_q = "011" else '0';
	stq_head_oh(4) <= '1' when stq_head_q = "100" else '0';
	stq_head_oh(5) <= '1' when stq_head_q = "101" else '0';
	stq_head_oh(6) <= '1' when stq_head_q = "110" else '0';
	stq_head_oh(7) <= '1' when stq_head_q = "111" else '0';
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
	store_is_older_0_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_0(0) ) ) when ldq_wen_0 else not stq_reset_0 and store_is_older_0_q(0);
	store_is_older_0_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_0(1) ) ) when ldq_wen_0 else not stq_reset_1 and store_is_older_0_q(1);
	store_is_older_0_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_0(2) ) ) when ldq_wen_0 else not stq_reset_2 and store_is_older_0_q(2);
	store_is_older_0_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_0(3) ) ) when ldq_wen_0 else not stq_reset_3 and store_is_older_0_q(3);
	store_is_older_0_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_0(4) ) ) when ldq_wen_0 else not stq_reset_4 and store_is_older_0_q(4);
	store_is_older_0_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_0(5) ) ) when ldq_wen_0 else not stq_reset_5 and store_is_older_0_q(5);
	store_is_older_0_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_0(6) ) ) when ldq_wen_0 else not stq_reset_6 and store_is_older_0_q(6);
	store_is_older_0_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_0(7) ) ) when ldq_wen_0 else not stq_reset_7 and store_is_older_0_q(7);
	store_is_older_1_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_1(0) ) ) when ldq_wen_1 else not stq_reset_0 and store_is_older_1_q(0);
	store_is_older_1_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_1(1) ) ) when ldq_wen_1 else not stq_reset_1 and store_is_older_1_q(1);
	store_is_older_1_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_1(2) ) ) when ldq_wen_1 else not stq_reset_2 and store_is_older_1_q(2);
	store_is_older_1_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_1(3) ) ) when ldq_wen_1 else not stq_reset_3 and store_is_older_1_q(3);
	store_is_older_1_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_1(4) ) ) when ldq_wen_1 else not stq_reset_4 and store_is_older_1_q(4);
	store_is_older_1_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_1(5) ) ) when ldq_wen_1 else not stq_reset_5 and store_is_older_1_q(5);
	store_is_older_1_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_1(6) ) ) when ldq_wen_1 else not stq_reset_6 and store_is_older_1_q(6);
	store_is_older_1_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_1(7) ) ) when ldq_wen_1 else not stq_reset_7 and store_is_older_1_q(7);
	store_is_older_2_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_2(0) ) ) when ldq_wen_2 else not stq_reset_0 and store_is_older_2_q(0);
	store_is_older_2_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_2(1) ) ) when ldq_wen_2 else not stq_reset_1 and store_is_older_2_q(1);
	store_is_older_2_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_2(2) ) ) when ldq_wen_2 else not stq_reset_2 and store_is_older_2_q(2);
	store_is_older_2_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_2(3) ) ) when ldq_wen_2 else not stq_reset_3 and store_is_older_2_q(3);
	store_is_older_2_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_2(4) ) ) when ldq_wen_2 else not stq_reset_4 and store_is_older_2_q(4);
	store_is_older_2_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_2(5) ) ) when ldq_wen_2 else not stq_reset_5 and store_is_older_2_q(5);
	store_is_older_2_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_2(6) ) ) when ldq_wen_2 else not stq_reset_6 and store_is_older_2_q(6);
	store_is_older_2_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_2(7) ) ) when ldq_wen_2 else not stq_reset_7 and store_is_older_2_q(7);
	store_is_older_3_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_3(0) ) ) when ldq_wen_3 else not stq_reset_0 and store_is_older_3_q(0);
	store_is_older_3_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_3(1) ) ) when ldq_wen_3 else not stq_reset_1 and store_is_older_3_q(1);
	store_is_older_3_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_3(2) ) ) when ldq_wen_3 else not stq_reset_2 and store_is_older_3_q(2);
	store_is_older_3_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_3(3) ) ) when ldq_wen_3 else not stq_reset_3 and store_is_older_3_q(3);
	store_is_older_3_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_3(4) ) ) when ldq_wen_3 else not stq_reset_4 and store_is_older_3_q(4);
	store_is_older_3_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_3(5) ) ) when ldq_wen_3 else not stq_reset_5 and store_is_older_3_q(5);
	store_is_older_3_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_3(6) ) ) when ldq_wen_3 else not stq_reset_6 and store_is_older_3_q(6);
	store_is_older_3_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_3(7) ) ) when ldq_wen_3 else not stq_reset_7 and store_is_older_3_q(7);
	store_is_older_4_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_4(0) ) ) when ldq_wen_4 else not stq_reset_0 and store_is_older_4_q(0);
	store_is_older_4_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_4(1) ) ) when ldq_wen_4 else not stq_reset_1 and store_is_older_4_q(1);
	store_is_older_4_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_4(2) ) ) when ldq_wen_4 else not stq_reset_2 and store_is_older_4_q(2);
	store_is_older_4_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_4(3) ) ) when ldq_wen_4 else not stq_reset_3 and store_is_older_4_q(3);
	store_is_older_4_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_4(4) ) ) when ldq_wen_4 else not stq_reset_4 and store_is_older_4_q(4);
	store_is_older_4_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_4(5) ) ) when ldq_wen_4 else not stq_reset_5 and store_is_older_4_q(5);
	store_is_older_4_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_4(6) ) ) when ldq_wen_4 else not stq_reset_6 and store_is_older_4_q(6);
	store_is_older_4_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_4(7) ) ) when ldq_wen_4 else not stq_reset_7 and store_is_older_4_q(7);
	store_is_older_5_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_5(0) ) ) when ldq_wen_5 else not stq_reset_0 and store_is_older_5_q(0);
	store_is_older_5_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_5(1) ) ) when ldq_wen_5 else not stq_reset_1 and store_is_older_5_q(1);
	store_is_older_5_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_5(2) ) ) when ldq_wen_5 else not stq_reset_2 and store_is_older_5_q(2);
	store_is_older_5_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_5(3) ) ) when ldq_wen_5 else not stq_reset_3 and store_is_older_5_q(3);
	store_is_older_5_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_5(4) ) ) when ldq_wen_5 else not stq_reset_4 and store_is_older_5_q(4);
	store_is_older_5_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_5(5) ) ) when ldq_wen_5 else not stq_reset_5 and store_is_older_5_q(5);
	store_is_older_5_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_5(6) ) ) when ldq_wen_5 else not stq_reset_6 and store_is_older_5_q(6);
	store_is_older_5_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_5(7) ) ) when ldq_wen_5 else not stq_reset_7 and store_is_older_5_q(7);
	store_is_older_6_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_6(0) ) ) when ldq_wen_6 else not stq_reset_0 and store_is_older_6_q(0);
	store_is_older_6_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_6(1) ) ) when ldq_wen_6 else not stq_reset_1 and store_is_older_6_q(1);
	store_is_older_6_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_6(2) ) ) when ldq_wen_6 else not stq_reset_2 and store_is_older_6_q(2);
	store_is_older_6_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_6(3) ) ) when ldq_wen_6 else not stq_reset_3 and store_is_older_6_q(3);
	store_is_older_6_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_6(4) ) ) when ldq_wen_6 else not stq_reset_4 and store_is_older_6_q(4);
	store_is_older_6_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_6(5) ) ) when ldq_wen_6 else not stq_reset_5 and store_is_older_6_q(5);
	store_is_older_6_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_6(6) ) ) when ldq_wen_6 else not stq_reset_6 and store_is_older_6_q(6);
	store_is_older_6_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_6(7) ) ) when ldq_wen_6 else not stq_reset_7 and store_is_older_6_q(7);
	store_is_older_7_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_7(0) ) ) when ldq_wen_7 else not stq_reset_0 and store_is_older_7_q(0);
	store_is_older_7_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_7(1) ) ) when ldq_wen_7 else not stq_reset_1 and store_is_older_7_q(1);
	store_is_older_7_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_7(2) ) ) when ldq_wen_7 else not stq_reset_2 and store_is_older_7_q(2);
	store_is_older_7_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_7(3) ) ) when ldq_wen_7 else not stq_reset_3 and store_is_older_7_q(3);
	store_is_older_7_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_7(4) ) ) when ldq_wen_7 else not stq_reset_4 and store_is_older_7_q(4);
	store_is_older_7_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_7(5) ) ) when ldq_wen_7 else not stq_reset_5 and store_is_older_7_q(5);
	store_is_older_7_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_7(6) ) ) when ldq_wen_7 else not stq_reset_6 and store_is_older_7_q(6);
	store_is_older_7_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_7(7) ) ) when ldq_wen_7 else not stq_reset_7 and store_is_older_7_q(7);
	-- Reduction Begin
	-- Reduce(ldq_not_empty, ldq_alloc, or)
	TEMP_1_res_0 <= ldq_alloc_0_q or ldq_alloc_4_q;
	TEMP_1_res_1 <= ldq_alloc_1_q or ldq_alloc_5_q;
	TEMP_1_res_2 <= ldq_alloc_2_q or ldq_alloc_6_q;
	TEMP_1_res_3 <= ldq_alloc_3_q or ldq_alloc_7_q;
	-- Layer End
	TEMP_2_res_0 <= TEMP_1_res_0 or TEMP_1_res_2;
	TEMP_2_res_1 <= TEMP_1_res_1 or TEMP_1_res_3;
	-- Layer End
	ldq_not_empty <= TEMP_2_res_0 or TEMP_2_res_1;
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
	stq_alloc_5_q when (stq_head_q = "101") else
	stq_alloc_6_q when (stq_head_q = "110") else
	stq_alloc_7_q when (stq_head_q = "111") else
	'0';
	-- MuxLookUp End

	stq_empty <= not stq_not_empty;
	empty_o <= ldq_empty and stq_empty;
	-- WrapAdd Begin
	-- WrapAdd(ldq_tail, ldq_tail, num_loads, 8)
	ldq_tail_d <= std_logic_vector(unsigned(ldq_tail_q) + unsigned(num_loads));
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_tail, stq_tail, num_stores, 8)
	stq_tail_d <= std_logic_vector(unsigned(stq_tail_q) + unsigned(num_stores));
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_issue, stq_issue, 1, 8)
	stq_issue_d <= std_logic_vector(unsigned(stq_issue_q) + 1);
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_resp, stq_resp, 1, 8)
	stq_resp_d <= std_logic_vector(unsigned(stq_resp_q) + 1);
	-- WrapAdd End

	-- Bits To One-Hot Begin
	-- BitsToOH(ldq_tail_oh, ldq_tail)
	ldq_tail_oh(0) <= '1' when ldq_tail_q = "000" else '0';
	ldq_tail_oh(1) <= '1' when ldq_tail_q = "001" else '0';
	ldq_tail_oh(2) <= '1' when ldq_tail_q = "010" else '0';
	ldq_tail_oh(3) <= '1' when ldq_tail_q = "011" else '0';
	ldq_tail_oh(4) <= '1' when ldq_tail_q = "100" else '0';
	ldq_tail_oh(5) <= '1' when ldq_tail_q = "101" else '0';
	ldq_tail_oh(6) <= '1' when ldq_tail_q = "110" else '0';
	ldq_tail_oh(7) <= '1' when ldq_tail_q = "111" else '0';
	-- Bits To One-Hot End

	-- Priority Masking Begin
	-- CyclicPriorityMask(ldq_head_next_oh, ldq_alloc_next, ldq_tail_oh)
	TEMP_3_double_in(0) <= ldq_alloc_next_0;
	TEMP_3_double_in(8) <= ldq_alloc_next_0;
	TEMP_3_double_in(1) <= ldq_alloc_next_1;
	TEMP_3_double_in(9) <= ldq_alloc_next_1;
	TEMP_3_double_in(2) <= ldq_alloc_next_2;
	TEMP_3_double_in(10) <= ldq_alloc_next_2;
	TEMP_3_double_in(3) <= ldq_alloc_next_3;
	TEMP_3_double_in(11) <= ldq_alloc_next_3;
	TEMP_3_double_in(4) <= ldq_alloc_next_4;
	TEMP_3_double_in(12) <= ldq_alloc_next_4;
	TEMP_3_double_in(5) <= ldq_alloc_next_5;
	TEMP_3_double_in(13) <= ldq_alloc_next_5;
	TEMP_3_double_in(6) <= ldq_alloc_next_6;
	TEMP_3_double_in(14) <= ldq_alloc_next_6;
	TEMP_3_double_in(7) <= ldq_alloc_next_7;
	TEMP_3_double_in(15) <= ldq_alloc_next_7;
	TEMP_3_double_out <= TEMP_3_double_in and not std_logic_vector( unsigned( TEMP_3_double_in ) - unsigned( "00000000" & ldq_tail_oh ) );
	ldq_head_next_oh <= TEMP_3_double_out(7 downto 0) or TEMP_3_double_out(15 downto 8);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(ldq_head_sel, ldq_alloc_next, or)
	TEMP_4_res_0 <= ldq_alloc_next_0 or ldq_alloc_next_4;
	TEMP_4_res_1 <= ldq_alloc_next_1 or ldq_alloc_next_5;
	TEMP_4_res_2 <= ldq_alloc_next_2 or ldq_alloc_next_6;
	TEMP_4_res_3 <= ldq_alloc_next_3 or ldq_alloc_next_7;
	-- Layer End
	TEMP_5_res_0 <= TEMP_4_res_0 or TEMP_4_res_2;
	TEMP_5_res_1 <= TEMP_4_res_1 or TEMP_4_res_3;
	-- Layer End
	ldq_head_sel <= TEMP_5_res_0 or TEMP_5_res_1;
	-- Reduction End

	-- One-Hot To Bits Begin
	-- OHToBits(ldq_head_next, ldq_head_next_oh)
	TEMP_6_in_0_0 <= '0';
	TEMP_6_in_0_1 <= ldq_head_next_oh(1);
	TEMP_6_in_0_2 <= '0';
	TEMP_6_in_0_3 <= ldq_head_next_oh(3);
	TEMP_6_in_0_4 <= '0';
	TEMP_6_in_0_5 <= ldq_head_next_oh(5);
	TEMP_6_in_0_6 <= '0';
	TEMP_6_in_0_7 <= ldq_head_next_oh(7);
	TEMP_7_res_0 <= TEMP_6_in_0_0 or TEMP_6_in_0_4;
	TEMP_7_res_1 <= TEMP_6_in_0_1 or TEMP_6_in_0_5;
	TEMP_7_res_2 <= TEMP_6_in_0_2 or TEMP_6_in_0_6;
	TEMP_7_res_3 <= TEMP_6_in_0_3 or TEMP_6_in_0_7;
	-- Layer End
	TEMP_8_res_0 <= TEMP_7_res_0 or TEMP_7_res_2;
	TEMP_8_res_1 <= TEMP_7_res_1 or TEMP_7_res_3;
	-- Layer End
	TEMP_6_out_0 <= TEMP_8_res_0 or TEMP_8_res_1;
	ldq_head_next(0) <= TEMP_6_out_0;
	TEMP_8_in_1_0 <= '0';
	TEMP_8_in_1_1 <= '0';
	TEMP_8_in_1_2 <= ldq_head_next_oh(2);
	TEMP_8_in_1_3 <= ldq_head_next_oh(3);
	TEMP_8_in_1_4 <= '0';
	TEMP_8_in_1_5 <= '0';
	TEMP_8_in_1_6 <= ldq_head_next_oh(6);
	TEMP_8_in_1_7 <= ldq_head_next_oh(7);
	TEMP_9_res_0 <= TEMP_8_in_1_0 or TEMP_8_in_1_4;
	TEMP_9_res_1 <= TEMP_8_in_1_1 or TEMP_8_in_1_5;
	TEMP_9_res_2 <= TEMP_8_in_1_2 or TEMP_8_in_1_6;
	TEMP_9_res_3 <= TEMP_8_in_1_3 or TEMP_8_in_1_7;
	-- Layer End
	TEMP_10_res_0 <= TEMP_9_res_0 or TEMP_9_res_2;
	TEMP_10_res_1 <= TEMP_9_res_1 or TEMP_9_res_3;
	-- Layer End
	TEMP_8_out_1 <= TEMP_10_res_0 or TEMP_10_res_1;
	ldq_head_next(1) <= TEMP_8_out_1;
	TEMP_10_in_2_0 <= '0';
	TEMP_10_in_2_1 <= '0';
	TEMP_10_in_2_2 <= '0';
	TEMP_10_in_2_3 <= '0';
	TEMP_10_in_2_4 <= ldq_head_next_oh(4);
	TEMP_10_in_2_5 <= ldq_head_next_oh(5);
	TEMP_10_in_2_6 <= ldq_head_next_oh(6);
	TEMP_10_in_2_7 <= ldq_head_next_oh(7);
	TEMP_11_res_0 <= TEMP_10_in_2_0 or TEMP_10_in_2_4;
	TEMP_11_res_1 <= TEMP_10_in_2_1 or TEMP_10_in_2_5;
	TEMP_11_res_2 <= TEMP_10_in_2_2 or TEMP_10_in_2_6;
	TEMP_11_res_3 <= TEMP_10_in_2_3 or TEMP_10_in_2_7;
	-- Layer End
	TEMP_12_res_0 <= TEMP_11_res_0 or TEMP_11_res_2;
	TEMP_12_res_1 <= TEMP_11_res_1 or TEMP_11_res_3;
	-- Layer End
	TEMP_10_out_2 <= TEMP_12_res_0 or TEMP_12_res_1;
	ldq_head_next(2) <= TEMP_10_out_2;
	-- One-Hot To Bits End

	ldq_head_d <= ldq_head_next when ldq_head_sel else ldq_tail_q;
	-- Bits To One-Hot Begin
	-- BitsToOH(stq_tail_oh, stq_tail)
	stq_tail_oh(0) <= '1' when stq_tail_q = "000" else '0';
	stq_tail_oh(1) <= '1' when stq_tail_q = "001" else '0';
	stq_tail_oh(2) <= '1' when stq_tail_q = "010" else '0';
	stq_tail_oh(3) <= '1' when stq_tail_q = "011" else '0';
	stq_tail_oh(4) <= '1' when stq_tail_q = "100" else '0';
	stq_tail_oh(5) <= '1' when stq_tail_q = "101" else '0';
	stq_tail_oh(6) <= '1' when stq_tail_q = "110" else '0';
	stq_tail_oh(7) <= '1' when stq_tail_q = "111" else '0';
	-- Bits To One-Hot End

	-- WrapAdd Begin
	-- WrapAdd(stq_head_next, stq_head, 1, 8)
	stq_head_next <= std_logic_vector(unsigned(stq_head_q) + 1);
	-- WrapAdd End

	stq_head_sel <= wresp_valid_0_i;
	stq_head_d <= stq_head_next when stq_head_sel else stq_head_q;
	handshake_lsq_lsq4_core_ga : entity work.handshake_lsq_lsq4_core_ga
		port map(
			rst => rst,
			clk => clk,
			group_init_valid_0_i => group_init_valid_0_i,
			group_init_valid_1_i => group_init_valid_1_i,
			group_init_valid_2_i => group_init_valid_2_i,
			group_init_ready_0_o => group_init_ready_0_o,
			group_init_ready_1_o => group_init_ready_1_o,
			group_init_ready_2_o => group_init_ready_2_o,
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
			num_loads_o => num_loads,
			ldq_port_idx_0_o => ldq_port_idx_0_d,
			ldq_port_idx_1_o => ldq_port_idx_1_d,
			ldq_port_idx_2_o => ldq_port_idx_2_d,
			ldq_port_idx_3_o => ldq_port_idx_3_d,
			ldq_port_idx_4_o => ldq_port_idx_4_d,
			ldq_port_idx_5_o => ldq_port_idx_5_d,
			ldq_port_idx_6_o => ldq_port_idx_6_d,
			ldq_port_idx_7_o => ldq_port_idx_7_d,
			stq_wen_0_o => stq_wen_0,
			stq_wen_1_o => stq_wen_1,
			stq_wen_2_o => stq_wen_2,
			stq_wen_3_o => stq_wen_3,
			stq_wen_4_o => stq_wen_4,
			stq_wen_5_o => stq_wen_5,
			stq_wen_6_o => stq_wen_6,
			stq_wen_7_o => stq_wen_7,
			stq_port_idx_0_o => stq_port_idx_0_d,
			stq_port_idx_1_o => stq_port_idx_1_d,
			stq_port_idx_2_o => stq_port_idx_2_d,
			stq_port_idx_3_o => stq_port_idx_3_d,
			stq_port_idx_4_o => stq_port_idx_4_d,
			stq_port_idx_5_o => stq_port_idx_5_d,
			stq_port_idx_6_o => stq_port_idx_6_d,
			stq_port_idx_7_o => stq_port_idx_7_d,
			ga_ls_order_0_o => ga_ls_order_0,
			ga_ls_order_1_o => ga_ls_order_1,
			ga_ls_order_2_o => ga_ls_order_2,
			ga_ls_order_3_o => ga_ls_order_3,
			ga_ls_order_4_o => ga_ls_order_4,
			ga_ls_order_5_o => ga_ls_order_5,
			ga_ls_order_6_o => ga_ls_order_6,
			ga_ls_order_7_o => ga_ls_order_7,
			num_stores_o => num_stores
		);
	handshake_lsq_lsq4_core_lda_dispatcher : entity work.handshake_lsq_lsq4_core_lda
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
			entry_alloc_4_i => ldq_alloc_4_q,
			entry_alloc_5_i => ldq_alloc_5_q,
			entry_alloc_6_i => ldq_alloc_6_q,
			entry_alloc_7_i => ldq_alloc_7_q,
			entry_payload_valid_0_i => ldq_addr_valid_0_q,
			entry_payload_valid_1_i => ldq_addr_valid_1_q,
			entry_payload_valid_2_i => ldq_addr_valid_2_q,
			entry_payload_valid_3_i => ldq_addr_valid_3_q,
			entry_payload_valid_4_i => ldq_addr_valid_4_q,
			entry_payload_valid_5_i => ldq_addr_valid_5_q,
			entry_payload_valid_6_i => ldq_addr_valid_6_q,
			entry_payload_valid_7_i => ldq_addr_valid_7_q,
			entry_port_idx_0_i => ldq_port_idx_0_q,
			entry_port_idx_1_i => ldq_port_idx_1_q,
			entry_port_idx_2_i => ldq_port_idx_2_q,
			entry_port_idx_3_i => ldq_port_idx_3_q,
			entry_port_idx_4_i => ldq_port_idx_4_q,
			entry_port_idx_5_i => ldq_port_idx_5_q,
			entry_port_idx_6_i => ldq_port_idx_6_q,
			entry_port_idx_7_i => ldq_port_idx_7_q,
			entry_payload_0_o => ldq_addr_0_d,
			entry_payload_1_o => ldq_addr_1_d,
			entry_payload_2_o => ldq_addr_2_d,
			entry_payload_3_o => ldq_addr_3_d,
			entry_payload_4_o => ldq_addr_4_d,
			entry_payload_5_o => ldq_addr_5_d,
			entry_payload_6_o => ldq_addr_6_d,
			entry_payload_7_o => ldq_addr_7_d,
			entry_wen_0_o => ldq_addr_wen_0,
			entry_wen_1_o => ldq_addr_wen_1,
			entry_wen_2_o => ldq_addr_wen_2,
			entry_wen_3_o => ldq_addr_wen_3,
			entry_wen_4_o => ldq_addr_wen_4,
			entry_wen_5_o => ldq_addr_wen_5,
			entry_wen_6_o => ldq_addr_wen_6,
			entry_wen_7_o => ldq_addr_wen_7,
			queue_head_oh_i => ldq_head_oh
		);
	handshake_lsq_lsq4_core_ldd_dispatcher : entity work.handshake_lsq_lsq4_core_ldd
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
			entry_alloc_4_i => ldq_alloc_4_q,
			entry_alloc_5_i => ldq_alloc_5_q,
			entry_alloc_6_i => ldq_alloc_6_q,
			entry_alloc_7_i => ldq_alloc_7_q,
			entry_payload_valid_0_i => ldq_data_valid_0_q,
			entry_payload_valid_1_i => ldq_data_valid_1_q,
			entry_payload_valid_2_i => ldq_data_valid_2_q,
			entry_payload_valid_3_i => ldq_data_valid_3_q,
			entry_payload_valid_4_i => ldq_data_valid_4_q,
			entry_payload_valid_5_i => ldq_data_valid_5_q,
			entry_payload_valid_6_i => ldq_data_valid_6_q,
			entry_payload_valid_7_i => ldq_data_valid_7_q,
			entry_port_idx_0_i => ldq_port_idx_0_q,
			entry_port_idx_1_i => ldq_port_idx_1_q,
			entry_port_idx_2_i => ldq_port_idx_2_q,
			entry_port_idx_3_i => ldq_port_idx_3_q,
			entry_port_idx_4_i => ldq_port_idx_4_q,
			entry_port_idx_5_i => ldq_port_idx_5_q,
			entry_port_idx_6_i => ldq_port_idx_6_q,
			entry_port_idx_7_i => ldq_port_idx_7_q,
			entry_payload_0_i => ldq_data_0_q,
			entry_payload_1_i => ldq_data_1_q,
			entry_payload_2_i => ldq_data_2_q,
			entry_payload_3_i => ldq_data_3_q,
			entry_payload_4_i => ldq_data_4_q,
			entry_payload_5_i => ldq_data_5_q,
			entry_payload_6_i => ldq_data_6_q,
			entry_payload_7_i => ldq_data_7_q,
			entry_reset_0_o => ldq_reset_0,
			entry_reset_1_o => ldq_reset_1,
			entry_reset_2_o => ldq_reset_2,
			entry_reset_3_o => ldq_reset_3,
			entry_reset_4_o => ldq_reset_4,
			entry_reset_5_o => ldq_reset_5,
			entry_reset_6_o => ldq_reset_6,
			entry_reset_7_o => ldq_reset_7,
			queue_head_oh_i => ldq_head_oh
		);
	handshake_lsq_lsq4_core_sta_dispatcher : entity work.handshake_lsq_lsq4_core_sta
		port map(
			rst => rst,
			clk => clk,
			port_payload_0_i => stp_addr_0_i,
			port_payload_1_i => stp_addr_1_i,
			port_ready_0_o => stp_addr_ready_0_o,
			port_ready_1_o => stp_addr_ready_1_o,
			port_valid_0_i => stp_addr_valid_0_i,
			port_valid_1_i => stp_addr_valid_1_i,
			entry_alloc_0_i => stq_alloc_0_q,
			entry_alloc_1_i => stq_alloc_1_q,
			entry_alloc_2_i => stq_alloc_2_q,
			entry_alloc_3_i => stq_alloc_3_q,
			entry_alloc_4_i => stq_alloc_4_q,
			entry_alloc_5_i => stq_alloc_5_q,
			entry_alloc_6_i => stq_alloc_6_q,
			entry_alloc_7_i => stq_alloc_7_q,
			entry_payload_valid_0_i => stq_addr_valid_0_q,
			entry_payload_valid_1_i => stq_addr_valid_1_q,
			entry_payload_valid_2_i => stq_addr_valid_2_q,
			entry_payload_valid_3_i => stq_addr_valid_3_q,
			entry_payload_valid_4_i => stq_addr_valid_4_q,
			entry_payload_valid_5_i => stq_addr_valid_5_q,
			entry_payload_valid_6_i => stq_addr_valid_6_q,
			entry_payload_valid_7_i => stq_addr_valid_7_q,
			entry_port_idx_0_i => stq_port_idx_0_q,
			entry_port_idx_1_i => stq_port_idx_1_q,
			entry_port_idx_2_i => stq_port_idx_2_q,
			entry_port_idx_3_i => stq_port_idx_3_q,
			entry_port_idx_4_i => stq_port_idx_4_q,
			entry_port_idx_5_i => stq_port_idx_5_q,
			entry_port_idx_6_i => stq_port_idx_6_q,
			entry_port_idx_7_i => stq_port_idx_7_q,
			entry_payload_0_o => stq_addr_0_d,
			entry_payload_1_o => stq_addr_1_d,
			entry_payload_2_o => stq_addr_2_d,
			entry_payload_3_o => stq_addr_3_d,
			entry_payload_4_o => stq_addr_4_d,
			entry_payload_5_o => stq_addr_5_d,
			entry_payload_6_o => stq_addr_6_d,
			entry_payload_7_o => stq_addr_7_d,
			entry_wen_0_o => stq_addr_wen_0,
			entry_wen_1_o => stq_addr_wen_1,
			entry_wen_2_o => stq_addr_wen_2,
			entry_wen_3_o => stq_addr_wen_3,
			entry_wen_4_o => stq_addr_wen_4,
			entry_wen_5_o => stq_addr_wen_5,
			entry_wen_6_o => stq_addr_wen_6,
			entry_wen_7_o => stq_addr_wen_7,
			queue_head_oh_i => stq_head_oh
		);
	handshake_lsq_lsq4_core_std_dispatcher : entity work.handshake_lsq_lsq4_core_std
		port map(
			rst => rst,
			clk => clk,
			port_payload_0_i => stp_data_0_i,
			port_payload_1_i => stp_data_1_i,
			port_ready_0_o => stp_data_ready_0_o,
			port_ready_1_o => stp_data_ready_1_o,
			port_valid_0_i => stp_data_valid_0_i,
			port_valid_1_i => stp_data_valid_1_i,
			entry_alloc_0_i => stq_alloc_0_q,
			entry_alloc_1_i => stq_alloc_1_q,
			entry_alloc_2_i => stq_alloc_2_q,
			entry_alloc_3_i => stq_alloc_3_q,
			entry_alloc_4_i => stq_alloc_4_q,
			entry_alloc_5_i => stq_alloc_5_q,
			entry_alloc_6_i => stq_alloc_6_q,
			entry_alloc_7_i => stq_alloc_7_q,
			entry_payload_valid_0_i => stq_data_valid_0_q,
			entry_payload_valid_1_i => stq_data_valid_1_q,
			entry_payload_valid_2_i => stq_data_valid_2_q,
			entry_payload_valid_3_i => stq_data_valid_3_q,
			entry_payload_valid_4_i => stq_data_valid_4_q,
			entry_payload_valid_5_i => stq_data_valid_5_q,
			entry_payload_valid_6_i => stq_data_valid_6_q,
			entry_payload_valid_7_i => stq_data_valid_7_q,
			entry_port_idx_0_i => stq_port_idx_0_q,
			entry_port_idx_1_i => stq_port_idx_1_q,
			entry_port_idx_2_i => stq_port_idx_2_q,
			entry_port_idx_3_i => stq_port_idx_3_q,
			entry_port_idx_4_i => stq_port_idx_4_q,
			entry_port_idx_5_i => stq_port_idx_5_q,
			entry_port_idx_6_i => stq_port_idx_6_q,
			entry_port_idx_7_i => stq_port_idx_7_q,
			entry_payload_0_o => stq_data_0_d,
			entry_payload_1_o => stq_data_1_d,
			entry_payload_2_o => stq_data_2_d,
			entry_payload_3_o => stq_data_3_d,
			entry_payload_4_o => stq_data_4_d,
			entry_payload_5_o => stq_data_5_d,
			entry_payload_6_o => stq_data_6_d,
			entry_payload_7_o => stq_data_7_d,
			entry_wen_0_o => stq_data_wen_0,
			entry_wen_1_o => stq_data_wen_1,
			entry_wen_2_o => stq_data_wen_2,
			entry_wen_3_o => stq_data_wen_3,
			entry_wen_4_o => stq_data_wen_4,
			entry_wen_5_o => stq_data_wen_5,
			entry_wen_6_o => stq_data_wen_6,
			entry_wen_7_o => stq_data_wen_7,
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
	addr_valid_1(0) <= ldq_addr_valid_1_q and stq_addr_valid_0_q;
	addr_valid_1(1) <= ldq_addr_valid_1_q and stq_addr_valid_1_q;
	addr_valid_1(2) <= ldq_addr_valid_1_q and stq_addr_valid_2_q;
	addr_valid_1(3) <= ldq_addr_valid_1_q and stq_addr_valid_3_q;
	addr_valid_1(4) <= ldq_addr_valid_1_q and stq_addr_valid_4_q;
	addr_valid_1(5) <= ldq_addr_valid_1_q and stq_addr_valid_5_q;
	addr_valid_1(6) <= ldq_addr_valid_1_q and stq_addr_valid_6_q;
	addr_valid_1(7) <= ldq_addr_valid_1_q and stq_addr_valid_7_q;
	addr_valid_2(0) <= ldq_addr_valid_2_q and stq_addr_valid_0_q;
	addr_valid_2(1) <= ldq_addr_valid_2_q and stq_addr_valid_1_q;
	addr_valid_2(2) <= ldq_addr_valid_2_q and stq_addr_valid_2_q;
	addr_valid_2(3) <= ldq_addr_valid_2_q and stq_addr_valid_3_q;
	addr_valid_2(4) <= ldq_addr_valid_2_q and stq_addr_valid_4_q;
	addr_valid_2(5) <= ldq_addr_valid_2_q and stq_addr_valid_5_q;
	addr_valid_2(6) <= ldq_addr_valid_2_q and stq_addr_valid_6_q;
	addr_valid_2(7) <= ldq_addr_valid_2_q and stq_addr_valid_7_q;
	addr_valid_3(0) <= ldq_addr_valid_3_q and stq_addr_valid_0_q;
	addr_valid_3(1) <= ldq_addr_valid_3_q and stq_addr_valid_1_q;
	addr_valid_3(2) <= ldq_addr_valid_3_q and stq_addr_valid_2_q;
	addr_valid_3(3) <= ldq_addr_valid_3_q and stq_addr_valid_3_q;
	addr_valid_3(4) <= ldq_addr_valid_3_q and stq_addr_valid_4_q;
	addr_valid_3(5) <= ldq_addr_valid_3_q and stq_addr_valid_5_q;
	addr_valid_3(6) <= ldq_addr_valid_3_q and stq_addr_valid_6_q;
	addr_valid_3(7) <= ldq_addr_valid_3_q and stq_addr_valid_7_q;
	addr_valid_4(0) <= ldq_addr_valid_4_q and stq_addr_valid_0_q;
	addr_valid_4(1) <= ldq_addr_valid_4_q and stq_addr_valid_1_q;
	addr_valid_4(2) <= ldq_addr_valid_4_q and stq_addr_valid_2_q;
	addr_valid_4(3) <= ldq_addr_valid_4_q and stq_addr_valid_3_q;
	addr_valid_4(4) <= ldq_addr_valid_4_q and stq_addr_valid_4_q;
	addr_valid_4(5) <= ldq_addr_valid_4_q and stq_addr_valid_5_q;
	addr_valid_4(6) <= ldq_addr_valid_4_q and stq_addr_valid_6_q;
	addr_valid_4(7) <= ldq_addr_valid_4_q and stq_addr_valid_7_q;
	addr_valid_5(0) <= ldq_addr_valid_5_q and stq_addr_valid_0_q;
	addr_valid_5(1) <= ldq_addr_valid_5_q and stq_addr_valid_1_q;
	addr_valid_5(2) <= ldq_addr_valid_5_q and stq_addr_valid_2_q;
	addr_valid_5(3) <= ldq_addr_valid_5_q and stq_addr_valid_3_q;
	addr_valid_5(4) <= ldq_addr_valid_5_q and stq_addr_valid_4_q;
	addr_valid_5(5) <= ldq_addr_valid_5_q and stq_addr_valid_5_q;
	addr_valid_5(6) <= ldq_addr_valid_5_q and stq_addr_valid_6_q;
	addr_valid_5(7) <= ldq_addr_valid_5_q and stq_addr_valid_7_q;
	addr_valid_6(0) <= ldq_addr_valid_6_q and stq_addr_valid_0_q;
	addr_valid_6(1) <= ldq_addr_valid_6_q and stq_addr_valid_1_q;
	addr_valid_6(2) <= ldq_addr_valid_6_q and stq_addr_valid_2_q;
	addr_valid_6(3) <= ldq_addr_valid_6_q and stq_addr_valid_3_q;
	addr_valid_6(4) <= ldq_addr_valid_6_q and stq_addr_valid_4_q;
	addr_valid_6(5) <= ldq_addr_valid_6_q and stq_addr_valid_5_q;
	addr_valid_6(6) <= ldq_addr_valid_6_q and stq_addr_valid_6_q;
	addr_valid_6(7) <= ldq_addr_valid_6_q and stq_addr_valid_7_q;
	addr_valid_7(0) <= ldq_addr_valid_7_q and stq_addr_valid_0_q;
	addr_valid_7(1) <= ldq_addr_valid_7_q and stq_addr_valid_1_q;
	addr_valid_7(2) <= ldq_addr_valid_7_q and stq_addr_valid_2_q;
	addr_valid_7(3) <= ldq_addr_valid_7_q and stq_addr_valid_3_q;
	addr_valid_7(4) <= ldq_addr_valid_7_q and stq_addr_valid_4_q;
	addr_valid_7(5) <= ldq_addr_valid_7_q and stq_addr_valid_5_q;
	addr_valid_7(6) <= ldq_addr_valid_7_q and stq_addr_valid_6_q;
	addr_valid_7(7) <= ldq_addr_valid_7_q and stq_addr_valid_7_q;
	addr_same_0(0) <= '1' when ldq_addr_0_q = stq_addr_0_q else '0';
	addr_same_0(1) <= '1' when ldq_addr_0_q = stq_addr_1_q else '0';
	addr_same_0(2) <= '1' when ldq_addr_0_q = stq_addr_2_q else '0';
	addr_same_0(3) <= '1' when ldq_addr_0_q = stq_addr_3_q else '0';
	addr_same_0(4) <= '1' when ldq_addr_0_q = stq_addr_4_q else '0';
	addr_same_0(5) <= '1' when ldq_addr_0_q = stq_addr_5_q else '0';
	addr_same_0(6) <= '1' when ldq_addr_0_q = stq_addr_6_q else '0';
	addr_same_0(7) <= '1' when ldq_addr_0_q = stq_addr_7_q else '0';
	addr_same_1(0) <= '1' when ldq_addr_1_q = stq_addr_0_q else '0';
	addr_same_1(1) <= '1' when ldq_addr_1_q = stq_addr_1_q else '0';
	addr_same_1(2) <= '1' when ldq_addr_1_q = stq_addr_2_q else '0';
	addr_same_1(3) <= '1' when ldq_addr_1_q = stq_addr_3_q else '0';
	addr_same_1(4) <= '1' when ldq_addr_1_q = stq_addr_4_q else '0';
	addr_same_1(5) <= '1' when ldq_addr_1_q = stq_addr_5_q else '0';
	addr_same_1(6) <= '1' when ldq_addr_1_q = stq_addr_6_q else '0';
	addr_same_1(7) <= '1' when ldq_addr_1_q = stq_addr_7_q else '0';
	addr_same_2(0) <= '1' when ldq_addr_2_q = stq_addr_0_q else '0';
	addr_same_2(1) <= '1' when ldq_addr_2_q = stq_addr_1_q else '0';
	addr_same_2(2) <= '1' when ldq_addr_2_q = stq_addr_2_q else '0';
	addr_same_2(3) <= '1' when ldq_addr_2_q = stq_addr_3_q else '0';
	addr_same_2(4) <= '1' when ldq_addr_2_q = stq_addr_4_q else '0';
	addr_same_2(5) <= '1' when ldq_addr_2_q = stq_addr_5_q else '0';
	addr_same_2(6) <= '1' when ldq_addr_2_q = stq_addr_6_q else '0';
	addr_same_2(7) <= '1' when ldq_addr_2_q = stq_addr_7_q else '0';
	addr_same_3(0) <= '1' when ldq_addr_3_q = stq_addr_0_q else '0';
	addr_same_3(1) <= '1' when ldq_addr_3_q = stq_addr_1_q else '0';
	addr_same_3(2) <= '1' when ldq_addr_3_q = stq_addr_2_q else '0';
	addr_same_3(3) <= '1' when ldq_addr_3_q = stq_addr_3_q else '0';
	addr_same_3(4) <= '1' when ldq_addr_3_q = stq_addr_4_q else '0';
	addr_same_3(5) <= '1' when ldq_addr_3_q = stq_addr_5_q else '0';
	addr_same_3(6) <= '1' when ldq_addr_3_q = stq_addr_6_q else '0';
	addr_same_3(7) <= '1' when ldq_addr_3_q = stq_addr_7_q else '0';
	addr_same_4(0) <= '1' when ldq_addr_4_q = stq_addr_0_q else '0';
	addr_same_4(1) <= '1' when ldq_addr_4_q = stq_addr_1_q else '0';
	addr_same_4(2) <= '1' when ldq_addr_4_q = stq_addr_2_q else '0';
	addr_same_4(3) <= '1' when ldq_addr_4_q = stq_addr_3_q else '0';
	addr_same_4(4) <= '1' when ldq_addr_4_q = stq_addr_4_q else '0';
	addr_same_4(5) <= '1' when ldq_addr_4_q = stq_addr_5_q else '0';
	addr_same_4(6) <= '1' when ldq_addr_4_q = stq_addr_6_q else '0';
	addr_same_4(7) <= '1' when ldq_addr_4_q = stq_addr_7_q else '0';
	addr_same_5(0) <= '1' when ldq_addr_5_q = stq_addr_0_q else '0';
	addr_same_5(1) <= '1' when ldq_addr_5_q = stq_addr_1_q else '0';
	addr_same_5(2) <= '1' when ldq_addr_5_q = stq_addr_2_q else '0';
	addr_same_5(3) <= '1' when ldq_addr_5_q = stq_addr_3_q else '0';
	addr_same_5(4) <= '1' when ldq_addr_5_q = stq_addr_4_q else '0';
	addr_same_5(5) <= '1' when ldq_addr_5_q = stq_addr_5_q else '0';
	addr_same_5(6) <= '1' when ldq_addr_5_q = stq_addr_6_q else '0';
	addr_same_5(7) <= '1' when ldq_addr_5_q = stq_addr_7_q else '0';
	addr_same_6(0) <= '1' when ldq_addr_6_q = stq_addr_0_q else '0';
	addr_same_6(1) <= '1' when ldq_addr_6_q = stq_addr_1_q else '0';
	addr_same_6(2) <= '1' when ldq_addr_6_q = stq_addr_2_q else '0';
	addr_same_6(3) <= '1' when ldq_addr_6_q = stq_addr_3_q else '0';
	addr_same_6(4) <= '1' when ldq_addr_6_q = stq_addr_4_q else '0';
	addr_same_6(5) <= '1' when ldq_addr_6_q = stq_addr_5_q else '0';
	addr_same_6(6) <= '1' when ldq_addr_6_q = stq_addr_6_q else '0';
	addr_same_6(7) <= '1' when ldq_addr_6_q = stq_addr_7_q else '0';
	addr_same_7(0) <= '1' when ldq_addr_7_q = stq_addr_0_q else '0';
	addr_same_7(1) <= '1' when ldq_addr_7_q = stq_addr_1_q else '0';
	addr_same_7(2) <= '1' when ldq_addr_7_q = stq_addr_2_q else '0';
	addr_same_7(3) <= '1' when ldq_addr_7_q = stq_addr_3_q else '0';
	addr_same_7(4) <= '1' when ldq_addr_7_q = stq_addr_4_q else '0';
	addr_same_7(5) <= '1' when ldq_addr_7_q = stq_addr_5_q else '0';
	addr_same_7(6) <= '1' when ldq_addr_7_q = stq_addr_6_q else '0';
	addr_same_7(7) <= '1' when ldq_addr_7_q = stq_addr_7_q else '0';
	ld_st_conflict_0(0) <= stq_alloc_0_q and store_is_older_0_q(0) and ( addr_same_0(0) or not stq_addr_valid_0_q );
	ld_st_conflict_0(1) <= stq_alloc_1_q and store_is_older_0_q(1) and ( addr_same_0(1) or not stq_addr_valid_1_q );
	ld_st_conflict_0(2) <= stq_alloc_2_q and store_is_older_0_q(2) and ( addr_same_0(2) or not stq_addr_valid_2_q );
	ld_st_conflict_0(3) <= stq_alloc_3_q and store_is_older_0_q(3) and ( addr_same_0(3) or not stq_addr_valid_3_q );
	ld_st_conflict_0(4) <= stq_alloc_4_q and store_is_older_0_q(4) and ( addr_same_0(4) or not stq_addr_valid_4_q );
	ld_st_conflict_0(5) <= stq_alloc_5_q and store_is_older_0_q(5) and ( addr_same_0(5) or not stq_addr_valid_5_q );
	ld_st_conflict_0(6) <= stq_alloc_6_q and store_is_older_0_q(6) and ( addr_same_0(6) or not stq_addr_valid_6_q );
	ld_st_conflict_0(7) <= stq_alloc_7_q and store_is_older_0_q(7) and ( addr_same_0(7) or not stq_addr_valid_7_q );
	ld_st_conflict_1(0) <= stq_alloc_0_q and store_is_older_1_q(0) and ( addr_same_1(0) or not stq_addr_valid_0_q );
	ld_st_conflict_1(1) <= stq_alloc_1_q and store_is_older_1_q(1) and ( addr_same_1(1) or not stq_addr_valid_1_q );
	ld_st_conflict_1(2) <= stq_alloc_2_q and store_is_older_1_q(2) and ( addr_same_1(2) or not stq_addr_valid_2_q );
	ld_st_conflict_1(3) <= stq_alloc_3_q and store_is_older_1_q(3) and ( addr_same_1(3) or not stq_addr_valid_3_q );
	ld_st_conflict_1(4) <= stq_alloc_4_q and store_is_older_1_q(4) and ( addr_same_1(4) or not stq_addr_valid_4_q );
	ld_st_conflict_1(5) <= stq_alloc_5_q and store_is_older_1_q(5) and ( addr_same_1(5) or not stq_addr_valid_5_q );
	ld_st_conflict_1(6) <= stq_alloc_6_q and store_is_older_1_q(6) and ( addr_same_1(6) or not stq_addr_valid_6_q );
	ld_st_conflict_1(7) <= stq_alloc_7_q and store_is_older_1_q(7) and ( addr_same_1(7) or not stq_addr_valid_7_q );
	ld_st_conflict_2(0) <= stq_alloc_0_q and store_is_older_2_q(0) and ( addr_same_2(0) or not stq_addr_valid_0_q );
	ld_st_conflict_2(1) <= stq_alloc_1_q and store_is_older_2_q(1) and ( addr_same_2(1) or not stq_addr_valid_1_q );
	ld_st_conflict_2(2) <= stq_alloc_2_q and store_is_older_2_q(2) and ( addr_same_2(2) or not stq_addr_valid_2_q );
	ld_st_conflict_2(3) <= stq_alloc_3_q and store_is_older_2_q(3) and ( addr_same_2(3) or not stq_addr_valid_3_q );
	ld_st_conflict_2(4) <= stq_alloc_4_q and store_is_older_2_q(4) and ( addr_same_2(4) or not stq_addr_valid_4_q );
	ld_st_conflict_2(5) <= stq_alloc_5_q and store_is_older_2_q(5) and ( addr_same_2(5) or not stq_addr_valid_5_q );
	ld_st_conflict_2(6) <= stq_alloc_6_q and store_is_older_2_q(6) and ( addr_same_2(6) or not stq_addr_valid_6_q );
	ld_st_conflict_2(7) <= stq_alloc_7_q and store_is_older_2_q(7) and ( addr_same_2(7) or not stq_addr_valid_7_q );
	ld_st_conflict_3(0) <= stq_alloc_0_q and store_is_older_3_q(0) and ( addr_same_3(0) or not stq_addr_valid_0_q );
	ld_st_conflict_3(1) <= stq_alloc_1_q and store_is_older_3_q(1) and ( addr_same_3(1) or not stq_addr_valid_1_q );
	ld_st_conflict_3(2) <= stq_alloc_2_q and store_is_older_3_q(2) and ( addr_same_3(2) or not stq_addr_valid_2_q );
	ld_st_conflict_3(3) <= stq_alloc_3_q and store_is_older_3_q(3) and ( addr_same_3(3) or not stq_addr_valid_3_q );
	ld_st_conflict_3(4) <= stq_alloc_4_q and store_is_older_3_q(4) and ( addr_same_3(4) or not stq_addr_valid_4_q );
	ld_st_conflict_3(5) <= stq_alloc_5_q and store_is_older_3_q(5) and ( addr_same_3(5) or not stq_addr_valid_5_q );
	ld_st_conflict_3(6) <= stq_alloc_6_q and store_is_older_3_q(6) and ( addr_same_3(6) or not stq_addr_valid_6_q );
	ld_st_conflict_3(7) <= stq_alloc_7_q and store_is_older_3_q(7) and ( addr_same_3(7) or not stq_addr_valid_7_q );
	ld_st_conflict_4(0) <= stq_alloc_0_q and store_is_older_4_q(0) and ( addr_same_4(0) or not stq_addr_valid_0_q );
	ld_st_conflict_4(1) <= stq_alloc_1_q and store_is_older_4_q(1) and ( addr_same_4(1) or not stq_addr_valid_1_q );
	ld_st_conflict_4(2) <= stq_alloc_2_q and store_is_older_4_q(2) and ( addr_same_4(2) or not stq_addr_valid_2_q );
	ld_st_conflict_4(3) <= stq_alloc_3_q and store_is_older_4_q(3) and ( addr_same_4(3) or not stq_addr_valid_3_q );
	ld_st_conflict_4(4) <= stq_alloc_4_q and store_is_older_4_q(4) and ( addr_same_4(4) or not stq_addr_valid_4_q );
	ld_st_conflict_4(5) <= stq_alloc_5_q and store_is_older_4_q(5) and ( addr_same_4(5) or not stq_addr_valid_5_q );
	ld_st_conflict_4(6) <= stq_alloc_6_q and store_is_older_4_q(6) and ( addr_same_4(6) or not stq_addr_valid_6_q );
	ld_st_conflict_4(7) <= stq_alloc_7_q and store_is_older_4_q(7) and ( addr_same_4(7) or not stq_addr_valid_7_q );
	ld_st_conflict_5(0) <= stq_alloc_0_q and store_is_older_5_q(0) and ( addr_same_5(0) or not stq_addr_valid_0_q );
	ld_st_conflict_5(1) <= stq_alloc_1_q and store_is_older_5_q(1) and ( addr_same_5(1) or not stq_addr_valid_1_q );
	ld_st_conflict_5(2) <= stq_alloc_2_q and store_is_older_5_q(2) and ( addr_same_5(2) or not stq_addr_valid_2_q );
	ld_st_conflict_5(3) <= stq_alloc_3_q and store_is_older_5_q(3) and ( addr_same_5(3) or not stq_addr_valid_3_q );
	ld_st_conflict_5(4) <= stq_alloc_4_q and store_is_older_5_q(4) and ( addr_same_5(4) or not stq_addr_valid_4_q );
	ld_st_conflict_5(5) <= stq_alloc_5_q and store_is_older_5_q(5) and ( addr_same_5(5) or not stq_addr_valid_5_q );
	ld_st_conflict_5(6) <= stq_alloc_6_q and store_is_older_5_q(6) and ( addr_same_5(6) or not stq_addr_valid_6_q );
	ld_st_conflict_5(7) <= stq_alloc_7_q and store_is_older_5_q(7) and ( addr_same_5(7) or not stq_addr_valid_7_q );
	ld_st_conflict_6(0) <= stq_alloc_0_q and store_is_older_6_q(0) and ( addr_same_6(0) or not stq_addr_valid_0_q );
	ld_st_conflict_6(1) <= stq_alloc_1_q and store_is_older_6_q(1) and ( addr_same_6(1) or not stq_addr_valid_1_q );
	ld_st_conflict_6(2) <= stq_alloc_2_q and store_is_older_6_q(2) and ( addr_same_6(2) or not stq_addr_valid_2_q );
	ld_st_conflict_6(3) <= stq_alloc_3_q and store_is_older_6_q(3) and ( addr_same_6(3) or not stq_addr_valid_3_q );
	ld_st_conflict_6(4) <= stq_alloc_4_q and store_is_older_6_q(4) and ( addr_same_6(4) or not stq_addr_valid_4_q );
	ld_st_conflict_6(5) <= stq_alloc_5_q and store_is_older_6_q(5) and ( addr_same_6(5) or not stq_addr_valid_5_q );
	ld_st_conflict_6(6) <= stq_alloc_6_q and store_is_older_6_q(6) and ( addr_same_6(6) or not stq_addr_valid_6_q );
	ld_st_conflict_6(7) <= stq_alloc_7_q and store_is_older_6_q(7) and ( addr_same_6(7) or not stq_addr_valid_7_q );
	ld_st_conflict_7(0) <= stq_alloc_0_q and store_is_older_7_q(0) and ( addr_same_7(0) or not stq_addr_valid_0_q );
	ld_st_conflict_7(1) <= stq_alloc_1_q and store_is_older_7_q(1) and ( addr_same_7(1) or not stq_addr_valid_1_q );
	ld_st_conflict_7(2) <= stq_alloc_2_q and store_is_older_7_q(2) and ( addr_same_7(2) or not stq_addr_valid_2_q );
	ld_st_conflict_7(3) <= stq_alloc_3_q and store_is_older_7_q(3) and ( addr_same_7(3) or not stq_addr_valid_3_q );
	ld_st_conflict_7(4) <= stq_alloc_4_q and store_is_older_7_q(4) and ( addr_same_7(4) or not stq_addr_valid_4_q );
	ld_st_conflict_7(5) <= stq_alloc_5_q and store_is_older_7_q(5) and ( addr_same_7(5) or not stq_addr_valid_5_q );
	ld_st_conflict_7(6) <= stq_alloc_6_q and store_is_older_7_q(6) and ( addr_same_7(6) or not stq_addr_valid_6_q );
	ld_st_conflict_7(7) <= stq_alloc_7_q and store_is_older_7_q(7) and ( addr_same_7(7) or not stq_addr_valid_7_q );
	can_bypass_0(0) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_0_q and addr_same_0(0) and addr_valid_0(0);
	can_bypass_0(1) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_1_q and addr_same_0(1) and addr_valid_0(1);
	can_bypass_0(2) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_2_q and addr_same_0(2) and addr_valid_0(2);
	can_bypass_0(3) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_3_q and addr_same_0(3) and addr_valid_0(3);
	can_bypass_0(4) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_4_q and addr_same_0(4) and addr_valid_0(4);
	can_bypass_0(5) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_5_q and addr_same_0(5) and addr_valid_0(5);
	can_bypass_0(6) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_6_q and addr_same_0(6) and addr_valid_0(6);
	can_bypass_0(7) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_7_q and addr_same_0(7) and addr_valid_0(7);
	can_bypass_1(0) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_0_q and addr_same_1(0) and addr_valid_1(0);
	can_bypass_1(1) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_1_q and addr_same_1(1) and addr_valid_1(1);
	can_bypass_1(2) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_2_q and addr_same_1(2) and addr_valid_1(2);
	can_bypass_1(3) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_3_q and addr_same_1(3) and addr_valid_1(3);
	can_bypass_1(4) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_4_q and addr_same_1(4) and addr_valid_1(4);
	can_bypass_1(5) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_5_q and addr_same_1(5) and addr_valid_1(5);
	can_bypass_1(6) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_6_q and addr_same_1(6) and addr_valid_1(6);
	can_bypass_1(7) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_7_q and addr_same_1(7) and addr_valid_1(7);
	can_bypass_2(0) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_0_q and addr_same_2(0) and addr_valid_2(0);
	can_bypass_2(1) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_1_q and addr_same_2(1) and addr_valid_2(1);
	can_bypass_2(2) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_2_q and addr_same_2(2) and addr_valid_2(2);
	can_bypass_2(3) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_3_q and addr_same_2(3) and addr_valid_2(3);
	can_bypass_2(4) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_4_q and addr_same_2(4) and addr_valid_2(4);
	can_bypass_2(5) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_5_q and addr_same_2(5) and addr_valid_2(5);
	can_bypass_2(6) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_6_q and addr_same_2(6) and addr_valid_2(6);
	can_bypass_2(7) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_7_q and addr_same_2(7) and addr_valid_2(7);
	can_bypass_3(0) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_0_q and addr_same_3(0) and addr_valid_3(0);
	can_bypass_3(1) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_1_q and addr_same_3(1) and addr_valid_3(1);
	can_bypass_3(2) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_2_q and addr_same_3(2) and addr_valid_3(2);
	can_bypass_3(3) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_3_q and addr_same_3(3) and addr_valid_3(3);
	can_bypass_3(4) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_4_q and addr_same_3(4) and addr_valid_3(4);
	can_bypass_3(5) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_5_q and addr_same_3(5) and addr_valid_3(5);
	can_bypass_3(6) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_6_q and addr_same_3(6) and addr_valid_3(6);
	can_bypass_3(7) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_7_q and addr_same_3(7) and addr_valid_3(7);
	can_bypass_4(0) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_0_q and addr_same_4(0) and addr_valid_4(0);
	can_bypass_4(1) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_1_q and addr_same_4(1) and addr_valid_4(1);
	can_bypass_4(2) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_2_q and addr_same_4(2) and addr_valid_4(2);
	can_bypass_4(3) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_3_q and addr_same_4(3) and addr_valid_4(3);
	can_bypass_4(4) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_4_q and addr_same_4(4) and addr_valid_4(4);
	can_bypass_4(5) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_5_q and addr_same_4(5) and addr_valid_4(5);
	can_bypass_4(6) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_6_q and addr_same_4(6) and addr_valid_4(6);
	can_bypass_4(7) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_7_q and addr_same_4(7) and addr_valid_4(7);
	can_bypass_5(0) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_0_q and addr_same_5(0) and addr_valid_5(0);
	can_bypass_5(1) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_1_q and addr_same_5(1) and addr_valid_5(1);
	can_bypass_5(2) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_2_q and addr_same_5(2) and addr_valid_5(2);
	can_bypass_5(3) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_3_q and addr_same_5(3) and addr_valid_5(3);
	can_bypass_5(4) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_4_q and addr_same_5(4) and addr_valid_5(4);
	can_bypass_5(5) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_5_q and addr_same_5(5) and addr_valid_5(5);
	can_bypass_5(6) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_6_q and addr_same_5(6) and addr_valid_5(6);
	can_bypass_5(7) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_7_q and addr_same_5(7) and addr_valid_5(7);
	can_bypass_6(0) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_0_q and addr_same_6(0) and addr_valid_6(0);
	can_bypass_6(1) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_1_q and addr_same_6(1) and addr_valid_6(1);
	can_bypass_6(2) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_2_q and addr_same_6(2) and addr_valid_6(2);
	can_bypass_6(3) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_3_q and addr_same_6(3) and addr_valid_6(3);
	can_bypass_6(4) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_4_q and addr_same_6(4) and addr_valid_6(4);
	can_bypass_6(5) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_5_q and addr_same_6(5) and addr_valid_6(5);
	can_bypass_6(6) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_6_q and addr_same_6(6) and addr_valid_6(6);
	can_bypass_6(7) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_7_q and addr_same_6(7) and addr_valid_6(7);
	can_bypass_7(0) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_0_q and addr_same_7(0) and addr_valid_7(0);
	can_bypass_7(1) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_1_q and addr_same_7(1) and addr_valid_7(1);
	can_bypass_7(2) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_2_q and addr_same_7(2) and addr_valid_7(2);
	can_bypass_7(3) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_3_q and addr_same_7(3) and addr_valid_7(3);
	can_bypass_7(4) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_4_q and addr_same_7(4) and addr_valid_7(4);
	can_bypass_7(5) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_5_q and addr_same_7(5) and addr_valid_7(5);
	can_bypass_7(6) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_6_q and addr_same_7(6) and addr_valid_7(6);
	can_bypass_7(7) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_7_q and addr_same_7(7) and addr_valid_7(7);
	-- Reduction Begin
	-- Reduce(load_conflict_0, ld_st_conflict_0, or)
	TEMP_13_res(0) <= ld_st_conflict_0(0) or ld_st_conflict_0(4);
	TEMP_13_res(1) <= ld_st_conflict_0(1) or ld_st_conflict_0(5);
	TEMP_13_res(2) <= ld_st_conflict_0(2) or ld_st_conflict_0(6);
	TEMP_13_res(3) <= ld_st_conflict_0(3) or ld_st_conflict_0(7);
	-- Layer End
	TEMP_14_res(0) <= TEMP_13_res(0) or TEMP_13_res(2);
	TEMP_14_res(1) <= TEMP_13_res(1) or TEMP_13_res(3);
	-- Layer End
	load_conflict_0 <= TEMP_14_res(0) or TEMP_14_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_1, ld_st_conflict_1, or)
	TEMP_15_res(0) <= ld_st_conflict_1(0) or ld_st_conflict_1(4);
	TEMP_15_res(1) <= ld_st_conflict_1(1) or ld_st_conflict_1(5);
	TEMP_15_res(2) <= ld_st_conflict_1(2) or ld_st_conflict_1(6);
	TEMP_15_res(3) <= ld_st_conflict_1(3) or ld_st_conflict_1(7);
	-- Layer End
	TEMP_16_res(0) <= TEMP_15_res(0) or TEMP_15_res(2);
	TEMP_16_res(1) <= TEMP_15_res(1) or TEMP_15_res(3);
	-- Layer End
	load_conflict_1 <= TEMP_16_res(0) or TEMP_16_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_2, ld_st_conflict_2, or)
	TEMP_17_res(0) <= ld_st_conflict_2(0) or ld_st_conflict_2(4);
	TEMP_17_res(1) <= ld_st_conflict_2(1) or ld_st_conflict_2(5);
	TEMP_17_res(2) <= ld_st_conflict_2(2) or ld_st_conflict_2(6);
	TEMP_17_res(3) <= ld_st_conflict_2(3) or ld_st_conflict_2(7);
	-- Layer End
	TEMP_18_res(0) <= TEMP_17_res(0) or TEMP_17_res(2);
	TEMP_18_res(1) <= TEMP_17_res(1) or TEMP_17_res(3);
	-- Layer End
	load_conflict_2 <= TEMP_18_res(0) or TEMP_18_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_3, ld_st_conflict_3, or)
	TEMP_19_res(0) <= ld_st_conflict_3(0) or ld_st_conflict_3(4);
	TEMP_19_res(1) <= ld_st_conflict_3(1) or ld_st_conflict_3(5);
	TEMP_19_res(2) <= ld_st_conflict_3(2) or ld_st_conflict_3(6);
	TEMP_19_res(3) <= ld_st_conflict_3(3) or ld_st_conflict_3(7);
	-- Layer End
	TEMP_20_res(0) <= TEMP_19_res(0) or TEMP_19_res(2);
	TEMP_20_res(1) <= TEMP_19_res(1) or TEMP_19_res(3);
	-- Layer End
	load_conflict_3 <= TEMP_20_res(0) or TEMP_20_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_4, ld_st_conflict_4, or)
	TEMP_21_res(0) <= ld_st_conflict_4(0) or ld_st_conflict_4(4);
	TEMP_21_res(1) <= ld_st_conflict_4(1) or ld_st_conflict_4(5);
	TEMP_21_res(2) <= ld_st_conflict_4(2) or ld_st_conflict_4(6);
	TEMP_21_res(3) <= ld_st_conflict_4(3) or ld_st_conflict_4(7);
	-- Layer End
	TEMP_22_res(0) <= TEMP_21_res(0) or TEMP_21_res(2);
	TEMP_22_res(1) <= TEMP_21_res(1) or TEMP_21_res(3);
	-- Layer End
	load_conflict_4 <= TEMP_22_res(0) or TEMP_22_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_5, ld_st_conflict_5, or)
	TEMP_23_res(0) <= ld_st_conflict_5(0) or ld_st_conflict_5(4);
	TEMP_23_res(1) <= ld_st_conflict_5(1) or ld_st_conflict_5(5);
	TEMP_23_res(2) <= ld_st_conflict_5(2) or ld_st_conflict_5(6);
	TEMP_23_res(3) <= ld_st_conflict_5(3) or ld_st_conflict_5(7);
	-- Layer End
	TEMP_24_res(0) <= TEMP_23_res(0) or TEMP_23_res(2);
	TEMP_24_res(1) <= TEMP_23_res(1) or TEMP_23_res(3);
	-- Layer End
	load_conflict_5 <= TEMP_24_res(0) or TEMP_24_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_6, ld_st_conflict_6, or)
	TEMP_25_res(0) <= ld_st_conflict_6(0) or ld_st_conflict_6(4);
	TEMP_25_res(1) <= ld_st_conflict_6(1) or ld_st_conflict_6(5);
	TEMP_25_res(2) <= ld_st_conflict_6(2) or ld_st_conflict_6(6);
	TEMP_25_res(3) <= ld_st_conflict_6(3) or ld_st_conflict_6(7);
	-- Layer End
	TEMP_26_res(0) <= TEMP_25_res(0) or TEMP_25_res(2);
	TEMP_26_res(1) <= TEMP_25_res(1) or TEMP_25_res(3);
	-- Layer End
	load_conflict_6 <= TEMP_26_res(0) or TEMP_26_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_7, ld_st_conflict_7, or)
	TEMP_27_res(0) <= ld_st_conflict_7(0) or ld_st_conflict_7(4);
	TEMP_27_res(1) <= ld_st_conflict_7(1) or ld_st_conflict_7(5);
	TEMP_27_res(2) <= ld_st_conflict_7(2) or ld_st_conflict_7(6);
	TEMP_27_res(3) <= ld_st_conflict_7(3) or ld_st_conflict_7(7);
	-- Layer End
	TEMP_28_res(0) <= TEMP_27_res(0) or TEMP_27_res(2);
	TEMP_28_res(1) <= TEMP_27_res(1) or TEMP_27_res(3);
	-- Layer End
	load_conflict_7 <= TEMP_28_res(0) or TEMP_28_res(1);
	-- Reduction End

	load_req_valid_0 <= ldq_alloc_0_q and not ldq_issue_0_q and ldq_addr_valid_0_q;
	load_req_valid_1 <= ldq_alloc_1_q and not ldq_issue_1_q and ldq_addr_valid_1_q;
	load_req_valid_2 <= ldq_alloc_2_q and not ldq_issue_2_q and ldq_addr_valid_2_q;
	load_req_valid_3 <= ldq_alloc_3_q and not ldq_issue_3_q and ldq_addr_valid_3_q;
	load_req_valid_4 <= ldq_alloc_4_q and not ldq_issue_4_q and ldq_addr_valid_4_q;
	load_req_valid_5 <= ldq_alloc_5_q and not ldq_issue_5_q and ldq_addr_valid_5_q;
	load_req_valid_6 <= ldq_alloc_6_q and not ldq_issue_6_q and ldq_addr_valid_6_q;
	load_req_valid_7 <= ldq_alloc_7_q and not ldq_issue_7_q and ldq_addr_valid_7_q;
	can_load_0 <= not load_conflict_0 and load_req_valid_0;
	can_load_1 <= not load_conflict_1 and load_req_valid_1;
	can_load_2 <= not load_conflict_2 and load_req_valid_2;
	can_load_3 <= not load_conflict_3 and load_req_valid_3;
	can_load_4 <= not load_conflict_4 and load_req_valid_4;
	can_load_5 <= not load_conflict_5 and load_req_valid_5;
	can_load_6 <= not load_conflict_6 and load_req_valid_6;
	can_load_7 <= not load_conflict_7 and load_req_valid_7;
	-- Priority Masking Begin
	-- CyclicPriorityMask(load_idx_oh_0, can_load, ldq_head_oh)
	TEMP_29_double_in(0) <= can_load_0;
	TEMP_29_double_in(8) <= can_load_0;
	TEMP_29_double_in(1) <= can_load_1;
	TEMP_29_double_in(9) <= can_load_1;
	TEMP_29_double_in(2) <= can_load_2;
	TEMP_29_double_in(10) <= can_load_2;
	TEMP_29_double_in(3) <= can_load_3;
	TEMP_29_double_in(11) <= can_load_3;
	TEMP_29_double_in(4) <= can_load_4;
	TEMP_29_double_in(12) <= can_load_4;
	TEMP_29_double_in(5) <= can_load_5;
	TEMP_29_double_in(13) <= can_load_5;
	TEMP_29_double_in(6) <= can_load_6;
	TEMP_29_double_in(14) <= can_load_6;
	TEMP_29_double_in(7) <= can_load_7;
	TEMP_29_double_in(15) <= can_load_7;
	TEMP_29_double_out <= TEMP_29_double_in and not std_logic_vector( unsigned( TEMP_29_double_in ) - unsigned( "00000000" & ldq_head_oh ) );
	load_idx_oh_0 <= TEMP_29_double_out(7 downto 0) or TEMP_29_double_out(15 downto 8);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(load_en_0, can_load, or)
	TEMP_30_res_0 <= can_load_0 or can_load_4;
	TEMP_30_res_1 <= can_load_1 or can_load_5;
	TEMP_30_res_2 <= can_load_2 or can_load_6;
	TEMP_30_res_3 <= can_load_3 or can_load_7;
	-- Layer End
	TEMP_31_res_0 <= TEMP_30_res_0 or TEMP_30_res_2;
	TEMP_31_res_1 <= TEMP_30_res_1 or TEMP_30_res_3;
	-- Layer End
	load_en_0 <= TEMP_31_res_0 or TEMP_31_res_1;
	-- Reduction End

	st_ld_conflict(0) <= ldq_alloc_0_q and not store_is_older_0_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_0(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_0_q );
	st_ld_conflict(1) <= ldq_alloc_1_q and not store_is_older_1_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_1(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_1_q );
	st_ld_conflict(2) <= ldq_alloc_2_q and not store_is_older_2_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_2(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_2_q );
	st_ld_conflict(3) <= ldq_alloc_3_q and not store_is_older_3_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_3(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_3_q );
	st_ld_conflict(4) <= ldq_alloc_4_q and not store_is_older_4_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_4(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_4_q );
	st_ld_conflict(5) <= ldq_alloc_5_q and not store_is_older_5_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_5(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_5_q );
	st_ld_conflict(6) <= ldq_alloc_6_q and not store_is_older_6_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_6(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_6_q );
	st_ld_conflict(7) <= ldq_alloc_7_q and not store_is_older_7_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_7(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_7_q );
	-- Reduction Begin
	-- Reduce(store_conflict, st_ld_conflict, or)
	TEMP_32_res(0) <= st_ld_conflict(0) or st_ld_conflict(4);
	TEMP_32_res(1) <= st_ld_conflict(1) or st_ld_conflict(5);
	TEMP_32_res(2) <= st_ld_conflict(2) or st_ld_conflict(6);
	TEMP_32_res(3) <= st_ld_conflict(3) or st_ld_conflict(7);
	-- Layer End
	TEMP_33_res(0) <= TEMP_32_res(0) or TEMP_32_res(2);
	TEMP_33_res(1) <= TEMP_32_res(1) or TEMP_32_res(3);
	-- Layer End
	store_conflict <= TEMP_33_res(0) or TEMP_33_res(1);
	-- Reduction End

	-- MuxLookUp Begin
	-- MuxLookUp(store_valid, stq_alloc, stq_issue)
	store_valid <= 
	stq_alloc_0_q when (stq_issue_q = "000") else
	stq_alloc_1_q when (stq_issue_q = "001") else
	stq_alloc_2_q when (stq_issue_q = "010") else
	stq_alloc_3_q when (stq_issue_q = "011") else
	stq_alloc_4_q when (stq_issue_q = "100") else
	stq_alloc_5_q when (stq_issue_q = "101") else
	stq_alloc_6_q when (stq_issue_q = "110") else
	stq_alloc_7_q when (stq_issue_q = "111") else
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
	stq_data_valid_5_q when (stq_issue_q = "101") else
	stq_data_valid_6_q when (stq_issue_q = "110") else
	stq_data_valid_7_q when (stq_issue_q = "111") else
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
	stq_addr_valid_5_q when (stq_issue_q = "101") else
	stq_addr_valid_6_q when (stq_issue_q = "110") else
	stq_addr_valid_7_q when (stq_issue_q = "111") else
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
	stq_last_oh(4) <= '1' when stq_tail_q = "101" else '0';
	stq_last_oh(5) <= '1' when stq_tail_q = "110" else '0';
	stq_last_oh(6) <= '1' when stq_tail_q = "111" else '0';
	stq_last_oh(7) <= '1' when stq_tail_q = "000" else '0';
	-- Bits To One-Hot End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_0, ld_st_conflict_0, stq_last_oh)
	TEMP_34_double_in(0) <= ld_st_conflict_0(7);
	TEMP_34_double_in(8) <= ld_st_conflict_0(7);
	TEMP_34_double_in(1) <= ld_st_conflict_0(6);
	TEMP_34_double_in(9) <= ld_st_conflict_0(6);
	TEMP_34_double_in(2) <= ld_st_conflict_0(5);
	TEMP_34_double_in(10) <= ld_st_conflict_0(5);
	TEMP_34_double_in(3) <= ld_st_conflict_0(4);
	TEMP_34_double_in(11) <= ld_st_conflict_0(4);
	TEMP_34_double_in(4) <= ld_st_conflict_0(3);
	TEMP_34_double_in(12) <= ld_st_conflict_0(3);
	TEMP_34_double_in(5) <= ld_st_conflict_0(2);
	TEMP_34_double_in(13) <= ld_st_conflict_0(2);
	TEMP_34_double_in(6) <= ld_st_conflict_0(1);
	TEMP_34_double_in(14) <= ld_st_conflict_0(1);
	TEMP_34_double_in(7) <= ld_st_conflict_0(0);
	TEMP_34_double_in(15) <= ld_st_conflict_0(0);
	TEMP_34_base_rev(0) <= stq_last_oh(7);
	TEMP_34_base_rev(1) <= stq_last_oh(6);
	TEMP_34_base_rev(2) <= stq_last_oh(5);
	TEMP_34_base_rev(3) <= stq_last_oh(4);
	TEMP_34_base_rev(4) <= stq_last_oh(3);
	TEMP_34_base_rev(5) <= stq_last_oh(2);
	TEMP_34_base_rev(6) <= stq_last_oh(1);
	TEMP_34_base_rev(7) <= stq_last_oh(0);
	TEMP_34_double_out <= TEMP_34_double_in and not std_logic_vector( unsigned( TEMP_34_double_in ) - unsigned( "00000000" & TEMP_34_base_rev ) );
	bypass_idx_oh_0(7) <= TEMP_34_double_out(0) or TEMP_34_double_out(8);
	bypass_idx_oh_0(6) <= TEMP_34_double_out(1) or TEMP_34_double_out(9);
	bypass_idx_oh_0(5) <= TEMP_34_double_out(2) or TEMP_34_double_out(10);
	bypass_idx_oh_0(4) <= TEMP_34_double_out(3) or TEMP_34_double_out(11);
	bypass_idx_oh_0(3) <= TEMP_34_double_out(4) or TEMP_34_double_out(12);
	bypass_idx_oh_0(2) <= TEMP_34_double_out(5) or TEMP_34_double_out(13);
	bypass_idx_oh_0(1) <= TEMP_34_double_out(6) or TEMP_34_double_out(14);
	bypass_idx_oh_0(0) <= TEMP_34_double_out(7) or TEMP_34_double_out(15);
	-- Priority Masking End

	bypass_en_vec_0 <= bypass_idx_oh_0 and can_bypass_0;
	-- Reduction Begin
	-- Reduce(bypass_en_0, bypass_en_vec_0, or)
	TEMP_35_res(0) <= bypass_en_vec_0(0) or bypass_en_vec_0(4);
	TEMP_35_res(1) <= bypass_en_vec_0(1) or bypass_en_vec_0(5);
	TEMP_35_res(2) <= bypass_en_vec_0(2) or bypass_en_vec_0(6);
	TEMP_35_res(3) <= bypass_en_vec_0(3) or bypass_en_vec_0(7);
	-- Layer End
	TEMP_36_res(0) <= TEMP_35_res(0) or TEMP_35_res(2);
	TEMP_36_res(1) <= TEMP_35_res(1) or TEMP_35_res(3);
	-- Layer End
	bypass_en_0 <= TEMP_36_res(0) or TEMP_36_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_1, ld_st_conflict_1, stq_last_oh)
	TEMP_37_double_in(0) <= ld_st_conflict_1(7);
	TEMP_37_double_in(8) <= ld_st_conflict_1(7);
	TEMP_37_double_in(1) <= ld_st_conflict_1(6);
	TEMP_37_double_in(9) <= ld_st_conflict_1(6);
	TEMP_37_double_in(2) <= ld_st_conflict_1(5);
	TEMP_37_double_in(10) <= ld_st_conflict_1(5);
	TEMP_37_double_in(3) <= ld_st_conflict_1(4);
	TEMP_37_double_in(11) <= ld_st_conflict_1(4);
	TEMP_37_double_in(4) <= ld_st_conflict_1(3);
	TEMP_37_double_in(12) <= ld_st_conflict_1(3);
	TEMP_37_double_in(5) <= ld_st_conflict_1(2);
	TEMP_37_double_in(13) <= ld_st_conflict_1(2);
	TEMP_37_double_in(6) <= ld_st_conflict_1(1);
	TEMP_37_double_in(14) <= ld_st_conflict_1(1);
	TEMP_37_double_in(7) <= ld_st_conflict_1(0);
	TEMP_37_double_in(15) <= ld_st_conflict_1(0);
	TEMP_37_base_rev(0) <= stq_last_oh(7);
	TEMP_37_base_rev(1) <= stq_last_oh(6);
	TEMP_37_base_rev(2) <= stq_last_oh(5);
	TEMP_37_base_rev(3) <= stq_last_oh(4);
	TEMP_37_base_rev(4) <= stq_last_oh(3);
	TEMP_37_base_rev(5) <= stq_last_oh(2);
	TEMP_37_base_rev(6) <= stq_last_oh(1);
	TEMP_37_base_rev(7) <= stq_last_oh(0);
	TEMP_37_double_out <= TEMP_37_double_in and not std_logic_vector( unsigned( TEMP_37_double_in ) - unsigned( "00000000" & TEMP_37_base_rev ) );
	bypass_idx_oh_1(7) <= TEMP_37_double_out(0) or TEMP_37_double_out(8);
	bypass_idx_oh_1(6) <= TEMP_37_double_out(1) or TEMP_37_double_out(9);
	bypass_idx_oh_1(5) <= TEMP_37_double_out(2) or TEMP_37_double_out(10);
	bypass_idx_oh_1(4) <= TEMP_37_double_out(3) or TEMP_37_double_out(11);
	bypass_idx_oh_1(3) <= TEMP_37_double_out(4) or TEMP_37_double_out(12);
	bypass_idx_oh_1(2) <= TEMP_37_double_out(5) or TEMP_37_double_out(13);
	bypass_idx_oh_1(1) <= TEMP_37_double_out(6) or TEMP_37_double_out(14);
	bypass_idx_oh_1(0) <= TEMP_37_double_out(7) or TEMP_37_double_out(15);
	-- Priority Masking End

	bypass_en_vec_1 <= bypass_idx_oh_1 and can_bypass_1;
	-- Reduction Begin
	-- Reduce(bypass_en_1, bypass_en_vec_1, or)
	TEMP_38_res(0) <= bypass_en_vec_1(0) or bypass_en_vec_1(4);
	TEMP_38_res(1) <= bypass_en_vec_1(1) or bypass_en_vec_1(5);
	TEMP_38_res(2) <= bypass_en_vec_1(2) or bypass_en_vec_1(6);
	TEMP_38_res(3) <= bypass_en_vec_1(3) or bypass_en_vec_1(7);
	-- Layer End
	TEMP_39_res(0) <= TEMP_38_res(0) or TEMP_38_res(2);
	TEMP_39_res(1) <= TEMP_38_res(1) or TEMP_38_res(3);
	-- Layer End
	bypass_en_1 <= TEMP_39_res(0) or TEMP_39_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_2, ld_st_conflict_2, stq_last_oh)
	TEMP_40_double_in(0) <= ld_st_conflict_2(7);
	TEMP_40_double_in(8) <= ld_st_conflict_2(7);
	TEMP_40_double_in(1) <= ld_st_conflict_2(6);
	TEMP_40_double_in(9) <= ld_st_conflict_2(6);
	TEMP_40_double_in(2) <= ld_st_conflict_2(5);
	TEMP_40_double_in(10) <= ld_st_conflict_2(5);
	TEMP_40_double_in(3) <= ld_st_conflict_2(4);
	TEMP_40_double_in(11) <= ld_st_conflict_2(4);
	TEMP_40_double_in(4) <= ld_st_conflict_2(3);
	TEMP_40_double_in(12) <= ld_st_conflict_2(3);
	TEMP_40_double_in(5) <= ld_st_conflict_2(2);
	TEMP_40_double_in(13) <= ld_st_conflict_2(2);
	TEMP_40_double_in(6) <= ld_st_conflict_2(1);
	TEMP_40_double_in(14) <= ld_st_conflict_2(1);
	TEMP_40_double_in(7) <= ld_st_conflict_2(0);
	TEMP_40_double_in(15) <= ld_st_conflict_2(0);
	TEMP_40_base_rev(0) <= stq_last_oh(7);
	TEMP_40_base_rev(1) <= stq_last_oh(6);
	TEMP_40_base_rev(2) <= stq_last_oh(5);
	TEMP_40_base_rev(3) <= stq_last_oh(4);
	TEMP_40_base_rev(4) <= stq_last_oh(3);
	TEMP_40_base_rev(5) <= stq_last_oh(2);
	TEMP_40_base_rev(6) <= stq_last_oh(1);
	TEMP_40_base_rev(7) <= stq_last_oh(0);
	TEMP_40_double_out <= TEMP_40_double_in and not std_logic_vector( unsigned( TEMP_40_double_in ) - unsigned( "00000000" & TEMP_40_base_rev ) );
	bypass_idx_oh_2(7) <= TEMP_40_double_out(0) or TEMP_40_double_out(8);
	bypass_idx_oh_2(6) <= TEMP_40_double_out(1) or TEMP_40_double_out(9);
	bypass_idx_oh_2(5) <= TEMP_40_double_out(2) or TEMP_40_double_out(10);
	bypass_idx_oh_2(4) <= TEMP_40_double_out(3) or TEMP_40_double_out(11);
	bypass_idx_oh_2(3) <= TEMP_40_double_out(4) or TEMP_40_double_out(12);
	bypass_idx_oh_2(2) <= TEMP_40_double_out(5) or TEMP_40_double_out(13);
	bypass_idx_oh_2(1) <= TEMP_40_double_out(6) or TEMP_40_double_out(14);
	bypass_idx_oh_2(0) <= TEMP_40_double_out(7) or TEMP_40_double_out(15);
	-- Priority Masking End

	bypass_en_vec_2 <= bypass_idx_oh_2 and can_bypass_2;
	-- Reduction Begin
	-- Reduce(bypass_en_2, bypass_en_vec_2, or)
	TEMP_41_res(0) <= bypass_en_vec_2(0) or bypass_en_vec_2(4);
	TEMP_41_res(1) <= bypass_en_vec_2(1) or bypass_en_vec_2(5);
	TEMP_41_res(2) <= bypass_en_vec_2(2) or bypass_en_vec_2(6);
	TEMP_41_res(3) <= bypass_en_vec_2(3) or bypass_en_vec_2(7);
	-- Layer End
	TEMP_42_res(0) <= TEMP_41_res(0) or TEMP_41_res(2);
	TEMP_42_res(1) <= TEMP_41_res(1) or TEMP_41_res(3);
	-- Layer End
	bypass_en_2 <= TEMP_42_res(0) or TEMP_42_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_3, ld_st_conflict_3, stq_last_oh)
	TEMP_43_double_in(0) <= ld_st_conflict_3(7);
	TEMP_43_double_in(8) <= ld_st_conflict_3(7);
	TEMP_43_double_in(1) <= ld_st_conflict_3(6);
	TEMP_43_double_in(9) <= ld_st_conflict_3(6);
	TEMP_43_double_in(2) <= ld_st_conflict_3(5);
	TEMP_43_double_in(10) <= ld_st_conflict_3(5);
	TEMP_43_double_in(3) <= ld_st_conflict_3(4);
	TEMP_43_double_in(11) <= ld_st_conflict_3(4);
	TEMP_43_double_in(4) <= ld_st_conflict_3(3);
	TEMP_43_double_in(12) <= ld_st_conflict_3(3);
	TEMP_43_double_in(5) <= ld_st_conflict_3(2);
	TEMP_43_double_in(13) <= ld_st_conflict_3(2);
	TEMP_43_double_in(6) <= ld_st_conflict_3(1);
	TEMP_43_double_in(14) <= ld_st_conflict_3(1);
	TEMP_43_double_in(7) <= ld_st_conflict_3(0);
	TEMP_43_double_in(15) <= ld_st_conflict_3(0);
	TEMP_43_base_rev(0) <= stq_last_oh(7);
	TEMP_43_base_rev(1) <= stq_last_oh(6);
	TEMP_43_base_rev(2) <= stq_last_oh(5);
	TEMP_43_base_rev(3) <= stq_last_oh(4);
	TEMP_43_base_rev(4) <= stq_last_oh(3);
	TEMP_43_base_rev(5) <= stq_last_oh(2);
	TEMP_43_base_rev(6) <= stq_last_oh(1);
	TEMP_43_base_rev(7) <= stq_last_oh(0);
	TEMP_43_double_out <= TEMP_43_double_in and not std_logic_vector( unsigned( TEMP_43_double_in ) - unsigned( "00000000" & TEMP_43_base_rev ) );
	bypass_idx_oh_3(7) <= TEMP_43_double_out(0) or TEMP_43_double_out(8);
	bypass_idx_oh_3(6) <= TEMP_43_double_out(1) or TEMP_43_double_out(9);
	bypass_idx_oh_3(5) <= TEMP_43_double_out(2) or TEMP_43_double_out(10);
	bypass_idx_oh_3(4) <= TEMP_43_double_out(3) or TEMP_43_double_out(11);
	bypass_idx_oh_3(3) <= TEMP_43_double_out(4) or TEMP_43_double_out(12);
	bypass_idx_oh_3(2) <= TEMP_43_double_out(5) or TEMP_43_double_out(13);
	bypass_idx_oh_3(1) <= TEMP_43_double_out(6) or TEMP_43_double_out(14);
	bypass_idx_oh_3(0) <= TEMP_43_double_out(7) or TEMP_43_double_out(15);
	-- Priority Masking End

	bypass_en_vec_3 <= bypass_idx_oh_3 and can_bypass_3;
	-- Reduction Begin
	-- Reduce(bypass_en_3, bypass_en_vec_3, or)
	TEMP_44_res(0) <= bypass_en_vec_3(0) or bypass_en_vec_3(4);
	TEMP_44_res(1) <= bypass_en_vec_3(1) or bypass_en_vec_3(5);
	TEMP_44_res(2) <= bypass_en_vec_3(2) or bypass_en_vec_3(6);
	TEMP_44_res(3) <= bypass_en_vec_3(3) or bypass_en_vec_3(7);
	-- Layer End
	TEMP_45_res(0) <= TEMP_44_res(0) or TEMP_44_res(2);
	TEMP_45_res(1) <= TEMP_44_res(1) or TEMP_44_res(3);
	-- Layer End
	bypass_en_3 <= TEMP_45_res(0) or TEMP_45_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_4, ld_st_conflict_4, stq_last_oh)
	TEMP_46_double_in(0) <= ld_st_conflict_4(7);
	TEMP_46_double_in(8) <= ld_st_conflict_4(7);
	TEMP_46_double_in(1) <= ld_st_conflict_4(6);
	TEMP_46_double_in(9) <= ld_st_conflict_4(6);
	TEMP_46_double_in(2) <= ld_st_conflict_4(5);
	TEMP_46_double_in(10) <= ld_st_conflict_4(5);
	TEMP_46_double_in(3) <= ld_st_conflict_4(4);
	TEMP_46_double_in(11) <= ld_st_conflict_4(4);
	TEMP_46_double_in(4) <= ld_st_conflict_4(3);
	TEMP_46_double_in(12) <= ld_st_conflict_4(3);
	TEMP_46_double_in(5) <= ld_st_conflict_4(2);
	TEMP_46_double_in(13) <= ld_st_conflict_4(2);
	TEMP_46_double_in(6) <= ld_st_conflict_4(1);
	TEMP_46_double_in(14) <= ld_st_conflict_4(1);
	TEMP_46_double_in(7) <= ld_st_conflict_4(0);
	TEMP_46_double_in(15) <= ld_st_conflict_4(0);
	TEMP_46_base_rev(0) <= stq_last_oh(7);
	TEMP_46_base_rev(1) <= stq_last_oh(6);
	TEMP_46_base_rev(2) <= stq_last_oh(5);
	TEMP_46_base_rev(3) <= stq_last_oh(4);
	TEMP_46_base_rev(4) <= stq_last_oh(3);
	TEMP_46_base_rev(5) <= stq_last_oh(2);
	TEMP_46_base_rev(6) <= stq_last_oh(1);
	TEMP_46_base_rev(7) <= stq_last_oh(0);
	TEMP_46_double_out <= TEMP_46_double_in and not std_logic_vector( unsigned( TEMP_46_double_in ) - unsigned( "00000000" & TEMP_46_base_rev ) );
	bypass_idx_oh_4(7) <= TEMP_46_double_out(0) or TEMP_46_double_out(8);
	bypass_idx_oh_4(6) <= TEMP_46_double_out(1) or TEMP_46_double_out(9);
	bypass_idx_oh_4(5) <= TEMP_46_double_out(2) or TEMP_46_double_out(10);
	bypass_idx_oh_4(4) <= TEMP_46_double_out(3) or TEMP_46_double_out(11);
	bypass_idx_oh_4(3) <= TEMP_46_double_out(4) or TEMP_46_double_out(12);
	bypass_idx_oh_4(2) <= TEMP_46_double_out(5) or TEMP_46_double_out(13);
	bypass_idx_oh_4(1) <= TEMP_46_double_out(6) or TEMP_46_double_out(14);
	bypass_idx_oh_4(0) <= TEMP_46_double_out(7) or TEMP_46_double_out(15);
	-- Priority Masking End

	bypass_en_vec_4 <= bypass_idx_oh_4 and can_bypass_4;
	-- Reduction Begin
	-- Reduce(bypass_en_4, bypass_en_vec_4, or)
	TEMP_47_res(0) <= bypass_en_vec_4(0) or bypass_en_vec_4(4);
	TEMP_47_res(1) <= bypass_en_vec_4(1) or bypass_en_vec_4(5);
	TEMP_47_res(2) <= bypass_en_vec_4(2) or bypass_en_vec_4(6);
	TEMP_47_res(3) <= bypass_en_vec_4(3) or bypass_en_vec_4(7);
	-- Layer End
	TEMP_48_res(0) <= TEMP_47_res(0) or TEMP_47_res(2);
	TEMP_48_res(1) <= TEMP_47_res(1) or TEMP_47_res(3);
	-- Layer End
	bypass_en_4 <= TEMP_48_res(0) or TEMP_48_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_5, ld_st_conflict_5, stq_last_oh)
	TEMP_49_double_in(0) <= ld_st_conflict_5(7);
	TEMP_49_double_in(8) <= ld_st_conflict_5(7);
	TEMP_49_double_in(1) <= ld_st_conflict_5(6);
	TEMP_49_double_in(9) <= ld_st_conflict_5(6);
	TEMP_49_double_in(2) <= ld_st_conflict_5(5);
	TEMP_49_double_in(10) <= ld_st_conflict_5(5);
	TEMP_49_double_in(3) <= ld_st_conflict_5(4);
	TEMP_49_double_in(11) <= ld_st_conflict_5(4);
	TEMP_49_double_in(4) <= ld_st_conflict_5(3);
	TEMP_49_double_in(12) <= ld_st_conflict_5(3);
	TEMP_49_double_in(5) <= ld_st_conflict_5(2);
	TEMP_49_double_in(13) <= ld_st_conflict_5(2);
	TEMP_49_double_in(6) <= ld_st_conflict_5(1);
	TEMP_49_double_in(14) <= ld_st_conflict_5(1);
	TEMP_49_double_in(7) <= ld_st_conflict_5(0);
	TEMP_49_double_in(15) <= ld_st_conflict_5(0);
	TEMP_49_base_rev(0) <= stq_last_oh(7);
	TEMP_49_base_rev(1) <= stq_last_oh(6);
	TEMP_49_base_rev(2) <= stq_last_oh(5);
	TEMP_49_base_rev(3) <= stq_last_oh(4);
	TEMP_49_base_rev(4) <= stq_last_oh(3);
	TEMP_49_base_rev(5) <= stq_last_oh(2);
	TEMP_49_base_rev(6) <= stq_last_oh(1);
	TEMP_49_base_rev(7) <= stq_last_oh(0);
	TEMP_49_double_out <= TEMP_49_double_in and not std_logic_vector( unsigned( TEMP_49_double_in ) - unsigned( "00000000" & TEMP_49_base_rev ) );
	bypass_idx_oh_5(7) <= TEMP_49_double_out(0) or TEMP_49_double_out(8);
	bypass_idx_oh_5(6) <= TEMP_49_double_out(1) or TEMP_49_double_out(9);
	bypass_idx_oh_5(5) <= TEMP_49_double_out(2) or TEMP_49_double_out(10);
	bypass_idx_oh_5(4) <= TEMP_49_double_out(3) or TEMP_49_double_out(11);
	bypass_idx_oh_5(3) <= TEMP_49_double_out(4) or TEMP_49_double_out(12);
	bypass_idx_oh_5(2) <= TEMP_49_double_out(5) or TEMP_49_double_out(13);
	bypass_idx_oh_5(1) <= TEMP_49_double_out(6) or TEMP_49_double_out(14);
	bypass_idx_oh_5(0) <= TEMP_49_double_out(7) or TEMP_49_double_out(15);
	-- Priority Masking End

	bypass_en_vec_5 <= bypass_idx_oh_5 and can_bypass_5;
	-- Reduction Begin
	-- Reduce(bypass_en_5, bypass_en_vec_5, or)
	TEMP_50_res(0) <= bypass_en_vec_5(0) or bypass_en_vec_5(4);
	TEMP_50_res(1) <= bypass_en_vec_5(1) or bypass_en_vec_5(5);
	TEMP_50_res(2) <= bypass_en_vec_5(2) or bypass_en_vec_5(6);
	TEMP_50_res(3) <= bypass_en_vec_5(3) or bypass_en_vec_5(7);
	-- Layer End
	TEMP_51_res(0) <= TEMP_50_res(0) or TEMP_50_res(2);
	TEMP_51_res(1) <= TEMP_50_res(1) or TEMP_50_res(3);
	-- Layer End
	bypass_en_5 <= TEMP_51_res(0) or TEMP_51_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_6, ld_st_conflict_6, stq_last_oh)
	TEMP_52_double_in(0) <= ld_st_conflict_6(7);
	TEMP_52_double_in(8) <= ld_st_conflict_6(7);
	TEMP_52_double_in(1) <= ld_st_conflict_6(6);
	TEMP_52_double_in(9) <= ld_st_conflict_6(6);
	TEMP_52_double_in(2) <= ld_st_conflict_6(5);
	TEMP_52_double_in(10) <= ld_st_conflict_6(5);
	TEMP_52_double_in(3) <= ld_st_conflict_6(4);
	TEMP_52_double_in(11) <= ld_st_conflict_6(4);
	TEMP_52_double_in(4) <= ld_st_conflict_6(3);
	TEMP_52_double_in(12) <= ld_st_conflict_6(3);
	TEMP_52_double_in(5) <= ld_st_conflict_6(2);
	TEMP_52_double_in(13) <= ld_st_conflict_6(2);
	TEMP_52_double_in(6) <= ld_st_conflict_6(1);
	TEMP_52_double_in(14) <= ld_st_conflict_6(1);
	TEMP_52_double_in(7) <= ld_st_conflict_6(0);
	TEMP_52_double_in(15) <= ld_st_conflict_6(0);
	TEMP_52_base_rev(0) <= stq_last_oh(7);
	TEMP_52_base_rev(1) <= stq_last_oh(6);
	TEMP_52_base_rev(2) <= stq_last_oh(5);
	TEMP_52_base_rev(3) <= stq_last_oh(4);
	TEMP_52_base_rev(4) <= stq_last_oh(3);
	TEMP_52_base_rev(5) <= stq_last_oh(2);
	TEMP_52_base_rev(6) <= stq_last_oh(1);
	TEMP_52_base_rev(7) <= stq_last_oh(0);
	TEMP_52_double_out <= TEMP_52_double_in and not std_logic_vector( unsigned( TEMP_52_double_in ) - unsigned( "00000000" & TEMP_52_base_rev ) );
	bypass_idx_oh_6(7) <= TEMP_52_double_out(0) or TEMP_52_double_out(8);
	bypass_idx_oh_6(6) <= TEMP_52_double_out(1) or TEMP_52_double_out(9);
	bypass_idx_oh_6(5) <= TEMP_52_double_out(2) or TEMP_52_double_out(10);
	bypass_idx_oh_6(4) <= TEMP_52_double_out(3) or TEMP_52_double_out(11);
	bypass_idx_oh_6(3) <= TEMP_52_double_out(4) or TEMP_52_double_out(12);
	bypass_idx_oh_6(2) <= TEMP_52_double_out(5) or TEMP_52_double_out(13);
	bypass_idx_oh_6(1) <= TEMP_52_double_out(6) or TEMP_52_double_out(14);
	bypass_idx_oh_6(0) <= TEMP_52_double_out(7) or TEMP_52_double_out(15);
	-- Priority Masking End

	bypass_en_vec_6 <= bypass_idx_oh_6 and can_bypass_6;
	-- Reduction Begin
	-- Reduce(bypass_en_6, bypass_en_vec_6, or)
	TEMP_53_res(0) <= bypass_en_vec_6(0) or bypass_en_vec_6(4);
	TEMP_53_res(1) <= bypass_en_vec_6(1) or bypass_en_vec_6(5);
	TEMP_53_res(2) <= bypass_en_vec_6(2) or bypass_en_vec_6(6);
	TEMP_53_res(3) <= bypass_en_vec_6(3) or bypass_en_vec_6(7);
	-- Layer End
	TEMP_54_res(0) <= TEMP_53_res(0) or TEMP_53_res(2);
	TEMP_54_res(1) <= TEMP_53_res(1) or TEMP_53_res(3);
	-- Layer End
	bypass_en_6 <= TEMP_54_res(0) or TEMP_54_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_7, ld_st_conflict_7, stq_last_oh)
	TEMP_55_double_in(0) <= ld_st_conflict_7(7);
	TEMP_55_double_in(8) <= ld_st_conflict_7(7);
	TEMP_55_double_in(1) <= ld_st_conflict_7(6);
	TEMP_55_double_in(9) <= ld_st_conflict_7(6);
	TEMP_55_double_in(2) <= ld_st_conflict_7(5);
	TEMP_55_double_in(10) <= ld_st_conflict_7(5);
	TEMP_55_double_in(3) <= ld_st_conflict_7(4);
	TEMP_55_double_in(11) <= ld_st_conflict_7(4);
	TEMP_55_double_in(4) <= ld_st_conflict_7(3);
	TEMP_55_double_in(12) <= ld_st_conflict_7(3);
	TEMP_55_double_in(5) <= ld_st_conflict_7(2);
	TEMP_55_double_in(13) <= ld_st_conflict_7(2);
	TEMP_55_double_in(6) <= ld_st_conflict_7(1);
	TEMP_55_double_in(14) <= ld_st_conflict_7(1);
	TEMP_55_double_in(7) <= ld_st_conflict_7(0);
	TEMP_55_double_in(15) <= ld_st_conflict_7(0);
	TEMP_55_base_rev(0) <= stq_last_oh(7);
	TEMP_55_base_rev(1) <= stq_last_oh(6);
	TEMP_55_base_rev(2) <= stq_last_oh(5);
	TEMP_55_base_rev(3) <= stq_last_oh(4);
	TEMP_55_base_rev(4) <= stq_last_oh(3);
	TEMP_55_base_rev(5) <= stq_last_oh(2);
	TEMP_55_base_rev(6) <= stq_last_oh(1);
	TEMP_55_base_rev(7) <= stq_last_oh(0);
	TEMP_55_double_out <= TEMP_55_double_in and not std_logic_vector( unsigned( TEMP_55_double_in ) - unsigned( "00000000" & TEMP_55_base_rev ) );
	bypass_idx_oh_7(7) <= TEMP_55_double_out(0) or TEMP_55_double_out(8);
	bypass_idx_oh_7(6) <= TEMP_55_double_out(1) or TEMP_55_double_out(9);
	bypass_idx_oh_7(5) <= TEMP_55_double_out(2) or TEMP_55_double_out(10);
	bypass_idx_oh_7(4) <= TEMP_55_double_out(3) or TEMP_55_double_out(11);
	bypass_idx_oh_7(3) <= TEMP_55_double_out(4) or TEMP_55_double_out(12);
	bypass_idx_oh_7(2) <= TEMP_55_double_out(5) or TEMP_55_double_out(13);
	bypass_idx_oh_7(1) <= TEMP_55_double_out(6) or TEMP_55_double_out(14);
	bypass_idx_oh_7(0) <= TEMP_55_double_out(7) or TEMP_55_double_out(15);
	-- Priority Masking End

	bypass_en_vec_7 <= bypass_idx_oh_7 and can_bypass_7;
	-- Reduction Begin
	-- Reduce(bypass_en_7, bypass_en_vec_7, or)
	TEMP_56_res(0) <= bypass_en_vec_7(0) or bypass_en_vec_7(4);
	TEMP_56_res(1) <= bypass_en_vec_7(1) or bypass_en_vec_7(5);
	TEMP_56_res(2) <= bypass_en_vec_7(2) or bypass_en_vec_7(6);
	TEMP_56_res(3) <= bypass_en_vec_7(3) or bypass_en_vec_7(7);
	-- Layer End
	TEMP_57_res(0) <= TEMP_56_res(0) or TEMP_56_res(2);
	TEMP_57_res(1) <= TEMP_56_res(1) or TEMP_56_res(3);
	-- Layer End
	bypass_en_7 <= TEMP_57_res(0) or TEMP_57_res(1);
	-- Reduction End

	rreq_valid_0_o <= load_en_0;
	-- One-Hot To Bits Begin
	-- OHToBits(rreq_id_0, load_idx_oh_0)
	TEMP_58_in_0_0 <= '0';
	TEMP_58_in_0_1 <= load_idx_oh_0(1);
	TEMP_58_in_0_2 <= '0';
	TEMP_58_in_0_3 <= load_idx_oh_0(3);
	TEMP_58_in_0_4 <= '0';
	TEMP_58_in_0_5 <= load_idx_oh_0(5);
	TEMP_58_in_0_6 <= '0';
	TEMP_58_in_0_7 <= load_idx_oh_0(7);
	TEMP_59_res_0 <= TEMP_58_in_0_0 or TEMP_58_in_0_4;
	TEMP_59_res_1 <= TEMP_58_in_0_1 or TEMP_58_in_0_5;
	TEMP_59_res_2 <= TEMP_58_in_0_2 or TEMP_58_in_0_6;
	TEMP_59_res_3 <= TEMP_58_in_0_3 or TEMP_58_in_0_7;
	-- Layer End
	TEMP_60_res_0 <= TEMP_59_res_0 or TEMP_59_res_2;
	TEMP_60_res_1 <= TEMP_59_res_1 or TEMP_59_res_3;
	-- Layer End
	TEMP_58_out_0 <= TEMP_60_res_0 or TEMP_60_res_1;
	rreq_id_0_o(0) <= TEMP_58_out_0;
	TEMP_60_in_1_0 <= '0';
	TEMP_60_in_1_1 <= '0';
	TEMP_60_in_1_2 <= load_idx_oh_0(2);
	TEMP_60_in_1_3 <= load_idx_oh_0(3);
	TEMP_60_in_1_4 <= '0';
	TEMP_60_in_1_5 <= '0';
	TEMP_60_in_1_6 <= load_idx_oh_0(6);
	TEMP_60_in_1_7 <= load_idx_oh_0(7);
	TEMP_61_res_0 <= TEMP_60_in_1_0 or TEMP_60_in_1_4;
	TEMP_61_res_1 <= TEMP_60_in_1_1 or TEMP_60_in_1_5;
	TEMP_61_res_2 <= TEMP_60_in_1_2 or TEMP_60_in_1_6;
	TEMP_61_res_3 <= TEMP_60_in_1_3 or TEMP_60_in_1_7;
	-- Layer End
	TEMP_62_res_0 <= TEMP_61_res_0 or TEMP_61_res_2;
	TEMP_62_res_1 <= TEMP_61_res_1 or TEMP_61_res_3;
	-- Layer End
	TEMP_60_out_1 <= TEMP_62_res_0 or TEMP_62_res_1;
	rreq_id_0_o(1) <= TEMP_60_out_1;
	TEMP_62_in_2_0 <= '0';
	TEMP_62_in_2_1 <= '0';
	TEMP_62_in_2_2 <= '0';
	TEMP_62_in_2_3 <= '0';
	TEMP_62_in_2_4 <= load_idx_oh_0(4);
	TEMP_62_in_2_5 <= load_idx_oh_0(5);
	TEMP_62_in_2_6 <= load_idx_oh_0(6);
	TEMP_62_in_2_7 <= load_idx_oh_0(7);
	TEMP_63_res_0 <= TEMP_62_in_2_0 or TEMP_62_in_2_4;
	TEMP_63_res_1 <= TEMP_62_in_2_1 or TEMP_62_in_2_5;
	TEMP_63_res_2 <= TEMP_62_in_2_2 or TEMP_62_in_2_6;
	TEMP_63_res_3 <= TEMP_62_in_2_3 or TEMP_62_in_2_7;
	-- Layer End
	TEMP_64_res_0 <= TEMP_63_res_0 or TEMP_63_res_2;
	TEMP_64_res_1 <= TEMP_63_res_1 or TEMP_63_res_3;
	-- Layer End
	TEMP_62_out_2 <= TEMP_64_res_0 or TEMP_64_res_1;
	rreq_id_0_o(2) <= TEMP_62_out_2;
	TEMP_64_in_3_0 <= '0';
	TEMP_64_in_3_1 <= '0';
	TEMP_64_in_3_2 <= '0';
	TEMP_64_in_3_3 <= '0';
	TEMP_64_in_3_4 <= '0';
	TEMP_64_in_3_5 <= '0';
	TEMP_64_in_3_6 <= '0';
	TEMP_64_in_3_7 <= '0';
	TEMP_65_res_0 <= TEMP_64_in_3_0 or TEMP_64_in_3_4;
	TEMP_65_res_1 <= TEMP_64_in_3_1 or TEMP_64_in_3_5;
	TEMP_65_res_2 <= TEMP_64_in_3_2 or TEMP_64_in_3_6;
	TEMP_65_res_3 <= TEMP_64_in_3_3 or TEMP_64_in_3_7;
	-- Layer End
	TEMP_66_res_0 <= TEMP_65_res_0 or TEMP_65_res_2;
	TEMP_66_res_1 <= TEMP_65_res_1 or TEMP_65_res_3;
	-- Layer End
	TEMP_64_out_3 <= TEMP_66_res_0 or TEMP_66_res_1;
	rreq_id_0_o(3) <= TEMP_64_out_3;
	-- One-Hot To Bits End

	-- Mux1H Begin
	-- Mux1H(rreq_addr_0, ldq_addr, load_idx_oh_0)
	TEMP_67_mux_0 <= ldq_addr_0_q when load_idx_oh_0(0) = '1' else "0000000";
	TEMP_67_mux_1 <= ldq_addr_1_q when load_idx_oh_0(1) = '1' else "0000000";
	TEMP_67_mux_2 <= ldq_addr_2_q when load_idx_oh_0(2) = '1' else "0000000";
	TEMP_67_mux_3 <= ldq_addr_3_q when load_idx_oh_0(3) = '1' else "0000000";
	TEMP_67_mux_4 <= ldq_addr_4_q when load_idx_oh_0(4) = '1' else "0000000";
	TEMP_67_mux_5 <= ldq_addr_5_q when load_idx_oh_0(5) = '1' else "0000000";
	TEMP_67_mux_6 <= ldq_addr_6_q when load_idx_oh_0(6) = '1' else "0000000";
	TEMP_67_mux_7 <= ldq_addr_7_q when load_idx_oh_0(7) = '1' else "0000000";
	TEMP_68_res_0 <= TEMP_67_mux_0 or TEMP_67_mux_4;
	TEMP_68_res_1 <= TEMP_67_mux_1 or TEMP_67_mux_5;
	TEMP_68_res_2 <= TEMP_67_mux_2 or TEMP_67_mux_6;
	TEMP_68_res_3 <= TEMP_67_mux_3 or TEMP_67_mux_7;
	-- Layer End
	TEMP_69_res_0 <= TEMP_68_res_0 or TEMP_68_res_2;
	TEMP_69_res_1 <= TEMP_68_res_1 or TEMP_68_res_3;
	-- Layer End
	rreq_addr_0_o <= TEMP_69_res_0 or TEMP_69_res_1;
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
	stq_addr_5_q when (store_idx = "101") else
	stq_addr_6_q when (store_idx = "110") else
	stq_addr_7_q when (store_idx = "111") else
	"0000000";
	-- MuxLookUp End

	-- MuxLookUp Begin
	-- MuxLookUp(wreq_data_0, stq_data, store_idx)
	wreq_data_0_o <= 
	stq_data_0_q when (store_idx = "000") else
	stq_data_1_q when (store_idx = "001") else
	stq_data_2_q when (store_idx = "010") else
	stq_data_3_q when (store_idx = "011") else
	stq_data_4_q when (store_idx = "100") else
	stq_data_5_q when (store_idx = "101") else
	stq_data_6_q when (store_idx = "110") else
	stq_data_7_q when (store_idx = "111") else
	"00000000000000000000000000000000";
	-- MuxLookUp End

	stq_issue_en <= store_en and wreq_ready_0_i;
	read_idx_oh_0_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0000" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_0, rresp_data, read_idx_oh_0)
	TEMP_70_mux_0 <= rresp_data_0_i when read_idx_oh_0_0 = '1' else "00000000000000000000000000000000";
	read_data_0 <= TEMP_70_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_0, read_idx_oh_0, or)
	read_valid_0 <= read_idx_oh_0_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_0, stq_data, bypass_idx_oh_0)
	TEMP_71_mux_0 <= stq_data_0_q when bypass_idx_oh_0(0) = '1' else "00000000000000000000000000000000";
	TEMP_71_mux_1 <= stq_data_1_q when bypass_idx_oh_0(1) = '1' else "00000000000000000000000000000000";
	TEMP_71_mux_2 <= stq_data_2_q when bypass_idx_oh_0(2) = '1' else "00000000000000000000000000000000";
	TEMP_71_mux_3 <= stq_data_3_q when bypass_idx_oh_0(3) = '1' else "00000000000000000000000000000000";
	TEMP_71_mux_4 <= stq_data_4_q when bypass_idx_oh_0(4) = '1' else "00000000000000000000000000000000";
	TEMP_71_mux_5 <= stq_data_5_q when bypass_idx_oh_0(5) = '1' else "00000000000000000000000000000000";
	TEMP_71_mux_6 <= stq_data_6_q when bypass_idx_oh_0(6) = '1' else "00000000000000000000000000000000";
	TEMP_71_mux_7 <= stq_data_7_q when bypass_idx_oh_0(7) = '1' else "00000000000000000000000000000000";
	TEMP_72_res_0 <= TEMP_71_mux_0 or TEMP_71_mux_4;
	TEMP_72_res_1 <= TEMP_71_mux_1 or TEMP_71_mux_5;
	TEMP_72_res_2 <= TEMP_71_mux_2 or TEMP_71_mux_6;
	TEMP_72_res_3 <= TEMP_71_mux_3 or TEMP_71_mux_7;
	-- Layer End
	TEMP_73_res_0 <= TEMP_72_res_0 or TEMP_72_res_2;
	TEMP_73_res_1 <= TEMP_72_res_1 or TEMP_72_res_3;
	-- Layer End
	bypass_data_0 <= TEMP_73_res_0 or TEMP_73_res_1;
	-- Mux1H End

	ldq_data_0_d <= read_data_0 or bypass_data_0;
	ldq_data_wen_0 <= bypass_en_0 or read_valid_0;
	read_idx_oh_1_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0001" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_1, rresp_data, read_idx_oh_1)
	TEMP_74_mux_0 <= rresp_data_0_i when read_idx_oh_1_0 = '1' else "00000000000000000000000000000000";
	read_data_1 <= TEMP_74_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_1, read_idx_oh_1, or)
	read_valid_1 <= read_idx_oh_1_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_1, stq_data, bypass_idx_oh_1)
	TEMP_75_mux_0 <= stq_data_0_q when bypass_idx_oh_1(0) = '1' else "00000000000000000000000000000000";
	TEMP_75_mux_1 <= stq_data_1_q when bypass_idx_oh_1(1) = '1' else "00000000000000000000000000000000";
	TEMP_75_mux_2 <= stq_data_2_q when bypass_idx_oh_1(2) = '1' else "00000000000000000000000000000000";
	TEMP_75_mux_3 <= stq_data_3_q when bypass_idx_oh_1(3) = '1' else "00000000000000000000000000000000";
	TEMP_75_mux_4 <= stq_data_4_q when bypass_idx_oh_1(4) = '1' else "00000000000000000000000000000000";
	TEMP_75_mux_5 <= stq_data_5_q when bypass_idx_oh_1(5) = '1' else "00000000000000000000000000000000";
	TEMP_75_mux_6 <= stq_data_6_q when bypass_idx_oh_1(6) = '1' else "00000000000000000000000000000000";
	TEMP_75_mux_7 <= stq_data_7_q when bypass_idx_oh_1(7) = '1' else "00000000000000000000000000000000";
	TEMP_76_res_0 <= TEMP_75_mux_0 or TEMP_75_mux_4;
	TEMP_76_res_1 <= TEMP_75_mux_1 or TEMP_75_mux_5;
	TEMP_76_res_2 <= TEMP_75_mux_2 or TEMP_75_mux_6;
	TEMP_76_res_3 <= TEMP_75_mux_3 or TEMP_75_mux_7;
	-- Layer End
	TEMP_77_res_0 <= TEMP_76_res_0 or TEMP_76_res_2;
	TEMP_77_res_1 <= TEMP_76_res_1 or TEMP_76_res_3;
	-- Layer End
	bypass_data_1 <= TEMP_77_res_0 or TEMP_77_res_1;
	-- Mux1H End

	ldq_data_1_d <= read_data_1 or bypass_data_1;
	ldq_data_wen_1 <= bypass_en_1 or read_valid_1;
	read_idx_oh_2_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0010" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_2, rresp_data, read_idx_oh_2)
	TEMP_78_mux_0 <= rresp_data_0_i when read_idx_oh_2_0 = '1' else "00000000000000000000000000000000";
	read_data_2 <= TEMP_78_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_2, read_idx_oh_2, or)
	read_valid_2 <= read_idx_oh_2_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_2, stq_data, bypass_idx_oh_2)
	TEMP_79_mux_0 <= stq_data_0_q when bypass_idx_oh_2(0) = '1' else "00000000000000000000000000000000";
	TEMP_79_mux_1 <= stq_data_1_q when bypass_idx_oh_2(1) = '1' else "00000000000000000000000000000000";
	TEMP_79_mux_2 <= stq_data_2_q when bypass_idx_oh_2(2) = '1' else "00000000000000000000000000000000";
	TEMP_79_mux_3 <= stq_data_3_q when bypass_idx_oh_2(3) = '1' else "00000000000000000000000000000000";
	TEMP_79_mux_4 <= stq_data_4_q when bypass_idx_oh_2(4) = '1' else "00000000000000000000000000000000";
	TEMP_79_mux_5 <= stq_data_5_q when bypass_idx_oh_2(5) = '1' else "00000000000000000000000000000000";
	TEMP_79_mux_6 <= stq_data_6_q when bypass_idx_oh_2(6) = '1' else "00000000000000000000000000000000";
	TEMP_79_mux_7 <= stq_data_7_q when bypass_idx_oh_2(7) = '1' else "00000000000000000000000000000000";
	TEMP_80_res_0 <= TEMP_79_mux_0 or TEMP_79_mux_4;
	TEMP_80_res_1 <= TEMP_79_mux_1 or TEMP_79_mux_5;
	TEMP_80_res_2 <= TEMP_79_mux_2 or TEMP_79_mux_6;
	TEMP_80_res_3 <= TEMP_79_mux_3 or TEMP_79_mux_7;
	-- Layer End
	TEMP_81_res_0 <= TEMP_80_res_0 or TEMP_80_res_2;
	TEMP_81_res_1 <= TEMP_80_res_1 or TEMP_80_res_3;
	-- Layer End
	bypass_data_2 <= TEMP_81_res_0 or TEMP_81_res_1;
	-- Mux1H End

	ldq_data_2_d <= read_data_2 or bypass_data_2;
	ldq_data_wen_2 <= bypass_en_2 or read_valid_2;
	read_idx_oh_3_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0011" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_3, rresp_data, read_idx_oh_3)
	TEMP_82_mux_0 <= rresp_data_0_i when read_idx_oh_3_0 = '1' else "00000000000000000000000000000000";
	read_data_3 <= TEMP_82_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_3, read_idx_oh_3, or)
	read_valid_3 <= read_idx_oh_3_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_3, stq_data, bypass_idx_oh_3)
	TEMP_83_mux_0 <= stq_data_0_q when bypass_idx_oh_3(0) = '1' else "00000000000000000000000000000000";
	TEMP_83_mux_1 <= stq_data_1_q when bypass_idx_oh_3(1) = '1' else "00000000000000000000000000000000";
	TEMP_83_mux_2 <= stq_data_2_q when bypass_idx_oh_3(2) = '1' else "00000000000000000000000000000000";
	TEMP_83_mux_3 <= stq_data_3_q when bypass_idx_oh_3(3) = '1' else "00000000000000000000000000000000";
	TEMP_83_mux_4 <= stq_data_4_q when bypass_idx_oh_3(4) = '1' else "00000000000000000000000000000000";
	TEMP_83_mux_5 <= stq_data_5_q when bypass_idx_oh_3(5) = '1' else "00000000000000000000000000000000";
	TEMP_83_mux_6 <= stq_data_6_q when bypass_idx_oh_3(6) = '1' else "00000000000000000000000000000000";
	TEMP_83_mux_7 <= stq_data_7_q when bypass_idx_oh_3(7) = '1' else "00000000000000000000000000000000";
	TEMP_84_res_0 <= TEMP_83_mux_0 or TEMP_83_mux_4;
	TEMP_84_res_1 <= TEMP_83_mux_1 or TEMP_83_mux_5;
	TEMP_84_res_2 <= TEMP_83_mux_2 or TEMP_83_mux_6;
	TEMP_84_res_3 <= TEMP_83_mux_3 or TEMP_83_mux_7;
	-- Layer End
	TEMP_85_res_0 <= TEMP_84_res_0 or TEMP_84_res_2;
	TEMP_85_res_1 <= TEMP_84_res_1 or TEMP_84_res_3;
	-- Layer End
	bypass_data_3 <= TEMP_85_res_0 or TEMP_85_res_1;
	-- Mux1H End

	ldq_data_3_d <= read_data_3 or bypass_data_3;
	ldq_data_wen_3 <= bypass_en_3 or read_valid_3;
	read_idx_oh_4_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0100" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_4, rresp_data, read_idx_oh_4)
	TEMP_86_mux_0 <= rresp_data_0_i when read_idx_oh_4_0 = '1' else "00000000000000000000000000000000";
	read_data_4 <= TEMP_86_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_4, read_idx_oh_4, or)
	read_valid_4 <= read_idx_oh_4_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_4, stq_data, bypass_idx_oh_4)
	TEMP_87_mux_0 <= stq_data_0_q when bypass_idx_oh_4(0) = '1' else "00000000000000000000000000000000";
	TEMP_87_mux_1 <= stq_data_1_q when bypass_idx_oh_4(1) = '1' else "00000000000000000000000000000000";
	TEMP_87_mux_2 <= stq_data_2_q when bypass_idx_oh_4(2) = '1' else "00000000000000000000000000000000";
	TEMP_87_mux_3 <= stq_data_3_q when bypass_idx_oh_4(3) = '1' else "00000000000000000000000000000000";
	TEMP_87_mux_4 <= stq_data_4_q when bypass_idx_oh_4(4) = '1' else "00000000000000000000000000000000";
	TEMP_87_mux_5 <= stq_data_5_q when bypass_idx_oh_4(5) = '1' else "00000000000000000000000000000000";
	TEMP_87_mux_6 <= stq_data_6_q when bypass_idx_oh_4(6) = '1' else "00000000000000000000000000000000";
	TEMP_87_mux_7 <= stq_data_7_q when bypass_idx_oh_4(7) = '1' else "00000000000000000000000000000000";
	TEMP_88_res_0 <= TEMP_87_mux_0 or TEMP_87_mux_4;
	TEMP_88_res_1 <= TEMP_87_mux_1 or TEMP_87_mux_5;
	TEMP_88_res_2 <= TEMP_87_mux_2 or TEMP_87_mux_6;
	TEMP_88_res_3 <= TEMP_87_mux_3 or TEMP_87_mux_7;
	-- Layer End
	TEMP_89_res_0 <= TEMP_88_res_0 or TEMP_88_res_2;
	TEMP_89_res_1 <= TEMP_88_res_1 or TEMP_88_res_3;
	-- Layer End
	bypass_data_4 <= TEMP_89_res_0 or TEMP_89_res_1;
	-- Mux1H End

	ldq_data_4_d <= read_data_4 or bypass_data_4;
	ldq_data_wen_4 <= bypass_en_4 or read_valid_4;
	read_idx_oh_5_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0101" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_5, rresp_data, read_idx_oh_5)
	TEMP_90_mux_0 <= rresp_data_0_i when read_idx_oh_5_0 = '1' else "00000000000000000000000000000000";
	read_data_5 <= TEMP_90_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_5, read_idx_oh_5, or)
	read_valid_5 <= read_idx_oh_5_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_5, stq_data, bypass_idx_oh_5)
	TEMP_91_mux_0 <= stq_data_0_q when bypass_idx_oh_5(0) = '1' else "00000000000000000000000000000000";
	TEMP_91_mux_1 <= stq_data_1_q when bypass_idx_oh_5(1) = '1' else "00000000000000000000000000000000";
	TEMP_91_mux_2 <= stq_data_2_q when bypass_idx_oh_5(2) = '1' else "00000000000000000000000000000000";
	TEMP_91_mux_3 <= stq_data_3_q when bypass_idx_oh_5(3) = '1' else "00000000000000000000000000000000";
	TEMP_91_mux_4 <= stq_data_4_q when bypass_idx_oh_5(4) = '1' else "00000000000000000000000000000000";
	TEMP_91_mux_5 <= stq_data_5_q when bypass_idx_oh_5(5) = '1' else "00000000000000000000000000000000";
	TEMP_91_mux_6 <= stq_data_6_q when bypass_idx_oh_5(6) = '1' else "00000000000000000000000000000000";
	TEMP_91_mux_7 <= stq_data_7_q when bypass_idx_oh_5(7) = '1' else "00000000000000000000000000000000";
	TEMP_92_res_0 <= TEMP_91_mux_0 or TEMP_91_mux_4;
	TEMP_92_res_1 <= TEMP_91_mux_1 or TEMP_91_mux_5;
	TEMP_92_res_2 <= TEMP_91_mux_2 or TEMP_91_mux_6;
	TEMP_92_res_3 <= TEMP_91_mux_3 or TEMP_91_mux_7;
	-- Layer End
	TEMP_93_res_0 <= TEMP_92_res_0 or TEMP_92_res_2;
	TEMP_93_res_1 <= TEMP_92_res_1 or TEMP_92_res_3;
	-- Layer End
	bypass_data_5 <= TEMP_93_res_0 or TEMP_93_res_1;
	-- Mux1H End

	ldq_data_5_d <= read_data_5 or bypass_data_5;
	ldq_data_wen_5 <= bypass_en_5 or read_valid_5;
	read_idx_oh_6_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0110" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_6, rresp_data, read_idx_oh_6)
	TEMP_94_mux_0 <= rresp_data_0_i when read_idx_oh_6_0 = '1' else "00000000000000000000000000000000";
	read_data_6 <= TEMP_94_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_6, read_idx_oh_6, or)
	read_valid_6 <= read_idx_oh_6_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_6, stq_data, bypass_idx_oh_6)
	TEMP_95_mux_0 <= stq_data_0_q when bypass_idx_oh_6(0) = '1' else "00000000000000000000000000000000";
	TEMP_95_mux_1 <= stq_data_1_q when bypass_idx_oh_6(1) = '1' else "00000000000000000000000000000000";
	TEMP_95_mux_2 <= stq_data_2_q when bypass_idx_oh_6(2) = '1' else "00000000000000000000000000000000";
	TEMP_95_mux_3 <= stq_data_3_q when bypass_idx_oh_6(3) = '1' else "00000000000000000000000000000000";
	TEMP_95_mux_4 <= stq_data_4_q when bypass_idx_oh_6(4) = '1' else "00000000000000000000000000000000";
	TEMP_95_mux_5 <= stq_data_5_q when bypass_idx_oh_6(5) = '1' else "00000000000000000000000000000000";
	TEMP_95_mux_6 <= stq_data_6_q when bypass_idx_oh_6(6) = '1' else "00000000000000000000000000000000";
	TEMP_95_mux_7 <= stq_data_7_q when bypass_idx_oh_6(7) = '1' else "00000000000000000000000000000000";
	TEMP_96_res_0 <= TEMP_95_mux_0 or TEMP_95_mux_4;
	TEMP_96_res_1 <= TEMP_95_mux_1 or TEMP_95_mux_5;
	TEMP_96_res_2 <= TEMP_95_mux_2 or TEMP_95_mux_6;
	TEMP_96_res_3 <= TEMP_95_mux_3 or TEMP_95_mux_7;
	-- Layer End
	TEMP_97_res_0 <= TEMP_96_res_0 or TEMP_96_res_2;
	TEMP_97_res_1 <= TEMP_96_res_1 or TEMP_96_res_3;
	-- Layer End
	bypass_data_6 <= TEMP_97_res_0 or TEMP_97_res_1;
	-- Mux1H End

	ldq_data_6_d <= read_data_6 or bypass_data_6;
	ldq_data_wen_6 <= bypass_en_6 or read_valid_6;
	read_idx_oh_7_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0111" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_7, rresp_data, read_idx_oh_7)
	TEMP_98_mux_0 <= rresp_data_0_i when read_idx_oh_7_0 = '1' else "00000000000000000000000000000000";
	read_data_7 <= TEMP_98_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_7, read_idx_oh_7, or)
	read_valid_7 <= read_idx_oh_7_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_7, stq_data, bypass_idx_oh_7)
	TEMP_99_mux_0 <= stq_data_0_q when bypass_idx_oh_7(0) = '1' else "00000000000000000000000000000000";
	TEMP_99_mux_1 <= stq_data_1_q when bypass_idx_oh_7(1) = '1' else "00000000000000000000000000000000";
	TEMP_99_mux_2 <= stq_data_2_q when bypass_idx_oh_7(2) = '1' else "00000000000000000000000000000000";
	TEMP_99_mux_3 <= stq_data_3_q when bypass_idx_oh_7(3) = '1' else "00000000000000000000000000000000";
	TEMP_99_mux_4 <= stq_data_4_q when bypass_idx_oh_7(4) = '1' else "00000000000000000000000000000000";
	TEMP_99_mux_5 <= stq_data_5_q when bypass_idx_oh_7(5) = '1' else "00000000000000000000000000000000";
	TEMP_99_mux_6 <= stq_data_6_q when bypass_idx_oh_7(6) = '1' else "00000000000000000000000000000000";
	TEMP_99_mux_7 <= stq_data_7_q when bypass_idx_oh_7(7) = '1' else "00000000000000000000000000000000";
	TEMP_100_res_0 <= TEMP_99_mux_0 or TEMP_99_mux_4;
	TEMP_100_res_1 <= TEMP_99_mux_1 or TEMP_99_mux_5;
	TEMP_100_res_2 <= TEMP_99_mux_2 or TEMP_99_mux_6;
	TEMP_100_res_3 <= TEMP_99_mux_3 or TEMP_99_mux_7;
	-- Layer End
	TEMP_101_res_0 <= TEMP_100_res_0 or TEMP_100_res_2;
	TEMP_101_res_1 <= TEMP_100_res_1 or TEMP_100_res_3;
	-- Layer End
	bypass_data_7 <= TEMP_101_res_0 or TEMP_101_res_1;
	-- Mux1H End

	ldq_data_7_d <= read_data_7 or bypass_data_7;
	ldq_data_wen_7 <= bypass_en_7 or read_valid_7;
	rresp_ready_0_o <= '1';
	stq_reset_0 <= wresp_valid_0_i when ( stq_resp_q = "000" ) else '0';
	stq_reset_1 <= wresp_valid_0_i when ( stq_resp_q = "001" ) else '0';
	stq_reset_2 <= wresp_valid_0_i when ( stq_resp_q = "010" ) else '0';
	stq_reset_3 <= wresp_valid_0_i when ( stq_resp_q = "011" ) else '0';
	stq_reset_4 <= wresp_valid_0_i when ( stq_resp_q = "100" ) else '0';
	stq_reset_5 <= wresp_valid_0_i when ( stq_resp_q = "101" ) else '0';
	stq_reset_6 <= wresp_valid_0_i when ( stq_resp_q = "110" ) else '0';
	stq_reset_7 <= wresp_valid_0_i when ( stq_resp_q = "111" ) else '0';
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
		elsif (rising_edge(clk)) then
			ldq_alloc_0_q <= ldq_alloc_0_d;
			ldq_alloc_1_q <= ldq_alloc_1_d;
			ldq_alloc_2_q <= ldq_alloc_2_d;
			ldq_alloc_3_q <= ldq_alloc_3_d;
			ldq_alloc_4_q <= ldq_alloc_4_d;
			ldq_alloc_5_q <= ldq_alloc_5_d;
			ldq_alloc_6_q <= ldq_alloc_6_d;
			ldq_alloc_7_q <= ldq_alloc_7_d;
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
			if (ldq_wen_4 = '1') then
				ldq_port_idx_4_q <= ldq_port_idx_4_d;
			end if;
			if (ldq_wen_5 = '1') then
				ldq_port_idx_5_q <= ldq_port_idx_5_d;
			end if;
			if (ldq_wen_6 = '1') then
				ldq_port_idx_6_q <= ldq_port_idx_6_d;
			end if;
			if (ldq_wen_7 = '1') then
				ldq_port_idx_7_q <= ldq_port_idx_7_d;
			end if;
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
		elsif (rising_edge(clk)) then
			stq_alloc_0_q <= stq_alloc_0_d;
			stq_alloc_1_q <= stq_alloc_1_d;
			stq_alloc_2_q <= stq_alloc_2_d;
			stq_alloc_3_q <= stq_alloc_3_d;
			stq_alloc_4_q <= stq_alloc_4_d;
			stq_alloc_5_q <= stq_alloc_5_d;
			stq_alloc_6_q <= stq_alloc_6_d;
			stq_alloc_7_q <= stq_alloc_7_d;
		end if;
		if (rising_edge(clk)) then
			if (stq_wen_0 = '1') then
				stq_port_idx_0_q <= stq_port_idx_0_d;
			end if;
			if (stq_wen_1 = '1') then
				stq_port_idx_1_q <= stq_port_idx_1_d;
			end if;
			if (stq_wen_2 = '1') then
				stq_port_idx_2_q <= stq_port_idx_2_d;
			end if;
			if (stq_wen_3 = '1') then
				stq_port_idx_3_q <= stq_port_idx_3_d;
			end if;
			if (stq_wen_4 = '1') then
				stq_port_idx_4_q <= stq_port_idx_4_d;
			end if;
			if (stq_wen_5 = '1') then
				stq_port_idx_5_q <= stq_port_idx_5_d;
			end if;
			if (stq_wen_6 = '1') then
				stq_port_idx_6_q <= stq_port_idx_6_d;
			end if;
			if (stq_wen_7 = '1') then
				stq_port_idx_7_q <= stq_port_idx_7_d;
			end if;
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
		end if;
		if (rst = '1') then
			ldq_tail_q <= "000";
		elsif (rising_edge(clk)) then
			ldq_tail_q <= ldq_tail_d;
		end if;
		if (rst = '1') then
			ldq_head_q <= "000";
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
