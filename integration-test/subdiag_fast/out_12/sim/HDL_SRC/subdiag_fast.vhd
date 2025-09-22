library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity subdiag_fast is
  port (
    d1_loadData : in std_logic_vector(31 downto 0);
    d2_loadData : in std_logic_vector(31 downto 0);
    e_loadData : in std_logic_vector(31 downto 0);
    d1_start_valid : in std_logic;
    d2_start_valid : in std_logic;
    e_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    d1_end_ready : in std_logic;
    d2_end_ready : in std_logic;
    e_end_ready : in std_logic;
    end_ready : in std_logic;
    d1_start_ready : out std_logic;
    d2_start_ready : out std_logic;
    e_start_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    d1_end_valid : out std_logic;
    d2_end_valid : out std_logic;
    e_end_valid : out std_logic;
    end_valid : out std_logic;
    d1_loadEn : out std_logic;
    d1_loadAddr : out std_logic_vector(9 downto 0);
    d1_storeEn : out std_logic;
    d1_storeAddr : out std_logic_vector(9 downto 0);
    d1_storeData : out std_logic_vector(31 downto 0);
    d2_loadEn : out std_logic;
    d2_loadAddr : out std_logic_vector(9 downto 0);
    d2_storeEn : out std_logic;
    d2_storeAddr : out std_logic_vector(9 downto 0);
    d2_storeData : out std_logic_vector(31 downto 0);
    e_loadEn : out std_logic;
    e_loadAddr : out std_logic_vector(9 downto 0);
    e_storeEn : out std_logic;
    e_storeAddr : out std_logic_vector(9 downto 0);
    e_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of subdiag_fast is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork0_outs_2_valid : std_logic;
  signal fork0_outs_2_ready : std_logic;
  signal mem_controller3_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller3_ldData_0_valid : std_logic;
  signal mem_controller3_ldData_0_ready : std_logic;
  signal mem_controller3_memEnd_valid : std_logic;
  signal mem_controller3_memEnd_ready : std_logic;
  signal mem_controller3_loadEn : std_logic;
  signal mem_controller3_loadAddr : std_logic_vector(9 downto 0);
  signal mem_controller3_storeEn : std_logic;
  signal mem_controller3_storeAddr : std_logic_vector(9 downto 0);
  signal mem_controller3_storeData : std_logic_vector(31 downto 0);
  signal mem_controller4_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller4_ldData_0_valid : std_logic;
  signal mem_controller4_ldData_0_ready : std_logic;
  signal mem_controller4_memEnd_valid : std_logic;
  signal mem_controller4_memEnd_ready : std_logic;
  signal mem_controller4_loadEn : std_logic;
  signal mem_controller4_loadAddr : std_logic_vector(9 downto 0);
  signal mem_controller4_storeEn : std_logic;
  signal mem_controller4_storeAddr : std_logic_vector(9 downto 0);
  signal mem_controller4_storeData : std_logic_vector(31 downto 0);
  signal mem_controller5_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller5_ldData_0_valid : std_logic;
  signal mem_controller5_ldData_0_ready : std_logic;
  signal mem_controller5_memEnd_valid : std_logic;
  signal mem_controller5_memEnd_ready : std_logic;
  signal mem_controller5_loadEn : std_logic;
  signal mem_controller5_loadAddr : std_logic_vector(9 downto 0);
  signal mem_controller5_storeEn : std_logic;
  signal mem_controller5_storeAddr : std_logic_vector(9 downto 0);
  signal mem_controller5_storeData : std_logic_vector(31 downto 0);
  signal constant0_outs : std_logic_vector(0 downto 0);
  signal constant0_outs_valid : std_logic;
  signal constant0_outs_ready : std_logic;
  signal extsi3_outs : std_logic_vector(10 downto 0);
  signal extsi3_outs_valid : std_logic;
  signal extsi3_outs_ready : std_logic;
  signal mux0_outs : std_logic_vector(10 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer35_outs : std_logic_vector(10 downto 0);
  signal buffer35_outs_valid : std_logic;
  signal buffer35_outs_ready : std_logic;
  signal trunci0_outs : std_logic_vector(9 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal trunci1_outs : std_logic_vector(9 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal trunci2_outs : std_logic_vector(9 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal buffer34_outs : std_logic_vector(0 downto 0);
  signal buffer34_outs_valid : std_logic;
  signal buffer34_outs_ready : std_logic;
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant4_outs : std_logic_vector(31 downto 0);
  signal constant4_outs_valid : std_logic;
  signal constant4_outs_ready : std_logic;
  signal load0_addrOut : std_logic_vector(9 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal load1_addrOut : std_logic_vector(9 downto 0);
  signal load1_addrOut_valid : std_logic;
  signal load1_addrOut_ready : std_logic;
  signal load1_dataOut : std_logic_vector(31 downto 0);
  signal load1_dataOut_valid : std_logic;
  signal load1_dataOut_ready : std_logic;
  signal addf0_result : std_logic_vector(31 downto 0);
  signal addf0_result_valid : std_logic;
  signal addf0_result_ready : std_logic;
  signal load2_addrOut : std_logic_vector(9 downto 0);
  signal load2_addrOut_valid : std_logic;
  signal load2_addrOut_ready : std_logic;
  signal load2_dataOut : std_logic_vector(31 downto 0);
  signal load2_dataOut_valid : std_logic;
  signal load2_dataOut_ready : std_logic;
  signal mulf0_result : std_logic_vector(31 downto 0);
  signal mulf0_result_valid : std_logic;
  signal mulf0_result_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(0 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(0 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal fork1_outs_2 : std_logic_vector(0 downto 0);
  signal fork1_outs_2_valid : std_logic;
  signal fork1_outs_2_ready : std_logic;
  signal cmpf0_result : std_logic_vector(0 downto 0);
  signal cmpf0_result_valid : std_logic;
  signal cmpf0_result_ready : std_logic;
  signal andi0_result : std_logic_vector(0 downto 0);
  signal andi0_result_valid : std_logic;
  signal andi0_result_ready : std_logic;
  signal not0_outs : std_logic_vector(0 downto 0);
  signal not0_outs_valid : std_logic;
  signal not0_outs_ready : std_logic;
  signal passer8_result : std_logic_vector(11 downto 0);
  signal passer8_result_valid : std_logic;
  signal passer8_result_ready : std_logic;
  signal buffer36_outs : std_logic_vector(10 downto 0);
  signal buffer36_outs_valid : std_logic;
  signal buffer36_outs_ready : std_logic;
  signal extsi4_outs : std_logic_vector(11 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal passer3_result_valid : std_logic;
  signal passer3_result_ready : std_logic;
  signal extsi5_outs : std_logic_vector(11 downto 0);
  signal extsi5_outs_valid : std_logic;
  signal extsi5_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant1_outs : std_logic_vector(1 downto 0);
  signal constant1_outs_valid : std_logic;
  signal constant1_outs_ready : std_logic;
  signal extsi6_outs : std_logic_vector(11 downto 0);
  signal extsi6_outs_valid : std_logic;
  signal extsi6_outs_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant2_outs : std_logic_vector(10 downto 0);
  signal constant2_outs_valid : std_logic;
  signal constant2_outs_ready : std_logic;
  signal extsi7_outs : std_logic_vector(11 downto 0);
  signal extsi7_outs_valid : std_logic;
  signal extsi7_outs_ready : std_logic;
  signal buffer4_outs : std_logic_vector(11 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal fork2_outs_0 : std_logic_vector(11 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1 : std_logic_vector(11 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal fork2_outs_2 : std_logic_vector(11 downto 0);
  signal fork2_outs_2_valid : std_logic;
  signal fork2_outs_2_ready : std_logic;
  signal addi0_result : std_logic_vector(11 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal buffer5_outs : std_logic_vector(0 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal buffer6_outs : std_logic_vector(0 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(0 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(0 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal passer9_result : std_logic_vector(0 downto 0);
  signal passer9_result_valid : std_logic;
  signal passer9_result_ready : std_logic;
  signal andi1_result : std_logic_vector(0 downto 0);
  signal andi1_result_valid : std_logic;
  signal andi1_result_ready : std_logic;
  signal spec_v2_repeating_init0_outs : std_logic_vector(0 downto 0);
  signal spec_v2_repeating_init0_outs_valid : std_logic;
  signal spec_v2_repeating_init0_outs_ready : std_logic;
  signal buffer7_outs : std_logic_vector(0 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal fork4_outs_0 : std_logic_vector(0 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(0 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal spec_v2_repeating_init1_outs : std_logic_vector(0 downto 0);
  signal spec_v2_repeating_init1_outs_valid : std_logic;
  signal spec_v2_repeating_init1_outs_ready : std_logic;
  signal buffer8_outs : std_logic_vector(0 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal fork15_outs_0 : std_logic_vector(0 downto 0);
  signal fork15_outs_0_valid : std_logic;
  signal fork15_outs_0_ready : std_logic;
  signal fork15_outs_1 : std_logic_vector(0 downto 0);
  signal fork15_outs_1_valid : std_logic;
  signal fork15_outs_1_ready : std_logic;
  signal buffer9_outs : std_logic_vector(0 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant3_outs : std_logic_vector(0 downto 0);
  signal constant3_outs_valid : std_logic;
  signal constant3_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(0 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal spec_v2_repeating_init2_outs : std_logic_vector(0 downto 0);
  signal spec_v2_repeating_init2_outs_valid : std_logic;
  signal spec_v2_repeating_init2_outs_ready : std_logic;
  signal buffer10_outs : std_logic_vector(0 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal fork16_outs_0 : std_logic_vector(0 downto 0);
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1 : std_logic_vector(0 downto 0);
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;
  signal buffer11_outs : std_logic_vector(0 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal constant5_outs : std_logic_vector(0 downto 0);
  signal constant5_outs_valid : std_logic;
  signal constant5_outs_ready : std_logic;
  signal mux4_outs : std_logic_vector(0 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal spec_v2_repeating_init3_outs : std_logic_vector(0 downto 0);
  signal spec_v2_repeating_init3_outs_valid : std_logic;
  signal spec_v2_repeating_init3_outs_ready : std_logic;
  signal buffer12_outs : std_logic_vector(0 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal fork17_outs_0 : std_logic_vector(0 downto 0);
  signal fork17_outs_0_valid : std_logic;
  signal fork17_outs_0_ready : std_logic;
  signal fork17_outs_1 : std_logic_vector(0 downto 0);
  signal fork17_outs_1_valid : std_logic;
  signal fork17_outs_1_ready : std_logic;
  signal buffer13_outs : std_logic_vector(0 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal source5_outs_valid : std_logic;
  signal source5_outs_ready : std_logic;
  signal constant6_outs : std_logic_vector(0 downto 0);
  signal constant6_outs_valid : std_logic;
  signal constant6_outs_ready : std_logic;
  signal mux5_outs : std_logic_vector(0 downto 0);
  signal mux5_outs_valid : std_logic;
  signal mux5_outs_ready : std_logic;
  signal spec_v2_repeating_init4_outs : std_logic_vector(0 downto 0);
  signal spec_v2_repeating_init4_outs_valid : std_logic;
  signal spec_v2_repeating_init4_outs_ready : std_logic;
  signal buffer14_outs : std_logic_vector(0 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal fork18_outs_0 : std_logic_vector(0 downto 0);
  signal fork18_outs_0_valid : std_logic;
  signal fork18_outs_0_ready : std_logic;
  signal fork18_outs_1 : std_logic_vector(0 downto 0);
  signal fork18_outs_1_valid : std_logic;
  signal fork18_outs_1_ready : std_logic;
  signal buffer15_outs : std_logic_vector(0 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal source6_outs_valid : std_logic;
  signal source6_outs_ready : std_logic;
  signal constant7_outs : std_logic_vector(0 downto 0);
  signal constant7_outs_valid : std_logic;
  signal constant7_outs_ready : std_logic;
  signal mux6_outs : std_logic_vector(0 downto 0);
  signal mux6_outs_valid : std_logic;
  signal mux6_outs_ready : std_logic;
  signal spec_v2_repeating_init5_outs : std_logic_vector(0 downto 0);
  signal spec_v2_repeating_init5_outs_valid : std_logic;
  signal spec_v2_repeating_init5_outs_ready : std_logic;
  signal buffer16_outs : std_logic_vector(0 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal fork19_outs_0 : std_logic_vector(0 downto 0);
  signal fork19_outs_0_valid : std_logic;
  signal fork19_outs_0_ready : std_logic;
  signal fork19_outs_1 : std_logic_vector(0 downto 0);
  signal fork19_outs_1_valid : std_logic;
  signal fork19_outs_1_ready : std_logic;
  signal buffer17_outs : std_logic_vector(0 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal source7_outs_valid : std_logic;
  signal source7_outs_ready : std_logic;
  signal constant8_outs : std_logic_vector(0 downto 0);
  signal constant8_outs_valid : std_logic;
  signal constant8_outs_ready : std_logic;
  signal mux7_outs : std_logic_vector(0 downto 0);
  signal mux7_outs_valid : std_logic;
  signal mux7_outs_ready : std_logic;
  signal spec_v2_repeating_init6_outs : std_logic_vector(0 downto 0);
  signal spec_v2_repeating_init6_outs_valid : std_logic;
  signal spec_v2_repeating_init6_outs_ready : std_logic;
  signal buffer18_outs : std_logic_vector(0 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal fork20_outs_0 : std_logic_vector(0 downto 0);
  signal fork20_outs_0_valid : std_logic;
  signal fork20_outs_0_ready : std_logic;
  signal fork20_outs_1 : std_logic_vector(0 downto 0);
  signal fork20_outs_1_valid : std_logic;
  signal fork20_outs_1_ready : std_logic;
  signal buffer19_outs : std_logic_vector(0 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal source8_outs_valid : std_logic;
  signal source8_outs_ready : std_logic;
  signal constant9_outs : std_logic_vector(0 downto 0);
  signal constant9_outs_valid : std_logic;
  signal constant9_outs_ready : std_logic;
  signal mux8_outs : std_logic_vector(0 downto 0);
  signal mux8_outs_valid : std_logic;
  signal mux8_outs_ready : std_logic;
  signal spec_v2_repeating_init7_outs : std_logic_vector(0 downto 0);
  signal spec_v2_repeating_init7_outs_valid : std_logic;
  signal spec_v2_repeating_init7_outs_ready : std_logic;
  signal buffer20_outs : std_logic_vector(0 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal fork21_outs_0 : std_logic_vector(0 downto 0);
  signal fork21_outs_0_valid : std_logic;
  signal fork21_outs_0_ready : std_logic;
  signal fork21_outs_1 : std_logic_vector(0 downto 0);
  signal fork21_outs_1_valid : std_logic;
  signal fork21_outs_1_ready : std_logic;
  signal buffer21_outs : std_logic_vector(0 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal source9_outs_valid : std_logic;
  signal source9_outs_ready : std_logic;
  signal constant10_outs : std_logic_vector(0 downto 0);
  signal constant10_outs_valid : std_logic;
  signal constant10_outs_ready : std_logic;
  signal mux9_outs : std_logic_vector(0 downto 0);
  signal mux9_outs_valid : std_logic;
  signal mux9_outs_ready : std_logic;
  signal spec_v2_repeating_init8_outs : std_logic_vector(0 downto 0);
  signal spec_v2_repeating_init8_outs_valid : std_logic;
  signal spec_v2_repeating_init8_outs_ready : std_logic;
  signal buffer22_outs : std_logic_vector(0 downto 0);
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal fork22_outs_0 : std_logic_vector(0 downto 0);
  signal fork22_outs_0_valid : std_logic;
  signal fork22_outs_0_ready : std_logic;
  signal fork22_outs_1 : std_logic_vector(0 downto 0);
  signal fork22_outs_1_valid : std_logic;
  signal fork22_outs_1_ready : std_logic;
  signal buffer23_outs : std_logic_vector(0 downto 0);
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal source10_outs_valid : std_logic;
  signal source10_outs_ready : std_logic;
  signal constant11_outs : std_logic_vector(0 downto 0);
  signal constant11_outs_valid : std_logic;
  signal constant11_outs_ready : std_logic;
  signal mux10_outs : std_logic_vector(0 downto 0);
  signal mux10_outs_valid : std_logic;
  signal mux10_outs_ready : std_logic;
  signal spec_v2_repeating_init9_outs : std_logic_vector(0 downto 0);
  signal spec_v2_repeating_init9_outs_valid : std_logic;
  signal spec_v2_repeating_init9_outs_ready : std_logic;
  signal buffer24_outs : std_logic_vector(0 downto 0);
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
  signal fork23_outs_0 : std_logic_vector(0 downto 0);
  signal fork23_outs_0_valid : std_logic;
  signal fork23_outs_0_ready : std_logic;
  signal fork23_outs_1 : std_logic_vector(0 downto 0);
  signal fork23_outs_1_valid : std_logic;
  signal fork23_outs_1_ready : std_logic;
  signal buffer25_outs : std_logic_vector(0 downto 0);
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal source11_outs_valid : std_logic;
  signal source11_outs_ready : std_logic;
  signal constant12_outs : std_logic_vector(0 downto 0);
  signal constant12_outs_valid : std_logic;
  signal constant12_outs_ready : std_logic;
  signal mux11_outs : std_logic_vector(0 downto 0);
  signal mux11_outs_valid : std_logic;
  signal mux11_outs_ready : std_logic;
  signal spec_v2_repeating_init10_outs : std_logic_vector(0 downto 0);
  signal spec_v2_repeating_init10_outs_valid : std_logic;
  signal spec_v2_repeating_init10_outs_ready : std_logic;
  signal buffer26_outs : std_logic_vector(0 downto 0);
  signal buffer26_outs_valid : std_logic;
  signal buffer26_outs_ready : std_logic;
  signal fork24_outs_0 : std_logic_vector(0 downto 0);
  signal fork24_outs_0_valid : std_logic;
  signal fork24_outs_0_ready : std_logic;
  signal fork24_outs_1 : std_logic_vector(0 downto 0);
  signal fork24_outs_1_valid : std_logic;
  signal fork24_outs_1_ready : std_logic;
  signal buffer27_outs : std_logic_vector(0 downto 0);
  signal buffer27_outs_valid : std_logic;
  signal buffer27_outs_ready : std_logic;
  signal source12_outs_valid : std_logic;
  signal source12_outs_ready : std_logic;
  signal constant13_outs : std_logic_vector(0 downto 0);
  signal constant13_outs_valid : std_logic;
  signal constant13_outs_ready : std_logic;
  signal mux12_outs : std_logic_vector(0 downto 0);
  signal mux12_outs_valid : std_logic;
  signal mux12_outs_ready : std_logic;
  signal spec_v2_repeating_init11_outs : std_logic_vector(0 downto 0);
  signal spec_v2_repeating_init11_outs_valid : std_logic;
  signal spec_v2_repeating_init11_outs_ready : std_logic;
  signal buffer29_outs : std_logic_vector(0 downto 0);
  signal buffer29_outs_valid : std_logic;
  signal buffer29_outs_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(0 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(0 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal fork6_outs_2 : std_logic_vector(0 downto 0);
  signal fork6_outs_2_valid : std_logic;
  signal fork6_outs_2_ready : std_logic;
  signal fork6_outs_3 : std_logic_vector(0 downto 0);
  signal fork6_outs_3_valid : std_logic;
  signal fork6_outs_3_ready : std_logic;
  signal buffer33_outs : std_logic_vector(0 downto 0);
  signal buffer33_outs_valid : std_logic;
  signal buffer33_outs_ready : std_logic;
  signal init2_outs : std_logic_vector(0 downto 0);
  signal init2_outs_valid : std_logic;
  signal init2_outs_ready : std_logic;
  signal fork7_outs_0 : std_logic_vector(0 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1 : std_logic_vector(0 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal buffer28_outs : std_logic_vector(0 downto 0);
  signal buffer28_outs_valid : std_logic;
  signal buffer28_outs_ready : std_logic;
  signal buffer31_outs : std_logic_vector(0 downto 0);
  signal buffer31_outs_valid : std_logic;
  signal buffer31_outs_ready : std_logic;
  signal buffer32_outs : std_logic_vector(0 downto 0);
  signal buffer32_outs_valid : std_logic;
  signal buffer32_outs_ready : std_logic;
  signal source13_outs_valid : std_logic;
  signal source13_outs_ready : std_logic;
  signal constant14_outs : std_logic_vector(0 downto 0);
  signal constant14_outs_valid : std_logic;
  signal constant14_outs_ready : std_logic;
  signal mux13_outs : std_logic_vector(0 downto 0);
  signal mux13_outs_valid : std_logic;
  signal mux13_outs_ready : std_logic;
  signal fork26_outs_0 : std_logic_vector(0 downto 0);
  signal fork26_outs_0_valid : std_logic;
  signal fork26_outs_0_ready : std_logic;
  signal fork26_outs_1 : std_logic_vector(0 downto 0);
  signal fork26_outs_1_valid : std_logic;
  signal fork26_outs_1_ready : std_logic;
  signal fork26_outs_2 : std_logic_vector(0 downto 0);
  signal fork26_outs_2_valid : std_logic;
  signal fork26_outs_2_ready : std_logic;
  signal andi13_result : std_logic_vector(0 downto 0);
  signal andi13_result_valid : std_logic;
  signal andi13_result_ready : std_logic;
  signal fork27_outs_0 : std_logic_vector(0 downto 0);
  signal fork27_outs_0_valid : std_logic;
  signal fork27_outs_0_ready : std_logic;
  signal fork27_outs_1 : std_logic_vector(0 downto 0);
  signal fork27_outs_1_valid : std_logic;
  signal fork27_outs_1_ready : std_logic;
  signal andi14_result : std_logic_vector(0 downto 0);
  signal andi14_result_valid : std_logic;
  signal andi14_result_ready : std_logic;
  signal fork28_outs_0 : std_logic_vector(0 downto 0);
  signal fork28_outs_0_valid : std_logic;
  signal fork28_outs_0_ready : std_logic;
  signal fork28_outs_1 : std_logic_vector(0 downto 0);
  signal fork28_outs_1_valid : std_logic;
  signal fork28_outs_1_ready : std_logic;
  signal not1_outs : std_logic_vector(0 downto 0);
  signal not1_outs_valid : std_logic;
  signal not1_outs_ready : std_logic;
  signal buffer3_outs : std_logic_vector(11 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal passer5_result : std_logic_vector(11 downto 0);
  signal passer5_result_valid : std_logic;
  signal passer5_result_ready : std_logic;
  signal buffer0_outs : std_logic_vector(10 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal fork29_outs_0 : std_logic_vector(10 downto 0);
  signal fork29_outs_0_valid : std_logic;
  signal fork29_outs_0_ready : std_logic;
  signal fork29_outs_1 : std_logic_vector(10 downto 0);
  signal fork29_outs_1_valid : std_logic;
  signal fork29_outs_1_ready : std_logic;
  signal fork29_outs_2 : std_logic_vector(10 downto 0);
  signal fork29_outs_2_valid : std_logic;
  signal fork29_outs_2_ready : std_logic;
  signal fork29_outs_3 : std_logic_vector(10 downto 0);
  signal fork29_outs_3_valid : std_logic;
  signal fork29_outs_3_ready : std_logic;
  signal fork29_outs_4 : std_logic_vector(10 downto 0);
  signal fork29_outs_4_valid : std_logic;
  signal fork29_outs_4_ready : std_logic;
  signal passer10_result : std_logic_vector(10 downto 0);
  signal passer10_result_valid : std_logic;
  signal passer10_result_ready : std_logic;
  signal trunci3_outs : std_logic_vector(10 downto 0);
  signal trunci3_outs_valid : std_logic;
  signal trunci3_outs_ready : std_logic;
  signal buffer30_outs : std_logic_vector(0 downto 0);
  signal buffer30_outs_valid : std_logic;
  signal buffer30_outs_ready : std_logic;
  signal passer11_result_valid : std_logic;
  signal passer11_result_ready : std_logic;
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal fork30_outs_0_valid : std_logic;
  signal fork30_outs_0_ready : std_logic;
  signal fork30_outs_1_valid : std_logic;
  signal fork30_outs_1_ready : std_logic;
  signal fork30_outs_2_valid : std_logic;
  signal fork30_outs_2_ready : std_logic;
  signal passer7_result_valid : std_logic;
  signal passer7_result_ready : std_logic;
  signal mux1_outs : std_logic_vector(11 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal extsi8_outs : std_logic_vector(31 downto 0);
  signal extsi8_outs_valid : std_logic;
  signal extsi8_outs_ready : std_logic;
  signal control_merge2_outs_valid : std_logic;
  signal control_merge2_outs_ready : std_logic;
  signal control_merge2_index : std_logic_vector(0 downto 0);
  signal control_merge2_index_valid : std_logic;
  signal control_merge2_index_ready : std_logic;
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal fork5_outs_2_valid : std_logic;
  signal fork5_outs_2_ready : std_logic;

begin

  out0 <= extsi8_outs;
  out0_valid <= extsi8_outs_valid;
  extsi8_outs_ready <= out0_ready;
  d1_end_valid <= mem_controller5_memEnd_valid;
  mem_controller5_memEnd_ready <= d1_end_ready;
  d2_end_valid <= mem_controller4_memEnd_valid;
  mem_controller4_memEnd_ready <= d2_end_ready;
  e_end_valid <= mem_controller3_memEnd_valid;
  mem_controller3_memEnd_ready <= e_end_ready;
  end_valid <= fork0_outs_1_valid;
  fork0_outs_1_ready <= end_ready;
  d1_loadEn <= mem_controller5_loadEn;
  d1_loadAddr <= mem_controller5_loadAddr;
  d1_storeEn <= mem_controller5_storeEn;
  d1_storeAddr <= mem_controller5_storeAddr;
  d1_storeData <= mem_controller5_storeData;
  d2_loadEn <= mem_controller4_loadEn;
  d2_loadAddr <= mem_controller4_loadAddr;
  d2_storeEn <= mem_controller4_storeEn;
  d2_storeAddr <= mem_controller4_storeAddr;
  d2_storeData <= mem_controller4_storeData;
  e_loadEn <= mem_controller3_loadEn;
  e_loadAddr <= mem_controller3_loadAddr;
  e_storeEn <= mem_controller3_storeEn;
  e_storeAddr <= mem_controller3_storeAddr;
  e_storeData <= mem_controller3_storeData;

  fork0 : entity work.handshake_fork_0(arch)
    port map(
      ins_valid => start_valid,
      ins_ready => start_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork0_outs_0_valid,
      outs_valid(1) => fork0_outs_1_valid,
      outs_valid(2) => fork0_outs_2_valid,
      outs_ready(0) => fork0_outs_0_ready,
      outs_ready(1) => fork0_outs_1_ready,
      outs_ready(2) => fork0_outs_2_ready
    );

  mem_controller3 : entity work.handshake_mem_controller_0(arch)
    port map(
      loadData => e_loadData,
      memStart_valid => e_start_valid,
      memStart_ready => e_start_ready,
      ldAddr(0) => load2_addrOut,
      ldAddr_valid(0) => load2_addrOut_valid,
      ldAddr_ready(0) => load2_addrOut_ready,
      ctrlEnd_valid => fork5_outs_2_valid,
      ctrlEnd_ready => fork5_outs_2_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller3_ldData_0,
      ldData_valid(0) => mem_controller3_ldData_0_valid,
      ldData_ready(0) => mem_controller3_ldData_0_ready,
      memEnd_valid => mem_controller3_memEnd_valid,
      memEnd_ready => mem_controller3_memEnd_ready,
      loadEn => mem_controller3_loadEn,
      loadAddr => mem_controller3_loadAddr,
      storeEn => mem_controller3_storeEn,
      storeAddr => mem_controller3_storeAddr,
      storeData => mem_controller3_storeData
    );

  mem_controller4 : entity work.handshake_mem_controller_1(arch)
    port map(
      loadData => d2_loadData,
      memStart_valid => d2_start_valid,
      memStart_ready => d2_start_ready,
      ldAddr(0) => load1_addrOut,
      ldAddr_valid(0) => load1_addrOut_valid,
      ldAddr_ready(0) => load1_addrOut_ready,
      ctrlEnd_valid => fork5_outs_1_valid,
      ctrlEnd_ready => fork5_outs_1_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller4_ldData_0,
      ldData_valid(0) => mem_controller4_ldData_0_valid,
      ldData_ready(0) => mem_controller4_ldData_0_ready,
      memEnd_valid => mem_controller4_memEnd_valid,
      memEnd_ready => mem_controller4_memEnd_ready,
      loadEn => mem_controller4_loadEn,
      loadAddr => mem_controller4_loadAddr,
      storeEn => mem_controller4_storeEn,
      storeAddr => mem_controller4_storeAddr,
      storeData => mem_controller4_storeData
    );

  mem_controller5 : entity work.handshake_mem_controller_2(arch)
    port map(
      loadData => d1_loadData,
      memStart_valid => d1_start_valid,
      memStart_ready => d1_start_ready,
      ldAddr(0) => load0_addrOut,
      ldAddr_valid(0) => load0_addrOut_valid,
      ldAddr_ready(0) => load0_addrOut_ready,
      ctrlEnd_valid => fork5_outs_0_valid,
      ctrlEnd_ready => fork5_outs_0_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller5_ldData_0,
      ldData_valid(0) => mem_controller5_ldData_0_valid,
      ldData_ready(0) => mem_controller5_ldData_0_ready,
      memEnd_valid => mem_controller5_memEnd_valid,
      memEnd_ready => mem_controller5_memEnd_ready,
      loadEn => mem_controller5_loadEn,
      loadAddr => mem_controller5_loadAddr,
      storeEn => mem_controller5_storeEn,
      storeAddr => mem_controller5_storeAddr,
      storeData => mem_controller5_storeData
    );

  constant0 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => fork0_outs_0_valid,
      ctrl_ready => fork0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant0_outs,
      outs_valid => constant0_outs_valid,
      outs_ready => constant0_outs_ready
    );

  extsi3 : entity work.handshake_extsi_0(arch)
    port map(
      ins => constant0_outs,
      ins_valid => constant0_outs_valid,
      ins_ready => constant0_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi3_outs,
      outs_valid => extsi3_outs_valid,
      outs_ready => extsi3_outs_ready
    );

  mux0 : entity work.handshake_mux_0(arch)
    port map(
      index => fork7_outs_0,
      index_valid => fork7_outs_0_valid,
      index_ready => fork7_outs_0_ready,
      ins(0) => extsi3_outs,
      ins(1) => passer10_result,
      ins_valid(0) => extsi3_outs_valid,
      ins_valid(1) => passer10_result_valid,
      ins_ready(0) => extsi3_outs_ready,
      ins_ready(1) => passer10_result_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  buffer35 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork29_outs_0,
      ins_valid => fork29_outs_0_valid,
      ins_ready => fork29_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer35_outs,
      outs_valid => buffer35_outs_valid,
      outs_ready => buffer35_outs_ready
    );

  trunci0 : entity work.handshake_trunci_0(arch)
    port map(
      ins => buffer35_outs,
      ins_valid => buffer35_outs_valid,
      ins_ready => buffer35_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  trunci1 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork29_outs_1,
      ins_valid => fork29_outs_1_valid,
      ins_ready => fork29_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  trunci2 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork29_outs_2,
      ins_valid => fork29_outs_2_valid,
      ins_ready => fork29_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  buffer34 : entity work.handshake_buffer_1(arch)
    port map(
      ins => fork7_outs_1,
      ins_valid => fork7_outs_1_valid,
      ins_ready => fork7_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer34_outs,
      outs_valid => buffer34_outs_valid,
      outs_ready => buffer34_outs_ready
    );

  mux3 : entity work.handshake_mux_1(arch)
    port map(
      index => buffer34_outs,
      index_valid => buffer34_outs_valid,
      index_ready => buffer34_outs_ready,
      ins_valid(0) => fork0_outs_2_valid,
      ins_valid(1) => passer11_result_valid,
      ins_ready(0) => fork0_outs_2_ready,
      ins_ready(1) => passer11_result_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  source0 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant4 : entity work.handshake_constant_1(arch)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant4_outs,
      outs_valid => constant4_outs_valid,
      outs_ready => constant4_outs_ready
    );

  load0 : entity work.handshake_load_0(arch)
    port map(
      addrIn => trunci2_outs,
      addrIn_valid => trunci2_outs_valid,
      addrIn_ready => trunci2_outs_ready,
      dataFromMem => mem_controller5_ldData_0,
      dataFromMem_valid => mem_controller5_ldData_0_valid,
      dataFromMem_ready => mem_controller5_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load0_addrOut,
      addrOut_valid => load0_addrOut_valid,
      addrOut_ready => load0_addrOut_ready,
      dataOut => load0_dataOut,
      dataOut_valid => load0_dataOut_valid,
      dataOut_ready => load0_dataOut_ready
    );

  load1 : entity work.handshake_load_1(arch)
    port map(
      addrIn => trunci1_outs,
      addrIn_valid => trunci1_outs_valid,
      addrIn_ready => trunci1_outs_ready,
      dataFromMem => mem_controller4_ldData_0,
      dataFromMem_valid => mem_controller4_ldData_0_valid,
      dataFromMem_ready => mem_controller4_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load1_addrOut,
      addrOut_valid => load1_addrOut_valid,
      addrOut_ready => load1_addrOut_ready,
      dataOut => load1_dataOut,
      dataOut_valid => load1_dataOut_valid,
      dataOut_ready => load1_dataOut_ready
    );

  addf0 : entity work.handshake_addf_0(arch)
    port map(
      lhs => load0_dataOut,
      lhs_valid => load0_dataOut_valid,
      lhs_ready => load0_dataOut_ready,
      rhs => load1_dataOut,
      rhs_valid => load1_dataOut_valid,
      rhs_ready => load1_dataOut_ready,
      clk => clk,
      rst => rst,
      result => addf0_result,
      result_valid => addf0_result_valid,
      result_ready => addf0_result_ready
    );

  load2 : entity work.handshake_load_2(arch)
    port map(
      addrIn => trunci0_outs,
      addrIn_valid => trunci0_outs_valid,
      addrIn_ready => trunci0_outs_ready,
      dataFromMem => mem_controller3_ldData_0,
      dataFromMem_valid => mem_controller3_ldData_0_valid,
      dataFromMem_ready => mem_controller3_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load2_addrOut,
      addrOut_valid => load2_addrOut_valid,
      addrOut_ready => load2_addrOut_ready,
      dataOut => load2_dataOut,
      dataOut_valid => load2_dataOut_valid,
      dataOut_ready => load2_dataOut_ready
    );

  mulf0 : entity work.handshake_mulf_0(arch)
    port map(
      lhs => addf0_result,
      lhs_valid => addf0_result_valid,
      lhs_ready => addf0_result_ready,
      rhs => constant4_outs,
      rhs_valid => constant4_outs_valid,
      rhs_ready => constant4_outs_ready,
      clk => clk,
      rst => rst,
      result => mulf0_result,
      result_valid => mulf0_result_valid,
      result_ready => mulf0_result_ready
    );

  fork1 : entity work.handshake_fork_1(arch)
    port map(
      ins => cmpf0_result,
      ins_valid => cmpf0_result_valid,
      ins_ready => cmpf0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork1_outs_0,
      outs(1) => fork1_outs_1,
      outs(2) => fork1_outs_2,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_valid(2) => fork1_outs_2_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready,
      outs_ready(2) => fork1_outs_2_ready
    );

  cmpf0 : entity work.handshake_cmpf_0(arch)
    port map(
      lhs => load2_dataOut,
      lhs_valid => load2_dataOut_valid,
      lhs_ready => load2_dataOut_ready,
      rhs => mulf0_result,
      rhs_valid => mulf0_result_valid,
      rhs_ready => mulf0_result_ready,
      clk => clk,
      rst => rst,
      result => cmpf0_result,
      result_valid => cmpf0_result_valid,
      result_ready => cmpf0_result_ready
    );

  andi0 : entity work.handshake_andi_0(arch)
    port map(
      lhs => not1_outs,
      lhs_valid => not1_outs_valid,
      lhs_ready => not1_outs_ready,
      rhs => fork1_outs_1,
      rhs_valid => fork1_outs_1_valid,
      rhs_ready => fork1_outs_1_ready,
      clk => clk,
      rst => rst,
      result => andi0_result,
      result_valid => andi0_result_valid,
      result_ready => andi0_result_ready
    );

  not0 : entity work.handshake_not_0(arch)
    port map(
      ins => fork1_outs_0,
      ins_valid => fork1_outs_0_valid,
      ins_ready => fork1_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => not0_outs,
      outs_valid => not0_outs_valid,
      outs_ready => not0_outs_ready
    );

  passer8 : entity work.handshake_passer_0(arch)
    port map(
      data => extsi4_outs,
      data_valid => extsi4_outs_valid,
      data_ready => extsi4_outs_ready,
      ctrl => fork27_outs_1,
      ctrl_valid => fork27_outs_1_valid,
      ctrl_ready => fork27_outs_1_ready,
      clk => clk,
      rst => rst,
      result => passer8_result,
      result_valid => passer8_result_valid,
      result_ready => passer8_result_ready
    );

  buffer36 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork29_outs_3,
      ins_valid => fork29_outs_3_valid,
      ins_ready => fork29_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer36_outs,
      outs_valid => buffer36_outs_valid,
      outs_ready => buffer36_outs_ready
    );

  extsi4 : entity work.handshake_extsi_1(arch)
    port map(
      ins => buffer36_outs,
      ins_valid => buffer36_outs_valid,
      ins_ready => buffer36_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi4_outs,
      outs_valid => extsi4_outs_valid,
      outs_ready => extsi4_outs_ready
    );

  passer3 : entity work.handshake_passer_1(arch)
    port map(
      data_valid => fork30_outs_2_valid,
      data_ready => fork30_outs_2_ready,
      ctrl => fork27_outs_0,
      ctrl_valid => fork27_outs_0_valid,
      ctrl_ready => fork27_outs_0_ready,
      clk => clk,
      rst => rst,
      result_valid => passer3_result_valid,
      result_ready => passer3_result_ready
    );

  extsi5 : entity work.handshake_extsi_1(arch)
    port map(
      ins => fork29_outs_4,
      ins_valid => fork29_outs_4_valid,
      ins_ready => fork29_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => extsi5_outs,
      outs_valid => extsi5_outs_valid,
      outs_ready => extsi5_outs_ready
    );

  source1 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant1 : entity work.handshake_constant_2(arch)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant1_outs,
      outs_valid => constant1_outs_valid,
      outs_ready => constant1_outs_ready
    );

  extsi6 : entity work.handshake_extsi_2(arch)
    port map(
      ins => constant1_outs,
      ins_valid => constant1_outs_valid,
      ins_ready => constant1_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi6_outs,
      outs_valid => extsi6_outs_valid,
      outs_ready => extsi6_outs_ready
    );

  source2 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant2 : entity work.handshake_constant_3(arch)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant2_outs,
      outs_valid => constant2_outs_valid,
      outs_ready => constant2_outs_ready
    );

  extsi7 : entity work.handshake_extsi_1(arch)
    port map(
      ins => constant2_outs,
      ins_valid => constant2_outs_valid,
      ins_ready => constant2_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi7_outs,
      outs_valid => extsi7_outs_valid,
      outs_ready => extsi7_outs_ready
    );

  buffer4 : entity work.handshake_buffer_2(arch)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer4_outs,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  fork2 : entity work.handshake_fork_2(arch)
    port map(
      ins => buffer4_outs,
      ins_valid => buffer4_outs_valid,
      ins_ready => buffer4_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork2_outs_0,
      outs(1) => fork2_outs_1,
      outs(2) => fork2_outs_2,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_valid(2) => fork2_outs_2_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready,
      outs_ready(2) => fork2_outs_2_ready
    );

  addi0 : entity work.handshake_addi_0(arch)
    port map(
      lhs => extsi5_outs,
      lhs_valid => extsi5_outs_valid,
      lhs_ready => extsi5_outs_ready,
      rhs => extsi6_outs,
      rhs_valid => extsi6_outs_valid,
      rhs_ready => extsi6_outs_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  buffer5 : entity work.handshake_buffer_3(arch)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer5_outs,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  buffer6 : entity work.handshake_buffer_4(arch)
    port map(
      ins => buffer5_outs,
      ins_valid => buffer5_outs_valid,
      ins_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  fork3 : entity work.handshake_fork_3(arch)
    port map(
      ins => buffer6_outs,
      ins_valid => buffer6_outs_valid,
      ins_ready => buffer6_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch)
    port map(
      lhs => fork2_outs_0,
      lhs_valid => fork2_outs_0_valid,
      lhs_ready => fork2_outs_0_ready,
      rhs => extsi7_outs,
      rhs_valid => extsi7_outs_valid,
      rhs_ready => extsi7_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  passer9 : entity work.handshake_passer_2(arch)
    port map(
      data => andi1_result,
      data_valid => andi1_result_valid,
      data_ready => andi1_result_ready,
      ctrl => fork26_outs_2,
      ctrl_valid => fork26_outs_2_valid,
      ctrl_ready => fork26_outs_2_ready,
      clk => clk,
      rst => rst,
      result => passer9_result,
      result_valid => passer9_result_valid,
      result_ready => passer9_result_ready
    );

  andi1 : entity work.handshake_andi_0(arch)
    port map(
      lhs => fork1_outs_2,
      lhs_valid => fork1_outs_2_valid,
      lhs_ready => fork1_outs_2_ready,
      rhs => fork3_outs_0,
      rhs_valid => fork3_outs_0_valid,
      rhs_ready => fork3_outs_0_ready,
      clk => clk,
      rst => rst,
      result => andi1_result,
      result_valid => andi1_result_valid,
      result_ready => andi1_result_ready
    );

  spec_v2_repeating_init0 : entity work.handshake_spec_v2_repeating_init_0(arch)
    port map(
      ins => passer9_result,
      ins_valid => passer9_result_valid,
      ins_ready => passer9_result_ready,
      clk => clk,
      rst => rst,
      outs => spec_v2_repeating_init0_outs,
      outs_valid => spec_v2_repeating_init0_outs_valid,
      outs_ready => spec_v2_repeating_init0_outs_ready
    );

  buffer7 : entity work.handshake_buffer_5(arch)
    port map(
      ins => spec_v2_repeating_init0_outs,
      ins_valid => spec_v2_repeating_init0_outs_valid,
      ins_ready => spec_v2_repeating_init0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer7_outs,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  fork4 : entity work.handshake_fork_3(arch)
    port map(
      ins => buffer7_outs,
      ins_valid => buffer7_outs_valid,
      ins_ready => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork4_outs_0,
      outs(1) => fork4_outs_1,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready
    );

  spec_v2_repeating_init1 : entity work.handshake_spec_v2_repeating_init_0(arch)
    port map(
      ins => fork4_outs_0,
      ins_valid => fork4_outs_0_valid,
      ins_ready => fork4_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => spec_v2_repeating_init1_outs,
      outs_valid => spec_v2_repeating_init1_outs_valid,
      outs_ready => spec_v2_repeating_init1_outs_ready
    );

  buffer8 : entity work.handshake_buffer_5(arch)
    port map(
      ins => spec_v2_repeating_init1_outs,
      ins_valid => spec_v2_repeating_init1_outs_valid,
      ins_ready => spec_v2_repeating_init1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  fork15 : entity work.handshake_fork_3(arch)
    port map(
      ins => buffer8_outs,
      ins_valid => buffer8_outs_valid,
      ins_ready => buffer8_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork15_outs_0,
      outs(1) => fork15_outs_1,
      outs_valid(0) => fork15_outs_0_valid,
      outs_valid(1) => fork15_outs_1_valid,
      outs_ready(0) => fork15_outs_0_ready,
      outs_ready(1) => fork15_outs_1_ready
    );

  buffer9 : entity work.handshake_buffer_3(arch)
    port map(
      ins => fork15_outs_0,
      ins_valid => fork15_outs_0_valid,
      ins_ready => fork15_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer9_outs,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  source3 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant3 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant3_outs,
      outs_valid => constant3_outs_valid,
      outs_ready => constant3_outs_ready
    );

  mux2 : entity work.handshake_mux_2(arch)
    port map(
      index => buffer9_outs,
      index_valid => buffer9_outs_valid,
      index_ready => buffer9_outs_ready,
      ins(0) => constant3_outs,
      ins(1) => fork4_outs_1,
      ins_valid(0) => constant3_outs_valid,
      ins_valid(1) => fork4_outs_1_valid,
      ins_ready(0) => constant3_outs_ready,
      ins_ready(1) => fork4_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  spec_v2_repeating_init2 : entity work.handshake_spec_v2_repeating_init_0(arch)
    port map(
      ins => fork15_outs_1,
      ins_valid => fork15_outs_1_valid,
      ins_ready => fork15_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => spec_v2_repeating_init2_outs,
      outs_valid => spec_v2_repeating_init2_outs_valid,
      outs_ready => spec_v2_repeating_init2_outs_ready
    );

  buffer10 : entity work.handshake_buffer_5(arch)
    port map(
      ins => spec_v2_repeating_init2_outs,
      ins_valid => spec_v2_repeating_init2_outs_valid,
      ins_ready => spec_v2_repeating_init2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  fork16 : entity work.handshake_fork_3(arch)
    port map(
      ins => buffer10_outs,
      ins_valid => buffer10_outs_valid,
      ins_ready => buffer10_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork16_outs_0,
      outs(1) => fork16_outs_1,
      outs_valid(0) => fork16_outs_0_valid,
      outs_valid(1) => fork16_outs_1_valid,
      outs_ready(0) => fork16_outs_0_ready,
      outs_ready(1) => fork16_outs_1_ready
    );

  buffer11 : entity work.handshake_buffer_6(arch)
    port map(
      ins => fork16_outs_0,
      ins_valid => fork16_outs_0_valid,
      ins_ready => fork16_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  source4 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready
    );

  constant5 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => source4_outs_valid,
      ctrl_ready => source4_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant5_outs,
      outs_valid => constant5_outs_valid,
      outs_ready => constant5_outs_ready
    );

  mux4 : entity work.handshake_mux_2(arch)
    port map(
      index => buffer11_outs,
      index_valid => buffer11_outs_valid,
      index_ready => buffer11_outs_ready,
      ins(0) => constant5_outs,
      ins(1) => mux2_outs,
      ins_valid(0) => constant5_outs_valid,
      ins_valid(1) => mux2_outs_valid,
      ins_ready(0) => constant5_outs_ready,
      ins_ready(1) => mux2_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  spec_v2_repeating_init3 : entity work.handshake_spec_v2_repeating_init_0(arch)
    port map(
      ins => fork16_outs_1,
      ins_valid => fork16_outs_1_valid,
      ins_ready => fork16_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => spec_v2_repeating_init3_outs,
      outs_valid => spec_v2_repeating_init3_outs_valid,
      outs_ready => spec_v2_repeating_init3_outs_ready
    );

  buffer12 : entity work.handshake_buffer_5(arch)
    port map(
      ins => spec_v2_repeating_init3_outs,
      ins_valid => spec_v2_repeating_init3_outs_valid,
      ins_ready => spec_v2_repeating_init3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  fork17 : entity work.handshake_fork_3(arch)
    port map(
      ins => buffer12_outs,
      ins_valid => buffer12_outs_valid,
      ins_ready => buffer12_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork17_outs_0,
      outs(1) => fork17_outs_1,
      outs_valid(0) => fork17_outs_0_valid,
      outs_valid(1) => fork17_outs_1_valid,
      outs_ready(0) => fork17_outs_0_ready,
      outs_ready(1) => fork17_outs_1_ready
    );

  buffer13 : entity work.handshake_buffer_7(arch)
    port map(
      ins => fork17_outs_0,
      ins_valid => fork17_outs_0_valid,
      ins_ready => fork17_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  source5 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source5_outs_valid,
      outs_ready => source5_outs_ready
    );

  constant6 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => source5_outs_valid,
      ctrl_ready => source5_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant6_outs,
      outs_valid => constant6_outs_valid,
      outs_ready => constant6_outs_ready
    );

  mux5 : entity work.handshake_mux_2(arch)
    port map(
      index => buffer13_outs,
      index_valid => buffer13_outs_valid,
      index_ready => buffer13_outs_ready,
      ins(0) => constant6_outs,
      ins(1) => mux4_outs,
      ins_valid(0) => constant6_outs_valid,
      ins_valid(1) => mux4_outs_valid,
      ins_ready(0) => constant6_outs_ready,
      ins_ready(1) => mux4_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux5_outs,
      outs_valid => mux5_outs_valid,
      outs_ready => mux5_outs_ready
    );

  spec_v2_repeating_init4 : entity work.handshake_spec_v2_repeating_init_0(arch)
    port map(
      ins => fork17_outs_1,
      ins_valid => fork17_outs_1_valid,
      ins_ready => fork17_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => spec_v2_repeating_init4_outs,
      outs_valid => spec_v2_repeating_init4_outs_valid,
      outs_ready => spec_v2_repeating_init4_outs_ready
    );

  buffer14 : entity work.handshake_buffer_5(arch)
    port map(
      ins => spec_v2_repeating_init4_outs,
      ins_valid => spec_v2_repeating_init4_outs_valid,
      ins_ready => spec_v2_repeating_init4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  fork18 : entity work.handshake_fork_3(arch)
    port map(
      ins => buffer14_outs,
      ins_valid => buffer14_outs_valid,
      ins_ready => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork18_outs_0,
      outs(1) => fork18_outs_1,
      outs_valid(0) => fork18_outs_0_valid,
      outs_valid(1) => fork18_outs_1_valid,
      outs_ready(0) => fork18_outs_0_ready,
      outs_ready(1) => fork18_outs_1_ready
    );

  buffer15 : entity work.handshake_buffer_8(arch)
    port map(
      ins => fork18_outs_0,
      ins_valid => fork18_outs_0_valid,
      ins_ready => fork18_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer15_outs,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  source6 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source6_outs_valid,
      outs_ready => source6_outs_ready
    );

  constant7 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => source6_outs_valid,
      ctrl_ready => source6_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant7_outs,
      outs_valid => constant7_outs_valid,
      outs_ready => constant7_outs_ready
    );

  mux6 : entity work.handshake_mux_2(arch)
    port map(
      index => buffer15_outs,
      index_valid => buffer15_outs_valid,
      index_ready => buffer15_outs_ready,
      ins(0) => constant7_outs,
      ins(1) => mux5_outs,
      ins_valid(0) => constant7_outs_valid,
      ins_valid(1) => mux5_outs_valid,
      ins_ready(0) => constant7_outs_ready,
      ins_ready(1) => mux5_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux6_outs,
      outs_valid => mux6_outs_valid,
      outs_ready => mux6_outs_ready
    );

  spec_v2_repeating_init5 : entity work.handshake_spec_v2_repeating_init_0(arch)
    port map(
      ins => fork18_outs_1,
      ins_valid => fork18_outs_1_valid,
      ins_ready => fork18_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => spec_v2_repeating_init5_outs,
      outs_valid => spec_v2_repeating_init5_outs_valid,
      outs_ready => spec_v2_repeating_init5_outs_ready
    );

  buffer16 : entity work.handshake_buffer_5(arch)
    port map(
      ins => spec_v2_repeating_init5_outs,
      ins_valid => spec_v2_repeating_init5_outs_valid,
      ins_ready => spec_v2_repeating_init5_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer16_outs,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  fork19 : entity work.handshake_fork_3(arch)
    port map(
      ins => buffer16_outs,
      ins_valid => buffer16_outs_valid,
      ins_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork19_outs_0,
      outs(1) => fork19_outs_1,
      outs_valid(0) => fork19_outs_0_valid,
      outs_valid(1) => fork19_outs_1_valid,
      outs_ready(0) => fork19_outs_0_ready,
      outs_ready(1) => fork19_outs_1_ready
    );

  buffer17 : entity work.handshake_buffer_9(arch)
    port map(
      ins => fork19_outs_0,
      ins_valid => fork19_outs_0_valid,
      ins_ready => fork19_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  source7 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source7_outs_valid,
      outs_ready => source7_outs_ready
    );

  constant8 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => source7_outs_valid,
      ctrl_ready => source7_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant8_outs,
      outs_valid => constant8_outs_valid,
      outs_ready => constant8_outs_ready
    );

  mux7 : entity work.handshake_mux_2(arch)
    port map(
      index => buffer17_outs,
      index_valid => buffer17_outs_valid,
      index_ready => buffer17_outs_ready,
      ins(0) => constant8_outs,
      ins(1) => mux6_outs,
      ins_valid(0) => constant8_outs_valid,
      ins_valid(1) => mux6_outs_valid,
      ins_ready(0) => constant8_outs_ready,
      ins_ready(1) => mux6_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux7_outs,
      outs_valid => mux7_outs_valid,
      outs_ready => mux7_outs_ready
    );

  spec_v2_repeating_init6 : entity work.handshake_spec_v2_repeating_init_0(arch)
    port map(
      ins => fork19_outs_1,
      ins_valid => fork19_outs_1_valid,
      ins_ready => fork19_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => spec_v2_repeating_init6_outs,
      outs_valid => spec_v2_repeating_init6_outs_valid,
      outs_ready => spec_v2_repeating_init6_outs_ready
    );

  buffer18 : entity work.handshake_buffer_5(arch)
    port map(
      ins => spec_v2_repeating_init6_outs,
      ins_valid => spec_v2_repeating_init6_outs_valid,
      ins_ready => spec_v2_repeating_init6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer18_outs,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  fork20 : entity work.handshake_fork_3(arch)
    port map(
      ins => buffer18_outs,
      ins_valid => buffer18_outs_valid,
      ins_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork20_outs_0,
      outs(1) => fork20_outs_1,
      outs_valid(0) => fork20_outs_0_valid,
      outs_valid(1) => fork20_outs_1_valid,
      outs_ready(0) => fork20_outs_0_ready,
      outs_ready(1) => fork20_outs_1_ready
    );

  buffer19 : entity work.handshake_buffer_10(arch)
    port map(
      ins => fork20_outs_0,
      ins_valid => fork20_outs_0_valid,
      ins_ready => fork20_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  source8 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source8_outs_valid,
      outs_ready => source8_outs_ready
    );

  constant9 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => source8_outs_valid,
      ctrl_ready => source8_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant9_outs,
      outs_valid => constant9_outs_valid,
      outs_ready => constant9_outs_ready
    );

  mux8 : entity work.handshake_mux_2(arch)
    port map(
      index => buffer19_outs,
      index_valid => buffer19_outs_valid,
      index_ready => buffer19_outs_ready,
      ins(0) => constant9_outs,
      ins(1) => mux7_outs,
      ins_valid(0) => constant9_outs_valid,
      ins_valid(1) => mux7_outs_valid,
      ins_ready(0) => constant9_outs_ready,
      ins_ready(1) => mux7_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux8_outs,
      outs_valid => mux8_outs_valid,
      outs_ready => mux8_outs_ready
    );

  spec_v2_repeating_init7 : entity work.handshake_spec_v2_repeating_init_0(arch)
    port map(
      ins => fork20_outs_1,
      ins_valid => fork20_outs_1_valid,
      ins_ready => fork20_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => spec_v2_repeating_init7_outs,
      outs_valid => spec_v2_repeating_init7_outs_valid,
      outs_ready => spec_v2_repeating_init7_outs_ready
    );

  buffer20 : entity work.handshake_buffer_5(arch)
    port map(
      ins => spec_v2_repeating_init7_outs,
      ins_valid => spec_v2_repeating_init7_outs_valid,
      ins_ready => spec_v2_repeating_init7_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  fork21 : entity work.handshake_fork_3(arch)
    port map(
      ins => buffer20_outs,
      ins_valid => buffer20_outs_valid,
      ins_ready => buffer20_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork21_outs_0,
      outs(1) => fork21_outs_1,
      outs_valid(0) => fork21_outs_0_valid,
      outs_valid(1) => fork21_outs_1_valid,
      outs_ready(0) => fork21_outs_0_ready,
      outs_ready(1) => fork21_outs_1_ready
    );

  buffer21 : entity work.handshake_buffer_11(arch)
    port map(
      ins => fork21_outs_0,
      ins_valid => fork21_outs_0_valid,
      ins_ready => fork21_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer21_outs,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  source9 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source9_outs_valid,
      outs_ready => source9_outs_ready
    );

  constant10 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => source9_outs_valid,
      ctrl_ready => source9_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant10_outs,
      outs_valid => constant10_outs_valid,
      outs_ready => constant10_outs_ready
    );

  mux9 : entity work.handshake_mux_2(arch)
    port map(
      index => buffer21_outs,
      index_valid => buffer21_outs_valid,
      index_ready => buffer21_outs_ready,
      ins(0) => constant10_outs,
      ins(1) => mux8_outs,
      ins_valid(0) => constant10_outs_valid,
      ins_valid(1) => mux8_outs_valid,
      ins_ready(0) => constant10_outs_ready,
      ins_ready(1) => mux8_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux9_outs,
      outs_valid => mux9_outs_valid,
      outs_ready => mux9_outs_ready
    );

  spec_v2_repeating_init8 : entity work.handshake_spec_v2_repeating_init_0(arch)
    port map(
      ins => fork21_outs_1,
      ins_valid => fork21_outs_1_valid,
      ins_ready => fork21_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => spec_v2_repeating_init8_outs,
      outs_valid => spec_v2_repeating_init8_outs_valid,
      outs_ready => spec_v2_repeating_init8_outs_ready
    );

  buffer22 : entity work.handshake_buffer_5(arch)
    port map(
      ins => spec_v2_repeating_init8_outs,
      ins_valid => spec_v2_repeating_init8_outs_valid,
      ins_ready => spec_v2_repeating_init8_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer22_outs,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  fork22 : entity work.handshake_fork_3(arch)
    port map(
      ins => buffer22_outs,
      ins_valid => buffer22_outs_valid,
      ins_ready => buffer22_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork22_outs_0,
      outs(1) => fork22_outs_1,
      outs_valid(0) => fork22_outs_0_valid,
      outs_valid(1) => fork22_outs_1_valid,
      outs_ready(0) => fork22_outs_0_ready,
      outs_ready(1) => fork22_outs_1_ready
    );

  buffer23 : entity work.handshake_buffer_12(arch)
    port map(
      ins => fork22_outs_0,
      ins_valid => fork22_outs_0_valid,
      ins_ready => fork22_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer23_outs,
      outs_valid => buffer23_outs_valid,
      outs_ready => buffer23_outs_ready
    );

  source10 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source10_outs_valid,
      outs_ready => source10_outs_ready
    );

  constant11 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => source10_outs_valid,
      ctrl_ready => source10_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant11_outs,
      outs_valid => constant11_outs_valid,
      outs_ready => constant11_outs_ready
    );

  mux10 : entity work.handshake_mux_2(arch)
    port map(
      index => buffer23_outs,
      index_valid => buffer23_outs_valid,
      index_ready => buffer23_outs_ready,
      ins(0) => constant11_outs,
      ins(1) => mux9_outs,
      ins_valid(0) => constant11_outs_valid,
      ins_valid(1) => mux9_outs_valid,
      ins_ready(0) => constant11_outs_ready,
      ins_ready(1) => mux9_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux10_outs,
      outs_valid => mux10_outs_valid,
      outs_ready => mux10_outs_ready
    );

  spec_v2_repeating_init9 : entity work.handshake_spec_v2_repeating_init_0(arch)
    port map(
      ins => fork22_outs_1,
      ins_valid => fork22_outs_1_valid,
      ins_ready => fork22_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => spec_v2_repeating_init9_outs,
      outs_valid => spec_v2_repeating_init9_outs_valid,
      outs_ready => spec_v2_repeating_init9_outs_ready
    );

  buffer24 : entity work.handshake_buffer_5(arch)
    port map(
      ins => spec_v2_repeating_init9_outs,
      ins_valid => spec_v2_repeating_init9_outs_valid,
      ins_ready => spec_v2_repeating_init9_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer24_outs,
      outs_valid => buffer24_outs_valid,
      outs_ready => buffer24_outs_ready
    );

  fork23 : entity work.handshake_fork_3(arch)
    port map(
      ins => buffer24_outs,
      ins_valid => buffer24_outs_valid,
      ins_ready => buffer24_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork23_outs_0,
      outs(1) => fork23_outs_1,
      outs_valid(0) => fork23_outs_0_valid,
      outs_valid(1) => fork23_outs_1_valid,
      outs_ready(0) => fork23_outs_0_ready,
      outs_ready(1) => fork23_outs_1_ready
    );

  buffer25 : entity work.handshake_buffer_13(arch)
    port map(
      ins => fork23_outs_0,
      ins_valid => fork23_outs_0_valid,
      ins_ready => fork23_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer25_outs,
      outs_valid => buffer25_outs_valid,
      outs_ready => buffer25_outs_ready
    );

  source11 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source11_outs_valid,
      outs_ready => source11_outs_ready
    );

  constant12 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => source11_outs_valid,
      ctrl_ready => source11_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant12_outs,
      outs_valid => constant12_outs_valid,
      outs_ready => constant12_outs_ready
    );

  mux11 : entity work.handshake_mux_2(arch)
    port map(
      index => buffer25_outs,
      index_valid => buffer25_outs_valid,
      index_ready => buffer25_outs_ready,
      ins(0) => constant12_outs,
      ins(1) => mux10_outs,
      ins_valid(0) => constant12_outs_valid,
      ins_valid(1) => mux10_outs_valid,
      ins_ready(0) => constant12_outs_ready,
      ins_ready(1) => mux10_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux11_outs,
      outs_valid => mux11_outs_valid,
      outs_ready => mux11_outs_ready
    );

  spec_v2_repeating_init10 : entity work.handshake_spec_v2_repeating_init_0(arch)
    port map(
      ins => fork23_outs_1,
      ins_valid => fork23_outs_1_valid,
      ins_ready => fork23_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => spec_v2_repeating_init10_outs,
      outs_valid => spec_v2_repeating_init10_outs_valid,
      outs_ready => spec_v2_repeating_init10_outs_ready
    );

  buffer26 : entity work.handshake_buffer_5(arch)
    port map(
      ins => spec_v2_repeating_init10_outs,
      ins_valid => spec_v2_repeating_init10_outs_valid,
      ins_ready => spec_v2_repeating_init10_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer26_outs,
      outs_valid => buffer26_outs_valid,
      outs_ready => buffer26_outs_ready
    );

  fork24 : entity work.handshake_fork_3(arch)
    port map(
      ins => buffer26_outs,
      ins_valid => buffer26_outs_valid,
      ins_ready => buffer26_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork24_outs_0,
      outs(1) => fork24_outs_1,
      outs_valid(0) => fork24_outs_0_valid,
      outs_valid(1) => fork24_outs_1_valid,
      outs_ready(0) => fork24_outs_0_ready,
      outs_ready(1) => fork24_outs_1_ready
    );

  buffer27 : entity work.handshake_buffer_14(arch)
    port map(
      ins => fork24_outs_0,
      ins_valid => fork24_outs_0_valid,
      ins_ready => fork24_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer27_outs,
      outs_valid => buffer27_outs_valid,
      outs_ready => buffer27_outs_ready
    );

  source12 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source12_outs_valid,
      outs_ready => source12_outs_ready
    );

  constant13 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => source12_outs_valid,
      ctrl_ready => source12_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant13_outs,
      outs_valid => constant13_outs_valid,
      outs_ready => constant13_outs_ready
    );

  mux12 : entity work.handshake_mux_2(arch)
    port map(
      index => buffer27_outs,
      index_valid => buffer27_outs_valid,
      index_ready => buffer27_outs_ready,
      ins(0) => constant13_outs,
      ins(1) => mux11_outs,
      ins_valid(0) => constant13_outs_valid,
      ins_valid(1) => mux11_outs_valid,
      ins_ready(0) => constant13_outs_ready,
      ins_ready(1) => mux11_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux12_outs,
      outs_valid => mux12_outs_valid,
      outs_ready => mux12_outs_ready
    );

  spec_v2_repeating_init11 : entity work.handshake_spec_v2_repeating_init_0(arch)
    port map(
      ins => fork24_outs_1,
      ins_valid => fork24_outs_1_valid,
      ins_ready => fork24_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => spec_v2_repeating_init11_outs,
      outs_valid => spec_v2_repeating_init11_outs_valid,
      outs_ready => spec_v2_repeating_init11_outs_ready
    );

  buffer29 : entity work.handshake_buffer_5(arch)
    port map(
      ins => spec_v2_repeating_init11_outs,
      ins_valid => spec_v2_repeating_init11_outs_valid,
      ins_ready => spec_v2_repeating_init11_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer29_outs,
      outs_valid => buffer29_outs_valid,
      outs_ready => buffer29_outs_ready
    );

  fork6 : entity work.handshake_fork_4(arch)
    port map(
      ins => buffer29_outs,
      ins_valid => buffer29_outs_valid,
      ins_ready => buffer29_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs(2) => fork6_outs_2,
      outs(3) => fork6_outs_3,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_valid(2) => fork6_outs_2_valid,
      outs_valid(3) => fork6_outs_3_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready,
      outs_ready(2) => fork6_outs_2_ready,
      outs_ready(3) => fork6_outs_3_ready
    );

  buffer33 : entity work.handshake_buffer_5(arch)
    port map(
      ins => fork6_outs_3,
      ins_valid => fork6_outs_3_valid,
      ins_ready => fork6_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer33_outs,
      outs_valid => buffer33_outs_valid,
      outs_ready => buffer33_outs_ready
    );

  init2 : entity work.handshake_init_0(arch)
    port map(
      ins => buffer33_outs,
      ins_valid => buffer33_outs_valid,
      ins_ready => buffer33_outs_ready,
      clk => clk,
      rst => rst,
      outs => init2_outs,
      outs_valid => init2_outs_valid,
      outs_ready => init2_outs_ready
    );

  fork7 : entity work.handshake_fork_3(arch)
    port map(
      ins => init2_outs,
      ins_valid => init2_outs_valid,
      ins_ready => init2_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork7_outs_0,
      outs(1) => fork7_outs_1,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready
    );

  buffer28 : entity work.handshake_buffer_3(arch)
    port map(
      ins => mux12_outs,
      ins_valid => mux12_outs_valid,
      ins_ready => mux12_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer28_outs,
      outs_valid => buffer28_outs_valid,
      outs_ready => buffer28_outs_ready
    );

  buffer31 : entity work.handshake_buffer_3(arch)
    port map(
      ins => fork6_outs_2,
      ins_valid => fork6_outs_2_valid,
      ins_ready => fork6_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer31_outs,
      outs_valid => buffer31_outs_valid,
      outs_ready => buffer31_outs_ready
    );

  buffer32 : entity work.handshake_buffer_4(arch)
    port map(
      ins => buffer31_outs,
      ins_valid => buffer31_outs_valid,
      ins_ready => buffer31_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer32_outs,
      outs_valid => buffer32_outs_valid,
      outs_ready => buffer32_outs_ready
    );

  source13 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source13_outs_valid,
      outs_ready => source13_outs_ready
    );

  constant14 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => source13_outs_valid,
      ctrl_ready => source13_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant14_outs,
      outs_valid => constant14_outs_valid,
      outs_ready => constant14_outs_ready
    );

  mux13 : entity work.handshake_mux_2(arch)
    port map(
      index => buffer32_outs,
      index_valid => buffer32_outs_valid,
      index_ready => buffer32_outs_ready,
      ins(0) => constant14_outs,
      ins(1) => buffer28_outs,
      ins_valid(0) => constant14_outs_valid,
      ins_valid(1) => buffer28_outs_valid,
      ins_ready(0) => constant14_outs_ready,
      ins_ready(1) => buffer28_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux13_outs,
      outs_valid => mux13_outs_valid,
      outs_ready => mux13_outs_ready
    );

  fork26 : entity work.handshake_fork_1(arch)
    port map(
      ins => mux13_outs,
      ins_valid => mux13_outs_valid,
      ins_ready => mux13_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork26_outs_0,
      outs(1) => fork26_outs_1,
      outs(2) => fork26_outs_2,
      outs_valid(0) => fork26_outs_0_valid,
      outs_valid(1) => fork26_outs_1_valid,
      outs_valid(2) => fork26_outs_2_valid,
      outs_ready(0) => fork26_outs_0_ready,
      outs_ready(1) => fork26_outs_1_ready,
      outs_ready(2) => fork26_outs_2_ready
    );

  andi13 : entity work.handshake_andi_0(arch)
    port map(
      lhs => not0_outs,
      lhs_valid => not0_outs_valid,
      lhs_ready => not0_outs_ready,
      rhs => fork26_outs_0,
      rhs_valid => fork26_outs_0_valid,
      rhs_ready => fork26_outs_0_ready,
      clk => clk,
      rst => rst,
      result => andi13_result,
      result_valid => andi13_result_valid,
      result_ready => andi13_result_ready
    );

  fork27 : entity work.handshake_fork_3(arch)
    port map(
      ins => andi13_result,
      ins_valid => andi13_result_valid,
      ins_ready => andi13_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork27_outs_0,
      outs(1) => fork27_outs_1,
      outs_valid(0) => fork27_outs_0_valid,
      outs_valid(1) => fork27_outs_1_valid,
      outs_ready(0) => fork27_outs_0_ready,
      outs_ready(1) => fork27_outs_1_ready
    );

  andi14 : entity work.handshake_andi_0(arch)
    port map(
      lhs => andi0_result,
      lhs_valid => andi0_result_valid,
      lhs_ready => andi0_result_ready,
      rhs => fork26_outs_1,
      rhs_valid => fork26_outs_1_valid,
      rhs_ready => fork26_outs_1_ready,
      clk => clk,
      rst => rst,
      result => andi14_result,
      result_valid => andi14_result_valid,
      result_ready => andi14_result_ready
    );

  fork28 : entity work.handshake_fork_3(arch)
    port map(
      ins => andi14_result,
      ins_valid => andi14_result_valid,
      ins_ready => andi14_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork28_outs_0,
      outs(1) => fork28_outs_1,
      outs_valid(0) => fork28_outs_0_valid,
      outs_valid(1) => fork28_outs_1_valid,
      outs_ready(0) => fork28_outs_0_ready,
      outs_ready(1) => fork28_outs_1_ready
    );

  not1 : entity work.handshake_not_0(arch)
    port map(
      ins => fork3_outs_1,
      ins_valid => fork3_outs_1_valid,
      ins_ready => fork3_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => not1_outs,
      outs_valid => not1_outs_valid,
      outs_ready => not1_outs_ready
    );

  buffer3 : entity work.handshake_buffer_15(arch)
    port map(
      ins => fork2_outs_1,
      ins_valid => fork2_outs_1_valid,
      ins_ready => fork2_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  passer5 : entity work.handshake_passer_0(arch)
    port map(
      data => buffer3_outs,
      data_valid => buffer3_outs_valid,
      data_ready => buffer3_outs_ready,
      ctrl => fork28_outs_1,
      ctrl_valid => fork28_outs_1_valid,
      ctrl_ready => fork28_outs_1_ready,
      clk => clk,
      rst => rst,
      result => passer5_result,
      result_valid => passer5_result_valid,
      result_ready => passer5_result_ready
    );

  buffer0 : entity work.handshake_buffer_16(arch)
    port map(
      ins => mux0_outs,
      ins_valid => mux0_outs_valid,
      ins_ready => mux0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer0_outs,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready
    );

  fork29 : entity work.handshake_fork_5(arch)
    port map(
      ins => buffer0_outs,
      ins_valid => buffer0_outs_valid,
      ins_ready => buffer0_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork29_outs_0,
      outs(1) => fork29_outs_1,
      outs(2) => fork29_outs_2,
      outs(3) => fork29_outs_3,
      outs(4) => fork29_outs_4,
      outs_valid(0) => fork29_outs_0_valid,
      outs_valid(1) => fork29_outs_1_valid,
      outs_valid(2) => fork29_outs_2_valid,
      outs_valid(3) => fork29_outs_3_valid,
      outs_valid(4) => fork29_outs_4_valid,
      outs_ready(0) => fork29_outs_0_ready,
      outs_ready(1) => fork29_outs_1_ready,
      outs_ready(2) => fork29_outs_2_ready,
      outs_ready(3) => fork29_outs_3_ready,
      outs_ready(4) => fork29_outs_4_ready
    );

  passer10 : entity work.handshake_passer_3(arch)
    port map(
      data => trunci3_outs,
      data_valid => trunci3_outs_valid,
      data_ready => trunci3_outs_ready,
      ctrl => fork6_outs_1,
      ctrl_valid => fork6_outs_1_valid,
      ctrl_ready => fork6_outs_1_ready,
      clk => clk,
      rst => rst,
      result => passer10_result,
      result_valid => passer10_result_valid,
      result_ready => passer10_result_ready
    );

  trunci3 : entity work.handshake_trunci_1(arch)
    port map(
      ins => fork2_outs_2,
      ins_valid => fork2_outs_2_valid,
      ins_ready => fork2_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => trunci3_outs,
      outs_valid => trunci3_outs_valid,
      outs_ready => trunci3_outs_ready
    );

  buffer30 : entity work.handshake_buffer_1(arch)
    port map(
      ins => fork6_outs_0,
      ins_valid => fork6_outs_0_valid,
      ins_ready => fork6_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer30_outs,
      outs_valid => buffer30_outs_valid,
      outs_ready => buffer30_outs_ready
    );

  passer11 : entity work.handshake_passer_1(arch)
    port map(
      data_valid => fork30_outs_1_valid,
      data_ready => fork30_outs_1_ready,
      ctrl => buffer30_outs,
      ctrl_valid => buffer30_outs_valid,
      ctrl_ready => buffer30_outs_ready,
      clk => clk,
      rst => rst,
      result_valid => passer11_result_valid,
      result_ready => passer11_result_ready
    );

  buffer1 : entity work.handshake_buffer_17(arch)
    port map(
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  buffer2 : entity work.handshake_buffer_18(arch)
    port map(
      ins_valid => buffer1_outs_valid,
      ins_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  fork30 : entity work.handshake_fork_0(arch)
    port map(
      ins_valid => buffer2_outs_valid,
      ins_ready => buffer2_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork30_outs_0_valid,
      outs_valid(1) => fork30_outs_1_valid,
      outs_valid(2) => fork30_outs_2_valid,
      outs_ready(0) => fork30_outs_0_ready,
      outs_ready(1) => fork30_outs_1_ready,
      outs_ready(2) => fork30_outs_2_ready
    );

  passer7 : entity work.handshake_passer_1(arch)
    port map(
      data_valid => fork30_outs_0_valid,
      data_ready => fork30_outs_0_ready,
      ctrl => fork28_outs_0,
      ctrl_valid => fork28_outs_0_valid,
      ctrl_ready => fork28_outs_0_ready,
      clk => clk,
      rst => rst,
      result_valid => passer7_result_valid,
      result_ready => passer7_result_ready
    );

  mux1 : entity work.handshake_mux_3(arch)
    port map(
      index => control_merge2_index,
      index_valid => control_merge2_index_valid,
      index_ready => control_merge2_index_ready,
      ins(0) => passer8_result,
      ins(1) => passer5_result,
      ins_valid(0) => passer8_result_valid,
      ins_valid(1) => passer5_result_valid,
      ins_ready(0) => passer8_result_ready,
      ins_ready(1) => passer5_result_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  extsi8 : entity work.handshake_extsi_3(arch)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi8_outs,
      outs_valid => extsi8_outs_valid,
      outs_ready => extsi8_outs_ready
    );

  control_merge2 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => passer3_result_valid,
      ins_valid(1) => passer7_result_valid,
      ins_ready(0) => passer3_result_ready,
      ins_ready(1) => passer7_result_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge2_outs_valid,
      outs_ready => control_merge2_outs_ready,
      index => control_merge2_index,
      index_valid => control_merge2_index_valid,
      index_ready => control_merge2_index_ready
    );

  fork5 : entity work.handshake_fork_0(arch)
    port map(
      ins_valid => control_merge2_outs_valid,
      ins_ready => control_merge2_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_valid(2) => fork5_outs_2_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready,
      outs_ready(2) => fork5_outs_2_ready
    );

end architecture;
