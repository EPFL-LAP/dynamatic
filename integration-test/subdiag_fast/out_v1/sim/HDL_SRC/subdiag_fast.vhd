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
  signal non_spec0_dataOut_valid : std_logic;
  signal non_spec0_dataOut_ready : std_logic;
  signal non_spec0_dataOut_spec : std_logic_vector(0 downto 0);
  signal buffer7_outs : std_logic_vector(0 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal buffer8_outs_spec : std_logic_vector(0 downto 0);
  signal spec_commit0_outs_valid : std_logic;
  signal spec_commit0_outs_ready : std_logic;
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
  signal buffer9_outs : std_logic_vector(0 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal buffer10_outs_spec : std_logic_vector(0 downto 0);
  signal spec_commit1_outs_valid : std_logic;
  signal spec_commit1_outs_ready : std_logic;
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
  signal buffer11_outs : std_logic_vector(0 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal buffer12_outs_spec : std_logic_vector(0 downto 0);
  signal spec_commit2_outs_valid : std_logic;
  signal spec_commit2_outs_ready : std_logic;
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
  signal extsi3_outs : std_logic_vector(31 downto 0);
  signal extsi3_outs_valid : std_logic;
  signal extsi3_outs_ready : std_logic;
  signal non_spec1_dataOut : std_logic_vector(31 downto 0);
  signal non_spec1_dataOut_valid : std_logic;
  signal non_spec1_dataOut_ready : std_logic;
  signal non_spec1_dataOut_spec : std_logic_vector(0 downto 0);
  signal mux0_outs : std_logic_vector(31 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal mux0_outs_spec : std_logic_vector(0 downto 0);
  signal buffer0_outs : std_logic_vector(31 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal buffer0_outs_spec : std_logic_vector(0 downto 0);
  signal buffer1_outs : std_logic_vector(31 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal buffer1_outs_spec : std_logic_vector(0 downto 0);
  signal fork1_outs_0 : std_logic_vector(31 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork1_outs_1 : std_logic_vector(31 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal fork1_outs_1_spec : std_logic_vector(0 downto 0);
  signal fork1_outs_2 : std_logic_vector(31 downto 0);
  signal fork1_outs_2_valid : std_logic;
  signal fork1_outs_2_ready : std_logic;
  signal fork1_outs_2_spec : std_logic_vector(0 downto 0);
  signal fork1_outs_3 : std_logic_vector(31 downto 0);
  signal fork1_outs_3_valid : std_logic;
  signal fork1_outs_3_ready : std_logic;
  signal fork1_outs_3_spec : std_logic_vector(0 downto 0);
  signal fork1_outs_4 : std_logic_vector(31 downto 0);
  signal fork1_outs_4_valid : std_logic;
  signal fork1_outs_4_ready : std_logic;
  signal fork1_outs_4_spec : std_logic_vector(0 downto 0);
  signal fork1_outs_5 : std_logic_vector(31 downto 0);
  signal fork1_outs_5_valid : std_logic;
  signal fork1_outs_5_ready : std_logic;
  signal fork1_outs_5_spec : std_logic_vector(0 downto 0);
  signal trunci0_outs : std_logic_vector(9 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal trunci0_outs_spec : std_logic_vector(0 downto 0);
  signal trunci1_outs : std_logic_vector(9 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal trunci1_outs_spec : std_logic_vector(0 downto 0);
  signal trunci2_outs : std_logic_vector(9 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal trunci2_outs_spec : std_logic_vector(0 downto 0);
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_outs_spec : std_logic_vector(0 downto 0);
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal control_merge0_index_spec : std_logic_vector(0 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal buffer3_outs_spec : std_logic_vector(0 downto 0);
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal fork5_outs_1_spec : std_logic_vector(0 downto 0);
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal source0_outs_spec : std_logic_vector(0 downto 0);
  signal constant1_outs : std_logic_vector(1 downto 0);
  signal constant1_outs_valid : std_logic;
  signal constant1_outs_ready : std_logic;
  signal constant1_outs_spec : std_logic_vector(0 downto 0);
  signal fork2_outs_0 : std_logic_vector(1 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork2_outs_1 : std_logic_vector(1 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal fork2_outs_1_spec : std_logic_vector(0 downto 0);
  signal extsi4_outs : std_logic_vector(9 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal extsi4_outs_spec : std_logic_vector(0 downto 0);
  signal extsi1_outs : std_logic_vector(31 downto 0);
  signal extsi1_outs_valid : std_logic;
  signal extsi1_outs_ready : std_logic;
  signal extsi1_outs_spec : std_logic_vector(0 downto 0);
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal source1_outs_spec : std_logic_vector(0 downto 0);
  signal constant5_outs : std_logic_vector(31 downto 0);
  signal constant5_outs_valid : std_logic;
  signal constant5_outs_ready : std_logic;
  signal constant5_outs_spec : std_logic_vector(0 downto 0);
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal source2_outs_spec : std_logic_vector(0 downto 0);
  signal constant2_outs : std_logic_vector(10 downto 0);
  signal constant2_outs_valid : std_logic;
  signal constant2_outs_ready : std_logic;
  signal constant2_outs_spec : std_logic_vector(0 downto 0);
  signal extsi2_outs : std_logic_vector(31 downto 0);
  signal extsi2_outs_valid : std_logic;
  signal extsi2_outs_ready : std_logic;
  signal extsi2_outs_spec : std_logic_vector(0 downto 0);
  signal load0_addrOut : std_logic_vector(9 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal load0_dataOut_spec : std_logic_vector(0 downto 0);
  signal addi0_result : std_logic_vector(9 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal addi0_result_spec : std_logic_vector(0 downto 0);
  signal load1_addrOut : std_logic_vector(9 downto 0);
  signal load1_addrOut_valid : std_logic;
  signal load1_addrOut_ready : std_logic;
  signal load1_dataOut : std_logic_vector(31 downto 0);
  signal load1_dataOut_valid : std_logic;
  signal load1_dataOut_ready : std_logic;
  signal load1_dataOut_spec : std_logic_vector(0 downto 0);
  signal addf0_result : std_logic_vector(31 downto 0);
  signal addf0_result_valid : std_logic;
  signal addf0_result_ready : std_logic;
  signal addf0_result_spec : std_logic_vector(0 downto 0);
  signal addi1_result : std_logic_vector(31 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal addi1_result_spec : std_logic_vector(0 downto 0);
  signal load2_addrOut : std_logic_vector(9 downto 0);
  signal load2_addrOut_valid : std_logic;
  signal load2_addrOut_ready : std_logic;
  signal load2_dataOut : std_logic_vector(31 downto 0);
  signal load2_dataOut_valid : std_logic;
  signal load2_dataOut_ready : std_logic;
  signal load2_dataOut_spec : std_logic_vector(0 downto 0);
  signal mulf0_result : std_logic_vector(31 downto 0);
  signal mulf0_result_valid : std_logic;
  signal mulf0_result_ready : std_logic;
  signal mulf0_result_spec : std_logic_vector(0 downto 0);
  signal buffer4_outs : std_logic_vector(31 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal buffer4_outs_spec : std_logic_vector(0 downto 0);
  signal cmpf0_result : std_logic_vector(0 downto 0);
  signal cmpf0_result_valid : std_logic;
  signal cmpf0_result_ready : std_logic;
  signal cmpf0_result_spec : std_logic_vector(0 downto 0);
  signal buffer2_outs : std_logic_vector(31 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal buffer2_outs_spec : std_logic_vector(0 downto 0);
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal cmpi0_result_spec : std_logic_vector(0 downto 0);
  signal andi0_result : std_logic_vector(0 downto 0);
  signal andi0_result_valid : std_logic;
  signal andi0_result_ready : std_logic;
  signal andi0_result_spec : std_logic_vector(0 downto 0);
  signal speculator0_outs : std_logic_vector(0 downto 0);
  signal speculator0_outs_valid : std_logic;
  signal speculator0_outs_ready : std_logic;
  signal speculator0_outs_spec : std_logic_vector(0 downto 0);
  signal speculator0_ctrl_save : std_logic_vector(0 downto 0);
  signal speculator0_ctrl_save_valid : std_logic;
  signal speculator0_ctrl_save_ready : std_logic;
  signal speculator0_ctrl_commit : std_logic_vector(0 downto 0);
  signal speculator0_ctrl_commit_valid : std_logic;
  signal speculator0_ctrl_commit_ready : std_logic;
  signal speculator0_ctrl_sc_save : std_logic_vector(2 downto 0);
  signal speculator0_ctrl_sc_save_valid : std_logic;
  signal speculator0_ctrl_sc_save_ready : std_logic;
  signal speculator0_ctrl_sc_commit : std_logic_vector(2 downto 0);
  signal speculator0_ctrl_sc_commit_valid : std_logic;
  signal speculator0_ctrl_sc_commit_ready : std_logic;
  signal speculator0_ctrl_sc_branch : std_logic_vector(0 downto 0);
  signal speculator0_ctrl_sc_branch_valid : std_logic;
  signal speculator0_ctrl_sc_branch_ready : std_logic;
  signal cond_br5_trueOut : std_logic_vector(2 downto 0);
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_falseOut : std_logic_vector(2 downto 0);
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal merge0_outs : std_logic_vector(2 downto 0);
  signal merge0_outs_valid : std_logic;
  signal merge0_outs_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(2 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(2 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal fork6_outs_2 : std_logic_vector(2 downto 0);
  signal fork6_outs_2_valid : std_logic;
  signal fork6_outs_2_ready : std_logic;
  signal fork7_outs_0 : std_logic_vector(0 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork7_outs_1 : std_logic_vector(0 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal fork7_outs_1_spec : std_logic_vector(0 downto 0);
  signal fork7_outs_2 : std_logic_vector(0 downto 0);
  signal fork7_outs_2_valid : std_logic;
  signal fork7_outs_2_ready : std_logic;
  signal fork7_outs_2_spec : std_logic_vector(0 downto 0);
  signal fork7_outs_3 : std_logic_vector(0 downto 0);
  signal fork7_outs_3_valid : std_logic;
  signal fork7_outs_3_ready : std_logic;
  signal fork7_outs_3_spec : std_logic_vector(0 downto 0);
  signal fork7_outs_4 : std_logic_vector(0 downto 0);
  signal fork7_outs_4_valid : std_logic;
  signal fork7_outs_4_ready : std_logic;
  signal fork7_outs_4_spec : std_logic_vector(0 downto 0);
  signal buffer13_outs : std_logic_vector(0 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal cond_br0_trueOut : std_logic_vector(0 downto 0);
  signal cond_br0_trueOut_valid : std_logic;
  signal cond_br0_trueOut_ready : std_logic;
  signal cond_br0_falseOut : std_logic_vector(0 downto 0);
  signal cond_br0_falseOut_valid : std_logic;
  signal cond_br0_falseOut_ready : std_logic;
  signal fork8_outs_0 : std_logic_vector(0 downto 0);
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1 : std_logic_vector(0 downto 0);
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal fork8_outs_2 : std_logic_vector(0 downto 0);
  signal fork8_outs_2_valid : std_logic;
  signal fork8_outs_2_ready : std_logic;
  signal fork8_outs_3 : std_logic_vector(0 downto 0);
  signal fork8_outs_3_valid : std_logic;
  signal fork8_outs_3_ready : std_logic;
  signal speculating_branch0_trueOut : std_logic_vector(0 downto 0);
  signal speculating_branch0_trueOut_valid : std_logic;
  signal speculating_branch0_trueOut_ready : std_logic;
  signal speculating_branch0_falseOut : std_logic_vector(0 downto 0);
  signal speculating_branch0_falseOut_valid : std_logic;
  signal speculating_branch0_falseOut_ready : std_logic;
  signal buffer5_outs : std_logic_vector(0 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(0 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(0 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal buffer14_outs : std_logic_vector(0 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal cond_br4_trueOut : std_logic_vector(0 downto 0);
  signal cond_br4_trueOut_valid : std_logic;
  signal cond_br4_trueOut_ready : std_logic;
  signal cond_br4_falseOut : std_logic_vector(0 downto 0);
  signal cond_br4_falseOut_valid : std_logic;
  signal cond_br4_falseOut_ready : std_logic;
  signal spec_save_commit0_outs : std_logic_vector(31 downto 0);
  signal spec_save_commit0_outs_valid : std_logic;
  signal spec_save_commit0_outs_ready : std_logic;
  signal spec_save_commit0_outs_spec : std_logic_vector(0 downto 0);
  signal cond_br1_trueOut : std_logic_vector(31 downto 0);
  signal cond_br1_trueOut_valid : std_logic;
  signal cond_br1_trueOut_ready : std_logic;
  signal cond_br1_trueOut_spec : std_logic_vector(0 downto 0);
  signal cond_br1_falseOut : std_logic_vector(31 downto 0);
  signal cond_br1_falseOut_valid : std_logic;
  signal cond_br1_falseOut_ready : std_logic;
  signal cond_br1_falseOut_spec : std_logic_vector(0 downto 0);
  signal spec_save_commit1_outs_valid : std_logic;
  signal spec_save_commit1_outs_ready : std_logic;
  signal spec_save_commit1_outs_spec : std_logic_vector(0 downto 0);
  signal cond_br2_trueOut_valid : std_logic;
  signal cond_br2_trueOut_ready : std_logic;
  signal cond_br2_trueOut_spec : std_logic_vector(0 downto 0);
  signal cond_br2_falseOut_valid : std_logic;
  signal cond_br2_falseOut_ready : std_logic;
  signal cond_br2_falseOut_spec : std_logic_vector(0 downto 0);
  signal spec_save_commit2_outs : std_logic_vector(31 downto 0);
  signal spec_save_commit2_outs_valid : std_logic;
  signal spec_save_commit2_outs_ready : std_logic;
  signal spec_save_commit2_outs_spec : std_logic_vector(0 downto 0);
  signal cond_br3_trueOut : std_logic_vector(31 downto 0);
  signal cond_br3_trueOut_valid : std_logic;
  signal cond_br3_trueOut_ready : std_logic;
  signal cond_br3_trueOut_spec : std_logic_vector(0 downto 0);
  signal cond_br3_falseOut : std_logic_vector(31 downto 0);
  signal cond_br3_falseOut_valid : std_logic;
  signal cond_br3_falseOut_ready : std_logic;
  signal cond_br3_falseOut_spec : std_logic_vector(0 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal fork4_outs_1_spec : std_logic_vector(0 downto 0);
  signal fork4_outs_2_valid : std_logic;
  signal fork4_outs_2_ready : std_logic;
  signal fork4_outs_2_spec : std_logic_vector(0 downto 0);
  signal buffer6_outs : std_logic_vector(31 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal buffer6_outs_spec : std_logic_vector(0 downto 0);
  signal buffer15_outs : std_logic_vector(0 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal buffer16_outs : std_logic_vector(31 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal buffer16_outs_spec : std_logic_vector(0 downto 0);
  signal spec_commit3_outs : std_logic_vector(31 downto 0);
  signal spec_commit3_outs_valid : std_logic;
  signal spec_commit3_outs_ready : std_logic;

begin

  out0 <= spec_commit3_outs;
  out0_valid <= spec_commit3_outs_valid;
  spec_commit3_outs_ready <= out0_ready;
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

  non_spec0 : entity work.handshake_non_spec_0(arch)
    port map(
      dataIn_valid => fork0_outs_2_valid,
      dataIn_ready => fork0_outs_2_ready,
      clk => clk,
      rst => rst,
      dataOut_valid => non_spec0_dataOut_valid,
      dataOut_ready => non_spec0_dataOut_ready,
      dataOut_spec => non_spec0_dataOut_spec
    );

  buffer7 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork8_outs_0,
      ins_valid => fork8_outs_0_valid,
      ins_ready => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer7_outs,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  buffer8 : entity work.handshake_buffer_1(arch)
    port map(
      ins_valid => fork4_outs_2_valid,
      ins_ready => fork4_outs_2_ready,
      ins_spec => fork4_outs_2_spec,
      clk => clk,
      rst => rst,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready,
      outs_spec => buffer8_outs_spec
    );

  spec_commit0 : entity work.handshake_spec_commit_0(arch)
    port map(
      ins_valid => buffer8_outs_valid,
      ins_ready => buffer8_outs_ready,
      ins_spec => buffer8_outs_spec,
      ctrl => buffer7_outs,
      ctrl_valid => buffer7_outs_valid,
      ctrl_ready => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => spec_commit0_outs_valid,
      outs_ready => spec_commit0_outs_ready
    );

  mem_controller3 : entity work.handshake_mem_controller_0(arch)
    port map(
      loadData => e_loadData,
      memStart_valid => e_start_valid,
      memStart_ready => e_start_ready,
      ldAddr(0) => load2_addrOut,
      ldAddr_valid(0) => load2_addrOut_valid,
      ldAddr_ready(0) => load2_addrOut_ready,
      ctrlEnd_valid => spec_commit0_outs_valid,
      ctrlEnd_ready => spec_commit0_outs_ready,
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

  buffer9 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork8_outs_1,
      ins_valid => fork8_outs_1_valid,
      ins_ready => fork8_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer9_outs,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  buffer10 : entity work.handshake_buffer_1(arch)
    port map(
      ins_valid => fork4_outs_1_valid,
      ins_ready => fork4_outs_1_ready,
      ins_spec => fork4_outs_1_spec,
      clk => clk,
      rst => rst,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready,
      outs_spec => buffer10_outs_spec
    );

  spec_commit1 : entity work.handshake_spec_commit_0(arch)
    port map(
      ins_valid => buffer10_outs_valid,
      ins_ready => buffer10_outs_ready,
      ins_spec => buffer10_outs_spec,
      ctrl => buffer9_outs,
      ctrl_valid => buffer9_outs_valid,
      ctrl_ready => buffer9_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => spec_commit1_outs_valid,
      outs_ready => spec_commit1_outs_ready
    );

  mem_controller4 : entity work.handshake_mem_controller_1(arch)
    port map(
      loadData => d2_loadData,
      memStart_valid => d2_start_valid,
      memStart_ready => d2_start_ready,
      ldAddr(0) => load1_addrOut,
      ldAddr_valid(0) => load1_addrOut_valid,
      ldAddr_ready(0) => load1_addrOut_ready,
      ctrlEnd_valid => spec_commit1_outs_valid,
      ctrlEnd_ready => spec_commit1_outs_ready,
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

  buffer11 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork8_outs_2,
      ins_valid => fork8_outs_2_valid,
      ins_ready => fork8_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  buffer12 : entity work.handshake_buffer_1(arch)
    port map(
      ins_valid => fork4_outs_0_valid,
      ins_ready => fork4_outs_0_ready,
      ins_spec => fork4_outs_0_spec,
      clk => clk,
      rst => rst,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready,
      outs_spec => buffer12_outs_spec
    );

  spec_commit2 : entity work.handshake_spec_commit_0(arch)
    port map(
      ins_valid => buffer12_outs_valid,
      ins_ready => buffer12_outs_ready,
      ins_spec => buffer12_outs_spec,
      ctrl => buffer11_outs,
      ctrl_valid => buffer11_outs_valid,
      ctrl_ready => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => spec_commit2_outs_valid,
      outs_ready => spec_commit2_outs_ready
    );

  mem_controller5 : entity work.handshake_mem_controller_2(arch)
    port map(
      loadData => d1_loadData,
      memStart_valid => d1_start_valid,
      memStart_ready => d1_start_ready,
      ldAddr(0) => load0_addrOut,
      ldAddr_valid(0) => load0_addrOut_valid,
      ldAddr_ready(0) => load0_addrOut_ready,
      ctrlEnd_valid => spec_commit2_outs_valid,
      ctrlEnd_ready => spec_commit2_outs_ready,
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

  non_spec1 : entity work.handshake_non_spec_1(arch)
    port map(
      dataIn => extsi3_outs,
      dataIn_valid => extsi3_outs_valid,
      dataIn_ready => extsi3_outs_ready,
      clk => clk,
      rst => rst,
      dataOut => non_spec1_dataOut,
      dataOut_valid => non_spec1_dataOut_valid,
      dataOut_ready => non_spec1_dataOut_ready,
      dataOut_spec => non_spec1_dataOut_spec
    );

  mux0 : entity work.handshake_mux_0(arch)
    port map(
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready,
      index_spec => control_merge0_index_spec,
      ins(0) => non_spec1_dataOut,
      ins(1) => cond_br1_trueOut,
      ins_valid(0) => non_spec1_dataOut_valid,
      ins_valid(1) => cond_br1_trueOut_valid,
      ins_ready(0) => non_spec1_dataOut_ready,
      ins_ready(1) => cond_br1_trueOut_ready,
      ins_0_spec => non_spec1_dataOut_spec,
      ins_1_spec => cond_br1_trueOut_spec,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready,
      outs_spec => mux0_outs_spec
    );

  buffer0 : entity work.handshake_buffer_2(arch)
    port map(
      ins => mux0_outs,
      ins_valid => mux0_outs_valid,
      ins_ready => mux0_outs_ready,
      ins_spec => mux0_outs_spec,
      clk => clk,
      rst => rst,
      outs => buffer0_outs,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready,
      outs_spec => buffer0_outs_spec
    );

  buffer1 : entity work.handshake_buffer_3(arch)
    port map(
      ins => buffer0_outs,
      ins_valid => buffer0_outs_valid,
      ins_ready => buffer0_outs_ready,
      ins_spec => buffer0_outs_spec,
      clk => clk,
      rst => rst,
      outs => buffer1_outs,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready,
      outs_spec => buffer1_outs_spec
    );

  fork1 : entity work.handshake_fork_1(arch)
    port map(
      ins => buffer1_outs,
      ins_valid => buffer1_outs_valid,
      ins_ready => buffer1_outs_ready,
      ins_spec => buffer1_outs_spec,
      clk => clk,
      rst => rst,
      outs(0) => fork1_outs_0,
      outs(1) => fork1_outs_1,
      outs(2) => fork1_outs_2,
      outs(3) => fork1_outs_3,
      outs(4) => fork1_outs_4,
      outs(5) => fork1_outs_5,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_valid(2) => fork1_outs_2_valid,
      outs_valid(3) => fork1_outs_3_valid,
      outs_valid(4) => fork1_outs_4_valid,
      outs_valid(5) => fork1_outs_5_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready,
      outs_ready(2) => fork1_outs_2_ready,
      outs_ready(3) => fork1_outs_3_ready,
      outs_ready(4) => fork1_outs_4_ready,
      outs_ready(5) => fork1_outs_5_ready,
      outs_0_spec => fork1_outs_0_spec,
      outs_1_spec => fork1_outs_1_spec,
      outs_2_spec => fork1_outs_2_spec,
      outs_3_spec => fork1_outs_3_spec,
      outs_4_spec => fork1_outs_4_spec,
      outs_5_spec => fork1_outs_5_spec
    );

  trunci0 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork1_outs_0,
      ins_valid => fork1_outs_0_valid,
      ins_ready => fork1_outs_0_ready,
      ins_spec => fork1_outs_0_spec,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready,
      outs_spec => trunci0_outs_spec
    );

  trunci1 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork1_outs_1,
      ins_valid => fork1_outs_1_valid,
      ins_ready => fork1_outs_1_ready,
      ins_spec => fork1_outs_1_spec,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready,
      outs_spec => trunci1_outs_spec
    );

  trunci2 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork1_outs_2,
      ins_valid => fork1_outs_2_valid,
      ins_ready => fork1_outs_2_ready,
      ins_spec => fork1_outs_2_spec,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready,
      outs_spec => trunci2_outs_spec
    );

  control_merge0 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => non_spec0_dataOut_valid,
      ins_valid(1) => cond_br2_trueOut_valid,
      ins_ready(0) => non_spec0_dataOut_ready,
      ins_ready(1) => cond_br2_trueOut_ready,
      ins_0_spec => non_spec0_dataOut_spec,
      ins_1_spec => cond_br2_trueOut_spec,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      outs_spec => control_merge0_outs_spec,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready,
      index_spec => control_merge0_index_spec
    );

  buffer3 : entity work.handshake_buffer_4(arch)
    port map(
      ins_valid => control_merge0_outs_valid,
      ins_ready => control_merge0_outs_ready,
      ins_spec => control_merge0_outs_spec,
      clk => clk,
      rst => rst,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready,
      outs_spec => buffer3_outs_spec
    );

  fork5 : entity work.handshake_fork_2(arch)
    port map(
      ins_valid => buffer3_outs_valid,
      ins_ready => buffer3_outs_ready,
      ins_spec => buffer3_outs_spec,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready,
      outs_0_spec => fork5_outs_0_spec,
      outs_1_spec => fork5_outs_1_spec
    );

  source0 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready,
      outs_spec => source0_outs_spec
    );

  constant1 : entity work.handshake_constant_1(arch)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      ctrl_spec => source0_outs_spec,
      clk => clk,
      rst => rst,
      outs => constant1_outs,
      outs_valid => constant1_outs_valid,
      outs_ready => constant1_outs_ready,
      outs_spec => constant1_outs_spec
    );

  fork2 : entity work.handshake_fork_3(arch)
    port map(
      ins => constant1_outs,
      ins_valid => constant1_outs_valid,
      ins_ready => constant1_outs_ready,
      ins_spec => constant1_outs_spec,
      clk => clk,
      rst => rst,
      outs(0) => fork2_outs_0,
      outs(1) => fork2_outs_1,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready,
      outs_0_spec => fork2_outs_0_spec,
      outs_1_spec => fork2_outs_1_spec
    );

  extsi4 : entity work.handshake_extsi_1(arch)
    port map(
      ins => fork2_outs_0,
      ins_valid => fork2_outs_0_valid,
      ins_ready => fork2_outs_0_ready,
      ins_spec => fork2_outs_0_spec,
      clk => clk,
      rst => rst,
      outs => extsi4_outs,
      outs_valid => extsi4_outs_valid,
      outs_ready => extsi4_outs_ready,
      outs_spec => extsi4_outs_spec
    );

  extsi1 : entity work.handshake_extsi_2(arch)
    port map(
      ins => fork2_outs_1,
      ins_valid => fork2_outs_1_valid,
      ins_ready => fork2_outs_1_ready,
      ins_spec => fork2_outs_1_spec,
      clk => clk,
      rst => rst,
      outs => extsi1_outs,
      outs_valid => extsi1_outs_valid,
      outs_ready => extsi1_outs_ready,
      outs_spec => extsi1_outs_spec
    );

  source1 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready,
      outs_spec => source1_outs_spec
    );

  constant5 : entity work.handshake_constant_2(arch)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      ctrl_spec => source1_outs_spec,
      clk => clk,
      rst => rst,
      outs => constant5_outs,
      outs_valid => constant5_outs_valid,
      outs_ready => constant5_outs_ready,
      outs_spec => constant5_outs_spec
    );

  source2 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready,
      outs_spec => source2_outs_spec
    );

  constant2 : entity work.handshake_constant_3(arch)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      ctrl_spec => source2_outs_spec,
      clk => clk,
      rst => rst,
      outs => constant2_outs,
      outs_valid => constant2_outs_valid,
      outs_ready => constant2_outs_ready,
      outs_spec => constant2_outs_spec
    );

  extsi2 : entity work.handshake_extsi_3(arch)
    port map(
      ins => constant2_outs,
      ins_valid => constant2_outs_valid,
      ins_ready => constant2_outs_ready,
      ins_spec => constant2_outs_spec,
      clk => clk,
      rst => rst,
      outs => extsi2_outs,
      outs_valid => extsi2_outs_valid,
      outs_ready => extsi2_outs_ready,
      outs_spec => extsi2_outs_spec
    );

  load0 : entity work.handshake_load_0(arch)
    port map(
      addrIn => trunci2_outs,
      addrIn_valid => trunci2_outs_valid,
      addrIn_ready => trunci2_outs_ready,
      addrIn_spec => trunci2_outs_spec,
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
      dataOut_ready => load0_dataOut_ready,
      dataOut_spec => load0_dataOut_spec
    );

  addi0 : entity work.handshake_addi_0(arch)
    port map(
      lhs => trunci0_outs,
      lhs_valid => trunci0_outs_valid,
      lhs_ready => trunci0_outs_ready,
      lhs_spec => trunci0_outs_spec,
      rhs => extsi4_outs,
      rhs_valid => extsi4_outs_valid,
      rhs_ready => extsi4_outs_ready,
      rhs_spec => extsi4_outs_spec,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready,
      result_spec => addi0_result_spec
    );

  load1 : entity work.handshake_load_1(arch)
    port map(
      addrIn => addi0_result,
      addrIn_valid => addi0_result_valid,
      addrIn_ready => addi0_result_ready,
      addrIn_spec => addi0_result_spec,
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
      dataOut_ready => load1_dataOut_ready,
      dataOut_spec => load1_dataOut_spec
    );

  addf0 : entity work.handshake_addf_0(arch)
    port map(
      lhs => load0_dataOut,
      lhs_valid => load0_dataOut_valid,
      lhs_ready => load0_dataOut_ready,
      lhs_spec => load0_dataOut_spec,
      rhs => load1_dataOut,
      rhs_valid => load1_dataOut_valid,
      rhs_ready => load1_dataOut_ready,
      rhs_spec => load1_dataOut_spec,
      clk => clk,
      rst => rst,
      result => addf0_result,
      result_valid => addf0_result_valid,
      result_ready => addf0_result_ready,
      result_spec => addf0_result_spec
    );

  addi1 : entity work.handshake_addi_1(arch)
    port map(
      lhs => fork1_outs_5,
      lhs_valid => fork1_outs_5_valid,
      lhs_ready => fork1_outs_5_ready,
      lhs_spec => fork1_outs_5_spec,
      rhs => extsi1_outs,
      rhs_valid => extsi1_outs_valid,
      rhs_ready => extsi1_outs_ready,
      rhs_spec => extsi1_outs_spec,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready,
      result_spec => addi1_result_spec
    );

  load2 : entity work.handshake_load_2(arch)
    port map(
      addrIn => trunci1_outs,
      addrIn_valid => trunci1_outs_valid,
      addrIn_ready => trunci1_outs_ready,
      addrIn_spec => trunci1_outs_spec,
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
      dataOut_ready => load2_dataOut_ready,
      dataOut_spec => load2_dataOut_spec
    );

  mulf0 : entity work.handshake_mulf_0(arch)
    port map(
      lhs => addf0_result,
      lhs_valid => addf0_result_valid,
      lhs_ready => addf0_result_ready,
      lhs_spec => addf0_result_spec,
      rhs => constant5_outs,
      rhs_valid => constant5_outs_valid,
      rhs_ready => constant5_outs_ready,
      rhs_spec => constant5_outs_spec,
      clk => clk,
      rst => rst,
      result => mulf0_result,
      result_valid => mulf0_result_valid,
      result_ready => mulf0_result_ready,
      result_spec => mulf0_result_spec
    );

  buffer4 : entity work.handshake_buffer_5(arch)
    port map(
      ins => load2_dataOut,
      ins_valid => load2_dataOut_valid,
      ins_ready => load2_dataOut_ready,
      ins_spec => load2_dataOut_spec,
      clk => clk,
      rst => rst,
      outs => buffer4_outs,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready,
      outs_spec => buffer4_outs_spec
    );

  cmpf0 : entity work.handshake_cmpf_0(arch)
    port map(
      lhs => buffer4_outs,
      lhs_valid => buffer4_outs_valid,
      lhs_ready => buffer4_outs_ready,
      lhs_spec => buffer4_outs_spec,
      rhs => mulf0_result,
      rhs_valid => mulf0_result_valid,
      rhs_ready => mulf0_result_ready,
      rhs_spec => mulf0_result_spec,
      clk => clk,
      rst => rst,
      result => cmpf0_result,
      result_valid => cmpf0_result_valid,
      result_ready => cmpf0_result_ready,
      result_spec => cmpf0_result_spec
    );

  buffer2 : entity work.handshake_buffer_6(arch)
    port map(
      ins => fork1_outs_4,
      ins_valid => fork1_outs_4_valid,
      ins_ready => fork1_outs_4_ready,
      ins_spec => fork1_outs_4_spec,
      clk => clk,
      rst => rst,
      outs => buffer2_outs,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready,
      outs_spec => buffer2_outs_spec
    );

  cmpi0 : entity work.handshake_cmpi_0(arch)
    port map(
      lhs => buffer2_outs,
      lhs_valid => buffer2_outs_valid,
      lhs_ready => buffer2_outs_ready,
      lhs_spec => buffer2_outs_spec,
      rhs => extsi2_outs,
      rhs_valid => extsi2_outs_valid,
      rhs_ready => extsi2_outs_ready,
      rhs_spec => extsi2_outs_spec,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready,
      result_spec => cmpi0_result_spec
    );

  andi0 : entity work.handshake_andi_0(arch)
    port map(
      lhs => cmpi0_result,
      lhs_valid => cmpi0_result_valid,
      lhs_ready => cmpi0_result_ready,
      lhs_spec => cmpi0_result_spec,
      rhs => cmpf0_result,
      rhs_valid => cmpf0_result_valid,
      rhs_ready => cmpf0_result_ready,
      rhs_spec => cmpf0_result_spec,
      clk => clk,
      rst => rst,
      result => andi0_result,
      result_valid => andi0_result_valid,
      result_ready => andi0_result_ready,
      result_spec => andi0_result_spec
    );

  speculator0 : entity work.handshake_speculator_0(arch)
    port map(
      ins => andi0_result,
      ins_valid => andi0_result_valid,
      ins_ready => andi0_result_ready,
      ins_spec => andi0_result_spec,
      trigger_valid => fork5_outs_1_valid,
      trigger_ready => fork5_outs_1_ready,
      trigger_spec => fork5_outs_1_spec,
      clk => clk,
      rst => rst,
      outs => speculator0_outs,
      outs_valid => speculator0_outs_valid,
      outs_ready => speculator0_outs_ready,
      outs_spec => speculator0_outs_spec,
      ctrl_save => speculator0_ctrl_save,
      ctrl_save_valid => speculator0_ctrl_save_valid,
      ctrl_save_ready => speculator0_ctrl_save_ready,
      ctrl_commit => speculator0_ctrl_commit,
      ctrl_commit_valid => speculator0_ctrl_commit_valid,
      ctrl_commit_ready => speculator0_ctrl_commit_ready,
      ctrl_sc_save => speculator0_ctrl_sc_save,
      ctrl_sc_save_valid => speculator0_ctrl_sc_save_valid,
      ctrl_sc_save_ready => speculator0_ctrl_sc_save_ready,
      ctrl_sc_commit => speculator0_ctrl_sc_commit,
      ctrl_sc_commit_valid => speculator0_ctrl_sc_commit_valid,
      ctrl_sc_commit_ready => speculator0_ctrl_sc_commit_ready,
      ctrl_sc_branch => speculator0_ctrl_sc_branch,
      ctrl_sc_branch_valid => speculator0_ctrl_sc_branch_valid,
      ctrl_sc_branch_ready => speculator0_ctrl_sc_branch_ready
    );

  cond_br5 : entity work.handshake_cond_br_0(arch)
    port map(
      condition => cond_br4_trueOut,
      condition_valid => cond_br4_trueOut_valid,
      condition_ready => cond_br4_trueOut_ready,
      data => speculator0_ctrl_sc_commit,
      data_valid => speculator0_ctrl_sc_commit_valid,
      data_ready => speculator0_ctrl_sc_commit_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br5_trueOut,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut => cond_br5_falseOut,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  sink7 : entity work.handshake_sink_0(arch)
    port map(
      ins => cond_br5_falseOut,
      ins_valid => cond_br5_falseOut_valid,
      ins_ready => cond_br5_falseOut_ready,
      clk => clk,
      rst => rst
    );

  merge0 : entity work.handshake_merge_0(arch)
    port map(
      ins(0) => speculator0_ctrl_sc_save,
      ins(1) => cond_br5_trueOut,
      ins_valid(0) => speculator0_ctrl_sc_save_valid,
      ins_valid(1) => cond_br5_trueOut_valid,
      ins_ready(0) => speculator0_ctrl_sc_save_ready,
      ins_ready(1) => cond_br5_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => merge0_outs,
      outs_valid => merge0_outs_valid,
      outs_ready => merge0_outs_ready
    );

  fork6 : entity work.handshake_fork_4(arch)
    port map(
      ins => merge0_outs,
      ins_valid => merge0_outs_valid,
      ins_ready => merge0_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs(2) => fork6_outs_2,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_valid(2) => fork6_outs_2_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready,
      outs_ready(2) => fork6_outs_2_ready
    );

  fork7 : entity work.handshake_fork_5(arch)
    port map(
      ins => speculator0_outs,
      ins_valid => speculator0_outs_valid,
      ins_ready => speculator0_outs_ready,
      ins_spec => speculator0_outs_spec,
      clk => clk,
      rst => rst,
      outs(0) => fork7_outs_0,
      outs(1) => fork7_outs_1,
      outs(2) => fork7_outs_2,
      outs(3) => fork7_outs_3,
      outs(4) => fork7_outs_4,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_valid(2) => fork7_outs_2_valid,
      outs_valid(3) => fork7_outs_3_valid,
      outs_valid(4) => fork7_outs_4_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready,
      outs_ready(2) => fork7_outs_2_ready,
      outs_ready(3) => fork7_outs_3_ready,
      outs_ready(4) => fork7_outs_4_ready,
      outs_0_spec => fork7_outs_0_spec,
      outs_1_spec => fork7_outs_1_spec,
      outs_2_spec => fork7_outs_2_spec,
      outs_3_spec => fork7_outs_3_spec,
      outs_4_spec => fork7_outs_4_spec
    );

  sink2 : entity work.handshake_sink_1(arch)
    port map(
      ins => speculator0_ctrl_save,
      ins_valid => speculator0_ctrl_save_valid,
      ins_ready => speculator0_ctrl_save_ready,
      clk => clk,
      rst => rst
    );

  buffer13 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork9_outs_0,
      ins_valid => fork9_outs_0_valid,
      ins_ready => fork9_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  cond_br0 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => buffer13_outs,
      condition_valid => buffer13_outs_valid,
      condition_ready => buffer13_outs_ready,
      data => speculator0_ctrl_commit,
      data_valid => speculator0_ctrl_commit_valid,
      data_ready => speculator0_ctrl_commit_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br0_trueOut,
      trueOut_valid => cond_br0_trueOut_valid,
      trueOut_ready => cond_br0_trueOut_ready,
      falseOut => cond_br0_falseOut,
      falseOut_valid => cond_br0_falseOut_valid,
      falseOut_ready => cond_br0_falseOut_ready
    );

  fork8 : entity work.handshake_fork_6(arch)
    port map(
      ins => cond_br0_falseOut,
      ins_valid => cond_br0_falseOut_valid,
      ins_ready => cond_br0_falseOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork8_outs_0,
      outs(1) => fork8_outs_1,
      outs(2) => fork8_outs_2,
      outs(3) => fork8_outs_3,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_valid(2) => fork8_outs_2_valid,
      outs_valid(3) => fork8_outs_3_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready,
      outs_ready(2) => fork8_outs_2_ready,
      outs_ready(3) => fork8_outs_3_ready
    );

  sink3 : entity work.handshake_sink_1(arch)
    port map(
      ins => cond_br0_trueOut,
      ins_valid => cond_br0_trueOut_valid,
      ins_ready => cond_br0_trueOut_ready,
      clk => clk,
      rst => rst
    );

  speculating_branch0 : entity work.handshake_speculating_branch_0(arch)
    port map(
      spec_tag_data => fork7_outs_0,
      spec_tag_data_valid => fork7_outs_0_valid,
      spec_tag_data_ready => fork7_outs_0_ready,
      spec_tag_data_spec => fork7_outs_0_spec,
      data => fork7_outs_3,
      data_valid => fork7_outs_3_valid,
      data_ready => fork7_outs_3_ready,
      data_spec => fork7_outs_3_spec,
      clk => clk,
      rst => rst,
      trueOut => speculating_branch0_trueOut,
      trueOut_valid => speculating_branch0_trueOut_valid,
      trueOut_ready => speculating_branch0_trueOut_ready,
      falseOut => speculating_branch0_falseOut,
      falseOut_valid => speculating_branch0_falseOut_valid,
      falseOut_ready => speculating_branch0_falseOut_ready
    );

  sink4 : entity work.handshake_sink_1(arch)
    port map(
      ins => speculating_branch0_falseOut,
      ins_valid => speculating_branch0_falseOut_valid,
      ins_ready => speculating_branch0_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer5 : entity work.handshake_buffer_7(arch)
    port map(
      ins => speculating_branch0_trueOut,
      ins_valid => speculating_branch0_trueOut_valid,
      ins_ready => speculating_branch0_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer5_outs,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  fork9 : entity work.handshake_fork_7(arch)
    port map(
      ins => buffer5_outs,
      ins_valid => buffer5_outs_valid,
      ins_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready
    );

  buffer14 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork9_outs_1,
      ins_valid => fork9_outs_1_valid,
      ins_ready => fork9_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  cond_br4 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => speculator0_ctrl_sc_branch,
      condition_valid => speculator0_ctrl_sc_branch_valid,
      condition_ready => speculator0_ctrl_sc_branch_ready,
      data => buffer14_outs,
      data_valid => buffer14_outs_valid,
      data_ready => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br4_trueOut,
      trueOut_valid => cond_br4_trueOut_valid,
      trueOut_ready => cond_br4_trueOut_ready,
      falseOut => cond_br4_falseOut,
      falseOut_valid => cond_br4_falseOut_valid,
      falseOut_ready => cond_br4_falseOut_ready
    );

  sink5 : entity work.handshake_sink_1(arch)
    port map(
      ins => cond_br4_falseOut,
      ins_valid => cond_br4_falseOut_valid,
      ins_ready => cond_br4_falseOut_ready,
      clk => clk,
      rst => rst
    );

  spec_save_commit0 : entity work.handshake_spec_save_commit_0(arch)
    port map(
      ins => addi1_result,
      ins_valid => addi1_result_valid,
      ins_ready => addi1_result_ready,
      ins_spec => addi1_result_spec,
      ctrl => fork6_outs_2,
      ctrl_valid => fork6_outs_2_valid,
      ctrl_ready => fork6_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => spec_save_commit0_outs,
      outs_valid => spec_save_commit0_outs_valid,
      outs_ready => spec_save_commit0_outs_ready,
      outs_spec => spec_save_commit0_outs_spec
    );

  cond_br1 : entity work.handshake_cond_br_2(arch)
    port map(
      condition => fork7_outs_4,
      condition_valid => fork7_outs_4_valid,
      condition_ready => fork7_outs_4_ready,
      condition_spec => fork7_outs_4_spec,
      data => spec_save_commit0_outs,
      data_valid => spec_save_commit0_outs_valid,
      data_ready => spec_save_commit0_outs_ready,
      data_spec => spec_save_commit0_outs_spec,
      clk => clk,
      rst => rst,
      trueOut => cond_br1_trueOut,
      trueOut_valid => cond_br1_trueOut_valid,
      trueOut_ready => cond_br1_trueOut_ready,
      trueOut_spec => cond_br1_trueOut_spec,
      falseOut => cond_br1_falseOut,
      falseOut_valid => cond_br1_falseOut_valid,
      falseOut_ready => cond_br1_falseOut_ready,
      falseOut_spec => cond_br1_falseOut_spec
    );

  sink0 : entity work.handshake_sink_2(arch)
    port map(
      ins => cond_br1_falseOut,
      ins_valid => cond_br1_falseOut_valid,
      ins_ready => cond_br1_falseOut_ready,
      ins_spec => cond_br1_falseOut_spec,
      clk => clk,
      rst => rst
    );

  spec_save_commit1 : entity work.handshake_spec_save_commit_1(arch)
    port map(
      ins_valid => fork5_outs_0_valid,
      ins_ready => fork5_outs_0_ready,
      ins_spec => fork5_outs_0_spec,
      ctrl => fork6_outs_1,
      ctrl_valid => fork6_outs_1_valid,
      ctrl_ready => fork6_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => spec_save_commit1_outs_valid,
      outs_ready => spec_save_commit1_outs_ready,
      outs_spec => spec_save_commit1_outs_spec
    );

  cond_br2 : entity work.handshake_cond_br_3(arch)
    port map(
      condition => fork7_outs_2,
      condition_valid => fork7_outs_2_valid,
      condition_ready => fork7_outs_2_ready,
      condition_spec => fork7_outs_2_spec,
      data_valid => spec_save_commit1_outs_valid,
      data_ready => spec_save_commit1_outs_ready,
      data_spec => spec_save_commit1_outs_spec,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br2_trueOut_valid,
      trueOut_ready => cond_br2_trueOut_ready,
      trueOut_spec => cond_br2_trueOut_spec,
      falseOut_valid => cond_br2_falseOut_valid,
      falseOut_ready => cond_br2_falseOut_ready,
      falseOut_spec => cond_br2_falseOut_spec
    );

  spec_save_commit2 : entity work.handshake_spec_save_commit_0(arch)
    port map(
      ins => fork1_outs_3,
      ins_valid => fork1_outs_3_valid,
      ins_ready => fork1_outs_3_ready,
      ins_spec => fork1_outs_3_spec,
      ctrl => fork6_outs_0,
      ctrl_valid => fork6_outs_0_valid,
      ctrl_ready => fork6_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => spec_save_commit2_outs,
      outs_valid => spec_save_commit2_outs_valid,
      outs_ready => spec_save_commit2_outs_ready,
      outs_spec => spec_save_commit2_outs_spec
    );

  cond_br3 : entity work.handshake_cond_br_2(arch)
    port map(
      condition => fork7_outs_1,
      condition_valid => fork7_outs_1_valid,
      condition_ready => fork7_outs_1_ready,
      condition_spec => fork7_outs_1_spec,
      data => spec_save_commit2_outs,
      data_valid => spec_save_commit2_outs_valid,
      data_ready => spec_save_commit2_outs_ready,
      data_spec => spec_save_commit2_outs_spec,
      clk => clk,
      rst => rst,
      trueOut => cond_br3_trueOut,
      trueOut_valid => cond_br3_trueOut_valid,
      trueOut_ready => cond_br3_trueOut_ready,
      trueOut_spec => cond_br3_trueOut_spec,
      falseOut => cond_br3_falseOut,
      falseOut_valid => cond_br3_falseOut_valid,
      falseOut_ready => cond_br3_falseOut_ready,
      falseOut_spec => cond_br3_falseOut_spec
    );

  sink1 : entity work.handshake_sink_2(arch)
    port map(
      ins => cond_br3_trueOut,
      ins_valid => cond_br3_trueOut_valid,
      ins_ready => cond_br3_trueOut_ready,
      ins_spec => cond_br3_trueOut_spec,
      clk => clk,
      rst => rst
    );

  fork4 : entity work.handshake_fork_8(arch)
    port map(
      ins_valid => cond_br2_falseOut_valid,
      ins_ready => cond_br2_falseOut_ready,
      ins_spec => cond_br2_falseOut_spec,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_valid(2) => fork4_outs_2_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready,
      outs_ready(2) => fork4_outs_2_ready,
      outs_0_spec => fork4_outs_0_spec,
      outs_1_spec => fork4_outs_1_spec,
      outs_2_spec => fork4_outs_2_spec
    );

  buffer6 : entity work.handshake_buffer_6(arch)
    port map(
      ins => cond_br3_falseOut,
      ins_valid => cond_br3_falseOut_valid,
      ins_ready => cond_br3_falseOut_ready,
      ins_spec => cond_br3_falseOut_spec,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready,
      outs_spec => buffer6_outs_spec
    );

  buffer15 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork8_outs_3,
      ins_valid => fork8_outs_3_valid,
      ins_ready => fork8_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer15_outs,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  buffer16 : entity work.handshake_buffer_8(arch)
    port map(
      ins => buffer6_outs,
      ins_valid => buffer6_outs_valid,
      ins_ready => buffer6_outs_ready,
      ins_spec => buffer6_outs_spec,
      clk => clk,
      rst => rst,
      outs => buffer16_outs,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready,
      outs_spec => buffer16_outs_spec
    );

  spec_commit3 : entity work.handshake_spec_commit_1(arch)
    port map(
      ins => buffer16_outs,
      ins_valid => buffer16_outs_valid,
      ins_ready => buffer16_outs_ready,
      ins_spec => buffer16_outs_spec,
      ctrl => buffer15_outs,
      ctrl_valid => buffer15_outs_valid,
      ctrl_ready => buffer15_outs_ready,
      clk => clk,
      rst => rst,
      outs => spec_commit3_outs,
      outs_valid => spec_commit3_outs_valid,
      outs_ready => spec_commit3_outs_ready
    );

end architecture;
