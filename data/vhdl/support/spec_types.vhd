library ieee;                                                                                                                                                                                             
  use ieee.std_logic_1164.all;                                                                                                                                                                              
                                                                                                                                                                                                            
  package spec_types is                                                                                                                                                                                     
    type State_type is (IDLE, KILL, KILL_ONLY_DATA);                                                                                                                                                        
    type Control_type is (CONTROL_SPEC, CONTROL_NO_CMP, CONTROL_CMP_CORRECT, CONTROL_RESEND, CONTROL_KILL, CONTROL_CORRECT_SPEC);                                                                           
    function to_slv(ctrl : Control_type) return std_logic_vector;                                                                                                                                           
    function from_slv(slv : std_logic_vector(2 downto 0)) return Control_type;                                                                                                                              
  end package;                                                                                                                                                                                              
                  
  package body spec_types is                                                                                                                                                                                
    function to_slv(ctrl : Control_type) return std_logic_vector is
    begin                                                                                                                                                                                                   
      case ctrl is
        when CONTROL_SPEC         => return "000";                                                                                                                                                          
        when CONTROL_NO_CMP       => return "001";
        when CONTROL_CMP_CORRECT  => return "010";
        when CONTROL_RESEND       => return "011";                                                                                                                                                          
        when CONTROL_KILL         => return "100";
        when CONTROL_CORRECT_SPEC => return "101";                                                                                                                                                          
      end case;   
    end function;
                                                                                                                                                                                                            
    function from_slv(slv : std_logic_vector(2 downto 0)) return Control_type is
    begin                                                                                                                                                                                                   
      case slv is 
        when "000"  => return CONTROL_SPEC;
        when "001"  => return CONTROL_NO_CMP;
        when "010"  => return CONTROL_CMP_CORRECT;                                                                                                                                                          
        when "011"  => return CONTROL_RESEND;
        when "100"  => return CONTROL_KILL;                                                                                                                                                                 
        when "101"  => return CONTROL_CORRECT_SPEC;
        when others => return CONTROL_SPEC;                                                                                                                                                                 
      end case;
    end function;                                                                                                                                                                                           
  end package body;
