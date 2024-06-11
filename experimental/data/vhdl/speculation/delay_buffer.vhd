Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;

entity delay_buffer is
generic (
  SIZE: integer := 32
);
port(
  clk, rst : in std_logic; 
  valid_in : in std_logic;
  ready_in : in std_logic;
  valid_out : out std_logic);
end entity;

architecture arch of delay_buffer is

    type mem is array (SIZE - 1 downto 0) of std_logic;
    signal regs : mem;

begin 

    gen_assignements : for i in 0 to SIZE - 1 generate 
        first_assignment : if i = 0 generate 
            process (clk) begin
                if rising_edge(clk) then
                    if (ready_in = '1' or rst = '1') then
                    	regs(i) <= valid_in;
		    end if;
                end if;
            end process;
        end generate first_assignment;

        other_assignments : if i > 0 generate
            process (clk) begin
                if rising_edge(clk) then
                    if (rst = '1') then
                        regs(i) <= '0';
                    elsif (ready_in = '1') then
                        regs(i) <= regs(i - 1);
                    end if;
                end if;
            end process;
        end generate other_assignments;

    end generate gen_assignements;

    valid_out <= regs(SIZE - 1);

end architecture;
