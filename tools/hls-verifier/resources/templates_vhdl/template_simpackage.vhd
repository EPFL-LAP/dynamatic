library ieee;
use ieee.std_logic_arith.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use ieee.std_logic_textio.all;
use std.textio.all;

---------------------------------------------------------------------------
package sim_package is
	-- Procedures to read tokens from input textfile
	procedure read_token(file filename	: text;
	                          line_num	: inout line;
	                          token 	: out string);

	-- Functions for type conversions
	function hex_str_to_logicVec(RHS : string; data_width : integer) 
			return std_logic_vector;
	function hex_logicVec_to_str(logicVec : std_logic_vector) 
			return string;

end package;

---------------------------------------------------------------------------
-- Main body of the simulation package ------------------------------------
package body sim_package is

---------------------------------------------------------------------------
	-- Procedure to read tokens from input textfile
	procedure read_token(file filename	: text;
	                          line_num	: inout line;
	                          token 	: out string) is
		-- Internal signals
		variable i 			: integer;
		variable end_flag 	: boolean;
		variable whitespace : character;
		variable buffr 		: string(1 to token'length);
	begin
		-- Initialization
		i 		 := 1;
		end_flag := false;

		-- Main loop to parse through the textfile
		main_loop: while not endfile(filename) loop

			-- Read next line if end of line
			if line_num = null or line_num'length = 0 then
				readline(filename, line_num);
			end if;
			-- Inner loop to remove whitespaces
			rm_whitespace: while line_num'length > 0 loop
				if  line_num(line_num'left) = LF or line_num(line_num'left) = HT or
					line_num(line_num'left) = CR or line_num(line_num'left) = ' ' then
				    read(line_num, whitespace);
				else
					exit rm_whitespace;
				end if;
			end loop;
			-- Inner loop to read tokens
			rd_tokens: while line_num'length > 0 and i <= buffr'length loop
				if  line_num(line_num'left) = LF or line_num(line_num'left) = HT or
					line_num(line_num'left) = CR or line_num(line_num'left) = ' ' then
				    exit rd_tokens;
				else
					read(line_num, buffr(i));
					i := i + 1;
				end if;
				end_flag := true;
			end loop;
			-- Check loop exit condition
			if end_flag = true then
				exit main_loop;
			end if;
		end loop;

		-- Assign output token (note, token length = i - 1)
		buffr(i)  := ' ';
		token 	  := buffr;

	end procedure read_token;

---------------------------------------------------------------------------
	-- Function to convert hexadecimal string value to std_logic_vector
	function hex_str_to_logicVec(RHS : string; data_width : integer) 
			return std_logic_vector is
		-- Internal signals
		variable idx 	   : integer;
		variable ext_width : integer := data_width + 4;
		variable buffr	   : std_logic_vector(ext_width - 1 downto 0);
		variable check	   : std_logic_vector(ext_width - 1 downto 0);
		variable ret 	   : std_logic_vector(data_width - 1 downto 0);  
	begin
		-- Initialization
		buffr := (others => '0');
		idx   := 3;
		-- Input string value starts from index 3, after '0x....'
		if (RHS(1) /= '0' and (RHS(2) /= 'x' or RHS(2) /= 'X')) then
			report "ERROR: The input string must support the hexadecimal format (0x....)";
		end if;

		-- Main loop to parse through the input string
		main_loop: while true loop
			-- New stuff here
			case RHS(idx) is
				when '0'     => buffr := buffr(ext_width - 5 downto 0) & "0000";
				when '1'     => buffr := buffr(ext_width - 5 downto 0) & "0001";
				when '2'     => buffr := buffr(ext_width - 5 downto 0) & "0010";
				when '3'     => buffr := buffr(ext_width - 5 downto 0) & "0011";
				when '4'     => buffr := buffr(ext_width - 5 downto 0) & "0100";
				when '5'     => buffr := buffr(ext_width - 5 downto 0) & "0101";
				when '6'     => buffr := buffr(ext_width - 5 downto 0) & "0110";
				when '7'     => buffr := buffr(ext_width - 5 downto 0) & "0111";
				when '8'     => buffr := buffr(ext_width - 5 downto 0) & "1000";
				when '9'     => buffr := buffr(ext_width - 5 downto 0) & "1001";
				when 'a'|'A' => buffr := buffr(ext_width - 5 downto 0) & "1010";
				when 'b'|'B' => buffr := buffr(ext_width - 5 downto 0) & "1011";
				when 'c'|'C' => buffr := buffr(ext_width - 5 downto 0) & "1100";
				when 'd'|'D' => buffr := buffr(ext_width - 5 downto 0) & "1101";
				when 'e'|'E' => buffr := buffr(ext_width - 5 downto 0) & "1110";
				when 'f'|'F' => buffr := buffr(ext_width - 5 downto 0) & "1111";
				when 'x'|'X' => buffr := buffr(ext_width - 5 downto 0) & "XXXX";
				when ' '     => exit main_loop;
				when others  => report "ERROR: Wrong hex char " & RHS(idx) & " at idx " & integer'image(idx);
								exit main_loop;
			end case;

			-- Increment index
			idx := idx + 1;
		end loop;

		-- Initialize output variable
		check := (others => '0');
		ret   := (others => '0');

		-- Sanity check for small data_width (lte 4)
		if (data_width <= 4) then
			check(data_width - 1 downto 0) := buffr(data_width - 1 downto 0);

			-- Check zero-padded value vs full converted value 
			if (check /= buffr) then 
				report "ERROR: Unknown error in coversion using small data_width";
				return ret;
			end if;
		end if;

		-- Take the lowest data_width bits
		ret := buffr(data_width - 1 downto 0);

		-- Return converted std_logic_vector
		return ret;
	end function hex_str_to_logicVec;

---------------------------------------------------------------------------
	-- Function to convert hexadecimal std_logic_vector to string value
	function hex_logicVec_to_str(logicVec : std_logic_vector) 
			return string is
		-- Internal signals
		constant inp_len : integer := logicVec'length;
		constant str_len : integer := (inp_len + 3) / 4;
		variable conv	 : string(1 to str_len); 
		variable ret 	 : string (conv'range);

	begin
		-- Use in-built function (won't report errors)
		conv := to_hex_string(logicVec);	

		-- Convert string to lower case (alternatively change token compare function)
		for i in conv'range loop
			case (conv(i)) is
				when 'A' 	=> ret(i) := 'a';
				when 'B' 	=> ret(i) := 'b';
				when 'C' 	=> ret(i) := 'c';
				when 'D' 	=> ret(i) := 'd';
				when 'E' 	=> ret(i) := 'e';
				when 'F' 	=> ret(i) := 'f';
				when others => ret(i) := conv(i);
			end case;
     	end loop;

		-- Return the converted string
		return ret;
	end function hex_logicVec_to_str;

---------------------------------------------------------------------------
end package body;
-- End of code
