
`timescale 1 ns / 1 ps

module two_port_RAM (
    clk,
    rst,
    ce0,
    we0,
    address0,
    din0,
    dout0,
    ce1,
    we1,
    address1,
    din1,
    dout1,
    done
);

//------------------------Local signal-------------------
parameter TV_IN  = "";
parameter TV_OUT = "";
parameter DATA_WIDTH = 32'd 32;
parameter ADDR_WIDTH = 32'd 10;
parameter DEPTH = 32'd 1000;
parameter DLY  = 0.1;

// Input and Output
input clk;
input rst;
input ce0, ce1;
input we0, we1;
input [ADDR_WIDTH - 1 : 0] address0, address1;
input [DATA_WIDTH - 1 : 0] din0, din1;
output reg [DATA_WIDTH - 1 : 0] dout0, dout1;
input done;

// Inner signals
reg [DATA_WIDTH - 1 : 0] mem [0 : DEPTH - 1];
initial begin : initialize_mem
    integer i;
    for (i = 0; i < DEPTH; i = i + 1) begin
        mem[i] = 0;
    end
end
reg writed_flag;
event write_process_done;
//------------------------Task and function--------------
task read_token;
    input integer fp;
    output reg [127 :0] token;
    integer ret;
    begin
        token = "";
        ret = 0;
        ret = $fscanf(fp,"%s",token);
    end
endtask

//------------------------Read array-------------------

// Read data form file to array
initial begin : read_file_process
    integer fp;
    integer err;
    integer ret;
    reg [127 : 0] token;
    reg [ 8*5 : 1] str;
    reg [ DATA_WIDTH - 1 : 0 ] mem_tmp;
    integer transaction_idx;
    integer i;
    transaction_idx = 0;

    wait(rst === 0);
    @(write_process_done);
    fp = $fopen(TV_IN,"r");
    if(fp == 0) begin       // Failed to open file
        $display("Failed to open file \"%s\"!", TV_IN);
        $finish;
    end
    read_token(fp, token);
    if (token != "[[[runtime]]]") begin             // Illegal format
        $display("ERROR: Simulation using HLS TB failed.");
        $finish;
    end
    read_token(fp, token);
    while (token != "[[[/runtime]]]") begin
        if (token != "[[transaction]]") begin
            $display("ERROR: Simulation using HLS TB failed.");
		  $finish;
        end
        read_token(fp, token);              // skip transaction number
        for(i = 0; i < DEPTH; i = i + 1) begin
            read_token(fp, token);
            ret = $sscanf(token, "0x%x", mem_tmp);
            mem[i] = mem_tmp;
            if (ret != 1) begin
                $display("Failed to parse token!");
                $finish;
            end
        end
        @(write_process_done);
        read_token(fp, token);
        if(token != "[[/transaction]]") begin
            $display("ERROR: Simulation using HLS TB failed.");
            $finish;
        end
        read_token(fp, token);
        transaction_idx = transaction_idx + 1;
    end
    $fclose(fp);
end

// Read data from array to RTL
always @ (posedge clk or rst) begin
    if(rst === 1) begin
        dout0 <= 0;
    end
    else begin
	  if((we0 == 0) && (ce0 == 1) && (ce1 == 1) && (we1 == 1) && (address0 == address1))
	      dout0 <= #DLY din1;
	  else if(ce0 == 1)
	      dout0 <= #DLY mem[address0];
        else ;
    end
end

always @ (posedge clk or rst) begin
    if(rst === 1) begin
        dout1 <= 0;
    end
    else begin
	  if((we0 == 1) && (ce0 == 1) && (ce1 == 1) && (we1 == 0) && (address0 == address1))
            dout1 <= #DLY din0;
	  else if(ce1 == 1)
            dout1 <= #DLY mem[address1];
        else ;
    end
end

//------------------------Write array-------------------

// Write data from RTL to array
always @ (posedge clk) begin
    if((we0 == 1) && (ce0 == 1) && (ce1 == 1) && (we1 == 1) && (address0 == address1))
        mem[address0] <= #DLY din1;
    else if ((we0 == 1) && (ce0 == 1))
        mem[address0] <= #DLY din0;
end

always @ (posedge clk) begin
    if((ce1 == 1) && (we1 == 1))
        mem[address1] <= #DLY din1;
end

// Write data from array to file
initial begin : write_file_proc
    integer fp;
    integer transaction_num;
    reg [ 8*5 : 1] str;
    integer i;
    transaction_num = 0;
    writed_flag = 1;
    wait(rst === 0);
    @(negedge clk);
    while(1) begin
        while(done == 0) begin
            -> write_process_done;
            @(negedge clk);
        end
        fp = $fopen(TV_OUT, "a");
        if(fp == 0) begin       // Failed to open file
            $display("Failed to open file \"%s\"!", TV_OUT);
            $finish;
        end
        $fdisplay(fp, "[[transaction]] %d", transaction_num);
	      for (i = 0; i < DEPTH; i = i + 1) begin
            $fdisplay(fp,"0x%x",mem[i]);
        end
        $fdisplay(fp, "[[/transaction]]");
        transaction_num = transaction_num + 1;
        $fclose(fp);
        writed_flag = 1;
        -> write_process_done;
        @(negedge clk);
    end
end

//------------------------conflict check-------------------
always @ (posedge clk) begin
    if ((we0 == 1) && (ce0 == 1) && (ce1 == 1) && (we1 == 1) && (address0 == address1))
        $display($time,"WARNING:write conflict----port0 and port1 write to the same address:%h at the same clock. Port1 has the high priority.",address0);
end

always @ (posedge clk) begin
    if ((we0 == 1) && (ce0 == 1) && (ce1 == 1) && (we1 == 0) && (address0 == address1))
        $display($time,"NOTE:read & write conflict----port0 write and port1 read to the same address:%h at the same clock. Write first Mode.",address0);
end

always @ (posedge clk) begin
    if ((we0 == 0) && (ce0 == 1) && (ce1 == 1) && (we1 == 1) && (address0 == address1))
        $display($time,"NOTE:read & write conflict----port0 read and port1 write to the same address:%h at the same clock. Write first Mode.",address0);
end

endmodule
