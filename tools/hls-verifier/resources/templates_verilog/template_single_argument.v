
`timescale 1 ns / 1 ps

module single_argument (
    clk,
    rst,
    ce0,
    we0,
    din0,
    dout0,
    dout0_ready,
    dout0_valid,
    done
);


//------------------------Local signal-------------------
parameter TV_IN = "";
parameter TV_OUT = "";
parameter DATA_WIDTH = 32'd 32;

// Input and Output
input clk;
input rst;
input ce0;
input we0;
input [DATA_WIDTH-1:0] din0;
output reg [DATA_WIDTH -1 : 0] dout0;
input dout0_ready;
output reg dout0_valid;
input done;


// Inner signals
reg [DATA_WIDTH - 1 : 0] mem;
reg tokenEmitted;

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

    if(TV_IN != "")begin

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
            read_token(fp,token);
            ret = $sscanf(token, "0x%x", mem_tmp);
            mem = mem_tmp;
            if (ret != 1) begin
                $display("Failed to parse token!");
                $finish;
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
end

// Read data from array to RTL
always @ (posedge clk or posedge rst) begin
    if(rst) begin
        tokenEmitted <= 1'b0;
        dout0 <= {DATA_WIDTH{1'b0}};
        dout0_valid <= 1'b0;
    end else  begin
	    if(!(tokenEmitted)) begin
            tokenEmitted <= 1'b1;
	        dout0 <= mem;
            dout0_valid <= 1'b1;
        end else begin
            if (dout0_ready)
                dout0_valid <=1'b0;
        end
    end
end


//------------------------Write array-------------------

// Write data from RTL to array
always @ (posedge clk) begin
    if((we0 == 1) && (ce0 == 1)) begin
        mem <= din0;
        $display("din0: %b",din0);
    end
end


// Write data from array to file
initial begin : write_file_proc
    integer fp;
    integer transaction_num;
    reg [ 8*5 : 1] str;
    integer i;
    transaction_num = 0;
    writed_flag = 1;

    if(TV_OUT !="") begin
        wait(rst === 0);

        //add the initialization step from single_argument.vhd to skip the first '1' from 'done'

        while(done == 0) begin
                -> write_process_done;
                @(negedge clk);
        end

        wait(done === 0);

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
            $display("mem: %b", mem);
            $fdisplay(fp,"0x%x",mem);
            $fdisplay(fp, "[[/transaction]]");
            transaction_num = transaction_num + 1;
            $fclose(fp);
            writed_flag = 1;
            -> write_process_done;
            @(negedge clk);
        end
    end
end


endmodule
