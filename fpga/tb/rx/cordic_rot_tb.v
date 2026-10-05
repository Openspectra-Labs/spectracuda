// Unit test for cordic_rot. Scoring is in check_rot.py against numpy.
`timescale 1ns / 1ps
`include "build/rot_params.vh"

module cordic_rot_tb;
    localparam integer DATA_W = 16, ANGLE_W = 16, STAGES = 16;
    localparam integer N = `ROT_N_VEC;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg  signed [DATA_W-1:0]  in_x = 0, in_y = 0;
    reg  signed [ANGLE_W-1:0] in_z = 0;
    reg                       in_valid = 0;
    wire signed [DATA_W-1:0]  out_x, out_y;
    wire                      out_valid;

    cordic_rot #(.DATA_W(DATA_W), .ANGLE_W(ANGLE_W), .STAGES(STAGES)) dut (
        .clk(clk), .rst(rst), .in_x(in_x), .in_y(in_y), .in_z(in_z),
        .in_valid(in_valid), .out_x(out_x), .out_y(out_y), .out_valid(out_valid)
    );

    reg [47:0] stim [0:N-1];
    integer fd, i, got = 0;

    always @(posedge clk) begin
        if (!rst && out_valid) begin
            $fwrite(fd, "%0d %0d\n", $signed(out_x), $signed(out_y));
            got = got + 1;
        end
    end

    initial begin
        $readmemh("build/rot_stim.hex", stim);
        fd = $fopen("build/rot_out.txt", "w");
        if (fd == 0) begin $display("FAIL: cannot open rot_out.txt"); $finish; end
        repeat (4) @(posedge clk); rst = 0; @(posedge clk);
        for (i = 0; i < N; i = i + 1) begin
            @(negedge clk);
            in_x = stim[i][47:32]; in_y = stim[i][31:16]; in_z = stim[i][15:0];
            in_valid = 1'b1;
        end
        @(negedge clk); in_valid = 1'b0;
        repeat (STAGES + 8) @(posedge clk);
        $fclose(fd);
        $display("cordic_rot_tb: fed %0d vectors, got %0d results", N, got);
        $finish;
    end
endmodule
