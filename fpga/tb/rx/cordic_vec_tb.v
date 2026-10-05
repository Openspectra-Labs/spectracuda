// Unit test for cordic_vec: feeds the vectors from
// gen_cordic_vectors.py and dumps the angle it produced. Scoring is in
// check_cordic.py against numpy's arctan2 -- the block is compared to
// the reference, never to a second Verilog implementation of the same
// formula.
`timescale 1ns / 1ps
`include "build/cordic_params.vh"

module cordic_vec_tb;

    localparam integer DATA_W  = 20;
    localparam integer ANGLE_W = 16;
    localparam integer STAGES  = 16;
    localparam integer N       = `CORDIC_N_VEC;

    reg clk = 0;
    reg rst = 1;
    always #5 clk = ~clk;

    reg  signed [DATA_W-1:0]  in_x = 0, in_y = 0;
    reg                       in_valid = 0;
    wire signed [ANGLE_W-1:0] out_angle;
    wire                      out_valid;

    cordic_vec #(.DATA_W(DATA_W), .ANGLE_W(ANGLE_W), .STAGES(STAGES)) dut (
        .clk(clk), .rst(rst),
        .in_x(in_x), .in_y(in_y), .in_valid(in_valid),
        .out_angle(out_angle), .out_valid(out_valid)
    );

    reg [2*DATA_W-1:0] stim [0:N-1];
    integer fd, i, got = 0;

    always @(posedge clk) begin
        if (!rst && out_valid) begin
            $fwrite(fd, "%0d\n", $signed(out_angle));
            got = got + 1;
        end
    end

    initial begin
        $readmemh("build/cordic_stim.hex", stim);
        fd = $fopen("build/cordic_out.txt", "w");
        if (fd == 0) begin
            $display("FAIL: cannot open build/cordic_out.txt");
            $finish;
        end

        repeat (4) @(posedge clk);
        rst = 0;
        @(posedge clk);

        for (i = 0; i < N; i = i + 1) begin
            @(negedge clk);
            in_x     = stim[i][2*DATA_W-1:DATA_W];
            in_y     = stim[i][DATA_W-1:0];
            in_valid = 1'b1;
        end
        @(negedge clk);
        in_valid = 1'b0;

        // Drain the pipeline.
        repeat (STAGES + 8) @(posedge clk);
        $fclose(fd);
        $display("cordic_vec_tb: fed %0d vectors, got %0d angles", N, got);
        $finish;
    end
endmodule
