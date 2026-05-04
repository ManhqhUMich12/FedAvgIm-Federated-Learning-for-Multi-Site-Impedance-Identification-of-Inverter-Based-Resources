clear; clc; close all;

%% =========================
% Fixed parameters
% ==========================
p.Tsam = 1e-5;
% Circuit and controller parameters
f0 = 50;
w1 = 100*pi;
p.Vdc = 1150;
p.Vg  = 575;
p.w1  = 100*pi;
p.fs  = 5e3;
p.Ts  = 1/p.fs;
p.Td  = 1.5*p.Ts;
p.fsw = 5e3;

% LCL filter
p.Rf1 = 3e-3;
p.Lf1 = 250e-6;
p.Rf2 = 3e-3;
p.Lf2 = 250e-6;
p.Cf  = 50e-6;

% Current controller
p.Kpi = 1.7391e-4;
p.Kii = 0.0348;

% PLL
p.Kppll = 40/(p.Vdc/2);
p.Kipll = 400/(p.Vdc/2);
p.beta  = 0;

% Grid
p.Lg = 250e-6;
p.Rg = 3e-3;

%% =========================
% Simulink model
% ==========================
mdl = "GFLI.slx";
open_system(mdl);

axisBlk = 'GFLI/AISTool/AxisSlt';

%% =========================
% Frequency vectors
% ==========================
fHz_meas = logspace(log10(1), log10(200), 20);
w_meas   = 2*pi*fHz_meas;

fHz_ana = logspace(log10(1), log10(200),600);
w_ana   = 2*pi*fHz_ana;

%% =========================
% Run 2 operating points
% ==========================
op1 = run_one_op(mdl, axisBlk, 3195, 0,    p, fHz_meas, w_meas, fHz_ana, w_ana);
op2 = run_one_op(mdl, axisBlk, 1095, 0,    p, fHz_meas, w_meas, fHz_ana, w_ana);

%% =========================
% Common style settings
% =========================
blue_col   = [0.00 0.45 0.74];   % xanh dương
orange_col = [0.85 0.33 0.10];   % cam tối

lw_ana   = 2.0;
lw_meas  = 1.6;
ms_meas  = 6;

fig_pos = [100 100 600 400];   % [left bottom width height]

font_name     = 'Times New Roman';
font_size_ax  = 11;
font_size_lb  = 11;
font_size_ti  = 11;
font_size_leg = 10;

leg_text = { ...
    'Analytical OP1 (I_d=3195, I_q=0)', ...
    'Measured OP1 (I_d=3195, I_q=0)', ...
    'Analytical OP2 (I_d=1095, I_q=0)', ...
    'Measured OP2 (I_d=1095, I_q=0)'};

%% =========================
% Create 2x2 tiled figure
% Top row   : Magnitude
% Bottom row: Phase
% =========================
fig = figure;
set(fig, 'Position', fig_pos);

t = tiledlayout(2,2, 'Padding','compact', 'TileSpacing','compact');

%% =========================
% Top-left: Ydd Magnitude
% =========================
ax1 = nexttile;
semilogx(op1.fHz_ana, 20*log10(abs(op1.Ydd_ana)), ...
    '-', 'Color', blue_col, 'LineWidth', lw_ana); hold on;

semilogx(op1.fHz_meas, 20*log10(abs(op1.Ydd_meas)), ...
    'o', 'Color', blue_col, 'LineWidth', lw_meas, ...
    'MarkerSize', ms_meas, 'MarkerFaceColor', 'none');

semilogx(op2.fHz_ana, 20*log10(abs(op2.Ydd_ana)), ...
    '--', 'Color', orange_col, 'LineWidth', lw_ana);

semilogx(op2.fHz_meas, 20*log10(abs(op2.Ydd_meas)), ...
    's', 'Color', orange_col, 'LineWidth', lw_meas, ...
    'MarkerSize', ms_meas, 'MarkerFaceColor', 'none');

grid on;
box(ax1,'on');
xlim([1 1e4]);

title('Y_{dd}', 'FontName', font_name,'FontAngle','italic', 'FontSize', font_size_ti, 'FontWeight', 'normal');
ylabel('Magnitude (dB)', 'FontName', font_name, 'FontSize', font_size_lb);
ax1.FontName = font_name;
ax1.FontSize = font_size_ax;
ax1.LineWidth = 1.2;

%% =========================
% Top-right: Yqq Magnitude
% =========================
ax2 = nexttile; 
semilogx(op1.fHz_ana, 20*log10(abs(op1.Yqq_ana)), ...
    '-', 'Color', blue_col, 'LineWidth', lw_ana); hold on;

semilogx(op1.fHz_meas, 20*log10(abs(op1.Yqq_meas)), ...
    'o', 'Color', blue_col, 'LineWidth', lw_meas, ...
    'MarkerSize', ms_meas, 'MarkerFaceColor', 'none');

semilogx(op2.fHz_ana, 20*log10(abs(op2.Yqq_ana)), ...
    '--', 'Color', orange_col, 'LineWidth', lw_ana);

semilogx(op2.fHz_meas, 20*log10(abs(op2.Yqq_meas)), ...
    's', 'Color', orange_col, 'LineWidth', lw_meas, ...
    'MarkerSize', ms_meas, 'MarkerFaceColor', 'none');

grid on;
box(ax2,'on');
xlim([1 1e4]);
title('Y_{qq}', 'FontName', font_name,'FontAngle','italic', 'FontSize', font_size_ti, 'FontWeight', 'normal');
ax2.FontName = font_name;
ax2.FontSize = font_size_ax;
ax2.LineWidth = 1.2;

% lgd = legend(leg_text, 'Location', 'best', 'NumColumns', 2);
lgd.FontName = font_name;
lgd.FontSize = font_size_leg;

%% =========================
% Bottom-left: Ydd Phase
% =========================
ax3 = nexttile;
semilogx(op1.fHz_ana, wrapTo180(rad2deg(angle(op1.Ydd_ana))), ...
    '-', 'Color', blue_col, 'LineWidth', lw_ana); hold on;

semilogx(op1.fHz_meas, wrapTo180(rad2deg(angle(op1.Ydd_meas))), ...
    'o', 'Color', blue_col, 'LineWidth', lw_meas, ...
    'MarkerSize', ms_meas, 'MarkerFaceColor', 'none');

semilogx(op2.fHz_ana, wrapTo180(rad2deg(angle(op2.Ydd_ana))), ...
    '--', 'Color', orange_col, 'LineWidth', lw_ana);

semilogx(op2.fHz_meas, wrapTo180(rad2deg(angle(op2.Ydd_meas))), ...
    's', 'Color', orange_col, 'LineWidth', lw_meas, ...
    'MarkerSize', ms_meas, 'MarkerFaceColor', 'none');

grid on;
box(ax3,'on');
xlim([1 1e4]);
ylim([-180 180]);
% title('Y_{dd} Phase', 'FontName', font_name, 'FontSize', font_size_ti);
ylabel('Phase (deg)', 'FontName', font_name, 'FontSize', font_size_lb);
ax3.FontName = font_name;
ax3.FontSize = font_size_ax;
ax3.LineWidth = 1.2;

%% =========================
% Bottom-right: Yqq Phase
% =========================
ax4 = nexttile;
semilogx(op1.fHz_ana, wrapTo180(rad2deg(angle(op1.Yqq_ana))), ...
    '-', 'Color', blue_col, 'LineWidth', lw_ana); hold on;

semilogx(op1.fHz_meas, wrapTo180(rad2deg(angle(op1.Yqq_meas))), ...
    'o', 'Color', blue_col, 'LineWidth', lw_meas, ...
    'MarkerSize', ms_meas, 'MarkerFaceColor', 'none');

semilogx(op2.fHz_ana, wrapTo180(rad2deg(angle(op2.Yqq_ana))), ...
    '--', 'Color', orange_col, 'LineWidth', lw_ana);

semilogx(op2.fHz_meas, wrapTo180(rad2deg(angle(op2.Yqq_meas))), ...
    's', 'Color', orange_col, 'LineWidth', lw_meas, ...
    'MarkerSize', ms_meas, 'MarkerFaceColor', 'none');

grid on;
box(ax4,'on');
xlim([1 1e4]);
ylim([-180 180]);
% title('Y_{qq} Phase', 'FontName', font_name, 'FontSize', font_size_ti);
ax4.FontName = font_name;
ax4.FontSize = font_size_ax;
ax4.LineWidth = 1.2;

%% =========================
% Shared x-label
% =========================
xlabel(t, 'Frequency (Hz)', 'FontName', font_name, 'FontSize', font_size_lb);

%% =========================================================
% Local function: run one operating point
% =========================================================
function out = run_one_op(mdl, axisBlk, I2d, I2q, p, fHz_meas, w_meas, fHz_ana, w_ana)

    %% ---- Push variables to base workspace for Simulink
    assignin('base','Tsam', p.Tsam);
    assignin('base','Vdc',  p.Vdc);
    assignin('base','Vg',   p.Vg);
    assignin('base','w1',   p.w1);
    assignin('base','fs',   p.fs);
    assignin('base','Ts',   p.Ts);
    assignin('base','Td',   p.Td);
    assignin('base','fsw',  p.fsw);

    assignin('base','Rf1',  p.Rf1);
    assignin('base','Lf1',  p.Lf1);
    assignin('base','Rf2',  p.Rf2);
    assignin('base','Lf2',  p.Lf2);
    assignin('base','Cf',   p.Cf);

    assignin('base','Kpi',  p.Kpi);
    assignin('base','Kii',  p.Kii);

    assignin('base','Kppll',p.Kppll);
    assignin('base','Kipll',p.Kipll);
    assignin('base','beta', p.beta);

    assignin('base','Lg',   p.Lg);
    assignin('base','Rg',   p.Rg);

    assignin('base','I2d',  I2d);
    assignin('base','I2q',  I2q);
    assignin('base','V2d',  p.Vg);

    assignin('base','fHz_meas', fHz_meas);
    assignin('base','w_meas',   w_meas);
    assignin('base','w',        w_meas);   % if Simulink model uses variable w

    %% ---- Measurement Ydd: d-axis injection
    set_param(axisBlk, 'Value', '1');
    simOut_d = sim(mdl, 'SrcWorkspace', 'base');
    
    SigGenDataLog_Ydd = simOut_d.SigGenDataLog_Ydd;
    sys_estim_Ydd = frestimate(SigGenDataLog_Ydd, w_meas, "rad/s");
    Ydd_meas = -squeeze(sys_estim_Ydd.ResponseData(1,1,:));

    %% ---- Measurement Yqq: q-axis injection
    set_param(axisBlk, 'Value', '0');
    simOut_q = sim(mdl, 'SrcWorkspace', 'base');
    
    SigGenDataLog_Yqq = simOut_q.SigGenDataLog_Yqq;
    sys_estim_Yqq = frestimate(SigGenDataLog_Yqq, w_meas, "rad/s");
    Yqq_meas = -squeeze(sys_estim_Yqq.ResponseData(1,1,:));

    %% ---- Build analytical model
    % Current control loop
    Ai = zeros(2,2);
    Bi = [1 0; 0 1];
    Ci = (p.Vdc/2) * [p.Kii 0; 0 p.Kii];
    Di = (p.Vdc/2) * [p.Kpi 0; 0 p.Kpi];

    % Computational delay
    Td = p.Td;
    Adel = [0, 1, 0, 0, 0, 0; 
            0, 0, 1, 0, 0, 0; 
            -120/Td^3, -60/Td^2, -12/Td, 0, 0, 0;
            0, 0, 0, 0, 1, 0; 
            0, 0, 0, 0, 0, 1; 
            0, 0, 0, -120/Td^3, -60/Td^2, -12/Td];
    Bdel = [0, 0;
            0, 0;
            1, 0;
            0, 0;
            0, 0;
            0, 1];
    Cdel = [240/Td^3 0 24/Td 0 0 0;
            0 0 0 240/Td^3 0 24/Td];
    Ddel = [-1 0; 0 -1];

    % LCL filter
    Alcl = [-p.Rf1/p.Lf1,  p.w1,          0,             0,         -1/p.Lf1, 0;
            -p.w1,        -p.Rf1/p.Lf1,   0,             0,          0,       -1/p.Lf1;
             0,            0,            -p.Rf2/p.Lf2,   p.w1,       1/p.Lf2, 0;
             0,            0,            -p.w1,         -p.Rf2/p.Lf2,0,        1/p.Lf2;
             1/p.Cf,       0,            -1/p.Cf,        0,          0,        p.w1;
             0,            1/p.Cf,        0,            -1/p.Cf,    -p.w1,     0];
    Blcl = [1/p.Lf1 0       0        0;
            0       1/p.Lf1 0        0;
            0       0      -1/p.Lf2  0;
            0       0       0       -1/p.Lf2;
            0       0       0        0;
            0       0       0        0];
    Clcl = [0 0 1 0 0 0;
            0 0 0 1 0 0];
    Dlcl = zeros(2,4);

    % PLL
    Apll = [0 p.Kipll;
            0 0];
    Bpll = [p.Kppll;
            1];
    Cpll = [1 0];
    Dpll = 0;

    % Interconnection
    R3 = [1 0;
          0 1;
         -I2q I2d]';

    R2 = [1 0 0 0; 
          0 1 0 0; 
          0 0 1 0; 
          0 0 0 1; 
          0 0 0 1];

    R1 = [0 0 0; 
          0 0 0; 
          0 0 0; 
          0 0 -p.Vg; 
          0 0 -p.Vg];

    R4 = zeros(2,4);

    R_3 = [0 0 0 0 -1 0; 
           0 0 0 0 0 -1; 
           1 0 0 0 0 -p.w1*(p.Lf1 + p.Lf2); 
           0 1 0 0 p.w1*(p.Lf1 + p.Lf2) 0;
           0 0 1 0 0 0; 
           0 0 0 1 0 0; 
           0 0 0 0 0 0; 
           0 0 0 0 0 0];

    R_2 = [1 0 0 0; 
           0 1 0 0; 
           0 0 p.beta 0; 
           0 0 0 p.beta; 
           0 0 0 0; 
           0 0 0 0; 
           0 0 1 0; 
           0 0 0 1];

    R_1 = [0 0 0 0 1 0; 
           0 0 0 0 0 1];

    R_0 = zeros(2,4);

    % Stack matrix
    Ast = blkdiag(Ai, Adel, Alcl);
    Bst = blkdiag(Bi, Bdel, Blcl);
    Cst = blkdiag(Ci, Cdel, Clcl);
    Dst = blkdiag(Di, Ddel, Dlcl);

    % First interconnection
    Avsc = Ast + Bst*R_3*((eye(6) - Dst*R_3)\Cst);
    Bvsc = Bst*R_3*((eye(6) - Dst*R_3)\(Dst*R_2)) + Bst*R_2;
    Cvsc = R_1*((eye(6) - Dst*R_3)\Cst);
    Dvsc = R_1*((eye(6) - Dst*R_3)\(Dst*R_2)) + R_0;

    % Add PLL
    Ast1 = blkdiag(Avsc, Apll);
    Bst1 = blkdiag(Bvsc, Bpll);
    Cst1 = blkdiag(Cvsc, Cpll);
    Dst1 = blkdiag(Dvsc, Dpll);

    % Final interconnection
    Avsc2 = Ast1 + Bst1*R1*((eye(3) - Dst1*R1)\Cst1);
    Bvsc2 = Bst1*R1*((eye(3) - Dst1*R1)\(Dst1*R2)) + Bst1*R2;
    Cvsc2 = R3*((eye(3) - Dst1*R1)\Cst1);
    Dvsc2 = R3*((eye(3) - Dst1*R1)\(Dst1*R2)) + R4;

    % Derive Yvsc
    s = tf('s');
    G_closed = Cvsc2 * inv(s*eye(16) - Avsc2) * Bvsc2 + Dvsc2;
    Yvsc = -minreal(G_closed(:,3:4));

    Y11 = minreal(Yvsc(1,1));   % Ydd
    Y22 = minreal(Yvsc(2,2));   % Yqq

    Ydd_ana = squeeze(freqresp(Y11, w_ana));
    Yqq_ana = squeeze(freqresp(Y22, w_ana));

    %% ---- Output
    out.I2d = I2d;
    out.I2q = I2q;

    out.fHz_meas = fHz_meas(:);
    out.fHz_ana  = fHz_ana(:);

    out.Ydd_meas = Ydd_meas(:);
    out.Yqq_meas = Yqq_meas(:);

    out.Ydd_ana  = Ydd_ana(:);
    out.Yqq_ana  = Yqq_ana(:);
end