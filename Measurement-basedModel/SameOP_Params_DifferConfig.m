clear; clc; close all;

%% =========================================================
% Compare measured admittance of two IBRs at SAME OP and SAME parameters
%
% IBR1: GFLI.slx   -> LCL filter model
% IBR2: GFLI_L.slx -> L filter model
%
% Only measured admittance is compared.
% Do NOT reuse LCL analytical model for GFLI_L unless you derive L-filter model.
%% =========================================================

%% =========================
% Fixed parameters
% ==========================
p.Tsam = 5e-5;

% Circuit and controller parameters
p.Vdc = 1150;
p.Vg  = 575;
p.w1  = 100*pi;
p.fs  = 5e3;
p.Ts  = 1/p.fs;
p.Td  = 1.5*p.Ts;
p.fsw = 5e3;

% LCL / L filter parameters
% For GFLI.slx, these are used as LCL parameters.
% For GFLI_L.slx, use the variables that your L-filter model needs.
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
% Two Simulink models
% ==========================
mdl_ibr1 = 'GFLI';     % GFLI.slx, LCL filter
mdl_ibr2 = 'GFLI_L';   % GFLI_L.slx, L filter

open_system([mdl_ibr1 '.slx']);
open_system([mdl_ibr2 '.slx']);

axisBlk_ibr1 = [mdl_ibr1 '/AISTool/AxisSlt'];
axisBlk_ibr2 = [mdl_ibr2 '/AISTool/AxisSlt'];

%% =========================
% Frequency vector
% ==========================
fHz_meas = logspace(log10(1), log10(200), 20).';
w_meas   = 2*pi*fHz_meas;

%% =========================
% SAME operating point for both IBRs
% ==========================
I2d = 3195;
I2q = 0;

op_ibr1 = run_one_op_measured_only(mdl_ibr1, axisBlk_ibr1, I2d, I2q, p, ...
    fHz_meas, w_meas);

op_ibr2 = run_one_op_measured_only(mdl_ibr2, axisBlk_ibr2, I2d, I2q, p, ...
    fHz_meas, w_meas);

%% =========================
% Plot settings
% ==========================
blue_col   = [0.00 0.45 0.74];
orange_col = [0.85 0.33 0.10];

lw  = 1.8;
ms  = 6;

font_name = 'Times New Roman';
font_size_ax  = 11;
font_size_lb  = 11;
font_size_ti  = 11;
font_size_leg = 10;

%% =========================================================
% Figure: Ydd and Yqq comparison
%% =========================================================
fig = figure;
set(fig, 'Position', [100 100 650 430]);

t = tiledlayout(2,2, 'Padding','compact', 'TileSpacing','compact');

%% =========================
% Ydd magnitude
% ==========================
ax1 = nexttile;

semilogx(op_ibr1.fHz_meas, 20*log10(abs(op_ibr1.Ydd_meas)), ...
    '-o', 'Color', blue_col, 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'none'); hold on;

semilogx(op_ibr2.fHz_meas, 20*log10(abs(op_ibr2.Ydd_meas)), ...
    '--s', 'Color', orange_col, 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'none');

grid on; box on;
title('Y_{dd}', 'FontName', font_name, 'FontAngle','italic', ...
    'FontSize', font_size_ti, 'FontWeight','normal');
ylabel('Magnitude (dB)', 'FontName', font_name, 'FontSize', font_size_lb);
xlim([1 200]);

ax1.FontName = font_name;
ax1.FontSize = font_size_ax;
ax1.LineWidth = 1.2;

%% =========================
% Yqq magnitude
% ==========================
ax2 = nexttile;

semilogx(op_ibr1.fHz_meas, 20*log10(abs(op_ibr1.Yqq_meas)), ...
    '-o', 'Color', blue_col, 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'none'); hold on;

semilogx(op_ibr2.fHz_meas, 20*log10(abs(op_ibr2.Yqq_meas)), ...
    '--s', 'Color', orange_col, 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'none');

grid on; box on;
title('Y_{qq}', 'FontName', font_name, 'FontAngle','italic', ...
    'FontSize', font_size_ti, 'FontWeight','normal');
xlim([1 200]);

ax2.FontName = font_name;
ax2.FontSize = font_size_ax;
ax2.LineWidth = 1.2;

legend({ ...
    'IBR1: GFLI, LCL filter', ...
    'IBR2: GFLI\_L, L filter'}, ...
    'Location','best', ...
    'FontName', font_name, ...
    'FontSize', font_size_leg);

%% =========================
% Ydd phase
% ==========================
ax3 = nexttile;

semilogx(op_ibr1.fHz_meas, wrapTo180(rad2deg(angle(op_ibr1.Ydd_meas))), ...
    '-o', 'Color', blue_col, 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'none'); hold on;

semilogx(op_ibr2.fHz_meas, wrapTo180(rad2deg(angle(op_ibr2.Ydd_meas))), ...
    '--s', 'Color', orange_col, 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'none');

grid on; box on;
ylabel('Phase (deg)', 'FontName', font_name, 'FontSize', font_size_lb);
xlim([1 200]);
ylim([-180 180]);

ax3.FontName = font_name;
ax3.FontSize = font_size_ax;
ax3.LineWidth = 1.2;

%% =========================
% Yqq phase
% ==========================
ax4 = nexttile;

semilogx(op_ibr1.fHz_meas, wrapTo180(rad2deg(angle(op_ibr1.Yqq_meas))), ...
    '-o', 'Color', blue_col, 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'none'); hold on;

semilogx(op_ibr2.fHz_meas, wrapTo180(rad2deg(angle(op_ibr2.Yqq_meas))), ...
    '--s', 'Color', orange_col, 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'none');

grid on; box on;
xlim([1 200]);
ylim([-180 180]);

ax4.FontName = font_name;
ax4.FontSize = font_size_ax;
ax4.LineWidth = 1.2;

xlabel(t, 'Frequency (Hz)', 'FontName', font_name, 'FontSize', font_size_lb);

%% =========================================================
% Local function: measured admittance only
%% =========================================================
function out = run_one_op_measured_only(mdl, axisBlk, I2d, I2q, p, ...
    fHz_meas, w_meas)

    %% ---- Push variables to base workspace for Simulink
    assignin('base','Tsam', p.Tsam);

    assignin('base','Vdc',  p.Vdc);
    assignin('base','Vg',   p.Vg);
    assignin('base','w1',   p.w1);
    assignin('base','fs',   p.fs);
    assignin('base','Ts',   p.Ts);
    assignin('base','Td',   p.Td);
    assignin('base','fsw',  p.fsw);

    assignin('base','Rf1',  p.Rf1*2);
    assignin('base','Lf1',  p.Lf1*2);
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
    assignin('base','w',        w_meas);

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

    %% ---- Output
    out.model = mdl;
    out.I2d = I2d;
    out.I2q = I2q;

    out.Kpi = p.Kpi;
    out.Kii = p.Kii;
    out.Kppll = p.Kppll;
    out.Kipll = p.Kipll;

    out.fHz_meas = fHz_meas(:);

    out.Ydd_meas = Ydd_meas(:);
    out.Yqq_meas = Yqq_meas(:);
end