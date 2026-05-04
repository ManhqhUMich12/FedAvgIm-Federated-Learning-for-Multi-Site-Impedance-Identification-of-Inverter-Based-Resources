clear; clc; close all;

%% =========================
% Fixed parameters
% =========================
p.Tsam = 5e-5;
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
% Noise options for synthetic analytical data
% =========================
noiseOpt.enable = true;
noiseOpt.seed   = 1;

% Relative noise
noiseOpt.rel_sigma_low  = 0.0005;
noiseOpt.rel_sigma_high = 0.15;

% Absolute noise floor
noiseOpt.abs_floor_ratio_low  = 0.0005;
noiseOpt.abs_floor_ratio_high = 0.01;

% Transition frequency
noiseOpt.fc = 10;
noiseOpt.p  = 100.0;

% Bias
noiseOpt.bias_mag_high = 0.1;
noiseOpt.bias_ph_high  = 4.0;
noiseOpt.bias_fc       = 50;

%% =========================
% Simulink model
% =========================
mdl = "GFLI.slx";
open_system(mdl);

axisBlk = 'GFLI/AISTool/AxisSlt';

%% =========================
% Frequency vectors
% =========================
fHz_meas = logspace(log10(1), log10(200), 20);
w_meas   = 2*pi*fHz_meas;

fHz_ana  = logspace(log10(1), log10(200), 20);
w_ana    = 2*pi*fHz_ana;

%% =========================
% Operating point
% =========================
I2d = 3195;
I2q = 0;

%% =========================
% Run clean OP only ONCE
% =========================
op1 = run_one_op_clean(mdl, axisBlk, I2d, I2q, p, fHz_meas, w_meas, fHz_ana, w_ana);

%% =========================
% Apply noise the first time
% =========================
op1 = apply_noise_to_op(op1, noiseOpt, noiseOpt.seed);

%% =========================
% Create figure once
% =========================
[hFig, hPlot] = create_op_plot(op1);
%% =========================================================
% FUNCTION: run clean operating point
% Only measurement + analytical clean
% =========================================================
function out = run_one_op_clean(mdl, axisBlk, I2d, I2q, p, fHz_meas, w_meas, fHz_ana, w_ana)

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

    %% ---- Build analytical model
    Ai = zeros(2,2);
    Bi = [1 0; 0 1];
    Ci = (p.Vdc/2) * [p.Kii 0; 0 p.Kii];
    Di = (p.Vdc/2) * [p.Kpi 0; 0 p.Kpi];

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

    Apll = [0 p.Kipll;
            0 0];
    Bpll = [p.Kppll;
            1];
    Cpll = [1 0];
    Dpll = 0;

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

    Ast = blkdiag(Ai, Adel, Alcl);
    Bst = blkdiag(Bi, Bdel, Blcl);
    Cst = blkdiag(Ci, Cdel, Clcl);
    Dst = blkdiag(Di, Ddel, Dlcl);

    Avsc = Ast + Bst*R_3*((eye(6) - Dst*R_3)\Cst);
    Bvsc = Bst*R_3*((eye(6) - Dst*R_3)\(Dst*R_2)) + Bst*R_2;
    Cvsc = R_1*((eye(6) - Dst*R_3)\Cst);
    Dvsc = R_1*((eye(6) - Dst*R_3)\(Dst*R_2)) + R_0;

    Ast1 = blkdiag(Avsc, Apll);
    Bst1 = blkdiag(Bvsc, Bpll);
    Cst1 = blkdiag(Cvsc, Cpll);
    Dst1 = blkdiag(Dvsc, Dpll);

    Avsc2 = Ast1 + Bst1*R1*((eye(3) - Dst1*R1)\Cst1);
    Bvsc2 = Bst1*R1*((eye(3) - Dst1*R1)\(Dst1*R2)) + Bst1*R2;
    Cvsc2 = R3*((eye(3) - Dst1*R1)\Cst1);
    Dvsc2 = R3*((eye(3) - Dst1*R1)\(Dst1*R2)) + R4;

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

    % initialize noisy data = clean analytical first
    out.Ydd_syn = Ydd_ana(:);
    out.Yqq_syn = Yqq_ana(:);
end

%% =========================================================
% FUNCTION: apply noise only
% No Simulink rerun, no operating-point rerun
% =========================================================
function op = apply_noise_to_op(op, noiseOpt, seed)

    if nargin < 3
        seed = 1;
    end

    if ~isfield(noiseOpt, 'enable') || ~noiseOpt.enable
        op.Ydd_syn = op.Ydd_ana;
        op.Yqq_syn = op.Yqq_ana;
        return;
    end

    rng(seed, 'twister');

    op.Ydd_syn = add_frf_noise(op.Ydd_ana, op.fHz_ana, noiseOpt);
    op.Yqq_syn = add_frf_noise(op.Yqq_ana, op.fHz_ana, noiseOpt);
end

%% =========================================================
% FUNCTION: create plot once
% =========================================================
function [hFig, h] = create_op_plot(op)

    % Style
    blue_col   = [0.00 0.45 0.74];
    orange_col = [0.85 0.33 0.10];
    gray_col   = [0.35 0.35 0.35];

    lw_syn   = 1.4;
    lw_meas  = 1.6;
    lw_ana   = 1.2;

    ms_ana   = 5;
    ms_syn   = 5;
    ms_meas  = 6;

    fig_pos = [100 100 700 450];

    font_name     = 'Times New Roman';
    font_size_ax  = 11;
    font_size_lb  = 11;
    font_size_ti  = 11;
    font_size_leg = 10;

    hFig = figure;
    set(hFig, 'Position', fig_pos);

    h.t = tiledlayout(2,2, 'Padding','compact', 'TileSpacing','compact');

    % -------------------------
    % Ydd magnitude
    % -------------------------
    h.ax1 = nexttile;
    h.lnYddAnaMag = semilogx(op.fHz_ana, 20*log10(abs(op.Ydd_ana)), ...
        's--', 'Color', blue_col, 'LineWidth', lw_ana, ...
        'MarkerSize', ms_ana, 'MarkerFaceColor', 'none'); hold on;

    h.lnYddSynMag = semilogx(op.fHz_ana, 20*log10(abs(op.Ydd_syn)), ...
        'x--', 'Color', orange_col, 'LineWidth', lw_syn, ...
        'MarkerSize', ms_syn);

    h.lnYddMeasMag = semilogx(op.fHz_meas, 20*log10(abs(op.Ydd_meas)), ...
        'o--', 'Color', gray_col, 'LineWidth', lw_meas, ...
        'MarkerSize', ms_meas, 'MarkerFaceColor', 'none');

    grid on; box(h.ax1,'on');
    xlim([1 200]);
    title('Y_{dd}', 'FontName', font_name,'FontAngle','italic', ...
        'FontSize', font_size_ti, 'FontWeight', 'normal');
    ylabel('Magnitude (dB)', 'FontName', font_name, 'FontSize', font_size_lb);
    h.ax1.FontName = font_name;
    h.ax1.FontSize = font_size_ax;
    h.ax1.LineWidth = 1.2;

    % -------------------------
    % Yqq magnitude
    % -------------------------
    h.ax2 = nexttile;
    h.lnYqqAnaMag = semilogx(op.fHz_ana, 20*log10(abs(op.Yqq_ana)), ...
        's--', 'Color', blue_col, 'LineWidth', lw_ana, ...
        'MarkerSize', ms_ana, 'MarkerFaceColor', 'none'); hold on;

    h.lnYqqSynMag = semilogx(op.fHz_ana, 20*log10(abs(op.Yqq_syn)), ...
        'x--', 'Color', orange_col, 'LineWidth', lw_syn, ...
        'MarkerSize', ms_syn);

    h.lnYqqMeasMag = semilogx(op.fHz_meas, 20*log10(abs(op.Yqq_meas)), ...
        'o--', 'Color', gray_col, 'LineWidth', lw_meas, ...
        'MarkerSize', ms_meas, 'MarkerFaceColor', 'none');

    grid on; box(h.ax2,'on');
    xlim([1 200]);
    title('Y_{qq}', 'FontName', font_name,'FontAngle','italic', ...
        'FontSize', font_size_ti, 'FontWeight', 'normal');
    h.ax2.FontName = font_name;
    h.ax2.FontSize = font_size_ax;
    h.ax2.LineWidth = 1.2;

    h.lgd = legend(h.ax2, ...
        'Analytical clean', ...
        'Analytical synthetic-noisy', ...
        'Measured', ...
        'Location','best');
    h.lgd.FontName = font_name;
    h.lgd.FontSize = font_size_leg;

    % -------------------------
    % Ydd phase
    % -------------------------
    h.ax3 = nexttile;
    h.lnYddAnaPh = semilogx(op.fHz_ana, wrapTo180(rad2deg(angle(op.Ydd_ana))), ...
        's--', 'Color', blue_col, 'LineWidth', lw_ana, ...
        'MarkerSize', ms_ana, 'MarkerFaceColor', 'none'); hold on;

    h.lnYddSynPh = semilogx(op.fHz_ana, wrapTo180(rad2deg(angle(op.Ydd_syn))), ...
        'x--', 'Color', orange_col, 'LineWidth', lw_syn, ...
        'MarkerSize', ms_syn);

    h.lnYddMeasPh = semilogx(op.fHz_meas, wrapTo180(rad2deg(angle(op.Ydd_meas))), ...
        'o--', 'Color', gray_col, 'LineWidth', lw_meas, ...
        'MarkerSize', ms_meas, 'MarkerFaceColor', 'none');

    grid on; box(h.ax3,'on');
    xlim([1 200]);
    ylim([-180 180]);
    ylabel('Phase (deg)', 'FontName', font_name, 'FontSize', font_size_lb);
    h.ax3.FontName = font_name;
    h.ax3.FontSize = font_size_ax;
    h.ax3.LineWidth = 1.2;

    % -------------------------
    % Yqq phase
    % -------------------------
    h.ax4 = nexttile;
    h.lnYqqAnaPh = semilogx(op.fHz_ana, wrapTo180(rad2deg(angle(op.Yqq_ana))), ...
        's--', 'Color', blue_col, 'LineWidth', lw_ana, ...
        'MarkerSize', ms_ana, 'MarkerFaceColor', 'none'); hold on;

    h.lnYqqSynPh = semilogx(op.fHz_ana, wrapTo180(rad2deg(angle(op.Yqq_syn))), ...
        'x--', 'Color', orange_col, 'LineWidth', lw_syn, ...
        'MarkerSize', ms_syn);

    h.lnYqqMeasPh = semilogx(op.fHz_meas, wrapTo180(rad2deg(angle(op.Yqq_meas))), ...
        'o--', 'Color', gray_col, 'LineWidth', lw_meas, ...
        'MarkerSize', ms_meas, 'MarkerFaceColor', 'none');

    grid on; box(h.ax4,'on');
    xlim([1 200]);
    ylim([-180 180]);
    h.ax4.FontName = font_name;
    h.ax4.FontSize = font_size_ax;
    h.ax4.LineWidth = 1.2;

    xlabel(h.t, 'Frequency (Hz)', 'FontName', font_name, 'FontSize', font_size_lb);
end

%% =========================================================
% FUNCTION: update plot only
% =========================================================
function update_op_plot(h, op)

    % ---- Ydd magnitude
    set(h.lnYddAnaMag,  'XData', op.fHz_ana,  'YData', 20*log10(abs(op.Ydd_ana)));
    set(h.lnYddSynMag,  'XData', op.fHz_ana,  'YData', 20*log10(abs(op.Ydd_syn)));
    set(h.lnYddMeasMag, 'XData', op.fHz_meas, 'YData', 20*log10(abs(op.Ydd_meas)));

    % ---- Yqq magnitude
    set(h.lnYqqAnaMag,  'XData', op.fHz_ana,  'YData', 20*log10(abs(op.Yqq_ana)));
    set(h.lnYqqSynMag,  'XData', op.fHz_ana,  'YData', 20*log10(abs(op.Yqq_syn)));
    set(h.lnYqqMeasMag, 'XData', op.fHz_meas, 'YData', 20*log10(abs(op.Yqq_meas)));

    % ---- Ydd phase
    set(h.lnYddAnaPh,   'XData', op.fHz_ana,  'YData', wrapTo180(rad2deg(angle(op.Ydd_ana))));
    set(h.lnYddSynPh,   'XData', op.fHz_ana,  'YData', wrapTo180(rad2deg(angle(op.Ydd_syn))));
    set(h.lnYddMeasPh,  'XData', op.fHz_meas, 'YData', wrapTo180(rad2deg(angle(op.Ydd_meas))));

    % ---- Yqq phase
    set(h.lnYqqAnaPh,   'XData', op.fHz_ana,  'YData', wrapTo180(rad2deg(angle(op.Yqq_ana))));
    set(h.lnYqqSynPh,   'XData', op.fHz_ana,  'YData', wrapTo180(rad2deg(angle(op.Yqq_syn))));
    set(h.lnYqqMeasPh,  'XData', op.fHz_meas, 'YData', wrapTo180(rad2deg(angle(op.Yqq_meas))));

    drawnow;
end

%% =========================================================
% FUNCTION: add direct complex noise to FRF
% =========================================================
function Ynoisy = add_frf_noise(Yclean, fHz, noiseOpt)

    Yclean = Yclean(:);
    fHz    = fHz(:);

    % Noise weight: low freq nhỏ, high freq lớn
    w_hf = (fHz ./ noiseOpt.fc).^noiseOpt.p;
    w_hf = w_hf ./ (1 + w_hf);

    rel_sigma_f = noiseOpt.rel_sigma_low + ...
        (noiseOpt.rel_sigma_high - noiseOpt.rel_sigma_low) .* w_hf;

    eta_rel = (randn(size(Yclean)) + 1j*randn(size(Yclean))) / sqrt(2);

    floor_low  = noiseOpt.abs_floor_ratio_low  * max(abs(Yclean));
    floor_high = noiseOpt.abs_floor_ratio_high * max(abs(Yclean));
    beta_f = floor_low + (floor_high - floor_low) .* w_hf;

    eta_abs = (randn(size(Yclean)) + 1j*randn(size(Yclean))) / sqrt(2);

    % Bias weight
    w_bias = (fHz ./ noiseOpt.bias_fc).^noiseOpt.p;
    w_bias = w_bias ./ (1 + w_bias);

    mag_bias = 1 + noiseOpt.bias_mag_high .* w_bias;
    ph_bias  = deg2rad(noiseOpt.bias_ph_high .* w_bias);

    Ynoisy = Yclean .* mag_bias .* exp(1j*ph_bias) ...
           + Yclean .* (rel_sigma_f .* eta_rel) ...
           + beta_f .* eta_abs;
end