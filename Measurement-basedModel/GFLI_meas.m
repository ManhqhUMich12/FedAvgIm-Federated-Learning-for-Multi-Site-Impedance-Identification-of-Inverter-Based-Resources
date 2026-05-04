clear all;
clc
Tsam = 5e-5;
%% Circuit and controller parameters of the GFLI
Vdc = 1150;     
Vg  = 575;   
f0 = 50;
w1  = 2*pi*f0;   
fs  = 5e3;
Ts = 1/fs;
Td = 1.5*Ts;
fsw = 5e3;

% LCL filter
Rf1 = 3e-3;
Lf1 = 250e-6;
Rf2 = 3e-3;
Lf2 = 250e-6;
Cf  = 50e-6;

% Current controller
Kpi = 1.7391e-4;
Kii = 0.0348;

% PLL
Kppll = 40/(Vdc/2);
Kipll = 400/(Vdc/2);
% Kppll = 0;
% Kipll = 0;
beta  = 0;

% Operating point
I2d = 3195;
% I2d = 5000;
I2q = -2132;
V2d = 575;
Lg  = 250e-6;          
Rg = 3e-3;

%% ------------ Simulink model and measurement setup
mdl = "GFLI.slx";
open_system(mdl)

axisBlk = 'GFLI/AISTool/AxisSlt';

% ===== Measured frequencies =====
fHz_meas = logspace(log10(1), log10(200), 10);
w_meas   = 2*pi*fHz_meas;
w = w_meas;   % if model uses variable w from workspace

%% ===== Run simulation for d-axis injection: get Ydd, Yqd =====
set_param(axisBlk, 'Value', '1');
sim(mdl);

sys_estim_Ydd = frestimate(SigGenDataLog_Ydd, w_meas, "rad/s");
sys_estim_Yqd = frestimate(SigGenDataLog_Yqd, w_meas, "rad/s");

%% ===== Run simulation for q-axis injection: get Ydq, Yqq =====
set_param(axisBlk, 'Value', '0');
sim(mdl);

sys_estim_Ydq = frestimate(SigGenDataLog_Ydq, w_meas, "rad/s");
sys_estim_Yqq = frestimate(SigGenDataLog_Yqq, w_meas, "rad/s");

%% Building state-space model of sub-systems
% Current control loop
Ai = zeros(2,2);
Bi = [1 , 0; 0, 1];
Ci = (Vdc/2) * [Kii, 0; 0, Kii];
Di = (Vdc/2) * [Kpi, 0; 0, Kpi];

% Computational delay time
Td = 1.5*Ts;
Adel = [0, 1, 0, 0, 0, 0; 
        0, 0, 1, 0, 0, 0; 
        -120/Td^3, -60/Td^2, -12/Td, 0, 0, 0;...
        0, 0, 0, 0, 1, 0; 
        0, 0, 0, 0, 0, 1; 
        0, 0, 0, -120/Td^3, -60/Td^2, -12/Td];
Bdel = [0, 0; 0, 0; 1, 0; 0, 0; 0, 0; 0, 1];
Cdel = [240/Td^3 0 24/Td 0 0 0; 
        0 0 0 240/Td^3 0 24/Td];
Ddel = [-1 0; 0 -1];

% LCL filter
Alcl = [-Rf1/Lf1 w1 0 0 -1/Lf1 0; 
        -w1 -Rf1/Lf1 0 0 0 -1/Lf1; 
        0 0 -Rf2/Lf2 w1 1/Lf2 0;...
        0 0 -w1 -Rf2/Lf2 0 1/Lf2; 
        1/Cf 0 -1/Cf 0 0 w1; 
        0 1/Cf 0 -1/Cf -w1 0];
Blcl = [1/Lf1 0 0 0; 
        0 1/Lf1 0 0; 
        0 0 -1/Lf2 0; 
        0 0 0 -1/Lf2; 
        0 0 0 0; 
        0 0 0 0];
Clcl = [0 0 1 0 0 0; 
        0 0 0 1 0 0];
Dlcl = zeros(2,4);

% PLL
Apll = [0 Kipll; 0 0];
Bpll = [Kppll; 1];
Cpll = [1 0];
Dpll = 0;

% Inter-connection
R3 = [1 0; 0 1; -I2q I2d]';
R2 = [1 0 0 0; 
      0 1 0 0; 
      0 0 1 0; 
      0 0 0 1; 
      0 0 0 1];
R1 = [0 0 0; 
      0 0 0; 
      0 0 0; 
      0 0 -V2d; 
      0 0 -V2d];
R4 = zeros(2,4);

R_3 = [0 0 0 0 -1 0; 
       0 0 0 0 0 -1; 
       1 0 0 0 0 -w1*(Lf1 + Lf2); 
       0 1 0 0 w1*(Lf1+Lf2) 0;...
       0 0 1 0 0 0; 
       0 0 0 1 0 0; 
       0 0 0 0 0 0; 
       0 0 0 0 0 0];
R_2 = [1 0 0 0; 
       0 1 0 0; 
       0 0 beta 0; 
       0 0 0 beta; 
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

%% Final state-space representation
Avsc = Ast + Bst*R_3*inv(eye(6) - Dst*R_3)*Cst;
Bvsc = Bst*R_3*inv(eye(6)-Dst*R_3)*Dst*R_2 + Bst*R_2;
Cvsc = R_1*inv(eye(6)-Dst*R_3)*Cst;
Dvsc = R_1*inv(eye(6)-Dst*R_3)*Dst*R_2 + R_0;

Ast1 = blkdiag(Avsc, Apll);
Bst1 = blkdiag(Bvsc, Bpll);
Cst1 = blkdiag(Cvsc, Cpll);
Dst1 = blkdiag(Dvsc, Dpll);

Avsc2 = Ast1 + Bst1*R1*inv(eye(3)-Dst1*R1)*Cst1;
Bvsc2 = Bst1*R1*inv(eye(3)-Dst1*R1)*Dst1*R2 + Bst1*R2;
Cvsc2 = R3*inv(eye(3)-Dst1*R1)*Cst1;
Dvsc2 = R3*inv(eye(3)-Dst1*R1)*Dst1*R2 + R4;

%% Derive Yvsc
s = tf('s');
G_closed = Cvsc2 * inv(s*eye(16) - Avsc2) * Bvsc2 + Dvsc2;
Yvsc = -minreal(G_closed(:,3:4));

Y11 = minreal(Yvsc(1,1));   % Ydd
Y12 = minreal(Yvsc(1,2));   % Ydq
Y21 = minreal(Yvsc(2,1));   % Yqd
Y22 = minreal(Yvsc(2,2));   % Yqq

%% ===== Analytical frequencies =====
fHz_ana = logspace(0,4,600);
w_ana   = 2*pi*fHz_ana;

H11_ana = squeeze(freqresp(Y11, w_ana));
H12_ana = squeeze(freqresp(Y12, w_ana));
H21_ana = squeeze(freqresp(Y21, w_ana));
H22_ana = squeeze(freqresp(Y22, w_ana));

mag11_ana = abs(H11_ana);
mag12_ana = abs(H12_ana);
mag21_ana = abs(H21_ana);
mag22_ana = abs(H22_ana);

ph11_ana = mod(rad2deg(angle(H11_ana)) + 180, 360) - 180;
ph12_ana = mod(rad2deg(angle(H12_ana)) + 180, 360) - 180;
ph21_ana = mod(rad2deg(angle(H21_ana)) + 180, 360) - 180;
ph22_ana = mod(rad2deg(angle(H22_ana)) + 180, 360) - 180;

%% ===== Measured data =====
Ymeas_dd = -squeeze(sys_estim_Ydd.ResponseData(1,1,:));
Ymeas_dq = -squeeze(sys_estim_Ydq.ResponseData(1,1,:));
Ymeas_qd = -squeeze(sys_estim_Yqd.ResponseData(1,1,:));
Ymeas_qq = -squeeze(sys_estim_Yqq.ResponseData(1,1,:));

mag_meas_dd = abs(Ymeas_dd);
mag_meas_dq = abs(Ymeas_dq);
mag_meas_qd = abs(Ymeas_qd);
mag_meas_qq = abs(Ymeas_qq);

ph_meas_dd = mod(rad2deg(angle(Ymeas_dd)) + 180, 360) - 180;
ph_meas_dq = mod(rad2deg(angle(Ymeas_dq)) + 180, 360) - 180;
ph_meas_qd = mod(rad2deg(angle(Ymeas_qd)) + 180, 360) - 180;
ph_meas_qq = mod(rad2deg(angle(Ymeas_qq)) + 180, 360) - 180;

%% ===== Plot Ydd comparison =====
figure;

subplot(2,1,1)
semilogx(fHz_ana, 20*log10(mag11_ana), 'b-', 'LineWidth', 1.5); hold on;
semilogx(fHz_meas, 20*log10(mag_meas_dd), 'ro', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Y_{dd}: Analytical vs Measured');
legend('Analytical Y_{dd}', 'Measured Y_{dd}', 'Location', 'best');

subplot(2,1,2)
semilogx(fHz_ana, ph11_ana, 'b-', 'LineWidth', 1.5); hold on;
semilogx(fHz_meas, ph_meas_dd, 'ro', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (deg)');
ylim([-180 180]);
legend('Analytical Y_{dd}', 'Measured Y_{dd}', 'Location', 'best');

%% ===== Plot Ydq comparison =====
figure;

subplot(2,1,1)
semilogx(fHz_ana, 20*log10(mag12_ana), 'b-', 'LineWidth', 1.5); hold on;
semilogx(fHz_meas, 20*log10(mag_meas_dq), 'ro', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Y_{dq}: Analytical vs Measured');
legend('Analytical Y_{dq}', 'Measured Y_{dq}', 'Location', 'best');

subplot(2,1,2)
semilogx(fHz_ana, ph12_ana, 'b-', 'LineWidth', 1.5); hold on;
semilogx(fHz_meas, ph_meas_dq, 'ro', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (deg)');
ylim([-180 180]);
legend('Analytical Y_{dq}', 'Measured Y_{dq}', 'Location', 'best');

%% ===== Plot Yqd comparison =====
figure;

subplot(2,1,1)
semilogx(fHz_ana, 20*log10(mag21_ana), 'b-', 'LineWidth', 1.5); hold on;
semilogx(fHz_meas, 20*log10(mag_meas_qd), 'ro', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Y_{qd}: Analytical vs Measured');
legend('Analytical Y_{qd}', 'Measured Y_{qd}', 'Location', 'best');

subplot(2,1,2)
semilogx(fHz_ana, ph21_ana, 'b-', 'LineWidth', 1.5); hold on;
semilogx(fHz_meas, ph_meas_qd, 'ro', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (deg)');
ylim([-180 180]);
legend('Analytical Y_{qd}', 'Measured Y_{qd}', 'Location', 'best');

%% ===== Plot Yqq comparison =====
figure;

subplot(2,1,1)
semilogx(fHz_ana, 20*log10(mag22_ana), 'b-', 'LineWidth', 1.5); hold on;
semilogx(fHz_meas, 20*log10(mag_meas_qq), 'ro', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Y_{qq}: Analytical vs Measured');
legend('Analytical Y_{qq}', 'Measured Y_{qq}', 'Location', 'best');

subplot(2,1,2)
semilogx(fHz_ana, ph22_ana, 'b-', 'LineWidth', 1.5); hold on;
semilogx(fHz_meas, ph_meas_qq, 'ro', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (deg)');
ylim([-180 180]);
legend('Analytical Y_{qq}', 'Measured Y_{qq}', 'Location', 'best');