clear all
clc

%% Circuit and controller parameters of the GFLI
Vdc = 1150;     %dc-voltage
Vg  = 33e3;     % grid voltage
w1  = 100*pi;   % angular freq
fs  = 5e3;
Ts = 1/fs;
fsw = 5e3;

% LCL filter
Rf1 = 3e-3;
Lf1 =  250e-6;
Rf2 = 3e-3;
Lf2 = 250e-6;
Cf  = 50e-6;

% Current controller
Kpi = 1.7391e-4;
Kii = 0.0348;
%PLL
Kppll = 20/(Vdc/2);
Kipll = 200/(Vdc/2);
beta  = 0;
%Operating point
I2d = 3195;
I2q = 0;
V2d = 575;

%% Building state-space model of sub-systems
% Current control loop
Ai = zeros(2,2);
Bi = [1 , 0; 0, 1];
Ci = (Vdc/2) * [Kii, 0; 0, Kii];
Di = (Vdc/2) * [Kpi, 0; 0, Kpi];

% Computational deplay time
Td = 1.5*Ts;
Adel = [0, 1, 0, 0, 0, 0; 0, 0, 1, 0, 0, 0; -120/Td^3, -60/Td^2, -12/Td, 0, 0, 0;...
    0, 0, 0, 0, 1, 0; 0, 0, 0, 0, 0, 1; 0, 0, 0, -120/Td^3, -60/Td^2, -12/Td];
Bdel = [0, 0; 0, 0; 1, 0; 0, 0; 0, 0; 0, 1];
Cdel = [240/Td^3 0 24/Td 0 0 0; 0 0 0 240/Td^3 0 24/Td];
Ddel = [-1 0; 0 -1];

%LCL-filter
Alcl = [-Rf1/Lf1 w1 0 0 -1/Lf1 0; -w1 -Rf1/Lf1 0 0 0 -1/Lf1; 0 0 -Rf2/Lf2 w1 1/Lf2 0;...
    0 0 -w1 -Rf2/Lf2 0 1/Lf2; 1/Cf 0 -1/Cf 0 0 w1; 0 1/Cf 0 -1/Cf -w1 0];
Blcl = [1/Lf1 0 0 0; 0 1/Lf1 0 0; 0 0 -1/Lf2 0; 0 0 0 -1/Lf2; 0 0 0 0; 0 0 0 0];
Clcl = [0 0 1 0 0 0; 0 0 0 1 0 0];
Dlcl = zeros(2,4);

%PLL
Apll = [0 Kipll; 0 0];
Bpll = [Kppll; 1];
Cpll = [1 0];
Dpll = 0;

%Inter-connection
R3 = [1 0; 0 1; -I2q I2d]';
R2 = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1; 0 0 0 1];
R1 = [0 0 0; 0 0 0; 0 0 0; 0 0 -V2d; 0 0 -V2d];
R4 = zeros(2,4);

%Inter-connection
R_3 = [0 0 0 0 -1 0; 0 0 0 0 0 -1; 1 0 0 0 0 -w1*(Lf1 + Lf2); 0 1 0 0 w1*(Lf1+Lf2) 0;...
    0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0];
R_2 = [1 0 0 0; 0 1 0 0; 0 0 beta 0; 0 0 0 beta; 0 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 1];
R_1 = [0 0 0 0 1 0; 0 0 0 0 0 1];
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
Dvsc = R_1*inv(eye(6)-Dst*R_3)*Dst*R_2+R_0;

Ast1 = blkdiag(Avsc, Apll);
Bst1 = blkdiag(Bvsc, Bpll);
Cst1 = blkdiag(Cvsc, Cpll);
Dst1 = blkdiag(Dvsc, Dpll);

Avsc2 = Ast1 + Bst1*R1*inv(eye(3)-Dst1*R1)*Cst1;
Bvsc2 = Bst1*R1*inv(eye(3)-Dst1*R1)*Dst1*R2 + Bst1*R2;
Cvsc2 = R3*inv(eye(3)-Dst1*R1)*Cst1;
Dvsc2 = R3*inv(eye(3)-Dst1*R1)*Dst1*R2+R4;

% %% Step response validation
% G = ss(Avsc2, Bvsc2, Cvsc2, Dvsc2);
% % step(G)
% eig(Avsc2)
% det(Avsc2)
% det(Avsc)

%% Formulate closed-loop transfer function => Derive Yvsc
s = tf('s');
G_closed = Cvsc2 * inv(s*eye(16) - Avsc2) * Bvsc2 + Dvsc2;
Yvsc2 = -minreal(G_closed(:,3:4));

% sys_high = ss(G);     % chuyển sang dạng state-space nếu chưa
% order_desired = 15;       % bậc mong muốn
% 
% % Giảm bậc bằng Balanced Truncation
% sys_low = balred(sys_high, order_desired);
% 
% % Hiển thị bậc và kiểm tra sai số
% order(sys_high)
% order(sys_low)
% bode(sys_high, sys_low)
% legend('Original','Reduced')

%% Plot validation
Y11 = minreal(Yvsc2(1,1));
Y12 = minreal(Yvsc2(1,2));
Y21 = minreal(Yvsc2(2,1));
Y22 = minreal(Yvsc2(2,2));

% Tần số: 1 -> 1e4 Hz
fHz = logspace(0,4,600);
w   = 2*pi*fHz;

% Lấy dữ liệu Bode (|G| theo abs, pha theo deg)
[mag11, ph11] = bode(Y11, w); mag11 = squeeze(mag11); ph11 = squeeze(ph11);
[mag12, ph12] = bode(Y12, w); mag12 = squeeze(mag12); ph12 = squeeze(ph12); ph12_wrap = mod(ph12 + 180, 360) - 180; % wrap về [-180,180];
[mag21, ph21] = bode(Y21, w); mag21 = squeeze(mag21); ph21 = squeeze(ph21); ph21_wrap = mod(ph21 + 180, 360) - 180;
[mag22, ph22] = bode(Y22, w); mag22 = squeeze(mag22); ph22 = squeeze(ph22); ph22_wrap = mod(ph22 + 180, 360) - 180;

tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

% Ydd
nexttile;
yyaxis left;  semilogx(fHz, 20*log10(mag11), 'LineWidth', 1.2); ylabel('Mag (dB)');
yyaxis right; semilogx(fHz, ph11,            'LineWidth', 1.0);  ylabel('Phase (deg)');
grid on; xlim([1 1e4]); title('Y_{dd} = i_d / v_{g,d}');

% Ydq
nexttile;
yyaxis left;  semilogx(fHz, 20*log10(mag12), 'LineWidth', 1.2); ylabel('Mag (dB)');
yyaxis right; semilogx(fHz, ph12_wrap,            'LineWidth', 1.0);  ylabel('Phase (deg)');
grid on; xlim([1 1e4]); title('Y_{dq} = i_d / v_{g,q}');

% Yqd
nexttile;
yyaxis left;  semilogx(fHz, 20*log10(mag21), 'LineWidth', 1.2); ylabel('Mag (dB)');
yyaxis right; semilogx(fHz, ph21_wrap,            'LineWidth', 1.0);  ylabel('Phase (deg)');
grid on; xlim([1 1e4]); title('Y_{qd} = i_q / v_{g,d}');

% Yqq
nexttile;
yyaxis left;  semilogx(fHz, 20*log10(mag22), 'LineWidth', 1.2); ylabel('Mag (dB)');
yyaxis right; semilogx(fHz, ph22_wrap,            'LineWidth', 1.0);  ylabel('Phase (deg)');
grid on; xlim([1 1e4]); title('Y_{qq} = i_q / v_{g,q}');