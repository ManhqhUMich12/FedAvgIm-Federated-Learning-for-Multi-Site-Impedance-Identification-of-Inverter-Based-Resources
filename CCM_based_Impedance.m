%% Circuit and controller parameters of the GFLI
Vdc = 1150;     %dc-voltage
Vg  = 33e3;     % grid voltage
w1  = 100*pi;   % angular freq
fs  = 5e3;
Ts = 1/fs;
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
%PLL
Kppll = 20;
Kipll = 200;
beta  = 0;

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

%Inter-connection
R3 = [0 0 0 0 -1 0; 0 0 0 0 0 -1; 1 0 0 0 0 -w1*(Lf1 + Lf2); 0 1 0 0 w1*(Lf1+Lf2) 0;...
    0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0];
R2 = [1 0 0 0; 0 1 0 0; 0 0 beta 0; 0 0 0 beta; 0 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 1];
R1 = [0 0 0 0 1 0; 0 0 0 0 0 1];
R0 = zeros(2,4);

% Stack matrix
Ast = blkdiag(Ai, Adel, Alcl);
Bst = blkdiag(Bi, Bdel, Blcl);
Cst = blkdiag(Ci, Cdel, Clcl);
Dst = blkdiag(Di, Ddel, Dlcl);

%% Final state-space representation
Avsc = Ast + Bst*R3*inv(eye(6) - Dst*R3)*Cst;
Bvsc = Bst*R3*inv(eye(6)-Dst*R3)*Dst*R2 + Bst*R2;
Cvsc = R1*inv(eye(6)-Dst*R3)*Cst;
Dvsc = R1*inv(eye(6)-Dst*R3)*Dst*R2+R0;

% %% Step response validation
% G = ss(Avsc, Bvsc, Cvsc, Dvsc);
% % step(G)
% % eig(Avsc)
% 
% % ==== REPRO Fig. 3(a) with the linear SSR ====
% t_end = 30;              % simulate 30 s as in the figure’s narrative
% dt    = 1e-4;            % small step for smooth lsim (0.1 ms)
% t     = (0:dt:t_end).';  % time vector
% 
% % Build input U = [vgd, vgq, iref_d, iref_q]
% U = zeros(length(t), 4);
% 
% % Ideal PLL case in SSR: keep grid-voltage perturbations = 0 (small-signal)
% % -> we only inject steps on the current references (delta around op point).
% 
% % Step on iref_d: + (3408 - 3195) = +213 A at t = 10 s
% dI_d = 3408 - 3195;   % 213 A
% U(t >= 10, 1) = dI_d;
% 
% % Step on iref_q: +426 A at t = 20 s
% dI_q = 426;           % 426 A
% U(t >= 20, 2) = dI_q;
% 
% % Simulate
% [y, t_out, x] = lsim(G, U, t);
% 
% % y is expected to be [i_c,d ; i_c,q] (converter-side currents) per your Cvsc
% id = y(:,1);  iq = y(:,2);
% 
% % Plot
% figure; 
% subplot(2,1,1); plot(t_out, id, 'LineWidth', 1.2); grid on;
% xlabel('Time (s)'); ylabel('\Delta i_{c,d} (A)');
% title('Linear SSR response: \Delta i_{c,d}');
% xline(10,'--'); xline(20,'--');
% 
% subplot(2,1,2); plot(t_out, iq, 'LineWidth', 1.2); grid on;
% xlabel('Time (s)'); ylabel('\Delta i_{c,q} (A)');
% title('Linear SSR response: \Delta i_{c,q}');
% xline(10,'--'); xline(20,'--');

%% Formulate closed-loop transfer function => Derive Yvsc
s = tf('s');
G_closed = Cvsc * inv(s*eye(14) - Avsc) * Bvsc + Dvsc;
Yvsc = -minreal(G_closed(:,3:4));

%% Plot validation
% Tách 4 phần tử SISO
Y11 = minreal(Yvsc(1,1));
Y12 = minreal(Yvsc(1,2));
Y21 = minreal(Yvsc(2,1));
Y22 = minreal(Yvsc(2,2));

% Tần số: 1 -> 1e4 Hz
fHz = logspace(0,4,600);
w   = 2*pi*fHz;

% Lấy dữ liệu Bode (|G| theo abs, pha theo deg)
[mag11, ph11] = bode(Y11, w); mag11 = squeeze(mag11); ph11 = squeeze(ph11);
[mag12, ph12] = bode(Y12, w); mag12 = squeeze(mag12); ph12 = squeeze(ph12); ph12_wrap = mod(ph12 + 180, 360) - 180; % wrap về [-180,180];
[mag21, ph21] = bode(Y21, w); mag21 = squeeze(mag21); ph21 = squeeze(ph21); ph21_wrap = mod(ph21 + 180, 360) - 180;
[mag22, ph22] = bode(Y22, w); mag22 = squeeze(mag22); ph22 = squeeze(ph22);

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
yyaxis right; semilogx(fHz, ph22,            'LineWidth', 1.0);  ylabel('Phase (deg)');
grid on; xlim([1 1e4]); title('Y_{qq} = i_q / v_{g,q}');