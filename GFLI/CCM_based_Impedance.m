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

% ==== Grid admittance Yg from Zg = Rf2 + j*w*Lf2*50 ====
Zg   = Rf2 + 1j*w*Lf2*20;   % grid impedance
Yg   = 1 ./ Zg;             % grid admittance
mag_g = abs(Yg);            % magnitude
ph_g  = angle(Yg)*180/pi;   % phase in degrees


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
% Ydd
yyaxis left;
semilogx(fHz, 20*log10(mag11), 'LineWidth', 1.2); hold on;
semilogx(fHz, 20*log10(mag_g), '--', 'LineWidth', 1.0); % Grid admittance
ylabel('Mag (dB)');

yyaxis right;
semilogx(fHz, ph11, 'LineWidth', 1.0); hold on;
semilogx(fHz, ph_g, '--', 'LineWidth', 1.0);           % Grid phase
ylabel('Phase (deg)');

grid on; xlim([1 1e4]);
title('Y_{dd} = i_d / v_{g,d}');
legend('IBR','Grid');


% Ydq
nexttile;
yyaxis left;  semilogx(fHz, 20*log10(mag12), 'LineWidth', 1.2); ylabel('Mag (dB)');
yyaxis right; semilogx(fHz, ph12_wrap,            'LineWidth', 1.0);  ylabel('Phase (deg)');
grid on; xlim([1 1e4]); title('Y_{dq} = i_d / v_{g,q}');
yyaxis left;
semilogx(fHz, 20*log10(mag12), 'LineWidth', 1.2); hold on;
semilogx(fHz, 20*log10(mag_g), '--', 'LineWidth', 1.0); % Grid admittance
ylabel('Mag (dB)');

yyaxis right;
semilogx(fHz, ph12_wrap, 'LineWidth', 1.0); hold on;
semilogx(fHz, ph_g, '--', 'LineWidth', 1.0);            % Grid phase
ylabel('Phase (deg)');

grid on; xlim([1 1e4]);
title('Y_{dq} = i_d / v_{g,q}');
legend('IBR','Grid');

% Yqd
nexttile;
yyaxis left;  semilogx(fHz, 20*log10(mag21), 'LineWidth', 1.2); ylabel('Mag (dB)');
yyaxis right; semilogx(fHz, ph21_wrap,            'LineWidth', 1.0);  ylabel('Phase (deg)');
grid on; xlim([1 1e4]); title('Y_{qd} = i_q / v_{g,d}');
yyaxis left;
semilogx(fHz, 20*log10(mag21), 'LineWidth', 1.2); hold on;
semilogx(fHz, 20*log10(mag_g), '--', 'LineWidth', 1.0); % Grid admittance
ylabel('Mag (dB)');

yyaxis right;
semilogx(fHz, ph21_wrap, 'LineWidth', 1.0); hold on;
semilogx(fHz, ph_g, '--', 'LineWidth', 1.0);            % Grid phase
ylabel('Phase (deg)');

grid on; xlim([1 1e4]);
title('Y_{qd} = i_q / v_{g,d}');
legend('IBR','Grid');

% Yqq
nexttile;
yyaxis left;  semilogx(fHz, 20*log10(mag22), 'LineWidth', 1.2); ylabel('Mag (dB)');
yyaxis right; semilogx(fHz, ph22,            'LineWidth', 1.0);  ylabel('Phase (deg)');
grid on; xlim([1 1e4]); title('Y_{qq} = i_q / v_{g,q}');
yyaxis left;
semilogx(fHz, 20*log10(mag22), 'LineWidth', 1.2); hold on;
semilogx(fHz, 20*log10(mag_g), '--', 'LineWidth', 1.0); % Grid admittance
ylabel('Mag (dB)');

yyaxis right;
semilogx(fHz, ph22, 'LineWidth', 1.0); hold on;
semilogx(fHz, ph_g, '--', 'LineWidth', 1.0);            % Grid phase
ylabel('Phase (deg)');

grid on; xlim([1 1e4]);
title('Y_{qq} = i_q / v_{g,q}');
legend('IBR','Grid');

%% ----- Loop gain / impedance ratio (dd, qq channels) -----
%% ----- Grid admittance as TF -----
s = tf('s');
Zg_tf = Rf2 + s*Lf2*20;   % Zg(s) = Rf2 + s*Lf2*50
Yg_tf = 1/Zg_tf;          % Yg(s)

%% ----- Loop gain / impedance ratio (dd, qq) -----
L_dd = minreal( Yg_tf / Y11 );   % L(s) = Yg / Yvsc
L_qq = minreal( Yg_tf / Y22 );

% Dải tần khảo sát
fLoop = logspace(-1,2,500);   % 0.1 -> 100 Hz
wLoop = 2*pi*fLoop;

[magLdd, phLdd] = bode(L_dd, wLoop);
magLdd = squeeze(magLdd);  phLdd = squeeze(phLdd);
[magLqq, phLqq] = bode(L_qq, wLoop);
magLqq = squeeze(magLqq);  phLqq = squeeze(phLqq);

% ----- Plot -----
figure;
subplot(2,1,1);
semilogx(fLoop, 20*log10(magLdd), 'LineWidth', 1.4); hold on;
semilogx(fLoop, 20*log10(magLqq), '--', 'LineWidth', 1.4);
yline(0, 'k:');         % 0 dB
xline(2, 'r:');         % đánh dấu 2 Hz
grid on;
ylabel('|L(j\omega)| (dB)');
legend('L_{dd}','L_{qq}','0 dB','2 Hz','Location','best');
title('Loop gain / impedance ratio');

subplot(2,1,2);
semilogx(fLoop, phLdd, 'LineWidth', 1.4); hold on;
semilogx(fLoop, phLqq, '--', 'LineWidth', 1.4);
yline(-180, 'k:');      % -180 deg
xline(2, 'r:');
grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (deg)');
legend('L_{dd}','L_{qq}','-180^\circ','2 Hz','Location','best');

%% ===== GNC plot (Generalized Nyquist Criterion) =====
% Return ratio matrix L(s) = Yg * inv(Yvsc)
s = tf('s');
Zg_tf = Rf2 + s*Lf2*20;      % Zg(s)
Yg_tf = 1/Zg_tf;             % Yg(s) scalar

Lmat = minreal( Yg_tf * inv(Yvsc) );   % 2x2 return-ratio matrix

% Dải tần GNC (có thể mở rộng nếu muốn)
fGNC = logspace(-1, 2, 400);     % 0.1 -> 100 Hz
wGNC = 2*pi*fGNC;

% Tính đáp ứng tần số ma trận L(jw)
Lw = freqresp(Lmat, wGNC);       % size: [2 x 2 x N]

lambda1 = zeros(1, numel(wGNC));
lambda2 = zeros(1, numel(wGNC));
detIplusL = zeros(1, numel(wGNC));

for k = 1:numel(wGNC)
    Ak = Lw(:,:,k);             % 2x2 complex matrix at this freq
    
    % Eigenvalues (GNC eigenloci)
    lam = eig(Ak);
    lambda1(k) = lam(1);
    lambda2(k) = lam(2);
    
    % det(I + L) for scalar Nyquist
    detIplusL(k) = det( eye(2) + Ak );
end

% Eigenloci:
lambda1_pos = lambda1;                 % đã tính cho w > 0
lambda2_pos = lambda2;

lambda1_neg = conj(fliplr(lambda1_pos));  % tương ứng w < 0
lambda2_neg = conj(fliplr(lambda2_pos));

lambda1_all = [lambda1_neg, lambda1_pos];
lambda2_all = [lambda2_neg, lambda2_pos];

figure;
plot(real(lambda1_all), imag(lambda1_all), 'b-', 'LineWidth', 1.5); hold on;
plot(real(lambda2_all), imag(lambda2_all), 'r--', 'LineWidth', 1.5);
plot(-1, 0, 'kx', 'MarkerSize', 10, 'LineWidth', 2);
grid on; axis equal;
xlabel('Re\{\lambda(L(j\omega))\}');
ylabel('Im\{\lambda(L(j\omega))\}');
title('GNC eigenloci (both +\omega and -\omega)');
legend('\lambda_1','\lambda_2','-1+j0','Location','best');

% det(I+L):
det_pos = detIplusL;
det_neg = conj(fliplr(det_pos));
det_all = [det_neg, det_pos];

figure;
plot(real(det_all), imag(det_all), 'LineWidth', 1.5); hold on;
plot(0,0,'rx','MarkerSize',10,'LineWidth',2);
grid on; axis equal;
xlabel('Re\{\det(I+L(j\omega))\}');
ylabel('Im\{\det(I+L(j\omega))\}');
title('GNC Nyquist of det(I+L) (both +\omega and -\omega)');