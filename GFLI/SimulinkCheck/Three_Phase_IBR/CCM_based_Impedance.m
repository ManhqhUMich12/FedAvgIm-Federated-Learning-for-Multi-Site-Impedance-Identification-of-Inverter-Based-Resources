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
Lf1 = 250e-6;
Rf2 = 3e-3;
Lf2 = 250e-6;
Cf  = 50e-6;

% Current controller
Kpi = 1.7391e-4;
Kii = 0.0348;
% Kpi = 8.6955e-4;
% Kii = 0.17391;
% Kpi = 1.7391e-3;
% Kii = 0.348;
%PLL
% Kppll = 20;
% Kipll = 200;
Kppll = 0;
Kipll = 0;
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
Yvsc = -minreal(G_closed(:,3:4));   % 2x2 admittance i_{dq} / v_{g,dq}

% Tách 4 phần tử SISO
Y11 = minreal(Yvsc(1,1));   % Ydd
Y12 = minreal(Yvsc(1,2));   % Ydq
Y21 = minreal(Yvsc(2,1));   % Yqd
Y22 = minreal(Yvsc(2,2));   % Yqq

% Tần số: 1 -> 1e4 Hz
fHz = logspace(0,4,600);
w   = 2*pi*fHz;

% ==== Bode của Yvsc (IBR admittance) ====
mag_ibr = cell(4,1);
ph_ibr  = cell(4,1);

[mag11, ph11] = bode(Y11, w); mag11 = squeeze(mag11); ph11 = squeeze(ph11);
[mag12, ph12] = bode(Y12, w); mag12 = squeeze(mag12); ph12 = squeeze(ph12);
[mag21, ph21] = bode(Y21, w); mag21 = squeeze(mag21); ph21 = squeeze(ph21);
[mag22, ph22] = bode(Y22, w); mag22 = squeeze(mag22); ph22 = squeeze(ph22);

mag_ibr{1} = mag11; ph_ibr{1} = ph11;  % dd
mag_ibr{2} = mag12; ph_ibr{2} = ph12;  % dq
mag_ibr{3} = mag21; ph_ibr{3} = ph21;  % qd
mag_ibr{4} = mag22; ph_ibr{4} = ph22;  % qq

%% ==== Grid admittance Yg trên dq-axis (4 thành phần) ====
Lg  = Lf2;          % L phía lưới sau khi nhân 20
I2  = eye(2);
J   = [0 -1; 1 0];

nFreq = numel(w);
Yg_dq = zeros(2,2,nFreq);

for k = 1:nFreq
    s_k = 1j*w(k);
    % Zg(dq) = (R + sL)I + w1*L*J
    Zg_k = (Rf2 + s_k*Lg)*I2 + w1*Lg*J;
    % Yg(dq) = Zg(dq)^{-1}
    Yg_dq(:,:,k) = Zg_k \ I2;
end

Yg11 = squeeze(Yg_dq(1,1,:));
Yg12 = squeeze(Yg_dq(1,2,:));
Yg21 = squeeze(Yg_dq(2,1,:));
Yg22 = squeeze(Yg_dq(2,2,:));

mag_g = cell(4,1);
ph_g  = cell(4,1);

mag_g{1} = abs(Yg11); ph_g{1} = angle(Yg11)*180/pi;   % dd
mag_g{2} = abs(Yg12); ph_g{2} = angle(Yg12)*180/pi;   % dq
mag_g{3} = abs(Yg21); ph_g{3} = angle(Yg21)*180/pi;   % qd
mag_g{4} = abs(Yg22); ph_g{4} = angle(Yg22)*180/pi;   % qq

%% ----- Plot magnitude & phase (2x2: dd, dq, qd, qq) -----
labels = {'dd','dq','qd','qq'};

tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

for k = 1:4
    % wrap phase IBR
    ph_ibr_rad = deg2rad(ph_ibr{k});
    ph_ibr_wr  = atan2(sin(ph_ibr_rad), cos(ph_ibr_rad));   % [-pi,pi]
    ph_ibr_deg = rad2deg(ph_ibr_wr);

    % wrap phase Grid
    ph_g_rad = deg2rad(ph_g{k});
    ph_g_wr  = atan2(sin(ph_g_rad), cos(ph_g_rad));         % [-pi,pi]
    ph_g_deg = rad2deg(ph_g_wr);

    nexttile;

    yyaxis left;
    semilogx(fHz, 20*log10(mag_ibr{k}), 'LineWidth', 1.2); hold on;
    semilogx(fHz, 20*log10(mag_g{k}), '--', 'LineWidth', 1.0);
    ylabel('Mag (dB)');

    yyaxis right;
    semilogx(fHz, ph_ibr_deg, 'LineWidth', 1.2); hold on;
    semilogx(fHz, ph_g_deg,   '--', 'LineWidth', 1.0);
    ylabel('Phase (deg)');

    grid on; xlim([1 1e4]);
    title(['Y_{' labels{k} '}']);

    if k == 3 || k == 4
        xlabel('Frequency (Hz)');
    end
    if k == 1
        legend('IBR','Grid (dq)','Location','best');
    end
end

sgtitle('Admittance trên dq-axis: IBR vs Grid');




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