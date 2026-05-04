clear all
clc
%% Circuit and controller parameters of the GFLI
Vdc = 1150;     %dc-voltage
Vg  = 575;     % grid voltage
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
Kppll = 40/(Vdc/2);
Kipll = 400/(Vdc/2);
beta  = 0;
%Operating point
I2d = 3478.26;
% I2d = 2782.61;
% I2q = 2086.96;
I2q = 0;
V2d = 575;
Lg  = 250e-6;          
Rg = 3e-3;

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
Yvsc = -minreal(G_closed(:,3:4));

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

I2  = eye(2);
J   = [0 -1; 1 0];

nFreq = numel(w);
Yg_dq = zeros(2,2,nFreq);

for k = 1:nFreq
    s_k = 1j*w(k);
    % Zg(dq) = (R + sL)I + w1*L*J
    Zg_k = (Rg + s_k*Lg)*I2 + w1*Lg*J;
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

% %% ----- Plot magnitude & phase (2x2: dd, dq, qd, qq) -----
% labels = {'dd','dq','qd','qq'};
% 
% tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
% 
% for k = 1:4
%     % wrap phase IBR
%     ph_ibr_rad = deg2rad(ph_ibr{k});
%     ph_ibr_wr  = atan2(sin(ph_ibr_rad), cos(ph_ibr_rad));   % [-pi,pi]
%     ph_ibr_deg = rad2deg(ph_ibr_wr);
% 
%     % wrap phase Grid
%     ph_g_rad = deg2rad(ph_g{k});
%     ph_g_wr  = atan2(sin(ph_g_rad), cos(ph_g_rad));         % [-pi,pi]
%     ph_g_deg = rad2deg(ph_g_wr);
% 
%     nexttile;
% 
%     yyaxis left;
%     semilogx(fHz, 20*log10(mag_ibr{k}), 'LineWidth', 1.2); hold on;
%     semilogx(fHz, 20*log10(mag_g{k}), '--', 'LineWidth', 1.0);
%     ylabel('Mag (dB)');
% 
%     yyaxis right;
%     semilogx(fHz, ph_ibr_deg, 'LineWidth', 1.2); hold on;
%     semilogx(fHz, ph_g_deg,   '--', 'LineWidth', 1.0);
%     ylabel('Phase (deg)');
% 
%     grid on; xlim([1 1e4]);
%     title(['Y_{' labels{k} '}']);
% 
%     if k == 3 || k == 4
%         xlabel('Frequency (Hz)');
%     end
%     if k == 1
%         legend('IBR','Grid (dq)','Location','best');
%     end
% end
% 
% sgtitle('Admittance trên dq-axis: IBR vs Grid');




%% ===== IEEE-TSG style GNC Nyquist (3.5 in × 1.75 in) =====
% Assumes you already have:
%   - Yvsc : 2x2 transfer function (dq admittance of IBR)
%   - Rg, Lg, w1 : grid parameters
% Output:
%   - Figure 1: eigenloci of L(jw)
%   - Figure 2: Nyquist of det(I+L)

%% -------------------- Frequency grid --------------------
fGNC = logspace(log10(1), log10(200), 20);
wGNC = 2*pi*fGNC;
nGNC = numel(wGNC);

%% -------------------- Frequency response --------------------
Yibr_resp = freqresp(Yvsc, wGNC);   % 2 x 2 x nGNC (complex)

I2 = eye(2);
J  = [0 -1; 1 0];

Yg_resp = zeros(2,2,nGNC);
for k = 1:nGNC
    s_k = 1j*wGNC(k);
    Zg_k = (Rg + s_k*Lg)*I2 + w1*Lg*J;  % Zg(dq)
    Yg_resp(:,:,k) = Zg_k \ I2;         % Yg(dq) = inv(Zg)
end

%% -------------------- Loop matrix + eigenvalues --------------------
lambda1   = zeros(1, nGNC);
lambda2   = zeros(1, nGNC);
detIplusL = zeros(1, nGNC);

prev_lam = [];  % for continuity tracking

for k = 1:nGNC
    Yibr_k = Yibr_resp(:,:,k);
    Yg_k   = Yg_resp(:,:,k);

    Lk  = Yibr_k / Yg_k;         % return ratio matrix
    lam = eig(Lk);               % 2 eigenvalues
    detIplusL(k) = det(I2 + Lk);

    % ---- enforce continuous ordering of eigenvalues (avoid swapping) ----
    if k == 1
        % initialize: sort by real part (or magnitude) for consistency
        [~, idx] = sort(real(lam), 'descend');
        lam = lam(idx);
    else
        % match current eigenvalues to previous by nearest distance
        d11 = abs(lam(1) - prev_lam(1));  d12 = abs(lam(1) - prev_lam(2));
        d21 = abs(lam(2) - prev_lam(1));  d22 = abs(lam(2) - prev_lam(2));
        if (d11 + d22) <= (d12 + d21)
            % keep order
        else
            lam = flipud(lam);
        end
    end

    lambda1(k) = lam(1);
    lambda2(k) = lam(2);
    prev_lam   = lam;
end

%% -------------------- Build full Nyquist (+w and -w) --------------------
lambda1_all = [conj(fliplr(lambda1)), lambda1];
lambda2_all = [conj(fliplr(lambda2)), lambda2];

det_all = [conj(fliplr(detIplusL)), detIplusL];

% %% ==================== FIGURE 1: Eigenloci ====================
% fig1 = figure('Color','w');
% apply_ieee_tsg_style(fig1, 3.5, 1.5);   % width=3.5in, height=1.75in
% ax1 = axes(fig1); hold(ax1,'on'); grid(ax1,'on'); box(ax1,'on'); axis(ax1,'equal');
% 
% p1 = plot(ax1, real(lambda1_all), imag(lambda1_all), '-',  'LineWidth', 1.5);
% p2 = plot(ax1, real(lambda2_all), imag(lambda2_all), '--', 'LineWidth', 1.5);
% p3 = plot(ax1, -1, 0, 'x', 'LineWidth', 1.5, 'MarkerSize', 6);
% 
% xlabel(ax1,'$\Re\{\lambda(L(j\omega))\}$','Interpreter','latex');
% ylabel(ax1,'$\Im\{\lambda(L(j\omega))\}$','Interpreter','latex');
% 
% % (khuyến nghị) tick cũng dùng latex cho đồng bộ
% set(ax1,'TickLabelInterpreter','latex');
% legend(ax1,[p1 p2 p3], {'$\lambda_1$','$\lambda_2$','$-1+j0$'}, ...
%     'Interpreter','latex','Location','best','Box','off');
% 
% % Export (vector PDF/EPS recommended for IEEE)
% exportgraphics(fig1,'GNC_Eigenloci_IEEE.pdf','ContentType','vector');

%% ==================== FIGURE 1: Eigenloci ====================
fig1 = figure('Color','w', 'Units','inches', 'Position',[1 1 0.8 0.8]);

ax1 = axes(fig1);
set(ax1, 'Units','normalized', 'Position',[0.18 0.18 0.76 0.76]);

hold(ax1,'on'); 
grid(ax1,'on'); 
box(ax1,'on');

p1 = plot(ax1, real(lambda1_all), imag(lambda1_all), '-',  'LineWidth', 1.0);
p2 = plot(ax1, real(lambda2_all), imag(lambda2_all), '--', 'LineWidth', 1.0);
p3 = plot(ax1, -1, 0, 'x', 'LineWidth', 1.0, 'MarkerSize', 4);

xlim(ax1, [-1.2 -0.8]);
ylim(ax1, [-0.2 0.2]);

axis(ax1,'square');

set(ax1,'FontName','Times New Roman','FontSize',6);
set(ax1,'TickLabelInterpreter','latex');

exportgraphics(fig1,'GNC_Eigenloci_IEEE.pdf','ContentType','vector');


%% ==================== FIGURE 2: det(I+L) ====================
fig2 = figure('Color','w');
apply_ieee_tsg_style(fig2, 3.5, 3.5/2);
ax2 = axes(fig2); hold(ax2,'on'); grid(ax2,'on'); box(ax2,'on'); axis(ax2,'equal');

plot(ax2, real(det_all), imag(det_all), '-', 'LineWidth', 1.1);
plot(ax2, 0, 0, 'x', 'LineWidth', 1.2, 'MarkerSize', 6);

xlabel(ax2,'$\mathrm{Re}\{\det(I+L(j\omega))\}$','Interpreter','latex');
ylabel(ax2,'$\mathrm{Im}\{\det(I+L(j\omega))\}$','Interpreter','latex');

exportgraphics(fig2,'GNC_detIplusL_IEEE.pdf','ContentType','vector');

%% ==================== Helper: IEEE-TSG style ====================
function apply_ieee_tsg_style(fig, w_in, h_in)
    set(fig, 'Units','inches', 'Position',[1 1 w_in h_in]);
    set(fig, 'PaperUnits','inches', 'PaperPosition',[0 0 w_in h_in], ...
             'PaperSize',[w_in h_in]);

    % IEEE-like typography (common choice)
    set(findall(fig,'-property','FontName'),'FontName','Times New Roman');
    set(findall(fig,'-property','FontSize'),'FontSize',8);

    % Axis cosmetics
    ax = findall(fig,'Type','axes');
    set(ax, 'LineWidth',0.75, 'TickDir','out', 'TickLength',[0.02 0.02], ...
            'XMinorTick','on','YMinorTick','on', 'GridAlpha',0.15, 'MinorGridAlpha',0.08);
end

function add_direction_arrows(ax, z, n_arrows)
    z = z(:);
    N = numel(z);
    if N < (n_arrows+2), return; end

    idx = round(linspace(2, N-1, n_arrows));
    for i = 1:numel(idx)
        k = idx(i);
        p0 = z(k-1); p1 = z(k);
        v  = p1 - p0;
        if abs(v) < 1e-12, continue; end
        v  = v / abs(v);     % normalize direction only

        % short arrow for clean look
        quiver(ax, real(p0), imag(p0), 0.10*real(v), 0.10*imag(v), 0, ...
            'LineWidth',0.75, 'MaxHeadSize',1.5);
    end
end

%% ===== Zg(dq) viết trực tiếp (2x2) và poles của L(s) =====
s = tf('s');

Zscalar = Rg + Lg*s;      % R + Ls
jTerm   = w1*Lg;          % w1*L

% Ma trận impedance dq: v_dq = Zg * i_dq
Zg_tf = [ Zscalar,   -jTerm;
          jTerm,      Zscalar ];

Zg_ss = ss(Zg_tf);        % state-space của Zg(s)
sysYvsc = ss(Yvsc);       % state-space của admittance VSC: v_dq -> i_dq

% Return ratio L(s) = Yvsc(s) * Zg(s)
L_ss = sysYvsc * Zg_ss;   % v_dq -> i_dq -> v_dq

% Cực của L(s)
poles_L = pole(L_ss);
disp('Poles of L(s) = Yvsc(s)*Zg(s):');
disp(poles_L);

% Cực bên phải (nếu có)
RHP_poles = poles_L(real(poles_L) > 0);
disp('RHP poles of L(s):');
disp(RHP_poles);
