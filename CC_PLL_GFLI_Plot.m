%% === Params như bạn giữ nguyên ở trên ===
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
% PLL
Kppll = 20/(Vdc/2);
Kipll = 200/(Vdc/2);
beta  = 0;

% Điện áp d-axis tại nút (giữ như bạn)
V2d = 575;

%% ====== VẼ 3 ĐƯỜNG VỚI 3 ĐIỂM LÀM VIỆC ======
ops = [ 3195   0;     % (I2d, I2q) = (3195, 0)
        1065   0;     % (1065, 0)
        3195 -2132 ]; % (3195, -2132)

labels = { 'I_{2d}=3195A, I_{2q}=0A', ...
           'I_{2d}=1065A, I_{2q}=0A', ...
           'I_{2d}=3195A, I_{2q}=-2132A' };

styles = {'-','.','+'};   % kiểu nét cho dễ phân biệt

% Tần số: 1 -> 1e4 Hz
fHz = logspace(0,4,400);
w   = 2*pi*fHz;

% Chuẩn bị container dữ liệu cho 4 phần tử Y
mag11_all = []; ph11_all = [];
mag12_all = []; ph12_all = [];
mag21_all = []; ph21_all = [];
mag22_all = []; ph22_all = [];

for k = 1:size(ops,1)
    I2d = ops(k,1);
    I2q = ops(k,2);

    %--- Xây model & lấy ma trận Yvsc2 tại điểm làm việc (I2d,I2q) ---
    Yvsc2 = build_Y_matrix(Vdc, w1, Ts, Rf1, Lf1, Rf2, Lf2, Cf, ...
                           Kpi, Kii, Kppll, Kipll, beta, I2d, I2q, V2d);

    Y11 = minreal(Yvsc2(1,1));
    Y12 = minreal(Yvsc2(1,2));
    Y21 = minreal(Yvsc2(2,1));
    Y22 = minreal(Yvsc2(2,2));

    % Bode data
    [mag11, ph11] = bode(Y11, w); mag11 = squeeze(mag11); ph11 = squeeze(ph11);
    [mag12, ph12] = bode(Y12, w); mag12 = squeeze(mag12); ph12 = squeeze(ph12);
    [mag21, ph21] = bode(Y21, w); mag21 = squeeze(mag21); ph21 = squeeze(ph21);
    [mag22, ph22] = bode(Y22, w); mag22 = squeeze(mag22); ph22 = squeeze(ph22);

    % Wrap phase về [-180, 180] cho các phần tử chéo chéo (giống code gốc)
    ph12 = wrapTo180(ph12);
    ph21 = wrapTo180(ph21);
    ph22 = wrapTo180(ph22); % (có thể giữ nguyên ph11 nếu bạn muốn dạng 0..-360)

    % Lưu lại
    mag11_all(:,k) = mag11; ph11_all(:,k) = ph11;
    mag12_all(:,k) = mag12; ph12_all(:,k) = ph12;
    mag21_all(:,k) = mag21; ph21_all(:,k) = ph21;
    mag22_all(:,k) = mag22; ph22_all(:,k) = ph22;
end

%% ====== VẼ: mỗi window theo format bạn đưa ======
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

% Ydd
ax = nexttile; hold(ax,'on');
ax.XScale = 'log';                % đặt log cho trục x
ax.XLim   = [1 1e4];
ax.XTick  = [1 10 100 1e3 1e4];   % tùy chọn
grid(ax,'on');
yyaxis left;
hL = gobjects(1,size(ops,1));
for k = 1:size(ops,1)
    hL(k) = semilogx(fHz, 20*log10(mag11_all(:,k)), styles{k}, 'LineWidth', 1.2);
end
ylabel('Mag (dB)');
yyaxis right;
for k = 1:size(ops,1)
    semilogx(fHz, ph11_all(:,k), styles{k}, 'LineWidth', 1.0);
end
ylabel('Phase (deg)');
grid on; xlim([1 1e4]); title('Y_{dd} = i_d / v_{g,d}');
legend(hL, labels, 'Location','best'); hold off;

% Ydq
ax = nexttile; hold(ax,'on');
ax.XScale = 'log';                % đặt log cho trục x
ax.XLim   = [1 1e4];
ax.XTick  = [1 10 100 1e3 1e4];   % tùy chọn
grid(ax,'on');
yyaxis left;
hL = gobjects(1,size(ops,1));
for k = 1:size(ops,1)
    hL(k) = semilogx(fHz, 20*log10(mag12_all(:,k)), styles{k}, 'LineWidth', 1.2);
end
ylabel('Mag (dB)');
yyaxis right;
for k = 1:size(ops,1)
    semilogx(fHz, ph12_all(:,k), styles{k}, 'LineWidth', 1.0); % ph12_all đã wrap
end
ylabel('Phase (deg)');
grid on; xlim([1 1e4]); title('Y_{dq} = i_d / v_{g,q}');
legend(hL, labels, 'Location','best'); hold off;

% Yqd
ax = nexttile; hold(ax,'on');
ax.XScale = 'log';                % đặt log cho trục x
ax.XLim   = [1 1e4];
ax.XTick  = [1 10 100 1e3 1e4];   % tùy chọn
grid(ax,'on');
yyaxis left;
hL = gobjects(1,size(ops,1));
for k = 1:size(ops,1)
    hL(k) = semilogx(fHz, 20*log10(mag21_all(:,k)), styles{k}, 'LineWidth', 1.2);
end
ylabel('Mag (dB)');
yyaxis right;
for k = 1:size(ops,1)
    semilogx(fHz, ph21_all(:,k), styles{k}, 'LineWidth', 1.0); % ph21_all đã wrap
end
ylabel('Phase (deg)');
grid on; xlim([1 1e4]); title('Y_{qd} = i_q / v_{g,d}');
legend(hL, labels, 'Location','best'); hold off;

% Yqq
ax = nexttile; hold(ax,'on');
ax.XScale = 'log';                % đặt log cho trục x
ax.XLim   = [1 1e4];
ax.XTick  = [1 10 100 1e3 1e4];   % tùy chọn
grid(ax,'on');
yyaxis left;
hL = gobjects(1,size(ops,1));
for k = 1:size(ops,1)
    hL(k) = semilogx(fHz, 20*log10(mag22_all(:,k)), styles{k}, 'LineWidth', 1.2);
end
ylabel('Mag (dB)');
yyaxis right;
for k = 1:size(ops,1)
    semilogx(fHz, ph22_all(:,k), styles{k}, 'LineWidth', 1.0); % ph22_all đã wrap
end
ylabel('Phase (deg)');
grid on; xlim([1 1e4]); title('Y_{qq} = i_q / v_{g,q}');
legend(hL, labels, 'Location','best'); hold off;

%% ====== HÀM PHỤ: Xây dựng Yvsc2 tại điểm (I2d, I2q) ======
function Yvsc2 = build_Y_matrix(Vdc, w1, Ts, Rf1, Lf1, Rf2, Lf2, Cf, ...
                                Kpi, Kii, Kppll, Kipll, beta, I2d, I2q, V2d)
    % Current control loop
    Ai = zeros(2,2);
    Bi = [1 , 0; 0, 1];
    Ci = (Vdc/2) * [Kii, 0; 0, Kii];
    Di = (Vdc/2) * [Kpi, 0; 0, Kpi];

    % Computational delay
    Td = 1.5*Ts;
    Adel = [0, 1, 0, 0, 0, 0; 0, 0, 1, 0, 0, 0; -120/Td^3, -60/Td^2, -12/Td, 0, 0, 0;...
            0, 0, 0, 0, 1, 0; 0, 0, 0, 0, 0, 1; 0, 0, 0, -120/Td^3, -60/Td^2, -12/Td];
    Bdel = [0, 0; 0, 0; 1, 0; 0, 0; 0, 0; 0, 1];
    Cdel = [240/Td^3 0 24/Td 0 0 0; 0 0 0 240/Td^3 0 24/Td];
    Ddel = [-1 0; 0 -1];

    % LCL-filter
    Alcl = [-Rf1/Lf1 w1 0 0 -1/Lf1 0; -w1 -Rf1/Lf1 0 0 0 -1/Lf1; 0 0 -Rf2/Lf2 w1 1/Lf2 0;...
            0 0 -w1 -Rf2/Lf2 0 1/Lf2; 1/Cf 0 -1/Cf 0 0 w1; 0 1/Cf 0 -1/Cf -w1 0];
    Blcl = [1/Lf1 0 0 0; 0 1/Lf1 0 0; 0 0 -1/Lf2 0; 0 0 0 -1/Lf2; 0 0 0 0; 0 0 0 0];
    Clcl = [0 0 1 0 0 0; 0 0 0 1 0 0];
    Dlcl = zeros(2,4);

    % PLL
    Apll = [0 Kipll; 0 0];
    Bpll = [Kppll; 1];
    Cpll = [1 0];
    Dpll = 0;

    % Inter-connection (phần phụ thuộc I2d, I2q nằm ở đây)
    R3 = [1 0; 0 1; -I2q I2d]';
    R2 = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1; 0 0 0 1];
    R1 = [0 0 0; 0 0 0; 0 0 0; 0 0 -V2d; 0 0 -V2d];
    R4 = zeros(2,4);

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

    % Final state-space representation
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

    s = tf('s');
    G_closed = Cvsc2 * inv(s*eye(16) - Avsc2) * Bvsc2 + Dvsc2;
    Yvsc2 = -minreal(G_closed(:,3:4));
end

% Helper: wrap phase về [-180, 180]
function ph = wrapTo180(ph_deg)
    ph = mod(ph_deg + 180, 360) - 180;
end
