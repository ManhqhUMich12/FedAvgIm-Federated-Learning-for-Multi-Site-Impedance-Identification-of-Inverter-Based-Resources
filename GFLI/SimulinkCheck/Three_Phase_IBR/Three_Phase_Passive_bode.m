clear all
clc

%% ----- Tham số LCL -----
Rf1 = 3e-3;
Lf1 = 250e-6;
Rf2 = 3e-3;
Lf2 = 20e-6;
Cf  = 50e-6;
w1  = 100*pi;   % 50 Hz
Kppll = 20;
Kipll = 200;
% Tần số: 1 -> 1e4 Hz

%% ----- Mô hình state-space -----
Alcl = [-Rf1/Lf1 w1        0         0       -1/Lf1   0; ...
        -w1      -Rf1/Lf1  0         0        0     -1/Lf1; ...
         0        0       -Rf2/Lf2   w1      1/Lf2   0; ...
         0        0       -w1       -Rf2/Lf2 0      1/Lf2; ...
         1/Cf     0       -1/Cf      0       0       w1; ...
         0        1/Cf     0        -1/Cf   -w1      0];

Blcl = [1/Lf1 0     0      0; ...
        0     1/Lf1 0      0; ...
        0     0    -1/Lf2  0; ...
        0     0     0     -1/Lf2; ...
        0     0     0      0; ...
        0     0     0      0];

Clcl = [0 0 1 0 0 0; ...
        0 0 0 1 0 0];

Dlcl = zeros(2,4);

s = tf('s');
G_lcl   = Clcl * inv(s*eye(6) - Alcl) * Blcl + Dlcl;
Ylcl_ss = -minreal(G_lcl(:,3:4));   % admittance nhìn từ lưới (state-space)

%% ----- Mô hình dq-impedance: Z2 + (Z1 // Zc) -----
I2 = eye(2);
J  = [0 -1; 1 0];

% --- RL trong dq dưới dạng impedance ---
Z1 = (Rf1 + s*Lf1)*I2 + w1*Lf1*J;
Z2 = (Rf2 + s*Lf2)*I2 + w1*Lf2*J;

% --- Chuyển qua admittance (dùng \ thay vì inv) ---
Y1 = Z1 \ I2;     % tương đương inv(Z1)
Y2 = Z2 \ I2;     % tương đương inv(Z2)

% --- Tụ trên dq: đã là admittance ---
Yc = Cf*(s*I2 - w1*J);

% --- Y_in nhìn từ phía lưới: Y2 - Y2*(Y2+Y1+Yc)^(-1)*Y2 ---
M       = Y2 + Y1 + Yc;         % ma trận Y2 + Y1 + Yc
Ylcl_dq = Y2 - Y2 * (M \ Y2);   % M\Y2 ~ inv(M)*Y2

%% ----- Bode: vẽ cả 4 thành phần dd, dq, qd, qq -----
fLoop = logspace(-1,4,1000);   % 0.1 -> 100 Hz
wLoop = 2*pi*fLoop;

labels = {'dd','dq','qd','qq'};
idx    = [1 1; 1 2; 2 1; 2 2];

mag_ss = cell(4,1);  ph_ss = cell(4,1);
mag_dq = cell(4,1);  ph_dq = cell(4,1);

for k = 1:4
    i = idx(k,1); j = idx(k,2);

    [m_ss, p_ss] = bode(Ylcl_ss(i,j), wLoop);
    [m_dq, p_dq] = bode(Ylcl_dq(i,j),  wLoop);

    mag_ss{k} = squeeze(m_ss);
    ph_ss{k}  = squeeze(p_ss);
    mag_dq{k} = squeeze(m_dq);
    ph_dq{k}  = squeeze(p_dq);
end
%% ==== Load admittance Yload trên dq-axis (4 thành phần) ====
Lf2p = Lf2*20;          % L phía lưới sau khi nhân 20

I2   = eye(2);
J    = [0 -1; 1 0];

% đảm bảo dùng cùng w với LCL
w    = wLoop;
nFreq = numel(w);

Yload_dq = zeros(2,2,nFreq);

for k = 1:nFreq
    s = 1j*w(k);
    % Z_load(dq) = (R + sL)I + w1*L*J
    Zk = (Rf2 + s*Lf2p)*I2 + w1*Lf2p*J;

    % Y_load(dq) = Z_load(dq)^{-1}
    Yload_dq(:,:,k) = Zk \ I2;   % dùng backslash thay cho inv(Zk)
end

% Tách 4 thành phần
Ydd = squeeze(Yload_dq(1,1,:));
Ydq = squeeze(Yload_dq(1,2,:));
Yqd = squeeze(Yload_dq(2,1,:));
Yqq = squeeze(Yload_dq(2,2,:));

% Đưa vào cell để đồng bộ với mag_ss, mag_dq, ph_ss, ph_dq
mag_load = cell(4,1);
ph_load  = cell(4,1);

mag_load{1} = abs(Ydd);  ph_load{1} = angle(Ydd)*180/pi;   % dd
mag_load{2} = abs(Ydq);  ph_load{2} = angle(Ydq)*180/pi;   % dq
mag_load{3} = abs(Yqd);  ph_load{3} = angle(Yqd)*180/pi;   % qd
mag_load{4} = abs(Yqq);  ph_load{4} = angle(Yqq)*180/pi;   % qq


%% ----- Plot magnitude (2x2: dd, dq, qd, qq) -----
figure;
for k = 1:4
    subplot(2,2,k);
    % State-space
    semilogx(fLoop, 20*log10(mag_ss{k}), 'LineWidth', 1.4); hold on;
    % dq-impedance (LCL)
    semilogx(fLoop, 20*log10(mag_dq{k}), '--', 'LineWidth', 1.4);
    % dq-impedance (Grid admittance)
    semilogx(fLoop, 20*log10(mag_load{k}), ':', 'LineWidth', 1.2);
    grid on;
    ylabel('|Y(j\omega)| (dB)');
    title(['Y_{' labels{k} '}']);
    if k == 3 || k == 4
        xlabel('Frequency (Hz)');
    end
    if k == 1
        legend('State-space','LCL dq-imp','Grid dq-imp', 'Location','best');
    end
end
sgtitle('Magnitude của 4 thành phần admittance LCL & Grid nhìn từ lưới');

%% ----- Plot phase (2x2: dd, dq, qd, qq) -----
figure;
for k = 1:4
    % wrap phase về [-pi,pi] (radian) rồi đổi lại deg
    ph_ss_rad   = deg2rad(ph_ss{k});
    ph_dq_rad   = deg2rad(ph_dq{k});
    ph_load_rad = deg2rad(ph_load{k});

    ph_ss_wrap   = atan2(sin(ph_ss_rad),   cos(ph_ss_rad));   % [-pi,pi]
    ph_dq_wrap   = atan2(sin(ph_dq_rad),   cos(ph_dq_rad));   % [-pi,pi]
    ph_load_wrap = atan2(sin(ph_load_rad), cos(ph_load_rad)); % [-pi,pi]

    ph_ss_deg   = rad2deg(ph_ss_wrap);
    ph_dq_deg   = rad2deg(ph_dq_wrap);
    ph_load_deg = rad2deg(ph_load_wrap);

    subplot(2,2,k);
    semilogx(fLoop, ph_ss_deg, 'LineWidth', 1.4); hold on;
    semilogx(fLoop, ph_dq_deg, '--', 'LineWidth', 1.4);
    semilogx(fLoop, ph_load_deg, ':', 'LineWidth', 1.2);
    grid on;
    ylabel('Phase (deg)');
    title(['Y_{' labels{k} '}']);
    if k == 3 || k == 4
        xlabel('Frequency (Hz)');
    end
    if k == 1
        legend('State-space','LCL dq-imp','Grid dq-imp', 'Location','best');
    end
end
sgtitle('Phase của 4 thành phần admittance LCL & Grid (wrapped -\pi..\pi)');
