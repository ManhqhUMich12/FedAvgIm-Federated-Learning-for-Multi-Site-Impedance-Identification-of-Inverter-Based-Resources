function make_gfli_impedance_datasetnew(ibr_id)
% MAKE_GFLI_IMPEDANCE_DATASETNEW(ibr_id)
%   ibr_id = 1..9
%
%   Sinh ra HAI file:
%     - gfliX_impedance_dataset.mat       : train (lưới tần số thưa hơn)
%     - gfliX_test_impedance_dataset.mat  : test  (lưới tần số dày, full)
%
%   GFLI 1..9: cùng mô hình vật lý, nhưng
%   - f_eval_Hz: phân bố khác nhau (low/mid/high/random…)
%   - V2d_range, P_range, Q_range: khác pattern giữa các client
%
%   Mục tiêu: tạo non-IID nhưng vẫn cùng domain vật lý cho PFL.

clc;

if nargin < 1
    error('Call make_gfli_impedance_datasetnew(ibr_id) with ibr_id = 1..9');
end

%% -------------------- USER CONFIG THEO IBR --------------------
use_custom_ops = false;   % một số case sẽ override bằng ops thủ công
custom_ops     = [];      % [V2d, P, Q] nếu dùng radial PF

switch ibr_id
    % ---------------------------------------------------------------------
    % GFLI1: baseline, full grid (thưa), logspace 1–200 Hz
    % ---------------------------------------------------------------------
    case 1
        f_eval_Hz = logspace(0, log10(200), 20).';   % 1..200 Hz (log)
        V2d_range = [0.95, 1.00, 1.05];
        P_range   = -1:0.5:1;       % [-1 -0.5 0 0.5 1]
        Q_range   = -1:0.5:1;       % [-1 -0.5 0 0.5 1]
        save_path = 'gfli1_impedance_dataset.mat';

    % ---------------------------------------------------------------------
    % GFLI2: low-V, low-f dense (1–30 Hz dense + phần còn lại thưa)
    % ---------------------------------------------------------------------
    case 2
        f_low  = linspace(1, 30, 14);                               % dense 1–30
        f_rest = logspace(log10(30), log10(200), 6);                % thưa 30–200
        f_eval_Hz = unique([f_low, f_rest]);                        % ~20 điểm
        f_eval_Hz = f_eval_Hz(:);
        V2d_range = [0.90, 0.95, 1.00];                             % thấp điện áp
        P_range   = -1:0.5:1;                                       % full P
        Q_range   = [-0.5, 0, 0.5];                                 % Q nhỏ
        save_path = 'gfli2_impedance_dataset.mat';

    % ---------------------------------------------------------------------
    % GFLI3: high-V, mid-f dense (30–80 Hz)
    % ---------------------------------------------------------------------
    case 3
        f_low  = logspace(0, log10(30), 6);                         % thưa 1–30
        f_mid  = linspace(30, 80, 10);                              % dense 30–80
        f_high = logspace(log10(80), log10(200), 4);                % thưa 80–200
        f_eval_Hz = unique([f_low, f_mid, f_high]);
        f_eval_Hz = f_eval_Hz(:);
        V2d_range = [1.00, 1.05, 1.10];                             % over-voltage
        P_range   = [-0.5, 0, 0.5];                                 % P vừa
        Q_range   = -1:0.5:1;                                       % Q full
        save_path = 'gfli3_impedance_dataset.mat';

    % ---------------------------------------------------------------------
    % GFLI4: high-f dense (80–200 Hz), PF ≈ 1 (radial quanh trục P)
    % ---------------------------------------------------------------------
    case 4
        f_low  = logspace(0, log10(80), 6);                         % thưa 1–80
        f_high = linspace(80, 200, 14);                             % dense 80–200
        f_eval_Hz = unique([f_low, f_high]);
        f_eval_Hz = f_eval_Hz(:);

        V2d_range = [0.95, 1.00];                                  % vừa
        % Radial PF ≈ 1: Q nhỏ, P gần ±1
        phi = linspace(-pi/8, pi/8, 5);                            % góc PF hẹp quanh 0
        S   = 1.0;
        P_pf = S*cos(phi);                                         % ~[0.92..1.0]
        Q_pf = S*sin(phi);                                         % nhỏ

        % Tạo ops radial cho PF ~1
        use_custom_ops = true;
        custom_ops = [];
        for v = V2d_range
            custom_ops = [custom_ops; [v*ones(numel(phi),1), P_pf(:), Q_pf(:)]]; %#ok<AGROW>
        end
        save_path = 'gfli4_impedance_dataset.mat';

    % ---------------------------------------------------------------------
    % GFLI5: random logspace (1–200 Hz), Q-dominant (|Q| lớn, |P| nhỏ)
    % ---------------------------------------------------------------------
    case 5
        rng(5);  % fix seed cho reproducible
        f_eval_Hz = 10.^(rand(20,1) * log10(200));   % uniform trong log10(1..200)
        f_eval_Hz = sort(f_eval_Hz);

        V2d_range = [0.9, 1.0, 1.1];

        % P nhỏ, Q lớn: Q-dominant
        P_range = [-0.3, 0, 0.3];
        Q_range = [-1, -0.5, 0, 0.5, 1];
        save_path = 'gfli5_impedance_dataset.mat';

    % ---------------------------------------------------------------------
    % GFLI6: random logspace (1–200 Hz), P-dominant (|P| lớn, |Q| nhỏ)
    % ---------------------------------------------------------------------
    case 6
        rng(6);
        f_eval_Hz = 10.^(rand(20,1) * log10(200));
        f_eval_Hz = sort(f_eval_Hz);

        V2d_range = [0.95, 1.05];

        % P lớn, Q nhỏ: P-dominant
        P_range = [-1, -0.5, 0, 0.5, 1];
        Q_range = [-0.3, 0, 0.3];
        save_path = 'gfli6_impedance_dataset.mat';

    % ---------------------------------------------------------------------
    % GFLI7: low+mid (1–80 Hz), radial PF từ lagging đến leading (~0.8)
    % ---------------------------------------------------------------------
    case 7
        f_low  = logspace(0, log10(30), 8);                         % 1–30
        f_mid  = linspace(30, 80, 12);                              % 30–80
        f_eval_Hz = unique([f_low, f_mid]);
        f_eval_Hz = f_eval_Hz(:);

        V2d_range = [0.9, 1.0];

        % Radial PF từ 0.8 lagging → leading
        pf = 0.8;
        phi = linspace(-acos(pf), acos(pf), 7);   % góc tương ứng PF=0.8
        S   = 1.0;
        P_pf = S*cos(phi);
        Q_pf = S*sin(phi);

        use_custom_ops = true;
        custom_ops = [];
        for v = V2d_range
            custom_ops = [custom_ops; [v*ones(numel(phi),1), P_pf(:), Q_pf(:)]]; %#ok<AGROW>
        end
        save_path = 'gfli7_impedance_dataset.mat';

    % ---------------------------------------------------------------------
    % GFLI8: mid+high (30–200 Hz), radial PF full (lagging ↔ leading)
    % ---------------------------------------------------------------------
    case 8
        f_mid  = logspace(log10(30), log10(80), 8);
        f_high = logspace(log10(80), log10(200), 12);
        f_eval_Hz = unique([f_mid, f_high]);
        f_eval_Hz = f_eval_Hz(:);

        V2d_range = [1.0, 1.1];

        phi = linspace(-pi/2, pi/2, 9);  % full lagging→leading
        S   = 1.0;
        P_pf = S*cos(phi);
        Q_pf = S*sin(phi);

        use_custom_ops = true;
        custom_ops = [];
        for v = V2d_range
            custom_ops = [custom_ops; [v*ones(numel(phi),1), P_pf(:), Q_pf(:)]]; %#ok<AGROW>
        end
        save_path = 'gfli8_impedance_dataset.mat';

    % ---------------------------------------------------------------------
    % GFLI9: full dải nhưng rất thưa (10 f), vài OP “interesting”
    %       → inverter “khó”, ít data
    % ---------------------------------------------------------------------
    case 9
        f_eval_Hz = logspace(0, log10(200), 10).';   % 10 điểm tần số

        V2d_range = [0.95, 1.05];

        % Một số OP "đặc biệt": PF=1, PF=0.8 lag, PF=0.8 lead, Q-only
        P_candidates = [ 1.0, 0.8,  0.8,  0.0];
        Q_candidates = [ 0.0, 0.6, -0.6,  1.0];

        use_custom_ops = true;
        custom_ops = [];
        for v = V2d_range
            custom_ops = [custom_ops; [v*ones(numel(P_candidates),1), ...
                                       P_candidates(:), Q_candidates(:)]]; %#ok<AGROW>
        end
        save_path = 'gfli9_impedance_dataset.mat';

    otherwise
        error('ibr_id phải từ 1 đến 9.');
end

% File test tương ứng: gfliX_test_impedance_dataset.mat
save_path_test = strrep(save_path, '_impedance_dataset.mat', '_test_impedance_dataset.mat');

%% -------------------- PARAMETERS (pu) – GIỮ NGUYÊN --------------------
S3_base = 5e3; VLL_base = 690; Vdc = 2000;
w1  = 100*pi; fs = 5e3; Ts = 1/fs; fsw = 5e3;
Rf1 = 3e-3; Lf1 = 250e-6; Rf2 = 3e-3; Lf2 = 250e-6; Cf = 50e-6;
Kpi = 1.7391e-4; Kii = 0.0348; beta = 0;

Zb = VLL_base^2 / S3_base;
Vb = sqrt(2/3)*VLL_base; Ib = Vb / Zb; m_base = (Vdc/2)/Vb;

Rf1_pu = Rf1 / Zb;  Lf1_pu = Lf1 / Zb;
Rf2_pu = Rf2 / Zb;  Lf2_pu = Lf2 / Zb;
Cf_pu  = Cf  * Zb;

Kpi_pu = m_base*Kpi*Ib;  Kii_pu = m_base*Kii*Ib;

Kppll = 20/(Vdc/2);  Kipll = 200/(Vdc/2);
Kppll_pu = Kppll*Vb; Kipll_pu = Kipll*Vb;

par = struct('w1',w1,'Ts',Ts, ...
    'Rf1_pu',Rf1_pu,'Lf1_pu',Lf1_pu,'Rf2_pu',Rf2_pu,'Lf2_pu',Lf2_pu,'Cf_pu',Cf_pu, ...
    'Kpi_pu',Kpi_pu,'Kii_pu',Kii_pu,'beta',beta, ...
    'Kppll_pu',Kppll_pu,'Kipll_pu',Kipll_pu);

%% -------------------- OP GRID --------------------
f_eval_Hz = f_eval_Hz(:);     % đảm bảo cột
N_freq    = numel(f_eval_Hz);

if ~use_custom_ops
    % Dùng grid V×P×Q + mask P^2+Q^2<=1 (như code gốc)
    [V2d_grid, P_grid, Q_grid] = ndgrid(V2d_range, P_range, Q_range);
    ops_all = [V2d_grid(:), P_grid(:), Q_grid(:)];
else
    % Dùng custom_ops đã build ở trên (radial, interesting OP,…)
    ops_all = custom_ops;
end

% Lọc các điểm over-rated power: P^2 + Q^2 <= 1
S2        = ops_all(:,2).^2 + ops_all(:,3).^2;
mask_ok   = (S2 <= 1 + 1e-12);
ops       = ops_all(mask_ok, :);
removed_count = sum(~mask_ok);

N_ops = size(ops,1);

%% -------------------- PREALLOC CHO FULL GRID (DENSE) --------------------
X   = zeros(N_ops*N_freq, 4);
Y_Y = zeros(N_ops*N_freq, 8);
Y_Z = zeros(N_ops*N_freq, 8);

%% -------------------- MAIN LOOP (FULL FREQUENCY GRID) --------------------
use_parfor = false;   % bạn có thể bật nếu có Parallel Toolbox

if use_parfor
    Xc  = cell(N_ops,1); YYc = cell(N_ops,1); YZc = cell(N_ops,1);
    parfor k = 1:N_ops
        [Xc{k}, YYc{k}, YZc{k}] = one_op_block(ops(k,:), f_eval_Hz, par);
    end
    X   = vertcat(Xc{:});
    Y_Y = vertcat(YYc{:});
    Y_Z = vertcat(YZc{:});
else
    row = 0;
    for k = 1:N_ops
        [Xk, YYk, YZk] = one_op_block(ops(k,:), f_eval_Hz, par);
        idx = row + (1:N_freq);
        X(idx,:)   = Xk;
        Y_Y(idx,:) = YYk;
        Y_Z(idx,:) = YZk;
        row = row + N_freq;
    end
end

%% -------------------- SPLIT TRAIN / TEST THEO FREQUENCY --------------------
% CÁCH 2:
%   - Test: dùng full f_eval_Hz (dense)
%   - Train: dùng lưới thưa hơn theo index frequency
%            (ở đây demo: lấy các index 1,3,5,... N_freq)

train_freq_idx = 1:2:N_freq;        % <-- chỉnh ở đây nếu muốn lưới thưa/dày khác
N_freq_tr = numel(train_freq_idx);

% Tính global row indices cho TRAIN
train_rows = zeros(N_ops * N_freq_tr, 1);
ptr = 0;
for op_i = 1:N_ops
    base = (op_i-1)*N_freq;
    idx_local = base + train_freq_idx;      % các row của OP này dùng cho train
    train_rows(ptr + (1:N_freq_tr)) = idx_local;
    ptr = ptr + N_freq_tr;
end

% TEST dùng full grid: tất cả rows
test_rows = (1:(N_ops*N_freq)).';

% Tạo train / test arrays
X_train   = X(train_rows, :);
Y_Y_train = Y_Y(train_rows, :);
Y_Z_train = Y_Z(train_rows, :);

X_test    = X(test_rows, :);
Y_Y_test  = Y_Y(test_rows, :);
Y_Z_test  = Y_Z(test_rows, :);

%% -------------------- META CHUNG --------------------
meta_common = struct();
meta_common.description   = sprintf('GFLI%d dq-admittance/impedance dataset (features: [V2d_pu,P_pu,Q_pu,f_Hz]).', ibr_id);
meta_common.created_on    = datestr(now);
meta_common.V2d_range     = V2d_range;
if ~use_custom_ops
    meta_common.P_range   = P_range;
    meta_common.Q_range   = Q_range;
else
    meta_common.P_range   = [];
    meta_common.Q_range   = [];
    meta_common.ops_used  = ops;      % lưu lại OP đã dùng
end
meta_common.params_pu     = par;
meta_common.N_ops_raw     = size(ops_all,1);
meta_common.N_ops_kept    = N_ops;
meta_common.N_ops_removed = removed_count;
meta_common.removal_rule  = 'Keep only points with P^2 + Q^2 <= 1 (p.u)';
meta_common.ibr_id        = ibr_id;

% Meta cho TRAIN (freq thưa)
meta_train           = meta_common;
meta_train.f_eval_Hz = f_eval_Hz(train_freq_idx);
meta_train.N_freq    = numel(train_freq_idx);
meta_train.freq_idx  = train_freq_idx(:);   % optional: lưu index trong full grid

% Meta cho TEST (freq dense)
meta_test           = meta_common;
meta_test.f_eval_Hz = f_eval_Hz;
meta_test.N_freq    = N_freq;
meta_test.freq_idx  = (1:N_freq).';

%% -------------------- SAVE 2 FILE --------------------
% Train file: gfliX_impedance_dataset.mat, biến tên Dataset
Dataset = struct('X',X_train,'Y_Y',Y_Y_train,'Y_Z',Y_Z_train,'meta',meta_train); %#ok<NASGU>
save(save_path, 'Dataset', '-v7.3');
fprintf('Saved TRAIN dataset to %s\n', save_path);

% Test file: gfliX_test_impedance_dataset.mat, biến tên Dataset
Dataset = struct('X',X_test,'Y_Y',Y_Y_test,'Y_Z',Y_Z_test,'meta',meta_test); %#ok<NASGU>
save(save_path_test, 'Dataset', '-v7.3');
fprintf('Saved TEST  dataset to %s\n', save_path_test);

fprintf('IBR%d: Kept %d OPs, removed %d OPs (P^2+Q^2>1).\n', ibr_id, N_ops, removed_count);

end

% =====================================================================
% =                            SUBFUNCS                               =
% =====================================================================

function [Xk, YYk, YZk] = one_op_block(op_row, f_eval_Hz, par)
% op_row = [V2d_pu, P_pu, Q_pu]
V2d = op_row(1);
Ppu = op_row(2);
Qpu = op_row(3);

% Quy đổi công suất -> dòng (V2q = 0)
I2d =  Ppu/(1.5*V2d);
I2q = -Qpu/(1.5*V2d);

% Lấy state-space đóng mạch (phụ thuộc OP)
[A,B,C,D] = build_gfli_ss(V2d, I2d, I2q, par);

% Đáp ứng theo tần số
w = 2*pi*f_eval_Hz(:).';
Nf = numel(w);
Y = zeros(2,2,Nf,'like',1i);

I = eye(size(A,1));
for i = 1:Nf
    G = C / (1j*w(i)*I - A) * B + D;    % G(jw)
    % Trong mô hình này, Y_dq(jw) = -G(:,3:4)
    Y(:,:,i) = -G(:,3:4);
end

% Chuyển thành nhãn
Ydd = squeeze(Y(1,1,:)); Ydq = squeeze(Y(1,2,:));
Yqd = squeeze(Y(2,1,:)); Yqq = squeeze(Y(2,2,:));

% Z = inv(Y) cho từng tần số
Z = zeros(2,2,Nf,'like',1i);
for i = 1:Nf
    Ydd_i = Y(1,1,i); Ydq_i = Y(1,2,i); Yqd_i = Y(2,1,i); Yqq_i = Y(2,2,i);
    detY = Ydd_i*Yqq_i - Ydq_i*Yqd_i;
    if abs(detY) < 1e-12
        Z(:,:,i) = NaN(2);   % tránh singular
    else
        Z(:,:,i) = (1/detY) * [ Yqq_i, -Ydq_i; -Yqd_i, Ydd_i ];
    end
end
Zdd = squeeze(Z(1,1,:)); Zdq = squeeze(Z(1,2,:));
Zqd = squeeze(Z(2,1,:)); Zqq = squeeze(Z(2,2,:));

% Gói block
Xk  = [repmat(V2d,Nf,1), repmat(Ppu,Nf,1), repmat(Qpu,Nf,1), f_eval_Hz];
YYk = [real(Ydd), imag(Ydd), real(Ydq), imag(Ydq), real(Yqd), imag(Yqd), real(Yqq), imag(Yqq)];
YZk = [real(Zdd), imag(Zdd), real(Zdq), imag(Zdq), real(Zqd), imag(Zqd), real(Zqq), imag(Zqq)];
end

function [Avsc2, Bvsc2, Cvsc2, Dvsc2] = build_gfli_ss(V2d_pu, I2d_pu, I2q_pu, p)

w1 = p.w1; Ts = p.Ts;

Ai = zeros(2,2);
Bi = eye(2);
Ci = [p.Kii_pu 0; 0 p.Kii_pu];
Di = [p.Kpi_pu 0; 0 p.Kpi_pu];

Td = 1.5*Ts;
Adel = [0,1,0,0,0,0;
        0,0,1,0,0,0;
        -120/Td^3,-60/Td^2,-12/Td,0,0,0;
        0,0,0,0,1,0;
        0,0,0,0,0,1;
        0,0,0,-120/Td^3,-60/Td^2,-12/Td];
Bdel = [0,0; 0,0; 1,0; 0,0; 0,0; 0,1];
Cdel = [240/Td^3,0,24/Td,0,0,0;
        0,0,0,240/Td^3,0,24/Td];
Ddel = [-1,0; 0,-1];

Alcl = [ -p.Rf1_pu/p.Lf1_pu,  w1,                   0,                  0,          -1/p.Lf1_pu,      0;
         -w1,                -p.Rf1_pu/p.Lf1_pu,    0,                  0,           0,               -1/p.Lf1_pu;
          0,                  0,                   -p.Rf2_pu/p.Lf2_pu,  w1,           1/p.Lf2_pu,       0;
          0,                  0,                   -w1,                -p.Rf2_pu/p.Lf2_pu, 0,          1/p.Lf2_pu;
          1/p.Cf_pu,          0,                   -1/p.Cf_pu,         0,           0,                 w1;
          0,                  1/p.Cf_pu,            0,                -1/p.Cf_pu,  -w1,                0 ];
Blcl = [ 1/p.Lf1_pu, 0,        0,         0;
         0,          1/p.Lf1_pu, 0,       0;
         0,          0,       -1/p.Lf2_pu,  0;
         0,          0,        0,        -1/p.Lf2_pu;
         0,          0,        0,         0;
         0,          0,        0,         0];
Clcl = [0 0 1 0 0 0;
        0 0 0 1 0 0];
Dlcl = zeros(2,4);

Apll = [0 p.Kipll_pu; 0 0];
Bpll = [p.Kppll_pu; 1];
Cpll = [1 0];
Dpll = 0;

R3 = [1 0; 0 1; -I2q_pu I2d_pu]';
R2 = [1 0 0 0;
      0 1 0 0;
      0 0 1 0;
      0 0 0 1;
      0 0 0 1];
R1 = [0 0 0;
      0 0 0;
      0 0 0;
      0 0 -V2d_pu;
      0 0 -V2d_pu];
R4 = zeros(2,4);

R_3 = [0 0 0 0 -1 0;
       0 0 0 0  0 -1;
       1 0 0 0  0 -w1*(p.Lf1_pu + p.Lf2_pu);
       0 1 0 0  w1*(p.Lf1_pu + p.Lf2_pu) 0;
       0 0 1 0  0 0;
       0 0 0 1  0 0;
       0 0 0 0  0 0;
       0 0 0 0  0 0];
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

E6 = eye(6);
Avsc = Ast + Bst*R_3/(E6 - Dst*R_3)*Cst;
Bvsc = Bst*R_3/(E6 - Dst*R_3)*Dst*R_2 + Bst*R_2;
Cvsc = R_1/(E6 - Dst*R_3)*Cst;
Dvsc = R_1/(E6 - Dst*R_3)*Dst*R_2 + R_0;

Ast1 = blkdiag(Avsc, Apll);
Bst1 = blkdiag(Bvsc, Bpll);
Cst1 = blkdiag(Cvsc, Cpll);
Dst1 = blkdiag(Dvsc, Dpll);

E3 = eye(3);
Avsc2 = Ast1 + Bst1*R1/(E3 - Dst1*R1)*Cst1;
Bvsc2 = Bst1*R1/(E3 - Dst1*R1)*Dst1*R2 + Bst1*R2;
Cvsc2 = R3/(E3 - Dst1*R1)*Cst1;
Dvsc2 = R3/(E3 - Dst1*R1)*Dst1*R2 + R4;
end


