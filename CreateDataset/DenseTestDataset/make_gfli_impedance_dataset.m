function make_gfli_impedance_dataset(ibr_id)
% MAKE_GFLI_IMPEDANCE_DATASETNEW(ibr_id)
%   ibr_id = 1..9
%
%   Sinh ra HAI file:
%     - gfliX_impedance_dataset.mat       : TRAIN (OP thưa + freq thưa)
%     - gfliX_test_impedance_dataset.mat  : TEST  (OP dày hơn + freq full)
%
%   Mục tiêu: test khó hơn nhưng vẫn cùng domain vật lý.
%   Thiết kế: ops_train ⊂ ops_test, f_train ⊂ f_test.

clc;

if nargin < 1
    error('Call make_gfli_impedance_datasetnew(ibr_id) with ibr_id = 1..9');
end

%% -------------------- USER CONFIG --------------------
use_parfor = false;      % bật nếu có Parallel Toolbox
train_freq_stride = 2;   % train lấy 1 điểm / stride (vd 2 => 1,3,5,...)

%% -------------------- SWITCH THEO IBR --------------------
use_custom_ops = false;

% --- TRAIN ranges (nếu grid) ---
V2d_range_tr = [];
P_range_tr   = [];
Q_range_tr   = [];

% --- TEST ranges (nếu grid) ---
V2d_range_te = [];
P_range_te   = [];
Q_range_te   = [];

% --- custom ops (nếu radial/interesting) ---
custom_ops_tr = [];
custom_ops_te = [];

switch ibr_id
    % ---------------------------------------------------------------------
    % GFLI1: baseline
    % ---------------------------------------------------------------------
    case 1
        f_eval_Hz_full = logspace(0, log10(200), 20).';   % 1..200 Hz (log)
        V2d_range_tr = [0.95, 1.00, 1.05];
        P_range_tr   = -1:0.5:1;
        Q_range_tr   = -1:0.5:1;

        % TEST OP dày hơn
        V2d_range_te = densify_lin(V2d_range_tr, max(7, 2*numel(V2d_range_tr)+1));
        P_range_te   = densify_lin(P_range_tr,   max(17, 2*numel(P_range_tr)+1));
        Q_range_te   = densify_lin(Q_range_tr,   max(17, 2*numel(Q_range_tr)+1));

        save_path = 'gfli1_impedance_dataset.mat';

    % ---------------------------------------------------------------------
    % GFLI2: low-V, low-f dense
    % ---------------------------------------------------------------------
    case 2
        f_low  = linspace(1, 30, 14);
        f_rest = logspace(log10(30), log10(200), 6);
        f_eval_Hz_full = unique([f_low, f_rest]).';
        f_eval_Hz_full = f_eval_Hz_full(:);

        V2d_range_tr = [0.90, 0.95, 1.00];
        P_range_tr   = -1:0.5:1;
        Q_range_tr   = [-0.5, 0, 0.5];

        % TEST OP dày hơn
        V2d_range_te = densify_lin(V2d_range_tr, max(7, 2*numel(V2d_range_tr)+1));
        P_range_te   = densify_lin(P_range_tr,   max(17, 2*numel(P_range_tr)+1));
        Q_range_te   = densify_lin(Q_range_tr,   max(9,  2*numel(Q_range_tr)+1));

        save_path = 'gfli2_impedance_dataset.mat';

    % ---------------------------------------------------------------------
    % GFLI3: high-V, mid-f dense (30–80 Hz)
    % ---------------------------------------------------------------------
    case 3
        f_low  = logspace(0, log10(30), 6);
        f_mid  = linspace(30, 80, 10);
        f_high = logspace(log10(80), log10(200), 4);
        f_eval_Hz_full = unique([f_low, f_mid, f_high]).';
        f_eval_Hz_full = f_eval_Hz_full(:);

        V2d_range_tr = [1.00, 1.05, 1.10];
        P_range_tr   = [-0.5, 0, 0.5];
        Q_range_tr   = -1:0.5:1;

        % TEST OP dày hơn
        V2d_range_te = densify_lin(V2d_range_tr, max(7, 2*numel(V2d_range_tr)+1));
        P_range_te   = densify_lin(P_range_tr,   max(9,  2*numel(P_range_tr)+1));
        Q_range_te   = densify_lin(Q_range_tr,   max(17, 2*numel(Q_range_tr)+1));

        save_path = 'gfli3_impedance_dataset.mat';

    % ---------------------------------------------------------------------
    % GFLI4: high-f dense (80–200 Hz), PF ≈ 1 radial
    % ---------------------------------------------------------------------
    case 4
        f_low  = logspace(0, log10(80), 6);
        f_high = linspace(80, 200, 14);
        f_eval_Hz_full = unique([f_low, f_high]).';
        f_eval_Hz_full = f_eval_Hz_full(:);

        use_custom_ops = true;

        V2d_range_tr = [0.95, 1.00];

        % TRAIN: phi thưa
        phi_tr = linspace(-pi/8, pi/8, 5);
        S   = 1.0;
        P_pf_tr = S*cos(phi_tr);
        Q_pf_tr = S*sin(phi_tr);

        custom_ops_tr = [];
        for v = V2d_range_tr
            custom_ops_tr = [custom_ops_tr; [v*ones(numel(phi_tr),1), P_pf_tr(:), Q_pf_tr(:)]]; %#ok<AGROW>
        end

        % TEST: phi dày hơn + V dày hơn
        V2d_range_te = densify_lin(V2d_range_tr, 5);
        phi_te = linspace(-pi/8, pi/8, 13);
        P_pf_te = S*cos(phi_te);
        Q_pf_te = S*sin(phi_te);

        custom_ops_te = [];
        for v = V2d_range_te(:).'
            custom_ops_te = [custom_ops_te; [v*ones(numel(phi_te),1), P_pf_te(:), Q_pf_te(:)]];
        end


        save_path = 'gfli4_impedance_dataset.mat';

    % ---------------------------------------------------------------------
    % GFLI5: random logspace, Q-dominant
    % ---------------------------------------------------------------------
    case 5
        rng(5);
        f_eval_Hz_full = 10.^(rand(20,1) * log10(200));
        f_eval_Hz_full = sort(f_eval_Hz_full(:));

        V2d_range_tr = [0.9, 1.0, 1.1];
        P_range_tr   = [-0.3, 0, 0.3];
        Q_range_tr   = [-1, -0.5, 0, 0.5, 1];

        % TEST OP dày hơn (mượt theo min/max)
        V2d_range_te = densify_lin(V2d_range_tr, 7);
        P_range_te   = densify_lin(P_range_tr,   9);
        Q_range_te   = densify_lin(Q_range_tr,   17);

        save_path = 'gfli5_impedance_dataset.mat';

    % ---------------------------------------------------------------------
    % GFLI6: random logspace, P-dominant
    % ---------------------------------------------------------------------
    case 6
        rng(6);
        f_eval_Hz_full = 10.^(rand(20,1) * log10(200));
        f_eval_Hz_full = sort(f_eval_Hz_full(:));

        V2d_range_tr = [0.95, 1.05];
        P_range_tr   = [-1, -0.5, 0, 0.5, 1];
        Q_range_tr   = [-0.3, 0, 0.3];

        V2d_range_te = densify_lin(V2d_range_tr, 7);
        P_range_te   = densify_lin(P_range_tr,   17);
        Q_range_te   = densify_lin(Q_range_tr,   9);

        save_path = 'gfli6_impedance_dataset.mat';

    % ---------------------------------------------------------------------
    % GFLI7: low+mid (1–80 Hz), PF~0.8 lagging→leading radial
    % ---------------------------------------------------------------------
    case 7
        f_low  = logspace(0, log10(30), 8);
        f_mid  = linspace(30, 80, 12);
        f_eval_Hz_full = unique([f_low, f_mid]).';
        f_eval_Hz_full = f_eval_Hz_full(:);

        use_custom_ops = true;

        V2d_range_tr = [0.9, 1.0];

        pf = 0.8;
        phi_tr = linspace(-acos(pf), acos(pf), 7);
        S = 1.0;
        P_pf_tr = S*cos(phi_tr);
        Q_pf_tr = S*sin(phi_tr);

        custom_ops_tr = [];
        for v = V2d_range_tr
            custom_ops_tr = [custom_ops_tr; [v*ones(numel(phi_tr),1), P_pf_tr(:), Q_pf_tr(:)]]; %#ok<AGROW>
        end

        % TEST: V dày hơn + phi dày hơn
        V2d_range_te = densify_lin(V2d_range_tr, 5);
        phi_te = linspace(-acos(pf), acos(pf), 15);
        P_pf_te = S*cos(phi_te);
        Q_pf_te = S*sin(phi_te);

        custom_ops_te = [];
        for v = V2d_range_te(:).'
            custom_ops_te = [custom_ops_te; [v*ones(numel(phi_te),1), P_pf_te(:), Q_pf_te(:)]];
        end


        save_path = 'gfli7_impedance_dataset.mat';

    % ---------------------------------------------------------------------
    % GFLI8: mid+high (30–200 Hz), PF full radial
    % ---------------------------------------------------------------------
    case 8
        f_mid  = logspace(log10(30), log10(80), 8);
        f_high = logspace(log10(80), log10(200), 12);
        f_eval_Hz_full = unique([f_mid, f_high]).';
        f_eval_Hz_full = f_eval_Hz_full(:);

        use_custom_ops = true;

        V2d_range_tr = [1.0, 1.1];

        phi_tr = linspace(-pi/2, pi/2, 9);
        S = 1.0;
        P_pf_tr = S*cos(phi_tr);
        Q_pf_tr = S*sin(phi_tr);

        custom_ops_tr = [];
        for v = V2d_range_tr
            custom_ops_tr = [custom_ops_tr; [v*ones(numel(phi_tr),1), P_pf_tr(:), Q_pf_tr(:)]]; %#ok<AGROW>
        end

        % TEST: V dày hơn + phi dày hơn
        V2d_range_te = densify_lin(V2d_range_tr, 5);
        phi_te = linspace(-pi/2, pi/2, 21);
        P_pf_te = S*cos(phi_te);
        Q_pf_te = S*sin(phi_te);

        custom_ops_te = [];
        for v = V2d_range_te(:).'
            custom_ops_te = [custom_ops_te; [v*ones(numel(phi_te),1), P_pf_te(:), Q_pf_te(:)]];
        end


        save_path = 'gfli8_impedance_dataset.mat';

    % ---------------------------------------------------------------------
    % GFLI9: very sparse freq, few “interesting” OP
    % ---------------------------------------------------------------------
    case 9
        f_eval_Hz_full = logspace(0, log10(200), 10).';

        use_custom_ops = true;

        V2d_range_tr = [0.95, 1.05];

        % TRAIN: 4 OP/voltage
        P_candidates_tr = [ 1.0, 0.8,  0.8,  0.0];
        Q_candidates_tr = [ 0.0, 0.6, -0.6,  1.0];

        custom_ops_tr = [];
        for v = V2d_range_tr
            custom_ops_tr = [custom_ops_tr; [v*ones(numel(P_candidates_tr),1), ...
                                             P_candidates_tr(:), Q_candidates_tr(:)]]; %#ok<AGROW>
        end

        % TEST: dày hơn bằng cách thêm nhiều mức PF & Q-only hơn + V dày hơn
        V2d_range_te = densify_lin(V2d_range_tr, 5);

        % PF set dày hơn
        pf_list = [1.0, 0.95, 0.9, 0.85, 0.8];
        phi = [];
        for pf = pf_list
            phi = [phi; -acos(pf); 0; acos(pf)]; %#ok<AGROW>
        end
        phi = unique(phi);
        S = 1.0;
        P_pf_te = S*cos(phi);
        Q_pf_te = S*sin(phi);

        % thêm Q-only levels
        P_extra = zeros(5,1);
        Q_extra = [-1; -0.5; 0; 0.5; 1];

        custom_ops_te = [];
        for v = V2d_range_te(:).'
            ops_pf    = [v*ones(numel(P_pf_te),1), P_pf_te(:), Q_pf_te(:)];
            ops_qonly = [v*ones(numel(P_extra),1), P_extra, Q_extra];
            custom_ops_te = [custom_ops_te; ops_pf; ops_qonly]; %#ok<AGROW>
        end


        save_path = 'gfli9_impedance_dataset.mat';

    otherwise
        error('ibr_id phải từ 1 đến 9.');
end

save_path_test = strrep(save_path, '_impedance_dataset.mat', '_test_impedance_dataset.mat');

%% -------------------- PARAMETERS (pu) – GIỮ NGUYÊN --------------------
S3_base = 5e3; VLL_base = 690; Vdc = 2000;
w1  = 100*pi; fs = 5e3; Ts = 1/fs; fsw = 5e3; %#ok<NASGU>
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

%% -------------------- BUILD OPS TRAIN / TEST --------------------
if ~use_custom_ops
    % --- TRAIN ops ---
    [Vt, Pt, Qt] = ndgrid(V2d_range_tr, P_range_tr, Q_range_tr);
    ops_all_tr = [Vt(:), Pt(:), Qt(:)];

    % --- TEST ops (dày hơn) ---
    [Ve, Pe, Qe] = ndgrid(V2d_range_te, P_range_te, Q_range_te);
    ops_all_te = [Ve(:), Pe(:), Qe(:)];
else
    ops_all_tr = custom_ops_tr;
    ops_all_te = custom_ops_te;

    if isempty(ops_all_te)
        ops_all_te = ops_all_tr; % fallback an toàn
    end
end

% Filter power constraint: P^2+Q^2<=1
[ops_train, removed_tr] = filter_ops_unit_circle(ops_all_tr);
[ops_test,  removed_te] = filter_ops_unit_circle(ops_all_te);

N_ops_tr = size(ops_train,1);
N_ops_te = size(ops_test,1);

%% -------------------- BUILD FREQ TRAIN / TEST --------------------
f_test = f_eval_Hz_full(:);
N_freq_full = numel(f_test);

train_freq_idx = 1:train_freq_stride:N_freq_full;
f_train = f_test(train_freq_idx);

%% -------------------- BUILD DATASETS --------------------
[X_train, Y_Y_train, Y_Z_train] = build_dataset(ops_train, f_train, par, use_parfor);
[X_test,  Y_Y_test,  Y_Z_test ] = build_dataset(ops_test,  f_test,  par, use_parfor);

%% -------------------- META --------------------
meta_common = struct();
meta_common.description   = sprintf('GFLI%d dq-admittance/impedance dataset (features: [V2d_pu,P_pu,Q_pu,f_Hz]).', ibr_id);
meta_common.created_on    = datestr(now);
meta_common.params_pu     = par;
meta_common.ibr_id        = ibr_id;
meta_common.removal_rule  = 'Keep only points with P^2 + Q^2 <= 1 (p.u)';

% TRAIN meta
meta_train = meta_common;
meta_train.f_eval_Hz   = f_train(:);
meta_train.N_freq      = numel(f_train);
meta_train.freq_idx_in_full = train_freq_idx(:);
meta_train.ops_used    = ops_train;
meta_train.N_ops_raw   = size(ops_all_tr,1);
meta_train.N_ops_kept  = N_ops_tr;
meta_train.N_ops_removed = removed_tr;

if ~use_custom_ops
    meta_train.V2d_range = V2d_range_tr;
    meta_train.P_range   = P_range_tr;
    meta_train.Q_range   = Q_range_tr;
end

% TEST meta
meta_test = meta_common;
meta_test.f_eval_Hz   = f_test(:);
meta_test.N_freq      = N_freq_full;
meta_test.freq_idx_in_full = (1:N_freq_full).';
meta_test.ops_used    = ops_test;
meta_test.N_ops_raw   = size(ops_all_te,1);
meta_test.N_ops_kept  = N_ops_te;
meta_test.N_ops_removed = removed_te;

if ~use_custom_ops
    meta_test.V2d_range = V2d_range_te;
    meta_test.P_range   = P_range_te;
    meta_test.Q_range   = Q_range_te;
end

%% -------------------- SAVE --------------------
Dataset = struct('X',X_train,'Y_Y',Y_Y_train,'Y_Z',Y_Z_train,'meta',meta_train); %#ok<NASGU>
save(save_path, 'Dataset', '-v7.3');
fprintf('Saved TRAIN dataset to %s\n', save_path);

Dataset = struct('X',X_test,'Y_Y',Y_Y_test,'Y_Z',Y_Z_test,'meta',meta_test); %#ok<NASGU>
save(save_path_test, 'Dataset', '-v7.3');
fprintf('Saved TEST  dataset to %s\n', save_path_test);

fprintf('IBR%d:\n', ibr_id);
fprintf('  TRAIN: Kept %d OPs, removed %d OPs (P^2+Q^2>1). Freq=%d\n', N_ops_tr, removed_tr, numel(f_train));
fprintf('  TEST : Kept %d OPs, removed %d OPs (P^2+Q^2>1). Freq=%d\n', N_ops_te, removed_te, numel(f_test));

end

% =====================================================================
% =                            SUBFUNCS                               =
% =====================================================================

function v = densify_lin(v_in, n_out)
% Densify vector bằng linspace(min,max,n_out), giữ trong [-1,1] nếu cần
v_in = unique(v_in(:).');
if numel(v_in) == 1
    v = v_in(:);
    return;
end
vmin = min(v_in); vmax = max(v_in);
v = linspace(vmin, vmax, n_out);
v = unique(v(:));
end

function [ops, removed_count] = filter_ops_unit_circle(ops_all)
S2 = ops_all(:,2).^2 + ops_all(:,3).^2;
mask_ok = (S2 <= 1 + 1e-12);
ops = ops_all(mask_ok, :);
removed_count = sum(~mask_ok);
end

function [X, Y_Y, Y_Z] = build_dataset(ops, f_eval_Hz, par, use_parfor)
f_eval_Hz = f_eval_Hz(:);
N_ops  = size(ops,1);
N_freq = numel(f_eval_Hz);

X   = zeros(N_ops*N_freq, 4);
Y_Y = zeros(N_ops*N_freq, 8);
Y_Z = zeros(N_ops*N_freq, 8);

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
end

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
        Z(:,:,i) = NaN(2);
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
% for i = 1:9 
% make_gfli_impedance_dataset(i); 
% end