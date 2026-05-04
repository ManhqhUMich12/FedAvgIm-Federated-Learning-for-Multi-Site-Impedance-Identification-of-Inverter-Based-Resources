function make_gfli_impedance_dataset()
% MAKE_GFLI_IMPEDANCE_DATASET
% GFLI 1: Original value -> ok
% GFLI 2: 0.5x current control bandwidth-> ok
% GFLI 3: 2x current control bandwidth-> ok
% GFLI 4: 0.5x PLL control bandwidth-> ok
% GFLI 5: 1.5x PLL control bandwidth-> ok
% GFLI 6: 2x PLL control bandwidth-> ok
% GFLI 7: 0.5x L1, R1-> ok
% GFLI 8: 2x L1, R1-> ok
% GFLI 9: 0.5x Cf-> ok

%GFLI10_Test: LC
clc;

%% -------------------- USER CONFIG --------------------
f_eval_Hz = logspace(0, log10(200), 30).';   % 1..200 Hz
V2d_range = linspace(0.9, 1.1, 3);            % p.u.
P_range   = -1:1:1;                         % p.u.
Q_range   = -1:0.5:1;                         % p.u.


save_path  = 'gfli10_impedance_dataset10.mat';
use_parfor = false;                           % bật nếu có Parallel Toolbox

%% -------------------- PARAMETERS (pu) --------------------

S3_base = 5e3; VLL_base = 690; Vdc = 2000;
w1  = 100*pi; fs = 5e3; Ts = 1/fs; fsw = 5e3;
Rf1 = 3e-3; Lf1 = 250e-6; Rf2 = 0.5*3e-3; Lf2 = 0.5*250e-6; Cf = 50e-6;
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
[V2d_grid, P_grid, Q_grid] = ndgrid(V2d_range, P_range, Q_range);
ops_all = [V2d_grid(:), P_grid(:), Q_grid(:)];

% === NEW: Lọc các điểm over-rated power: P^2 + Q^2 <= 1 ===
S2 = ops_all(:,2).^2 + ops_all(:,3).^2;     % P^2 + Q^2
mask_ok = (S2 <= 1 + 1e-12);                % dung sai nhỏ để tránh lỗi số
ops = ops_all(mask_ok, :);
removed_count = sum(~mask_ok);

N_ops  = size(ops,1);
N_freq = numel(f_eval_Hz);

%% -------------------- PREALLOC --------------------
X   = zeros(N_ops*N_freq, 4);
Y_Y = zeros(N_ops*N_freq, 8);
Y_Z = zeros(N_ops*N_freq, 8);

%% -------------------- MAIN --------------------
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

%% -------------------- SAVE --------------------
meta = struct();
meta.description   = 'GFLI dq-admittance/impedance dataset (features: [V2d_pu,P_pu,Q_pu,f_Hz]).';
meta.created_on    = datestr(now);
meta.f_eval_Hz     = f_eval_Hz;
meta.V2d_range     = V2d_range;
meta.P_range       = P_range;
meta.Q_range       = Q_range;
meta.params_pu     = par;
meta.N_ops_raw     = size(ops_all,1);       % <--- NEW
meta.N_ops_kept    = N_ops;                 % <--- NEW
meta.N_ops_removed = removed_count;         % <--- NEW
meta.removal_rule  = 'Keep only points with P^2 + Q^2 <= 1 (p.u)'; % <--- NEW
meta.N_freq        = N_freq;

Dataset = struct('X',X,'Y_Y',Y_Y,'Y_Z',Y_Z,'meta',meta); %#ok<NASGU>
save(save_path, 'Dataset', '-v7.3');
fprintf('Saved dataset to %s\n', save_path);
fprintf('Kept %d OPs, removed %d OPs (P^2+Q^2>1).\n', N_ops, removed_count);

end % ===== end main =====

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
% Dựng state-space cho GFLI ở per-unit, theo mô tả bạn cung cấp
% (V2d_pu đóng vai trò giống "Vgd" — điều kiện V2q=0)

w1 = p.w1; Ts = p.Ts;

% Current PI (dq)
Ai = zeros(2,2);
Bi = eye(2);
Ci = [p.Kii_pu 0; 0 p.Kii_pu];
Di = [p.Kpi_pu 0; 0 p.Kpi_pu];

% Computational delay (6th-order approx.)
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

% LCL filter in pu
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

% PLL (pu)
Apll = [0 p.Kipll_pu; 0 0];
Bpll = [p.Kppll_pu; 1];
Cpll = [1 0];
Dpll = 0;

% Interconnections (pu) — theo bạn cung cấp
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

% Stack lần 1
Ast = blkdiag(Ai, Adel, Alcl);
Bst = blkdiag(Bi, Bdel, Blcl);
Cst = blkdiag(Ci, Cdel, Clcl);
Dst = blkdiag(Di, Ddel, Dlcl);

% Dùng phép chia ma trận thay vì inv() để ổn định số
E6 = eye(6);
Avsc = Ast + Bst*R_3/(E6 - Dst*R_3)*Cst;
Bvsc = Bst*R_3/(E6 - Dst*R_3)*Dst*R_2 + Bst*R_2;
Cvsc = R_1/(E6 - Dst*R_3)*Cst;
Dvsc = R_1/(E6 - Dst*R_3)*Dst*R_2 + R_0;

% Stack với PLL
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
