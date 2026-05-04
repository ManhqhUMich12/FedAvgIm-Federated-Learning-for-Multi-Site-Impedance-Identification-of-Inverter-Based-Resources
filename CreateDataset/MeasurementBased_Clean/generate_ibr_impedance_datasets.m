clear;
clc;

%% =========================================================
%  AUTO Y/Z MEASUREMENT FOR 9 FREQUENCY PROFILES x 9 IBR VARIANTS
%
%  Key idea:
%  - The old "case_id" controlled only frequency/OP profiles.
%  - The new "ibr_id" controls model/parameter variations.
%  - You can run either:
%       run_mode = 'pairwise'         -> profile i with IBR i
%       run_mode = 'all_combinations' -> every profile x every IBR
%
%  Notes:
%  - Saved Dataset.X uses pu features:
%       X   = [V2d_pu, P_pu, Q_pu, f_Hz]
%       Y_Y = [Re(Ydd), Im(Ydd), Re(Ydq), Im(Ydq), Re(Yqd), Im(Yqd), Re(Yqq), Im(Yqq)]
%       Y_Z = [Re(Zdd), Im(Zdd), Re(Zdq), Im(Zdq), Re(Zqd), Im(Zqd), Re(Zqq), Im(Zqq)]
%
%  - IMPORTANT mapping assumptions from your request:
%       * "Kpllp, Kplli" -> interpreted as Kppll, Kipll
%       * "R1, L1"       -> interpreted as Rf1, Lf1
%       * IBR4/7/8 use model: GFLI_L.slx
%
%  Output naming:
%       Pairwise mode:
%         gfli_ibr%d_impedance_dataset_new.mat
%         gfli_ibr%d_test_impedance_dataset_new.mat
%
%       All-combinations mode:
%         gfli_profile%d_ibr%d_impedance_dataset_new.mat
%         gfli_profile%d_ibr%d_test_impedance_dataset_new.mat
%% =========================================================

%% ================= USER SETTINGS =================
run_mode    = 'pairwise';   % 'pairwise' or 'all_combinations'
profile_list = 2:9;         % old frequency/OP profiles
ibr_list     = 2:9;         % new IBR variants

% ---- Base values for pu <-> SI conversion ----
base.V_base  = 575;      % [V]
base.S_base  = 2e6;      % [VA]
base.kPQ     = 1.0;      % 1.0 or 1.5 depending on dq power convention
base.Smax_pu = 1.0;      % keep only points with P_pu^2 + Q_pu^2 <= 1

% ---- Timing variables placed into Simulink blocks ----
timing_cfg.Nperiod        = 3;
timing_cfg.t_inj_start    = 0.1;
timing_cfg.t_margin_pert  = 0.5;
timing_cfg.t_margin_sim   = 0.5;

% ---- Train/test split by frequency ----
split_cfg.train_freq_rule = 'odd_index';

%% ================= BASE PARAMETERS =================
base_params = get_base_parameters();

%% ================= BUILD JOB LIST =================
jobs = build_job_list(run_mode, profile_list, ibr_list);

%% ================= MAIN LOOP =================
for ij = 1:numel(jobs)
    profile_id = jobs(ij).profile_id;
    ibr_id     = jobs(ij).ibr_id;

    % ---------- Frequency/OP profile ----------
    c = build_profile_data(profile_id, base);

    % ---------- IBR configuration ----------
    ibr = build_ibr_config(ibr_id, base_params);

    % ---------- Timing ----------
    timing = build_case_timing(c.f_eval_Hz, timing_cfg);

    fprintf('\n====================================================\n');
    fprintf('PROFILE %d | IBR %d | model = %s | %d OPs | %d freqs\n', ...
        profile_id, ibr_id, ibr.mdl, size(c.ops_si,1), numel(c.f_eval_Hz));
    fprintf('Removed %d invalid operating points by power constraint\n', c.n_removed);
    fprintf('Tsig = %.6f s | Tpert = %.6f s | Tsim = %.6f s\n', ...
        timing.Tsig, timing.Tpert, timing.Tsim);
    fprintf('====================================================\n');

    % ---------- Open correct model ----------
    prepare_model(ibr.mdl, timing);
    axisBlk = [ibr.mdl '/AISTool/AxisSlt'];

    % ---------- Push parameters to base workspace ----------
    push_all_parameters_to_base(ibr.params, timing);

    % ---------- Push frequency variables ----------
    fHz_meas = c.f_eval_Hz(:);
    w_meas   = 2*pi*fHz_meas;
    w        = w_meas; %#ok<NASGU>

    assignin('base','fHz_meas', fHz_meas);
    assignin('base','w_meas',   w_meas);
    assignin('base','w',        w);

    % ---------- Preallocate full Dataset arrays ----------
    Nops  = size(c.ops_pu, 1);
    Nfreq = numel(c.f_eval_Hz);
    Nrows = Nops * Nfreq;

    X_full   = zeros(Nrows, 4);
    Y_Y_full = zeros(Nrows, 8);
    Y_Z_full = zeros(Nrows, 8);

    row = 1;

    % ---------- Loop operating points ----------
    for iop = 1:Nops
        % pu operating point for saved Dataset.X
        V2d_pu = c.ops_pu(iop,1);
        P_pu   = c.ops_pu(iop,2);
        Q_pu   = c.ops_pu(iop,3);

        % SI operating point for simulation
        V2d = c.ops_si(iop,1);
        P   = c.ops_si(iop,2);
        Q   = c.ops_si(iop,3);

        [I2d, I2q] = pq_to_idiq(P, Q, V2d, base.kPQ);

        fprintf('Profile %d | IBR %d | OP %3d/%3d | Vpu = %.4f | Ppu = %.4f | Qpu = %.4f\n', ...
            profile_id, ibr_id, iop, Nops, V2d_pu, P_pu, Q_pu);

        assignin('base','V2d',V2d);
        assignin('base','P',P);
        assignin('base','Q',Q);
        assignin('base','I2d',I2d);
        assignin('base','I2q',I2q);

        meas = run_Y_measurement(ibr.mdl, axisBlk, fHz_meas);

        for k = 1:Nfreq
            X_full(row,:) = [V2d_pu, P_pu, Q_pu, fHz_meas(k)];

            Y_Y_full(row,:) = complex2row( ...
                meas.Ydd(k), meas.Ydq(k), meas.Yqd(k), meas.Yqq(k));

            Y_Z_full(row,:) = complex2row( ...
                meas.Zdd(k), meas.Zdq(k), meas.Zqd(k), meas.Zqq(k));

            row = row + 1;
        end
    end

    %% ---------- Train/test split by frequency ----------
    train_freq_idx = 1:2:Nfreq;
    test_freq_idx  = 1:Nfreq;

    train_rows = build_rows_from_freq_idx(Nops, Nfreq, train_freq_idx);
    test_rows  = build_rows_from_freq_idx(Nops, Nfreq, test_freq_idx);

    X_train   = X_full(train_rows, :);
    Y_Y_train = Y_Y_full(train_rows, :);
    Y_Z_train = Y_Z_full(train_rows, :);

    X_test   = X_full(test_rows, :);
    Y_Y_test = Y_Y_full(test_rows, :);
    Y_Z_test = Y_Z_full(test_rows, :);

    %% ---------- Build Dataset structs ----------
    meta_train = build_meta(profile_id, ibr, c, base, timing, train_freq_idx, split_cfg, 'train');
    meta_test  = build_meta(profile_id, ibr, c, base, timing, test_freq_idx,  split_cfg, 'test');

    Dataset = struct('X', X_train, 'Y_Y', Y_Y_train, 'Y_Z', Y_Z_train, 'meta', meta_train); %#ok<NASGU>
    save_name_train = build_save_name(run_mode, profile_id, ibr_id, 'train');
    save(save_name_train, 'Dataset', '-v7.3');
    fprintf('Saved TRAIN dataset to %s\n', save_name_train);

    Dataset = struct('X', X_test, 'Y_Y', Y_Y_test, 'Y_Z', Y_Z_test, 'meta', meta_test); %#ok<NASGU>
    save_name_test = build_save_name(run_mode, profile_id, ibr_id, 'test');
    save(save_name_test, 'Dataset', '-v7.3');
    fprintf('Saved TEST  dataset to %s\n', save_name_test);
end

disp('Done. All requested train/test Dataset files have been saved.');

%% =========================================================
%% LOCAL FUNCTIONS
%% =========================================================

function p = get_base_parameters()
    p = struct();

    p.Tsam = 5e-5;

    p.Vdc = 1150;
    p.Vg  = 575;
    p.f0  = 50;
    p.w1  = 2*pi*p.f0;
    p.fs  = 5e3;
    p.Ts  = 1/p.fs;
    p.Td  = 1.5*p.Ts;
    p.fsw = 5e3;

    % Filter / network parameters
    p.Rf1 = 3e-3;
    p.Lf1 = 250e-6;
    p.Rf2 = 3e-3;
    p.Lf2 = 250e-6;
    p.Cf  = 50e-6;

    % Current controller
    p.Kpi = 1.7391e-4;
    p.Kii = 0.0348;

    % PLL
    p.Kppll = 40/(p.Vdc/2);
    p.Kipll = 400/(p.Vdc/2);
    p.beta  = 0;

    % Grid side
    p.Lg = 250e-6;
    p.Rg = 3e-3;
end

function jobs = build_job_list(run_mode, profile_list, ibr_list)
    switch lower(run_mode)
        case 'pairwise'
            if numel(profile_list) ~= numel(ibr_list)
                error('For pairwise mode, profile_list and ibr_list must have the same length.');
            end
            jobs = struct('profile_id', num2cell(profile_list(:)), ...
                          'ibr_id',     num2cell(ibr_list(:)));

        case 'all_combinations'
            idx = 0;
            jobs = repmat(struct('profile_id',[],'ibr_id',[]), numel(profile_list)*numel(ibr_list), 1);
            for ip = 1:numel(profile_list)
                for ii = 1:numel(ibr_list)
                    idx = idx + 1;
                    jobs(idx).profile_id = profile_list(ip);
                    jobs(idx).ibr_id     = ibr_list(ii);
                end
            end

        otherwise
            error('Unsupported run_mode = %s', run_mode);
    end
end

function prepare_model(mdl, timing)
    mdlFile = [mdl '.slx'];
    if ~bdIsLoaded(mdl)
        open_system(mdlFile);
    end
    set_param(mdl, 'StopTime', 'Tsim');

    assignin('base','Nperiod',       timing.Nperiod);
    assignin('base','t_inj_start',   timing.t_inj_start);
    assignin('base','t_margin_pert', timing.t_margin_pert);
    assignin('base','t_margin_sim',  timing.t_margin_sim);
    assignin('base','Tsig',          timing.Tsig);
    assignin('base','Tpert',         timing.Tpert);
    assignin('base','Tsim',          timing.Tsim);
end

function push_all_parameters_to_base(p, timing)
    %#ok<INUSD>
    names = fieldnames(p);
    for i = 1:numel(names)
        assignin('base', names{i}, p.(names{i}));
    end
end

function ibr = build_ibr_config(ibr_id, p0)
    p = p0;
    mdl = 'GFLI';
    desc = '';

    switch ibr_id
        case 1
            desc = 'Default GFLI';

        case 2
            p.Kpi = 0.9 * p0.Kpi;
            p.Kii = 0.9 * p0.Kii;
            p.desc = 'Same as IBR1, but Kpi = 0.5*Kpi, Kii = 0.5*Kii';

        case 3
            p.Kpi = 1.1 * p0.Kpi;
            p.Kii = 1.1 * p0.Kii;
            p.desc = 'Same as IBR1, but Kpi = 2*Kpi, Kii = 2*Kii';

        case 4
            mdl = 'GFLI_L';
            p.Rf1 = 2.0 * p0.Rf1;
            p.Lf1 = 2.0 * p0.Lf1;
            desc = 'L1 filter only, using GFLI_L';

        case 5
            p.Kppll = 0.95 * p0.Kppll;
            p.Kipll = 0.95 * p0.Kipll;
            p.desc = 'Same as IBR1, but Kppll = 0.95*Kppll, Kipll = 0.95*Kipll';

        case 6
            p.Kppll = 1.05 * p0.Kppll;
            p.Kipll = 1.05 * p0.Kipll;
            p.desc = 'Same as IBR1, but Kppll = 1.05*Kppll, Kipll = 1.05*Kipll';

        case 7
            mdl = 'GFLI_L';
            p.Rf1 = 2.5 * p0.Rf1;
            p.Lf1 = 2.5 * p0.Lf1;
            desc = 'Same as IBR4, but Rf1 = 2*Rf1, Lf1 = 2*Lf1';

        case 8
            mdl = 'GFLI_L';
            p.Kppll = 1.05 * p0.Kppll;
            p.Kipll = 1.05 * p0.Kipll;
            desc = 'Same as IBR4, but Kpi = 2*Kpi, Kii = 2*Kii';

        case 9
            p.Kppll = 0.85 * p0.Kppll;
            p.Kipll = 0.85 * p0.Kipll;
            p.desc = 'Same as IBR1, but Kpi = 0.85*Kpi, Kii = 0.85*Kii';

        otherwise
            error('Unsupported ibr_id = %d', ibr_id);
    end

    ibr = struct();
    ibr.ibr_id      = ibr_id;
    ibr.mdl         = mdl;
    ibr.description = desc;
    ibr.params      = p;
end

function c = build_profile_data(profile_id, base)
    use_custom_ops = false;
    custom_ops_pu  = zeros(0,3);

    V2d_range_pu = [];
    P_range_pu   = [];
    Q_range_pu   = [];
    params_pu    = struct();

    switch profile_id
        case 1
            f_eval_Hz = logspace(0, log10(200), 20).';
            V2d_range_pu = [0.95, 1.00, 1.05];
            P_range_pu   = -1:0.5:1;
            Q_range_pu   = -1:0.5:1;

        case 2
            f_low  = linspace(1, 30, 14);
            f_rest = logspace(log10(30), log10(200), 6);
            f_eval_Hz = [f_low, f_rest].';
            V2d_range_pu = [0.90, 0.95, 1.00];
            P_range_pu   = -1:0.5:1;
            Q_range_pu   = [-0.5, 0, 0.5];

        case 3
            f_low  = logspace(0, log10(30), 6);
            f_mid  = linspace(30, 80, 10);
            f_high = logspace(log10(80), log10(200), 4);
            f_eval_Hz = [f_low, f_mid, f_high].';
            V2d_range_pu = [1.00, 1.05, 1.10];
            P_range_pu   = [-0.5, 0, 0.5];
            Q_range_pu   = -1:0.5:1;

        case 4
            f_low  = logspace(0, log10(80), 6);
            f_high = linspace(80, 200, 14);
            f_eval_Hz = [f_low, f_high].';
            V2d_range_pu = [0.95, 1.00];
            phi = linspace(-pi/8, pi/8, 5);
            S   = 1.0;
            P_pf = S*cos(phi);
            Q_pf = S*sin(phi);

            use_custom_ops = true;
            for v = V2d_range_pu
                custom_ops_pu = [custom_ops_pu; ...
                    [v*ones(numel(phi),1), P_pf(:), Q_pf(:)]]; %#ok<AGROW>
            end

            params_pu.phi  = phi;
            params_pu.S    = S;
            params_pu.P_pf = P_pf;
            params_pu.Q_pf = Q_pf;

        case 5
            rng(5);
            f_eval_Hz = sort(10 .^ (rand(20,1) * log10(200)));
            V2d_range_pu = [0.9, 1.0, 1.1];
            P_range_pu   = [-0.3, 0, 0.3];
            Q_range_pu   = [-1, -0.5, 0, 0.5, 1];

        case 6
            rng(6);
            f_eval_Hz = sort(10 .^ (rand(20,1) * log10(200)));
            V2d_range_pu = [0.95, 1.05];
            P_range_pu   = [-1, -0.5, 0, 0.5, 1];
            Q_range_pu   = [-0.3, 0, 0.3];

        case 7
            f_low  = logspace(0, log10(30), 8);
            f_mid  = linspace(30, 80, 12);
            f_eval_Hz = [f_low, f_mid].';
            V2d_range_pu = [0.9, 1.0];
            pf  = 0.8;
            phi = linspace(-acos(pf), acos(pf), 7);
            S   = 1.0;
            P_pf = S*cos(phi);
            Q_pf = S*sin(phi);

            use_custom_ops = true;
            for v = V2d_range_pu
                custom_ops_pu = [custom_ops_pu; ...
                    [v*ones(numel(phi),1), P_pf(:), Q_pf(:)]]; %#ok<AGROW>
            end

            params_pu.pf   = pf;
            params_pu.phi  = phi;
            params_pu.S    = S;
            params_pu.P_pf = P_pf;
            params_pu.Q_pf = Q_pf;

        case 8
            f_mid  = logspace(log10(30), log10(80), 8);
            f_high = logspace(log10(80), log10(200), 12);
            f_eval_Hz = [f_mid, f_high].';
            V2d_range_pu = [1.0, 1.1];
            phi = linspace(-pi/2, pi/2, 9);
            S   = 1.0;
            P_pf = S*cos(phi);
            Q_pf = S*sin(phi);

            use_custom_ops = true;
            for v = V2d_range_pu
                custom_ops_pu = [custom_ops_pu; ...
                    [v*ones(numel(phi),1), P_pf(:), Q_pf(:)]]; %#ok<AGROW>
            end

            params_pu.phi  = phi;
            params_pu.S    = S;
            params_pu.P_pf = P_pf;
            params_pu.Q_pf = Q_pf;

        case 9
            f_eval_Hz = logspace(0, log10(200), 10).';
            V2d_range_pu = [0.95, 1.05];
            P_candidates = [1.0, 0.8, 0.8, 0.0];
            Q_candidates = [0.0, 0.6, -0.6, 1.0];

            use_custom_ops = true;
            for v = V2d_range_pu
                custom_ops_pu = [custom_ops_pu; ...
                    [v*ones(numel(P_candidates),1), P_candidates(:), Q_candidates(:)]]; %#ok<AGROW>
            end

            params_pu.P_candidates = P_candidates;
            params_pu.Q_candidates = Q_candidates;

        otherwise
            error('Unsupported profile_id = %d', profile_id);
    end

    f_eval_Hz = sanitize_frequency_vector(f_eval_Hz);

    if use_custom_ops
        ops_pu = custom_ops_pu;
    else
        [VV, PP, QQ] = ndgrid(V2d_range_pu, P_range_pu, Q_range_pu);
        ops_pu = [VV(:), PP(:), QQ(:)];
    end

    n_before_filter = size(ops_pu, 1);
    tol = 1e-12;
    mask_valid = (ops_pu(:,2).^2 + ops_pu(:,3).^2) <= (base.Smax_pu^2 + tol);
    ops_pu = ops_pu(mask_valid, :);
    n_after_filter = size(ops_pu, 1);
    n_removed = n_before_filter - n_after_filter;

    ops_si = zeros(size(ops_pu));
    ops_si(:,1) = ops_pu(:,1) * base.V_base;
    ops_si(:,2) = ops_pu(:,2) * base.S_base;
    ops_si(:,3) = ops_pu(:,3) * base.S_base;

    params_pu.use_custom_ops = use_custom_ops;
    params_pu.V2d_range_pu   = V2d_range_pu;
    params_pu.P_range_pu     = P_range_pu;
    params_pu.Q_range_pu     = Q_range_pu;
    params_pu.custom_ops_pu  = custom_ops_pu;

    c = struct();
    c.profile_id      = profile_id;
    c.f_eval_Hz       = f_eval_Hz(:);
    c.ops_pu          = ops_pu;
    c.ops_si          = ops_si;
    c.V2d_range_pu    = V2d_range_pu;
    c.P_range_pu      = P_range_pu;
    c.Q_range_pu      = Q_range_pu;
    c.params_pu       = params_pu;
    c.n_before_filter = n_before_filter;
    c.n_after_filter  = n_after_filter;
    c.n_removed       = n_removed;
end

function timing = build_case_timing(f_eval_Hz, cfg)
    Tsig  = sum(cfg.Nperiod ./ f_eval_Hz(:));
    Tpert = Tsig + cfg.t_margin_pert;
    Tsim  = cfg.t_inj_start + Tpert + cfg.t_margin_sim;

    timing = struct();
    timing.Nperiod       = cfg.Nperiod;
    timing.t_inj_start   = cfg.t_inj_start;
    timing.t_margin_pert = cfg.t_margin_pert;
    timing.t_margin_sim  = cfg.t_margin_sim;
    timing.Tsig          = Tsig;
    timing.Tpert         = Tpert;
    timing.Tsim          = Tsim;
end

function [Id, Iq] = pq_to_idiq(P, Q, Vd, kPQ)
    if abs(Vd) < 1e-12
        error('Vd is too close to zero.');
    end

    Id = P / (kPQ * Vd);
    Iq = -Q / (kPQ * Vd);
end

function meas = run_Y_measurement(mdl, axisBlk, fHz_meas)
    fHz_meas = sanitize_frequency_vector(fHz_meas);
    w_meas   = 2*pi*fHz_meas(:);

    set_param(axisBlk, 'Value', '1');
    sim(mdl);

    sys_estim_Ydd = frestimate(SigGenDataLog_Ydd, w_meas, "rad/s");
    sys_estim_Yqd = frestimate(SigGenDataLog_Yqd, w_meas, "rad/s");

    set_param(axisBlk, 'Value', '0');
    sim(mdl);

    sys_estim_Ydq = frestimate(SigGenDataLog_Ydq, w_meas, "rad/s");
    sys_estim_Yqq = frestimate(SigGenDataLog_Yqq, w_meas, "rad/s");

    Ydd = -squeeze(sys_estim_Ydd.ResponseData(1,1,:));
    Ydq = -squeeze(sys_estim_Ydq.ResponseData(1,1,:));
    Yqd = -squeeze(sys_estim_Yqd.ResponseData(1,1,:));
    Yqq = -squeeze(sys_estim_Yqq.ResponseData(1,1,:));

    Nf = numel(fHz_meas);
    Y = complex(zeros(2,2,Nf));
    Z = complex(nan(2,2,Nf));

    Y(1,1,:) = Ydd;
    Y(1,2,:) = Ydq;
    Y(2,1,:) = Yqd;
    Y(2,2,:) = Yqq;

    for k = 1:Nf
        Yk = Y(:,:,k);
        if rcond(Yk) > 1e-12
            Z(:,:,k) = inv(Yk);
        end
    end

    meas = struct();
    meas.f_Hz = fHz_meas(:);
    meas.w    = w_meas;
    meas.Y    = Y;
    meas.Z    = Z;

    meas.Ydd = Ydd;
    meas.Ydq = Ydq;
    meas.Yqd = Yqd;
    meas.Yqq = Yqq;

    meas.Zdd = squeeze(Z(1,1,:));
    meas.Zdq = squeeze(Z(1,2,:));
    meas.Zqd = squeeze(Z(2,1,:));
    meas.Zqq = squeeze(Z(2,2,:));
end

function row8 = complex2row(a11, a12, a21, a22)
    row8 = [ ...
        real(a11), imag(a11), ...
        real(a12), imag(a12), ...
        real(a21), imag(a21), ...
        real(a22), imag(a22)];
end

function rows = build_rows_from_freq_idx(Nops, Nfreq, freq_idx)
    Nfreq_sel = numel(freq_idx);
    rows = zeros(Nops * Nfreq_sel, 1);

    ptr = 0;
    for op_i = 1:Nops
        base_row = (op_i - 1) * Nfreq;
        idx_this_op = base_row + freq_idx(:);
        rows(ptr + (1:Nfreq_sel)) = idx_this_op;
        ptr = ptr + Nfreq_sel;
    end
end

function meta = build_meta(profile_id, ibr, c, base, timing, freq_idx, split_cfg, split_name)
    meta = struct();

    meta.description = sprintf( ...
        'Profile %d with IBR %d (%s), features: [V2d_pu,P_pu,Q_pu,f_Hz].', ...
        profile_id, ibr.ibr_id, ibr.description);

    meta.created_on = datestr(now, 'dd-mmm-yyyy HH:MM:SS');

    meta.profile_id = profile_id;
    meta.ibr_id     = ibr.ibr_id;
    meta.model_name = ibr.mdl;
    meta.ibr_description = ibr.description;
    meta.ibr_params = ibr.params;

    meta.V2d_range = c.V2d_range_pu;
    meta.P_range   = c.P_range_pu;
    meta.Q_range   = c.Q_range_pu;
    meta.params_pu = c.params_pu;

    meta.N_ops_raw     = c.n_before_filter;
    meta.N_ops_kept    = c.n_after_filter;
    meta.N_ops_removed = c.n_removed;

    if abs(base.Smax_pu - 1.0) < 1e-12
        meta.removal_rule = 'Keep only points with P^2 + Q^2 <= 1 (p.u.)';
    else
        meta.removal_rule = sprintf( ...
            'Keep only points with P^2 + Q^2 <= %.6g^2 (p.u.)', base.Smax_pu);
    end

    meta.f_eval_Hz = c.f_eval_Hz(freq_idx).';
    meta.N_freq    = numel(freq_idx);
    meta.freq_idx  = freq_idx(:);

    meta.split_name      = split_name;
    meta.train_freq_rule = split_cfg.train_freq_rule;

    meta.Nperiod       = timing.Nperiod;
    meta.t_inj_start   = timing.t_inj_start;
    meta.t_margin_pert = timing.t_margin_pert;
    meta.t_margin_sim  = timing.t_margin_sim;
    meta.Tsig          = timing.Tsig;
    meta.Tpert         = timing.Tpert;
    meta.Tsim          = timing.Tsim;

    meta.base_V_base = base.V_base;
    meta.base_S_base = base.S_base;
    meta.base_kPQ    = base.kPQ;
    meta.Smax_pu     = base.Smax_pu;

    meta.X_columns = {'V2d_pu','P_pu','Q_pu','f_Hz'};
    meta.Y_columns = {'ReYdd','ImYdd','ReYdq','ImYdq','ReYqd','ImYqd','ReYqq','ImYqq'};
    meta.Z_columns = {'ReZdd','ImZdd','ReZdq','ImZdq','ReZqd','ImZqd','ReZqq','ImZqq'};
end

function name = build_save_name(run_mode, profile_id, ibr_id, split_name)
    switch lower(run_mode)
        case 'pairwise'
            if strcmpi(split_name, 'train')
                name = sprintf('gfli_ibr%d_impedance_dataset.mat', ibr_id);
            else
                name = sprintf('gfli_ibr%d_test_impedance_dataset.mat', ibr_id);
            end

        case 'all_combinations'
            if strcmpi(split_name, 'train')
                name = sprintf('gfli_profile%d_ibr%d_impedance_dataset.mat', profile_id, ibr_id);
            else
                name = sprintf('gfli_profile%d_ibr%d_test_impedance_dataset.mat', profile_id, ibr_id);
            end

        otherwise
            error('Unsupported run_mode = %s', run_mode);
    end
end

function f = sanitize_frequency_vector(f)
    f = sort(f(:));

    if isempty(f)
        return;
    end

    tol = 1e-10 * max(1, max(abs(f)));
    keep = true(size(f));
    for i = 2:numel(f)
        if abs(f(i) - f(i-1)) <= tol
            keep(i) = false;
        end
    end
    f = f(keep);
end
