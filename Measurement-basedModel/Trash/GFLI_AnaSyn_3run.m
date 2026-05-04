clear; clc; close all;

%% =========================================================
% USER SETTINGS
% =========================================================
nSeeds = 100;
seedList = 1:nSeeds;

% Number of repeated measurements per operating point
nMeasRepeats = 3;

% Cache
forceRerunMeasurement = false;
cacheFile = '1percent_ops_cache_3OP_avgErr3runs_magPhaseOnly.mat';

% Acceptance criteria
acceptCfg.metricBandPct = [5, 95];
acceptCfg.minCoverage   = 0.6;

% Plot flags
plotCfg.makeSummaryFigure              = true;
plotCfg.makeErrorBandFigures           = true;
plotCfg.makeRepresentativeBodeFigures  = true;
plotCfg.makeMeasurementRepeatFigures   = true;

%% =========================================================
% FIXED PARAMETERS
% =========================================================
p.Tsam = 5e-5;
p.Vdc  = 1150;
p.Vg   = 575;
p.w1   = 100*pi;
p.fs   = 5e3;
p.Ts   = 1/p.fs;
p.Td   = 1.5*p.Ts;
p.fsw  = 5e3;

% LCL filter
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

% Grid
p.Lg = 250e-6;
p.Rg = 3e-3;

%% =========================================================
% NOISE OPTIONS FOR SYNTHETIC ANALYTICAL DATA
% =========================================================
noiseOpt.enable = true;
noiseOpt.seed   = 1;

noiseOpt.rel_sigma_low  = 0.001;
noiseOpt.rel_sigma_high = 0.15;

noiseOpt.abs_floor_ratio_low  = 0.005;
noiseOpt.abs_floor_ratio_high = 0.03;

noiseOpt.fc = 15;
noiseOpt.p  = 4;

noiseOpt.bias_mag_high = 0.1;
noiseOpt.bias_ph_high  = 6.0;
noiseOpt.bias_fc       = 40;
noiseOpt.random_bias_sign = false;

%% =========================================================
% SIMULINK MODEL
% =========================================================
mdl = "GFLI.slx";
open_system(mdl);
axisBlk = 'GFLI/AISTool/AxisSlt';

%% =========================================================
% FREQUENCY VECTORS
% =========================================================
fHz_meas = logspace(log10(1), log10(200), 20);
w_meas   = 2*pi*fHz_meas;

fHz_ana  = logspace(log10(1), log10(200), 20);
w_ana    = 2*pi*fHz_ana;

%% =========================================================
% OPERATING POINTS
% =========================================================
opsSpec(1).name = 'OP1';
opsSpec(1).I2d  = 3195;
opsSpec(1).I2q  = 0;

opsSpec(2).name = 'OP2';
opsSpec(2).I2d  = 1065;
opsSpec(2).I2q  = 0;

opsSpec(3).name = 'OP3';
opsSpec(3).I2d  = 3915;
opsSpec(3).I2q  = -2132;

nOP = numel(opsSpec);

%% =========================================================
% STEP 1: LOAD OR RUN CLEAN MEASURED OPERATING POINTS
% =========================================================
if ~forceRerunMeasurement && exist(cacheFile, 'file')
    S = load(cacheFile, 'opsClean');
    opsClean = S.opsClean;
    fprintf('\nLoaded clean OP cache from "%s"\n', cacheFile);
else
    fprintf('\nRunning expensive measurement for %d operating points...\n', nOP);
    fprintf('Each OP will be measured %d times.\n', nMeasRepeats);

    opsClean = struct([]);

    for k = 1:nOP
        fprintf('  Measuring %s: I2d = %.1f, I2q = %.1f\n', ...
            opsSpec(k).name, opsSpec(k).I2d, opsSpec(k).I2q);

        op = run_one_op_clean_repeated(mdl, axisBlk, ...
            opsSpec(k).I2d, opsSpec(k).I2q, ...
            p, fHz_meas, w_meas, fHz_ana, w_ana, nMeasRepeats);

        op.name = opsSpec(k).name;

        if isempty(opsClean)
            opsClean = repmat(op, nOP, 1);
        end

        opsClean(k) = op;
    end

    save(cacheFile, 'opsClean');
    fprintf('Saved clean OP cache to "%s"\n', cacheFile);
end

%% =========================================================
% STEP 2: SYNTHETIC VALIDATION OVER 100 SEEDS
% =========================================================
results = repmat(struct(), nOP, 1);

for k = 1:nOP
    fprintf('\nProcessing %s (I2d = %.1f, I2q = %.1f)\n', ...
        opsClean(k).name, opsClean(k).I2d, opsClean(k).I2q);

    op = opsClean(k);
    nF = numel(op.fHz_ana);

    % ---------- measured metrics from 3 repeated runs
    measYdd = compute_error_metrics_from_runs(op.Ydd_ana, op.Ydd_meas_runs);
    measYqq = compute_error_metrics_from_runs(op.Yqq_ana, op.Yqq_meas_runs);

    % ---------- synthetic distributions
    syn.Ydd.mag_rmse = zeros(nSeeds,1);
    syn.Ydd.ph_rmse  = zeros(nSeeds,1);
    syn.Ydd.emag     = zeros(nSeeds, nF);
    syn.Ydd.eph      = zeros(nSeeds, nF);

    syn.Yqq.mag_rmse = zeros(nSeeds,1);
    syn.Yqq.ph_rmse  = zeros(nSeeds,1);
    syn.Yqq.emag     = zeros(nSeeds, nF);
    syn.Yqq.eph      = zeros(nSeeds, nF);

    % representative seed: closest in mag/phase RMSE only
    bestSeed  = seedList(1);
    bestDist  = inf;
    bestSynOp = [];

    for sIdx = 1:nSeeds
        seed = seedList(sIdx);

        opSyn = apply_noise_to_op(op, noiseOpt, seed);

        synYdd = compute_error_metrics_single(op.Ydd_ana, opSyn.Ydd_syn);
        synYqq = compute_error_metrics_single(op.Yqq_ana, opSyn.Yqq_syn);

        syn.Ydd.mag_rmse(sIdx) = synYdd.mag_rmse;
        syn.Ydd.ph_rmse(sIdx)  = synYdd.ph_rmse;
        syn.Ydd.emag(sIdx,:)   = synYdd.emag.';
        syn.Ydd.eph(sIdx,:)    = synYdd.eph.';

        syn.Yqq.mag_rmse(sIdx) = synYqq.mag_rmse;
        syn.Yqq.ph_rmse(sIdx)  = synYqq.ph_rmse;
        syn.Yqq.emag(sIdx,:)   = synYqq.emag.';
        syn.Yqq.eph(sIdx,:)    = synYqq.eph.';

        d = seed_distance_mag_phase(measYdd, synYdd, measYqq, synYqq);
        if d < bestDist
            bestDist  = d;
            bestSeed  = seed;
            bestSynOp = opSyn;
        end
    end

    % ---------- summarize
    sumYdd = summarize_channel_distribution_mag_phase_only(measYdd, syn.Ydd, acceptCfg);
    sumYqq = summarize_channel_distribution_mag_phase_only(measYqq, syn.Yqq, acceptCfg);

    % ---------- store
    results(k).name         = op.name;
    results(k).I2d          = op.I2d;
    results(k).I2q          = op.I2q;
    results(k).fHz          = op.fHz_ana(:);
    results(k).nMeasRepeats = op.nMeasRepeats;

    results(k).opClean    = op;
    results(k).bestSeed   = bestSeed;
    results(k).bestSynOp  = bestSynOp;

    results(k).Ydd.meas   = measYdd;
    results(k).Ydd.syn    = syn.Ydd;
    results(k).Ydd.sum    = sumYdd;

    results(k).Yqq.meas   = measYqq;
    results(k).Yqq.syn    = syn.Yqq;
    results(k).Yqq.sum    = sumYqq;

    fprintf('  Best representative seed = %d\n', bestSeed);
    fprintf('  Ydd acceptable = %s | Yqq acceptable = %s\n', ...
        tf2str(sumYdd.acceptable), tf2str(sumYqq.acceptable));
end

%% =========================================================
% STEP 3: BUILD SUMMARY TABLE
% =========================================================
T = build_summary_table_mag_phase_only(results);

disp(' ');
disp('================ VALIDATION SUMMARY TABLE ================');
disp(T);

%% =========================================================
% STEP 4: PLOTS
% =========================================================
if plotCfg.makeSummaryFigure
    plot_summary_metrics_mag_phase_only(results, acceptCfg);
end

if plotCfg.makeErrorBandFigures
    for k = 1:nOP
        plot_error_band_figure_mag_phase_only(results(k), acceptCfg);
    end
end

if plotCfg.makeRepresentativeBodeFigures
    for k = 1:nOP
        plot_representative_bode(results(k));
    end
end

if plotCfg.makeMeasurementRepeatFigures
    for k = 1:nOP
        plot_measurement_repeats(results(k).opClean);
    end
end

%% =========================================================
% LOCAL FUNCTION: summarize channel distribution
% mag/phase only
% =========================================================
function summary = summarize_channel_distribution_mag_phase_only(meas, syn, acceptCfg)

    pct = [acceptCfg.metricBandPct(1), 50, acceptCfg.metricBandPct(2)];

    % RMSE distributions
    qMag = myprctile_rows(syn.mag_rmse, pct);
    qPh  = myprctile_rows(syn.ph_rmse,  pct);

    summary.mag.measured = meas.mag_rmse;
    summary.mag.pLow     = qMag(1);
    summary.mag.p50      = qMag(2);
    summary.mag.pHigh    = qMag(3);
    summary.mag.inBand   = is_in_band(meas.mag_rmse, qMag(1), qMag(3));

    summary.ph.measured  = meas.ph_rmse;
    summary.ph.pLow      = qPh(1);
    summary.ph.p50       = qPh(2);
    summary.ph.pHigh     = qPh(3);
    summary.ph.inBand    = is_in_band(meas.ph_rmse, qPh(1), qPh(3));

    % Frequency-wise error bands
    qEmag = myprctile_rows(syn.emag, pct);
    qEph  = myprctile_rows(syn.eph,  pct);

    summary.errBand.mag.pLow  = qEmag(1,:).';
    summary.errBand.mag.p50   = qEmag(2,:).';
    summary.errBand.mag.pHigh = qEmag(3,:).';

    summary.errBand.ph.pLow   = qEph(1,:).';
    summary.errBand.ph.p50    = qEph(2,:).';
    summary.errBand.ph.pHigh  = qEph(3,:).';

    inMag = meas.emag >= summary.errBand.mag.pLow & meas.emag <= summary.errBand.mag.pHigh;
    inPh  = meas.eph  >= summary.errBand.ph.pLow  & meas.eph  <= summary.errBand.ph.pHigh;

    summary.mag.coverage = mean(inMag);
    summary.ph.coverage  = mean(inPh);

    summary.mag.coveragePass = summary.mag.coverage >= acceptCfg.minCoverage;
    summary.ph.coveragePass  = summary.ph.coverage  >= acceptCfg.minCoverage;

    summary.acceptable = summary.mag.inBand && summary.ph.inBand && ...
                         summary.mag.coveragePass && summary.ph.coveragePass;
end

%% =========================================================
% LOCAL FUNCTION: measured metrics from repeated runs
% Average magnitude error and phase error separately
% =========================================================
function out = compute_error_metrics_from_runs(Yana, Yruns)

    Yana = Yana(:);
    tiny = 1e-12;
    nRuns = size(Yruns, 2);

    mag_ana_db = 20*log10(abs(Yana) + tiny);
    ph_ana_deg = wrapTo180(rad2deg(angle(Yana)));

    emag_runs = zeros(size(Yruns));
    eph_runs  = zeros(size(Yruns));

    for r = 1:nRuns
        Ycur = Yruns(:,r);

        mag_cur_db = 20*log10(abs(Ycur) + tiny);
        ph_cur_deg = wrapTo180(rad2deg(angle(Ycur)));

        emag_runs(:,r) = mag_cur_db - mag_ana_db;
        eph_runs(:,r)  = wrapTo180(ph_cur_deg - ph_ana_deg);
    end

    % Average magnitude error directly
    out.emag = mean(emag_runs, 2);

    % Circular mean for phase error
    out.eph = rad2deg(angle(mean(exp(1j*deg2rad(eph_runs)), 2)));

    out.mag_rmse = sqrt(mean(out.emag.^2));
    out.mag_mae  = mean(abs(out.emag));

    out.ph_rmse = sqrt(mean(out.eph.^2));
    out.ph_mae  = mean(abs(out.eph));

    out.emag_runs = emag_runs;
    out.eph_runs  = eph_runs;
end

%% =========================================================
% LOCAL FUNCTION: synthetic metrics from single FRF
% =========================================================
function out = compute_error_metrics_single(Yana, Yother)

    Yana   = Yana(:);
    Yother = Yother(:);
    tiny = 1e-12;

    mag_ana   = 20*log10(abs(Yana)   + tiny);
    mag_other = 20*log10(abs(Yother) + tiny);
    out.emag  = mag_other - mag_ana;

    ph_ana   = wrapTo180(rad2deg(angle(Yana)));
    ph_other = wrapTo180(rad2deg(angle(Yother)));
    out.eph  = wrapTo180(ph_other - ph_ana);

    out.mag_rmse = sqrt(mean(out.emag.^2));
    out.mag_mae  = mean(abs(out.emag));

    out.ph_rmse = sqrt(mean(out.eph.^2));
    out.ph_mae  = mean(abs(out.eph));
end

%% =========================================================
% LOCAL FUNCTION: distance to choose representative seed
% mag/phase only
% =========================================================
function d = seed_distance_mag_phase(measYdd, synYdd, measYqq, synYqq)

    vMeas = [ ...
        measYdd.mag_rmse, measYdd.ph_rmse, ...
        measYqq.mag_rmse, measYqq.ph_rmse];

    vSyn = [ ...
        synYdd.mag_rmse, synYdd.ph_rmse, ...
        synYqq.mag_rmse, synYqq.ph_rmse];

    scale = max(abs(vMeas), [1e-6, 1, 1e-6, 1]);
    d = sum(((vSyn - vMeas) ./ scale).^2);
end

%% =========================================================
% LOCAL FUNCTION: build summary table
% mag/phase only
% =========================================================
function T = build_summary_table_mag_phase_only(results)

    nOP = numel(results);
    nRows = 2*nOP;

    OP      = strings(nRows,1);
    Channel = strings(nRows,1);
    I2d     = zeros(nRows,1);
    I2q     = zeros(nRows,1);

    MagRMSE_meas = zeros(nRows,1);
    MagRMSE_p5   = zeros(nRows,1);
    MagRMSE_p50  = zeros(nRows,1);
    MagRMSE_p95  = zeros(nRows,1);

    PhRMSE_meas  = zeros(nRows,1);
    PhRMSE_p5    = zeros(nRows,1);
    PhRMSE_p50   = zeros(nRows,1);
    PhRMSE_p95   = zeros(nRows,1);

    MagCoverage  = zeros(nRows,1);
    PhCoverage   = zeros(nRows,1);
    Acceptable   = strings(nRows,1);

    row = 0;
    for k = 1:nOP
        % Ydd
        row = row + 1;
        OP(row)      = string(results(k).name);
        Channel(row) = "Ydd";
        I2d(row)     = results(k).I2d;
        I2q(row)     = results(k).I2q;

        S = results(k).Ydd.sum;
        MagRMSE_meas(row) = S.mag.measured;
        MagRMSE_p5(row)   = S.mag.pLow;
        MagRMSE_p50(row)  = S.mag.p50;
        MagRMSE_p95(row)  = S.mag.pHigh;

        PhRMSE_meas(row)  = S.ph.measured;
        PhRMSE_p5(row)    = S.ph.pLow;
        PhRMSE_p50(row)   = S.ph.p50;
        PhRMSE_p95(row)   = S.ph.pHigh;

        MagCoverage(row) = S.mag.coverage;
        PhCoverage(row)  = S.ph.coverage;
        Acceptable(row)  = string(tf2str(S.acceptable));

        % Yqq
        row = row + 1;
        OP(row)      = string(results(k).name);
        Channel(row) = "Yqq";
        I2d(row)     = results(k).I2d;
        I2q(row)     = results(k).I2q;

        S = results(k).Yqq.sum;
        MagRMSE_meas(row) = S.mag.measured;
        MagRMSE_p5(row)   = S.mag.pLow;
        MagRMSE_p50(row)  = S.mag.p50;
        MagRMSE_p95(row)  = S.mag.pHigh;

        PhRMSE_meas(row)  = S.ph.measured;
        PhRMSE_p5(row)    = S.ph.pLow;
        PhRMSE_p50(row)   = S.ph.p50;
        PhRMSE_p95(row)   = S.ph.pHigh;

        MagCoverage(row) = S.mag.coverage;
        PhCoverage(row)  = S.ph.coverage;
        Acceptable(row)  = string(tf2str(S.acceptable));
    end

    T = table(OP, Channel, I2d, I2q, ...
        MagRMSE_meas, MagRMSE_p5, MagRMSE_p50, MagRMSE_p95, ...
        PhRMSE_meas, PhRMSE_p5, PhRMSE_p50, PhRMSE_p95, ...
        MagCoverage, PhCoverage, Acceptable);
end

%% =========================================================
% LOCAL FUNCTION: summary plot
% mag/phase only
% =========================================================
function plot_summary_metrics_mag_phase_only(results, acceptCfg)

    labels = {};
    magMeas = [];
    magP5 = [];
    magP50 = [];
    magP95 = [];

    phMeas = [];
    phP5 = [];
    phP50 = [];
    phP95 = [];

    for k = 1:numel(results)
        labels{end+1} = sprintf('%s-Ydd', results(k).name);
        labels{end+1} = sprintf('%s-Yqq', results(k).name);

        S = results(k).Ydd.sum;
        magMeas(end+1) = S.mag.measured;
        magP5(end+1)   = S.mag.pLow;
        magP50(end+1)  = S.mag.p50;
        magP95(end+1)  = S.mag.pHigh;

        phMeas(end+1)  = S.ph.measured;
        phP5(end+1)    = S.ph.pLow;
        phP50(end+1)   = S.ph.p50;
        phP95(end+1)   = S.ph.pHigh;

        S = results(k).Yqq.sum;
        magMeas(end+1) = S.mag.measured;
        magP5(end+1)   = S.mag.pLow;
        magP50(end+1)  = S.mag.p50;
        magP95(end+1)  = S.mag.pHigh;

        phMeas(end+1)  = S.ph.measured;
        phP5(end+1)    = S.ph.pLow;
        phP50(end+1)   = S.ph.p50;
        phP95(end+1)   = S.ph.pHigh;
    end

    x = 1:numel(labels);

    figure('Name','Summary Metrics','Position',[80 80 950 700]);
    t = tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

    % Magnitude RMSE
    ax1 = nexttile;
    hold on;
    errorbar(x, magP50, magP50-magP5, magP95-magP50, 'o', 'LineWidth',1.2);
    plot(x, magMeas, 'x', 'LineWidth',1.6, 'MarkerSize',8);
    yline(0, ':');
    grid on; box on;
    ylabel('Magnitude RMSE (dB)');
    title(sprintf('Synthetic median and %d-%d percentile vs measured', ...
        acceptCfg.metricBandPct(1), acceptCfg.metricBandPct(2)));
    set(ax1, 'XTick', x, 'XTickLabel', labels, 'XTickLabelRotation', 20);

    % Phase RMSE
    ax2 = nexttile;
    hold on;
    errorbar(x, phP50, phP50-phP5, phP95-phP50, 'o', 'LineWidth',1.2);
    plot(x, phMeas, 'x', 'LineWidth',1.6, 'MarkerSize',8);
    yline(0, ':');
    grid on; box on;
    ylabel('Phase RMSE (deg)');
    xlabel(t, 'Operating point / channel');
    set(ax2, 'XTick', x, 'XTickLabel', labels, 'XTickLabelRotation', 20);

    legend(ax1, 'Synthetic median with percentile band', 'Measured', 'Location','best');
end

%% =========================================================
% LOCAL FUNCTION: error-band figure for one OP
% mag/phase only
% =========================================================
function plot_error_band_figure_mag_phase_only(R, acceptCfg)

    fHz = R.fHz(:);
    pctLow  = acceptCfg.metricBandPct(1);
    pctHigh = acceptCfg.metricBandPct(2);

    figure('Name',sprintf('%s Error Bands', R.name), 'Position',[100 100 1100 720]);
    t = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

    % Ydd magnitude error
    ax1 = nexttile;
    hold on;
    fill_between_semilogx(fHz, R.Ydd.sum.errBand.mag.pLow, R.Ydd.sum.errBand.mag.pHigh);
    semilogx(fHz, R.Ydd.sum.errBand.mag.p50, '--', 'LineWidth',1.2);
    semilogx(fHz, R.Ydd.meas.emag, 'o-', 'LineWidth',1.4, 'MarkerSize',5);
    grid on; box on;
    title(sprintf('%s: Y_{dd} magnitude error', R.name));
    ylabel('\Delta Magnitude (dB)');
    legend(sprintf('Synthetic %d-%d%%', pctLow, pctHigh), 'Synthetic median', ...
        sprintf('Measured avg error (%d runs)', R.nMeasRepeats), 'Location','best');

    % Yqq magnitude error
    ax2 = nexttile;
    hold on;
    fill_between_semilogx(fHz, R.Yqq.sum.errBand.mag.pLow, R.Yqq.sum.errBand.mag.pHigh);
    semilogx(fHz, R.Yqq.sum.errBand.mag.p50, '--', 'LineWidth',1.2);
    semilogx(fHz, R.Yqq.meas.emag, 'o-', 'LineWidth',1.4, 'MarkerSize',5);
    grid on; box on;
    title(sprintf('%s: Y_{qq} magnitude error', R.name));

    % Ydd phase error
    ax3 = nexttile;
    hold on;
    fill_between_semilogx(fHz, R.Ydd.sum.errBand.ph.pLow, R.Ydd.sum.errBand.ph.pHigh);
    semilogx(fHz, R.Ydd.sum.errBand.ph.p50, '--', 'LineWidth',1.2);
    semilogx(fHz, R.Ydd.meas.eph, 'o-', 'LineWidth',1.4, 'MarkerSize',5);
    grid on; box on;
    ylabel('\Delta Phase (deg)');
    title(sprintf('%s: Y_{dd} phase error', R.name));

    % Yqq phase error
    ax4 = nexttile;
    hold on;
    fill_between_semilogx(fHz, R.Yqq.sum.errBand.ph.pLow, R.Yqq.sum.errBand.ph.pHigh);
    semilogx(fHz, R.Yqq.sum.errBand.ph.p50, '--', 'LineWidth',1.2);
    semilogx(fHz, R.Yqq.meas.eph, 'o-', 'LineWidth',1.4, 'MarkerSize',5);
    grid on; box on;
    title(sprintf('%s: Y_{qq} phase error', R.name));

    xlabel(t, sprintf('%s  |  I2d = %.1f, I2q = %.1f  |  repeats = %d  |  Ydd acceptable = %s, Yqq acceptable = %s', ...
        R.name, R.I2d, R.I2q, R.nMeasRepeats, ...
        tf2str(R.Ydd.sum.acceptable), tf2str(R.Yqq.sum.acceptable)));
end

%% =========================================================
% LOCAL FUNCTION: representative bode plot
% measured = average FRF only for visualization
% =========================================================
function plot_representative_bode(R)

    op  = R.opClean;
    syn = R.bestSynOp;

    blue_col   = [0.00 0.45 0.74];
    orange_col = [0.85 0.33 0.10];
    gray_col   = [0.35 0.35 0.35];

    figure('Name',sprintf('%s Representative Bode', R.name), ...
        'Position',[110 110 1050 720]);

    t = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

    % Ydd mag
    nexttile;
    semilogx(op.fHz_ana, 20*log10(abs(op.Ydd_ana)), 's--', 'Color', blue_col, ...
        'LineWidth',1.2, 'MarkerSize',5); hold on;
    semilogx(op.fHz_ana, 20*log10(abs(syn.Ydd_syn)), 'x--', 'Color', orange_col, ...
        'LineWidth',1.4, 'MarkerSize',5);
    semilogx(op.fHz_meas, 20*log10(abs(op.Ydd_meas_vis)), 'o--', 'Color', gray_col, ...
        'LineWidth',1.6, 'MarkerSize',6);
    grid on; box on;
    ylabel('Magnitude (dB)');
    title(sprintf('%s | Y_{dd}', R.name));

    % Yqq mag
    nexttile;
    semilogx(op.fHz_ana, 20*log10(abs(op.Yqq_ana)), 's--', 'Color', blue_col, ...
        'LineWidth',1.2, 'MarkerSize',5); hold on;
    semilogx(op.fHz_ana, 20*log10(abs(syn.Yqq_syn)), 'x--', 'Color', orange_col, ...
        'LineWidth',1.4, 'MarkerSize',5);
    semilogx(op.fHz_meas, 20*log10(abs(op.Yqq_meas_vis)), 'o--', 'Color', gray_col, ...
        'LineWidth',1.6, 'MarkerSize',6);
    grid on; box on;
    title(sprintf('%s | Y_{qq}', R.name));
    legend('Analytical clean', sprintf('Synthetic seed %d', R.bestSeed), ...
        sprintf('Measured mean FRF for view (%d runs)', R.nMeasRepeats), 'Location','best');

    % Ydd phase
    nexttile;
    semilogx(op.fHz_ana, wrapTo180(rad2deg(angle(op.Ydd_ana))), 's--', 'Color', blue_col, ...
        'LineWidth',1.2, 'MarkerSize',5); hold on;
    semilogx(op.fHz_ana, wrapTo180(rad2deg(angle(syn.Ydd_syn))), 'x--', 'Color', orange_col, ...
        'LineWidth',1.4, 'MarkerSize',5);
    semilogx(op.fHz_meas, wrapTo180(rad2deg(angle(op.Ydd_meas_vis))), 'o--', 'Color', gray_col, ...
        'LineWidth',1.6, 'MarkerSize',6);
    grid on; box on;
    ylabel('Phase (deg)');

    % Yqq phase
    nexttile;
    semilogx(op.fHz_ana, wrapTo180(rad2deg(angle(op.Yqq_ana))), 's--', 'Color', blue_col, ...
        'LineWidth',1.2, 'MarkerSize',5); hold on;
    semilogx(op.fHz_ana, wrapTo180(rad2deg(angle(syn.Yqq_syn))), 'x--', 'Color', orange_col, ...
        'LineWidth',1.4, 'MarkerSize',5);
    semilogx(op.fHz_meas, wrapTo180(rad2deg(angle(op.Yqq_meas_vis))), 'o--', 'Color', gray_col, ...
        'LineWidth',1.6, 'MarkerSize',6);
    grid on; box on;

    xlabel(t, sprintf('%s representative seed = %d', R.name, R.bestSeed));
end

%% =========================================================
% LOCAL FUNCTION: plot raw measurement repeats
% =========================================================
function plot_measurement_repeats(op)

    figure('Name', sprintf('%s Measurement Repeats', op.name), ...
        'Position', [120 120 1000 700]);
    t = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

    % Ydd magnitude
    nexttile; hold on;
    for r = 1:op.nMeasRepeats
        semilogx(op.fHz_meas, 20*log10(abs(op.Ydd_meas_runs(:,r))), '--');
    end
    semilogx(op.fHz_meas, 20*log10(abs(op.Ydd_meas_vis)), 'ko-', 'LineWidth',1.5);
    grid on; box on; title(sprintf('%s Y_{dd} magnitude', op.name));

    % Yqq magnitude
    nexttile; hold on;
    for r = 1:op.nMeasRepeats
        semilogx(op.fHz_meas, 20*log10(abs(op.Yqq_meas_runs(:,r))), '--');
    end
    semilogx(op.fHz_meas, 20*log10(abs(op.Yqq_meas_vis)), 'ko-', 'LineWidth',1.5);
    grid on; box on; title(sprintf('%s Y_{qq} magnitude', op.name));

    % Ydd phase
    nexttile; hold on;
    for r = 1:op.nMeasRepeats
        semilogx(op.fHz_meas, wrapTo180(rad2deg(angle(op.Ydd_meas_runs(:,r)))), '--');
    end
    semilogx(op.fHz_meas, wrapTo180(rad2deg(angle(op.Ydd_meas_vis))), 'ko-', 'LineWidth',1.5);
    grid on; box on; title(sprintf('%s Y_{dd} phase', op.name));

    % Yqq phase
    nexttile; hold on;
    for r = 1:op.nMeasRepeats
        semilogx(op.fHz_meas, wrapTo180(rad2deg(angle(op.Yqq_meas_runs(:,r)))), '--');
    end
    semilogx(op.fHz_meas, wrapTo180(rad2deg(angle(op.Yqq_meas_vis))), 'ko-', 'LineWidth',1.5);
    grid on; box on; title(sprintf('%s Y_{qq} phase', op.name));

    xlabel(t, sprintf('%s | measured repeats = %d', op.name, op.nMeasRepeats));
end

%% =========================================================
% LOCAL FUNCTION: fill band on semilog-x axis
% =========================================================
function fill_between_semilogx(x, yLow, yHigh)
    x = x(:);
    yLow = yLow(:);
    yHigh = yHigh(:);
    X = [x; flipud(x)];
    Y = [yLow; flipud(yHigh)];
    fill(X, Y, [0.85 0.90 1.00], 'EdgeColor','none', 'FaceAlpha',0.5);
    set(gca, 'XScale', 'log');
end

%% =========================================================
% LOCAL FUNCTION: percentile across rows
% Input:
%   X = [nSamples x nVariables]
% Output:
%   q = [numel(pcts) x nVariables]
% =========================================================
function q = myprctile_rows(X, pcts)
    if isvector(X)
        X = X(:);
    end

    Xs = sort(X, 1, 'ascend');
    n = size(Xs,1);
    q = zeros(numel(pcts), size(Xs,2));

    for i = 1:numel(pcts)
        p = pcts(i);
        pos = 1 + (p/100) * (n - 1);
        lo = floor(pos);
        hi = ceil(pos);
        a = pos - lo;

        if lo == hi
            q(i,:) = Xs(lo,:);
        else
            q(i,:) = (1-a).*Xs(lo,:) + a.*Xs(hi,:);
        end
    end
end

%% =========================================================
% LOCAL FUNCTION: check scalar in band
% =========================================================
function tf = is_in_band(x, lowVal, highVal)
    tf = (x >= lowVal) && (x <= highVal);
end

%% =========================================================
% LOCAL FUNCTION: logical to YES/NO string
% =========================================================
function s = tf2str(tf)
    if tf
        s = 'YES';
    else
        s = 'NO';
    end
end

%% =========================================================
% LOCAL FUNCTION: run repeated measurement + analytical clean
% =========================================================
function out = run_one_op_clean_repeated(mdl, axisBlk, I2d, I2q, p, fHz_meas, w_meas, fHz_ana, w_ana, nMeasRepeats)

    %% ---- Push variables to base workspace for Simulink
    assignin('base','Tsam', p.Tsam);
    assignin('base','Vdc',  p.Vdc);
    assignin('base','Vg',   p.Vg);
    assignin('base','w1',   p.w1);
    assignin('base','fs',   p.fs);
    assignin('base','Ts',   p.Ts);
    assignin('base','Td',   p.Td);
    assignin('base','fsw',  p.fsw);

    assignin('base','Rf1',  p.Rf1);
    assignin('base','Lf1',  p.Lf1);
    assignin('base','Rf2',  p.Rf2);
    assignin('base','Lf2',  p.Lf2);
    assignin('base','Cf',   p.Cf);

    assignin('base','Kpi',  p.Kpi);
    assignin('base','Kii',  p.Kii);

    assignin('base','Kppll',p.Kppll);
    assignin('base','Kipll',p.Kipll);
    assignin('base','beta', p.beta);

    assignin('base','Lg',   p.Lg);
    assignin('base','Rg',   p.Rg);

    assignin('base','I2d',  I2d);
    assignin('base','I2q',  I2q);
    assignin('base','V2d',  p.Vg);

    assignin('base','fHz_meas', fHz_meas);
    assignin('base','w_meas',   w_meas);
    assignin('base','w',        w_meas);

    %% ---- Repeated measurement
    nF_meas = numel(w_meas);

    Ydd_meas_runs = zeros(nF_meas, nMeasRepeats);
    Yqq_meas_runs = zeros(nF_meas, nMeasRepeats);

    for r = 1:nMeasRepeats
        fprintf('    Repeat %d/%d for I2d = %.1f, I2q = %.1f\n', ...
            r, nMeasRepeats, I2d, I2q);

        % ---- Measurement Ydd
        set_param(axisBlk, 'Value', '1');
        simOut_d = sim(mdl, 'SrcWorkspace', 'base');

        SigGenDataLog_Ydd = simOut_d.SigGenDataLog_Ydd;
        sys_estim_Ydd = frestimate(SigGenDataLog_Ydd, w_meas, "rad/s");
        Ydd_meas_runs(:, r) = -squeeze(sys_estim_Ydd.ResponseData(1,1,:));

        % ---- Measurement Yqq
        set_param(axisBlk, 'Value', '0');
        simOut_q = sim(mdl, 'SrcWorkspace', 'base');

        SigGenDataLog_Yqq = simOut_q.SigGenDataLog_Yqq;
        sys_estim_Yqq = frestimate(SigGenDataLog_Yqq, w_meas, "rad/s");
        Yqq_meas_runs(:, r) = -squeeze(sys_estim_Yqq.ResponseData(1,1,:));
    end

    % For visualization only: mean FRF in complex domain
    Ydd_meas_vis = mean(Ydd_meas_runs, 2);
    Yqq_meas_vis = mean(Yqq_meas_runs, 2);

    %% ---- Build analytical model
    Ai = zeros(2,2);
    Bi = [1 0; 0 1];
    Ci = (p.Vdc/2) * [p.Kii 0; 0 p.Kii];
    Di = (p.Vdc/2) * [p.Kpi 0; 0 p.Kpi];

    Td = p.Td;
    Adel = [0, 1, 0, 0, 0, 0;
            0, 0, 1, 0, 0, 0;
            -120/Td^3, -60/Td^2, -12/Td, 0, 0, 0;
            0, 0, 0, 0, 1, 0;
            0, 0, 0, 0, 0, 1;
            0, 0, 0, -120/Td^3, -60/Td^2, -12/Td];
    Bdel = [0, 0;
            0, 0;
            1, 0;
            0, 0;
            0, 0;
            0, 1];
    Cdel = [240/Td^3 0 24/Td 0 0 0;
            0 0 0 240/Td^3 0 24/Td];
    Ddel = [-1 0; 0 -1];

    Alcl = [-p.Rf1/p.Lf1,  p.w1,          0,             0,         -1/p.Lf1, 0;
            -p.w1,        -p.Rf1/p.Lf1,   0,             0,          0,       -1/p.Lf1;
             0,            0,            -p.Rf2/p.Lf2,   p.w1,       1/p.Lf2, 0;
             0,            0,            -p.w1,         -p.Rf2/p.Lf2,0,        1/p.Lf2;
             1/p.Cf,       0,            -1/p.Cf,        0,          0,        p.w1;
             0,            1/p.Cf,        0,            -1/p.Cf,    -p.w1,     0];
    Blcl = [1/p.Lf1 0       0        0;
            0       1/p.Lf1 0        0;
            0       0      -1/p.Lf2  0;
            0       0       0       -1/p.Lf2;
            0       0       0        0;
            0       0       0        0];
    Clcl = [0 0 1 0 0 0;
            0 0 0 1 0 0];
    Dlcl = zeros(2,4);

    Apll = [0 p.Kipll;
            0 0];
    Bpll = [p.Kppll;
            1];
    Cpll = [1 0];
    Dpll = 0;

    R3 = [1 0;
          0 1;
         -I2q I2d]';

    R2 = [1 0 0 0;
          0 1 0 0;
          0 0 1 0;
          0 0 0 1;
          0 0 0 1];

    R1 = [0 0 0;
          0 0 0;
          0 0 0;
          0 0 -p.Vg;
          0 0 -p.Vg];

    R4 = zeros(2,4);

    R_3 = [0 0 0 0 -1 0;
           0 0 0 0 0 -1;
           1 0 0 0 0 -p.w1*(p.Lf1 + p.Lf2);
           0 1 0 0 p.w1*(p.Lf1 + p.Lf2) 0;
           0 0 1 0 0 0;
           0 0 0 1 0 0;
           0 0 0 0 0 0;
           0 0 0 0 0 0];

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

    Avsc = Ast + Bst*R_3*((eye(6) - Dst*R_3)\Cst);
    Bvsc = Bst*R_3*((eye(6) - Dst*R_3)\(Dst*R_2)) + Bst*R_2;
    Cvsc = R_1*((eye(6) - Dst*R_3)\Cst);
    Dvsc = R_1*((eye(6) - Dst*R_3)\(Dst*R_2)) + R_0;

    Ast1 = blkdiag(Avsc, Apll);
    Bst1 = blkdiag(Bvsc, Bpll);
    Cst1 = blkdiag(Cvsc, Cpll);
    Dst1 = blkdiag(Dvsc, Dpll);

    Avsc2 = Ast1 + Bst1*R1*((eye(3) - Dst1*R1)\Cst1);
    Bvsc2 = Bst1*R1*((eye(3) - Dst1*R1)\(Dst1*R2)) + Bst1*R2;
    Cvsc2 = R3*((eye(3) - Dst1*R1)\Cst1);
    Dvsc2 = R3*((eye(3) - Dst1*R1)\(Dst1*R2)) + R4;

    s = tf('s');
    G_closed = Cvsc2 * inv(s*eye(16) - Avsc2) * Bvsc2 + Dvsc2;
    Yvsc = -minreal(G_closed(:,3:4));

    Y11 = minreal(Yvsc(1,1));
    Y22 = minreal(Yvsc(2,2));

    Ydd_ana = squeeze(freqresp(Y11, w_ana));
    Yqq_ana = squeeze(freqresp(Y22, w_ana));

    %% ---- Output
    out.I2d = I2d;
    out.I2q = I2q;

    out.fHz_meas = fHz_meas(:);
    out.fHz_ana  = fHz_ana(:);

    out.nMeasRepeats = nMeasRepeats;

    out.Ydd_meas_runs = Ydd_meas_runs;
    out.Yqq_meas_runs = Yqq_meas_runs;

    % visualization-only average FRF
    out.Ydd_meas_vis = Ydd_meas_vis(:);
    out.Yqq_meas_vis = Yqq_meas_vis(:);

    out.Ydd_ana = Ydd_ana(:);
    out.Yqq_ana = Yqq_ana(:);

    out.Ydd_syn = Ydd_ana(:);
    out.Yqq_syn = Yqq_ana(:);
end

%% =========================================================
% LOCAL FUNCTION: apply noise only
% =========================================================
function op = apply_noise_to_op(op, noiseOpt, seed)

    if nargin < 3
        seed = 1;
    end

    if ~isfield(noiseOpt, 'enable') || ~noiseOpt.enable
        op.Ydd_syn = op.Ydd_ana;
        op.Yqq_syn = op.Yqq_ana;
        return;
    end

    rng(seed, 'twister');
    op.Ydd_syn = add_frf_noise(op.Ydd_ana, op.fHz_ana, noiseOpt);
    op.Yqq_syn = add_frf_noise(op.Yqq_ana, op.fHz_ana, noiseOpt);
end

%% =========================================================
% LOCAL FUNCTION: add direct complex noise to FRF
% =========================================================
function Ynoisy = add_frf_noise(Yclean, fHz, noiseOpt)

    Yclean = Yclean(:);
    fHz    = fHz(:);

    w_hf = (fHz ./ noiseOpt.fc).^noiseOpt.p;
    w_hf = w_hf ./ (1 + w_hf);

    rel_sigma_f = noiseOpt.rel_sigma_low + ...
        (noiseOpt.rel_sigma_high - noiseOpt.rel_sigma_low) .* w_hf;

    eta_rel = (randn(size(Yclean)) + 1j*randn(size(Yclean))) / sqrt(2);

    floor_low  = noiseOpt.abs_floor_ratio_low  * max(abs(Yclean));
    floor_high = noiseOpt.abs_floor_ratio_high * max(abs(Yclean));
    beta_f = floor_low + (floor_high - floor_low) .* w_hf;

    eta_abs = (randn(size(Yclean)) + 1j*randn(size(Yclean))) / sqrt(2);

    w_bias = (fHz ./ noiseOpt.bias_fc).^noiseOpt.p;
    w_bias = w_bias ./ (1 + w_bias);

    if isfield(noiseOpt, 'random_bias_sign') && noiseOpt.random_bias_sign
        % sMag = sign(randn(1));
        % sPh  = sign(randn(1));
        sMag = sign(1);
        sPh  = sign(-1);
        if sMag == 0, sMag = 1; end
        if sPh  == 0, sPh  = 1; end
    else
        sMag = 1;
        sPh  = -1;
    end

    mag_bias = 1 + sMag * noiseOpt.bias_mag_high .* w_bias;
    ph_bias  = deg2rad(sPh * noiseOpt.bias_ph_high .* w_bias);

    Ynoisy = Yclean .* mag_bias .* exp(1j*ph_bias) ...
           + Yclean .* (rel_sigma_f .* eta_rel) ...
           + beta_f .* eta_abs;
end