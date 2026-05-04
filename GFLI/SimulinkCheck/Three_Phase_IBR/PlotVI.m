% close all; clearvars;

%% ===================== IEEE-style plot settings =====================
S.fontname        = 'Times New Roman';
S.fontsize        = 8;
S.fontsizeLegend  = 8;

S.figUnit         = 'centimeters';
S.figPos          = [5 10];            % [x y] on screen (cm)
% S.figSize         = [8.9 4.0];         % ~IEEE 1-column width ≈ 3.5in = 8.9cm
S.figSize         = [4.4 4.0];         % ~IEEE 1-column width ≈ 3.5in = 8.9cm

S.lineWidth       = 1.0;               % curves
S.axisLineWidth   = 0.75;              % axes box

S.gridOn          = true;
S.gridAlpha       = 0.12;
S.minorGridAlpha  = 0.08;

S.xLimits         = [0 0.5];
S.yLimitsV        = [-2e3 2e3];
S.yLimitsI        = [-5.7e3 5.7e3];

% IEEE-friendly: color OK (online) nhưng vẫn phân biệt bằng linestyle
% (Nếu muốn B/W hoàn toàn: đặt 3 màu đều 'k' và chỉ đổi linestyle)
% IEEE-friendly (color + still distinguishable if printed)
col_a = [0.0000 0.4470 0.7410];  ls_a = '-';   % blue
col_b = [0.8500 0.3250 0.0980];  ls_b = '-';  % orange
col_c = [0.4660 0.6740 0.1880];  ls_c = '-';   % green

%% ===================== Extract data =====================
t     = Vpcc.time(:);

Vpcca = Vpcc.signals.values(:,1);
Vpccb = Vpcc.signals.values(:,2);
Vpccc = Vpcc.signals.values(:,3);

Ipcca = Ipcc.signals.values(:,1);
Ipccb = Ipcc.signals.values(:,2);
Ipccc = Ipcc.signals.values(:,3);

%% ===================== Figure 1: v_pcc (abc) =====================
fig1 = figure('Units', S.figUnit, 'Position', [S.figPos S.figSize], ...
              'Color','w');

ax1 = axes(fig1); hold(ax1,'on'); box(ax1,'on');

hVa = plot(ax1, t, Vpcca, 'LineStyle', ls_a, 'Color', col_a, 'LineWidth', S.lineWidth);
hVb = plot(ax1, t, Vpccb, 'LineStyle', ls_b, 'Color', col_b, 'LineWidth', S.lineWidth);
hVc = plot(ax1, t, Vpccc, 'LineStyle', ls_c, 'Color', col_c, 'LineWidth', S.lineWidth);


xlim(ax1, S.xLimits); ylim(ax1, S.yLimitsV);

xlabel(ax1, 'Time (s)', 'Interpreter','latex');
% ylabel(ax1, '$v_{\mathrm{pcc}}$ (V)', 'Interpreter','latex');

applyIEEEStyle(ax1, S);

% lg1 = legend(ax1, [hVa hVb hVc], {'$v_a$','$v_b$','$v_c$'}, ...
%     'Interpreter','latex', 'Orientation','horizontal', ...
%     'Location','southoutside', 'NumColumns', 1);
% styleLegendIEEE(lg1, S);

%% ===================== Figure 2: i_pcc (abc) =====================
fig2 = figure('Units', S.figUnit, 'Position', [S.figPos S.figSize], ...
              'Color','w');

ax2 = axes(fig2); hold(ax2,'on'); box(ax2,'on');

hIa = plot(ax2, t, Ipcca, 'LineStyle', ls_a, 'Color', col_a, 'LineWidth', S.lineWidth);
hIb = plot(ax2, t, Ipccb, 'LineStyle', ls_b, 'Color', col_b, 'LineWidth', S.lineWidth);
hIc = plot(ax2, t, Ipccc, 'LineStyle', ls_c, 'Color', col_c, 'LineWidth', S.lineWidth);

xlim(ax2, S.xLimits); ylim(ax2, S.yLimitsI);

xlabel(ax2, 'Time (s)', 'Interpreter','latex');
% ylabel(ax2, '$i_{\mathrm{pcc}}$ (A)', 'Interpreter','latex');

applyIEEEStyle(ax2, S);

% lg2 = legend(ax2, [hIa hIb hIc], {'$i_a$','$i_b$','$i_c$'}, ...
%     'Interpreter','latex', 'Orientation','horizontal', ...
%     'Location','southoutside', 'NumColumns', 2);
% styleLegendIEEE(lg2, S);

%% ===================== Export (PDF/EPS, vector) =====================
% Khuyến nghị IEEE: PDF/EPS vector, font embed.
% Bạn đổi tên file theo paper.
exportgraphics(fig1, 'Fig_Vpcc.pdf', 'ContentType','vector');
exportgraphics(fig2, 'Fig_Ipcc.pdf', 'ContentType','vector');

% Nếu journal yêu cầu EPS:
% exportgraphics(fig1, 'Fig_Vpcc.eps', 'ContentType','vector');
% exportgraphics(fig2, 'Fig_Ipcc.eps', 'ContentType','vector');

%% ===================== Local functions =====================
function applyIEEEStyle(ax, S)
    set(ax, 'FontName', S.fontname, 'FontSize', S.fontsize, ...
        'LineWidth', S.axisLineWidth, ...
        'TickLabelInterpreter','latex', ...
        'XColor','k','YColor','k');

    % Grid: nhẹ để không “rối” khi in
    if S.gridOn
        grid(ax,'on'); grid(ax,'minor');
        ax.GridAlpha      = S.gridAlpha;
        ax.MinorGridAlpha = S.minorGridAlpha;
    else
        grid(ax,'off');
    end

    % Tight layout kiểu IEEE (giảm khoảng trắng thừa)
    ax.TickDir = 'out';
    ax.Layer   = 'top';
end

function styleLegendIEEE(lg, S)
    set(lg, 'FontName', S.fontname, 'FontSize', S.fontsizeLegend);
    lg.Box = 'off';
end
