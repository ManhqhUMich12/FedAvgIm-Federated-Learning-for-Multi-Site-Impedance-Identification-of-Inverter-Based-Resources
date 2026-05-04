%% ==== Grid admittance Yg trên dq-axis (4 thành phần) ====
Lg  = 2*250e-6;          
Rg = 2*3e-3;
w1  = 100*pi;
%% -------------------- Frequency response --------------------

I2 = eye(2);
J  = [0 -1; 1 0];
nGNC = length(fGNC);
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
%% -------------------- Winding number around critical point --------------------
% Critical point for eigenvalue loci: -1 + j0
pcrit = -1 + 1j*0;

N_lambda1 = winding_number(lambda1_all, pcrit);
N_lambda2 = winding_number(lambda2_all, pcrit);

% For det(I+L), critical point is 0 + j0
N_det = winding_number(det_all, 0 + 1j*0);

fprintf('\n===== GNC Winding Number Results =====\n');
fprintf('Winding number of lambda1 around -1+j0 = %.6f\n', N_lambda1);
fprintf('Winding number of lambda2 around -1+j0 = %.6f\n', N_lambda2);
fprintf('Winding number of det(I+L) around 0+j0  = %.6f\n', N_det);

fprintf('\nRounded values:\n');
fprintf('N_lambda1 = %d\n', round(N_lambda1));
fprintf('N_lambda2 = %d\n', round(N_lambda2));
fprintf('N_det     = %d\n', round(N_det));

if round(N_lambda1) == 0 && round(N_lambda2) == 0
    fprintf('\nEigenloci result: No encirclement of -1+j0.\n');
else
    fprintf('\nEigenloci result: Encirclement of -1+j0 detected.\n');
end

if round(N_det) == 0
    fprintf('det(I+L) result: No encirclement of the origin.\n');
else
    fprintf('det(I+L) result: Encirclement of the origin detected.\n');
end
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

% %% ==================== FIGURE 2: det(I+L) ====================
% fig2 = figure('Color','w');
% apply_ieee_tsg_style(fig2, 3.5, 3.5/2);
% ax2 = axes(fig2); hold(ax2,'on'); grid(ax2,'on'); box(ax2,'on'); axis(ax2,'equal');
% 
% plot(ax2, real(det_all), imag(det_all), '-', 'LineWidth', 1.1);
% plot(ax2, 0, 0, 'x', 'LineWidth', 1.2, 'MarkerSize', 6);
% 
% xlabel(ax2,'$\mathrm{Re}\{\det(I+L(j\omega))\}$','Interpreter','latex');
% ylabel(ax2,'$\mathrm{Im}\{\det(I+L(j\omega))\}$','Interpreter','latex');
% 
% exportgraphics(fig2,'GNC_detIplusL_IEEE.pdf','ContentType','vector');

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

function N = winding_number(z, p)
    %WINDING_NUMBER Computes winding number of a closed complex curve z around point p.
    %
    % z : complex vector describing the Nyquist curve
    % p : critical point, e.g., -1+1j*0 or 0+1j*0
    %
    % N > 0: counter-clockwise encirclement
    % N < 0: clockwise encirclement
    % N = 0: no encirclement

    z = z(:);

    % Remove NaN/Inf points if any
    valid = isfinite(real(z)) & isfinite(imag(z));
    z = z(valid);

    % Close the curve if it is not exactly closed
    if abs(z(end) - z(1)) > 1e-10
        z = [z; z(1)];
    end

    % Avoid singular case: curve passing exactly through the critical point
    if any(abs(z - p) < 1e-8)
        warning('Nyquist curve passes very close to the critical point. Winding number may be unreliable.');
    end

    theta = unwrap(angle(z - p));
    N = (theta(end) - theta(1)) / (2*pi);
end