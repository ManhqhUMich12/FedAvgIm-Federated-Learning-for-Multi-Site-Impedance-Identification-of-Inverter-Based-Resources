clear all
clc

%% ----- 
Rf2 = 1e-3;
Lf2 = 1e-3;
Cf = 500000e-6;
%% ----- 
s = tf('s');
Zs_tf = Rf2 + s*Lf2 +1/(s*Cf);   

Zl_tf = Rf2 + s*Lf2*10;
Hs = 1/(1+Zs_tf/Zl_tf);

fLoop = logspace(-1,2,500);   % 0.1 -> 100 Hz
wLoop = 2*pi*fLoop;

[mag, ph] = bode(Zs_tf, wLoop);
[magl, phl] = bode(Zl_tf, wLoop);
[magH, phH] = bode(Hs, wLoop);
mag = squeeze(mag);  ph = squeeze(ph);
magl = squeeze(magl);  phl = squeeze(phl);
magH = squeeze(magH);  phH = squeeze(phH);

% ----- Plot -----
figure;
subplot(2,1,1);
semilogx(fLoop, 20*log10(mag), 'LineWidth', 1.4); hold on;
semilogx(fLoop, 20*log10(magl), '--', 'LineWidth', 1.4);
semilogx(fLoop, 20*log10(magH), '-', 'LineWidth', 1.4);
grid on;
ylabel('|L(j\omega)| (dB)');
legend('Z_{s}','Z_{load}','Location','best');
title('Loop gain / impedance ratio');

subplot(2,1,2);
semilogx(fLoop, ph, 'LineWidth', 1.4); hold on;
semilogx(fLoop, phl, '--', 'LineWidth', 1.4);
semilogx(fLoop, phH, '-', 'LineWidth', 1.4);
grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (deg)');
legend('Z_{s}','Z_{load}','-180^\circ','2 Hz','Location','best');