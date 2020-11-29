% some short simulations to see what robust detrending does to data

%% create signals
sr = 2048;
dur = 100;      %in seconds
t = (0:(dur*sr)-1)/sr;

% simulate sine wave
freq = 1;      %in Hz
sine_wave = sin(2*pi*freq*t)';

% sums of sine waves?

% white\pink noise
power = 2;
cn = dsp.ColoredNoise(power,length(t));
noise = cn();

%% plot signal
signal = noise;
fft_cutoff = 6; %hz

ERPfigure()
subplot(3,2,1)
plot(t,signal)
subplot(3,2,2)
[ spectrum, x_axis ] = plotFFT(signal, sr);
plot(x_axis(x_axis<fft_cutoff),spectrum(x_axis<fft_cutoff))

%% do robust detrending

ord = 10;   %order of polynom
win=50*sr; %size of window (in samples, because of the *sr)

[detrended,w] = nt_detrend(signal,ord,[],[],[],[],win);
%[detrended,w,r] = nt_detrend(signal,ord);

subplot(3,2,3)
plot(t,detrended)
subplot(3,2,4)
[ spectrum, x_axis ] = plotFFT(detrended, sr);
plot(x_axis(x_axis<fft_cutoff),spectrum(x_axis<fft_cutoff))

% plot signal before and after detrending, and psd before and after
% change paratmeters everywhere
%compare to hfp
% show changes with repetitions

%% plot filtered data

filt_cutoff = 0.1;
[ filtered ] = HPF( signal, sr, filt_cutoff);
subplot(3,2,5)
hold on
plot(t,filtered)
subplot(3,2,6)
hold on
[ spectrum, x_axis ] = plotFFT(filtered, sr);
plot(x_axis(x_axis<fft_cutoff),spectrum(x_axis<fft_cutoff))

%% plot third order polynom that looks like sine~

x=-1:0.01:1;
%x2=2*pi*x;
%y=  x2 +(-1/6)*x2.^3;% + (1/24)*x.^5; %beginning of taylor series
y = -x.^3 + x;
y = (1/max(y)) * y;
figure();
plot(x,y);
hold on
plot(x,sin(pi*x));

legend('3-rd deg polynom','sine wave (0.5Hz)')

%%

figure()
for i=1:4
    subplot(2,2,i)
    plot(r(:,i));
end