%% assignment document for Paul Kumar 1002240657
% Q1 Quantization simulation


% q = Vmax/(2^b - 1) * V

clear all

Ts = 0.002;
t = linspace(0, 1, 1000);
sineWave = sin((2*pi*4)*t);
%sanity check
%plot(t, sineWave)

%create quantized waveform
sineWaveQuant_4bit = quantization(sineWave, 4);
sineWaveQuant_8bit = quantization(sineWave, 8);
sineWaveQuant_12bit = quantization(sineWave, 12);
sineWaveQuant_16bit = quantization(sineWave, 16);


figure(1)
subplot(4,1,1);
plot(t, sineWave, '-b');
hold on
plot(t, sineWaveQuant_4bit, '-r');
hold off
legend("Continuous sine wave", "4-bit Quantized Wave");

subplot(4,1,2);
plot(t, sineWave, '-b');
hold on 
plot(t, sineWaveQuant_8bit, '-r');
hold off
legend("Original sine wave", "8-bit Quantized Wave");

subplot(4,1,3);
plot(t, sineWave, '-b');
hold on
plot(t, sineWaveQuant_12bit, '-r');
hold off
legend("Continuous sine wave", "12-bit Quantized Wave");

subplot(4,1,4);
plot(t, sineWave, '-b');
hold on
plot(t, sineWaveQuant_16bit, '-r');
hold off
legend("Continuous sine wave", "16-bit Quantized Wave");

% now find the difference in the signal - this is quantization equation
% value

error_4bit = sineWave - sineWaveQuant_4bit;
error_8bit = sineWave - sineWaveQuant_8bit;
error_12bit = sineWave - sineWaveQuant_12bit;
error_16bit = sineWave - sineWaveQuant_16bit;

quant_4bit = (1/(2^4-1));
quant_8bit = (1/(2^8-1));
quant_12bit = (1/(2^12-1));
quant_16bit = (1/(2^16-1));

%% question 2
f = 5;
T = 1/f;
Ts = 1/7;
t = linspace(0,2,1000);
sineWave = sin(2*pi*f*t);

%create a time array of samples. recall a sampled signal is simply
%(singal)*delta(t - Ts*k), where k is some shift
tSampled = 0:Ts:2;
aliasedSine = sin(2*pi*f*tSampled);

%reconstructed aliased sine wave. note the resulting freq of the aliased
%sine wave is 2Hz ie Fa element of the set of (0.5Fs, Fs), then Fa, the
%aliased frequency is Fa = Fs - F, where F is the original frequency

%create new 100pt array to show for completeness
tAliased = linspace(0,2,1000);
sineContAliased = -1*sin(2*pi*2*tAliased);

figure(1)
plot(t, sineWave, 'LineWidth', 1.5)
title("Aliasing Simulation")
xlabel("Time [s]")
ylabel("Amplitude [V]")
hold on
stem(tSampled, aliasedSine, 'LineWidth', 1.5)
plot(tAliased, sineContAliased, 'LineWidth', 1.5);
legend("Original Signal", "Sampled Signal", "Aliased Signal")


%% question 3

%load dtaa
data1 = load('data_c1.mat');

%grab obj
data1 = data1.x;

%segment the data into equal segments
L = length(data1);
segLength = floor(L/10);

segMoments = zeros(1,10);
segVar = zeros(1,10);

% we are determining whether signal is stationary. Recall a signal is
% stationary if all moments and joint moments do not depend on time. ie if
% the mean and variance do not change in time (meaning that for any time
% window we take, the mean and the variance will be the same), then the
% signal is atationary
%%
% **
% 
% <<FILENAME.PNG>>
% 
stationary = 1;
for i = 1:segLength:L
    if(i+segLength-1>L)
        tempMean = mean(data1(i:end));
        tempVar = var(data1(i:end));
        
        segMoments((i+segLength-1)/segLength) = tempMean;
        segVar((i+segLength-1)/segLength) = tempVar;
       break
    end
    
    tempMean = mean(data1(i:i+segLength-1));
    tempVar = var(data1(i:i+segLength-1));
        
    segMoments((i+segLength-1)/segLength) = tempMean;
    segVar((i+segLength-1)/segLength) = tempVar;
    
    index = (i+segLength-1)/segLength;
    
    if(index ~= 1)
        if(segMoments(index) ~= segMoments(index-1) || segVar(index) ~= segMoments(index-1))
            stationary = 0;
        end
    end
end
    
   
disp(segMoments);
disp(segVar);


if(~stationary)
    %applied the detrend function
    disp("Not stationary!")
    scrapedData = detrend(data1);
    
    segMoments = zeros(1,10);
    segVar = zeros(1,10);
    
    stationary = 1;
    for i = 1:segLength:L
        if(i+segLength-1>L)
            tempMean = mean(scrapedData(i:end));
            tempVar = var(scrapedData(i:end));

            segMoments((i+segLength-1)/segLength) = tempMean;
            segVar((i+segLength-1)/segLength) = tempVar;
           break
        end

        tempMean = mean(scrapedData(i:i+segLength-1));
        tempVar = var(scrapedData(i:i+segLength-1));

        segMoments((i+segLength-1)/segLength) = tempMean;
        segVar((i+segLength-1)/segLength) = tempVar;

        index = (i+segLength-1)/segLength;

        if(index ~= 1)
            if(segMoments(index) ~= segMoments(index-1) || segVar(index) ~= segMoments(index-1))
                stationary = 0;
            end
        end
    end
end

disp(segMoments)
disp(segVar)

if(stationary)
    disp("Stantionary after detrend!")
else
    disp("Not stationary pt2!")
end

%after detrending the signal is still non stationary. However, the detrend
%method did remove the clear linear trend in the data. plotting the
%original signal against the detrended, it is clear though the "DC" bias
%has been removed from the signal

subplot(211);
title("Non Stationary Signals")
xlabel("Samples")
ylabel("Amplitude")
plot(data1)
hold on
plot(scrapedData);

%now the associated mean values of the entire signal
x = linspace(1,1000,1000);
line(x, mean(data1)*ones(1,length(data1)))
line(x, mean(scrapedData)*ones(1,length(scrapedData)))

hold off
subplot(212)
    

% note! the moments and variance are not the same

%% question 4

%length of signal
L = 512;
Fs = 1000;
T = 1/Fs;

t = (0:L-1)*T;
f1 = 200;
f2 = 400;
f3 = 900;

x1 = sin(2*pi*f1*t)+sin(2*pi*f2*t);
x2 = sin(2*pi*f1*t)+sin(2*pi*f3*t);

%plot the waveforms in seperate subplots
subplot(311)
plot(t, x1);
hold on
plot(t, x2);

title("Constructed time signals")
xlabel("samples")
ylabel("Amplitude [V]")
hold off

%create the fft
fftX1 = fft(x1);
fftX2 = fft(x2);

spectrum = abs(fftX1/L);
spectrum = spectrum(1:L/2+1);

spectrum(2:end-1) = 2*spectrum(2:end-1);

%create freq axis
f = Fs*(0:(L/2))/L;

subplot(312)

plot(f, spectrum);
title("Magnitude spectrum of Sin(2*pi*200t)+Sin(2*pi*400t)")
xlabel("Frequency [Hz]");
ylabel("X1(f) [V]")

%now for second wave
spectrum2 = abs(fftX2/L);
spectrum2 = spectrum2(1:L/2+1);

spectrum2(2:end-1) = 2*spectrum2(2:end-1);

subplot(313)

plot(f, spectrum2);
title("Magnitude spectrum of Sin(2*pi*200t)+Sin(2*pi*900t)")
xlabel("Frequency [Hz]");
ylabel("X1(f) [V]")

%final plot
figure(2)
plot(f, spectrum, '-b')
hold on
plot(f, spectrum2, '--r')

legend("Correctly sampled signal", "aliased signal")
title("Comparison of sufficiently and undersampled signal")
xlabel("frequency [Hz]")
ylabel("Amplitude [V]")


%% question 5

data = load('sines1.mat');

signalX = data.x;
signalY = data.y;

[corr, lag] = xcorr(signalX,signalY);

plot(lag, corr);


%% question 6

%load data
eeg = load('eeg_data.mat');
eeg = eeg.eeg;

Fs = 50;
T = 1/Fs;

%create the 801pt arrays. not the time is between 0 and 16.02s
t = linspace(0, 16.02, 801);

%now create array of sinusoidal waves
correlations = [];
minCorrelations = [];
frequencies = [];

for i = 1:25/0.25
    temp = sin(2*pi*(0.25*i)*t);
    [tempCorr, tempLag] = xcorr(eeg, temp);
    
    correlations(i) = max(tempCorr);
    minCorrelations(i) = min(tempCorr);
    frequencies(i) = i*0.25;
    
end

plot(frequencies, correlations,'-b', "LineWidth", 1.2);
title("Cross correlation of EEG and various sine waves")
xlabel("time [s]")
ylabel("Correlation amplitude");
hold on;

[Max, Index] = max(correlations);

%sort the array to get 5 max values
sortedCorr = sort(correlations);

indices = [];
maxes = [];
counter = 1;
for i = 2:length(correlations)-1
    if(correlations(i-1) < correlations(i) && correlations(i) > correlations(i+1))
        maxes(counter) = correlations(i);
        indices(counter) = i;
        counter=counter+1;
    end
end

%sort indices and take first 5
sortedMax = sort(maxes);
topMax =[frequencies(find(correlations==sortedMax(end))),frequencies(find(correlations==sortedMax(end-1))),frequencies(find(correlations==sortedMax(end-2))),frequencies(find(correlations==sortedMax(end-3))),frequencies(find(correlations==sortedMax(end-4)))];
topCorr = [correlations(find(correlations==sortedMax(end))), correlations(find(correlations==sortedMax(end-1))), correlations(find(correlations==sortedMax(end-2))), correlations(find(correlations==sortedMax(end-3))), correlations(find(correlations==sortedMax(end-4)))];

stem(topMax, topCorr);
hold off

%NOTE: The freqs which have the largest correlations are given in the below array
topMax

%now same follows for cosine

for i = 1:25/0.25
    temp2 = cos(2*pi*(0.25*i)*t);
    [tempCorr2, tempLag2] = xcorr(eeg, temp2);
    
    correlations2(i) = max(tempCorr2);
    minCorrelations2(i) = min(tempCorr2);
    frequencies2(i) = i*0.25;
    
end

figure(2)
plot(frequencies2, correlations2,'-b', "LineWidth", 1.2);
title("Cross correlation of EEG and various sine waves")
xlabel("time [s]")
ylabel("Correlation amplitude");
hold on;

[Max2, Index2] = max(correlations2);

%sort the array to get 5 max values
sortedCorr2 = sort(correlations2);

indices2 = [];
maxes2 = [];
counter2 = 1;
for i = 2:length(correlations2)-1
    if(correlations2(i-1) < correlations2(i) && correlations2(i) > correlations2(i+1))
        maxes2(counter2) = correlations2(i);
        indices2(counter2) = i;
        counter2=counter2+1;
    end
end

%sort indices and take first 5
sortedMax2 = sort(maxes2);
topMax2 =[frequencies2(find(correlations2==sortedMax2(end))),frequencies2(find(correlations2==sortedMax2(end-1))),frequencies2(find(correlations2==sortedMax2(end-2))),frequencies2(find(correlations2==sortedMax2(end-3))),frequencies2(find(correlations2==sortedMax2(end-4)))];
topCorr2 = [correlations2(find(correlations2==sortedMax2(end))), correlations2(find(correlations2==sortedMax2(end-1))), correlations2(find(correlations2==sortedMax2(end-2))), correlations2(find(correlations2==sortedMax2(end-3))), correlations2(find(correlations2==sortedMax2(end-4)))];

stem(topMax2, topCorr2);

%topmax for cosine
topMax2

%% question 7
H = [1/3 1/3 1/3];
whitenoise = randn(1, 512);

%filter
y = [];
for i = 3:length(whitenoise)
    y(i-2) = (1/3)*(whitenoise(i) + whitenoise(i-1) + whitenoise(i-2));
end

t = (0:length(whitenoise)-1)*(1/(length(whitenoise)-1));
plot(t, whitenoise);

figure(2)
[noiseCorr, lags] = xcorr(whitenoise, whitenoise);
normCorr = (noiseCorr-min(noiseCorr))./(max(noiseCorr) - min(noiseCorr));
plot(lags, normCorr);
hold on

[filtCorr,lags] = xcorr(y, y);
normCorr = (filtCorr-min(filtCorr))./(max(filtCorr) - min(filtCorr));
plot(lags, normCorr);

hold off
title("Autocorrelation Comparison")
xlabel("Lags [samples]")
ylabel("Normalized Correlation")
legend("Unfiltered Noise", "Filtered Noise")

%%by normalizing the autocorrelations,it is clear that although the
%%whitenoise is still very much uncorrelated sample to sample, the moving
%%average filter has caused some correlation between samples, 
%%because the filter output depends on the value of preceeding samples.
%% question 8

%create the auto correlation
%recall sampled data autocorrelation is 

[waveform_noise, time, waveform, snr_out] = sig_noise(300, -12, 256);

N = length(waveform_noise);
Fs = 1000;

f = Fs/(N-1) * (0:N-1);
fh = f(1:end/2);

%now take the fft of the noisey signal
Wfft = abs(fft(waveform_noise));
Wffth = Wfft(1:N/2);

figure(1)
subplot(211)
plot(fh, Wffth)
title("One side amplitude spectrum")
xlabel("Frequency [Hz]")
ylabel("Magnitude")

subplot(212)
plot(fh, Wffth.^2)
title("PSD from direct FFT")
xlabel("Frequency [Hz]")
ylabel("PSD Magnitude")

%now take the autocorrelation of the waveform
%note - can use xcorr since autocorrelation is simply the corss with the
%same function being shifted

[Rxx, lags] = xcorr(waveform_noise, waveform_noise);

N2 = length(Rxx);
Fs = 1000;
f2 = Fs/(N2-1) * (0:N2-1);
f2h = f2(1:end/2);

RxxFft = abs(fft(Rxx));
RxxFfth = RxxFft(1:N2/2);

figure(2);
subplot(211);

%plotting the single sidded fft
plot(f2h, RxxFfth)

subplot(212)
plot(f2h, RxxFfth.^2)

%Looking at the two spectra, its clear that the spectra are similar. They
%differ slightly but this could be due to the different scaling factor
%between the two. However, if the peaks are observed, the peaks have the
%same corresponding freq

%% question 9

%load data
ver1 = load('ver_problem2.mat').ver1;
ver = load('ver_problem2.mat').ver;
actual = load('ver_problem2.mat').actual_ver;

%ensable averages
avgVer = mean(ver);
avgVer1 = mean(ver1);

Ts = 0.005;
Fs = 1/Ts;

N = length(ver);
t = (0:N-1)*Fs;

%plotting the ensemble averages
figure(1);
plot(t, avgVer);
hold on 
plot(t, avgVer1);
legend('Fixed Refernce', 'Deviating Reference')

hold off

%analyze deviation from acutal

%calculate noise deviation by taking the difference between the two signals
verErr = actual - avgVer;
ver1Err = actual - avgVer1;

%taking LMS error
stdVer = std(verErr.^2)
stdVer1 = std(ver1Err.^2)

figure(2)
plot(t, verErr)
hold on
plot(t, ver1Err)

legend('Ver err signal', 'Ver1 error signal');
title("Error Signals")
xlabel("time [s]")
ylabel("Amplitude")

%clearly the LMS error of the signals show that the non varying refernce
%results in a VER measurement that is much more similar to the non noise
%contaminated signal

%% Question 10

Fc = 40;
Fs = 1000;
order = 65;

wc = Fc/(Fs/2);

b = fir1(order, wc, 'low', blackmanharris(order+1));

% applying the two methods

%grab the signal
signal = load('sawth.mat').x;
t=(0:length(signal)-1)/Fs;

filt_out_FIR = filter(b,1,signal);
filt_out_conv = conv(signal, b, 'same');

%plot the waveforms with the present group delay in the FIR filter.

plot(t, signal, "-b","LineWidth", 1.3)
hold on
plot(t, filt_out_FIR,"-r", "LineWidth", 1.3)
plot(t, filt_out_conv,"-g", "LineWidth", 1.3)
hold off

legend("original", "FIR Filter", "Convolution Filter");

%Can see due to the introduced delay from the casaul filter, the system
%starts later (peaks show up later in time) whereas the filtering done 
%through convolution shows the output of the filtered signal is 
%occurring before the actual signal, indicating non causality. 

%remove group delay and compare roundeness of peaks
delay = mean(grpdelay(b));

tt = t(1:end-delay);
convNew = filt_out_conv(1:end-delay);

signalf = filt_out_FIR;
signalf(1:delay+1) = [];

figure(2)
plot(tt,signalf)
hold on
plot(tt, convNew, '--r')
hold off
%the two filter outputs are the same, just shifted.
%most notably, the filtered signals are more rounded and more smoothley
%transition between minima and maxima, due to the degree of high order low
%pass filter. The filtered wave looks essentially like a moving average.

%% question 11

Fs = 1000;
Ts = 1/Fs;
Fc = 100;
wc = Fc/(Fs/2);

ord = 33;

fir = fir1(ord, wc, 'low', blackmanharris(ord+1));

impulse = zeros(1, 256);
impulse(1) = 1;

filt_FIR = filter(fir, 1, impulse);
filt_Cov = conv(impulse, fir, 'same');

t = (0:length(impulse)-1)*Ts;

index = find(t==0.05);

%plot the responses
plot(t(1:index), filt_FIR(1:index), '-b');
hold on
plot(t(1:index), filt_Cov(1:index), '-r');
hold off

fftFIR = fft(filt_FIR)/L;
fftCov = fft(filt_Cov)/L;
L = length(impulse);
f = Fs*(0:(L/2))/L;

%plot the amplitude spectrums
figure(2)

subplot(211)

plot(f, 2*abs(fftFIR(1:L/2+1)))
hold on 
plot(f, 2*abs(fftCov(1:L/2+1)))
hold off
title("Normalized Amplitude Spectrum")
xlabel("Frequency [Hz]")
ylabel("Amplitude |Y(f)|")

subplot(212)
phaseFIR = unwrap(angle(fftFIR(1:L/2+1)));
phaseCov = unwrap(angle(fftCov(1:L/2+1)));
plot(f.*(2*pi), phaseFIR)
hold on
plot(f.*(2*pi), phaseCov)
hold off
title("Phase Spectrum")
xlabel("Frequency [rad/s]")
ylabel("Phase |Y(f)|")
%now applying the shift
impulse = [zeros(1,10) 1 zeros(1,245)];

filt_FIR = filter(fir, 1, impulse);
filt_Cov = conv(impulse, fir, 'same');

fftFIR = fft(filt_FIR)/L;
fftCov = fft(filt_Cov)/L;

figure(3)
subplot(211)
plot(f, 2*abs(fftFIR(1:L/2+1)))
hold on 
plot(f, 2*abs(fftCov(1:L/2+1)))
hold off

subplot(212)
phaseFIR = unwrap(angle(fftFIR(1:L/2+1)));
phaseCov = unwrap(angle(fftCov(1:L/2+1)));
plot(f.*(2*pi), phaseFIR)
hold on
plot(f.*(2*pi), phaseCov)

%from the plots it can be clearly seen the frequency response as modelled
%from the amplitude spectrum is much different for non causal filters when
%the system starts right away. when the input signal is delayed, the freq
%response is the same, while for the most part maintaining the phase of the
%signal at higher frequencies.

%% question 12

ord = 8;
Fc = 100;
Fs = 1000;

wc = Fc/(Fs/2);

impulse = [1 zeros(1,255)];
L = length(impulse);

[b,a] = butter(ord, wc);

yf = filter(b,a,impulse);
yf2 = filtfilt(b,a, impulse);

t = (0:length(impulse)-1).*(1/Fs);

index = find(t==0.05);

figure(1)
plot(t(1:index), yf(1:index));
hold on 
plot(t(1:index), yf2(1:index))
hold off

legend("Butterworth by Filter", "IIR Butterwroth")

%now the magnitude and phase respone
fftBW = fft(yf)/L;
fftFF = fft(yf2)/L;

f = Fs*(0:(L/2))/L;
fftBW = fftBW(1:(L/2)+1);

fftFF = fftFF(1:(L/2)+1);

figure(2)
subplot(211)
plot(f, abs(fftBW).*2)
hold on
plot(f, abs(fftFF).*2)
legend("BW Filter", "IIR Filter")
hold off

%phase
subplot(212)
plot(f.*(2*pi), unwrap(angle(fftBW(1:(L/2)+1))))
hold on
plot(f.*(2*pi), unwrap(angle(fftFF(1:(L/2)+1))))
legend("BW Filter", "IIR Filter")
hold off



%rerun to show absence of ripple
impulse = [zeros(1,20) 1 zeros(1, 235)];

yf = filter(b,a,impulse);
yf2 = filtfilt(b,a, impulse);

fftBW = fft(yf)/L;
fftFF = fft(yf2)/L;

f = Fs*(0:(L/2))/L;
fftBW = fftBW(1:(L/2)+1);

fftFF = fftFF(1:(L/2)+1);

figure(3)
subplot(211)
plot(f, abs(fftBW).*2)
hold on
plot(f, abs(fftFF).*2)
legend("BW Filter", "IIR Filter")
hold off

%phase
subplot(212)
plot(f.*(2*pi), unwrap(angle(fftBW(1:(L/2)+1))))
hold on
plot(f.*(2*pi), unwrap(angle(fftFF(1:(L/2)+1))))
legend("BW Filter", "IIR Filter")
hold off

