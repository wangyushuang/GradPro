%读取bin文件
%data(:,ch)表示第ch通道的数据 ch=1:12
%data(:,13)表示标准水听器的输出数据 
%%
clear all;
close all;
pname='F:\2018\March\Data\data3_9\bp\bp90KHz-100KHz_LFM_40dB\';

fname='Board0_ADC_0.bin';
mode = 3;
FS=250e3;
PackPoint=17;

gain = 1;

%f0 =100e3;  % find the position of sin in FFT  

CH = 1;

%Test_Noise2(mode,0,0,FS,CH,PackPoint,pname,fname,gain);         %mode 1: draw ch1 2:draw 16 PSD

%Test_bwnoise(mode,1,1,FS,PackPoint,pname,fname,gain,f0); %Test_bwnoise(mode,from,to,FS,PackPoint,pname,fname,gain)

Test_Noise3(mode,0,0,FS,CH,PackPoint,pname,fname,gain); 

%%
clear all;

load tmp;
