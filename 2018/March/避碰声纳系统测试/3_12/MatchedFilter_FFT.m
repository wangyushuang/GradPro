function yout=MatchedFilter_FFT(s, ref)
%% 频域实现
N=length(s);
yout=ifftshift(ifft(fft(s,N*2-1).*conj(fft(ref,N*2-1))));

%% 测试案例
% t=linspace(0,1,1000);
% f=20e3;
% ref=sin(2*pi*f*t);
% s(1:1000)=0;
% s(1001:2000)=ref;
% s(2001:3000)=0;
% yout=MatchedFilter_FFT(s,ref);
% subplot 311
% plot(ref);
% xlim([0 3000])
% title('参考信号')
% subplot 312 
% plot(s)
% title('输入信号')
% subplot 313
% plot(yout)
% title('输出信号')