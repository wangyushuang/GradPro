function yout=MatchedFilter(s, ref)
% s-信号
%ref-参考信号
Ns=length(s);
Nr=length(ref);
yout=zeros(Ns,1);
% 需要保证Nr为奇数
if(rem(Nr,2)==0) %如果为偶数，则去掉最后一个点
    Nr=Nr-1;
end
Nrh=(Nr+1)/2;%参考信号中点
for k=1:Ns
%     for h=max(1,Nrh-k+1):min(Nr,Nrh+Ns-k)    %循环导致处理时间长    
%         yout(k)=yout(k)+ref(h)*s(k-Nrh+h);
%     end    
    range=[max(1,Nrh-k+1):min(Nr,Nrh+Ns-k)];
    yout(k)=ref(range)*s(k-Nrh+range)'; %利用矩阵乘法替代内层循环
end

%% 测试案例
% t=linspace(0,1,1000);
% f=10e3;
% ref=sin(2*pi*f*t);
% s(1:1000)=0;
% s(1001:2000)=ref;
% s(2001:3000)=0;
% yout=MatchedFilter(s,ref);
% subplot 311
% plot(ref);
% title('参考信号')
% subplot 312 
% plot(s)
% title('输入信号')
% subplot 313
% plot(yout)
% title('输出信号')


