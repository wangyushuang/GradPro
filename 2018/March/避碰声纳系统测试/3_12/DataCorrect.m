%% 由于水听器与采集器正负接线分不清，因此根据波形校正相位
%以1号阵元为参考，与其相位相反的波形进行反相处理
%内外圈差0.7个周期
for ch=[2 7 8 10 12]
    data(:,ch)=-data(:,ch);
end

%  figure
%  i=1;
%  for ch=1:4
%      subplot(4,1,i)
%      i=i+1;
%      plot(data(1:psec,ch),'.-')
%      xlim([1.75768e5 1.75798e5])
%      ylim([-2 2])
%  end