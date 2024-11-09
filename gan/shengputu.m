mkdir('C:\Users\11148\Desktop\声纹识别\specgram\13\');%新建存放语谱图的文件夹
file='D:\1\lol语音包\瑞文\';%声音信号所在文件夹
file1=strcat(file,'*.wav');%file1='C:\Users\如初\Desktop\matlab程序\*.wav'（strcat:连接多个字符串）
file2=dir(file1);%获取文件夹下所有的wav文件（dir:读取文件夹下所有文件和文件夹/特定后缀的文件）
k=length(file2);%计算wav文件个数（length:数组长度,行数或者列数的较大值  size:数组的行数和列数 numel:元素总数）
R=1024;%窗长
window=hamming(R);%汉明窗
N=1024;%fft个数，与窗长相同
L=512;%步长
overlap=R-L;%帧长重叠部分，也叫帧移，一般重叠部分为50%
for i=1:k
    file3=strcat(file,file2(i).name);%单个音频的绝对路径
    [x,fs]=audioread(file3);%读取单个音频，fs是采样频率
    x1=x(:,1);%x为双声道，所以后面取单声道数据
    %plot(x1)%可以画波形图
    file4=strcat('C:\Users\11148\Desktop\声纹识别\specgram\13\',file2(i).name,'.jpg');%语谱图命名方式和存储文件夹
    %x1=awgn(x1,100,'measured','linear');%可以加白噪
    figure(i);
    specgram(x1,N,fs,window,overlap);%第一种：specgram无输出的时候，直接画出来
    saveas(gca,file4);%保存语谱图图片
end
    %[B,f,t]=specgram(x(:,1),N,fs,window,overlap);%第二种：specgram输出返回值，其中B为振幅，f为频率，t为时间
    %imagesc(t,f,20*log10(abs(B)));%根据返回参数，画出语谱图（imagesc:画图，参数分别为横坐标范围、纵坐标范围和数值），除了语谱图还可以画别的
    %plot(f,20*log10(abs(B(:,21))))%频域：B(:,21)对应的是t[21]=0.23s时的频谱
    %plot(t,20*log10(abs(B(21,:))))%时域：B（21:,：)对应的是f[21]=816.32Hz时的时域波形
    %colormap(cool);%图形变色系
    %axis xy%注意axis ij和axis xy的区别
    %xlabel('时间/s');%横坐标命名
    %ylabel('频率/kHz');%纵坐标命名