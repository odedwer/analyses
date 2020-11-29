%% define parameters
recording_type = 'Visual'; % 'Visual' or 'Auditory'
subj_num = 2;
num_of_files = 3;
datadir=sprintf('S:/Lab-Shared/Experiments/HighDenseGamma/results/EEG/%s/Raw/',recording_type);

%% load all of the auditory data
ft_defaults
file_names = [];
for i=2:num_of_files
    file_names = [file_names sprintf("vis_s%d_%d.bdf",subj_num,i)];
end
savedir=[];%'./PP/'; % directory to save results
data_array = cell(size(file_names));
header_array = cell(size(file_names));
event_array = cell(size(file_names));
for i=1:length(file_names)
    header_array{i} = ft_read_header([datadir,convertStringsToChars(file_names(i))]);
    data_array{i} = ft_read_data([datadir,convertStringsToChars(file_names(i))])';
    event_array{i} = ft_read_event([datadir,convertStringsToChars(file_names(i))])';
end
%% save things for easy access
blk = 1;
channel = 'A3';

chan_num = find(strcmp(header_array{blk}.label,channel));
data = data_array{blk}(:,chan_num);

%get indices of onsets
onsets = [];
for i=1:length(event_array{blk})
    if event_array{blk}(i).value==12
        onsets = [onsets;event_array{blk}(i).sample];
    elseif event_array{blk}(i).value==22
        onsets = [onsets;event_array{blk}(i).sample];
    end    
end

%% plot our chosen channel

%  tmp = zeros(size(data));
%  tmp(2048*140:end)=100;
%  data = data +tmp;

ERPfigure();subnum=1;nsubplts=4;
hax = [];
hax(subnum)=subplot(nsubplts,1,subnum);
%plot((1:length(data))/2048,data-mean(data))
plot(data)
title('Data')
subnum=subnum+1;

%% destep - decided not to do this, erase our changes to their code!!!
[y,stepList]=nt_destep(data,[],8*2048); %,thresh,guard,depth,minstep)
tit = 'desteped';

hax(subnum)=subplot(nsubplts,1,subnum);
%plot((1:length(data))/2048,y);hold on
a = plot(y);hold on
title(tit)
subnum=subnum+1;

%% do robust detrending - polynomial

ord = 10;
tit = sprintf('Ord %d',ord);
[y,w] = nt_detrend(data,ord);

hax(subnum)=subplot(nsubplts,1,subnum);
plot((1:length(data))/2048,y);hold on
%scatter(find(~w),ones(1,length(find(~w))),'r*')
title(tit)
subnum=subnum+1;

%% do robust detrending - polynomial with windows

ord = 10;win=25*2048;
tit = sprintf('Ord %d Win %d',ord,win);
[y,w] = nt_detrend(data,ord,[],[],[],[],win);

hax(subnum)=subplot(nsubplts,1,subnum);
%plot((1:length(data))/2048,y);hold on
plot(y);hold on
scatter(find(~w),ones(1,length(find(~w))),'b*')
title(tit)
subnum=subnum+1;

%% robust detrending no erp
ord = 10;
window = 0:round(0.5*2048); %num of timepoints to exclude 

tit = sprintf('Ord %d, no ERP',ord);
w=ones(size(data));
w(onsets+window)=0;
[y,w,r] = nt_detrend(data,ord,w);

hax(subnum)=subplot(nsubplts,1,subnum);
plot(y);hold on
scatter(find(~w),ones(1,length(find(~w))),'r*')
title(tit)
subnum=subnum+1;

%% do robust detrending - sinusoids

ord = 6;
[y,w,r]=nt_detrend(data,ord,[],'sinusoids');
tit = sprintf('Ord %d, sinusoids',ord);

hax(subnum)=subplot(nsubplts,1,subnum);
plot(y);hold on
scatter(find(~w),ones(1,length(find(~w))),'r*')
title(tit)
subnum=subnum+1;

%% compare to HPF
cutoff = 1;

y = HPF(data,2048,cutoff);
tit = sprintf('HPF %0.1fHz',cutoff);

hax(subnum)=subplot(nsubplts,1,subnum);

plot(y);
%plot((1:length(data))/2048,y);

title(tit)
subnum=subnum+1;

%%
xlim(hax,[1000 1100])

%% detrend all data of subject, save as 3d cell array of detrended blocks

detrended_data = cell(size(data_array));
ord = 10;
window = 0:round(0.5*2048); %num of timepoints to exclude 
relevant_events = [12 22];
for i=1:length(detrended_data)
    disp(i)
    onsets = [];
    disp('getting onsets...')
    for j=1:length(event_array{i})
        if event_array{i}(j).value==12
            onsets = [onsets;event_array{i}(j).sample];
        elseif event_array{i}(j).value==22
            onsets = [onsets;event_array{i}(j).sample];
        end    
    end
    w=ones(size(data_array{i}(:,1)));
    w(onsets+window)=0;
    disp('starting detrending...')
    [y,w] = nt_detrend(data_array{i},ord,w,[],[],[],10*2048);
    detrended_data{i} = y/(10^6);  % fix units for mne
    disp('done.')
end
disp('saving...')
save(sprintf("S:/Lab-Shared/Experiments/HighDenseGamma/Analyses/Python/DensegridPreprocessing/RawFiles/S3/detrended_%s_s%d.mat",lower(recording_type),subj_num),'detrended_data','-v7.3')
disp('done.')
