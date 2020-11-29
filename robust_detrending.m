%% define parameters
ft_defaults
num_of_files = 6;
%%
for subj_num = 2:3
    %load data
    for recording_type = ["Visual","Auditory"]
        if recording_type == "Visual"
            file_start_string = "vis";
        else
            file_start_string = "aud";
        end
        datadir = sprintf('//ems.elsc.huji.ac.il/deouell-lab/Lab-Shared/Experiments/HighDenseGamma/results/EEG/%s/Raw/',recording_type);
        cd(datadir)
        file_names = [];
        for i=1:num_of_files
            if isfile(sprintf("%s_s%d_%d.bdf",file_start_string,subj_num,i))
                file_names = [file_names sprintf("%s_s%d_%d.bdf",file_start_string,subj_num,i)];
            end
        end
        data_array = cell(size(file_names));
        header_array = cell(size(file_names));
        event_array = cell(size(file_names));
        for i=1:length(file_names)
            header_array{i} = ft_read_header([datadir,convertStringsToChars(file_names(i))]);
            data_array{i} = ft_read_data([datadir,convertStringsToChars(file_names(i))])';
            event_array{i} = ft_read_event([datadir,convertStringsToChars(file_names(i))])';
        end

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
        save(sprintf("//ems.elsc.huji.ac.il/deouell-lab/Lab-Shared/Experiments/HighDenseGamma/Analyses/Python/DensegridPreprocessing/RawFiles/S%d/detrended_%s_s%d.mat",subj_num,lower(recording_type),subj_num),'detrended_data','-v7.3')
        disp('done.')
    end
end