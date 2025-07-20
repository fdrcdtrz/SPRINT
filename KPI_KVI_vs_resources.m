%% Two different plots: total KPI e KVI totali VS N_servizi and N_risorse for the considered approaches


base_path = 'C:\Users\Federica de Trizio\PycharmProjects\CutAndSolve'; % Correct path

%% Folders
folders_proposed = {'benchmark_100_50_0.05_ris', 'benchmark_100_55_0.05_ris', 'benchmark_100_60_0.05_ris', 'benchmark_100_65_0.05_ris', 'benchmark_100_70_0.05_ris', 'benchmark_100_75_0.05_ris', 'benchmark_100_80_0.05_ris', 'benchmark_100_85_0.05_ris', 'benchmark_100_90_0.05_ris'};
folders_benchmark = {'benchmark_100_50_0.05_ris', 'benchmark_100_55_0.05_ris', 'benchmark_100_60_0.05_ris', 'benchmark_100_65_0.05_ris', 'benchmark_100_70_0.05_ris', 'benchmark_100_75_0.05_ris', 'benchmark_100_80_0.05_ris', 'benchmark_100_85_0.05_ris', 'benchmark_100_90_0.05_ris'};


[kpi_values_proposed, kvi_values_proposed] = extract_pareto_max(folders_proposed, base_path);
[kpi_values_greedy_kpi, kvi_values_greedy_kpi] = extract_greedy_kpi_max(folders_benchmark, base_path);
[kpi_values_greedy_kvi, kvi_values_greedy_kvi] = extract_greedy_kvi_max(folders_benchmark, base_path);
[kpi_values_random, kvi_values_random] = extract_random_max(folders_benchmark, base_path);


%% 

num_resources = [50, 55, 60, 65, 70, 75, 80, 85, 90]; % # of resources

kpi_data = [kpi_values_proposed; kpi_values_greedy_kpi; kpi_values_random; kpi_values_greedy_kvi];
kvi_data = [kvi_values_proposed; kvi_values_greedy_kpi; kvi_values_random; kvi_values_greedy_kvi];

% Generates the plots
plot_kpi_kvi(num_resources, kpi_data, kvi_data);



%% FUNCTIONS

function [kpi_vector, kvi_vector] = extract_pareto_max(selected_folders, base_path)
    kpi_vector = [];
    kvi_vector = [];

    for i = 1:length(selected_folders)
        folder_path = fullfile(base_path, selected_folders{i});
        file_path = fullfile(folder_path, 'pareto_solutions.csv');

        if isfile(file_path)
            data = readmatrix(file_path);
            if isempty(data)
                warning('File empty: %s', file_path);
                continue;
            end

            numRows = size(data, 1);
            
            % Find the middle row index
            midIdx = ceil(numRows / 2);
            
            % Extract it
            midRow = data(midIdx, :);

            
            % Select max KPI and KVI value
            kpi_max = midRow(1);%max(data(:, 1)); %
            kvi_max = midRow(2);%max(data(:, 2)); % 

            kpi_vector = [kpi_vector, kpi_max];
            kvi_vector = [kvi_vector, kvi_max];
        else
            warning('File not found: %s', file_path);
        end
    end
end


function [kpi_vector, kvi_vector] = extract_greedy_kpi_max(selected_folders, base_path)
    kpi_vector = [];
    kvi_vector = [];

    for i = 1:length(selected_folders)
        folder_path = fullfile(base_path, selected_folders{i});
        file_path = fullfile(folder_path, 'greedy_kpi_results.csv');

        if isfile(file_path)
            rawData = readlines(file_path);
            validRows = rawData(~strcmp(strtrim(rawData), "") & strlength(strtrim(rawData)) > 0);
            if isempty(validRows)
                warning("File empty: %s", file_path);
                continue;
            end
            
            lastRow = strtrim(validRows(end));
            values = regexp(lastRow, '[+-]?\d*\.?\d+', 'match');

            if length(values) >= 2
                kpi_vector = [kpi_vector, str2double(values{1})];
                kvi_vector = [kvi_vector, str2double(values{2})];
            else
                warning("Data not valid in %s", file_path);
            end
        else
            warning("File not found: %s", file_path);
        end
    end
end

function [kpi_vector, kvi_vector] = extract_random_max(selected_folders, base_path)
    kpi_vector = [];
    kvi_vector = [];

    for i = 1:length(selected_folders)
        folder_path = fullfile(base_path, selected_folders{i});
        file_path = fullfile(folder_path, 'random_results.csv');

        if isfile(file_path)
            rawData = readlines(file_path);
            validRows = rawData(~strcmp(strtrim(rawData), "") & strlength(strtrim(rawData)) > 0);
            if isempty(validRows)
                warning("File empty: %s", file_path);
                continue;
            end
            
            lastRow = strtrim(validRows(end));
            values = regexp(lastRow, '[+-]?\d*\.?\d+', 'match');

            if length(values) >= 2
                kpi_vector = [kpi_vector, str2double(values{1})];
                kvi_vector = [kvi_vector, str2double(values{2})];
            else
                warning("Data not valid in %s", file_path);
            end
        else
            warning("File non trovato: %s", file_path);
        end
    end
end


function [kpi_vector, kvi_vector] = extract_greedy_kvi_max(selected_folders, base_path)
    kpi_vector = [];
    kvi_vector = [];

    for i = 1:length(selected_folders)
        folder_path = fullfile(base_path, selected_folders{i});
        file_path = fullfile(folder_path, 'greedy_kvi_results.csv');

        if isfile(file_path)
            rawData = readlines(file_path);
            validRows = rawData(~strcmp(strtrim(rawData), "") & strlength(strtrim(rawData)) > 0);
            if isempty(validRows)
                warning("File empty: %s", file_path);
                continue;
            end
            
            lastRow = strtrim(validRows(end));
            values = regexp(lastRow, '[+-]?\d*\.?\d+', 'match');

            if length(values) >= 2
                kpi_vector = [kpi_vector, str2double(values{1})];
                kvi_vector = [kvi_vector, str2double(values{2})];
            else
                warning("Data not valid in %s", file_path);
            end
        else
            warning("File not found: %s", file_path);
        end
    end
end

function plot_kpi_kvi(num_resources, kpi_data, kvi_data)
    % Styles
    styles = {'-o', '-s', '-d', '-'};
    colors = {'k', 'k', 'k'}; 
    %plot(num_resources, kpi_data(i,:), styles{i}, 'Color', colors{i}, 'LineWidth', 2.5, 'MarkerSize', 8);
    
    figure;
    hold on;
    for i = 1:4
        plot(num_resources, kpi_data(i,:), styles{i}, 'LineWidth', 3.5, 'MarkerSize', 8);
    end
    set(gca, 'FontSize', 40, 'GridAlpha', 0.3, 'FontName', 'Times New Roman');
    xlabel('Number of Resources');
    ylabel('Total network quality performance');
    legend({'This work', 'PGM', 'RM', 'VGM'}, 'Location', 'best');
    grid on;
    hold off;
    
    figure;
    hold on;
    for i = 1:4
        plot(num_resources, kvi_data(i,:), styles{i}, 'LineWidth', 3.5, 'MarkerSize', 8);
    end
    set(gca, 'FontSize', 40, 'GridAlpha', 0.3, 'FontName', 'Times New Roman');
    xlabel('Number of Resources');
    ylabel('Total network social and ethical value');
    legend({'This work', 'PGM', 'RM', 'VGM'}, 'Location', 'best');
    grid on;
    hold off;
end


