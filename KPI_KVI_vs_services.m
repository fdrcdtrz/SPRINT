%% Two separate plots: total KPI and KVI VS N_servizi + N_risorse per i quattro approtest_results_cci


base_path = 'C:\Users\Federica de Trizio\PycharmProjects\CutAndSolve'; % Path

% Select folders with the files 

% folders_proposed = {'benchmark_80_80_0.1_ris_tutte_scarse', 'benchmark_85_80_0.1_ris_tutte_scarse', 'benchmark_90_80_0.1_ris_tutte_scarse', 'benchmark_95_80_0.1_ris_tutte_scarse', 'benchmark_100_80_0.1_ris_tutte_scarse', 'benchmark_105_80_0.1_ris_tutte_scarse', 'benchmark_110_80_0.1_ris_tutte_scarse', 'benchmark_115_80_0.1_ris_tutte_scarse', 'benchmark_120_80_0.1_ris_tutte_scarse'};
% folders_benchmark = {'benchmark_80_80_0.1_ris_tutte_scarse', 'benchmark_85_80_0.1_ris_tutte_scarse', 'benchmark_90_80_0.1_ris_tutte_scarse', 'benchmark_95_80_0.1_ris_tutte_scarse', 'benchmark_100_80_0.1_ris_tutte_scarse', 'benchmark_105_80_0.1_ris_tutte_scarse', 'benchmark_110_80_0.1_ris_tutte_scarse', 'benchmark_115_80_0.1_ris_tutte_scarse', 'benchmark_120_80_0.1_ris_tutte_scarse'};

folders_proposed = {'benchmark_80_80_0.05_ris', 'benchmark_85_80_0.05_ris', 'benchmark_90_80_0.05_ris', 'benchmark_95_80_0.05_ris', 'benchmark_100_80_0.05_ris', 'benchmark_105_80_0.05_ris', 'benchmark_110_80_0.05_ris', 'benchmark_115_80_0.05_ris', 'benchmark_120_80_0.05_ris'};
folders_benchmark = {'benchmark_80_80_0.05_ris', 'benchmark_85_80_0.05_ris', 'benchmark_90_80_0.05_ris', 'benchmark_95_80_0.05_ris', 'benchmark_100_80_0.05_ris', 'benchmark_105_80_0.05_ris', 'benchmark_110_80_0.05_ris', 'benchmark_115_80_0.05_ris', 'benchmark_120_80_0.05_ris'};

% Extract data for all considered approaches
[kpi_values_proposed, kvi_values_proposed] = extract_pareto_max(folders_proposed, base_path);
[kpi_values_greedy_kpi, kvi_values_greedy_kpi] = extract_greedy_kpi_max(folders_benchmark, base_path);
[kpi_values_random, kvi_values_random] = extract_random_max(folders_benchmark, base_path);
[kpi_values_greedy_kvi, kvi_values_greedy_kvi] = extract_greedy_kvi_max(folders_benchmark, base_path);

%%  Plot

num_services = [80, 85, 90, 95, 100, 105, 110, 115, 120]; % # of services

% Construct matrices for plot
kpi_data = [kpi_values_proposed; kpi_values_greedy_kpi; kpi_values_random; kpi_values_greedy_kvi];
kvi_data = [kvi_values_proposed; kvi_values_greedy_kpi; kvi_values_random; kvi_values_greedy_kvi];

% Generate plot
plot_kpi_kvi(num_services, kpi_data, kvi_data);



%% Functions
function [kpi_vector, kvi_vector] = extract_pareto_max(selected_folders, base_path)
    kpi_vector = [];
    kvi_vector = [];

    for i = 1:length(selected_folders)
        folder_path = fullfile(base_path, selected_folders{i});
        file_path = fullfile(folder_path, 'pareto_solutions.csv');

        if isfile(file_path)
            data = readmatrix(file_path);
            if isempty(data)
                warning('File vuoto: %s', file_path);
                continue;
            end
            
            numRows = size(data, 1);
            
            % Find index of the middle row
            midIdx = ceil(numRows / 2);
            
            % Extract it
            midRow = data(midIdx, :);

            
            % Select max KPI and KVI
            kpi_max = midRow(1);%max(data(:, 1));
            kvi_max = midRow(2);%max(data(:, 2)); 

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
                warning("File is empty: %s", file_path);
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
                warning("Dati not valid in %s", file_path);
            end
        else
            warning("File not found: %s", file_path);
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
                warning("File vuoto: %s", file_path);
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

function plot_kpi_kvi(num_services, kpi_data, kvi_data)
    % Different styles
    styles = {'-o', '-s', '-d', '-'};
    colors = {'b', 'r', 'g'}; 
    
    % --- Plot KPI ---
    figure;
    hold on;
    for i = 1:4
        plot(num_services, kpi_data(i,:), styles{i}, 'LineWidth', 3.5, 'MarkerSize', 8);
    end
    set(gca, 'FontSize', 40, 'GridAlpha', 0.3, 'FontName', 'Times New Roman');
    xlabel('Number of Services');
    ylabel('Total network quality performance');
    %title('KPI vs Number of Services');
    legend({'This work', 'PGM', 'RM', 'VGM'}, 'Location', 'best');
    grid on;
    hold off;
    
    % --- Plot KVI ---
    figure;
    hold on;
    for i = 1:4
        plot(num_services, kvi_data(i,:), styles{i}, 'LineWidth', 3.5, 'MarkerSize', 8);
    end
    set(gca, 'FontSize', 40, 'GridAlpha', 0.3, 'FontName', 'Times New Roman');
    xlabel('Number of Services');
    ylabel('Total network social and ethical value');
    %title('KVI vs Number of Services');
    legend({'This work', 'PGM', 'RM', 'VGM'}, 'Location', 'best');
    grid on;
    hold off;
end


