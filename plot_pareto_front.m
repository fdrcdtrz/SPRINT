%% Pareto front 

clc; clear; close all;

% Set the main path
main_folder = 'C:\Users\Federica de Trizio\PycharmProjects\CutAndSolve'; % to modify with the path

% Finds all the folders names "results_*"
folders = dir(fullfile(main_folder, 'results_*'));

% Select specific folders with files to plot
result_names = ["fig3_120_80_0.1", "fig3_240_160_0.1","fig3_480_320_0.1", "fig3_720_480_0.1", "fig3_960_640_0.1", "fig3_1200_800_0.1"]; % da cambiare: prima tutti 150/160/170/180/190, fisso pesi, vario delta, poi vario pesi e fisso delta. 10 plot in totale


% Either set different colors for each curve by default or by specifying
% them
%colors = lines(length(result_names));

colorList = {
    '#1f77b4',  % blu
    '#ff7f0e',  % arancio
    '#2ca02c',  % verde
    '#d62728',  % rosso
    '#9467bd',  % viola
    '#8c564b',  % marrone
    '#e377c2',  % rosa
    '#7f7f7f',  % grigio
    '#bcbd22',  % giallo-verde oliva
    '#17becf',  % azzurro
    '#aec7e8',  % blu chiaro
    '#ffbb78',  % arancio chiaro
    '#98df8a',  % verde chiaro
    '#ff9896',  % rosso salmone
    '#c5b0d5'   % lilla chiaro
};


% Variables for axes limits
all_KPI = [];
all_KVI = [];

figure; % Create a single plot
hold on;
legend_entries = {};

% Loop on each result folder
for i = 1:length(result_names)
    folder_name = result_names(i);
    folder_path = fullfile(main_folder, folder_name);

    % Pareto file path
    pareto_file = fullfile(folder_path, 'pareto_solutions.csv');

    if exist(pareto_file, 'file')
        % Read Pareto files
        data = readtable(pareto_file);

        % Extract total KPI and KVI 
        KPI_Totale = data.KPI_Totale;
        KVI_Totale = data.KVI_Totale;

        % Store all the points for limits computation on axes
        all_KPI = [all_KPI; KPI_Totale];
        all_KVI = [all_KVI; KVI_Totale];

        % Plot Pareto curves with different colors. Replace with the other
        % lines if the other color option is chosen.
        plot(KPI_Totale, KVI_Totale, 'o-', 'LineWidth', 3, ...
            'MarkerSize', 8, ...
            'Color', colorList{i});
        % plot(KPI_Totale, KVI_Totale, 'o-', 'LineWidth', 3, ...
        % 'MarkerSize', 8, ...
        % 'Color', colors(i,:));

        % Add legend entries
        %legend_entries{end+1} = strrep(folder_name, '_', ' '); % Replaces
        %_ with spaces to enhance readability
    else
        disp(['Pareto file Pareto not found: ' pareto_file]); 
    end
end

% Improvement of the plot aesthetic-wise
xlabel('Network quality performance');
ylabel('Network social and ethical value');
%title('Pareto Fronts for Different Configurations');
grid on;


% Set the legend la legenda
legend({'\delta = 0.01', '\delta = 0.1', '\delta = 1'}, 'Location', 'Best');
legend({'J = 100, N = 35','J = 240, N = 160','J = 480, N = 320','J = 720, N = 480','J = 960, N = 640', 'J = 1200, N = 800'}, 'Location', 'Best');


% Set axes limitations wrt to the collected data
xlim([min(all_KPI) - 0.05 * range(all_KPI), max(all_KPI) + 0.05 * range(all_KPI)]);
ylim([min(all_KVI) - 0.05 * range(all_KVI), max(all_KVI) + 0.05 * range(all_KVI)]);

% Set font and sizes
set(gca, 'FontSize', 40, 'GridAlpha', 0.3, 'FontName', 'Times New Roman');

% Keep aspect ratio
%pbaspect([1 1 1]);
%exportgraphics(gcf,'120_80.eps','ContentType','vector','BackgroundColor','none')

% Save plot in .eps format
%export_file = fullfile(main_folder, 'pareto_front_all.fig');
%exportgraphics(gcf, export_file, 'ContentType', 'vector', 'BackgroundColor', 'none');

hold off;
