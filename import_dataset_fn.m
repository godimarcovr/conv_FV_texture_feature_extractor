function import_dataset_fn(dataset)
dataset.classes = dir(fullfile(dataset.root, dataset.sets{1}));
dataset.classes = {dataset.classes(3:end).name};
for set=dataset.sets
	set=char(set)
	dataset.(set).path = fullfile(dataset.root, set);
	dataset.(set).all = {};
	dataset.(set).all_labels = [];
	
	count_cl = 0;
	for cl=dataset.classes
		count_cl = count_cl + 1;
		cl = char(cl);
		tmp = dir(fullfile(dataset.(set).path, cl, '*.*'));
		tmp = tmp(3:end);
		dataset.(set).(cl) = cell(size(tmp, 1), 1);
		for i=1:size(tmp, 1)
			dataset.(set).(cl){i} = fullfile(tmp(i).folder, tmp(i).name);
			% [filepath,name,ext] = fileparts(dataset.(set).(cl){i});
		end
		dataset.(set).all = [dataset.(set).all; dataset.(set).(cl)];
		dataset.(set).all_labels = [dataset.(set).all_labels; ones(size(dataset.(set).(cl), 1), 1) .* count_cl];
	end
end
save('dataset.mat', 'dataset')
end