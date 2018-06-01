function features = compute_conv_features_fn(features_file,features_file_cp, dataset, net)
% TODO parametrize checkpoint saving
features = struct;

if exist(features_file, 'file')
    load(features_file)
    'loaded features'
end

if exist(features_file_cp, 'file')
    fprintf('Trovato checkpoint, loading...\n');
    checkpoint = load(features_file_cp);
end

for set=dataset.sets
    set=char(set);
    if ~isfield(features, set)
        tmp_set = cell(size(dataset.(set).all, 1), 1);
        step = 1000;
        for cursor=0:step:size(dataset.(set).all, 1)+step
            if exist('checkpoint', 'var') && (cursor+step) <= checkpoint.i && strcmp(checkpoint.set,  set)
                tmp_set(cursor+1:cursor+step) = checkpoint.tmp_set(cursor+1:cursor+step);
                continue
            end
            parfor i=cursor+1:min(step+cursor, size(dataset.(set).all, 1))
                i
                tmp_path = dataset.(set).all{i};
                tmp_feats = compute_img_features_fn(tmp_path, [size(imread(tmp_path), 1) size(imread(tmp_path), 2)], net, []);
                tmp_feats.feats = vl_colsubset(tmp_feats.feats, 1000);
                tmp_set{i} = tmp_feats;
            end
            i = min(step+cursor, size(dataset.(set).all, 1));
            fprintf('Salvo checkpoint....\n')
%             save(features_file_cp, 'set', 'i', 'tmp_set', '-v7.3');
        end

        features.(set) = tmp_set;
        save(features_file, 'features', '-v7.3')
    end
    
end
end