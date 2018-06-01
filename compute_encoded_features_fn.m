function compute_encoded_features_fn(encoded_features_file, dataset, sets, features, encoders, pca_enabled)
% TODO no pca_enabled, just check if preprocessing step is present
if exist(encoded_features_file, 'file')
    load(encoded_features_file)
else
    labels = zeros(0);
    enc_features = [];
    cursor = 0;

    for set = sets
        set = set{1}
        
        for enum_i=1:size(dataset.(set).all, 1)
            labels = [labels dataset.(set).all_labels(enum_i)];
        end
        
        for enum_i=1:size(dataset.(set).all, 1)
            enum_i
            feats = [];
            tmp_feat = features.(set){enum_i}.feats;
			tmp_feat = (tmp_feat - encoders.preprocessing.mean) ./ encoders.preprocessing.std;
            if pca_enabled
                tmp_feat = tmp_feat' * encoders.pca.coeff;
				tmp_feat = tmp_feat';
			end
			tmp_feat = vl_fisher(tmp_feat, encoders.encoder.means, encoders.encoder.covariances, encoders.encoder.priors, 'Improved') ; %single vs multi??
            feats = [feats tmp_feat];

            if isempty(enc_features)
                total_samples = 0;
                for f=fields(features)'
                    f = char(f);
                    total_samples = total_samples + numel(features.(f));
                end
                enc_features = zeros(total_samples, numel(feats));
            end
            cursor = cursor + 1;
            enc_features(cursor, :) = feats;
        end
    end
    save(encoded_features_file, 'enc_features', 'labels','-v7.3')
end
end