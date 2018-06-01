function encoders = train_gmm_fn(dataset, features, encoder_file, set, n_imgs, max_descrs_per_img, pca_enabled, dcnn, numWords)
encoders = struct;
if exist(encoder_file, 'file')
    load(encoder_file)
else
    if ~isfield(encoders, 'postpca')
        encoders.encoder.numWords = numWords; %da 
        encoders.encoder.projection = 1 ;
        encoders.encoder.projectionCenter = 0 ;
        encoders.encoder.encoderType = dcnn.opts.encoderType;
        encoders.encoder.renormalize = false ;
        enum_i = 0;
        index_subset = randperm(numel(dataset.(set).all));
        index_subset = index_subset(1:n_imgs);
        labels = zeros(0);
        descrs = zeros(512, 0);
        for enum_i=index_subset
            feats = [];
            feats = [feats vl_colsubset(features.(set){enum_i}.feats, max_descrs_per_img)];
            descrs = [descrs feats];
        end
		encoders.preprocessing.mean = mean(descrs, 2);
		encoders.preprocessing.std = std(descrs,0,2);
		descrs = (descrs - encoders.preprocessing.mean) ./ encoders.preprocessing.std;
        if pca_enabled
            [coeff,~,latent] = pca(descrs');
            expl_vars = cumsum(latent) ./ sum(latent);
            n_dims = find(expl_vars >= .975, 1)
            coeff = coeff(:, 1:n_dims);
            encoders.pca.coeff = coeff;

            descrs = (descrs' * coeff)';
        end
		[encoders.encoder.means, encoders.encoder.covariances, encoders.encoder.priors] = vl_gmm(descrs, encoders.encoder.numWords, 'Initialization', 'kmeans', 'CovarianceBound', double(max(var(descrs)')*0.0001), 'NumRepetitions', 3);
		encoders.encoder.means = single(encoders.encoder.means);
		encoders.encoder.covariances = single(encoders.encoder.covariances);
		encoders.encoder.priors = single(encoders.encoder.priors);
        save(encoder_file, 'encoders','-v7.3')
        clear descrs_ifv;
    end
end
end