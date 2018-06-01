function feat_vect = compute_img_features_fn(img_path,imsize,  net, encoder)
    
    img = imread(img_path);
    sc_x = size(img, 2) / imsize(2);
    sc_y = size(img, 1) / imsize(1);
    img = imresize(img, imsize);

    showFigures = 0;
    if showFigures
        figure
        imshow(img)
        hold on
    end


    ymin = 1;
    ymax = imsize(1);
    xmin = 1;
    xmax =  imsize(2);
    regions = create_region_basis_fn(ymin, ymax, xmin, xmax, img, imsize);
    [dcnn_feats, dcnn_locs] = get_dcnn_features(net, img, regions, 'encoder', encoder);
    feat_vect = extract_feats_and_locs(dcnn_feats, dcnn_locs, regions.overlap_ratio);
    if showFigures
        for k=1:size(feat_vect.locs, 2)
            plot(feat_vect.locs(1, k),feat_vect.locs(2, k),'c*');
        end
        plot(x,y,'co', 'MarkerSize', 12);
    end




    feat_vect = [feat_vect];

end

function regions = create_region_basis_fn(ymin, ymax, xmin, xmax, img, imsize)
    regions.basis = zeros(size(img,1), size(img, 2));
    regions.basis(floor(ymin):floor(ymax), floor(xmin):floor(xmax)) = 1;
    regions.overlap_ratio = 0;
    regions.labels = {1};
end

function feat_vect = extract_feats_and_locs(dcnn_feats, dcnn_locs, overlap_ratio)
    feat_vect.feats = dcnn_feats{1};
    if iscell(feat_vect.feats)
        feat_vect.feats = feat_vect.feats{1};
    else
        feat_vect.feats = feat_vect.feats(:, 1)';
    end
    %feat_vect.feats = vl_colsubset(feat_vect.feats,50);
    feat_vect.locs = dcnn_locs{1}{1};
    feat_vect.overlap_ratio = overlap_ratio;
end