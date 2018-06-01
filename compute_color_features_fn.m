function feat_vect = compute_color_features_fn(img_path, w2c, parsing_path)
feat_vect = cell(size(pose_info.coords, 2), 1);

if nargin < 3
    parsing_path = empty;
end

img = imread(img_path);
imsize = size(img);
imsize = imsize(1:2);

showFigures = 0;
if showFigures
    figure
    imshow(img)
    hold on
end

[parsingfilter, overlap_ratio] = create_parsingfilter_fn(img, parsing_path, imsize);
tmp_patch = double(img);
if numel(size(tmp_patch)) == 2
    tmp_patch = cat(3, tmp_patch, tmp_patch, tmp_patch);
end
tmp_colors =im2c(tmp_patch, w2c, 0);
feat_vect{j} = extract_feats_and_locs(tmp_colors, parsingfilter, overlap_ratio);


element_count = 0;

for j=1:size(pose_info.coords, 2)
    element_count = element_count + 1;
    x = floor(pose_info.coords(1, j) / sc_x);
    y = floor(pose_info.coords(2, j) / sc_y);
    ymin = max(y-jpatch_size/2, 1);
    ymax = min(y+jpatch_size/2, imsize(1));
    xmin = max(x-jpatch_size/2, 1);
    xmax = min(x+jpatch_size/2, imsize(2));
    [parsingfilter, overlap_ratio] = create_parsingfilter_fn(ymin, ymax, xmin, xmax, img, parsing_path, imsize);
    tmp_patch = double(img(floor(ymin):floor(ymax), floor(xmin):floor(xmax), :));
    if numel(size(tmp_patch)) == 2
        tmp_patch = cat(3, tmp_patch, tmp_patch, tmp_patch);
    end
    tmp_colors =im2c(tmp_patch, w2c, 0);
    feat_vect{j} = extract_feats_and_locs(tmp_colors, [xmin, xmax, ymin, ymax], element_count, parsingfilter, overlap_ratio);
end

feat_vect_segm = cell(size(pose_info.segments, 2), 1);
for k=1:size(pose_info.segments, 2)
    element_count = element_count + 1;
    xs = [pose_info.segments(1,k), pose_info.segments(3,k)] ./ sc_x;
    ys = [pose_info.segments(2,k), pose_info.segments(4,k)] ./ sc_y;
    
    ymin = max(min(ys), 1);
    ymax = min(max(ys), imsize(1));
    xmin = max(min(xs), 1);
    xmax = min(max(xs), imsize(2));
    if xmax - xmin < jpatch_size
        xmin = max(xmin - jpatch_size/2, 1);
        xmax = min(xmax + jpatch_size/2, imsize(2));
    end
    if ymax - ymin < jpatch_size
        ymin = max(ymin - jpatch_size/2, 1);
        ymax = min(ymax + jpatch_size/2, imsize(1));
    end
    
    [parsingfilter, overlap_ratio] = create_parsingfilter_fn(ymin, ymax, xmin, xmax, img, parsing_path, imsize);
    tmp_patch = double(img(floor(ymin):floor(ymax), floor(xmin):floor(xmax), :));
    if numel(size(tmp_patch)) == 2
        tmp_patch = cat(3, tmp_patch, tmp_patch, tmp_patch);
    end
    tmp_colors =im2c(tmp_patch, w2c, 0);
    feat_vect_segm{k} = extract_feats_and_locs(tmp_colors, [xmin, xmax, ymin, ymax], element_count, parsingfilter, overlap_ratio);
end

feat_vect = [feat_vect; feat_vect_segm];

end
function [parsingfilter, overlap_ratio] = create_parsingfilter_fn(ymin, ymax, xmin, xmax, img, parsing_path, imsize)
    parsingfilter = false(size(img,1), size(img, 2));
    parsingfilter(floor(ymin):floor(ymax), floor(xmin):floor(xmax)) = true;
    if exist(parsing_path, 'file')
        parsing_img = imread(parsing_path);
        parsing_img = imresize(parsing_img, imsize, 'nearest');
        parsing_img = parsing_img(:, :, 1) > 2; %0 BG 1 SKIN 2 HAIR
        overlap_ratio = sum(sum(double(parsingfilter & parsing_img))) / sum(sum(double(parsingfilter)));
        if any(any(parsingfilter & parsing_img))
            parsingfilter = parsingfilter & parsing_img;
        end
    else
        overlap_ratio = 0;
    end
    parsingfilter = parsingfilter(floor(ymin):floor(ymax), floor(xmin):floor(xmax));
end

function feat_vect = extract_feats_and_locs(feats, locs, element_count, parsingfilter, overlap_ratio)
    feats = feats(parsingfilter);
    unf = unique(feats);
    conteggi = histc(feats(:),unf);
    feat_vect.feats = zeros(1, 11);
    feat_vect.feats(unf) = conteggi;
    feat_vect.feats(unf) = feat_vect.feats(unf) ./ numel(feats);
    feat_vect.locs = locs;
    feat_vect.element = element_count;
    feat_vect.overlap_ratio = overlap_ratio;
    
    if sum(feat_vect.feats) - 1 < 0.01
        return
    else
        'Color sum != 1'
    end
end







