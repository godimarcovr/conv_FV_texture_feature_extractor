%% init net
clear all
% feature settings
deep_fbanks_dir = '../deep-fbanks/';
addpath(deep_fbanks_dir)
setup; %vlfeat and matconvnet
path_model_vgg_m = [deep_fbanks_dir 'data/models/imagenet-vgg-m.mat'];
% path_model_vgg_m = [deep_fbanks_dir 'data/models/imagenet-vgg-verydeep-19.mat']; %fix average image! fix offset!

dcnn.opts.model = path_model_vgg_m;
dcnn.opts.layer = 13;
dcnn.opts.encoderType = 'fv';

opts.useGpu = true;
opts.gpuId = 1;

if opts.useGpu
  gpuDevice(opts.gpuId) ;
end

net = load(dcnn.opts.model) ;
net.layers = net.layers(1:dcnn.opts.layer) ;
if opts.useGpu
  net = vl_simplenn_move(net, 'gpu') ;
  net.useGpu = true ;
else
  net = vl_simplenn_move(net, 'cpu') ;
  net.useGpu = false ;
end

%% import data

if exist('dataset.mat', 'file')
    load('dataset.mat')
else
	dataset.root = [ '/media/vips/data/cjoppi/datasets_texture/dtd/']; %folder continaing a folder for each set continaing a folder for each class containing images of that class
	dataset.sets = {'images'};
	import_dataset_fn(dataset)
	load('dataset.mat')
end

%% compute conv features
features_folder = '/media/vips/data/mgodi/tmp_texture/'; %where to save output features
features_file = fullfile(features_folder, 'conv_features_all.mat');
features_file_cp = fullfile(features_folder, 'conv_features_all_cp.mat');
features = compute_conv_features_fn(features_file,features_file_cp, dataset, net);

%% load encoders
load("std_ifv_encoder_dtd.mat") %loads "encoders" variable

%% build feat vectors
sets = {'images'};
encoded_features_file = fullfile(features_folder, 'pca_ifv_features_train.mat');
[enc_features, labels] = compute_encoded_features_fn(encoded_features_file, dataset, sets, features, encoders, pca_enabled)

%% compute features ved47
load("ved47_svm_parameters.mat")

for c=1:47
   pred = w{c}'*enc_features' + b{c} ;
   scores{c} = pred ;
end

enc_features_ved47 = cat(1, scores{:}) ;

