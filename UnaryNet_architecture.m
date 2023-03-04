%% UnaryNet architecture for mf-CNNCRF
% Notes
%% Create the Layer Graph

clear all; close all; clc;
lgraph = layerGraph();
%% Layer Branches
tempLayers = imageInputLayer([224 224 1],"Name","imageinput","Normalization","none");
lgraph = addLayers(lgraph,tempLayers);

%% Sobel_y_branch
tempLayers = [
    SobelVLayer('Sobel_y')
    convolution2dLayer([3 3],16,"Name","conv_V1","Padding","same")
    reluLayer("Name","relu_V1")
    convolution2dLayer([3 3],16,"Name","conv_V2","Padding","same")
    reluLayer("Name","relu_V2")
    convolution2dLayer([3 3],16,"Name","conv_V3","Padding","same")
    reluLayer("Name","relu_V3")
    convolution2dLayer([3 3],1,"Name","conv_V","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

%% Sobel_x_branch
tempLayers = [
    SobelHLayer('Sobel_x')
    convolution2dLayer([3 3],16,"Name","conv_H1","Padding","same")
    reluLayer("Name","relu_H1")
    convolution2dLayer([3 3],16,"Name","conv_H2","Padding","same")
    reluLayer("Name","relu_H2")
    convolution2dLayer([3 3],16,"Name","conv_H3","Padding","same")
    reluLayer("Name","relu_H3")
    convolution2dLayer([3 3],1,"Name","conv_H","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

%% Intensity_branch
tempLayers = [
    convolution2dLayer([3 3],16,"Name","conv_I1","Padding","same")
    reluLayer("Name","relu_I1")
    convolution2dLayer([3 3],16,"Name","conv_I2","Padding","same")
    reluLayer("Name","relu_I2")
    convolution2dLayer([3 3],16,"Name","conv_I3","Padding","same")
    reluLayer("Name","relu_I3")
    convolution2dLayer([3 3],1,"Name","conv_I","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);
%% Depth_Concatenation and Output

tempLayers = [
    depthConcatenationLayer(3,"Name","Concatenation")
    convolution2dLayer([3 3],1,"Name","conv_out","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);
%% Layer Connections

lgraph = connectLayers(lgraph,"imageinput","Sobel_y");
lgraph = connectLayers(lgraph,"imageinput","Sobel_x");
lgraph = connectLayers(lgraph,"imageinput","conv_I1");
lgraph = connectLayers(lgraph,"conv_H","Concatenation/in2");
lgraph = connectLayers(lgraph,"conv_V","Concatenation/in3");
lgraph = connectLayers(lgraph,"conv_I","Concatenation/in1");
%%
clear tempLayers;
%% Plot the Layers
plot(lgraph);
%% Save Architecture

save('./UnaryNet_graph.mat','lgraph')
%% Sobel_x

function graphSobelHLayer=SobelHLayer(name)
    Hkern=[
    1 0 -1;
    2 0 -2;
    1 0 -1];

    FilterSize=[3,3];
    NumFiltersPerGroup=1; NumGroups=1;   
    W(:,:,1,1,1)=Hkern;
    b=zeros(1,1,NumFiltersPerGroup,NumGroups);
    
    graphSobelHLayer=groupedConvolution2dLayer(FilterSize,NumFiltersPerGroup,"channel-wise",'Weights',W,"Bias",b,'WeightLearnRateFactor',0,"BiasLearnRateFactor",0,"Name",name,'Padding','same','WeightsInitializer',"zeros");
end
%% Sobel_y

function graphSobelVLayer=SobelVLayer(name)
   Vkern= [
           1 2 1; 
           0 0 0; 
          -1 -2 -1];

    FilterSize=[3,3];
    NumFiltersPerGroup=1; NumGroups=1;   
    W(:,:,1,1,1)=Vkern;
    b=zeros(1,1,NumFiltersPerGroup,NumGroups);
    
    graphSobelVLayer=groupedConvolution2dLayer(FilterSize,NumFiltersPerGroup,"channel-wise",'Weights',W,"Bias",b,'WeightLearnRateFactor',0,"BiasLearnRateFactor",0,"Name",name,'Padding','same','WeightsInitializer',"zeros");
end