%% SmoothnessNet architecture for mf-CNNCRF
%% Layers
tempLayers = [
    imageInputLayer([224 224 2],"Name","imageinput","Normalization","none")
    convolution2dLayer([3 3],16,"Name","V_conv1","Padding","same")
    tanhLayer("Name","tanh_1")
    convolution2dLayer([3 3],16,"Name","V_conv2","Padding","same")
    tanhLayer("Name","tanh_2")
    convolution2dLayer([3 3],16,"Name","V_conv3","Padding","same")
    tanhLayer("Name","tanh_3")
    convolution2dLayer([3 3],16,"Name","V_conv4","Padding","same")
    tanhLayer("Name","tanh_4")
    convolution2dLayer([3 3],4,"Name","V_out","Padding","same")];
lgraph=layerGraph(tempLayers); clear tempLayers;
%% Save
save('./SmoothnessNet_graph.mat','lgraph')