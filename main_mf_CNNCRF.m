%% mf-CNNCRF
% Code for mf-CNNCRF

cutoff=.01; % for the negative log-likelihoods
%% Load Unet

load('./Unet.mat','dlnet')
Unet=dlnet; clear dlnet;
CombName='mfCNNCRF';
%% Load Vnet

load('./Vnet.mat','dlnet');
Vnet=dlnet; clear dlnet;
%% Load Image set

x1=imread('imgA.jpg'); x1g=rgb2gray(x1); x1g=double(x1g);
x2=imread('imgB.jpg'); x2g=rgb2gray(x2); x2g=double(x2g);
e=cat(4,x1,x2);
 
Xg=cat(3,x1g,x2g); clear x1 x2 x1g x2g
X=Xg/255; clear Xg
%% Estimation of Unary term: U

U=get_Unary(X,Unet,cutoff);
%% Estimation of Smoothness Probabilities in 4 directions: PH, PV, PD1, PD2

[PH,PV,PD1,PD2]=get_HVD1D2(X,Vnet);
%% Application of Conditional Random Field

DecisionMap=CRFapp(U,PH,PV,PD1,PD2,cutoff);
%% Fused Image

Fused=Fused_Given_Labels(e,DecisionMap);

return
%%
function p=apply_cutoff(p,cutoff)
% Used for the negative log-likelihood
    p(p<cutoff)=cutoff;
    p(p>1-cutoff)=1-cutoff;
end


function [U]=get_Unary(x,Unet,cutoff)
    % x: input grayscale images
    % Unet: the Unary Network
    % cutoff: used for negative log-likelihood
    % U: Unary term
    
    dX1=x(:,:,1); dX2=x(:,:,2); 
    dX1=dlarray(dX1,'SSCB'); dX1=gpuArray(dX1);
    dX2=dlarray(dX2,'SSCB'); dX2=gpuArray(dX2);
    
    % forward of Unet for each branch
    W1 = forward(Unet,dX1); % [224,224,1,M]
    W2 = forward(Unet,dX2); % [224,224,1,M]
    
    I1=W1(:,:,1,:);
    I2=W2(:,:,1,:); 
    PL=cat(3,I1,I2);
    P=softmax(PL);
    p=extractdata(gather(P));
    
    % p(p<.01)=.01; p(p>.99)=.99;
    P_unary=apply_cutoff(p,cutoff);
    
    U=-log(P_unary);
end



function [PH,PV,PD1,PD2]=get_HVD1D2(x,Vnet)
% Get probabilities in all 4 directions of N8-grid
    x=single(x);
    
    X0=x(:,:,1); X1=x(:,:,2); 
    % All input image combinations
    dX00=dlarray(cat(3,X0,X0),'SSCB'); dX00=gpuArray(dX00);
    dX01=dlarray(cat(3,X0,X1),'SSCB'); dX01=gpuArray(dX01);
    dX10=dlarray(cat(3,X1,X0),'SSCB'); dX10=gpuArray(dX10);
    dX11=dlarray(cat(3,X1,X1),'SSCB'); dX11=gpuArray(dX11);
    
    % Network outputs
    d00=forward(Vnet,dX00); clear dX00
    d01=forward(Vnet,dX01); clear dX01
    d10=forward(Vnet,dX10); clear dX10
    d11=forward(Vnet,dX11); clear dX11
    
    h =softmax(cat(3,d00(:,:,1,:),d01(:,:,1,:),d10(:,:,1,:),d11(:,:,1,:)));
    v =softmax(cat(3,d00(:,:,2,:),d01(:,:,2,:),d10(:,:,2,:),d11(:,:,2,:)));
    d1=softmax(cat(3,d00(:,:,3,:),d01(:,:,3,:),d10(:,:,3,:),d11(:,:,3,:)));
    d2=softmax(cat(3,d00(:,:,4,:),d01(:,:,4,:),d10(:,:,4,:),d11(:,:,4,:)));
    
    h=gather(extractdata(h));
    v=gather(extractdata(v));
    d1=gather(extractdata(d1));
    d2=gather(extractdata(d2));
        
    Ph0011=h(:,:,1)+h(:,:,4); % P_h(lpq=00)+P_h(lpq=11)
    Ph0110=h(:,:,2)+h(:,:,3); % P_h(lpq=01)+P_h(lpq=10)
    
    Pv0011=v(:,:,1)+v(:,:,4); % P_v(lpq=00)+P_v(lpq=11)
    Pv0110=v(:,:,2)+v(:,:,3); % P_v(lpq=01)+P_v(lpq=10)
    
    Pd10011=d1(:,:,1)+d1(:,:,4); % P_d1(lpq=00)+P_d1(lpq=11)
    Pd10110=d1(:,:,2)+d1(:,:,3); % P_d1(lpq=01)+P_d2(lpq=10)
    
    Pd20011=d2(:,:,1)+d2(:,:,4); % P_d2(lpq=00)+P_d2(lpq=11)
    Pd20110=d2(:,:,2)+d2(:,:,3); % P_d2(lpq=01)+P_d2(lpq=10)
    
    % Concatenation of probabilities in all 4 directions of N8-grid
    PH=cat(3,Ph0011,Ph0110);
    PV=cat(3,Pv0011,Pv0110);
    PD1=cat(3,Pd10011,Pd10110);
    PD2=cat(3,Pd20011,Pd20110);

end

function L=CRFapp(u,PH,PV,PD1,PD2,cutoff)  
% Application of CRF model
% L: decision map

    ph=PH;   ph=apply_cutoff(ph,cutoff);
    pv=PV;   pv=apply_cutoff(pv,cutoff);
    pd1=PD1; pd1=apply_cutoff(pd1,cutoff);
    pd2=PD2; pd2=apply_cutoff(pd2,cutoff);
        
    u1=u(:,:,1);
    u2=u(:,:,2);
    U=[u1(:), u2(:)]';
    [N,M,Ne]=size(u);
    
    % Probabilities in all 4 directions of N8 grid
    ph0011=ph(:,:,1); ph0110=ph(:,:,2);     % Horizontal: Ph(lp=lq),   Ph(lp!=lq)
    pv0011=pv(:,:,1); pv0110=pv(:,:,2);     % Vertical:   Pv(lp=lq),   Pv(lp!=lq)
    pd10011=pd1(:,:,1); pd10110=pd1(:,:,2); % Diagonal:   Pd1(lp=lq), Pd1(lp!=lq)
    pd20011=pd2(:,:,1); pd20110=pd2(:,:,2); % Diagonal:   Pd2(lp=lq), Pd2(lp!=lq)
    
    % Negative log-likelihood of probabilities
    h=-log(ph0110); v=-log(pv0110); d1=-log(pd10110); d2=-log(pd20110);
    
    [ii, jj] = sparse_adj_matrix([N, M], 1, inf); % p = ii , q = jj;
    vv=zeros(size(ii));
    
    [i1,j1]=ind2sub([N,M],ii); % [p1, q1];
    [i2,j2]=ind2sub([N,M],jj); % [p2, q2];
    
    % N8-Grid Construction:
    
    % Horizontal
    Hidx1=i2==i1 & j2==j1+1;
    Hidx2=i2==i1 & j2==j1-1;
    
    vv(Hidx1)=h(ii(Hidx1)); % [i,j]=>[i,j+1]
    vv(Hidx2)=h(jj(Hidx2));
    
    % Vertical
    Vidx1=i2==i1+1 & j2==j1;
    Vidx2=i2==i1-1 & j2==j1;
    vv(Vidx1)=v(ii(Vidx1)); % [i,j]=>[i,j+1]
    vv(Vidx2)=v(jj(Vidx2));
    
    % Diagonal - 1
    D1idx1=i2==i1-1 & j2==j1+1; % [i-1,j]=>[i,j+1]
    vv(D1idx1)=d1(ii(D1idx1))*(sqrt(2)^-1);
    D1idx2=i2==i1+1 & j2==j1-1;
    vv(D1idx2)=d1(jj(D1idx2))*(sqrt(2)^-1);
    
    % Diagonal - 2
    D2idx1=i2==i1+1 & j2==j1+1;
    vv(D2idx1)=d2(ii(D2idx1))*(sqrt(2)^-1);
    D2idx2=i2==i1-1 & j2==j1-1;
    vv(D2idx2)=d2(jj(D2idx2))*(sqrt(2)^-1);
    
    SparseSmoothness = sparse(ii, jj, vv, N*M, N*M);
        
    Sc=ones(Ne)-eye(Ne);
    
    % GraphCut
    iters=1;
    gch = GraphCut('open', U, Sc, SparseSmoothness); 
    [gch, L0] = GraphCut('expand', gch, iters);  
    gch = GraphCut('close', gch);  
    
    % Reshape Decision Map L and convert to binary Map
    LL=reshape(L0,N,M);
    L=logical(LL);
end

function [ Fused ] = Fused_Given_Labels(e, T)
    % e: input images
    % T: decision map
    ed=double(e); % convert images to double
    T=logical(T); % convert map to binary
    
    if size(ed,4)~=1 % rgb input images
        Fused=(~T).*ed(:,:,:,1)+(T).*ed(:,:,:,2);
    else % grayscale input images
         Fused=(~T).*ed(:,:,1)+(T).*ed(:,:,2);
    end

    if isinteger(e)
        Fused=uint8(Fused);
    end
end