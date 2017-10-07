function u = ChannelModel(seq, W, var)
    % seq : the input sequence
    % W   : distortion parameter
    % var : the variance of noise
    % u   : the output sequence
    h_tmp = RaisedCosine(1:3,W);
    h = [0,h_tmp,0];
    u_tmp = conv(h,seq);
    v = normrnd(0,sqrt(var),1,size(u_tmp,2));
    u = u_tmp + v;
    %u = u_tmp;
end