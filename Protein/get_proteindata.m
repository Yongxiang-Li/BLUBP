function [design, trail] = get_proteindata(k, m)
%GET_ROADDATA 此处显示有关此函数的摘要
%   此处显示详细说明
    proteindata = load('protein.mat', 'data');
    Xd=proteindata.data(:,1:end-1);    Yd = proteindata.data(:,end);
    Xd = 10*(Xd - min(Xd))./(max(Xd) - min(Xd));
    level = struct('S', {[]}, 'Y', {[]});
    design = repmat(level, k, 1);
    I = reshape(randperm(k*m,k*m), m, k);
    for j = 1 : k
        design(j).S = Xd(I(:,j),:);
        design(j).Y = Yd(I(:,j),:);
    end
    index = true(size(Yd));    index(I(:)) = false;
    trail = struct('S', {[]}, 'Y', {[]});
    trail.S = Xd(index,:);    trail.Y = Yd(index);
end

