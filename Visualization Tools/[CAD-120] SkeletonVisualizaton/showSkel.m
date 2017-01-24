% need to create a figure first, for example:
% fid = figure;
% hold on
% showSkel(...)
% hold off
function showSkel(data, data_pos, figureTitle)
% actual visualization

lineLen=50;
linewidth=2;
for i=1:11
    xyz = data(i,[11,12,13]);
    x=xyz(1);
    y=xyz(2);
    z=xyz(3);
    plot3(x,y,z,'r.', 'MarkerSize',15);        
        
    vect = data(i,[1,4,7])*lineLen;
    vect = xyz+vect;
    xd = vect(1);
    yd = vect(2);
    zd = vect(3);
    line([x,xd],[y,yd],[z,zd],'Color','r', 'LineWidth',linewidth);
    
    vect = data(i,[2,5,8])*lineLen;
    vect = xyz+vect;
    xd = vect(1);
    yd = vect(2);
    zd = vect(3);
    line([x,xd],[y,yd],[z,zd],'Color','g', 'LineWidth',linewidth);
    
    vect = data(i,[3,6,9])*lineLen;
    vect = xyz+vect;
    xd = vect(1);
    yd = vect(2);
    zd = vect(3);
    line([x,xd],[y,yd],[z,zd],'Color','b', 'LineWidth',linewidth);    
end

for i=1:4
    xyz = data_pos(i,[1,2,3]);
    x=xyz(1);
    y=xyz(2);
    z=xyz(3);
    
    plot3(x,y,z,'B.','MarkerSize',15); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

line([data(1,11), data(2,11)], [data(1,12), data(2,12)], [data(1,13), data(2,13)], 'Color','g', 'LineWidth',1)
line([data(2,11), data(3,11)], [data(2,12), data(3,12)], [data(2,13), data(3,13)], 'Color','g', 'LineWidth',1)

line([data(2,11), data(4,11)], [data(2,12), data(4,12)], [data(2,13), data(4,13)], 'Color','g', 'LineWidth',1)
line([data(4,11), data(5,11)], [data(4,12), data(5,12)], [data(4,13), data(5,13)], 'Color','g', 'LineWidth',1)

line([data(2,11), data(6,11)], [data(2,12), data(6,12)], [data(2,13), data(6,13)], 'Color','g', 'LineWidth',1)
line([data(6,11), data(7,11)], [data(6,12), data(7,12)], [data(6,13), data(7,13)], 'Color','g', 'LineWidth',1)

line([data(3,11), data(8,11)], [data(3,12), data(8,12)], [data(3,13), data(8,13)], 'Color','g', 'LineWidth',1)
line([data(8,11), data(9,11)], [data(8,12), data(9,12)], [data(8,13), data(9,13)], 'Color','g', 'LineWidth',1)

line([data(3,11), data(10,11)], [data(3,12), data(10,12)], [data(3,13), data(10,13)], 'Color','g', 'LineWidth',1)
line([data(10,11), data(11,11)], [data(10,12), data(11,12)], [data(10,13), data(11,13)], 'Color','g', 'LineWidth',1)

line([data(5,11), data_pos(1,1)], [data(5,12), data_pos(1,2)], [data(5,13), data_pos(1,3)], 'Color','g', 'LineWidth',1)
line([data(7,11), data_pos(2,1)], [data(7,12), data_pos(2,2)], [data(5,13), data_pos(2,3)], 'Color','g', 'LineWidth',1)
line([data(9,11), data_pos(3,1)], [data(9,12), data_pos(3,2)], [data(9,13), data_pos(3,3)], 'Color','g', 'LineWidth',1)
line([data(11,11), data_pos(4,1)], [data(11,12), data_pos(4,2)], [data(11,13), data_pos(4,3)], 'Color','g', 'LineWidth',1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

axis square
xlabel('x')
ylabel('y')
zlabel('z')
title(figureTitle);
legend('joints','x','y','z');

fprintf('figure done..\n');

end

