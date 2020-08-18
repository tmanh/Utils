function showSkeleton()
    %% Data path.
    data = '/home/anhtruong/Downloads/Dataset/Simple/CAD-120/Skeleton/Subject1_skeleton/stacking_objects/1204145234.txt';
    
    %% Set up the figure.
    fid = figure;
    hold on
    view(0,90)
    
    %% Set up the movie.
    writerObj = VideoWriter('out.avi'); % Name it.
    writerObj.FrameRate = 30; % How many frames per second.
    
    %% Record the movie.
    open(writerObj);
    
    for i = 1:601
        visualizeSkeleton(data, i); % plotting function - replace your own
        frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
        writeVideo(writerObj, frame);
        pause(0.1);
        cla;
        pause(0.2);
    end

    hold off
    
    close(writerObj); % Saves the movie.
end

