% assume all csv files are located within data at present path...

files = dir('data/*csv'); 
mkdir post_proc

% 59 columns per file 
h = waitbar(0,'...'); 
for i = 1 : length(files)
    d = parse_frame(fullfile('data',files(i).name),'-delimiter',',',...
        '-spec',repmat('%f',[1,59]));
    x = zeros(length(d.Time), 57); 
    xproc = zeros(size(x,1),57,5); 
    vn = fieldnames(d); 
    vn(1) = [];
    vn(end) = [];
    t = d.Time; 
    feedback = d.FeedBackEvent;
    xvals = cell(size(vn)); 
    
    parfor j = 1 : length(vn)
        [xvals{j}, xproc_vals{j}] = example_post_process(d.(vn{j})); 
    end
    for j = 1 : length(vn)
        x(:,j) = xvals{j}; 
        tmp = xproc_vals{j}; 
        for k = 1 : size(tmp,2)
            xproc(:,j,k) = tmp(:,k); 
        end
    end
    [~,fn] = fileparts(files(i).name); 
    save(fullfile('post_proc',fn), ...
        'xproc','feedback','vn','t','x'); 
    waitbar(i/length(files),h); 
end
close(h); 

%% now load back in bandpass filtered or single channels

f = dir('post_proc/*mat'); 

X = []; h = waitbar(0,'...'); w = 250; 
sid = cell(size(f)); label = [];
session = [];
subj = [];
for i = 1 : length(f)
    
    load(fullfile('post_proc',f(i).name),...
        't','feedback','vn','x'); % load xproc if desired
    
    num_cases = sum(feedback);
    idx = find(f(i).name=='_'); 
    for j = 1 : num_cases
        tmp = num2str(j); 
        while length(tmp) < 3
            tmp = ['0',tmp]; 
        end
        label = [label ; {[pullname(f(i).name(6:end)), '_FB',tmp]}];
        subj = [subj ; {f(i).name(idx(1)+1:idx(2)-1)}]; 
        session = [session ; {pullname(f(i).name(idx(2)+1:end))}]; 
    end
    sid{i} = pullname(f(i).name); 
    
    idx = find(feedback); 
    for j = 1 : length(idx)
        X = [X ; [t(idx(j)), x(idx(j):idx(j)+w,strcmp('Cz',vn))',...
            j]];
    end
    waitbar(i/length(f),h); 
end
close(h); 

labels = parse_frame('train_labels.csv','-delimiter',',','-spec','%s%f');

y = labels.Prediction;


%% now fit the model
[~,i,j] = intersect(labels.IdFeedBack,label);

xnew = [X, grp2idx(subj),grp2idx(session)];
% LB AUC=0.7
ens = fitensemble(xnew(j,:),y(i),'adaboostm1',500,'tree',...
    'prior','uniform','type','classification','learnrate',.05,'holdout',.3);

