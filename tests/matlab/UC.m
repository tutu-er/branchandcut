clear;
clc;
%%
mpc = case39_UC;
mpc=ext2int(mpc);
load load3996.mat;
time_mode;

T = size(Pd, 2);

%% Preparation
% define_constants;
[GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, ...
    MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PC1, PC2, QC1MIN, QC1MAX, ...
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF] = idx_gen;
% [PQ, PV, REF, NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, ...
%     VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN] = idx_bus;
% [F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, ...
%     TAP, SHIFT, BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST, ...
%     ANGMIN, ANGMAX, MU_ANGMIN, MU_ANGMAX] = idx_brch;

[ref, pv, pq] = bustypes(mpc.bus, mpc.gen);

baseMVA = mpc.baseMVA;
bus = mpc.bus;
gen = mpc.gen;
branch = mpc.branch;
gencost = mpc.gencost;

nb = size(bus, 1);
nl = size(branch, 1);
ng = size(gen, 1);

%% Transition Matrix
G = zeros(nb, ng);
for k = 1:ng
    G(mpc.gen(k, GEN_BUS), k) = 1;
end

%% obj
obj = 0;

%% Variables
pg = sdpvar(ng, T, 'full');
cpower = sdpvar(ng, T, 'full');

x = binvar(ng, T, 'full');
coc = sdpvar(ng, T-1, 'full');

%% Constraints
cons = [];

%% Active Power Balance Constraints
cons = [cons, sum(pg, 1) == sum(Pd, 1)];

%% Thermal Upper and Lower Bound Constraints
cons = [cons, repmat(gen(:, PMIN), 1, T) .* x <= pg <= repmat(gen(:, PMAX), 1, T) .* x];

%% New Energy Upper and Lower Bound Constraints
% cons = [cons, 0 <= pgnew, pgnew <= pgnew_max];

%% Up-Down Ramp rate Constraints
Ru = 0.4 * gen(:, PMAX)/T_delta;
Rd = 0.4 * gen(:, PMAX)/T_delta;

Ru = repmat( Ru , 1, T-1);
Rd = repmat( Rd , 1, T-1);

Ru_co = repmat( 0.3 * gen(:, PMAX) , 1, T-1);
Rd_co = repmat( 0.3 * gen(:, PMAX) , 1, T-1);

ramp_mat = diag(ones(T,1)) + diag(-ones(T-1,1), -1);
% ramp_mat(1, T) = -1; %% 是否认为循环
cons = [cons, -Ru_co - (Ru - Ru_co) .* x(:, 1:end-1) <= pg * ramp_mat(:, 1:T-1) ...
                <= Rd_co + (Rd - Rd_co) .* x(:, 2:end)];

%% Unit commitment costs Constraints
start_cost = gencost(:, 2);
shut_cost = gencost(:, 3);

cons = [cons, coc >= - repmat(start_cost, 1, T-1) .* (x * ramp_mat(:, 1:T-1) )];
cons = [cons, coc >= + repmat(shut_cost, 1, T-1) .* (x * ramp_mat(:, 1:T-1) )];
cons = [cons, coc >= 0];

%% Minium duration parameters
Ton = 4 * 4;
Toff = 4 * 4;

%% Minium duration Constrains - for type edtion
for t = 1:min(Ton, T-1)
    cons = [cons, -x * ramp_mat(:, 1:T-t) <= x(:, 1+t:end)]; % t列：x(:,t+1)-x(:,t)
end
for t = 1:min(Toff, T-1)
    cons = [cons, x * ramp_mat(:, 1:T-t) <= 1-x(:, 1+t:end)]; % t列：x(:,t+1)-x(:,t)
end


%% Operation costs Constraints
cons = [cons, cpower >= repmat(gencost(:, end-1)/T_delta, 1, T) .* pg + repmat(gencost(:, end)/T_delta, 1, T) .* x];

%% DCPF Constraints
H = makePTDF(mpc);
cons = [cons, repmat(-branch(:, 6), 1, T) <= H * (G*pg - Pd) <= repmat(branch(:, 6), 1, T)];

%% Objective function
obj = obj +sum(cpower, "all");
obj = obj +sum(coc, 'all');

%% Solve
ops = sdpsettings('solver', 'gurobi', 'verbose', 0);
sol = optimize(cons, obj, ops);

%% Results
pg = value(pg);
cpower = value(cpower);
obj = value(obj);
coc = sum(value(coc), 'all');
total_cost = obj;
x = value(x);

%% disp
fprintf('总运行成本%d美元\n', obj);
fprintf('机组启停成本%d美元\n', coc);

fprintf('yalmip建模时间%.2fs\n', sol.yalmiptime);
fprintf('gurobi求解时间%.2fs\n', sol.solvertime);

if 0
%% plot line
time = minutes(0:15:15*(T-1));

figure('Position',[100, 100, 1200, 700]);

on_pg = find(sum(x,2) > 0);

plot(time ,pg(on_pg, :)','LineWidth',1.2);
title('机组出力折线图')
legend_strs = {};
for i = 1:length(on_pg)
    legend_strs = [legend_strs; sprintf('机组%d', on_pg(i))];
end
legend(legend_strs)


%% plot area
figure('Position',[100, 100, 1200, 700])
time = minutes(0:15:15*(T-1));
g_order = [7, 9, 2, 5, 3, 10]; 
[g_sorted_ordr,handel_order] = sort(g_order); 
handels = area(time, pg(g_order, :)');
colors = slanCM(179);
color_idx = floor(1:size(colors,1)/6:256);
colororder(colors(color_idx,:))

legend_strs = {};
for i = 1:length(g_order)
    legend_strs = [legend_strs; sprintf('机组%d', g_sorted_ordr(i))];
end
legend(handels(handel_order), legend_strs)

xlim([0, max(time)])
ylabel('P/MW')

title('机组出力堆叠图')


%% plot
% 绘制热力图
figure('Position', [100, 400, 1200, 300]);
h = heatmap(x);
h.Title = '机组启停状态';
h.XLabel = '时间 (小时)';
h.YLabel = '机组编号';
h.ColorbarVisible = 'off';

annotation('textbox', [0.75, 0.88, 0.1, 0.1], 'String', '蓝色=运行, 白色=停机', ...
    'FitBoxToText', 'on', 'BackgroundColor', 'w');

h.XDisplayLabels = repmat({''}, 1, T);
h.XDisplayLabels(8:8:T) = {'2','4','6','8','10','12','14','16','18','20','22','24'};

end
%% save
save UC_result.mat x


