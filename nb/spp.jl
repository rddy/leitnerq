#=
Module for solving the Leitner Queue Network static planning problem
@author Siddharth Reddy <sgr45@cornell.edu>
=#

using JuMP
using Ipopt

eps = 1e-9

function allocate_work_rates(num_decks, work_rate_budget, difficulty)
    m = Model(with_optimizer(Ipopt.Optimizer)) # Uses IP-OPT solver by default
    @variable(m, u[1:num_decks] >= 0)
    @variable(m, a >= 0)
    @variable(m, l[1:num_decks] >= 0)
    @constraint(m, a + sum(u) <= work_rate_budget)
    for i = 1:num_decks
        @constraint(m, l[i] <= u[i])
    end
    @NLconstraint(m, l[1] == a + (1 - (u[1] - l[1]) / (u[1] - l[1] + difficulty)) * l[1] + 
            (1 - (u[2] - l[2]) / (u[2] - l[2] + difficulty / 2)) * l[2])
    for i = 2:num_decks-1
        @NLconstraint(m, l[i] == (u[i-1] - l[i-1]) / (u[i-1] - l[i-1] + difficulty / (i-1)) * 
                l[i-1] + (1 - (u[i+1] - l[i+1]) / (u[i+1] - l[i+1] + difficulty / (i+1))) * l[i+1])
    end
    @NLconstraint(m, l[num_decks] == (u[num_decks-1] - l[num_decks-1]) /
            (u[num_decks-1] - l[num_decks-1] + difficulty / (num_decks-1)) * l[num_decks-1])
    @NLobjective(m, Max, a)
    TT = stdout
    redirect_stdout()
    optimize!(m)
    status = termination_status(m)
    redirect_stdout(TT)
    return objective_value(m), value.(u), value.(l)
end

# Assuming fixed work rates, what is the corresponding equilibrium?
function throughput_for_work_rates(num_decks, u, difficulty)
    m = Model(with_optimizer(Ipopt.Optimizer))
    @variable(m, a >= 0)
    @variable(m, l[1:num_decks] >= 0)
    for i = 1:num_decks
        @constraint(m, l[i] <= u[i])
    end
    @NLconstraint(m, l[1] == a + (1 - (u[1] - l[1]) / (u[1] - l[1] + difficulty)) * l[1] + 
            (1 - (u[2] - l[2]) / (u[2] - l[2] + difficulty / 2)) * l[2])
    for i = 2:num_decks-1
        @NLconstraint(m, l[i] == (u[i-1] - l[i-1]) / (u[i-1] - l[i-1] + difficulty / (i-1)) * 
                l[i-1] + (1 - (u[i+1] - l[i+1]) / (u[i+1] - l[i+1] + difficulty / (i+1))) * l[i+1])
    end
    @NLconstraint(m, l[num_decks] == (u[num_decks-1] - l[num_decks-1]) /
            (u[num_decks-1] - l[num_decks-1] + difficulty / (num_decks-1)) * l[num_decks-1])
    @NLobjective(m, Max, a)
    TT = stdout
    redirect_stdout()
    optimize!(m)
    status = termination_status(m)
    redirect_stdout(TT)
    return objective_value(m)
end

# Assuming fixed work rates and arrival rate, what is the corresponding equilibrium?
function eq_flow_rates_for_work_rates_and_arrival_rate(num_decks, u, difficulty, a)
    m = Model(with_optimizer(Ipopt.Optimizer))
    @variable(m, l[1:num_decks] >= 0)
    for i = 1:num_decks
        @constraint(m, l[i] <= u[i])
    end
    @NLconstraint(m, l[1] == a + (1 - (u[1] - l[1]) / (u[1] - l[1] + difficulty)) * l[1] + 
            (1 - (u[2] - l[2]) / (u[2] - l[2] + difficulty / 2)) * l[2])
    for i = 2:num_decks-1
        @NLconstraint(m, l[i] == (u[i-1] - l[i-1]) / (u[i-1] - l[i-1] + difficulty / (i-1)) * 
                l[i-1] + (1 - (u[i+1] - l[i+1]) / (u[i+1] - l[i+1] + difficulty / (i+1))) * l[i+1])
    end
    @NLconstraint(m, l[num_decks] == (u[num_decks-1] - l[num_decks-1]) /
            (u[num_decks-1] - l[num_decks-1] + difficulty / (num_decks-1)) * l[num_decks-1])
    @NLobjective(m, Max, l[num_decks])
    TT = stdout
    redirect_stdout()
    optimize!(m)
    status = termination_status(m)
    redirect_stdout(TT)
    return objective_value(m), value.(l)
end

# Use a heuristic to generate a work rate allocation that satisfies a budget
function work_rates_of_heuristic(num_decks, work_rate_budget, f)
    work_rates = map(f, 1:num_decks)
    return work_rates / sum(work_rates) * work_rate_budget
end

