using PyCall
using Dates
using Random

include("optimisation_models.jl")
include("MCP_heuristics.jl")

function main_MIRSA(index_dict, deployment_dict, D, c, N, I, E, T_init, R, run)

  index_dict = Dict([(convert(Int64, k), convert(Int64, index_dict[k])) for k in keys(index_dict)])
  deployment_dict = Dict([(convert(Int64, k), convert(Int64, deployment_dict[k])) for k in keys(deployment_dict)])
  D  = convert.(Float64, D)

  c = convert(Float64, c)
  N = convert(Int64, N)
  I = convert(Int64, I)
  E = convert(Int64, E)
  T_init = convert(Float64, T_init)
  R = convert(Int64, R)
  run = string(run)

  W, L = size(D)

  P = maximum(values(index_dict))
  n_partitions = [deployment_dict[i] for i in 1:P]

  if run == "SALS"

    x_sol, LB_sol, obj_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R), Array{Float64, 2}(undef, R, I)
    x_init = solve_MILP_partitioning(D, c, n_partitions, index_dict, "Gurobi")
    
    for r = 1:R
      println("Run ", r, "/", R)
      x_sol[r, :], LB_sol[r], obj_sol[r, :] = simulated_annealing_local_search_partition(D, c, n_partitions, N, I, E, x_init, T_init, index_dict)
    end

  elseif run == "RSSA"

    x_sol, LB_sol, obj_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R), Array{Float64, 2}(undef, R, I)
    x_init = zeros(Int64, L)
    ind_set = [l for l in 1:L]
    n_while = n_partitions[1]

    while n_while > 0
      loc = sample(ind_set)
      deleteat!(ind_set, findall(x->x==loc, ind_set))
      x_init[loc] = 1
      n_while -= 1
    end

    x_init = convert.(Float64, x_init)

    for r = 1:R
      println("Run ", r, "/", R)
      x_sol[r, :], LB_sol[r], obj_sol[r, :] = simulated_annealing_local_search_partition(D, c, n_partitions, N, I, E, x_init, T_init, index_dict)
    end

  else
    println("No such run available.")
    throw(ArgumentError)
  end

  return x_sol, LB_sol, obj_sol

end

function main_GRED(deployment_dict, D, c, R, p, run)

  deployment_dict = Dict([(convert(Int64, k), convert(Int64, deployment_dict[k])) for k in keys(deployment_dict)])
  D  = convert.(Float64, D)
  c = convert(Float64, c)
  R = convert(Int64, R)

  W, L = size(D)

  if run == "TGH"
    x_sol, LB_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R)
    n = convert(Float64, deployment_dict[1])
    for r = 1:R
      println("Run ", r, "/", R)
      x_sol[r, :], LB_sol[r] = threshold_greedy_algorithm(D, c, n)
    end
  elseif run == "STGH"
    x_sol, LB_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R)
    n = convert(Float64, deployment_dict[1])
    p = convert(Float64, p)
    for r = 1:R
      println("Run ", r, "/", R)
      x_sol[r, :], LB_sol[r] = stochastic_threshold_greedy_algorithm(D, c, n, p)
    end
  else
    println("No such run available.")
    throw(ArgumentError)
  end

  return x_sol, LB_sol

end

function main_RAND(deployment_dict, D, c, I, R, run)

  deployment_dict = Dict([(convert(Int64, k), convert(Int64, deployment_dict[k])) for k in keys(deployment_dict)])
  D  = convert.(Float64, D)
  c = convert(Float64, c)
  I = convert(Int64, I)
  R = convert(Int64, R)

  W, L = size(D)

  if run == "RS"
    x_sol, LB_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R)
    n = convert(Float64, deployment_dict[1])
    for r = 1:R
      println("Run ", r, "/", R)
      x_sol[r, :], LB_sol[r] = random_search(D, c, n, I)
    end

  else
    println("No such run available.")
    throw(ArgumentError)
  end

  return x_sol, LB_sol

end

# Local Search algorithm
function main_LSEA(index_dict, deployment_dict, D, c, N, I, E, run)

  index_dict = Dict([(convert(Int64, k), convert(Int64, index_dict[k])) for k in keys(index_dict)])
  deployment_dict = Dict([(convert(Int64, k), convert(Int64, deployment_dict[k])) for k in keys(deployment_dict)])
  D  = convert.(Float64, D)

  c = convert(Float64, c)
  N = convert(Int64, N)
  I = convert(Int64, I)
  E = convert(Int64, E)
  T_init = convert(Float64, T_init)
  R = convert(Int64, R)
  run = string(run)

  W, L = size(D)

  P = maximum(values(index_dict))
  n_partitions = [deployment_dict[i] for i in 1:P]

  if run == "GLS"
    x_sol, LB_sol, obj_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R), Array{Float64, 2}(undef, R, I)
    x_init = solve_MILP_partitioning(D, c, n_partitions, index_dict, "Gurobi")
    for r = 1:R
      println("Run ", r, "/", R)
      x_sol[r, :], LB_sol[r], obj_sol[r, :] = greedy_local_search_partition(D, c, n_partitions, N, I, E, x_init, index_dict)
    end
  else
    println("No such run available.")
    throw(ArgumentError)
  end

  return x_sol, LB_sol, obj_sol

end