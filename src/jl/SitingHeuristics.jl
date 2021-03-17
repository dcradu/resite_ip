using PyCall
using Dates

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

    t1 = now()
    for r = 1:R
      println("Run ", r, "/", R)
      x_sol[r, :], LB_sol[r], obj_sol[r, :] = simulated_annealing_local_search_partition(D, c, n_partitions, N, I, E, x_init, T_init, index_dict)
    end
    t2 = now()
    dt = (t2 - t1)/R
    println(dt)

  elseif run == "SALSR"

    x_sol, LB_sol, obj_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R), Array{Float64, 2}(undef, R, I)
    x_init = zeros(Float64, L)
    ind = collect(1:L)
    x_ind = sample(ind, convert(Int64, deployment_dict[1]))
    x_init[x_ind] .= 1.
    n = convert(Float64, deployment_dict[1])

    for r = 1:R
      println("Run ", r, "/", R)
      x_sol[r, :], LB_sol[r], obj_sol[r, :] = simulated_annealing_local_search(D, c, n, N, I, E, x_init, T_init)
    end

  else
    println("No such run available.")
    throw(ArgumentError)
  end

  return x_sol, LB_sol, obj_sol

end

function main_GRED(deployment_dict, D, c, R, eps, run)

  deployment_dict = Dict([(convert(Int64, k), convert(Int64, deployment_dict[k])) for k in keys(deployment_dict)])
  D  = convert.(Float64, D)
  c = convert(Float64, c)
  R = convert(Int64, R)

  W, L = size(D)

  if run == "RGH"
    x_sol, LB_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R)
    n = convert(Float64, deployment_dict[1])
    for r = 1:R
      println("Run ", r, "/", R)
      x_sol[r, :], LB_sol[r] = randomised_greedy_heuristic(D, c, n)
    end
  elseif run == "CGA"
    x_sol, LB_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R)
    n = convert(Float64, deployment_dict[1])
    for r = 1:R
      println("Run ", r, "/", R)
      x_sol[r, :], LB_sol[r] = classic_greedy_algorithm(D, c, n)
    end
  elseif run == "SGA"
    x_sol, LB_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R)
    n = convert(Float64, deployment_dict[1])
    eps = convert(Float64, 0.001)
    for r = 1:R
      println("Run ", r, "/", R)
      x_sol[r, :], LB_sol[r] = stochastic_greedy_algorithm(D, c, n, eps)
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