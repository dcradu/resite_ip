using PyCall

include("optimisation_models.jl")
include("MCP_heuristics.jl")

function main_call(index_dict, deployment_dict, D, c, N, I, E, T_init, R, run)

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

  x_init = solve_MILP_partitioning(D, c, n_partitions, index_dict, "Gurobi")

  if run == "GLS"
    x_sol, LB_sol, obj_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R), Array{Float64, 2}(undef, R, I)
    for r = 1:R
      println("Run ", r, "/", R)
      x_sol[r, :], LB_sol[r], obj_sol[r, :] = greedy_local_search_partition(D, c, n_partitions, N, I, E, x_init, index_dict)
    end
  elseif run == "SALS"
    x_sol, LB_sol, obj_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R), Array{Float64, 2}(undef, R, I)
    for r = 1:R
      println("Run ", r, "/", R)
      x_sol[r, :], LB_sol[r], obj_sol[r, :] = simulated_annealing_local_search_partition(D, c, n_partitions, N, I, E, x_init, T_init, index_dict)
    end
  else
    println("No such run available.")
    throw(ArgumentError)
  end

  x = x_sol[findmax(LB_sol), :]
  print(size(x))

  return x
end