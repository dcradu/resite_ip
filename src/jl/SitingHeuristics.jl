using PyCall
using Dates

include("models.jl")
include("heuristics.jl")

function main_SA(index_dict, deployment_dict, legacy_index_list,
                 criticality_matrix, c,
                 neighborhood_radius=1, T_init=200, p=0.05,
                 no_iterations=1000, no_epochs=500, no_runs=100, no_runs_init=100,
                 init_sol_algorithm="MIR")

  index_dict = Dict([(convert(Int64, k), convert(Int64, index_dict[k])) for k in keys(index_dict)])
  deployment_dict = Dict([(convert(Int64, k), convert(Int64, deployment_dict[k])) for k in keys(deployment_dict)])
  legacy_index_list = convert.(Int64, legacy_index_list)

  D  = convert.(Float64, criticality_matrix)
  c = convert(Float64, c)
  N = convert(Int64, neighborhood_radius)
  I = convert(Int64, no_iterations)
  E = convert(Int64, no_epochs)
  T_init = convert(Float64, T_init)
  p = convert(Float64, p)
  R = convert(Int64, no_runs)
  R_init = convert(Int64, no_runs_init)
  init_sol_algorithm = string(init_sol_algorithm)

  W, L = size(D)

  P = maximum(values(index_dict))
  n_partitions = [deployment_dict[i] for i in 1:P]

  x_sol, LB_sol, obj_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R), Array{Float64, 2}(undef, R, I)

  if init_sol_algorithm == "MIR"
    x_init = solve_MILP_partitioning(D, c, n_partitions, index_dict, legacy_index_list, "Gurobi")

  elseif init_sol_algorithm == "SGH"
    x_init_alg, LB_init_alg = Array{Float64, 2}(undef, R_init, L), Array{Float64, 1}(undef, R_init)
    for r = 1:R_init
      if (div(r, 10) > 0) & (mod(r, 10) == 0)
        @info "$(Dates.format(now(), "HH:MM:SS")) Run $(r)/$(R_init) of $(init_sol_algorithm)"
      end
      x_init_alg[r, :], LB_init_alg[r] = randomised_greedy_heuristic_partition(D, c, n_partitions, p,
                                                                               index_dict, legacy_index_list)
    end
    LB_init_alg_best = argmax(LB_init_alg)
    x_init = round.(x_init_alg[LB_init_alg_best, :])

  else
    println("No such algorithm available.")
    throw(ArgumentError)
  end

  println("Initial solution retrieved. Starting local search.")
  for r = 1:R
    if (div(r, 10) > 0) & (mod(r, 10) == 0)
      @info "$(Dates.format(now(), "HH:MM:SS")) Run $(r)/$(R) of LS"
    end
    x_sol[r, :], LB_sol[r], obj_sol[r, :] = simulated_annealing_local_search_partition(D, c, n_partitions,
                                                                                       N, I, E, T_init,
                                                                                       x_init, index_dict, legacy_index_list)
  end

  return x_sol, LB_sol, obj_sol

end

function main_SGH(index_dict, deployment_dict, legacy_index_list,
                  criticality_matrix, c,
                  p=0.05, no_runs=100, algorithm="SGH")

  index_dict = Dict([(convert(Int64, k), convert(Int64, index_dict[k])) for k in keys(index_dict)])
  deployment_dict = Dict([(convert(Int64, k), convert(Int64, deployment_dict[k])) for k in keys(deployment_dict)])
  legacy_index_list = Vector{Int64}(vec(legacy_index_list))

  D  = convert.(Float64, criticality_matrix)
  c = convert(Float64, c)
  R = convert(Int64, no_runs)
  p = convert(Float64, p)
  algorithm = string(algorithm)

  W, L = size(D)

  P = maximum(values(index_dict))
  n_partitions = Vector{Int64}(vec([deployment_dict[i] for i in 1:P]))

  if algorithm == "SGH"
    x_sol, LB_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R)
    for r = 1:R
      if (div(r, 10) > 0) & (mod(r, 10) == 0)
        @info "$(Dates.format(now(), "HH:MM:SS")) Run $(r)/$(R)"
      end
      x_sol[r, :], LB_sol[r] = randomised_greedy_heuristic_partition(D, c, n_partitions, p,
                                                                     index_dict, legacy_index_list)
    end

  else
    println("No such algorithm available.")
    throw(ArgumentError)
  end

  return x_sol, LB_sol

end

function main_DGH(index_dict, deployment_dict, legacy_index_list,
                  criticality_matrix, c,
                  no_runs=100, algorithm="DGH")

  index_dict = Dict([(convert(Int64, k), convert(Int64, index_dict[k])) for k in keys(index_dict)])
  deployment_dict = Dict([(convert(Int64, k), convert(Int64, deployment_dict[k])) for k in keys(deployment_dict)])
  legacy_index_list = Vector{Int64}(vec(legacy_index_list))

  D  = convert.(Float64, criticality_matrix)
  c = convert(Float64, c)
  R = convert(Int64, no_runs)
  algorithm = string(algorithm)

  W, L = size(D)

  P = maximum(values(index_dict))
  n_partitions = Vector{Int64}(vec([deployment_dict[i] for i in 1:P]))

  if algorithm == "DGH"
    x_sol, LB_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R)
    for r = 1:R
      if (div(r, 2) > 0) & (mod(r, 2) == 0)
        @info "$(Dates.format(now(), "HH:MM:SS")) Run $(r)/$(R)"
      end
      x_sol[r, :], LB_sol[r] = greedy_heuristic_partition(D, c, n_partitions,
                                                          index_dict, legacy_index_list)
    end

  else
    println("No such algorithm available.")
    throw(ArgumentError)
  end

  return x_sol, LB_sol

end
