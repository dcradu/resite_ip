using PyCall
using Dates

include("models.jl")
include("heuristics.jl")

function main_SA(index_dict::Dict{Any, Any}, deployment_dict::Dict{Any, Any}, legacy_index_list::Vector{Int64},
                 D::Array{Float64, 2}, c::Float64,
                 neighborhood_radius::Int64=1, T_init::Float64=200., p::Float64=0.05,
                 no_iterations::Int64=1000, no_epochs::Int64=500, no_runs::Int64=100, no_runs_init::Int64=100,
                 init_sol_algorithm::String="MIR")

  index_dict = Dict([(convert(Int64, k), convert(Int64, index_dict[k])) for k in keys(index_dict)])
  deployment_dict = Dict([(convert(Int64, k), convert(Int64, deployment_dict[k])) for k in keys(deployment_dict)])

  W, L = size(D)
  n_partitions = [deployment_dict[i] for i in 1:maximum(values(index_dict))]

  x_sol, LB_sol, obj_sol = Array{Float64, 2}(undef, no_runs, L), Array{Float64, 1}(undef, no_runs), Array{Float64, 2}(undef, no_runs, no_iterations)

  if init_sol_algorithm == "MIR"
    x_init = solve_MILP_partitioning(D, c, n_partitions, index_dict, legacy_index_list, "Gurobi")

  elseif init_sol_algorithm == "SGH"
    x_init_alg, LB_init_alg = Array{Float64, 2}(undef, no_runs_init, L), Array{Float64, 1}(undef, no_runs_init)
    for r = 1:no_runs_init
      if (div(r, 10) > 0) & (mod(r, 10) == 0)
        @info "$(Dates.format(now(), "HH:MM:SS")) Run $(r)/$(no_runs_init) of $(init_sol_algorithm)"
      end
      x_init_alg[r, :], LB_init_alg[r] = randomised_greedy_heuristic_partition(D, c, n_partitions, p,
                                                                               index_dict, legacy_index_list)
    end
    LB_init_alg_best = argmax(LB_init_alg)
    x_init = x_init_alg[LB_init_alg_best, :]

  else
    println("No such algorithm available.")
    throw(ArgumentError)
  end

  println("Initial solution retrieved. Starting local search.")
  for r = 1:no_runs
    if (div(r, 10) > 0) & (mod(r, 10) == 0)
      @info "$(Dates.format(now(), "HH:MM:SS")) Run $(r)/$(no_runs) of LS"
    end
    x_sol[r, :], LB_sol[r], obj_sol[r, :] = simulated_annealing_local_search_partition(D, c, n_partitions,
                                                                                       neighborhood_radius, no_iterations, no_epochs, T_init,
                                                                                       x_init, index_dict, legacy_index_list)
  end

  return x_sol, LB_sol, obj_sol

end

function main_SGH(index_dict::Dict{Any, Any}, deployment_dict::Dict{Any, Any}, legacy_index_list::Vector{Int64},
                  D::Array{Float64, 2}, c::Float64,
                  p::Float64=0.05, no_runs::Int64=100, algorithm::String="SGH")

  index_dict = Dict([(convert(Int64, k), convert(Int64, index_dict[k])) for k in keys(index_dict)])
  deployment_dict = Dict([(convert(Int64, k), convert(Int64, deployment_dict[k])) for k in keys(deployment_dict)])

  W, L = size(D)
  n_partitions = Vector{Int64}(vec([deployment_dict[i] for i in 1:maximum(values(index_dict))]))

  if algorithm == "SGH"
    x_sol, LB_sol = Array{Float64, 2}(undef, no_runs, L), Array{Float64, 1}(undef, no_runs)
    for r = 1:no_runs
      if (div(r, 10) > 0) & (mod(r, 10) == 0)
        @info "$(Dates.format(now(), "HH:MM:SS")) Run $(r)/$(no_runs)"
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

function main_DGH(index_dict::Dict{Any, Any}, deployment_dict::Dict{Any, Any}, legacy_index_list::Vector{Int64},
                  D::Array{Float64, 2}, c::Float64,
                  no_runs::Int64=100, algorithm::String="DGH")

  index_dict = Dict([(convert(Int64, k), convert(Int64, index_dict[k])) for k in keys(index_dict)])
  deployment_dict = Dict([(convert(Int64, k), convert(Int64, deployment_dict[k])) for k in keys(deployment_dict)])

  W, L = size(D)
  n_partitions = Vector{Int64}(vec([deployment_dict[i] for i in 1:maximum(values(index_dict))]))

  if algorithm == "DGH"
    x_sol, LB_sol = Array{Float64, 2}(undef, no_runs, L), Array{Float64, 1}(undef, no_runs)
    for r = 1:no_runs
      if (div(r, 2) > 0) & (mod(r, 2) == 0)
        @info "$(Dates.format(now(), "HH:MM:SS")) Run $(r)/$(no_runs)"
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
