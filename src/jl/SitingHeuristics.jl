using PyCall
using Dates
using Random

include("optimisation_models.jl")
include("MCP_heuristics.jl")
include("cross_validation_tools.jl")

pickle = pyimport("pickle")

#################### Useful Functions #######################

function myunpickle(filename)

  r = nothing
  @pywith pybuiltin("open")(filename,"rb") as f begin

    r = pickle.load(f)
  end
  return r

end

function main_MIRSA(index_dict, deployment_dict, D, c, N, I, E, T_init, R, run, p, data_path)

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
  p = string(p)
  data_path = string(data_path)
  legacy_index = Vector{Int64}(undef, 0)

  W, L = size(D)

  P = maximum(values(index_dict))
  n = convert(Float64, deployment_dict[1])

  if run == "MIR"

    x_sol, LB_sol, obj_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R), Array{Float64, 2}(undef, R, I)
    x_init = solve_MILP(D, c, n, "Gurobi")

    for r = 1:R
      x_sol[r, :], LB_sol[r], obj_sol[r, :] = simulated_annealing_local_search(D, c, n, N, I, E, x_init, T_init, legacy_index)
    end

  elseif run == "RS"

    x_sol, LB_sol, obj_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R), Array{Float64, 2}(undef, R, I)
    x_init = zeros(Int64, L)
    ind_set = [l for l in 1:L]
    n_while = n

    while n_while > 0
      loc = sample(ind_set)
      deleteat!(ind_set, findall(x->x==loc, ind_set))
      x_init[loc] = 1
      n_while -= 1
    end

    x_init = convert.(Float64, x_init)

    for r = 1:R
      x_sol[r, :], LB_sol[r], obj_sol[r, :] = simulated_annealing_local_search(D, c, n, N, I, E, x_init, T_init, legacy_index)
    end

  elseif run == "SGH"

    x_sol, LB_sol, obj_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R), Array{Float64, 2}(undef, R, I)

    # Read LB_init from file
    LB_init = myunpickle(joinpath(data_path, "objective_vector.p"))
    # Read x_init from file
    x_init = myunpickle(joinpath(data_path, "solution_matrix.p"))

    LB_init_best = argmax(LB_init)
    x_init_best = x_init[LB_init_best, :]

    for r = 1:R
      x_sol[r, :], LB_sol[r], obj_sol[r, :] = simulated_annealing_local_search(D, c, n, N, I, E, x_init_best, T_init, legacy_index)
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
      x_sol[r, :], LB_sol[r] = threshold_greedy_algorithm(D, c, n)
    end

  elseif run == "STGH"
    x_sol, LB_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R)
    n = convert(Float64, deployment_dict[1])
    p = convert(Float64, p)
    for r = 1:R
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
      x_sol[r, :], LB_sol[r] = random_search(D, c, n, I)
    end

  else
    println("No such run available.")
    throw(ArgumentError)
  end

  return x_sol, LB_sol

end

function main_CROSS(D, c, n, k, number_years, number_years_training, number_years_testing, number_experiments, number_runs_per_experiment, criterion, cross_validation_method)

  D = convert.(Float64, D)
  c = convert(Float64, c)
  n = convert(Int64, n)
  k = convert(Int64, k)
  number_years = convert(Int64, number_years)
  number_years_training = convert(Int64, number_years_training)
  number_years_testing = convert(Int64, number_years_testing)
  number_experiments = convert(Int64, number_experiments)
  number_runs_per_experiment = convert(Int64, number_runs_per_experiment)
  criterion = convert(String, criterion)
  cross_validation_method = convert(String, cross_validation_method)

  if cross_validation_method == "custom"
      obj_training, obj_testing, ind_training, ind_testing = custom_cross_validation(D, c, n, number_years, number_years_training, number_years_testing, number_experiments, number_runs_per_experiment, criterion);
  elseif cross_validation_method == "k_fold"
      obj_training, obj_testing, ind_training, ind_testing = k_fold_cross_validation(D, c, n, k, number_years, number_runs_per_experiment, criterion);
  end

  return obj_training, obj_testing, ind_training, ind_testing

end
