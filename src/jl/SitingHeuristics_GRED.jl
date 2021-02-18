using PyCall

include("optimisation_models.jl")
include("MCP_heuristics.jl")

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
