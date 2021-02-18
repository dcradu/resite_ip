using PyCall

include("optimisation_models.jl")
include("MCP_heuristics.jl")

function main_RAND(deployment_dict, D, c, run)

  deployment_dict = Dict([(convert(Int64, k), convert(Int64, deployment_dict[k])) for k in keys(deployment_dict)])
  D  = convert.(Float64, D)
  c = convert(Float64, c)

  if run == "RS"

    x_sol, LB_sol = Array{Float64, 2}(undef, R, L), Array{Float64, 1}(undef, R)
    n = convert(Float64, deployment_dict[1])
    runs = convert(Int64, I*E)

    for r = 1:R
      println("Run ", r, "/", R)
      x_sol[r, :], LB_sol[r] = random_search(D, c, n, runs)
    end

  else
    println("No such run available.")
    throw(ArgumentError)
  end

  return x_sol, LB_sol

end
