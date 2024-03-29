using JuMP
using Gurobi

function solve_MILP_partitioning(D::Array{Float64, 2},
                                 c::Float64,
                                 n::Array{Int64, 1},
                                 partitions_indices::Dict{Int64, Int64},
                                 legacy_indices::Vector{Int64},
                                 solver::String)

  W = size(D)[1]
  L = size(D)[2]
  P = length(n)

  # Computes number of locations in each partition
  cnt = zeros(Int64, P)
  for i = 1:L
    cnt[partitions_indices[i]] += 1
  end

  # Computes indices of partitions
  ind_part = Vector{Int64}(undef, P+1)
  ind_part[1] = 1
  for i = 1:P
    ind_part[i+1] = ind_part[i] + cnt[i]
  end

  # Selects solver
  if solver == "Gurobi"
    MILP_model = Model(optimizer_with_attributes(Gurobi.Optimizer,
                 "TimeLimit" => 7200., "MIPGap" => 0.01, "LogToConsole" => 0, "Threads" => 1))
  else
      println("Please use Gurobi. No other solver currently supported.")
      throw(ArgumentError)
  end

  # Defines variables
  @variable(MILP_model, x[1:L], Bin)
  if !(isempty(legacy_indices))
      for v in x[legacy_indices]
          set_lower_bound(v, 1)
          set_upper_bound(v, 1)
      end
  end
  @variable(MILP_model, 0 <= y[1:W] <= 1)

  # Defines Constraints
  @constraint(MILP_model,
              cardinality[i=1:P],
              sum(x[ind_part[i]:(ind_part[i+1]-1)]) == n[i])
  @constraint(MILP_model, covering, D * x .>= c * y)

  # Defines objective function
  @objective(MILP_model, Max, sum(y))

  # Solves model
  optimize!(MILP_model)

  # Extracts solution
  x_sol = round.(value.(x))

  return x_sol

end
