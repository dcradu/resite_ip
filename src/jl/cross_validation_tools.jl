using StatsBase
using JuMP
using Gurobi
using Statistics
using PyCall

pickle = pyimport("pickle")

#################### Functions to Pickle and Unpickle #######################

function myunpickle(filename)

  r = nothing
  @pywith pybuiltin("open")(filename,"rb") as f begin
    r = pickle.load(f)
  end
  return r

end

function mypickle(filename, obj)

  out = open(filename,"w")
  pickle.dump(obj, out)
  close(out)

 end

 #################### k-fold Cross Validation #######################

function k_fold_cross_validation(D::Array{Float64, 2}, c::Float64, n::Int64, k::Int64, number_years::Int64, number_runs_per_experiment::Int64, criterion::String, algorithm::String="rga", number_periods_per_year::Int64=8760, p::Float64=0.05, N::Int64=1, I::Int64=2000, E::Int64=500, T_init::Float64=100.)

    W, L = size(D)
    years_pool, windows_pool = [i for i = 1:number_years], [[j+number_periods_per_year*(i-1) for j = 1:number_periods_per_year] for i = 1:number_years]
    years_dict = Dict(years_pool .=> windows_pool)
    if mod(number_years, k) != 0
        unused_years = mod(number_years, k)
        years_dropped = sample(years_pool, unused_years, replace=false)
        filter!(d -> !(d.first in years_dropped), years_dict)
        filter!(a -> !(a in years_dropped), years_pool)
    end

    if mod(number_runs_per_experiment, 2) == 0
        number_runs_per_experiment += 1
    end
    runs = [r for r = 1:number_runs_per_experiment]

    years_pool  .= sample(years_pool, length(years_pool), replace=false)
    sample_size = convert(Int64, length(years_pool) / k)
    samples, years_subsets = [i for i = 1:k], [years_pool[1+sample_size*(i-1):sample_size*i] for i = 1:k]
    samples_dict = Dict(samples .=> years_subsets)

    samples_testing, samples_training = 0, Vector{Int64}(undef, k-1)
    years_testing, years_training = Vector{Int64}(undef, sample_size), Vector{Int64}(undef, length(years_pool)-sample_size)
    n_rows_testing, n_rows_training = sample_size * number_periods_per_year, (length(years_pool)-sample_size) * number_periods_per_year
    col_tmp_testing, col_tmp_training = Vector{Float64}(undef, n_rows_testing), Vector{Float64}(undef, n_rows_training)
    col_list_testing, col_list_training = [l for l = 1:L], [l for l = 1:L]
    col_zeros_testing, col_zeros_training = zeros(Float64, n_rows_testing), zeros(Float64, n_rows_training)
    y_tmp_testing, y_tmp_training = Vector{Float64}(undef, n_rows_testing), Vector{Float64}(undef, n_rows_training)
    D_testing, D_training = Array{Float64, 2}(undef, n_rows_testing, L), Array{Float64, 2}(undef, n_rows_training, L)
    ind_testing, ind_training = Array{Int64, 2}(undef, k, n), Array{Int64, 2}(undef, k, n)
    ind_tmp_testing, ind_tmp_training = Array{Int64, 2}(undef, number_runs_per_experiment, n), Array{Int64, 2}(undef, number_runs_per_experiment, n)
    ind_tmp = Vector{Int64}(undef, n)
    obj_testing, obj_training = Vector{Float64}(undef, k), Vector{Float64}(undef, k)
    obj_tmp_testing, obj_tmp_training = Vector{Float64}(undef, number_runs_per_experiment), Vector{Float64}(undef, number_runs_per_experiment)

    @inbounds for s in samples
        samples_testing = samples[s]
        samples_training .= filter(a -> a != samples[s], samples)
        years_testing .= samples_dict[samples_testing]
        years_training .= reduce(vcat,collect(samples_dict[j] for j in samples_training))

        matrix_slicer!(D, D_training, years_training, years_dict, number_periods_per_year, col_list_training)
        matrix_slicer!(D, D_testing, years_testing, years_dict, number_periods_per_year, col_list_testing)

        println("Training Experiment ", s)
        @inbounds for run in runs
            println("Training Run ", run)
            if algorithm == "mirsa"
                ind_tmp .= mirsa(D_training, c, n, N, I, E, T_init)
            elseif algorithm == "rga"
                ind_tmp .= randomised_greedy_algorithm(D_training, c, n, p)
            elseif algorithm == "ga"
                ind_tmp .= greedy_algorithm(D_training, c, n)
            end
            obj_tmp_training[run] = evaluate_obj(D_training, c, ind_tmp, col_tmp_training, col_zeros_training, y_tmp_training)
            ind_tmp_training[run,:] .= ind_tmp
        end
        if criterion == "median"
            reference_run = findall(obj_tmp_training .== median(obj_tmp_training))[1]
        elseif criterion == "max"
            reference_run = findall(obj_tmp_training .== maximum(obj_tmp_training))[1]
        end
        ind_tmp .= view(ind_tmp_training, reference_run, :)
        obj_training[s] = evaluate_obj(D_testing, c, ind_tmp, col_tmp_testing, col_zeros_testing, y_tmp_testing)
        ind_training[s, :] .= ind_tmp

        println("Testing Experiment ", s)
        @inbounds for run in runs
            println("Testing Run ", run)
            if algorithm == "mirsa"
                ind_tmp .= mirsa(D_testing, c, n, N, I, E, T_init)
            elseif algorithm == "rga"
                ind_tmp .= randomised_greedy_algorithm(D_testing, c, n, p)
            elseif algorithm == "ga"
                ind_tmp .= greedy_algorithm(D_testing, c, n)
            end
            obj_tmp_testing[run] = evaluate_obj(D_testing, c, ind_tmp, col_tmp_testing, col_zeros_testing, y_tmp_testing)
            ind_tmp_testing[run,:] .= ind_tmp
        end
        if criterion == "median"
            reference_run = findall(obj_tmp_testing .== median(obj_tmp_testing))[1]
        elseif criterion == "max"
            reference_run = findall(obj_tmp_testing .== maximum(obj_tmp_testing))[1]
        end
        obj_testing[s] = obj_tmp_testing[reference_run]
        ind_testing[s, :] .= view(ind_tmp_testing, reference_run, :)
    end
    return ind_training, ind_testing, obj_training, obj_testing
end

function time_k_fold_cross_validation(D::Array{Float64, 2}, c::Float64, n::Int64, k::Int64, number_years::Int64, number_runs_per_experiment::Int64, criterion::String, algorithm::String="rga", number_periods_per_year::Int64=8760, p::Float64=0.05, N::Int64=1, I::Int64=2000, E::Int64=500, T_init::Float64=100.)
  @time k_fold_cross_validation(D, c, n, k, number_years, number_runs_per_experiment, criterion)
end

#################### Custom Cross Validation #######################

function custom_cross_validation(D::Array{Float64, 2}, c::Float64, n::Int64, number_years::Int64, number_years_training::Int64, number_years_testing::Int64, number_experiments::Int64, number_runs_per_experiment::Int64, criterion::String, algorithm::String="rga", number_periods_per_year::Int64=8760, p::Float64=0.05, N::Int64=1, I::Int64=2000, E::Int64=500, T_init::Float64=100.)

    W, L = size(D)
    years_pool, windows_pool = [i for i = 1:number_years], [[j+number_periods_per_year*(i-1) for j = 1:number_periods_per_year] for i = 1:number_years]
    years_windows_dict = Dict(years_pool .=> windows_pool)

    if (number_years_testing + number_years_training) != number_years
        unused_years = number_years - (number_years_testing + number_years_training)
        years_dropped = sample(years_pool, unused_years, replace=false)
        filter!(d -> !(d.first in years_dropped), years_windows_dict)
        filter!(a -> !(a in years_dropped), years_pool)
    end

    if mod(number_runs_per_experiment, 2) == 0
        number_runs_per_experiment += 1
    end

    runs = [r for r = 1:number_runs_per_experiment]
    experiments = [e for e = 1:number_experiments]

    years_training_dict = Dict([(e, Vector{Int64}(undef, number_years_training)) for e in experiments])
    years_testing_dict = Dict([(e, Vector{Int64}(undef, number_years_testing)) for e in experiments])
    @inbounds for e in experiments
        years_training_dict[e]  .= sample(years_pool, number_years_training, replace=false)
        years_testing_dict[e] .= filter(a -> !(a in years_training_dict[e]), years_pool)
    end

    number_rows_testing, number_rows_training = number_years_testing * number_periods_per_year, number_years_training * number_periods_per_year
    col_tmp_testing, col_tmp_training = Vector{Float64}(undef, number_rows_testing), Vector{Float64}(undef, number_rows_training)
    col_list_testing, col_list_training = [l for l = 1:L], [l for l = 1:L]
    col_zeros_testing, col_zeros_training = zeros(Float64, number_rows_testing), zeros(Float64, number_rows_training)
    y_tmp_testing, y_tmp_training = Vector{Float64}(undef, number_rows_testing), Vector{Float64}(undef, number_rows_training)
    D_testing, D_training = Array{Float64, 2}(undef, number_rows_testing, L), Array{Float64, 2}(undef, number_rows_training, L)
    ind_testing, ind_training = Array{Int64, 2}(undef, number_experiments, n), Array{Int64, 2}(undef, number_experiments, n)
    ind_tmp_testing, ind_tmp_training = Array{Int64, 2}(undef, number_runs_per_experiment, n), Array{Int64, 2}(undef, number_runs_per_experiment, n)
    ind_tmp = Vector{Int64}(undef, n)
    obj_testing, obj_training = Vector{Float64}(undef, number_experiments), Vector{Float64}(undef, number_experiments)
    obj_tmp_testing, obj_tmp_training = Vector{Float64}(undef, number_runs_per_experiment), Vector{Float64}(undef, number_runs_per_experiment)

    @inbounds for e in experiments

        println("Training Years: ", years_training_dict[e])
        println("Testing Years: ", years_testing_dict[e])
        matrix_slicer!(D, D_training, years_training_dict[e], years_windows_dict, number_periods_per_year, col_list_training)
        matrix_slicer!(D, D_testing, years_testing_dict[e], years_windows_dict, number_periods_per_year, col_list_testing)

        println("Training Experiment ", e)
        @inbounds for run in runs
            println("Training Run ", run)
            if algorithm == "mirsa"
                ind_tmp .= mirsa(D_training, c, n, N, I, E, T_init)
            elseif algorithm == "rga"
                ind_tmp .= randomised_greedy_algorithm(D_training, c, n, p)
            elseif algorithm == "ga"
                ind_tmp .= greedy_algorithm(D_training, c, n)
            end
            obj_tmp_training[run] = evaluate_obj(D_training, c, ind_tmp, col_tmp_training, col_zeros_training, y_tmp_training)
            ind_tmp_training[run,:] .= ind_tmp
        end
        if criterion == "median"
            reference_run = findall(obj_tmp_training .== median(obj_tmp_training))[1]
        elseif criterion == "max"
            reference_run = findall(obj_tmp_training .== maximum(obj_tmp_training))[1]
        end
        ind_tmp .= view(ind_tmp_training, reference_run, :)
        obj_training[e] = evaluate_obj(D_testing, c, ind_tmp, col_tmp_testing, col_zeros_testing, y_tmp_testing)
        ind_training[e, :] .= ind_tmp

        println("Testing Experiment ", e)
        @inbounds for run in runs
            println("Testing Run ", run)
            if algorithm == "mirsa"
                ind_tmp .= mirsa(D_testing, c, n, N, I, E, T_init)
            elseif algorithm == "rga"
                ind_tmp .= randomised_greedy_algorithm(D_testing, c, n, p)
            elseif algorithm == "ga"
                ind_tmp .= greedy_algorithm(D_testing, c, n)
            end
            obj_tmp_testing[run] = evaluate_obj(D_testing, c, ind_tmp, col_tmp_testing, col_zeros_testing, y_tmp_testing)
            ind_tmp_testing[run,:] .= ind_tmp
        end
        if criterion == "median"
            reference_run = findall(obj_tmp_testing .== median(obj_tmp_testing))[1]
        elseif criterion == "max"
            reference_run = findall(obj_tmp_testing .== maximum(obj_tmp_testing))[1]
        end
        obj_testing[e] = obj_tmp_testing[reference_run]
        ind_testing[e, :] .= view(ind_tmp_testing, reference_run, :)
    end
    return ind_training, ind_testing, obj_training, obj_testing
end

function time_custom_cross_validation(D::Array{Float64, 2}, c::Float64, n::Int64, number_years::Int64, number_years_training::Int64, number_years_testing::Int64, number_experiments::Int64, number_runs_per_experiment::Int64, criterion::String, algorithm::String="rga", number_periods_per_year::Int64=8760, p::Float64=0.05, N::Int64=1, I::Int64=2000, E::Int64=500, T_init::Float64=100.)
  @time custom_cross_validation(D, c, n, number_years, number_years_training, number_years_testing, number_experiments, number_runs_per_experiment, criterion)
end

#################### Auxiliary Functions #######################

function evaluate_obj(D_tmp::Array{Float64, 2}, c::Float64, ind_locations::Vector{Int64}, col_tmp::Vector{Float64}, col_zeros::Vector{Float64}, y_tmp::Vector{Float64})

    col_tmp .= col_zeros
    @inbounds for ind in ind_locations
        col_tmp .+= view(D_tmp, :, ind)
    end
    y_tmp .= col_tmp .>= c
    return sum(y_tmp)

end

function matrix_slicer!(D::Array{Float64, 2}, D_tmp::Array{Float64, 2}, keys::Vector{Int64}, row_dict::Dict{Int64, Vector{Int64}}, number_rows::Int64, column_list::Vector{Int64})

    pointer_col = 1
    @inbounds for ind in column_list
        pointer_start, pointer_end = 1, number_rows
        @inbounds for key in keys
            pointer_start_row_tmp, pointer_end_row_tmp = row_dict[key][1], row_dict[key][number_rows]
            D_tmp[pointer_start:pointer_end, pointer_col] .= view(D, pointer_start_row_tmp:pointer_end_row_tmp, ind)
            pointer_start += number_rows
            pointer_end += number_rows
        end
        pointer_col += 1
    end
end

function time_matrix_slicer!(D::Array{Float64, 2}, D_tmp::Array{Float64, 2}, row_dict::Dict{Int64, Vector{Int64}}, number_rows::Int64, column_list::Vector{Int64})
    @time matrix_slicer!(D, D_tmp, row_dict, number_rows, column_list)
end

#################### MIRSA #######################

function mirsa(D::Array{Float64, 2}, c::Float64, n::Int64, N::Int64, I::Int64, E::Int64, T_init::Float64)

  W, L = size(D)

  # Solve Mixed-Integer Relaxation

  MILP_model = Model(optimizer_with_attributes(Gurobi.Optimizer, "TimeLimit" => 3600., "MIPGap" => 0.05, "LogToConsole" => 0))

  @variable(MILP_model, x[1:L], Bin)
  @variable(MILP_model, 0 <= y[1:W] <= 1)

  @constraint(MILP_model, cardinality, sum(x) == n)
  @constraint(MILP_model, covering, D * x .>= c * y)

  @objective(MILP_model, Max, sum(y))

  optimize!(MILP_model)

  x_init = round.(value.(x))

  # Pre-allocate x-related arrays
  ind_ones_incumbent = Vector{Int64}(undef, n)
  ind_ones_incumbent_filtered = Vector{Int64}(undef, n-N)
  ind_zeros_incumbent = Vector{Int64}(undef, L-n)
  ind_zeros_incumbent_filtered = Vector{Int64}(undef, L-n-N)
  ind_ones2zeros_candidate = Vector{Int64}(undef, N)
  ind_zeros2ones_candidate = Vector{Int64}(undef, N)
  ind_ones2zeros_tmp = Vector{Int64}(undef, N)
  ind_zeros2ones_tmp = Vector{Int64}(undef, N)

  # Pre-allocate y-related arrays
  y_incumbent = Vector{Float64}(undef, W)
  y_tmp = Vector{Float64}(undef, W)

  Dx_incumbent = zeros(Float64, W)
  Dx_tmp = Vector{Float64}(undef, W)

  # Initialise
  ind_ones_incumbent .= findall(x_init .== 1.)
  ind_zeros_incumbent .= findall(x_init .== 0.)
  @inbounds for ind in ind_ones_incumbent
    Dx_incumbent .+= view(D, :, ind)
  end
  y_incumbent .= Dx_incumbent .>= c
  obj_incumbent = sum(y_incumbent)

  # Simulated Annealing Local Search
  @inbounds for i = 1:I
    delta_candidate = -100000
    @inbounds for e = 1:E
      # Sample from neighbourhood
      sample!(ind_ones_incumbent, ind_ones2zeros_tmp, replace=false)
      sample!(ind_zeros_incumbent, ind_zeros2ones_tmp, replace=false)

      # Compute y and associated objective value
      Dx_tmp .= Dx_incumbent
      @inbounds for j = 1:N
        Dx_tmp .+= view(D, :, ind_zeros2ones_tmp[j])
        Dx_tmp .-= view(D, :, ind_ones2zeros_tmp[j])
      end
      y_tmp .= Dx_tmp .>= c

      # Update objective difference
      delta_tmp = sum(y_tmp) - obj_incumbent

      # Update candidate solution
      if delta_tmp > delta_candidate
        ind_ones2zeros_candidate .= ind_ones2zeros_tmp
        ind_zeros2ones_candidate .= ind_zeros2ones_tmp
        delta_candidate = delta_tmp
      end
    end
    if delta_candidate > 0
      ind_ones_incumbent_filtered .= filter(a -> !(a in ind_ones2zeros_candidate), ind_ones_incumbent)
      ind_ones_incumbent[1:(n-N)] .= ind_ones_incumbent_filtered
      ind_ones_incumbent[(n-N+1):n] .= ind_zeros2ones_candidate
      ind_zeros_incumbent_filtered .= filter(a -> !(a in ind_zeros2ones_candidate), ind_zeros_incumbent)
      ind_zeros_incumbent[1:(L-n-N)] .= ind_zeros_incumbent_filtered
      ind_zeros_incumbent[(L-n-N+1):(L-n)] .= ind_ones2zeros_candidate
      @inbounds for j = 1:N
        Dx_incumbent .+= view(D, :, ind_zeros2ones_candidate[j])
        Dx_incumbent .-= view(D, :, ind_ones2zeros_candidate[j])
      end
      y_incumbent .= Dx_incumbent .>= c
      obj_incumbent = sum(y_incumbent)
    else
      T = T_init * exp(-10*i/I)
      p = exp(delta_candidate / T)
      d = Binomial(1, p)
      b = rand(d)
      if b == 1
        ind_ones_incumbent_filtered .= filter(a -> !(a in ind_ones2zeros_candidate), ind_ones_incumbent)
        ind_ones_incumbent[1:(n-N)] .= ind_ones_incumbent_filtered
        ind_ones_incumbent[(n-N+1):n] .= ind_zeros2ones_candidate
        ind_zeros_incumbent_filtered .= filter(a -> !(a in ind_zeros2ones_candidate), ind_zeros_incumbent)
        ind_zeros_incumbent[1:(L-n-N)] .= ind_zeros_incumbent_filtered
        ind_zeros_incumbent[(L-n-N+1):(L-n)] .= ind_ones2zeros_candidate
        @inbounds for j = 1:N
          Dx_incumbent .+= view(D, :, ind_zeros2ones_candidate[j])
          Dx_incumbent .-= view(D, :, ind_ones2zeros_candidate[j])
        end
        y_incumbent .= Dx_incumbent .>= c
        obj_incumbent = sum(y_incumbent)
      end
    end
  end
  return ind_ones_incumbent
end

#################### Randomised Greedy Algorithm #######################

function randomised_greedy_algorithm(D::Array{Float64,2}, c::Float64, n::Int64, p::Float64)

  W, L = size(D)
  s = convert(Int64, round(L*p))
  random_ind_set = Vector{Int64}(undef, s)
  ind_compl_incumbent = [i for i in 1:L]
  ind_incumbent = Vector{Int64}(undef, convert(Int64, n))
  Dx_incumbent = zeros(Float64, W)
  obj_incumbent = 0
  Dx_tmp = Vector{Float64}(undef, W)
  y_tmp = Vector{Float64}(undef, W)
  ind_candidate_list = zeros(Int64, L)
  locations_added, threshold = 0, 0
  @inbounds while locations_added < n
    if locations_added < c
      threshold = locations_added + 1
      obj_candidate = 0
    else
      obj_candidate = obj_incumbent
    end
    ind_candidate_pointer = 1
    sample!(ind_compl_incumbent, random_ind_set, replace=false)
    @inbounds for ind in random_ind_set
        Dx_tmp .= Dx_incumbent .+ view(D, :, ind)
        y_tmp .= Dx_tmp .>= threshold
        obj_tmp = sum(y_tmp)
        if obj_tmp > obj_candidate
          ind_candidate_pointer = 1
          ind_candidate_list[ind_candidate_pointer] = ind
          obj_candidate = obj_tmp
          ind_candidate_pointer += 1
        elseif obj_tmp == obj_candidate
          ind_candidate_list[ind_candidate_pointer] = ind
          ind_candidate_pointer += 1
        end
    end
    ind_candidate = sample(view(ind_candidate_list, 1:ind_candidate_pointer-1))
    ind_incumbent[locations_added+1] = ind_candidate
    filter!(a -> a != ind_candidate, ind_compl_incumbent)
    Dx_incumbent .= Dx_incumbent .+ view(D, :, ind_candidate)
    obj_incumbent = obj_candidate
    locations_added += 1
  end
  return ind_incumbent
end


function greedy_algorithm(D::Array{Float64,2}, c::Float64, n::Int64)

  W, L = size(D)
  n = convert(Int64, n)
  ind_compl_incumbent = [i for i in 1:L]
  ind_incumbent = Vector{Int64}(undef, n)
  Dx_incumbent = zeros(Float64, W)
  obj_incumbent = 0
  Dx_tmp = Vector{Float64}(undef, W)
  y_tmp = Vector{Float64}(undef, W)
  ind_candidate_list = zeros(Int64, L)
  locations_added, threshold = 0, 0
  @inbounds while locations_added < n
    if locations_added < c
      threshold = locations_added + 1
      obj_candidate = 0
    else
      obj_candidate = obj_incumbent
    end
    ind_candidate_pointer = 1
    @inbounds for ind in ind_compl_incumbent
        Dx_tmp .= Dx_incumbent .+ view(D, :, ind)
        y_tmp .= Dx_tmp .>= threshold
        obj_tmp = sum(y_tmp)
        if obj_tmp > obj_candidate
          ind_candidate_pointer = 1
          ind_candidate_list[ind_candidate_pointer] = ind
          obj_candidate = obj_tmp
          ind_candidate_pointer += 1
        elseif obj_tmp == obj_candidate
          ind_candidate_list[ind_candidate_pointer] = ind
          ind_candidate_pointer += 1
        end
    end
    ind_candidate = sample(view(ind_candidate_list, 1:ind_candidate_pointer-1))
    ind_incumbent[locations_added+1] = ind_candidate
    filter!(a -> a != ind_candidate, ind_compl_incumbent)
    Dx_incumbent .= Dx_incumbent .+ view(D, :, ind_candidate)
    obj_incumbent = obj_candidate
    locations_added += 1
  end
  return ind_incumbent
end
