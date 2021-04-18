using StatsBase
using Distributions

#################### Random Search Algorithm #######################

function random_search(D::Array{Float64, 2}, c::Float64, n::Float64, R::Int64)

  W, L = size(D)
  x_incumbent = zeros(Float64, L)
  ind_set = [l for l in 1:L]
  ind_incumbent = Vector{Int64}(undef, convert(Int64, n))
  ind_candidate = Vector{Int64}(undef, convert(Int64, n))
  Dx_candidate = Vector{Float64}(undef, W)
  y_candidate = Vector{Float64}(undef, W)

  LB_incumbent = 0
  for r in 1:R
    ind_candidate .= sample(ind_set, convert(Int64, n), replace=false)
    Dx_candidate = sum(view(D, :, ind_candidate), dims = 2)
    y_candidate = Dx_candidate .>= c
    LB_candidate = sum(y_candidate)
    if LB_candidate >= LB_incumbent
      ind_incumbent .= ind_candidate
      LB_incumbent = LB_candidate
    end
  end
  x_incumbent[ind_incumbent] .= 1.
  return x_incumbent, LB_incumbent
end

#################### Greedy Local Search w/ Partitioning Constraints (Dict Implementation) #######################

# Description: function implementing a greedy local search for geographical regions partitioned into a set of subregions
#
# Comments: 1) types of inputs should match those declared in argument list
#           2) implementation uses both dict and array data structures
#
# Inputs: D - criticality matrix with entries in {0, 1}, where rows represent time windows and columns represent locations
#         c - global criticality threshold
#         n - number of sites to deploy
#         N - number of locations to swap in order to obtain a neighbour of the incumbent solution
#         I - number of iterations (outer loop), defines the number of times the incumbent solution may be updated
#         E - number of epochs (inner loop), defines the number of neighbours of the incumbent solution sampled at each iteration
#         x_init - initial solution, vector with entries in {0, 1}, with cardinality n and whose dimension is compatible with D
#         locations_regions_mapping - dictionary associating its subregion (value) to each location (key)
#
# Outputs: x_incumbent - vector with entries in {0, 1} and cardinality n representing the incumbent solution at the last iteration
#          LB_incumbent - objective value of incumbent solution, provides a lower bound on optimal objective
#          obj - vector storing the incumbent objective value at each iteration
#

function greedy_local_search_partition(D::Array{Float64, 2}, c::Float64, n::Vector{Int64}, N::Int64, I::Int64, E::Int64, x_init::Array{Float64, 1}, locations_regions_mapping::Dict{Int64, Int64})

  W, L = size(D)
  P = maximum(values(locations_regions_mapping))

  # Pre-allocate lower bound vector
  obj = Vector{Int64}(undef, I)

  # Pre-allocate x-related containers
  x_incumbent = Vector{Float64}(undef, L)
  ind_ones2zeros_candidate = Vector{Int64}(undef, N)
  ind_zeros2ones_candidate = Vector{Int64}(undef, N)
  ind_ones2zeros_tmp = Vector{Int64}(undef, N)
  ind_zeros2ones_tmp = Vector{Int64}(undef, N)

  regions = [i for i in 1:P]
  sample_count_per_region = Vector{Int64}(undef, P)
  init_sample_count_per_region = zeros(Int64, P)
  ind_samples_per_region_tmp = Vector{Int64}(undef, P+1)
  ind_samples_per_region_candidate = Vector{Int64}(undef, P+1)
  locations_count_per_region = zeros(Int64, P)
  index_range_per_region = Vector{Int64}(undef, P+1)

  @inbounds for i = 1:L
    locations_count_per_region[locations_regions_mapping[i]] += 1
  end

  ind_ones_incumbent = Dict([(r, Vector{Int64}(undef, n[r])) for r in regions])
  ind_zeros_incumbent = Dict([(r, Vector{Int64}(undef, locations_count_per_region[r]-n[r])) for r in regions])

  index_range_per_region[1] = 1
  @inbounds for j = 1:P
    index_range_per_region[j+1] = index_range_per_region[j] + locations_count_per_region[j]
  end

  # Pre-allocate y-related arrays
  y_incumbent = Array{Bool}(undef, W, 1)
  y_tmp = Array{Bool}(undef, W, 1)
  c_threshold = c .* ones(Float64, W, 1)

  Dx_incumbent = zeros(Float64, W, 1)
  Dx_tmp = Array{Float64}(undef, W, 1)

  # Initialise
  ind_ones, counter_ones = findall(x_init .== 1.), zeros(Int64, P)
  @inbounds for ind in ind_ones
    p = locations_regions_mapping[ind]
    counter_ones[p] += 1
    ind_ones_incumbent[p][counter_ones[p]] = ind
    Dx_incumbent .+= view(D, :, ind)
  end
  y_incumbent .= Dx_incumbent .>= c_threshold

  ind_zeros, counter_zeros = findall(x_init .== 0.), zeros(Int64, P)
  for ind in ind_zeros
    p = locations_regions_mapping[ind]
    counter_zeros[p] += 1
    ind_zeros_incumbent[p][counter_zeros[p]] = ind
  end
  ind_samples_per_region_tmp[1] = 1

  # Iterate
  @inbounds for i = 1:I
    obj[i] = sum(y_incumbent)
    delta_candidate = -1000000
    @inbounds for e = 1:E
      # Sample from neighbourhood
      sample_count_per_region .= init_sample_count_per_region
      @inbounds while sum(sample_count_per_region) < N
        p = sample(regions)
        if (sample_count_per_region[p] < n[p]) && (sample_count_per_region[p] < locations_count_per_region[p] - n[p])
          sample_count_per_region[p] += 1
        end
      end

      @inbounds for i = 1:P
        ind_samples_per_region_tmp[i+1] = ind_samples_per_region_tmp[i] + sample_count_per_region[i]
        if sample_count_per_region[i] != 0
          view(ind_ones2zeros_tmp, ind_samples_per_region_tmp[i]:(ind_samples_per_region_tmp[i+1]-1)) .= sample(ind_ones_incumbent[i], sample_count_per_region[i], replace=false)
          view(ind_zeros2ones_tmp, ind_samples_per_region_tmp[i]:(ind_samples_per_region_tmp[i+1]-1)) .= sample(ind_zeros_incumbent[i], sample_count_per_region[i], replace=false)
        end
      end

      # Compute y and associated objective value
      Dx_tmp .= Dx_incumbent .+ sum(view(D, :, ind_zeros2ones_tmp), dims = 2) .- sum(view(D, :, ind_ones2zeros_tmp), dims = 2)
      y_tmp .= Dx_tmp .>= c_threshold

      # Update objective difference
      delta_tmp = sum(y_tmp) - obj[i]

      # Update candidate solution
      if delta_tmp > delta_candidate
        ind_ones2zeros_candidate .= ind_ones2zeros_tmp
        ind_zeros2ones_candidate .= ind_zeros2ones_tmp
        ind_samples_per_region_candidate .= ind_samples_per_region_tmp
        delta_candidate = delta_tmp
      end
    end
    if delta_candidate > 0
      @inbounds for i = 1:P
        ind_ones_incumbent[i] .= union(setdiff(ind_ones_incumbent[i], view(ind_ones2zeros_candidate, ind_samples_per_region_candidate[i]:(ind_samples_per_region_candidate[i+1]-1))), view(ind_zeros2ones_candidate, ind_samples_per_region_candidate[i]:(ind_samples_per_region_candidate[i+1]-1)))
        ind_zeros_incumbent[i] .= union(setdiff(ind_zeros_incumbent[i], view(ind_zeros2ones_candidate, ind_samples_per_region_candidate[i]:(ind_samples_per_region_candidate[i+1]-1))), view(ind_ones2zeros_candidate, ind_samples_per_region_candidate[i]:(ind_samples_per_region_candidate[i+1]-1)))
      end
      Dx_incumbent .= Dx_incumbent .+ sum(view(D, :, ind_zeros2ones_candidate), dims = 2) .- sum(view(D, :, ind_ones2zeros_candidate), dims = 2)
      y_incumbent .= Dx_incumbent .>= c_threshold
    end
  end
  @inbounds for i in 1:P
    x_incumbent[ind_ones_incumbent[i]] .= 1.
    x_incumbent[ind_zeros_incumbent[i]] .= 0.
  end
  LB = sum(y_incumbent)
  return x_incumbent, LB, obj

end

#################### Simulated Annealing Local Search w/ Partitioning Constraints (Dict Implementation) #######################

# Description: function implementing a simulated annealing-inspired local search for geographical regions partitioned into a set of subregions
#
# Comments: 1) types of inputs should match those declared in argument list
#           2) implementation uses both dict and array data structures
#
# Inputs: D - criticality matrix with entries in {0, 1}, where rows represent time windows and columns represent locations
#         c - global criticality threshold
#         n - number of sites to deploy
#         N - number of locations to swap in order to obtain a neighbour of the incumbent solution
#         I - number of iterations (outer loop), defines the number of times the incumbent solution may be updated
#         E - number of epochs (inner loop), defines the number of neighbours of the incumbent solution sampled at each iteration
#         x_init - initial solution, vector with entries in {0, 1}, with cardinality n and whose dimension is compatible with D
#         T_init - initial temperature from which the (exponentially-decreasing) temperature schedule is constructed
#         locations_regions_mapping - dictionary associating its subregion (value) to each location (key)
#
# Outputs: x_incumbent - vector with entries in {0, 1} and cardinality n representing the incumbent solution at the last iteration
#          LB_incumbent - objective value of incumbent solution, provides a lower bound on optimal objective
#          obj - vector storing the incumbent objective value at each iteration
#

function simulated_annealing_local_search_partition(D::Array{Float64, 2}, c::Float64, n::Vector{Int64}, N::Int64, I::Int64, E::Int64, x_init::Array{Float64, 1}, T_init::Float64, locations_regions_mapping::Dict{Int64, Int64})

  W, L = size(D)
  P = maximum(values(locations_regions_mapping))

  # Pre-allocate lower bound vector
  obj = Vector{Int64}(undef, I)

  # Pre-allocate x-related containers
  x_incumbent = Vector{Float64}(undef, L)
  ind_ones2zeros_candidate = Vector{Int64}(undef, N)
  ind_zeros2ones_candidate = Vector{Int64}(undef, N)
  ind_ones2zeros_tmp = Vector{Int64}(undef, N)
  ind_zeros2ones_tmp = Vector{Int64}(undef, N)

  regions = [i for i in 1:P]
  sample_count_per_region = Vector{Int64}(undef, P)
  init_sample_count_per_region = zeros(Int64, P)
  ind_samples_per_region_tmp = Vector{Int64}(undef, P+1)
  ind_samples_per_region_candidate = Vector{Int64}(undef, P+1)
  locations_count_per_region = zeros(Int64, P)
  index_range_per_region = Vector{Int64}(undef, P+1)

  @inbounds for i = 1:L
    locations_count_per_region[locations_regions_mapping[i]] += 1
  end

  ind_ones_incumbent = Dict([(r, Vector{Int64}(undef, n[r])) for r in regions])
  ind_zeros_incumbent = Dict([(r, Vector{Int64}(undef, locations_count_per_region[r]-n[r])) for r in regions])

  index_range_per_region[1] = 1
  @inbounds for j = 1:P
    index_range_per_region[j+1] = index_range_per_region[j] + locations_count_per_region[j]
  end

  # Pre-allocate y-related arrays
  y_incumbent = Array{Bool}(undef, W, 1)
  y_tmp = Array{Bool}(undef, W, 1)
  c_threshold = c .* ones(Float64, W, 1)

  Dx_incumbent = zeros(Float64, W, 1)
  Dx_tmp = Array{Float64}(undef, W, 1)

  # Initialise
  ind_ones, counter_ones = findall(x_init .== 1.), zeros(Int64, P)
  @inbounds for ind in ind_ones
    p = locations_regions_mapping[ind]
    counter_ones[p] += 1
    ind_ones_incumbent[p][counter_ones[p]] = ind
    Dx_incumbent .+= view(D, :, ind)
  end
  y_incumbent .= Dx_incumbent .>= c_threshold

  ind_zeros, counter_zeros = findall(x_init .== 0.), zeros(Int64, P)
  for ind in ind_zeros
    p = locations_regions_mapping[ind]
    counter_zeros[p] += 1
    ind_zeros_incumbent[p][counter_zeros[p]] = ind
  end
  ind_samples_per_region_tmp[1] = 1

  # Iterate
  @inbounds for i = 1:I
    obj[i] = sum(y_incumbent)
    delta_candidate = -1000000
    @inbounds for e = 1:E
      # Sample from neighbourhood
      sample_count_per_region .= init_sample_count_per_region
      @inbounds while sum(sample_count_per_region) < N
        p = sample(regions)
        if (sample_count_per_region[p] < n[p]) && (sample_count_per_region[p] < locations_count_per_region[p] - n[p])
          sample_count_per_region[p] += 1
        end
      end

      @inbounds for i = 1:P
        ind_samples_per_region_tmp[i+1] = ind_samples_per_region_tmp[i] + sample_count_per_region[i]
        if sample_count_per_region[i] != 0
          view(ind_ones2zeros_tmp, ind_samples_per_region_tmp[i]:(ind_samples_per_region_tmp[i+1]-1)) .= sample(ind_ones_incumbent[i], sample_count_per_region[i], replace=false)
          view(ind_zeros2ones_tmp, ind_samples_per_region_tmp[i]:(ind_samples_per_region_tmp[i+1]-1)) .= sample(ind_zeros_incumbent[i], sample_count_per_region[i], replace=false)
        end
      end

      # Compute y and associated objective value
      Dx_tmp .= Dx_incumbent .+ sum(view(D, :, ind_zeros2ones_tmp), dims = 2) .- sum(view(D, :, ind_ones2zeros_tmp), dims = 2)
      y_tmp .= Dx_tmp .>= c_threshold

      # Update objective difference
      delta_tmp = sum(y_tmp) - obj[i]

      # Update candidate solution
      if delta_tmp > delta_candidate
        ind_ones2zeros_candidate .= ind_ones2zeros_tmp
        ind_zeros2ones_candidate .= ind_zeros2ones_tmp
        ind_samples_per_region_candidate .= ind_samples_per_region_tmp
        delta_candidate = delta_tmp
      end
    end
    if delta_candidate > 0
      @inbounds for i = 1:P
        ind_ones_incumbent[i] .= union(setdiff(ind_ones_incumbent[i], view(ind_ones2zeros_candidate, ind_samples_per_region_candidate[i]:(ind_samples_per_region_candidate[i+1]-1))), view(ind_zeros2ones_candidate, ind_samples_per_region_candidate[i]:(ind_samples_per_region_candidate[i+1]-1)))
        ind_zeros_incumbent[i] .= union(setdiff(ind_zeros_incumbent[i], view(ind_zeros2ones_candidate, ind_samples_per_region_candidate[i]:(ind_samples_per_region_candidate[i+1]-1))), view(ind_ones2zeros_candidate, ind_samples_per_region_candidate[i]:(ind_samples_per_region_candidate[i+1]-1)))
      end
      Dx_incumbent .= Dx_incumbent .+ sum(view(D, :, ind_zeros2ones_candidate), dims = 2) .- sum(view(D, :, ind_ones2zeros_candidate), dims = 2)
      y_incumbent .= Dx_incumbent .>= c_threshold
    else
      T = T_init * exp(-10*i/I)
      p = exp(delta_candidate / T)
      d = Binomial(1, p)
      b = rand(d)
      if b == 1
        @inbounds for i = 1:P
          ind_ones_incumbent[i] .= union(setdiff(ind_ones_incumbent[i], view(ind_ones2zeros_candidate, ind_samples_per_region_candidate[i]:(ind_samples_per_region_candidate[i+1]-1))), view(ind_zeros2ones_candidate, ind_samples_per_region_candidate[i]:(ind_samples_per_region_candidate[i+1]-1)))
          ind_zeros_incumbent[i] .= union(setdiff(ind_zeros_incumbent[i], view(ind_zeros2ones_candidate, ind_samples_per_region_candidate[i]:(ind_samples_per_region_candidate[i+1]-1))), view(ind_ones2zeros_candidate, ind_samples_per_region_candidate[i]:(ind_samples_per_region_candidate[i+1]-1)))
        end
        Dx_incumbent .= Dx_incumbent .+ sum(view(D, :, ind_zeros2ones_candidate), dims = 2) .- sum(view(D, :, ind_ones2zeros_candidate), dims = 2)
        y_incumbent .= Dx_incumbent .>= c_threshold
      end
    end
  end
  @inbounds for i in 1:P
    x_incumbent[ind_ones_incumbent[i]] .= 1.
    x_incumbent[ind_zeros_incumbent[i]] .= 0.
  end
  LB = sum(y_incumbent)
  return x_incumbent, LB, obj

end

################### Multi-Threshold Greedy Heuristic #######################

# Description: function implementing a multi-threshold greedy algorithm with random selection of locations leading to same objective gain for unpartitioned geographical regions (single cardinality constraint)
#
# Comments: 1) types of inputs should match those declared in argument list
#           2) at every iteration, randomisation is used to select the location that should be removed from the locations set when several locations are "tied", i.e., removing each of them leads to the same decrease in objective value
#
# Inputs: D - criticality matrix with entries in {0, 1}, where rows represent time windows and columns represent locations
#         c - global criticality threshold
#         n - number of sites to deploy
#
# Outputs: x_incumbent - vector with entries in {0, 1} and cardinality n representing the incumbent solution at the last iteration
#          LB_incumbent - objective value of incumbent solution, provides a lower bound on optimal objective
#

function threshold_greedy_algorithm(D::Array{Float64,2}, c::Float64, n::Float64)

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
  return ind_incumbent, obj_incumbent
end

function time_threshold_greedy_algorithm(D::Array{Float64,2}, c::Float64, n::Float64)
  @time threshold_greedy_algorithm(D, c, n)
end

#################### Stochastic Multi-Threshold Greedy Algorithm #######################

# Description: function implementing the stochastic multi-threshold greedy algorithm (which generalises the stochastic greedy algorithm for submodular maximization (Mirzasoleiman et al, 2015))
#
# Comments: 1) types of inputs should match those declared in argument list
#           2) at every iteration, randomisation is used to select the location that should be removed from the locations set when several locations are "tied", i.e., removing each of them leads to the same decrease in objective value
#
# Inputs: D - criticality matrix with entries in {0, 1}, where rows represent time windows and columns represent locations
#         c - global criticality threshold
#         n - number of sites to deploy
#         p - factor (between 0 and 1) defines the size of the random subset s = (L/n) x p
#
# Outputs: x_incumbent - vector with entries in {0, 1} and cardinality n representing the incumbent solution at the last iteration
#          LB_incumbent - objective value of incumbent solution, provides a lower bound on optimal objective
#

function stochastic_threshold_greedy_algorithm(D::Array{Float64,2}, c::Float64, n::Float64, p::Float64)

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
  return ind_incumbent, obj_incumbent
end

function stochastic_threshold_greedy_algorithm_trajectory(D::Array{Float64,2}, c::Float64, n::Float64, p::Float64)
  W, L = size(D)
  s = convert(Int64, round((L/n)*p))
  random_ind_set = Vector{Int64}(undef, s)
  ind_compl_incumbent = [i for i in 1:L]
  ind_incumbent = zeros(Int64, 0)
  Dx_incumbent = zeros(Float64, W)
  y_incumbent = zeros(Float64, W)
  obj_incumbent = 0
  obj_incumbent_vec = zeros(Float64, convert(Int64,n))
  Dx_tmp = Vector{Float64}(undef, W)
  y_tmp = Vector{Float64}(undef, W)
  locations_added, threshold = 0, 0
  @inbounds while locations_added < n
    if locations_added < c
      threshold = locations_added + 1
      obj_candidate = 0
    else
      obj_candidate = obj_incumbent
    end
    ind_candidate_list = Vector{Int64}(undef, 0)
    random_ind_set .= sample(ind_compl_incumbent, s, replace=false)
    @inbounds for ind in random_ind_set
        Dx_tmp .= Dx_incumbent .+ view(D, :, ind)
        y_tmp .= Dx_tmp .>= threshold
        obj_tmp = sum(y_tmp)
        if obj_tmp > obj_candidate
          ind_candidate_list = [ind]
          obj_candidate = obj_tmp
        elseif obj_tmp == obj_candidate
          ind_candidate_list = union(ind, ind_candidate_list)
        end
    end
    ind_candidate = sample(ind_candidate_list)
    ind_incumbent = union(ind_incumbent, ind_candidate)
    ind_compl_incumbent = setdiff(ind_compl_incumbent, ind_candidate)
    Dx_incumbent .= Dx_incumbent .+ view(D, :, ind_candidate)
    y_incumbent .= Dx_incumbent .>= threshold
    obj_incumbent = sum(y_incumbent)
    obj_incumbent_vec[locations_added+1] = obj_incumbent
    locations_added += 1
  end
  return ind_incumbent, obj_incumbent

end

#################### Simulated Annealing Local Search #######################

# Description: function implementing a simulated annealing-inspired local search for unpartitioned geographical regions
#
# Comments: 1) types of inputs should match those declared in argument list
#
# Inputs: D - criticality matrix with entries in {0, 1}, where rows represent time windows and columns represent locations
#         c - global criticality threshold
#         n - number of sites to deploy
#         N - number of locations to swap in order to obtain a neighbour of the incumbent solution
#         I - number of iterations (outer loop), defines the number of times the incumbent solution may be updated
#         E - number of epochs (inner loop), defines the number of neighbours of the incumbent solution sampled at each iteration
#         x_init - initial solution, vector with entries in {0, 1}, with cardinality n and whose dimension is compatible with D
#         T_init - initial temperature from which the (exponentially-decreasing) temperature schedule is constructed
#
# Outputs: x_incumbent - vector with entries in {0, 1} and cardinality n representing the incumbent solution at the last iteration
#          LB_incumbent - objective value of incumbent solution, provides a lower bound on optimal objective
#          obj - vector storing the incumbent objective value at each iteration
#

function simulated_annealing_local_search(D::Array{Float64, 2}, c::Float64, n::Float64, N::Int64, I::Int64, E::Int64, x_init::Array{Float64, 1}, T_init::Float64, legacy_locations::Vector{Int64})

  W, L = size(D)
  n = convert(Int64, n)
  n_new = n - length(legacy_locations)

  # Pre-allocate lower bound vector
  obj = Vector{Int64}(undef, I)

  # Pre-allocate x-related arrays
  ind_ones_incumbent = Vector{Int64}(undef, n_new)
  ind_ones_incumbent_filtered = Vector{Int64}(undef, n_new-N)
  ind_zeros_incumbent = Vector{Int64}(undef, L-n)
  ind_zeros_incumbent_filtered = Vector{Int64}(undef, L-n-N)
  ind_ones2zeros_candidate = Vector{Int64}(undef, N)
  ind_zeros2ones_candidate = Vector{Int64}(undef, N)
  ind_ones2zeros_tmp = Vector{Int64}(undef, N)
  ind_zeros2ones_tmp = Vector{Int64}(undef, N)

  # Pre-allocate y-related arrays
  y_incumbent = Array{Bool}(undef, W, 1)
  y_tmp = Array{Bool}(undef, W, 1)

  Dx_incumbent = Array{Float64}(undef, W, 1)
  Dx_tmp = Array{Float64}(undef, W, 1)

  # Initialise
  ind_deployed = findall(x_init .== 1.)
  Dx_incumbent .= sum(view(D, :, ind_deployed), dims = 2)
  y_incumbent .= Dx_incumbent .>= c
  ind_ones_incumbent .= filter(a -> !(a in legacy_locations), ind_deployed)
  ind_zeros_incumbent .= findall(x_init .== 0.)

  # Iterate
  @inbounds for i = 1:I
    obj[i] = sum(y_incumbent)
    delta_candidate = -100000
    @inbounds for e = 1:E
      # Sample from neighbourhood
      sample!(ind_ones_incumbent, ind_ones2zeros_tmp, replace=false)
      sample!(ind_zeros_incumbent, ind_zeros2ones_tmp, replace=false)

      # Compute y and associated objective value
      @inbounds for j = 1:N
        Dx_tmp .= Dx_incumbent .+ view(D, :, ind_zeros2ones_tmp[j]) .- view(D, :, ind_ones2zeros_tmp[j])
      end
      y_tmp .= Dx_tmp .>= c

      # Update objective difference
      delta_tmp = sum(y_tmp) - obj[i]

      # Update candidate solution
      if delta_tmp > delta_candidate
        ind_ones2zeros_candidate .= ind_ones2zeros_tmp
        ind_zeros2ones_candidate .= ind_zeros2ones_tmp
        delta_candidate = delta_tmp
      end
    end
    if delta_candidate > 0
      ind_ones_incumbent_filtered .= filter(a -> !(a in ind_ones2zeros_candidate), ind_ones_incumbent)
      ind_ones_incumbent[1:(n_new-N)] .= ind_ones_incumbent_filtered
      ind_ones_incumbent[(n_new-N+1):n_new] .= ind_zeros2ones_candidate
      ind_zeros_incumbent_filtered .= filter(a -> !(a in ind_zeros2ones_candidate), ind_zeros_incumbent)
      ind_zeros_incumbent[1:(L-n-N)] .= ind_zeros_incumbent_filtered
      ind_zeros_incumbent[(L-n-N+1):(L-n)] .= ind_ones2zeros_candidate
      @inbounds for j = 1:N
        Dx_incumbent .= Dx_incumbent .+ view(D, :, ind_zeros2ones_candidate[j]) .- view(D, :, ind_ones2zeros_candidate[j])
      end
      y_incumbent .= Dx_incumbent .>= c
    else
      T = T_init * exp(-10*i/I)
      p = exp(delta_candidate / T)
      d = Binomial(1, p)
      b = rand(d)
      if b == 1
        ind_ones_incumbent_filtered .= filter(a -> !(a in ind_ones2zeros_candidate), ind_ones_incumbent)
        ind_ones_incumbent[1:(n_new-N)] .= ind_ones_incumbent_filtered
        ind_ones_incumbent[(n_new-N+1):n_new] .= ind_zeros2ones_candidate
        ind_zeros_incumbent_filtered .= filter(a -> !(a in ind_zeros2ones_candidate), ind_zeros_incumbent)
        ind_zeros_incumbent[1:(L-n-N)] .= ind_zeros_incumbent_filtered
        ind_zeros_incumbent[(L-n-N+1):(L-n)] .= ind_ones2zeros_candidate
        @inbounds for j = 1:N
          Dx_incumbent .= Dx_incumbent .+ view(D, :, ind_zeros2ones_candidate[j]) .- view(D, :, ind_ones2zeros_candidate[j])
        end
        y_incumbent .= Dx_incumbent .>= c
      end
    end
  end
  obj_incumbent = sum(y_incumbent)
  return ind_ones_incumbent, obj_incumbent, obj
end