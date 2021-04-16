using StatsBase
using Distributions

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

function simulated_annealing_local_search_partition(D::Array{Float64, 2}, c::Float64, n::Vector{Int64}, N::Int64, I::Int64, E::Int64, T_init::Float64, x_init::Array{Float64, 1}, locations_regions_mapping::Dict{Int64, Int64}, legacy_locations::Vector{Int64})

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
  legacy_locations_count_per_region = zeros(Int64, P)
  index_range_per_region = Vector{Int64}(undef, P+1)

  @inbounds for i = 1:L
    if i in legacy_locations
      legacy_locations_count_per_region[locations_regions_mapping[i]] += 1
    end
    locations_count_per_region[locations_regions_mapping[i]] += 1
  end

  ind_ones_incumbent = Dict([(r, Vector{Int64}(undef, n[r]-legacy_locations_count_per_region[r])) for r in regions])
  ind_zeros_incumbent = Dict([(r, Vector{Int64}(undef, locations_count_per_region[r]-n[r])) for r in regions])

  index_range_per_region[1] = 1
  @inbounds for j = 1:P
    index_range_per_region[j+1] = index_range_per_region[j] + locations_count_per_region[j]
  end

  # Pre-allocate y-related arrays
  y_incumbent = Array{Bool}(undef, W, 1)
  y_tmp = Array{Bool}(undef, W, 1)

  Dx_incumbent = zeros(Float64, W, 1)
  Dx_tmp = Array{Float64}(undef, W, 1)

  # Initialise
  ind_ones, counter_ones = findall(x_init .== 1.), zeros(Int64, P)
  Dx_incumbent .= sum(view(D, :, ind_ones), dims=2)[:,1]
  filter!(a -> !(a in legacy_locations), ind_ones)
  @inbounds for ind in ind_ones
    p = locations_regions_mapping[ind]
    counter_ones[p] += 1
    ind_ones_incumbent[p][counter_ones[p]] = ind
  end
  y_incumbent .= Dx_incumbent .>= c

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
        if (sample_count_per_region[p] < n[p] - legacy_locations_count_per_region[p]) && (sample_count_per_region[p] < locations_count_per_region[p] - n[p] + legacy_locations_count_per_region[p])
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
      y_tmp .= Dx_tmp .>= c

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
      y_incumbent .= Dx_incumbent .>= c
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
        y_incumbent .= Dx_incumbent .>= c
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

#################### Randomised Threshold Greedy Heuristic with Partitioning Constraints #######################

# Description: function implementing a randomised greedy heuristic for geographical regions partitioned into a set of subregions
#
# Comments: 1) types of inputs should match those declared in argument list
#           2) the implementation relies both on dict and array data structures (as opposed to an array-only implementation)
#
# Inputs: D - criticality matrix with entries in {0, 1}, where rows represent time windows and columns represent locations
#         c - global criticality threshold
#         n - number of sites to deploy
#         p - the proportion of locations that should be sampled at each iteration
#         locations_regions_mapping - dictionary associating its subregion (value) to each location (key)
#         legacy_locations - array storing the indices of existing sites
#
#
# Outputs: ind_incumbent - vector of cardinality storing the indices of the n locations selected by the algorithm
#          obj_incumbent - objective value of incumbent solution, provides a lower bound on optimal objective
#

function randomised_threshold_greedy_heuristic_partition(D::Array{Float64,2}, c::Float64, n::Vector{Int64}, p::Float64,
                                                         locations_regions_mapping::Dict{Int64, Int64},
                                                         legacy_locations::Vector{Int64})

  W, L = size(D)
  s = convert(Int64, round(L*p))
  Dx_incumbent = sum(D[:, legacy_locations], dims=2)[:,1]
  y_incumbent = zeros(Int64, W)
  obj_incumbent = 0
  Dx_tmp = Vector{Float64}(undef, W)
  y_tmp = Vector{Float64}(undef, W)

  P = length(n)
  regions = [i for i = 1:P]
  sample_count_per_region = Vector{Int64}(undef, P)
  init_sample_count_per_region = zeros(Int64, P)
  locations_count_per_region, locations_added_per_region = zeros(Int64, P), zeros(Int64, P)
  @inbounds for ind = 1:L
    if ind in legacy_locations
      locations_added_per_region[locations_regions_mapping[ind]] += 1
    else
      locations_count_per_region[locations_regions_mapping[ind]] += 1
    end
  end

  ind_incumbent = legacy_locations
  ind_ones = [i for i in 1:L]
  filter!(a -> !(a in legacy_locations), ind_ones)
  ind_compl_incumbent = Dict([(r, Vector{Int64}(undef, locations_count_per_region[r])) for r in regions])
  regions_start_pointer = 1
  @inbounds for r in regions
    regions_end_pointer = regions_start_pointer + locations_count_per_region[r]
    ind_compl_incumbent[r] = ind_ones[regions_start_pointer:(regions_end_pointer-1)]
    regions_start_pointer = regions_end_pointer
  end

  @inbounds while sum(locations_added_per_region) < sum(n)
    if sum(locations_added_per_region) < c
      threshold = sum(locations_added_per_region) + 1
      obj_candidate = 0
    else
      threshold = c
      obj_candidate = obj_incumbent
    end

    iter_count = 0
    sample_count_per_region .= init_sample_count_per_region
    @inbounds while sum(sample_count_per_region) < s && iter_count < 10 * s
      r = sample(regions)
      if locations_added_per_region[r] < n[r] && sample_count_per_region[r] < length(ind_compl_incumbent[r])
        sample_count_per_region[r] += 1
      end
      iter_count += 1
    end

    random_ind_set = Vector{Int64}(undef, 0)
    @inbounds for r in regions
      random_ind_set = union(random_ind_set, sample(ind_compl_incumbent[r], sample_count_per_region[r], replace=false))
    end

    ind_candidate_list = Vector{Int64}(undef, 0)
    @inbounds for ind in random_ind_set
      Dx_tmp .= Dx_incumbent .+ view(D, :, ind)
      y_tmp .= Dx_tmp .>= threshold
      obj_tmp = sum(y_tmp)
      if obj_tmp >= obj_candidate
        if obj_tmp > obj_candidate
          ind_candidate_list = [ind]
          obj_candidate = obj_tmp
        else
          ind_candidate_list = union(ind_candidate_list, ind)
        end
      end
    end
    ind_candidate = sample(ind_candidate_list)
    ind_compl_incumbent[locations_regions_mapping[ind_candidate]] =
                                   setdiff(ind_compl_incumbent[locations_regions_mapping[ind_candidate]], ind_candidate)
    ind_incumbent = union(ind_incumbent, ind_candidate)
    locations_added_per_region[locations_regions_mapping[ind_candidate]] += 1
    Dx_incumbent .= Dx_incumbent .+ view(D, :, ind_candidate)
    y_incumbent .= Dx_incumbent .>= c
    obj_incumbent = sum(y_incumbent)
  end

  x_incumbent = zeros(Float64, L)
  x_incumbent[ind_incumbent] .= 1.

  return x_incumbent, obj_incumbent

end
