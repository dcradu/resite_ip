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
  R = maximum(values(locations_regions_mapping))

  iterations = [i for i in 1:I]
  epochs = [e for e in 1:E]
  component_updates = [j for j in 1:N]

  # Pre-allocate lower bound vector
  obj = Vector{Int64}(undef, I)

  # Pre-allocate x-related containers
  locations = [l for l in 1:L]
  x_incumbent = zeros(Float64, L)
  ind_ones2zeros_candidate = Vector{Int64}(undef, N)
  ind_zeros2ones_candidate = Vector{Int64}(undef, N)
  ind_ones2zeros_tmp = Vector{Int64}(undef, N)
  ind_zeros2ones_tmp = Vector{Int64}(undef, N)

  regions = [r for r in 1:R]
  sample_count_per_region_tmp = Vector{Int64}(undef, R)
  sample_count_per_region_candidate = Vector{Int64}(undef, R)
  init_sample_count_per_region = zeros(Int64, R)
  ind_samples_per_region_tmp = Vector{Int64}(undef, R+1)
  ind_samples_per_region_candidate = Vector{Int64}(undef, R+1)
  index_range_per_region = Vector{Int64}(undef, R+1)
  locations_count_per_region = zeros(Int64, R)
  legacy_locations_count_per_region = zeros(Int64, R)

  @inbounds for l in locations
    if l in legacy_locations
      legacy_locations_count_per_region[locations_regions_mapping[l]] += 1
    end
    locations_count_per_region[locations_regions_mapping[l]] += 1
  end

  ind_ones_incumbent = Dict([(r, Vector{Int64}(undef, n[r]-legacy_locations_count_per_region[r])) for r in regions])
  ind_ones_incumbent_filtered = Dict([(r, Vector{Int64}(undef, n[r]-legacy_locations_count_per_region[r])) for r in regions])
  ind_zeros_incumbent = Dict([(r, Vector{Int64}(undef, locations_count_per_region[r]-n[r])) for r in regions])
  ind_zeros_incumbent_filtered = Dict([(r, Vector{Int64}(undef, locations_count_per_region[r]-n[r])) for r in regions])

  index_range_per_region[1] = 1
  @inbounds for r in regions
    index_range_per_region[r+1] = index_range_per_region[r] + locations_count_per_region[r]
  end

  # Pre-allocate y-related arrays
  y_incumbent = Vector{Float64}(undef, W)
  y_tmp = Vector{Float64}(undef, W)

  Dx_incumbent = zeros(Float64, W)
  Dx_tmp = Vector{Float64}(undef, W)

  # Initialise Dx_incumbent and y_incumbent
  ind_ones = findall(x_init .== 1.)
  @inbounds for ind in ind_ones
    Dx_incumbent .= Dx_incumbent .+ view(D, :, ind)
  end
  y_incumbent .= Dx_incumbent .>= c

  filter!(a -> !(a in legacy_locations), ind_ones)
  ind_ones_incumbent_pointer = zeros(Int64, R)
  @inbounds for ind in ind_ones
    r = locations_regions_mapping[ind]
    ind_ones_incumbent_pointer[r] += 1
    ind_ones_incumbent[r][ind_ones_incumbent_pointer[r]] = ind
  end

  ind_zeros = findall(x_init .== 0.)
  ind_zeros_incumbent_pointer = zeros(Int64, R)
  for ind in ind_zeros
    r = locations_regions_mapping[ind]
    ind_zeros_incumbent_pointer[r] += 1
    ind_zeros_incumbent[r][ind_zeros_incumbent_pointer[r]] = ind
  end

  # Iterate
  @inbounds for i in iterations
    obj[i] = sum(y_incumbent)
    delta_candidate = -1000000
    @inbounds for e in epochs
      # Sample from neighbourhood
      sample_count = 0
      sample_count_per_region_tmp .= init_sample_count_per_region
      @inbounds while sample_count < N
        r = sample(regions)
        if (sample_count_per_region_tmp[r] < n[r] - legacy_locations_count_per_region[r]) && (sample_count_per_region_tmp[r] < locations_count_per_region[r] - n[r])
          sample_count_per_region_tmp[r] += 1
          sample_count += 1
        end
      end

      ind_samples_per_region_tmp[1] = 1
      @inbounds for r in regions
        ind_samples_per_region_tmp[r+1] = ind_samples_per_region_tmp[r] + sample_count_per_region_tmp[r]
        if sample_count_per_region_tmp[r] != 0
          ind_ones2zeros_tmp[ind_samples_per_region_tmp[r]:(ind_samples_per_region_tmp[r+1]-1)] .= sample(ind_ones_incumbent[r], sample_count_per_region_tmp[r], replace=false)
          ind_zeros2ones_tmp[ind_samples_per_region_tmp[r]:(ind_samples_per_region_tmp[r+1]-1)] .= sample(ind_zeros_incumbent[r], sample_count_per_region_tmp[r], replace=false)
        end
      end

      # Compute y and associated objective value
      Dx_tmp .= Dx_incumbent
      @inbounds for j in component_updates
        Dx_tmp .+= view(D, :, ind_zeros2ones_tmp[j])
        Dx_tmp .-= view(D, :, ind_ones2zeros_tmp[j])
      end
      y_tmp .= Dx_tmp .>= c

      # Update objective difference
      delta_tmp = sum(y_tmp) - obj[i]

      # Update candidate solution
      if delta_tmp > delta_candidate
        ind_ones2zeros_candidate .= ind_ones2zeros_tmp
        ind_zeros2ones_candidate .= ind_zeros2ones_tmp
        ind_samples_per_region_candidate .= ind_samples_per_region_tmp
        sample_count_per_region_candidate .= sample_count_per_region_tmp
        delta_candidate = delta_tmp
      end
    end
    if delta_candidate > 0
      @inbounds for r in regions
        if sample_count_per_region_candidate[r] != 0
          view(ind_ones_incumbent_filtered[r], 1:(n[r]-legacy_locations_count_per_region[r]-sample_count_per_region_candidate[r])) .= filter(a -> !(a in view(ind_ones2zeros_candidate, ind_samples_per_region_candidate[r]:(ind_samples_per_region_candidate[r+1]-1))), ind_ones_incumbent[r])
          view(ind_ones_incumbent[r], 1:(n[r]-legacy_locations_count_per_region[r]-sample_count_per_region_candidate[r])) .= view(ind_ones_incumbent_filtered[r], 1:(n[r]-legacy_locations_count_per_region[r]-sample_count_per_region_candidate[r]))
          view(ind_ones_incumbent[r], (n[r]-legacy_locations_count_per_region[r]-sample_count_per_region_candidate[r]+1):(n[r]-legacy_locations_count_per_region[r])) .= view(ind_zeros2ones_candidate, ind_samples_per_region_candidate[r]:(ind_samples_per_region_candidate[r+1]-1))
          view(ind_zeros_incumbent_filtered[r], 1:(locations_count_per_region[r]-n[r]-sample_count_per_region_candidate[r])) .= filter(a -> !(a in view(ind_zeros2ones_candidate, ind_samples_per_region_candidate[r]:(ind_samples_per_region_candidate[r+1]-1))), ind_zeros_incumbent[r])
          view(ind_zeros_incumbent[r], 1:(locations_count_per_region[r]-n[r]-sample_count_per_region_candidate[r])) .= view(ind_zeros_incumbent_filtered[r], 1:(locations_count_per_region[r]-n[r]-sample_count_per_region_candidate[r]))
          view(ind_zeros_incumbent[r], (locations_count_per_region[r]-n[r]-sample_count_per_region_candidate[r]+1):(locations_count_per_region[r]-n[r])) .= view(ind_ones2zeros_candidate, ind_samples_per_region_candidate[r]:(ind_samples_per_region_candidate[r+1]-1))
        end
      end
      @inbounds for j in component_updates
        Dx_incumbent .+= view(D, :, ind_zeros2ones_candidate[j])
        Dx_incumbent .-= view(D, :, ind_ones2zeros_candidate[j])
      end
      y_incumbent .= Dx_incumbent .>= c
    else
      T = T_init * exp(-10*i/I)
      p = exp(delta_candidate / T)
      d = Binomial(1, p)
      b = rand(d)
      if b == 1
        @inbounds for r in regions
          if sample_count_per_region_candidate[r] != 0
            view(ind_ones_incumbent_filtered[r], 1:(n[r]-legacy_locations_count_per_region[r]-sample_count_per_region_candidate[r])) .= filter(a -> !(a in view(ind_ones2zeros_candidate, ind_samples_per_region_candidate[r]:(ind_samples_per_region_candidate[r+1]-1))), ind_ones_incumbent[r])
            view(ind_ones_incumbent[r], 1:(n[r]-legacy_locations_count_per_region[r]-sample_count_per_region_candidate[r])) .= view(ind_ones_incumbent_filtered[r], 1:(n[r]-legacy_locations_count_per_region[r]-sample_count_per_region_candidate[r]))
            view(ind_ones_incumbent[r], (n[r]-legacy_locations_count_per_region[r]-sample_count_per_region_candidate[r]+1):(n[r]-legacy_locations_count_per_region[r])) .= view(ind_zeros2ones_candidate, ind_samples_per_region_candidate[r]:(ind_samples_per_region_candidate[r+1]-1))
            view(ind_zeros_incumbent_filtered[r], 1:(locations_count_per_region[r]-n[r]-sample_count_per_region_candidate[r])) .= filter(a -> !(a in view(ind_zeros2ones_candidate, ind_samples_per_region_candidate[r]:(ind_samples_per_region_candidate[r+1]-1))), ind_zeros_incumbent[r])
            view(ind_zeros_incumbent[r], 1:(locations_count_per_region[r]-n[r]-sample_count_per_region_candidate[r])) .= view(ind_zeros_incumbent_filtered[r], 1:(locations_count_per_region[r]-n[r]-sample_count_per_region_candidate[r]))
            view(ind_zeros_incumbent[r], (locations_count_per_region[r]-n[r]-sample_count_per_region_candidate[r]+1):(locations_count_per_region[r]-n[r])) .= view(ind_ones2zeros_candidate, ind_samples_per_region_candidate[r]:(ind_samples_per_region_candidate[r+1]-1))
          end
        end
        @inbounds for j in component_updates
          Dx_incumbent .+= view(D, :, ind_zeros2ones_candidate[j])
          Dx_incumbent .-= view(D, :, ind_ones2zeros_candidate[j])
        end
        y_incumbent .= Dx_incumbent .>= c
      end
     end
  end
  x_incumbent[legacy_locations] .= 1.
  @inbounds for r in regions
    x_incumbent[ind_ones_incumbent[r]] .= 1.
    x_incumbent[ind_zeros_incumbent[r]] .= 0.
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

function randomised_greedy_heuristic_partition(D::Array{Float64,2}, c::Float64, n::Vector{Int64}, p::Float64, locations_regions_mapping::Dict{Int64, Int64}, legacy_locations::Vector{Int64})

  W, L = size(D)
  n_total = sum(n)
  s = convert(Int64, round(L*p))
  random_ind_set = Vector{Int64}(undef, s)
  Dx_incumbent = zeros(Float64, W)
  @inbounds for ind in legacy_locations
    Dx_incumbent .= Dx_incumbent .+ view(D, :, ind)
  end
  x_incumbent = zeros(Float64, L)
  y_incumbent = Vector{Float64}(undef, W)
  obj_incumbent = 0
  Dx_tmp = Vector{Float64}(undef, W)
  y_tmp = Vector{Float64}(undef, W)

  R = length(n)
  regions = [r for r = 1:R]
  locations = [l for l = 1:L]
  sample_count_per_region = Vector{Int64}(undef, R)
  init_sample_count_per_region = zeros(Int64, R)
  candidate_locations_count_per_region, locations_added_per_region = zeros(Int64, R), zeros(Int64, R)
  @inbounds for ind in locations
    if ind in legacy_locations
      locations_added_per_region[locations_regions_mapping[ind]] += 1
    else
      candidate_locations_count_per_region[locations_regions_mapping[ind]] += 1
    end
  end

  ind_incumbent = Vector{Int64}(undef, n_total)
  ind_incumbent[1:length(legacy_locations)] .= legacy_locations
  ind_candidate_list = Vector{Int64}(undef, s)
  ind_ones = Vector{Int64}(undef, L)
  ind_ones .= locations
  filter!(a -> !(a in legacy_locations), ind_ones)
  ind_compl_incumbent = Dict([(r, Vector{Int64}(undef, candidate_locations_count_per_region[r])) for r in regions])
  regions_start_pointer = 1
  @inbounds for r in regions
    regions_end_pointer = regions_start_pointer + candidate_locations_count_per_region[r]
    ind_compl_incumbent[r] .= ind_ones[regions_start_pointer:(regions_end_pointer-1)]
    regions_start_pointer = regions_end_pointer
  end

  locations_added = sum(locations_added_per_region)
  @inbounds while locations_added < n_total
    if locations_added < c
      threshold = locations_added + 1
      obj_candidate = 0
    else
      threshold = c
      obj_candidate = obj_incumbent
    end

    iter_count, sample_count = 0, 0
    sample_count_per_region .= init_sample_count_per_region
    @inbounds while sample_count < s && iter_count < 10 * s
      r = sample(regions)
      if locations_added_per_region[r] < n[r] && sample_count_per_region[r] < length(ind_compl_incumbent[r])
        sample_count_per_region[r] += 1
        sample_count += 1
      end
      iter_count += 1
    end

    sample_start_pointer, sample_end_pointer = 1, 0
    @inbounds for r in regions
      if sample_count_per_region[r] != 0
        sample_end_pointer = sample_start_pointer + sample_count_per_region[r]
        random_ind_set[sample_start_pointer:(sample_end_pointer-1)] .= sample(ind_compl_incumbent[r], sample_count_per_region[r], replace=false)
        sample_start_pointer = sample_end_pointer
      end
    end

    ind_candidate_pointer = 1
    @inbounds for ind in view(random_ind_set, 1:(sample_end_pointer-1))
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
    locations_added_per_region[locations_regions_mapping[ind_candidate]] += 1
    locations_added = sum(locations_added_per_region)
    ind_incumbent[locations_added] = ind_candidate
    Dx_incumbent .= Dx_incumbent .+ view(D, :, ind_candidate)
    y_incumbent .= Dx_incumbent .>= threshold
    obj_incumbent = sum(y_incumbent)
    filter!(a -> a != ind_candidate, ind_compl_incumbent[locations_regions_mapping[ind_candidate]])
  end

  x_incumbent[ind_incumbent] .= 1.
  return x_incumbent, obj_incumbent
end

#################### Threshold Greedy Heuristic with Partitioning Constraints #######################

# Description: function implementing a greedy heuristic for geographical regions partitioned into a set of subregions
#
# Comments: 1) types of inputs should match those declared in argument list
#           2) the implementation relies both on dict and array data structures (as opposed to an array-only implementation)
#
# Inputs: D - criticality matrix with entries in {0, 1}, where rows represent time windows and columns represent locations
#         c - global criticality threshold
#         n - number of sites to deploy
#         locations_regions_mapping - dictionary associating its subregion (value) to each location (key)
#         legacy_locations - array storing the indices of existing sites
#
#
# Outputs: ind_incumbent - vector of cardinality storing the indices of the n locations selected by the algorithm
#          obj_incumbent - objective value of incumbent solution, provides a lower bound on optimal objective
#

function greedy_heuristic_partition(D::Array{Float64,2}, c::Float64, n::Vector{Int64}, locations_regions_mapping::Dict{Int64, Int64}, legacy_locations::Vector{Int64})

  W, L = size(D)
  n_total = sum(n)
  Dx_incumbent = zeros(Float64, W)
  @inbounds for ind in legacy_locations
    Dx_incumbent .= Dx_incumbent .+ view(D, :, ind)
  end
  x_incumbent = zeros(Float64, L)
  y_incumbent = Vector{Float64}(undef, W)
  obj_incumbent = 0
  Dx_tmp = Vector{Float64}(undef, W)
  y_tmp = Vector{Float64}(undef, W)

  R = length(n)
  regions = [r for r = 1:R]
  locations = [l for l = 1:L]
  candidate_locations_count_per_region, locations_added_per_region = zeros(Int64, R), zeros(Int64, R)
  @inbounds for ind in locations
    if ind in legacy_locations
      locations_added_per_region[locations_regions_mapping[ind]] += 1
    else
      candidate_locations_count_per_region[locations_regions_mapping[ind]] += 1
    end
  end

  ind_incumbent = Vector{Int64}(undef, n_total)
  ind_incumbent[1:length(legacy_locations)] .= legacy_locations
  ind_candidate_list = Vector{Int64}(undef, L)
  ind_ones = Vector{Int64}(undef, L)
  ind_ones .= locations
  filter!(a -> !(a in legacy_locations), ind_ones)
  ind_compl_incumbent = Dict([(r, Vector{Int64}(undef, candidate_locations_count_per_region[r])) for r in regions])
  regions_start_pointer = 1
  @inbounds for r in regions
    regions_end_pointer = regions_start_pointer + candidate_locations_count_per_region[r]
    ind_compl_incumbent[r] .= ind_ones[regions_start_pointer:(regions_end_pointer-1)]
    regions_start_pointer = regions_end_pointer
  end

  locations_added = sum(locations_added_per_region)
  @inbounds while locations_added < n_total
    if locations_added < c
      threshold = locations_added + 1
      obj_candidate = 0
    else
      threshold = c
      obj_candidate = obj_incumbent
    end
    ind_candidate_pointer = 1
    @inbounds for r in regions
        if locations_added_per_region[r] < n[r]
          @inbounds for ind in ind_compl_incumbent[r]
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
        end
    end
    ind_candidate = sample(view(ind_candidate_list, 1:ind_candidate_pointer-1))
    filter!(a -> a != ind_candidate, ind_compl_incumbent[locations_regions_mapping[ind_candidate]])
    ind_incumbent[locations_added+1] = ind_candidate
    Dx_incumbent .= Dx_incumbent .+ view(D, :, ind_candidate)
    y_incumbent .= Dx_incumbent .>= c
    obj_incumbent = sum(y_incumbent)
    locations_added_per_region[locations_regions_mapping[ind_candidate]] += 1
    locations_added += 1
  end

  x_incumbent[ind_incumbent] .= 1.
  return x_incumbent, obj_incumbent
end
