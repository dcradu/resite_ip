using BandedMatrices
using Statistics

### Objective type definitions ###

abstract type Objective end

struct ElectricityOutput <: Objective end

struct MaxVariability <: Objective end

struct ElectricityOutput <: Objective end

struct ElectricityOutput <: Objective end

struct AverageVariability <: Objective end

struct MaxResidualDemand <: Objective end

struct AverageResidualDemand <: Objective end

struct Criticality <: Objective
    c::Float64
end

struct Correlation <: Objective end

### PREPROCESSING ###

function compute_criticality_matrix(A::Array{Float64, 2}, demand::Vector{Float64}, potential::Vector{Float64}, N::Int64, varsigma::Float64)::Array{Float64, 2}

    T::Int64, L::Int64 = size(A)
    D::Array{Float64, 2} = zeros(Float64, T, L)
    alpha_tl::Float64 = 0.0
    t::Int64, l::Int64 = 1, 1
    @inbounds while l <= L
        t = 1
        @inbounds while t <= T
            alpha_tl = (varsigma * demand[t]) / (potential[l] * N)
            if A[t, l] >= alpha_tl
                D[t, l] = 1.0
            end
            t += 1
        end
        l += 1
    end
    return D
end

function time_compute_criticality_matrix(A::Array{Float64, 2}, demand::Vector{Float64}, potential::Vector{Float64}, N::Int64, varsigma::Float64)
    @time compute_criticality_matrix(A, demand, potential, N, varsigma)
end

function compute_correlation_matrix(A::Array{Float64, 2})::Array{Float64, 2}

    return cor(A, dims=1)

end

function time_compute_correlation_matrix(A::Array{Float64 ,2})
    @time compute_correlation_matrix(A)
end

### OBJECTIVES ###

function compute_objective(A::Array{Float64}, L::Vector{Int64}, obj::ElectricityOutput)::Float64

    p::Float64, T::Int64 = 0.0, size(A)[1]
    @inbounds for l in L
        t::Int64 = 1
        @inbounds while t <= T
            p += A[t, l]
            t += 1
        end
    end
    return p

end

function time_compute_objective(A::Array{Float64}, L::Vector{Int64}, obj::ElectricityOutput)
    @time compute_objective(A, L, obj)
end

function compute_objective(A::Array{Float64}, p::Vector{Float64}, L::Int64, obj::ElectricityOutput)::Float64

    T::Int64, p_tmp::Float64 = length(p), 0.0
    t::Int64 = 1
    @inbounds while t <= T
        p_tmp += (p[t] + A[t, L])
        t += 1
    end
    return p_tmp

end

function time_compute_objective(A::Array{Float64}, L::Int64, obj::ElectricityOutput)
    @time compute_objective(A, L, obj)
end

function compute_objective(A::Array{Float64}, d::Vector{Float64}, L::Vector{Int64}, obj::MaxVariability)::Float64

    var::Float64, var_max::Float64 = 0.0, -1e9
    p_tmp::Float64, p_tmp1::Float64, rd_t::Float64, rd_tm1::Float64 = 0.0, sum(A[1, l] for l in L), 0.0, 0.0
    t::Int64 = 2
    @inbounds while t <= length(d)
        p_tmp, rd_t, rd_tm1 = 0.0, 0.0, 0.0
        @inbounds for l in L
            p_tmp += A[t, l]
        end
        if p_tmp < d[t]
            rd_t = (d[t] - p_tmp)
        end
        if p_tmp1 < d[t-1]
            rd_tm1 = (d[t-1] - p_tmp1)
        end
        var = rd_t - rd_tm1
        if var > var_max
            var_max = var
        end
        t += 1
        p_tmp1 = p_tmp
    end
    return var_max

end

function time_compute_objective(A::Array{Float64}, d::Vector{Float64}, L::Vector{Int64}, obj::MaxVariability)
    @time compute_objective(A, d, L, obj)
end

function compute_objective(A::Array{Float64}, p::Vector{Float64}, d::Vector{Float64}, l::Int64, obj::MaxVariability)::Float64

    var::Float64, var_max::Float64 = 0.0, -1e9
    p_tmp::Float64, p_tmp1::Float64, rd_t::Float64, rd_tm1::Float64 = 0.0, p[1] + A[1, l], 0.0, 0.0
    t::Int64 = 2
    @inbounds while t <= length(d)
        p_tmp, rd_t, rd_tm1 = p[t] + A[t, l], 0.0, 0.0
        if p_tmp < d[t]
            rd_t = (d[t] - p_tmp)
        end
        if p_tmp1 < d[t-1]
            rd_tm1 = (d[t-1] - p_tmp1)
        end
        var = rd_t - rd_tm1
        if var > var_max
            var_max = var
        end
        t += 1
        p_tmp1 = p_tmp
    end
    return var_max

end

function time_compute_objective(A::Array{Float64}, p::Vector{Float64}, d::Vector{Float64}, l::Int64, obj::MaxVariability)
    @time compute_objective(A, p, d, l, obj)
end

function compute_objective(A::Array{Float64}, d::Vector{Float64}, L::Vector{Int64}, obj::AverageVariability)::Float64

    var::Float64, rd_t::Float64, rd_tm1::Float64 = 0.0, 0.0, 0.0
    p_tmp::Float64, p_tmp1::Float64 = 0.0, 0.0
    @inbounds for l in L
        p_tmp1 += A[1, l]
    end
    t::Int64 = 2
    @inbounds while t <= length(d)
        p_tmp, rd_t, rd_tm1 = 0.0, 0.0, 0.0
        @inbounds for l in L
            p_tmp += A[t, l]
        end
        if p_tmp < d[t]
            rd_t = (d[t] - p_tmp)
        end
        if p_tmp1 < d[t-1]
            rd_tm1 = (d[t-1] - p_tmp1)
        end
        var += abs(rd_t - rd_tm1)
        p_tmp1 = p_tmp
        t += 1
    end
    return var/(length(d)-1)

end

function time_compute_objective(A::Array{Float64}, d::Vector{Float64}, L::Vector{Int64}, obj::AverageVariability)
    @time compute_objective(A, d, L, obj)
end

function compute_objective(A::Array{Float64}, p::Vector{Float64}, d::Vector{Float64}, l::Int64, obj::AverageVariability)::Float64

    var::Float64, rd_t::Float64, rd_tm1::Float64 = 0.0, 0.0, 0.0
    p_tmp::Float64, p_tmp1::Float64 = 0.0, p[1] + A[1, l]
    t::Int64 = 2
    @inbounds while t <= length(d)
        p_tmp, rd_t, rd_tm1 = p[t] + A[t, l], 0.0, 0.0
        if p_tmp < d[t]
            rd_t = (d[t] - p_tmp)
        end
        if p_tmp1 < d[t-1]
            rd_tm1 = (d[t-1] - p_tmp1)
        end
        var += abs(rd_t - rd_tm1)
        p_tmp1 = p_tmp
        t += 1
    end
    return var/(length(d)-1)

end

function time_compute_objective(A::Array{Float64}, p::Vector{Float64}, d::Vector{Float64}, l::Int64, obj::AverageVariability)
    @time compute_objective(A, p, d, l, obj)
end

function compute_objective(A::Array{Float64}, d::Vector{Float64}, L::Vector{Int64}, obj::MaxResidualDemand)::Float64

    p_tmp::Float64, rd::Float64, rd_max::Float64 = 0.0, 0.0, 0.0
    t::Int64 = 1
    @inbounds while t <= length(d)
        p_tmp = 0.0
        @inbounds for l in L
            p_tmp += A[t, l]
        end
        if p_tmp < d[t]
           rd = (d[t] - p_tmp)
           if rd > rd_max
               rd_max = rd
           end
        end
        t += 1
    end
    return rd_max

end

function time_compute_objective(A::Array{Float64}, d::Vector{Float64}, L::Vector{Int64}, obj::MaxResidualDemand)
    @time compute_objective(A, d, L, obj)
end

function compute_objective(A::Array{Float64}, p::Vector{Float64}, d::Vector{Float64}, l::Int64, obj::MaxResidualDemand)::Float64

    p_tmp::Float64, rd::Float64, rd_max::Float64 = 0.0, 0.0, -1e+9
    t::Int64 = 1
    @inbounds while t <= length(d)
        p_tmp = p[t] + A[t, l]
        if p_tmp < d[t]
            rd = (d[t] - p_tmp)
            if rd > rd_max
                rd_max = rd
            end
        end
        t += 1
    end
    return rd_max

end

function time_compute_objective(A::Array{Float64}, p::Vector{Float64}, d::Vector{Float64}, l::Int64, obj::MaxResidualDemand)
    @time compute_objective(A, p, d, l, obj)
end

function compute_objective(A::Array{Float64}, d::Vector{Float64}, L::Vector{Int64}, obj::AverageResidualDemand)::Float64

    p_tmp::Float64, rd::Float64 = 0.0, 0.0
    t::Int64 = 1
    @inbounds while t <= length(d)
        p_tmp = 0.0
        @inbounds for l in L
            p_tmp += A[t, l]
        end
        if p_tmp < d[t]
           rd += (d[t] - p_tmp)
        end
        t += 1
    end
    return rd/length(d)

end

function time_compute_objective(A::Array{Float64}, d::Vector{Float64}, L::Vector{Int64}, obj::AverageResidualDemand)
    @time compute_objective(A, d, L, obj)
end

function compute_objective(A::Array{Float64}, p::Vector{Float64}, d::Vector{Float64}, l::Int64, obj::AverageResidualDemand)::Float64

    p_tmp::Float64, rd::Float64 = 0.0, 0.0
    t::Int64 = 1
    @inbounds while t <= length(d)
        p_tmp = p[t] + A[t, l]
        if p_tmp < d[t]
            rd += (d[t] - p_tmp)
        end
        t += 1
    end
    return rd/length(d)

end

function time_compute_objective(A::Array{Float64}, p::Vector{Float64}, d::Vector{Float64}, l::Int64, obj::AverageResidualDemand)
    @time compute_objective(A, p, d, l, obj)
end

function compute_objective(A::Array{Float64}, L::Vector{Int64}, obj::Criticality)::Float64

    d_tmp::Float64, ncrit::Float64 = 0.0, 0.0
    w::Int64 = 1
    @inbounds while w <= size(A)[1]
        d_tmp = 0.0
        @inbounds for l in L
            d_tmp += A[w, l]
        end
        if d_tmp >= obj.c
            ncrit += 1.0
        end
        w += 1
    end
    return ncrit/size(A)[1]

end

function time_compute_objective(A::Array{Float64}, L::Vector{Int64}, obj::Criticality)
    @time compute_objective(A, L, obj)
end

function compute_objective(A::Array{Float64}, d::Vector{Float64}, thres::Float64, l::Int64, obj::Criticality)::Float64

    d_tmp::Float64, ncrit::Float64 = 0.0, 0.0
    w::Int64 = 1
    @inbounds while w <= length(d)
        d_tmp = d[w] + A[w, l]
        if d_tmp >= thres
            ncrit += 1.0
        end
        w += 1
    end
    return ncrit/size(A)[1]

end

function time_compute_objective(A::Array{Float64}, d::Vector{Float64}, l::Int64, obj::Criticality)
    @time compute_objective(A, d, l, obj)
end

function compute_objective(A::Array{Float64}, L::Vector{Int64}, obj::Correlation)

    a_tmp::Float64, corr::Float64 = 0.0, 0.0
    @inbounds for l1 in L
        @inbounds for l2 in L
            if l1 != l2
                corr += A[l1, l2]
            end
        end
    end
    corr = corr / 2.0
    return corr

end

function time_compute_objective(A::Array{Float64}, L::Vector{Int64}, obj::Correlation)
    @time compute_objective(A, L, obj)
end

function compute_objective(A::Array{Float64}, l::Int64, L::Vector{Int64}, obj::Correlation)

    corr::Float64 = 0.0
    @inbounds for loc in L
        corr += A[l, loc]
    end
    return corr

end

function time_compute_objective(A::Array{Float64}, l::Int64, L::Vector{Int64}, obj::Correlation)
    @time compute_objective(A, l, L, obj)
end

############# GREEDY ALGORITHMS ###########

### SINGLE OBJECTIVE ALGORITHMS ###

function greedy_algorithm(A::Array{Float64}, N::Int64, obj::ElectricityOutput)

    T, L = size(A)
    ind_compl_incumbent::Vector{Int64}, ind_incumbent::Vector{Int64}, ind_tmp::Vector{Int64}, ind_candidate::Int64 = [l for l = 1:L], Vector{Int64}(undef, N), zeros(Int64, 0), 0
    p_incumbent::Vector{Float64} = zeros(Float64, T)
    v::Float64, v_max::Float64 = 0.0, 0.0
    n::Int64 = 0
    @inbounds while n < N
        v_max = -1e+9
        @inbounds for ind in ind_compl_incumbent
            v = compute_objective(A, p_incumbent, ind, obj)
            if v > v_max
                filter!(a -> !(a in ind_tmp), ind_tmp)
                push!(ind_tmp, ind)
                v_max = v
            elseif v == v_max
                push!(ind_tmp, ind)
            end
        end
        ind_candidate = rand(ind_tmp)
        p_incumbent .+= view(A, :, ind_candidate)
        ind_incumbent[n+1] = ind_candidate
        filter!(a -> a != ind_candidate, ind_compl_incumbent)
        n += 1
    end
    return ind_incumbent

end

function time_greedy_algorithm(A::Array{Float64}, N::Int64, obj::ElectricityOutput)
    @time greedy_algorithm(A, N, obj)
end

function greedy_algorithm(A::Array{Float64}, d::Vector{Float64}, N::Int64, obj::MaxResidualDemand)

    T, L = size(A)
    ind_compl_incumbent::Vector{Int64}, ind_incumbent::Vector{Int64}, ind_tmp::Vector{Int64}, ind_candidate::Int64 = [l for l = 1:L], zeros(Int64, 0), zeros(Int64, 0), 0
    p_incumbent::Vector{Float64} = zeros(Float64, T)
    v::Float64, v_min::Float64 = 0.0, 0.0
    n::Int64 = 0
    @inbounds while n < N
        v_min = 1e+9
        @inbounds for ind in ind_compl_incumbent
            v = compute_objective(A, p_incumbent, d, ind, obj)
            if v < v_min
                filter!(a -> !(a in ind_tmp), ind_tmp)
                push!(ind_tmp, ind)
                v_min = v
            elseif v == v_min
                push!(ind_tmp, ind)
            end
        end
        ind_candidate = rand(ind_tmp)
        p_incumbent .+= view(A, :, ind_candidate)
        push!(ind_incumbent, ind_candidate)
        filter!(a -> a != ind_candidate, ind_compl_incumbent)
        n += 1
    end
    return ind_incumbent

end


function time_greedy_algorithm(A::Array{Float64}, d::Vector{Float64}, N::Int64, obj::MaxResidualDemand)
    @time greedy_algorithm(A, d, N, obj)
end

function greedy_algorithm(A::Array{Float64}, d::Vector{Float64}, N::Int64, obj::AverageResidualDemand)

    T, L = size(A)
    ind_compl_incumbent::Vector{Int64}, ind_incumbent::Vector{Int64}, ind_tmp::Vector{Int64}, ind_candidate::Int64 = [l for l = 1:L], zeros(Int64, 0), zeros(Int64, 0), 0
    p_incumbent::Vector{Float64} = zeros(Float64, T)
    v::Float64, v_min::Float64 = 0.0, 0.0
    n::Int64 = 0
    @inbounds while n < N
        v_min = 1e+9
        @inbounds for ind in ind_compl_incumbent
            v = compute_objective(A, p_incumbent, d, ind, obj)
            if v < v_min
                filter!(a -> !(a in ind_tmp), ind_tmp)
                push!(ind_tmp, ind)
                v_min = v
            elseif v == v_min
                push!(ind_tmp, ind)
            end
        end
        ind_candidate = rand(ind_tmp)
        p_incumbent .+= view(A, :, ind_candidate)
        push!(ind_incumbent, ind_candidate)
        filter!(a -> a != ind_candidate, ind_compl_incumbent)
        n += 1
    end
    return ind_incumbent

end

function time_greedy_algorithm(A::Array{Float64}, d::Vector{Float64}, N::Int64, obj::AverageResidualDemand)
    @time greedy_algorithm(A, d, N, obj)
end

function greedy_algorithm(A::Array{Float64}, d::Vector{Float64}, N::Int64, obj::MaxVariability)

    T, L = size(A)
    ind_compl_incumbent::Vector{Int64}, ind_incumbent::Vector{Int64}, ind_tmp::Vector{Int64}, ind_candidate::Int64 = [l for l = 1:L], zeros(Int64, 0), zeros(Int64, 0), 0
    p_incumbent::Vector{Float64} = zeros(Float64, T)
    v::Float64, v_min::Float64 = 0.0, 0.0
    n::Int64 = 0
    @inbounds while n < N
        v_min = 1e+9
        @inbounds for ind in ind_compl_incumbent
            v = compute_objective(A, p_incumbent, d, ind, obj)
            if v < v_min
                filter!(a -> !(a in ind_tmp), ind_tmp)
                push!(ind_tmp, ind)
                v_min = v
            elseif v == v_min
                push!(ind_tmp, ind)
            end
        end
        ind_candidate = rand(ind_tmp)
        p_incumbent .+= view(A, :, ind_candidate)
        push!(ind_incumbent, ind_candidate)
        filter!(a -> a != ind_candidate, ind_compl_incumbent)
        n += 1
    end
    return ind_incumbent

end

function time_greedy_algorithm(A::Array{Float64}, d::Vector{Float64}, N::Int64, obj::MaxVariability)
    @time greedy_algorithm(A, d, N, obj)
end

function greedy_algorithm(A::Array{Float64}, d::Vector{Float64}, N::Int64, obj::AverageVariability)

    T, L = size(A)
    ind_compl_incumbent::Vector{Int64}, ind_incumbent::Vector{Int64}, ind_tmp::Vector{Int64}, ind_candidate::Int64 = [l for l = 1:L], zeros(Int64, 0), zeros(Int64, 0), 0
    p_incumbent::Vector{Float64} = zeros(Float64, T)
    v::Float64, v_min::Float64 = 0.0, 0.0
    n::Int64 = 0
    @inbounds while n < N
        v_min = 1e+9
        @inbounds for ind in ind_compl_incumbent
            v = compute_objective(A, p_incumbent, d, ind, obj)
            if v < v_min
                filter!(a -> !(a in ind_tmp), ind_tmp)
                push!(ind_tmp, ind)
                v_min = v
            elseif v == v_min
                push!(ind_tmp, ind)
            end
        end
        ind_candidate = rand(ind_tmp)
        p_incumbent .+= view(A, :, ind_candidate)
        push!(ind_incumbent, ind_candidate)
        filter!(a -> a != ind_candidate, ind_compl_incumbent)
        n += 1
    end
    return ind_incumbent

end

function time_greedy_algorithm(A::Array{Float64}, d::Vector{Float64}, N::Int64, obj::AverageVariability)
    @time greedy_algorithm(A, d, N, obj)
end

function greedy_algorithm(A::Array{Float64}, N::Int64, obj::Criticality)

    W, L = size(A)
    ind_compl_incumbent::Vector{Int64}, ind_incumbent::Vector{Int64}, ind_tmp::Vector{Int64}, ind_candidate::Int64 = [l for l = 1:L], zeros(Int64, 0), zeros(Int64, 0), 0
    d_incumbent::Vector{Float64} = zeros(Float64, W)
    v::Float64, v_max::Float64 = 0.0, 0.0
    n::Int64, threshold::Float64 = 0, 0.0
    @inbounds while n < N
        v_max = -1e+9
        if threshold < obj.c
            threshold += 1.0
        else
            threshold = obj.c
        end
        @inbounds for ind in ind_compl_incumbent
            v = compute_objective(A, d_incumbent, threshold, ind, obj)
            if v > v_max
                filter!(a -> !(a in ind_tmp), ind_tmp)
                push!(ind_tmp, ind)
                v_max = v
            elseif v == v_max
                push!(ind_tmp, ind)
            end
        end
        ind_candidate = rand(ind_tmp)
        d_incumbent .+= view(A, :, ind_candidate)
        push!(ind_incumbent, ind_candidate)
        filter!(a -> a != ind_candidate, ind_compl_incumbent)
        n += 1
    end
    return ind_incumbent

end

function time_greedy_algorithm(A::Array{Float64}, N::Int64, obj::Criticality)
    @time greedy_algorithm(A, N, obj)
end

function greedy_algorithm(A::Array{Float64}, N::Int64, obj::Correlation)

    L = size(A)[1]
    ind_compl_incumbent::Vector{Int64}, ind_incumbent::Vector{Int64}, ind_tmp::Vector{Int64}, ind_candidate::Int64 = [l for l = 1:L], zeros(Int64, 0), zeros(Int64, 0), 0
    v::Float64, v_min::Float64 = 0.0, 0.0
    n::Int64 = 0
    @inbounds while n < N
        v_min = 1e+9
        @inbounds for ind in ind_compl_incumbent
            if n > 1
                v = compute_objective(A, ind, ind_incumbent, obj)
            else
                v = compute_objective(A, ind, ind_compl_incumbent, obj)
            end
            if v < v_min
                filter!(a -> !(a in ind_tmp), ind_tmp)
                push!(ind_tmp, ind)
                v_min = v
            elseif v == v_min
                push!(ind_tmp, ind)
            end
        end
        ind_candidate = rand(ind_tmp)
        push!(ind_incumbent, ind_candidate)
        filter!(a -> a != ind_candidate, ind_compl_incumbent)
        n += 1
    end
    return ind_incumbent

end

function time_greedy_algorithm(A::Array{Float64}, N::Int64, obj::Correlation)
    @time greedy_algorithm(A, N, obj)
end

### MULTI OBJECTIVE ALGORITHMS ###

# TBA

### FUNCTION TO CALL FROM PYTHON ###

function siting_method(capacity_factor_matrix::Array{Float64}, demand::Vector{Float64}, potential::Vector{Float64}, deployment_target::Int64, c::Float64, varsigma::Float64, criterion::String)

    production_matrix::Array{Float64, 2} = potential' .* capacity_factor_matrix
    criticality_matrix::Array{Float64, 2}, correlation_matrix::Array{Float64, 2} = compute_criticality_matrix(capacity_factor_matrix, demand, potential, deployment_target, varsigma), compute_correlation_matrix(capacity_factor_matrix)
    criteria = ["ElectricityOutput", "AverageVariability", "MaxVariability", "AverageResidualDemand", "MaxResidualDemand", "Criticality", "Correlation"]
    obj_types = [ElectricityOutput(), AverageVariability(), MaxVariability(), AverageResidualDemand(), MaxResidualDemand(), Criticality(c), Correlation()]
    obj_mapping = Dict(criteria .=> obj_types)
    locations::Vector{Int64} = zeros(Int64, deployment_target)
    if criterion == "Criticality"
        locations .= greedy_algorithm(criticality_matrix, deployment_target, obj_mapping[criterion])
    elseif criterion == "ElectricityOutput"
        locations .= greedy_algorithm(production_matrix, deployment_target, obj_mapping[criterion])
    elseif criterion == "Correlation"
        locations .= greedy_algorithm(correlation_matrix, deployment_target, obj_mapping[criterion])
    else
        locations .= greedy_algorithm(production_matrix, demand, deployment_target, obj_mapping[criterion])
    end
    x::Vector{Float64}, obj_values::Vector{Float64} = zeros(Float64, length(potential)), zeros(Float64, 0)
    @inbounds for crit in criteria
        if crit == "Criticality"
            push!(obj_values, 1.0-compute_objective(criticality_matrix, locations, obj_mapping[crit]))
        elseif crit == "ElectricityOutput"
            push!(obj_values, compute_objective(production_matrix, locations, obj_mapping[crit]))
        elseif crit == "Correlation"
            push!(obj_values, compute_objective(correlation_matrix, locations, obj_mapping[crit]))
        else
            push!(obj_values, compute_objective(production_matrix, demand, locations, obj_mapping[crit]))
        end
    end
    x[locations] .= 1.0
    return x, Dict(criteria .=> obj_values)

end

function time_siting_method(capacity_factor_matrix::Array{Float64}, demand::Vector{Float64}, potential::Vector{Float64}, deployment_target::Int64, c::Float64, varsigma::Float64, criterion::String)
    @time siting_method(capacity_factor_matrix, demand, potential, deployment_target, c, varsigma, criterion)
end

function compute_objectives(locations::Vector{Int64}, capacity_factor_matrix::Array{Float64}, demand::Vector{Float64}, potential::Vector{Float64}, deployment_target::Int64, c::Float64, varsigma::Float64)

    production_matrix::Array{Float64, 2} = potential' .* capacity_factor_matrix
    criticality_matrix::Array{Float64, 2}, correlation_matrix::Array{Float64, 2} = compute_criticality_matrix(capacity_factor_matrix, demand, potential, deployment_target, varsigma), compute_correlation_matrix(capacity_factor_matrix)
    criteria = ["ElectricityOutput", "AverageVariability", "MaxVariability", "AverageResidualDemand", "MaxResidualDemand", "Criticality", "Correlation"]
    obj_types = [ElectricityOutput(), AverageVariability(), MaxVariability(), AverageResidualDemand(), MaxResidualDemand(), Criticality(c), Correlation()]
    obj_mapping = Dict(criteria .=> obj_types)
    obj_values::Vector{Float64} = zeros(Float64, 0)
    @inbounds for crit in criteria
        if crit == "Criticality"
            push!(obj_values, 1.0-compute_objective(criticality_matrix, locations, obj_mapping[crit]))
        elseif crit == "ElectricityOutput"
            push!(obj_values, compute_objective(production_matrix, locations, obj_mapping[crit]))
        elseif crit == "Correlation"
            push!(obj_values, compute_objective(correlation_matrix, locations, obj_mapping[crit]))
        else
            push!(obj_values, compute_objective(production_matrix, demand, locations, obj_mapping[crit]))
        end
    end
    return Dict(criteria .=> obj_values)

end

function compute_objectives(locations::Vector{Int64}, capacity_factor_matrix::Array{Float64}, demand::Vector{Float64}, potential::Vector{Float64}, deployment_target::Int64, c::Float64, delta::Int64, varsigma::Float64, tau::Int64)

    production_matrix::Array{Float64, 2} = potential' .* capacity_factor_matrix
    criticality_matrix::Array{Float64, 2}, correlation_matrix::Array{Float64, 2} = compute_criticality_matrix(capacity_factor_matrix, demand, potential, delta, deployment_target, varsigma), compute_correlation_matrix(capacity_factor_matrix)
    criteria = ["ElectricityOutput", "AverageVariability", "MaxVariability", "AverageResidualDemand", "MaxResidualDemand", "Criticality", "Correlation"]
    obj_types = [ElectricityOutput(), AverageVariability(), MaxVariability(tau), AverageResidualDemand(), MaxResidualDemand(), Criticality(c), Correlation()]
    obj_mapping = Dict(criteria .=> obj_types)
    obj_values::Vector{Float64} = zeros(Float64, 0)
    @inbounds for crit in criteria
        if crit == "Criticality"
            push!(obj_values, 1.0-compute_objective(criticality_matrix, locations, obj_mapping[crit]))
        elseif crit == "ElectricityOutput"
            push!(obj_values, compute_objective(production_matrix, locations, obj_mapping[crit]))
        elseif crit == "Correlation"
            push!(obj_values, compute_objective(correlation_matrix, locations, obj_mapping[crit]))
        else
            push!(obj_values, compute_objective(production_matrix, demand, locations, obj_mapping[crit]))
        end
    end
    return Dict(criteria .=> obj_values)

end

function compute_objectives(locations::Vector{Int64}, capacity_factor_matrix::Array{Float64}, demand::Vector{Float64}, potential::Vector{Float64}, deployment_target::Int64, c::Float64, delta::Int64, varsigma::Float64, tau::Int64)

    production_matrix::Array{Float64, 2} = potential' .* capacity_factor_matrix
    criticality_matrix::Array{Float64, 2}, correlation_matrix::Array{Float64, 2} = compute_criticality_matrix(capacity_factor_matrix, demand, potential, delta, deployment_target, varsigma), compute_correlation_matrix(capacity_factor_matrix)
    criteria = ["ElectricityOutput", "AverageVariability", "MaxVariability", "AverageResidualDemand", "MaxResidualDemand", "Criticality", "Correlation"]
    obj_types = [ElectricityOutput(), AverageVariability(), MaxVariability(tau), AverageResidualDemand(), MaxResidualDemand(), Criticality(c), Correlation()]
    obj_mapping = Dict(criteria .=> obj_types)
    obj_values::Vector{Float64} = zeros(Float64, 0)
    @inbounds for crit in criteria
        if crit == "Criticality"
            push!(obj_values, 1.0-compute_objective(criticality_matrix, locations, obj_mapping[crit]))
        elseif crit == "ElectricityOutput"
            push!(obj_values, compute_objective(production_matrix, locations, obj_mapping[crit]))
        elseif crit == "Correlation"
            push!(obj_values, compute_objective(correlation_matrix, locations, obj_mapping[crit]))
        else
            push!(obj_values, compute_objective(production_matrix, demand, locations, obj_mapping[crit]))
        end
    end
    return Dict(criteria .=> obj_values)

end
