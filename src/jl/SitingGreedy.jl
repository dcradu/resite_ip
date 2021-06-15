using BandedMatrices
using Statistics

### Objective type definitions ###

abstract type Objective end

struct MaxVariability <: Objective
    tau::Int64
end

struct AverageVariability <: Objective end

struct MaxResidualDemand <: Objective end

struct AverageResidualDemand <: Objective end

struct Criticality <: Objective
    c::Float64
end

struct Correlation <: Objective end

# Incomplete Type Initialization using Inner Constructor
# mutable struct InputData
#     A::Array{Float64}
#     d::Vector{Float64}
#     T::Vector{Int64}
#     InputData(A,d) = new(A, d)
# end

# Recursive Type Initialization using Inner Constructor (i.e., T depends on d)
# mutable struct InputData
#     A::Array{Float64}
#     d::Vector{Float64}
#     T::Vector{Int64}
#     InputData(A,d) = new(A, d, [t for t = 1:length(d)])
# end

# struct InputData
#     A::Array{Float64}
#     d::Vector{Float64}
# end
#
# objective_mv = MaxVariability(10)
# objective_av = AverageVariability()
# objective_mrd = MaxResidualDemand()
# objective_ard = AverageResidualDemand()
# objective_crit = Criticality(10.0)
# objective_corr = Correlation()

### PREPROCESSING ###

function compute_criticality_matrix(A::Array{Float64, 2}, demand::Vector{Float64}, potential::Vector{Float64}, delta::Int64, N::Int64, varsigma::Float64)::Array{Float64, 2}

    T::Int64, L::Int64 = size(A)
    W::Int64 = T-delta+1
    S::Array{Float64, 2} = BandedMatrix{Float64}(Ones(W, T), (0, delta-1))
    A_smth::Array{Float64, 2} = S * A
    d_smth::Vector{Float64} = S * demand
    D::Array{Float64, 2} = zeros(Float64, W, L)
    alpha_l::Float64 = 0.0
    w::Int64 = 1
    @inbounds while w <= W
        l::Int64 = 1
        @inbounds while l <= L
            alpha_lw = (varsigma * d_smth[w]) / (potential[l] * N)
            if A_smth[w, l] > alpha_lw
                D[w, l] = 1.0
            end
            l += 1
        end
        w += 1
    end
    return D
end

function time_compute_criticality_matrix(A::Array{Float64, 2}, demand::Vector{Float64}, potential::Vector{Float64}, delta::Int64, N::Int64, varsigma::Float64)
    @time compute_criticality_matrix(A, demand, potential, delta, N, varsigma)
end

function compute_correlation_matrix(A::Array{Float64, 2})::Array{Float64, 2}

    return cor(A, dims=1)

end

function time_compute_correlation_matrix(A::Array{Float64 ,2})
    @time compute_correlation_matrix(A)
end

### OBJECTIVES ###

function compute_objective(A::Array{Float64}, d::Vector{Float64}, L::Vector{Int64}, obj::MaxVariability)::Float64

    rd::Float64, rd_max::Float64 = 0.0, -1e9
    p_t::Float64, p_tmi::Float64, rd_t::Float64, rd_tmi::Float64 = 0.0, 0.0, 0.0, 0.0
    i::Int64, t::Int64 = 1, 1
    @inbounds while i <= obj.tau
        t = i+1
        @inbounds while t <= length(d)
            p_t, p_tmi, rd_t, rd_tmi = 0.0, 0.0, 0.0, 0.0
            @inbounds for l in L
                p_t += A[t, l]
                p_tmi += A[t-i, l]
            end
            if p_t < d[t]
                rd_t = (d[t] - p_t)
            elseif p_tmi < d[t-i]
                rd_tmi = (d[t-i] - p_tmi)
            end
            rd = rd_t - rd_tmi
            if rd > rd_max
                rd_max = rd
            end
            t += 1
        end
        i += 1
    end
    return rd_max

end

function time_compute_objective(A::Array{Float64}, d::Vector{Float64}, L::Vector{Int64}, obj::MaxVariability)
    @time compute_objective(A, d, L, obj)
end

function compute_objective(A::Array{Float64}, p::Vector{Float64}, d::Vector{Float64}, l::Int64, obj::MaxVariability)::Float64

    rd::Float64, rd_max::Float64 = 0.0, -1e9
    p_t::Float64, p_tmi::Float64, rd_t::Float64, rd_tmi::Float64 = 0.0, 0.0, 0.0, 0.0
    i::Int64, t::Int64 = 1, 1
    @inbounds while i <= obj.tau
        t = i+1
        @inbounds while t <= length(d)
            p_t, p_tmi, rd_t, rd_tmi = p[t] + A[t, l], p[t-i] + A[t-i, l], 0.0, 0.0
            if p_t < d[t]
                rd_t = (d[t] - p_t)
            elseif p_tmi < d[t-i]
                rd_tmi = (d[t-i] - p_tmi)
            end
            rd = rd_t - rd_tmi
            if rd > rd_max
                rd_max = rd
            end
            t += 1
        end
        i += 1
    end
    return rd_max

end

function time_compute_objective(A::Array{Float64}, p::Vector{Float64}, d::Vector{Float64}, l::Int64, obj::MaxVariability)
    @time compute_objective(A, p, d, l, obj)
end

function compute_objective(A::Array{Float64}, d::Vector{Float64}, L::Vector{Int64}, obj::AverageVariability)::Float64

    rd::Float64, rd_t::Float64, rd_tm1::Float64 = 0.0, 0.0, 0.0
    p_t::Float64, p_tm1::Float64 = 0.0, 0.0
    @inbounds for l in L
        p_tm1 += A[1, l]
    end
    t::Int64 = 2
    @inbounds while t <= length(d)
        p_t, rd_t, rd_tm1 = 0.0, 0.0, 0.0
        @inbounds for l in L
            p_t += A[t, l]
        end
        if p_t < d[t]
            rd_t = (d[t] - p_t)
        elseif p_tm1 < d[t-1]
            rd_tm1 = (d[t-1] - p_tm1)
        end
        rd += abs(rd_t - rd_tm1)
        p_tm1 = p_t
        t += 1
    end
    return rd/(length(d)-2)

end

function time_compute_objective(A::Array{Float64}, d::Vector{Float64}, L::Vector{Int64}, obj::AverageVariability)
    @time compute_objective(A, d, L, obj)
end

function compute_objective(A::Array{Float64}, p::Vector{Float64}, d::Vector{Float64}, l::Int64, obj::AverageVariability)::Float64

    rd::Float64, rd_t::Float64, rd_tm1::Float64 = 0.0, 0.0, 0.0
    p_t::Float64, p_tm1::Float64 = 0.0, p[1] + A[1, l]
    t::Int64 = 2
    @inbounds while t <= length(d)
        p_t, rd_t, rd_tm1 = p[t] + A[t, l], 0.0, 0.0
        if p_t < d[t]
            rd_t = (d[t] - p_t)
        elseif p_tm1 < d[t-1]
            rd_tm1 = (d[t-1] - p_tm1)
        end
        rd += abs(rd_t - rd_tm1)
        p_tm1 = p_t
        t += 1
    end
    return rd/(length(d)-2)

end

function time_compute_objective(A::Array{Float64}, p::Vector{Float64}, d::Vector{Float64}, l::Int64, obj::AverageVariability)
    @time compute_objective(A, p, d, l, obj)
end

function compute_objective(A::Array{Float64}, d::Vector{Float64}, L::Vector{Int64}, obj::MaxResidualDemand)::Float64

    p::Float64, rd::Float64, rd_max::Float64 = 0.0, 0.0, 0.0
    t::Int64 = 1
    @inbounds while t <= length(d)
        p = 0.0
        @inbounds for l in L
            p += A[t, l]
        end
        if p < d[t]
           rd = (d[t] - p)
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

    p_t::Float64, rd::Float64, rd_max::Float64 = 0.0, 0.0, -1e+9
    t::Int64 = 1
    @inbounds while t <= length(d)
        p_t = p[t] + A[t, l]
        if p_t < d[t]
            rd = (d[t] - p_t)
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

    p::Float64, rd::Float64 = 0.0, 0.0
    t::Int64 = 1
    @inbounds while t <= length(d)
        p = 0.0
        @inbounds for l in L
            p += A[t, l]
        end
        if p < d[t]
           rd += (d[t] - p)
        end
        t += 1
    end
    return rd/length(d)

end

function time_compute_objective(A::Array{Float64}, d::Vector{Float64}, L::Vector{Int64}, obj::AverageResidualDemand)
    @time compute_objective(A, d, L, obj)
end

function compute_objective(A::Array{Float64}, p::Vector{Float64}, d::Vector{Float64}, l::Int64, obj::AverageResidualDemand)::Float64

    p_t::Float64, rd::Float64 = 0.0, 0.0
    t::Int64 = 1
    @inbounds while t <= length(d)
        p_t = p[t] + A[t, l]
        if p_t < d[t]
            rd += (d[t] - p_t)
        end
        t += 1
    end
    return rd/length(d)

end

function time_compute_objective(A::Array{Float64}, p::Vector{Float64}, d::Vector{Float64}, l::Int64, obj::AverageResidualDemand)
    @time compute_objective(A, p, d, l, obj)
end

function compute_objective(A::Array{Float64}, L::Vector{Int64}, obj::Criticality)::Float64

    d_w::Float64, v::Float64 = 0.0, 0.0
    w::Int64 = 1
    @inbounds while w <= size(A)[1]
        d_w = 0.0
        @inbounds for l in L
            d_w += A[w, l]
        end
        if d_w > obj.c
            v += 1.0
        end
        w += 1
    end
    return v

end

function time_compute_objective(A::Array{Float64}, L::Vector{Int64}, obj::Criticality)
    @time compute_objective(A, L, obj)
end

function compute_objective(A::Array{Float64}, d::Vector{Float64}, l::Int64, obj::Criticality)::Float64

    d_w::Float64, v::Float64 = 0.0, 0.0
    w::Int64 = 1
    @inbounds while w <= length(d)
        d_w = d[w] + A[w, l]
        if d_w > obj.c
            v += 1.0
        end
        w += 1
    end
    return v

end

function time_compute_objective(A::Array{Float64}, d::Vector{Float64}, l::Int64, obj::Criticality)
    @time compute_objective(A, d, l, obj)
end

function compute_objective(A::Array{Float64}, L::Vector{Int64}, obj::Correlation)

    a_tmp::Float64, v::Float64 = 0.0, 0.0
    @inbounds for l1 in L
        @inbounds for l2 in L
            if l1 != l2
                v += A[l1, l2]
            end
        end
    end
    v = v / 2.0
    return v

end

function time_compute_objective(A::Array{Float64}, L::Vector{Int64}, obj::Correlation)
    @time compute_objective(A, L, obj)
end

function compute_objective(A::Array{Float64}, l::Int64, L::Vector{Int64}, obj::Correlation)

    v::Float64 = 0.0
    @inbounds for loc in L
        v += A[l, loc]
    end
    return v

end

function time_compute_objective(A::Array{Float64}, l::Int64, L::Vector{Int64}, obj::Correlation)
    @time compute_objective(A, l, L, obj)
end

############# GREEDY ALGORITHMS ###########

### SINGLE OBJECTIVE ALGORITHMS ###

function greedy_algorithm(A::Array{Float64}, d::Vector{Float64}, N::Int64, obj::MaxResidualDemand)

    T, L = size(A)
    ind_compl_incumbent::Vector{Int64}, ind_incumbent::Vector{Int64}, ind_candidate::Int64 = [l for l = 1:L], zeros(Int64, 0), 0
    p_incumbent::Vector{Float64} = zeros(Float64, T)
    v::Float64, v_min::Float64 = 0.0, 0.0
    n::Int64 = 0
    @inbounds while n < N
        ind_candidate, v_min = 0, 1e+9
        @inbounds for ind in ind_compl_incumbent
            v = compute_objective(A, p_incumbent, d, ind, obj)
            if v < v_min
                ind_candidate = ind
                v_min = v
            end
        end
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
    ind_compl_incumbent::Vector{Int64}, ind_incumbent::Vector{Int64}, ind_candidate::Int64 = [l for l = 1:L], zeros(Int64, 0), 0
    p_incumbent::Vector{Float64} = zeros(Float64, T)
    v::Float64, v_min::Float64 = 0.0, 0.0
    n::Int64 = 0
    @inbounds while n < N
        ind_candidate, v_min = 0, 1e+9
        @inbounds for ind in ind_compl_incumbent
            v = compute_objective(A, p_incumbent, d, ind, obj)
            if v < v_min
                ind_candidate = ind
                v_min = v
            end
        end
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
    ind_compl_incumbent::Vector{Int64}, ind_incumbent::Vector{Int64}, ind_candidate::Int64 = [l for l = 1:L], zeros(Int64, 0), 0
    p_incumbent::Vector{Float64} = zeros(Float64, T)
    v::Float64, v_min::Float64 = 0.0, 0.0
    n::Int64 = 0
    @inbounds while n < N
        ind_candidate, v_min = 0, 1e+9
        @inbounds for ind in ind_compl_incumbent
            v = compute_objective(A, p_incumbent, d, ind, obj)
            if v < v_min
                ind_candidate = ind
                v_min = v
            end
        end
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
    ind_compl_incumbent::Vector{Int64}, ind_incumbent::Vector{Int64}, ind_candidate::Int64 = [l for l = 1:L], zeros(Int64, 0), 0
    p_incumbent::Vector{Float64} = zeros(Float64, T)
    v::Float64, v_min::Float64 = 0.0, 0.0
    n::Int64 = 0
    @inbounds while n < N
        ind_candidate, v_min = 0, 1e+9
        @inbounds for ind in ind_compl_incumbent
            v = compute_objective(A, p_incumbent, d, ind, obj)
            if v < v_min
                ind_candidate = ind
                v_min = v
            end
        end
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
    ind_compl_incumbent::Vector{Int64}, ind_incumbent::Vector{Int64}, ind_candidate::Int64 = [l for l = 1:L], zeros(Int64, 0), 0
    d_incumbent::Vector{Float64} = zeros(Float64, W)
    v::Float64, v_max::Float64 = 0.0, 0.0
    n::Int64 = 0
    @inbounds while n < N
        ind_candidate, v_max = 0, -1e+9
        @inbounds for ind in ind_compl_incumbent
            v = compute_objective(A, d_incumbent, ind, obj)
            if v > v_max
                ind_candidate = ind
                v_max = v
            end
        end
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
    ind_compl_incumbent::Vector{Int64}, ind_incumbent::Vector{Int64}, ind_candidate::Int64 = [l for l = 1:L], zeros(Int64, 0), 0
    v::Float64, v_min::Float64 = 0.0, 0.0
    n::Int64 = 0
    @inbounds while n < N
        ind_candidate, v_min = 0, 1e+9
        @inbounds for ind in ind_compl_incumbent
            if n > 1
                v = compute_objective(A, ind, ind_incumbent, obj)
            else
                v = compute_objective(A, ind, ind_compl_incumbent, obj)
            end
            if v < v_min
                ind_candidate = ind
                v_min = v
            end
        end
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

function siting_method(capacity_factor_matrix::Array{Float64,2}, demand::Vector{Float64}, potential::Vector{Float64}, deployment_target::Int64, c::Float64, delta::Int64, varsigma::Float64, tau::Int64, criterion::String)

    production_matrix::Array{Float64, 2} = potential' .* capacity_factor_matrix
    criticality_matrix::Array{Float64, 2}, correlation_matrix::Array{Float64, 2} = compute_criticality_matrix(capacity_factor_matrix, demand, potential, delta, deployment_target, varsigma), compute_correlation_matrix(capacity_factor_matrix)
    criteria = ["AverageVariability", "MaxVariability", "AverageResidualDemand", "MaxResidualDemand", "Criticality", "Correlation"]
    obj_types = [AverageVariability(), MaxVariability(tau), AverageResidualDemand(), MaxResidualDemand(), Criticality(c), Correlation()]
    obj_mapping = Dict(criteria .=> obj_types)
    locations::Vector{Int64} = zeros(Int64, deployment_target)
    if criterion == "Criticality"
        locations .= greedy_algorithm(criticality_matrix, deployment_target, obj_mapping[criterion])
    elseif criterion == "Correlation"
        locations .= greedy_algorithm(correlation_matrix, deployment_target, obj_mapping[criterion])
    else
        locations .= greedy_algorithm(production_matrix, demand, deployment_target, obj_mapping[criterion])
    end
    x::Vector{Float64}, obj_values::Vector{Float64} = zeros(Float64, length(potential)), zeros(Float64, 0)
    @inbounds for crit in criteria
        if crit == "Criticality"
            push!(obj_values, compute_objective(criticality_matrix, locations, obj_mapping[crit]))
        elseif crit == "Correlation"
            push!(obj_values, compute_objective(correlation_matrix, locations, obj_mapping[crit]))
        else
            push!(obj_values, compute_objective(production_matrix, demand, locations, obj_mapping[crit]))
        end
    end
    x[locations] .= 1.
    return x, Dict(criteria .=> obj_values)

end

function time_siting_method(capacity_factor_matrix::Array{Float64}, demand::Vector{Float64}, potential::Vector{Float64}, deployment_target::Int64, c::Float64, delta::Int64, varsigma::Float64, tau::Int64, criterion::String)
    @time siting_method(capacity_factor_matrix, demand, potential, deployment_target, c, delta, varsigma, tau, criterion)
end
