using Ipopt
using Infiltrator
using GLMakie
using ForwardDiff


ns::Int64 = 7
nu::Int64 = 3
nt = ns + nu
N::Int64 = 100
dt::Float64 = 0.0001
n::Int64 = N*nt

# Total number of variables n = N*(ns + nu)
# Define objective functions and dynamics
function L(y, u)
    return (u[1]^2 )
end


cross(a, b) = [a[2]*b[3]-a[3]*b[2], a[3]*b[1]-a[1]*b[3], a[1]*b[2]-a[2]*b[1]]

function dyn(x, u)
    tau = -1.0
    T = 4.446618e-3
    Isp = 450
    mu = 1.407645794e16
    gs = 32.174
    Re = 20925662.73
    J2 = 1082.639e-6
    J3 = -2.565e-6
    J4 = -1.608e-6
    # Extract state, control, and parameter for problem
    p, f, g, h, k, L, w = x
    ur, ut, uh = u
    # Gravitational disturbing acceleration
    q = 1 + f * cos(L) + g * sin(L)
    r = p / q
    alpha2 = h * h - k * k
    chi = sqrt(h * h + k * k)
    s2 = 1 + chi * chi
    rX = (r / s2) * (cos(L) + alpha2 * cos(L) + 2 * h * k * sin(L))
    rY = (r / s2) * (sin(L) - alpha2 * sin(L) + 2 * h * k * cos(L))
    rZ = (2 * r / s2) * (h * sin(L) - k * cos(L))
    rVec = [rX, rY, rZ]
    rMag = sqrt(rX^2 + rY^2 + rZ^2)
    rXZMag = sqrt(rX^2 + rZ^2)
    vX = -(1 / s2) * sqrt(mu / p) * (sin(L) + alpha2 * sin(L) - 2 * h * k * cos(L) + g - 2 * f * h * k + alpha2 * g)
    vY = -(1 / s2) * sqrt(mu / p) * (-cos(L) + alpha2 * cos(L) + 2 * h * k * sin(L) - f + 2 * g * h * k + alpha2 * f)
    vZ = (2 / s2) * sqrt(mu / p) * (h * cos(L) + k * sin(L) + f * h + g * k)
    vVec = [vX, vY, vZ]
    rCrossv = cross(rVec, vVec)
    rCrossvMag = sqrt(rCrossv[1]^2 + rCrossv[2]^2 + rCrossv[3]^2)
    rCrossvCrossr = cross(rCrossv, rVec)
    ir1 = rVec[1] / rMag
    ir2 = rVec[2] / rMag
    ir3 = rVec[3] / rMag
    ir = [ir1, ir2, ir3]
    it1 = rCrossvCrossr[1] / (rCrossvMag * rMag)
    it2 = rCrossvCrossr[2] / (rCrossvMag * rMag)
    it3 = rCrossvCrossr[3] / (rCrossvMag * rMag)
    it = [it1, it2, it3]
    ih1 = rCrossv[1] / rCrossvMag
    ih2 = rCrossv[2] / rCrossvMag
    ih3 = rCrossv[3] / rCrossvMag
    ih = [ih1, ih2, ih3]
    enir = ir3
    enirir1 = enir * ir1
    enirir2 = enir * ir2
    enirir3 = enir * ir3
    enenirir1 = 0 - enirir1
    enenirir2 = 0 - enirir2
    enenirir3 = 1 - enirir3
    enenirirMag = sqrt(enenirir1^2 + enenirir2^2 + enenirir3^2)
    in1 = enenirir1 / enenirirMag
    in2 = enenirir2 / enenirirMag
    in3 = enenirir3 / enenirirMag
    # Geocentric latitude
    sinphi = rZ / rXZMag
    cosphi = sqrt(1 - sinphi^2)
    # Legendre polynomials
    P2 = (3 * sinphi^2 - 2) / 2
    P3 = (5 * sinphi^3 - 3 * sinphi) / 2
    P4 = (35 * sinphi^4 - 30 * sinphi^2 + 3) / 8
    dP2 = 3 * sinphi
    dP3 = (15 * sinphi - 3) / 2
    dP4 = (140 * sinphi^3 - 60 * sinphi) / 8
    # Oblate earth perturbations
    sumn = (Re / r)^2 * dP2 * J2 + (Re / r)^3 * dP3 * J3 + (Re / r)^4 * dP4 * J4
    sumr = (2 + 1) * (Re / r)^2 * P2 * J2 + (3 + 1) * (Re / r)^3 * P3 * J3 + (4 + 1) * (Re / r)^4 * P4 * J4
    ∆gn = -(mu * cosphi / (r^2)) * sumn
    ∆gr = -(mu / (r^2)) * sumr
    ∆gnin1 = ∆gn * in1
    ∆gnin2 = ∆gn * in2
    ∆gnin3 = ∆gn * in3
    ∆grir1 = ∆gr * ir1
    ∆grir2 = ∆gr * ir2
    ∆grir3 = ∆gr * ir3
    ∆g1 = ∆gnin1 - ∆grir1
    ∆g2 = ∆gnin2 - ∆grir2
    ∆g3 = ∆gnin3 - ∆grir3
    Deltag1 = ir[1] * ∆g1 + ir[2] * ∆g2 + ir[3] * ∆g3
    Deltag2 = it[1] * ∆g1 + it[2] * ∆g2 + it[3] * ∆g3
    Deltag3 = ih[1] * ∆g1 + ih[2] * ∆g2 + ih[3] * ∆g3
    # Thrust acceleration
    DeltaT1 = ((gs * T * (1 + 0.01 * tau)) / w) * ur
    DeltaT2 = ((gs * T * (1 + 0.01 * tau)) / w) * ut
    DeltaT3 = ((gs * T * (1 + 0.01 * tau)) / w) * uh
    # Total acceleration
    Delta1 = Deltag1 + DeltaT1
    Delta2 = Deltag2 + DeltaT2
    Delta3 = Deltag3 + DeltaT3
    # Differential equations of motion
    dp = (2 * p / q) * sqrt(p / mu) * Delta2
    df = sqrt(p / mu) * sin(L) * Delta1 + sqrt(p / mu) * (1 / q) * ((q + 1) * cos(L) + f) * Delta2 - sqrt(p / mu) * (g / q) * (h * sin(L) - k * cos(L)) * Delta3
    dg = -sqrt(p / mu) * cos(L) * Delta1 + sqrt(p / mu) * (1 / q) * ((q + 1) * sin(L) + g) * Delta2 + sqrt(p / mu) * (f / q) * (h * sin(L) - k * cos(L)) * Delta3
    dh = sqrt(p / mu) * (s2 * cos(L) / (2 * q)) * Delta3
    dk = sqrt(p / mu) * (s2 * sin(L) / (2 * q)) * Delta3
    dL = sqrt(p / mu) * (1 / q) * (h * sin(L) - k * cos(L)) * Delta3 + sqrt(mu * p) * ((q / p)^2)
    dw = -(T * (1 + 0.01 * tau) / Isp)
    return [dp, df, dg, dh, dk, dL, dw]
end


function closure_L(x)
    y = @view x[1:ns]
    u = @view x[ns+1:end]

    return L(y, u)
end

function L_der(y, u)
    x = [y; u]
    return ForwardDiff.gradient(closure_L, x)
end

function L_dder(y, u)
    x = [y; u]

    return ForwardDiff.hessian(closure_L, x)
end


function closure_dyn(x)
    y = @view x[1:ns]
    u = @view x[ns+1:end]

    return dyn(y, u)
end

function dyn_der(y, u)
    x = [y; u]
    return ForwardDiff.jacobian(closure_dyn, x)
end

function split_matrix_chunks(A::Matrix{T}, ns) where T
    mats = Matrix{T}[]
   
    m = size(A, 1)
    iter = Int(round(m/ns))
    count = 0
    for i=1:iter
        push!(mats, A[(i-1)*ns+1:i*ns, :])
        count = i
    end
    # push!(mats, A[count*ns+1:end])

    return mats
end
function dyn_dder(y, u)
    x = [y; u]

    H = ForwardDiff.jacobian(x -> ForwardDiff.jacobian(closure_dyn, x), x)

    return split_matrix_chunks(H, ns+nu)
end
# Objective function
# Using trapezoidal method
# Number of mesh points = N
# Number of mesh intervals = N-1
function eval_f(x::Vector{Float64})
    sum = 0
    for i=1:N-1
        y = @view x[((i-1)*nt+1):((i-1)*nt+ns)]
        u = @view x[((i-1)*nt+ns+1):((i-1)*nt+ns+nu)]
        yn = @view x[((i)*nt+1):((i)*nt+ns)]
        un = @view x[((i)*nt+ns+1):((i)*nt+ns+nu)]

        sum = sum + ((L(y, u) + L(yn, un))/2)*dt
    end

    return sum
end

# Number of collocation constraints is (N-1)*ns
function eval_g(x::Vector{Float64}, g::Vector{Float64})
    # Bad: g    = zeros(2)  # Allocates new array
    # OK:  g[:] = zeros(2)  # Modifies 'in place'
    for i=1:N-1
        y = @view x[((i-1)*nt+1):((i-1)*nt+ns)]
        u = @view x[((i-1)*nt+ns+1):((i-1)*nt+ns+nu)]
        yn = @view x[((i)*nt+1):((i)*nt+ns)]
        un = @view x[((i)*nt+ns+1):((i)*nt+ns+nu)]

        # Trapezoidal method
        g[((i-1)*ns+1):((i)*ns)] .= yn - (y + ((dyn(y,u) + dyn(yn, un))/2)*dt) 
    end

    return true
end

function eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64})
    # grad_f is a vector of size n
    for i = 1:N
        y = @view x[((i-1)*nt+1):((i-1)*nt+ns)]
        u = @view x[((i-1)*nt+ns+1):((i-1)*nt+ns+nu)]

        # Trapezoidal method
        if i == 1 || i == N
            grad_f[((i-1)*nt+1):((i)*nt)] .= L_der(y, u)*dt / 2
        else
            grad_f[((i-1)*nt+1):((i)*nt)] .= L_der(y, u)*dt / 2 + L_der(y, u)*dt / 2
        end
    end

    return true
end

#-------------------------------------------------------------------
# Jacobian Computations
#-------------------------------------------------------------------
nnzg = (N-1)*ns*2*nt
neg = (N-1)*ns*ns
ratio_j = nnzg/neg
function compute_indices_indices_jacobian!(rows, cols)
    for i = 1:N-1
        row_start_ind = (i - 1) * ns
        col_start_ind = (i - 1) * nt
        start_index = (i - 1) * (2*nt*ns) + 1 
        count = 0
        for j = 1:ns  # ns rows corresponding to each grid-point
            for k = 1:2*nt
                rows[start_index+count] = row_start_ind + j
                cols[start_index+count] = col_start_ind + k
                count = count + 1
            end
        end
    end
end

function compute_values_jacobian!(values, x)
    val_array = zeros(ns, 2nt)
    for i=1:N-1
        val_array .= zero(Float64)
        y = @view x[((i-1)*nt+1):((i-1)*nt+ns)]
        u = @view x[((i-1)*nt+ns+1):((i-1)*nt+ns+nu)]
        yn = @view x[((i)*nt+1):((i)*nt+ns)]
        un = @view x[((i)*nt+ns+1):((i)*nt+ns+nu)]

        val_array[1:ns,1:nt] = -dyn_der(y, u)*dt/2
        val_array[1:ns,nt+1:2nt] = -dyn_der(yn, un)*dt/2
        for j = 1:ns
            val_array[j, j] = val_array[j, j] - 1.0
            val_array[j, nt+j] = val_array[j, nt+j] + 1.0
        end
        count = 0
        start_ind = (i-1)*2*ns*nt + 1
        for j = 1:ns
            for k = 1:2nt
                values[start_ind + count] = val_array[j, k]
                count = count + 1
            end
        end
    end
end

function eval_jac_g(
    x::Vector{Float64},
    rows::Vector{Int32},
    cols::Vector{Int32},
    values::Union{Nothing,Vector{Float64}},
)
    if values === nothing
        compute_indices_indices_jacobian!(rows, cols)
    else
        compute_values_jacobian!(values, x)
    end
    return true
end

#-------------------------------------------------------------------
# Hessian Computations
#-------------------------------------------------------------------
# Total number of variables = n
# Size of Hessian Matrix: n*n = (N*nt)*(N*nt)
# Non-zero entries in Hessian matrix: N*nt^2
neH = (N*nt)*(N*nt)
nnzh = Int(round(N*nt*(nt+1)/2))
ratio_h = nnzh / neH
function compute_indices_hessian!(rows, cols)
    for i = 1:N
        row_start_ind = (i - 1) * nt
        col_start_ind = (i - 1) * nt
        start_index = Int(round((i - 1) * (nt*(nt+1)/2) + 1))
        count = 0
        for j = 1:nt  # ns rows corresponding to each grid-point
            for k = 1:j
                rows[start_index+count] = row_start_ind + j
                cols[start_index+count] = col_start_ind + k
                count = count + 1
            end
        end
    end
end

function compute_values_hessian!(values, x, obj_factor, lambda)
    val_array = zeros(nt, nt)
    lambda_ind = 1
    for i=1:N
        val_array .= zero(Float64)
        y = @view x[((i-1)*nt+1):((i-1)*nt+ns)]
        u = @view x[((i-1)*nt+ns+1):((i-1)*nt+ns+nu)]
        # yn = @view x[((i)*nt+1):((i)*nt+ns)]
        # un = @view x[((i)*nt+ns+1):((i)*nt+ns+nu)]

        # Compute Hessian for objective
        if i == 1 || i == N
            val_array[1:nt,1:nt] = L_dder(y, u)*dt*obj_factor/2
        else
            val_array[1:nt,1:nt] = L_dder(y,u)*dt*obj_factor/2 + L_dder(y,u)*dt*obj_factor/2
        end

        # Compute hessian for constraints
        dd = dyn_dder(y,u)
        # ddn = dyn_dder(yn,un)

        if i == 1
            for j = 1:length(dd)
                val_array[1:nt, 1:nt] .= val_array[1:nt, 1:nt] .- lambda[lambda_ind] * dd[j] * dt / 2
                # val_array[nt+1:2nt, nt+1:2nt] .= val_array[nt+1:2nt, nt+1:2nt] .- lambda[i]*ddn[i]/2
                lambda_ind = lambda_ind + 1
            end
        elseif i == N
            for j = 1:length(dd)
                val_array[1:nt, 1:nt] .= val_array[1:nt, 1:nt] .- lambda[lambda_ind-ns] * dd[j] * dt / 2
                # val_array[nt+1:2nt, nt+1:2nt] .= val_array[nt+1:2nt, nt+1:2nt] .- lambda[i]*ddn[i]/2
                lambda_ind = lambda_ind + 1
            end
        else
            for j = 1:length(dd)
                val_array[1:nt, 1:nt] .= val_array[1:nt, 1:nt] .- lambda[lambda_ind-ns] * dd[j] * dt / 2 .- lambda[lambda_ind] * dd[j] * dt / 2
                # val_array[nt+1:2nt, nt+1:2nt] .= val_array[nt+1:2nt, nt+1:2nt] .- lambda[i]*ddn[i]/2
                lambda_ind = lambda_ind + 1
            end
        end

        count = 0
        start_ind = Int(round((i - 1) * (nt*(nt+1)/2) + 1))
        for j = 1:nt
            for k = 1:j
                values[start_ind + count] = val_array[j, k]
                count = count + 1
            end
        end
    end
end
    
function eval_h(
    x::Vector{Float64},
    rows::Vector{Int32},
    cols::Vector{Int32},
    obj_factor::Float64,
    lambda::Vector{Float64},
    values::Union{Nothing,Vector{Float64}},
)
    if values === nothing
        compute_indices_hessian!(rows, cols)
    else
        compute_values_hessian!(values, x, obj_factor, lambda)
    end
    return
end

n = N*(ns + nu)
x_L = zeros(n)
x_U = zeros(n)
function add_variable_constraints!(x_L, x_U, y_ll, y_ul, u_ll, u_ul)
    for i = 1:N
        x_L[((i-1)*nt+1):((i-1)*nt+nt)] .= [y_ll; u_ll]
        x_U[((i-1)*nt+1):((i-1)*nt+nt)] .= [y_ul; u_ul]
    end
end

x0 = [21837080.052835, 0, 0, -0.25396764647494, 0, pi, 1]
y_ll = [20_000_000, -1, -1, -1, -1, x0[6], 0.1]
y_ul = [60_000_000, 1, 1, 1, 1, 9*2*pi, x0[7]]

u_ll = [-1.0, -1.0, -1.0]
u_ul = [1.0, 1.0, 1.0]

add_variable_constraints!(x_L, x_U, y_ll, y_ul, u_ll, u_ul)

function initial_state!(x_L, x_U, x0)
    x_L[1:ns] = x0
    x_U[1:ns] = x0
end

initial_state!(x_L,x_U, x0)


m = (N-1)*ns
g_L = zeros(m)
g_U = zeros(m)
nnzg = m*2*nt

prob = Ipopt.CreateIpoptProblem(
    n,
    x_L,
    x_U,
    m,
    g_L,
    g_U,
    nnzg,
    nnzh,
    eval_f,
    eval_g,
    eval_grad_f,
    eval_jac_g,
    eval_h
)

Ipopt.AddIpoptStrOption(prob, "hessian_approximation", "limited-memory")
prob.x = zeros(n)
prob.x[1:ns] = x0
solvestat = Ipopt.IpoptSolve(prob)

function plot_results(x)
    y_mat = zeros(ns, N)
    u_mat = zeros(nu, N)
    for i = 1:N
        y_mat[:,i] = @view x[((i-1)*nt+1):((i-1)*nt+ns)]
        u_mat[:,i] = @view x[((i-1)*nt+ns+1):((i-1)*nt+ns+nu)]
    end
    return y_mat, u_mat
end

y_mat, u_mat = plot_results(prob.x)
lines(y_mat[1,:], y_mat[2, :])