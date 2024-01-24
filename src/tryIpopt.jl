using Ipopt
using Infiltrator
using GLMakie
using ForwardDiff

# hs071
# min x1 * x4 * (x1 + x2 + x3) + x3
# st  x1 * x2 * x3 * x4 >= 25
#     x1^2 + x2^2 + x3^2 + x4^2 = 40
#     1 <= x1, x2, x3, x4 <= 5
# Start at (1,5,5,1)
# End at (1.000..., 4.743..., 3.821..., 1.379...)

# First try to optimize by using only Euler integration with a double integrator dynamics
# yd[1] = y[2] where xd = \dot{x}
# yd[2] = u
ns::Int64 = 2 
nu::Int64 = 1
nt = ns + nu
N::Int64 = 10000
dt::Float64 = 0.001
n::Int64 = N*nt
# Total number of variables n = N*(ns + nu)
function L(y, u)
    return (y[1]^2 + y[2]^2 + 0*u[1]^2)
end

function L_der(y, u)
    return [2*y[1], 2*y[2], 0*2*u[1]]
end

function L_dder(y, u)
    return [2.0 0.0 0.0;
            0.0 2.0 0.0;
            0.0 0.0 0*2.0]
end


function dyn(y, u)
    yn1 = y[2]
    yn2 = u[1]

    return [yn1, yn2]
end

function dyn_der(y, u)
    return [0.0 1.0 0.0;
            0.0 0.0 1.0]
end

function dyn_dder(y, u)
    g1_dder = [0.0 0.0 0.0;
               0.0 0.0 0.0;
               0.0 0.0 0.0]
    
    g2_dder = [0.0 0.0 0.0;
                0.0 0.0 0.0;
                0.0 0.0 0.0]
    
    return [g1_dder, g2_dder]
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
    

rows = zeros(Int64, nnzh)
cols = zeros(Int64, nnzh)
vals = zeros(Float64, nnzh)
obj_factor = 1.0
lambda = ones((N-1)*ns)
x = ones(N*nt)
compute_indices_hessian!(rows, cols)
compute_values_hessian!(vals, x, obj_factor, lambda)

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

y_ll = [-10.0, -10.0]
y_ul = [10.0, 10.0]
u_ll = [-1.0]
u_ul = [1.0]

add_variable_constraints!(x_L, x_U, y_ll, y_ul, u_ll, u_ul)

function initial_state!(x_L, x_U, x0)
    x_L[1:ns] = x0
    x_U[1:ns] = x0
end

x0 = [8.0, 0.0]

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

Ipopt.AddIpoptStrOption(prob, "hessian_approximation", "exact")
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

function closure_L(x)
    y = @view x[1:2]
    u = @view x[3:3]

    return L(y, u)
end

function auto_grad_L(y, u)
    x = [y; u]
    return ForwardDiff.gradient(closure_L, x)
end

function auto_hess_L(y, u)
    x = [y; u]

    return ForwardDiff.hessian(closure_L, x)
end

function closure_dyn(x)
    y = @view x[1:2]
    u = @view x[3:3]

    return dyn(y, u)
end

function auto_jac_dyn(y, u)
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
function auto_hess_dyn(y, u)
    x = [y; u]

    H = ForwardDiff.jacobian(x -> ForwardDiff.jacobian(closure_dyn, x), x)

    return split_matrix_chunks(H, ns+nu)
end

y = ones(2)
u = ones(1)
ag = auto_grad_L(y, u)
ah = auto_hess_L(y, u)
aJ = auto_jac_dyn(y, u)
aH = auto_hess_dyn(y, u)