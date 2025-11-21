module SimpleManifolds

"""
SimpleManifolds.jl
A tiny example manifold library with:
  - Sphere manifold `Sphere`
  - Basic Riemannian operations
  - Riemannian gradient descent
"""

using LinearAlgebra
using Random 

# ----------------------------
# Abstract manifold interface
# ----------------------------

# Creation of an abstract class, acts as a tag for manifolds
abstract type AbstractManifold end

# A simple N-sphere S^n embedded in ℝ^(n+1)
struct Sphere <: AbstractManifold # sphere is a subtype of AbstractManifold
    n::Int  # field with type annotation 
end

dimension(M::Sphere) = M.n
ambient_dimension(M::Sphere) = M.n + 1

# ----------------------------
# Sphere operations
# ----------------------------

"Project a vector in ℝ^(n+1) onto the unit sphere."
function project(M::Sphere, x::AbstractVector) 
    return x / norm(x) # normed vector gives us a point on the sphere
end

"Project ambient vector v to the tangent space T_p S^n."
function to_tangent(M::Sphere, p::AbstractVector, v::AbstractVector)
    # p is a unit vector on the sphere, the tangent space is all v s.t. dot(v,p) = 0
    # project to tangent by subtracting parts of v that are normal to T_pM.
    return v .- (dot(v, p)) .* p # .- .* perform element-wise subtract and multiply
end

"Riemannian inner product = Euclidean dot product."
# inner does not enforce tangency as it will be called many times (wasted dot product calcs)
inner(M::Sphere, p, u, v) = dot(u, v)  # inputs should be from tangent space -> needs insurance

"Riemannian norm on the tangent space."
norm(M::Sphere, p, u) = sqrt(inner(M, p, u, u)) 

"Exponential map on S^n at point p with tangent vector u."
# Takes a tangent vector and returns a point on the sphere after following geodesic
function exp(M::Sphere, p::AbstractVector, u::AbstractVector)
    θ = LinearAlgebra.norm(u)
    if θ < 1e-12
        return p
    end
    # geodescic on sphere when norm u is small, stay at p. 
    return cos(θ) .* p .+ sin(θ) .* (u ./ θ)
end

"Simple retraction on S^n: step in ambient space then renormalize."
function retract(M::Sphere, p::AbstractVector, u::AbstractVector)
    # step forward on smbient manifold then project back down to sphere
    return project(M, p .+ u)
end

"Random point on S^n."
function random_point(M::Sphere)
    x = randn(ambient_dimension(M))
    return x / norm(x)
end

"Random tangent vector at p on S^n."
function random_tangent(M::Sphere, p::AbstractVector)
    v = randn(ambient_dimension(M))
    return to_tangent(M, p, v)
end

# ----------------------------
# Generic Riemannian gradient descent
# ----------------------------

"""
    manifold_gradient_descent(M, f, gradf, p0; α=0.1, maxiter=100)

Run simple Riemannian gradient descent on a manifold `M`.

Arguments:
  - M      : manifold (e.g. `Sphere(2)`)
  - f      : cost function f(p)::Real
  - gradf  : gradient function gradf(p)::AbstractVector (ambient)
  - p0     : initial point (ambient representation)

Keyword arguments:
  - α      : step size
  - maxiter: number of iterations
"""
function manifold_gradient_descent(
        M::AbstractManifold,
        f::Function,
        gradf::Function,
        p0;
        α::Real = 0.1,
        maxiter::Integer = 100,
    )

    p = project(M, p0)

    for k in 1:maxiter
        g = gradf(p)
        g_tan = to_tangent(M, p, g)
        p = retract(M, p, -α .* g_tan)
    end

    return p
end

# ----------------------------
# Exports
# ----------------------------

export AbstractManifold,
       Sphere,
       dimension,
       ambient_dimension,
       project,
       to_tangent,
       inner,
       norm,
       exp,
       retract,
       random_point,
       random_tangent,
       manifold_gradient_descent

end # module
