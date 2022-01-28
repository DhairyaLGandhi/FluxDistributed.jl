function Optimisers.update(o, x::T, x̄, state) where T
  if x̄ === nothing
    return x, state
  elseif Functors.isleaf(x)
    return Optimisers._update(o, x, x̄, state)
  else
    x̄, _  = _functor(typeof(x), x̄)
    x, restructure = _functor(typeof(x), x)
    xstate = map((x, x̄, state) -> Optimisers.update(o, x, x̄, state), x, x̄, state)
    return restructure(map(first, xstate)), map(x -> x[2], xstate)
  end
end

function Optimisers.state(o, x)
  if Functors.isleaf(x)
    return Optimisers.init(o, x)
  else
    x, _ = _functor(x)
    return map(x -> Optimisers.state(o, x), x)
  end
end

_functor(x) = Functors.functor(x)
_functor(ref::Base.RefValue) = Functors.functor(ref.x)
_functor(T, x) = Functors.functor(T, x)
_functor(T, ref::Base.RefValue) = Functors.functor(T, ref.x)

Optimisers.init(o, x) = nothing

_accum(x, y) = Zygote.accum(x, y)
_accum(x::Base.RefValue, y::Base.RefValue) = Ref(Zygote.accum(x.x, y.x))

_dodiv(::Nothing, y) = nothing
_dodiv(nt::NamedTuple, y) = NamedTuple{keys(nt)}((_dodiv(nt[k], y) for k in keys(nt)))
_dodiv(x, y) = x / y
_dodiv(x::Base.RefValue, y::Real) = Ref(_dodiv(x.x, y))
_dodiv(x::AbstractArray, y::Real) = x / y


_show_stats(::Nothing) = nothing
_show_stats(nt::NamedTuple) = NamedTuple{keys(nt)}((_show_stats(nt[k]) for k in keys(nt)))
_show_stats(x) = @show mean(x), sum(x), maximum(x), minimum(x)
_show_stats(x::Base.RefValue) = Ref(_show_stats(x[]))
