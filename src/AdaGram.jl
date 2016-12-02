module AdaGram

using ArrayViews
using Devectorize

sigmoid(x) = 1. / (1. + exp(-x))
log_sigmoid(x) = -log(1. + exp(-x))

Tsf = Float32
Tw = Int32

include("softmax.jl")

import ArrayViews.view
import ArrayViews.Subs
import Base.vec


abstract Locality
immutable Local <: Locality end
immutable Shared <: Locality end


immutable Dictionary{S<:AbstractString}
	word2id::Dict{S, Tw}
	id2word::Array{S}
end

function Dictionary{S<:AbstractString}(id2word::Array{S})
	word2id = Dict(word=>Tw(id) for (id,word) in enumerate(id2word))
	Dictionary(word2id, id2word)
end

type VectorModel
	frequencies::DenseArray{Int64}
	code::DenseArray{Int8, 2}
	path::DenseArray{Int32, 2}
	In::DenseArray{Tsf, 3}
	Out::DenseArray{Tsf, 2}
	alpha::Float64
	d::Float64
	counts::DenseArray{Float32, 2}
end

M(vm::VectorModel) = size(vm.In, 1) #dimensionality of word vectors
T(vm::VectorModel) = size(vm.In, 2) #number of meanings
V(vm::VectorModel) = size(vm.In, 3) #number of words

view(x::SharedArray, i1::Subs, i2::Subs) = view(sdata(x), i1, i2)
view(x::SharedArray, i1::Subs, i2::Subs, i3::Subs) = view(sdata(x), i1, i2, i3)

Base.rand{T}(dims::Tuple, norm::T) = (rand(T, dims) .- 0.5) ./ norm

function shared_rand{T}(dims::Tuple, norm::T)
	S = SharedArray(T, dims; init = S -> begin
			chunk = localindexes(S)
			chunk_size = length(chunk)
			data = rand(T, chunk_size) #GOLDPLATE: this can be done inplace
			@devec data = (data - 0.5) ./ norm 
			S[chunk] = data
		end)
	return S
end

function shared_zeros{T}(::Type{T}, dims::Tuple)
	S = SharedArray(T, dims; init = S -> begin
			chunk = localindexes(S)
			chunk_size = length(chunk)
			S[chunk] = 0.
		end)
	return S
end





"""
 - `V` vocabulary size
 - `M` dimentionality of vectors
 - `T` max number of meanings
"""
function VectorModel(::Type{Shared}, max_length::Int64, V::Int64, M::Int64, T::Int64=1, alpha::Float64=1e-2, d::Float64=0.)
	In = shared_rand((M, T, V), Float32(M))
	Out = shared_rand((M, V), Float32(M))

	counts = shared_zeros(Float32, (T, V))
	frequencies = shared_zeros(Int64, (V,))
	
	path = shared_zeros(Int32, (max_length, V))
	code = shared_zeros(Int8, (max_length, V))

	code[:] = -1

	return VectorModel(frequencies, code, path, In, Out, alpha, d, counts)
end

VectorModel(max_length::Int64, V::Int64, M::Int64, T::Int64=1, alpha::Float64=1e-2, d::Float64=0.) = VectorModel(Shared, max_length, V, M, T, alpha, d)

function VectorModel(::Type{Local}, max_length::Int64, V::Int64, M::Int64, T::Int64=1, alpha::Float64=1e-2, d::Float64=0.)
	In = rand((M, T, V), Float32(M))
	Out = rand((M, V), Float32(M))

	counts = zeros(Float32, (T, V))
	frequencies = zeros(Int64, (V,))
	
	path = zeros(Int32, (max_length, V))
	code = zeros(Int8, (max_length, V))

	code[:] = -1

	return VectorModel(frequencies, code, path, In, Out, alpha, d, counts)
end

function get_huffman(freqs)
	V=length(freqs)
	nodes = build_huffman_tree(freqs)
	outputs = convert_huffman_tree(nodes, V)
end



function VectorModel(freqs::Array{Int64}, M::Int64, T::Int64=1, alpha::Float64=1e-2,
	d::Float64=0., huffman_outputs::Vector{HierarchicalOutput}=get_huffman(freqs);	
	locality=Shared)
	V = length(freqs)
	
	max_length = maximum(map(x -> length(x.code), huffman_outputs))

	vm = VectorModel(locality, max_length, V,M,T,alpha,d)
	vm.frequencies[:] = freqs


	for v in 1:V
		vm.code[:, v] = -1
		for i in 1:length(huffman_outputs[v])
			vm.code[i, v] = huffman_outputs[v].code[i]
			vm.path[i, v] = huffman_outputs[v].path[i]
		end
	end

	return vm 
end

view(vm::VectorModel, v::Integer, s::Integer) = view(vm.In, :, s, v)

function exp_normalize!(x)
	max_x = maximum(x)
	sum_x = 0.
	for i in 1:length(x)
		x[i] = exp(x[i] - max_x)
		sum_x += x[i]
	end
	for i in 1:length(x)
		x[i] /= sum_x
	end
end

const superlib =  Base.Libdl.find_library(Pkg.dir("AdaGram")*"/lib/superlib.so")
superlib!="" || error("Library: superlib Not Found")

include("kahan.jl")
include("skip_gram.jl")
include("stick_breaking.jl")
include("textutil.jl")
include("gradient.jl")
include("predict.jl")
include("util.jl")

export VectorModel, gradient!
export get_gradient, apply_gradient!
export V, T, M, L
export train_vectors!, inplace_train_vectors!
export vec, closest_words
export finalize!
export save_model, read_from_file, dict_from_file, build_from_file
export disambiguate, disambiguate!,  write_dictionary
export likelihood, parallel_likelihood
export expected_pi!, expected_pi
export load_model

export Dictionary
export Local, Shared

end
