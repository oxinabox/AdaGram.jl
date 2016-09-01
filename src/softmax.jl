using Base.Collections
using Base.Order

type HierarchicalSoftmaxNode
	parent::Int32
	branch::Bool
end

type HierarchicalOutput
	code::Array{Int8}
	path::Array{Int}
end

import Base.length

length(out::HierarchicalOutput) = length(out.path)

function HierarchicalSoftmaxNode()
	return HierarchicalSoftmaxNode(Int32(0), false)
end

function softmax_path(nodes::Array{HierarchicalSoftmaxNode},
		vocab_len::Integer, id::Integer)
	Task() do
		while true
			node = nodes[id]
			if node.parent == 0 break; end
			@assert node.parent > vocab_len
			produce((Int32(node.parent - vocab_len), node.branch))
			id = node.parent
		end
	end
end

function build_huffman_tree{Tf <: Number}(freqs::Array{Tf})
	vocab_len = length(freqs)
	nodes = Array(HierarchicalSoftmaxNode, vocab_len)
	for v in 1:vocab_len
		nodes[v] = HierarchicalSoftmaxNode()
	end

	freq_ord = By(wf -> wf[2])
	heap = heapify!([(nodes[v], freqs[v]) for v in 1:vocab_len], freq_ord)

	function pop_initialize!(parent::Int, branch::Bool)
		node = heappop!(heap, freq_ord)
		node[1].parent = Int32(parent)
		node[1].branch = branch
		return node[2]
	end

	id = vocab_len
	while length(heap) > 1
		id += 1
		node = HierarchicalSoftmaxNode()
		push!(nodes, node)

		freq = pop_initialize!(id, true) + pop_initialize!(id, false)
		heappush!(heap, (node, freq), freq_ord)
	end

	@assert length(heap) == 1

	return nodes
end

function convert_huffman_tree(nodes::Array{HierarchicalSoftmaxNode}, vocab_len::Integer)
	outputs = Array(HierarchicalOutput, vocab_len)
	for layer_num in 1:vocab_len
		code = Array(Int8, 0)
		path = Array(Int, 0)

		for (parent_id, branch) in softmax_path(nodes, vocab_len, layer_num)
			push!(code, Int8(branch))
			push!(path, parent_id)
		end

		outputs[v] = HierarchicalOutput(code, path)
	end

	return outputs
end

export HierarchicalSoftmaxNode
