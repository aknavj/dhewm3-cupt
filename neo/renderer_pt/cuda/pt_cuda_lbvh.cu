#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/device/device_radix_sort.cuh>

#include "renderer_pt/cuda/pt_cuda_device.h"

/*
================================================================================
Morton Code Utilities
================================================================================
*/

__device__ __forceinline__ unsigned int expandBits(unsigned int v) {
	// spread 10 bits across 30 bits with 2 gaps between each bit
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

__device__ __forceinline__ unsigned int morton3D(float x, float y, float z) {
	// quantize to 10-bit integers [0, 1023]
	unsigned int ix = (unsigned int)fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
	unsigned int iy = (unsigned int)fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
	unsigned int iz = (unsigned int)fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
	return (expandBits(iz) << 2) | (expandBits(iy) << 1) | expandBits(ix);
}

/*
================================================================================
Kernel 1: Compute Morton Codes
================================================================================
*/

__global__ void ComputeMortonCodesKernel(
	const ptTriangle_t* triangles,
	int numTriangles,
	unsigned int* mortonCodes,
	int* indices,
	float sceneMinX, float sceneMinY, float sceneMinZ,
	float sceneInvExtX, float sceneInvExtY, float sceneInvExtZ
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numTriangles) return;

	const ptTriangle_t& tri = triangles[idx];

	// Centroid from precomputed bounds
	float cx = (tri.bounds[0] + tri.bounds[3]) * 0.5f;
	float cy = (tri.bounds[1] + tri.bounds[4]) * 0.5f;
	float cz = (tri.bounds[2] + tri.bounds[5]) * 0.5f;

	// Normalize to [0, 1]
	float nx = (cx - sceneMinX) * sceneInvExtX;
	float ny = (cy - sceneMinY) * sceneInvExtY;
	float nz = (cz - sceneMinZ) * sceneInvExtZ;

	mortonCodes[idx] = morton3D(nx, ny, nz);
	indices[idx] = idx;
}

/*
================================================================================
Delta Function
================================================================================
*/

__device__ __forceinline__ int lbvhDelta(
	const unsigned int* sortedCodes,
	int numPrimitives,
	int i,
	int j
) {
	if (j < 0 || j >= numPrimitives) return -1;
	unsigned int ci = sortedCodes[i];
	unsigned int cj = sortedCodes[j];
	if (ci == cj) {
		// tiebreaker: use index XOR for duplicate Morton codes
		return 32 + __clz(i ^ j);
	}
	return __clz(ci ^ cj);
}

/*
================================================================================
Kernel 2: Build Radix Tree
================================================================================
*/

__global__ void BuildRadixTreeKernel(
	const unsigned int* sortedMortonCodes,
	int numPrimitives,
	ptBVHNode_t* nodes,
	int* parents
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numPrimitives - 1) return;

	int N = numPrimitives;

	// determine direction of the range (+1 or -1)
	int deltaRight = lbvhDelta(sortedMortonCodes, N, i, i + 1);
	int deltaLeft  = lbvhDelta(sortedMortonCodes, N, i, i - 1);
	int d = (deltaRight >= deltaLeft) ? 1 : -1;

	// compute upper bound for range length
	int deltaMin = lbvhDelta(sortedMortonCodes, N, i, i - d);
	int lmax = 2;
	while (lbvhDelta(sortedMortonCodes, N, i, i + lmax * d) > deltaMin) {
		lmax <<= 1;
		if (lmax > N) { lmax = N; break; }  // Safety bound
	}

	// binary search for the actual range length
	int l = 0;
	for (int t = lmax >> 1; t >= 1; t >>= 1) {
		if (lbvhDelta(sortedMortonCodes, N, i, i + (l + t) * d) > deltaMin) {
			l += t;
		}
	}
	int j = i + l * d;  // other end of the range

	// find the split position
	int deltaNode = lbvhDelta(sortedMortonCodes, N, i, j);
	int s = 0;
	int maxLen = l;
	// binary search within the range for the highest split point
	for (int div = 2; ; div <<= 1) {
		int t = (maxLen + div - 1) / div;
		if (t <= 0) break;
		if (lbvhDelta(sortedMortonCodes, N, i, i + (s + t) * d) > deltaNode) {
			s += t;
		}
		if (t == 1) break;
	}
	int gamma = i + s * d + min(d, 0);

	// assign children
	int first = min(i, j);
	int last  = max(i, j);

	// left child: covers [first, gamma]
	int leftChild;
	if (first == gamma) {
		leftChild = N - 1 + gamma;       // single leaf -> leaf node
	} else {
		leftChild = gamma;               // multiple leaves -> internal node
	}

	// right child: covers [gamma+1, last]
	int rightChild;
	if (last == gamma + 1) {
		rightChild = N - 1 + gamma + 1;  // single leaf -> leaf node
	} else {
		rightChild = gamma + 1;          // multiple leaves -> internal node
	}

	// store internal node
	nodes[i].leftChild  = leftChild;
	nodes[i].rightChild = rightChild;
	nodes[i].primitiveCount = 0;  // internal node — no direct primitives
	nodes[i].firstPrimitive = 0;

	// store parent pointers (children -> this node)
	parents[leftChild]  = i;
	parents[rightChild] = i;
}

/*
================================================================================
Kernel 3: Initialize Leaf Nodes
================================================================================
*/

__global__ void InitLeafNodesKernel(
	const ptTriangle_t* triangles,
	const int* sortedTriIndices,
	ptBVHNode_t* nodes,
	int numPrimitives
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numPrimitives) return;

	int nodeIdx = numPrimitives - 1 + i;
	int triIdx  = sortedTriIndices[i];

	// mark as leaf
	nodes[nodeIdx].leftChild  = -1;
	nodes[nodeIdx].rightChild = -1;
	nodes[nodeIdx].firstPrimitive = i;       // offset into triIndices
	nodes[nodeIdx].primitiveCount = 1;

	// copy bounds with epsilon padding
	const float EPSILON = 0.001f;
	const ptTriangle_t& tri = triangles[triIdx];
	nodes[nodeIdx].bounds[0] = tri.bounds[0] - EPSILON;
	nodes[nodeIdx].bounds[1] = tri.bounds[1] - EPSILON;
	nodes[nodeIdx].bounds[2] = tri.bounds[2] - EPSILON;
	nodes[nodeIdx].bounds[3] = tri.bounds[3] + EPSILON;
	nodes[nodeIdx].bounds[4] = tri.bounds[4] + EPSILON;
	nodes[nodeIdx].bounds[5] = tri.bounds[5] + EPSILON;
}

__global__ void BuildEscapeLinksKernel(ptBVHNode_t* nodes, const int* parents, int numNodes) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numNodes) return;

	int child = idx;
	int parent = parents[child];
	int escape = -1;
	while (parent >= 0) {
		if (nodes[parent].leftChild == child) {
			escape = nodes[parent].rightChild;
			break;
		}
		child = parent;
		parent = parents[child];
	}
	nodes[idx].escapeIndex = escape;
}

/*
================================================================================
Kernel 4: Bottom-Up AABB Propagation
================================================================================
*/

__global__ void ComputeNodeBoundsKernel(
	ptBVHNode_t* nodes,
	const int* parents,
	int* atomicCounters,
	int numPrimitives
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numPrimitives) return;

	// start at leaf node
	int parent = parents[numPrimitives - 1 + i];

	while (parent >= 0) {
		// publish this subtree's bounds before signalling arrival, otherwise the
		// sibling thread may merge stale data
		__threadfence();

		// first child to arrive increments 0->1 and exits,
		// the second one (1->2) merges both children
		if (atomicAdd(&atomicCounters[parent], 1) == 0) {
			return;
		}

		// the sibling's bounds were written by another SM in this launch; bypass
		// the non-coherent L1 and read them from L2
		const int left  = __ldcg(&nodes[parent].leftChild);
		const int right = __ldcg(&nodes[parent].rightChild);

		float lb[6], rb[6];
		for (int k = 0; k < 6; k++) {
			lb[k] = __ldcg(&nodes[left].bounds[k]);
			rb[k] = __ldcg(&nodes[right].bounds[k]);
		}

		nodes[parent].bounds[0] = fminf(lb[0], rb[0]);
		nodes[parent].bounds[1] = fminf(lb[1], rb[1]);
		nodes[parent].bounds[2] = fminf(lb[2], rb[2]);
		nodes[parent].bounds[3] = fmaxf(lb[3], rb[3]);
		nodes[parent].bounds[4] = fmaxf(lb[4], rb[4]);
		nodes[parent].bounds[5] = fmaxf(lb[5], rb[5]);

		parent = parents[parent];
	}
}

/*
================================================================================
Kernel 5: Single-Triangle Fallback
================================================================================
*/

__global__ void InitSingleLeafKernel(
	const ptTriangle_t* triangles,
	ptBVHNode_t* nodes,
	int* triIndices
) {
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

	triIndices[0] = 0;

	nodes[0].leftChild  = -1;
	nodes[0].rightChild = -1;
	nodes[0].firstPrimitive = 0;
	nodes[0].primitiveCount = 1;
	nodes[0].escapeIndex = -1;

	const float EPSILON = 0.001f;
	nodes[0].bounds[0] = triangles[0].bounds[0] - EPSILON;
	nodes[0].bounds[1] = triangles[0].bounds[1] - EPSILON;
	nodes[0].bounds[2] = triangles[0].bounds[2] - EPSILON;
	nodes[0].bounds[3] = triangles[0].bounds[3] + EPSILON;
	nodes[0].bounds[4] = triangles[0].bounds[4] + EPSILON;
	nodes[0].bounds[5] = triangles[0].bounds[5] + EPSILON;
}

// morton3D produces 30 bit codes, the radix sort can skip the top two bits
#define LBVH_MORTON_BITS	30

/*
================================================================================
LBVHSortTempBytes
================================================================================
*/
extern "C" size_t LBVHSortTempBytes(int numTriangles) {
	if (numTriangles <= 1) return 0;
	size_t bytes = 0;
	cub::DoubleBuffer<unsigned int> keys(NULL, NULL);
	cub::DoubleBuffer<int> vals(NULL, NULL);
	cub::DeviceRadixSort::SortPairs(NULL, bytes, keys, vals, numTriangles, 0, LBVH_MORTON_BITS);
	return bytes;
}

/*
================================================================================
Launch Wrapper: LaunchBuildLBVH

All work stays on the GPU and is ordered on 'stream'. The host only
provides the scene bounds (computed by idPathTracerScene).
================================================================================
*/

extern "C" void LaunchBuildLBVH(
	const ptTriangle_t* d_triangles,
	int numTriangles,
	ptBVHNode_t* d_bvhNodes,
	int* d_triIndices,
	unsigned int* d_mortonCodes,
	unsigned int* d_mortonAlt,
	int* d_sortedIndices,
	void* d_sortTemp,
	size_t sortTempBytes,
	int* d_parents,
	int* d_atomicCounters,
	const float* sceneBounds,
	cudaStream_t stream
) {
	if (numTriangles <= 0) return;

	// single triangle
	if (numTriangles == 1) {
		InitSingleLeafKernel<<<1, 1, 0, stream>>>(d_triangles, d_bvhNodes, d_triIndices);
		return;
	}

	// scene extent for normalization
	float sceneMin[3] = { sceneBounds[0], sceneBounds[1], sceneBounds[2] };
	float sceneMax[3] = { sceneBounds[3], sceneBounds[4], sceneBounds[5] };
	float invExt[3];
	for (int i = 0; i < 3; i++) {
		float ext = sceneMax[i] - sceneMin[i];
		invExt[i] = (ext > 1e-6f) ? (1.0f / ext) : 0.0f;
	}

	const int BLOCK = 256;
	int grid = (numTriangles + BLOCK - 1) / BLOCK;
	int numInternalNodes = numTriangles - 1;
	int numNodes = 2 * numTriangles - 1;

	// morton codes
	ComputeMortonCodesKernel<<<grid, BLOCK, 0, stream>>>(
		d_triangles, numTriangles,
		d_mortonCodes, d_sortedIndices,
		sceneMin[0], sceneMin[1], sceneMin[2],
		invExt[0], invExt[1], invExt[2]
	);

	// radix sort by Morton code on our stream, using the backend's persistent scratch
	// buffer (thrust allocated and freed temporary storage on every build)
	cub::DoubleBuffer<unsigned int> keys(d_mortonCodes, d_mortonAlt);
	cub::DoubleBuffer<int> vals(d_sortedIndices, d_triIndices);
	cub::DeviceRadixSort::SortPairs(d_sortTemp, sortTempBytes, keys, vals, numTriangles, 0, LBVH_MORTON_BITS, stream);

	const unsigned int* sortedCodes = keys.Current();
	if (vals.Current() != d_triIndices) {
		cudaMemcpyAsync(d_triIndices, vals.Current(), numTriangles * sizeof(int), cudaMemcpyDeviceToDevice, stream);
	}

	// initialize parent array to -1
	cudaMemsetAsync(d_parents, 0xFF, numNodes * sizeof(int), stream);

	// initialize atomic counters to 0
	cudaMemsetAsync(d_atomicCounters, 0, numInternalNodes * sizeof(int), stream);

	// build radix tree topology
	int treeGrid = (numInternalNodes + BLOCK - 1) / BLOCK;
	BuildRadixTreeKernel<<<treeGrid, BLOCK, 0, stream>>>(
		sortedCodes, numTriangles, d_bvhNodes, d_parents
	);

	// initialize leaf nodes
	InitLeafNodesKernel<<<grid, BLOCK, 0, stream>>>(
		d_triangles, d_triIndices, d_bvhNodes, numTriangles
	);

	const int nodeGrid = (numNodes + BLOCK - 1) / BLOCK;
	BuildEscapeLinksKernel<<<nodeGrid, BLOCK, 0, stream>>>(d_bvhNodes, d_parents, numNodes);

	// bottom-up AABB propagation
	ComputeNodeBoundsKernel<<<grid, BLOCK, 0, stream>>>(
		d_bvhNodes, d_parents, d_atomicCounters, numTriangles
	);
}
