// Diagnostic permutation of an already-built candidate list. No geometry is queried here.
__device__ void order_candidates(cg::grid_group all, int worker, State &f, const MeshView &mesh,
                                 const PatchIndex &patch, int *scratch) {
    if (f.nc < 2)
        return;
    int n = mesh.n, words = (n + 31) / 32, stride = gridDim.x * blockDim.x;
    const int *sortedIds = scratch;
    int *epoch = scratch + n;
    unsigned *masks = reinterpret_cast<unsigned *>(epoch + n);
    int *offsets = reinterpret_cast<int *>(masks + words);
    // The rebuild number marks current membership without clearing an entire array at each pose.
    for (int i = worker; i < f.nc; i += stride)
        epoch[patch.active[i]] = f.rebuilds;
    all.sync();
    int lane = worker % 32;
    for (int w = worker / 32; w < words; w += stride / 32) {
        int rank = 32 * w + lane;
        bool keep = rank < n && epoch[sortedIds[rank]] == f.rebuilds;
        unsigned mask = __ballot_sync(0xffffffff, keep);
        if (!lane)
            masks[w] = mask;
    }
    all.sync();
    if (!worker) {
        int total = 0;
        for (int w = 0; w < words; ++w) {
            offsets[w] = total;
            total += __popc(masks[w]);
        }
        // The final scratch word is a sticky check for duplicated or missing IDs.
        if (total != f.nc)
            offsets[words] = 1;
    }
    all.sync();
    for (int rank = worker; rank < n; rank += stride) {
        unsigned mask = masks[rank / 32];
        if (mask & (1u << lane))
            patch.active[offsets[rank / 32] + __popc(mask & ((1u << lane) - 1))] = sortedIds[rank];
    }
    all.sync();
}
