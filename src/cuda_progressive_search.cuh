// Each warp advances one ordered face group without a global round barrier.
__global__ void progressive_round(BatchControl *control) {
    auto &g=*control;
    if(!batch_enabled(g)) return;
    const int lane=threadIdx.x;
    for(int job=blockIdx.x;job<g.offsets[g.scene.np];job+=gridDim.x) {
        const int pair=batch_owner(g.offsets,g.scene.np,job), id=job-g.offsets[pair];
        const auto &q=g.pairs[pair].query; auto &w=*q.work;
        auto &s=w.search_progress[id];
        const auto group=w.group[id];
        if(!lane) {
            const auto &seed=w.seed_attempts[id];
            s.next=group.begin; s.done=seed.skip || seed.hit || seed.fault.code || group.begin==group.end;
        }
        __syncwarp();
        while(!s.done) {
            const int begin=s.next, width=32;
            const int end=min(begin+width,group.end), k=begin+lane;
            bool stop=false;
            if(k<end) {
                const auto a=q.proposal_attempt(group,k);
                w.attempts[k]=a;
                stop=a.hit || a.fault.code;
            }
            const unsigned stops=__ballot_sync(0xffffffffu,stop);
            const int winner=stops?begin+__ffs(stops)-1:group.end;
            unsigned unused=__ballot_sync(0xffffffffu,k<end && k>winner && w.attempts[k].projected);
            unsigned saved=0;
            if(stops) {
                // The preceding two-batch implementation executes all of the first
                // 32 candidates; a later winner also requires its entire tail.
                const int old_end=winner<group.begin+32?min(group.begin+32,group.end):group.end;
                for(int j=end+lane;j<old_end;j+=32) saved+=w.unique_proposals[j]!=0;
            }
            for(int shift=16;shift;shift/=2) saved+=__shfl_down_sync(0xffffffffu,saved,shift);
            __syncwarp();
            if(!lane) {
                s.next=end; s.done=stops || end==group.end;
                atomicAdd(&w.projection_rounds,1ULL);
                atomicAdd(&w.avoided_projections,static_cast<unsigned long long>(saved));
                atomicAdd(&w.unused_projections,static_cast<unsigned long long>(__popc(unused)));
            }
            __syncwarp();
        }
    }
}
