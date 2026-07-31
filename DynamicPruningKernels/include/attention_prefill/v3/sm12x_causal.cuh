#pragma once
#include "attention_prefill/v3/traits.cuh"

// main kernel func
template <
    uint32_t kHq, uint32_t kHk, uint32_t kD, uint32_t kNG, bool kIsCausal, bool kIsLeftpad,
    uint32_t kBM, uint32_t kBN, uint32_t kPipelineK, uint32_t kPipelineV, uint32_t kPipelineReg,
    typename TMADescK, typename TMADescV, class WarpLayoutTraits_, class MMATraits_,
    bool kThrSkipG2S, bool kWarpSkipG2S, bool kWarpSkipMMA, typename DType, uint32_t kWarps=4
>
__global__ __launch_bounds__(256) void attention_head_v3_causal(const __grid_constant__ AttentionParams<TMADescK, TMADescV> params) {
    using namespace cute;
    const auto& [Q, K, V, Mask, Index, Leftpad, O, B, Tq, Tk, scale, tma_desc_k, tma_desc_v] = params;
    constexpr uint32_t Hq = kHq;
    constexpr uint32_t Hk = kHk;
    constexpr uint32_t D = kD;
    constexpr uint32_t NG = kNG;
    constexpr bool IsCausal = kIsCausal;
    constexpr bool IsLeftpad = kIsLeftpad;
    constexpr uint32_t BM = kBM;
    constexpr uint32_t BN = kBN;
    constexpr uint32_t PipelineK = kPipelineK;
    constexpr uint32_t PipelineV = kPipelineV;
    constexpr uint32_t PipelineReg = kPipelineReg;

    constexpr uint32_t G2SQColPerCTA = WarpLayoutTraits_::G2SQColPerCTA;
    constexpr uint32_t threadsPerCTA = WarpLayoutTraits_::threadsPerCTA;
    constexpr uint32_t MMARow = WarpLayoutTraits_::MMARow;
    constexpr uint32_t MMACol = WarpLayoutTraits_::MMACol;
    constexpr uint32_t MMAIter = WarpLayoutTraits_::MMAIter;

    using tiled_mma_qk = typename MMATraits_::tiled_mma_qk;
    using tiled_mma_pv = typename MMATraits_::tiled_mma_pv;
    using sQLayout = typename MMATraits_::sQLayout;
    using sKLayout = typename MMATraits_::sKLayout;
    using sVLayout = typename MMATraits_::sVLayout;
    using sOLayout = typename MMATraits_::sOLayout;
    using g2sQ_tiled_copy_s = typename MMATraits_::g2sQ_tiled_copy_s;
    using g2sQ_tiled_copy_g = typename MMATraits_::g2sQ_tiled_copy_g;
    using s2rQ_tiled_copy = typename MMATraits_::s2rQ_tiled_copy;
    using s2rK_tiled_copy = typename MMATraits_::s2rK_tiled_copy;
    using s2rV_tiled_copy = typename MMATraits_::s2rV_tiled_copy;
    using r2sO_tiled_copy = typename MMATraits_::r2sO_tiled_copy;
    using s2gO_tiled_copy_s = typename MMATraits_::s2gO_tiled_copy_s;
    using s2gO_tiled_copy_g = typename MMATraits_::s2gO_tiled_copy_g;

    // grid: (cdiv(Tq, BM), Hq, B)
    const uint32_t tidx = threadIdx.x;
    const uint32_t bidx = blockIdx.x;
    const uint32_t bidy = blockIdx.y;
    const uint32_t bidz = blockIdx.z;
    const uint32_t warp_id = tidx / device::kWarpThreads;
    const uint32_t lane_id = tidx % device::kWarpThreads;

    constexpr uint32_t kv_group_size = Hq / Hk;
    constexpr uint32_t group_size = Hq / NG;
    const uint32_t query_head = bidy;
    const uint32_t key_head = bidy / kv_group_size;
    const uint32_t group_id = query_head / group_size;
    const float scale_ln2 = scale * 1.44269504;

    const uint32_t base_off_m = bidx * BM;
    const uint32_t thread_off_m_1 = (warp_id / MMACol) * 16 + lane_id / 4;
    const uint32_t thread_off_m_2 = thread_off_m_1 + 8; // CTA-local row, aligned with mma layout
    const uint32_t max_off_m = cute::min(Tq, base_off_m + BM);
    constexpr uint32_t MMARowPerCTA = MMARow * 16;

    //
    // prepare tensors and smem layout
    //
    Tensor mQ = make_tensor(
        make_gmem_ptr(static_cast<const DType*>(Q)),
        make_ordered_layout(
            make_shape(Tq, Int<D>{}, Int<Hq>{}, B),
            make_step(_2{}, _0{}, _1{}, _3{})
        )
    );
    Tensor mK = make_tensor(
        make_gmem_ptr(static_cast<const DType*>(K)),
        make_ordered_layout(
            make_shape(Tk, Int<D>{}, Int<Hk>{}, B),
            make_step(_2{}, _0{}, _1{}, _3{})
        )
    );
    Tensor mV = make_tensor(
        make_gmem_ptr(static_cast<const DType*>(V)),
        make_ordered_layout(
            make_shape(Tk, Int<D>{}, Int<Hk>{}, B),
            make_step(_2{}, _0{}, _1{}, _3{})
        )
    );
    Tensor mO = make_tensor(
        make_gmem_ptr(static_cast<DType*>(O)),
        make_ordered_layout(
            make_shape(Tq, Int<D>{}, Int<Hq>{}, B),
            make_step(_2{}, _0{}, _1{}, _3{})
        )
    );
    Tensor mMask = make_tensor(
        make_gmem_ptr(static_cast<const uint8_t*>(Mask)),
        make_layout(make_shape(B, Tq, Int<NG>{}), LayoutRight{})
    );
    Tensor mIndex = make_tensor(
        make_gmem_ptr(reinterpret_cast<const uint32_t*>(Index)),
        make_layout(
            make_shape(B, Tq, Int<NG>{}),
            make_stride(Tq * NG * 2, Int<NG * 2>{}, _2{})
        )
    ); // skip high 32-bit
    Tensor mLeftpad = make_tensor(
        make_gmem_ptr(static_cast<const uint32_t*>(Leftpad)),
        make_layout(make_shape(B), LayoutRight{})
    );

    extern __shared__ uint8_t shared_memory[];
    using SharedStorageKV = SharedMemoryKV<DType, sKLayout, sVLayout, PipelineK, PipelineV>;
    SharedStorageKV &smem_kv = *reinterpret_cast<SharedStorageKV*>(shared_memory);

    using SharedStorageQ = SharedMemoryQ<DType, sQLayout, kWarps>;
    using SharedStorageO = SharedMemoryO<DType, sOLayout>;
    SharedStorageQ &smem_q = *reinterpret_cast<SharedStorageQ*>(shared_memory);
    SharedStorageO &smem_o = *reinterpret_cast<SharedStorageO*>(shared_memory);

    Tensor sK = make_tensor(
        make_smem_ptr(smem_kv.K.begin()),
        sKLayout{}
    );
    Tensor sV = make_tensor(
        make_smem_ptr(smem_kv.V.begin()),
        sVLayout{}
    );
    Tensor sQ = make_tensor(
        make_smem_ptr(smem_q.Q.begin()),
        sQLayout{}
    );
    Tensor sO = make_tensor(
        make_smem_ptr(smem_o.O.begin()),
        sOLayout{}
    );

    // load mask & index
    SkipHelper<MMAIter, kWarpSkipG2S, kWarpSkipMMA> skip_helper;
    update_skip_metadata<kWarpSkipG2S, kWarpSkipMMA, MMARow, MMACol, MMAIter, decltype(mMask), decltype(mIndex)>(
        skip_helper, mMask, mIndex, B, Tq, max_off_m, base_off_m, group_id, warp_id, lane_id, bidz
    );
    if (skip_helper.execute_cta == 0) return;

    uint32_t producer_k = 0;
    uint32_t producer_v = 0;

    //
    // prologue, load Q from gmem -> smem -> RF
    //
    Tensor gQ = local_tile(
        mQ,
        make_tile(_1{}, Int<D>{}),
        make_coord(_, 0, query_head, bidz)
    );
    auto g2sQ_thr_copy_g = g2sQ_tiled_copy_g{}.get_slice(tidx % G2SQColPerCTA);
    auto g2sQ_thr_copy_s = g2sQ_tiled_copy_s{}.get_slice(tidx % G2SQColPerCTA);
    auto pSgQ = g2sQ_thr_copy_g.partition_S(gQ);
    auto pDsQ = g2sQ_thr_copy_s.partition_D(sQ);

    CUTE_UNROLL
    for (uint32_t i=0, off1=thread_off_m_1, off2=thread_off_m_2; i < MMAIter; ++i, off1+=MMARowPerCTA, off2+=MMARowPerCTA) {
        if constexpr (kThrSkipG2S) {
            if (skip_helper.rMask[i*2]) copy(g2sQ_tiled_copy_g{}, pSgQ(_, _0{}, _, skip_helper.rIndex[i*2]), pDsQ(_, off1, _));
            if (skip_helper.rMask[i*2+1]) copy(g2sQ_tiled_copy_g{}, pSgQ(_, _0{}, _, skip_helper.rIndex[i*2+1]), pDsQ(_, off2, _));
        }
        else if constexpr (kWarpSkipG2S) {
            if (skip_helper.execute_g2s_warp[i*2]) copy(g2sQ_tiled_copy_g{}, pSgQ(_, _0{}, _, skip_helper.rIndex[i*2]), pDsQ(_, off1, _));
            if (skip_helper.execute_g2s_warp[i*2+1]) copy(g2sQ_tiled_copy_g{}, pSgQ(_, _0{}, _, skip_helper.rIndex[i*2+1]), pDsQ(_, off2, _));
        }
        else {
            copy(g2sQ_tiled_copy_g{}, pSgQ(_, _0{}, _, skip_helper.rIndex[i*2]), pDsQ(_, off1, _));
            copy(g2sQ_tiled_copy_g{}, pSgQ(_, _0{}, _, skip_helper.rIndex[i*2+1]), pDsQ(_, off2, _));
        }
    }
    cp_async_fence();

    // fill the pipeline for K & V
    // 1. get Tk start and end
    const uint32_t key_start = IsLeftpad ? mLeftpad(make_coord(bidz)) : 0;
    uint32_t key_end = Tk;
    uint32_t cta_min_query = 0;
    uint32_t cta_max_query = 0;

    // count causal range
    uint32_t local_max_query = 0;
    uint32_t local_min_query = Tk;
    CUTE_UNROLL
    for (uint32_t i=0; i < MMAIter * 2; ++i) {
        if (skip_helper.rMask[i]) {
            local_max_query = cute::max(local_max_query, skip_helper.rIndex[i]);
            local_min_query = cute::min(local_min_query, skip_helper.rIndex[i]);
        }
    }
    uint32_t warp_max_query = __reduce_max_sync(0xffffffff, local_max_query);
    uint32_t warp_min_query = __reduce_min_sync(0xffffffff, local_min_query);

    if (lane_id == 0) {
        smem_q.cta_reduce_cache[warp_id] = warp_max_query;
        smem_q.cta_reduce_cache[kWarps + warp_id] = warp_min_query;
    }
    __syncthreads();

    uint32_t x_max = 0;
    uint32_t x_min = Tk;
    if (warp_id == 0 && lane_id < kWarps) {
        x_max = smem_q.cta_reduce_cache[lane_id];
        x_min = smem_q.cta_reduce_cache[kWarps + lane_id];
    }
    if (warp_id == 0) {
        x_max = __reduce_max_sync(0xffffffff, x_max);
        x_min = __reduce_min_sync(0xffffffff, x_min);
        if (lane_id == 0) {
            smem_q.cta_reduce_cache[kWarps * 2] = x_max;
            smem_q.cta_reduce_cache[kWarps * 2 + 1] = x_min;
        }
    }
    __syncthreads();
    cta_max_query = smem_q.cta_reduce_cache[kWarps * 2];
    cta_min_query = smem_q.cta_reduce_cache[kWarps * 2 + 1];
    key_end = cute::min(Tk, cta_max_query + 1);

    // K & V load helper
    auto tma_mK = tma_desc_k.get_tma_tensor(make_shape(Tk, Int<D>{}, Int<Hk>{}, B));
    auto tma_mV = tma_desc_v.get_tma_tensor(make_shape(Int<D>{}, Tk, Int<Hk>{}, B));

    auto tma_k_thr = tma_desc_k.get_slice(_0{});
    auto tma_v_thr = tma_desc_v.get_slice(_0{});

    auto gK = local_tile(
        tma_mK,
        make_tile(Int<BN>{}, Int<D>{}),
        make_coord(_, _0{}, key_head, bidz)
    );
    auto gV = local_tile(
        tma_mV,
        make_tile(Int<D>{}, Int<BN>{}),
        make_coord(_0{}, _, key_head, bidz)
    );
    auto pSgK = tma_k_thr.partition_S(gK);
    auto pDsK = tma_k_thr.partition_D(sK);
    auto pSgV = tma_v_thr.partition_S(gV);
    auto pDsV = tma_v_thr.partition_D(sV);

    const uint32_t key_start_tile = key_start / BN;
    const uint32_t key_end_tile = cute::ceil_div(key_end, BN);
    const uint32_t key_tile_num = key_end_tile - key_start_tile;
    constexpr size_t kv_load_bytes = BN * D * sizeof(DType);

    auto tma_issue_k = [&](uint32_t tile_pos, uint32_t write_stage) {
        if (warp_id == 0 && cute::elect_one_sync()) {
            set_barrier_transaction_bytes(
                smem_kv.tma_k_barrier[write_stage],
                kv_load_bytes
            );
            copy(
                tma_desc_k.with(smem_kv.tma_k_barrier[write_stage]),
                pSgK(_, _0{}, _0{}, tile_pos), pDsK(_, _0{}, _0{}, write_stage)
            );
        }
    };
    auto tma_issue_v = [&](uint32_t tile_pos, uint32_t write_stage) {
        if (warp_id == 0 && cute::elect_one_sync()) {
            set_barrier_transaction_bytes(
                smem_kv.tma_v_barrier[write_stage],
                kv_load_bytes
            );
            copy(
                tma_desc_v.with(smem_kv.tma_v_barrier[write_stage]),
                pSgV(_, _0{}, _0{}, tile_pos), pDsV(_, _0{}, _0{}, write_stage)
            );
        }
    };

    // wait gmem -> smem, then LDSM to rQ
    ThrMMA thr_mma_qk = tiled_mma_qk{}.get_slice(tidx);
    ThrMMA thr_mma_pv = tiled_mma_pv{}.get_slice(tidx);
    auto rQ = thr_mma_qk.partition_fragment_A(sQ);
    auto rK = thr_mma_qk.partition_fragment_B(sK(_, _, _0{}));
    auto rV = thr_mma_pv.partition_fragment_B(sV(_, _, _0{}));

    Tensor rP_logical = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
    auto rP = thr_mma_qk.partition_fragment_C(rP_logical);
    auto cP = thr_mma_qk.partition_C(rP_logical);

    auto rP_operand = make_tensor(
        static_cast<DType*>(nullptr),
        make_layout(make_shape(Int<BM>{}, Int<BN>{}), LayoutRight{})
    );
    auto rPlA = thr_mma_pv.partition_fragment_A(rP_operand);

    Tensor gO_tile = local_tile(
        mO,
        make_tile(Int<BM>{}, Int<D>{}),
        make_coord(bidx, _0{}, query_head, bidz)
    );
    auto rO = thr_mma_pv.partition_fragment_C(gO_tile); // acc fragment
    clear(rO);

    auto rM = make_fragment_like<float>(rP(make_coord(_0{}, _), _, _0{}));
    auto rL = make_fragment_like<float>(rP(make_coord(_0{}, _), _, _0{}));
    fill(rM, -MAXFLOAT);
    clear(rL);

    auto s2rQ_thr_copy = s2rQ_tiled_copy{}.get_slice(tidx);
    auto pSsQ = s2rQ_thr_copy.partition_S(sQ);
    auto pDrQ = s2rQ_thr_copy.retile_D(rQ);

    auto s2rK_thr_copy = s2rK_tiled_copy{}.get_slice(tidx);
    auto pSsK = s2rK_thr_copy.partition_S(sK);
    auto pDrK = s2rK_thr_copy.retile_D(rK);

    auto s2rV_thr_copy = s2rV_tiled_copy{}.get_slice(tidx);
    auto pSsV = s2rV_thr_copy.partition_S(sV);
    auto pDrV = s2rV_thr_copy.retile_D(rV);

    cp_async_wait<0>();
    __syncthreads();
    copy(s2rQ_tiled_copy{}, pSsQ, pDrQ);
    __syncthreads();

    // Q * scale
    // f16 version
    auto scale2_rn = Float22Half2<DType, true>()(scale_ln2, scale_ln2);
    if constexpr (is_same_v<DType, __half>) {
        auto rQ_pack = recast<__half2>(rQ);
        for (uint32_t i=0; i < size(rQ_pack); ++i) {
            rQ_pack(i) = __hmul2_rn(rQ_pack(i), scale2_rn);
        }
    }
    else {
        auto rQ_pack = recast<__nv_bfloat162>(rQ);
        for (uint32_t i=0; i < size(rQ_pack); ++i) {
            rQ_pack(i) = __hmul2_rn(rQ_pack(i), scale2_rn);
        }
    }

    // float version
    // if constexpr (is_same_v<DType, __half>) {
    //     auto rQ_cast = recast<__half>(rQ);
    //     auto scale_cast = __float2half_rn(scale_ln2);
    //     CUTE_UNROLL
    //     for (uint32_t i=0; i < size(rQ_cast); ++i) rQ_cast(i) = rQ_cast(i) * scale_cast;
    // }
    // else {
    //      auto rQ_cast = recast<__nv_bfloat16>(rQ);
    //     auto scale_cast = __float2bfloat16_rn(scale_ln2);
    //     CUTE_UNROLL
    //     for (uint32_t i=0; i < size(rQ_cast); ++i) rQ_cast(i) = rQ_cast(i) * scale_cast;
    // }

    // fill the pipeline for K & V
    // init barrier
    if (tidx == 0) {
        CUTE_UNROLL
        for (uint32_t i=0; i < PipelineK; ++i) {
            cute::initialize_barrier(smem_kv.tma_k_barrier[i]);
        }
        CUTE_UNROLL
        for (uint32_t i=0; i < PipelineV; ++i) {
            cute::initialize_barrier(smem_kv.tma_v_barrier[i]);
        }
    }
    __syncthreads();

    CUTE_UNROLL
    for (int32_t p=0; p < cute::max(PipelineK, PipelineV) - 1; ++p) {
        if (producer_k < key_tile_num && p < PipelineK - 1) {
            tma_issue_k(producer_k + key_start_tile, pipe_stage<PipelineK>(producer_k));
            ++producer_k;
        }
        if (producer_v < key_tile_num && p < PipelineV - 1) {
            tma_issue_v(producer_v + key_start_tile, pipe_stage<PipelineV>(producer_v));
            ++producer_v;
        }
    }

    // mainloop
    CUTE_NO_UNROLL
    for (uint32_t k_tile=0; k_tile < key_tile_num; ++k_tile) {
        const uint32_t tile_start = (key_start_tile + k_tile) * BN;
        const uint32_t tile_end = tile_start + BN;
        const bool tile_all_valid = (tile_start >= key_start) && (tile_end <= key_end) && (tile_end - 1 <= cta_min_query);

        if (producer_k < key_tile_num) {
            tma_issue_k(producer_k + key_start_tile, pipe_stage<PipelineK>(producer_k));
            ++producer_k;
        }
        if (producer_v < key_tile_num) {
            tma_issue_v(producer_v + key_start_tile, pipe_stage<PipelineV>(producer_v));
            ++producer_v;
        }
        clear(rP);

        // wait until consumer's K block arrive
        const uint32_t read_stage_k = pipe_stage<PipelineK>(k_tile);
        const uint32_t read_phase_k = pipe_phase<PipelineK>(k_tile);
        wait_barrier(smem_kv.tma_k_barrier[read_stage_k], read_phase_k);
        __syncthreads();

        // S2R for Key, then Q@K^T
        copy(s2rK_tiled_copy{}, pSsK(_, _, _, make_coord(0, read_stage_k)), pDrK);
        CUTE_UNROLL
        for (uint32_t i=0; i < size<1>(rQ); ++i) {
            if constexpr (kWarpSkipMMA) {
                if (skip_helper.execute_mma_warp[i]) {
                    CUTE_UNROLL
                    for (uint32_t j=0; j < size<1>(rK); ++j) {
                        CUTE_UNROLL
                        for (uint32_t k=0; k < size<2>(rK); ++k) {
                            gemm(tiled_mma_qk{}, rQ(_, i, k), rK(_, j, k), rP(_, i, j));
                        }
                    }
                }
            }
            else {
                CUTE_UNROLL
                for (uint32_t j=0; j < size<1>(rK); ++j) {
                    CUTE_UNROLL
                    for (uint32_t k=0; k < size<2>(rK); ++k) {
                        gemm(tiled_mma_qk{}, rQ(_, i, k), rK(_, j, k), rP(_, i, j));
                    }
                }
            }
        }
        // row-wise apply mask & partial softmax update
        if (tile_all_valid) {
            CUTE_UNROLL
            for (uint32_t mi=0; mi < size<1>(rP); ++mi) {
                CUTE_UNROLL
                for (uint32_t mma_i=0; mma_i < size<0>(rP) / 2; ++mma_i) {
                    float rm = -MAXFLOAT;
                    const uint32_t local_row = mi * 2 + mma_i;
                    // update max
                    CUTE_UNROLL
                    for (uint32_t ni=0; ni < size<2>(rP); ++ni) {
                        rm = cute::max(rm, cute::max(rP(mma_i * 2, mi, ni), rP(mma_i * 2 + 1, mi, ni)));
                    }

                    rm = cute::max(rm, __shfl_xor_sync(0xffffffff, rm, 1, 4));
                    rm = cute::max(rm, __shfl_xor_sync(0xffffffff, rm, 2, 4));
                    float old_max = rM(local_row);
                    float new_max = cute::max(rm, old_max);
                    const bool rescale_acc = new_max > old_max;
                    rM(local_row) = new_max;

                    if (rescale_acc && old_max != -MAXFLOAT) {
                        const float score_scale = exp2f(old_max - new_max);
                        rL(local_row) *= score_scale;
                        CUTE_UNROLL
                        for (uint32_t ni=0; ni < size<2>(rO); ++ni) {
                            rO(mma_i * 2, mi, ni) *= score_scale;
                            rO(mma_i * 2 + 1, mi, ni) *= score_scale;
                        }
                    }

                    float rl = 0.0f;
                    float safe_max = (new_max == -MAXFLOAT) ? 0.0f : new_max;
                    CUTE_UNROLL
                    for (uint32_t ni=0; ni < size<2>(rP); ++ni) {
                        float rp0 = exp2f(rP(mma_i * 2, mi, ni) - safe_max);
                        float rp1 = exp2f(rP(mma_i * 2 + 1, mi, ni) - safe_max);

                        rP(mma_i * 2, mi, ni) = rp0;
                        rP(mma_i * 2 + 1, mi, ni) = rp1;
                        rl += rp0 + rp1;
                    }
                    rL(local_row) += rl;
                }
            }
        }
        else {
            CUTE_UNROLL
            for (uint32_t mi=0; mi < size<1>(rP); ++mi) {
                CUTE_UNROLL
                for (uint32_t mma_i=0; mma_i < size<0>(rP) / 2; ++mma_i) {
                    float rm = -MAXFLOAT;
                    const uint32_t local_row = mi * 2 + mma_i;
                    const uint32_t abs_q_pos = skip_helper.rIndex[local_row];
                    // update max
                    CUTE_UNROLL
                    for (uint32_t ni=0; ni < size<2>(rP); ++ni) {
                        auto mn = cP(mma_i * 2, mi, ni);
                        uint32_t n = get<1>(mn);
                        uint32_t abs_k_pos = tile_start + n;

                        const bool valid1 = (abs_k_pos >= key_start) && (abs_k_pos < key_end) && (abs_k_pos <= abs_q_pos);
                        const bool valid2 = (abs_k_pos + 1 >= key_start) && (abs_k_pos + 1 < key_end) && (abs_k_pos + 1 <= abs_q_pos);
                        rP(mma_i * 2, mi, ni) = valid1 ? rP(mma_i * 2, mi, ni) : -MAXFLOAT;
                        rP(mma_i * 2 + 1, mi, ni) = valid2 ? rP(mma_i * 2 + 1, mi, ni) : -MAXFLOAT;
                        rm = cute::max(rm, cute::max(rP(mma_i * 2, mi, ni), rP(mma_i * 2 + 1, mi, ni)));
                    }

                    rm = cute::max(rm, __shfl_xor_sync(0xffffffff, rm, 1, 4));
                    rm = cute::max(rm, __shfl_xor_sync(0xffffffff, rm, 2, 4));
                    float old_max = rM(local_row);
                    float new_max = cute::max(rm, old_max);
                    const bool rescale_acc = new_max > old_max;
                    rM(local_row) = new_max;

                    if (rescale_acc && old_max != -MAXFLOAT) {
                        const float score_scale = exp2f(old_max - new_max);
                        rL(local_row) *= score_scale;
                        CUTE_UNROLL
                        for (uint32_t ni=0; ni < size<2>(rO); ++ni) {
                            rO(mma_i * 2, mi, ni) *= score_scale;
                            rO(mma_i * 2 + 1, mi, ni) *= score_scale;
                        }
                    }

                    float rl = 0.0f;
                    float safe_max = (new_max == -MAXFLOAT) ? 0.0f : new_max;
                    CUTE_UNROLL
                    for (uint32_t ni=0; ni < size<2>(rP); ++ni) {
                        float rp0 = exp2f(rP(mma_i * 2, mi, ni) - safe_max);
                        float rp1 = exp2f(rP(mma_i * 2 + 1, mi, ni) - safe_max);

                        rP(mma_i * 2, mi, ni) = rp0;
                        rP(mma_i * 2 + 1, mi, ni) = rp1;
                        rl += rp0 + rp1;
                    }
                    rL(local_row) += rl;
                }
            }
        }
        // P@V mma
        // convert QK-C layout to PV-A layout
        convert_QKrC_to_PVrA<decltype(rP), decltype(rPlA), DType>(rP, rPlA);

        const uint32_t read_stage_v = pipe_stage<PipelineV>(k_tile);
        const uint32_t read_phase_v = pipe_phase<PipelineV>(k_tile);
        wait_barrier(smem_kv.tma_v_barrier[read_stage_v], read_phase_v);
        __syncthreads();

        copy(s2rV_tiled_copy{}, pSsV(_, _, _, make_coord(0, read_stage_v)), pDrV);
        CUTE_UNROLL
        for (uint32_t i=0; i < size<1>(rPlA); ++i) {
            if constexpr (kWarpSkipMMA) {
                if (skip_helper.execute_mma_warp[i]) {
                    CUTE_UNROLL
                    for (uint32_t j=0; j < size<1>(rV); ++j) {
                        CUTE_UNROLL
                        for (uint32_t k=0; k < size<2>(rV); ++k) {
                            gemm(tiled_mma_pv{}, rPlA(_, i, k), rV(_, j, k), rO(_, i, j));
                        }
                    }
                }
            }
            else {
                CUTE_UNROLL
                for (uint32_t j=0; j < size<1>(rV); ++j) {
                    CUTE_UNROLL
                    for (uint32_t k=0; k < size<2>(rV); ++k) {
                        gemm(tiled_mma_pv{}, rPlA(_, i, k), rV(_, j, k), rO(_, i, j));
                    }
                }
            }
        }
        __syncthreads();
    }
    // epilogue
    CUTE_UNROLL
    for (uint32_t mi=0; mi < size<1>(rO); ++mi) {
        CUTE_UNROLL
        for (uint32_t mma_i=0; mma_i < size<0>(rO) / 2; ++mma_i) {
            const uint32_t local_row = mi * 2 + mma_i;
            float row_sum = rL(local_row);
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, 1, 4);
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, 2, 4);
            const float inv_row_sum = __frcp_rn(row_sum);
            
            CUTE_UNROLL
            for (uint32_t ni=0; ni < size<2>(rO); ++ni) {
                rO(mma_i * 2, mi, ni) *= inv_row_sum;
                rO(mma_i * 2 + 1, mi, ni) *= inv_row_sum;
            }
        }
    }

    // write back
    auto r2sO_thr_copy = r2sO_tiled_copy{}.get_slice(tidx);
    auto pSrO = r2sO_thr_copy.retile_S(rO);
    auto pDsO = r2sO_thr_copy.partition_D(sO);
    auto pSrO_cast = make_tensor_like<DType>(pSrO);
    copy(pSrO, pSrO_cast);
    copy(r2sO_tiled_copy{}, pSrO_cast, pDsO);
    __syncthreads();

    auto gO_epi = local_tile(
        mO,
        make_tile(_1{}, Int<D>{}),
        make_coord(_, _0{}, query_head, bidz)
    );
    auto s2gO_thr_copy_g = s2gO_tiled_copy_g{}.get_slice(tidx % G2SQColPerCTA);
    auto s2gO_thr_copy_s = s2gO_tiled_copy_s{}.get_slice(tidx % G2SQColPerCTA);
    auto pSsO = s2gO_thr_copy_s.partition_S(sO);
    auto pDgO = s2gO_thr_copy_g.partition_D(gO_epi);
    CUTE_UNROLL
    for (uint32_t i=0, off1=thread_off_m_1, off2=thread_off_m_2; i < MMAIter; ++i, off1+=MMARowPerCTA, off2+=MMARowPerCTA) {
        if (skip_helper.rMask[i*2]) copy(s2gO_tiled_copy_g{}, pSsO(_, off1, _), pDgO(_, _0{}, _, skip_helper.rIndex[i*2]));
        if (skip_helper.rMask[i*2+1]) copy(s2gO_tiled_copy_g{}, pSsO(_, off2, _), pDgO(_, _0{}, _, skip_helper.rIndex[i*2+1]));
    }
}
