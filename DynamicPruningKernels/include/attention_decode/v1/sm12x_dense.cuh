#pragma once
#include "attention_decode/v1/traits.cuh"

// main kernel func
template <
    uint32_t kHq, uint32_t kHk, uint32_t kD, uint32_t kNG, bool kIsLeftpad,
    uint32_t kBM, uint32_t kBN, uint32_t kPipelineK, uint32_t kPipelineV,
    uint32_t kSplitRound,
    typename TMADescK, typename TMADescV, class WarpLayoutTraits_, class MMATraits_,
    typename DType, uint32_t kWarps=4
>
__global__ __launch_bounds__(256) void attention_head_v1_dense(const __grid_constant__ AttentionParams<TMADescK, TMADescV> params) {
    using namespace cute;
    const auto& [Q, K, V, Mask, Leftpad, O, pO, LSE, B, Tk, Split, scale, tma_desc_k, tma_desc_v] = params;
    constexpr uint32_t Hq = kHq;
    constexpr uint32_t Hk = kHk;
    constexpr uint32_t D = kD;
    constexpr uint32_t NG = kNG;
    constexpr bool IsLeftpad = kIsLeftpad;
    constexpr uint32_t BM = kBM;
    constexpr uint32_t BN = kBN;
    constexpr uint32_t PipelineK = kPipelineK;
    constexpr uint32_t PipelineV = kPipelineV;
    constexpr uint32_t SplitRound = kSplitRound;

    constexpr uint32_t G2SQRowPerCTA = WarpLayoutTraits_::G2SQRowPerCTA;
    constexpr uint32_t G2SQColPerCTA = WarpLayoutTraits_::G2SQColPerCTA;
    constexpr uint32_t threadsPerCTA = WarpLayoutTraits_::threadsPerCTA;
    constexpr uint32_t MMARow = WarpLayoutTraits_::MMARow;
    constexpr uint32_t MMACol = WarpLayoutTraits_::MMACol;
    constexpr uint32_t MMA_M = MMATraits_::MMA_M;
    constexpr uint32_t MMAIter = WarpLayoutTraits_::MMAIter;

    using tiled_mma_qk = typename MMATraits_::tiled_mma_qk;
    using tiled_mma_pv = typename MMATraits_::tiled_mma_pv;
    using sQLayout = typename MMATraits_::sQLayout;
    using sKLayout = typename MMATraits_::sKLayout;
    using sVLayout = typename MMATraits_::sVLayout;
    using sOLayout = typename MMATraits_::sOLayout;
    using sOPartialLayout = typename MMATraits_::sOPartialLayout;
    using g2sQ_tiled_copy = typename MMATraits_::g2sQ_tiled_copy;
    using s2rQ_tiled_copy = typename MMATraits_::s2rQ_tiled_copy;
    using s2rK_tiled_copy = typename MMATraits_::s2rK_tiled_copy;
    using s2rV_tiled_copy = typename MMATraits_::s2rV_tiled_copy;
    using r2sO_tiled_copy = typename MMATraits_::r2sO_tiled_copy;
    using s2gO_tiled_copy = typename MMATraits_::s2gO_tiled_copy;

    // grid: (Split, max(NG, Hk), B)
    const uint32_t tidx = threadIdx.x;
    const uint32_t bidx = blockIdx.x;
    const uint32_t bidy = blockIdx.y;
    const uint32_t bidz = blockIdx.z;
    const uint32_t warp_id = tidx / device::kWarpThreads;
    const uint32_t lane_id = tidx % device::kWarpThreads;

    constexpr uint32_t kv_group_size = Hq / Hk;
    constexpr uint32_t group_size = Hq / NG;
    const uint32_t query_head = BM * bidy;
    const uint32_t key_head = query_head / kv_group_size;
    const uint32_t group_id = query_head / group_size;
    const float scale_ln2 = scale * 1.44269504;

    const uint32_t lane_mma_row = lane_id / 4;
    const uint32_t lane_mma_col = lane_id % 4;

    //
    // prepare tensors and smem layout
    //
    Tensor mQ = make_tensor(
        make_gmem_ptr(static_cast<const DType*>(Q)),
        make_ordered_layout(
            make_shape(Int<Hq>{}, Int<D>{}, 1, B),
            make_step(_1{}, _0{}, _2{}, _3{})
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
            make_shape(Int<Hq>{}, Int<D>{}, 1, B),
            make_step(_1{}, _0{}, _2{}, _3{})
        )
    );
    Tensor mpO = make_tensor(
        make_gmem_ptr(static_cast<DType*>(pO)),
        make_ordered_layout(
            make_shape(Int<Hq>{}, Int<D>{}, Int<SplitRound>{}, B),
            make_step(_1{}, _0{}, _2{}, _3{})
        )
    );
    Tensor mLSE = make_tensor(
        make_gmem_ptr(static_cast<float*>(LSE)),
        make_ordered_layout(
            make_shape(Int<Hq>{}, Int<SplitRound>{}, B),
            make_step(_1{}, _0{}, _2{})
        )
    );
    Tensor mMask = make_tensor(
        make_gmem_ptr(static_cast<const uint8_t*>(Mask)),
        make_layout(make_shape(B, _1{}, Int<NG>{}), LayoutRight{})
    );
    Tensor mLeftpad = make_tensor(
        make_gmem_ptr(static_cast<const uint32_t*>(Leftpad)),
        make_layout(make_shape(B), LayoutRight{})
    );

    constexpr bool IsCTAReduce = MMACol > 1;
    extern __shared__ uint8_t shared_memory[];
    using SharedStorageKV = SharedMemoryKV<DType, sKLayout, sVLayout, PipelineK, PipelineV, IsCTAReduce, MMACol>;
    SharedStorageKV &smem_kv = *reinterpret_cast<SharedStorageKV*>(shared_memory);

    using SharedStorageQ = SharedMemoryQ<DType, sQLayout>;
    using SharedStorageO = SharedMemoryO<DType, sOLayout>;
    using SharedStoragePartialO = SharedMemoryO<float, sOPartialLayout>;
    SharedStorageQ &smem_q = *reinterpret_cast<SharedStorageQ*>(shared_memory);
    SharedStorageO &smem_o = *reinterpret_cast<SharedStorageO*>(shared_memory);
    SharedStoragePartialO &smem_partial_o = *reinterpret_cast<SharedStoragePartialO*>(shared_memory);

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
    Tensor sOPartial = make_tensor(
        make_smem_ptr(smem_partial_o.O.begin()),
        sOPartialLayout{}
    );

    // load mask
    if (mMask(make_coord(bidz, 0, group_id)) == 0) return;

    uint32_t producer_k = 0;
    uint32_t producer_v = 0;

    //
    // prologue, load Q from gmem -> smem -> RF
    //
    Tensor gQ = local_tile(
        mQ,
        make_tile(Int<BM>{}, Int<D>{}),
        make_coord(bidy, 0, 0, bidz)
    );
    auto g2sQ_thr_copy = g2sQ_tiled_copy{}.get_slice(tidx);
    auto pSgQ = g2sQ_thr_copy.partition_S(gQ);
    auto pDsQ = g2sQ_thr_copy.partition_D(sQ);

    // g2sQ pred
    auto prQ = make_identity_tensor(make_shape(Int<BM>{}, Int<D>{}));
    auto pSpQ = g2sQ_thr_copy.partition_S(prQ);
    auto g2sQ_pred = cute::lazy::transform(pSpQ, [](auto c) {
        return get<0>(c) < Int<BM>{};
    });

    copy_if(g2sQ_tiled_copy{}, g2sQ_pred, pSgQ, pDsQ);
    cp_async_fence();

    // fill the pipeline for K & V
    // 1. get Tk start and end
    const uint32_t total_k_tiles = cute::ceil_div(Tk, BN);
    const uint32_t partial_k_tiles = cute::ceil_div(total_k_tiles, Split);
    const uint32_t base_k_tile = bidx * partial_k_tiles;
    const uint32_t key_end = cute::min(Tk, (base_k_tile + partial_k_tiles) * BN);

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

    const uint32_t key_start_tile = base_k_tile;
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

    auto issue_cta_reduce_max = [&](float local_max) {
        if constexpr (MMACol == 1) return local_max;
        else {
            if (lane_id % 4 == 0) smem_kv.cta_reduce_cache[lane_mma_row * kWarps + warp_id] = local_max;
            __syncthreads();

            float tmp_max = -MAXFLOAT;
            if (warp_id == 0 && lane_mma_col < kWarps) {
                tmp_max = smem_kv.cta_reduce_cache[lane_mma_row * kWarps + lane_mma_col];
            }
            if (warp_id == 0) {
                CUTE_UNROLL
                for (int32_t off = kWarps / 2; off > 0; off >>= 1) {
                    tmp_max = cute::max(tmp_max, __shfl_xor_sync(0xffffffff, tmp_max, off, 4));
                }
                if (lane_mma_col == 0) smem_kv.cta_reduce_cache[lane_mma_row] = tmp_max;
            }
            __syncthreads();
            return smem_kv.cta_reduce_cache[lane_mma_row];
        }
    };
    auto issue_cta_reduce_sum = [&](float local_sum) {
        if constexpr (MMACol == 1) return local_sum;
        else {
            if (lane_id % 4 == 0) smem_kv.cta_reduce_cache[lane_mma_row * kWarps + warp_id] = local_sum;
            __syncthreads();

            float tmp_sum = 0;
            if (warp_id == 0 && lane_mma_col < kWarps) {
                tmp_sum = smem_kv.cta_reduce_cache[lane_mma_row * kWarps + lane_mma_col];
            }
            if (warp_id == 0) {
                CUTE_UNROLL
                for (int32_t off = kWarps / 2; off > 0; off >>= 1) {
                    tmp_sum += __shfl_xor_sync(0xffffffff, tmp_sum, off, 4);
                }
                if (lane_mma_col == 0) smem_kv.cta_reduce_cache[lane_mma_row] = tmp_sum;
            }
            __syncthreads();
            return smem_kv.cta_reduce_cache[lane_mma_row];
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
        make_layout(make_shape(Int<MMA_M>{}, Int<BN>{}), LayoutRight{})
    );
    auto rPlA = thr_mma_pv.partition_fragment_A(rP_operand);

    Tensor gO_tile = local_tile(
        mO,
        make_tile(Int<BM>{}, Int<D>{}),
        make_coord(bidy, 0, 0, bidz)
    );
    Tensor rO_logical = make_identity_tensor(make_shape(Int<MMA_M>{}, Int<D>{}));
    auto rO = thr_mma_pv.partition_fragment_C(rO_logical); // split-K acc fragment
    auto cO = thr_mma_pv.partition_C(rO_logical);
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
            CUTE_UNROLL
            for (uint32_t j=0; j < size<1>(rK); ++j) {
                CUTE_UNROLL
                for (uint32_t k=0; k < size<2>(rK); ++k) {
                    gemm(tiled_mma_qk{}, rQ(_, i, k), rK(_, j, k), rP(_, i, j));
                }
            }
        }

        // Row-wise warp-local online softmax. The CTA combines warp-K
        // partials once in the epilogue.
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
        // QK-C and PV-A share the same per-thread coordinates.
        convert_QKrC_to_PVrA<decltype(rP), decltype(rPlA), DType>(rP, rPlA);

        const uint32_t read_stage_v = pipe_stage<PipelineV>(k_tile);
        const uint32_t read_phase_v = pipe_phase<PipelineV>(k_tile);
        wait_barrier(smem_kv.tma_v_barrier[read_stage_v], read_phase_v);
        __syncthreads();

        copy(s2rV_tiled_copy{}, pSsV(_, _, _, make_coord(0, read_stage_v)), pDrV);
        CUTE_UNROLL
        for (uint32_t i=0; i < size<1>(rPlA); ++i) {
            CUTE_UNROLL
            for (uint32_t j=0; j < size<1>(rV); ++j) {
                CUTE_UNROLL
                for (uint32_t k=0; k < size<2>(rV); ++k) {
                    gemm(tiled_mma_pv{}, rPlA(_, i, k), rV(_, j, k), rO(_, i, j));
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
            const float warp_max = rM(local_row);
            const float row_max = issue_cta_reduce_max(warp_max);
            float row_sum = rL(local_row);
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, 1, 4);
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, 2, 4);
            const float warp_scale = row_max == -MAXFLOAT
                ? 0.0f
                : exp2f(warp_max - row_max);
            row_sum = issue_cta_reduce_sum(row_sum * warp_scale);
            rM(local_row) = row_max;
            rL(local_row) = row_sum;
            const float partial_scale = row_sum == 0.0f
                ? 0.0f
                : warp_scale * __frcp_rn(row_sum);
            
            CUTE_UNROLL
            for (uint32_t ni=0; ni < size<2>(rO); ++ni) {
                rO(mma_i * 2, mi, ni) *= partial_scale;
                rO(mma_i * 2 + 1, mi, ni) *= partial_scale;
            }
        }
    }

    // The PV warp layout splits K. Preserve FP32 until all warp partials are
    // reduced, then let warp 0 perform the single DType conversion and STSM.
    if constexpr (MMACol > 1) {
        CUTE_UNROLL
        for (uint32_t i=0; i < size(rO); ++i) {
            auto coord = cO(i);
            sOPartial(get<0>(coord), get<1>(coord), warp_id) = rO(i);
        }
        __syncthreads();

        if (warp_id == 0) {
            CUTE_UNROLL
            for (uint32_t i=0; i < size(rO); ++i) {
                auto coord = cO(i);
                float partial = 0.0f;
                CUTE_UNROLL
                for (uint32_t warp=0; warp < MMACol; ++warp) {
                    partial += sOPartial(get<0>(coord), get<1>(coord), warp);
                }
                rO(i) = partial;
            }
            __syncwarp();
        }
    }

    if (warp_id == 0) {
        auto r2sO_thr_copy = r2sO_tiled_copy{}.get_slice(tidx);
        auto pSrO = r2sO_thr_copy.retile_S(rO);
        auto pDsO = r2sO_thr_copy.partition_D(sO);
        auto pSrO_cast = make_tensor_like<DType>(pSrO);
        copy(pSrO, pSrO_cast);
        copy(r2sO_tiled_copy{}, pSrO_cast, pDsO);
    }
    __syncthreads();

    if (Split == 1) {
        auto s2gO_thr_copy = s2gO_tiled_copy{}.get_slice(tidx);
        auto pSsO = s2gO_thr_copy.partition_S(sO);
        auto pDgO = s2gO_thr_copy.partition_D(gO_tile);

        // s2gO pred
        auto prO = make_identity_tensor(make_shape(Int<BM>{}, Int<D>{}));
        auto pDpO = s2gO_thr_copy.partition_D(prO);
        auto s2gO_pred = cute::lazy::transform(pDpO, [](auto c) {
            return get<0>(c) < Int<BM>{};
        });

        copy_if(s2gO_tiled_copy{}, s2gO_pred, pSsO, pDgO);
    }
    else {
        auto gpO = local_tile(
            mpO,
            make_tile(Int<BM>{}, Int<D>{}),
            make_coord(bidy, 0, bidx, bidz)
        );
        auto s2gO_thr_copy = s2gO_tiled_copy{}.get_slice(tidx);
        auto pSsO = s2gO_thr_copy.partition_S(sO);
        auto pDgO = s2gO_thr_copy.partition_D(gpO);

        // s2gO pred
        auto prO = make_identity_tensor(make_shape(Int<BM>{}, Int<D>{}));
        auto pDpO = s2gO_thr_copy.partition_D(prO);
        auto s2gO_pred = cute::lazy::transform(pDpO, [](auto c) {
            return get<0>(c) < Int<BM>{};
        });

        copy_if(s2gO_tiled_copy{}, s2gO_pred, pSsO, pDgO);

        // write back LSE
        if (warp_id == 0 && lane_mma_col == 0) {
            CUTE_UNROLL
            for (uint32_t mi=0; mi < size<1>(rP); ++mi) {
                CUTE_UNROLL
                for (uint32_t mma_i=0; mma_i < size<0>(rP) / 2; ++mma_i) {
                    const uint32_t local_row = mi * 2 + mma_i;
                    const uint32_t query_row = get<0>(cP(mma_i * 2, mi, _0{}));
                    if (query_row < BM) {
                        float lse = log2f(rL(local_row)) + rM(local_row);
                        mLSE(make_coord(query_head + query_row, bidx, bidz)) = lse;
                    }
                }
            }
        }
    }
}
