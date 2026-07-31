## Version 3
Thread-level G2S skip version of GEMM_MN.
- Thread-level skip for `cp.async` load A row, which is better than warp-level skip A, since `cp.async` only requires the thread to issue the instruction, with no need for subsequent serial waiting. At the same time, there are only two states (0 or 1) within a warp, rendering the impact of warp divergence negligible.