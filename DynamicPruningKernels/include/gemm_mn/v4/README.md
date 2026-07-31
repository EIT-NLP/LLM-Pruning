## Version 4
Warp-level G2S skip version of GEMM_MN.
- Warp-level skip for `cp.async` load A row
- For sm8x, the warp-level is suboptimal
- For sm10x, sm12x, which support TMA gather4 and requires each warp to handle 4 rows of A per instruction, the warp-level decision is required
