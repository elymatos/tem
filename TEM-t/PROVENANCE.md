# Provenance

Vendored from https://github.com/Ryan-SHU/TEM-t at commit
`c36ec3a7c7a164c160d02e94ad95177ce7c1019a` (2026-05-27).

Unofficial reproduction of Whittington, Warren & Behrens, "Relating Transformers
to Models and Neural Representations of the Hippocampal Formation", ICLR 2022
(arXiv:2112.04035). No official implementation of TEM-t was ever released; the
paper's "code will be released on publication" was not fulfilled.

Upstream `.git` was not copied, so this lives in the parent repo's history.
Upstream declares MIT in its README badge but ships no LICENSE file, and the
GitHub API reports no license.

Known deviations from the paper (verified against the PDF, Appendix D):
- Memory deduplication is NOT in the paper; it is an upstream addition, though
  the upstream README lists it among "constraints from the paper".
- Adaptive temperature is beta0 * log(m + 1); the paper specifies log(n_memories).
