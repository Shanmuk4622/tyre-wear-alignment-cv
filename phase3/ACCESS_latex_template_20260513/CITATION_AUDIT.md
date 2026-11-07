# Literature and venue-source audit

**Checked 2–3 November and extended 6 November 2026.** This is a focused primary-source bibliography,
not a systematic literature review or comprehensive retraction-database search.
No performance number from an unrelated dataset is used as a direct comparator.

## Cited sources

| BibTeX key | Primary record checked | Use in this paper |
|---|---|---|
| `tireeye` | [PHM Society, Huber/Preindl/Betz, 2022](https://papers.phmsociety.org/index.php/phmconf/article/view/3242) | Physical optical measurement differs from proxy classification |
| `tread2021` | [Publisher-hosted paper, Petrović et al., 2021](https://hrcak.srce.hr/file/385709) | HOG/SVM tread inspection; compound family name Ristić Durrant retained |
| `mil2025` | [Springer, Mignot et al., 2025](https://link.springer.com/article/10.1007/s00170-025-15740-3) | Local defect evidence aggregated into manufacturing decisions; different task |
| `resnet` | [Author preprint, He/Zhang/Ren/Sun](https://arxiv.org/abs/1512.03385) | Residual-network context, not a claimed new baseline run |
| `segformer` | [NeurIPS 2021 proceedings](https://proceedings.neurips.cc/paper_files/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html) | Transformer segmentation context |
| `caruana` | [Publisher, Machine Learning 28:41–75, 1997](https://doi.org/10.1023/A:1007379606734) | Shared multitask learning |
| `pcgrad` | [NeurIPS 2020 proceedings](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) | Inspiration for explicitly different symmetric two-task diagnostic |
| `sanity` | [Author preprint, Adebayo et al.](https://arxiv.org/abs/1810.03292) | Visual plausibility alone does not validate explanations |
| `shortcuts` | [Nature Machine Intelligence, Geirhos et al., 2020](https://www.nature.com/articles/s42256-020-00257-z) | Incidental predictive cues; no claim that every failure here is proven shortcut learning |
| `mae` | [Author preprint, He et al.](https://arxiv.org/abs/2111.06377) | Missing-content reconstruction context; CVPR 2022 identity checked through official paper record |
| `maskfeat` | [Author preprint, Wei et al.](https://arxiv.org/abs/2112.09133) | Fixed descriptor prediction/local contrast; no replication claim |
| `vicreg` | [Author preprint, Bardes/Ponce/LeCun](https://arxiv.org/abs/2105.04906) | Paired invariance/variance/covariance principle; six-dimensional adaptation |
| `barlow` | [PMLR 139:12310–12320](https://proceedings.mlr.press/v139/zbontar21a.html) | Redundancy-reduction context |
| `rankme` | [PMLR 202:10929–10974](https://proceedings.mlr.press/v202/garrido23a.html) | Rank assessment context; paper explicitly distinguishes covariance-energy weighting |
| `cka` | [PMLR 97:3519–3529](https://proceedings.mlr.press/v97/kornblith19a.html) | Centered kernel alignment in exploratory mask-layout follow-up |
| `groups` | [Ecography 40:913–929, Roberts et al.](https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881) | Dependence-aware grouping/evaluation |
| `selectionbias` | [JMLR 11(70):2079–2107](https://www.jmlr.org/papers/v11/cawley10a.html) | Selection bias on finite validation data |
| `calibration` | [PMLR 70:1321–1330](https://proceedings.mlr.press/v70/guo17a.html) | Probability-quality context; not a new temperature-scaling experiment |
| eigondepth | [Authors' paper](https://arxiv.org/abs/1406.2283) | Absolute scale ambiguity in monocular depth estimation |
| `fcanet` | [Authors' paper](https://arxiv.org/abs/2012.11879) | Global averaging as the constant DCT coefficient. The paper uses DCT as a matched spatial baseline. |
| `tokenpool` | [CVF WACV proceedings](https://openaccess.thecvf.com/content/WACV2023/html/Marin_Token_Pooling_in_Vision_Transformers_for_Image_Classification_WACV_2023_paper.html) | Token-set approximation by clustering. Our basis is common and tied to fixed canonical positions. |
| `matsubara` | [Authors' WACV 2022 preprint](https://arxiv.org/abs/2108.11898) | Learned task-driven edge compression, with separate rate and latency aims |
| `zahm` | [SIAM publisher record](https://epubs.siam.org/doi/10.1137/18M1221837) | Gradient-based dimension reduction for vector-valued functions. Our fixed spatial projection addresses a different constrained objective. |

Title, author, venue/year and available identifiers were checked against the
records above. Some direct DOI/OpenReview/CVF openings were blocked by provider
verification or unsupported redirects; primary publisher search records or the
authors' arXiv records supplied the usable fallback. No secondary blog is used
for a technical claim. Conference page ranges not confidently resolved were
omitted rather than guessed. The supplied bibliography style prints canonical
DOI URLs so journal records remain directly findable.

## Current IEEE Access preparation

| Official source | Applied action |
|---|---|
| [Preparing your article](https://ieeeaccess.ieee.org/authors/preparing-your-article/) | Genuine template, evidence-supported claims, specific AI-assistance acknowledgment |
| [Submission guidelines](https://ieeeaccess.ieee.org/authors/submission-guidelines/) | Source/PDF pairing, both biographies, AI system identification and disclosure scope, review checklist |
| [IEEE article structure](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/structure-your-article/) | Single self-contained abstract below 250 words |
| [Current official LaTeX template](https://ieeeaccess.ieee.org/wp-content/uploads/2026/05/ACCESS_latex_template_20260513-1-1.zip) | Core class/style files compared to the supplied ones by SHA-256 |

The official template download is inspected in memory; its ZIP is a publisher
template distribution, not a training-data publication. No ZIP is saved or
uploaded. [Current template receipt](audit/CURRENT_TEMPLATE_CHECK.json).

A single acknowledgment identifies and links OpenAI Codex and specifies assistance in the abstract, Sections I–X, captions, and code. The reviewers requested consolidation. Verify citation form and placement against the venue rules at submission.

This audit does not promise journal acceptance or replace final checking of
reference updates, retractions, originality and author declarations.
