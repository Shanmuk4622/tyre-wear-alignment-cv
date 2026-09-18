# Current status: revised bibliography, 18 September 2026

31 unique references. The revision audit at the end supersedes the original
counts, split interpretation, and figure inventory below. The earlier audit is
retained as a record of what was checked for the first draft.

# Citation audit — 16 September 2026

The manuscript has 16 unique references. Citations support the adjacent concept;
none supplies a numerical external benchmark that is compared to this dataset.
The older Markdown bibliography contained duplicate HRNet and ChArUco entries;
the paper uses unique BibTeX keys and does not copy those duplicates.

Verification covers primary landing pages/abstracts and official documentation,
plus DOI metadata where noted. It is not independent replication of the cited
work or a claim to have checked every sentence of every full paper.

| Key | Primary source checked | Scope and correction |
|---|---|---|
| tireeye | https://papers.phmsociety.org/index.php/phmconf/article/view/3242 | Authors, 2022, 14(1), DOI; groove outline and physical scale reference. No imported depth accuracy. |
| resnet | https://arxiv.org/abs/1512.03385 ; https://doi.org/10.1109/CVPR.2016.90 | Primary abstract plus Crossref DOI metadata confirms CVPR 2016, pp. 770–778. |
| coral | https://arxiv.org/abs/1901.07884 ; https://doi.org/10.1016/j.patrec.2020.11.008 | Corrected older preprint-only citation to Pattern Recognition Letters 140 (2020), 325–331; primary arXiv record and DOI metadata agree. |
| gradcam | https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html | Verified ICCV 2017 version, pp. 618–626; do not confuse the later IJCV version with this conference entry. |
| sanity | https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html | Authors, title, NeurIPS 31 (2018); visual plausibility does not establish parameter dependence. |
| unet | https://arxiv.org/abs/1505.04597 | Architecture and MICCAI 2015 identification; no unverified DOI/pages added. |
| deeplab | https://arxiv.org/abs/1802.02611 | Author list and ECCV 2018 camera-ready identification. |
| segformer | https://proceedings.neurips.cc/paper_files/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html | Primary NeurIPS 34 (2021) record; architecture only, no transplanted benchmark score. |
| yolo | https://docs.ultralytics.com/models/yolo26/ | Official software documentation, explicitly distinct from a research paper and executed Ultralytics 8.4.20. |
| rtdetr | https://huggingface.co/PekingU/rtdetr_v2_r18vd | Exact publisher model card; no substitution with RT-DETR-L. |
| calibration | https://proceedings.mlr.press/v70/guo17a.html | Authors, ICML/PMLR 70 (2017), 1321–1330, temperature-scaling motivation. |
| conformal | https://arxiv.org/abs/2107.07511 | Explicitly cited preprint record, 2021. No guarantee asserted for dependent project data. |
| hrnet | https://arxiv.org/abs/1902.09212 | Official primary paper, accepted CVPR 2019; architecture foundation, not tyre performance. |
| charuco | https://docs.opencv.org/4.11.0/df/d4a/tutorial_charuco_detection.html | Version-specific official documentation matches the local OpenCV 4.11 family. |
| mobilenet | https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5647_ECCV_2024_paper.php | Official ECCV 2024 record and authors; not a claim of phone performance for this workstation. |
| leakage | https://doi.org/10.1016/j.patter.2023.100804 ; https://arxiv.org/abs/2207.07048 | Primary abstract/DOI landing plus Crossref confirms Patterns 4(9), 100804 (2023). General leakage motivation only. |

Some direct CVF/publisher fetches returned access errors; primary arXiv records,
indexed CVF results, and publisher-deposited DOI metadata were used as identified
above. No secondary blog was used to substantiate a technical claim.

## Claim checks

- Proxy labels remain mileage labels; no measured depth, healthy label or safety conclusion.
- Historical split flags remain visible despite later operator identity confirmation.
- Selected and final classification endpoints remain distinct; no test-set ranking.
- Classification resolution/batch exceptions are taken from the actual registry.
- The explanation gate is sanity delta >0.05 plus a non-missing insertion/deletion difference, not a positive-faithfulness requirement.
- Ordinal crop/fusion decisions are distinct from the inherited probability-ensemble calibration argmax.
- Geometry error uses native width minus one, not resized width; 432 records are correlated observations from two test tyres.
- Point versus dense-mask supervision is stated; 100% comparator coverage means no fallback helped the final result.
- UI/video/synthetic alignment checks are software evidence; no physical accuracy claim.

## Figure checks

Five analytic/schematic vector figures are regenerated from frozen records or
explicitly labelled as a schematic. Four source screenshots form three figure
environments. Screenshot bytes remain unchanged; cropping, altered predictions,
synthetic replacement photographs, and invented accuracy/error bars are absent.
Error whiskers in the localisation figure are sample SD across three seeds, not
population confidence intervals. The per-tyre comparison retains all seed values.

Full numerical inputs and their SHA-256 hashes are in `NUMERICAL_AUDIT.json`.
Rendering/layout findings and build integrity are recorded separately in
`VERIFICATION.json` after the final PDF build.

## Revision audit — 18 September 2026 (supersedes original counts)

The revised bibliography contains 31 unique entries. The original 16 entries
remain, with 11 application/validation articles or reports, three localisation
method references, and one explicit AI-software disclosure reference added.
The 10-row related-study comparison distinguishes tasks, references, and assessed
validation scope; NR means not established in the consulted primary text, not
an allegation that the original work omitted it. No external performance number
is ranked against this project's results.

Publisher-deposited DOI metadata for 11 additions is preserved in
`revision_evidence/citation_metadata.json`; none reports an `update-to` notice
in the fetched record. This metadata check is not an exhaustive retraction audit.
Primary sources consulted:

- Tread segmentation (2025): https://doi.org/10.1016/j.pes.2025.100080
- Behaviour-aware wear classification (2026): https://doi.org/10.1016/j.measurement.2026.121509
- Tyre-level multiple-instance decisions (2025): https://doi.org/10.1007/s00170-025-15740-3
- Laser-plane depth (2019): https://doi.org/10.1177/1687814019837828
- Drive-over sensors (2014): https://doi.org/10.4271/2014-01-0069
- Camera camber measurement (2010): https://doi.org/10.1016/j.sna.2010.04.004
- Rim/point-cloud alignment (2026): https://re.public.polimi.it/handle/11311/1312834 ; https://doi.org/10.3390/metrology6010004
- Surface damage (2024): https://www.mdpi.com/1424-8220/24/9/2778
- RGB-depth tread dimensions (2024): https://www.mdpi.com/2076-3417/14/15/6625
- Grouped validation: https://nsojournals.onlinelibrary.wiley.com/doi/full/10.1111/ecog.02881
- Shortcut learning: https://doi.org/10.1038/s42256-020-00257-z
- Boundary-sensitive evaluation: https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html
- Coordinate expectation: https://arxiv.org/abs/1801.07372
- Coordinate classification: https://arxiv.org/abs/2107.03332v3
- AI software identification: https://openai.com/codex/

Some publisher pages were available only through indexed primary excerpts or
abstracts; comparisons are limited to those supported claims. SimCC is cited as
the explicitly verified revised preprint rather than guessing conference pages.
Author diacritics are encoded with LaTeX accents for the supplied Type-1 fonts.

The old blanket interpretation of classification overlap concerns is superseded:
`REVISION_ANALYSIS.json` verifies zero shared recorded group IDs per fold. The
paper differentiates this audit, the operator's physical identity confirmation,
and the earlier appearance-similarity suspicion. The new per-image plot is a
retrospective aggregation of existing errors, not a new independent experiment.
The outdated workstation screenshot and synthetic alignment interface are removed
from the article. The historical evidence files themselves remain unchanged.
