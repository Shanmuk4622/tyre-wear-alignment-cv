# References and source-verification notes

Primary landing pages, abstracts and official documentation were checked on 12 September 2026. This is bibliographic and claim-scope verification, not a claim that every cited paper was independently reproduced. Preprint records are explicitly identified; software documentation is not described as peer-reviewed research. The report makes no numerical state-of-the-art comparison with these external studies.

<!-- bibliography:start -->
1. Huber, S., Preindl, P., and Betz, J. (2022). **TireEye: Optical On-board Tire Wear Detection.** *Annual Conference of the PHM Society*, 14(1). DOI: 10.36001/phmconf.2022.v14i1.3242. [Primary publication](https://papers.phmsociety.org/index.php/phmconf/article/view/3242).

2. He, K., Zhang, X., Ren, S., and Sun, J. (2016). **Deep Residual Learning for Image Recognition.** *CVPR*, 770–778. [Primary publication](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).

3. Cao, W., Mirjalili, V., and Raschka, S. (2019). **Rank consistent ordinal regression for neural networks with application to age estimation.** arXiv:1901.07884, preprint record. [Primary preprint](https://arxiv.org/abs/1901.07884).

4. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., and Batra, D. (2017). **Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.** *ICCV*; preprint arXiv:1610.02391. [Primary preprint record](https://arxiv.org/abs/1610.02391).

5. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., and Kim, B. (2018). **Sanity Checks for Saliency Maps.** *NeurIPS*, 31. [Primary publication](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html).

6. Ronneberger, O., Fischer, P., and Brox, T. (2015). **U-Net: Convolutional Networks for Biomedical Image Segmentation.** arXiv:1505.04597, preprint record. [Primary preprint](https://arxiv.org/abs/1505.04597).

7. Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., and Adam, H. (2018). **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.** arXiv:1802.02611, preprint record. [Primary preprint](https://arxiv.org/abs/1802.02611).

8. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., and Luo, P. (2021). **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.** *NeurIPS*, 34. [Primary publication](https://proceedings.neurips.cc/paper_files/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html).

9. Ultralytics. **YOLO26 model documentation.** Software documentation, accessed 12 September 2026. [Official documentation](https://docs.ultralytics.com/models/yolo26/). Executed study library pin: Ultralytics 8.4.20; the current web page is not an immutable specification of that version.

10. Peking University / Hugging Face. **RT-DETRv2-R18 model card and Transformers integration.** Model/software documentation, accessed 12 September 2026. [Exact model card](https://huggingface.co/PekingU/rtdetr_v2_r18vd), [official integration documentation](https://huggingface.co/docs/transformers/model_doc/rt_detr_v2). The study uses its frozen protocol revision rather than mutable model-card examples.

11. Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). **On Calibration of Modern Neural Networks.** *ICML*, PMLR 70, 1321–1330. [Primary publication](https://proceedings.mlr.press/v70/guo17a.html).

12. Angelopoulos, A. N., and Bates, S. (2021). **A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.** arXiv:2107.07511, preprint record. [Primary preprint](https://arxiv.org/abs/2107.07511).
<!-- bibliography:end -->

## Scope of citation use

| References | Used for | Not used to claim |
|---|---|---|
| 1 | Distinguishing optical physical measurement from proxy classification | Comparable macro-F1, a verified project depth error, or clinical/safety validity |
| 2–3 | Residual recognition and ordinal decision motivation | Every sweep architecture has identical pretraining or a new ordinal method was invented |
| 4–5 | CAM interpretation and need for parameter-dependence checks | Passing one gate proves causality or universally valid explanations |
| 6–10 | Dense-model families and software/model identities | Equal-loss/equal-pretraining comparison or a new detection architecture |
| 11–12 | Calibration and prediction-set foundations | Guaranteed deployment coverage on this dependent dataset |

## Historical-reference quarantine

`docs/09_RELATED_WORK.md` remains an earlier research/planning record. Its detailed performance numbers and broad novelty claims have not all been verified for this submission. They are not copied into the manuscript as established facts. In particular, do not import tyre-depth millimetre errors, alignment precision, or future/publication-year claims without reading the relevant primary result and identifying its dataset and target.

The selected references support the concepts actually discussed. They are not an exhaustive systematic review. If a venue expects an architecture-by-architecture literature survey or a specific bibliographic style, expand this list from primary sources and preserve the distinction between papers, preprints, model cards and software releases. Do not invent DOI, volume, page or publication details.
