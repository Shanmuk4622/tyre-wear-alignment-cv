"""Retain selected verified primary references and add the representation literature."""
from pathlib import Path
import json, re
ROOT=Path(__file__).resolve().parent
old=(ROOT.parents[1]/'ACCESS_latex_template_20260513/references.bib').read_text(encoding='utf-8')
keys=['tireeye','tread2021','mil2025','resnet','segformer','sanity','shortcuts','groups','selectionbias','calibration']
entries=[]
for key in keys:
    start=re.search(r'@\w+\{'+key+r',',old).start()
    end=re.search(r'\n@',old[start+1:])
    entries.append(old[start:start+1+end.start()].strip() if end else old[start:].strip())
extra=r'''
@article{caruana,
 author={Caruana, Rich}, title={Multitask Learning},
 journal={Machine Learning}, volume={28}, pages={41--75}, year={1997},
 doi={10.1023/A:1007379606734}}
@inproceedings{pcgrad,
 author={Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
 title={Gradient Surgery for Multi-Task Learning},
 booktitle={Advances in Neural Information Processing Systems}, volume={33}, year={2020},
 url={https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html}}
@inproceedings{mae,
 author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
 title={Masked Autoencoders Are Scalable Vision Learners},
 booktitle={Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)}, year={2022},
 url={https://arxiv.org/abs/2111.06377}}
@inproceedings{maskfeat,
 author={Wei, Chen and Fan, Haoqi and Xie, Saining and Wu, Chao-Yuan and Yuille, Alan and Feichtenhofer, Christoph},
 title={Masked Feature Prediction for Self-Supervised Visual Pre-Training},
 booktitle={Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)}, year={2022},
 url={https://arxiv.org/abs/2112.09133}}
@inproceedings{vicreg,
 author={Bardes, Adrien and Ponce, Jean and LeCun, Yann},
 title={{VICReg}: Variance-Invariance-Covariance Regularization for Self-Supervised Learning},
 booktitle={Proc. Int. Conf. Learning Representations (ICLR)}, year={2022},
 url={https://arxiv.org/abs/2105.04906}}
@inproceedings{barlow,
 author={Zbontar, Jure and Jing, Li and Misra, Ishan and LeCun, Yann and Deny, Stephane},
 title={Barlow Twins: Self-Supervised Learning via Redundancy Reduction},
 booktitle={Proc. Int. Conf. Machine Learning}, series={Proceedings of Machine Learning Research},
 volume={139}, pages={12310--12320}, year={2021},
 url={https://proceedings.mlr.press/v139/zbontar21a.html}}
@inproceedings{rankme,
 author={Garrido, Quentin and Balestriero, Randall and Najman, Laurent and LeCun, Yann},
 title={{RankMe}: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank},
 booktitle={Proc. Int. Conf. Machine Learning}, series={Proceedings of Machine Learning Research},
 volume={202}, pages={10929--10974}, year={2023},
 url={https://proceedings.mlr.press/v202/garrido23a.html}}
@inproceedings{cka,
 author={Kornblith, Simon and Norouzi, Mohammad and Lee, Honglak and Hinton, Geoffrey},
 title={Similarity of Neural Network Representations Revisited},
 booktitle={Proc. Int. Conf. Machine Learning}, series={Proceedings of Machine Learning Research},
 volume={97}, pages={3519--3529}, year={2019},
 url={https://proceedings.mlr.press/v97/kornblith19a.html}}
@inproceedings{eigondepth,
 author={Eigen, David and Puhrsch, Christian and Fergus, Rob},
 title={Depth Map Prediction from a Single Image Using a Multi-Scale Deep Network},
 booktitle={Advances in Neural Information Processing Systems}, volume={27}, year={2014},
 url={https://arxiv.org/abs/1406.2283}}
@inproceedings{fcanet,
 author={Qin, Zequn and Zhang, Pengyi and Wu, Fei and Li, Xi},
 title={{FcaNet}: Frequency Channel Attention Networks},
 booktitle={Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)}, year={2021},
 url={https://doi.org/10.1109/ICCV48922.2021.00082}}
@inproceedings{tokenpool,
 author={Marin, Dmitrii and Chang, Jen-Hao Rick and Ranjan, Anurag and Prabhu, Anish and Rastegari, Mohammad and Tuzel, Oncel},
 title={Token Pooling in Vision Transformers for Image Classification},
 booktitle={Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV)},
 pages={12--21}, year={2023}, url={https://arxiv.org/abs/2110.03860}}
@article{zahm,
 author={Zahm, Olivier and Constantine, Paul G. and Prieur, Cl{\'e}mentine and Marzouk, Youssef M.},
 title={Gradient-Based Dimension Reduction of Multivariate Vector-Valued Functions},
 journal={SIAM Journal on Scientific Computing}, volume={42}, number={1},
 pages={A534--A558}, year={2020}, url={https://doi.org/10.1137/18M1221837}}
@inproceedings{matsubara,
 author={Matsubara, Yoshitomo and Yang, Ruihan and Levorato, Marco and Mandt, Stephan},
 title={Supervised Compression for Resource-Constrained Edge Computing Systems},
 booktitle={Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV)},
 year={2022}, url={https://arxiv.org/abs/2108.11898}}
'''
# IEEEtran's supplied bibliography style does not print a DOI-only field.
# Preserve the DOI and provide its canonical resolver as the online location.
entries=[re.sub(r'doi=\{([^}]+)\}',r'doi={\1}, url={https://doi.org/\1}',entry) for entry in entries]
entries=[entry.replace(r"Danijela Risti{\'c} Durrant",r"{Risti{\'c} Durrant}, Danijela") for entry in entries]
(ROOT/'references.bib').write_text('\n\n'.join(entries)+'\n'+extra,encoding='utf-8')
print('23 primary research references.')
