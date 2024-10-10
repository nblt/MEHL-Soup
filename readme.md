# Learning Scalable Model Soup on a Single GPU: An Efficient Subspace Training Strategy

Tao Li*, Weisen Jiang*, Fanghui Liu, Xiaolin Huang, James T Kwok 

(*Equally contribution)

**Paper:** https://arxiv.org/abs/2407.03641

**ECCV 2024**

## Introduction
Model soup is an effective strategy for enhancing model performance by averaging multiple models fine-tuned from different hyper-parameter configurations into a single "soup model" in weight space. Learned soup is promising to achieve better performance than greedy soup due to its better flexibility in learning coefficients. However, it is often less perferred in practice due to its huge memory requirements (e.g. ~250GB of memory required to average 72 CLIP ViT-B/32 models). In this work, we propose an efficient and scalable strategy that allows enjoying the advantages of learned soup while maintaining similiar level of memory burden to that of greedy soup.

Our approach includes two key components: 1) a hyper-plane optimization target that enhances performance by facilitating coefficient expolation, and 2) a mini-batch model optimization strategy that allows the memory requirements are only related to the batch size, making it scalable.

The code is raw and still under construction. We will release more friendly interface/implementation in the next couple months.