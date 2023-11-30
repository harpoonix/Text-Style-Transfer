# Text-Style-Transfer

Text style transfer is an important task in natural language generation, which aims to control
certain attributes in the generated text, such as politeness, emotion, humor, and many others.
It has a long history in the field of natural language processing, and recently has re-gained
significant attention thanks to the promising performance brought by deep neural models. In this project, we study two methods and architectures employed for this task, compare their generated results and the results of our attempt at reproducibility. In some places, we have also attempted to analyse how our modifications to these architectures compare to the ones designed by the authors. We end with some insights on how the task of text style transfer can be approached, with possible extensions to our modifications that can make it work better.  

## [Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation](https://arxiv.org/abs/1905.12926)

*Paper by Ke Wang, Hang Hua and Xiojun Wan*

Our attempt at reproducibility with some architecture modifications is in [this](./controllable-text-attribute-transfer) folder.

## [Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation](https://arxiv.org/abs/1905.05621)

*Paper by Ning Dai, Jianze Liang, Xipeng Qiu, Xuanjing Huang*

Our attempt at reproducibility with some architecture modifications is in [this](./style-transformer) folder.

## Demo

A demo of the style transfer task using our best trained model can be found in [this python script](./style-transformer/testany.py).

## Our Saved weights

- Wang et. al model: trained on [amazon](./controllable-text-attribute-transfer/method/mymodel-amazon/save/1700501011/), [yelp](./controllable-text-attribute-transfer/method/mymodel-yelp/save/1699447740/)
- Dai et. al model: [multi-class](./style-transformer/save/Nov07025149/ckpts/)

