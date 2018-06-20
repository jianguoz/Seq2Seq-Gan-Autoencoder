# Seq2Seq-Gan
**Jianguo Zhang, June 20, 2018**

Related implementations for **sequence to sequence**, **generative adversarial networks(GAN)** and **Autoencoder**

## Sequence to Sequence

![image1](https://github.com/JianguoZhang1994/Seq2Seq-Gan-Autoencoder/blob/master/seq2seq/images/sequence-to-sequence-inference-decoder.png)

## Generative Adversarial Networks

### gan_diagram

<div align=center>

![image2](https://github.com/JianguoZhang1994/Seq2Seq-Gan-Autoencoder/blob/master/gan_mnist/assets/gan_diagram.png)
</div>

### dcgan

<div align=center>

![image6](https://github.com/JianguoZhang1994/Seq2Seq-Gan-Autoencoder/blob/master/dcgan-svhn/assets/dcgan.png)
</div>


### svhn_gan

<div align=center>

![image7](https://github.com/JianguoZhang1994/Seq2Seq-Gan-Autoencoder/blob/master/dcgan-svhn/assets/svhn_gan.png)
</div>

### gan_network

<div align=center>

![image3](https://github.com/JianguoZhang1994/Seq2Seq-Gan-Autoencoder/blob/master/gan_mnist/assets/gan_network.png)
</div>

## Autoencoder

### simple_autoencoder

<div align=center>

![image4](https://github.com/JianguoZhang1994/Seq2Seq-Gan-Autoencoder/blob/master/autoencoder/assets/simple_autoencoder.png)
</div>

### convolutional_autoencoder

<div align=center>

![image5](https://github.com/JianguoZhang1994/Seq2Seq-Gan-Autoencoder/blob/master/autoencoder/assets/convolutional_autoencoder.png)
</div>


### Semi-supervised learning

[This](https://github.com/JianguoZhang1994/Seq2Seq-Gan-Autoencoder/tree/master/Semi-supervised) is an implementation for [Improve techniques for training Gans](https://arxiv.org/pdf/1606.03498.pdf). Which can be modified for semi-supervised learning 

### Seq2Seq Twitter-Chatbot

[This](https://github.com/jianguoz/Seq2Seq-Gan-Autoencoder/tree/master/seq2seq-twitter-chatbot) is 200-line codes of Seq2Seq model for twitter chatbot, the dataset is already uploaded, and the code can be ran directly.

Requirements: Tensorflow 1.8.0, [Tensorlayer>=1.8.4](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#tensorlayer.layers.Seq2Seq), cuda>=9.0.

After tweaking the dataset to bring down the overall % of unknown tokens to 3%, increasing vocabulary to 8000, the results are promising.

| Query					| Reply					|
| ------------- | ------------- |
| donald trump won last nights presidential debate according to snap online polls | thought he was a joke |
| trump campaign please never delete this | he was a joke |
| s new scoop is unk the castro connection how trumps company violated the us embargo against cuba | trump is a disgrace to the debate and the media is not a liar |
| who won the first presidential debate | trump will be a better time for a man |
| just wanna live in unk everything is 10x better there | i was just on the same side and i was like it was a good time |
| the lil girl i forgot her name scares the shit out of me n probably always will lmaooo | she was so cute and she was so cute and she was a bitch |
| question of the day who do you think won last nights debate weigh in using | hillary is a trump supporter and a woman who has a better job than a woman |
| trump is a total embarrassment hes like a unk child who happens to be a professional liar and con man | trump is a lot of money to get a great country |
| chill the fuck out | thats what i said |
| so fun to see what the special effects team created for | wow that is the best thing ive ever seen |
| she was fine too unk voice | she is a good guy |
| the important ones are still here | i dont think so |
| been drinking pumpkin spice protein shake every morning for a week and now i literally cannot even | me too i just got it |
| lmao i just lost my job and i cant even leave early to be in the unk video today | literally the same thing |
| hey happy birthday have a nice day | thank you |

_______________________________________________________________
