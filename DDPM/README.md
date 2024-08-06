Pytorch Implementation of vanilla DDPM with einops
 
* Used 2D Linear Attention instead of vanilla self-attention for reasonable memory usage

* Used CelebA-HQ 256x256 resized dataset : https://www.kaggle.com/badasstechie/celebahq-resized-256x256

#### Result from Trained model
Result for MNIST(32x32), CIFAR10(32x32), CelebA-HQ(128x128) respectively

<img src="https://user-images.githubusercontent.com/48702949/139891786-922270ea-a833-4760-8374-50e2599b4d34.jpg" width="240" height="240"/> <img src="https://user-images.githubusercontent.com/48702949/139892619-3f7986f9-202b-4a04-9ccd-20a379df9dbc.jpg" width="240" height="240"/> <img src="https://user-images.githubusercontent.com/48702949/139892655-55423eab-3304-41df-b680-b60958e0090a.jpg" width="240" height="240"/>

---

Also implemented conditional diffusion model which can be applied to a task like Super Resolution. It is similar to SR3.

* Used 2D Linear Attention instead of vanilla self-attention for reasonable memory usage as above.

* Trained with FFHQ thumbnails(128x128) dataset : https://github.com/NVlabs/ffhq-dataset

* Tested with CelebA-HQ 256x256 resized dataset : https://www.kaggle.com/badasstechie/celebahq-resized-256x256

#### Examples of images pairs of training dataset
<img src="https://user-images.githubusercontent.com/48702949/136547999-45a613aa-67eb-42d8-8cbf-16b931164659.jpg" width="866" height="123"/>

#### Result from Trained model
Trained to implement Super Resolution task of 32x32 to 128x128 resolution
* Total training epoch = 250
* Random images were selected from the above CelebA-HQ and resized to 32x32 low resolution inputs to test the trained model
<img src="https://user-images.githubusercontent.com/48702949/136547491-cb8dc04c-c52e-446d-84ee-c315558581a4.jpg" width="866" height="123"/>
