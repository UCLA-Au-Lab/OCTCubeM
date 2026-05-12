## OCTCube-M - A multi-modal foundation model for OCT and *en face* retinal imaging


This is the official repo for [OCTCube-M: A 3D multimodal optical coherence tomography foundation model for retinal and systemic diseases with cross-cohort and cross-device validation](https://arxiv.org/abs/2408.11227).

Please contact 	**zucksliu@cs.washington.edu** or **swang@cs.washington.edu** if you have questions.

### Update `2025-06-29`
- Docker available! Check out the [docker](https://hub.docker.com/repositories/zucksliu)
- You can try the basic development environment by running:
``` docker pull zucksliu/octcubem_pytorch_2.4.1-cuda12.4-cudnn9 ```
- To run the inference notebook, you can use the following docker:
``` docker pull zucksliu/octcube-multi-cls-tool/general ```
- We update the inference notebook with two threshold options, one for general healthy / diseased classification (distribution based on the training set of OCTCube), and one for screening purposes: this setting is more conservative and will only classify a volume as diseased if it is very likely to be diseased. This is useful for screening purposes where you want to minimize false positives. For more details, please refer to the [inference notebook](https://github.com/ZucksLiu/OCTCubeM/blob/main/inference_OCTCube.ipynb)
- Have fun computing with OCTCube-M! 🚀

### 🚀Checklist
- [] Release a tri-modal OCTCube-EF model instance trained on AI-READI dataset.
- [] Prepare the inference code for using our OCTCube-IR model.
- [] Release saliency generation code to visualize where OCTCube is looking.👀
- [x] Prepare how to prepare the public datasets used in this study.
- [x] Prepare the inference code and example OCT data volumes to try the classification model.
- [x] Release the model on hugging face! Check [here](https://huggingface.co/zucksliu/OCTCubeM). We have the source OCTCube model, the bi-modal OCTCube-IR model, and a multi-tasking classification model for 8 retinal diseases.
- [x] Release the code for pre-training, fine-tuning and multi-modal training.



### 📝Key features of OCTCube

- OCTCube is pre-trained on 26,685 3D OCT volumes encompassing 1.62 million 2D OCT images.
- Retinal diseases prediction: Best performance on predicting 8 retinal diseases (AMD, DME, POAG, DR, ERM, CRAO/CRVO, VD, RNV) with strong generalizability.
- Cross-cohort, cross-devices and cross-modality ability: Predicts cross-organ nodule malignancy, low cardiac ejection fraction, diabetes, and hypertension.

### 📝Key features of OCTCube-M
- OCTCube-IR allows accurate retrieval between OCT and IR images.
- OCTCube-EF excels in predicting the growth rate of geographic atrophy (GA) by integraing OCT, FAF and IR images.


### Repo structure

This repository is divided into three main parts:

1. **Pre-training**: This section contains the code and instructions for pre-training the models. Detailed information can be found in the `OCTCubeM/Pre-training` directory.

2. **OCTCube**: This README will mainly cover the `OCTCubeM/OCTCube` part, which includes fine-tuning and using the OCTCube models for various tasks.

3. **retinal-COEM**: This section focuses on multimodal tasks and can be found in the `OCTCubeM/retinal-COEM` directory.

For more details on the Pre-training and retinal-COEM parts, please refer to their respective README files inside their directories. We also provide a `.gitignore` file and a `requirement.txt` file in each directory to help you avoid uploading unnecessary files if you only want to fork one part of the repository.


### 🔧 Install Environment

1. **Create environment with conda:**

    ```sh
    conda create -n octcube python=3.10 -y
    conda activate octcube
    ```

2. **Install dependencies:**

    ```sh
    git clone https://github.com/ZucksLiu/OCTCubeM.git
    cd OCTCubeM
    ```

3. **Install PyTorch and CUDA:**

    - To install PyTorch via conda:

        ```sh
        conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
        ```

    - To install PyTorch via pip:

        ```sh
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
        ```

4. **Install additional dependencies:**

    ```sh
    pip install -r requirement_no_torch.txt
    ```

5. **Install flash attention:**

    ```sh
    pip install ninja  # To make it faster
    pip install packaging
    pip install flash-attn==2.5.2 --no-build-isolation
    ```

    [Optional] If your platform cannot directly compile flash attention due to CUDA version incompatibility, you can try to install cuda first via conda:

    ```sh
    conda install -c nvidia cuda==11.8 cuda-nvcc=11.8
    ```

Following the above step, you should have all the necessary dependencies installed for all three parts of the repository.

### Assets, checkpoints and Data preparation
We introduce how to prepare the data, model weights and assets for pre-training, fine-tuning and inference.

In `OCTCubeM/ckpt/`, put all the model checkpoints you want to use, including in case you want to use [RETFound OCT model](https://github.com/rmaphoh/RETFound_MAE) to fine-tune on your own data or to initialize as the *en face* encoder.

In `OCTCubeM/assets/`, there are several stuff:
- `oct_examples/`: Example OCT data volumes provided for trying our inference botebook!
- `ext_oph_datasets/`: Public datasets used in this study. Please refer to [OCTCubeM/assets/BENCHMARK.md] for how to prepare those datasets using our provided tools.
- `Oph_cls_task`:" Example to show how to prepare your in-house data for fine-tuning.
- `SLIViT/`, scripts to process the data used by SLIViT and cross-modality experiments, and the place to put SLIViT pre-trained weights.
- `aireadi_v1_patient.txt`: Patient id list used for reproducing our experiment on AI-READI-1.0 dataset, this is to help to better reproduce our results at the era of AI-READI-2.0:)

In `OCTCubeM/OCTCube/assets/`, we put the folder for saving the NodleCT3D dataset and the metadata of the Echo dataset.

#### What should I do here?
In general, if you just want to try our inference notebook, just download our `OCTCube_multitask_cls.pth` and put it in `OCTCubeM/ckpt/`. Then, download the [example OCT data volumes](https://drive.google.com/drive/folders/1Xxa2C5UrKgtBkP6RYBpm3kGLCci55DvH?usp=sharing) and put it in `OCTCubeM/assets/oct_examples/`, and you should be fine with running `inference_OCTCube.ipynb`!🚀

If you want to try fine-tune OCTCube on the public datasets, follow the instruction in [OCTCubeM/assets/BENCHMARK.md] to prepare the datasets and put them in `OCTCubeM/assets/ext_oph_datasets/`. Then, download the pre-trained weights `OCTCube.pth` and put them in `OCTCubeM/ckpt/`. Then look into `OCTCubeM/OCTCube/README.md` for how to fine-tune the model.

If you want to try mutli-modal part of OCTCube, download `mm_octcube_ir.pt` into `OCTCubeM/ckpt/`, and then look into `OCTCubeM/retinal-COEM/README.md` for how to prepare the data and run the code.


### 🌱Play with `OCTCube_inference.ipynb`
In the notebook, we provide the minimum example to show how to load out trained multi-tasking classification model and use it to predict the disease labels of a given OCT data volume. You can also use your own data by changing the path in the notebook. We aim to add gradient saliency support very soon, stay tuned!


### Trouble-shooting
If you have any trouble running the code, please feel free to open an issue or contact us via **zucksliu@cs.washington.edu**. We are welcome to any help that improve the repo!


### 📃Citation

If you find this repository useful, please consider citing this paper:
```
@article{liu2024octcube,
  title={OCTCube: a 3D foundation model for optical coherence tomography that improves cross-dataset, cross-disease, cross-device and cross-modality analysis},
  author={Liu, Zixuan and Xu, Hanwen and Woicik, Addie and Shapiro, Linda G and Blazes, Marian and Wu, Yue and Lee, Cecilia S and Lee, Aaron Y and Wang, Sheng},
  journal={arXiv preprint arXiv:2408.11227},
  year={2024}
}
```


