# Audio Captioning with BEATs, Conformer & BART
**Winning model of DCASE Challenge 2023 Task 6A**, with the follow-up publication:
- **Improving Audio Captioning Models with Fine-grained Audio Features, Text Embedding Supervision, and LLM Mix-up Augmentation**  
  Shih-Lun Wu, Xuankai Chang, Gordon Wichern, Jee-weon Jung, François Germain, Jonathan Le Roux, and Shinji Watanabe  
  Int. Conf. on Acoustics, Speech, and Signal Processing (**ICASSP**) 2024  
  [[arXiv page](https://arxiv.org/abs/2309.17352)] [[DCASE results](https://dcase.community/challenge2023/task-automated-audio-captioning-results)]
- BibTex citation
  ```
  @inproceedings{wu2024improving,
    title={Improving Audio Captioning Models with Fine-grained Audio Features, Text Embedding Supervision, and LLM Mix-up Augmentation},
    author={Wu, Shih-Lun and Chang, Xuankai and Wichern, Gordon and Jung, Jee-weon and Germain, Fran{\c{c}}ois and Le Roux, Jonathan and Watanabe, Shinji},
    booktitle={Proc. Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP)},
    year={2024}
  }
  ```

## Install Packages
- (Recommended) [Create Conda environment](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) with **Python 3.9**
- [Install PyTorch](https://pytorch.org/get-started/locally/) with the correct CUDA version
- Install dependencies for SPICE metric
  ```
  cd caption_evaluation_tools/coco_caption
  bash get_stanford_models.sh
  cd ../../
  ```
- Install other dependencies
  ```
  pip install -r requirements.txt
  ```

## Download Dataset & Pretrained Model
- Install p7zip (required for unpacking dataset)
  ```
  # if using conda
  conda install bioconda::p7zip
  # if installing to system
  # sudo apt-get install p7zip-full
  ```
- Download Clotho dataset
  ```
  bash download_clotho.sh
  ```
- Install Git-LFS
  ```
  # if using conda
  conda install conda-forge::git-lfs
  git-lfs install

  # if installing to system
  # curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
  # sudo apt-get install git-lfs
  # git-lfs install
  ```
- Get pretrained model
  ```
  bash download_model.sh
  ```

## Reproduce Best Model Results
- Run inference & evaluation code
  ```
  bash run_sampling_reranking.sh
  ```
