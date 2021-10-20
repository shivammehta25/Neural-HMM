# Neural HMMs are all you need (for high-quality attention-free TTS)
---

[paper_link]: https://arxiv.org/abs/2108.13320
[demo_page]: https://shivammehta007.github.io/Neural-HMM/
[ljspeech_link]: https://keithito.com/LJ-Speech-Dataset/
[github_link]: https://github.com/shivammehta007/Neural-HMM.git
[github_new_issue_link]: https://github.com/shivammehta007/Neural-HMM/issues/new
[docker_install_link]: https://docs.docker.com/get-docker/
[tacotron2_link]: https://github.com/NVIDIA/tacotron2
[pretrained_model_link]: https://www.test.com
[nvidia_waveglow_link]: https://drive.google.com/file/d/1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF/view
[pytorch_lightning_link]: https://github.com/PyTorchLightning/pytorch-lightning
[pytorch_dataloader_issue_link]: https://github.com/pytorch/pytorch/issues/57273

This is the official code repository for [Neural HMMs are all you need (for high-quality attention-free TTS)][paper_link].

## Abstract
 Neural sequence-to-sequence TTS has achieved significantly better output quality than statistical speech synthesis using HMMs. However, neural TTS is generally not probabilistic and the use of non-monotonic attention both increases training time and introduces "babbling" failure modes that are unacceptable in production. This paper demonstrates that the old and new paradigms can be combined to obtain the advantages of both worlds, by replacing the attention in Tacotron 2 with an autoregressive left-right no-skip hidden Markov model defined by a neural network. This leads to an HMM-based neural TTS model with monotonic alignment, trained to maximise the full sequence likelihood without approximations. We discuss how to combine innovations from both classical and contemporary TTS for best results. The final system is smaller and simpler than Tacotron 2, and learns to speak with fewer iterations and less data, whilst achieving the same naturalness prior to the post-net. Unlike Tacotron 2, our system also allows easy control over speaking rate. 
 
 Visit our [demo page][demo_page] for audio examples.


## Training and setup using LJSpeech
1. Download and extract the [LJSpeech dataset][ljspeech_link]. Place it in `data` folder such that the directory becomes `data/LJSpeech-1.1`. otherwise update the filelists in `data/filelists` accordingly.
2. Clone this repository ```git clone https://github.com/shivammehta007/Neural-HMM.git``` 
3. Initalise the submodules ```git submodule init; git submodule update```
4. Make sure you have [docker installed][docker_install_link] and running.
    * It is recommended to use docker (it manages the CUDA runtime libraries and Python dependencies itself specified in Dockerfile)
    * Alternatively, If you do not intend to use docker, you can use pip to install the dependencies using ```pip install -r requirements.txt``` 
5. Run ``bash start.sh`` and it will install all the dependencies and run the container.
6. Check `src/hparams.py` for hyperparameters and set gpus.
    1. For multigpu training, set gpus to ```[0, 1 ..]```
    2. For CPU training (Not recommended), set gpus to an empty list ```[]```
    3. Check the location of transcriptions 
7. Run ```python train.py``` to train the model.
    1. Checkpoints will be saved in the `hparams.checkpoint_dir`.
    2. Tensorboard logs will be saved in the `hparams.tensorboard_log_dir`.
8. To resume training, run ```python train.py -c <CHECKPOINT_PATH>```

## Synthesis
1. Download our `[pre-trained model]` (Available soon).
2. Download Nvidia's [WaveGlow model][nvidia_waveglow_link].
3. Run jupyter notebook and open ```synthesis.ipynb```.


## Miscellaneous
### Mix Precision Training or Full Precision Training
- In ```src.hparams.py``` change ```hparams.precision``` to ```16``` for mix precision and ```32``` for full precision.
### Multi-GPU Training or Single GPU Training
- Since the code uses PyTorch Lightning, Setting gpus more than one element in the list will enable multi-gpu training. So change ```hparams.gpus``` to ```[0, 1, 2]``` for multi-gpu training and single element ```[0]``` for single GPU training.


### Known Issues/Warnings

#### PyTorch dataloader
* If you encounter this error message ```[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)```. This is a known issue in [PyTorch Dataloader][pytorch_dataloader_issue_link]. 
* It will be fixed when PyTorch will release new Docker container image with updated version of Torch. If you are not using docker this can be removed with ```torch > 1.9.1```

## Citation Information
If you use Neural HMMs are all you need (for high-quality attention-free TTS) for your research, please cite our paper:
```
@article{mehta2021neural,
  title={Neural {HMM}s are all you need (for high-quality attention-free {TTS})},
  author={Mehta, Shivam and Sz{\'e}kely, {\'E}va and Beskow, Jonas and Henter, Gustav Eje},
  journal={arXiv preprint arXiv:2108.13320},
  year={2021}
}
```

## Support
If you have any questions or comments, please open an [issue][github_new_issue_link] on our GitHub repository.

## Acknowledgement
The code implementation is inspired by NVIDIA team's implementation of [Tacotron 2][tacotron2_link] and uses [PyTorch Lightning][pytorch_lightning_link] for boilerplate-free code.
