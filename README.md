# See, Hear, Explore: Curiosity via Audio-Visual Association
[[Website]](https://vdean.github.io/audio-curiosity.html)&nbsp;&nbsp;&nbsp;&nbsp;[[Video]](https://youtu.be/DMiW5hwsoeo)

[Victoria Dean](https://vdean.github.io/)&nbsp;&nbsp;&nbsp;&nbsp;[Shubham Tulsiani](https://shubhtuls.github.io/)&nbsp;&nbsp;&nbsp;&nbsp;[Abhinav Gupta](http://www.cs.cmu.edu/~abhinavg/)

Carnegie Mellon University, Facebook AI Research

This is an implementation of our paper on curiosity via audio-visual association. In this paper, we introduce a form of curiosity that rewards novel associations between different sensory modalities. Our approach exploits multiple modalities to provide a stronger signal for more efficient exploration. Our method is inspired by the fact that, for humans, both sight and sound play a critical role in exploration. We present results on Atari and Habitat (a photorealistic navigation simulator), showing the benefits of using an audio-visual association model for intrinsically guiding learning agents in the absence of external rewards.

This code trains an audio-visual exploration agent in Atari environments. It does not yet have support for the Habitat navigation setting, as the underlying environment is not open-sourced.

## Installation
```bash
git clone git@github.com:vdean/audio-curiosity.git
cd audio-curiosity
conda env create -f environment.yml
```
### Retro Setup 
You will need to download and import the Atari 2600 game ROMs to retro. The below commands should do this automatically (you may need to install unrar). For more details, see: https://github.com/openai/retro/issues/53
```bash
wget http://www.atarimania.com/roms/Roms.rar && unrar x Roms.rar && unzip Roms/ROMS.zip
python3 -m retro.import ROMS/
```

To add audio support, copy our modified retro_env.py into retro. If you set up a conda environment as instructed above, this command should work:
```bash
cp retro_env.py ~/anaconda3/envs/venv_audio_curiosity/lib/python3.7/site-packages/retro/retro_env.py
```

### Baselines Setup
Modify the following line in ~/anaconda3/envs/venv_audio_curiosity/lib/python3.7/site-packages/baselines/logger.py:
```python
summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
```
to
```python
summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items() if v != None])
```

## Usage
### Training
The following command should train an audio-visual exploration agent on Breakout with default experiment parameters.
```bash
python run.py --env_kind=Breakout --feature_space=fft --train_discriminator=True --discriminator_weighted=True
```

To train a visual prediction baseline agent on Breakout:
```bash
python run.py --env_kind=Breakout --feature_space=visual
```

### Creating Plots
To create a figure with the 12 Atari environments we used (after you have trained), run:
```bash
python make_plots.py --all=True --mean=True
```

### Acknowledgement
Code built off the open-source reposity from Large-Scale Study of Curiosity-Driven Learning [1]: https://github.com/openai/large-scale-curiosity

[1] [Yuri Burda, Harri Edwards, Deepak Pathak, Amos Storkey, Trevor Darrell, and Alexei A Efros. Large-scale study of curiosity-driven learning. arXiv preprint arXiv:1808.04355, 2018.](https://arxiv.org/abs/1808.04355)
