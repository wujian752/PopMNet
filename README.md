# PopMNet

[PopMNet](https://wujian752.github.io/PopMNet/) is a model for generating structured pop music melodies.
It comprises a CNN-based Structure Generation Net (SGN) and an RNN-based Melody Generation Net (MGN).
The SGN generates melody structures, while the MGN generates melodies conditioned on these structures and chord progressions.

## Dependencies

### python

To run the code in this repository, Python 3.6 and several other dependencies are required. 
To install these dependencies, execute the following command:

```python
pip install -r requirements.txt
```

### Magenta

[Magenta](https://github.com/tensorflow/magenta) is a research project that explores the role of machine learning in the creation of art and music. 
We utilize some code from Magenta to process our data.

## Dataset
The MusicXML files should be stored in the folder ```raw_data```.

## Usage
### Data process
The initial step is to preprocess the data:

```bash
python data_process.py --input-dir raw_data
```

The MusicXML files in the folder ```raw_data``` will be converted to ```magenta.music.LeadSheet``` objects and saved in ```data/LS.pickle```.


### SGN
Train the SGN:

```bash
python sgn/main.py  --use_cuda
```

The checkpoints will be saved in ```results/sgn/checkpoints```.
If no Nvidia GPU is available, please remove the ```--use_cuda``` option.

To generate melody structures:

```bash
python sgn/generate.py --use_cuda  --sample_num 100
```

Melody structures will be saved in ```results/sgn/samples``` in json formats.

### MGN

To train the MGN:

```bash
python mgn/main.py data/LS.pickle --trainer plan --plan --chords --encoder-decoder onehot --save-dir results/mgn --render-chords
```

The checkpoints will be saved in ```results/sgn/checkpoints```.

To generate melodies conditioned on melody structures:

```bash
python mgn/main.py data/LS.pickle --inference --trainer plan --generate-dir results/mgn/generate \
								  --restore-file results/mgn/checkpoints/best_checkpoint \
								  --generate-num 100 \
								  --relations results/sgn/samples
```

Note that the ```--generate-num``` should be less than the ```--sample_num``` because the generation of melodies requires generated structures as a condition.

# Citing

Please cite the following paper if you use the code provided in this repository.

```bibtex
@article{WU2020103303,
title = {PopMNet: Generating structured pop music melodies using neural networks},
journal = {Artificial Intelligence},
volume = {286},
pages = {103303},
year = {2020},
issn = {0004-3702},
doi = {https://doi.org/10.1016/j.artint.2020.103303},
url = {https://www.sciencedirect.com/science/article/pii/S000437022030062X},
author = {Jian Wu and Xiaoguang Liu and Xiaolin Hu and Jun Zhu},
}
```
