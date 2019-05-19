Timbrer: Musical Timbre Transfer
====================

Installation:

install anaconda
run anaconda prompt

run:
conda env create -f environment.yml

## Dataset generation
Run `generate_set_singlethread.py`. The input midi files will be downloaded and a dataset of spectrogram pairs will be created.

## Training the network
Modify `pix2pixHD/train.py` by setting the correct test and dataset names. Start the training process by running the file. Intermediate results will be visible in the `pix2pixHD/checkpoints/<test_name>/web/`folder.

## Running inference
Modify `pix2pixHD/test.py` by setting the test and dataset names. Run the file, which will generate the output spectrograms into the `pix2pixHD\results\<test_name>` folder.

## Reconstructing the waveforms
Run `reconstruct.py` and give it a list of `.npy` spectrograms. They will be processed in batches for greater throughput. Expect around ~5s/image on a modern GPU. The resulting waveforms will be placed in the same folder as the inputs.

## Copyright notice
This work is a derivative of [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) by NVIDIA Corporation.
