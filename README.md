# Parakeet Wrapper

## Description

CLAMS app that wraps around [parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
to perform ASR on audio or video files.

## User instructions

General user instructions for CLAMS apps are available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp/).

### System requirements

This app requires Python 3.10 or higher.
For local installation of required Python modules, see [requirements.txt](https://github.com/clamsproject/app-parakeet-wrapper/blob/main/requirements.txt). 

### Configurable runtime parameters

For the full list of parameters, please refer to the app metadata from the [CLAMS App Directory](https://apps.clams.ai/)
or the [`metadata.py`](https://github.com/clamsproject/app-parakeet-wrapper/blob/main/metadata.py).

### Input and output details

The app takes a `MMIF file` with an `AudioDocument` or `VideoDocument` and outputs a `TextDocument` annotation with an `Alignment` for each `Token` and `Sentence`. 
