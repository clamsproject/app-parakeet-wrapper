"""
The purpose of this file is to define the metadata of the app with minimal imports.

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from lapps.discriminators import Uri


# DO NOT CHANGE the function name
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification.
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """
    
    # first set up some basic information
    metadata = AppMetadata(
        name="Parakeet Wrapper",
        description="A CLAMS wrapper for NVIDIA NeMo Parakeet ASR models available on huggingface-hub with support for "
                    "punctuation, capitalization, and word-level timestamping.",
        app_license="Apache-2.0",
        identifier="parakeet-wrapper",  # should be a single string without whitespaces. If you don't intent to publish this app to the CLAMS app-directory, please use a full IRI format.
        url="https://github.com/clamsproject/app-parakeet-wrapper",  # a website where the source code and full documentation of the app is hosted
        # (if you are on the CLAMS team, this MUST be "https://github.com/clamsproject/app-parakeet-wrapper"
        # (see ``.github/README.md`` file in this directory for the reason)
        # analyzer_version='version_X', # use this IF THIS APP IS A WRAPPER of an existing computational analysis algorithm
        # (it is very important to pinpoint the primary analyzer version for reproducibility)
        # (for example, when the app's implementation uses ``torch``, it doesn't make the app a "torch-wrapper")
        # (but, when the app doesn't implementaion any additional algorithms/model/architecture, but simply use API's of existing, for exmaple, OCR software, it is a wrapper)
        # if the analyzer is a python app, and it's specified in the requirements.txt
        # this trick can also be useful (replace ANALYZER_NAME with the pypi dist name)
        analyzer_version='20250714',
        analyzer_license="cc-by-4.0",  # short name for a software license
    )
    # Input: audio or video document
    metadata.add_input_oneof(DocumentTypes.AudioDocument, DocumentTypes.VideoDocument)
    # Output: ASR results (text, tokens, timeframes, alignments, sentences)
    metadata.add_output(DocumentTypes.TextDocument)
    metadata.add_output(AnnotationTypes.TimeFrame)
    metadata.add_output(AnnotationTypes.Alignment)
    metadata.add_output(Uri.TOKEN)
    metadata.add_output(Uri.SENTENCE)
    # Parameters for model selection and runtime
    metadata.add_parameter(
        name='contextSize',
        description='Local attention context size for the model. Can be any positive integer, or 0 to set global (full-context) attention. Larger context sizes may improve performance but require a lot more memory. For desktop CUDA device with 12GB VRAM, a context size of around 100 is recommended for full utilization of VRAM. Default is 96',
        type='integer',
        default='96'
    )
    metadata.add_parameter(
        name='modelSize',
        description='Parakeet model size to use. Choices: 110m, 0.6b, 1.1b',
        type='string',
        choices=['110m', '0.6b', '1.1b'],
        default='0.6b'
    )
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
