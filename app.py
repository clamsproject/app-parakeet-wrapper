"""
CLAMS wrapper for NVIDIA NeMo Parakeet ASR models. Supports model selection (size/type), word-level timestamping, and outputs MMIF annotations for text, tokens, timeframes, alignments, and sentences.
"""

import argparse
import logging
import tempfile

import ffmpeg
import nemo.collections.asr as nemo_asr
from clams import ClamsApp, Restifier
# For an NLP tool we need to import the LAPPS vocabulary items
from lapps.discriminators import Uri
from mmif import Mmif, AnnotationTypes, DocumentTypes

# Imports needed for Clams and MMIF.
# Non-NLP Clams applications will require AnnotationTypes

PARAKEET_MODEL_SIZE_MAP = {
    '110m': "nvidia/parakeet-tdt_ctc-110m",
    '0.6b': "nvidia/parakeet-tdt-0.6b-v2",
    '1.1b': "nvidia/parakeet-tdt_ctc-1.1b",
}
PARAKEET_MODEL_VERSIONS = {
    "nvidia/parakeet-tdt_ctc-110m": "431a349f3051ab85c22b9b7a2741b5fe77065665",
    "nvidia/parakeet-tdt-0.6b-v2": "d97f7ac5d85e7185b7a7c4771c883c0e26d1d16f",
    "nvidia/parakeet-tdt_ctc-1.1b": "675e78684c83ae21e2a8fb042726b66d91b9ba3d",
}
# as of writing (analyzer_ver) other models does not support punctuation and capitalization, so we do not support them


class ParakeetWrapper(ClamsApp):

    def __init__(self):
        super().__init__()
        self.model_cache = {}

    def _appmetadata(self):
        from metadata import appmetadata
        return appmetadata()

    def _get_model(self, model_size):
        model_name = PARAKEET_MODEL_SIZE_MAP[model_size]
        if model_name not in self.model_cache:
            if model_name not in PARAKEET_MODEL_VERSIONS:
                raise ValueError(f"Unsupported model size {model_size}, note that this wrapper does not "
                                 f"support all the Parakeet models. See parameters specification in the appmetadata.")
            # meno api does not support model versioning, and always downloads the `main` HEAD
            # we need to pre-download model using HF api 
            from huggingface_hub import snapshot_download

            # Download the model repository to the local cache
            snapshot_download(repo_id=model_name, revision=PARAKEET_MODEL_VERSIONS[model_name])
            self.model_cache[model_name] = nemo_asr.models.ASRModel.from_pretrained(model_name)
        return self.model_cache[model_name]

    @staticmethod
    def convert_to_16k_wav_bytes(input_path):
        """
        Converts an audio or video file to 16kHz mono WAV format using ffmpeg-python.
        Returns the WAV data as bytes (in-memory, no output file).
        """
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_wav.close()  # Close the file so ffmpeg can write to it
        ffmpeg.input(input_path).output(temp_wav.name, format='wav', ac=1, ar=16000).overwrite_output().run()
        return temp_wav.name  # Return the path to the temporary WAV file
    
    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        model_size = parameters['modelSize']
        model = self._get_model(model_size)
        # Find audio/video documents
        for srcdoc in mmif.get_documents_by_type(DocumentTypes.AudioDocument) \
                   + mmif.get_documents_by_type(DocumentTypes.VideoDocument):
            audio_path = srcdoc.location_path()
            resampled = self.convert_to_16k_wav_bytes(audio_path)
            
            # disable global attention to reduce memory usage
            model.change_attention_model("rel_pos_local_attn", [parameters['contextSize'], parameters['contextSize']])
            model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select

            # Run ASR with word-level timestamping
            result = model.transcribe([resampled], timestamps=True)
            # result is a list of `Hypothesis` objects (maybe one for each audio file?)
            result = result[0]  # Assuming single audio file input, take the first result
            # not result has (among others) `text` (str), `words` (list of str) and `timestamp` (dict) attributes
            # and the `timestamp` dict has keys 'timestep', 'word', 'segment', and 'char' with lists of timestamps
            # and values are list of dicts with 'start', 'end', and the corresponding segmentation type (e.g., 'word', 'segment', 'char')
            # time notations are in seconds.

            # create a new view per audio/video input document
            view = mmif.new_view()
            self.sign_view(view, parameters)

            # convert result.text to a TextDocument annotation
            raw_text = result.text.strip()
            td_ann = view.new_textdocument(raw_text, lang='en')
            view.new_annotation(AnnotationTypes.Alignment, source=srcdoc.long_id, target=td_ann.long_id)
            char_offset = 0
            segment_token_ids = []
            segment_idx = 0
            segments_offset = 0
            for word_dict in result.timestamp["word"]:
                
                raw_token = word_dict["word"]
                # find this token’s position in the entire text
                tok_start = raw_text.index(raw_token, char_offset)
                tok_end = tok_start + len(raw_token)
                char_offset = tok_end

                token = view.new_annotation(
                    Uri.TOKEN,
                    word=raw_token,
                    start=tok_start,
                    end=tok_end,
                    document=f'{td_ann.long_id}'
                )
                segment_token_ids.append(token.long_id)

                tf_start = int(word_dict["start"] * 1000)
                tf_end = int(word_dict["end"] * 1000)
                tf = view.new_annotation(
                    AnnotationTypes.TimeFrame,
                    label="speech",
                    start=tf_start,
                    end=tf_end
                )
                view.new_annotation(
                    AnnotationTypes.Alignment,
                    source=tf.long_id,
                    target=token.long_id
                )
                # simultaneously track the segment index while looping through words
                # note that char_offset is tracking offset from the very beginning of the text
                # while segments_offset is tracking offset from the beginning of the current segment
                cur_segment_text = result.timestamp["segment"][segment_idx]['segment'].strip()
                # when we have reached the end of the current segment, see next segment
                if char_offset - segments_offset > len(cur_segment_text):
                    view.new_annotation(
                        Uri.SENTENCE,
                        targets=segment_token_ids,
                        text=cur_segment_text,
                    )
                    segment_idx += 1
                    segments_offset += len(cur_segment_text) + 1  # +1 for the space after each segment 
                    segment_token_ids = []  # reset for the next segment
        return mmif

def get_app():
    return ParakeetWrapper()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # add more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    # if get_app() call requires any "configurations", they should be set now as global variables
    # and referenced in the get_app() function. NOTE THAT you should not change the signature of get_app()
    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()

