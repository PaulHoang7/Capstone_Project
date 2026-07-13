"""
Extended data loaders with tone sequence support for Variant B+.

TextAudioSpeakerToneLoader: Extends base loader to also return tone sequences.
TextAudioSpeakerToneCollate: Extends base collate to pad tone sequences.

Imports base classes from vits2_pytorch/data_utils.py without modifying them.
"""

import sys
import os

import torch

# Add vits2_pytorch to path
_VITS2_DIR = os.path.join(os.path.dirname(__file__), '../../vits2_pytorch')
if _VITS2_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_VITS2_DIR))

import commons
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate

# Add Capstone_project root to path
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '../..')
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))

from Capstone_project.tone_encoder.tone_utils import (
    cleaned_text_to_tone_sequence,
    text_to_tone_sequence,
)


class TextAudioSpeakerToneLoader(TextAudioSpeakerLoader):
    """Extends TextAudioSpeakerLoader to also return tone sequences.

    Each item returned is: (text, tone, spec, wav, sid)
    where tone is a LongTensor of tone IDs (0-7) with the same length as text.
    """

    def get_tone(self, text_str):
        """Extract tone sequence from text string, aligned with get_text output.

        Uses the same cleaning and intersperse logic as get_text to ensure
        len(tone) == len(text) at every stage.
        """
        if self.cleaned_text:
            tone_norm = cleaned_text_to_tone_sequence(text_str)
        else:
            tone_norm = text_to_tone_sequence(text_str, self.text_cleaners)
        if self.add_blank:
            tone_norm = commons.intersperse(tone_norm, 0)
        return torch.LongTensor(tone_norm)

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        audiopath, sid, text_str = (
            audiopath_sid_text[0],
            audiopath_sid_text[1],
            audiopath_sid_text[2],
        )
        text = self.get_text(text_str)    # base class: LongTensor of symbol IDs
        tone = self.get_tone(text_str)    # new: LongTensor of tone IDs
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        return (text, tone, spec, wav, sid)

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])


class TextAudioSpeakerToneCollate:
    """Collate function that pads text, tone, spec, wav, and sid.

    Returns: (text_padded, text_lengths, tone_padded,
              spec_padded, spec_lengths, wav_padded, wav_lengths, sid)
    """

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        # batch items: (text, tone, spec, wav, sid)
        # Sort by spec length (descending) for efficient packing
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[2].size(1) for x in batch]),  # spec at index 2
            dim=0,
            descending=True,
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[2].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(
            len(batch), batch[0][2].size(0), max_spec_len
        )
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

        text_padded.zero_()
        tone_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            tone = row[1]
            tone_padded[i, :tone.size(0)] = tone

            spec = row[2]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[3]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[4]

        if self.return_ids:
            return (
                text_padded,
                text_lengths,
                tone_padded,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                sid,
                ids_sorted_decreasing,
            )
        return (
            text_padded,
            text_lengths,
            tone_padded,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            sid,
        )
