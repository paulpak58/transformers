# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from transformers import AutoProcessor, DiaFeatureExtractor, DiaProcessor, DiaTokenizer
from transformers.testing_utils import require_torch
from transformers.utils import PROCESSOR_NAME, is_torch_available


if is_torch_available():
    import torch

    from transformers import DacConfig, DacModel


def tiny_codec():
    config = DacConfig(
        encoder_hidden_size=4,
        downsampling_ratios=[2, 2],
        decoder_hidden_size=8,
        n_codebooks=2,
        codebook_size=16,
        codebook_dim=4,
    )
    return DacModel(config).eval()


@require_torch
class ProcessorMixinAudioTokenizerTest(unittest.TestCase):
    """The base class saves a bundled audio tokenizer in a subfolder of the processor directory and loads it back."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def processor(self, audio_tokenizer):
        return DiaProcessor(DiaFeatureExtractor(), DiaTokenizer(), audio_tokenizer=audio_tokenizer)

    def assert_same_weights(self, actual, expected):
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)

    def test_bundled_codec_round_trips_through_the_saved_directory(self):
        codec = tiny_codec()
        processor = self.processor(codec)
        saved = os.path.join(self.tmpdir, "saved")
        processor.save_pretrained(saved)
        with open(os.path.join(saved, PROCESSOR_NAME), encoding="utf-8") as f:
            config = json.load(f)
        self.assertEqual(config["audio_tokenizer_subfolder"], "audio_tokenizer")
        self.assertNotIn("audio_tokenizer", config)
        self.assertTrue(os.path.isfile(os.path.join(saved, "audio_tokenizer", "config.json")))
        # the codec is still held by the processor, not detached by the serialization
        self.assertIs(processor.audio_tokenizer, codec)

        reloaded = DiaProcessor.from_pretrained(saved)
        self.assertIsInstance(reloaded.audio_tokenizer, DacModel)
        self.assert_same_weights(reloaded.audio_tokenizer, codec)
        self.assertEqual(reloaded.to_dict(), processor.to_dict())

        moved = os.path.join(self.tmpdir, "moved")
        shutil.move(saved, moved)
        reloaded = AutoProcessor.from_pretrained(moved, local_files_only=True)
        self.assert_same_weights(reloaded.audio_tokenizer, codec)
        # a second save bundles the codec again, so the copy stays self-contained
        resaved = os.path.join(self.tmpdir, "resaved")
        reloaded.save_pretrained(resaved)
        self.assertTrue(os.path.isfile(os.path.join(resaved, "audio_tokenizer", "config.json")))
        self.assert_same_weights(DiaProcessor.from_pretrained(resaved).audio_tokenizer, codec)

    def test_nested_subfolder_from_a_hand_written_config(self):
        codec = tiny_codec()
        saved = os.path.join(self.tmpdir, "saved")
        self.processor(codec).save_pretrained(saved)
        shutil.move(os.path.join(saved, "audio_tokenizer"), os.path.join(saved, "codecs", "dac"))
        with open(os.path.join(saved, PROCESSOR_NAME), encoding="utf-8") as f:
            config = json.load(f)
        config["audio_tokenizer_subfolder"] = "codecs/dac"
        with open(os.path.join(saved, PROCESSOR_NAME), "w", encoding="utf-8") as f:
            json.dump(config, f)

        reloaded = DiaProcessor.from_pretrained(saved)
        self.assert_same_weights(reloaded.audio_tokenizer, codec)
        self.assertEqual(reloaded.to_dict()["audio_tokenizer_subfolder"], "codecs/dac")
        resaved = os.path.join(self.tmpdir, "resaved")
        reloaded.save_pretrained(resaved)
        self.assertTrue(os.path.isfile(os.path.join(resaved, "codecs", "dac", "config.json")))
        self.assert_same_weights(DiaProcessor.from_pretrained(resaved).audio_tokenizer, codec)

    def test_bundled_codec_follows_the_repository_revision_and_subfolder(self):
        codec = tiny_codec()
        saved = os.path.join(self.tmpdir, "repo", "nested")
        self.processor(codec).save_pretrained(saved)
        repo = os.path.join(self.tmpdir, "repo")
        with patch.object(DacModel, "from_pretrained", wraps=DacModel.from_pretrained) as loader:
            reloaded = DiaProcessor.from_pretrained(repo, subfolder="nested", revision="v1")
        self.assert_same_weights(reloaded.audio_tokenizer, codec)
        loader.assert_called_once()
        self.assertEqual(loader.call_args.args, (repo,))
        self.assertEqual(loader.call_args.kwargs["subfolder"], os.path.join("nested", "audio_tokenizer"))
        self.assertEqual(loader.call_args.kwargs["revision"], "v1")
        self.assertNotIn("tokenizer", loader.call_args.kwargs)

    def test_referenced_codec_keeps_its_descriptor(self):
        codec_dir = os.path.join(self.tmpdir, "codec")
        tiny_codec().save_pretrained(codec_dir)
        codec = DacModel.from_pretrained(codec_dir)
        processor = self.processor(codec)
        self.assertEqual(
            processor.to_dict()["audio_tokenizer"],
            {"audio_tokenizer_class": "DacModel", "audio_tokenizer_name_or_path": codec_dir},
        )
        saved = os.path.join(self.tmpdir, "saved")
        processor.save_pretrained(saved)
        self.assertFalse(os.path.exists(os.path.join(saved, "audio_tokenizer")))
        reloaded = DiaProcessor.from_pretrained(saved)
        self.assertEqual(reloaded.audio_tokenizer.name_or_path, codec_dir)
        self.assert_same_weights(reloaded.audio_tokenizer, codec)

    def test_codec_passed_to_from_pretrained_replaces_the_saved_one(self):
        saved = os.path.join(self.tmpdir, "saved")
        self.processor(tiny_codec()).save_pretrained(saved)
        codec = tiny_codec()
        with patch.object(DacModel, "from_pretrained", wraps=DacModel.from_pretrained) as loader:
            processor = DiaProcessor.from_pretrained(saved, audio_tokenizer=codec)
        loader.assert_not_called()
        self.assertIs(processor.audio_tokenizer, codec)
        # the model never reaches the tokenizer's init kwargs, which are serialized on save
        self.assertNotIn("audio_tokenizer", processor.tokenizer.init_kwargs)
        resaved = os.path.join(self.tmpdir, "resaved")
        processor.save_pretrained(resaved)
        self.assert_same_weights(DiaProcessor.from_pretrained(resaved).audio_tokenizer, codec)
