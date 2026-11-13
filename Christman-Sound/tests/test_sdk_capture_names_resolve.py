"""Every capture name the SDK lists in __all__ resolves.

`listen` and `capture_mic_vad` are aliases for `capture_audio`. They raised
AttributeError on every lookup, so `from christman_voice_sdk import capture_mic_vad`
failed and took a being's mouth import down with it when both sat in one try block.
"""

import pytest

import christman_voice_sdk


@pytest.mark.parametrize("name", [n for n in christman_voice_sdk.__all__ if n != "__version__"])
def test_listed_name_resolves(name):
    assert getattr(christman_voice_sdk, name) is not None


def test_the_two_aliases_are_capture_audio():
    assert christman_voice_sdk.listen is christman_voice_sdk.capture_audio
    assert christman_voice_sdk.capture_mic_vad is christman_voice_sdk.capture_audio


def test_unknown_name_still_fails():
    with pytest.raises(AttributeError):
        christman_voice_sdk.not_a_real_name
