# voice-dataset-labeler

Acoustic labelling of voice recordings: f0, loudness and formants per window, from classical signal processing (numpy and scipy).

- `extractors/`: formant extractors behind one interface: `tvlp.py` (time-varying linear prediction with temporal smoothing, numpy and the standard library only) and `praat.py`.
- `labeller/`: the windowed labelling pipeline (acoustic features, formants, post-processing).
- `synth/`: synthetic test signals.

The TVLP tracker is scored as a baseline in [voice-analyser](https://github.com/hannahsq/voice-analyser), which pins this repo as a submodule (decision D-007 there); that project is where current work happens.

Licence: MIT.
