# VisionScribe v2 validation

This records development checks, not a caption-quality benchmark. No model was
trained and no dataset was downloaded. Tests ran on Windows 11 with Python 3.11.17,
PyTorch 2.6.0+cpu, Transformers 4.49.0, and an Intel Core i5-1235U with about
15.7 GiB usable RAM. Inference used two CPU threads. Timings vary with host load
and disk-cache state.

## Generation comparison

The completed comparison used three local photos. Three and four beams produced
identical captions on all three, with no observed repeated phrases:

| Photo | Three beams | Four beams | Caption |
| --- | --- | --- | --- |
| Ultralytics-installed bus sample | 47.46 s | 63.20 s | A couple of people standing next to a bus. |
| Street photo embedded in the v1 notebook | 53.76 s | 77.96 s | This little boy was running around the corner of the street. |
| Dogs photo embedded in the v1 notebook | 49.22 s | 53.62 s | Two dogs playing with each other on a road. |

Other settings were `max_new_tokens=40`, `no_repeat_ngram_size=2`,
`early_stopping=True`, and `do_sample=False`. Three beams were retained because
four added 9–45% latency without improving these examples. Three-beam inference
sampled peaks were 1015–1036 MiB, versus 1058–1072 MiB for four beams. Resident
memory after each call was 983–986 MiB versus 1009–1012 MiB. Sampling ran every
50 ms and may miss brief peaks. This small comparison does not establish globally
optimal decoding parameters or formal accuracy.

## Qualitative legacy comparison

The notebook stores v1 captions for its example images. V2 was tested on the
embedded rendered photos (which include plotting borders), so this is not a
controlled rerun on identical original image files:

| Visible scene | Historical v1 output | V2 output |
| --- | --- | --- |
| Two dogs on pavement | Two dogs play with each other on the grass. | Two dogs playing with each other on a road. |
| One child in a street | Two children are walking aross the street. | This little boy was running around the corner of the street. |

V2 improved the surface description and person count in these examples. It can
still misinterpret actions; these results do not demonstrate universal accuracy
improvement or support a BLEU/CIDEr claim. V1 and v2 were not loaded together.

## Resource checks

The original first-use model/processor load, including checkpoint download, took
about 49.9 seconds, excluding startup imports and caption generation. Download
time alone was not isolated, and the checkpoint was not downloaded again for
these follow-up checks.

A fresh follow-up process with the already-populated disk/OS cache took 4.83 s
to load the processor (including its lazy Transformers imports) and 0.47 s to
construct the model: 5.30 s combined, excluding the earlier PyTorch import. The
first bus caption took 47.46 s; repeating it with the production configuration
took 52.40 s. Both returned the same text. Earlier resource loads took 6–11 s;
CPU load, disk caching, and caption length affect these observations.

Three-beam resident memory settled around 983–988 MiB, with sampled inference
peaks of 1015–1036 MiB. The roughly 293 MiB load-only reading is misleading:
memory-mapped checkpoint pages become resident during inference. Memory here is
whole-process RSS, not just parameter storage, and may vary by operating system.

Dynamic INT8 quantization of Linear layers was completed in a separate development
process with the same cached checkpoint and three-beam settings. The model loaded
and generated captions on all three photos; neither the original checkpoint nor
production resources were modified. Cached resources took 5.28 s to load, followed
by 2.90 s for in-place quantization (8.18 s combined). Conversion sampled a
1103 MiB peak. No quantized checkpoint was written.

| Photo | FP32 inference | INT8 inference | INT8 caption |
| --- | --- | --- | --- |
| Bus | 47.46 s | 41.76 s | A couple of people standing next to each other. |
| Child in street | 53.76 s | 44.90 s | A young boy stands in front of the camera. |
| Two dogs | 49.22 s | 53.64 s | A close - up of the legs of a man standing on the ground. |

INT8 omitted the bus, lost the street description, and misidentified the dogs
photo entirely. Its resident memory settled around 1150–1151 MiB, with sampled
inference peaks of 1177–1185 MiB, higher than FP32 here. Two examples ran faster
but the dogs example ran slower. Quantization was **rejected**: it did not deliver
a consistent latency or memory advantage and substantially changed caption
content. Production stays full precision and contains no quantization branch.

The public checkpoint (706,587,995 bytes) was stored in the normal Hugging Face
cache outside the repository. No new weights were added to Git. V2 startup is
lazy: opening the page does not import PyTorch or load a model. Runtime checks
confirm that TensorFlow is not imported, and repeat calls return the same cached
model object. One shared inference lock limits simultaneous generation allocations.

Community Cloud suitability still requires testing the deployed allocation.
Local memory readings do not include a guarantee about Linux framework overhead,
all concurrent upload sessions, or the host's available resources. Python 3.11
Linux dependencies were resolved successfully; local execution used Windows.

The actual Streamlit worker was also measured after browser generation and
regeneration: 965 MiB resident memory and a Windows-reported peak of 1073 MiB
for one small uploaded photo. The Python launcher was excluded. These readings
make a 690 MB allocation insufficient. Streamlit's published resource page gives
an approximate range up to 2.7 GB, explicitly subject to change, so this backend
is plausible on the larger allocation but is not proven reliable on every
Community Cloud app. Additional uploads/sessions need headroom; a roughly 2 GiB
or larger host is a practical deployment target, not a measured minimum.
See [Streamlit's resource documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).
No backend replacement or keep-alive mechanism was added.

## Automated and UI checks

The repeated default suite passed **27 tests, with 1 integration test skipped,
in 9.04 s** under Python 3.11.17. Hugging Face offline mode was enabled, and no
weights were downloaded. Syntax, production imports, and `pip check` passed. Unit tests
cover cache identity, eval/inference mode, generation arguments, friendly errors,
image handling, and Streamlit generation, replacement, regeneration, and reset.

The opt-in integration test also passed under Python 3.11. A local Matplotlib
portrait sample produced “An older woman in a military uniform with a flag behind
her.” Both requests matched. The repeated offline opt-in run passed **1 test in
98.87 s**. The first request from the disk cache took 52.06 s; the subsequent
request took 46.42 s. Cached model identity, evaluation mode,
nonempty caption output, and the absence of TensorFlow were checked. These timings
are local observations, not expected Cloud latency guarantees.

Headless Edge checks against the actual local Streamlit server confirmed branding,
the active GIT sidebar, upload/preview, disabled/enabled Generate Caption, real
GIT generation, caption.txt download with matching text, actual Regenerate, and
Clear Image behavior. A fresh server's first UI request produced “Two dogs
playing with each other on a road.” in 65.4 s including resource loading; its
loading status remained visible alongside the preview and controls. A completed
browser recheck using the same process took 54.5 s to generate and 49.9 s to
regenerate, returning identical captions.

The 1480-pixel desktop view and 390-pixel narrow view were visually inspected.
Columns stack on the narrow view, captions and controls remain readable, and
document width stayed 390 pixels with no horizontal overflow. The analysis stage
is brief; its callback order is covered by unit tests, while loading and generation
statuses were observed in the browser. An initial automation assertion read stale
controls during a rerun; waiting for generation to finish resolved the test race,
without changing production UI code. Download and Clear reruns preserved/reset
results correctly. The README screenshot remains a real local v2 result capture.

README links, screenshot paths, production requirements/imports, preserved
artifact hashes, and Git changes were reviewed. No credential-pattern findings,
new weights, or unignored virtual environments/caches were found. One-off
benchmark/browser scripts, result files, logs, and local test-photo copies were
removed after recording these observations. The reusable integration test,
documentation, screenshot, environments, and external Hub cache remain.

The v1 H5, tokenizer, and notebook retained their original SHA-256 values after
moving to `legacy/`. The legacy implementation, tests, and dependencies are
separated from the default v2 runtime and test suite.
