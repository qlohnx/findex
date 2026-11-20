# Lab 08 showcase plan (recording still required)

## 2–3 minute demo-video script

Record your own screen and voice after the Lab 08 build is deployed. Do not
present the currently public Lab 07 instance as if it already has semantic
search.

| Time | Screen | Narration cue |
| --- | --- | --- |
| 0:00–0:20 | README first screen and live URL | Findex turns streamed text/Wikipedia pages into an inverted index and searchable web app. |
| 0:20–0:55 | Web UI, exact keyword query | Show BM25 results, highlighted snippets and a document page. |
| 0:55–1:25 | Semantic then hybrid toggle | Try a paraphrase that keyword search misses; explain best-chunk cosine and RRF briefly. |
| 1:25–2:00 | Profiling report and flame graph | Show that JSON dominates cold/build workflows, while scalar BM25 was the warm-query hotspot. |
| 2:00–2:30 | Speedup table and tests | Point to the measured 4.52× three-term ranking gain and explain why the rare term does not speed up. |
| 2:30–2:50 | Limitations | Model memory, limited corpus slice, and cold JSON load remain honest next steps. |

## Five-minute class showcase

1. Live demo: keyword, semantic paraphrase and hybrid; API `?mode=`.
2. Measurement: 9.554 ms scalar vs 2.116 ms NumPy on the 5,000-document,
   three-term warm query, measured as the median of three 100-query batches.
3. Bug story: `py-spy` could not inspect the Windows venv launcher. Switching
   the profiled child process to the base Python interpreter, with the project
   and venv site-packages on `PYTHONPATH`, produced valid flame graphs. This
   changed the profiling method, not the application code.
4. Why RRF: BM25 and cosine scores have unrelated scales, so combine rank
   positions instead of summing raw values.
5. Limitations and next step: JSON persistence remains expensive, and the
   optional model must fit the target deployment memory budget.
