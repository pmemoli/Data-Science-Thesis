helm-run --conf-paths src/eval-engine/src/helm/benchmark/presentation/run_entries_custom.conf \
 --models-to-run microsoft/phi-3.5-mini-instruct \
 --suite test-script \
 --output-path src/data/helm/ \
 --max-eval-instances 150

python -m src.scripts.compute_benchmarks
