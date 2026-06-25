To check whether your RGB images have GPS:

.\.venv\Scripts\python.exe benchmarks\feature_stage.py --rgb-dir "C:\path\to\rgb_images" --check-gps-only
It will print each image as GPS or NO GPS.

To run feature extraction only:

.\.venv\Scripts\python.exe benchmarks\feature_stage.py --rgb-dir "C:\path\to\rgb_images" --output-dir output\feature_benchmark --skip-matching

To run feature extraction plus GPS-filtered matching:

.\.venv\Scripts\python.exe benchmarks\feature_stage.py --rgb-dir "C:\path\to\rgb_images" --output-dir output\feature_benchmark --n-neighbors 8

Because your current environment is Python 3.14 on Windows, this uses the COLMAP CLI path by default, so colmap.exe must be installed and available on PATH.

After the run, outputs are here:

output/feature_benchmark/database.dbfeat
output/feature_benchmark/rgb_images/
output/feature_benchmark/match_pairs.