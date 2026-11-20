@echo off
REM Re-runs every experiment reported in the paper (12 training runs), then builds tables and figures.
REM Usage: double-click or run "run_all.bat" (about 1-2 hours on a normal laptop CPU)
cd /d %~dp0
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
for %%s in (13 42 2026) do (
  if "%%s"=="42" (
    python src\qwe.py --model fact --rank 4 --seed %%s --save
    python src\qwe.py --model dense --seed %%s --save
    python src\qwe.py --model real --seed %%s --save
  ) else (
    python src\qwe.py --model fact --rank 4 --seed %%s
    python src\qwe.py --model dense --seed %%s
    python src\qwe.py --model real --seed %%s
  )
)
python src\qwe.py --model fact --rank 2 --seed 42
python src\qwe.py --model fact --rank 8 --seed 42
python src\qwe.py --model fact_noimag --rank 4 --seed 42
python src\profile_res.py dense
python src\profile_res.py fact
python src\profile_res.py real
python src\analyze.py
echo All done. See results\ and figures\
pause
