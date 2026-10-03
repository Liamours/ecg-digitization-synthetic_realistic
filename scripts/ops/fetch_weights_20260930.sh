#!/bin/sh
cd /c/researches/research-ecg-digitization/external
for i in 1 2 3 4 5 6; do
  if [ ! -d Open-ECG-Digitizer/.git ]; then rm -rf Open-ECG-Digitizer; git clone --depth 1 https://github.com/Ahus-AIM/Open-ECG-Digitizer Open-ECG-Digitizer && echo CLONE_OK; fi
  if [ -d Open-ECG-Digitizer/.git ]; then (cd Open-ECG-Digitizer && git lfs pull && echo OPEN_LFS_OK) && break; fi
  echo "retry $i"; sleep 20
done
for i in 1 2 3 4 5 6; do
  (cd ECG-Digitiser && git lfs pull --include="models/M3/**" && echo KRONES_LFS_OK) && break
  echo "krones retry $i"; sleep 20
done
echo FETCH_ALL_DONE
