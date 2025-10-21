python ../prep_vctk.py `
  --file-list  speaker1-train-files.txt `
  --in-dir ../VCTK-Corpus/wav48/p225 `
  --out vctk-speaker1-train.4.16000.8192.4096.h5 `
  --scale 4 `
  --sr 16000 `
  --dimension 8192 `
  --stride 4096 `
  --interpolate `
  --low-pass

python ../prep_vctk.py `
  --file-list speaker1-val-files.txt `
  --in-dir ../VCTK-Corpus/wav48/p225 `
  --out vctk-speaker1-val.4.16000.8192.4096.h5.tmp `
  --scale 4 `
  --sr 16000 `
  --dimension 8192 `
  --stride 4096 `
  --interpolate `
  --low-pass

python ../prep_vctk.py `
  --file-list  speaker1-train-files.txt `
  --in-dir ../VCTK-Corpus/wav48/p225 `
  --out vctk-speaker1-train.4.16000.-1.4096.h5 `
  --scale 4 `
  --sr 16000 `
  --dimension -1 `
  --stride 4096 `
  --interpolate `
  --low-pass

python ../prep_vctk.py `
  --file-list speaker1-val-files.txt `
  --in-dir ../VCTK-Corpus/wav48/p225 `
  --out vctk-speaker1-val.4.16000.-1.4096.h5.tmp `
  --scale 4 `
  --sr 16000 `
  --dimension -1 `
  --stride 4096 `
  --interpolate `
  --low-pass