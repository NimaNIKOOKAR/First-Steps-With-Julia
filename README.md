# First Steps with Julia – Street‑View Character Recognition

A **hands‑on starter project** that walks you through building image‑classification models in Julia using the **Chars74K Street‑View** dataset.  
You’ll learn:

* Julia basics & package ecosystem  
* Image loading and preprocessing  
* Training a **Random Forest** baseline in 20 lines  
* Rolling your own **k‑Nearest Neighbour** with fast Leave‑One‑Out CV  
* Effortless parallelisation with one macro  
* How to craft a Kaggle submission (Classification Accuracy)

---

## Table of Contents
1. [Project structure](#project-structure)  
2. [Prerequisites](#prerequisites)  
3. [Dataset overview](#dataset-overview)  
4. [Random‑Forest baseline](#random‑forest-baseline)  
5. [K‑NN from scratch + LOO‑CV](#k‑nn-from-scratch--loo‑cv)  
6. [Parallelisation in one line](#parallelisation-in-one-line)  
7. [Hyper‑parameter tuning](#hyper‑parameter-tuning)  
8. [Submission format](#submission-format)  
9. [Citation & licence](#citation--licence)

---

## Project structure
```text
.
├── data/
│   ├── trainResized/     # 20×20 .Bmp images
│   ├── testResized/
│   ├── trainLabels.csv
│   └── sampleSubmission.csv
├── src/
│   ├── read_data.jl
│   ├── rf_baseline.jl
│   ├── knn_loocv.jl
│   └── utils.jl
└── README.md
```
All paths are centralised in `src/utils.jl`; change one string, everything works.

---

## Prerequisites
* Julia ≥ 1.10  
* Packages (install once):
  ```julia
  using Pkg
  Pkg.add.(["Images", "DataFrames", "CSV", "DecisionTree"])
  ```

Optional for parallel speed‑ups:
```julia
import Distributed; Distributed.addprocs(Threads.nthreads()-1)
```

---

## Dataset overview
* **Chars74K Street‑View subset** – 62 classes (`A–Z`, `a–z`, `0–9`)  
* Images already resized to **20×20 (400 pixels)** for convenience.  
* Train: 7 192 images, Test: 3 060 images.  
* Evaluation metric: **Classification Accuracy** on unseen test labels.

---

## Random‑Forest baseline (50 trees)
```julia
include("src/read_data.jl")      # xTrain, xTest, yTrain in Float32
using DecisionTree

model = build_forest(yTrain, xTrain, round(Int, √size(xTrain,2)), 50, 1.0)
y_pred = apply_forest(model, xTest)

# write submission
using CSV, DataFrames
sub = DataFrame(ImageId = 1:size(xTest,1), Class = Char.(y_pred))
CSV.write("rf_submission.csv", sub)
```
Produces **≈ 0.57 accuracy** in < 30 s on a laptop.

---

## K‑NN from scratch + LOO‑CV
`src/knn_loocv.jl` implements:

* **Euclidean distance** with an explicit loop (faster & memory‑friendly).  
* `get_k_nearest_neighbors` – returns sorted indices in O(N) per point.  
* `assign_label_each_k` – one pass gives predictions for *all* k ∈ 1…`maxK`.  
* Leave‑One‑Out CV fully parallelised:

```julia
@everywhere include("src/knn_loocv.jl")
maxK = 20
acc = loocv_knn(xTrain', yTrain, maxK)   # returns Vector{Float32}
println("Best k = $(argmax(acc)), accuracy = $(maximum(acc))")
```

Typical result: **k = 3** with **≈ 0.66 accuracy**; runtime ≈ 40 s on 4 cores.

---

## Parallelisation in one line
```julia
using Distributed; addprocs(4)   # choose your core count
preds = @parallel (vcat) for i in 1:n_samples
    assign_label(xTrain, yTrain, 3, view(xTest', :, i))
end
```
Julia ships with green‑threading and shared‑memory workers—no boilerplate.

---

## Hyper‑parameter tuning
Because `assign_label_each_k` returns predictions for **all k in one call**, tuning costs **O(N)** instead of *k×O(N)*:

```julia
maxK = 20
acc = loocv_knn(xTrain', yTrain, maxK)
using Plots; plot(1:maxK, acc; xlabel="k", ylabel="LOO Accuracy")
```
Pick the elbow. Submit with your chosen k:

```julia
y_pred = knn_predict(xTrain', yTrain, xTest', k=3)
CSV.write("knn_submission.csv",
          DataFrame(ImageId=1:length(y_pred), Class=Char.(y_pred)))
```

---

## Submission format
```
ImageId,Class
6284,A
6285,b
6286,0
...
```
Upload `*.csv` to Kaggle → leaderboard score appears instantly.

---

## Citation & licence
```
@inproceedings{Campos2009Chars74k,
  title={Character recognition in natural images},
  author={de Campos, T. and Babu, B.R. and Varma, M.},
  booktitle={Proc. International Conference on Computer Vision Theory and Applications},
  year={2009}
}
```
Tutorial code © 2025 MIT‑licensed. Dataset © UCL / Google Street View (see original terms).

---

*Enjoy hacking—Julia’s speed means you can iterate like C while scripting like Python!*
