# MSD-Datasets

## News

- The version **v0.1.0** has been released. There might be many imperfections in the project, but I will keep improving it.

## TODO

- [x] Add Introduction
- [ ] Add Dataset use method
    - [x] Add Python single-thread version
    - [x] Add Python multi-thread version
    - [ ] Add C++ single-thread version
    - [ ] Add C++ multi-thread version
- [ ] Add the generated Datastes
- [x] Add Citation
- [ ] (coming soon) Add MSM CODE
- [ ] Optimize the size of the dataset (Now 146~GB, There is too much repetitive data)

## Introduction

Due to the lack of scale-related analysis in the existing datasets in the field of feature matching, this has to some extent affected the research or evaluation of feature matching algorithms for scale differences. For this purpose, we have proposed the **Scale Difference Index (SDI)** to describe scale differences, which can be derived respectively from the **Angle Difference Index (ADI)** and the **pixel Difference Index (PDI)**. Then, we constructed a **Multi-Scale Dataset (MSD)** based on SDI.

The MSD comprises two components.

1. The first is a scale-difference dataset based on the existing Megadepth (Li and Snavely Citation2018) dataset. It offers the advantage of a large amount of data and images that are very close to nature.
2. The second was a self-constructed scale-difference dataset based on remote sensing imagery from UAVs. It offers the advantage of very clear images, which are suitable for research in remote sensing from UAVs and other areas.

The main function of this repository is to provide the MSD-Dataset based on MegaDepth.

For more details, please check [**Paper section 5**](https://www.tandfonline.com/doi/full/10.1080/17538947.2025.2543562#d1e4493).

![img](./Docs/assets/figure_5.jpeg)

Examples of Multi-Scale Dataset. The content of the subheading is the image resolution. (a, b) low ADI, low PDI; (c, d) low ADI, high PDI; (e, f) high ADI, low PDI; (h, g) high ADI, high PDI.

## Get the Dataset

### Download Megadepth

There are many versions of this dataset. The version adopted in this study is the one used in [**DISK**](https://github.com/cvlab-epfl/disk).

For the download, you can refer to the official guidelines of [**DISK**](https://github.com/cvlab-epfl/disk), and please pay close attention to the file [**download_dataset**](https://github.com/cvlab-epfl/disk/blob/master/download_dataset)。

### Method 1: Run The Code

To ensure computational stability during data integration (which may involve time-consuming processes), the algorithm automatically creates a `.cache` directory in the current working directory. This stores intermediate results during execution and aggregates the final output files only after all computations complete.

⚠️ **Important**: The `.cache` directory is not automatically cleaned up after execution. Users must manually delete this directory to avoid unnecessary disk space usage.

1. **Clone this project**

```bash
git clone git@github.com:KevenGe/MSD-Datasets.git
```

2. **Configure the Python environment properly.** 

This project uses [uv](https://github.com/astral-sh/uv) to manage the package environment. Therefore, after you configure the [uv](https://github.com/astral-sh/uv) environment, you only need 
```bash
uv sync
```
to reproduce the environment of this project and activate it through 
```bash
.venv\Scripts\activate
```
3. **Execute the code**

Then just run

```bash
python ./main.py < megadepth_path >
```

Then you will see the MSD-Dataset json file.

### Method 2: The Generated Dataset Based On Megadepth

If it's not convenient for you to generate the dataset yourself through code, we also provide pre-calculated datasets, which can help you use them more quickly.

- Baidu Yun: https://pan.baidu.com/s/1YDU92yX69yWI3YGV8tDxcA?pwd=8ki4
- Google Drive: (Coming Soon)
- GitHub Release: (Coming Soon)

### Analysis of the dataset file format

For `.cache` directory, the file format is as follows:

```text
├─.cache                                                    # cache directory
│  ├─0001                                                   # scene ID
│  │  ├─5008984_74a994ce1c_62638625_7283cc3777.json         # cache file           
│  │  ├─5008984_74a994ce1c_534552203_52c5f133f0.json        # cache file     
│  │  ├─5008984_74a994ce1c_883453033_cc28e4b2a9             # cache file     
│  │  ├─......
│  ├─0004
│  ├─0005
│  ├─0007
│  ├─.....
```

For cache file, the file format is as follows:

```json
{
  "data": {
    "img1_path": "scenes/0001/images/5008984_74a994ce1c_o.jpg",
    "img1_depth_path": "scenes/0001/depth_maps/5008984_74a994ce1c_o.h5",
    "img1_calib_path": "scenes/0001/calibration/calibration_5008984_74a994ce1c_o.jpg.h5",
    "img2_path": "scenes/0001/images/62638625_7283cc3777_o.jpg",
    "img2_depth_path": "scenes/0001/depth_maps/62638625_7283cc3777_o.h5",
    "img2_calib_path": "scenes/0001/calibration/calibration_62638625_7283cc3777_o.jpg.h5",
    "ADI": 0.594368945806114,
    "PDI": 0.24401540959600815,
    "SDI": 0.8383843554021222
  }
}
```

For the result

```json
{
  "ADI": {
    "Low": [],
    "Medium": [],
    "High": [],
    "Ultra": []
  },
  "PDI": {
    "Low": [],
    "Medium": [],
    "High": [],
    "Ultra": []
  },
  "SDI": {
    "Low": [],
    "Medium": [],
    "High": [],
    "Ultra": []
  }
}
```

## FAQ

## Citation

Should you find this dataset beneficial to your work, we kindly request that you cite it using the provided BibTeX
entry.

```bibtex
@article{Ge25082025,
    author = {Qifeng Ge and Xiaoping Du and Chen Xu and Jianhao Xu and Zhenzhen Yan and Xiangtao Fan},
    title = {MSM: a scaling-based feature matching algorithm for images with large-scale differences},
    journal = {International Journal of Digital Earth},
    volume = {18},
    number = {1},
    pages = {2543562},
    year = {2025},
    publisher = {Taylor \& Francis},
    doi = {10.1080/17538947.2025.2543562},
    URL = {https://doi.org/10.1080/17538947.2025.2543562},
    eprint = {https://doi.org/10.1080/17538947.2025.2543562}
}
```

## Acknowledgements
