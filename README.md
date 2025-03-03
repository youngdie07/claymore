# ClaymoreUW
# Claymore Multi-GPU Material Point Method for Engineering Projects

Authors:
[Justin Bonus](https://github.com/JustinBonus)\, 
[Pedro Arduino](https://github.com/parduino)\

<div align="left">
    <a href="https://claymore.readthedocs.io/en/latest/"> Documentation </a>
</div>

## Data-Repositories

Comparison data of ClaymoreUW MPM to DualSPHysics SPH and STAR-CCM+ CFD against stochastic experiments by Goserberg et al. 2016. For comparison details, see Bonus et al. 2025, "Tsunami Debris Motion and Loads in a Scaled Port Setting: Comparative Analysis of Three State-of-the-Art Numerical Methods",

https://www.designsafe-ci.org/data/browser/projects/PRJ-5846/workdir

## Description

Opensource code for Multi-GPU MPM. It is a modified fork from opensource code [page](https://github.com/penn-graphics-research/claymore) for the SIGGRAPH 2020 paper:

**A Massively Parallel and Scalable Multi-GPU Material Point Method** 

[page](https://sites.google.com/view/siggraph2020-multigpu)\, [pdf](https://www.seas.upenn.edu/~cffjiang/research/wang2020multigpu/wang2020multigpu.pdf)\, [supp](https://www.seas.upenn.edu/~cffjiang/research/wang2020multigpu/supp.pdf)\, [video](https://vimeo.com/414136257)


<!--
<p float="left">
<img src="Data/Clips/faceless.gif" height="128px"/>
<img src="Data/Clips/flow.gif" height="128px"/>
<img src="Data/Clips/chains.gif" height="128px"/>
<img src="Data/Clips/cat.gif" height="128px"/>
</p>
-->

## Compilation
This is a cross-platform C++/CUDA cmake project. The minimum version requirement of cmake is 3.16, yet the latest version is generally recommended. The required CUDA version is 10.2 or 11.

Currently, *supported OS* includes Windows 10 and Ubuntu (>=18.04), and *tested compilers* includes gcc8.4, msvc v142, clang-9 (includes msvc version). 

### Build
Run the following command in the *root directory*. Note that adding "--config Release" to the last command is needed when compiling using msvc.
```mkdir build
cd build
cmake ..
cmake --build .
```

Or configure the project using the *CMake Tools* extension in *Visual Studio Code* (recommended).

### Formulation

> Time-Integration (Explicit)
> Material models (Fixed-corotated, NA Cam-Clay, Drucker-Prager, Weakly Comp. Fluid)
> Shape Function (2nd Order B-Spline)
> Transfer Scheme (Affine Particle-In-Cell)
> Kernel (Grid-to-Particle-to-Grid)

### Input

JSON input files are used to configure some simulation settings.

Currently, binary position data and the level-set (signed distance field) [data](https://github.com/littlemine/Data) are accepted as input files for particles. Uniformly sampling particles from analytic geometries is another viable way for the initialization of models.

Starting from an \*.obj or \*.stl file, SDFGen [page](https://github.com/wdas/SDFGen) can make appropiate \*.sdf files. 

### Output

Particle output includes 
> Position (x, y, z)
> Stress ($\sigma_{1}$, $\sigma_{2}$, $\sigma_{3}$)
> Deformation Gradient ($J = ||F||$)

Particle data is segmented by object and device.

Grid output includes
> Index (x, y, z)
> Mass (m)
> Momentum ($M_{x}$, $M_{y}$, $M_{z}$)

Grid data is dynamic (inactive cells not included) and down-sampled (one block output per 4x4x4 cells)

Outputs are binary geometry files (\*.bgeo). Houdini Apprentice can render these efficiently.


### Run Demos
The project provides the following GPU-based schemes for MPM:
- **GMPM**: Improved single-GPU pipeline
- **MGSP**: Static geometry (particle) partitioning multi-GPU pipeline
- **WASIRF**: Single-GPU simulation of flume experiments. 
<!--
- dynamic spatial partitioning multi-GPU pipeline
-->

Go to *Projects/\*\**, run the executable.

## Code Usage
> Use the codebase in another cmake c++ project.

Directly include the codebase as a submodule, and follow the examples in the *Projects*.

> Develop upon the codebase.

Create a sub-folder in *Projects* with a cmake file at its root.

## Bibtex

Please cite the original paper if you use this code for your research: 
```
@article{Wang2020multiGMPM,
    author = {Xinlei Wang* and Yuxing Qiu* and Stuart R. Slattery and Yu Fang and Minchen Li and Song-Chun Zhu and Yixin Zhu and Min Tang and Dinesh Manocha and Chenfanfu Jiang},
    title = {A Massively Parallel and Scalable Multi-GPU Material Point Method},
    journal = {ACM Transactions on Graphics},
    year = {2020},
    volume = {39},
    number = {4},
    articleno = {Article 30}
}
```

## Credits
This project draws inspirations from [Taichi](https://github.com/taichi-dev/taichi), [GMPM](https://github.com/kuiwuchn/GPUMPM).



### Dependencies
The following libraries are adopted in our project development:

- [cub](http://nvlabs.github.io/cub/) (now replaced by Thrust)
- [fmt](https://fmt.dev/latest/index.html)

For particle data IO and generation, we use these two libraries in addition:

- [partio](http://partio.us/)
- [SDFGen](https://github.com/christopherbatty/SDFGen)

Due to the C++ standard requirement (at most C++14) for compiling CUDA (10.2) code, we import these following libraries as well:

- [function_ref](https://github.com/TartanLlama/function_ref)
- [optional](https://github.com/TartanLlama/optional)
- [variant](https://github.com/mpark/variant)
