#!/bin/bash

# Sets up the build for ClaymoreUW Multi-GPU MPM
# For more info on use of ClaymoreUW see Bonus 2023 dissertation or the HydroUQ / ClaymoreUW documentation
#
# Relies on CMake 3.15+, CUDA 10.2+, and GCC 7.5+
# Requires an NVIDIA GPU currently, recommended to have 
# atleast a compute-capability (CCXY) 6.0+ (sm_60), with 7.5+ being the most tested
# Typically fulfilled by any model made after ~2018, consumer or HPC
# 
# Currently tested on:
# TACC Lonestar6 - Triple GPU NVIDIA A100 40GB PCIe sm_80 (Ampere Architecture) - GCC 9.4
# TACC Frontera - Quad. GPU NVIDIA RTX Quadro 5000 16GB sm_75 () - GCC 9.4
# Local Desktop - Single GPU NVIDIA RTX 4060 ti 16GB sm_89 (Ada Architecture) - GCC 8.9 - Intel i5 
# Local Desktop (May be deprecated) - Single GPU NVIDIA GTX 780 ti 8 GB sm_60 - GCC 7.5 - Intel i3 
# ACCESS TAMU ACES - Dual GPU NVIDIA H100 80GB PCIe sm_90 (Hopper Architecture) - ICC ... (note: some issues with intel specific compiler on ACES in the past, contact bonus@berkeley.edu for assistance) - Intel Sky/Meteor-Lake?
# 
# Some newer CPUs (e.g. most limited release intel CPU lines as seen in exp. HPC systems such as TAMU ACES)
# may have issues with threading depending on the OS. Recommended to use just 1 CPU thread in these cases, 
# or just 1 thread per socket on the node. Limits speed of IO to some extent but typically trivial. 
# Hyperthreading not extensively tested.
# 
# Justin Bonus, Oct. 2024

echo "Make build directory"
rm -rf build
mkdir -p build
cd build

echo "Configure build directory with CMake (3.15+)"
cmake ..

echo "Build ClaymoreUW and External Libraries"
cmake --build .

echo "Set permissions of executables (allows others to run your compiled programs on TACC systems if made available)"
#sudo chmod a+rwx ./Projects/FBAR/fbar
sudo chmod a+rwx ./Projects/OSU_LWF/osu_lwf
#sudo chmod a+rwx ./Projects/DSA/dsa

echo "Copy executables from build subdirectories to upper-level subdirectories..."
#cp ./Projects/FBAR/fbar ../Projects/FBAR/ 
cp ./Projects/OSU_LWF/osu_lwf ../Projects/OSU_LWF/ 
#cp ./Projects/DSA/dsa ../Projects/DSA/

echo "Copy experimental wave-maker paddle-motion resources to Project subfolders..."
#cp ../Data/WaveMaker/wmdisp_hydro2sec_1200hz_smooth_14032023.csv ../Projects/FBAR/
#cp ../Data/WaveMaker/wmdisp_LWF_Unbroken_Amp4_SF500_twm10sec_1200hz_14032023.csv ../Projects/FBAR/
#cp ../Data/WaveMaker/wmdisp_TWB_Amp2_SF350_twm10sec_1200hz_14032023.csv ../Projects/FBAR/
#cp ../Data/WaveMaker/wmdisp_TWB_Amp2_SF375_twm10sec_1200hz_16052023.csv ../Projects/FBAR/
cp ../Data/WaveMaker/wmdisp_hydro2sec_1200hz_smooth_14032023.csv ../Projects/OSU_LWF/
cp ../Data/WaveMaker/wmdisp_LWF_Unbroken_Amp4_SF500_twm10sec_1200hz_14032023.csv ../Projects/OSU_LWF/
cp ../Data/WaveMaker/wmdisp_TWB_Amp2_SF350_twm10sec_1200hz_14032023.csv ../Projects/OSU_LWF/
cp ../Data/WaveMaker/wmdisp_TWB_Amp2_SF375_twm10sec_1200hz_16052023.csv ../Projects/OSU_LWF/

