# ESA SNAP + Python Integration Guide

## Step 1: Java Installation and Environment Setup
ESA SNAP requires Java to run, so the first step is to ensure that Java is installed and the necessary environment variables are set up.

### 1.1 Install Java
You need to install OpenJDK to run ESA SNAP, which is Java-based.


sudo apt update
sudo apt install openjdk-11-jdk


**Verify Java installation:**

java -version

Expected output should resemble:

openjdk version "11.0.x"
OpenJDK Runtime Environment (build 11.0.x+...)
OpenJDK 64-Bit Server VM (build 11.0.x+...)


### 1.2 Set Java Environment Variables
Configure the environment variables to use the installed Java version.

**Find the location of Java:**

sudo update-alternatives --config java

Note the path of the selected version.

**Add Java to your environment variables by editing the `.bashrc` file:**

nano ~/.bashrc

Add these lines at the end of the file:

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

_Adjust the path to match your installed Java version._

**Apply the changes:**

source ~/.bashrc
echo $JAVA_HOME

You should see the correct Java path.

---

## Step 2: Installing and Coupling ESA SNAP with Python

### 2.1 Install ESA SNAP

**Install wget if not already installed:**

sudo apt install wget


**Download the latest ESA SNAP installer for Linux:**

cd /tmp
wget https://download.esa.int/step/snap/10_0/installers/esa-snap_all_linux-10.0.0.sh


**Make the installer executable and run it:**

chmod +x esa-snap_all_linux-10.0.0.sh
bash esa-snap_all_linux-10.0.0.sh

Follow the instructions to complete installation. Choose to install all toolboxes and **say NO** when asked to install Python.

### 2.2 Creating a Conda Virtual Environment

**Install Anaconda (if not already installed):**

wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
chmod +x Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh


**Option 1: Use the provided YAML file**

Using the provided YAML file `snappy_env.yml`, run:

conda env create -f snappy_env.yml


**Option 2: Create the environment manually:**

conda create -n snappy -c conda-forge python=3.10 geopandas rioxarray dask xarray tqdm requests -y
conda activate snappy
pip install python-snappy


### 2.3 Configure SNAP Python Interface (Snappy)

**Navigate to the SNAP bin directory:**

cd /home/eouser/esa-snap/bin


**Generate Snappy bindings:**

./snappy-conf /home/eouser/anaconda3/envs/snappy/bin/python /home/eouser/anaconda3/envs/snappy/lib/

Successful message:

Configuration finished successful!
The SNAP-Python interface is located in '/home/eouser/anaconda3/envs/snappy/lib/esa_snappy'


### 2.4 Using Snappy in Python Code

#### Option 1: Copy `esa_snappy` to site-packages

**Find site-packages directory:**

python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"

Example output:

/home/user/anaconda3/envs/snappy/lib/python3.10/site-packages


**Copy `esa_snappy`:**

cp -r /home/user/anaconda3/envs/snappy/lib/esa_snappy /home/user/anaconda3/envs/snappy/lib/python3.10/site-packages/

_Adjust the path if your Python version differs._

**Now you can import it easily:**
python
from snappy import ProductIO


#### Option 2: Append path in code (only if option 1 doenst work)

**In your Python script or interactive session:**

conda activate snappy
python
import sys
sys.path.append('/home/eouser/anaconda3/envs/snappy/lib')
from esa_snappy import ProductIO


This lets you call SNAP functions directly from your Python scripts.

