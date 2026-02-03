![IdeAtlas header](https://github.com/user-attachments/assets/496adbd5-2d4f-47f1-ace7-f397584b2fea)

# About the project
IDEAtlas, an ESA-funded project, aims to develop, implement, validate, and showcase advanced AI-based methods to automatically map and characterize the spatial extent of deprived urban areas (DUAs) (slums or informal settlements) using Earth Observation (EO) data. This initiative supports national and local governments, as well as civil society, in monitoring progress on SDG indicator 11.1.1, which focuses on the proportion of the urban population living in slums, informal settlements, or inadequate housing. 

Adopting a user-centered approach, IDEAtlas engages various stakeholders—including local communities, national authorities, and international organizations—in the co-design and co-development of AI-based solutions. The project builds upon previous and ongoing efforts, collaborating with initiatives such as IDEAMAPS and SLUMAP. Developed algorithms are integrated into a cloud-based, end-to-end processing system, demonstrated across eight pilot cities. 

IDEAtlas collaborates with early adopters in eight pilot cities across four continents:

- Mexico City, Mexico
- Medellín, Colombia
- Salvador, Brazil
- Buenos Aires, Argentina
- Lagos, Nigeria
- Nairobi, Kenya
- Mumbai, India
- Jakarta, Indonesia

We are currently expanding our reach and are already actively working in additional cities together with local partners, which includes:

- Tegucigalpa, Honduras
- Guatemala City, Guatemala
- Bissau, Guinea-Bissau
- Pereira, Colombia
- Rio de Janeiro, Brazil
- Kisumu, Kenya
- San Jose, Costa Rica
- Kigali, Rwanda
- Johannesburg, South Africa
- Juba, South Sudan

Further expansion is underway in additional urban centers, with active initiatives being launched in key cities across Latin America, Africa, and beyond to broaden our global impact.

<p align="center">
<img width="850" alt="image" src="https://github.com/user-attachments/assets/649ee0ba-3b18-4bb5-9c86-dff7d7c4838f" />
</p>


Case Study Cities

The IDEAtlas team comprises experts from leading scientific and industrial organizations:

- University of Twente Faculty ITC: Specializing in geospatial sciences and Earth Observation.
- GeoVille: An industrial leader in geoinformatics and EO-based solutions.
This partnership ensures a robust foundation for developing and implementing cost-effective EO solutions for monitoring informal settlements. 
- Advisory Board: includes distinguished members from European Space Agency, United Nations Statistics Division, UN-Habitat and IDEAMAPS.

# IDEABench Benchmark Dataset
Through the invaluable contributions from our local Co-anchors, we have built "IDEABench," a comprehensive dataset that serves as the foundation for training and testing AI-based models. IDEABench is not a static benchmark dataset but a continuously improving dataset built considering the feedback of local communities collected through the User Portal. The quality and geographical diversity of data collected within each city are essential for training accurate AI models and obtaining a sound quantification of model uncertainties.

For each city, the input for the model was constructed from four data sources, as illustrated in Fig. 2 below:

- Sentinel 1 GRD images closest to the acquisition time of the S2 image.
- Cloud-free Sentinel 2 L2A images.
- A pre-computed built-up density (PBD) derived from Google Open Buildings, used as ancillary information.
- A reference label with three classes.

![datasource](https://github.com/user-attachments/assets/65bb5856-e2c3-449e-bad2-3bac8aa5b085)

Input data to train the AI model

The benchmark dataset can be freely downloaded from Data Archiving and Networked Services (DANS) using the following link: https://doi.org/10.17026/PT/X4NJII

# Model
Working closely with local communities, local governments, and a range of (inter)national stakeholders, we co-designed an AI-driven strategy utilizing open Earth Observation (EO) and geospatial data to map DUAs in eight cities worldwide. We designed a tailored Multi-Branch Convolutional Neural Network (MB-CNN) architecture to fuse multi-modal data sources such as Sentinel 1 (S1), Sentinel 2 (S2), and pre-computed built-up density (PBD) for semantic segmentation of DUAs. The network consists of encoder and decoder blocks interconnected by skip connections to preserve fine-grained spatial information.


To enhance computational efficiency and reduce model complexity, we reduced the number of levels in both the encoder and decoder blocks and decreased the number of filters, resulting in a more lightweight and efficient model. We adopted an early fusion with feature adaptation approach, which has demonstrated stable performance compared to middle and late fusion techniques when combining multi-source inputs.

![MBCNN](https://github.com/user-attachments/assets/c494da59-6601-4567-941c-ca4601e85f77)


MB-CNN model architecture


## Using the code

Clone the repository:
```bash
git clone https://github.com/IDEAtlas/ai-dua-mapping.git
```
```bash
cd ai-dua-mapping
```

**Option 1: Using Conda**

For CPU:
```bash
conda env create -f env/environment-cpu.yaml
```

For GPU:
```bash
conda env create -f env/environment-gpu.yaml
```

Activate the environment:
```bash
conda activate ideatlas
```

**Option 2: Using Docker**

Build the image for CPU:
```bash
docker build -f env/Dockerfile-cpu.tf -t ideatlas .
```

Or for GPU:
```bash
docker build -f env/Dockerfile-gpu.tf -t ideatlas .
```

Run the container in detached mode:
```bash
docker run -dit --gpus all --name ideatlas -v $(pwd):/workspace ideatlas
```

Enter the container:
```bash
docker exec -it ideatlas bash
```

Configure settings by editing `config.yaml` with your desired model parameters if needed.

## Training from Scratch
Train a new model on complete dataset:
```bash
python main.py --task train --city nairobi --country kenya --year 2025
```

## Fine-tuning with Pre-trained Weights
Fine-tune a pre-trained model:
```bash
# Using default global weights
python main.py --task finetune --city nairobi --country kenya --year 2025

# Using custom weights
python main.py --task finetune --city nairobi --country kenya --year 2025 --weights checkpoint/custom.h5
```

## Classification/Inference
Generate segmentation maps from trained model:
```bash
# Using city-specific weights
python main.py --task classify --city nairobi --country kenya --year 2025

# Using custom weights
python main.py --task classify --city nairobi --country kenya --year 2025 --weights checkpoint/custom.h5
```

## SDG Statistics
Compute SDG 11.1.1 statistics from classified rasters:
```bash
python main.py --task sdg_stats --city nairobi --country kenya --year 2025
```

**Note**: Users can customize the acquisition period of Sentinel 2 data in the preprocessing scripts located in the `preprocessing/` directory (e.g., `stac_api.py`).

---

### Contact Information

- **Project Website**: https://ideatlas.eu/
- **User Portal**: https://portal.ideatlas.eu/
- **Email**: ideatlas@utwente.nl

---
