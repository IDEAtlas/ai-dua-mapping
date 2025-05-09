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

<p align="center">
<img width="850" alt="image" src="https://github.com/user-attachments/assets/649ee0ba-3b18-4bb5-9c86-dff7d7c4838f" />
</p>


Fig. 1: Case Study Cities

The IDEAtlas team comprises experts from leading scientific and industrial organizations:

- University of Twente Faculty ITC: Specializing in geospatial sciences and Earth Observation.
- GeoVille: An industrial leader in geoinformatics and EO-based solutions.
This partnership ensures a robust foundation for developing and implementing cost-effective EO solutions for monitoring informal settlements. 
- Advisory Board: includes distinguished members from European Space Agency, United Nations Statistics Division, UN-Habitat and IDEAMAPS.

# Data
Through the invaluable contributions from our local Co-anchors, we have built "IDEABench," a comprehensive dataset that serves as the foundation for training and testing AI-based models. IDEABench is not a static benchmark dataset but a continuously improving dataset built considering the feedback of local communities collected through the User Portal. The quality and geographical diversity of data collected within each city are essential for training accurate AI models and obtaining a sound quantification of model uncertainties.

For each city, the input for the model was constructed from four data sources, as illustrated in Fig. 2 below:

- Sentinel 1 GRD images closest to the acquisition time of the S2 image.
- Cloud-free Sentinel 2 L2A images.
- A pre-computed built-up density (PBD) derived from Google Open Buildings, used as ancillary information.
- A reference label with three classes.

![datasource](https://github.com/user-attachments/assets/65bb5856-e2c3-449e-bad2-3bac8aa5b085)

Fig. 2: Input data to train the AI model

The benchmark dataset can be freely downloaded from Data Archiving and Networked Services (DANS) using the following link: https://doi.org/10.17026/PT/X4NJII

# AI Model
Working closely with local communities, local governments, and a range of (inter)national stakeholders, we co-designed an AI-driven strategy utilizing open Earth Observation (EO) and geospatial data to map DUAs in eight cities worldwide. We designed a tailored Multi-Branch Convolutional Neural Network (MB-CNN) architecture to fuse multi-modal data sources such as Sentinel 1 (S1), Sentinel 2 (S2), and pre-computed built-up density (PBD) for semantic segmentation of DUAs. The network consists of encoder and decoder blocks interconnected by skip connections to preserve fine-grained spatial information.

To enhance computational efficiency and reduce model complexity, we reduced the number of levels in both the encoder and decoder blocks and decreased the number of filters, resulting in a more lightweight and efficient model. We adopted an early fusion with feature adaptation approach, which has demonstrated stable performance compared to middle and late fusion techniques when combining multi-source inputs.

![MBCNN](https://github.com/user-attachments/assets/0f92f097-312f-4a70-a229-4b1e858514a7)

Fig. 3: Model architecture

# Results
Our experiments suggests that combining multi-spectral data with urban morphometric features, particularly Sentinel-2 imagery and built-up density, delivers the highest accuracy in identifying and mapping DUAs. The improved reference data (Reference V2) via the IDEAtlas platform showed an increase in accuracy. However, the significant variability in accuracy across the sample cities highlights the complexity of the problem and emphasizes the need for supplementary geospatial data to enhance limitation of EO data.


Table 1: Performance of the MB-CNN model across 8 cities (P: precision, R: recall, F1: F1-score)
<p align="center">
<img width="641" alt="table4" src="https://github.com/user-attachments/assets/e96d67af-68a2-4876-9003-a72584832a8a" />
</p>

Stay updated with our progress and contribute to the development of AI-based solutions for mapping deprived urban areas.

- Website: https://ideatlas.eu/
- User Portal: https://portal.ideatlas.eu/

