# ALTEGRAD-2024-2025---Generating-Graphs-with-Specified-Properties
Projet for my master's degree with Emilie Pic and Garance Gérard. We ranked 4th out of 58 participants at this challenge.

### Challenge Description: 
The goal of this project is to study and apply machine learning/artificial intelligence techniques
to generate graphs of specific properties. One of the most challenging tasks of machine learning
on graphs is that of graph generation. Graph generation has attracted a lot of attention recently,
and its main objective is to create novel and realistic graphs. For instance, in chemo-informatics,
graph generative models are employed to generate novel, realistic molecular graphs that also
exhibit desired properties (e.g., high drug-likeness). Recently, there has been a surge of interest
in developing new graph generative models, and most of the proposed models typically fall into
one of the following five families of models: (1) Auto-Regressive models; (2) Variational
Autoencoders; (3) Generative Adversarial Networks; (4) Normalizing Flows; and (5) Diffusion
models. These models can capture the complex
structural and semantic information of graphs. In this challenge, given a text query that
describes some properties of the structure of the graph, you need to generate the graph
corresponding to the description. The pipeline to deal with this task can be achieved by training
a latent diffusion for conditional graph generation

### Code details 
We built our code upon the baseline implementation provided by the instructors, following [Evdaimon et al., 2024](https://arxiv.org/pdf/2403.01535).

Each folder represents a different direction taken in the project. The VAE folder (based on the baseline) was the most conclusive and the direction leading to the est score. 

A file "Rapport_ALTEGRAD.pdf" details more precisely what we did during the challenge. 
