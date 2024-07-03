# Enhancing Customer Support with open-source LLMs: Development of a Question Answering Application
This repository contains the source code for a Question Answering (QA) application designed to enhance customer support capabilities in Small and Medium-sized Enterprises (SMEs) using open-source Large Language Models (LLMs).

## Abstract
Today's customers expect 24/7 personalised service across multiple channels, putting immense pressure on organisations to deliver exceptional experiences while maintaining efficiency. Generative Artificial Intelligence (AI) is emerging as a potential solution, promising to automate tasks and personalise interactions. However, this technology remains largely out of reach for Small and Medium-sized Enterprises (SMEs) due to financial and technical constraints. This research investigates the potential of Large Language Models (LLMs) to enhance customer support capabilities in SMEs, with a particular focus on technical question answering. The research develops a Question Answering (QA) application that uses Retrieval-Augmented Generation (RAG) to optimise information retrieval from technical documents. The application developed exclusively using open-source tools and LLMs, aims to provide a cost-efficient solution for SMEs to lower the barrier of AI adoption. Guided by a Design Science Research Methodology (DSRM), the research details the development of the application and evaluates its performance and usability. The results show that while the application successfully improves factual accuracy and explainability, it suffers from high latency issues, with response times unsuitable for real-world use. The method used to incorporate human feedback when the LLM cannot answer a question demonstrated potential as a continuous learning mechanism for the application. However, this judging mechanism still needs further development to achieve consistent performance. In summary, while the open-source LLM-based QA application shows potential for improving SME customer support, significant improvements in computational resources and alternative judging approaches are required to fully realise its capabilities. Future work should focus on increasing processing speeds and exploring more powerful LLMs to reduce latency and improve answer relevance and correctness. 

## Getting Started

### Hardware Requirements:

While not strictly necessary for running the core application, NVIDIA CUDA Toolkit may be beneficial for improved performance on certain tasks. If you plan to utilize NVIDIA CUDA for potential performance enhancements, refer to the official NVIDIA CUDA Toolkit documentation for installation instructions: https://docs.nvidia.com/cuda/

### Software Requirements:

1. Clone this repository.
2. Install dependencies: pip install -r requirements.txt

## Code Structure

The src directory contains the application source code, organized into subfolders for specific functionalities:

- application: Handles core logic of the QA application.
- evaluation: Contains scripts for evaluating the application's performance and usability.
- notebooks: Contains Jupyter Notebooks for data processing, evaluation, and running the application.
- utils: Contains utility functions used throughout the project.

## Additional Information

License: Refer to the LICENSE file for licensing information.