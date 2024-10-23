<!--
 Copyright 2024 Google, LLC
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     https://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->
# Langchain End-to-End

## Prerequisites

Install pipx, and poetry.

Have a Google Token for running the queries (https://aistudio.google.com/app/apikey)

Login to your Google CLI project

### Google APIs required

* Vertex AI API
* Generative Language API

## The Examples

There are two examples in this kit so far:

* A sarcastic yet useful agent for simple Q&A (no session)
* A Retrieval Augmented Generation (RAG) model using Gemini and Vertex embeddings.

## Run the example

From your terminal:

```shell
cd <project-directory>

# Change to a virtual environment
poetry shell

# Install the dependencies in the virtual environment
poetry install

# Run the simple agent program
example agent 

# Run the RAG agent
example books

# Note, the books example will use quota from the embeddings client and each query is executed against your Gemini Flash Quota
```
