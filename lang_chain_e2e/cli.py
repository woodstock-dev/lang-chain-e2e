# Copyright 2024 Google, LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging

from lang_chain_e2e.books import main as Books
from lang_chain_e2e.agent import main as Agent

FORMAT = "[%(asctime)s] (%(filename)s . %(funcName)s :: %(lineno)d) -- %(message)s"

def main():
    logging.basicConfig(filename="lc-e2e.log", level=logging.INFO, format=FORMAT)
    
    parser = argparse.ArgumentParser(
        prog="cli",
        description="A simple CLI for demonstrating gemini using langchain",
        epilog="\nGoogle Cloud Platform",
    )
    
    subparsers = parser.add_subparsers(dest="action", help="Command Help")
    
    agent = subparsers.add_parser("agent", help="Runs a simple Q&A bot with a sarcastic twist.")
    books = subparsers.add_parser("books", help="Runs a RAG model against the book data.")
    
    args = parser.parse_args()
    
    match args.action:
        case "agent":
            Agent()
        case "books":
            Books()
        