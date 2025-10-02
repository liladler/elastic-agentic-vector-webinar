#!/usr/bin/env python3
"""
Simple MCP Server for Elasticsearch Search
Exposes Elasticsearch search capabilities through Model Context Protocol
"""

import os
import json
import asyncio
from typing import Any
import torch
import numpy as np
from elasticsearch import Elasticsearch
from colpali_engine.models import ColPali, ColPaliProcessor
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize ColPali model
print("Loading ColPali model for MCP server...")
model_name = "vidore/colpali-v1.2"
colpali_model = ColPali.from_pretrained(model_name)
colpali_processor = ColPaliProcessor.from_pretrained(model_name)
print("✓ ColPali ready!")

# Initialize Elasticsearch client
ELASTIC_URL = os.getenv("ELASTIC_URL")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")
INDEX_NAME = "technical_docs_webinar"

es_client = Elasticsearch(
    ELASTIC_URL,
    api_key=ELASTIC_API_KEY,
    verify_certs=True
)

# Create MCP server
app = Server("elastic-search-server")

def create_col_pali_query_vectors(query: str):
    """Generate ColPali query vectors (both multi-vectors and avg_vector)"""
    try:
        batch_queries = colpali_processor.process_queries([query]).to(colpali_model.device)
        
        with torch.no_grad():
            query_embeddings = colpali_model(**batch_queries)
            multi_vectors = query_embeddings[0].cpu().numpy()
            
            # Calculate normalized average vector for kNN retrieval
            avg_vector = multi_vectors.mean(axis=0)
            avg_vector_norm = np.linalg.norm(avg_vector)
            if avg_vector_norm > 0:
                avg_vector = avg_vector / avg_vector_norm
        
        return {
            "multi_vectors": multi_vectors.tolist(),
            "avg_vector": avg_vector.tolist()
        }
    except Exception as e:
        print(f"Error generating query vectors: {e}")
        return None

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available Elasticsearch tools"""
    return [
        Tool(
            name="elasticsearch_search",
            description="Search the Elasticsearch knowledge base using hybrid search (text + vector) with ColPali rescoring. Returns relevant document chunks with metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information"
                    },
                    "size": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="elasticsearch_get_document",
            description="Retrieve a specific document by slide number from the knowledge base",
            inputSchema={
                "type": "object",
                "properties": {
                    "slide_number": {
                        "type": "integer",
                        "description": "The slide/page number to retrieve"
                    }
                },
                "required": ["slide_number"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    
    if name == "elasticsearch_search":
        query = arguments.get("query")
        size = arguments.get("size", 3)
        
        try:
            # Generate ColPali query vectors
            query_vectors = create_col_pali_query_vectors(query)
            
            # Use RRF to combine text + avg_vector, then rescore with ColPali
            search_body = {
                    "_source": ["title", "slide_text", "image_path", "metadata"],
                    "retriever": {
                        "rescorer": {
                            "retriever": {
                                "rrf": {
                                    "retrievers": [
                                        {
                                            "standard": {
                                                "query": {
                                                    "multi_match": {
                                                        "query": query,
                                                        "fields": ["title", "slide_text"],
                                                        "type": "best_fields"
                                                    }
                                                }
                                            }
                                        },
                                        {
                                            "standard": {
                                                "query": {
                                                    "semantic": {
                                                        "query": query,
                                                        "field": "content"
                                                    }
                                                }
                                            }
                                        },
                                        {
                                            "knn": {
                                                "field": "avg_vector",
                                                "query_vector": query_vectors["avg_vector"],
                                                "k": 10,
                                                "num_candidates": 100
                                            }
                                        }
                                    ],
                                    "rank_window_size": 50
                                }
                            },
                            "rescore": {
                                "window_size": 10,
                                "query": {
                                    "rescore_query": {
                                        "script_score": {
                                            "query": {"match_all": {}},
                                            "script": {
                                                "source": "maxSimDotProduct(params.query_vector, 'col_pali_vectors')",
                                                "params": {"query_vector": query_vectors["multi_vectors"]}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "size": size
            }
            
            response = es_client.search(index=INDEX_NAME, body=search_body)
            
            # Format results
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                results.append({
                    "title": source.get('title', ''),
                    "slide_text": source.get('slide_text', '')[:500],  # Truncate for MCP
                    "slide_number": source.get('metadata', {}).get('page_number', ''),
                    "image_path": source.get('image_path', ''),
                    "score": hit['_score']
                })
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "query": query,
                    "num_results": len(results),
                    "results": results
                }, indent=2)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    elif name == "elasticsearch_get_document":
        slide_number = arguments.get("slide_number")
        
        try:
            search_body = {
                "query": {
                    "term": {
                        "metadata.page_number": slide_number
                    }
                },
                "size": 1,
                "_source": ["title", "slide_text", "image_path", "metadata"]
            }
            
            response = es_client.search(index=INDEX_NAME, body=search_body)
            
            if response['hits']['hits']:
                hit = response['hits']['hits'][0]
                source = hit['_source']
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "slide_number": slide_number,
                        "title": source.get('title', ''),
                        "slide_text": source.get('slide_text', ''),
                        "image_path": source.get('image_path', ''),
                        "metadata": source.get('metadata', {})
                    }, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Slide {slide_number} not found"}, indent=2)
                )]
                
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    else:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"}, indent=2)
        )]

async def main():
    """Run the MCP server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main()) 