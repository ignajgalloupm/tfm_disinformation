import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from embedding_generation import EmbeddingGenerator
from qdrant_client.http import models
import time
import os


class VectorDatabase():

    def __init__(self, client='local', ip='localhost', port=6333, encoder='mpnet', version='v1', wiki=None):
        self.collection = f'{encoder}_{version}'
        if client == 'local':
            self.client = QdrantClient(':memory:')
            self.start_collection(wiki, encoder, version)

        elif client == 'docker':
            self.client = QdrantClient(f'{ip}:{port}')
            if self.collection in [c.name for c in self.client.get_collections().collections]:
                ## delete collection
                self.client.delete_collection(collection_name=self.collection)
            self.start_collection(wiki, encoder, version)
        else:
            raise Exception('Invalid client type')
        

        
    def search(self, queries, top=10):
        collection_names = [collection.name for collection in self.client.get_collections().collections]

        if self.collection not in collection_names:
            raise Exception('Collection does not exist')
        
        search_queries = [models.SearchRequest(vector=queries[i], limit=top, with_payload=True) for i in range(len(queries))]
        return self.client.search_batch(collection_name=self.collection, requests=search_queries)
        

    def start_collection(self, wiki, encoder, version):
        if wiki is None:
            raise Exception('wiki is required to create collection')
        init = time.time()

        print('Generating embeddings')
        emb_gen = EmbeddingGenerator(encoder=encoder, version=version)
        emb_gen.generate(wiki)
        print('Time of embedding generation:', time.time() - init)

        print('Creating collection')
        for i, pages in enumerate(wiki):
            data, vector_size = self.loadData(i, pages)
            print(len(data))
            if i == 0:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
            self.client.upload_points(
                collection_name=self.collection,
                points=data,
                parallel=4,
                max_retries=3,
            )
        print('Time to create collection:', time.time() - init)

        
    def loadData(self, i, pages):
        ids, texts = pages['id'], pages['text'] #, pages['lines']
        embeddings = np.load(f'embeddings/{self.collection}/{i}.npy')
        num_pages, vector_size = embeddings.shape[0], embeddings.shape[1]
        data = [PointStruct(id=i*num_pages+j, vector=embeddings[j], payload={"id": id, "text": text}) 
                for j, id, text in zip(range(num_pages), ids, texts)] #, "lines": lines
        return data, vector_size

