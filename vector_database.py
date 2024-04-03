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
            if self.collection not in [c.name for c in self.client.get_collections().collections]:
                self.start_collection(wiki, encoder, version)
        else:
            raise Exception('Invalid client type')
        

        
    def search(self, queries, top=10):
        collection_names = [collection.name for collection in self.client.get_collections().collections]

        if self.collection not in collection_names:
            raise Exception('Collection does not exist')
        
        search_queries = [models.SearchRequest(vector=queries[i].tolist(), limit=top, with_payload=True) for i in range(len(queries))]
        return self.client.search_batch(collection_name=self.collection, requests=search_queries)
        

    def start_collection(self, wiki, encoder, version):
        if wiki is None:
            raise Exception('wiki is required to create collection')
        
        init = time.time()

        print('Generating embeddings')
        emb_gen = EmbeddingGenerator(encoder=encoder, version=version)
        #emb_gen.generate(wiki)
        print('Time of generation:', time.time() - init)

        print('Creating collection')
        # count the number of files in folder embeddings
        for i, pages in enumerate(wiki):
            data, vector_size = self.loadData(i, pages)
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
        print('Time:', time.time() - init)

        
    def loadData(self, i, pages):
        ids, texts, lines = pages['id'], pages['text'], pages['lines']

        embeddings = np.load(f'embeddings/{self.collection}/{i}.npy')
        vector_size = embeddings.shape[1]
        data = [PointStruct(id=id, vector=embeddings[j].tolist(), payload={"text": text, "lines": lines}) for j, id, text, lines in zip(range(embeddings.shape[0]), ids, texts, lines)]
        print(data)
        # for j in range(embeddings.shape[0]):
        #     page = wiki[i*embeddings.shape[0] + j]
        #     id, text, lines = page['id'], page['text'], page['lines']
        #     ps = PointStruct(id=id, vector=embeddings[j].tolist(), payload={"text": text, "lines": lines})
        #     data.append(ps)
        return data, vector_size

