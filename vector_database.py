import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from embedding_generation import EmbeddingGenerator
from qdrant_client.http import models
import time
import uuid

# log and warning suppression
import logging
logging.getLogger('qdrant_client').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)

PAGES_PER_FILE = 50000


class VectorDatabase():

    def __init__(self, client='local', ip='localhost', port=6333, encoder='mpnet', version='v1', wiki_loader=None):
        self.collection = f'{encoder}_{version}'
        if client == 'local':
            self.client = QdrantClient(':memory:')
            self.__start_collection__(wiki_loader, encoder, version)

        elif client == 'docker':
            self.client = QdrantClient(f'{ip}:{port}')
            if self.collection in [c.name for c in self.client.get_collections().collections]:
                ## delete collection
                self.client.delete_collection(collection_name=self.collection)
            self.__start_collection__(wiki_loader, encoder, version)
        else:
            raise Exception('Invalid client type')

        

    def __start_collection__(self, wiki_loader, encoder, version):
        print('Creating collection')
        if wiki_loader is None:
            raise Exception('wiki is required to create collection')
        init = time.time()

        emb_gen = EmbeddingGenerator(encoder=encoder, version=version)

        for i, pages in enumerate(wiki_loader):
            data, vector_size = self.__load_data__(pages, emb_gen)
            if i == 0:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
            self.client.upload_points(
                collection_name=self.collection,
                points=data,
                parallel=2,
                max_retries=3,
            )
            print(f'Block {i+1}/{len(wiki_loader)} done')
        print('Time to create collection:', time.time() - init)


    def __uuid__(self, id):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id))

        
    def __load_data__(self, pages, emb_gen):
        ids, texts = pages['id'], pages['text'] #, pages['lines']
        embeddings = emb_gen(texts)
        num_pages, vector_size = embeddings.shape[0], embeddings.shape[1]
        data = [PointStruct(id=self.__uuid__(id), vector=embeddings[j], payload={'id': id,'text': text}) 
                for j, id, text in zip(range(num_pages), ids, texts)] #, "lines": lines
        return data, vector_size


    def search_similar(self, queries, top=10):
        collection_names = [collection.name for collection in self.client.get_collections().collections]

        if self.collection not in collection_names:
            raise Exception('Collection does not exist')
        
        search_queries = [models.SearchRequest(vector=queries[i], limit=top, with_payload=True, with_vector=True) for i in range(len(queries))]
        return self.client.search_batch(collection_name=self.collection, requests=search_queries)
    
    
    def search_ids(self, ids):
        ids = [self.__uuid__(id) for id in ids]
        return self.client.retrieve(collection_name=self.collection, ids=ids, with_payload=True, with_vectors=True)
