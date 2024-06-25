from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models
import time
import uuid
import torch

# log and warning suppression
import logging
logging.getLogger('qdrant_client').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)

PAGES_PER_FILE = 50000


class VectorDatabase():

    def __init__(self, wiki_loader, emb_gen, host='local', ip='localhost', port=6333, version='v1'):
        self.collection = version
        self.wiki_loader = wiki_loader
        self.host = host
        self.ip = ip
        self.port = port

        self.__initialization__(emb_gen)
        
    
    def __initialization__(self, emb_gen):
        if self.host == 'local':
            self.client = QdrantClient(':memory:')
            self.__start_collection__(emb_gen)
        elif self.host == 'docker':
            self.client = QdrantClient(f'{self.ip}:{self.port}', timeout=1000)
            if (self.collection == 'v0' or self.collection == 'v1') and self.collection in [c.name for c in self.client.get_collections().collections]:
                ## connect
                print('Collection already exists')
                ## delete collection
            elif self.collection in [c.name for c in self.client.get_collections().collections]:
                self.client.delete_collection(collection_name=self.collection)
                self.__start_collection__(emb_gen)
            else:
                self.__start_collection__(emb_gen)
        else:
            raise Exception('Invalid host type' + self.host)


    def __start_collection__(self, emb_gen):
        print('Creating collection')
        init = time.time()
        emb_gen.eval()
        for i, pages in enumerate(self.wiki_loader):
            data, vector_size = self.__load_data__(pages, emb_gen)
            if i == 0:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    hnsw_config=models.HnswConfigDiff(
                        ef_construct=100,
                        m=16,
                        max_indexing_threads=0,
                        full_scan_threshold=10000,
                        on_disk=True,
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        deleted_threshold=0.2,
                        vacuum_min_vector_number=1000,
                        default_segment_number=0,
                        max_segment_size=None,
                        memmap_threshold=20000,
                        indexing_threshold=20000,
                        flush_interval_sec=5,
                        max_optimization_threads=None,
                    ),
                    wal_config=models.WalConfigDiff(
                        wal_capacity_mb=32,
                        wal_segments_ahead=2,
                    ),
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            always_ram=False,
                        ),
                    ),
                    on_disk_payload=True,
                )
            self.client.upload_points(
                collection_name=self.collection,
                points=data,
                parallel=2,
                max_retries=3,
            )
            print(f'Block {i+1}/{len(self.wiki_loader)} done')
        print('Time to create collection:', time.time() - init)


    def __uuid__(self, id):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id))

    @torch.no_grad()
    def __load_data__(self, pages, emb_gen):
        ids, texts = pages['id'], pages['text'] #, pages['lines']
        embeddings = emb_gen(texts).detach().cpu().half().numpy()
        num_pages, vector_size = embeddings.shape[0], embeddings.shape[1]
        data = [PointStruct(id=self.__uuid__(id), vector=embeddings[j], payload={'id': id,'text': text}) 
                for j, id, text in zip(range(num_pages), ids, texts)] #, "lines": lines
        return data, vector_size
    

    def refresh(self, emb_gen):
        self.wiki_loader.dataset.refresh()
        self.collection = 'vx'
        self.__initialization__(emb_gen)

        
    def search_similar(self, queries, top=10, with_payload=False, with_vector=False):
        collection_names = [collection.name for collection in self.client.get_collections().collections]

        if self.collection not in collection_names:
            raise Exception('Collection does not exist')
        
        queries = queries.detach().cpu().numpy()
        search_queries = [models.SearchRequest(vector=queries[i], limit=top, with_payload=with_payload, with_vector=with_vector) for i in range(len(queries))]
        return self.client.search_batch(collection_name=self.collection, requests=search_queries)
    

    def search_dissimilar(self, top=10):
        pages = self.wiki_loader.dataset.get_random_ids(top)
        return pages

    
    def search_ids(self, ids):
        ids = [self.__uuid__(id) for id in ids]
        return self.client.retrieve(collection_name=self.collection, ids=ids, with_payload=True, with_vectors=False)
