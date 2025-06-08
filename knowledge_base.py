import os
import json
import asyncio
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

class KnowledgeBase:
    """
    向量知识库管理系统
    支持文档存储、检索和管理
    """
    
    def __init__(self, db_path: str = "./knowledge_db", model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # 初始化 ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(model_name)
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.logger = logging.getLogger(__name__)
    
    async def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        添加文档到知识库
        
        Args:
            content: 文档内容
            metadata: 文档元数据
            doc_id: 文档ID，如果不提供则自动生成
        
        Returns:
            str: 文档ID
        """
        try:
            # 生成文档ID
            if not doc_id:
                doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(content)}"
            
            # 生成嵌入向量
            embedding = await asyncio.to_thread(
                self.embedding_model.encode, content
            )
            
            # 准备元数据
            if not metadata:
                metadata = {}
            metadata.update({
                "created_at": datetime.now().isoformat(),
                "content_length": len(content)
            })
            
            # 添加到集合
            self.collection.add(
                documents=[content],
                embeddings=[embedding.tolist()],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            self.logger.info(f"成功添加文档: {doc_id}")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"添加文档失败: {e}")
            raise
    
    async def search_documents(
        self,
        query: str,
        n_results: int = 5,  # 正确的参数名是n_results，不是top_k
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 搜索查询
            n_results: 返回结果数量
            where: 过滤条件
        
        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            # 生成查询嵌入向量
            query_embedding = await asyncio.to_thread(
                self.embedding_model.encode, query
            )
            
            # 执行搜索
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # 格式化结果
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": 1 - results["distances"][0][i]  # 转换为相似度
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"搜索文档失败: {e}")
            raise
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        删除文档
        
        Args:
            doc_id: 文档ID
        
        Returns:
            bool: 是否删除成功
        """
        try:
            self.collection.delete(ids=[doc_id])
            self.logger.info(f"成功删除文档: {doc_id}")
            return True
        except Exception as e:
            self.logger.error(f"删除文档失败: {e}")
            return False
    
    async def update_document(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        更新文档
        
        Args:
            doc_id: 文档ID
            content: 新的文档内容
            metadata: 新的元数据
        
        Returns:
            bool: 是否更新成功
        """
        try:
            update_data = {}
            
            if content is not None:
                # 生成新的嵌入向量
                embedding = await asyncio.to_thread(
                    self.embedding_model.encode, content
                )
                update_data["documents"] = [content]
                update_data["embeddings"] = [embedding.tolist()]
            
            if metadata is not None:
                metadata["updated_at"] = datetime.now().isoformat()
                update_data["metadatas"] = [metadata]
            
            if update_data:
                update_data["ids"] = [doc_id]
                self.collection.update(**update_data)
                self.logger.info(f"成功更新文档: {doc_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"更新文档失败: {e}")
            return False
    
    async def get_document_count(self) -> int:
        """
        获取文档总数
        
        Returns:
            int: 文档总数
        """
        try:
            return self.collection.count()
        except Exception as e:
            self.logger.error(f"获取文档数量失败: {e}")
            return 0
    
    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        列出文档
        
        Args:
            limit: 限制数量
            offset: 偏移量
        
        Returns:
            List[Dict]: 文档列表
        """
        try:
            results = self.collection.get(
                limit=limit,
                offset=offset,
                include=["documents", "metadatas"]
            )
            
            documents = []
            for i in range(len(results["ids"])):
                documents.append({
                    "id": results["ids"][i],
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i]
                })
            
            return documents
            
        except Exception as e:
            self.logger.error(f"列出文档失败: {e}")
            return []
    
    async def initialize_if_needed(self):
        """如果需要则初始化知识库"""
        try:
            # 检查是否已有数据
            count = self.collection.count()
            if count == 0:
                # 添加一些默认知识
                await self._add_default_knowledge()
                self.logger.info("已添加默认知识库内容")
            else:
                self.logger.info(f"知识库已存在 {count} 条记录")
        except Exception as e:
            self.logger.error(f"知识库初始化失败: {e}")
    
    async def _add_default_knowledge(self):
        """添加默认知识"""
        default_docs = [
            {
                "content": "我是一个智能AI助手，可以帮助您进行对话、查询天气、学习知识等。",
                "metadata": {"type": "system", "category": "introduction"}
            },
            {
                "content": "要查询天气，您可以说'北京今天天气怎么样'或'上海明天会下雨吗'。",
                "metadata": {"type": "system", "category": "weather_help"}
            },
            {
                "content": "要让我学习新知识，您可以说'学习：这里是要学习的内容'。",
                "metadata": {"type": "system", "category": "learning_help"}
            }
        ]
        
        for doc in default_docs:
            await self.add_document(
                content=doc["content"],
                metadata=doc["metadata"]
            )
    
    async def get_conversation_context(self, session_id: str, limit: int = 5) -> List[Dict]:
        """获取对话上下文"""
        try:
            results = self.collection.query(
                where={"session_id": session_id},
                n_results=limit
            )
            
            return [
                {
                    "content": content,
                    "metadata": metadata
                }
                for content, metadata in zip(results["documents"][0], results["metadatas"][0])
            ]
        except Exception as e:
            self.logger.error(f"获取对话上下文失败: {e}")
            return []