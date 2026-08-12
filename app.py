# 基于fastapi搭建智能问答服务提供智能问答web服务，提供http接口，WebSocket流式问答，静态资源服务。会话历史管理，跨域配置，健康探针

# 导入FastAPI相关模块 用于构建API和Websocket
from fastapi import FastAPI, WebSocket, HTTPException, Query, Depends
# 导入FastAPI响应类型，用于流式响应和文件服务
from fastapi.responses import StreamingResponse, FileResponse
# 导入CORS中间件，支持跨域请求
from fastapi.middleware.cors import CORSMiddleware
# 导入静态文件服务模块
from fastapi.staticfiles import StaticFiles
# 导入WebSocket断开连接
from starlette.websockets import WebSocketDisconnect

import os
#  Pydantic模型用于验证请求
from pydantic import BaseModel
# 导入异步事件循环模块
import asyncio
# 导入JSON 处理模块
import json
# 导入UUID模块， 生成唯一会话ID
import uuid
from typing import Optional, List, Dict, Any

import time

import re


from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MySqlClient
from rag_qa.core.vector_store import VectorStore
from main import IntegratedQASystem



app = FastAPI(title='汽车贷款智能顾问 API', description='集成 FAQ、OCR 和 RAG 的汽车贷款问答系统')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  #允许所有前端域名访问
    allow_credentials=True, #允许凭证  允许请求携带Cookie等身份凭证
    allow_methods=['*'],  # 允许所有HTTP方法
    allow_headers=['*'],  # 允许所有头部
)


qa_system = None

# 管理数据库生命周期
@app.on_event("startup")
async def startup():
    """执行 startup 函数。
        
        params:
            无。
        
        return:
            函数返回值。"""
    global qa_system
    mysql_client = MySqlClient()
    mysql_client = mysql_client.__enter__()
    redis_client = RedisClient()
    redis_client = redis_client.__enter__()
    vector_store = VectorStore()
    vector_store = vector_store.__enter__()
    qa_system = IntegratedQASystem(mysql_client, redis_client, vector_store)
    print("系统初始化完成")

@app.on_event("shutdown")
async def shutdown():
    """执行 shutdown 函数。
        
        params:
            无。
        
        return:
            函数返回值。"""
    global qa_system
    if qa_system:
        qa_system.vector_store.client.close()
        qa_system.mysql_client.cursor.close()
        qa_system.mysql_client.connection.close()
        qa_system.redis_client.client.close()
        # 如果 VectorStore 需要关闭，同样处理
    print("系统已关闭")



# Pydantic请求/响应数据类型定义
 ## 非流式拆查询接口入参结构体 -> 接受Post请求Json请求
class QueryRequest(BaseModel):
    query: str                            # 查询内容，必填
    source_filter: Optional[str] = None   # 汽车金融知识类别过滤，可选
    session_id: Optional[str] = None      # 会话ID，可选

class QueryResponse(BaseModel):
    answer: str                          #答案
    is_streaming: bool                   #是否流式响应    True代表需要切换WebSocket接口
    session_id: str                      # 会话ID
    processing_time: float          # 处理时间

# 将本地static目录挂载至服务器根路径
app.mount('/static', StaticFiles(directory='static'), name='static')

@app.get('/')        # http://ip:port/
async def get_root():
    # 返回 static目录首页下的html文件
    """执行 get_root 函数。
        
        params:
            无。
        
        return:
            函数返回值。"""
    return FileResponse('static/index.html')



# HTTP核心业务接口
#  创建对话ID
@app.post('/api/create_session')
async def create_session():
    """执行 create_session 函数。
        
        params:
            无。
        
        return:
            函数返回值。"""
    session_id = str(uuid.uuid4())
    return {'session_id': session_id}


# 查询历史消息
@app.get('/api/history/{session_id}')
async def get_history(session_id: str):
    """执行 get_history 函数。
        
        params:
            session_id: 参数说明。
        
        return:
            函数返回值。"""
    try:
        history = qa_system.get_session_history(session_id)
        return {'session_id': session_id, 'history': history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'获取历史记录失败: {str(e)}')

# 根据对话ID清除历史接口
@app.delete('/api/history/{session_id}')
async def clear_history(session_id: str):
    """执行 clear_history 函数。
        
        params:
            session_id: 参数说明。
        
        return:
            函数返回值。"""
    success = qa_system.clear_session_history(session_id)
    if success:
        return {'status': success, 'message': '历史记录已清除'}
    else:
        raise HTTPException(status_code = 500, detail='清除历史记录失败')

# 同步非主流式问答post, 一次性返回完整回答，不支持分段流式输出
# 非流式查询接口
@app.post("/api/query")
async def query(request: QueryRequest):
    """执行 query 函数。
        
        params:
            request: 参数说明。
        
        return:
            函数返回值。"""
    start_time = time.time()  # 记录开始时间
    # 使用请求中的 session_id 或生成新 ID
    session_id = request.session_id or str(uuid.uuid4())

    # 执行 BM25 搜索
    answer, need_rag = qa_system.bm25_search.search(request.query)
    if need_rag:
        # 需要 RAG，提示使用 WebSocket
        return {
            "answer": "请使用WebSocket接口获取流式响应",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    # 返回 MySQL 答案
    return {
        "answer": answer,
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time
    }

@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    """执行 websocket_endpoint 函数。
        
        params:
            websocket: 参数说明。
        
        return:
            函数返回值。"""
    await websocket.accept()  # 接受 WebSocket 连接
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            request_data = json.loads(data)  # 解析 JSON 数据
            # 获取查询参数
            query = request_data.get("query")
            source_filter = request_data.get("source_filter")
            session_id = request_data.get("session_id", str(uuid.uuid4()))
            start_time = time.time()  # 记录开始时间
            # 发送开始标志
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.send_json({
                    "type": "start",
                    "session_id": session_id
                })
            collected_answer = ""
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                collected_answer += token  # 累积答案
                if is_complete and not collected_answer:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        # 发送结束标志
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break
                if token and websocket.client_state == websocket.client_state.CONNECTED:
                    # 发送 token 数据
                    await websocket.send_json({
                        "type": "token",
                        "token": token,
                        "session_id": session_id
                    })
                if is_complete:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        # 发送结束标志
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break
                await asyncio.sleep(0.01)  # 控制流式输出的速度
    except WebSocketDisconnect as e:
        # 记录 WebSocket 断开信息
        print(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    except Exception as e:
        # 记录错误信息
        print(f"WebSocket error: {str(e)}")
        if websocket.client_state == websocket.client_state.CONNECTED:
            # 发送错误消息
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
    finally:
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                # 关闭 WebSocket 连接
                await websocket.close()
        except Exception as e:
            # 记录关闭连接时的错误
            print(f"Error closing WebSocket: {str(e)}")




@app.get("/health")
async def health_check():
    # 返回健康状态标记, k8s调用该接口返回200则判断服务正常运行.
    """执行 health_check 函数。
        
        params:
            无。
        
        return:
            函数返回值。"""
    return {"status": "healthy"}  # 返回健康状态

# 获取有效知识类别接口
@app.get("/api/sources")
async def get_sources():
    """执行 get_sources 函数。
        
        params:
            无。
        
        return:
            函数返回值。"""
    return {"sources": qa_system.config.APP_VALID_SOURCES}


# 主程序入口
if __name__ == "__main__":
    # springboot = springcore + spirngmvc + tomcat
    # fastapi = springmvc (url -> 方法调用)
    # uvicorn = tomcat (服务容器，负责处理多线程、高并发等)

    import uvicorn      # uvicorn库 -> 异步web服务容器, 用于启动FastAPI应用.
    import os

    # 从环境变量获取主机和端口，默认值为 0.0.0.0:8080
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8080))

    # 运行 FastAPI 应用，监听指定的主机和端口
    # reload=False 关闭热重载, 生产环境禁用, 避免性能损耗.
    uvicorn.run("app:app", host=host, port=port, reload=False)
