from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse
import aiohttp
import uvicorn
import json
import asyncio
import base64
import io
import time
import os
import re
from PIL import Image
from collections import defaultdict

app = FastAPI()

# -------------------------- 核心配置 --------------------------
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
OLLAMA_CHAT_MODEL = "deepseek-r1:14b"
OLLAMA_URL = "http://192.168.2.246:11434"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60
TEMP_IMAGE_DIR = "/tmp/affine_images"
CONVERSATION_HISTORY = defaultdict(list)
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
os.chmod(TEMP_IMAGE_DIR, 0o777)
# --------------------------------------------------------------------------------

# -------------------------- 图片处理（强制解析）--------------------------
def is_base64(s: str) -> bool:
    """判断字符串是否为Base64数据"""
    try:
        return len(s) % 4 == 0 and re.match('^[A-Za-z0-9+/]+[=]*$', s) is not None
    except:
        return False

def save_base64_image(base64_data: str) -> str:
    """强制解析任何可能的Base64图片数据"""
    try:
        if "," in base64_data:
            base64_data = base64_data.split(",", 1)[1]
        padding = len(base64_data) % 4
        if padding != 0:
            base64_data += "=" * (4 - padding)
        
        if not is_base64(base64_data):
            raise ValueError("不是有效的Base64数据")
        
        image_bytes = base64.b64decode(base64_data)
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.thumbnail((1024, 1024))
            image_name = f"final_img_{int(time.time() * 1000)}.jpg"
            image_path = os.path.join(TEMP_IMAGE_DIR, image_name)
            img.save(image_path, "JPEG", quality=80)
        
        print(f"✅ 图片处理成功：{image_path}（尺寸：{img.size}，大小：{os.path.getsize(image_path)/1024:.1f}KB）")
        return image_path
    except Exception as e:
        print(f"❌ 图片处理失败：{str(e)}")
        raise

# -------------------------- 会话管理（确保稳定）--------------------------
async def get_session_id(request: Request) -> str:  # 改为异步函数
    """优先从请求体提取会话ID，确保同一对话ID不变"""
    # 1. 尝试从请求体提取（最可靠）
    try:
        body = json.loads(await request.body())  # 现在可以在异步函数中使用await了
        session_id = body.get("session_id") or body.get("conversation_id")
        if session_id:
            return str(session_id)
    except:
        pass
    
    # 2. 尝试从 headers/params 提取
    session_id = request.headers.get("X-Session-ID") or request.query_params.get("session_id")
    if session_id:
        return session_id
    
    # 3. 最后用客户端IP+用户代理生成稳定ID
    client_ip = request.client.host
    user_agent = request.headers.get("User-Agent", "unknown")
    return f"stable_{hash(f'{client_ip}_{user_agent}') % 100000}"

def update_conversation_history(session_id: str, role: str, content: str, image_path: str = None):
    """强制保留历史，增加长度限制保护"""
    history = CONVERSATION_HISTORY[session_id]
    history.append({"role": role, "content": content, "image_path": image_path, "timestamp": time.time()})
    if len(history) > 30:
        CONVERSATION_HISTORY[session_id] = history[-30:]
    print(f"📝 会话{session_id}历史长度：{len(CONVERSATION_HISTORY[session_id])}轮")

# -------------------------- 主请求处理 --------------------------
@app.post("/v1/models/{model_path:path}")
async def proxy_request(model_path: str, request: Request):
    # 清理过期图片
    asyncio.create_task(asyncio.to_thread(lambda: [
        os.remove(os.path.join(TEMP_IMAGE_DIR, f)) 
        for f in os.listdir(TEMP_IMAGE_DIR) 
        if os.path.isfile(os.path.join(TEMP_IMAGE_DIR, f)) and 
        time.time() - os.path.getmtime(os.path.join(TEMP_IMAGE_DIR, f)) > 600
    ]))
    
    # 获取稳定的会话ID（调用异步函数需要用await）
    session_id = await get_session_id(request)
    print(f"\n📌 处理会话：{session_id}，请求路径：{model_path}")

    # 解析请求体（保留原始数据用于调试）
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except Exception as e:
        print(f"解析请求体错误：{str(e)}，原始数据：{raw_body[:200]}...")
        return StreamingResponse(iter([f"无效JSON：{str(e)}"]), status_code=400)

    # 1. 处理嵌入请求
    if "gemini-embedding-001" in model_path and "embedContent" in model_path:
        text = body.get("content", {}).get("parts", [{}])[0].get("text", "")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                ) as resp:
                    if resp.status == 200:
                        return {"embedding": {"values": (await resp.json()).get("embedding", [])}}
                    error = await resp.text()
                    print(f"嵌入请求失败：{error}")
                    return StreamingResponse(iter([f"嵌入错误：{error}"]), status_code=resp.status)
        except Exception as e:
            return StreamingResponse(iter([f"嵌入请求异常：{str(e)}"]), status_code=500)

    # 2. 处理多模态聊天（核心修复）
    elif "gemini-2.5-flash" in model_path and "streamGenerateContent" in model_path:
        try:
            current_text = ""
            current_image_path = None
            gemini_messages = body.get("contents", [])

            for msg in gemini_messages:
                for part in msg.get("parts", []):
                    if isinstance(part, dict) and "text" in part:
                        current_text += str(part["text"]) + "\n"
                    
                    if isinstance(part, dict):
                        for key, value in part.items():
                            if (any(k in key.lower() for k in ["image", "media", "img", "pic"]) and 
                                isinstance(value, str) and len(value) > 500):
                                print(f"🔍 发现疑似图片字段：{key}（长度：{len(value)}）")
                                try:
                                    current_image_path = save_base64_image(value)
                                    break
                                except:
                                    print(f"⚠️  字段{key}不是有效图片，继续检测")
                        
                        if not current_image_path and "data" in part:
                            data = part["data"]
                            if isinstance(data, str) and len(data) > 500:
                                print(f"🔍 检测到data字段，尝试解析图片...")
                                try:
                                    current_image_path = save_base64_image(data)
                                except:
                                    pass

            current_text = current_text.strip() or "请分析图片内容"
            update_conversation_history(session_id, "user", current_text, current_image_path)

            full_history = []
            for msg in CONVERSATION_HISTORY[session_id]:
                content = msg["content"]
                if msg["image_path"]:
                    content += "\n【该消息包含图片，请结合图片内容回答】"
                full_history.append({"role": msg["role"], "content": content})

            ollama_body = {
                "model": OLLAMA_CHAT_MODEL,
                "messages": full_history,
                "stream": True,
                "timeout": REQUEST_TIMEOUT,
                "options": {"num_ctx": 4096}
            }
            if current_image_path:
                ollama_body["image"] = current_image_path
                print(f"🚀 发送带图片请求：模型={OLLAMA_CHAT_MODEL}，历史={len(full_history)}轮")
            else:
                print(f"🚀 发送文本请求：历史={len(full_history)}轮（若上传了图片则未检测到）")

            async def sse_generator():
                model_reply = ""
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(f"{OLLAMA_URL}/api/chat", json=ollama_body) as resp:
                            print(f"LLaVA响应状态：{resp.status}")
                            if resp.status != 200:
                                error = await resp.text()
                                yield f"data: {json.dumps({'error': f'模型错误：{error}'})}\n\n"
                                return

                            async for line in resp.content:
                                if line:
                                    line_str = line.decode().strip()
                                    if not line_str:
                                        continue
                                    try:
                                        ollama_line = json.loads(line_str)
                                    except:
                                        yield f"data: {json.dumps({'error': '响应格式错误'})}\n\n"
                                        continue

                                    content = ollama_line.get("message", {}).get("content", "")
                                    done = ollama_line.get("done", False)
                                    if content:
                                        model_reply += content
                                        sse_data = json.dumps({
                                            "streamType": "text",
                                            "candidates": [{
                                                "content": {"parts": [{"text": content}], "role": "model"},
                                                "finishReason": None
                                            }]
                                        })
                                        yield f"data: {sse_data}\n\n"
                                    
                                    if done:
                                        update_conversation_history(session_id, "model", model_reply.strip())
                                        yield "data: [DONE]\n\n"
                                        break
                except Exception as e:
                    yield f"data: {json.dumps({'error': f'处理错误：{str(e)}'})}\n\n"
                finally:
                    if current_image_path and os.path.exists(current_image_path):
                        os.remove(current_image_path)
                        print(f"🗑️ 清理临时图片：{current_image_path}")

            return StreamingResponse(
                content=sse_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        except Exception as e:
            error = f"聊天处理错误：{str(e)}"
            print(error)
            return StreamingResponse(iter([error]), status_code=500)

    return StreamingResponse(iter([f"路径未找到：{model_path}"]), status_code=404)

if __name__ == "__main__":
    print(f"🚀 启动最终版多模态代理（解决图片和上下文问题），模型：{OLLAMA_CHAT_MODEL}")
    uvicorn.run(app, host="0.0.0.0", port=4000, log_level="info")
