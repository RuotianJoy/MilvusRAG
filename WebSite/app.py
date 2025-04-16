#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MilvusRAG Web应用
提供基于Flask的可视化界面，用于上传测试文件、执行测试并显示Milvus数据库内容
"""

import os
import pandas as pd
import csv
import json
import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, session, Response
from werkzeug.utils import secure_filename
from pymilvus import connections, utility, Collection, DataType
import sys
import uuid
from datetime import datetime
import numpy as np
import math
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import re
import traceback

# 禁用PyTorch MPS加速，避免Apple Silicon上的图形处理器错误
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# 添加自定义JSON编码器
class NumpyEncoder(json.JSONEncoder):
    """处理NumPy数据类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# 添加API密钥环境变量检查
# 这里您需要替换为自己的API密钥
DEFAULT_API_KEY = "sk-qvJtRhsJdn97oivEXcM3pxG1FClBwvXPCxfenxOfIc11Xeyy"  # 在这里填入您的OpenAI API密钥
if "OPENAI_API_KEY_GPT" not in os.environ and DEFAULT_API_KEY:
    print("设置默认OpenAI API密钥环境变量")
    os.environ["OPENAI_API_KEY_GPT"] = DEFAULT_API_KEY

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 从RAGTesting导入测试评估函数
sys.path.append(os.path.join(project_root, "RAGTesting"))
from RAGTesting import RAG_TestingAndEvaluation

# 配置应用
app = Flask(__name__)
app.secret_key = "milvusrag_secret_key"
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 限制上传文件大小为32MB
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}
app.config['SESSION_TYPE'] = 'filesystem'  # 使用文件系统存储会话数据
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # 禁用JSON美化，减少响应大小
app.config['JSON_SORT_KEYS'] = False  # 禁用JSON键排序，保持原始顺序
app.json.encoder = NumpyEncoder  # 使用自定义JSON编码器

# Milvus连接配置
MILVUS_HOST = '127.0.0.1'  # 默认本地主机
MILVUS_PORT = '19530'  # 默认端口

# 结果历史记录文件
app.config['RESULTS_HISTORY_FILE'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_history.json')
# 确保数据目录存在
os.makedirs(os.path.dirname(app.config['RESULTS_HISTORY_FILE']), exist_ok=True)

# 确保历史记录文件存在
if not os.path.exists(app.config['RESULTS_HISTORY_FILE']):
    with open(app.config['RESULTS_HISTORY_FILE'], 'w', encoding='utf-8') as f:
        json.dump([], f)

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 创建一个全局字典来存储每个会话的进度信息
session_progress = {}

# 集合名称映射 - 将技术名称映射为人类可读的名称
COLLECTION_NAME_MAP = {
    "arwu_text": "ARWU文本集合",
    "arwu_score": "ARWU评分集合",
    "arwu_enhanced": "ARWU增强集合",
    "us_colleges": "美国高校集合",
    "university_subjects": "大学学科集合",
    "usnews2025_school_subject_relations_text": "USNews学校学科关系集合",
    "usnews2025_school_subject_relations": "USNews学校学科关系集合",
    "usnews2025_subjects": "USNews学科集合",
    "university_base": "大学基础信息集合",
    "university_summary": "大学摘要集合",
    "the2025_subjects": "THE2025学科集合",
    "university_statistics": "大学统计数据集合",
    "the2025_basic_info": "THE2025基本信息集合",
    "the2025_metrics": "THE2025指标集合",
    "the2025_meta": "THE2025元数据集合",
    "university_indicators": "大学指标集合",
    "usnews2025_schools": "USNews2025高校集合",
    "us_colleges_WIKI": "美国高校维基数据集合",
}

def normalize_source_name(source):
    """规范化数据来源名称，将技术名称转换为人类可读名称"""
    source = str(source).strip()
    
    # 如果为空，直接返回
    if not source:
        return ""
        
    # 处理嵌套格式（如"xxx:yyy"，保留最后部分）
    if ":" in source:
        parts = source.split(":")
        source = parts[-1].strip()
    
    # 处理关键词搜索格式
    if "关键词搜索" in source:
        keyword_match = re.search(r'\(([^)]+)\)', source)
        if keyword_match:
            source = keyword_match.group(1).strip()
    
    # 处理"来源: xxx"格式
    if "来源:" in source:
        source_match = re.search(r'来源: ([^)]+)', source)
        if source_match:
            source = source_match.group(1).strip()
            
    # 应用集合名称映射
    if source in COLLECTION_NAME_MAP:
        return COLLECTION_NAME_MAP[source]
        
    # 对技术名称进行一些简单处理
    if source.startswith("arwu_"):
        prefix = "ARWU"
        suffix = source[5:]
        return f"{prefix}{suffix.capitalize()}集合"
    
    if source.startswith("the2025_"):
        prefix = "THE2025"
        suffix = source[8:]
        return f"{prefix}{suffix.capitalize()}集合"
        
    if source.startswith("usnews"):
        prefix = "USNews"
        suffix = source[6:]
        return f"{prefix}{suffix}集合"
        
    # 如果没有匹配到任何规则，原样返回
    return source

def allowed_file(filename):
    """检查文件扩展名是否允许上传"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """渲染首页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    app.logger.info("接收到文件上传请求")
    
    # 检查是否存在文件
    if 'file' not in request.files:
        app.logger.warning("上传请求中没有文件部分")
        flash('没有选择文件', 'warning')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # 检查文件名是否为空
    if file.filename == '':
        app.logger.warning("没有选择文件")
        flash('没有选择文件', 'warning')
        return redirect(url_for('index'))
    
    # 输出文件信息进行调试
    app.logger.info(f"上传的文件: {file.filename}, 类型: {file.content_type}")
    
    # 检查文件格式
    if file and allowed_file(file.filename):
        # 创建会话ID
        session_id = str(int(time.time()))
        
        # 获取原始文件扩展名
        original_extension = os.path.splitext(file.filename)[1].lower()
        
        # 创建新的标准化文件名
        filename = f"RAG{session_id}{original_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # 保存文件
        try:
            file.save(filepath)
            app.logger.info(f"文件已保存到: {filepath}，统一命名为: {filename}")
        except Exception as e:
            app.logger.error(f"保存文件失败: {str(e)}")
            flash(f'保存文件失败: {str(e)}', 'danger')
            return redirect(url_for('index'))
        
        flash(f'文件上传成功', 'success')
        
        # 解析上传的文件
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:  # xlsx
                df = pd.read_excel(filepath)
            
            # 检查文件格式是否符合要求
            if 'Questions' not in df.columns or 'Answers' not in df.columns:
                app.logger.warning(f"文件格式错误: 缺少必要的列")
                flash('文件格式错误，必须包含Questions和Answers列', 'danger')
                return redirect(url_for('index'))
            
            # 保存文件路径和会话ID到会话
            session['uploaded_file'] = filepath
            session['session_id'] = session_id
            app.logger.info(f"会话ID: {session_id}, 文件路径已保存到会话")
            
            # 重定向到处理页面
            return redirect(url_for('process_file', filename=filename, session_id=session_id))
        
        except Exception as e:
            app.logger.error(f'文件处理错误: {str(e)}')
            flash(f'文件处理错误: {str(e)}', 'danger')
            return redirect(url_for('index'))
    
    app.logger.warning(f"不支持的文件类型: {file.filename}")
    flash('不支持的文件类型', 'warning')
    return redirect(url_for('index'))

@app.route('/process/<filename>/<session_id>')
def process_file(filename, session_id):
    """处理上传的测试文件并显示进度页面"""
    print(f"处理文件: {filename}, 会话ID: {session_id}")
    
    # 从会话中获取文件路径，或者重建路径
    filepath = session.get('uploaded_file')
    if not filepath:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"从参数重建文件路径: {filepath}")
    
    # 检查文件是否存在
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        flash(f'文件不存在，请重新上传', 'danger')
        return redirect(url_for('index'))
    
    try:
        # 读取数据
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:  # xlsx
            df = pd.read_excel(filepath)
        
        # 调试输出DataFrame信息
        print(f"DataFrame信息: 列={df.columns.tolist()}, 行数={df.shape[0]}")
        
        # 检查必要的列是否存在
        if 'Questions' not in df.columns or 'Answers' not in df.columns:
            print("缺少必要的Questions或Answers列")
            flash('文件格式错误，必须包含Questions和Answers列', 'danger')
            return redirect(url_for('index'))
        
        # 转换为列表，确保处理各种数据类型
        try:
            # 确保转换为字符串并创建列表
            questions = [str(q) for q in df['Questions'].tolist()]
            answers = [str(a) for a in df['Answers'].tolist()]
            
            if not questions or not answers:
                raise ValueError("Questions或Answers列为空")
                
            questions_count = len(questions)
            print(f"成功读取文件，包含 {questions_count} 个问题")
            
            return render_template('process.html', 
                                 filename=filename,
                                 session_id=session_id,
                                 questions=questions, 
                                 answers=answers,
                                 file_path=filepath)
        except Exception as data_error:
            print(f"处理数据时出错: {str(data_error)}")
            flash(f'数据格式错误: {str(data_error)}. 请确保文件包含有效的Questions和Answers列', 'danger')
            return redirect(url_for('index'))
    
    except Exception as e:
        print(f'处理文件时出错: {str(e)}')
        import traceback
        print(traceback.format_exc())  # 添加详细的错误跟踪
        flash(f'文件处理错误: {str(e)}', 'danger')
        return redirect(url_for('index'))

def add_to_history(test_id, filename, question_count, result_file, avg_scores):
    """将测试结果添加到历史记录"""
    try:
        # 读取现有历史记录
        with open(app.config['RESULTS_HISTORY_FILE'], 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # 添加新记录
        history.append({
            'id': test_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filename': filename,
            'question_count': question_count,
            'result_file': result_file,
            'avg_scores': avg_scores
        })
        
        # 保存历史记录
        with open(app.config['RESULTS_HISTORY_FILE'], 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
        app.logger.info(f"测试结果已添加到历史记录: {test_id}")
        return True
    except Exception as e:
        app.logger.error(f"保存历史记录失败: {str(e)}")
        return False

def get_test_history():
    """获取测试历史记录"""
    try:
        with open(app.config['RESULTS_HISTORY_FILE'], 'r', encoding='utf-8') as f:
            history = json.load(f)
        # 按时间倒序排序
        history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return history
    except Exception as e:
        app.logger.error(f"读取历史记录失败: {str(e)}")
        return []

def get_test_result(test_id):
    """获取指定测试ID的结果详情"""
    history = get_test_history()
    for item in history:
        if item.get('id') == test_id:
            return item
    return None

@app.route('/history')
def test_history():
    """显示测试历史记录页面"""
    history = get_test_history()
    return render_template('history.html', history=history)

@app.route('/result/<test_id>')
def test_result(test_id):
    """显示特定测试的结果页面"""
    result = get_test_result(test_id)
    if not result:
        flash('未找到指定的测试结果', 'warning')
        return redirect(url_for('test_history'))
        
    # 读取结果文件中的详细数据
    try:
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result['result_file'])
        if os.path.exists(result_path):
            df = pd.read_csv(result_path)
            
            # 检查是否是多模型测试（通过检查avg_scores结构或CSV中的Model列）
            is_multi_model = False
            if 'Model' in df.columns and len(df['Model'].unique()) > 1:
                is_multi_model = True
            elif isinstance(result['avg_scores'], dict):
                # 现在avg_scores总是一个字典，它的第一层键是模型名称
                # 如果有多个模型，那就是多模型测试
                if len(result['avg_scores']) > 1:
                    is_multi_model = True
                # 单模型情况下也是以模型名称为键的字典
                else:
                    is_multi_model = False
            
            # 统计数据来源
            source_stats = {}
            if 'Sources' in df.columns:
                for source_list in df['Sources'].dropna():
                    # 确保source_list是字符串
                    if isinstance(source_list, str):
                        sources = source_list.split(', ')
                        for source in sources:
                            source = source.strip()
                            if not source:
                                continue
                                
                            # 使用规范化函数处理数据来源
                            normalized_source = normalize_source_name(source)
                            
                            if normalized_source:  # 确保不是空字符串
                                if normalized_source in source_stats:
                                    source_stats[normalized_source] += 1
                                else:
                                    source_stats[normalized_source] = 1
                    else:
                        app.logger.warning(f"跳过非字符串类型的数据来源: {source_list}")
            
            # 根据是否多模型测试处理数据
            results_data = []
            
            if is_multi_model:
                # 多模型测试情况 - 按问题分组
                questions = df['Question'].unique()
                
                for question in questions:
                    question_df = df[df['Question'] == question].reset_index(drop=True)
                    
                    if len(question_df) == 0:
                        continue
                        
                    # 基本信息
                    # 安全处理Sources列
                    sources = []
                    if 'Sources' in question_df.columns:
                        sources_value = question_df.iloc[0].get('Sources', '')
                        if isinstance(sources_value, str):
                            sources = sources_value.split(', ')
                    
                    item = {
                        'question': question,
                        'reference': question_df.iloc[0]['Reference'],
                        'retrieval_time': float(question_df.iloc[0]['Retrieval_Time']),
                        'is_multi_model': True,
                        'model_answers': {},
                        'model_metrics': {},
                        'sources': sources,
                        'context': question_df.iloc[0].get('Context', '') if 'Context' in question_df.columns else ''
                    }
                    
                    # 填充各模型的数据
                    for _, row in question_df.iterrows():
                        model_name = row['Model']
                        item['model_answers'][model_name] = row['Generated']
                        
                        # 添加该模型的指标
                        metrics = {}
                        for col in row.index:
                            if col not in ['Question', 'Reference', 'Generated', 'Retrieval_Time', 'Model', 'Sources', 'Context']:
                                try:
                                    # 确保值是浮点数
                                    metrics[col] = float(row[col])
                                except (TypeError, ValueError):
                                    # 如果无法转换为浮点数，则跳过
                                    app.logger.warning(f"无法将指标 {col} 的值 {row[col]} 转换为浮点数")
                                    
                        item['model_metrics'][model_name] = metrics
                        
                    results_data.append(item)
            else:
                # 单模型测试情况
                for _, row in df.iterrows():
                    # 安全处理Sources列
                    sources = []
                    if 'Sources' in df.columns:
                        sources_value = row.get('Sources', '')
                        if isinstance(sources_value, str):
                            sources = sources_value.split(', ')
                    
                    item = {
                        'question': row['Question'],
                        'reference': row['Reference'],
                        'generated': row['Generated'],
                        'retrieval_time': float(row['Retrieval_Time']),
                        'is_multi_model': False,
                        'metrics': {},
                        'sources': sources,
                        'context': row.get('Context', '') if 'Context' in df.columns else ''
                    }
                    # 添加指标
                    for col in df.columns:
                        if col not in ['Question', 'Reference', 'Generated', 'Retrieval_Time', 'Model', 'Sources', 'Context']:
                            try:
                                # 确保值是浮点数
                                item['metrics'][col] = float(row[col])
                            except (TypeError, ValueError):
                                # 如果无法转换为浮点数，则跳过
                                app.logger.warning(f"无法将指标 {col} 的值 {row[col]} 转换为浮点数")
                                
                    results_data.append(item)
                
            return render_template('result.html', 
                                result=result, 
                                results_data=results_data,
                                is_multi_model=is_multi_model,
                                source_stats=source_stats)
        else:
            flash('结果文件不存在', 'warning')
            return redirect(url_for('test_history'))
    except Exception as e:
        app.logger.error(f"读取结果文件失败: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        flash(f'读取结果文件失败: {str(e)}', 'danger')
        return redirect(url_for('test_history'))

@app.route('/progress/<session_id>')
def progress_stream(session_id):
    """为特定会话ID返回SSE进度流"""
    def generate():
        # 初始化该会话的进度
        if session_id not in session_progress:
            session_progress[session_id] = {"current": 0, "total": 0, "results": []}
        
        # 发送初始数据
        yield f"data: {json.dumps(session_progress[session_id], cls=NumpyEncoder)}\n\n"
        
        # 使用循环和休眠来等待进度更新
        last_progress = -1
        while True:
            current_progress = session_progress.get(session_id, {}).get("current", 0)
            total = session_progress.get(session_id, {}).get("total", 0)
            
            # 只有当进度发生变化或收到结果时才发送更新
            if current_progress != last_progress:
                yield f"data: {json.dumps(session_progress[session_id], cls=NumpyEncoder)}\n\n"
                last_progress = current_progress
            
            # 如果测试完成，中断流
            if total > 0 and current_progress >= total:
                yield f"data: {json.dumps({'completed': True, **session_progress[session_id]}, cls=NumpyEncoder)}\n\n"
                # 不再在这里删除会话数据，让客户端有机会获取完整结果
                # 会话数据将在GET /run_test接口中被清理
                break
                
            time.sleep(0.5)  # 减少服务器负载
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/run_test', methods=['POST', 'GET'])
def run_test():
    """执行测试并返回结果"""
    # 如果是GET请求，则返回完整结果
    if request.method == 'GET':
        session_id = request.args.get('session_id')
        if not session_id or session_id not in session_progress:
            return jsonify({
                'status': 'error',
                'message': '找不到对应的会话数据'
            }), 404
        
        # 获取会话数据
        progress_data = session_progress[session_id]
        
        # 计算平均得分
        avg_scores = calculate_average_scores(progress_data['results'])
        
        # 如果已经有结果文件，返回文件名
        result_filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            # 保存结果到CSV
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            save_results_to_csv(progress_data['results'], result_filepath)
            
            # 创建唯一测试ID
            test_id = str(uuid.uuid4())
            
            # 检查第一个结果项以确定是否为多模型测试
            use_multi_model = progress_data['results'][0].get('multi_model', False) if progress_data['results'] else False
            
            # 统计数据来源使用情况
            source_stats = {}
            for item in progress_data['results']:
                for source in item.get('sources', []):
                    # 跳过空或None的source
                    if not source:
                        continue
                    
                    # 使用规范化函数处理数据来源
                    normalized_source = normalize_source_name(source)
                    
                    if normalized_source:  # 确保不是空字符串
                        if normalized_source in source_stats:
                            source_stats[normalized_source] += 1
                        else:
                            source_stats[normalized_source] = 1
            
            # 对于单模型情况，将avg_scores包装在以模型名为键的字典中，以与历史记录格式匹配
            if not use_multi_model:
                # 获取模型名称
                model_name = "default"
                if progress_data['results'] and len(progress_data['results']) > 0:
                    # 尝试从结果中获取模型名称
                    for result in progress_data['results']:
                        if 'model' in result:
                            model_name = result['model']
                            break
                
                # 将avg_scores包装在以模型名为键的字典中
                wrapped_avg_scores = {model_name: avg_scores}
                
                # 添加到历史记录
                add_to_history(test_id, f"测试_{datetime.now().strftime('%Y%m%d')}", len(progress_data['results']), result_filename, wrapped_avg_scores)
            else:
                # 多模型情况不需要额外处理
                add_to_history(test_id, f"测试_{datetime.now().strftime('%Y%m%d')}", len(progress_data['results']), result_filename, avg_scores)
            
            # 从会话数据中删除，节约内存
            if session_id in session_progress:
                del session_progress[session_id]
                
        except Exception as e:
            app.logger.error(f"保存结果文件失败: {str(e)}")
            result_filename = None
        
        return jsonify({
            'status': 'success',
            'results': progress_data['results'],
            'avg_scores': avg_scores,
            'result_file': result_filename,
            'test_id': test_id,
            'models': list(progress_data['results'][0].get('model_answers', {}).keys()) if use_multi_model and progress_data['results'] else ["deepseek-chat"],
            'multi_model': use_multi_model,
            'source_stats': source_stats
        })
    
    # 以下是POST请求的处理逻辑
    app.logger.info("收到执行测试请求")
    
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            app.logger.error("请求中没有JSON数据")
            return jsonify({
                'status': 'error',
                'message': '请求数据为空'
            }), 400
            
        filepath = data.get('filepath')
        session_id = data.get('session_id')
        filename = os.path.basename(filepath) if filepath else "未知文件"
        # 获取要使用的模型列表
        models = data.get('models', ["deepseek-chat"])
        if isinstance(models, str):
            models = [models]  # 确保是列表
        
        use_multi_model = len(models) > 1  # 是否使用多模型对比
        
        app.logger.info(f"测试文件路径: {filepath}, 会话ID: {session_id}, 文件名: {filename}")
        app.logger.info(f"使用的模型: {models}, 多模型模式: {use_multi_model}")
        
        # 检查filepath是否为完整路径还是仅为文件名
        if filepath and not os.path.isabs(filepath) and not os.path.exists(filepath):
            # 尝试在上传目录中查找文件
            possible_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(filepath))
            app.logger.info(f"尝试在上传目录中查找文件: {possible_path}")
            if os.path.exists(possible_path):
                filepath = possible_path
                app.logger.info(f"找到文件: {filepath}")
        
        if not filepath or not os.path.exists(filepath):
            app.logger.error(f"文件不存在: {filepath}")
            return jsonify({
                'status': 'error',
                'message': '文件不存在或路径无效'
            }), 404
        
        # 连接Milvus
        app.logger.info("尝试连接Milvus数据库")
        if not connect_to_milvus():
            app.logger.error("无法连接Milvus数据库")
            return jsonify({
                'status': 'error',
                'message': '无法连接Milvus数据库'
            }), 500
        
        app.logger.info("成功连接Milvus数据库")
        
        # 读取数据
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:  # xlsx
                df = pd.read_excel(filepath)
            
            app.logger.info(f"成功读取测试文件，共{len(df)}条记录")
        except Exception as e:
            app.logger.error(f"读取文件失败: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'读取文件失败: {str(e)}'
            }), 500
        
        # 准备结果列表
        results_data = []
        
        # 获取可用的集合
        try:
            available_collections = utility.list_collections()
            app.logger.info(f"获取到的Milvus集合: {available_collections}")
        except Exception as e:
            app.logger.error(f"获取集合列表失败: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'获取集合列表失败: {str(e)}'
            }), 500
        
        knowledge_collections = {}
        
        # 加载集合
        for coll_name in available_collections:
            try:
                knowledge_collections[coll_name] = Collection(name=coll_name)
                knowledge_collections[coll_name].load()
                app.logger.info(f"已加载集合: {coll_name}")
            except Exception as e:
                app.logger.error(f"加载集合 {coll_name} 失败: {str(e)}")
        
        if not knowledge_collections:
            app.logger.error("没有可用的向量集合")
            return jsonify({
                'status': 'error',
                'message': '没有可用的向量集合'
            }), 500
        
        # 初始化会话进度
        max_questions = min(len(df), 50)  # 限制处理数量，避免过长时间
        session_progress[session_id] = {
            "current": 0,
            "total": max_questions,
            "results": []
        }
        app.logger.info(f"将处理前{max_questions}个问题")
        
        # 创建唯一测试ID
        test_id = str(uuid.uuid4())
        
        # 对每个问题执行测试
        for idx, row in df.iterrows():
            if idx >= max_questions:
                break
                
            try:
                # 确保数据为字符串
                user_input = str(row["Questions"])
                reference_answer = str(row["Answers"])
                
                app.logger.info(f"处理问题 {idx+1}/{max_questions}: {user_input[:30]}...")
                
                start_time = time.time()
                
                # 检索知识
                app.logger.info("开始检索知识...")
                context = RAG_TestingAndEvaluation.load_knowledge_variables(knowledge_collections, user_input, top_k=3)
                
                # 提取数据来源信息
                sources = []
                try:
                    for line in context.split('\n'):
                        if line.startswith("以下信息来自"):
                            # 处理基本格式：以下信息来自XXX:
                            source_text = line.replace("以下信息来自", "").strip()
                            
                            # 使用规范化函数处理数据来源
                            normalized_source = normalize_source_name(source_text)
                            if normalized_source and normalized_source not in sources:
                                sources.append(normalized_source)
                                
                        # 处理搜索结果格式：搜索结果 #n (ID: xx, 来源: xxx)
                        elif "搜索结果" in line and "来源:" in line:
                            source_match = re.search(r'来源: ([^)]+)(?:\))?', line)
                            if source_match:
                                source = source_match.group(1).strip()
                                normalized_source = normalize_source_name(source)
                                if normalized_source and normalized_source not in sources:
                                    sources.append(normalized_source)
                except Exception as source_error:
                    app.logger.error(f"提取数据来源时出错: {str(source_error)}")
                    app.logger.error(traceback.format_exc())
                    sources = ["未知数据来源"]
                
                app.logger.info(f"获取到数据来源: {sources}")
                
                # 根据设置决定是使用单一模型还是多模型对比
                if use_multi_model:
                    # 多模型生成答案并对比
                    app.logger.info(f"开始使用多模型生成回答: {models}...")
                    model_answers = RAG_TestingAndEvaluation.generate_answers_with_comparison(user_input, context, models)
                    
                    end_time = time.time()
                    retrieval_time = end_time - start_time
                    
                    app.logger.info(f"多模型生成回答完成，耗时: {retrieval_time:.2f}秒")
                    
                    # 评估每个模型的结果
                    app.logger.info("开始评估多模型结果...")
                    model_metrics = RAG_TestingAndEvaluation.evaluate_multi_model_answers(model_answers, reference_answer)
                    
                    # 添加到结果列表
                    result_item = {
                        'question': user_input,
                        'reference': reference_answer,
                        'model_answers': model_answers,
                        'model_metrics': model_metrics,
                        'retrieval_time': retrieval_time,
                        'multi_model': True,
                        'sources': sources,
                        'context': context
                    }
                else:
                    # 单一模型生成
                    model = models[0] if models else "deepseek-chat"
                    app.logger.info(f"开始使用模型 {model} 生成回答...")
                    generated_answer = RAG_TestingAndEvaluation.generate_answer(user_input, context, model_name=model)
                    
                    end_time = time.time()
                    retrieval_time = end_time - start_time
                    
                    app.logger.info(f"生成回答完成，耗时: {retrieval_time:.2f}秒")
                    
                    # 评估结果
                    app.logger.info("开始评估结果...")
                    metrics = RAG_TestingAndEvaluation.evaluate_with_metrics(generated_answer, reference_answer)
                    
                    # 添加到结果列表
                    result_item = {
                        'question': user_input,
                        'reference': reference_answer,
                        'generated': generated_answer,
                        'retrieval_time': retrieval_time,
                        'metrics': metrics,
                        'multi_model': False,
                        'sources': sources,
                        'context': context
                    }
                
                results_data.append(result_item)
                
                # 更新进度信息
                session_progress[session_id]["current"] = idx + 1
                session_progress[session_id]["results"].append(result_item)
                
                app.logger.info(f"问题 {idx+1} 处理完成")
                
            except Exception as question_error:
                app.logger.error(f"处理问题 {idx+1} 时出错: {str(question_error)}")
        
        # 计算平均分数
        avg_scores = {}
        if results_data:
            if use_multi_model:
                # 如果使用多模型，为每个模型计算平均分
                for model in models:
                    model_avg = {}
                    for metric in ['rouge_1', 'rouge_2', 'rouge_l', 'keyword_match',
                                'precision_1', 'recall_1', 'precision_l', 'recall_l',
                                'keyword_precision', 'keyword_recall']:
                        try:
                            model_avg[f'avg_{metric}'] = sum(item['model_metrics'][model].get(metric, 0) 
                                                         for item in results_data 
                                                         if model in item.get('model_metrics', {})) / len(results_data)
                        except:
                            model_avg[f'avg_{metric}'] = 0
                    avg_scores[model] = model_avg
            else:
                # 单一模型情况
                count = len(results_data)
                avg_metrics = {
                    'avg_rouge_1': 0,
                    'avg_rouge_2': 0,
                    'avg_rouge_l': 0,
                    'avg_keyword_match': 0,
                    'avg_precision_1': 0,
                    'avg_recall_1': 0,
                    'avg_precision_l': 0,
                    'avg_recall_l': 0,
                    'avg_keyword_precision': 0,
                    'avg_keyword_recall': 0
                }
                
                for result in results_data:
                    for metric_name, value in result.get('metrics', {}).items():
                        avg_key = f'avg_{metric_name}'
                        if avg_key in avg_metrics:
                            avg_metrics[avg_key] += value
                
                # 计算平均值
                if count > 0:
                    for key in avg_metrics:
                        avg_metrics[key] = avg_metrics[key] / count
                
                avg_scores = avg_metrics
        
        app.logger.info(f"计算得到的平均分数: {avg_scores}")
        
        # 统计数据来源使用情况
        source_stats = {}
        for item in results_data:
            for source in item.get('sources', []):
                # 跳过空或None的source
                if not source:
                    continue
                
                # 使用规范化函数处理数据来源
                normalized_source = normalize_source_name(source)
                
                if normalized_source:  # 确保不是空字符串
                    if normalized_source in source_stats:
                        source_stats[normalized_source] += 1
                    else:
                        source_stats[normalized_source] = 1
        
        # 生成测试ID
        test_id = str(uuid.uuid4())
        
        # 保存结果到CSV
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_filename = f"test_results_{test_id}_{timestamp}.csv"
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        
        # 转换为DataFrame并保存
        try:
            result_rows = []
            
            if use_multi_model:
                # 多模型结果保存
                for item in results_data:
                    base_row = {
                        'Question': item['question'],
                        'Reference': item['reference'],
                        'Retrieval_Time': item['retrieval_time'],
                        'Sources': ', '.join([str(src) for src in normalize_sources_list(item.get('sources', []))]),  # 规范化数据来源
                        'Context': item.get('context', '')[:1000]  # 限制上下文长度
                    }
                    
                    # 为每个模型创建一行
                    for model in models:
                        if model in item.get('model_answers', {}):
                            row = base_row.copy()
                            row['Model'] = model
                            row['Generated'] = item['model_answers'].get(model, "")
                            
                            # 添加该模型的所有指标
                            model_metrics = item.get('model_metrics', {}).get(model, {})
                            for metric, value in model_metrics.items():
                                row[metric] = value
                                
                            result_rows.append(row)
            else:
                # 单一模型结果
                for item in results_data:
                    row = {
                        'Question': item['question'],
                        'Reference': item['reference'],
                        'Generated': item['generated'],
                        'Retrieval_Time': item['retrieval_time'],
                        'Model': item.get('model', models[0] if 'models' in locals() else "deepseek-chat"),
                        'Sources': ', '.join([str(src) for src in normalize_sources_list(item.get('sources', []))]),  # 规范化数据来源
                        'Context': item.get('context', '')[:1000]  # 限制上下文长度
                    }
                    # 添加所有指标
                    for metric, value in item['metrics'].items():
                        row[metric] = value
                    result_rows.append(row)
            
            result_df = pd.DataFrame(result_rows)
            result_df.to_csv(result_filepath, index=False, encoding='utf-8-sig')
            app.logger.info(f"结果已保存到文件: {result_filepath}")
        except Exception as save_error:
            app.logger.error(f"保存结果文件失败: {str(save_error)}")
        
        # 添加到历史记录
        add_to_history(test_id, filename, len(results_data), result_filename, avg_scores)
        
        return jsonify({
            'status': 'success',
            'results': results_data,
            'avg_scores': avg_scores,
            'result_file': result_filename,
            'test_id': test_id,
            'models': models,
            'multi_model': use_multi_model,
            'source_stats': source_stats
        })
    
    except Exception as e:
        app.logger.error(f"执行测试时发生未知错误: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'执行测试失败: {str(e)}'
        }), 500

@app.route('/milvus')
def milvus_data():
    """显示Milvus数据库内容页面"""
    try:
        connect_to_milvus()
        collections = utility.list_collections()
        return render_template('milvus.html', collections=collections)
    except Exception as e:
        flash(f'连接Milvus失败: {str(e)}')
        return redirect(url_for('index'))

@app.route('/api/collection/<collection_name>')
def get_collection_info(collection_name):
    """获取指定集合的信息"""
    try:
        print(f"尝试获取集合信息: {collection_name}")
        
        # 确保Milvus连接
        connection_status = connect_to_milvus()
        if not connection_status:
            print(f"无法连接到Milvus服务器")
            return jsonify({
                'status': 'error',
                'message': '无法连接到Milvus服务器'
            }), 500
        
        # 尝试加载集合
        try:
            collection = Collection(name=collection_name)
            collection.load()  # 确保集合已加载
        except Exception as coll_error:
            print(f"加载集合 {collection_name} 失败: {str(coll_error)}")
            return jsonify({
                'status': 'error',
                'message': f'加载集合失败: {str(coll_error)}'
            }), 500
        
        # 获取集合统计信息
        try:
            stats = collection.get_statistics()
            row_count = stats["row_count"]
        except Exception as stats_error:
            print(f"获取集合统计信息失败: {str(stats_error)}")
            row_count = 0  # 默认值
        
        # 获取集合架构
        fields = []
        try:
            schema = collection.schema
            for field in schema.fields:
                field_info = {
                    'name': field.name,
                    'type': str(field.dtype),
                    'is_primary': getattr(field, 'is_primary', False)
                }
                if hasattr(field, 'params') and field.params:
                    field_info['params'] = field.params
                fields.append(field_info)
        except Exception as schema_error:
            print(f"获取集合架构失败: {str(schema_error)}")
        
        # 获取索引信息
        indexes = []
        try:
            index_infos = collection.index().info
            for field_name, index_info in index_infos.items():
                indexes.append({
                    'field': field_name,
                    'info': index_info
                })
        except Exception as index_error:
            print(f"获取索引信息失败: {str(index_error)}")
        
        # 释放集合
        try:
            collection.release()
        except:
            pass
            
        return jsonify({
            'name': collection_name,
            'row_count': row_count,
            'fields': fields,
            'indexes': indexes
        })
    
    except Exception as e:
        print(f"获取集合信息时发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/collection/<collection_name>/data')
def get_collection_data(collection_name):
    """获取集合中的数据样本，支持分页"""
    try:
        print(f"尝试获取集合数据: {collection_name}")
        
        # 获取分页参数
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 10))
        offset = (page - 1) * page_size
        
        # 确保Milvus连接
        if not connect_to_milvus():
            print(f"无法连接到Milvus服务器")
            return jsonify({
                'status': 'error',
                'message': '无法连接到Milvus服务器'
            }), 500
        
        # 打开集合
        try:
            collection = Collection(name=collection_name)
            collection.load()
        except Exception as coll_error:
            print(f"加载集合 {collection_name} 失败: {str(coll_error)}")
            return jsonify({
                'status': 'error',
                'message': f'加载集合失败: {str(coll_error)}'
            }), 500
            
        # 获取非向量字段
        non_vector_fields = []
        for field in collection.schema.fields:
            # 检查字段类型，DataType.FLOAT_VECTOR对应值是100
            if not hasattr(field, 'dtype') or field.dtype != DataType.FLOAT_VECTOR:
                non_vector_fields.append(field.name)
        
        # 如果没有非向量字段，至少返回ID字段
        if not non_vector_fields:
            non_vector_fields = ["id"]
        
        # 获取总数据量 - 先测试collection.num_entities
        try:
            # 先尝试使用num_entities属性
            total_count = collection.num_entities
            print(f"使用num_entities获取总数: {total_count}")
        except:
            try:
                # 如果失败，尝试使用count方法
                total_count = collection.query(expr="", output_fields=["count(*)"])[0]["count(*)"]
                print(f"使用count(*)查询获取总数: {total_count}")
            except:
                try:
                    # 如果以上都失败，尝试使用统计信息
                    stats = collection.get_statistics()
                    total_count = int(stats["row_count"])
                    print(f"使用get_statistics获取总数: {total_count}")
                except Exception as stats_error:
                    # 最后的fallback，先查询所有数据计数
                    print(f"获取集合统计信息失败，尝试获取所有数据: {str(stats_error)}")
                    try:
                        all_data = collection.query(expr="", output_fields=["id"])
                        total_count = len(all_data)
                        print(f"通过查询所有ID获取总数: {total_count}")
                    except:
                        print("所有获取总数方法都失败，设置为默认值0")
                        total_count = 0  # 默认值
        
        # 打印调试信息
        print(f"计算分页 - 总数据量: {total_count}, 页大小: {page_size}")
        
        # 计算总页数
        total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1
        print(f"计算得到总页数: {total_pages}")
            
        # 查询数据
        try:
            results = collection.query(
                expr="",
                output_fields=non_vector_fields,
                offset=offset,
                limit=page_size
            )
            print(f"查询到数据条数: {len(results)}")
        except Exception as query_error:
            print(f"查询集合数据失败: {str(query_error)}")
            return jsonify({
                'status': 'error',
                'message': f'查询数据失败: {str(query_error)}'
            }), 500
            
        # 释放集合
        try:
            collection.release()
        except:
            pass
            
        # 处理结果中的NumPy类型
        processed_results = []
        for item in results:
            processed_item = {}
            for key, value in item.items():
                # 处理NumPy类型
                if isinstance(value, np.ndarray):
                    if key.endswith('_vector') or len(value) > 20:  # 向量字段或大数组
                        processed_item[key] = {
                            "type": "vector",
                            "dim": len(value)
                        }
                    else:
                        processed_item[key] = value.tolist()
                elif isinstance(value, np.integer):
                    processed_item[key] = int(value)
                elif isinstance(value, np.floating):
                    processed_item[key] = float(value)
                else:
                    processed_item[key] = value
            processed_results.append(processed_item)
            
        response_data = {
            'status': 'success',
            'data': processed_results,
            'count': len(processed_results),
            'total_count': total_count,
            'total_pages': total_pages,
            'current_page': page,
            'page_size': page_size
        }
        
        print(f"返回分页数据: 当前页={page}, 总页数={total_pages}, 总记录数={total_count}")
        return jsonify(response_data)
    
    except Exception as e:
        print(f"获取集合数据时发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/uploads/<filename>')
def download_file(filename):
    """下载生成的结果文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def connect_to_milvus():
    """连接到Milvus服务器"""
    try:
        # 检查是否已连接
        try:
            if utility.has_collection("_dummy_check_"):
                return True  # 已连接
        except:
            pass  # 继续尝试连接
            
        # 重新连接
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"成功连接到Milvus服务器: {MILVUS_HOST}:{MILVUS_PORT}")
        return True
    except Exception as e:
        print(f"Milvus连接失败: {e}")
        return False

def get_embedding_model(required_dim):
    """根据需要的维度返回合适的嵌入模型"""
    if required_dim == 384:
        return SentenceTransformer('all-MiniLM-L6-v2')
    elif required_dim == 768:
        return SentenceTransformer('all-MiniLM-L6-v2')
    else:
        # 默认使用384维模型，然后进行适当的降维
        return SentenceTransformer('all-MiniLM-L6-v2')

def adapt_vector_dimension(vector, target_dim, method='pca'):
    """调整向量维度的高级方法"""
    if len(vector) == target_dim:
        return vector
    
    if len(vector) < target_dim:
        # 从低维到高维，使用更智能的上采样
        if method == 'repeat':
            # 重复向量元素并截断
            repeat_factor = math.ceil(target_dim / len(vector))
            extended = np.tile(vector, repeat_factor)[:target_dim]
            return extended
        elif method == 'interpolation':
            # 使用插值方法
            x = np.linspace(0, 1, len(vector))
            x_new = np.linspace(0, 1, target_dim)
            return np.interp(x_new, x, vector)
    else:
        # 从高维到低维，使用降维方法
        if method == 'pca':
            # 使用PCA降维
            pca = PCA(n_components=target_dim)
            return pca.fit_transform([vector])[0]
        elif method == 'truncate':
            # 简单截断
            return vector[:target_dim]
    
    return vector  # 默认返回原向量

def prepare_vector_for_metric(vector, metric_type):
    """根据度量类型准备向量"""
    if metric_type.upper() == 'COSINE':
        # COSINE度量需要归一化向量
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
    # L2度量不需要特殊处理
    return vector

# 添加计算平均分数的辅助函数
def calculate_average_scores(results):
    """计算所有测试结果的平均分数"""
    if not results:
        return {}
    
    # 检查是否使用多模型
    use_multi_model = results[0].get('multi_model', False) if results else False
    
    if use_multi_model:
        # 多模型情况下，为每个模型单独计算平均分
        model_metrics = {}
        
        # 初始化模型列表
        for result in results:
            for model_name in result.get('model_metrics', {}).keys():
                if model_name not in model_metrics:
                    model_metrics[model_name] = {
                        'count': 0,
                        'avg_rouge_1': 0,
                        'avg_rouge_2': 0,
                        'avg_rouge_l': 0,
                        'avg_keyword_match': 0,
                        'avg_precision_1': 0,
                        'avg_recall_1': 0,
                        'avg_precision_l': 0,
                        'avg_recall_l': 0,
                        'avg_keyword_precision': 0,
                        'avg_keyword_recall': 0
                    }
        
        # 累加每个模型的指标
        for result in results:
            for model_name, metrics in result.get('model_metrics', {}).items():
                model_metrics[model_name]['count'] += 1
                
                for metric_name, value in metrics.items():
                    avg_key = f'avg_{metric_name}'
                    if avg_key in model_metrics[model_name]:
                        model_metrics[model_name][avg_key] += value
        
        # 计算每个模型的平均值
        for model_name, metrics in model_metrics.items():
            count = metrics['count']
            if count > 0:
                for key in list(metrics.keys()):
                    if key != 'count':
                        metrics[key] = metrics[key] / count
        
        return model_metrics
    else:
        # 单一模型情况
        count = len(results)
        avg_metrics = {
            'avg_rouge_1': 0,
            'avg_rouge_2': 0,
            'avg_rouge_l': 0,
            'avg_keyword_match': 0,
            'avg_precision_1': 0,
            'avg_recall_1': 0,
            'avg_precision_l': 0,
            'avg_recall_l': 0,
            'avg_keyword_precision': 0,
            'avg_keyword_recall': 0
        }
        
        for result in results:
            for metric_name, value in result.get('metrics', {}).items():
                avg_key = f'avg_{metric_name}'
                if avg_key in avg_metrics:
                    avg_metrics[avg_key] += value
        
        # 计算平均值
        if count > 0:
            for key in avg_metrics:
                avg_metrics[key] = avg_metrics[key] / count
        
        return avg_metrics

def normalize_sources_list(sources):
    """规范化数据来源列表，用于统一显示"""
    if not sources:
        return []
    
    normalized_sources = []
    unique_sources = set()  # 用于去重
    
    for source in sources:
        normalized = normalize_source_name(source)
        if normalized and normalized not in unique_sources:
            normalized_sources.append(normalized)
            unique_sources.add(normalized)
            
    return normalized_sources

def save_results_to_csv(results, filepath):
    """将测试结果保存到CSV文件"""
    if not results:
        return
    
    # 检查是否为多模型
    use_multi_model = results[0].get('multi_model', False) if results else False
    
    result_rows = []
    if use_multi_model:
        # 多模型结果
        for item in results:
            base_row = {
                'Question': item['question'],
                'Reference': item['reference'],
                'Retrieval_Time': item['retrieval_time'],
                'Sources': ', '.join([str(src) for src in normalize_sources_list(item.get('sources', []))]),  # 规范化数据来源
                'Context': item.get('context', '')[:1000]  # 限制上下文长度
            }
            
            # 为每个模型添加一行
            for model_name, answer in item.get('model_answers', {}).items():
                row = base_row.copy()
                row['Model'] = model_name
                row['Generated'] = answer
                
                # 添加该模型的评估指标
                metrics = item.get('model_metrics', {}).get(model_name, {})
                for metric_name, value in metrics.items():
                    row[metric_name] = value
                
                result_rows.append(row)
    else:
        # 单一模型结果
        for item in results:
            row = {
                'Question': item['question'],
                'Reference': item['reference'],
                'Generated': item['generated'],
                'Retrieval_Time': item['retrieval_time'],
                'Model': item.get('model', models[0] if 'models' in locals() else "deepseek-chat"),
                'Sources': ', '.join([str(src) for src in normalize_sources_list(item.get('sources', []))]),  # 规范化数据来源
                'Context': item.get('context', '')[:1000]  # 限制上下文长度
            }
            
            # 添加所有指标
            for metric_name, value in item.get('metrics', {}).items():
                row[metric_name] = value
            
            result_rows.append(row)
    
    # 创建DataFrame并保存
    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(filepath, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    # 配置日志
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    app.logger.setLevel(logging.DEBUG)
    
    # 提供更详细的错误信息
    app.config['PROPAGATE_EXCEPTIONS'] = True
    
    # 运行应用程序，启用调试模式
    app.run(debug=True, host='0.0.0.0', port=5555)