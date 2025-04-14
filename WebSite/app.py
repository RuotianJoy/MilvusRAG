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
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename
from pymilvus import connections, utility, Collection, DataType
import sys
import uuid
from datetime import datetime

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

# Milvus连接配置
MILVUS_HOST = '47.115.47.33'  # 默认本地主机
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
            results_data = []
            for _, row in df.iterrows():
                item = {
                    'question': row['Question'],
                    'reference': row['Reference'],
                    'generated': row['Generated'],
                    'retrieval_time': row['Retrieval_Time'],
                    'metrics': {}
                }
                # 添加指标
                for col in df.columns:
                    if col not in ['Question', 'Reference', 'Generated', 'Retrieval_Time']:
                        item['metrics'][col] = row[col]
                results_data.append(item)
                
            return render_template('result.html', 
                                result=result, 
                                results_data=results_data)
        else:
            flash('结果文件不存在', 'warning')
            return redirect(url_for('test_history'))
    except Exception as e:
        app.logger.error(f"读取结果文件失败: {str(e)}")
        flash(f'读取结果文件失败: {str(e)}', 'danger')
        return redirect(url_for('test_history'))

@app.route('/run_test', methods=['POST'])
def run_test():
    """执行测试并返回结果"""
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
        
        app.logger.info(f"测试文件路径: {filepath}, 会话ID: {session_id}, 文件名: {filename}")
        
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
        
        # 对每个问题执行测试
        max_questions = min(len(df), 10)  # 限制处理数量，避免过长时间
        app.logger.info(f"将处理前{max_questions}个问题")
        
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
                
                # 生成回答
                app.logger.info("开始生成回答...")
                generated_answer = RAG_TestingAndEvaluation.generate_answer(user_input, context)
                
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
                    'metrics': metrics
                }
                results_data.append(result_item)
                app.logger.info(f"问题 {idx+1} 处理完成")
            except Exception as question_error:
                app.logger.error(f"处理问题 {idx+1} 时出错: {str(question_error)}")
                import traceback
                app.logger.error(traceback.format_exc())
        
        # 计算平均分数
        avg_scores = {}
        if results_data:
            for metric in results_data[0]['metrics']:
                try:
                    avg_scores[f'avg_{metric}'] = sum(item['metrics'].get(metric, 0) for item in results_data) / len(results_data)
                except:
                    avg_scores[f'avg_{metric}'] = 0
        
        app.logger.info(f"计算得到的平均分数: {avg_scores}")
        
        # 生成测试ID
        test_id = str(uuid.uuid4())
        
        # 保存结果到CSV
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_filename = f"test_results_{test_id}_{timestamp}.csv"
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        
        # 转换为DataFrame并保存
        try:
            result_rows = []
            for item in results_data:
                row = {
                    'Question': item['question'],
                    'Reference': item['reference'],
                    'Generated': item['generated'],
                    'Retrieval_Time': item['retrieval_time']
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
            'test_id': test_id
        })
    
    except Exception as e:
        app.logger.error(f"执行测试时发生未知错误: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
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
            
        response_data = {
            'status': 'success',
            'data': results,
            'count': len(results),
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
    app.run(debug=True, host='0.0.0.0', port=5000) 