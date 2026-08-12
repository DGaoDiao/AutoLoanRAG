# 基于数据训练 将用户查询分为'通用知识'和'专业知识'
# 加载BERT -> 数据预处理 -> 模型训练 -> 模型评估 -> 预测查询类别

import json
import os
from pathlib import Path
import torch
import sys
from base.logger import logger
from base.config import Config
import numpy as np

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class QueryClassifier:
    AUTO_LOAN_TERMS = (
        '车贷', '汽车贷款', '车辆贷款', '购车贷款', '贷款', '首付', '月供',
        '利率', '利息', '年化', '还款', '提前还款', '逾期', '征信', '授信',
        '审批', '放款', '抵押', '解押', '合同', '违约金', '担保费', '服务费',
        '新车', '二手车', '车架号', 'vin', '发动机号', '车辆评估', '购车',
    )
    def __init__(self, model_path = Config().ROOT_DIR/'model'/'bert_query_classifier' ):
        """初始化对象。
                
                params:
                    model_path: 参数说明。
                
                return:
                    无。"""
        self.logger = logger
        self.model_path = model_path
        self.bert_path = Config().ROOT_DIR/'model'/'bert_base_chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'使用设备{self.device}')

        # 定义映射标签 通用知识0 专业知识1
        self.label_map = {'通用知识': 0, '专业咨询': 1}

        self.load_model()

    def load_model(self):
        """如果有分类模型则加载，如果没有就加载一个预训练
                                :return:
                        
                        params:
                            无。
                
                return:
                    函数返回值。"""
        if self.model_path.exists():
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.logger.info(f'加载模型{self.model_path}')
        else:
            self.model = BertForSequenceClassification.from_pretrained(self.bert_path, num_labels=2)
            self.model.to(self.device)
            self.logger.info(f'加载模型{self.bert_path}')

    def save_model(self):
        """执行 save_model 函数。
                
                params:
                    无。
                
                return:
                    函数返回值。"""
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        self.logger.info(f'保存模型至{self.model_path}')

    def preprocess_data(self, texts, labels):
        """:param texts: 待处理文本列表
                                :param labels: 标签
                                :return:
                        
                        params:
                            texts: 参数说明。
                            labels: 参数说明。
                
                return:
                    函数返回值。"""

        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)

        return encodings, [self.label_map[label] for label in labels]

    def create_dataset(self, encodings, labels):

        """执行 create_dataset 函数。
                
                params:
                    encodings: 参数说明。
                    labels: 参数说明。
                
                return:
                    函数返回值。"""
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                """初始化对象。
                                
                                params:
                                    encodings: 参数说明。
                                    labels: 参数说明。
                                
                                return:
                                    无。"""
                self.encodings = encodings
                self.labels = labels
            def __getitem__(self, idx):
                """执行 __getitem__ 函数。
                                
                                params:
                                    idx: 参数说明。
                                
                                return:
                                    函数返回值。"""
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                """执行 __len__ 函数。
                                
                                params:
                                    无。
                                
                                return:
                                    函数返回值。"""
                return len(self.labels)

        return Dataset(encodings, labels)

    def train_model(self, data_file = Config().ROOT_DIR/'rag_qa'/'classify_data'/'model_generic_5000.json'):
        """训练bert分类模型 区分通用知识，和专业查询
                                :param data_file:
                                :return:
                        
                        params:
                            data_file: 参数说明。
                
                return:
                    函数返回值。"""
        if not data_file.exists():
            logger.error(f'数据集文件{data_file}不存在')
            raise FileNotFoundError(f'数据集文件{data_file}不存在')
        with open(data_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]
        texts = [item['query'] for item in data]
        labels = [item['label'] for item in data]

        x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=114514)

        train_encodings, train_labels = self.preprocess_data(x_train, y_train)
        test_encodings, test_labels = self.preprocess_data(x_test, y_test)

        train_dataset = self.create_dataset(train_encodings, train_labels)
        test_dataset = self.create_dataset(test_encodings, test_labels)

        training_args = TrainingArguments(
            output_dir=Config().ROOT_DIR/'training'/'bert_results',
            num_train_epochs = 7,
            per_device_train_batch_size = 8,
            per_device_eval_batch_size = 8,
            warmup_steps = 20,
            weight_decay = 1e-2,
            logging_dir = Config().ROOT_DIR/'training'/'bert_logs',
            logging_steps = 10,
            save_strategy = 'epoch',
            load_best_model_at_end=True,
            evaluation_strategy = 'epoch',
            save_total_limit=1,
            metric_for_best_model='eval_loss',
            fp16 = True,
        )

        trainer = Trainer(
            model = self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics
        )

        logger.info('开始训练模型')
        trainer.train()
        self.save_model()

        self.evaluate_model(x_test, test_labels)

    def compute_metrics(self, eval_pred):
        """计算分类任务的评估
                                :param eval_pred: 包含模型输出logits 和 真实标签
                                :return: 准确率字典
                        
                        params:
                            eval_pred: 参数说明。
                
                return:
                    函数返回值。"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {'accuracy': accuracy}

    def evaluate_model(self, texts, labels):
        """函数功能: 在给定文本和标签上评估模型，输出分类报告和混淆矩阵
                                :param texts: 待评估的文本列表
                                :param labels: 文本对应的真实标签
                                :return:
                        
                        params:
                            texts: 参数说明。
                            labels: 参数说明。
                
                return:
                    函数返回值。"""
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)

        # 创建评估数据集对象
        dataset = self.create_dataset(encodings, labels)

        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)

        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = labels

        logger.info('分类报告')
        logger.info(classification_report(true_labels,
                                          pred_labels,
                                          target_names=['通用知识', '专业咨询']))

        logger.info('混淆矩阵')
        logger.info(confusion_matrix(true_labels, pred_labels))

    # 预测类别
    def predict_category(self, query):
        """对单个查询进行意图识别
                                :param query: 待分类的查询文本
                                :return: 类别名称
                        
                        params:
                            query: 参数说明。
                
                return:
                    函数返回值。"""
        normalized_query = query.lower().strip()
        # The bundled BERT checkpoint was trained for the former education domain.
        # Use an explicit automotive-finance gate so unrelated questions never
        # retrieve loan documents. The model remains available for retraining.
        if any(term in normalized_query for term in self.AUTO_LOAN_TERMS):
            return '专业咨询'
        return '通用知识'


if __name__ == '__main__':
    query_classifier = QueryClassifier()
    res = query_classifier.predict_category('AI的课程大纲是什么')
    print(res)
