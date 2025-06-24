import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from options import parse_args
from data import *
from tqdm import tqdm
from utils_our.compute_fisher_diag import compute_fisher_diag
from tqdm.auto import trange, tqdm
import copy
import sys
import random
from net import get_model
from torch.optim import Optimizer
import datetime
import tensorflow as tf
import sys
import tensorflow_privacy as tfp
import numpy as np
from tensorflow_privacy.privacy.dp_query.quantile_adaptive_clip_sum_query import QuantileAdaptiveClipSumQuery
from tensorflow_privacy.privacy.dp_query.normalized_query import NormalizedQuery
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from typing import Any, NamedTuple, Optional
import pandas as pd
import logging
from datetime import datetime
import warnings
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from options import parse_args
from data import *
from net import *
from tqdm import tqdm
from tqdm.auto import trange, tqdm
import copy
import sys
import random
from torch.optim import Optimizer
import datetime
import tensorflow as tf
import sys
import tensorflow_privacy as tfp
import numpy as np
from typing import Any, NamedTuple, Optional
import pandas as pd
import logging
from datetime import datetime
import warnings

try:
    from opacus.accountants import RDPAccountant
except ImportError:
    # 如果没有安装 Opacus，设置一个替代类
    class RDPAccountant:
        def __init__(self):
            print("警告: Opacus未安装，使用简化版隐私计算")
            self.steps = 0
            self.noise = 0
            
        def step(self, noise_multiplier, sample_rate):
            self.steps += 1
            self.noise = noise_multiplier
            
        def get_epsilon(self, delta=1e-5):
            # 简化版隐私计算，仅用于测试
            return (self.steps ** 0.5) * (1.0/self.noise) * np.sqrt(2 * np.log(1/delta))

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
num_clients = args.num_clients
local_epoch = args.local_epoch
global_epoch = args.global_epoch
batch_size = args.batch_size
target_epsilon = args.target_epsilon
target_delta = args.target_delta
clipping_bound = args.clipping_bound
dataset = args.dataset
user_sample_rate = args.user_sample_rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cumulative_epsilon = 0

def setup_logging():
    # 创建带有时间戳的唯一日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./logs/training_log_{dataset}_{timestamp}.log"
    
    # 同时创建一个纯文本文件用于记录所有输出
    txt_filename = f"./outputs/terminal_output_{dataset}_{timestamp}.txt"
    
    # Excel文件名
    excel_filename = f"./outputs/training_results_{dataset}_{timestamp}.xlsx"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    os.makedirs(os.path.dirname(txt_filename), exist_ok=True)
    os.makedirs(os.path.dirname(excel_filename), exist_ok=True)
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logging.info(f"=== 开始训练 ===")
    logging.info(f"数据集: {dataset}")
    logging.info(f"客户端数量: {num_clients}")
    logging.info(f"本地训练轮次: {local_epoch}")
    logging.info(f"全局聚合轮次: {global_epoch}")
    logging.info(f"批量大小: {batch_size}")
    logging.info(f"目标隐私预算 (ε): {target_epsilon}")
    logging.info(f"目标隐私预算 (δ): {target_delta}")
    logging.info(f"初始裁剪阈值: {clipping_bound}")
    logging.info(f"最大裁剪阈值: {args.max_clip_threshold if args.max_clip_threshold > 0 else '无限制'}")
    logging.info(f"Fisher阈值: {args.fisher_threshold}")
    logging.info(f"噪声乘数: {args.noise_multiplier}")
    
    return log_filename, txt_filename, excel_filename

if args.store == True:
    saved_stdout = sys.stdout
    file = open(
        f'./txt/{args.dirStr}/'
        f'dataset {dataset} '
        f'--num_clients {num_clients} '
        f'--local_epoch {local_epoch} '
        f'--global_epoch {global_epoch} '
        f'--batch_size {batch_size} '
        f'--target_epsilon {target_epsilon} '
        f'--target_delta {target_delta} '
        f'--clipping_bound {clipping_bound} '
        f'--fisher_threshold {args.fisher_threshold} '
        f'--lambda_1 {args.lambda_1} '
        f'--lambda_2 {args.lambda_2} '
        f'--lr {args.lr} '
        f'--alpha {args.dir_alpha}'
        f'.txt'
        ,'a'
        )
    sys.stdout = file

def tf_tensor_to_torch(tensors):
    if isinstance(tensors, list):
        return [torch.tensor(t.numpy()) for t in tensors]
    else:
        return torch.tensor(tensors.numpy())

def torch_tensor_to_tf(tensors):
    if isinstance(tensors, list):
        return [tf.convert_to_tensor(t.detach().cpu().numpy(), dtype=tf.float32) for t in tensors]
    else:
        return tf.convert_to_tensor(tensors.detach().cpu().numpy(), dtype=tf.float32)
        
def get_epsilon(
    num_examples,
    batch_size,
    noise_multiplier,
    epochs,
    delta=1e-2
) -> float:
    """返回当前训练设置下的 epsilon 值，使用 Opacus RDP 会计"""
    try:
        from opacus.accountants import RDPAccountant
    
        # 计算采样概率和步数
        sampling_probability = batch_size / num_examples
        steps = int(epochs * num_examples / batch_size)
    
        # 初始化 RDP 会计
        accountant = RDPAccountant()
        

        for _ in range(steps):
                accountant.step(
                    noise_multiplier=noise_multiplier,
                    sample_rate=sampling_probability
                )
    
        # 获取当前 epsilon 值
        epsilon = accountant.get_epsilon(delta=delta)
        return epsilon
        
    except Exception as e:
        # 如果 Opacus 出错，回退到 TensorFlow Privacy
        print(f"Opacus 计算出错: {e}，尝试使用 TensorFlow Privacy...")
        from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
        
        # 计算每个 epoch 中的迭代次数
        steps = epochs * num_examples // batch_size
        
        # 直接使用 compute_dp_sgd_privacy 计算 epsilon 值
        epsilon, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
	    n=num_examples,
	    batch_size=batch_size,
	    noise_multiplier=noise_multiplier,
	    epochs=epochs,
	    delta=delta
	)
        return epsilon
        
    except Exception as e:
        # 如果 Opacus 出错，回退到 TensorFlow Privacy
        print(f"Opacus 计算出错: {e}，尝试使用 TensorFlow Privacy...")
        from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
        
        # 计算每个 epoch 中的迭代次数
        steps = epochs * num_examples // batch_size
        
        # 直接使用 compute_dp_sgd_privacy 计算 epsilon 值
        epsilon, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
            n=num_examples,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            epochs=epochs,
            delta=delta
        )
        
        return epsilon
    
def adaptive_clip_noise_params(
    noise_multiplier: float,
    expected_clients_per_round: float,
    clipped_count_stddev: Optional[float] = None,
) -> tuple[float, float]:
  if noise_multiplier > 0.0:
    if clipped_count_stddev is None:
      clipped_count_stddev = 0.05 * expected_clients_per_round

    if noise_multiplier >= 2 * clipped_count_stddev:
      raise ValueError(
          f'clipped_count_stddev = {clipped_count_stddev} (defaults to '
          '0.05 * `expected_clients_per_round` if not specified) is too low '
          'to achieve the desired effective `noise_multiplier` '
          f'({noise_multiplier}). You must either increase '
          '`clipped_count_stddev` or decrease `noise_multiplier`.'
      )

    value_noise_multiplier = (
        noise_multiplier**-2 - (2 * clipped_count_stddev) ** -2
    ) ** -0.5

    added_noise_factor = value_noise_multiplier / noise_multiplier
    if added_noise_factor >= 2:
      warnings.warn(
          f'A significant amount of noise ({added_noise_factor:.2f}x) has to '
          'be added for record aggregation to achieve the desired effective '
          f'`noise_multiplier` ({noise_multiplier}). If you are manually '
          'specifying `clipped_count_stddev` you may want to increase it. Or '
          'you may need more `expected_clients_per_round`.'
      )
  else:
    if clipped_count_stddev is None:
      clipped_count_stddev = 0.0
    value_noise_multiplier = 0.0

  return value_noise_multiplier, clipped_count_stddev

def local_update(model, dataloader, global_model, clip=None, value_noise_multiplier=None,clipped_count_stddev=None):


    fisher_threshold = args.fisher_threshold
    model = model.to(device)
    global_model = global_model.to(device)

    w_glob = [param.clone().detach() for param in global_model.parameters()]

    fisher_diag = compute_fisher_diag(model, dataloader)


    u_loc, v_loc = [], []
    for param, fisher_value in zip(model.parameters(), fisher_diag):
        u_param = (param * (fisher_value > fisher_threshold)).clone().detach()
        v_param = (param * (fisher_value <= fisher_threshold)).clone().detach()
        u_loc.append(u_param)
        v_loc.append(v_param)

    u_glob, v_glob = [], []
    for global_param, fisher_value in zip(global_model.parameters(), fisher_diag):
        u_param = (global_param * (fisher_value > fisher_threshold)).clone().detach()
        v_param = (global_param * (fisher_value <= fisher_threshold)).clone().detach()
        u_glob.append(u_param)
        v_glob.append(v_param)

    for u_param, v_param, model_param in zip(u_loc, v_glob, model.parameters()):
        model_param.data = u_param + v_param

    saved_u_loc = [u.clone() for u in u_loc]

    def custom_loss(outputs, labels, param_diffs, reg_type):
        ce_loss = F.cross_entropy(outputs, labels)
        if reg_type == "R1":
            reg_loss = (args.lambda_1 / 2) * torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))

        elif reg_type == "R2":
            C = args.clipping_bound
            norm_diff = torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))
            reg_loss = (args.lambda_2 / 2) * torch.norm(norm_diff - C)

        else:
            raise ValueError("Invalid regularization type")

        return ce_loss + reg_loss
    

    optimizer1 = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.local_epoch):
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer1.zero_grad()
            outputs = model(data)
            param_diffs = [u_new - u_old for u_new, u_old in zip(model.parameters(), w_glob)]
            loss = custom_loss(outputs, labels, param_diffs, "R1")
            loss.backward()
            with torch.no_grad():
                for model_param, u_param in zip(model.parameters(), u_loc):
                    model_param.grad *= (u_param != 0)
            optimizer1.step()
    optimizer2 = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.local_epoch):
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer2.zero_grad()
            outputs = model(data)
            param_diffs = [model_param - w_old for model_param, w_old in zip(model.parameters(), w_glob)]
            loss = custom_loss(outputs, labels, param_diffs, "R2")
            loss.backward()
            with torch.no_grad():
                for model_param, v_param in zip(model.parameters(), v_glob):
                    model_param.grad *= (v_param != 0)
            optimizer2.step()

    with torch.no_grad():
        update = [(new_param - old_param).clone() for new_param, old_param in zip(model.parameters(), w_glob)]

    #### 客户端加噪处理
    if clip is not None and value_noise_multiplier is not None and clipped_count_stddev is not None:
        # 将更新转换为TensorFlow格式并进行裁剪和加噪
        tf_update = torch_tensor_to_tf(update)
        
        # 创建查询对象进行裁剪和加噪
        query = QuantileAdaptiveClipSumQuery(
            initial_l2_norm_clip=clip,
            noise_multiplier=value_noise_multiplier,
            target_unclipped_quantile=args.target_unclipped_quantile,
            learning_rate=args.adaptive_clip_learning_rate,
            clipped_count_stddev=clipped_count_stddev,
            expected_num_records=1,  # 单个客户端
            geometric_update=True)
        
        # 初始化状态
        global_state = query.initial_global_state()
        sample_params = query.derive_sample_params(global_state)
        sample_state = query.initial_sample_state(tf_update)
        
        # 预处理并加噪
        record = query.preprocess_record(sample_params, tf_update)
        sample_state = query.accumulate_preprocessed_record(sample_state, record)
        noised_result, _, _ = query.get_noised_result(sample_state, global_state)
        
        # 转换回PyTorch格式
        update = tf_tensor_to_torch(noised_result)

    model = model.to('cpu')
    return update

def test(client_model, client_testloader):
    client_model.eval()
    client_model = client_model.to(device)

    num_data = 0


    correct = 0
    with torch.no_grad():
        for data, labels in client_testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = client_model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            num_data += labels.size(0)
    
    accuracy = 100.0 * correct / num_data

    client_model = client_model.to('cpu')

    return accuracy

output_dir = f"./output_txt_{args.target_epsilon}_{args.dir_alpha}"
def save_round_info(round_data, epoch, client_data, clients_accuracies, epsilon, clip, value_noise_multiplier, 
                   sample_client_num, N):
    
    # 在函数内部定义output_dir，不再作为参数传入
    output_dir = f"./output_txt_{args.target_epsilon}"
    
    # 只有在第一次（epoch == 0）时创建文件夹
    if epoch == 0:
        os.makedirs(output_dir, exist_ok=True)  # 创建文件夹
    details_file_path = f"{output_dir}/training_details.txt"
    
    # 写入文件
    file_mode = 'w' if epoch == 0 else 'a'  # 第一次写入文件头，后续轮次追加内容
    with open(details_file_path, file_mode) as f:
        if epoch == 0:
            f.write("客户端详情:\n")
            f.write("轮次 | 平均准确率 | 客户端准确率 | 累计隐私预算 (ε) | 裁剪阈值 | 噪声乘子 | 参与客户端数量 | 参与样本总量\n")
            f.write("="*80 + "\n")
        
        f.write(f"第{epoch+1}轮:\n")
        f.write(f"平均准确率: {sum(clients_accuracies)/len(clients_accuracies):.2f}%\n")
        f.write(f"客户端准确率: {clients_accuracies}\n")
        f.write(f"累计隐私预算 (ε): {epsilon:.4f}\n")
        f.write(f"目标隐私预算: {target_epsilon:.4f}\n")
        f.write(f"当前裁剪阈值: {clip:.4f}\n")
        f.write(f"噪声乘子: {value_noise_multiplier:.4f}\n")
        f.write(f"参与客户端数量: {sample_client_num}/{num_clients}\n")
        f.write(f"参与样本总量: {N}\n")
        f.write("="*80 + "\n")
    
    # 将每一轮的统计数据存储到 round_data 中
    round_data["轮次"].append(epoch + 1)
    round_data["平均准确率"].append(sum(clients_accuracies)/len(clients_accuracies))
    round_data["客户端准确率"].append(clients_accuracies)
    round_data["累计隐私预算"].append(epsilon)
    round_data["裁剪阈值"].append(clip)
    round_data["噪声乘子"].append(value_noise_multiplier)
    round_data["参与客户端数量"].append(sample_client_num)
    round_data["参与样本总量"].append(N)
    
    return round_data


# 准备配置数据的函数
def prepare_config_data():
    config_data = {
        "参数名称": [
            "数据集", "客户端数量", "本地训练轮次", "全局聚合轮次",
            "批量大小", "目标隐私预算(ε)", "目标隐私预算(δ)",
            "初始裁剪阈值", "最大裁剪阈值", "Fisher阈值", "噪声乘数",
            "用户采样率", "学习率", "lambda_1", "lambda_2",
            "目标未裁剪分位数", "自适应裁剪学习率", "Dirichlet alpha"
        ],
        "参数值": [
            dataset, num_clients, local_epoch, global_epoch,
            batch_size, target_epsilon, target_delta,
            clipping_bound, args.max_clip_threshold, args.fisher_threshold, args.noise_multiplier,
            args.user_sample_rate, args.lr, args.lambda_1, args.lambda_2,
            args.target_unclipped_quantile, args.adaptive_clip_learning_rate, args.dir_alpha
        ]
    }
    return pd.DataFrame(config_data)


# 将训练数据保存到Excel
def save_to_excel(config_df, rounds_data, client_acc_matrix, mean_acc_list, excel_filename):
    # 创建ExcelWriter对象
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # 保存配置信息
        config_df.to_excel(writer, sheet_name='配置参数', index=False)
        
        # 保存每轮训练摘要信息
        summary_df = pd.DataFrame({
            "轮次": rounds_data["轮次"],
            "平均准确率": rounds_data["平均准确率"],
            "累计隐私预算": rounds_data["累计隐私预算"],
            "裁剪阈值": rounds_data["裁剪阈值"],
            "噪声乘子": rounds_data["噪声乘子"],
            "参与客户端数量": rounds_data["参与客户端数量"],
            "参与样本总量": rounds_data["参与样本总量"]
        })
        summary_df.to_excel(writer, sheet_name='训练摘要', index=False)
        
        # 保存客户端准确率矩阵
        client_acc_df = pd.DataFrame(client_acc_matrix)
        client_acc_df.columns = [f"客户端{i+1}" for i in range(client_acc_df.shape[1])]
        client_acc_df.index = [f"轮次{i+1}" for i in range(client_acc_df.shape[0])]
        client_acc_df.to_excel(writer, sheet_name='客户端准确率矩阵')
        
        # 保存平均准确率曲线数据
        mean_acc_df = pd.DataFrame({"轮次": list(range(1, len(mean_acc_list)+1)), "平均准确率": mean_acc_list})
        mean_acc_df.to_excel(writer, sheet_name='平均准确率曲线', index=False)
        
        # 为每个轮次创建详细数据表
        for i, round_num in enumerate(rounds_data["轮次"]):
            # 创建该轮次的客户端详细数据
            clients_data = []
            for j, acc in enumerate(rounds_data["客户端准确率"][i]):
                clients_data.append({
                    "客户端ID": j+1,
                    "准确率(%)": acc
                })
            round_detail_df = pd.DataFrame(clients_data)
            round_detail_df.to_excel(writer, sheet_name=f'轮次{round_num}详情', index=False)

###--------主函数-------####
def main():
    # 设置日志
    log_filename, txt_filename, excel_filename = setup_logging()
    
    # 创建一个记录所有输出的文件
    output_file = open(txt_filename, 'a')
    
    # 保存原始的stdout和stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # 重定向stdout和stderr到文件
    sys.stdout = output_file
    sys.stderr = output_file
    
    try:
        logging.info("初始化模型和数据加载器...")
        mean_acc_s = []
        acc_matrix = []
        
        # 初始化轮次数据收集字典
        rounds_data = {
            "轮次": [],
            "平均准确率": [],
            "客户端准确率": [],
            "累计隐私预算": [],
            "裁剪阈值": [],
            "噪声乘子": [],
            "参与客户端数量": [],
            "参与样本总量": []
        }
        
        # 准备配置数据
        config_df = prepare_config_data()
        
        # 原有的模型初始化和训练代码保持不变
        if dataset == 'MNIST':
            train_dataset, test_dataset = get_mnist_datasets()
            clients_train_set = get_clients_datasets(train_dataset, num_clients)
            client_data_sizes = [len(client_dataset) for client_dataset in clients_train_set]
            clients_train_loaders = [DataLoader(client_dataset, batch_size=batch_size) for client_dataset in clients_train_set]
            clients_test_loaders = [DataLoader(test_dataset) for _ in range(num_clients)]
            num_classes = 10
            model_input_channels = 1
        elif dataset == 'CIFAR10':
            clients_train_loaders, clients_test_loaders, client_data_sizes = get_CIFAR10(args.dir_alpha, num_clients)
            num_classes = 10
            model_input_channels = 3
        elif dataset == 'FEMNIST':
            clients_train_loaders, clients_test_loaders, client_data_sizes = get_FEMNIST(num_clients)
            num_classes = 62
            model_input_channels = 1
        elif dataset == 'SVHN':
            clients_train_loaders, clients_test_loaders, client_data_sizes = get_SVHN(args.dir_alpha, num_clients)
            num_classes = 10
            model_input_channels = 3
        else:
            print('Undefined dataset')
            assert 1 == 0
    
        # 统一初始化模型
        clients_models = [get_model(args, num_classes=num_classes, model_input_channels=model_input_channels) for _ in range(num_clients)]
        global_model = get_model(args, num_classes=num_classes, model_input_channels=model_input_channels)
    

        for client_model in clients_models:
            client_model.load_state_dict(global_model.state_dict())
        clip=args.clipping_bound
        value_noise_multiplier = None
        clipped_count_stddev = None

        # 在 main() 函数开头初始化
        accountant = RDPAccountant()
        cumulative_epsilon = 0

        for epoch in trange(global_epoch):
            sampled_client_indices = random.sample(range(num_clients), max(1, int(user_sample_rate * num_clients)))
            sampled_clients_models = [clients_models[i] for i in sampled_client_indices]
            sampled_clients_train_loaders = [clients_train_loaders[i] for i in sampled_client_indices]
            sampled_clients_test_loaders = [clients_test_loaders[i] for i in sampled_client_indices]
            clients_model_updates = []
            clients_accuracies = []
            
            ####------- client Train-------#####
            # 先计算参与的客户端数量和样本总量
            sample_client_num = len(sampled_client_indices)
            N = sum([client_data_sizes[i] for i in sampled_client_indices])
            
            # 在第一轮初始化噪声参数
            if epoch == 0:
                value_noise_multiplier, clipped_count_stddev = adaptive_clip_noise_params(
                    noise_multiplier=args.noise_multiplier,
                    expected_clients_per_round=sample_client_num,
                    clipped_count_stddev=None)
                
            for idx, (client_model, client_trainloader, client_testloader) in enumerate(zip(sampled_clients_models, sampled_clients_train_loaders, sampled_clients_test_loaders)):
                if not args.store:
                    tqdm.write(f'client:{idx+1}/{args.num_clients}')
                # 传递加噪参数给客户端
                client_update = local_update(client_model, client_trainloader, global_model, 
                                           clip=clip, value_noise_multiplier=value_noise_multiplier, clipped_count_stddev=clipped_count_stddev)
                clients_model_updates.append(client_update)
                accuracy = test(client_model, client_testloader)
                clients_accuracies.append(accuracy)
            logging.info(f"客户端准确率: {clients_accuracies}")
            mean_acc_s.append(sum(clients_accuracies)/len(clients_accuracies))
            acc_matrix.append(clients_accuracies)
            
            ####------- Server Aggregator-------#####
            logging.info("开始服务器端聚合...")
            query_global = QuantileAdaptiveClipSumQuery(
                    initial_l2_norm_clip=clip,
                    noise_multiplier=0.0,  # 服务器端不添加噪声
                    target_unclipped_quantile=args.target_unclipped_quantile,
                    learning_rate=args.adaptive_clip_learning_rate,
                    clipped_count_stddev=0.0, 
                    expected_num_records=sample_client_num,
                    geometric_update=True)
            query_global = NormalizedQuery(query_global, denominator=sample_client_num)
            # 初始化全局状态（第一次聚合时）
            global_state = query_global.initial_global_state()
            sample_params = query_global.derive_sample_params(global_state) # 获取采样参数
            # 初始化样本状态（累加器）
            example_update = torch_tensor_to_tf(clients_model_updates[0])  # 获取一个样例更新
            sample_state = query_global.initial_sample_state(example_update)  # 使用实际结构生成状态
            tf_clients_model_updates = []
            for update in clients_model_updates:
                tf_update = torch_tensor_to_tf(update)  # update 是 List[Tensor]
                tf_clients_model_updates.append(tf_update)
            # 预处理并累加所有客户端更新
            for update in tf_clients_model_updates:
                record = query_global.preprocess_record(sample_params, update)
                sample_state = query_global.accumulate_preprocessed_record(sample_state, record)
            # 聚合并加噪输出
            result, new_global_state, _ = query_global.get_noised_result(sample_state, global_state)
            clip = new_global_state.numerator_state.sum_state.l2_norm_clip.numpy()

            # 检查是否超过最大裁剪阈值
            if args.max_clip_threshold > 0 and clip > args.max_clip_threshold:
                logging.info(f"裁剪阈值 {clip:.4f} 超过最大限制，将其限制为 {args.max_clip_threshold}")
                # 仅更新clip变量，不尝试修改state
                clip = args.max_clip_threshold
            
            # # 直接对已加噪的客户端更新进行平均
            # with torch.no_grad():
            #     # 计算平均更新
            #     avg_update = []
            #     for i in range(len(clients_model_updates[0])):  # 遍历每个参数层
            #         layer_updates = torch.stack([update[i] for update in clients_model_updates])
            #         avg_layer_update = torch.mean(layer_updates, dim=0)
            #         avg_update.append(avg_layer_update)
                
            #     # 将平均更新应用到全局模型
            #     avg_update = [update.to(device) for update in avg_update]
            #     for global_param, update in zip(global_model.parameters(), avg_update):
            #         global_param.add_(update)
            
            # # 更新裁剪阈值（简化版本，可以根据需要调整）
            # if epoch > 0:
            #     # 简单的裁剪阈值更新策略
            #     update_norms = [torch.norm(torch.cat([u.flatten() for u in update])).item() 
            #                    for update in clients_model_updates]
            #     avg_norm = sum(update_norms) / len(update_norms)
            #     clip = clip * 0.95 + avg_norm * 0.05  # 简单的移动平均更新
                
            #     # 检查是否超过最大裁剪阈值
            #     if args.max_clip_threshold > 0 and clip > args.max_clip_threshold:
            #         logging.info(f"裁剪阈值 {clip:.4f} 超过最大限制，将其限制为 {args.max_clip_threshold}")
            #         clip = args.max_clip_threshold
            
            logging.info(f"\n更新后的裁剪范数: {clip}")
            pytorch_update = [tf_tensor_to_torch(t) for t in result]
            pytorch_update = [update.to(device) for update in pytorch_update]  # 确保在同一设备
            # Apply the aggregated updates to the global model parameters
            with torch.no_grad():
                for global_param, update in zip(global_model.parameters(), pytorch_update):
                    global_param.add_(update)
            logging.info('Z_delta是：')    
            logging.info(value_noise_multiplier)
            logging.info('标准差是：')    
            logging.info(clipped_count_stddev)
            epsilon = get_epsilon(
                num_examples=N,
                batch_size=batch_size,
                noise_multiplier=value_noise_multiplier,
                epochs=args.local_epoch,
                delta=args.target_delta
            )
            logging.info(f"第{epoch}轮累计使用到的隐私预算为 ε ≈ {epsilon:.4f}")
            # 创建详细输出
            logging.info("\n" + "="*80)
            logging.info(f"第 {epoch+1}/{global_epoch} 轮训练详情:")
            logging.info("-"*80)
            
            # 1. 客户端准确率表格
            client_data = {
                "客户端ID": [f"客户端 {sampled_client_indices[i]}" for i in range(len(clients_accuracies))],
                "准确率 (%)": [f"{acc:.2f}" for acc in clients_accuracies],
                "训练样本数": [client_data_sizes[i] for i in sampled_client_indices]
            }
            client_df = pd.DataFrame(client_data)
            logging.info("\n客户端详情:")
            logging.info(client_df)
            
            # 2. 隐私和聚合参数
            logging.info("\n隐私与聚合参数:")
            logging.info(f"平均准确率: {sum(clients_accuracies)/len(clients_accuracies):.2f}%")
            logging.info(f"累计隐私预算 (ε): {epsilon:.4f}")
            logging.info(f"目标隐私预算: {target_epsilon:.4f}")
            logging.info(f"当前裁剪阈值: {clip:.4f}")
            logging.info(f"噪声乘子: {value_noise_multiplier:.4f}")
            logging.info(f"参与客户端数量: {sample_client_num}/{num_clients}")
            logging.info(f"参与样本总量: {N}")
            logging.info("="*80)
            
            # 再检查是否达到隐私预算上限
            if epsilon >= args.target_epsilon:
                logging.info(f"已达到隐私预算 ε={epsilon:.4f}，停止训练。")
                torch.save(global_model.state_dict(), f"early_stop_model_e{epoch}.pt")
                break

            # 在每轮训练结束时调用
            rounds_data = save_round_info(
                rounds_data,
                epoch,
                client_data,
                clients_accuracies,
                epsilon,
                clip,
                value_noise_multiplier,
                sample_client_num,
                N
            )

            # 更新隐私消耗（每轮迭代后）
            sampling_probability = batch_size / N
            steps = int(args.local_epoch * N / batch_size)
            
            for _ in range(steps):
                accountant.step(
                    noise_multiplier=value_noise_multiplier,
                    sample_rate=sampling_probability
                )
            
            cumulative_epsilon = accountant.get_epsilon(delta=args.target_delta)
            logging.info(f"第{epoch}轮累计隐私预算为 ε ≈ {cumulative_epsilon:.4f}")
            
            # 检查是否超过阈值
            if cumulative_epsilon >= args.target_epsilon:
                logging.info(f"已达到隐私预算 ε={cumulative_epsilon:.4f}，停止训练。")
                torch.save(global_model.state_dict(), f"early_stop_model_e{epoch}.pt")
                break
                    
        char_set = '1234567890abcdefghijklmnopqrstuvwxyz'
        ID = ''
        for ch in random.sample(char_set, 5):
            ID = f'{ID}{ch}'
        logging.info(
            f'===============================================================\n'
            f'task_ID : {ID}\n'
            f'main_yxy\n'
            f'mean accuracy : \n'
            f'{mean_acc_s}\n'
            f'acc matrix : \n'
            f'{torch.tensor(acc_matrix)}\n'
            f'===============================================================\n'
        )
        logging.info(f"训练完成！日志文件保存在: {log_filename}")
        
        # 训练结束后，保存所有收集的数据到Excel
        save_to_excel(config_df, rounds_data, acc_matrix, mean_acc_s, excel_filename)
        logging.info(f"训练结果已保存到Excel文件: {excel_filename}")
    finally:
        # 恢复原始输出
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logging.info(f"训练完成！所有输出已保存到: {txt_filename}")
        logging.info(f"训练参数和结果已保存到Excel: {excel_filename}")
    

if __name__ == '__main__':
    main()
