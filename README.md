# MTF-Transformer
## 环境
[multi-view.tar.gz](https://pan.baidu.com/s/134vlOJmFKJSH7tiATfA6BQ?pwd=ise7)
```bash
cp multi-view.tar.gz ${path}/anaconda3/envs
cd ${path}/anaconda3/envs
mkdir -p multi-view
tar -xzf multi-view.tar.gz -C multi-view
conda activate multi-view
```
## 数据
[H36M_data 提取码：i6dd ](https://pan.baidu.com/s/1Wu6XEEuAtQLpttIAYQaE4Q?pwd=i6dd)
## 预训练模型
[H36M_checkpoint 提取码：i6dd ](https://pan.baidu.com/s/1Wu6XEEuAtQLpttIAYQaE4Q?pwd=i6dd)
## 训练
### H36M(有相机参数)
```bash
Ours(T=7): python run_h36m.py --cfg ./cfg/submit/t_7_dim_4.yaml --gpu 2 3
```
### H36M(无相机参数)
```bash
Ours(T=7):python run_h36m.py --cfg ./cfg/submit/t_7_dim_4.yaml --gpu 2 3

Ours(T=27):python run_h36m.py --cfg ./cfg/submit/t_27_dim_4.yaml --gpu 2 3
```

## 测试
### H36M(有相机参数)
```bash
Ours(T=7): python run_h36m.py --cfg ./cfg/submit/gt_trans_t_7_no_res.yaml --eval --checkpoint ./checkpoint/submit/gt_trans_t_7_no_res_2022-02-27-02-21/model.bin --gpu 2 3 --eval_n_frames 1 --eval_n_views 4 --eval_batch_size 500 --n_frames 7

Triangulate: python run_h36m.py --triangulate --eval
```
### H36M(无相机参数)
```bash
Ours(T=7): python run_h36m.py --cfg ./cfg/submit/t_7_dim_4.yaml --eval --checkpoint ./checkpoint/submit/t_7_dim_4_2022-03-24-17-56/model.bin --gpu 2 3 --eval_n_frames 1 --eval_n_views 4 --eval_batch_size 500 --n_frames 7

Ours(T=27)python run_h36m.py --cfg ./cfg/submit/t_27_dim_4.yaml --eval --checkpoint ./checkpoint/submit/t_27_dim_4_2022-03-20-21-16/model.bin --gpu 2 3 --eval_n_frames 1 --eval_n_views 4 --eval_batch_size 500 --n_frames 27
```
