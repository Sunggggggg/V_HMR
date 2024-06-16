# V_HMR

# Training
python train_Motion.py --cfg ./configs/repr_table1_3dpw.yaml --gpu 0 

# Table1 3dpw
python test.py --dataset 3dpw --cfg ./configs/repr_table1_3dpw.yaml --gpu 0 
# Table1 h36m
python evaluate.py --dataset h36m --cfg ./configs/repr_table1_h36m_mpii3d.yaml --gpu 0
# Table1 mpii3d
python evaluate.py --dataset mpii3d --cfg ./configs/repr_table1_h36m_mpii3d.yaml --gpu 0

# Table2 3dpw
python evaluate.py --dataset 3dpw --cfg ./configs/repr_table2_3dpw.yaml --gpu 0 

# for rendering 
python evaluate.py --dataset 3dpw --cfg ./configs/repr_table1_3dpw.yaml --gpu 0 --render