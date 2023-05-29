python train.py --name main_no_pretrain --dataset ogbg-molpcba_main --roc_auc --batch_size 64 --num_accumulation_steps 8 --num_epochs 16 --warmup_proportion 0.5 --hidden_dim 1024 --num_heads 16 --num_layers 24 --ema_update_after_step 1000 --amp --device cuda:0
python train.py --name main_no_pretrain --dataset ogbg-molmuv_main --roc_auc --batch_size 128 --num_accumulation_steps 4 --num_epochs 32 --warmup_proportion 0.5 --amp --device cuda:0
python train.py --name main_no_pretrain --dataset ogbg-molhiv_main --batch_size 128 --num_accumulation_steps 4 --num_epochs 32 --warmup_proportion 0.5 --amp --device cuda:0
python train.py --name main_no_pretrain --dataset ogbg-moltoxcast_main --batch_size 64 --num_accumulation_steps 1 --num_epochs 100 --warmup_proportion 0.5 --amp --device cuda:0
python train.py --name main_no_pretrain --dataset ogbg-moltox21_main --batch_size 64 --num_accumulation_steps 1 --num_epochs 100 --warmup_proportion 0.5 --amp --device cuda:0
python train.py --name main_no_pretrain --dataset ogbg-molbbbp_main --batch_size 64 --num_accumulation_steps 1 --num_epochs 100 --warmup_proportion 0.5 --amp --device cuda:0
python train.py --name main_no_pretrain --dataset ogbg-molbace_main --batch_size 64 --num_accumulation_steps 1 --num_epochs 100 --warmup_proportion 0.5 --amp --device cuda:0
python train.py --name main_no_pretrain --dataset ogbg-molclintox_main --batch_size 64 --num_accumulation_steps 1 --num_epochs 100 --warmup_proportion 0.5 --amp --device cuda:0
python train.py --name main_no_pretrain --dataset ogbg-molsider_main --batch_size 64 --num_accumulation_steps 1 --num_epochs 100 --warmup_proportion 0.5 --amp --device cuda:0
python train.py --name main_no_pretrain --dataset ogbg-mollipo_main --batch_size 64 --num_accumulation_steps 1 --num_epochs 100 --warmup_proportion 0.5 --amp --device cuda:0
python train.py --name main_no_pretrain --dataset ogbg-molesol_main --batch_size 64 --num_accumulation_steps 1 --num_epochs 100 --warmup_proportion 0.5 --amp --device cuda:0
python train.py --name main_no_pretrain --dataset ogbg-molfreesolv_main --batch_size 64 --num_accumulation_steps 1 --num_epochs 100 --warmup_proportion 0.5 --amp --device cuda:0
