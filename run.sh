# Ablation Experiment
echo -e "Ablation Experiment\n"

if [ ! -d "./ablation_exp" ]; then
  mkdir "./ablation_exp"
fi

echo -e "harm\n"

if [ ! -d "./ablation_exp/harm" ]; then
  mkdir "./ablation_exp/harm"
fi

# pre_output_layers
echo -e "\tpre_putput_layers=1, 3, 5\n"
python main.py --mydataset harm --num_pre_output_layers 1 --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --fusion weighted_sum --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/harm/num_pre_putput_layers=1.log" 2>>"./ablation_exp/harm/num_pre_putput_layers=1.log"
python main.py --mydataset harm --num_pre_output_layers 3 --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --fusion weighted_sum --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap  1>>"./ablation_exp/harm/num_pre_putput_layers=3.log" 2>>"./ablation_exp/harm/num_pre_putput_layers=3.log"
python main.py --mydataset harm --num_pre_output_layers 5 --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --fusion weighted_sum --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/harm/num_pre_putput_layers=5.log" 2>>"./ablation_exp/harm/num_pre_putput_layers=5.log"

## map_layers
echo -e "\twith_pro_cap=false\n"
python main.py --mydataset harm --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --fusion weighted_sum --num_pre_output_layers 3 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 1>>"./ablation_exp/harm/with_pro_cap=f.log" 2>>"./ablation_exp/harm/with_pro_cap=f.log"

# weight_generator
echo -e "\tweight_generator='acc', 'direct', 'fixed'\n"

echo -e "\tweight_generator='acc'\n"
python main.py --mydataset harm --weight_generator acc --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --fusion weighted_sum --num_pre_output_layers 3 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/harm/weight_generator=acc.log" 2>>"./ablation_exp/harm/weight_generator=acc.log"
echo -e "\tweight_generator='direct'\n"
python main.py --mydataset harm --weight_generator direct --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --fusion weighted_sum --num_pre_output_layers 3 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/harm/weight_generator=direct.log" 2>>"./ablation_exp/harm/weight_generator=direct.log"
echo -e "\tweight_generator='fixed'\n"
python main.py --mydataset harm --weight_generator fixed --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --fusion weighted_sum --num_pre_output_layers 3 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/harm/weight_generator=fixed.log" 2>>"./ablation_exp/harm/weight_generator=fixed.log"

# fusion
echo -e "\tfusion='weighted_sum', 'weighted_align'\n"

echo -e "\tfusion='weighted_sum'\n"
python main.py --mydataset harm --fusion weighted_sum --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --num_pre_output_layers 3 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/harm/fusion=weighted_sum.log" 2>>"./ablation_exp/harm/fusion=weighted_sum.log"
echo -e "\tfusion='weighted_align'\n"
python main.py --mydataset harm --fusion weighted_align --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --num_pre_output_layers 3 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/harm/fusion=weighted_align.log" 2>>"./ablation_exp/harm/fusion=weighted_align.log"

# pre_output_layers=5
echo -e "\tfusion='sum', with_pro_cap=TRUE\n"
python main.py --mydataset harm --fusion sum --with_pro_cap --num_pre_output_layers 5 --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 1>>"./ablation_exp/harm/fusion=sum_and_with_pro_cap=TRUE.log" 2>>"./ablation_exp/harm/fusion=sum_and_with_pro_cap=TRUE.log"

echo -e "\tfusion='weighted_sum', with_pro_cap=FALSE\n"
python main.py --mydataset harm --fusion weighted_sum --num_pre_output_layers 5 --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 1>>"./ablation_exp/harm/fusion=weighted_sum_and_with_pro_cap=FALSE.log" 2>>"./ablation_exp/harm/fusion=weighted_sum_and_with_pro_cap=FALSE.log"

echo -e "\tfusion='concat', with_pro_cap=TRUE\n"
python main.py --mydataset harm --fusion concat --with_pro_cap --num_pre_output_layers 5 --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 1>>"./ablation_exp/harm/fusion=concat_and_with_pro_cap=TRUE.log" 2>>"./ablation_exp/harm/fusion=concat_and_with_pro_cap=TRUE.log"


echo -e "mimc\n"

if [ ! -d "./ablation_exp/mimc" ]; then
  mkdir "./ablation_exp/mimc"
fi

# pre_output_layers
echo -e "\tpre_putput_layers=1, 3, 5\n"
python main.py --mydataset mimc --num_pre_output_layers 1 --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --fusion weighted_sum --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/mimc/num_pre_putput_layers=1.log" 2>>"./ablation_exp/mimc/num_pre_putput_layers=1.log"
python main.py --mydataset mimc --num_pre_output_layers 3 --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --fusion weighted_sum --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap  1>>"./ablation_exp/mimc/num_pre_putput_layers=3.log" 2>>"./ablation_exp/mimc/num_pre_putput_layers=3.log"
python main.py --mydataset mimc --num_pre_output_layers 5 --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --fusion weighted_sum --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/mimc/num_pre_putput_layers=5.log" 2>>"./ablation_exp/mimc/num_pre_putput_layers=5.log"

# with_pro_cap
echo -e "\twith_pro_cap=false\n"
python main.py --mydataset mimc --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --fusion weighted_sum --num_pre_output_layers 3 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 1>>"./ablation_exp/mimc/with_pro_cap=f.log" 2>>"./ablation_exp/mimc/with_pro_cap=f.log"

# weight_generator
echo -e "\tweight_generator='acc', 'direct', 'fixed'\n"

echo -e "\tweight_generator='acc'\n"
python main.py --mydataset mimc --weight_generator acc --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --fusion weighted_sum --num_pre_output_layers 3 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/mimc/weight_generator=acc.log" 2>>"./ablation_exp/mimc/weight_generator=acc.log"
echo -e "\tweight_generator='direct'\n"
python main.py --mydataset mimc --weight_generator direct --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --fusion weighted_sum --num_pre_output_layers 3 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/mimc/weight_generator=direct.log" 2>>"./ablation_exp/mimc/weight_generator=direct.log"
echo -e "\tweight_generator='fixed'\n"
python main.py --mydataset mimc --weight_generator fixed --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --fusion weighted_sum --num_pre_output_layers 3 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/mimc/weight_generator=fixed.log" 2>>"./ablation_exp/mimc/weight_generator=fixed.log"

# fusion
echo -e "\tfusion='weighted_sum', 'weighted_align'\n"

#echo -e "\tfusion='weighted_sum'\n"
#python main.py --mydataset mimc --fusion weighted_sum --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --num_pre_output_layers 3 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/mimc/fusion=weighted_sum.log" 2>>"./ablation_exp/mimc/fusion=weighted_sum.log"
echo -e "\tfusion='weighted_align'\n"
python main.py --mydataset mimc --fusion weighted_align --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --num_pre_output_layers 3 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/mimc/fusion=weighted_align.log" 2>>"./ablation_exp/mimc/fusion=weighted_align.log"

# pre_output_layers=5
echo -e "\tfusion='sum', with_pro_cap=TRUE\n"
python main.py --mydataset mimc --fusion sum --with_pro_cap --num_pre_output_layers 5 --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 1>>"./ablation_exp/mimc/fusion=sum_and_with_pro_cap=TRUE.log" 2>>"./ablation_exp/mimc/fusion=sum_and_with_pro_cap=TRUE.log"

echo -e "\tfusion='weighted_sum', with_pro_cap=FALSE\n"
python main.py --mydataset mimc --fusion weighted_sum --num_pre_output_layers 5 --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 1>>"./ablation_exp/mimc/fusion=weighted_sum_and_with_pro_cap=FALSE.log" 2>>"./ablation_exp/mimc/fusion=weighted_sum_and_with_pro_cap=FALSE.log"

echo -e "\tfusion='concat', with_pro_cap=TRUE\n"
python main.py --mydataset mimc --fusion concat --with_pro_cap --num_pre_output_layers 5 --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 1>>"./ablation_exp/mimc/fusion=concat_and_with_pro_cap=TRUE.log" 2>>"./ablation_exp/mimc/fusion=concat_and_with_pro_cap=TRUE.log"

echo -e "additional experiments\n"
#   1. pre_output_layers=7
#   2. fusion=weighted_concat
#   3. fusion=weighted_align_concat

echo -e "harm\n"
python main.py --mydataset harm --num_pre_output_layers 7 --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --fusion weighted_sum --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/harm/num_pre_putput_layers=7.log" 2>>"./ablation_exp/harm/num_pre_putput_layers=7.log"
python main.py --mydataset harm --fusion weighted_concat --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --num_pre_output_layers 3 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/harm/fusion=weighted_concat.log" 2>>"./ablation_exp/harm/fusion=weighted_concat.log"
python main.py --mydataset harm --fusion weighted_align_concat --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --num_pre_output_layers 3 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/harm/fusion=weighted_align_concat.log" 2>>"./ablation_exp/harm/fusion=weighted_align_concat.log"

echo -e "mimc\n"
python main.py --mydataset mimc --num_pre_output_layers 7 --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --fusion weighted_sum --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/mimc/num_pre_putput_layers=7.log" 2>>"./ablation_exp/mimc/num_pre_putput_layers=7.log"
python main.py --mydataset mimc --fusion weighted_concat --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --num_pre_output_layers 3 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/mimc/fusion=weighted_concat.log" 2>>"./ablation_exp/mimc/fusion=weighted_concat.log"
python main.py --mydataset mimc --fusion weighted_align_concat --dataset original --labels original --multilingual_tokenizer_path "none" --clip_pretrained_model "openai/clip-vit-large-patch14" --local_pretrained_weights "none" --caption_mode "none" --use_pretrained_map t --num_mapping_layers 1 --map_dim 32 --num_pre_output_layers 3 --drop_probs 0.2 0.4 0.1 --freeze_image_encoder t --freeze_text_encoder t --gpus 0 --batch_size 16 --lr 0.0001 --weight_fine_grained_loss 0 --weight_super_loss 0 --max_epochs 20 --with_pro_cap 1>>"./ablation_exp/mimc/fusion=weighted_align_concat.log" 2>>"./ablation_exp/mimc/fusion=weighted_align_concat.log"
