INPUT_JSON="./data/p04_left/input.json"

python ct_update.py --input $INPUT_JSON --desc "output/depth_anything_5mm"

python evaluation.py --input $INPUT_JSON --desc "output/depth_anything_5mm"

# python ct_update_depth.py --input $INPUT_JSON --desc "output/depth_ps_depth_ablation_2"

# python evaluation.py --input $INPUT_JSON --desc "output/depth_ps_depth_ablation_2"

# python ct_update_reg.py --input $INPUT_JSON --desc "output/depth_ps_reg_ablation"

# python evaluation.py --input $INPUT_JSON --desc "output/depth_ps_reg_ablation"


# * variable sequence length
# BASE_INPUT="./data/p04_left/input"

# python ct_update_depth.py --input $BASE_INPUT"_1.json" --desc "output/seq_length/gt_depth/len_1"
# python ct_update_depth.py --input $BASE_INPUT"_10.json" --desc "output/seq_length/gt_depth/len_10"
# python ct_update_depth.py --input $BASE_INPUT"_20.json" --desc "output/seq_length/gt_depth/len_20"
# python ct_update_depth.py --input $BASE_INPUT"_30.json" --desc "output/seq_length/gt_depth/len_30"
# python ct_update_depth.py --input $BASE_INPUT"_40.json" --desc "output/seq_length/gt_depth/len_40"
# python ct_update_depth.py --input $BASE_INPUT"_50.json" --desc "output/seq_length/gt_depth/len_50"
# python ct_update_depth.py --input $BASE_INPUT"_60.json" --desc "output/seq_length/gt_depth/len_60"
# python ct_update_depth.py --input $BASE_INPUT"_70.json" --desc "output/seq_length/gt_depth/len_70"
# python ct_update_depth.py --input $BASE_INPUT"_80.json" --desc "output/seq_length/gt_depth/len_80"

# python evaluation.py --input $INPUT_JSON --desc "output/seq_length/gt_depth/len_1"
# python evaluation.py --input $INPUT_JSON --desc "output/seq_length/gt_depth/len_10"
# python evaluation.py --input $INPUT_JSON --desc "output/seq_length/gt_depth/len_20"
# python evaluation.py --input $INPUT_JSON --desc "output/seq_length/gt_depth/len_30"
# python evaluation.py --input $INPUT_JSON --desc "output/seq_length/gt_depth/len_40"
# python evaluation.py --input $INPUT_JSON --desc "output/seq_length/gt_depth/len_50"
# python evaluation.py --input $INPUT_JSON --desc "output/seq_length/gt_depth/len_60"
# python evaluation.py --input $INPUT_JSON --desc "output/seq_length/gt_depth/len_70"
# python evaluation.py --input $INPUT_JSON --desc "output/seq_length/gt_depth/len_80"