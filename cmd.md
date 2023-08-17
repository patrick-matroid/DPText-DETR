python demo/demo.py --config-file configs/DPText_DETR/ArT/R_50_poly.yaml --input ../DeepSolo/datasets/ic15/test_images --output ic15_test --opts MODEL.WEIGHTS ckpts/art_final.pth


python tools/train_net.py --config-file configs/DPText_DETR/IC15/R_50_poly_finetuned_ArT_for_ic15.yaml --eval-only MODEL.WEIGHTS ckpts/art_final.pth



python tools/train_net.py --config-file configs/DPText_DETR/TotalText/R_50_poly_recog.yaml --eval-only MODEL.WEIGHTS ckpts/totaltext_final.pth

python tools/train_net.py --config-file configs/DPText_DETR/IC15/R_50_poly_finetuned_ArT_for_ic15_art_imgsize.yaml --eval-only MODEL.WEIGHTS ckpts/art_final.pth

python tools/train_net.py --config-file configs/DPText_DETR/IC15/R_50_poly_finetuned_ArT_for_ic15_art_imgsize_0.2th.yaml --eval-only MODEL.WEIGHTS ckpts/art_final.pth



python tools/train_net.py --config-file configs/DPText_DETR/IC15/R_50_poly_finetuned_ArT_for_ic15_art_imgsize_generic.yaml --eval-only MODEL.WEIGHTS ckpts/art_final.pth

python tools/train_net.py --config-file configs/DPText_DETR/IC15/R_50_poly_finetuned_ArT_for_ic15_art_imgsize_strong.yaml --eval-only MODEL.WEIGHTS ckpts/art_final.pth

python tools/train_net.py --config-file configs/DPText_DETR/IC15/R_50_poly_finetuned_ArT_for_ic15_art_imgsize_weak.yaml --eval-only MODEL.WEIGHTS ckpts/art_final.pth


python tools/train_net.py --config-file configs/DPText_DETR/IC15/R_50_poly_pretrain_ArT_for_ic15_art_imgsize_generic.yaml --eval-only MODEL.WEIGHTS ckpts/pretrain_art.pth

python tools/train_net.py --config-file configs/DPText_DETR/IC15/R_50_poly_pretrain_ArT_for_ic15_art_imgsize_strong.yaml --eval-only MODEL.WEIGHTS ckpts/pretrain_art.pth

python tools/train_net.py --config-file configs/DPText_DETR/IC15/R_50_poly_pretrain_ArT_for_ic15_art_imgsize_weak.yaml --eval-only MODEL.WEIGHTS ckpts/pretrain_art.pth